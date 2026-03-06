"""
py2wl.pool — KernelPool 分布式内核池
======================================

把多台设备上的 Wolfram 内核统一管理，自动调度任务到最合适的内核。

用法：
    # 环境变量配置（推荐）
    export PY2WL_KERNELS=local,192.168.1.100:9999,192.168.1.102:9999
    export PY2WL_KERNEL_CPU_SCORES=8,16,4   # 对应每个内核的算力权重
    export PY2WL_SCHEDULER=/path/to/my_scheduler.py  # 可选，自定义调度

    from py2wl.pool import KernelPool
    pool = KernelPool()
    result = pool.execute("Eigenvalues[RandomReal[{0,1},{500,500}]]")
    pool.close()

    # 也可以直接构造
    pool = KernelPool(["local", "192.168.1.100:9999"])

自定义调度器脚本格式（PY2WL_SCHEDULER 指向的 .py 文件）：

    def scheduler(kernels, expr):
        \"\"\"
        kernels: list[KernelInfo]  所有内核的只读快照
        expr:    str               即将执行的 WL 表达式
        返回:    int               选中的内核 id
        \"\"\"
        idle = [k for k in kernels if k.status == "idle"]
        if not idle:
            idle = [k for k in kernels if k.status != "offline"]
        return max(idle, key=lambda k: k.cpu_score).id

默认调度策略：
    1. 优先选 idle 且 queue_len 最小的
    2. 全部 busy 时选 avg_ms 最小的（任务最快结束）
    3. offline 的内核自动跳过
    4. 全部 offline 时抛出 RuntimeError
"""

import importlib.util
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

log = logging.getLogger("py2wl.pool")

# 滑动平均窗口大小
_AVG_WINDOW = 10
# 心跳间隔（秒）
_HEARTBEAT_INTERVAL = 5


@dataclass
class KernelInfo:
    """单个内核的状态快照（线程安全读取，写入需持有 _lock）"""
    id:         int
    host:       str    # "local" 或 "192.168.x.x:port"
    status:     str    # "idle" | "busy" | "offline"
    queue_len:  int
    avg_ms:     float  # 滑动平均耗时（毫秒）
    cpu_score:  int    # 算力权重，越大越强

    # 内部字段，不暴露给调度器
    _kernel:    object         = field(repr=False, compare=False)
    _lock:      threading.Lock = field(default_factory=threading.Lock,
                                       repr=False, compare=False)
    _history:   list           = field(default_factory=list,
                                       repr=False, compare=False)

    def snapshot(self) -> "KernelInfo":
        """返回只读快照，供调度器使用（不含内部字段）"""
        with self._lock:
            return KernelInfo(
                id=self.id, host=self.host, status=self.status,
                queue_len=self.queue_len, avg_ms=self.avg_ms,
                cpu_score=self.cpu_score,
                _kernel=None, _lock=threading.Lock(), _history=[],
            )

    def record_time(self, ms: float) -> None:
        """记录一次执行耗时，更新滑动平均"""
        with self._lock:
            self._history.append(ms)
            if len(self._history) > _AVG_WINDOW:
                self._history.pop(0)
            self.avg_ms = sum(self._history) / len(self._history)


class KernelPool:
    """
    多内核统一调度池。

    线程安全：execute() 可以从多个线程并发调用。
    """

    def __init__(self, kernels: Optional[List[str]] = None):
        """
        参数：
            kernels: 内核地址列表，如 ["local", "192.168.1.100:9999"]
                     None 时从环境变量 PY2WL_KERNELS 读取
        """
        self.kernels: List[KernelInfo] = []
        self.scheduler: Callable = self._default_scheduler
        self._stop = threading.Event()
        self._pool_lock = threading.Lock()

        self._init_kernels(kernels)
        self._load_scheduler()
        self._start_stats_thread()

        log.info(f"KernelPool 已初始化，共 {len(self.kernels)} 个内核: "
                 f"{[k.host for k in self.kernels]}")

    # ── 初始化 ───────────────────────────────────────────────

    def _init_kernels(self, kernels: Optional[List[str]]) -> None:
        """解析内核地址列表，建立连接"""
        if kernels is None:
            raw = os.environ.get("PY2WL_KERNELS", "local")
            kernels = [s.strip() for s in raw.split(",") if s.strip()]

        scores_raw = os.environ.get("PY2WL_KERNEL_CPU_SCORES", "")
        scores = []
        if scores_raw:
            try:
                scores = [int(x.strip()) for x in scores_raw.split(",")]
            except ValueError:
                log.warning("PY2WL_KERNEL_CPU_SCORES 格式错误，忽略")

        for i, addr in enumerate(kernels):
            cpu_score = scores[i] if i < len(scores) else self._detect_cpu_score(addr)
            info = self._make_kernel_info(i, addr, cpu_score)
            if info is not None:
                self.kernels.append(info)

        if not self.kernels:
            raise RuntimeError("KernelPool：没有可用的内核，请检查 PY2WL_KERNELS 配置")

    def _make_kernel_info(self, idx: int, addr: str,
                          cpu_score: int) -> Optional[KernelInfo]:
        """尝试连接一个内核，返回 KernelInfo 或 None"""
        if addr == "local":
            try:
                from .kernel import WolframKernel
                kernel = WolframKernel()
                log.info(f"内核 #{idx} 本地内核连接成功")
                return KernelInfo(
                    id=idx, host="local", status="idle",
                    queue_len=0, avg_ms=0.0, cpu_score=cpu_score,
                    _kernel=kernel,
                )
            except Exception as e:
                log.warning(f"本地内核初始化失败: {e}")
                return None
        else:
            try:
                from .remote import RemoteKernel
                kernel = RemoteKernel(addr)
                # 发一次 ping 验证连通性
                kernel.ping()
                log.info(f"内核 #{idx} 远程 {addr} 连接成功")
                return KernelInfo(
                    id=idx, host=addr, status="idle",
                    queue_len=0, avg_ms=0.0, cpu_score=cpu_score,
                    _kernel=kernel,
                )
            except Exception as e:
                log.warning(f"远程内核 {addr} 连接失败（标记 offline）: {e}")
                # 仍然保留条目，心跳线程会定期重试
                from .remote import RemoteKernel
                try:
                    kernel = RemoteKernel.__new__(RemoteKernel)
                    kernel.host, port = addr.split(":")
                    kernel.port = int(port)
                    kernel.sock = None
                    kernel._lock = __import__("threading").Lock()
                except Exception:
                    return None
                return KernelInfo(
                    id=idx, host=addr, status="offline",
                    queue_len=0, avg_ms=0.0, cpu_score=cpu_score,
                    _kernel=kernel,
                )

    def _detect_cpu_score(self, addr: str) -> int:
        """本地内核探测 CPU 核心数，远程默认 4"""
        if addr == "local":
            try:
                import multiprocessing
                return multiprocessing.cpu_count()
            except Exception:
                return 4
        return 4

    # ── 调度器 ───────────────────────────────────────────────

    def _load_scheduler(self) -> None:
        """动态加载用户自定义调度脚本"""
        path = os.environ.get("PY2WL_SCHEDULER")
        if not path:
            return
        if not os.path.exists(path):
            log.warning(f"PY2WL_SCHEDULER 文件不存在: {path}，使用默认调度")
            return
        try:
            spec = importlib.util.spec_from_file_location("_py2wl_scheduler", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "scheduler"):
                log.warning(f"调度脚本中未找到 scheduler() 函数，使用默认调度")
                return
            self.scheduler = mod.scheduler
            log.info(f"已加载自定义调度器: {path}")
        except Exception as e:
            log.warning(f"加载调度脚本失败: {e}，使用默认调度")

    def _default_scheduler(self, kernels: List[KernelInfo], expr: str) -> int:
        """
        默认调度策略：
          1. idle 内核中选 queue_len 最小的（同等时选 cpu_score 最大的）
          2. 全部 busy 时选 avg_ms 最小的
          3. 跳过 offline
        """
        online = [k for k in kernels if k.status != "offline"]
        if not online:
            raise RuntimeError("所有内核均处于 offline 状态")

        idle = [k for k in online if k.status == "idle"]
        if idle:
            return min(idle, key=lambda k: (k.queue_len, -k.cpu_score)).id

        # 全部 busy：选预计最快完成的
        return min(online, key=lambda k: k.avg_ms if k.avg_ms > 0 else float("inf")).id

    # ── 心跳线程 ─────────────────────────────────────────────

    def _start_stats_thread(self) -> None:
        t = threading.Thread(target=self._stats_loop, daemon=True,
                             name="py2wl-pool-heartbeat")
        t.start()

    def _stats_loop(self) -> None:
        """定期对所有远程内核发送 ping，更新 online/offline 状态"""
        while not self._stop.wait(_HEARTBEAT_INTERVAL):
            for info in self.kernels:
                if info.host == "local":
                    continue  # 本地内核不需要心跳
                try:
                    info._kernel.ping()
                    with info._lock:
                        if info.status == "offline":
                            info.status = "idle"
                            log.info(f"内核 #{info.id} ({info.host}) 重新上线")
                except Exception:
                    with info._lock:
                        if info.status != "offline":
                            log.warning(f"内核 #{info.id} ({info.host}) 心跳失败，标记 offline")
                        info.status = "offline"

    # ── 核心执行 ─────────────────────────────────────────────

    def execute(self, expr: str) -> Any:
        """
        将表达式路由到最合适的内核执行。

        线程安全，可并发调用。
        调度 → 分配 → 执行 → 计时 → 归还
        """
        # 获取快照供调度器使用（不持锁，快照是只读副本）
        snapshots = [k.snapshot() for k in self.kernels]

        # 调度器选择目标内核 id
        try:
            target_id = self.scheduler(snapshots, expr)
        except Exception as e:
            raise RuntimeError(f"调度器异常: {e}") from e

        # 找到对应的真实 KernelInfo
        target = next((k for k in self.kernels if k.id == target_id), None)
        if target is None:
            raise RuntimeError(f"调度器返回了不存在的内核 id: {target_id}")

        # 标记 busy，增加队列计数
        with target._lock:
            target.queue_len += 1
            target.status = "busy"

        log.debug(f"任务 → 内核 #{target_id} ({target.host}), "
                  f"queue={target.queue_len}")

        t0 = time.perf_counter()
        try:
            result = target._kernel.evaluate(expr)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            target.record_time(elapsed_ms)
            log.debug(f"内核 #{target_id} 完成，耗时 {elapsed_ms:.0f}ms")
            return result
        except Exception as e:
            # 执行失败：若是远程内核标记 offline
            if target.host != "local":
                with target._lock:
                    target.status = "offline"
                log.warning(f"内核 #{target_id} 执行失败，标记 offline: {e}")
            raise
        finally:
            with target._lock:
                target.queue_len = max(0, target.queue_len - 1)
                if target.queue_len == 0 and target.status == "busy":
                    target.status = "idle"

    def status(self) -> List[dict]:
        """返回所有内核当前状态，供监控/日志使用"""
        return [
            {
                "id":        k.id,
                "host":      k.host,
                "status":    k.status,
                "queue_len": k.queue_len,
                "avg_ms":    round(k.avg_ms, 1),
                "cpu_score": k.cpu_score,
            }
            for k in self.kernels
        ]

    def close(self) -> None:
        """停止心跳线程，关闭所有内核连接"""
        self._stop.set()
        for info in self.kernels:
            try:
                info._kernel.close()
                log.debug(f"内核 #{info.id} ({info.host}) 已关闭")
            except Exception as e:
                log.debug(f"关闭内核 #{info.id} 时发生异常（非致命）: {e}")
        log.info("KernelPool 已关闭")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
