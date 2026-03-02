"""
kernel.py — Wolfram 内核单例（WSTP / WolframLanguageSession）

关键设计决策
──────────────
- 自动检测 CPU 核心数并设置多线程环境变量（OMP_NUM_THREADS 等）
- 自动启动 Wolfram 并行子内核（Parallel[] 所需）
- close() 必须 join controller 线程才能真正等到内核退出。
- wolframclient 的 terminate() / stop() 返回的 future 在 STOP 任务被
  出队时立即 set_result，但实际杀进程 (_kernel_stop) 在 controller
  线程的 finally 块里异步执行。不 join 直接退出 → controller 线程成
  孤儿 → Quit[] 未发送 → license 未释放 → 下次启动被 license 挡住。
"""
import atexit
import concurrent.futures
import logging
import os
import queue
import shutil
import signal
import threading
import time
import uuid

from wolframclient.evaluation import WolframLanguageSession

# 导入内部清理模块
from .compat._core.cleaner import cleanup as _cleanup

log = logging.getLogger("py2wl")

CACHE_DIR = os.environ.get("WOLFRAM_CACHE_DIR", "/sdcard/wolfram_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# controller 线程最长等待：TERMINATE_TIMEOUT(默认3s) + 缓冲
_CTRL_JOIN_TIMEOUT = 10

def _atexit_close():
    """
    模块级 atexit，只注册一次。
    永远操作当前 WolframKernel._instance，不绑定具体对象。
    避免多次实例化导致 atexit 累积多个 close，按逆序乱跑的问题。
    """
    inst = WolframKernel._instance
    if inst is not None:
        inst.close()

atexit.register(_atexit_close)


def _compute_hash(expr: str) -> str:
    import hashlib
    return hashlib.sha256(expr.encode()).hexdigest()


def _set_thread_env():
    """
    自动设置多线程环境变量，让底层 BLAS/LAPACK 库使用所有可用 CPU 核心。
    用户可通过 WOLFRAM_NUM_THREADS 环境变量覆盖。
    """
    # 如果用户已显式设置，则尊重用户设置
    if "WOLFRAM_NUM_THREADS" in os.environ:
        num = os.environ["WOLFRAM_NUM_THREADS"]
    else:
        # 否则检测 CPU 核心数（在容器中可能返回限制后的数量）
        try:
            import multiprocessing
            num = str(multiprocessing.cpu_count())
        except:
            num = "1"  # 保底
        os.environ["WOLFRAM_NUM_THREADS"] = num

    # 设置常见 BLAS 库的线程数环境变量
    os.environ.setdefault("OMP_NUM_THREADS", num)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", num)
    os.environ.setdefault("MKL_NUM_THREADS", num)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", num)
    log.info(f"启用多线程计算，线程数: {num}")


def _shutdown_session(session) -> None:
    """
    彻底关闭一个 WolframLanguageSession：

    1. 优雅关闭（发 Quit[]，等内核自行退出，最多 TERMINATE_TIMEOUT=3s）
    2. join controller 线程（等 _kernel_stop finally 真正跑完）
    3. SIGKILL 兜底（controller join 超时时手动杀进程）
    """
    if session is None:
        return

    ctrl = getattr(session, "kernel_controller", None)

    # 拿 PID（terminate 后 kernel_proc 会被置 None，提前取）
    pid = None
    try:
        if ctrl and ctrl.kernel_proc:
            pid = ctrl.kernel_proc.pid
    except Exception:
        pass

    # ── Step 1: 优雅停止（发 Quit[]）─────────────────────────
    try:
        # gracefully=True → 发 Quit[]，等内核自行退出后再 kill
        # future.result() 在 STOP 出队时返回，不等 _kernel_stop 完成
        session.stop_future(gracefully=True).result(timeout=1)
    except Exception:
        pass

    # ── Step 2: join controller 线程（等 _kernel_stop 跑完）──
    if ctrl is not None and ctrl.is_alive():
        ctrl.join(timeout=_CTRL_JOIN_TIMEOUT)
        if ctrl.is_alive():
            log.warning("controller 线程 join 超时，强制 SIGKILL")

    # ── Step 3: SIGKILL 兜底──────────────────────────────────
    if pid is not None:
        try:
            os.kill(pid, 0)          # 检测是否还活着
            log.debug(f"SIGKILL kernel PID {pid}")
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass                     # 已经死了，正常
        except Exception as e:
            log.debug(f"SIGKILL failed (non-critical): {e}")


class WolframKernel:
    _instance = None
    _lock     = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._initialized = False
                cls._instance = obj
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 在启动内核前设置多线程环境变量
        _set_thread_env()

        self._kernel_path = os.environ.get("WOLFRAM_EXEC")
        if not self._kernel_path:
            raise RuntimeError("WOLFRAM_EXEC 环境变量未设置")

        self._session = None
        self._running = True
        self._req_queue = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="wolfram-worker")

        self._parallel_launched = False   # 懒加载标志
        self._worker.start()
        self._start_session()
        log.info("WolframKernel 已初始化（并行子内核将在首次调用时懒启动）")

    # ── session 管理 ──────────────────────────────────────────

    # wolframclient 默认 socket 超时 20s，在资源紧张的宿主机上首次启动
    # 偶尔会误触发（内核进程已跑起来但尚未握手完毕），导致不必要的重启。
    # 调高到 60s，也可通过 WOLFRAM_INIT_TIMEOUT 环境变量覆盖。
    _SESSION_INIT_TIMEOUT = int(os.environ.get("WOLFRAM_INIT_TIMEOUT", "60"))

    def _start_session(self):
        old, self._session = self._session, None
        if old is not None:
            _shutdown_session(old)
        sess = WolframLanguageSession(
            kernel=self._kernel_path,
            kernel_loglevel=logging.WARNING,
            initiate_timeout=self._SESSION_INIT_TIMEOUT,
        )
        # controller 线程在 request_kernel_start()（即首次 evaluate）前尚未 start()。
        # 在 start() 之前把它设为 daemon，这样 threading._shutdown() 就不会
        # 无限等待它——atexit 里我们已经主动 join，daemon 只是最后的安全网。
        ctrl = getattr(sess, "kernel_controller", None)
        if ctrl is not None:
            ctrl.daemon = True
        self._session = sess
        log.info("WSTP 会话已启动")

    def _restart_session(self):
        log.warning("内核异常，重启会话…")
        with self._lock:
            self._start_session()
            self._parallel_launched = False   # 重启后并行子内核需重新初始化

    def _launch_parallel_kernels(self):
        """
        懒加载：启动 Wolfram 并行子内核，用于 Parallel 系列函数。
        由首次 evaluate() 经 _ensure_parallel() 触发，而非在 __init__ 里调用。

        原因：
          1. 主内核冷启动（~7s）期间无需等待子内核，减少感知延迟。
          2. 避免主内核尚未就绪时 LaunchKernels[] 进入队列，与首次实际请求
             竞争，误触发 "Failed to communicate" 后不必要地重启。
        """
        if self._parallel_launched:
            return
        self._parallel_launched = True   # 先置位，防止并发重入
        try:
            num_kernels = os.environ.get("WOLFRAM_PARALLEL_KERNELS")
            cmd = f"LaunchKernels[{num_kernels}]" if num_kernels else "LaunchKernels[]"
            self.evaluate(cmd, _skip_parallel_init=True)
            count = self.evaluate("$KernelCount", _skip_parallel_init=True)
            log.info(f"已启动 {count} 个并行子内核")
        except Exception as e:
            self._parallel_launched = False   # 失败时复位，允许下次重试
            log.warning(f"启动并行内核失败（不影响主内核）: {e}")

    def _ensure_parallel(self):
        """在首次 evaluate 时懒启动并行子内核（幂等，仅执行一次）。"""
        if not self._parallel_launched:
            self._launch_parallel_kernels()

    # ── worker loop ───────────────────────────────────────────

    def _worker_loop(self):
        while self._running:
            try:
                item = self._req_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break
            expr, future = item
            if future.cancelled():
                continue
            sess = self._session
            if sess is None or not self._running:
                future.cancel()
                continue
            try:
                future.set_result(sess.evaluate(expr))
            except Exception as e:
                if not self._running:
                    future.set_exception(e)
                    break
                log.warning(f"执行失败，重启: {e}")
                self._restart_session()
                try:
                    future.set_result(self._session.evaluate(expr))
                except Exception as e2:
                    future.set_exception(e2)

    # ── public API ────────────────────────────────────────────

    def evaluate(self, expr: str, _skip_parallel_init: bool = False):
        # 首次真正的业务调用时懒启动并行子内核。
        # _launch_parallel_kernels 内部调用 evaluate 时传 _skip_parallel_init=True
        # 避免递归触发。
        if not _skip_parallel_init:
            self._ensure_parallel()
        f = concurrent.futures.Future()
        self._req_queue.put((expr, f))
        return f.result()

    def evaluate_to_file(self, expr: str, fmt: str = "wxf",
                         out_dir: str = "/sdcard/wolfram_out",
                         no_cache: bool = False) -> str:
        fmt_map = {
            "json":"JSON","txt":"Text","text":"Text","csv":"CSV","tsv":"TSV",
            "png":"PNG","jpg":"JPEG","jpeg":"JPEG","gif":"GIF","bmp":"BMP",
            "tif":"TIFF","tiff":"TIFF","pdf":"PDF","svg":"SVG","eps":"EPS",
            "wdx":"WDX","tex":"TeX","table":"Table","list":"List","mx":"MX",
        }
        wl_fmt = fmt_map.get(fmt.lower())
        if wl_fmt is None:
            raise ValueError(f"不支持的格式 '{fmt}'")

        if not no_cache:
            cache = os.path.join(CACHE_DIR, f"{_compute_hash(expr)}.{fmt}")
            if os.path.exists(cache):
                return cache
        else:
            cache = None

        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, f"{uuid.uuid4().hex}.{fmt}")
        self.evaluate(f'Export["{filepath}", {expr}, "{wl_fmt}"]')

        for _ in range(20):
            if os.path.exists(filepath):
                if cache:
                    shutil.copy2(filepath, cache)
                return filepath
            time.sleep(0.5)
        raise RuntimeError(f"Export 未生成文件：{filepath}")

    def close(self):
        if not self._running:
            return                   # 幂等
        self._running = False

        # ── Step 1: 先关闭并行子内核（直接调用 session，不走 worker queue）
        # 必须在 put(None) 之前、worker 退出之前执行；否则 evaluate() 把
        # future 放入队列后无人消费，f.result() 永久阻塞进程无法退出。
        try:
            sess = self._session
            if sess is not None:
                sess.evaluate("CloseKernels[]")
            log.debug("已关闭并行子内核")
        except Exception as e:
            log.debug(f"关闭并行内核时发生异常: {e}")

        # ── Step 2: 通知 worker 退出
        self._req_queue.put(None)    # 唤醒 worker，使其 break out of loop

        sess, self._session = self._session, None
        _shutdown_session(sess)      # join controller + SIGKILL 兜底

        self._worker.join(timeout=3)

        with self._lock:
            WolframKernel._instance = None
        self._initialized = False   # 允许本对象被重新 __init__（防御性重置）
        log.info("内核已关闭")

        # 每次内核关闭后自动清理残留进程（异步）
        try:
            if _cleanup():
                log.debug("清理脚本已杀死残留进程")
        except Exception as e:
            log.warning(f"清理时发生异常: {e}")

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()