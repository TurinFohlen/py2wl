"""
compat/_proxy_base.py
---------------------
通用代理基础模块：把 Python 函数调用路由到 Wolfram Kernel。
numpy.py / torch.py / sympy.py 等都继承自此，只需声明自己的 root 命名空间。

容错策略由环境变量 WOLFRAM_FAULT_MODE 控制：
  strict       (默认) 任何错误直接抛出
  auto-ai      AI 有高置信候选时自动重试，否则降为 interactive
  interactive  遇错暂停，控制台询问用户
"""

import os
import sys
import logging
from pathlib import Path
from ._state import _state

log = logging.getLogger("py2wl.compat")

_DEFAULT_MAPPINGS = str(Path(__file__).parent / "mappings")


# ── 延迟初始化：resolver ────────────────────────────────────────
def _get_resolver():
    if _state["resolver"] is not None:
        return _state["resolver"]
    from ._core.metadata  import MetadataRepository
    from ._core.resolver  import ResolutionEngine
    from ._core.ai_plugin import AIPlugin

    mappings_dir = os.environ.get("WOLFRAM_MAPPINGS_DIR", _DEFAULT_MAPPINGS)
    repo = MetadataRepository(mappings_dir)
    ai   = AIPlugin() if os.environ.get("WOLFRAM_AI_PLUGIN") else None
    _state["resolver"] = ResolutionEngine(repo, ai)
    log.info(f"兼容层初始化，映射目录：{mappings_dir}，共 {len(repo.all_rules)} 条规则")
    return _state["resolver"]


# ── 延迟初始化：kernel ──────────────────────────────────────────
def _get_kernel():
    if _state["kernel"] is not None:
        return _state["kernel"]
    from py2wl.kernel import WolframKernel
    _state["kernel"] = WolframKernel()
    return _state["kernel"]


# ── 延迟初始化：fault_handler ───────────────────────────────────
def _get_fault_handler():
    if _state.get("fault_handler") is not None:
        return _state["fault_handler"]
    from ._core.fault_handler import FaultHandler, FaultMode
    from ._core.ai_plugin     import AIPlugin

    resolver = _get_resolver()
    ai = AIPlugin() if os.environ.get("WOLFRAM_AI_PLUGIN") else None

    mode_str = os.environ.get("WOLFRAM_FAULT_MODE", "strict")
    try:
        mode = FaultMode(mode_str)
    except ValueError:
        log.warning(f"未知容错模式 '{mode_str}'，回退到 strict")
        mode = FaultMode.STRICT

    _state["fault_handler"] = FaultHandler(resolver._repo, ai, mode)
    log.info(f"容错系统初始化，模式：{mode.value}")
    return _state["fault_handler"]


# ── 双哈希缓存层 ────────────────────────────────────────────────
from ._core.result_cache import _cache as _CACHE, ResultCache

def _maybe_cached(rule, args, kwargs, core_expr: str):
    """
    命中 → 直接返回缓存结果，绕过序列化 + 内核 + 反序列化。
    未命中 → 返回 None。
    cacheable: false 的规则永远不查缓存。
    """
    if rule and rule.get("cacheable") is False:
        return None
    cmd_hash = ResultCache.hash_expr(core_expr)
    return _CACHE.get(cmd_hash)

def _store_in_cache(rule, args, kwargs, core_expr: str, value) -> None:
    """
    将 (cmd_hash, result_hash, value) 存入缓存。
    result_hash 用于跨表达式结果去重。
    cacheable: false 的规则跳过。
    存入前对浮点数结果做精度截断（与 hash_result 保持一致），
    使缓存返回值和直接计算结果在数值上完全相同。
    """
    if rule and rule.get("cacheable") is False:
        return
    from ._core.result_cache import _normalize_for_hash
    value = _normalize_for_hash(value)
    cmd_hash = ResultCache.hash_expr(core_expr)
    _CACHE.put(cmd_hash, value)

def cache_stats() -> dict:
    """返回缓存统计（条目数/命中率/去重数）。"""
    return _CACHE.stats()

def cache_clear() -> None:
    """清空结果缓存。"""
    _CACHE.clear()


# ── 核心调用类 ──────────────────────────────────────────────────
class _WolframCallable:
    """封装单个 Wolfram 函数调用，含完整容错循环。"""
    __slots__ = ("_path",)

    def __init__(self, path: str):
        self._path = path

    def __call__(self, *args, **kwargs):
        return self._call_with_fault(self._path, args, kwargs, rule=None)

    def _call_with_fault(self, path, args, kwargs, rule=None):
        """
        带容错的调用循环。
        rule=None 表示走正常解析流程；rule!=None 表示直接用指定规则（容错重试）。
        """
        resolver = _get_resolver()
        handler  = _get_fault_handler()

        # ── Step 1：解析规则 ─────────────────────────────────────
        if rule is None:
            rule = resolver.resolve(path, args=args, kwargs=kwargs)
            if rule is None:
                exc = AttributeError(
                    f"未找到 '{path}' 的 Wolfram 映射。"
                    + (f"\n候选：{[r['python_path'] for r in resolver.candidates_for_hint(path)[:5]]}"
                       if resolver.candidates_for_hint(path) else "")
                )
                return self._handle_fault(exc, path, args, kwargs,
                                          raw_wl=None, handler=handler)

        # ── Step 2：构造核心表达式 ───────────────────────────────
        try:
            core_expr = resolver.build_wl_expr(rule, args, kwargs)
        except Exception as e:
            return self._handle_fault(e, path, args, kwargs,
                                      raw_wl=None, handler=handler)

        # ── Step 3：缓存检查 ─────────────────────────────────────
        cached = _maybe_cached(rule, args, kwargs, core_expr)
        if cached is not None:
            return cached

        # ── Step 4：内核执行 ─────────────────────────────────────
        kernel = _get_kernel()
        oc = rule.get("output_converter", "from_wxf")

        try:
            if oc == "from_wl_image":
                # 图像输出：通过 evaluate_to_file 获得文件路径
                fmt = rule.get("image_format", "png")
                result = kernel.evaluate_to_file(core_expr, fmt=fmt)
            else:
                # 其他类型：直接 evaluate 获得 Python 对象
                # _normalize：PackedArray/WLFunction → Python 原生类型
                from ._core.converters import _normalize
                result = _normalize(kernel.evaluate(core_expr))

            # ── Step 5：存入缓存 ─────────────────────────────────
            _store_in_cache(rule, args, kwargs, core_expr, result)

            return result

        except Exception as e:
            return self._handle_fault(e, path, args, kwargs,
                                      raw_wl=core_expr, handler=handler)

    def _handle_fault(self, exc, path, args, kwargs, raw_wl, handler):
        """
        交给 FaultHandler 决定下一步：重试 / 跳过 / 重抛。
        """
        from ._core.fault_handler import ActionKind

        action = handler.handle(exc, path, args, kwargs, raw_wl)

        if action.kind == ActionKind.RETRY_RULE:
            log.info(f"容错重试：{path} → {action.rule['python_path']}")
            return self._call_with_fault(path, args, kwargs, rule=action.rule)

        if action.kind == ActionKind.RETRY_EXPR:
            log.info(f"用户自定义表达式重试：{action.custom_expr[:60]}")
            kernel = _get_kernel()
            try:
                raw = kernel.evaluate(action.custom_expr)
                from ._core.converters import convert_output
                return convert_output(raw, "from_wl_passthrough")
            except Exception as e2:
                raise RuntimeError(
                    f"自定义表达式执行失败：{e2}") from e2

        if action.kind == ActionKind.SKIP:
            log.warning(f"已跳过调用：{path}，返回 None")
            return None

        # ActionKind.RAISE — 重抛原始异常
        raise exc

    def __repr__(self):
        return f"<WolframCallable '{self._path}'>"


# ── 命名空间代理 ────────────────────────────────────────────────
class LibraryProxy:
    """
    通用命名空间代理。
    - 精确路径匹配 → _WolframCallable（或常量立即求值）
    - 其余 → 递归 LibraryProxy（支持 np.fft.fft 链式访问）
    """
    def __init__(self, path: str):
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_const_cache", {})
        object.__setattr__(self, "_const_lock", __import__("threading").Lock())

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        cache = object.__getattribute__(self, "_const_cache")
        if name in cache:
            return cache[name]

        new_path = f"{object.__getattribute__(self, '_path')}.{name}"
        rule = _get_resolver()._repo.get_rule(new_path)

        if rule is not None:
            if rule.get("constant"):
                lock = object.__getattribute__(self, "_const_lock")
                with lock:
                    if name in cache:
                        return cache[name]
                    value = _WolframCallable(new_path)()
                    cache[name] = value
                return value
            return _WolframCallable(new_path)

        return LibraryProxy(new_path)

    def __call__(self, *args, **kwargs):
        return _WolframCallable(
            object.__getattribute__(self, "_path"))(*args, **kwargs)

    def __repr__(self):
        return f"<WolframProxy '{object.__getattribute__(self, '_path')}'>"


# ── 便利函数 ────────────────────────────────────────────────────
def list_mappings():
    """列出所有已加载的映射规则。"""
    return _get_resolver()._repo.all_rules

def search(query: str):
    """按关键词 / 标签 / 路径前缀搜索映射。"""
    return _get_resolver()._repo.search_rules(query)

def reload_mappings(directory: str = None):
    """热重载映射目录（同时重置容错缓存）。"""
    _state["resolver"]      = None
    _state["fault_handler"] = None
    if directory:
        os.environ["WOLFRAM_MAPPINGS_DIR"] = directory
    _get_resolver()

def inject_kernel(kernel):
    """测试用：注入 MockKernel。"""
    _state["kernel"] = kernel

def fault_summary():
    """返回本次会话所有自动纠错记录，供调试用。"""
    h = _state.get("fault_handler")
    return h.correction_summary() if h else []

def set_fault_mode(mode: str):
    """
    运行时切换容错模式，无需重启。
    mode: 'strict' | 'auto-ai' | 'interactive'
    """
    from ._core.fault_handler import FaultMode
    h = _get_fault_handler()
    h.set_mode(FaultMode(mode))
    log.info(f"容错模式已切换为：{mode}")