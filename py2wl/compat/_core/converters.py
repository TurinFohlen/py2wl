"""
compat/_core/converters.py
--------------------------
输出管道（WXF-only）：
  from_wxf      → WXF 文件路径 → wolframclient 无损反序列化（唯一数值出口）
  from_wl_image → PNG 临时文件 → 复制到持久路径后返回（防止被 finally 删除）
"""

import os
import math
import shutil
import tempfile
import logging
from typing import Any

log = logging.getLogger("py2wl.compat")

try:
    from wolframclient.deserializers import binary_deserialize
    HAS_WOLFRAMCLIENT = True
except ImportError:
    HAS_WOLFRAMCLIENT = False
    log.debug("wolframclient 未安装，from_wxf 不可用。安装：pip install wolframclient")


# ═══════════════════════════════════════════════════════════════
#  输入转换器：Python → Wolfram 表达式字符串
# ═══════════════════════════════════════════════════════════════

def _float_to_wl(v) -> str:
    """
    Python float → Wolfram 数值字符串。

    关键：Python 科学计数法用 'e'（如 6.12e-17），
    而 WL 把 'e' 解析为变量 Global`e，必须改为 '*^'（如 6.12*^-17）。
    同时处理 inf / nan → WL 对应写法。
    """
    if isinstance(v, float):
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        if math.isnan(v):
            return "Indeterminate"
        s = repr(v)
        # e / E 只出现在科学计数法的指数位置，直接替换安全
        if 'e' in s or 'E' in s:
            s = s.replace('e', '*^').replace('E', '*^')
        return s
    return str(v)


def to_wl_list(value) -> str:
    """Python list / ndarray → Wolfram List {1,2,3}"""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        inner = ", ".join(to_wl_list(v) if isinstance(v, (list, tuple))
                          else _float_to_wl(v) if isinstance(v, float)
                          else str(v) for v in value)
        return "{" + inner + "}"
    return _float_to_wl(value) if isinstance(value, float) else str(value)

def to_wl_scalar(value) -> str:
    return _float_to_wl(value) if isinstance(value, float) else str(value)

def to_wl_matrix(value) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return "{" + ", ".join(to_wl_list(row) for row in value) + "}"

def to_wl_matrix_and_vector(value) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return to_wl_matrix(value[0]) + ", " + to_wl_list(value[1])
    raise ValueError(f"期望 (A, b) 元组，得到 {type(value)}")

def to_wl_string(value) -> str:
    return f'"{value}"'

def to_wl_passthrough(value) -> str:
    return str(value)


# ═══════════════════════════════════════════════════════════════
#  输出转换器
# ═══════════════════════════════════════════════════════════════

def _normalize(obj) -> Any:
    """
    递归将 wolframclient 返回的 WL 原生类型标准化为 Python 原生类型。

    转换策略基于一个清晰的边界：
      数据容器  → 转换为 Python 原生类型（有限的 4 种）
      符号表达式 → 原样保留（调用方本来就期望 WLFunction）

    数据容器映射：
      PackedArray / ndarray       → list        (.tolist())
      list / tuple                → list        (递归)
      WLFunction[List, ...]       → list        (Fourier/FFT 等返回此类型)
      WLFunction[Association,...] → dict        (关联数组)
      Rational / ExactReal        → float
      MachineComplex              → complex
      MachineInteger              → int

    符号表达式（保留）：
      WLFunction[Sin, ...], WLFunction[Rule, ...] 等
      → 调用方（如 sympy.diff / sympy.solve 包装层）会自行处理
    """
    # 1. PackedArray / numpy ndarray
    if hasattr(obj, "tolist"):
        result = obj.tolist()
        # wolframclient PackedArray 复数数组可能 .tolist() 返回 [[re,im],...]
        # 统一转为 Python complex 列表
        if (isinstance(result, list) and result
                and isinstance(result[0], (list, tuple))
                and len(result[0]) == 2
                and all(isinstance(x, (int, float)) for x in result[0])):
            # 看起来是 [re, im] 对，但也可能是 2 列矩阵 → 用 imag 部分判断
            # 保守起见：只有 dtype 含 complex 才转
            dtype_str = str(getattr(obj, "dtype", "")).lower()
            if "complex" in dtype_str:
                return [complex(float(r), float(i)) for r, i in result]
        return result

    # 2. Python 原生列表 / 元组（递归）
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]

    # 3. WLFunction：按 head 名称决定是数据容器还是符号表达式
    t = type(obj).__name__
    if "WLFunction" in t:
        head = getattr(obj, "head", None)
        head_name = getattr(head, "name", None) or str(head)
        args = getattr(obj, "args", ())

        if head_name == "List":
            # 特判：WLFunction[List, <单个 PackedArray>]
            #   wolframclient 有时把 Fourier/FFT 等结果包成
            #   List{ PackedArray{c1..cN} }，直接展平避免多一层嵌套
            if len(args) == 1 and hasattr(args[0], "tolist"):
                return _normalize(args[0])
            # 普通情况：递归展开每个元素
            return [_normalize(a) for a in args]

        if head_name == "Complex" and len(args) == 2:
            # WLFunction[Complex, re, im] → Python complex
            # wolframclient 在某些平台不自动把复数转 Python complex
            try:
                return complex(float(args[0]), float(args[1]))
            except (TypeError, ValueError):
                pass

        if head_name == "Association":
            # 关联数组：展开为 Python dict
            result = {}
            for item in args:
                item_head = getattr(getattr(item, "head", None), "name", None)
                if item_head == "Rule" and len(getattr(item, "args", ())) == 2:
                    k, v = item.args
                    result[_normalize(k)] = _normalize(v)
            return result if result else obj

        # 其他 WLFunction（Sin/D/Rule/Plus/...）→ 符号表达式，原样返回
        return obj

    # 4. WL 数值类型 → Python 数值
    if "Rational" in t or "ExactNumber" in t or "MachineReal" in t:
        try:    return float(obj)
        except: pass
    if "Complex" in t:
        try:    return complex(obj)
        except: pass
    if "Integer" in t and t != "int":
        try:    return int(obj)
        except: pass

    return obj


def from_wxf(result) -> Any:
    """
    WXF 文件路径或字节流 → Python 对象（wolframclient 反序列化 + 类型标准化）。

    标准化处理：
      PackedArray → Python list（修复 np.linalg.eig 返回 PackedArray 问题）
      Rational    → float（修复 Rational[5,2] 无法直接用于 Python 算术问题）
      空文件      → 抛出含诊断信息的 ValueError

    需要：pip install wolframclient
    """
    if not HAS_WOLFRAMCLIENT:
        raise ImportError(
            "from_wxf 需要 wolframclient。\n"
            "安装：pip install wolframclient"
        )
    if isinstance(result, (bytes, bytearray)):
        if not result:
            raise ValueError("from_wxf：收到空字节串——内核表达式可能存在语法错误")
        return _normalize(binary_deserialize(result))

    s = str(result).strip().strip('"')
    if os.path.exists(s):
        data = open(s, "rb").read()
        if not data:
            raise ValueError(
                f"from_wxf：WXF 文件为空：{s}\n"
                "可能原因：Wolfram 表达式语法错误、参数类型不匹配或内核超时。\n"
                "请检查对应规则的 wolfram_function 和 input_converter。"
            )
        return _normalize(binary_deserialize(data))

    raise ValueError(f"from_wxf：无效路径或字节流：{str(result)[:80]}")


def from_wl_image(wl_result: str) -> str:
    """
    将 WL Export 写出的 PNG 临时文件复制到持久临时路径后返回。

    必须复制：_proxy_base 的 finally 块会立刻删除原始 wlb_*.png，
    如果直接返回原路径，调用方拿到的是已删除文件的路径。
    复制后的文件由调用方负责清理，或使用后调用 os.unlink()。
    """
    s = wl_result.strip().strip('"')
    if not os.path.exists(s):
        raise FileNotFoundError(f"Wolfram 图像文件不存在：{s}")
    # 复制到持久临时文件（不使用 wlb_ 前缀，不会被自动删除）
    _, dst = tempfile.mkstemp(suffix=".png", prefix="wlfimg_")
    shutil.copy2(s, dst)
    return dst


# ═══════════════════════════════════════════════════════════════
#  注册表
# ═══════════════════════════════════════════════════════════════

INPUT_CONVERTERS = {
    "to_wl_list":               to_wl_list,
    "to_wl_scalar":             to_wl_scalar,
    "to_wl_matrix":             to_wl_matrix,
    "to_wl_matrix_and_vector":  to_wl_matrix_and_vector,
    "to_wl_string":             to_wl_string,
    "to_wl_passthrough":        to_wl_passthrough,
}


def from_wxf_dataframe(result) -> "WolframDataFrame":
    """
    WXF → WolframDataFrame。
    WL 返回的 Dataset/List 结构反序列化后包装为 WolframDataFrame。
    实际上 pandas 的 read_csv 等由 Python stdlib 直接处理，
    此转换器作为兜底，处理经内核返回的结构化数据。
    """
    raw = from_wxf(result)
    # 延迟导入避免循环
    from py2wl.compat.pandas import WolframDataFrame as WDF
    if isinstance(raw, WDF):
        return raw
    if isinstance(raw, (list, tuple)) and raw:
        if isinstance(raw[0], (list, tuple)):
            # 第一行作为列名
            cols = [str(c) for c in raw[0]]
            rows = [list(r) for r in raw[1:]]
            return WDF(cols, rows)
        # 平坦列表 → 单列
        return WDF(["value"], [[v] for v in raw])
    if isinstance(raw, dict):
        cols = list(raw.keys())
        n = max(len(v) for v in raw.values()) if raw else 0
        rows = [[list(raw[c])[i] if i < len(raw[c]) else None
                 for c in cols] for i in range(n)]
        return WDF(cols, rows)
    # 无法识别结构，包一层
    return WDF(["value"], [[raw]])

OUTPUT_CONVERTERS = {
    "from_wxf":       from_wxf,
    "from_wl_image":  from_wl_image,
    "from_wxf_dataframe": from_wxf_dataframe,
}


# ═══════════════════════════════════════════════════════════════
#  大数据文件传输层（Large-Data File Transfer）
# ═══════════════════════════════════════════════════════════════
#
#  设计：当 Python 序列包含超过 LARGE_THRESHOLD 个元素时，
#  改用 WXF 二进制文件传递而非在 PTY 管道中发送巨大字符串。
#
#  字符串模式（当前默认）：
#    Python list → to_wl_matrix() → "{{...19MB...}}"
#                                    ↓ PTY（5000 次 4KB 写入，慢）
#    kernel.evaluate('Export["out.wxf", Dot["{{...}}","{{...}}"], "WXF"]')
#
#  文件模式（本层，大数据自动触发）：
#    Python list → binary_serialize() → tmp_input.wxf（15ms 落盘）
#                                        ↓ PTY 只传一行 ~80 字节
#    kernel.evaluate('Export["out.wxf", N[Dot[Import["in.wxf","WXF"],
#                    Import["in2.wxf","WXF"]]], "WXF"]')
#
#  对 build_wl_expr / _WolframCallable 完全透明：
#    返回的字符串形如 'Import["path","WXF"]'，直接嵌入 WL 表达式。
#    tmp 文件路径通过全局注册表跟踪，由 build_wl_expr 的 finally 清理。

LARGE_THRESHOLD = 1_000   # 元素数超过此值走文件模式

# 待清理的临时输入文件（线程级注册表，key = 当前线程 id）
import threading
_pending_inputs: dict = {}   # {thread_id: [path1, path2, ...]}
_pending_lock = threading.Lock()

def _register_input_tmp(path: str):
    """登记待清理的输入临时文件。"""
    tid = threading.get_ident()
    with _pending_lock:
        _pending_inputs.setdefault(tid, []).append(path)

def flush_input_tmps():
    """
    清理本线程登记的所有临时输入文件。
    由 _call_with_fault 在 Step3/Step4 之后调用。
    """
    tid = threading.get_ident()
    with _pending_lock:
        paths = _pending_inputs.pop(tid, [])
    for p in paths:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except OSError:
            pass


def _count_elements(value) -> int:
    """递归统计嵌套列表/数组的总元素数（快速估算）。"""
    if hasattr(value, "size"):          # numpy array
        return int(value.size)
    if hasattr(value, "__len__"):
        n = len(value)
        if n == 0:
            return 0
        first = value[0] if not hasattr(value, "iloc") else None
        if first is not None and hasattr(first, "__len__") and not isinstance(first, str):
            # 二维：用第一行推断
            return n * _count_elements(first)
        return n
    return 1


def _to_wl_wxf_file(value) -> str:
    """
    将 Python 列表/数组序列化为 WXF 二进制文件，
    返回 WL Import 表达式字符串，Wolfram 端可直接使用结果。

    ── 核心设计 ────────────────────────────────────────────────────
    wolframclient 的 export() 对 numpy.ndarray 有专用编码路径：
      numpy array  → wl_export() → WXF PackedArray  ✓  BLAS 可用
      .tolist() 后 → wl.List()  → WXF List          ✗  通用算法，慢 100×

    因此：numpy array 必须直接传给 wl_export，绝不能先 .tolist()。
    对于纯 Python list/tuple，尝试先 numpy.asarray() 升级为数值数组；
    若无法升级（字符串、混合类型等），才退回 wl.List 路径。

    返回表达式带 Developer`ToPackedArray[N[...]] 双重包装：
      N[]                    — 将精确有理数/整数统一转为 MachinePrecision 浮点
      Developer`ToPackedArray — 强制连续内存布局（BLAS/LAPACK 必要条件）
    ────────────────────────────────────────────────────────────────
    """
    from wolframclient.serializers import export as wl_export
    from wolframclient.language import wl

    # ── 确定要序列化的对象 ────────────────────────────────────────
    if hasattr(value, "dtype"):
        # 已是 numpy array：直接序列化为 PackedArray
        serializable = value
    elif isinstance(value, (list, tuple)):
        # 纯 Python 序列：尝试升级为 numpy 数值数组
        try:
            import numpy as _np
            arr = _np.asarray(value)
            # 只有数值 dtype 才能编码为 PackedArray
            if arr.dtype.kind in ("f", "i", "u", "c"):
                serializable = arr
            else:
                serializable = _list_to_wl_expr(value, wl)
        except Exception:
            serializable = _list_to_wl_expr(value, wl)
    else:
        serializable = value

    # ── 写入临时 WXF 文件 ─────────────────────────────────────────
    fd, tmp_path = tempfile.mkstemp(suffix=".wxf", prefix="wlb_in_")
    try:
        with os.fdopen(fd, "wb") as f:
            wl_export(serializable, f, target_format="wxf")
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    _register_input_tmp(tmp_path)
    wl_path = tmp_path.replace("\\", "/")
    # Developer`ToPackedArray + N[] 保证 Wolfram 端得到连续 PackedArray
    return f'Developer`ToPackedArray[N[Import["{wl_path}", "WXF"]]]'


def _list_to_wl_expr(v, wl):
    """
    纯 Python list/tuple（含非数值元素）→ wolframclient wl.List 表达式。
    仅用于无法转 numpy 的兜底路径，产生 WXF List（非 PackedArray）。
    """
    if isinstance(v, (list, tuple)):
        return wl.List(*(_list_to_wl_expr(x, wl) for x in v))
    if isinstance(v, bool):
        return wl.True_ if v else wl.False_
    if isinstance(v, float):
        if math.isinf(v): return wl.Infinity if v > 0 else wl.DirectedInfinity(-1)
        if math.isnan(v): return wl.Indeterminate
    return v


def _large_aware(str_converter):
    """
    装饰器：为现有的字符串 input converter 添加大数据文件模式。

    当数据元素数超过 LARGE_THRESHOLD 时，自动走 WXF 文件路径；
    否则调用原始字符串转换器（零改动，完全向后兼容）。
    """
    def wrapper(value):
        if _count_elements(value) > LARGE_THRESHOLD:
            try:
                return _to_wl_wxf_file(value)
            except ImportError:
                log.debug("wolframclient 不可用，降级到字符串模式")
            except Exception as e:
                log.warning(f"WXF 文件序列化失败（{e}），降级到字符串模式")
        return str_converter(value)
    wrapper.__name__ = str_converter.__name__
    wrapper.__doc__  = (str_converter.__doc__ or "") + "  [large-data aware]"
    return wrapper


# 用文件感知版本替换大数据相关的 input converter
to_wl_list   = _large_aware(to_wl_list)
to_wl_matrix = _large_aware(to_wl_matrix)

# 更新注册表（已有引用不受影响，走注册表的路径会用新版本）
INPUT_CONVERTERS["to_wl_list"]   = to_wl_list
INPUT_CONVERTERS["to_wl_matrix"] = to_wl_matrix


def convert_input(value, converter_name: str) -> str:
    return INPUT_CONVERTERS.get(converter_name, to_wl_passthrough)(value)

def convert_output(wl_result: str, converter_name: str) -> Any:
    return OUTPUT_CONVERTERS.get(converter_name, from_wxf)(wl_result)

def register_input_converter(name: str, func):
    INPUT_CONVERTERS[name] = func

def register_output_converter(name: str, func):
    OUTPUT_CONVERTERS[name] = func
