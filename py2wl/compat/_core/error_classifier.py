"""
compat/_core/error_classifier.py
---------------------------------
错误分类器：区分"可恢复"与"不可恢复"错误，并提取诊断信息。

可恢复（FaultKind.RECOVERABLE）：
  - 函数名找不到在 YAML 映射里（AttributeError，rule is None）
  - Wolfram 内核返回消息（$Failed、语法错误等）
  - 参数类型不匹配（TypeError、ValueError）

不可恢复（FaultKind.FATAL）：
  - 内核进程崩溃 / PTY 断开
  - Python 系统级错误（MemoryError、KeyboardInterrupt、SystemExit）
"""

from __future__ import annotations
import re
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


class FaultKind(Enum):
    RECOVERABLE = auto()   # 可容错处理
    FATAL       = auto()   # 必须终止


class RecoverableCategory(Enum):
    RULE_NOT_FOUND  = auto()   # YAML 里没这条路径
    KERNEL_EVAL     = auto()   # 内核执行出错（$Failed、语法错误）
    ARG_MISMATCH    = auto()   # 参数数量 / 类型不对
    CONVERTER_ERROR = auto()   # 输出转换器失败


@dataclass
class ErrorInfo:
    kind:      FaultKind
    category:  Optional[RecoverableCategory]
    original:  Exception
    python_path: str              # 调用路径，如 "numpy.fft.fft"
    args:      tuple  = field(default_factory=tuple)
    kwargs:    dict   = field(default_factory=dict)
    raw_wl:    Optional[str] = None   # 传给内核的 WL 表达式（如有）
    hint:      str = ""               # 可供用户阅读的一句话说明


# ── 内核返回的 WL 错误模式 ──────────────────────────────────────
_WL_FAIL_PATTERNS = [
    r"\$Failed",
    r"Syntax::sntx",
    r"::\w+",          # Message[tag::name]
    r"Error:",
]
_WL_FAIL_RE = re.compile("|".join(_WL_FAIL_PATTERNS))


def classify(exc: Exception, python_path: str,
             args=(), kwargs=None, raw_wl: str = None) -> ErrorInfo:
    """主入口：给一个 Exception 返回 ErrorInfo。"""
    kwargs = kwargs or {}

    # ── 不可恢复类型 ──────────────────────────────────────────
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError, RecursionError)):
        return ErrorInfo(FaultKind.FATAL, None, exc, python_path, args, kwargs, raw_wl,
                         f"系统级错误，无法容错：{type(exc).__name__}")

    if isinstance(exc, OSError) and "PTY" in str(exc):
        return ErrorInfo(FaultKind.FATAL, None, exc, python_path, args, kwargs, raw_wl,
                         "内核 PTY 管道断开，需要重启")

    # ── 可恢复：函数未找到 ────────────────────────────────────
    if isinstance(exc, AttributeError) and "未找到" in str(exc):
        return ErrorInfo(
            FaultKind.RECOVERABLE, RecoverableCategory.RULE_NOT_FOUND,
            exc, python_path, args, kwargs, raw_wl,
            f"函数 '{python_path}' 不在映射数据库中"
        )

    # ── 可恢复：参数不匹配 ────────────────────────────────────
    if isinstance(exc, (TypeError, ValueError)):
        return ErrorInfo(
            FaultKind.RECOVERABLE, RecoverableCategory.ARG_MISMATCH,
            exc, python_path, args, kwargs, raw_wl,
            f"参数类型/数量不匹配：{exc}"
        )

    # ── 可恢复：内核执行错误 ──────────────────────────────────
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        if raw_wl and _WL_FAIL_RE.search(raw_wl):
            cat = RecoverableCategory.KERNEL_EVAL
            hint = f"Wolfram 内核返回错误（表达式可能有误）"
        elif "内核执行失败" in msg:
            cat = RecoverableCategory.KERNEL_EVAL
            hint = f"内核执行失败：{msg[:120]}"
        else:
            cat = RecoverableCategory.CONVERTER_ERROR
            hint = f"输出转换错误：{msg[:120]}"
        return ErrorInfo(FaultKind.RECOVERABLE, cat, exc,
                         python_path, args, kwargs, raw_wl, hint)

    # ── 默认：可恢复（未知错误保守处理）────────────────────────
    return ErrorInfo(
        FaultKind.RECOVERABLE, RecoverableCategory.KERNEL_EVAL,
        exc, python_path, args, kwargs, raw_wl,
        f"未知错误（{type(exc).__name__}）：{str(exc)[:100]}"
    )
