"""
compat/_core/interactor.py
---------------------------
交互式控制台 UI：在出错时暂停，展示候选列表，等用户做决定。

用户可选的动作：
  [1..N]  选择候选规则重试
  [e]     手动输入 Wolfram 表达式直接执行
  [s]     跳过本次调用（返回 None）
  [q]     终止程序（re-raise 原始异常）

返回 InteractorResult，由 FaultHandler 解读。
"""

from __future__ import annotations

import sys
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any


class UserChoice(Enum):
    USE_CANDIDATE  = auto()   # 用候选规则重试
    USE_CUSTOM_EXPR = auto()  # 直接用用户写的 WL 表达式
    SKIP           = auto()   # 跳过，返回 None
    QUIT           = auto()   # 终止


@dataclass
class InteractorResult:
    choice:     UserChoice
    rule:       Optional[Dict] = None   # USE_CANDIDATE 时有效
    custom_expr: Optional[str] = None  # USE_CUSTOM_EXPR 时有效


# ANSI 颜色（仅在 tty 时启用）
def _is_tty() -> bool:
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def _c(code: str, text: str) -> str:
    if not _is_tty():
        return text
    return f"\033[{code}m{text}\033[0m"


BOLD   = lambda t: _c("1",    t)
RED    = lambda t: _c("1;31", t)
YELLOW = lambda t: _c("1;33", t)
CYAN   = lambda t: _c("1;36", t)
GREEN  = lambda t: _c("1;32", t)
DIM    = lambda t: _c("2",    t)


def ask(error_info, candidates: List[Tuple[float, Dict]]) -> InteractorResult:
    """
    暂停执行，在 stderr 上展示错误信息和候选，读取用户选择。
    error_info: ErrorInfo dataclass
    candidates: [(score, rule), ...]
    """
    ei = error_info
    sep = "─" * 62

    print(file=sys.stderr)
    print(YELLOW(f"⚠  容错系统介入"), file=sys.stderr)
    print(DIM(sep), file=sys.stderr)
    print(f"  函数路径 : {BOLD(ei.python_path)}", file=sys.stderr)
    print(f"  错误类别 : {ei.category.name if ei.category else 'UNKNOWN'}", file=sys.stderr)
    print(f"  错误信息 : {RED(ei.hint)}", file=sys.stderr)
    if ei.args:
        arg_repr = ", ".join(repr(a)[:40] for a in ei.args)
        print(f"  调用参数 : ({arg_repr})", file=sys.stderr)
    print(DIM(sep), file=sys.stderr)

    if candidates:
        print(CYAN("  候选函数（按相关度排序）："), file=sys.stderr)
        for i, (score, rule) in enumerate(candidates, 1):
            pct = f"{score*100:.0f}%"
            desc = rule.get("description", "")[:50]
            wf   = rule.get("wolfram_function", "?")
            print(
                f"  [{BOLD(str(i))}] {rule['python_path']:<32}"
                f"→ {GREEN(wf):<20}  {DIM(desc)}  {DIM(pct)}",
                file=sys.stderr)
    else:
        print(YELLOW("  ⚠ 未找到相似候选。"), file=sys.stderr)

    print(DIM(sep), file=sys.stderr)
    print(f"  [{BOLD('e')}] 手动输入 Wolfram 表达式  "
          f"[{BOLD('s')}] 跳过此调用  "
          f"[{BOLD('q')}] 终止程序", file=sys.stderr)
    print(DIM(sep), file=sys.stderr)

    while True:
        try:
            raw = input("  你的选择 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return InteractorResult(UserChoice.QUIT)

        if not raw:
            continue

        # 数字选候选
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(candidates):
                _, rule = candidates[idx]
                print(GREEN(f"  ✓ 已选：{rule['python_path']}"), file=sys.stderr)
                return InteractorResult(UserChoice.USE_CANDIDATE, rule=rule)
            else:
                print(RED(f"  超出范围（1-{len(candidates)}），请重新输入"), file=sys.stderr)
                continue

        if raw.lower() == "e":
            try:
                expr = input("  WL 表达式 > ").strip()
            except (EOFError, KeyboardInterrupt):
                return InteractorResult(UserChoice.QUIT)
            if expr:
                print(GREEN(f"  ✓ 将执行：{expr[:60]}"), file=sys.stderr)
                return InteractorResult(UserChoice.USE_CUSTOM_EXPR, custom_expr=expr)
            continue

        if raw.lower() == "s":
            print(YELLOW("  ⟳ 跳过，继续执行（返回 None）"), file=sys.stderr)
            return InteractorResult(UserChoice.SKIP)

        if raw.lower() == "q":
            print(RED("  ✗ 用户终止"), file=sys.stderr)
            return InteractorResult(UserChoice.QUIT)

        print(RED(f"  无效输入：{raw!r}"), file=sys.stderr)
