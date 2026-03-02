"""
compat/_core/fault_handler.py
------------------------------
容错引擎：统一处理三种错误来源，根据策略决定下一步动作。

策略由环境变量 WOLFRAM_FAULT_MODE 控制：
  strict       默认，原样重抛异常
  auto-ai      AI 有唯一高置信候选时自动重试，否则降为 interactive
  interactive  总是暂停询问用户

返回 FaultAction，由 _WolframCallable 解读后执行。

会话级"纠错记忆"：
  _correction_cache[python_path] = rule
  同一路径第二次出错时直接复用上次用户的选择，不再询问。
"""

from __future__ import annotations

import os
import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING

from .error_classifier import ErrorInfo, FaultKind, classify
from .candidate_finder  import CandidateFinder
from .interactor        import ask, UserChoice, InteractorResult

if TYPE_CHECKING:
    from .metadata  import MetadataRepository
    from .ai_plugin import AIPlugin

log = logging.getLogger("py2wl.compat")


# ── 容错模式 ──────────────────────────────────────────────────
class FaultMode(Enum):
    STRICT      = "strict"
    AUTO_AI     = "auto-ai"
    INTERACTIVE = "interactive"

    @classmethod
    def from_env(cls) -> "FaultMode":
        val = os.environ.get("WOLFRAM_FAULT_MODE", "strict").lower()
        mapping = {m.value: m for m in cls}
        return mapping.get(val, cls.STRICT)


# ── 动作类型 ──────────────────────────────────────────────────
class ActionKind(Enum):
    RETRY_RULE   = auto()   # 用另一条 rule 重试
    RETRY_EXPR   = auto()   # 用自定义 WL 表达式重试
    SKIP         = auto()   # 返回 None，继续
    RAISE        = auto()   # 重抛原始异常


@dataclass
class FaultAction:
    kind:        ActionKind
    rule:        Optional[Dict] = None    # RETRY_RULE
    custom_expr: Optional[str] = None    # RETRY_EXPR


# ── AI 置信度阈值 ──────────────────────────────────────────────
_AI_AUTO_THRESHOLD = 0.75   # score > 此值且唯一时自动重试


class FaultHandler:
    """
    使用方式：

        handler = FaultHandler(repo, ai_plugin)
        action  = handler.handle(exc, python_path, args, kwargs, raw_wl)
        if action.kind == ActionKind.RETRY_RULE:
            ...
    """

    def __init__(self,
                 repo:      "MetadataRepository",
                 ai_plugin: Optional["AIPlugin"] = None,
                 mode:      Optional[FaultMode]  = None):
        self._mode   = mode or FaultMode.from_env()
        self._finder = CandidateFinder(repo, ai_plugin, top_k=6)
        # 会话级纠错记忆：{python_path: rule}
        self._correction_cache: Dict[str, Dict] = {}
        self._skip_cache: set = set()   # 用户选过"跳过"的路径

    @property
    def mode(self) -> FaultMode:
        return self._mode

    def set_mode(self, mode: FaultMode):
        self._mode = mode

    # ── 主入口 ────────────────────────────────────────────────
    def handle(self,
               exc:         Exception,
               python_path: str,
               args:        tuple = (),
               kwargs:      dict  = None,
               raw_wl:      str   = None) -> FaultAction:
        """
        给定异常，返回 FaultAction。
        调用方根据 action.kind 决定重试 / 跳过 / 重抛。
        """
        kwargs = kwargs or {}

        # 1. 分类错误
        ei = classify(exc, python_path, args, kwargs, raw_wl)
        log.debug(f"容错：{ei.kind.name} / {ei.category} / {ei.hint}")

        # 2. 不可恢复 → 直接重抛
        if ei.kind == FaultKind.FATAL:
            log.error(f"不可恢复错误：{ei.hint}")
            return FaultAction(ActionKind.RAISE)

        # 3. strict 模式 → 直接重抛
        if self._mode == FaultMode.STRICT:
            return FaultAction(ActionKind.RAISE)

        # 4. 查会话纠错记忆
        if python_path in self._skip_cache:
            log.debug(f"跳过缓存命中：{python_path}")
            return FaultAction(ActionKind.SKIP)
        if python_path in self._correction_cache:
            cached = self._correction_cache[python_path]
            log.info(f"纠错缓存命中：{python_path} → {cached['python_path']}")
            return FaultAction(ActionKind.RETRY_RULE, rule=cached)

        # 5. 找候选
        use_ai = self._mode in (FaultMode.AUTO_AI, FaultMode.INTERACTIVE)
        candidates = self._finder.find(
            python_path, error_hint=ei.hint,
            args=args, kwargs=kwargs, use_ai=use_ai)

        log.debug(f"候选数：{len(candidates)}，"
                  f"最高分：{candidates[0][0]:.2f}" if candidates else "无候选")

        # 6. auto-ai：高置信唯一候选 → 自动重试
        if self._mode == FaultMode.AUTO_AI and candidates:
            top_score, top_rule = candidates[0]
            second_score = candidates[1][0] if len(candidates) > 1 else 0.0
            # 唯一高置信（第一名显著高于第二名）
            if top_score >= _AI_AUTO_THRESHOLD and (top_score - second_score) > 0.15:
                log.info(
                    f"auto-ai 自动替换：{python_path} → "
                    f"{top_rule['python_path']} (score={top_score:.2f})")
                self._correction_cache[python_path] = top_rule
                return FaultAction(ActionKind.RETRY_RULE, rule=top_rule)
            # 置信不足 → 降为交互
            log.debug("auto-ai 置信不足，降为 interactive")

        # 7. interactive：询问用户
        result: InteractorResult = ask(ei, candidates)

        if result.choice == UserChoice.QUIT:
            return FaultAction(ActionKind.RAISE)

        if result.choice == UserChoice.SKIP:
            self._skip_cache.add(python_path)
            return FaultAction(ActionKind.SKIP)

        if result.choice == UserChoice.USE_CANDIDATE and result.rule:
            self._correction_cache[python_path] = result.rule
            return FaultAction(ActionKind.RETRY_RULE, rule=result.rule)

        if result.choice == UserChoice.USE_CUSTOM_EXPR and result.custom_expr:
            return FaultAction(ActionKind.RETRY_EXPR,
                               custom_expr=result.custom_expr)

        # 兜底
        return FaultAction(ActionKind.RAISE)

    # ── 工具方法 ──────────────────────────────────────────────
    def clear_cache(self):
        """清空会话纠错记忆（测试 / 重启用）。"""
        self._correction_cache.clear()
        self._skip_cache.clear()

    def correction_summary(self) -> List[Dict]:
        """返回本次会话已累计的所有自动纠错记录。"""
        return [
            {"original": orig, "corrected_to": rule["python_path"]}
            for orig, rule in self._correction_cache.items()
        ]
