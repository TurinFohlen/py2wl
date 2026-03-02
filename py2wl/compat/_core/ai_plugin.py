"""
compat/_core/ai_plugin.py
--------------------------
AI 插件：当 YAML 映射表找不到规则时，由 AI 给出 Wolfram 函数名建议；
也用于容错系统的候选重排（candidate_finder.py）。

环境变量：
  AI_PROVIDER          = deepseek | claude | gemini | groq  （默认 deepseek）
  DEEPSEEK_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY / GROQ_API_KEY
  WOLFRAM_AI_PLUGIN    = 1   （必须设置才启用，避免意外网络调用）
"""

import os
import importlib
import pkgutil
import logging
from typing import Optional

log = logging.getLogger("py2wl.compat")

_PROVIDERS_PKG = "py2wl.compat._core.ai_providers"
_HARDCODED_MAP = {
    "deepseek": "DeepSeekProvider",
    "claude":   "ClaudeProvider",
    "gemini":   "GeminiProvider",
    "groq":     "GroqProvider",
}
_ENV_KEY_MAP = {
    "deepseek": "DEEPSEEK_API_KEY",
    "claude":   "ANTHROPIC_API_KEY",
    "gemini":   "GOOGLE_API_KEY",
    "groq":     "GROQ_API_KEY",
}


class AIPlugin:
    def __init__(self, api_key: str = None, provider_name: str = None):
        self._name     = (provider_name or os.getenv("AI_PROVIDER", "deepseek")).lower()
        self._api_key  = api_key
        self._provider = None

    def _ensure_provider(self) -> bool:
        if self._provider is not None:
            return True

        pkg = importlib.import_module(_PROVIDERS_PKG)
        dynamic = {
            m: m.capitalize() + "Provider"
            for _, m, _ in pkgutil.iter_modules(pkg.__path__)
            if m not in ("__init__", "base")
        }
        full_map = {**dynamic, **_HARDCODED_MAP}

        if self._name not in full_map:
            log.warning(f"未知 AI 提供商：{self._name}，可用：{list(full_map)}")
            return False

        if not self._api_key:
            env = _ENV_KEY_MAP.get(self._name, self._name.upper() + "_API_KEY")
            self._api_key = os.getenv(env)
        if not self._api_key:
            log.warning(f"未找到 {self._name} API Key，AI 插件已禁用")
            return False

        try:
            mod = importlib.import_module(f"{_PROVIDERS_PKG}.{self._name}")
            self._provider = getattr(mod, full_map[self._name])(api_key=self._api_key)
            log.info(f"AI 插件就绪：{self._name} / {self._provider.model}")
            return True
        except Exception as e:
            log.warning(f"加载 AI 提供商失败：{e}")
            return False

    # ── 主功能 1：推断 Wolfram 函数名 ───────────────────────────
    def suggest_mapping(self, python_path: str,
                        context: str = "") -> Optional[str]:
        """
        给定 Python 函数路径，返回最可能对应的 Wolfram 函数名（仅函数名，无括号）。
        context 可附加参数示例或错误信息辅助判断。
        """
        if not self._ensure_provider():
            return None
        try:
            prompt = (
                f"你是 Wolfram Language 专家，熟悉 Python 科学计算生态（NumPy/SciPy/pandas/PyTorch 等）。\n"
                f"用户调用了 Python 函数：{python_path}\n"
                + (f"上下文：{context}\n" if context else "")
                + f"\n只回答对应的 Wolfram Language 内置函数名（如 Fourier、LinearSolve），"
                  f"不要括号，不要参数，不要解释，不要换行。"
            )
            result = self._provider.generate(prompt, max_tokens=40, temperature=0.1)
            # 取第一行，去掉可能残留的括号
            return result.splitlines()[0].split("[")[0].strip()
        except Exception as e:
            log.warning(f"AI 函数名建议失败：{e}")
            return None

    # ── 主功能 2：候选重排（供 candidate_finder 调用）───────────
    def rerank(self, python_path: str, error_hint: str,
               candidates: list) -> Optional[str]:
        """
        给定错误上下文和候选列表，返回"最合适候选"的编号序列，如 "2,1,4"。
        candidates: [(score, rule), ...]
        """
        if not self._ensure_provider():
            return None
        try:
            lines = "\n".join(
                f"  [{i+1}] {r['python_path']} → {r['wolfram_function']}"
                f"  （{r.get('description', '')}）"
                for i, (_, r) in enumerate(candidates)
            )
            prompt = (
                f"用户在 py2wl 中调用了 `{python_path}` 但失败了。\n"
                f"错误信息：{error_hint}\n\n"
                f"以下是按编辑距离预筛的 Python→Wolfram 候选映射：\n{lines}\n\n"
                f"请只输出编号序列（如 2,1,4），按'最可能是用户真正意图'排序，不要解释。"
            )
            return self._provider.generate(prompt, max_tokens=30, temperature=0.1)
        except Exception as e:
            log.warning(f"AI 重排失败：{e}")
            return None

    # ── 主功能 3：解释映射（供交互式 UI 调用）──────────────────
    def explain(self, python_path: str, rule: dict) -> Optional[str]:
        """向用户解释一条 Python→Wolfram 映射的语义和注意事项。"""
        if not self._ensure_provider():
            return None
        try:
            return self._provider.explain_mapping(python_path, rule)
        except Exception as e:
            log.warning(f"AI 映射解释失败：{e}")
            return None
