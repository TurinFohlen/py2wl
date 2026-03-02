"""
compat/_core/resolver.py
------------------------
查找引擎：三级解析策略（Trie 精确 → 倒排模糊 → AI 兜底）

输出通道：
  - 所有表达式直接构造为 WL 字符串，由上层决定是直接 evaluate 还是用于 evaluate_to_file。
  - 不再包含 Export 包装。
"""

import logging
from typing import Optional, Dict, List

log = logging.getLogger("py2wl.compat")


class ResolutionEngine:
    _instance = None

    @classmethod
    def get_instance(cls, metadata_repo=None, ai_plugin=None):
        if cls._instance is None:
            if metadata_repo is None:
                raise RuntimeError("首次初始化必须传入 metadata_repo")
            cls._instance = cls(metadata_repo, ai_plugin)
        return cls._instance

    def __init__(self, metadata_repo, ai_plugin=None):
        self._repo      = metadata_repo
        self._ai_plugin = ai_plugin
        self._cache: Dict[str, Optional[Dict]] = {}

    def resolve(self, python_path: str, args=(), kwargs=None,
                use_ai: bool = True) -> Optional[Dict]:
        if python_path in self._cache:
            return self._cache[python_path]
        rule = self._repo.get_rule(python_path)
        if rule:
            self._cache[python_path] = rule
            return rule
        candidates = self._repo.search_rules(python_path)
        if candidates:
            return candidates[0]
        if use_ai and self._ai_plugin:
            suggestion = self._ai_suggest(python_path)
            if suggestion:
                return suggestion
        return None

    def search(self, query: str) -> List[Dict]:
        return self._repo.search_rules(query)

    def _ai_suggest(self, python_path: str) -> Optional[Dict]:
        try:
            suggestion = self._ai_plugin.suggest_mapping(python_path)
            if suggestion:
                return {
                    "python_path":      python_path,
                    "wolfram_function": suggestion.strip(),
                    "input_converter":  "to_wl_list",
                    "output_converter": "from_wxf",
                    "tags":             [],
                    "description":      f"AI 建议：{suggestion}",
                    "match_type":       "ai_suggestion",
                }
        except Exception as e:
            log.warning(f"AI 兜底失败：{e}")
        return None

    def set_ai_plugin(self, plugin):
        self._ai_plugin = plugin

    def clear_cache(self):
        self._cache.clear()

    def candidates_for_hint(self, python_path: str) -> List[Dict]:
        return self._repo.search_rules(python_path)

    def build_wl_expr(self, rule: Dict, args: tuple, kwargs: dict) -> str:
        """
        构造 Wolfram 核心表达式字符串，不包含任何 Export 包装。
        用于直接 evaluate 或作为 evaluate_to_file 的核心表达式。

        参数：
          - rule: 映射规则
          - args: 位置参数
          - kwargs: 关键字参数

        返回：
          Wolfram 表达式字符串，例如 "Mean[{1,2,3}]"
        """
        from . import converters as _cv

        wf = rule.get("wolfram_function", "Identity")
        ics_names = rule.get("input_converters")
        ic_name = rule.get("input_converter", "to_wl_list")

        # 输入转换
        wl_parts = []
        if ics_names:
            for arg, cn in zip(args, ics_names):
                conv = getattr(_cv, cn, _cv.to_wl_list)
                wl_parts.append(conv(arg))
            default_conv = getattr(_cv, ic_name, _cv.to_wl_list)
            for arg in args[len(ics_names):]:
                wl_parts.append(default_conv(arg))
        else:
            conv = getattr(_cv, ic_name, _cv.to_wl_list)
            if len(args) == 0:
                wl_parts = []
            elif len(args) == 1:
                wl_parts = [conv(args[0])]
            else:
                wl_parts = [conv(a) for a in args]

        if kwargs:
            kw_conv = getattr(_cv, ic_name, _cv.to_wl_list)
            for k, v in kwargs.items():
                wl_parts.append(f"{k} -> {kw_conv(v)}")

        core_expr = f"{wf}[{', '.join(wl_parts)}]" if wl_parts else wf

        # numeric: true → 强制数值近似
        if rule.get("numeric"):
            core_expr = f"N[{core_expr}]"

        return core_expr