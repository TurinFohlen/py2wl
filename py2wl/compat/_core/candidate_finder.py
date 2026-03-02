"""
compat/_core/candidate_finder.py
----------------------------------
候选函数查找器：在出错时，从元数据库 + AI 两路找出最相似的备选规则。

评分策略（综合得分 0-1）：
  1. 编辑距离（Levenshtein）相似度  ×0.4
  2. 标签 / 关键词匹配数              ×0.4
  3. AI 语义排名（可选）              ×0.2

返回按得分降序排列的 [(score, rule), ...]。
"""

from __future__ import annotations
import logging
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .metadata import MetadataRepository
    from .ai_plugin import AIPlugin

log = logging.getLogger("py2wl.compat")


# ── 轻量 Levenshtein（零依赖）──────────────────────────────────
def _lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                            prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _lev_sim(a: str, b: str) -> float:
    max_len = max(len(a), len(b), 1)
    return 1.0 - _lev(a, b) / max_len


# ── 路径分词（对点分路径拆成词组进行匹配）──────────────────────
def _tokens(path: str) -> set:
    """numpy.fft.fft  →  {'numpy', 'fft'}"""
    return set(path.replace("_", ".").split("."))


class CandidateFinder:
    """
    给定一个 python_path（可能拼错 / 不存在）和错误上下文，
    返回按相关度排序的备选规则列表。
    """

    def __init__(self, repo: "MetadataRepository",
                 ai_plugin: Optional["AIPlugin"] = None,
                 top_k: int = 6):
        self._repo      = repo
        self._ai        = ai_plugin
        self._top_k     = top_k

    def find(self, python_path: str,
             error_hint: str = "",
             args: tuple = (),
             kwargs: dict = None,
             use_ai: bool = True) -> List[Tuple[float, Dict]]:
        """
        返回 [(score, rule), ...]，按 score 降序。
        """
        kwargs = kwargs or {}
        scored: Dict[str, Tuple[float, Dict]] = {}

        target_tokens = _tokens(python_path)
        target_ns     = python_path.split(".")[0]   # e.g. "numpy"

        for rule in self._repo.all_rules:
            path   = rule["python_path"]
            rule_ns = path.split(".")[0]

            # ── 1. 编辑距离相似度 ────────────────────────────
            lev_sc = _lev_sim(python_path, path)

            # ── 2. 词集合 Jaccard 相似度 ──────────────────────
            rule_tokens = _tokens(path)
            union   = target_tokens | rule_tokens
            inter   = target_tokens & rule_tokens
            jac_sc  = len(inter) / len(union) if union else 0.0

            # ── 3. 命名空间加成（同一库优先）─────────────────
            ns_bonus = 0.15 if rule_ns == target_ns else 0.0

            # ── 4. 关键词 / 错误提示匹配 ─────────────────────
            desc  = (rule.get("description") or "").lower()
            tags  = " ".join(rule.get("tags") or []).lower()
            hint_words = set(error_hint.lower().split())
            hint_sc = sum(1 for w in hint_words if w in desc or w in tags) * 0.05

            score = lev_sc * 0.45 + jac_sc * 0.35 + ns_bonus + min(hint_sc, 0.15)
            if score > 0.05:  # 过滤完全无关的
                scored[path] = (score, rule)

        # 按分数降序取 top_k
        ranked = sorted(scored.values(), key=lambda x: -x[0])[: self._top_k]

        # ── AI 二次排名（可选）───────────────────────────────
        if use_ai and self._ai and ranked:
            ranked = self._ai_rerank(python_path, error_hint, ranked)

        return ranked

    # ── AI 重排：让 AI 从 top_k 候选里选最合适的并排序 ──────
    def _ai_rerank(self, python_path: str, error_hint: str,
                   candidates: List[Tuple[float, Dict]]) -> List[Tuple[float, Dict]]:
        try:
            candidate_lines = "\n".join(
                f"  [{i+1}] {r['python_path']} → {r['wolfram_function']}  ({r.get('description','')})"
                for i, (_, r) in enumerate(candidates)
            )
            prompt = (
                f"用户调用了 Python 函数 `{python_path}` 但失败了（{error_hint}）。\n"
                f"以下是按编辑距离预筛的候选映射：\n{candidate_lines}\n\n"
                f"请只输出编号列表（如 2,1,4），按'最可能是用户真正意图'的顺序排列，不要解释。"
            )
            resp = self._ai._ensure_provider() and self._ai._provider.generate(
                prompt, max_tokens=60)
            if not resp or not isinstance(resp, str):
                return candidates

            # 解析 "2,1,4" 格式
            import re
            nums = [int(x) - 1 for x in re.findall(r"\d+", resp)
                    if 0 < int(x) <= len(candidates)]
            if not nums:
                return candidates

            seen = set()
            reordered = []
            for n in nums:
                if n not in seen:
                    seen.add(n)
                    # 稍微抬高 AI 排到前面的分数
                    sc, rule = candidates[n]
                    boost = (len(nums) - len(reordered)) * 0.03
                    reordered.append((min(sc + boost, 1.0), rule))
            # 把未被 AI 提及的追加在后
            for i, (sc, rule) in enumerate(candidates):
                if i not in seen:
                    reordered.append((sc * 0.9, rule))
            return reordered

        except Exception as e:
            log.debug(f"AI 重排失败（无影响）：{e}")
            return candidates
