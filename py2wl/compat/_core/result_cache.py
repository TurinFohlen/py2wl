"""
result_cache.py — Dual-hash result cache
=========================================
Two metadata fields per entry:
  cmd_hash:    SHA-256 of the WL expression string
  result_hash: SHA-256 of pickle(result)

Lookup path (cmd_hash → result):
  O(1) dict lookup, blocks at serialization boundary so the kernel
  is never called for the same expression twice.

Dedup path (result_hash → entry):
  Different expressions that produce identical results share one
  CacheEntry — the second expression gets its cmd_hash registered
  to the existing entry, zero extra memory for the result object.

Cache rules:
  - All calls are cached by default.
  - Rule YAML can opt out with  cacheable: false
    (use for non-deterministic: RandomInteger, Now, SessionTime, …)
  - Capacity limited by maxsize (LRU eviction on oldest .created).
"""

import hashlib
import logging
import math
import pickle
import threading
import time
from typing import Any, Optional

log = logging.getLogger("py2wl.compat")

# 浮点数精度截断位数：对结果哈希前将浮点数保留 FLOAT_ROUND_DIGITS 位有效数字，
# 使得数值上相等但因浮点误差略有偏差的结果（如 2.5 vs 2.500000000000000000004）
# 产生相同的哈希值，从而共享缓存条目。
FLOAT_ROUND_DIGITS = 12


def _normalize_for_hash(obj: Any) -> Any:
    """
    递归地将结果中的浮点数截断到 FLOAT_ROUND_DIGITS 位，
    使得浮点微小误差不影响哈希值。
    支持 float / list / tuple / dict / numpy 数组。
    """
    if isinstance(obj, float):
        if math.isfinite(obj):
            return round(obj, FLOAT_ROUND_DIGITS)
        return obj  # nan / inf 直接保留
    if isinstance(obj, (list, tuple)):
        normalized = [_normalize_for_hash(v) for v in obj]
        return type(obj)(normalized)
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(v) for k, v in obj.items()}
    # numpy / pandas 等有 tolist() 的对象
    if hasattr(obj, "tolist"):
        try:
            return _normalize_for_hash(obj.tolist())
        except Exception:
            pass
    return obj


# ── Cache entry ─────────────────────────────────────────────────
class CacheEntry:
    __slots__ = ("cmd_hash", "result_hash", "result", "hits", "created")

    def __init__(self, cmd_hash: str, result_hash: str, result: Any):
        self.cmd_hash    = cmd_hash
        self.result_hash = result_hash
        self.result      = result
        self.hits        = 0
        self.created     = time.monotonic()


# ── Cache ────────────────────────────────────────────────────────
class ResultCache:
    def __init__(self, maxsize: int = 2048):
        self._by_cmd:    dict[str, CacheEntry] = {}
        self._by_result: dict[str, CacheEntry] = {}
        self._lock       = threading.Lock()
        self._maxsize    = maxsize
        self._hits       = 0
        self._misses     = 0

    # ── hash helpers ────────────────────────────────────────────

    @staticmethod
    def hash_expr(expr: str) -> str:
        return hashlib.sha256(expr.encode()).hexdigest()

    @staticmethod
    def hash_result(result: Any) -> str:
        """
        对结果计算哈希前先做浮点数精度截断（_normalize_for_hash），
        使整数列表和等值浮点数列表（如 Mean[{1,2,3,4}]=2.5）产生相同哈希，
        从而共享缓存条目，避免因浮点微小误差导致缓存失效。
        """
        normalized = _normalize_for_hash(result)
        try:
            data = pickle.dumps(normalized, protocol=4)
        except Exception:
            data = repr(normalized).encode()
        return hashlib.sha256(data).hexdigest()

    # ── public API ───────────────────────────────────────────────

    def get(self, cmd_hash: str) -> Optional[Any]:
        """Return cached result or None (miss)."""
        with self._lock:
            entry = self._by_cmd.get(cmd_hash)
            if entry is not None:
                entry.hits += 1
                self._hits  += 1
                log.debug(f"cache HIT  {cmd_hash[:12]}… (hits={entry.hits})")
                return entry.result
            self._misses += 1
            return None

    def put(self, cmd_hash: str, result: Any) -> None:
        """Store result; deduplicate against identical result objects."""
        result_hash = self.hash_result(result)

        with self._lock:
            # Dedup: same result already cached under a different expression?
            existing = self._by_result.get(result_hash)
            if existing is not None:
                self._by_cmd[cmd_hash] = existing
                log.debug(f"cache DEDUP {cmd_hash[:12]}… → {existing.cmd_hash[:12]}…")
                return

            # Evict oldest if at capacity
            if len(self._by_cmd) >= self._maxsize:
                oldest = min(self._by_cmd, key=lambda k: self._by_cmd[k].created)
                old = self._by_cmd.pop(oldest)
                self._by_result.pop(old.result_hash, None)
                log.debug(f"cache EVICT {oldest[:12]}…")

            entry = CacheEntry(cmd_hash, result_hash, result)
            self._by_cmd[cmd_hash]       = entry
            self._by_result[result_hash] = entry
            log.debug(f"cache STORE {cmd_hash[:12]}… result={result_hash[:12]}…")

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries":        len(self._by_cmd),
                "unique_results": len(self._by_result),
                "hits":           self._hits,
                "misses":         self._misses,
                "hit_rate":       round(self._hits / max(1, total), 3),
            }

    def clear(self) -> None:
        with self._lock:
            self._by_cmd.clear()
            self._by_result.clear()
            self._hits = self._misses = 0


# ── process-level singleton ──────────────────────────────────────
_cache = ResultCache()
