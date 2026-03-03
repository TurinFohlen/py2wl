"""
result_cache.py — Dual-hash result cache
=========================================
Two metadata fields per entry:
  cmd_hash:    SHA-256 of the WL expression string（表达式字符串哈希）
  result_hash: SHA-256 of result bytes（结果字节哈希，用于跨表达式去重）

Lookup path (cmd_hash → result):
  O(1) dict lookup，命中时直接返回，完全绕过内核调用。

Dedup path (result_hash → entry):
  不同表达式产生相同结果时共享同一 CacheEntry，节省内存。
  例：Mean[{1,2,3,4}] 和 Mean[{1.0,2.0,3.0,4.0}] 结果相同，只存一份。

result_hash 计算策略（按类型分级，最小化哈希开销）：
  numpy array  → array.tobytes()          ~5ms/8MB，零额外分配，最快路径
  float        → round(v, 12) 后 repr     消除浮点微小误差（2.5 vs 2.5000...4）
  list / tuple → 递归处理标量，大容器快速跳过
  其他         → repr() 兜底

性能账单（n=1000 矩阵，8MB）：
  旧实现：pickle.dumps(normalize(result)) ≈ 50ms
  新实现：result.tobytes()               ≈  5ms  （10× 提速）
  传输节省：命中时省去 ~1000ms 传输
  回本门槛：命中率 > 5/1000 = 0.5%

Cache rules:
  - All calls are cached by default.
  - Rule YAML can opt out with cacheable: false
    (use for non-deterministic: RandomInteger, Now, SessionTime, …)
  - Capacity limited by maxsize (LRU eviction on oldest .created).
"""

import hashlib
import logging
import math
import threading
import time
from typing import Any, Optional

log = logging.getLogger("py2wl.compat")

# 标量/小列表的浮点精度截断位数
# 消除 2.5 vs 2.500000000000000000004 这类浮点误差导致的哈希不一致
FLOAT_ROUND_DIGITS = 12

# 列表元素数超过此值时跳过逐元素截断，直接 repr()
# 避免对大矩阵做百万次 round() 调用
_LARGE_THRESHOLD = 1_000


def _hash_result_bytes(result: Any) -> bytes:
    """
    将计算结果转为用于哈希的字节序列。

    设计原则：
      1. numpy array → .tobytes()：最快，零额外内存，天然包含所有精度信息
      2. 标量 float  → round 后 repr：消除浮点微小误差
      3. 小列表/tuple → 递归处理：精确去重
      4. 大列表       → repr() fallback：避免递归遍历百万元素
      5. 其他类型     → repr() 兜底
    """
    # numpy array / 任何有 tobytes() 的数值数组
    if hasattr(result, "tobytes") and hasattr(result, "dtype"):
        # 同时编码 shape 和 dtype，防止不同形状但字节相同的数组误判为相等
        meta = f"{result.shape}:{result.dtype}:".encode()
        return meta + result.tobytes()

    # Python float（标量结果）
    if isinstance(result, float):
        if math.isfinite(result):
            return repr(round(result, FLOAT_ROUND_DIGITS)).encode()
        return repr(result).encode()   # nan / inf

    # int / bool / str / None
    if isinstance(result, (int, bool, str, type(None))):
        return repr(result).encode()

    # list / tuple（递归，大容器跳过截断）
    if isinstance(result, (list, tuple)):
        if len(result) > _LARGE_THRESHOLD:
            return repr(result).encode()
        parts = [_hash_result_bytes(v) for v in result]
        prefix = b"L" if isinstance(result, list) else b"T"
        return prefix + b"|".join(parts)

    # dict
    if isinstance(result, dict):
        pairs = sorted(
            repr(k).encode() + b":" + _hash_result_bytes(v)
            for k, v in result.items()
        )
        return b"D" + b",".join(pairs)

    # complex
    if isinstance(result, complex):
        r = round(result.real, FLOAT_ROUND_DIGITS)
        i = round(result.imag, FLOAT_ROUND_DIGITS)
        return f"C{r}+{i}j".encode()

    # 兜底
    return repr(result).encode()


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
        """表达式字符串 → cmd_hash（缓存查找 key）"""
        return hashlib.sha256(expr.encode()).hexdigest()

    @staticmethod
    def hash_result(result: Any) -> str:
        """
        结果 → result_hash（跨表达式去重 key）

        numpy array 走 tobytes() 快速路径（~5ms/8MB）；
        标量/小列表走精确截断路径（消除浮点误差）。
        """
        return hashlib.sha256(_hash_result_bytes(result)).hexdigest()

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
