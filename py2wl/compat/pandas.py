"""
compat/pandas.py — 轻量级 Pandas 兼容层
========================================
设计原则：
  · WolframDataFrame 在 Python 侧持有数据（列表的列表 + 列名）
  · 简单操作（head/tail/sort/groupby 聚合）纯 Python 实现，零内核延迟
  · 复杂计算（merge/rolling/corr）委托给 WL 内核，通过 WXF 无损交换
  · read_csv 使用 Python stdlib，不过内核（解决原 Import[] 格式 bug）

用法：
    from py2wl.compat import pandas as pd
    df = pd.read_csv("/sdcard/data.csv")
    print(df.head())
    print(df.groupby("city").mean())
    df.to_csv("/sdcard/out.csv")
"""

import csv
import os
import tempfile
import logging
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _np = None
    _HAS_NP = False

log = logging.getLogger("py2wl.compat.pandas")


# ═══════════════════════════════════════════════════════
#  辅助：Python 值 → WL 字符串
# ═══════════════════════════════════════════════════════

def _to_wl(v) -> str:
    if v is None:           return "Missing[]"
    if isinstance(v, bool): return "True" if v else "False"
    if isinstance(v, int):  return str(v)
    if isinstance(v, float):return repr(v)
    s = str(v).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{s}"'

def _try_numeric(s: str):
    """尝试将字符串解析为数值，失败返回原字符串。"""
    try:   return int(s)
    except (ValueError, TypeError): pass
    try:   return float(s)
    except (ValueError, TypeError): pass
    return s


# ═══════════════════════════════════════════════════════
#  内核调用辅助（惰性，仅在需要时初始化）
# ═══════════════════════════════════════════════════════

def _kernel_eval_wxf(expr: str) -> Any:
    """执行 WL 表达式，写 WXF 临时文件，反序列化后返回。"""
    from ._proxy_base import _get_kernel
    from ._core.converters import _normalize
    from wolframclient.language import wlexpr
    return _normalize(_get_kernel().evaluate(wlexpr(expr)))

# ═══════════════════════════════════════════════════════
#  GroupByProxy
# ═══════════════════════════════════════════════════════

class _GroupByProxy:
    """groupby() 返回此对象，支持链式聚合调用。"""
    def __init__(self, df: "WolframDataFrame", by: Union[str, List[str]]):
        self._df = df
        self._by = [by] if isinstance(by, str) else by

    def _grouped(self):
        """分组：返回 {key_tuple: [rows]} 字典。"""
        by_idxs = [self._df._columns.index(b) for b in self._by]
        groups: Dict[tuple, list] = {}
        for row in self._df._rows:
            key = tuple(row[i] for i in by_idxs)
            groups.setdefault(key, []).append(row)
        return groups

    def _agg(self, func) -> "WolframDataFrame":
        groups = self._grouped()
        non_by = [c for c in self._df._columns if c not in self._by]
        non_by_idxs = [self._df._columns.index(c) for c in non_by]
        result_rows = []
        for key, rows in sorted(groups.items()):
            new_row = list(key)
            for i in non_by_idxs:
                vals = [r[i] for r in rows if isinstance(r[i], (int, float))]
                new_row.append(func(vals) if vals else None)
            result_rows.append(new_row)
        return WolframDataFrame(self._by + non_by, result_rows)

    def mean(self):   return self._agg(lambda vs: sum(vs) / len(vs))
    def sum(self):    return self._agg(sum)
    def min(self):    return self._agg(min)
    def max(self):    return self._agg(max)
    def count(self):  return self._agg(len)
    def first(self):  return self._agg(lambda vs: vs[0])
    def last(self):   return self._agg(lambda vs: vs[-1])

    def agg(self, func_or_dict):
        """支持 .agg("mean") 或 .agg({"col": "sum"})。"""
        FUNCS = {"mean": lambda vs: sum(vs)/len(vs), "sum": sum,
                 "min": min, "max": max, "count": len}
        if isinstance(func_or_dict, str):
            return self._agg(FUNCS.get(func_or_dict, lambda vs: None))
        raise NotImplementedError("agg(dict) 暂不支持")


# ═══════════════════════════════════════════════════════
#  RollingProxy（委托给 WL MovingMap）
# ═══════════════════════════════════════════════════════

class _RollingProxy:
    def __init__(self, df: "WolframDataFrame", window: int):
        self._df    = df
        self._win   = window

    def _wl_moving(self, wl_func: str) -> "WolframDataFrame":
        """用 WL MovingMap[func, list, window] 计算滚动聚合。"""
        non_key = self._df._columns
        result_cols = {}
        for i, col in enumerate(non_key):
            vals = [r[i] for r in self._df._rows if isinstance(r[i], (int, float))]
            wl_list = "{" + ", ".join(str(v) for v in vals) + "}"
            try:
                raw = _kernel_eval_wxf(f"MovingMap[{wl_func}, {wl_list}, {self._win}]")
                result_cols[col] = raw if isinstance(raw, list) else [raw]
            except Exception as e:
                log.warning(f"rolling.{wl_func} {col} 失败：{e}")
                result_cols[col] = [None] * len(vals)
        # 对齐长度（MovingMap 会截断）
        min_len = min(len(v) for v in result_cols.values()) if result_cols else 0
        rows = [[result_cols[c][i] for c in non_key]
                for i in range(min_len)]
        return WolframDataFrame(non_key, rows)

    def mean(self): return self._wl_moving("Mean")
    def sum(self):  return self._wl_moving("Total")
    def min(self):  return self._wl_moving("Min")
    def max(self):  return self._wl_moving("Max")


# ═══════════════════════════════════════════════════════
#  WolframDataFrame（核心类）
# ═══════════════════════════════════════════════════════

class WolframDataFrame:
    """
    轻量级 DataFrame，内部持有：
      _columns : List[str]       列名
      _rows    : List[List]      行数据（行优先）

    纯 Python 操作直接在内存上运行，复杂计算委托 WL 内核。
    """

    def __init__(self, columns: List[str], rows: List[List]):
        self._columns = list(columns)
        self._rows    = [list(r) for r in rows]

    # ── 属性 ───────────────────────────────────────────
    @property
    def columns(self) -> List[str]:  return list(self._columns)
    @property
    def shape(self):     return (len(self._rows), len(self._columns))
    @property
    def dtypes(self) -> Dict[str, str]:
        result = {}
        for i, col in enumerate(self._columns):
            vals = [r[i] for r in self._rows if r[i] is not None]
            if not vals:
                result[col] = "object"
            elif all(isinstance(v, bool) for v in vals):
                result[col] = "bool"
            elif all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
                result[col] = "int"
            elif all(isinstance(v, float) for v in vals):
                result[col] = "float"
            elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                result[col] = "float"
            elif all(isinstance(v, str) for v in vals):
                result[col] = "str"
            else:
                result[col] = "object"
        return result

    def __len__(self): return len(self._rows)

    # ── 表示 ───────────────────────────────────────────
    def __repr__(self) -> str:
        if not self._rows:
            return f"Empty WolframDataFrame  columns={self._columns}"
        cols   = self._columns
        widths = [max(len(str(c)), max((len(str(r[i])) for r in self._rows), default=0))
                  for i, c in enumerate(cols)]
        sep    = "  "
        header = sep.join(str(c).ljust(w) for c, w in zip(cols, widths))
        lines  = [header, "-" * len(header)]
        for ri, row in enumerate(self._rows[:20]):
            lines.append(sep.join(str(v).ljust(w) for v, w in zip(row, widths)))
        if len(self._rows) > 20:
            lines.append(f"... ({len(self._rows)} rows total)")
        lines.append(f"\n[{len(self._rows)} rows × {len(self._columns)} columns]")
        return "\n".join(lines)

    # ── 索引 / 列选择 ──────────────────────────────────
    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._columns:
                raise KeyError(f"列 '{key}' 不存在。现有列：{self._columns}")
            idx = self._columns.index(key)
            return [row[idx] for row in self._rows]
        if isinstance(key, list):
            idxs = [self._columns.index(k) for k in key]
            return WolframDataFrame(key, [[r[i] for i in idxs] for r in self._rows])
        if isinstance(key, int):
            if key < 0:
                key = len(self._rows) + key
            if not (0 <= key < len(self._rows)):
                raise IndexError(f"行索引越界：{key}，共 {len(self._rows)} 行")
            return dict(zip(self._columns, self._rows[key]))
        if isinstance(key, slice):
            return WolframDataFrame(list(self._columns), self._rows[key])
        raise TypeError(f"不支持的索引类型：{type(key)}")

    def __setitem__(self, key: str, values):
        if key in self._columns:
            idx = self._columns.index(key)
            for row, v in zip(self._rows, values):
                row[idx] = v
        else:
            self._columns.append(key)
            for row, v in zip(self._rows, values):
                row.append(v)

    @property
    def loc(self):
        return _LocIndexer(self)

    @property
    def iloc(self):
        return _ILocIndexer(self)

    # ── 纯 Python 操作 ─────────────────────────────────
    def head(self, n: int = 5) -> "WolframDataFrame":
        return WolframDataFrame(self._columns, self._rows[:n])

    def tail(self, n: int = 5) -> "WolframDataFrame":
        return WolframDataFrame(self._columns, self._rows[-n:])

    def copy(self) -> "WolframDataFrame":
        return WolframDataFrame(self._columns, self._rows)

    def rename(self, columns: Dict[str, str] = None, **kwargs) -> "WolframDataFrame":
        mapping = columns or kwargs
        new_cols = [mapping.get(c, c) for c in self._columns]
        return WolframDataFrame(new_cols, self._rows)

    def isna(self, col: str = None):
        """返回缺失值布尔掩码。col 指定列名，None 则检查每行。"""
        def _missing(v): return v is None or v == "" or (isinstance(v, float) and v != v)
        if col is not None:
            idx = self._columns.index(col)
            return [_missing(row[idx]) for row in self._rows]
        return [any(_missing(v) for v in row) for row in self._rows]

    def notna(self, col: str = None):
        return [not v for v in self.isna(col)]

    def dropna(self, axis=0, how="any") -> "WolframDataFrame":
        def _is_missing(v): return v is None or v == "" or v != v  # NaN check
        if axis == 0:
            if how == "any":
                rows = [r for r in self._rows if not any(_is_missing(v) for v in r)]
            else:
                rows = [r for r in self._rows if not all(_is_missing(v) for v in r)]
            return WolframDataFrame(self._columns, rows)
        raise NotImplementedError("axis=1 暂不支持")

    def fillna(self, value) -> "WolframDataFrame":
        def _fill(v): return value if (v is None or v == "") else v
        return WolframDataFrame(self._columns,
                                [[_fill(v) for v in r] for r in self._rows])

    def sort_values(self, by: Union[str, List[str]],
                    ascending: Union[bool, List[bool]] = True) -> "WolframDataFrame":
        keys = [by] if isinstance(by, str) else by
        ascs = [ascending] * len(keys) if isinstance(ascending, bool) else ascending
        idxs = [self._columns.index(k) for k in keys]

        def sort_key(row):
            return tuple(row[i] if row[i] is not None else "" for i in idxs)

        # numpy argsort：单列全数值时走向量化路径（稳定、快速）
        if _HAS_NP and len(keys) == 1:
            col_vals = [r[idxs[0]] for r in self._rows]
            if all(isinstance(v, (int, float)) for v in col_vals):
                order = _np.argsort(col_vals, stable=True)
                if not ascs[0]:
                    order = order[::-1]
                return WolframDataFrame(list(self._columns),
                                        [self._rows[int(i)] for i in order])

        rows = sorted(self._rows, key=sort_key,
                      reverse=(not all(ascs)))
        return WolframDataFrame(self._columns, rows)

    def reset_index(self, drop: bool = False) -> "WolframDataFrame":
        return self.copy()

    def unique(self, col: str) -> List:
        idx = self._columns.index(col)
        seen, out = set(), []
        for row in self._rows:
            v = row[idx]
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def value_counts(self, col: str = None) -> Dict:
        if col is None and len(self._columns) == 1:
            col = self._columns[0]
        if col is None:
            raise ValueError("请指定列名")
        idx = self._columns.index(col)
        counts: Dict = {}
        for row in self._rows:
            v = row[idx]
            counts[v] = counts.get(v, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def query(self, expr_str: str) -> "WolframDataFrame":
        """Query rows. Supports single and compound conditions (and/or).
        Examples: 'score > 85', 'score >= 88 and city == "BJ"'
        """
        import re

        OPS = {">": lambda a, b: a > b,  "<": lambda a, b: a < b,
               ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
               "==": lambda a, b: a == b, "!=": lambda a, b: a != b}

        ATOM_RE = re.compile(r"(\w+)\s*([><=!]+)\s*(['\"]?)(.+?)\\3\s*$")

        def _eval_atom(part, row):
            m2 = re.match(
                r"(\w+)\s*([><=!]+)\s*(?:['\"]?)(.+?)(?:['\"]?)\s*$",
                part.strip()
            )
            if not m2:
                raise ValueError(f"Cannot parse: {part!r}")
            col, op, val_str = m2.group(1), m2.group(2), m2.group(3)
            if col not in self._columns:
                raise KeyError(f"Column '{col}' not found")
            cell = row[self._columns.index(col)]
            val  = _try_numeric(val_str)
            fn   = OPS.get(op)
            if fn is None:
                raise ValueError(f"Unsupported operator: {op}")
            if isinstance(cell, str) and not isinstance(val, str):
                val = str(val)
            elif isinstance(val, str) and not isinstance(cell, str) and cell is not None:
                try:    val = type(cell)(val)
                except (ValueError, TypeError): pass
            return fn(cell, val)

        def _eval_row(expr, row):
            or_parts = re.split(r"\s+or\s+", expr, flags=re.IGNORECASE)
            if len(or_parts) > 1:
                return any(_eval_row(p, row) for p in or_parts)
            and_parts = re.split(r"\s+and\s+", expr, flags=re.IGNORECASE)
            return all(_eval_atom(p, row) for p in and_parts)

        # numpy 向量化路径：纯 and 链 + 全列数值时走快速掩码
        if _HAS_NP and self._rows:
            and_parts = re.split(r"\s+and\s+", expr_str, flags=re.IGNORECASE)
            or_parts  = re.split(r"\s+or\s+",  expr_str, flags=re.IGNORECASE)
            NP_OPS = {">":  lambda a,b: a>b,  "<":  lambda a,b: a<b,
                      ">=": lambda a,b: a>=b, "<=": lambda a,b: a<=b,
                      "==": lambda a,b: a==b, "!=": lambda a,b: a!=b}
            try:
                # 解析所有原子条件
                atoms = and_parts if len(or_parts) == 1 else or_parts
                logic = "and" if len(or_parts) == 1 else "or"
                parsed = []
                for part in atoms:
                    m2 = re.match(
                        r"(\w+)\s*([><=!]+)\s*(?:['\"\']?)(.+?)(?:['\"\']?)\s*$",
                        part.strip())
                    if not m2: raise ValueError("skip numpy path")
                    col, op_s, vs = m2.group(1), m2.group(2), m2.group(3)
                    if col not in self._columns: raise ValueError("skip numpy path")
                    parsed.append((self._columns.index(col), op_s, _try_numeric(vs)))
                # 提取列数组
                arrs = {idx: _np.array([r[idx] for r in self._rows])
                        for idx, _, _ in parsed}
                # 计算掩码
                mask = None
                for idx, op_s, val in parsed:
                    mi = NP_OPS[op_s](arrs[idx], val)
                    if mask is None:        mask = mi
                    elif logic == "or":     mask = mask | mi
                    else:                   mask = mask & mi
                rows = [r for r, keep in zip(self._rows, mask) if keep]
                return WolframDataFrame(list(self._columns), rows)
            except (TypeError, ValueError, KeyError):
                pass  # 降级到逐行 Python 路径

        rows = [r for r in self._rows if _eval_row(expr_str, r)]
        return WolframDataFrame(list(self._columns), rows)

    # ── 聚合（纯 Python）──────────────────────────────
    def mean(self) -> Dict[str, float]:
        result = {}
        for i, col in enumerate(self._columns):
            vals = [r[i] for r in self._rows if isinstance(r[i], (int, float))]
            result[col] = sum(vals) / len(vals) if vals else None
        return result

    def sum(self) -> Dict:
        result = {}
        for i, col in enumerate(self._columns):
            vals = [r[i] for r in self._rows if isinstance(r[i], (int, float))]
            result[col] = sum(vals) if vals else None
        return result

    def min(self) -> Dict:
        result = {}
        for i, col in enumerate(self._columns):
            vals = [r[i] for r in self._rows if r[i] is not None]
            result[col] = min(vals) if vals else None
        return result

    def max(self) -> Dict:
        result = {}
        for i, col in enumerate(self._columns):
            vals = [r[i] for r in self._rows if r[i] is not None]
            result[col] = max(vals) if vals else None
        return result

    def std(self) -> Dict[str, float]:
        result = {}
        for i, col in enumerate(self._columns):
            vals = [r[i] for r in self._rows if isinstance(r[i], (int, float))]
            if len(vals) < 2:
                result[col] = None
                continue
            if _HAS_NP:
                result[col] = float(_np.std(vals, ddof=1))
            else:
                import math
                m = sum(vals) / len(vals)
                result[col] = math.sqrt(sum((v-m)**2 for v in vals)/(len(vals)-1))
        return result

    def describe(self) -> "WolframDataFrame":
        import math
        stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        rows = []
        for stat in stats:
            row = [stat]
            for i, col in enumerate(self._columns):
                vals = sorted(v for r in self._rows
                              for v in [r[i]] if isinstance(v, (int, float)))
                if not vals:
                    row.append(None); continue
                n = len(vals)
                if   stat == "count": row.append(n)
                elif stat == "mean":  row.append(sum(vals) / n)
                elif stat == "std":
                    m = sum(vals) / n
                    row.append(math.sqrt(sum((v-m)**2 for v in vals) / max(n-1, 1)))
                elif stat == "min":   row.append(vals[0])
                elif stat == "25%":   row.append(vals[int(n*0.25)])
                elif stat == "50%":   row.append(vals[int(n*0.5)])
                elif stat == "75%":   row.append(vals[int(n*0.75)])
                elif stat == "max":   row.append(vals[-1])
            rows.append(row)
        return WolframDataFrame(["stat"] + self._columns, rows)

    def groupby(self, by: Union[str, List[str]]) -> _GroupByProxy:
        return _GroupByProxy(self, by)

    def rolling(self, window: int) -> _RollingProxy:
        return _RollingProxy(self, window)

    # ── IO ────────────────────────────────────────────
    def to_csv(self, path: str, index: bool = False) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self._columns)
            w.writerows(self._rows)
        log.info(f"已写入 {path}  ({len(self._rows)} 行)")

    def to_dict(self, orient: str = "records") -> Any:
        if orient == "records":
            return [dict(zip(self._columns, r)) for r in self._rows]
        if orient == "list":
            return {c: [r[i] for r in self._rows]
                    for i, c in enumerate(self._columns)}
        raise ValueError(f"不支持的 orient: {orient!r}")

    # ── WL 序列化 / 高级操作 ───────────────────────────
    def to_wl_dataset(self) -> str:
        """序列化为 WL Dataset[{<|...|>, ...}] 表达式字符串。"""
        rows = []
        for row in self._rows:
            pairs = ", ".join(
                f'"{c}" -> {_to_wl(v)}'
                for c, v in zip(self._columns, row)
            )
            rows.append(f"<|{pairs}|>")
        return "Dataset[{" + ", ".join(rows) + "}]"

    def corr(self) -> "WolframDataFrame":
        """相关矩阵，委托给 WL Correlation。"""
        numeric_cols = [c for c, t in self.dtypes.items() if "float" in t or "int" in t]
        if len(numeric_cols) < 2:
            raise ValueError("相关矩阵需要至少两列数值数据")
        idxs   = [self._columns.index(c) for c in numeric_cols]
        matrix = [[row[i] for i in idxs] for row in self._rows]
        wl_mat = "{" + ", ".join(
            "{" + ", ".join(str(v) for v in row) + "}"
            for row in matrix
        ) + "}"
        raw = _kernel_eval_wxf(f"Correlation[{wl_mat}]")
        if isinstance(raw, (list, tuple)):
            return WolframDataFrame(numeric_cols, raw)
        return raw

    def merge(self, right: "WolframDataFrame",
              on: Union[str, List[str]] = None,
              how: str = "inner") -> "WolframDataFrame":
        """合并两个 DataFrame（纯 Python 实现，支持 inner/left/outer）。"""
        return merge(self, right, on=on, how=how)

    def apply(self, wl_func: str, axis: int = 0) -> "WolframDataFrame":
        """
        对每列（axis=0）或每行（axis=1）应用 WL 函数字符串。
        例如：df.apply("Mean") 相当于 df.mean()，但走内核。
        """
        if axis == 0:
            result_cols, result_rows = [], [[]]
            for i, col in enumerate(self._columns):
                vals = [r[i] for r in self._rows]
                wl_list = "{" + ", ".join(_to_wl(v) for v in vals) + "}"
                raw = _kernel_eval_wxf(f"{wl_func}[{wl_list}]")
                result_cols.append(col)
                result_rows[0].append(raw)
            return WolframDataFrame(result_cols, result_rows)
        raise NotImplementedError("axis=1 的 apply 暂不支持")


# ═══════════════════════════════════════════════════════
#  loc / iloc 索引器
# ═══════════════════════════════════════════════════════

class _LocIndexer:
    def __init__(self, df: WolframDataFrame):
        self._df = df

    def __getitem__(self, key):
        df = self._df
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        # 行
        if isinstance(row_key, slice):
            rows = df._rows[row_key]
        elif isinstance(row_key, list):
            rows = [df._rows[i] for i in row_key]
        else:
            rows = [df._rows[row_key]]
        # 列
        if isinstance(col_key, slice):
            cols   = df._columns[col_key]
            idxs   = list(range(*col_key.indices(len(df._columns))))
            rows   = [[r[i] for i in idxs] for r in rows]
        elif isinstance(col_key, list):
            idxs   = [df._columns.index(c) for c in col_key]
            cols   = col_key
            rows   = [[r[i] for i in idxs] for r in rows]
        elif col_key == slice(None):
            cols   = df._columns
        else:
            idx  = df._columns.index(col_key)
            vals = [r[idx] for r in rows]
            return vals[0] if len(vals) == 1 else vals
        return WolframDataFrame(cols, rows)


class _ILocIndexer:
    def __init__(self, df: WolframDataFrame):
        self._df = df

    def __getitem__(self, key):
        df = self._df
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        # 行：整数 → 单行（保持为列表以便列选择），切片/列表 → 多行
        single_row = isinstance(row_key, int)
        if isinstance(row_key, slice):
            rows = df._rows[row_key]
        elif isinstance(row_key, list):
            rows = [df._rows[i] for i in row_key]
        else:
            rows = [df._rows[row_key]]
        # 列
        if isinstance(col_key, int):
            vals = [r[col_key] for r in rows]
            return vals[0] if single_row else vals
        if isinstance(col_key, slice):
            idxs = list(range(*col_key.indices(len(df._columns))))
            cols = [df._columns[i] for i in idxs]
            rows = [[r[i] for i in idxs] for r in rows]
        else:
            cols = df._columns
        return WolframDataFrame(cols, rows)


# ═══════════════════════════════════════════════════════
#  顶层函数
# ═══════════════════════════════════════════════════════

def read_csv(filepath_or_buffer: str,
             sep: str = ",",
             header: Union[int, str] = "infer",
             names: List[str] = None,
             index_col=None,
             dtype=None,
             encoding: str = "utf-8",
             skiprows: int = 0,
             nrows: int = None,
             **kwargs) -> WolframDataFrame:
    """
    使用 Python stdlib csv 模块读取文件，零内核依赖。
    支持：sep、header、names、encoding、skiprows、nrows。
    """
    with open(filepath_or_buffer, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=sep)
        all_rows = list(reader)

    if skiprows:
        all_rows = all_rows[skiprows:]

    if not all_rows:
        return WolframDataFrame(names or [], [])

    # 确定列名
    if names is not None:
        columns = list(names)
        data_rows = all_rows
    elif header == 0 or header == "infer":
        columns   = all_rows[0]
        data_rows = all_rows[1:]
    else:
        columns   = [f"col_{i}" for i in range(len(all_rows[0]))]
        data_rows = all_rows

    if nrows is not None:
        data_rows = data_rows[:nrows]

    # 解析数值
    parsed = []
    for row in data_rows:
        # 对齐列数（补 None 或截断）
        row = row + [""] * max(0, len(columns) - len(row))
        row = row[:len(columns)]
        parsed.append([_try_numeric(v.strip()) for v in row])

    return WolframDataFrame(columns, parsed)


def read_excel(filepath: str, sheet_name=0,
               header: int = 0, **kwargs) -> WolframDataFrame:
    """通过 WL Import 读取 Excel（需要 WolframEngine 支持）。"""
    raw = _kernel_eval_wxf(f'Import["{filepath}", "XLSX"]')
    if isinstance(raw, (list, tuple)) and raw:
        rows  = list(raw)
        cols  = [str(c) for c in rows[0]]
        data  = [[_try_numeric(str(v)) for v in r] for r in rows[1:]]
        return WolframDataFrame(cols, data)
    raise ValueError(f"read_excel：无法解析 {filepath}")


def DataFrame(data=None, columns: List[str] = None,
              index=None, dtype=None) -> WolframDataFrame:
    """
    从 dict / list-of-dicts / list-of-lists 创建 WolframDataFrame。
    """
    if data is None:
        return WolframDataFrame(columns or [], [])

    if isinstance(data, dict):
        cols = list(data.keys())
        iters = [list(v) for v in data.values()]
        n    = max(len(v) for v in iters) if iters else 0
        rows = [[iters[j][i] if i < len(iters[j]) else None
                 for j in range(len(cols))]
                for i in range(n)]
        return WolframDataFrame(cols, rows)

    if isinstance(data, list):
        if not data:
            return WolframDataFrame(columns or [], [])
        if isinstance(data[0], dict):
            cols = columns or list(data[0].keys())
            rows = [[r.get(c) for c in cols] for r in data]
            return WolframDataFrame(cols, rows)
        # list-of-lists
        if columns is None:
            columns = [f"col_{i}" for i in range(len(data[0]))]
        return WolframDataFrame(columns, data)

    raise TypeError(f"不支持的 data 类型：{type(data)}")


def merge(left: WolframDataFrame, right: WolframDataFrame,
          on: Union[str, List[str]] = None,
          left_on: str = None, right_on: str = None,
          how: str = "inner") -> WolframDataFrame:
    """
    合并两个 WolframDataFrame（纯 Python，支持 inner/left/right/outer）。
    """
    if on is None and left_on is None:
        # 自动找同名列
        common = [c for c in left._columns if c in right._columns]
        if not common:
            raise ValueError("merge：找不到公共列，请指定 on=")
        on = common[0]

    lk = on if isinstance(on, str) else left_on
    rk = on if isinstance(on, str) else right_on

    li = left._columns.index(lk)
    ri = right._columns.index(rk)

    # 右侧建索引
    r_idx: Dict = {}
    for row in right._rows:
        r_idx.setdefault(row[ri], []).append(row)

    # 合并列名（避免重复，右侧非 key 列加 _right 后缀）
    r_extra = [c for c in right._columns if c != rk]
    new_cols = left._columns + [f"{c}_right" if c in left._columns else c
                                 for c in r_extra]
    r_extra_idxs = [right._columns.index(c) for c in r_extra]

    rows = []
    matched_rkeys = set()

    for lrow in left._rows:
        key = lrow[li]
        rrows = r_idx.get(key, [])
        if rrows:
            matched_rkeys.add(key)
            for rrow in rrows:
                rows.append(lrow + [rrow[j] for j in r_extra_idxs])
        elif how in ("left", "outer"):
            rows.append(lrow + [None] * len(r_extra_idxs))

    if how in ("right", "outer"):
        for rrow in right._rows:
            if rrow[ri] not in matched_rkeys:
                rows.append([None] * len(left._columns) + [rrow[j] for j in r_extra_idxs])

    return WolframDataFrame(new_cols, rows)


def concat(objs: List[WolframDataFrame],
           axis: int = 0,
           ignore_index: bool = False) -> WolframDataFrame:
    """沿 axis=0（行拼接）或 axis=1（列拼接）合并多个 DataFrame。"""
    if not objs:
        return WolframDataFrame([], [])

    if axis == 0:
        # 取所有列的并集
        all_cols: List[str] = []
        for df in objs:
            for c in df._columns:
                if c not in all_cols:
                    all_cols.append(c)
        rows = []
        for df in objs:
            for row in df._rows:
                new_row = [row[df._columns.index(c)] if c in df._columns else None
                           for c in all_cols]
                rows.append(new_row)
        return WolframDataFrame(all_cols, rows)

    if axis == 1:
        new_cols = []
        for df in objs:
            new_cols.extend(df._columns)
        n = max(len(df._rows) for df in objs)
        rows = []
        for i in range(n):
            row = []
            for df in objs:
                row.extend(df._rows[i] if i < len(df._rows) else [None] * len(df._columns))
            rows.append(row)
        return WolframDataFrame(new_cols, rows)

    raise ValueError(f"axis 必须为 0 或 1，得到 {axis}")


def Series(data=None, name: str = None, **kwargs) -> WolframDataFrame:
    """轻量 Series：单列 WolframDataFrame。"""
    col = name or "value"
    if isinstance(data, list):
        return WolframDataFrame([col], [[v] for v in data])
    if isinstance(data, dict):
        cols = [col, "index"] if name else ["index", "value"]
        rows = [[k, v] for k, v in data.items()]
        return WolframDataFrame(cols, rows)
    return WolframDataFrame([col], [])


def isna(value) -> bool:
    return value is None or value != value


def notna(value) -> bool:
    return not isna(value)
