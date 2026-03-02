#!/usr/bin/env python3
"""
tests/test_pandas.py — WolframDataFrame 测试套件
=================================================
按实现类型分组，不依赖真实内核：

  A. 构造与属性    — __init__ / shape / columns / dtypes
  B. 行列访问      — __getitem__ / loc / iloc / head / tail
  C. 变形操作      — sort_values / dropna / fillna / rename / reset_index / copy
  D. 统计运算      — mean / std / describe / value_counts / unique
  E. 查询过滤      — query
  F. 分组聚合      — groupby (纯 Python 路径)
  G. 合并拼接      — merge / concat
  H. IO 往返       — read_csv / to_csv / to_dict / DataFrame 构造器
  I. 内核委托      — corr / rolling (需要真实内核，自动 skip)
"""

import os, sys, math, tempfile, unittest
from pathlib import Path

ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from py2wl.compat.pandas import WolframDataFrame as WDF


# ── 测试数据工厂 ───────────────────────────────────────────────
def make_scores():
    """6 行 × 4 列的成绩表。"""
    return WDF(
        ["name", "score", "grade", "city"],
        [["Alice", 92, "A", "BJ"],
         ["Bob",   78, "C", "SH"],
         ["Carol", 88, "B", "BJ"],
         ["Dave",  95, "A", "SZ"],
         ["Eve",   72, "D", "SH"],
         ["Frank", 85, "B", "BJ"]]
    )

def make_nums():
    """纯数值 DataFrame，方便统计测试。"""
    return WDF(
        ["x", "y"],
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0],
         [4.0, 40.0], [5.0, 50.0]]
    )

def make_with_na():
    """含空值的 DataFrame。"""
    return WDF(
        ["a", "b"],
        [[1, "x"], [None, "y"], [3, None], [4, "z"]]
    )


# ══════════════════════════════════════════════════════════════
#  A. 构造与属性
# ══════════════════════════════════════════════════════════════
class TestA_Construction(unittest.TestCase):

    def test_shape(self):
        df = make_scores()
        self.assertEqual(df.shape, (6, 4))

    def test_columns(self):
        df = make_scores()
        self.assertEqual(df.columns, ["name", "score", "grade", "city"])

    def test_len(self):
        self.assertEqual(len(make_scores()), 6)
        self.assertEqual(len(WDF(["a"], [])), 0)

    def test_empty_dataframe(self):
        df = WDF(["x", "y"], [])
        self.assertEqual(df.shape, (0, 2))
        self.assertEqual(len(df), 0)

    def test_dtypes(self):
        df = make_scores()
        dt = df.dtypes
        self.assertIn("score", dt)
        # score 列全是整数 → int 或 float
        self.assertIn(dt["score"], ("int", "float"))
        # name 列是字符串
        self.assertEqual(dt["name"], "str")

    def test_repr_smoke(self):
        # 不崩溃即可
        s = repr(make_scores())
        self.assertIn("Alice", s)


# ══════════════════════════════════════════════════════════════
#  B. 行列访问
# ══════════════════════════════════════════════════════════════
class TestB_Access(unittest.TestCase):

    def test_getitem_column(self):
        df = make_scores()
        names = df["name"]
        self.assertIsInstance(names, list)
        self.assertEqual(names[0], "Alice")
        self.assertEqual(len(names), 6)

    def test_getitem_row_by_index(self):
        df = make_scores()
        row = df[0]
        self.assertIsInstance(row, (list, dict))

    def test_head(self):
        df = make_scores()
        h = df.head(3)
        self.assertIsInstance(h, WDF)
        self.assertEqual(len(h), 3)
        self.assertEqual(h["name"][0], "Alice")

    def test_tail(self):
        df = make_scores()
        t = df.tail(2)
        self.assertEqual(len(t), 2)
        self.assertEqual(t["name"][-1], "Frank")

    def test_head_default(self):
        df = WDF(["a"], [[i] for i in range(10)])
        self.assertEqual(len(df.head()), 5)

    def test_column_assignment(self):
        df = make_scores()
        df["bonus"] = [10] * 6
        self.assertIn("bonus", df.columns)
        self.assertEqual(df["bonus"][0], 10)

    def test_isna(self):
        df = make_with_na()
        mask = df.isna("a")
        self.assertTrue(mask[1])   # None → True
        self.assertFalse(mask[0])

    def test_notna(self):
        df = make_with_na()
        mask = df.notna("a")
        self.assertFalse(mask[1])
        self.assertTrue(mask[0])


# ══════════════════════════════════════════════════════════════
#  C. 变形操作
# ══════════════════════════════════════════════════════════════
class TestC_Transform(unittest.TestCase):

    def test_sort_values_ascending(self):
        df = make_scores()
        s = df.sort_values("score")
        scores = s["score"]
        self.assertEqual(scores, sorted(scores))

    def test_sort_values_descending(self):
        df = make_scores()
        s = df.sort_values("score", ascending=False)
        scores = s["score"]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_dropna(self):
        df = make_with_na()
        cleaned = df.dropna()
        self.assertEqual(len(cleaned), 2)   # 只有 row[0] 和 row[3] 完整
        for row in cleaned._rows:
            self.assertTrue(all(v is not None and v != "" for v in row))

    def test_fillna(self):
        df = make_with_na()
        filled = df.fillna(0)
        flat = [v for row in filled._rows for v in row]
        self.assertNotIn(None, flat)

    def test_rename(self):
        df = make_scores()
        r = df.rename(columns={"score": "points"})
        self.assertIn("points", r.columns)
        self.assertNotIn("score", r.columns)

    def test_rename_does_not_mutate(self):
        df = make_scores()
        _ = df.rename(columns={"score": "points"})
        self.assertIn("score", df.columns)

    def test_reset_index(self):
        df = make_scores().head(3)
        r = df.reset_index()
        self.assertIsInstance(r, WDF)
        self.assertEqual(len(r), 3)

    def test_copy_independence(self):
        df = make_scores()
        cp = df.copy()
        cp["score"][0] = 999
        # 原 df 不受影响
        self.assertNotEqual(df["score"][0], 999)


# ══════════════════════════════════════════════════════════════
#  D. 统计运算（纯 Python）
# ══════════════════════════════════════════════════════════════
class TestD_Stats(unittest.TestCase):

    def test_mean(self):
        df = make_nums()
        m = df.mean()
        self.assertAlmostEqual(m["x"], 3.0, places=4)
        self.assertAlmostEqual(m["y"], 30.0, places=4)

    def test_sum(self):
        df = make_nums()
        s = df.sum()
        self.assertAlmostEqual(s["x"], 15.0, places=4)

    def test_min_max(self):
        df = make_nums()
        self.assertAlmostEqual(df.min()["x"], 1.0, places=4)
        self.assertAlmostEqual(df.max()["x"], 5.0, places=4)

    def test_std(self):
        # std([1,2,3,4,5]) 样本标准差 = sqrt(2.5) ≈ 1.5811
        df = make_nums()
        s = df.std()
        self.assertAlmostEqual(s["x"], math.sqrt(2.5), places=3)

    def test_describe_structure(self):
        df = make_nums()
        d = df.describe()
        self.assertIsInstance(d, WDF)
        stats = d["stat"]
        for s in ["count", "mean", "std", "min", "max"]:
            self.assertIn(s, stats, f"describe() 缺少 '{s}'")

    def test_describe_values(self):
        df = make_nums()
        d = df.describe()
        idx = d["stat"].index("mean")
        x_mean = float(d["x"][idx])
        self.assertAlmostEqual(x_mean, 3.0, places=4)

    def test_unique(self):
        df = make_scores()
        grades = df.unique("grade")
        self.assertIsInstance(grades, list)
        self.assertEqual(sorted(grades), ["A", "B", "C", "D"])

    def test_value_counts(self):
        df = make_scores()
        vc = df.value_counts("city")
        self.assertEqual(vc.get("BJ"), 3)
        self.assertEqual(vc.get("SH"), 2)


# ══════════════════════════════════════════════════════════════
#  E. 查询过滤
# ══════════════════════════════════════════════════════════════
class TestE_Query(unittest.TestCase):

    def test_query_numeric_gt(self):
        df = make_scores()
        r = df.query("score > 85")
        self.assertEqual(len(r), 3)
        self.assertTrue(all(s > 85 for s in r["score"]))

    def test_query_string_eq(self):
        df = make_scores()
        r = df.query("city == 'BJ'")
        self.assertEqual(len(r), 3)
        self.assertTrue(all(c == "BJ" for c in r["city"]))

    def test_query_combined(self):
        df = make_scores()
        r = df.query("score >= 88 and city == 'BJ'")
        self.assertEqual(len(r), 2)

    def test_query_no_match(self):
        df = make_scores()
        r = df.query("score > 100")
        self.assertEqual(len(r), 0)

    def test_query_returns_wdf(self):
        df = make_scores()
        self.assertIsInstance(df.query("score > 80"), WDF)


# ══════════════════════════════════════════════════════════════
#  F. 分组聚合
# ══════════════════════════════════════════════════════════════
class TestF_GroupBy(unittest.TestCase):

    def test_groupby_mean(self):
        df = make_scores()
        g = df.groupby("city").mean()
        self.assertIsInstance(g, WDF)
        cities = g["city"]
        self.assertIn("BJ", cities)
        bj_idx = cities.index("BJ")
        # BJ: 92, 88, 85 → mean = 88.33
        self.assertAlmostEqual(float(g["score"][bj_idx]),
                               (92+88+85)/3, places=2)

    def test_groupby_count(self):
        df = make_scores()
        g = df.groupby("city").count()
        cities = g["city"]
        bj_idx = cities.index("BJ")
        self.assertEqual(g["score"][bj_idx], 3)

    def test_groupby_sum(self):
        df = make_nums()
        g = df.groupby("x").sum()
        self.assertIsInstance(g, WDF)

    def test_groupby_multi_key(self):
        df = make_scores()
        g = df.groupby(["city", "grade"]).count()
        self.assertIsInstance(g, WDF)
        self.assertGreater(len(g), 0)


# ══════════════════════════════════════════════════════════════
#  G. 合并拼接
# ══════════════════════════════════════════════════════════════
class TestG_MergeConcat(unittest.TestCase):

    def _make_left(self):
        return WDF(["id","val"], [[1,10],[2,20],[3,30]])

    def _make_right(self):
        return WDF(["id","score"], [[2,100],[3,200],[4,300]])

    def test_merge_inner(self):
        from py2wl.compat.pandas import merge
        r = merge(self._make_left(), self._make_right(), on="id", how="inner")
        self.assertEqual(len(r), 2)
        self.assertIn("val",   r.columns)
        self.assertIn("score", r.columns)

    def test_merge_left(self):
        from py2wl.compat.pandas import merge
        r = merge(self._make_left(), self._make_right(), on="id", how="left")
        self.assertEqual(len(r), 3)

    def test_merge_returns_wdf(self):
        from py2wl.compat.pandas import merge
        r = merge(self._make_left(), self._make_right(), on="id")
        self.assertIsInstance(r, WDF)

    def test_concat_rows(self):
        from py2wl.compat.pandas import concat
        a = WDF(["x","y"], [[1,2],[3,4]])
        b = WDF(["x","y"], [[5,6],[7,8]])
        r = concat([a, b])
        self.assertEqual(len(r), 4)
        self.assertEqual(r["x"], [1,3,5,7])

    def test_concat_preserves_columns(self):
        from py2wl.compat.pandas import concat
        a = WDF(["a","b"], [[1,2]])
        b = WDF(["a","b"], [[3,4]])
        r = concat([a, b])
        self.assertEqual(r.columns, ["a", "b"])


# ══════════════════════════════════════════════════════════════
#  H. IO 往返
# ══════════════════════════════════════════════════════════════
class TestH_IO(unittest.TestCase):

    CSV_CONTENT = "name,score,city\nAlice,92,BJ\nBob,78,SH\nCarol,88,BJ\n"

    def test_read_csv_basic(self):
        from py2wl.compat.pandas import read_csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write(self.CSV_CONTENT)
            path = f.name
        try:
            df = read_csv(path)
            self.assertIsInstance(df, WDF)
            self.assertEqual(df.shape, (3, 3))
            self.assertEqual(df.columns, ["name","score","city"])
        finally:
            os.unlink(path)

    def test_read_csv_values(self):
        from py2wl.compat.pandas import read_csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write(self.CSV_CONTENT)
            path = f.name
        try:
            df = read_csv(path)
            self.assertEqual(df["name"][0], "Alice")
            self.assertEqual(df["score"][0], 92)
        finally:
            os.unlink(path)

    def test_to_csv_roundtrip(self):
        from py2wl.compat.pandas import read_csv
        df = WDF(["a","b","c"], [[1,2,3],[4,5,6],[7,8,9]])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            df.to_csv(path)
            df2 = read_csv(path)
            self.assertEqual(df.columns, df2.columns)
            self.assertEqual(len(df), len(df2))
            self.assertEqual(df["a"], [int(x) for x in df2["a"]])
        finally:
            os.unlink(path)

    def test_to_dict_records(self):
        df = WDF(["x","y"], [[1,2],[3,4]])
        records = df.to_dict("records")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["x"], 1)
        self.assertEqual(records[1]["y"], 4)

    def test_to_dict_list(self):
        df = WDF(["x","y"], [[1,2],[3,4]])
        d = df.to_dict("list")
        self.assertEqual(d["x"], [1, 3])
        self.assertEqual(d["y"], [2, 4])

    def test_dataframe_constructor_dict(self):
        from py2wl.compat.pandas import DataFrame
        df = DataFrame({"a": [1,2,3], "b": [4,5,6]})
        self.assertIsInstance(df, WDF)
        self.assertEqual(df.shape, (3, 2))
        self.assertEqual(df["a"], [1,2,3])

    def test_dataframe_constructor_list_of_dicts(self):
        from py2wl.compat.pandas import DataFrame
        df = DataFrame([{"x":1,"y":2}, {"x":3,"y":4}])
        self.assertIsInstance(df, WDF)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["x"], [1, 3])


# ══════════════════════════════════════════════════════════════
#  I. 内核委托（真实内核）
# ══════════════════════════════════════════════════════════════
def _has_real_kernel():
    try:
        from py2wl.compat._state import _state
        k = _state.get("kernel")
        return k is not None and not k.__class__.__name__.startswith("Mock")
    except Exception:
        return False

SKIP_NO_KERNEL = unittest.skipUnless(_has_real_kernel(),
    "需要真实 WolframEngine")

class TestI_KernelDelegated(unittest.TestCase):

    @SKIP_NO_KERNEL
    def test_corr_structure(self):
        """corr() → 相关矩阵，列数 = 原数值列数。"""
        df = make_nums()
        c = df.corr()
        self.assertIsInstance(c, WDF)
        self.assertEqual(len(c.columns), len(df.columns))

    @SKIP_NO_KERNEL
    def test_corr_diagonal_is_one(self):
        """自相关 = 1。"""
        df = make_nums()
        c = df.corr()
        for i, col in enumerate(c.columns):
            self.assertAlmostEqual(float(c[col][i]), 1.0, places=4)

    @SKIP_NO_KERNEL
    def test_rolling_mean(self):
        """rolling(3).mean() 返回 WDF，长度与原始相同。"""
        df = make_nums()
        r = df.rolling(3).mean()
        self.assertIsInstance(r, WDF)
        self.assertEqual(len(r), len(df))

    @SKIP_NO_KERNEL
    def test_rolling_sum(self):
        df = make_nums()
        r = df.rolling(2).sum()
        self.assertIsInstance(r, WDF)


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)
