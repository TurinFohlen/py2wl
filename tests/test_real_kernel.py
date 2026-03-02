#!/usr/bin/env python3
"""
tests/test_real_kernel.py — 真实内核集成测试
==============================================
按 Wolfram Language 返回数据类型分组，验证：
  1. WL 计算结果正确（数值精度 / 结构）
  2. _normalize() 类型标准化正确（PackedArray→list、Rational→float 等）
  3. output_converter 与 wolfram_function 配合无误

分组：
  A. Scalar      — 单个数值（float/int）
  B. List        — 一维 Python list
  C. Matrix      — 二维 Python list
  D. Nested      — 多层结构（Eigensystem / SVD / FindMinimum）
  E. Image       — PNG 文件路径（str）
  F. DataFrame   — WolframDataFrame

运行：
  python tests/test_real_kernel.py              # 全部
  python tests/test_real_kernel.py TypeA        # 只跑 scalar
  python -m pytest tests/test_real_kernel.py -v
"""

import os, sys, math, unittest, tempfile
from pathlib import Path
import sys
from pathlib import Path

def find_project_root():
    """向上查找包含 py2wl 的目录"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "py2wl").exists():
            return parent
    raise RuntimeError("无法找到项目根目录")

ROOT = str(find_project_root())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
# ── 跳过条件：无真实内核 ────────────────────────────────────
def _has_real_kernel():
    try:
        from py2wl.compat._state import _state
        k = _state.get("kernel")
        # MockKernel 没有 session / link 属性
        return k is not None and hasattr(k, "evaluate") and not k.__class__.__name__.startswith("Mock")
    except Exception:
        return False

SKIP_NO_KERNEL = unittest.skipUnless(
    _has_real_kernel(),
    "需要真实 WolframEngine（未检测到运行中的内核）"
)


# ══════════════════════════════════════════════════════════════
#  A. Scalar — WL 返回单个数值
#     文档参考: Mean/Det/Norm/NIntegrate 均返回 MachineReal
#     _normalize: MachineReal → float（已内建，此处验证）
# ══════════════════════════════════════════════════════════════

class TypeA_Scalar(unittest.TestCase):
    """scalar 类型：验证返回 Python float/int，数值精度 ≥ 4 位小数。"""

    def _check(self, func, args, expected, places=4):
        result = func(*args)
        self.assertIsInstance(result, (int, float),
            f"{func} 应返回 float，得到 {type(result).__name__}: {result!r}")
        self.assertAlmostEqual(float(result), expected, places=places,
            msg=f"{func}({args}) = {result}, 期望 {expected}")

    # numpy
    def test_np_mean(self):
        from py2wl.compat import numpy as np
        self._check(np.mean, ([1.0, 2.0, 3.0, 4.0],), 2.5)

    def test_np_std(self):
        from py2wl.compat import numpy as np
        # 样本标准差（ddof=1），WL StandardDeviation
        result = float(np.std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]))
        self.assertAlmostEqual(result, 2.138089935299395, places=4)

    def test_np_sum(self):
        from py2wl.compat import numpy as np
        self._check(np.sum, ([1.0, 2.0, 3.0, 4.0],), 10.0)

    def test_np_linalg_det_2x2(self):
        from py2wl.compat import numpy as np
        self._check(np.linalg.det, ([[1.0,2.0],[3.0,4.0]],), -2.0)

    def test_np_linalg_det_identity(self):
        from py2wl.compat import numpy as np
        self._check(np.linalg.det, ([[1.0,0.0],[0.0,1.0]],), 1.0)

    def test_np_linalg_norm_vector(self):
        from py2wl.compat import numpy as np
        self._check(np.linalg.norm, ([3.0, 4.0],), 5.0)

    def test_np_trace(self):
        from py2wl.compat import numpy as np
        self._check(np.trace, ([[1.0,0.0],[0.0,3.0]],), 4.0)

    def test_np_median(self):
        from py2wl.compat import numpy as np
        self._check(np.median, ([1.0, 2.0, 3.0, 4.0, 5.0],), 3.0)

    def test_np_max(self):
        from py2wl.compat import numpy as np
        self._check(np.max, ([3.0, 1.0, 4.0, 1.0, 5.0],), 5.0)

    def test_np_min(self):
        from py2wl.compat import numpy as np
        self._check(np.min, ([3.0, 1.0, 4.0, 1.0, 5.0],), 1.0)

    # scipy scalar
    def test_scipy_linalg_det(self):
        from py2wl.compat import scipy
        self._check(scipy.linalg.det, ([[2.0,1.0],[1.0,1.0]],), 1.0)

    def test_scipy_linalg_norm(self):
        from py2wl.compat import scipy
        self._check(scipy.linalg.norm, ([3.0, 4.0],), 5.0)

    # torch / tf scalar
    def test_torch_mean(self):
        from py2wl.compat import torch
        self._check(torch.mean, ([1.0,2.0,3.0,4.0],), 2.5)

    def test_torch_sum(self):
        from py2wl.compat import torch
        self._check(torch.sum, ([1.0,2.0,3.0,4.0],), 10.0)

    def test_tf_reduce_mean(self):
        from py2wl.compat import tensorflow as tf
        self._check(tf.reduce_mean, ([1.0,2.0,3.0,4.0],), 2.5)


# ══════════════════════════════════════════════════════════════
#  B. List — WL 返回一维列表
#     _normalize: PackedArray.tolist() 或 WLExpr → Python list
# ══════════════════════════════════════════════════════════════

class TypeB_List(unittest.TestCase):
    """list 类型：验证返回 Python list，元素类型为 float/int。"""

    def _check_list(self, result, expected, places=4):
        self.assertIsInstance(result, list,
            f"应返回 list，得到 {type(result).__name__}: {result!r}")
        self.assertEqual(len(result), len(expected),
            f"长度不匹配: {len(result)} != {len(expected)}")
        for i, (got, exp) in enumerate(zip(result, expected)):
            self.assertAlmostEqual(float(got), exp, places=places,
                msg=f"result[{i}]={got}, 期望 {exp}")

    def test_np_sort(self):
        from py2wl.compat import numpy as np
        result = np.sort([3.0, 1.0, 4.0, 1.0, 5.0])
        self.assertIsInstance(result, list)
        self.assertEqual(result, sorted(result))

    def test_np_cumsum(self):
        from py2wl.compat import numpy as np
        result = np.cumsum([1.0, 2.0, 3.0, 4.0])
        self._check_list(result, [1.0, 3.0, 6.0, 10.0])

    def test_np_diff(self):
        from py2wl.compat import numpy as np
        result = np.diff([1.0, 3.0, 6.0, 10.0])
        self._check_list(result, [2.0, 3.0, 4.0])

    def test_np_linalg_eigvals(self):
        from py2wl.compat import numpy as np
        result = np.linalg.eigvals([[3.0, 0.0], [0.0, 2.0]])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        # 特征值为 3 和 2（顺序不定）
        vals = sorted(float(v) for v in result)
        self.assertAlmostEqual(vals[0], 2.0, places=4)
        self.assertAlmostEqual(vals[1], 3.0, places=4)

    def test_np_linalg_solve(self):
        from py2wl.compat import numpy as np
        # 解 [[2,1],[1,1]]x = [1,2]  → x = [-1, 3]
        result = np.linalg.solve([[2.0,1.0],[1.0,1.0]], [1.0,2.0])
        self.assertIsInstance(result, list)
        self._check_list(result, [-1.0, 3.0])

    def test_np_flatten(self):
        from py2wl.compat import numpy as np
        result = np.flatten([[1.0, 2.0], [3.0, 4.0]])
        self._check_list(result, [1.0, 2.0, 3.0, 4.0])

    def test_scipy_fft(self):
        from py2wl.compat import scipy
        # FFT of [1,0,1,0] → 实部为 [2,0,2,0]（复数列表）
        result = scipy.fft.fft([1.0, 0.0, 1.0, 0.0])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)

    def test_scipy_linalg_eigvals(self):
        from py2wl.compat import scipy
        result = scipy.linalg.eigvals([[3.0, 0.0], [0.0, 2.0]])
        self.assertIsInstance(result, list)
        vals = sorted(float(v.real if hasattr(v,'real') else v) for v in result)
        self.assertAlmostEqual(vals[0], 2.0, places=4)
        self.assertAlmostEqual(vals[1], 3.0, places=4)

    # PackedArray 专项测试
    def test_packed_array_normalized(self):
        """验证 _normalize() 正确处理 PackedArray → Python list。"""
        from py2wl.compat import numpy as np
        result = np.linalg.eigvals([[5.0, 0.0], [0.0, 3.0]])
        # 关键：不应是 PackedArray，必须是原生 list
        self.assertIsInstance(result, list,
            f"PackedArray 未被 _normalize 展开，得到 {type(result)}")
        self.assertTrue(all(isinstance(v, (int, float, complex)) for v in result),
            f"元素类型应为数值，实际: {[type(v).__name__ for v in result]}")


# ══════════════════════════════════════════════════════════════
#  C. Matrix — WL 返回二维列表
#     Transpose / Inverse / Correlation / Covariance
# ══════════════════════════════════════════════════════════════

class TypeC_Matrix(unittest.TestCase):
    """matrix 类型：验证返回 list-of-list，元素为 float。"""

    def _check_matrix(self, result, expected, places=4):
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(expected))
        for i, (row_got, row_exp) in enumerate(zip(result, expected)):
            self.assertIsInstance(row_got, list,
                f"row[{i}] 应为 list，得到 {type(row_got)}")
            for j, (g, e) in enumerate(zip(row_got, row_exp)):
                self.assertAlmostEqual(float(g), e, places=places,
                    msg=f"[{i}][{j}]={g}, 期望 {e}")

    def test_np_transpose(self):
        from py2wl.compat import numpy as np
        result = np.transpose([[1.0, 2.0], [3.0, 4.0]])
        self._check_matrix(result, [[1.0, 3.0], [2.0, 4.0]])

    def test_np_linalg_inv(self):
        from py2wl.compat import numpy as np
        # 单位矩阵的逆是自身
        result = np.linalg.inv([[1.0, 0.0], [0.0, 1.0]])
        self._check_matrix(result, [[1.0, 0.0], [0.0, 1.0]])

    def test_np_linalg_inv_2x2(self):
        from py2wl.compat import numpy as np
        # [[4,7],[2,6]] 的逆
        result = np.linalg.inv([[4.0, 7.0], [2.0, 6.0]])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(float(result[0][0]),  0.6, places=4)
        self.assertAlmostEqual(float(result[0][1]), -0.7, places=4)

    def test_scipy_linalg_inv(self):
        from py2wl.compat import scipy
        result = scipy.linalg.inv([[2.0, 0.0], [0.0, 4.0]])
        self._check_matrix(result, [[0.5, 0.0], [0.0, 0.25]])

    def test_np_dot_matrix(self):
        from py2wl.compat import numpy as np
        # [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
        result = np.dot([[1.0,2.0],[3.0,4.0]], [[1.0,0.0],[0.0,1.0]])
        self._check_matrix(result, [[1.0,2.0],[3.0,4.0]])


# ══════════════════════════════════════════════════════════════
#  D. Nested — WL 返回多层结构
#     Eigensystem: {eigenvalues, eigenvectors}
#     SVD:         {U, Sigma, V}
#     FindMinimum: {fmin, x_at_min}  （已包装，Rule 已解开）
#     FindRoot:    x_val             （已包装）
#     NSolve:      [root1, root2, …] （已包装）
# ══════════════════════════════════════════════════════════════

class TypeD_Nested(unittest.TestCase):
    """nested 类型：验证多层结构正确，Rule 对象已被包装解开。"""

    def test_np_linalg_eig_structure(self):
        """Eigensystem → [[vals], [vecs]]，验证 PackedArray 被展开。"""
        from py2wl.compat import numpy as np
        result = np.linalg.eig([[3.0, 0.0], [0.0, 2.0]])
        # 结构：[[eigenvalues], [eigenvectors_rows]]
        self.assertIsInstance(result, list, f"应为 list，得 {type(result)}")
        self.assertEqual(len(result), 2, "应有 [eigenvalues, eigenvectors]")
        vals, vecs = result[0], result[1]
        self.assertIsInstance(vals, list, "eigenvalues 应为 list")
        self.assertIsInstance(vecs, list, "eigenvectors 应为 list")
        # 验证特征值（顺序不定）
        numeric_vals = sorted(float(v) for v in vals)
        self.assertAlmostEqual(numeric_vals[0], 2.0, places=4)
        self.assertAlmostEqual(numeric_vals[1], 3.0, places=4)

    def test_np_linalg_svd_structure(self):
        """SVD → [U, Sigma, V]，验证形状。"""
        from py2wl.compat import numpy as np
        result = np.linalg.svd([[3.0, 0.0], [0.0, 2.0]])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3, "SVD 应返回 [U, Sigma, V]")
        U, S, V = result
        self.assertIsInstance(U, list)
        self.assertIsInstance(S, list)
        self.assertIsInstance(V, list)
        # 奇异值为 3, 2（对角矩阵）
        sv = sorted(float(v) for v in S if isinstance(v, (int, float)))
        if sv:
            self.assertAlmostEqual(sv[-1], 3.0, places=3)

    def test_scipy_optimize_minimize_scalar_no_rule(self):
        """FindMinimum 包装后不含 Rule 对象，返回 {fmin, x_at_min}。"""
        from py2wl.compat import scipy
        # f(x) = (x-3)^2 + 1，最小值 1.0 在 x=3
        result = scipy.optimize.minimize_scalar("(x-3)^2 + 1")
        self.assertIsInstance(result, list,
            f"应返回 [fmin, x], 得 {type(result)}: {result!r}")
        self.assertEqual(len(result), 2)
        fmin, x_at_min = float(result[0]), float(result[1])
        self.assertAlmostEqual(fmin,     1.0, places=3)
        self.assertAlmostEqual(x_at_min, 3.0, places=3)
        # 确保没有 Rule 对象残留
        type_names = [type(x).__name__ for x in result]
        self.assertNotIn("WLFunction", type_names,
            f"结果包含未解开的 WL Rule 对象: {result}")

    def test_scipy_optimize_root_no_rule(self):
        """FindRoot 包装后返回纯数值。"""
        from py2wl.compat import scipy
        # x^2 - 4 = 0，从 x=1 出发找根
        result = scipy.optimize.root("x^2 - 4", 1)
        self.assertIsInstance(result, (int, float),
            f"应返回数值, 得 {type(result)}: {result!r}")
        self.assertAlmostEqual(abs(float(result)), 2.0, places=3)

    def test_scipy_optimize_brentq_no_rule(self):
        """FindRoot 区间包装后返回纯数值。"""
        from py2wl.compat import scipy
        result = scipy.optimize.brentq("x^2 - 4", 0, 3)
        self.assertIsInstance(result, (int, float))
        self.assertAlmostEqual(float(result), 2.0, places=3)

    def test_np_roots_no_rule(self):
        """NSolve 包装后返回实数根列表，无 Rule 对象。"""
        from py2wl.compat import numpy as np
        # x^2 - 5x + 6 = 0 → 根为 2, 3
        result = np.roots("x^2 - 5x + 6")
        self.assertIsInstance(result, list)
        roots = sorted(float(v) for v in result)
        self.assertAlmostEqual(roots[0], 2.0, places=3)
        self.assertAlmostEqual(roots[1], 3.0, places=3)

    def test_sympy_solve_no_rule(self):
        """Solve 包装后返回实数解列表。"""
        from py2wl.compat import sympy as sp
        result = sp.solve("x^2 - 4 == 0")
        self.assertIsInstance(result, list)
        roots = sorted(float(v) for v in result)
        self.assertAlmostEqual(roots[0], -2.0, places=3)
        self.assertAlmostEqual(roots[1],  2.0, places=3)

    def test_no_rule_objects_in_nested(self):
        """通用断言：nested 类型结果不含任何 WLFunction/Rule 对象。"""
        from py2wl.compat import numpy as np
        result = np.linalg.eig([[3.0, 0.0], [0.0, 2.0]])
        def _check_no_rule(obj, path=""):
            t = type(obj).__name__
            self.assertNotIn("WLFunction", t,
                f"发现 WLFunction 对象 at {path}: {obj!r}")
            self.assertNotIn("WLSymbol", t,
                f"发现 WLSymbol 对象 at {path}: {obj!r}")
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    _check_no_rule(v, f"{path}[{i}]")
        _check_no_rule(result)


# ══════════════════════════════════════════════════════════════
#  E. Image — WL 返回 PNG 路径
#     from_wl_image 复制到持久路径，验证文件存在且是有效 PNG
# ══════════════════════════════════════════════════════════════

class TypeE_Image(unittest.TestCase):
    """image 类型：验证返回持久 PNG 路径，文件头有效。"""

    PNG_MAGIC = b'\x89PNG'

    def _check_image(self, result):
        self.assertIsInstance(result, str,
            f"应返回 PNG 路径字符串，得 {type(result)}")
        self.assertTrue(os.path.exists(result),
            f"PNG 文件不存在：{result}")
        with open(result, "rb") as f:
            header = f.read(4)
        self.assertEqual(header, self.PNG_MAGIC,
            f"文件头不是 PNG：{header!r}")
        os.unlink(result)  # 清理

    def test_matplotlib_plot(self):
        from py2wl.compat import matplotlib as plt
        result = plt.pyplot.plot([1.0, 2.0, 3.0], [1.0, 4.0, 9.0])
        self._check_image(result)

    def test_matplotlib_scatter(self):
        from py2wl.compat import matplotlib as plt
        result = plt.pyplot.scatter([1.0, 2.0, 3.0], [1.0, 4.0, 9.0])
        self._check_image(result)

    def test_matplotlib_hist(self):
        from py2wl.compat import matplotlib as plt
        result = plt.pyplot.hist([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        self._check_image(result)

    def test_matplotlib_bar(self):
        from py2wl.compat import matplotlib as plt
        result = plt.pyplot.bar(["a","b","c"], [3.0, 1.0, 4.0])
        self._check_image(result)

    def test_image_is_not_temp_path(self):
        """验证 from_wl_image 复制后路径不含 wlb_ 前缀（临时文件已删）。"""
        from py2wl.compat import matplotlib as plt
        result = plt.pyplot.plot([1.0, 2.0], [1.0, 2.0])
        self.assertNotIn("wlb_", os.path.basename(result),
            "应返回持久路径，不是临时 wlb_ 路径")
        if os.path.exists(result):
            os.unlink(result)


# ══════════════════════════════════════════════════════════════
#  F. DataFrame — WolframDataFrame
# ══════════════════════════════════════════════════════════════

class TypeF_DataFrame(unittest.TestCase):
    """dataframe 类型：验证 read_csv 往返、groupby、merge 正确。"""

    def setUp(self):
        from py2wl.compat.pandas import WolframDataFrame
        self.WDF = WolframDataFrame
        # 创建临时 CSV
        fd, self.csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(self.csv_path, "w") as f:
            f.write("name,age,city\n")
            f.write("Alice,25,BJ\n")
            f.write("Bob,30,SH\n")
            f.write("Carol,28,BJ\n")

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.unlink(self.csv_path)

    def test_read_csv_returns_wdf(self):
        from py2wl.compat import pandas as pd
        df = pd.read_csv(self.csv_path)
        self.assertIsInstance(df, self.WDF)
        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(df.columns, ["name", "age", "city"])

    def test_read_csv_values_correct(self):
        from py2wl.compat import pandas as pd
        df = pd.read_csv(self.csv_path)
        self.assertEqual(df["age"], [25, 30, 28])

    def test_groupby_mean(self):
        from py2wl.compat import pandas as pd
        df = pd.read_csv(self.csv_path)
        result = df.groupby("city").mean()
        self.assertIsInstance(result, self.WDF)
        cities = result["city"]
        self.assertIn("BJ", cities)
        bj_row = result._rows[result["city"].index("BJ")]
        bj_age = bj_row[result._columns.index("age")]
        self.assertAlmostEqual(float(bj_age), 26.5, places=2)

    def test_merge(self):
        from py2wl.compat import pandas as pd
        left  = pd.DataFrame({"id": [1,2,3], "val": [10,20,30]})
        right = pd.DataFrame({"id": [2,3,4], "score": [100,200,300]})
        result = pd.merge(left, right, on="id", how="inner")
        self.assertIsInstance(result, self.WDF)
        self.assertEqual(len(result), 2)

    def test_to_csv_roundtrip(self):
        from py2wl.compat import pandas as pd
        df = pd.read_csv(self.csv_path)
        out = self.csv_path + ".out.csv"
        df.to_csv(out)
        df2 = pd.read_csv(out)
        os.unlink(out)
        self.assertEqual(df._columns, df2._columns)
        self.assertEqual(len(df), len(df2))

    def test_describe_structure(self):
        from py2wl.compat import pandas as pd
        df = pd.read_csv(self.csv_path)
        desc = df.describe()
        self.assertIsInstance(desc, self.WDF)
        self.assertIn("stat", desc._columns)
        stats = desc["stat"]
        for s in ["count", "mean", "std", "min", "max"]:
            self.assertIn(s, stats, f"describe() 缺少 {s}")


# ══════════════════════════════════════════════════════════════
#  G. 类型标准化专项（_normalize）
#     不依赖特定库，直接测试 from_wxf 的标准化行为
# ══════════════════════════════════════════════════════════════

class TypeG_Normalization(unittest.TestCase):
    """验证 _normalize() 对 wolframclient 各类型的处理。"""

    def test_scalar_is_float(self):
        """numeric:true 规则返回 Python float，不是 MachineReal。"""
        from py2wl.compat import numpy as np
        result = np.mean([1.0, 2.0, 3.0])
        self.assertIsInstance(result, float,
            f"Mean 应返回 float，得 {type(result).__name__}")

    def test_packed_array_is_list(self):
        """PackedArray 经 _normalize 后是 Python list。"""
        from py2wl.compat import numpy as np
        result = np.linalg.eigvals([[3.0, 0.0], [0.0, 2.0]])
        self.assertIsInstance(result, list,
            f"PackedArray 应被展开为 list，得 {type(result).__name__}")

    def test_nested_list_elements_are_list(self):
        """Eigensystem 的嵌套结构每一层都是 list（非 PackedArray）。"""
        from py2wl.compat import numpy as np
        result = np.linalg.eig([[3.0, 0.0], [0.0, 2.0]])
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, list,
                f"嵌套元素应为 list，得 {type(item).__name__}")

    def test_no_wlfunction_in_numeric_result(self):
        """数值函数结果中不含任何 WLFunction 对象。"""
        from py2wl.compat import numpy as np
        for func, args in [
            (np.mean,        ([1.0,2.0,3.0],)),
            (np.linalg.norm, ([3.0,4.0],)),
            (np.sum,         ([1.0,2.0,3.0],)),
        ]:
            result = func(*args)
            self.assertNotIn("WLFunction", type(result).__name__,
                f"{func} 返回了 WLFunction: {result!r}")


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 支持 `python test_real_kernel.py TypeA` 只跑某组
    if len(sys.argv) > 1 and sys.argv[1].startswith("Type"):
        suite = unittest.TestLoader().loadTestsFromName(
            sys.argv[1], sys.modules[__name__]
        )
        sys.argv.pop(1)
    else:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
