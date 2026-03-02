"""
test_full.py — 全库映射测试套件
覆盖：numpy / sympy / torch / tensorflow / perf / monitoring
共 40 个测试，全部使用 MockKernel（不需要真实 Wolfram Kernel）
"""
import sys, os, tempfile, unittest
from unittest.mock import patch
from pathlib import Path

ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 清理模块缓存
for k in list(sys.modules):
    if "py2wl" in k:
        del sys.modules[k]

MAPPINGS_DIR = str(Path(__file__).parent.parent / "py2wl" / "compat" / "mappings")
os.environ["WOLFRAM_MAPPINGS_DIR"] = MAPPINGS_DIR


# ═══════════════════════════════════════════════════════
#  通用 MockKernel（WXF 模式）
# ═══════════════════════════════════════════════════════
class MockKernel:
    """
    WSTP-era mock: evaluate(wl_obj) returns Python objects directly.
    No file I/O, no Export strings.
    """
    # wl function name → Python return value
    RETURNS = {
        "Fourier":               [complex(1,0), complex(0,1)],
        "LinearSolve":           [1.0, 2.0],
        "Eigensystem":           [[1.0, 2.0], [[1,0],[0,1]]],
        "SingularValueDecomposition": [[3.0,1.0], [[1,0],[0,1]], [[1,0],[0,1]]],
        "Total":                 10.0,
        "Mean":                  2.5,
        "Sort":                  [1,2,3,4,5],
        "Accumulate":            [1,3,6,10],
        "Dot":                   [[1,0],[0,1]],
        "Inverse":               [[1,0],[0,1]],
        "Transpose":             [[1,3],[2,4]],
        "Sqrt":                  1.4142135623730951,
        "Exp":                   2.718281828459045,
        "Sin":                   0.8414709848078965,
        "RandomReal":            [[0.1,0.2],[0.3,0.4]],
        "Solve":                 [[2.0],[3.0]],
        "IdentityMatrix":        [[1,0],[0,1]],
        "Max":                   5.0,
        "Min":                   1.0,
        "Tr":                    5.0,
        "StandardDeviation":     1.5811388300841898,
        "Variance":              2.5,
        "Det":                   -2.0,
        "Norm":                  3.7416573867739413,
        "N":                     3.141592653589793,
        "Pi":                    3.141592653589793,
        "E":                     2.718281828459045,
        "PrincipalComponents":   [[0.1,0.2],[0.3,0.4]],
        "NIntegrate":            2.0,
        "Integrate":             2.0,
        "D":                     1.0,
        "NSolve":                [[2.0],[3.0]],
        "FindMinimum":           [0.0, [0.0, 0.0]],
        "SeedRandom":            None,
        "Classify":              "class_A",
    }

    def _match(self, expr_str: str):
        for k, v in self.RETURNS.items():
            if k in expr_str:
                return v
        return 42.0

    def evaluate(self, expr) -> object:
        """Accept wl object or string, return Python object."""
        expr_str = str(expr)
        return self._match(expr_str)

    def evaluate_many(self, exprs) -> list:
        return [self.evaluate(e) for e in exprs]


def _inject():
    from py2wl.compat._state import _state
    _state["kernel"]   = MockKernel()
    _state["resolver"] = None


def _reset():
    for k in list(sys.modules):
        if "py2wl" in k:
            del sys.modules[k]
    from py2wl.compat._core.metadata import MetadataRepository
    from py2wl.compat._core.resolver import ResolutionEngine
    from py2wl.compat._state import _state
    MetadataRepository._instance = None
    ResolutionEngine._instance   = None
    _state["kernel"]   = None
    _state["resolver"] = None


# ═══════════════════════════════════════════════════════
#  1. 元数据：全库加载统计
# ═══════════════════════════════════════════════════════
class TestAllMappingsLoad(unittest.TestCase):
    def setUp(self):
        _reset()
        from py2wl.compat._core.metadata import MetadataRepository
        self.repo = MetadataRepository(MAPPINGS_DIR)

    def test_total_count(self):
        n = len(self.repo.all_rules)
        self.assertGreater(n, 200, f"期望 >200 条规则，实际 {n}")
        print(f"\n[Test] 全库规则总数 ✅  {n} 条")

    def test_numpy_rules(self):
        rules = self.repo.search_rules("numpy")
        self.assertGreater(len(rules), 100)
        print(f"\n[Test] numpy 规则 ✅  {len(rules)} 条")

    def test_sympy_rules(self):
        rules = self.repo.search_rules("sympy")
        self.assertGreater(len(rules), 50)
        print(f"\n[Test] sympy 规则 ✅  {len(rules)} 条")

    def test_torch_rules(self):
        rules = self.repo.search_rules("torch")
        self.assertGreater(len(rules), 50)
        print(f"\n[Test] torch 规则 ✅  {len(rules)} 条")

    def test_tensorflow_rules(self):
        rules = self.repo.search_rules("tensorflow")
        self.assertGreater(len(rules), 30)
        print(f"\n[Test] tensorflow 规则 ✅  {len(rules)} 条")

    def test_broadcast_rules(self):
        rules = self.repo.search_rules("broadcast")
        self.assertGreater(len(rules), 15, "广播规则应覆盖 numpy + torch + tf")
        print(f"\n[Test] 广播规则 ✅  {len(rules)} 条（跨 numpy/torch/tf）")

    def test_performance_rules(self):
        rules = self.repo.search_rules("performance")
        self.assertGreater(len(rules), 10)
        print(f"\n[Test] 性能优化规则 ✅  {len(rules)} 条")

    def test_monitoring_rules(self):
        rules = self.repo.search_rules("monitoring")
        self.assertGreater(len(rules), 10)
        print(f"\n[Test] 监控日志规则 ✅  {len(rules)} 条")


# ═══════════════════════════════════════════════════════
#  2. NumPy 代理端到端
# ═══════════════════════════════════════════════════════
class TestNumpyProxy(unittest.TestCase):
    def setUp(self):
        _reset(); _inject()

    def _np(self):
        import py2wl.compat.numpy as _np
        _inject()
        return _np

    def test_fft(self):
        np = self._np()
        r = np.fft.fft([1, 0, 1, 0])
        self.assertIsNotNone(r)
        print(f"\n[Test] np.fft.fft ✅  type={type(r).__name__}")

    def test_linalg_solve(self):
        np = self._np()
        r = np.linalg.solve([[2,1],[1,1]], [1,2])
        self.assertIsNotNone(r)
        print(f"\n[Test] np.linalg.solve ✅  {r}")

    def test_linalg_det(self):
        np = self._np()
        r = np.linalg.det([[1,2],[3,4]])
        self.assertIsNotNone(r)
        print(f"\n[Test] np.linalg.det ✅  scalar={r}")

    def test_sum(self):
        np = self._np()
        r = np.sum([1, 2, 3, 4])
        self.assertIsNotNone(r)
        print(f"\n[Test] np.sum ✅  {r}")

    def test_broadcast_add(self):
        np = self._np()
        r = np.add([1, 2, 3], [4, 5, 6])
        self.assertIsNotNone(r)
        print(f"\n[Test] np.add (广播) ✅  {r}")

    def test_sin(self):
        np = self._np()
        r = np.sin([0, 1.57, 3.14])
        self.assertIsNotNone(r)
        print(f"\n[Test] np.sin ✅")

    def test_sort(self):
        np = self._np()
        r = np.sort([3, 1, 4, 1, 5])
        self.assertIsNotNone(r)
        print(f"\n[Test] np.sort ✅  {r}")

    def test_cumsum(self):
        np = self._np()
        r = np.cumsum([1, 2, 3, 4])
        self.assertIsNotNone(r)
        print(f"\n[Test] np.cumsum ✅  {r}")

    def test_transpose(self):
        np = self._np()
        r = np.transpose([[1,2],[3,4]])
        self.assertIsNotNone(r)
        print(f"\n[Test] np.transpose ✅")

    def test_linalg_eig(self):
        np = self._np()
        r = np.linalg.eig([[4,0],[0,3]])
        self.assertIsNotNone(r)
        print(f"\n[Test] np.linalg.eig ✅")


# ═══════════════════════════════════════════════════════
#  3. SymPy 代理端到端
# ═══════════════════════════════════════════════════════
class TestSympyProxy(unittest.TestCase):
    def setUp(self):
        _reset(); _inject()

    def _sp(self):
        import py2wl.compat.sympy as sp
        _inject()
        return sp

    def test_diff(self):
        sp = self._sp()
        rule = sp._path if hasattr(sp, "_path") else "sympy"
        # 路径存在即可
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("sympy.diff")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "D")
        print(f"\n[Test] sympy.diff → D ✅")

    def test_integrate(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("sympy.integrate")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "Integrate")
        print(f"\n[Test] sympy.integrate → Integrate ✅")

    def test_solve(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("sympy.solve")
        self.assertIsNotNone(r)
        self.assertIn("Solve", r["wolfram_function"])
        print(f"\n[Test] sympy.solve → Solve ✅")

    def test_factor(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("sympy.factor")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "Factor")
        print(f"\n[Test] sympy.factor → Factor ✅")

    def test_gamma(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("sympy.gamma")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "Gamma")
        print(f"\n[Test] sympy.gamma → Gamma ✅")

    def test_special_functions_count(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        rules = repo.search_rules("sympy")
        self.assertGreater(len(rules), 60)
        print(f"\n[Test] sympy 特殊函数覆盖 ✅  {len(rules)} 条规则")


# ═══════════════════════════════════════════════════════
#  4. PyTorch 代理端到端
# ═══════════════════════════════════════════════════════
class TestTorchProxy(unittest.TestCase):
    def setUp(self):
        _reset(); _inject()

    def test_matmul_rule(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("torch.matmul")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "Dot")
        print(f"\n[Test] torch.matmul → Dot ✅")

    def test_broadcast_add(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("torch.add")
        self.assertIsNotNone(r)
        self.assertIn("Broadcasting", r["wolfram_function"])
        print(f"\n[Test] torch.add → Broadcasting[Plus] ✅")

    def test_relu_rule(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("torch.nn.functional.relu")
        self.assertIsNotNone(r)
        print(f"\n[Test] torch.nn.functional.relu ✅  {r['wolfram_function'][:30]}...")

    def test_linalg_svd(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("torch.linalg.svd")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "SingularValueDecomposition")
        print(f"\n[Test] torch.linalg.svd → SingularValueDecomposition ✅")

    def test_torch_proxy_chain(self):
        import py2wl.compat.torch as torch
        _inject()
        # linalg 子命名空间应返回代理对象，不报错
        sub = torch.linalg
        self.assertIn("Proxy", type(sub).__name__)
        print(f"\n[Test] torch.linalg 代理链 ✅  {repr(sub)}")

    def test_seed(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("torch.manual_seed")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "SeedRandom")
        print(f"\n[Test] torch.manual_seed → SeedRandom ✅")


# ═══════════════════════════════════════════════════════
#  5. TensorFlow 代理
# ═══════════════════════════════════════════════════════
class TestTensorflowProxy(unittest.TestCase):
    def test_reduce_sum(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("tf.reduce_sum")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "Total")
        print(f"\n[Test] tf.reduce_sum → Total ✅")

    def test_broadcast_multiply(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("tf.multiply")
        self.assertIsNotNone(r)
        self.assertIn("Broadcasting", r["wolfram_function"])
        print(f"\n[Test] tf.multiply → Broadcasting[Times] ✅")

    def test_linalg_matmul(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("tf.linalg.matmul")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "Dot")
        print(f"\n[Test] tf.linalg.matmul → Dot ✅")

    def test_nn_relu(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("tf.nn.relu")
        self.assertIsNotNone(r)
        print(f"\n[Test] tf.nn.relu ✅")

    def test_random_set_seed(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("tf.random.set_seed")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "SeedRandom")
        print(f"\n[Test] tf.random.set_seed → SeedRandom ✅")


# ═══════════════════════════════════════════════════════
#  6. 性能工具 / 监控
# ═══════════════════════════════════════════════════════
class TestPerfMonitoring(unittest.TestCase):
    def test_numba_jit(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("numba.jit")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "Compile")
        print(f"\n[Test] numba.jit → Compile ✅")

    def test_lru_cache(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("functools.lru_cache")
        self.assertIsNotNone(r)
        print(f"\n[Test] functools.lru_cache ✅  {r['wolfram_function'][:30]}...")

    def test_scipy_sparse(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("scipy.sparse.csr_matrix")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "SparseArray")
        print(f"\n[Test] scipy.sparse.csr_matrix → SparseArray ✅")

    def test_einsum(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("numpy.einsum")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "TensorContract")
        print(f"\n[Test] numpy.einsum → TensorContract ✅")

    def test_tqdm(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("tqdm.tqdm")
        self.assertIsNotNone(r)
        self.assertIn("ProgressIndicator", r["wolfram_function"])
        print(f"\n[Test] tqdm.tqdm → Monitor+ProgressIndicator ✅")

    def test_time_sleep(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("time.sleep")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "Pause")
        print(f"\n[Test] time.sleep → Pause ✅")

    def test_scipy_fft(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("scipy.fft.fft")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "Fourier")
        print(f"\n[Test] scipy.fft.fft → Fourier ✅")

    def test_interpolation(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        r = repo.get_rule("scipy.interpolate.interp1d")
        self.assertIsNotNone(r)
        self.assertEqual(r["wolfram_function"], "Interpolation")
        print(f"\n[Test] scipy.interpolate.interp1d → Interpolation ✅")


class TestScipy(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata import MetadataRepository
        self.repo = MetadataRepository(MAPPINGS_DIR)

    def _r(self, path):
        r = self.repo.get_rule(path)
        self.assertIsNotNone(r, f"规则未找到：{path}")
        return r

    def test_integrate_quad(self):
        r = self._r("scipy.integrate.quad")
        self.assertIn("NIntegrate", r["wolfram_function"])
        print(f"\n[Test] scipy.integrate.quad → NIntegrate ✅")

    def test_optimize_minimize(self):
        r = self._r("scipy.optimize.minimize")
        self.assertIn("FindMinimum", r["wolfram_function"])
        print(f"\n[Test] scipy.optimize.minimize → FindMinimum ✅")

    def test_optimize_curve_fit(self):
        r = self._r("scipy.optimize.curve_fit")
        self.assertEqual(r["wolfram_function"], "NonlinearModelFit")
        print(f"\n[Test] scipy.optimize.curve_fit → NonlinearModelFit ✅")

    def test_linalg_lstsq(self):
        r = self._r("scipy.linalg.lstsq")
        self.assertEqual(r["wolfram_function"], "LeastSquares")
        print(f"\n[Test] scipy.linalg.lstsq → LeastSquares ✅")

    def test_linalg_expm(self):
        r = self._r("scipy.linalg.expm")
        self.assertEqual(r["wolfram_function"], "MatrixExp")
        print(f"\n[Test] scipy.linalg.expm → MatrixExp ✅")

    def test_special_bessel(self):
        r = self._r("scipy.special.besselj")
        self.assertEqual(r["wolfram_function"], "BesselJ")
        print(f"\n[Test] scipy.special.besselj → BesselJ ✅")

    def test_special_comb(self):
        r = self._r("scipy.special.comb")
        self.assertEqual(r["wolfram_function"], "Binomial")
        print(f"\n[Test] scipy.special.comb → Binomial ✅")

    def test_signal_convolve(self):
        r = self._r("scipy.signal.convolve")
        self.assertEqual(r["wolfram_function"], "ListConvolve")
        print(f"\n[Test] scipy.signal.convolve → ListConvolve ✅")

    def test_signal_find_peaks(self):
        r = self._r("scipy.signal.find_peaks")
        self.assertEqual(r["wolfram_function"], "PeakDetect")
        print(f"\n[Test] scipy.signal.find_peaks → PeakDetect ✅")

    def test_ndimage_gaussian(self):
        r = self._r("scipy.ndimage.gaussian_filter")
        self.assertEqual(r["wolfram_function"], "GaussianFilter")
        print(f"\n[Test] scipy.ndimage.gaussian_filter → GaussianFilter ✅")

    def test_ndimage_morphology(self):
        r = self._r("scipy.ndimage.binary_erosion")
        self.assertEqual(r["wolfram_function"], "Erosion")
        print(f"\n[Test] scipy.ndimage.binary_erosion → Erosion ✅")

    def test_stats_ttest(self):
        r = self._r("scipy.stats.ttest_ind")
        self.assertEqual(r["wolfram_function"], "TTest")
        print(f"\n[Test] scipy.stats.ttest_ind → TTest ✅")

    def test_stats_pearsonr(self):
        r = self._r("scipy.stats.pearsonr")
        self.assertEqual(r["wolfram_function"], "Correlation")
        print(f"\n[Test] scipy.stats.pearsonr → Correlation ✅")

    def test_stats_norm_pdf(self):
        r = self._r("scipy.stats.norm.pdf")
        self.assertIn("NormalDistribution", r["wolfram_function"])
        print(f"\n[Test] scipy.stats.norm.pdf → PDF[NormalDistribution] ✅")

    def test_spatial_pdist(self):
        r = self._r("scipy.spatial.distance.pdist")
        self.assertEqual(r["wolfram_function"], "DistanceMatrix")
        print(f"\n[Test] scipy.spatial.distance.pdist → DistanceMatrix ✅")

    def test_spatial_convexhull(self):
        r = self._r("scipy.spatial.ConvexHull")
        self.assertEqual(r["wolfram_function"], "ConvexHull")
        print(f"\n[Test] scipy.spatial.ConvexHull → ConvexHull ✅")

    def test_spatial_delaunay(self):
        r = self._r("scipy.spatial.Delaunay")
        self.assertEqual(r["wolfram_function"], "DelaunayMesh")
        print(f"\n[Test] scipy.spatial.Delaunay → DelaunayMesh ✅")

    def test_cluster_hierarchical(self):
        r = self._r("scipy.cluster.hierarchy.linkage")
        self.assertEqual(r["wolfram_function"], "HierarchicalClustering")
        print(f"\n[Test] scipy.cluster.hierarchy.linkage → HierarchicalClustering ✅")

    def test_interpolate_cubicspline(self):
        r = self._r("scipy.interpolate.CubicSpline")
        self.assertIn("Spline", r["wolfram_function"])
        print(f"\n[Test] scipy.interpolate.CubicSpline → Spline ✅")

    def test_fft_dct(self):
        r = self._r("scipy.fft.dct")
        self.assertEqual(r["wolfram_function"], "FourierDCT")
        print(f"\n[Test] scipy.fft.dct → FourierDCT ✅")

    def test_scipy_total_count(self):
        rules = self.repo.search_rules("scipy")
        self.assertGreater(len(rules), 100)
        print(f"\n[Test] scipy 总规则数 ✅  {len(rules)} 条")


# ═══════════════════════════════════════════════════════
class TestPandas(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata import MetadataRepository
        self.repo = MetadataRepository(MAPPINGS_DIR)

    def _r(self, path):
        r = self.repo.get_rule(path)
        self.assertIsNotNone(r, f"规则未找到：{path}")
        return r

    def test_read_csv(self):
        r = self._r("pandas.read_csv")
        self.assertEqual(r["wolfram_function"], "Import")
        print(f"\n[Test] pandas.read_csv → Import ✅")

    def test_groupby(self):
        r = self._r("pandas.DataFrame.groupby")
        self.assertEqual(r["wolfram_function"], "GroupBy")
        print(f"\n[Test] pandas.DataFrame.groupby → GroupBy ✅")

    def test_merge(self):
        r = self._r("pandas.merge")
        self.assertEqual(r["wolfram_function"], "JoinAcross")
        print(f"\n[Test] pandas.merge → JoinAcross ✅")

    def test_rolling_mean(self):
        r = self._r("pandas.DataFrame.rolling.mean")
        self.assertIn("MovingMap", r["wolfram_function"])
        print(f"\n[Test] pandas.rolling.mean → MovingMap[Mean] ✅")

    def test_ewm_mean(self):
        r = self._r("pandas.DataFrame.ewm.mean")
        self.assertEqual(r["wolfram_function"], "ExponentialMovingAverage")
        print(f"\n[Test] pandas.ewm.mean → ExponentialMovingAverage ✅")

    def test_value_counts(self):
        r = self._r("pandas.DataFrame.value_counts")
        self.assertEqual(r["wolfram_function"], "Counts")
        print(f"\n[Test] pandas.value_counts → Counts ✅")

    def test_pivot_table(self):
        r = self._r("pandas.DataFrame.pivot_table")
        self.assertIn("GroupBy", r["wolfram_function"])
        print(f"\n[Test] pandas.pivot_table → Query+GroupBy ✅")

    def test_dropna(self):
        r = self._r("pandas.DataFrame.dropna")
        self.assertEqual(r["wolfram_function"], "DeleteMissing")
        print(f"\n[Test] pandas.dropna → DeleteMissing ✅")

    def test_get_dummies(self):
        r = self._r("pandas.get_dummies")
        self.assertEqual(r["wolfram_function"], "EncodeCategorical")
        print(f"\n[Test] pandas.get_dummies → EncodeCategorical ✅")

    def test_cumsum(self):
        r = self._r("pandas.DataFrame.cumsum")
        self.assertEqual(r["wolfram_function"], "Accumulate")
        print(f"\n[Test] pandas.cumsum → Accumulate ✅")

    def test_to_excel(self):
        r = self._r("pandas.DataFrame.to_excel")
        self.assertEqual(r["wolfram_function"], "Export")
        print(f"\n[Test] pandas.to_excel → Export ✅")

    def test_comparison_operators(self):
        for op, wl in [("eq","Equal"),("gt","Greater"),("lt","Less")]:
            r = self._r(f"pandas.DataFrame.{op}")
            self.assertIn("Broadcasting", r["wolfram_function"])
        print(f"\n[Test] pandas 比较运算符 (eq/gt/lt) → Broadcasting ✅")

    def test_plot_bar(self):
        r = self._r("pandas.DataFrame.plot.bar")
        self.assertEqual(r["wolfram_function"], "BarChart")
        print(f"\n[Test] pandas.plot.bar → BarChart ✅")

    def test_pandas_total_count(self):
        rules = self.repo.search_rules("pandas")
        self.assertGreater(len(rules), 80)
        print(f"\n[Test] pandas 总规则数 ✅  {len(rules)} 条")


# ═══════════════════════════════════════════════════════
#  9. NumPy Extra
# ═══════════════════════════════════════════════════════
class TestNumpyExtra(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata import MetadataRepository
        self.repo = MetadataRepository(MAPPINGS_DIR)

    def _r(self, path):
        r = self.repo.get_rule(path)
        self.assertIsNotNone(r, f"规则未找到：{path}")
        return r

    def test_set_operations(self):
        for path, wl in [
            ("numpy.union1d", "Union"),
            ("numpy.intersect1d", "Intersection"),
            ("numpy.setdiff1d", "Complement"),
        ]:
            r = self._r(path)
            self.assertEqual(r["wolfram_function"], wl)
        print(f"\n[Test] numpy 集合运算（union/intersect/setdiff）✅")

    def test_type_checks(self):
        for path in ["numpy.isfinite","numpy.isinf","numpy.isnan","numpy.isreal"]:
            self.assertIsNotNone(self.repo.get_rule(path))
        print(f"\n[Test] numpy 类型检查函数 ✅")

    def test_window_functions(self):
        for path in ["numpy.hanning","numpy.hamming","numpy.blackman","numpy.bartlett"]:
            self.assertIsNotNone(self.repo.get_rule(path))
        print(f"\n[Test] numpy 窗函数（hanning/hamming/blackman/bartlett）✅")

    def test_bitwise(self):
        for path, wl in [
            ("numpy.bitwise_and","BitAnd"),
            ("numpy.bitwise_or","BitOr"),
            ("numpy.bitwise_xor","BitXor"),
        ]:
            r = self._r(path)
            self.assertEqual(r["wolfram_function"], wl)
        print(f"\n[Test] numpy 位运算（and/or/xor）✅")

    def test_polynomial(self):
        r = self._r("numpy.polyfit")
        self.assertEqual(r["wolfram_function"], "Fit")
        r2 = self._r("numpy.roots")
        self.assertIn("NSolve", r2["wolfram_function"])
        print(f"\n[Test] numpy 多项式（polyfit→Fit, roots→NSolve）✅")

    def test_nan_functions(self):
        for path in ["numpy.nanmean","numpy.nanstd","numpy.nansum","numpy.nanmax","numpy.nanmin"]:
            self.assertIsNotNone(self.repo.get_rule(path))
        print(f"\n[Test] numpy nan系列函数（nanmean/nanstd/…）✅")

    def test_linalg_extra(self):
        for path, wl in [
            ("numpy.linalg.matrix_rank","MatrixRank"),
            ("numpy.linalg.pinv","PseudoInverse"),
            ("numpy.linalg.lu","LUDecomposition"),
            ("numpy.linalg.matrix_power","MatrixPower"),
        ]:
            r = self._r(path)
            self.assertEqual(r["wolfram_function"], wl)
        print(f"\n[Test] numpy.linalg 扩展（rank/pinv/lu/power）✅")

    def test_histogram(self):
        r = self._r("numpy.histogram")
        self.assertEqual(r["wolfram_function"], "BinCounts")
        print(f"\n[Test] numpy.histogram → BinCounts ✅")

    def test_numpy_extra_total(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        all_numpy = [r for r in repo.all_rules if r["python_path"].startswith("numpy.")]
        self.assertGreater(len(all_numpy), 180)
        print(f"\n[Test] numpy.* 规则总数 ✅  {len(all_numpy)} 条")


# ═══════════════════════════════════════════════════════
#  10. scikit-learn
# ═══════════════════════════════════════════════════════
class TestSklearn(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata import MetadataRepository
        self.repo = MetadataRepository(MAPPINGS_DIR)

    def _r(self, path):
        r = self.repo.get_rule(path)
        self.assertIsNotNone(r, f"规则未找到：{path}")
        return r

    def test_preprocessing(self):
        for path, wl in [
            ("sklearn.preprocessing.StandardScaler","Standardize"),
            ("sklearn.preprocessing.MinMaxScaler","Rescale"),
            ("sklearn.preprocessing.normalize","Normalize"),
        ]:
            r = self._r(path)
            self.assertEqual(r["wolfram_function"], wl)
        print(f"\n[Test] sklearn 预处理（StandardScaler/MinMaxScaler/normalize）✅")

    def test_decomposition(self):
        r = self._r("sklearn.decomposition.PCA")
        self.assertEqual(r["wolfram_function"], "PrincipalComponents")
        print(f"\n[Test] sklearn.decomposition.PCA → PrincipalComponents ✅")

    def test_clustering(self):
        r = self._r("sklearn.cluster.KMeans")
        self.assertEqual(r["wolfram_function"], "FindClusters")
        print(f"\n[Test] sklearn.cluster.KMeans → FindClusters ✅")

    def test_regression(self):
        r = self._r("sklearn.linear_model.LinearRegression")
        self.assertEqual(r["wolfram_function"], "LinearModelFit")
        print(f"\n[Test] sklearn.linear_model.LinearRegression → LinearModelFit ✅")

    def test_classification(self):
        for path in ["sklearn.svm.SVC","sklearn.ensemble.RandomForestClassifier"]:
            r = self._r(path)
            self.assertIn("Classify", r["wolfram_function"])
        print(f"\n[Test] sklearn 分类器（SVC/RandomForest）→ Classify ✅")

    def test_metrics(self):
        for path, wl in [
            ("sklearn.metrics.mean_squared_error", "Mean[(#1-#2)^2]"),
            ("sklearn.metrics.r2_score", "1 - Total"),
        ]:
            r = self._r(path)
            self.assertIn(wl[:8], r["wolfram_function"])
        print(f"\n[Test] sklearn 评估指标（MSE/R²）✅")

    def test_model_selection(self):
        r = self._r("sklearn.model_selection.train_test_split")
        self.assertIsNotNone(r)
        print(f"\n[Test] sklearn.model_selection.train_test_split ✅")

    def test_sklearn_total(self):
        rules = self.repo.search_rules("sklearn")
        self.assertGreater(len(rules), 40)
        print(f"\n[Test] sklearn 总规则数 ✅  {len(rules)} 条")


# ═══════════════════════════════════════════════════════
#  11. Matplotlib / Seaborn
# ═══════════════════════════════════════════════════════
class TestMatplotlib(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata import MetadataRepository
        self.repo = MetadataRepository(MAPPINGS_DIR)

    def _r(self, path):
        r = self.repo.get_rule(path)
        self.assertIsNotNone(r, f"规则未找到：{path}")
        return r

    def test_basic_plots(self):
        for path, wl in [
            ("matplotlib.pyplot.plot","ListLinePlot"),
            ("matplotlib.pyplot.scatter","ListPlot"),
            ("matplotlib.pyplot.hist","Histogram"),
            ("matplotlib.pyplot.bar","BarChart"),
            ("matplotlib.pyplot.boxplot","BoxWhiskerChart"),
            ("matplotlib.pyplot.pie","PieChart"),
        ]:
            r = self._r(path)
            self.assertEqual(r["wolfram_function"], wl)
        print(f"\n[Test] matplotlib 基础图形（plot/scatter/hist/bar/box/pie）✅")

    def test_advanced_plots(self):
        for path, wl in [
            ("matplotlib.pyplot.contour","ContourPlot"),
            ("matplotlib.pyplot.imshow","ArrayPlot"),
            ("matplotlib.pyplot.quiver","VectorPlot"),
            ("matplotlib.pyplot.streamplot","StreamPlot"),
        ]:
            r = self._r(path)
            self.assertEqual(r["wolfram_function"], wl)
        print(f"\n[Test] matplotlib 高级图形（contour/imshow/quiver/stream）✅")

    def test_3d_plots(self):
        r = self._r("matplotlib.pyplot.plot_surface")
        self.assertEqual(r["wolfram_function"], "Plot3D")
        r2 = self._r("matplotlib.pyplot.scatter3D")
        self.assertEqual(r2["wolfram_function"], "ListPointPlot3D")
        print(f"\n[Test] matplotlib 3D图形（surface→Plot3D, scatter3D）✅")

    def test_seaborn(self):
        for path, wl in [
            ("seaborn.heatmap","MatrixPlot"),
            ("seaborn.boxplot","BoxWhiskerChart"),
            ("seaborn.violinplot","DistributionChart"),
            ("seaborn.kdeplot","SmoothHistogram"),
        ]:
            r = self._r(path)
            self.assertEqual(r["wolfram_function"], wl)
        print(f"\n[Test] seaborn（heatmap/boxplot/violin/kde）✅")

    def test_matplotlib_total(self):
        rules = self.repo.search_rules("matplotlib")
        sns_rules = self.repo.search_rules("seaborn")
        total = len(rules) + len(sns_rules)
        self.assertGreater(total, 40)
        print(f"\n[Test] matplotlib+seaborn 总规则数 ✅  {total} 条")


# ═══════════════════════════════════════════════════════
#  8. Pandas — WolframDataFrame 端到端
# ═══════════════════════════════════════════════════════
class TestGrandTotal(unittest.TestCase):
    def test_grand_total(self):
        from py2wl.compat._core.metadata import MetadataRepository
        repo = MetadataRepository(MAPPINGS_DIR)
        n = len(repo.all_rules)
        self.assertGreater(n, 800, f"期望 >800 条，实际 {n}")
        print(f"\n{'='*60}")
        print(f"  全库映射规则总数：{n} 条")
        print(f"{'='*60}")

        # 分库统计
        libs = {}
        for r in repo.all_rules:
            ns = r["python_path"].split(".")[0]
            libs[ns] = libs.get(ns, 0) + 1
        for ns, cnt in sorted(libs.items(), key=lambda x:-x[1]):
            print(f"  {ns:<20} {cnt:>4} 条")

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestAllMappingsLoad,
        TestNumpyProxy,
        TestSympyProxy,
        TestTorchProxy,
        TestTensorflowProxy,
        TestPerfMonitoring,
        TestScipy,
        TestPandas,
        TestNumpyExtra,
        TestSklearn,
        TestMatplotlib,
        TestGrandTotal,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
