"""
test_compat.py — 核心组件单元测试
覆盖：MetadataRepository / Converters / ResolutionEngine / LibraryProxy
MockKernel 写 WXF-stub 文件（纯文本占位，同时 mock from_wxf 绕过 wolframclient）
"""
import sys, os, unittest
from pathlib import Path

ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

for k in list(sys.modules):
    if "py2wl" in k:
        del sys.modules[k]

MAPPINGS_DIR = str(Path(__file__).parent.parent / "py2wl" / "compat" / "mappings")
os.environ["WOLFRAM_MAPPINGS_DIR"] = MAPPINGS_DIR


# ── MockKernel：拦截 Export[path, expr, "WXF"]，写占位文本 ─────
class MockKernel:
    RESULTS = {
        "Fourier":     [complex(1,0), complex(0,1)],
        "LinearSolve": [1.0, 2.0],
        "Total":       10.0,
        "Mean":        2.5,
        "Sort":        [1,2,3,4,5],
        "Det":         1.0,
        "Norm":        3.74166,
        "Pi":          3.141592653589793,
        "N":           3.141592653589793,
    }

    def evaluate(self, expr) -> object:
        expr_str = str(expr)
        for k, v in self.RESULTS.items():
            if k in expr_str:
                return v
        return 42.0

    def evaluate_many(self, exprs) -> list:
        return [self.evaluate(e) for e in exprs]


# ══════════════════════════════════════════════════════
class TestMetadata(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata import MetadataRepository
        MetadataRepository._instance = None
        self.repo = MetadataRepository(MAPPINGS_DIR)

    def test_load(self):
        self.assertGreater(len(self.repo.all_rules), 800)
        print(f"\n[Test] YAML 加载 ✅  {len(self.repo.all_rules)} 条规则")

    def test_exact(self):
        rule = self.repo.get_rule("numpy.fft.fft")
        self.assertIsNotNone(rule)
        self.assertEqual(rule["wolfram_function"], "Fourier")
        print(f"\n[Test] 精确查找 ✅  numpy.fft.fft → Fourier")

    def test_constant_flag(self):
        rule = self.repo.get_rule("sympy.pi")
        self.assertIsNotNone(rule)
        self.assertTrue(rule.get("constant"), "sympy.pi 应有 constant: true")
        print(f"\n[Test] constant 标志 ✅  sympy.pi")

    def test_numeric_flag(self):
        rule = self.repo.get_rule("numpy.mean")
        self.assertIsNotNone(rule)
        self.assertTrue(rule.get("numeric"), "numpy.mean 应有 numeric: true")
        rule_sp = self.repo.get_rule("sympy.solve")
        self.assertFalse(rule_sp.get("numeric"), "sympy.solve 不应有 numeric: true")
        print(f"\n[Test] numeric 标志 ✅  numpy.mean=True, sympy.solve=False")

    def test_tag_search(self):
        results = self.repo.search_rules("fft")
        self.assertGreater(len(results), 0)
        print(f"\n[Test] 标签搜索 fft ✅  {len(results)} 条")

    def test_missing(self):
        self.assertIsNone(self.repo.get_rule("numpy.abc.xyz"))
        print("\n[Test] 不存在路径 ✅  正确返回 None")


# ══════════════════════════════════════════════════════
class TestConverters(unittest.TestCase):
    def test_to_wl_list(self):
        from py2wl.compat._core.converters import to_wl_list
        self.assertEqual(to_wl_list([1, 2, 3]), "{1, 2, 3}")
        print("\n[Test] to_wl_list ✅")

    def test_to_wl_matrix(self):
        from py2wl.compat._core.converters import to_wl_matrix
        r = to_wl_matrix([[1, 2], [3, 4]])
        self.assertIn("{1, 2}", r)
        print(f"\n[Test] to_wl_matrix ✅  {r}")

    def test_to_wl_matrix_and_vector(self):
        from py2wl.compat._core.converters import to_wl_matrix_and_vector
        r = to_wl_matrix_and_vector(([[1, 0], [0, 1]], [3, 4]))
        self.assertIn("3", r)
        print(f"\n[Test] to_wl_matrix_and_vector ✅")

    def test_output_converters_registry(self):
        from py2wl.compat._core.converters import OUTPUT_CONVERTERS
        # WXF-only：只应有 from_wxf 和 from_wl_image
        self.assertIn("from_wxf",      OUTPUT_CONVERTERS)
        self.assertIn("from_wl_image", OUTPUT_CONVERTERS)
        self.assertNotIn("from_wl_csv",     OUTPUT_CONVERTERS)
        self.assertNotIn("from_wl_numpy",   OUTPUT_CONVERTERS)
        self.assertNotIn("from_wl_scalar",  OUTPUT_CONVERTERS)
        print(f"\n[Test] OUTPUT_CONVERTERS WXF-only ✅  {list(OUTPUT_CONVERTERS)}")

    def test_from_wl_image_missing(self):
        from py2wl.compat._core.converters import from_wl_image
        with self.assertRaises(FileNotFoundError):
            from_wl_image("/nonexistent/path.png")
        print("\n[Test] from_wl_image FileNotFoundError ✅")


# ══════════════════════════════════════════════════════
class TestResolver(unittest.TestCase):
    def setUp(self):
        from py2wl.compat._core.metadata import MetadataRepository
        from py2wl.compat._core.resolver import ResolutionEngine
        MetadataRepository._instance = None
        ResolutionEngine._instance   = None
        self.res = ResolutionEngine(MetadataRepository(MAPPINGS_DIR))

    def test_resolve(self):
        rule = self.res.resolve("numpy.fft.fft")
        self.assertIsNotNone(rule)
        print(f"\n[Test] Resolver 精确匹配 ✅  → {rule['wolfram_function']}")

    def test_wxf_file_mode(self):
        """WSTP 模式：build_wl_expr 返回纯 WL 核心表达式，无 Export 包装。"""
        for p in ["numpy.fft.fft", "numpy.linalg.det", "numpy.mean", "sympy.solve"]:
            rule = self.res.resolve(p)
            expr = self.res.build_wl_expr(rule, ([1, 2, 3, 4],), {})
            self.assertIsInstance(expr, str, f"{p}: 应返回字符串")
            self.assertNotIn("Export[", expr, f"{p}: WSTP 模式不含 Export")
        print(f"\n[Test] build_wl_expr 返回纯核心表达式 ✅")

    def test_numeric_wraps_N(self):
        """numeric: true 的规则，表达式应包含 N[...]。"""
        rule = self.res.resolve("numpy.mean")
        self.assertTrue(rule.get("numeric"))
        expr = self.res.build_wl_expr(rule, ([1, 2, 3],), {})
        self.assertIn("N[", expr)
        print(f"\n[Test] numeric → N[] 包装 ✅  {expr[:60]}")

    def test_sympy_no_N(self):
        """sympy 规则不应有 N[] 包装。"""
        rule = self.res.resolve("sympy.solve")
        self.assertFalse(rule.get("numeric"))
        expr = self.res.build_wl_expr(rule, (["x**2 - 1", "x"],), {})
        self.assertNotIn("N[", expr)
        print(f"\n[Test] sympy 无 N[] 包装 ✅")

    def test_image_mode(self):
        """from_wl_image 规则：output_converter 为 from_wl_image。"""
        rule = self.res.resolve("matplotlib.pyplot.plot")
        self.assertEqual(rule.get("output_converter"), "from_wl_image")
        expr = self.res.build_wl_expr(rule, ([1, 2, 3],), {})
        self.assertIsInstance(expr, str)
        print(f"\n[Test] image 模式 output_converter ✅")

    def test_missing(self):
        self.assertIsNone(self.res.resolve("numpy.xyz.unknown"))
        print("\n[Test] Resolver 未知路径 ✅  返回 None")


# ══════════════════════════════════════════════════════
class TestProxy(unittest.TestCase):
    def setUp(self):
        for k in list(sys.modules):
            if "py2wl" in k:
                del sys.modules[k]
        from py2wl.compat._core.metadata import MetadataRepository
        from py2wl.compat._core.resolver import ResolutionEngine
        from py2wl.compat._state import _state
        MetadataRepository._instance = None
        ResolutionEngine._instance   = None
        _state["kernel"] = MockKernel()
        # mock from_wxf 绕过 wolframclient

    def test_fft_end_to_end(self):
        import py2wl.compat.numpy as _np
        result = _np.fft.fft([1, 0, 1, 0])
        self.assertIsNotNone(result)
        print(f"\n[Test] np.fft.fft 端到端 ✅  result={result!r}")

    def test_unknown_raises(self):
        import py2wl.compat.numpy as _np
        with self.assertRaises(AttributeError):
            _np.abc.xyz([1, 2, 3])
        print("\n[Test] 未知函数 AttributeError ✅")

    def test_constant_not_callable(self):
        """sympy.pi 应返回值，不是 callable。"""
        import py2wl.compat.sympy as sp
        val = sp.pi
        self.assertFalse(callable(val), f"sp.pi 不应是 callable，得到 {type(val)}")
        print(f"\n[Test] sympy.pi 非 callable ✅  {val!r}")

    def test_function_is_callable(self):
        """sympy.sin 应是 callable。"""
        import py2wl.compat.sympy as sp
        self.assertTrue(callable(sp.sin))
        print(f"\n[Test] sympy.sin 是 callable ✅")

    def test_scipy_proxy_exists(self):
        """scipy 代理文件应能正常导入。"""
        import py2wl.compat.scipy as sc
        self.assertIn("scipy", repr(sc))
        print(f"\n[Test] scipy 代理 ✅  {repr(sc)}")

    def test_repr(self):
        import py2wl.compat.numpy as _np
        self.assertIn("numpy", repr(_np))
        print(f"\n[Test] repr ✅  {repr(_np)}")


class TestConstantCaching(unittest.TestCase):
    """验证 constant:true 规则的实例级缓存行为。"""

    def setUp(self):
        from py2wl.compat._state import _state
        from py2wl.compat._core.metadata import MetadataRepository
        from py2wl.compat._core.resolver import ResolutionEngine
        MetadataRepository._instance = None
        ResolutionEngine._instance   = None
        _state["kernel"]   = MockKernel()
        _state["resolver"] = None
        from py2wl.compat._proxy_base import cache_clear
        cache_clear()

    def test_constant_returns_value_not_callable(self):
        """constant:true 属性访问返回值，不是 callable。"""
        from py2wl.compat import numpy as np
        val = np.pi
        self.assertFalse(callable(val), "np.pi 不应是 callable，应直接返回数值")

    def test_constant_cached_on_second_access(self):
        """同一代理实例第二次访问常量不再调用 evaluate。"""
        from py2wl.compat._proxy_base import LibraryProxy
        from py2wl.compat._state import _state
        call_count = [0]
        orig = _state["kernel"].evaluate
        def counting_evaluate(expr):
            if "Pi" in expr: call_count[0] += 1
            return orig(expr)
        _state["kernel"].evaluate = counting_evaluate
        proxy = LibraryProxy("numpy")
        for _ in range(3):
            proxy.pi
        self.assertEqual(call_count[0], 1,
            f"np.pi 应只触发 1 次内核调用，实际 {call_count[0]} 次")

    def test_constant_cached_in_loop(self):
        """列表推导中重复访问常量只触发一次内核调用（修复卡死问题）。"""
        from py2wl.compat._proxy_base import LibraryProxy
        from py2wl.compat._state import _state
        call_count = [0]
        orig = _state["kernel"].evaluate
        def counting_evaluate(expr):
            if "Pi" in expr: call_count[0] += 1
            return orig(expr)
        _state["kernel"].evaluate = counting_evaluate
        proxy = LibraryProxy("numpy")
        results = [proxy.pi * i for i in range(40)]
        self.assertEqual(call_count[0], 1,
            f"40 次循环中 np.pi 应只触发 1 次内核调用，实际 {call_count[0]} 次")
        self.assertEqual(len(results), 40)

    def test_different_instances_independent_cache(self):
        """不同代理实例各自独立缓存。"""
        from py2wl.compat._proxy_base import LibraryProxy
        p1, p2 = LibraryProxy("numpy"), LibraryProxy("numpy")
        self.assertEqual(p1.pi, p2.pi)
        c1 = object.__getattribute__(p1, "_const_cache")
        c2 = object.__getattribute__(p2, "_const_cache")
        self.assertIsNot(c1, c2)

    def test_non_constant_not_cached(self):
        """普通函数不写入 _const_cache。"""
        from py2wl.compat._proxy_base import LibraryProxy
        proxy = LibraryProxy("numpy")
        _ = proxy.mean
        cache = object.__getattribute__(proxy, "_const_cache")
        self.assertNotIn("mean", cache)

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestMetadata, TestConverters, TestResolver, TestProxy, TestConstantCaching]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

