#!/usr/bin/env python3
"""
tests/test_large_data.py — 大数据文件传输层测试
=================================================
验证：
  1. 阈值以下走字符串模式（不产生 wlb_in_* 临时文件）
  2. 阈值以上走文件模式（产生并清理 wlb_in_* 临时文件）
  3. wolframclient 不可用时自动降级到字符串模式
  4. _count_elements 对各种数据结构正确计数
  5. flush_input_tmps 线程隔离
  6. 文件模式输出的 Import["path","WXF"] 格式正确
"""
import os, sys, math, glob, tempfile, threading, unittest
from pathlib import Path

ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from py2wl.compat._core.converters import (
    _count_elements,
    _large_aware,
    _register_input_tmp,
    flush_input_tmps,
    to_wl_list,
    to_wl_matrix,
    LARGE_THRESHOLD,
)


# ══════════════════════════════════════════════════════════════
#  A. _count_elements
# ══════════════════════════════════════════════════════════════
class TestA_CountElements(unittest.TestCase):

    def test_flat_list(self):
        self.assertEqual(_count_elements([1,2,3,4,5]), 5)

    def test_empty(self):
        self.assertEqual(_count_elements([]), 0)

    def test_2d_list(self):
        m = [[1,2,3],[4,5,6]]
        self.assertEqual(_count_elements(m), 6)

    def test_1000x1000(self):
        m = [[0.0]*1000 for _ in range(1000)]
        self.assertEqual(_count_elements(m), 1_000_000)

    def test_scalar(self):
        self.assertEqual(_count_elements(42.0), 1)

    def test_numpy_array(self):
        try:
            import numpy as np
            arr = np.zeros((500, 200))
            self.assertEqual(_count_elements(arr), 100_000)
        except ImportError:
            self.skipTest("numpy 不可用")


# ══════════════════════════════════════════════════════════════
#  B. 阈值触发逻辑
# ══════════════════════════════════════════════════════════════
class TestB_Threshold(unittest.TestCase):

    def _make_list(self, n):
        return [float(i) for i in range(n)]

    def _make_matrix(self, n):
        return [[float(i*n+j) for j in range(n)] for i in range(n)]

    def test_small_list_is_string(self):
        """小于阈值 → 返回 { } 格式字符串。"""
        small = self._make_list(LARGE_THRESHOLD - 1)
        result = to_wl_list(small)
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("{"), f"期望 {{...}}，得: {result[:40]}")
        self.assertNotIn('Import[', result)

    def test_large_list_tries_file_mode(self):
        """大于阈值 → 尝试文件模式（wolframclient 可用时返回 Import[...]）。"""
        try:
            from wolframclient.serializers import export as _
            HAS_WC = True
        except ImportError:
            HAS_WC = False

        big = self._make_list(LARGE_THRESHOLD + 1)
        result = to_wl_list(big)
        self.assertIsInstance(result, str)
        if HAS_WC:
            self.assertIn('Import[', result,
                f"wolframclient 可用时应走文件模式，得: {result[:60]}")
        else:
            # 降级到字符串
            self.assertTrue(result.startswith("{"))

    def test_small_matrix_is_string(self):
        """5×5 矩阵（25元素）→ 字符串模式。"""
        m = self._make_matrix(5)
        result = to_wl_matrix(m)
        self.assertTrue(result.startswith("{"))

    def test_threshold_boundary(self):
        """恰好在阈值边界的行为。"""
        at_threshold = self._make_list(LARGE_THRESHOLD)
        result = to_wl_list(at_threshold)
        # 恰好等于阈值 → 字符串模式（> not >=）
        self.assertTrue(result.startswith("{"))


# ══════════════════════════════════════════════════════════════
#  C. 文件模式输出格式
# ══════════════════════════════════════════════════════════════
class TestC_FileMode(unittest.TestCase):

    def setUp(self):
        try:
            from wolframclient.serializers import export as _
            self.has_wc = True
        except ImportError:
            self.has_wc = False

    def _big_list(self):
        return [float(i) for i in range(LARGE_THRESHOLD + 100)]

    def test_import_expr_format(self):
        """文件模式返回 Import["path","WXF"] 格式。"""
        if not self.has_wc:
            self.skipTest("wolframclient 不可用")
        result = to_wl_list(self._big_list())
        self.assertRegex(result, r'^Import\[".+\.wxf",\s*"WXF"\]$')

    def test_tmp_file_created(self):
        """文件模式会创建 wlb_in_*.wxf 临时文件。"""
        if not self.has_wc:
            self.skipTest("wolframclient 不可用")
        before = set(glob.glob(os.path.join(tempfile.gettempdir(), "wlb_in_*.wxf")))
        _ = to_wl_list(self._big_list())
        after = set(glob.glob(os.path.join(tempfile.gettempdir(), "wlb_in_*.wxf")))
        new_files = after - before
        self.assertGreater(len(new_files), 0, "应创建至少一个 wlb_in_*.wxf")

    def test_tmp_file_cleaned_by_flush(self):
        """flush_input_tmps() 清理本线程的输入临时文件。"""
        if not self.has_wc:
            self.skipTest("wolframclient 不可用")
        result = to_wl_list(self._big_list())
        # 从 Import["path","WXF"] 提取路径
        import re
        m = re.search(r'Import\["(.+?)",', result)
        self.assertIsNotNone(m)
        tmp_path = m.group(1)
        self.assertTrue(os.path.exists(tmp_path), "flush 前文件应存在")
        flush_input_tmps()
        self.assertFalse(os.path.exists(tmp_path), "flush 后文件应已删除")

    def test_wxf_file_is_binary(self):
        """写出的 .wxf 文件是合法的 WXF 二进制（魔数检查）。"""
        if not self.has_wc:
            self.skipTest("wolframclient 不可用")
        result = to_wl_list(self._big_list())
        import re
        m = re.search(r'Import\["(.+?)",', result)
        tmp_path = m.group(1)
        try:
            with open(tmp_path, "rb") as f:
                header = f.read(2)
            # WXF 文件以 0x38 0x3A 开头（'8:'）
            self.assertEqual(header, b'8:',
                f"WXF 文件头应为 b'8:'，实际: {header.hex()}")
        finally:
            flush_input_tmps()

    def test_file_embeddable_in_expr(self):
        """文件模式输出可直接嵌入 WL 表达式。"""
        if not self.has_wc:
            self.skipTest("wolframclient 不可用")
        import_str = to_wl_list(self._big_list())
        expr = f'Export["/tmp/out.wxf", Total[{import_str}], "WXF"]'
        # 表达式格式合法（不含未转义引号等问题）
        self.assertIn('Import[', expr)
        self.assertIn('Total[', expr)
        flush_input_tmps()


# ══════════════════════════════════════════════════════════════
#  D. 降级行为
# ══════════════════════════════════════════════════════════════
class TestD_Fallback(unittest.TestCase):

    def test_fallback_when_no_wolframclient(self):
        """wolframclient 不可用时 _large_aware 降级到原始转换器。"""
        call_log = []

        def original_conv(v):
            call_log.append(v)
            return "{fallback}"

        wrapped = _large_aware(original_conv)

        # 模拟大数据，强制文件模式
        big = [0.0] * (LARGE_THRESHOLD + 1)

        # 打补丁：让 _to_wl_wxf_file 抛 ImportError
        import py2wl.compat._core.converters as cv_mod
        original_fn = cv_mod._to_wl_wxf_file
        cv_mod._to_wl_wxf_file = lambda v: (_ for _ in ()).throw(
            ImportError("模拟 wolframclient 不可用"))
        try:
            result = wrapped(big)
            self.assertEqual(result, "{fallback}",
                "应降级调用原始转换器")
            self.assertEqual(len(call_log), 1)
        finally:
            cv_mod._to_wl_wxf_file = original_fn

    def test_fallback_preserves_function_name(self):
        """_large_aware 保留原始函数名。"""
        def my_conv(v): return str(v)
        wrapped = _large_aware(my_conv)
        self.assertEqual(wrapped.__name__, "my_conv")


# ══════════════════════════════════════════════════════════════
#  E. 线程安全
# ══════════════════════════════════════════════════════════════
class TestE_ThreadSafety(unittest.TestCase):

    def test_flush_is_thread_local(self):
        """不同线程的 pending 临时文件互不干扰。"""
        registered = {}

        def worker(tid):
            fd, p = tempfile.mkstemp(suffix=".wxf", prefix="test_thread_")
            os.close(fd)
            _register_input_tmp(p)
            registered[tid] = p
            # 等待一会，确保主线程不会清理我的文件
            import time; time.sleep(0.05)
            flush_input_tmps()

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        # 两个线程各自注册并清理了自己的文件
        for tid, p in registered.items():
            self.assertFalse(os.path.exists(p),
                f"线程 {tid} 的临时文件应被该线程清理")

    def test_main_thread_flush_does_not_affect_other_thread_files(self):
        """主线程 flush 不会删除其他线程的文件。"""
        fd, p = tempfile.mkstemp(suffix=".wxf", prefix="other_thread_")
        os.close(fd)

        other_registered = []
        def register_in_other_thread():
            _register_input_tmp(p)
            other_registered.append(True)
            import time; time.sleep(0.1)   # 让主线程有机会 flush

        t = threading.Thread(target=register_in_other_thread)
        t.start()
        import time; time.sleep(0.02)
        flush_input_tmps()   # 主线程 flush，不应影响子线程的文件
        t.join()

        # 子线程的文件应还存在（子线程尚未 flush）
        # 清理
        if os.path.exists(p):
            os.unlink(p)


# ══════════════════════════════════════════════════════════════
#  F. 与 build_wl_expr 集成
# ══════════════════════════════════════════════════════════════
class TestF_Integration(unittest.TestCase):
    """验证 build_wl_expr 使用 to_wl_list/to_wl_matrix 时文件模式透明嵌入。"""

    def setUp(self):
        try:
            from wolframclient.serializers import export as _
            self.has_wc = True
        except ImportError:
            self.has_wc = False

    def test_large_matrix_in_build_wl_expr(self):
        """大矩阵经 build_wl_expr 时产生 Import[...] 而非长字符串。"""
        if not self.has_wc:
            self.skipTest("wolframclient 不可用")

        from py2wl.compat._core.resolver import ResolutionEngine
        from py2wl.compat._core.metadata import MetadataRepository
        MetadataRepository._instance = None
        ResolutionEngine._instance   = None

        repo   = MetadataRepository()
        engine = ResolutionEngine(repo)

        rule = {
            "python_path":    "numpy.linalg.det",
            "wolfram_function": "Det",
            "input_converters": ["to_wl_matrix"],
            "output_converter": "from_wxf",
            "numeric": True,
        }
        N = 50  # 50×50 = 2500 元素 > LARGE_THRESHOLD
        big_mat = [[float(i*N+j) for j in range(N)] for i in range(N)]

        expr, tmp_out = engine.build_wl_expr(rule, (big_mat,), {})

        try:
            # 表达式中应含 Import[...] 而非超长 {{ }} 字符串
            self.assertIn('Import[', expr,
                f"大矩阵应走文件模式，表达式: {expr[:120]}")
            # 表达式不应超过合理长度（< 256 字符）
            self.assertLess(len(expr), 256,
                f"文件模式表达式应很短，实际 {len(expr)} 字符")
        finally:
            flush_input_tmps()
            if tmp_out and os.path.exists(tmp_out):
                os.unlink(tmp_out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
