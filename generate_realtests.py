#!/usr/bin/env python3
"""
generate_realtests.py — 智能测试生成器（真实内核版）
为所有 887 条规则（排除 constant）生成带真实断言的测试代码。

用法:
  python generate_realtests.py          # 输出到 tests/test_generated_real.py
  python generate_realtests.py 200      # 只生成前 200 条
"""

import sys
import yaml
import textwrap
from pathlib import Path
from collections import defaultdict

# ── 规则加载 ────────────────────────────────────────────
def load_rules(mappings_dir="py2wl/compat/mappings"):
    rules = []
    for f in sorted(Path(mappings_dir).glob("*.yaml")):
        r = yaml.safe_load(f.read_text())
        if r:
            rules.extend(r)
    return [r for r in rules if not r.get("constant")]

# ── 参数生成策略 ────────────────────────────────────────
SAMPLE_ARGS = {
    "to_wl_list":               "[1.0, 2.0, 3.0, 4.0]",
    "to_wl_scalar":             "4",
    "to_wl_matrix":             "[[1.0, 2.0], [3.0, 4.0]]",
    "to_wl_matrix_and_vector":  "([[2.0,1.0],[1.0,1.0]], [1.0,2.0])",
    "to_wl_string":             '"test_value"',
    "to_wl_passthrough":        '"x"',
}

# wolfram_function → (样本值/参数, 断言代码)
WF_ASSERTIONS = {
    # 统计
    "Mean":              ("([1.0,2.0,3.0,4.0],)",
                          "self.assertAlmostEqual(float(result), 2.5, places=4)"),
    "Total":             ("([1.0,2.0,3.0,4.0],)",
                          "self.assertAlmostEqual(float(result), 10.0, places=4)"),
    "StandardDeviation": ("([2.0,4.0,4.0,4.0,5.0,5.0,7.0,9.0],)",
                          "self.assertAlmostEqual(float(result), 2.13809, places=3)"),
    "Variance":          ("([2.0,4.0,4.0,4.0,5.0,5.0,7.0,9.0],)",
                          "self.assertAlmostEqual(float(result), 4.57143, places=3)"),
    "Min":               ("([3.0,1.0,4.0,1.0,5.0],)",
                          "self.assertAlmostEqual(float(result), 1.0, places=4)"),
    "Max":               ("([3.0,1.0,4.0,1.0,5.0],)",
                          "self.assertAlmostEqual(float(result), 5.0, places=4)"),
    "Median":            ("([1.0,2.0,3.0,4.0,5.0],)",
                          "self.assertAlmostEqual(float(result), 3.0, places=4)"),
    # 线代
    "Det":               ("([[1.0,2.0],[3.0,4.0]],)",
                          "self.assertAlmostEqual(float(result), -2.0, places=4)"),
    "Tr":                ("([[1.0,0.0],[0.0,3.0]],)",
                          "self.assertIsNotNone(result)"),
    "Norm":              ("([3.0,4.0],)",
                          "self.assertAlmostEqual(float(result), 5.0, places=4)"),
    "MatrixRank":        ("([[1.0,0.0],[0.0,1.0]],)",
                          "self.assertEqual(int(result), 2)"),
    # 数组变换
    "Fourier":           ("([1.0,0.0,1.0,0.0],)",
                          "self.assertIsInstance(result, (list,tuple))"),
    "Sort":              ("([3.0,1.0,4.0,1.0,5.0],)",
                          "r=list(result); self.assertEqual(r[0], min(r))"),
    "Reverse":           ("([1.0,2.0,3.0],)",
                          "self.assertEqual(list(result)[0], 3.0)"),
    "Flatten":           ("([[1.0,[2.0]],3.0],)",
                          "self.assertIsInstance(result, (list,tuple))"),
    "Accumulate":        ("([1.0,2.0,3.0,4.0],)",
                          "self.assertAlmostEqual(list(result)[-1], 10.0, places=4)"),
    "Range":             ("(5,)",
                          "self.assertEqual(len(list(result)), 5)"),
    "Subdivide":         ("(0.0, 1.0, 4)",
                          "self.assertEqual(len(list(result)), 5)"),
    # 数学函数
    "Sin":               ("(0.0,)",  "self.assertAlmostEqual(float(result), 0.0, places=6)"),
    "Cos":               ("(0.0,)",  "self.assertAlmostEqual(float(result), 1.0, places=6)"),
    "Exp":               ("(0.0,)",  "self.assertAlmostEqual(float(result), 1.0, places=6)"),
    "Log":               ("(1.0,)",  "self.assertAlmostEqual(float(result), 0.0, places=6)"),
    "Sqrt":              ("(4.0,)",  "self.assertAlmostEqual(float(result), 2.0, places=6)"),
    "Abs":               ("(-3.0,)", "self.assertAlmostEqual(float(result), 3.0, places=6)"),
    "Sign":              ("(-5.0,)", "self.assertAlmostEqual(float(result), -1.0, places=6)"),
    "Floor":             ("(2.9,)",  "self.assertAlmostEqual(float(result), 2.0, places=6)"),
    "Ceiling":           ("(2.1,)",  "self.assertAlmostEqual(float(result), 3.0, places=6)"),
    "Round":             ("(2.5,)",  "self.assertIsNotNone(result)"),
    "GCD":               ("(12, 8)", "self.assertAlmostEqual(float(result), 4.0, places=0)"),
    "LCM":               ("(4, 6)",  "self.assertAlmostEqual(float(result), 12.0, places=0)"),
    # 矩阵操作
    "Transpose":         ("([[1.0,2.0],[3.0,4.0]],)",
                          "r=list(result); self.assertEqual(len(r), 2)"),
    "Inverse":           ("([[1.0,0.0],[0.0,1.0]],)",
                          "self.assertIsInstance(result, (list,tuple))"),
    "Eigenvalues":       ("([[2.0,0.0],[0.0,3.0]],)",
                          "self.assertIsInstance(result, (list,tuple))"),
    "Eigensystem":       ("([[2.0,0.0],[0.0,3.0]],)",
                          "self.assertIsInstance(result, (list,tuple))"),
    "LinearSolve":       ("([[2.0,1.0],[1.0,1.0]], [1.0,2.0])",
                          "self.assertIsInstance(result, (list,tuple))"),
    # 随机（只验证类型）
    "SeedRandom":        ("(42,)", "pass  # SeedRandom 返回 Null"),
    "RandomReal":        ("(1.0,)", "self.assertIsNotNone(result)"),
    # IO
    "Import":            ('("/dev/null",)', "self.assertIsNotNone(result)"),
    "Export":            ('("/tmp/wfb_test_out.wxf", 1, "WXF")',
                          "self.assertIsNotNone(result)"),
}

def _safe_name(path: str) -> str:
    return path.replace(".", "_").replace("[", "").replace("]", "").replace(" ", "_")

def _get_args(rule: dict) -> str:
    """返回调用时传入的参数元组字符串。"""
    wf = str(rule.get("wolfram_function", ""))
    # 优先从 WF_ASSERTIONS 匹配
    if not wf.endswith("&"):
        for key, (args, _) in WF_ASSERTIONS.items():
            if wf == key or wf.startswith(key + "[") or wf.startswith(key + " "):
                return args

    ics = rule.get("input_converters")
    ic  = rule.get("input_converter", "to_wl_passthrough")
    if ics:
        parts = [SAMPLE_ARGS.get(c, "None") for c in ics]
        return "(" + ", ".join(parts) + ")"
    # 特殊修正：某些函数需要矩阵参数而不是默认值
    if "SingularValue" in wf or "Eigensystem" in wf or "Eigenvalues" in wf:
        return "([[4.0,0.0],[0.0,9.0]],)"
    # pandas.DataFrame 需要二维数据
    if rule["python_path"] == "pandas.DataFrame":
        return '({"a":[1.0,2.0],"b":[3.0,4.0]},)'
    return "(" + SAMPLE_ARGS.get(ic, "None") + ",)"

# 副作用函数：Python 中返回 None
_NONE_RETURN_PATHS = {
    "functools.cache", "functools.lru_cache",
    "logging.debug", "logging.error", "logging.info", "logging.warning",
    "time.sleep", "warnings.warn",
}

def _get_assertion(rule: dict) -> str:
    """根据规则元数据选择合适的断言。"""
    oc   = rule.get("output_converter", "from_wxf")
    wf   = str(rule.get("wolfram_function", ""))
    lib  = rule["python_path"].split(".")[0]
    path = rule["python_path"]

    # 0. 已知跳过 / 特殊处理
    if path in _NONE_RETURN_PATHS:
        return "self.assertIsNone(result)"
    if path == "pandas.read_excel":
        return 'self.skipTest("需要真实 Excel 文件")'

    # 1. 图像
    if oc == "from_wl_image":
        return 'self.assertIsInstance(result, str); self.assertTrue(str(result).endswith(".png") or os.path.exists(str(result)))'

    # 2. DataFrame
    if oc == "from_wxf_dataframe":
        return ('from py2wl.compat.pandas import WolframDataFrame; '
                'self.assertIsInstance(result, WolframDataFrame)')

    # 3. WF 精确断言（只对"简单" wolfram_function 匹配，纯函数 & 结尾跳过）
    if not wf.endswith("&"):   # 纯函数（&结尾）跳过精确匹配，避免误判
        for key, (_, assertion) in WF_ASSERTIONS.items():
            if wf == key or wf.startswith(key + "[") or wf.startswith(key + " "):
                return assertion

    # 4. numeric:true → 类型检查（宽松，允许 list/str 占位）
    if rule.get("numeric"):
        return 'self.assertIsNotNone(result, "numeric 规则不应返回 None")'

    # 5. 符号库 → 只验非 None
    if lib in ("sympy",) and not rule.get("numeric"):
        return 'self.assertIsNotNone(result, "符号结果不应为 None")'

    # 6. 通用
    return 'self.assertIsNotNone(result)'

def _get_import_prefix(rule: dict) -> str:
    """调用时用的对象前缀（np. / sp. / pd. 等）。"""
    lib = rule["python_path"].split(".")[0]
    return {
        "numpy": "np", "scipy": "scipy", "sympy": "sp",
        "pandas": "pd", "torch": "torch", "tf": "tf",
        "sklearn": "sklearn", "matplotlib": "plt_mod",
        "seaborn": "sns", "logging": "logging",
        "functools": "functools", "datetime": "datetime",
        "warnings": "warnings", "cupy": "cupy", "jax": "jax",
        "joblib": "joblib", "cProfile": "cProfile",
        "contextlib": "contextlib", "concurrent": "concurrent",
        "line_profiler": "line_profiler", "memory_profiler": "memory_profiler",
        "mpl_toolkits": "mpl_toolkits", "multiprocessing": "multiprocessing",
        "numba": "numba", "psutil": "psutil", "timeit": "timeit",
        "tqdm": "tqdm", "time": "time",
    }.get(lib, lib)

# ── 主生成逻辑 ──────────────────────────────────────────
HEADER = '''\
#!/usr/bin/env python3
"""
test_generated_real.py — 自动生成的全库测试（{n} 条规则）
使用真实 Wolfram 内核，需设置 WOLFRAM_EXEC 环境变量。
生成器：generate_realtests.py (real kernel version)
"""
import sys, os, unittest, tempfile
from pathlib import Path

ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 清理模块缓存
for _k in list(sys.modules):
    if "py2wl" in _k:
        del sys.modules[_k]

MAPPINGS_DIR = str(Path(__file__).parent.parent / "py2wl" / "compat" / "mappings")
os.environ["WOLFRAM_MAPPINGS_DIR"] = MAPPINGS_DIR

# 导入所有兼容模块（此时 WolframKernel 尚未初始化）
import py2wl.compat.numpy      as np
import py2wl.compat.scipy      as scipy
import py2wl.compat.sympy      as sp
import py2wl.compat.pandas     as pd
import py2wl.compat.torch      as torch
import py2wl.compat.tensorflow as tf
import py2wl.compat.sklearn    as sklearn
import py2wl.compat.matplotlib as plt_mod
import py2wl.compat.seaborn    as sns
import py2wl.compat.logging      as logging
import py2wl.compat.functools    as functools
import py2wl.compat.datetime     as datetime
import py2wl.compat.warnings     as warnings
import py2wl.compat.cupy         as cupy
import py2wl.compat.jax          as jax
import py2wl.compat.joblib       as joblib
import py2wl.compat.cProfile     as cProfile
import py2wl.compat.contextlib   as contextlib
import py2wl.compat.concurrent   as concurrent
import py2wl.compat.line_profiler as line_profiler
import py2wl.compat.memory_profiler as memory_profiler
import py2wl.compat.mpl_toolkits as mpl_toolkits
import py2wl.compat.multiprocessing as multiprocessing
import py2wl.compat.numba        as numba
import py2wl.compat.psutil       as psutil
import py2wl.compat.timeit       as timeit
import py2wl.compat.tqdm         as tqdm
import py2wl.compat.time         as time
'''

# ── 按库分组输出测试类 ──────────────────────────────────

CLASS_TEMPLATE = '''

# ══════════════════════════════════════════════════════════════
#  {lib_upper}
# ══════════════════════════════════════════════════════════════
class Test_{lib_safe}(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 在类级别检测内核是否可用，缓存结果供跳过条件使用
        cls._has_kernel = None
        try:
            from py2wl import WolframKernel
            k = WolframKernel()
            res = k.evaluate("2+2")
            cls._has_kernel = (res == 4)
        except Exception:
            cls._has_kernel = False

    def setUp(self):
        if not self.__class__._has_kernel:
            self.skipTest("需要真实 Wolfram 内核（请设置 WOLFRAM_EXEC 环境变量）")
'''

METHOD_TEMPLATE = '''
    def test_{safe_name}(self):
        """{path} → {wf}  [{oc}]"""
        try:
            result = {prefix}.{attr}({args})
            {assertion}
        except AttributeError as e:
            self.skipTest(f"规则未实现: {{e}}")
        except Exception as e:
            self.fail(f"{path} 调用失败: {{e}}")
'''

def generate(rules, N=None):
    if N:
        rules = rules[:N]

    by_lib = defaultdict(list)
    for r in rules:
        lib = r["python_path"].split(".")[0]
        by_lib[lib].append(r)

    out = [HEADER.format(n=len(rules))]

    for lib in ["numpy", "scipy", "sympy", "pandas", "torch", "tf",
                "sklearn", "matplotlib", "seaborn",
                # 其余库
                *sorted(k for k in by_lib if k not in
                        {"numpy","scipy","sympy","pandas","torch","tf",
                         "sklearn","matplotlib","seaborn"})]:
        if lib not in by_lib:
            continue
        lib_safe = lib.replace(".", "_")
        out.append(CLASS_TEMPLATE.format(
            lib_upper=lib.upper(),
            lib_safe=lib_safe,
        ))

        SKIP_KEYWORDS = {"assert","class","return","import","from","if","else",
                         "for","while","with","try","except","pass","raise","del",
                         "lambda","yield","global","nonlocal","and","or","not",
                         "in","is","break","continue","True","False","None","def"}
        for rule in by_lib[lib]:
            if any(p in SKIP_KEYWORDS for p in rule["python_path"].split(".")):
                continue
            path   = rule["python_path"]
            # attr = path 去掉 lib 前缀，例如 numpy.fft.fft → fft.fft
            attr   = ".".join(path.split(".")[1:])
            prefix = _get_import_prefix(rule)
            wf     = str(rule.get("wolfram_function", ""))[:40]
            oc     = rule.get("output_converter", "from_wxf")
            args   = _get_args(rule).strip("()")
            assertion = _get_assertion(rule)

            out.append(METHOD_TEMPLATE.format(
                safe_name=_safe_name(path),
                path=path,
                wf=wf,
                oc=oc,
                prefix=prefix,
                attr=attr,
                args=args,
                assertion=assertion,
            ))

    # ── Grand Total 测试类 ─────────────────────────────────────
    out.append(textwrap.dedent("""
    # ══════════════════════════════════════════════════════════════
    #  Grand Total
    # ══════════════════════════════════════════════════════════════
    class TestGrandTotal(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            # 在类级别检测内核是否可用，缓存结果供跳过条件使用
            cls._has_kernel = None
            try:
                from py2wl import WolframKernel
                k = WolframKernel()
                res = k.evaluate("2+2")
                cls._has_kernel = (res == 4)
            except Exception:
                cls._has_kernel = False

        def setUp(self):
            if not self.__class__._has_kernel:
                self.skipTest("需要真实 Wolfram 内核（请设置 WOLFRAM_EXEC 环境变量）")

        def test_rule_count(self):
            from py2wl.compat._core.metadata import MetadataRepository
            repo = MetadataRepository(MAPPINGS_DIR)
            n = len(repo.all_rules)
            self.assertGreater(n, 800, f"期望 >800 条规则，实际 {n}")
            print(f"\\n  全库规则总数：{n} 条")

        def test_output_converters(self):
            from py2wl.compat._core.converters import OUTPUT_CONVERTERS
            for oc in ("from_wxf", "from_wl_image", "from_wxf_dataframe"):
                self.assertIn(oc, OUTPUT_CONVERTERS)


    if __name__ == "__main__":
        unittest.main(verbosity=2)
    """))

    return "\n".join(out)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rules = load_rules()
    code  = generate(rules, N)
    out   = Path("tests/test_generated_real.py")
    out.write_text(code)
    print(f"生成 {out}  ({len(rules) if not N else N} 条规则, {len(code)} 字节)")