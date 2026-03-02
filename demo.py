#!/usr/bin/env python3
"""
demo.py — py2wl v0.6 综合演示
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
运行：python demo.py
无需 Wolfram 内核，全程离线演示兼容层核心能力。
"""

import sys, os, time, textwrap
sys.path.insert(0, os.path.dirname(__file__))

# ── ANSI 颜色 ─────────────────────────────────────────────────
def _c(code, t): return f"\033[{code}m{t}\033[0m" if sys.stdout.isatty() else t
R  = lambda t: _c("1;31", t);  G  = lambda t: _c("1;32", t)
Y  = lambda t: _c("1;33", t);  B  = lambda t: _c("1;34", t)
M  = lambda t: _c("1;35", t);  C  = lambda t: _c("1;36", t)
W  = lambda t: _c("1;37", t);  DIM = lambda t: _c("2", t)
BG = lambda t: _c("48;5;234", t)

def pause(s=0.04):   time.sleep(s)
def hline(ch="─", n=66, color=DIM): print(color(ch * n))
def section(title):
    print(); hline("━", color=C); print(C(f"  {title}")); hline("━", color=C)

def typeprint(text, delay=0.012):
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
    print()


# ══════════════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════════════
BANNER = r"""
 ██╗    ██╗ ██████╗ ██╗     ███████╗██████╗  █████╗ ███╗   ███╗
 ██║    ██║██╔═══██╗██║     ██╔════╝██╔══██╗██╔══██╗████╗ ████║
 ██║ █╗ ██║██║   ██║██║     █████╗  ██████╔╝███████║██╔████╔██║
 ██║███╗██║██║   ██║██║     ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║
 ╚███╔███╔╝╚██████╔╝███████╗██║     ██║  ██║██║  ██║██║ ╚═╝ ██║
  ╚══╝╚══╝  ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
"""
SUBTITLE = "  bridge  ·  v0.6  ·  Python ⟷ Wolfram Language via PTY"

print(C(BANNER))
typeprint(M(SUBTITLE), delay=0.018)
print()
pause(0.3)


# ══════════════════════════════════════════════════════════════
#  1. 映射数据库总览
# ══════════════════════════════════════════════════════════════
section("① 映射数据库  —  835 条规则 · 11 个主库")

from py2wl.compat._core.metadata import MetadataRepository
from collections import Counter

repo = MetadataRepository(
    os.path.join(os.path.dirname(__file__), "py2wl/compat/mappings")
)

LIBS = [
    ("numpy",      "NumPy",      "1;34"),
    ("scipy",      "SciPy",      "1;36"),
    ("pandas",     "pandas",     "1;33"),
    ("sympy",      "SymPy",      "1;35"),
    ("torch",      "PyTorch",    "1;31"),
    ("sklearn",    "scikit-learn","1;32"),
    ("tf",         "TensorFlow", "1;33"),
    ("matplotlib", "Matplotlib", "1;34"),
    ("seaborn",    "Seaborn",    "1;36"),
]

counts = Counter(r["python_path"].split(".")[0] for r in repo.all_rules)
total  = len(repo.all_rules)
BAR_W  = 30

print()
for prefix, label, color in LIBS:
    n   = counts.get(prefix, 0)
    pct = n / total
    bar = "█" * int(pct * BAR_W) + "░" * (BAR_W - int(pct * BAR_W))
    line = f"  {_c(color, f'{label:<14}')}  {_c(color, bar)}  {W(str(n)):>5}"
    print(line)
    pause(0.06)

print()
print(f"  {DIM('监控/性能/其他')}  {DIM('─' * BAR_W)}  {W(str(total - sum(counts[p] for p,_,_ in LIBS))):>5}")
print()
pause(0.1)
print(G(f"  ✓ 共 {W(str(total))} 条映射规则已就绪"))
pause(0.3)


# ══════════════════════════════════════════════════════════════
#  2. Trie 精确查找 + 转换器管道展示
# ══════════════════════════════════════════════════════════════
section("② 三层解析引擎  —  Trie 精确查找")

DEMOS = [
    "numpy.linalg.eig",
    "scipy.integrate.quad",
    "pandas.DataFrame.groupby",
    "torch.nn.functional.softmax",
    "sympy.solve",
]

print()
for path in DEMOS:
    rule = repo.get_rule(path)
    pause(0.08)
    if rule:
        wf   = _c("1;32", rule["wolfram_function"])
        ic   = _c("2",    rule.get("input_converter",  "to_wl_list"))
        oc   = _c("2",    rule.get("output_converter", "from_wl_passthrough"))
        tags = DIM(" · ".join((rule.get("tags") or [])[:3]))
        print(f"  {C(path):<52}  →  {wf}")
        print(f"  {'':52}     {DIM('in:')} {ic}  {DIM('out:')} {oc}")
        print(f"  {'':52}     {tags}")
        print()
pause(0.2)


# ══════════════════════════════════════════════════════════════
#  3. 模糊搜索 + 候选排名
# ══════════════════════════════════════════════════════════════
section("③ 模糊搜索  —  标签 + 关键词倒排索引")

QUERIES = [
    ("fourier",  "按标签搜索"),
    ("matrix",   "按关键词搜索"),
    ("optimize", "跨库匹配"),
]

from py2wl.compat._core.candidate_finder import CandidateFinder
finder = CandidateFinder(repo, ai_plugin=None, top_k=4)

for q, hint in QUERIES:
    print(f"\n  {DIM('query:')} {W(repr(q))}  {DIM(f'({hint})')}")
    hits = repo.search_rules(q)[:4]
    for i, r in enumerate(hits):
        marker = G("▶") if i == 0 else DIM("·")
        print(f"    {marker} {r['python_path']:<42}→  {_c('1;32', r['wolfram_function'])}")
    pause(0.12)


# ══════════════════════════════════════════════════════════════
#  4. Levenshtein 候选打分可视化
# ══════════════════════════════════════════════════════════════
section("④ 容错候选打分  —  Levenshtein + Jaccard + 命名空间加成")

TYPOS = [
    ("numpy.linalg.eign",   "eig 拼错"),
    ("scipy.optmize.fmin",  "optimize 拼错"),
    ("pandas.grouby",       "groupby 拼错"),
]

print()
for bad_path, note in TYPOS:
    candidates = finder.find(bad_path, use_ai=False)
    print(f"  {R('✗')} {W(bad_path):<40}  {DIM(note)}")
    for score, rule in candidates[:3]:
        bar_len = int(score * 20)
        bar = "▓" * bar_len + "░" * (20 - bar_len)
        sc_color = G if score > 0.7 else Y if score > 0.4 else DIM
        print(f"      {sc_color(f'{score:.2f}')}  {_c('2', bar)}  {rule['python_path']}")
    print()
    pause(0.12)


# ══════════════════════════════════════════════════════════════
#  5. 错误分类器展示
# ══════════════════════════════════════════════════════════════
section("⑤ 错误分类器  —  可恢复 vs 不可恢复")

from py2wl.compat._core.error_classifier import classify, FaultKind

ERRORS = [
    (AttributeError("未找到 'numpy.linalg.eign' 的 Wolfram 映射"),
     "numpy.linalg.eign", "函数名拼写错误"),
    (TypeError("to_wl_matrix 期望二维列表，得到 int"),
     "numpy.dot", "参数类型错误"),
    (RuntimeError("内核执行失败 [scipy.linalg.lu]：$Failed"),
     "scipy.linalg.lu", "内核执行错误"),
    (MemoryError("out of memory"),
     "numpy.fft.fft", "系统内存耗尽"),
    (KeyboardInterrupt(),
     "pandas.read_csv", "用户中断"),
]

print()
for exc, path, desc in ERRORS:
    ei = classify(exc, path)
    kind_str  = G("RECOVERABLE") if ei.kind == FaultKind.RECOVERABLE else R("   FATAL   ")
    cat_str   = _c("2", ei.category.name if ei.category else "—")
    print(f"  [{kind_str}]  {W(f'{type(exc).__name__}'):<22}  {DIM(desc)}")
    print(f"            {DIM('category:')} {cat_str}")
    print(f"            {DIM('hint:')} {ei.hint[:70]}")
    print()
    pause(0.1)


# ══════════════════════════════════════════════════════════════
#  6. 真实内核计算演示
# ══════════════════════════════════════════════════════════════
section("⑥ 真实内核计算  —  需要 Wolfram Engine 已安装并配置 WOLFRAM_EXEC")

# 检查是否有内核可用
# 检查是否有内核可用
has_kernel = False
try:
    from py2wl import WolframKernel
    k = WolframKernel()
    # 使用文件模式获取结果，避免解析终端输出
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    out_dir = os.path.dirname(tmp)
    try:
        path = k.evaluate_to_file("2+2", fmt="txt", out_dir=out_dir, no_cache=True)
        if os.path.exists(path):
            with open(path) as f:
                content = f.read().strip()
            os.unlink(path)
            has_kernel = (content == "4")
        else:
            has_kernel = False
    except Exception:
        has_kernel = False
    del k
except Exception:
    has_kernel = False
    
            
if not has_kernel:
    print(Y("  ⚠ 未检测到可用的 Wolfram 内核，跳过真实计算演示"))
    print(Y("    请设置环境变量 WOLFRAM_EXEC 并确保内核可执行"))
else:
    print(G("  ✅ 检测到 Wolfram 内核，开始演示实际计算"))
    print()

    from py2wl.compat import numpy as np
    from py2wl.compat import scipy
    from py2wl.compat import sympy as sp
    from py2wl.compat import pandas as pd
    from py2wl.compat import matplotlib
    plt = matplotlib.pyplot
    import os

    # ── 数值计算：均值、行列式、FFT ──────────────────────────
    print(C("  📊 数值计算"))
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean_val = np.mean(data)
    det_val = np.linalg.det([[4.0, -2.0], [1.0, 1.0]])
    print(f"    {DIM('np.mean([1,2,3,4,5])')} → {G(f'{mean_val}')}")
    print(f"    {DIM('np.linalg.det([[4,-2],[1,1]])')} → {G(f'{det_val}')}")
    
    t = np.linspace(0, 1, 500)
	    # 生成信号
    t_vals = list(t)
  # t 已经是列表
    print("类型检查：", [type(2 * np.pi * 120 * x) for x in t_vals[:5]])
    print("前五个值：", [2 * np.pi * 120 * x for x in t_vals[:5]])
    a = np.sin([2 * np.pi * 50 * x for x in t_vals])
    b = np.sin([2 * np.pi * 120 * x for x in t_vals])
    b_half = np.multiply(0.5, b)   # 逐元素乘以 0.5
    sig = np.add(a, b_half)        # 逐元素相加
    spec = np.fft.fft(sig)
    print(f"    {DIM('FFT of 500-point signal')} → 频谱长度 {len(spec)}")
	    # ── 数值积分 ────────────────────────────────────────────
    quad_res, quad_err = scipy.integrate.quad("x^2", 0, 1)
    print(f"    {DIM('scipy.integrate.quad(\"x^2\",0,1)')} → {G(f'{quad_res:.6f} ± {quad_err}')}")
    pause(0.2)

    # ── 符号计算：导数、方程求解 ──────────────────────────────
    print(C("  🧠 符号计算"))
    x = sp.Symbol('x')
    f = sp.Mul(sp.sin(x), sp.exp(x))   # 注意：使用 Mul 显式构造乘法
    deriv = sp.diff(f, x)
    print(f"    {DIM('sp.diff(sp.sin(x)*sp.exp(x), x)')} → {G(str(deriv)[:60])}...")
    
    sol = sp.solve(x**2 - 5*x + 6, x)
    print(f"    {DIM('sp.solve(x² -5x +6)')} → {G(str(sol))}")
    pause(0.3)

    # ── 数据处理：简单 pandas 操作 ────────────────────────────
    print(C("  🐼 数据处理"))
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    means = df.mean()
    print(f"    {DIM('pd.DataFrame.mean()')} → {G(str(means))}")
    pause(0.2)

    # ── 图像生成（需要输出目录）────────────────────────────────
    try:
        os.makedirs("/sdcard/wolfram_out", exist_ok=True)
        img_path = plt.plot([1,2,3], [4,5,6])
        if os.path.exists(img_path):
            print(f"    {DIM('plt.plot 图像保存至')} {G(img_path)}")
        else:
            print("    ⚠ 图像生成失败")
    except Exception as e:
        print(f"    ⚠ 图像生成异常: {e}")

    print()
    hline("─", color=DIM)


# ══════════════════════════════════════════════════════════════
#  7. 容错策略模拟（不启动内核）
# ══════════════════════════════════════════════════════════════
section("⑦ 容错策略  —  strict / auto-ai / interactive 三模式")

from py2wl.compat._core.fault_handler import FaultHandler, FaultMode, ActionKind

# 构建 handler（不启动 AI，仅演示打分逻辑）
handler_strict = FaultHandler(repo, mode=FaultMode.STRICT)
handler_auto   = FaultHandler(repo, mode=FaultMode.AUTO_AI)

FAULT_CASES = [
    (AttributeError("未找到 'numpy.linalg.eign' 的 Wolfram 映射"), "numpy.linalg.eign"),
    (TypeError("参数类型不匹配"),                                    "scipy.optmize.fmin"),
]

print()
for exc, path in FAULT_CASES:
    print(f"  {DIM('调用路径:')} {W(path)}")
    print(f"  {DIM('异常类型:')} {Y(type(exc).__name__)}")

    # strict
    action = handler_strict.handle(exc, path)
    print(f"  {DIM('strict  →')} {R('RAISE')}  ← 直接重抛，不容错")

    # auto-ai（无 AI key，演示候选匹配部分）
    candidates = finder.find(path, use_ai=False)
    if candidates:
        top_score, top_rule = candidates[0]
        if top_score >= 0.75:
            verdict = G(f"AUTO-RETRY → {top_rule['python_path']}") + DIM(f"  (score={top_score:.2f})")
        else:
            verdict = Y(f"INTERACTIVE  ← 置信不足({top_score:.2f})，降为询问用户")
        print(f"  {DIM('auto-ai →')} {verdict}")

    print()
    pause(0.15)


# ══════════════════════════════════════════════════════════════
#  FINALE
# ══════════════════════════════════════════════════════════════
hline("═", color=M)
print()
SUMMARY = [
    ("835",  "映射规则"),
    ("11",   "兼容库"),
    ("3",    "容错模式"),
    ("4",    "AI 提供商"),
    ("0",    "运行时依赖（核心层）"),
]
for val, label in SUMMARY:
    pause(0.07)
    print(f"  {M(f'{val:>5}')}  {W(label)}")

print()
typeprint(G("  py2wl — 受限环境下的全能 Wolfram 桥梁 🐺"), delay=0.02)
print()
hline("═", color=M)
print()