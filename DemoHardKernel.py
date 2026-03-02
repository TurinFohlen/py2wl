#!/usr/bin/env python3
"""
DemoHardKernel.py
=================
wolfram-pty compat 层真实内核计算演示。
每一节调用真实 Wolfram Engine，展示科学计算可行性。

运行：
    python DemoHardKernel.py
    python DemoHardKernel.py --section lglg   # 只跑线代节
    python DemoHardKernel.py --fast             # 跳过慢速演示
"""

import sys
import time
import math
import argparse
import traceback

# ══════════════════════════════════════════════════════════════════════
#  ANSI 颜色
# ══════════════════════════════════════════════════════════════════════
R  = "\033[31m";  G  = "\033[32m";  Y  = "\033[33m"
B  = "\033[34m";  M  = "\033[35m";  C  = "\033[36m"
W  = "\033[37m";  DIM = "\033[2m";  BOLD = "\033[1m"
RESET = "\033[0m"

def c(color, text): return f"{color}{text}{RESET}"

# ══════════════════════════════════════════════════════════════════════
#  横幅
# ══════════════════════════════════════════════════════════════════════

BANNER = f"""
{C}╔══════════════════════════════════════════════════════════════════════╗{RESET}
{C}║{RESET}                                                                      {C}║{RESET}
{C}║{RESET}  {BOLD}{W} ██╗    ██╗ ██████╗ ██╗     ███████╗██████╗  █████╗ ███╗   ███╗{RESET}  {C}║{RESET}
{C}║{RESET}  {BOLD}{W} ██║    ██║██╔═══██╗██║     ██╔════╝██╔══██╗██╔══██╗████╗ ████║{RESET}  {C}║{RESET}
{C}║{RESET}  {BOLD}{C} ██║ █╗ ██║██║   ██║██║     █████╗  ██████╔╝███████║██╔████╔██║{RESET}  {C}║{RESET}
{C}║{RESET}  {BOLD}{C} ██║███╗██║██║   ██║██║     ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║{RESET}  {C}║{RESET}
{C}║{RESET}  {BOLD}{B} ╚███╔███╔╝╚██████╔╝███████╗██║     ██║  ██║██║  ██║██║ ╚═╝ ██║{RESET}  {C}║{RESET}
{C}║{RESET}  {BOLD}{B}  ╚══╝╚══╝  ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝{RESET}  {C}║{RESET}
{C}║{RESET}                                                                      {C}║{RESET}
{C}║{RESET}         {DIM}wolfram-pty  ·  compat layer  ·  real kernel demo{RESET}          {C}║{RESET}
{C}╚══════════════════════════════════════════════════════════════════════╝{RESET}
"""

# ══════════════════════════════════════════════════════════════════════
#  UI 工具
# ══════════════════════════════════════════════════════════════════════

def section(title: str):
    bar = "─" * 68
    print(f"\n{C}{bar}{RESET}")
    print(f"{BOLD}{Y}  {title}{RESET}")
    print(f"{C}{bar}{RESET}")

def ok(label, value, unit="", expect=None):
    v_str = f"{value}" if not isinstance(value, float) else f"{value:.6g}"
    exp_str = ""
    if expect is not None:
        ok_flag = abs(float(value) - float(expect)) < 1e-4 if isinstance(value,(int,float)) else True
        exp_str = f"  {G}✓{RESET}" if ok_flag else f"  {R}✗ 期望 {expect}{RESET}"
    print(f"  {G}●{RESET} {W}{label:<38}{RESET} {C}{v_str}{RESET}{DIM}{unit}{RESET}{exp_str}")

def info(msg):
    print(f"  {DIM}→ {msg}{RESET}")

def warn(label, err):
    print(f"  {Y}⚠{RESET}  {label}: {DIM}{str(err)[:80]}{RESET}")

def fail(label, err):
    print(f"  {R}✗{RESET}  {label}: {R}{str(err)[:80]}{RESET}")

def timing(t):
    return f"{t*1000:.1f}ms" if t < 1 else f"{t:.2f}s"

class Timer:
    def __enter__(self):
        self._t = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t


# ══════════════════════════════════════════════════════════════════════
#  各节演示函数
# ══════════════════════════════════════════════════════════════════════

def demo_constants():
    section("§1  数学常量  (constant: true — 实例级缓存验证)")
    from py2wl.compat import numpy as np
    from py2wl.compat import sympy as sp

    # 首次访问
    with Timer() as t1:
        pi_val = np.pi
    ok("np.pi  (首次，触发内核)", pi_val, expect=math.pi)
    info(f"首次耗时 {timing(t1.elapsed)}")

    # 缓存命中
    with Timer() as t2:
        for _ in range(1000):
            _ = np.pi
    info(f"1000 次缓存访问耗时 {timing(t2.elapsed)}  (平均 {t2.elapsed/1000*1e6:.2f}μs/次)")

    ok("np.e",            np.e,            expect=math.e)
    ok("np.euler_gamma",  np.euler_gamma,  expect=0.5772156649)
    ok("np.pi × 2",       np.pi * 2,       expect=2 * math.pi)

    ok("sp.pi (符号→数值)", float(sp.pi) if hasattr(sp.pi, '__float__') else sp.pi)
    ok("sp.E  (符号→数值)", float(sp.E)  if hasattr(sp.E,  '__float__') else sp.E)
    ok("sp.EulerGamma",   sp.EulerGamma)


def demo_linalg():
    section("§2  线性代数  (numpy.linalg → WL Eigensystem / SVD / LinearSolve)")
    from py2wl.compat import numpy as np

    # 特征值 / 特征向量
    A = [[4.0, 1.0], [2.0, 3.0]]
    with Timer() as t:
        vals, vecs = np.linalg.eig(A)
    ok("eig([[4,1],[2,3]]) 特征值 λ₁", float(vals[0]),  expect=5.0)
    ok("eig([[4,1],[2,3]]) 特征值 λ₂", float(vals[1]),  expect=2.0)
    info(f"耗时 {timing(t.elapsed)}")

    # 奇异值分解
    M = [[1.0,2.0,0.0],[0.0,0.0,3.0],[0.0,0.0,0.0],[0.0,2.0,0.0]]
    with Timer() as t:
        U, S, V = np.linalg.svd(M)
    # 如果 S 是二维列表，展平
    if S and isinstance(S[0], list):
        S = [s[0] for s in S]
    ok(f"svd(4×3) 奇异值数量", len(S))
    ok(f"svd 最大奇异值",       max(float(s) for s in S),  expect=3.0)
    info(f"耗时 {timing(t.elapsed)}")

    # 行列式
    B = [[3.0,1.0],[1.0,2.0]]
    ok("det([[3,1],[1,2]])",  np.linalg.det(B), expect=5.0)

    # 线性方程组
    A2 = [[2.0,1.0,-1.0],[3.0,0.0,2.0],[1.0,-2.0,3.0]]
    b  = [8.0,11.0,3.0]
    with Timer() as t:
        x = np.linalg.solve(A2, b)
    ok("solve Ax=b  x[0]", float(x[0]), expect=27/7)
    ok("solve Ax=b  x[1]", float(x[1]), expect=0.0)
    ok("solve Ax=b  x[2]", float(x[2]), expect=-2/7)
    info(f"耗时 {timing(t.elapsed)}")

    # 矩阵范数
    ok("norm([3,4])",  np.linalg.norm([3.0,4.0]), expect=5.0)

    # 矩阵指数（scipy）
    from py2wl.compat import scipy
    I2 = [[0.0,1.0],[-1.0,0.0]]
    with Timer() as t:
        eI = scipy.linalg.expm(I2)
    # e^([[0,1],[-1,0]]) ≈ [[cos1, sin1],[-sin1, cos1]]
    ok("expm([[0,1],[-1,0]])[0][0] ≈ cos(1)",
       float(eI[0][0]), expect=math.cos(1))
    info(f"耗时 {timing(t.elapsed)}")



def demo_fft():
    section("§3  信号处理  (numpy.fft → WL Fourier)")
    from py2wl.compat import numpy as np

    # 构造单频信号：cos(2πt), N=8
    N = 8
    signal = [math.cos(2 * math.pi * k / N) for k in range(N)]
    with Timer() as t:
        spectrum = np.fft.fft(signal)

    # ── 诊断：打印 spectrum 的真实结构（修复 FFT 后可删）──────
    print(f"  [FFT诊断] type={type(spectrum).__name__}  "
          f"len={len(spectrum) if hasattr(spectrum,'__len__') else 'N/A'}")
    if hasattr(spectrum, '__len__') and len(spectrum) > 0:
        e0 = spectrum[0]
        print(f"  [FFT诊断] spectrum[0] type={type(e0).__name__}  "
              f"len={len(e0) if hasattr(e0,'__len__') else 'N/A'}  "
              f"val={repr(e0)[:80]}")
        if hasattr(e0, '__len__') and len(e0) > 0:
            print(f"  [FFT诊断] spectrum[0][0] type={type(e0[0]).__name__}  "
                  f"val={repr(e0[0])[:60]}")

    # _normalize 已将 WLFunction[List,...] 展开为 Python list
    # WL Fourier 返回复数元素格式：list of [re, im] 或 complex
    ok("FFT(cos信号) 长度", len(spectrum), expect=N)

    def _to_complex(v):
        if isinstance(v, complex):            return v
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return complex(float(v[0]), float(v[1]))
        return complex(float(v))

    mags     = [abs(_to_complex(v)) for v in spectrum]
    dominant = mags.index(max(mags))
    ok("FFT 主频位置", dominant)
    info(f"耗时 {timing(t.elapsed)}")

    # 逆变换往返
    with Timer() as t:
        ifft = np.fft.ifft(spectrum)
    ifft_c   = [_to_complex(v) for v in ifft]
    max_err  = max(abs(ifft_c[i].real - signal[i]) for i in range(N))
    ok("IFFT 往返误差 < 1e-6", max_err < 1e-6)
    info(f"耗时 {timing(t.elapsed)}")




def demo_calculus():
    section("§4  符号 + 数值微积分  (sympy / scipy → WL D / Integrate / NIntegrate)")
    from py2wl.compat import sympy as sp
    from py2wl.compat import scipy

    # 符号微分
    with Timer() as t:
        dsin = sp.diff("Sin[x]", "x")
    ok("d/dx sin(x)",  dsin)
    info(f"耗时 {timing(t.elapsed)}")

    with Timer() as t:
        dxn = sp.diff("x^4 - 3*x^2 + 2", "x")
    ok("d/dx (x⁴-3x²+2)", dxn)
    info(f"耗时 {timing(t.elapsed)}")

    # 符号积分
    with Timer() as t:
        isin = sp.integrate("Sin[x]", "x")
    ok("∫sin(x)dx",  isin)
    info(f"耗时 {timing(t.elapsed)}")

    # 数值积分
    with Timer() as t:
        val, err = scipy.integrate.quad("Sin[x]", 0, math.pi)
    ok("∫₀^π sin(x)dx = 2",  float(val), expect=2.0)
    ok("误差估计",              float(err))
    info(f"耗时 {timing(t.elapsed)}")

    with Timer() as t:
        val2, _ = scipy.integrate.quad("Exp[-x^2]", 0, 10)
    ok("∫₀^∞ e^(-x²)dx ≈ √π/2", float(val2), expect=math.sqrt(math.pi)/2)
    info(f"耗时 {timing(t.elapsed)}")


def demo_solve():
    section("§5  方程求解  (sympy.solve / scipy.optimize → WL Solve / FindMinimum)")
    from py2wl.compat import sympy as sp
    from py2wl.compat import scipy

    # 多项式方程
    with Timer() as t:
        roots = sp.solve("x^2 - 5*x + 6 == 0")
    ok("x²-5x+6=0  根数",  len(roots), expect=2)
    ok("根 1",              float(roots[0]), expect=2.0)
    ok("根 2",              float(roots[1]), expect=3.0)
    info(f"耗时 {timing(t.elapsed)}")

    # 三次方程
    with Timer() as t:
        roots3 = sp.solve("x^3 - 6*x^2 + 11*x - 6 == 0")
    ok("x³-6x²+11x-6=0  根数", len(roots3), expect=3)
    info(f"耗时 {timing(t.elapsed)}")

    # 数值优化
    with Timer() as t:
        fmin, xmin = scipy.optimize.minimize_scalar("(x - 2.5)^2 + 1")
    ok("min (x-2.5)²+1  最小值",  float(fmin), expect=1.0)
    ok("min (x-2.5)²+1  极值点",  float(xmin), expect=2.5)
    info(f"耗时 {timing(t.elapsed)}")

    # 求根
    with Timer() as t:
        root = scipy.optimize.brentq("x^3 - x - 2", 1, 2)
    ok("x³-x-2=0 在 [1,2] 的根",  float(root), expect=1.52138)
    info(f"耗时 {timing(t.elapsed)}")


def demo_stats():
    section("§6  统计计算  (numpy / scipy.stats → WL Mean / StandardDeviation / …)")
    from py2wl.compat import numpy as np
    from py2wl.compat import scipy

    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    ok("mean",    np.mean(data),   expect=5.0)
    ok("std",     np.std(data),    expect=2.13809)
    ok("median",  np.median(data), expect=4.5)
    ok("min",     np.min(data),    expect=2.0)
    ok("max",     np.max(data),    expect=9.0)

    # 正态分布 PDF
    with Timer() as t:
        pdf0 = scipy.stats.norm(0, 1)   # N(0,1) 在 x=0 的 PDF
    ok("N(0,1) pdf(0) = 1/√(2π)",
       float(pdf0) if isinstance(pdf0, (int,float)) else pdf0,
       expect=1/math.sqrt(2*math.pi))
    info(f"耗时 {timing(t.elapsed)}")


def demo_pandas():
    section("§7  数据处理  (pandas compat 层 — WolframDataFrame)")
    import os, tempfile
    from py2wl.compat import pandas as pd
    from py2wl.compat.pandas import WolframDataFrame

    # 构造测试数据
    fd, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    rows = [
        "name,score,grade,city",
        "Alice,92,A,Beijing",
        "Bob,78,C,Shanghai",
        "Carol,88,B,Beijing",
        "Dave,95,A,Shenzhen",
        "Eve,72,D,Shanghai",
        "Frank,85,B,Beijing",
    ]
    with open(csv_path, "w") as f:
        f.write("\n".join(rows))

    try:
        # 读 CSV
        with Timer() as t:
            df = pd.read_csv(csv_path)
        ok("read_csv 行数",    len(df), expect=6)
        ok("read_csv 列数",    len(df.columns), expect=4)
        info(f"耗时 {timing(t.elapsed)}")

        # 过滤
        high = df.query("score > 85")
        ok("query(score>85) 行数", len(high), expect=3)

        # 分组聚合
        grouped = df.groupby("city").mean()
        ok("groupby('city') 分组数",  len(grouped))
        bj_scores = [row[grouped._columns.index("score")]
                     for row in grouped._rows
                     if row[grouped._columns.index("city")] == "Beijing"]
        if bj_scores:
            ok("Beijing 平均分", float(bj_scores[0]), expect=(92+88+85)/3)

        # 描述统计
        desc = df.describe()
        ok("describe() 行数 ≥ 5",  len(desc) >= 5)

        # 排序
        sorted_df = df.sort_values("score")
        scores = [row[sorted_df._columns.index("score")] for row in sorted_df._rows]
        ok("sort_values 单调递增",  all(scores[i] <= scores[i+1]
                                        for i in range(len(scores)-1)))

        # 列运算
        df["rank"] = [i+1 for i in range(len(df))]
        ok("新增列后列数",  len(df.columns), expect=5)

        # CSV 往返
        out_path = csv_path + ".out"
        df.to_csv(out_path)
        df2 = pd.read_csv(out_path)
        ok("CSV 往返行数一致",  len(df2), expect=len(df))
        os.unlink(out_path)

    finally:
        os.unlink(csv_path)


def demo_pca():
    section("§8  机器学习  (sklearn.decomposition.PCA → WL PrincipalComponents)")
    from py2wl.compat import sklearn
    from py2wl.compat import numpy as np

    # 二维数据：沿 (1,2) 方向的线性数据加噪声
    import random; random.seed(42)
    data = [[i + random.gauss(0, 0.1),
             2*i + random.gauss(0, 0.1)]
            for i in range(-5, 6)]

    with Timer() as t:
        pcs = sklearn.decomposition.PCA(data)
    ok("PCA 返回类型 list",    isinstance(pcs, list))
    ok("主成分数",              len(pcs))
    info(f"耗时 {timing(t.elapsed)}")

    # 标准化
    with Timer() as t:
        scaled = sklearn.preprocessing.StandardScaler(
            [float(row[0]) for row in data]
        )
    ok("Standardize 长度",  len(scaled) if isinstance(scaled,list) else 1)
    info(f"耗时 {timing(t.elapsed)}")


def demo_stress():
    section("§9  压力测试  (常量缓存 × 1000 / 批量矩阵 / 积分序列)")
    from py2wl.compat import numpy as np
    from py2wl.compat import scipy

    # ── 常量缓存压力 ──────────────────────────────────────────
    with Timer() as t:
        total = sum(np.pi * i for i in range(1000))
    ok("∑ np.pi×i (i=0..999)",  abs(total - math.pi * 999*1000/2) < 1e-3)
    info(f"1000 次 np.pi 访问耗时 {timing(t.elapsed)}（全部命中缓存）")

    # ── 100 个随机 2×2 矩阵的行列式 ─────────────────────────────
    import random
    random.seed(42)  # 可重现
    matrices = [
        [[random.uniform(0,5), random.uniform(0,5)],
         [random.uniform(0,5), random.uniform(0,5)]]
        for _ in range(100)
    ]
    with Timer() as t:
        dets = [np.linalg.det(m) for m in matrices]
    ok("100 个 2×2 行列式（批量）",  len(dets), expect=100)
    info(f"耗时 {timing(t.elapsed)}，平均 {timing(t.elapsed/100)}/次")

    # ── 积分序列 ──────────────────────────────────────────────
    with Timer() as t:
        integrals = [
            scipy.integrate.quad(f"x^{n}", 0, 1)[0]
            for n in range(1, 6)
        ]
    # ∫₀¹ xⁿ dx = 1/(n+1)
    for n, val in enumerate(integrals, 1):
        ok(f"∫₀¹ x^{n} dx = 1/{n+1}",
           abs(float(val) - 1/(n+1)) < 1e-4)
    info(f"5 次数值积分耗时 {timing(t.elapsed)}")

# ══════════════════════════════════════════════════════════════════════
#  总结
# ══════════════════════════════════════════════════════════════════════

def print_summary(results: dict, total_time: float):
    passed = sum(1 for v in results.values() if v == "ok")
    failed = sum(1 for v in results.values() if v == "fail")
    skipped= sum(1 for v in results.values() if v == "skip")

    print(f"\n{C}{'═'*68}{RESET}")
    print(f"{BOLD}{W}  演示结果汇总{RESET}")
    print(f"{C}{'─'*68}{RESET}")

    for name, status in results.items():
        icon  = f"{G}✓{RESET}" if status=="ok" else f"{R}✗{RESET}" if status=="fail" else f"{Y}–{RESET}"
        label = f"{G}通过{RESET}" if status=="ok" else f"{R}失败{RESET}" if status=="fail" else f"{Y}跳过{RESET}"
        print(f"  {icon}  {name:<30} {label}")

    print(f"{C}{'─'*68}{RESET}")
    print(f"  总计：{G}{passed} 通过{RESET}  {R}{failed} 失败{RESET}  {Y}{skipped} 跳过{RESET}  "
          f"  总耗时：{timing(total_time)}")
    print(f"{C}{'═'*68}{RESET}\n")


# ══════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════

SECTIONS = {
    "constants": ("§1 常量缓存",    demo_constants),
    "linalg":    ("§2 线性代数",    demo_linalg),
    "fft":       ("§3 信号处理",    demo_fft),
    "calculus":  ("§4 微积分",      demo_calculus),
    "solve":     ("§5 方程求解",    demo_solve),
    "stats":     ("§6 统计",        demo_stats),
    "pandas":    ("§7 数据处理",    demo_pandas),
    "pca":       ("§8 机器学习",    demo_pca),
    "stress":    ("§9 压力测试",    demo_stress),
}

def main():
    parser = argparse.ArgumentParser(description="wolfram-pty 真实内核演示")
    parser.add_argument("--section", "-s", help="只运行指定节（如 linalg）")
    parser.add_argument("--fast",    "-f", action="store_true",
                        help="跳过压力测试")
    parser.add_argument("--list",    "-l", action="store_true",
                        help="列出所有节")
    args = parser.parse_args()

    if args.list:
        print("可用节：")
        for k, (label, _) in SECTIONS.items():
            print(f"  {k:<12} {label}")
        return

    print(BANNER)
    print(f"  {DIM}Python {sys.version.split()[0]}  ·  "
          f"wolfram-pty compat layer{RESET}\n")

    to_run = {}
    if args.section:
        if args.section not in SECTIONS:
            print(f"未知节: {args.section}，可用: {list(SECTIONS)}")
            sys.exit(1)
        to_run = {args.section: SECTIONS[args.section]}
    else:
        to_run = {k: v for k, v in SECTIONS.items()
                  if not (args.fast and k == "stress")}

    results = {}
    t_start = time.perf_counter()

    for key, (label, fn) in to_run.items():
        try:
            fn()
            results[label] = "ok"
        except KeyboardInterrupt:
            print(f"\n{Y}  中断{RESET}")
            results[label] = "skip"
            break
        except Exception as e:
            results[label] = "fail"
            print(f"\n  {R}ERROR in {label}:{RESET}")
            traceback.print_exc()

    total_time = time.perf_counter() - t_start
    print_summary(results, total_time)


if __name__ == "__main__":
    main()
