#!/usr/bin/env python3
"""
perf_compare_v2.py — 性能对比：原生 NumPy vs wolfram-pty (WSTP 版本)
对比操作：矩阵乘法、特征值分解、奇异值分解

使用方法：
    python perf_compare_v2.py          # 默认使用传输模式
    python perf_compare_v2.py --direct # 使用直接模式（数据在 Wolfram 端生成，无传输开销）
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

from py2wl.compat import numpy as wnp

# 测试规模
SIZES = [100, 200, 400, 800]
ITERATIONS = 3
WARMUP = 1

OPERATIONS = [
    ("matmul", "矩阵乘法 (A @ B)"),
    ("eig",    "特征值分解 (eig)"),
    ("svd",    "奇异值分解 (svd)"),
]

def measure_time(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result

def run_benchmark(direct=False):
    results = {
        "numpy": {op: [] for op, _ in OPERATIONS},
        "wolfram": {op: [] for op, _ in OPERATIONS},
    }

    print(f"测试模式: {'直接 (Wolfram 端生成数据)' if direct else '传输 (Python 传入数据)'}")
    print("=" * 60)

    for size in SIZES:
        print(f"\n矩阵大小: {size}x{size}")

        # 生成 NumPy 矩阵（用于原生 NumPy 和可能的传输）
        A_np = np.random.rand(size, size)
        B_np = np.random.rand(size, size)

        if direct:
            # 直接模式：使用 wnp.random.rand 在 Wolfram 端生成
            A_w = wnp.random.rand(size, size)
            B_w = wnp.random.rand(size, size)
        else:
            # 传输模式：从 NumPy 数组转换（内部会走文件通道，但经过优化）
            A_w = wnp.array(A_np)   # 直接传入 numpy 数组，无需 .tolist()
            B_w = wnp.array(B_np)

        for op_name, op_desc in OPERATIONS:
            # ----- 原生 NumPy -----
            times_np = []
            for i in range(ITERATIONS + WARMUP):
                if op_name == "matmul":
                    func = lambda: np.matmul(A_np, B_np)
                elif op_name == "eig":
                    func = lambda: np.linalg.eig(A_np)
                elif op_name == "svd":
                    func = lambda: np.linalg.svd(A_np)
                t, _ = measure_time(func)
                if i >= WARMUP:
                    times_np.append(t)
            avg_np = np.mean(times_np)
            results["numpy"][op_name].append(avg_np)

            # ----- wolfram-pty -----
            times_w = []
            for i in range(ITERATIONS + WARMUP):
                if op_name == "matmul":
                    func = lambda: wnp.matmul(A_w, B_w)
                elif op_name == "eig":
                    func = lambda: wnp.linalg.eig(A_w)
                elif op_name == "svd":
                    func = lambda: wnp.linalg.svd(A_w)
                t, _ = measure_time(func)
                if i >= WARMUP:
                    times_w.append(t)
            avg_w = np.mean(times_w)
            results["wolfram"][op_name].append(avg_w)

            print(f"  {op_desc:20} | NumPy: {avg_np*1000:8.2f} ms | Wolfram: {avg_w*1000:8.2f} ms")

    return results

def plot_results(results, sizes, mode):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (op_name, op_desc) in zip(axes, OPERATIONS):
        ax.plot(sizes, results["numpy"][op_name], 'o-', label='NumPy')
        ax.plot(sizes, results["wolfram"][op_name], 's-', label='Wolfram-pty')
        ax.set_xlabel('矩阵大小 N')
        ax.set_ylabel('时间 (秒)')
        ax.set_title(f'{op_desc} ({mode})')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'perf_compare_{mode}.png')
    plt.show()
    print(f"图表已保存为 perf_compare_{mode}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="性能对比：NumPy vs wolfram-pty")
    parser.add_argument("--direct", action="store_true", help="使用直接模式（数据在 Wolfram 端生成）")
    args = parser.parse_args()

    mode = "direct" if args.direct else "transfer"
    results = run_benchmark(direct=args.direct)
    plot_results(results, SIZES, mode)