#!/usr/bin/env python3
"""
性能测试：新版 wolfram-pty 处理大矩阵
运行前确保已安装 numpy 和 matplotlib，并设置 WOLFRAM_EXEC。
"""
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
import time
import numpy as np
import matplotlib.pyplot as plt
from py2wl.compat import numpy as wnp

# 测试规模（可根据内存调整）
sizes = [100, 200, 400, 800, 1000]
ITERATIONS = 3  # 每个规模重复次数
WARMUP = 1      # 预热次数

def bench(op_name, func_np, func_w, sizes):
    """执行性能测试，返回 (sizes, times_np, times_w)"""
    times_np = []
    times_w = []
    for n in sizes:
        # 生成 NumPy 矩阵
        A_np = np.random.rand(n, n)
        B_np = np.random.rand(n, n)

        # 转换为 wolfram 代理对象（只做一次）
        A_w = wnp.array(A_np)
        B_w = wnp.array(B_np)

        # ----- 原生 NumPy -----
        t0 = time.perf_counter()
        for i in range(ITERATIONS + WARMUP):
            if i >= WARMUP:
                t0_inner = time.perf_counter()
            func_np(A_np, B_np)
        t1 = time.perf_counter()
        avg_np = (t1 - t0) / ITERATIONS
        times_np.append(avg_np)

        # ----- wolfram-pty -----
        t0 = time.perf_counter()
        for i in range(ITERATIONS + WARMUP):
            if i >= WARMUP:
                t0_inner = time.perf_counter()
            func_w(A_w, B_w)
        t1 = time.perf_counter()
        avg_w = (t1 - t0) / ITERATIONS
        times_w.append(avg_w)

        ratio = avg_w / avg_np
        print(f"{op_name} n={n}:  NumPy {avg_np:.3f}s  "
              f"Wolfram {avg_w:.3f}s  (ratio {ratio:.2f}x)")
    return sizes, times_np, times_w

# 定义操作
ops = [
    ("矩阵乘法", lambda A, B: A @ B, lambda A, B: wnp.matmul(A, B)),
    ("特征值分解", lambda A, _: np.linalg.eig(A), lambda A, _: wnp.linalg.eig(A)),
    ("奇异值分解", lambda A, _: np.linalg.svd(A), lambda A, _: wnp.linalg.svd(A)),
]

# 收集结果
results = {}
for op_name, func_np, func_w in ops:
    sizes, t_np, t_w = bench(op_name, func_np, func_w, sizes)
    results[op_name] = (t_np, t_w)

# 绘图
plt.figure(figsize=(15, 5))
for i, (op_name, (t_np, t_w)) in enumerate(results.items(), 1):
    plt.subplot(1, 3, i)
    plt.plot(sizes, t_np, 'o-', label='NumPy')
    plt.plot(sizes, t_w, 's-', label='Wolfram-pty')
    plt.xlabel('矩阵大小 N')
    plt.ylabel('时间 (秒)')
    plt.title(op_name)
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig('perf_optimized.png')
plt.show()
print("图表已保存为 perf_optimized.png")