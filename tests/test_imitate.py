#!/usr/bin/env python3
"""
test_imitate.py — 真实 Wolfram 内核调用手动验证脚本
输出转换器现已统一：
  数值数组 → CSV 文件 → numpy array
  标量/复合 → WL 字符串直接返回（passthrough）
  图像 → PNG 文件
"""
import sys, os, csv
sys.path.insert(0, '.')
from compat import numpy as np


def load_csv_file(path):
    """读取 CSV 文件，返回数值列表或二维列表。"""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    try:
        parsed = [[float(c) for c in row] for row in rows if row]
        return parsed[0] if len(parsed) == 1 else parsed
    except ValueError:
        return rows


print("🚀 测试真实 Wolfram 内核调用")
print("=" * 50)

# 1. 数值数组 → CSV
a = np.array([1, 2, 3, 4])
print(f"np.array 返回: {a}")

# 2. FFT → CSV
b = np.fft.fft([1, 2, 3, 4])
print(f"np.fft.fft 返回: {b}")

# 3. 行列式 → passthrough（标量）
c = np.linalg.det([[1, 2], [3, 4]])
print(f"np.linalg.det 返回: {c}")

print("=" * 50)
print("测试完成")
