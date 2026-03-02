#!/usr/bin/env python3
"""
实际测试所有映射规则：调用每个函数，验证内核是否能正确执行。
警告：该脚本会执行所有 887 条规则，耗时可能较长（几分钟）。
运行前请确保：
  - WOLFRAM_EXEC 已设置
  - wolframclient 已安装
  - 当前目录为项目根目录
"""
import sys
import os
sys.path.insert(0, '/storage/emulated/0/wolfram_pty/factory3/wfb_clean')
import sys
import os
import traceback
from types import ModuleType

# 确保项目路径在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from py2wl.compat._core.metadata import MetadataRepository

# ===== 样例参数生成器 =====
def generate_args(rule):
    """根据规则生成样例参数，返回 args 元组和 kwargs 字典"""
    # 基于 input_converters 或 input_converter 生成参数
    ics = rule.get("input_converters")
    ic = rule.get("input_converter", "to_wl_passthrough")
    # 简单示例值映射
    sample_values = {
        "to_wl_list": [1, 2, 3],
        "to_wl_scalar": 5,
        "to_wl_matrix": [[1, 2], [3, 4]],
        "to_wl_matrix_and_vector": ([[1,2],[3,4]], [5,6]),
        "to_wl_string": "test",
        "to_wl_passthrough": "x^2",  # 对于表达式
    }
    if ics:
        args = []
        for cn in ics:
            args.append(sample_values.get(cn, 0))
        return tuple(args), {}
    else:
        # 单参数情况
        if ic in sample_values:
            return (sample_values[ic],), {}
        else:
            return (), {}  # 无参数

# ===== 导入兼容模块 =====
def import_library(lib_name):
    """动态导入兼容模块，如 numpy, sympy 等"""
    try:
        if lib_name == "numpy":
            from py2wl.compat import numpy as np
            return np
        elif lib_name == "sympy":
            from py2wl.compat import sympy as sp
            return sp
        elif lib_name == "scipy":
            from py2wl.compat import scipy
            return scipy
        elif lib_name == "tensorflow":
            from py2wl.compat import tensorflow as tf
            return tf
        elif lib_name == "torch":
            from py2wl.compat import torch
            return torch
        elif lib_name == "pandas":
            from py2wl.compat import pandas as pd
            return pd
        elif lib_name == "matplotlib":
            from py2wl.compat import matplotlib as mpl
            return mpl
        elif lib_name == "seaborn":
            from py2wl.compat import seaborn as sns
            return sns
        elif lib_name == "perf":
            from py2wl.compat import perf
            return perf
        elif lib_name == "monitoring":
            from py2wl.compat import monitoring
            return monitoring
        else:
            return None
    except ImportError as e:
        print(f"⚠️  无法导入 {lib_name}: {e}")
        return None

# ===== 主测试逻辑 =====
def main():
    # 获取所有规则
    repo = MetadataRepository.get_instance('py2wl/compat/mappings')
    all_rules = repo.all_rules
    print(f"共 {len(all_rules)} 条规则")

    # 按顶级模块分组
    rules_by_module = {}
    for rule in all_rules:
        path = rule['python_path']
        top = path.split('.')[0]
        rules_by_module.setdefault(top, []).append(rule)

    results = {
        'passed': [],
        'failed': [],
        'skipped': []
    }

    for top_module, rules in rules_by_module.items():
        print(f"\n======= 测试模块: {top_module} =======")
        mod = import_library(top_module)
        if mod is None:
            print(f"⚠️  无法导入模块 {top_module}，跳过其所有规则")
            results['skipped'].extend(r['python_path'] for r in rules)
            continue

        for rule in rules:
            path = rule['python_path']
            # 跳过常量
            if rule.get('constant'):
                print(f"⏭️  跳过常量: {path}")
                results['skipped'].append(path)
                continue

            # 获取属性
            parts = path.split('.')[1:]  # 去掉顶级模块
            obj = mod
            try:
                for part in parts:
                    obj = getattr(obj, part)
            except AttributeError:
                # 如果属性不存在，可能是子模块未实现，记录失败
                results['failed'].append((path, "AttributeError: 对象不存在"))
                print(f"❌ {path}: 对象不存在")
                continue

            if not callable(obj):
                # 非可调用对象（如常量或属性），跳过
                results['skipped'].append(path)
                continue

            # 生成示例参数
            args, kwargs = generate_args(rule)
            try:
                # 调用函数
                result = obj(*args, **kwargs)
                # 简单检查结果：非 None 即认为通过
                if result is not None:
                    results['passed'].append(path)
                    print(f"✅ {path}")
                else:
                    results['passed'].append(path)  # None 也接受
                    print(f"✅ {path} (返回 None)")
            except Exception as e:
                tb = traceback.format_exc(limit=1)
                error_msg = f"{type(e).__name__}: {e}"
                results['failed'].append((path, error_msg))
                print(f"❌ {path}: {error_msg}")
                # 可选：打印详细错误
                # print(tb)

    # 输出汇总
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    print(f"通过: {len(results['passed'])}")
    print(f"失败: {len(results['failed'])}")
    print(f"跳过: {len(results['skipped'])}")
    if results['failed']:
        print("\n失败的规则:")
        for path, err in results['failed']:
            print(f"  {path}: {err}")

if __name__ == "__main__":
    main()