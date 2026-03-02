"""
compat/seaborn.py — Seaborn 绘图代理（规则在 matplotlib.yaml 内）
用法：
    from py2wl.compat import seaborn as sns
    img = sns.heatmap(matrix)
"""
import sys
from ._proxy_base import LibraryProxy
sys.modules[__name__] = LibraryProxy("seaborn")
