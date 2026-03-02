"""
compat/numpy.py — NumPy 兼容代理
用法：
    from py2wl.compat import numpy as np
    result = np.fft.fft([1, 2, 3, 4])
"""
import sys
from ._proxy_base import LibraryProxy, list_mappings, search, reload_mappings, inject_kernel

# 把本模块替换为代理对象（支持 np.fft.fft 链式访问）
sys.modules[__name__] = LibraryProxy("numpy")
