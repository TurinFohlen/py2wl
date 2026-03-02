"""
compat/scipy.py — SciPy 兼容代理
用法：
    from py2wl.compat import scipy
    result = scipy.integrate.quad(f, 0, 1)
    result = scipy.linalg.svd(matrix)
"""
import sys
from ._proxy_base import LibraryProxy
sys.modules[__name__] = LibraryProxy("scipy")
