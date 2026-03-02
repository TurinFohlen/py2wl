"""
compat/sympy.py — SymPy 兼容代理
用法：
    from py2wl.compat import sympy as sp
    result = sp.integrate("x^2", "x")
"""
import sys
from ._proxy_base import LibraryProxy

sys.modules[__name__] = LibraryProxy("sympy")
