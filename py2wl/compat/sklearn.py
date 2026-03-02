"""
compat/sklearn.py — scikit-learn 兼容代理
用法：
    from py2wl.compat import sklearn
    result = sklearn.preprocessing.StandardScaler([1,2,3,4,5])
"""
import sys
from ._proxy_base import LibraryProxy
sys.modules[__name__] = LibraryProxy("sklearn")
