"""
compat/monitoring.py — 监控/日志/计时兼容代理（tqdm / logging / time 等）
用法：
    from py2wl.compat import monitoring
"""
import sys
from ._proxy_base import LibraryProxy

sys.modules[__name__] = LibraryProxy("time")
