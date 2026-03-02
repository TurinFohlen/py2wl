"""
compat/torch.py — PyTorch 兼容代理
用法：
    from py2wl.compat import torch
    out = torch.matmul(A, B)
    z   = torch.nn.functional.relu(x)
"""
import sys
from ._proxy_base import LibraryProxy

sys.modules[__name__] = LibraryProxy("torch")
