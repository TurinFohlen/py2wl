"""
py2wl.compat
---------------------
NumPy/SciPy 兼容层入口。

用法：
    from py2wl.compat import numpy as np
    result = np.fft.fft([1, 2, 3, 4])
"""
from . import numpy
from ._core.converters import register_input_converter, register_output_converter

__all__ = ["numpy", "register_input_converter", "register_output_converter"]
