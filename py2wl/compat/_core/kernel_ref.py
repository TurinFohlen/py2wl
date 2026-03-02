"""
kernel_ref.py
单独模块存放 _get_kernel，方便测试 patch。
"""
import sys
from pathlib import Path

def get_kernel():
    """惰性获取 WolframKernel 单例"""
    try:
        from py2wl.kernel import WolframKernel
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "py2wl_kernel",
            Path(__file__).parent.parent.parent / "kernel.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["py2wl_kernel"] = mod
        WolframKernel = mod.WolframKernel
    return WolframKernel()
