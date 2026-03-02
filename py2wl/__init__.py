"""py2wl 包（懒加载核心，避免在没有 pexpect 的环境爆炸）"""

def __getattr__(name):
    if name == "WolframKernel":
        from .kernel import WolframKernel
        import sys
        sys.modules[__name__].WolframKernel = WolframKernel
        return WolframKernel
    if name == "WolframPipeline":
        # WolframPipeline 已在 WSTP 版本中移除，保留兼容名避免 ImportError
        raise AttributeError(
            "WolframPipeline 已移除，请直接使用 WolframKernel.evaluate()")
    raise AttributeError(f"module 'py2wl' has no attribute {name!r}")
