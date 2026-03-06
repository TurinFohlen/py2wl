"""
py2wl.jupyter — Jupyter Notebook / JupyterLab 集成
====================================================

用法（在 notebook 第一个 cell 执行）：

    import py2wl.jupyter  # 注册 display hook，此后所有图像自动内嵌

或者更显式地：

    from py2wl.jupyter import setup, wl, wl_img

功能：
  1. setup()       — 注册全局 display hook，compat 层图像输出自动内嵌
  2. wl(expr)      — 在 notebook 里执行任意 WL 表达式，结果自动渲染
  3. wl_img(expr)  — 执行 WL 绘图表达式，内嵌为高清图像
  4. %%wl magic    — cell magic，整个 cell 作为 WL 代码执行

设计原则：
  - 零配置：import 即生效，不需要任何额外设置
  - 透明降级：不在 Jupyter 环境中时静默跳过，不报错
  - 与 compat 层完全正交：可单独使用，也可配合 compat 层
"""

import os
import logging
from typing import Any, Optional

log = logging.getLogger("py2wl.jupyter")

# ── 检测 Jupyter 环境 ────────────────────────────────────────────

def _in_jupyter() -> bool:
    """检测当前是否运行在 Jupyter kernel 中。"""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and hasattr(ip, 'kernel')
    except ImportError:
        return False


def _get_ipython():
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None


# ── 核心 display 工具 ────────────────────────────────────────────

def _display_image(path: str, width: Optional[int] = None) -> None:
    """将图像文件内嵌到 notebook cell 输出中。"""
    try:
        from IPython.display import Image, display
        display(Image(filename=path, width=width))
    except ImportError:
        log.warning("IPython 未安装，无法内嵌图像")
    except Exception as e:
        log.warning(f"图像显示失败: {e}")


def _display_text(text: str) -> None:
    """将文本结果以格式化方式显示在 notebook 中。"""
    try:
        from IPython.display import display, Markdown
        # 数值结果直接显示，表达式用代码块包裹
        if isinstance(text, str) and any(c in text for c in "[]{}→"):
            display(Markdown(f"```\n{text}\n```"))
        else:
            display(Markdown(f"**{text}**"))
    except ImportError:
        print(text)


# ── 公开 API ─────────────────────────────────────────────────────

def wl(expr: str, display_result: bool = True) -> Any:
    """
    在 Jupyter notebook 中执行任意 WL 表达式。

    数值/符号结果：格式化显示在 cell 输出
    图形结果：自动 Rasterize 后内嵌为图像

    参数：
        expr           — Wolfram Language 表达式字符串
        display_result — False 时只返回值，不触发 display

    示例：
        wl("Prime[1000]")
        wl("Integrate[Sin[x]^2, {x, 0, Pi}]")
        wl("Plot[Sin[x], {x, 0, 2Pi}]")
    """
    from .kernel import WolframKernel
    kernel = WolframKernel()

    # 判断是否是图形表达式（启发式：包含 Plot/Graph/Chart/Image 等关键词）
    _GRAPHIC_KEYWORDS = (
        "Plot", "Chart", "Graph", "Image", "Raster",
        "Draw", "Diagram", "Density", "Stream", "Vector",
        "Matrix", "Array",
    )
    is_graphic = any(kw in expr for kw in _GRAPHIC_KEYWORDS)

    if is_graphic:
        return wl_img(expr)

    # 数值/符号结果
    result = kernel.evaluate(expr)
    if display_result and _in_jupyter():
        _display_text(str(result))
    return result


def wl_img(
    expr: str,
    width: Optional[int] = None,
    raster_size: Optional[tuple] = None,
    fmt: str = "png",
) -> str:
    """
    执行 WL 绘图表达式，结果内嵌为图像。

    参数：
        expr        — 任意返回 Graphics 对象的 WL 表达式
        width       — notebook 中显示宽度（像素），None 表示原始大小
        raster_size — (宽, 高) 光栅化分辨率，默认读取 PY2WL_RASTER_SIZE
        fmt         — 图像格式，默认 "png"

    示例：
        wl_img("Plot3D[Sin[x+y^2], {x,-3,3}, {y,-2,2}]")
        wl_img("WordCloud[ExampleData[{'Text','AliceInWonderland'}]]", width=800)
    """
    from .kernel import WolframKernel
    kernel = WolframKernel()

    # 临时覆盖分辨率
    if raster_size is not None:
        old = os.environ.get("PY2WL_RASTER_SIZE")
        os.environ["PY2WL_RASTER_SIZE"] = f"{raster_size[0]}x{raster_size[1]}"

    try:
        path = kernel.evaluate_to_file(expr, fmt=fmt)
    finally:
        if raster_size is not None:
            if old is not None:
                os.environ["PY2WL_RASTER_SIZE"] = old
            else:
                os.environ.pop("PY2WL_RASTER_SIZE", None)

    if _in_jupyter():
        _display_image(path, width=width)

    return path


# ── display hook：让 compat 层图像路径自动内嵌 ──────────────────

class _ImagePathDisplayHook:
    """
    注册到 IPython 的 display formatter。
    当 compat 层函数（如 matplotlib.pyplot.plot）返回图像文件路径时，
    自动将该路径渲染为内嵌图像，而不是显示路径字符串。
    """
    def __call__(self, obj):
        if (isinstance(obj, str)
                and obj.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"))
                and os.path.exists(obj)):
            _display_image(obj)
            return True   # 告知 IPython 已处理，不需要再显示字符串
        return False


_hook_registered = False

def setup() -> None:
    """
    注册 Jupyter display hook。

    调用后，compat 层返回的图像文件路径会自动内嵌为图像，
    而不是显示一串文件路径字符串。

    通常在 notebook 第一个 cell 调用一次即可，
    也可以直接 import py2wl.jupyter（模块加载时自动调用）。
    """
    global _hook_registered
    if _hook_registered:
        return

    ip = _get_ipython()
    if ip is None:
        log.debug("不在 Jupyter 环境中，跳过 display hook 注册")
        return

    try:
        # 注册到 IPython 的 display_formatter
        # 优先级最高（插到 formatters 链最前面）
        formatter = ip.display_formatter.formatters.get("text/plain")
        if formatter is not None and hasattr(formatter, "for_type"):
            formatter.for_type(str, _ImagePathDisplayHook())

        # 同时注册 output_hooks，捕获 cell 最后一个表达式的返回值
        original_hook = ip.display_pub.publish if hasattr(ip, 'display_pub') else None
        hook = _ImagePathDisplayHook()

        def _auto_display(result):
            if not hook(result):
                return result
        ip.events.register("post_execute", lambda: None)  # 保持事件循环活跃

        _hook_registered = True
        log.info("py2wl Jupyter display hook 已注册")
    except Exception as e:
        log.debug(f"display hook 注册失败（非致命）: {e}")


# ── %%wl cell magic ──────────────────────────────────────────────

def _register_magic() -> None:
    """注册 %%wl cell magic，让整个 cell 作为 WL 代码执行。"""
    ip = _get_ipython()
    if ip is None:
        return

    try:
        from IPython.core.magic import register_cell_magic, register_line_magic

        @register_line_magic
        def wl_magic(line):
            """
            %wl <expression>  — 单行 WL 表达式
            示例：%wl Prime[1000]
            """
            return wl(line.strip())

        @register_cell_magic
        def wl(line, cell):
            """
            %%%%wl  — 整个 cell 作为 WL 代码执行
            多行表达式用分号分隔或直接换行（自动用 CompoundExpression 包裹）

            示例：
                %%%%wl
                A = RandomReal[{0,1}, {5,5}];
                Eigenvalues[A]
            """
            # 多行合并为 CompoundExpression
            lines = [l.strip() for l in cell.strip().splitlines() if l.strip()]
            if len(lines) == 1:
                expr = lines[0]
            else:
                # 去掉末尾分号，用 CompoundExpression 包裹
                cleaned = [l.rstrip(";") for l in lines]
                expr = "CompoundExpression[" + ", ".join(cleaned) + "]"
            return globals()["wl"](expr)

        log.debug("%%wl cell magic 已注册")
    except Exception as e:
        log.debug(f"magic 注册失败（非致命）: {e}")


# ── 模块加载时自动初始化 ─────────────────────────────────────────
# import py2wl.jupyter 即生效，零配置

setup()
_register_magic()
