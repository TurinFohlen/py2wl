# py2wl - Python to Wolfram Bridge
# py2wl

**Python → Wolfram Language bridge**

把 Wolfram Engine 内核当作 Python 的计算后端来用。兼容层让现有的 NumPy / SciPy / pandas / SymPy / PyTorch / scikit-learn / Matplotlib 代码几乎不改一行就能路由到 Wolfram 内核执行。

```bash
pip install py2wl
```

> **前提**：系统中需要已安装 [Wolfram Engine](https://www.wolfram.com/engine/)（免费许可证可用），并设置环境变量 `WOLFRAM_EXEC` 指向内核可执行文件。

---

## 为什么用 py2wl？

| 场景 | py2wl vs NumPy |
|------|---------------|
| 特征值分解 n=1000 | **快 40×**（Wolfram 多核 LAPACK ） |
| 奇异值分解 n=800 | **快 10.4×** |
| 符号计算 | NumPy 不支持，Wolfram 原生 |
| 矩阵乘法 | 慢 5–8×（跨进程传输开销） |
|复杂密集计算| 手写cython千行，不如自动Compile一行|

Wolfram Engine（无 GUI）约 600MB，比 NumPy + SciPy + pandas + scikit-learn + matplotlib 全家桶（2–3GB）更小，能做的事更多。

传统 Python 科学栈要获得 C 级性能，用户必须自己动手：写 Cython、写 Numba 装饰器、或者手写 C extension。这是有门槛的，大多数用户不会做，没时间做，只能接受解释器速度。

py2wl 的路径是：用户写普通 Python → compat 层翻译成 WL 表达式 → Wolfram 内部 Compile[] 自动生成 C → clang O3 编译为原生 AArch64。整个过程对用户完全透明，他们甚至不知道底层发生了什么。

这其实是 Wolfram 语言设计哲学的核心优势——它是一门从一开始就把"符号表达式可以被分析和变换"作为第一公民的语言。`Compile` 能工作是因为 WL 表达式在语义上足够干净，编译器知道每个操作的确切类型和结构。Python 做不到这一点，因为 Python 的动态类型让静态分析极其困难，Cython 的本质就是在给 Python 加上 WL 本来就有的那些类型标注。

而py2wl 把 Wolfram 的编译基础设施免费送给了每一个 Python 用户。

---

## 快速开始

### 直接调用内核

```python
import os
os.environ["WOLFRAM_EXEC"] = "/path/to/WolframKernel"

from py2wl import WolframKernel

kernel = WolframKernel()

# 数值计算
result = kernel.evaluate("Eigenvalues[{{1,2},{3,4}}]")
print(result)   # [5.37228, -0.37228]

# 生成图像（无头模式自动 Rasterize，无需手动处理）
path = kernel.evaluate_to_file(
    "Plot3D[Sin[x + y^2], {x, -3, 3}, {y, -2, 2}]",
    fmt="png"
)
print(f"图片已保存: {path}")

kernel.close()
```

### 兼容层（drop-in 替换）

```python
# 把这一行
import numpy as np
# 改成
from py2wl.compat import numpy as np

# 以下代码无需任何修改
A = np.array([[1.0, 2.0], [3.0, 4.0]])
vals, vecs = np.linalg.eig(A)      # → Wolfram Eigensystem
result = np.fft.fft([1, 2, 3, 4])  # → Wolfram Fourier
```

```python
from py2wl.compat import scipy as sp

# SciPy 函数同样透明代理
result = sp.linalg.expm(A)         # → Wolfram MatrixExp
```

---

## 映射覆盖范围

共 **894 条**函数映射，覆盖主流科学计算库：

| 库 | 映射数 | 典型函数 |
|----|--------|---------|
| NumPy | 222 | `linalg.*`, `fft.*`, `random.*`, `polynomial.*` |
| SciPy | 144 | `linalg.*`, `signal.*`, `optimize.*`, `stats.*` |
| pandas | 125 | `DataFrame.*`, `read_csv`, `groupby`, `merge` |
| SymPy | 124 | `diff`, `integrate`, `solve`, `simplify` |
| scikit-learn | 60 | `LinearRegression`, `KMeans`, `PCA`, `RandomForest` |
| PyTorch | 66 | `tensor`, `matmul`, `nn.*` |
| TensorFlow | 55 | `constant`, `matmul`, `reduce_*` |
| Matplotlib | 47 | `plot`, `scatter`, `hist`, `plot_surface` |
| 监控/计时 | 24 | `time.*`, `psutil.*`, `tqdm` |
| 性能优化 | 27 | `numba.jit`, `joblib.Parallel`, `numpy.einsum` |

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WOLFRAM_EXEC` | —（必填） | Wolfram 内核可执行文件路径 |
| `WOLFRAM_PARALLEL_KERNELS` | 自动检测 CPU 数 | 并行子内核数量 |
| `WOLFRAM_INIT_TIMEOUT` | `60` | 内核启动超时（秒） |
| `WOLFRAM_CACHE_DIR` | `/sdcard/wolfram_cache` | 图像文件缓存目录 |
| `WOLFRAM_FAULT_MODE` | `strict` | 容错模式：`strict` / `auto-ai` / `interactive` |
| `WOLFRAM_MAPPINGS_DIR` | 内置 mappings/ | 自定义映射规则目录 |
| `PY2WL_RASTER_SIZE` | `1920x1080` | 图像输出分辨率（无头模式） |

---

## 架构

```
Python 调用
    │
    ▼
兼容层（py2wl.compat）
  sys.modules 劫持 → 透明代理
    │
    ▼
解析引擎（ResolutionEngine）
  Trie 精确查找 → 倒排模糊匹配 → AI 兜底
    │
    ▼
结果缓存（ResultCache）
  RAM LRU dict，浮点精度归一化，双哈希去重
    │  命中直接返回
    ▼ 未命中
数据序列化（converters）
  numpy array → WXF PackedArray（保留 BLAS 路径）
    │
    ▼
Wolfram 内核（WolframKernel）
  WSTP socket，worker 线程，并行子内核懒加载
    │
    ▼
结果反序列化 → Python 原生类型
```

**关键设计**：大矩阵走 WXF 二进制文件传输（而非字符串序列化），Wolfram 端用 `Developer\`ToPackedArray[N[...]]` 强制打包为连续内存，确保 BLAS/LAPACK 调用路径完整保留。

---

## 容错系统

遇到未映射函数或执行错误时，根据 `WOLFRAM_FAULT_MODE` 决定行为：

- **`strict`**（默认）：直接抛出异常
- **`auto-ai`**：调用配置的 AI provider 自动推断 Wolfram 等价函数并重试
- **`interactive`**：暂停到终端，展示候选函数列表，由用户选择

支持的 AI provider：Claude、DeepSeek、Gemini、Groq。

```bash
export WOLFRAM_FAULT_MODE=auto-ai
export WOLFRAM_AI_PLUGIN=claude
export ANTHROPIC_API_KEY=sk-...
```

---

## 扩展映射规则

在 YAML 文件中添加自定义函数映射，无需修改源码：

```yaml
- python_path: mylib.special_func
  wolfram_function: "MyWLFunc[#1, #2]&"
  input_converters: [to_wl_list, to_wl_scalar]
  output_converter: from_wxf
  tags: [mylib, special]
  description: "My custom function"
  cacheable: false   # 随机/时间函数加此项
  numeric: true   浮点模式加此项
```

```bash
export WOLFRAM_MAPPINGS_DIR=/path/to/my/mappings
```

---

## 安装

```bash
# 基础安装
pip install py2wl

# 完整安装（含 AI 容错）
pip install "py2wl[ai]"
```

设置内核路径：

```bash
# Linux / macOS
export WOLFRAM_EXEC=/usr/local/Wolfram/WolframEngine/14.1/Executables/WolframKernel

# Android（Termux）
export WOLFRAM_EXEC=/data/data/com.termux/files/home/wolfram-extract/opt/Wolfram/WolframEngine/14.1/Executables/math
```

注:py2wl 在计算密集型任务（如特征值分解、奇异值分解）上性能远超 NumPy，但在通信密集型任务（如矩阵乘法）上会略慢。这是跨进程计算的正常现象，因为数据在两个进程间传输需要物理时间。如果您需要极致的矩阵乘法速度，建议使用原生 NumPy；而对于复杂算法或需要符号计算的任务，py2wl 是您的首选。
---

## 许可证

MIT License

Wolfram Engine 需遵守 [Wolfram Engine 许可协议](https://www.wolfram.com/legal/agreements/wolfram-engine.html)，免费用于非生产用途。

