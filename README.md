# py2wl

**Python → Wolfram Language bridge**

把 Wolfram Engine 当作 Python 的计算后端。兼容层让现有的 NumPy / SciPy / pandas / SymPy / PyTorch / scikit-learn / Matplotlib 代码几乎不改一行就能路由到 Wolfram 内核执行，并通过自动 C 编译在计算密集型场景获得显著加速。

```bash
pip install py2wl
pip install "py2wl[jupyter]"   # 含 Jupyter 集成
pip install "py2wl[all]"       # 全功能
```

> **前提**：系统中需要已安装 [Wolfram Engine](https://www.wolfram.com/engine/)（提供免费非商业许可证），并设置环境变量 `WOLFRAM_EXEC` 指向内核可执行文件。

---

## 为什么用 py2wl？

| 场景 | py2wl vs NumPy |
|------|---------------|
| 特征值分解 n=1000 | **快 10×**（Wolfram 多核 LAPACK） |
| 奇异值分解 n=800  | **快 2.6×** |
| 自定义计算函数    | **快 4–9×**（自动 C 编译 + O3，用户零感知） |
| 符号计算          | NumPy 不支持，Wolfram 原生 |
| 矩阵乘法          | 慢 5–8×（跨进程传输开销，属正常现象） |

> 📱 **实测设备：OPPO A35 5G**
> 处理器：骁龙 480，台积电 8nm，8 核（2×2.0GHz + 6×1.8GHz）；内存：8GB；首发价约 289 USD。入门级消费机，非开发机，非服务器。**如果您的设备比这台更新，预期结果只会更好。**

Wolfram Engine 无 GUI 版约 **600MB**，比 NumPy + SciPy + pandas + scikit-learn + Matplotlib 全家桶（2–3GB）更小，覆盖的计算领域更广。用户不需要手写 Cython，不需要 Numba 装饰器，底层自动完成编译。

---

## 快速开始

### 直接调用内核

```python
import os
os.environ["WOLFRAM_EXEC"] = "/path/to/WolframKernel"

from py2wl import WolframKernel

kernel = WolframKernel()

# 数值计算
print(kernel.evaluate("Eigenvalues[{{1,2},{3,4}}]"))
# → [5.37228, -0.37228]

# 符号计算
print(kernel.evaluate("Integrate[Sin[x]^2, {x, 0, Pi}]"))
# → Pi/2

# 生成图像（无头模式自动 Rasterize，无需手动处理）
path = kernel.evaluate_to_file(
    "Plot3D[Sin[x + y^2], {x, -3, 3}, {y, -2, 2}]",
    fmt="png"
)

kernel.close()
```

### 兼容层（drop-in 替换）

```python
# 把这一行
import numpy as np
# 改成
from py2wl.compat import numpy as np

# 以下代码完全不用改
A = np.random.randn(1000, 1000)
vals, vecs = np.linalg.eig(A)      # → Wolfram Eigensystem，8核并行
result = np.fft.fft([1, 2, 3, 4])  # → Wolfram Fourier
```

```python
from py2wl.compat import scipy as sp
result = sp.linalg.expm(A)         # → Wolfram MatrixExp
```

### Jupyter 集成

```python
# 第一个 cell 执行一次，之后图像自动内嵌
import py2wl.jupyter
```

```python
from py2wl.jupyter import wl, wl_img

# 数值/符号结果自动格式化显示
wl("Prime[1000]")                          # → 7919
wl("Integrate[Sin[x]^2, {x, 0, Pi}]")     # → π/2

# 绘图自动内嵌为高清图像
wl("Plot3D[Sin[x + y^2], {x,-3,3}, {y,-2,2}]")
wl_img("WordCloud[ExampleData[{'Text','AliceInWonderland'}]]", width=800)
```

```
%%wl
A = RandomReal[{0, 1}, {500, 500}];
Eigenvalues[A]
```

---

## 分布式内核池

py2wl 支持将多台设备上的 Wolfram 内核统一管理，自动把计算请求分发到最合适的内核。

**两层并行，各司其职：**

```
KernelPool（请求调度层，跨设备）
    │  把不同的 Python 请求路由到不同设备
    │  目标：提升整体吞吐量
    │
    ├─ 手机  WolframKernel（8核，Wolfram Parallel 任务内并行）
    ├─ 电脑  WolframKernel（16核，Wolfram Parallel 任务内并行）
    └─ 平板  WolframKernel（4核，Wolfram Parallel 任务内并行）
              │  把单个大任务拆分到多个子内核
              │  目标：缩短单任务延迟
```

Wolfram 的 `Parallel` 族函数负责把一个大计算拆成子任务分发到多个子内核——这是**任务内并行**。`KernelPool` 解决的是另一个问题：当多个独立的 Python 请求同时到来（多个 notebook cell、多个 API 调用、流水线的多个组件），如何把它们分配到多台设备的空闲内核上——这是**请求级并发**。两层完全正交，叠加使用。
真·穷："手头有十几台退役的旧手机、平板，放着也是放着，接上电源跑 py2wl server，组个“垃圾佬集群”，低成本榨干剩余价值。这比租云服务器便宜多了（电费都不用自己交 😂）。"

**快速上手：**

```bash
# 远端电脑启动服务
export WOLFRAM_EXEC=/path/to/WolframKernel
python -m py2wl.server --port 9999
```

```python
# 手机/本机使用
import os
os.environ["PY2WL_KERNELS"] = "local,192.168.1.100:9999"

from py2wl.pool import KernelPool

with KernelPool() as pool:
    # 自动路由到最合适的内核
    result = pool.execute("Eigenvalues[RandomReal[{0,1},{1000,1000}]]")

    # 查看各内核实时状态
    print(pool.status())
    # [{'id':0,'host':'local','status':'idle','queue_len':0,'avg_ms':651,'cpu_score':8},
    #  {'id':1,'host':'192.168.1.100:9999','status':'idle','queue_len':0,'avg_ms':203,'cpu_score':16}]
```

**自定义调度策略：**

```python
# my_scheduler.py — 用户自定义调度器
def scheduler(kernels, expr):
    """
    kernels: 所有内核的状态快照列表
      .id, .host, .status, .queue_len, .avg_ms, .cpu_score
    expr: 即将执行的 WL 表达式字符串
    返回: 选中的内核 id
    """
    # 示例：大矩阵推给算力最强的设备
    if "1000" in expr or "2000" in expr:
        online = [k for k in kernels if k.status != "offline"]
        return max(online, key=lambda k: k.cpu_score).id
    # 其他任务选队列最短的
    idle = [k for k in kernels if k.status == "idle"]
    return min(idle or kernels, key=lambda k: k.queue_len).id
```

```bash
export PY2WL_SCHEDULER=/path/to/my_scheduler.py
export PY2WL_KERNEL_CPU_SCORES=8,16,4   # 手动指定各内核算力权重
```

默认调度策略：优先选空闲且队列最短的内核；全部繁忙时选历史平均耗时最短的；离线内核自动跳过并定期重连。

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
| Matplotlib / Seaborn | 47 | `plot`, `scatter`, `hist`, `plot_surface` |
| 监控 / 性能 | 51 | `time.*`, `psutil.*`, `numba.jit`, `joblib` |

---

## 自动 C 编译加速

内核启动后自动设置全局编译选项，对所有经过 `Compile[]` 的 WL 函数生效：

```wolfram
$CompilationTarget = "C"            (* 生成原生机器码，而非 WVM 字节码 *)
Compile`$CCompilerOptions = {"-O3"} (* 等价 gcc -O3，最高级别优化 *)
```

用户无需任何操作。C 编译失败时 Wolfram 自动退回 WVM，不会报错崩溃。在 Android ARM 设备上，C 编译相比默认 WVM 约快 **2–4×**，叠加 O3 优化后综合加速可达 **4–9×**。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WOLFRAM_EXEC` | —（必填） | Wolfram 内核可执行文件路径 |
| `WOLFRAM_PARALLEL_KERNELS` | 自动检测 CPU 数 | 并行子内核数量 |
| `WOLFRAM_INIT_TIMEOUT` | `60` | 内核启动超时（秒） |
| `WOLFRAM_CACHE_DIR` | `/sdcard/wolfram_cache` | 图像文件缓存目录 |
| `WOLFRAM_FAULT_MODE` | `strict` | 容错模式：`strict` / `auto-ai` / `interactive` |
| `WOLFRAM_MAPPINGS_DIR` | 内置 `mappings/` | 自定义映射规则目录 |
| `PY2WL_RASTER_SIZE` | `1920x1080` | 图像输出分辨率（无头模式） |
| `PY2WL_KERNELS` | `local` | 内核池地址列表，逗号分隔 |
| `PY2WL_KERNEL_CPU_SCORES` | 自动检测 | 各内核算力权重，逗号分隔 |
| `PY2WL_SCHEDULER` | 内置默认 | 自定义调度器脚本路径 |
| `PY2WL_SERVER_PORT` | `9999` | 远程服务端监听端口 |

---

## 架构

```
Python 调用
    │
    ▼
KernelPool（可选，分布式调度）
  请求级并发，跨设备路由，自定义调度策略
  心跳检测，自动重连，滑动平均耗时统计
    │  本地直连 WolframKernel
    │  远程通过 RemoteKernel → TCP → KernelServer
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
  RAM LRU dict，cmd_hash O(1) 查找
  result_hash 去重：numpy → tobytes()，标量 → round(12)+repr
    │  命中直接返回
    ▼ 未命中
数据序列化（converters）
  numpy array → WXF PackedArray（保留 BLAS 路径）
    │
    ▼
Wolfram 内核（WolframKernel）
  WSTP socket，worker 线程，懒加载并行子内核
  自动 C 编译：$CompilationTarget=C，$CCompilerOptions=-O3
    │  Wolfram Parallel 族函数在此层做任务内并行
    ▼
结果反序列化 → Python 原生类型
    │
    ▼
Jupyter display hook（可选）
  图像路径 → IPython.display.Image 自动内嵌
  %%wl magic → 整个 cell 执行 WL
```

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
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 扩展映射规则

用 YAML 添加自定义函数映射，无需修改源码：

```yaml
- python_path: mylib.special_func
  wolfram_function: "MyWLFunc[#1, #2]&"
  input_converters: [to_wl_list, to_wl_scalar]
  output_converter: from_wxf
  cacheable: false
  description: "My custom function"
```

```bash
export WOLFRAM_MAPPINGS_DIR=/path/to/my/mappings
```

---

## 安装

```bash
pip install py2wl
pip install "py2wl[jupyter]"   # 含 Jupyter 集成
pip install "py2wl[ai]"        # 含 AI 容错
pip install "py2wl[all]"       # 全功能
```

设置内核路径：

```bash
# Linux / macOS
export WOLFRAM_EXEC=/usr/local/Wolfram/WolframEngine/14.1/Executables/WolframKernel

# Android（Termux）
export WOLFRAM_EXEC=/data/data/com.termux/files/home/wolfram-extract/opt/Wolfram/WolframEngine/14.1/Executables/math
```

---

## 许可证

MIT License

Wolfram Engine 需遵守 [Wolfram Engine 许可协议](https://www.wolfram.com/legal/agreements/wolfram-engine.html)，免费用于非商业用途。
