wolfram-pty

在受限环境中通过 Python 无缝调用 Wolfram 引擎的轻量级桥梁，零科学库依赖，覆盖 800+ 常用函数。

https://img.shields.io/pypi/v/wolfram-pty
https://img.shields.io/badge/License-MIT-yellow.svg

🎯 它是什么？

· Wolfram 引擎的 Python 接口：让你能在 Python 中直接使用 Wolfram 语言的全部计算能力（数学、符号、数值、可视化）。
· 专为受限环境设计：完美运行于 Termux、Docker 容器、树莓派、CI/CD 流水线等无法安装庞大科学库的地方。
· 零科学库依赖：所有计算由 Wolfram 引擎完成，Python 只需标准库读写文件，彻底摆脱 numpy、pandas 等依赖地狱。
· 开箱即用的兼容层：内置 800+ 个 NumPy/SciPy/pandas/PyTorch 等函数的自动映射，一行代码切换（不用改变习惯）。

---

✨ 核心特性

· 🔥 零科学库依赖 – Python 仅用标准库，计算结果通过文件传递。
· 🚀 持久内核会话 – 单例内核一次启动，多次调用，状态保持。
· 📦 文件输出模式 – 结果直接保存为 JSON/PNG/TXT，稳定可靠。
· 🧠 自动缓存 – 相同表达式计算结果自动缓存，重复调用瞬间返回。
· 🧩 元数据驱动 – 目前已支持 835 个常用函数，覆盖 NumPy、SciPy、pandas、PyTorch、TensorFlow、SymPy、scikit-learn、Matplotlib 等。
· 🛠️ 优雅的错误恢复 – 内核崩溃自动重启，队列串行执行，无竞态。
· 🔌 即插即用 – 只需配置内核路径和密码文件，一行代码接入。

---

🚀 快速开始

安装

```bash
pip install wolfram-pty
```

⚠️ 注意：本库不包含 Wolfram Engine，您需要自行获取合法授权并安装。

配置内核路径

通过环境变量指定 Wolfram 内核位置：

```bash
export WOLFRAM_EXEC=/path/to/your/wolfram-kernel   # 例如 /usr/local/bin/math
export WOLFRAM_PWFILE=~/.Wolfram/Licensing/mathpass   # 密码文件路径（可选）
```

基本使用

```python
from wolfram_pty import WolframKernel

k = WolframKernel()          # 单例内核，只启动一次

# 直接计算，返回字符串
print(k.evaluate("2+2"))     # "4"

# 批量计算
results = k.batch_evaluate(["Range[5]", "Pi", "Det[{{1,2},{3,4}}]"])
print(results)               # ["{1,2,3,4,5}", "Pi", "-2"]

# 使用兼容的 NumPy 接口
from wolfram_pty.compat import numpy as np
a = np.array([1,2,3])        # 创建数组（短路，不走内核）
b = np.fft.fft(a)            # 实际调用 Wolfram 的 Fourier
print(b)                     # 复数数组
```

---

📁 文件输出模式（核心机制）

所有计算结果均通过文件返回，确保零依赖且稳定。

```python
# 将结果保存为 JSON 文件
file_path = k.evaluate_to_file("Range[5]", fmt="json")
print(file_path)             # /sdcard/wolfram_out/xxx.json

# 用标准库读取
import json
with open(file_path) as f:
    data = json.load(f)      # [1,2,3,4,5]

# 生成图像并保存为 PNG
img_path = k.evaluate_to_file("Plot[Sin[x], {x,0,2Pi}]", fmt="png")
```

---

🧠 自动缓存

相同表达式第二次调用时直接返回缓存文件，避免重复计算。

```python
# 第一次调用（触发内核计算）
path1 = k.evaluate_to_file("Range[10000]")

# 第二次调用（命中缓存，瞬间返回）
path2 = k.evaluate_to_file("Range[10000]")

# 强制重新计算（如随机数）
path3 = k.evaluate_to_file("RandomReal[1,1000]", no_cache=True)

# 清理7天前的缓存
k.clear_cache(older_than_days=7)
```

---

📚 已支持的库函数（835+）

通过 wolfram_pty.compat 子模块，您可以使用熟悉的科学计算库语法，底层自动映射到 Wolfram 引擎。

库 函数数量 示例
NumPy 200+ np.array, np.fft.fft, np.linalg.solve
SciPy 143 scipy.integrate.quad, scipy.optimize.minimize
pandas 125 pd.DataFrame, df.groupby, pd.read_csv
PyTorch 66 torch.tensor, torch.add, torch.nn.ReLU
TensorFlow 55 tf.constant, tf.matmul, tf.nn.softmax
SymPy 71 sympy.symbols, sympy.diff, sympy.solve
scikit-learn 60 sklearn.preprocessing.StandardScaler, sklearn.cluster.KMeans
Matplotlib 47 plt.plot, plt.imshow, seaborn
性能/监控 50+ tqdm, logging, time, psutil

使用方法：

```python
from wolfram_pty.compat import numpy as np
from wolfram_pty.compat import scipy
from wolfram_pty.compat import pandas as pd
# 其他库同理
```

---

⚙️ 配置选项

通过环境变量自定义行为：

变量 默认值 说明
WOLFRAM_EXEC /root/.../math Wolfram 内核可执行文件路径
WOLFRAM_PWFILE ~/.Wolfram/Licensing/mathpass 密码文件路径
WOLFRAM_CACHE_DIR /sdcard/wolfram_cache 缓存目录
WOLFRAM_OUT_DIR /sdcard/wolfram_out 输出文件默认目录

---

⚠️ 重要法律声明

本软件（wolfram-pty）仅提供与 Wolfram 引擎的通信接口，不包含 Wolfram Engine 本身。
用户需自行从 Wolfram 官方 获取并合法安装 Wolfram Engine，并确保遵守其授权协议。
本软件开发者不对因未授权使用 Wolfram Engine 而产生的任何法律问题承担责任。

---

📄 许可证

MIT License © 2025 TurinFohlen

---

🤝 贡献

欢迎提交 issue 和 PR！如果您希望增加新的函数映射，请在 mappings/ 下添加 YAML 文件，并确保通过测试。
