wolfram-pty

A lightweight Python bridge to the Wolfram Engine for restricted environments – zero scientific dependencies, 800+ mapped functions.

https://img.shields.io/pypi/v/wolfram-pty
https://img.shields.io/badge/License-MIT-yellow.svg

---

🎯 What is it?

wolfram-pty lets you call the full power of Wolfram Language directly from Python – perfect for constrained environments like Termux, Docker containers, Raspberry Pi, CI/CD pipelines where installing heavy scientific libraries (numpy, pandas, etc.) is impractical.

· Zero scientific dependencies – all heavy lifting is done by the Wolfram Engine; Python only needs the standard library.
· Persistent kernel session – a single kernel instance keeps state across calls.
· File‑based output – results are saved as JSON/PNG/TXT and read back with Python’s built‑in modules.
· Automatic caching – identical expressions return cached results instantly.
· Metadata‑driven compatibility layer – already maps 835+ functions from NumPy, SciPy, pandas, PyTorch, TensorFlow, SymPy, scikit‑learn, Matplotlib and more.
· Resilient – automatic kernel restart on failure, serialised request queue, no activation race conditions.

---

✨ Features

· 🔥 Zero scientific libraries – no numpy, no pandas, just plain Python.
· 🚀 Persistent kernel – start once, evaluate many times with full state retention.
· 📦 File output mode – reliable, parse‑free communication via files.
· 🧠 Smart cache – reuse results for identical expressions.
· 🧩 Drop‑in replacements – use familiar NumPy/SciPy syntax, powered by Wolfram.
· 🛠️ Auto‑recovery – kernel crashes? It restarts automatically.
· 🔌 Easy configuration – set two environment variables and you’re ready.

---

🚀 Quick Start

Installation

```bash
pip install wolfram-pty
```

⚠️ Note: This package does not include the Wolfram Engine. You must obtain a valid licence from Wolfram Research and install it yourself.

Set up the kernel path

Tell wolfram-pty where your Wolfram kernel is:

```bash
export WOLFRAM_EXEC=/path/to/your/wolfram-kernel   # e.g. /usr/local/bin/math
export WOLFRAM_PWFILE=~/.Wolfram/Licensing/mathpass   # optional, password file
```

Basic usage

```python
from wolfram_pty import WolframKernel

k = WolframKernel()          # singleton kernel, starts only once

# Evaluate an expression, get a string back
print(k.evaluate("2+2"))     # "4"

# Batch evaluation
results = k.batch_evaluate(["Range[5]", "Pi", "Det[{{1,2},{3,4}}]"])
print(results)               # ["{1,2,3,4,5}", "Pi", "-2"]

# Use the familiar NumPy interface (powered by Wolfram)
from wolfram_pty.compat import numpy as np
a = np.array([1,2,3])        # bypasses kernel – pure Python list
b = np.fft.fft(a)            # actually calls Wolfram's Fourier
print(b)                     # complex array (read from a JSON file)
```

---

📁 File Output Mode (Core Mechanism)

All results are written to files and read back with standard Python – no parsing hassle, no dependencies.

```python
# Save result as a JSON file
file_path = k.evaluate_to_file("Range[5]", fmt="json")
print(file_path)             # /sdcard/wolfram_out/xxx.json

# Read it with the built‑in json module
import json
with open(file_path) as f:
    data = json.load(f)      # [1,2,3,4,5]

# Generate a plot and save as PNG
img_path = k.evaluate_to_file("Plot[Sin[x], {x,0,2Pi}]", fmt="png")
```

---

🧠 Automatic Caching

The second time you ask for the same expression, the result is returned instantly from the cache.

```python
# First call – kernel runs
path1 = k.evaluate_to_file("Range[10000]")

# Second call – cache hit, almost instantaneous
path2 = k.evaluate_to_file("Range[10000]")

# Force recompute (e.g. for random numbers)
path3 = k.evaluate_to_file("RandomReal[1,1000]", no_cache=True)

# Clean cache files older than 7 days
k.clear_cache(older_than_days=7)
```

---

📚 Supported Libraries (835+ functions)

The wolfram_pty.compat submodule provides drop‑in replacements for popular scientific Python libraries – everything is translated to Wolfram calls.

Library Functions Example
NumPy 200+ np.array, np.fft.fft, np.linalg.solve
SciPy 143 scipy.integrate.quad, scipy.optimize.minimize
pandas 125 pd.DataFrame, df.groupby, pd.read_csv
PyTorch 66 torch.tensor, torch.add, torch.nn.ReLU
TensorFlow 55 tf.constant, tf.matmul, tf.nn.softmax
SymPy 71 sympy.symbols, sympy.diff, sympy.solve
scikit‑learn 60 sklearn.preprocessing.StandardScaler, sklearn.cluster.KMeans
Matplotlib 47 plt.plot, plt.imshow, seaborn
Performance 50+ tqdm, logging, time, psutil

How to use them:

```python
from wolfram_pty.compat import numpy as np
from wolfram_pty.compat import scipy
from wolfram_pty.compat import pandas as pd
# … and so on
```

---

⚙️ Configuration

All settings are controlled via environment variables:

Variable Default Description
WOLFRAM_EXEC /root/.../math Path to the Wolfram kernel executable
WOLFRAM_PWFILE ~/.Wolfram/Licensing/mathpass Path to the password file (optional)
WOLFRAM_CACHE_DIR /sdcard/wolfram_cache Directory for cached results
WOLFRAM_OUT_DIR /sdcard/wolfram_out Default output directory for files

---

⚠️ Important Legal Notice

This software only provides a communication interface to the Wolfram Engine and does not include, bundle, or distribute the Wolfram Engine itself.
Users must obtain a valid licence directly from Wolfram Research and comply with their terms.
The developers of wolfram-pty assume no liability for any unauthorised use of the Wolfram Engine.

---

📄 License

MIT License © 2025 TurinFohlen

---

🤝 Contributing

Issues and pull requests are welcome! If you’d like to add mappings for more functions, create a YAML file in mappings/ and make sure tests pass.

---

Happy computing – even in the most constrained environments!
