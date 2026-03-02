#!/usr/bin/env python3
"""
release.py — 一键打包并输出到 /mnt/user-data/outputs/
用法：python release.py [版本号]
      python release.py v1.0.7
      python release.py          # 自动递增
"""
import os, re, sys, shutil, subprocess, pathlib, glob

ROOT    = pathlib.Path(__file__).parent
OUTDIR  = pathlib.Path("/mnt/user-data/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── 确定版本号 ───────────────────────────────────────────────────
if len(sys.argv) > 1:
    version = sys.argv[1].lstrip("v")
else:
    # 自动从 outputs 目录找最新包，递增 patch
    existing = sorted(glob.glob(str(OUTDIR / "wolfram-pty_v*.7z")))
    if existing:
        m = re.search(r'v(\d+)\.(\d+)\.(\d+)', existing[-1])
        if m:
            major, minor, patch = int(m[1]), int(m[2]), int(m[3])
            version = f"{major}.{minor}.{patch+1}"
        else:
            version = "1.0.0"
    else:
        version = "1.0.0"

tag      = f"wolfram-pty_v{version}"
out_file = OUTDIR / f"{tag}.7z"

# ── 清理旧包 ─────────────────────────────────────────────────────
for old in OUTDIR.glob("wolfram-pty_v*.7z"):
    old.unlink()
    print(f"删除旧包: {old.name}")

# ── 清理 __pycache__ ─────────────────────────────────────────────
for cache in ROOT.rglob("__pycache__"):
    shutil.rmtree(cache, ignore_errors=True)

# ── 打包 ─────────────────────────────────────────────────────────
print(f"打包 → {out_file.name} ...")
result = subprocess.run(
    ["7z", "a", str(out_file), str(ROOT), "-xr!*.pyc", "-mx9"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("7z 错误:", result.stderr)
    sys.exit(1)

size_kb = out_file.stat().st_size // 1024
print(f"✅  {out_file.name}  ({size_kb} KB)")
