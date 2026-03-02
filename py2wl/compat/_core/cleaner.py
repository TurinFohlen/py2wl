"""
cleaner.py — 清理残留的 Wolfram 内核进程（跨平台）
在无法使用 pkill 的系统上，尝试使用 psutil 或遍历 /proc。
如果无法安装 psutil，则使用 subprocess 调用系统命令，并提供备选方案。
"""

import os
import signal
import subprocess
import sys
import logging

log = logging.getLogger(__name__)

def kill_process(pid):
    """尝试用 SIGKILL 杀死指定 PID 的进程"""
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        # 进程已不存在
        return False
    except PermissionError:
        log.warning(f"无权限杀死进程 {pid}")
        return False

def find_and_kill_wolfram_processes():
    """查找并杀死所有 WolframKernel 和 math 进程"""
    killed = False
    if sys.platform.startswith('linux'):
        # Linux 下尝试使用 pgrep（最快）
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'WolframKernel|math'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split()
                for pid_str in pids:
                    try:
                        pid = int(pid_str)
                        if kill_process(pid):
                            killed = True
                    except ValueError:
                        pass
                if killed:
                    log.debug(f"通过 pgrep 杀死 {len(pids)} 个进程")
        except FileNotFoundError:
            # 没有 pgrep，回退到遍历 /proc
            try:
                for pid in os.listdir('/proc'):
                    if not pid.isdigit():
                        continue
                    try:
                        with open(f'/proc/{pid}/cmdline', 'rb') as f:
                            cmdline = f.read().decode(errors='ignore')
                        if 'WolframKernel' in cmdline or 'math' in cmdline:
                            if kill_process(int(pid)):
                                killed = True
                    except (IOError, OSError):
                        continue
            except FileNotFoundError:
                # /proc 不可访问，放弃
                pass
    elif sys.platform == 'win32':
        # Windows 下使用 taskkill
        try:
            result = subprocess.run(
                ['taskkill', '/F', '/IM', 'math.exe'],
                capture_output=True,
                check=False
            )
            if result.returncode == 0:
                killed = True
            # WolframKernel 可能也有 .exe
            result2 = subprocess.run(
                ['taskkill', '/F', '/IM', 'WolframKernel.exe'],
                capture_output=True,
                check=False
            )
            if result2.returncode == 0:
                killed = True
        except FileNotFoundError:
            pass
    else:
        # macOS 或其他 Unix，尝试使用 pkill
        try:
            result = subprocess.run(
                ['pkill', '-9', '-f', 'WolframKernel|math'],
                capture_output=True,
                check=False
            )
            if result.returncode == 0:
                killed = True
        except FileNotFoundError:
            pass

    return killed

def cleanup():
    """执行清理，返回是否杀死过进程"""
    return find_and_kill_wolfram_processes()