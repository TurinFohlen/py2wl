import sys
sys.path.insert(0, '/root/wolfproject')  # 确保能找到 py2wl.py
from py2wl import WolframKernel
import time
import os

# 可选：设置环境变量指定内核路径（如果 py2wl.py 中未配置）
os.environ["WOLFRAM_EXEC"] = "/root/wolfram-extract/opt/Wolfram/WolframEngine/14.1/Executables/math"

def test_session():
    print("="*50)
    print("开始测试 WolframKernel 持续会话")
    print("="*50)
    
    # 1. 启动内核（单例）
    try:
        kernel = WolframKernel()
        print("✅ 内核启动成功（或已存在）")
    except Exception as e:
        print(f"❌ 内核启动失败: {e}")
        return

    # 2. 发送第一个表达式
    try:
        result1 = kernel.evaluate("2+2")  # evaluate 返回字符串
        print(f"第一次计算 (2+2) 结果: {result1}")
    except Exception as e:
        print(f"❌ 第一次计算失败: {e}")
        return

    # 3. 发送第二个表达式，定义变量
    try:
        result2 = kernel.evaluate("x = 5")  # 通常没有输出（可能返回 "5" 或空）
        print(f"定义变量 x = 5, 返回: {result2}")
    except Exception as e:
        print(f"❌ 定义变量失败: {e}")
        return

    # 4. 发送第三个表达式，引用变量
    try:
        result3 = kernel.evaluate("x")
        print(f"读取变量 x 的值: {result3}")
    except Exception as e:
        print(f"❌ 读取变量失败: {e}")
        return

    # 5. 等待几秒，再发送一个表达式，检查内核是否仍在运行
    print("等待 5 秒...")
    time.sleep(5)
    try:
        result4 = kernel.evaluate("x + 10")
        print(f"再次计算 (x+10) 结果: {result4}")
    except Exception as e:
        print(f"❌ 后续计算失败: {e}")
        return

    # 6. 检查内核进程是否存活（可选）
    import psutil
    if hasattr(kernel, '_child') and kernel._child:
        pid = kernel._child.pid
        if psutil.pid_exists(pid):
            print(f"✅ 内核进程 (PID {pid}) 仍在运行")
        else:
            print(f"❌ 内核进程 (PID {pid}) 已退出")
    else:
        print("⚠️ 无法获取内核进程信息")

    print("\n🎉 测试完成！如果以上所有步骤都成功，说明内核会话可以持续保持。")

if __name__ == "__main__":
    test_session()