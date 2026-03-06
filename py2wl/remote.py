"""
py2wl.remote — 远程 WolframKernel 客户端
==========================================

连接运行了 py2wl.server 的远端设备，接口与本地 WolframKernel 完全相同。

协议（length-prefixed TCP）：
    请求：[4字节大端长度][UTF-8 表达式]
    响应：[4字节大端长度][WXF 二进制结果]
    错误：[0xFFFFFFFF][4字节错误消息长度][UTF-8 错误消息]
    Ping：请求长度 = 0 → 响应长度 = 0

用法：
    kernel = RemoteKernel("192.168.1.100:9999")
    result = kernel.evaluate("Prime[1000]")
    kernel.close()
"""

import logging
import socket
import struct
import threading
import time
from typing import Any

log = logging.getLogger("py2wl.remote")

# 连接超时（秒）
_CONNECT_TIMEOUT = 10
# 接收超时（秒）— 大矩阵运算可能需要较长时间
_RECV_TIMEOUT    = 300
# 自动重连最大次数
_MAX_RECONNECT   = 3


class RemoteKernel:
    """
    远程 Wolfram 内核客户端。

    接口与 WolframKernel 保持一致：
      .evaluate(expr)              → Any
      .evaluate_to_file(expr, fmt) → str（本地临时路径）
      .ping()                      → None（连通性检查）
      .close()
    """

    def __init__(self, address: str):
        """
        参数：
            address: "host:port"，如 "192.168.1.100:9999"
        """
        parts = address.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"地址格式错误，应为 'host:port'，got: {address!r}")
        self.host = parts[0]
        self.port = int(parts[1])
        self.address = address
        self.sock: socket.socket = None
        self._lock = threading.Lock()
        self._connect()

    # ── 连接管理 ─────────────────────────────────────────────

    def _connect(self) -> None:
        """建立 TCP 连接"""
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(_CONNECT_TIMEOUT)
        s.connect((self.host, self.port))
        s.settimeout(_RECV_TIMEOUT)
        # 禁用 Nagle 算法，减少小包延迟
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        log.debug(f"已连接到远程内核 {self.address}")

    def _reconnect(self) -> None:
        """断线重连"""
        for attempt in range(1, _MAX_RECONNECT + 1):
            try:
                log.info(f"重连 {self.address} (第 {attempt}/{_MAX_RECONNECT} 次)…")
                self._connect()
                return
            except Exception as e:
                if attempt == _MAX_RECONNECT:
                    raise ConnectionError(
                        f"重连 {self.address} 失败（已重试 {_MAX_RECONNECT} 次）: {e}"
                    ) from e
                time.sleep(1.5 ** attempt)   # 指数退避

    # ── 底层收发 ─────────────────────────────────────────────

    def _recv_exact(self, n: int) -> bytes:
        """确保接收恰好 n 字节，否则抛出 ConnectionError"""
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(f"连接断开（期望 {n} 字节，已收 {len(buf)} 字节）")
            buf.extend(chunk)
        return bytes(buf)

    def _send_request(self, expr: str) -> bytes:
        """
        发送请求并返回 WXF 响应字节。
        加锁保证同一连接上的线程安全（KernelPool 每个内核独占一个连接）。
        """
        with self._lock:
            data = expr.encode("utf-8")
            try:
                # 发送：[4字节长度][表达式]
                self.sock.sendall(struct.pack(">I", len(data)) + data)

                # 接收响应头
                raw_len = self._recv_exact(4)
                resp_len = struct.unpack(">I", raw_len)[0]

                # 0xFFFFFFFF 表示服务端错误
                if resp_len == 0xFFFFFFFF:
                    # 再读 4 字节错误消息长度，然后读错误消息
                    err_len_raw = self._recv_exact(4)
                    err_len = struct.unpack(">I", err_len_raw)[0]
                    err_msg = self._recv_exact(err_len).decode("utf-8")
                    raise RuntimeError(f"远程内核错误: {err_msg}")

                return self._recv_exact(resp_len)

            except (ConnectionError, OSError, struct.error) as e:
                log.warning(f"连接中断，尝试重连: {e}")
                self._reconnect()
                # 重连后重发一次
                data2 = expr.encode("utf-8")
                self.sock.sendall(struct.pack(">I", len(data2)) + data2)
                raw_len = self._recv_exact(4)
                resp_len = struct.unpack(">I", raw_len)[0]
                if resp_len == 0xFFFFFFFF:
                    err_len = struct.unpack(">I", self._recv_exact(4))[0]
                    err_msg = self._recv_exact(err_len).decode("utf-8")
                    raise RuntimeError(f"远程内核错误: {err_msg}")
                return self._recv_exact(resp_len)

    # ── 公开接口 ─────────────────────────────────────────────

    def evaluate(self, expr: str) -> Any:
        """执行 WL 表达式，返回反序列化后的 Python 对象"""
        wxf = self._send_request(expr)
        from wolframclient.deserializers import WXFConsumer, binary_deserialize
        return binary_deserialize(wxf, consumer=WXFConsumer())

    def evaluate_to_file(self, expr: str, fmt: str = "png") -> str:
        """
        在远端执行绘图表达式，将结果文件内容传回并写入本地临时文件。

        协议扩展：在表达式前加 FILE: 前缀，服务端识别后返回文件原始字节。
        """
        import os, tempfile
        marker = f"FILE:{fmt}:"
        wxf_or_bytes = self._send_request(marker + expr)

        suffix = f".{fmt}"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="py2wl_remote_")
        try:
            os.write(fd, wxf_or_bytes)
        finally:
            os.close(fd)
        return path

    def ping(self) -> None:
        """发送心跳（长度=0），验证连接是否存活"""
        with self._lock:
            try:
                self.sock.sendall(struct.pack(">I", 0))
                raw = self._recv_exact(4)
                if struct.unpack(">I", raw)[0] != 0:
                    raise ConnectionError("ping 响应异常")
            except Exception as e:
                log.debug(f"ping {self.address} 失败: {e}")
                raise

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        log.debug(f"远程内核连接已关闭: {self.address}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
