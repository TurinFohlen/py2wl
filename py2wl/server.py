"""
py2wl.server — 远程 Wolfram 内核服务端
========================================

在局域网内的设备上运行此服务，让其他设备通过 RemoteKernel 连接使用其算力。

启动方式：
    python -m py2wl.server
    python -m py2wl.server --port 9999
    python -m py2wl.server --host 0.0.0.0 --port 9999

协议（length-prefixed TCP）：
    请求：[4字节大端长度][UTF-8 表达式]
    响应：[4字节大端长度][WXF 二进制结果]
    错误：[0xFFFFFFFF][4字节错误消息长度][UTF-8 错误消息]
    Ping：请求长度 = 0 → 响应长度 = 0
    文件：表达式以 "FILE:fmt:" 开头 → 响应为原始文件字节
"""

import logging
import os
import socket
import struct
import threading
import time

log = logging.getLogger("py2wl.server")

# 默认端口（可通过 PY2WL_SERVER_PORT 覆盖）
DEFAULT_PORT = int(os.environ.get("PY2WL_SERVER_PORT", "9999"))


class KernelServer:
    """
    TCP 服务端，包装本地 WolframKernel。

    每个客户端连接独占一个线程，请求串行转发给唯一的本地内核执行
    （WolframKernel 内部已有队列，线程安全）。
    """

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        self.host    = host
        self.port    = port
        self.running = False
        self.server: socket.socket = None
        self.kernel  = None   # 懒加载，start() 时初始化
        self._clients: list = []
        self._clients_lock = threading.Lock()

    def start(self) -> None:
        """启动内核和 TCP 监听，阻塞直到 close() 被调用"""
        # 初始化本地内核
        log.info("正在启动 Wolfram 内核…")
        from .kernel import WolframKernel
        self.kernel = WolframKernel()
        log.info("Wolfram 内核已就绪")

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(16)
        self.running = True

        log.info(f"py2wl server 监听 {self.host}:{self.port}")
        print(f"[py2wl server] 监听 {self.host}:{self.port}，按 Ctrl+C 停止")

        self.server.settimeout(1.0)   # 允许定期检查 self.running
        while self.running:
            try:
                client, addr = self.server.accept()
                log.info(f"新连接: {addr[0]}:{addr[1]}")
                t = threading.Thread(
                    target=self._handle_client,
                    args=(client, addr),
                    daemon=True,
                    name=f"client-{addr[0]}:{addr[1]}"
                )
                t.start()
                with self._clients_lock:
                    self._clients.append(t)
            except socket.timeout:
                continue
            except OSError:
                break   # server socket 被 close() 关闭

    def _handle_client(self, sock: socket.socket, addr) -> None:
        """处理单个客户端连接的所有请求"""
        sock.settimeout(300)   # 单次请求最长等待 5 分钟
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client_id = f"{addr[0]}:{addr[1]}"

        try:
            while self.running:
                # ── 读取请求头（4字节长度）────────────────────────
                raw_len = self._recv_exact(sock, 4)
                if raw_len is None:
                    log.debug(f"客户端 {client_id} 断开连接")
                    break
                msg_len = struct.unpack(">I", raw_len)[0]

                # ── Ping ─────────────────────────────────────────
                if msg_len == 0:
                    sock.sendall(struct.pack(">I", 0))
                    continue

                # ── 读取表达式 ────────────────────────────────────
                raw_expr = self._recv_exact(sock, msg_len)
                if raw_expr is None:
                    break
                expr = raw_expr.decode("utf-8")

                # ── 判断是否是文件请求 ────────────────────────────
                # 文件请求格式："FILE:fmt:实际表达式"
                if expr.startswith("FILE:"):
                    self._handle_file_request(sock, expr, client_id)
                else:
                    self._handle_eval_request(sock, expr, client_id)

        except Exception as e:
            log.error(f"客户端 {client_id} 处理异常: {e}")
        finally:
            sock.close()
            log.debug(f"客户端 {client_id} 连接已关闭")

    def _handle_eval_request(self, sock, expr: str, client_id: str) -> None:
        """执行表达式请求，返回 WXF 序列化结果"""
        log.debug(f"[{client_id}] eval: {expr[:60]}{'…' if len(expr)>60 else ''}")
        try:
            result = self.kernel.evaluate(expr)

            from wolframclient.serializers import export as wl_export
            wxf_data = wl_export(result, target_format="wxf")

            sock.sendall(struct.pack(">I", len(wxf_data)) + wxf_data)
            log.debug(f"[{client_id}] 响应 {len(wxf_data)} 字节")

        except Exception as e:
            self._send_error(sock, str(e))
            log.warning(f"[{client_id}] 执行错误: {e}")

    def _handle_file_request(self, sock, raw_expr: str, client_id: str) -> None:
        """
        执行绘图请求，返回图像文件原始字节。
        格式：FILE:png:Plot[Sin[x],{x,0,2Pi}]
        """
        try:
            _, fmt, expr = raw_expr.split(":", 2)
        except ValueError:
            self._send_error(sock, "FILE 请求格式错误，应为 FILE:fmt:expr")
            return

        log.debug(f"[{client_id}] file({fmt}): {expr[:60]}…")
        try:
            path = self.kernel.evaluate_to_file(expr, fmt=fmt)
            with open(path, "rb") as fh:
                file_bytes = fh.read()
            sock.sendall(struct.pack(">I", len(file_bytes)) + file_bytes)
            log.debug(f"[{client_id}] 图像响应 {len(file_bytes)} 字节")
        except Exception as e:
            self._send_error(sock, str(e))
            log.warning(f"[{client_id}] 文件请求错误: {e}")

    def _send_error(self, sock, msg: str) -> None:
        """发送错误响应：0xFFFFFFFF + 4字节长度 + 错误消息"""
        try:
            err_bytes = msg.encode("utf-8")
            sock.sendall(
                struct.pack(">I", 0xFFFFFFFF) +
                struct.pack(">I", len(err_bytes)) +
                err_bytes
            )
        except Exception:
            pass   # 发送错误响应时再出错就放弃

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int):
        """接收恰好 n 字节，连接断开返回 None"""
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except (socket.timeout, OSError):
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def close(self) -> None:
        self.running = False
        if self.server:
            try:
                self.server.close()
            except Exception:
                pass
        if self.kernel:
            try:
                self.kernel.close()
            except Exception:
                pass
        log.info("py2wl server 已停止")
        print("[py2wl server] 已停止")
