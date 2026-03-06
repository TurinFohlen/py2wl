"""
python -m py2wl.server 入口
"""
import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def main():
    parser = argparse.ArgumentParser(
        prog="python -m py2wl.server",
        description="py2wl 远程内核服务端"
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PY2WL_SERVER_PORT", "9999")),
                        help="监听端口（默认 9999，或 PY2WL_SERVER_PORT 环境变量）")
    args = parser.parse_args()

    from py2wl.server import KernelServer
    server = KernelServer(host=args.host, port=args.port)
    try:
        server.start()   # 阻塞
    except KeyboardInterrupt:
        pass
    finally:
        server.close()

if __name__ == "__main__":
    main()
