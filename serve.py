"""CPU HTTP inference: submit, poll incremental output, cancel, inspect metrics.

python serve.py --checkpoint checkpoints/tiny-trained.npz
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from nano_infer.engine import Engine
from nano_infer.paging import BlockPool
from nano_infer.service import BusyError, Service
from nano_infer.training import load_checkpoint


def start_server(service, host="127.0.0.1", port=8000):
    lock, wake, stop = threading.RLock(), threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/requests":
                return self.reply(404, dict(error="unknown endpoint"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("JSON body must contain 1..65536 bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                with lock:
                    key = service.submit(**body)
                wake.set()
                self.reply(202, dict(id=key))
            except BusyError as exc:
                self.reply(429, dict(error=str(exc)))
            except (ValueError, TypeError) as exc:
                self.reply(400, dict(error=str(exc)))

        def do_GET(self):
            with lock:
                if self.path == "/health":
                    return self.reply(200, dict(status="ok", pool=service.engine.pool.snapshot()))
                if not self.path.startswith("/requests/"):
                    return self.reply(404, dict(error="unknown endpoint"))
                try:
                    payload = service.snapshot(self.path.split("/")[-1])
                except KeyError:
                    return self.reply(404, dict(error="unknown or expired request"))
            self.reply(200, payload)

        def do_DELETE(self):
            if not self.path.startswith("/requests/"):
                return self.reply(404, dict(error="unknown endpoint"))
            try:
                with lock:
                    payload = service.cancel(self.path.split("/")[-1])
                self.reply(200, payload)
            except KeyError:
                self.reply(404, dict(error="unknown or expired request"))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)

    def work():
        while not stop.is_set():
            with lock:
                service.step()
                active = any(e.status in ("queued", "running") for e in service.entries.values())
            if not active:
                wake.wait(.05)
                wake.clear()
            else:
                stop.wait(.001)  # allow HTTP threads to acquire the owner lock

    worker = threading.Thread(target=work, daemon=True)
    http = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    http.start()

    def close():
        stop.set()
        wake.set()
        worker.join()
        server.shutdown()
        server.server_close()
        http.join()
        with lock:
            for key in list(service.entries):
                service.cancel(key)

    return server, close


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tiny-trained.npz")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-pending", type=int, default=16)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint)
    pool = BlockPool(args.blocks, args.block_size, model.width)
    engine = Engine(model, token_budget=16, max_batch=4, max_draft=4, batched=True, pool=pool)
    server, close = start_server(Service(engine, args.max_pending, max(64, args.max_pending)), port=args.port)
    print(f"Serving http://127.0.0.1:{server.server_port}; Ctrl+C to stop", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        close()


if __name__ == "__main__":
    main()
