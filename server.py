"""Local web UI for Queen: an interactive board with live, visualized analysis.

  .venv/bin/python server.py          # then open http://127.0.0.1:8765

Loads the model once, serves ui/ and streams analyses as NDJSON:
  POST /api/analyze {fen, history?, temperature?, max_tokens?}
       -> {"type":"start"} {"type":"delta","text":...}* {"type":"done","result":{...}} | {"type":"error"}
  POST /api/stop      stop the running generation
  GET  /api/info      device / quantization / prompt
Only one analysis runs at a time (the GPU is shared); others wait their turn.
"""
import argparse
import json
import mimetypes
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from queen_engine import QueenEngine

UI_DIR = Path(__file__).resolve().parent / "ui"
ENGINE: QueenEngine = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # stream until close; no chunked encoding needed

    def log_message(self, fmt, *args):
        if "/api/" in self.path:
            sys.stderr.write("[http] " + fmt % args + "\n")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/info":
            return self._json(ENGINE.info())
        rel = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        path = (UI_DIR / rel).resolve()
        if not path.is_file() or UI_DIR not in path.parents:
            return self.send_error(404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/api/stop":
            ENGINE.stop()
            return self._json({"ok": True})
        if self.path != "/api/analyze":
            return self.send_error(404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad JSON"}, 400)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(obj):
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()

        def on_text(delta):
            try:
                send({"type": "delta", "text": delta})
            except (BrokenPipeError, ConnectionResetError):
                ENGINE.stop()            # browser went away: stop generating

        try:
            send({"type": "start"})
            result = ENGINE.analyze(
                req["fen"], req.get("history") or [],
                temperature=float(req.get("temperature", 0.6)),
                max_tokens=int(req.get("max_tokens", 2048)),
                game_id=int(req.get("game_id", 0)), on_text=on_text)
            send({"type": "done", "result": result})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                send({"type": "error", "message": str(e)})
            except OSError:
                pass


def main():
    global ENGINE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", choices=["cuda", "cpu"])
    p.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    args = p.parse_args()
    # Claim the port before loading the model, so a second copy fails fast and clearly.
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError:
        sys.exit(f"[queen] Port {args.port} is in use: Queen's UI is probably already running at "
                 f"http://{args.host}:{args.port} (only one copy fits on the GPU). "
                 f"Stop it first, or pick another port with --port.")
    try:
        ENGINE = QueenEngine(device=args.device, quant=args.quant)
    except Exception as e:
        if type(e).__name__ == "OutOfMemoryError":
            sys.exit("[queen] Not enough free GPU memory (needs ~3 GB). Another program is using the GPU; "
                     "check with: nvidia-smi   (or run on CPU with --device cpu, much slower).")
        raise
    print(f"[queen] UI at http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
