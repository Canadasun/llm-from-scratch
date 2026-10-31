"""
A thin HTTP adapter over InferenceService, built on the standard library
(no extra dependency). It exposes:

    GET  /healthz   -> 200 service info
    POST /generate  -> {"completion": ...} or an error object

Production notes (honest): this is a threaded single-process server, fine for a
demo, a local dev endpoint, or behind one worker. For real deployment the same
InferenceService drops unchanged behind FastAPI/uvicorn or gunicorn — only this
file is replaced. Bind to 127.0.0.1 unless you intend to expose it.
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY_BYTES = 64 * 1024  # cap request bodies; prompt limit is enforced in validation


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "llm-from-scratch/serve"

        def _send(self, status, obj):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _client(self):
            # Note: trusts the socket peer only. Do NOT trust X-Forwarded-For
            # unless a trusted proxy sets it; we deliberately ignore it here.
            return self.client_address[0] if self.client_address else "-"

        def _bearer(self):
            # Pull an API key from `Authorization: Bearer <key>`. Returns None if
            # the header is absent or malformed; authentication itself is the
            # service's job, not the transport's.
            h = self.headers.get("Authorization", "")
            if h[:7].lower() == "bearer ":
                return h[7:].strip() or None
            return None

        def do_GET(self):
            if self.path.split("?")[0] == "/healthz":
                self._send(200, service.health())
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self):
            if self.path.split("?")[0] != "/generate":
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self._send(400, {"error": "invalid_request", "detail": "bad Content-Length"})
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(413, {"error": "payload_too_large"})
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._send(400, {"error": "invalid_request", "detail": "body is not valid JSON"})
                return
            if getattr(service, "requires_auth", False):
                # Secured service: the bearer token is the identity; a missing
                # or bad token becomes a 401 inside the service (and is audited).
                status, body = service.handle_generate(
                    payload, api_key=self._bearer(), client=self._client())
            else:
                status, body = service.handle_generate(payload, client=self._client())
            self._send(status, body)

        def log_message(self, *args):
            pass  # structured logging happens in the service, not here

    return Handler


def serve(service, host="127.0.0.1", port=8000):
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    logging.getLogger("serve").info(
        json.dumps({"event": "listening", "host": host, "port": port}))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
