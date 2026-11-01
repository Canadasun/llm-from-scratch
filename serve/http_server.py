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
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY_BYTES = 64 * 1024  # cap request bodies; prompt limit is enforced in validation
# Absolute wall-clock cap on READING a whole request (headers + body). The
# per-read socket `timeout` only bounds INACTIVITY, so a slow client that dribbles
# one byte just under that timeout could hold a worker indefinitely. This deadline
# tears the connection down regardless of such trickling, then is cancelled before
# generation so a legitimate request's model work is not time-limited by it.
MAX_REQUEST_READ_SECONDS = 30


class BoundedHTTPServer(ThreadingHTTPServer):
    """Bound accepted connection workers, including idle keep-alive clients.

    The kernel backlog is bounded separately. No worker thread is allocated for
    an excess connection; it receives a best-effort 503 and is closed. This is
    single-process admission control, not a distributed quota or production proxy.
    """
    request_queue_size = 32

    def __init__(self, address, handler, max_connections=32):
        if isinstance(max_connections, bool) or not isinstance(max_connections, int) or max_connections < 1:
            raise ValueError("max_connections must be a positive integer")
        self._connection_slots = threading.BoundedSemaphore(max_connections)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self._connection_slots.acquire(blocking=False):
            body = b'{"error":"overloaded"}'
            response = (b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n"
                        b"Content-Type: application/json\r\nRetry-After: 1\r\nContent-Length: "
                        + str(len(body)).encode() + b"\r\n\r\n" + body)
            try:
                request.settimeout(.1)
                request.sendall(response)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "llm-from-scratch/serve"
        # Per-connection socket timeout: without it a slow or stalled client can
        # hold a worker thread forever (a slowloris-style resource exhaustion),
        # and the body is read before authentication, so an unauthenticated peer
        # could tie up threads. A stalled read now raises and frees the thread.
        # (Defense in depth; in production a managed proxy also caps this.)
        timeout = 30
        max_request_read_seconds = MAX_REQUEST_READ_SECONDS

        def handle_one_request(self):
            # Start an ABSOLUTE read deadline for the request. do_POST cancels it
            # once the body is fully read, so generation time is not bounded by it;
            # for a GET or a stalled read it is cancelled in the finally below.
            self._read_watchdog = threading.Timer(
                self.max_request_read_seconds, self._abort_slow_read)
            self._read_watchdog.daemon = True
            self._read_watchdog.start()
            try:
                super().handle_one_request()
            finally:
                self._cancel_read_watchdog()

        def _cancel_read_watchdog(self):
            watchdog = getattr(self, "_read_watchdog", None)
            if watchdog is not None:
                watchdog.cancel()
                self._read_watchdog = None

        def _abort_slow_read(self):
            # Fired when the absolute read deadline passes: shut the socket down so
            # any in-progress read raises and the worker thread is freed. Defense
            # in depth with `timeout`; in production a managed proxy also caps this.
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        def _send(self, status, obj):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if status >= 400:
                # Unread/malformed bodies cannot become another keep-alive request.
                self.close_connection = True
                self.send_header("Connection", "close")
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
            route = self.path.split("?")[0]
            if route == "/healthz":
                self._send(200, service.health())
            elif route == "/metrics" and hasattr(service, "metrics_snapshot"):
                # Aggregate-only operational metrics (counts/rates/timings/cost).
                # Open like /healthz: it leaks no tenant data, prompts or outputs.
                self._send(200, service.metrics_snapshot())
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self):
            if self.path.split("?")[0] != "/generate":
                self._send(404, {"error": "not_found"})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1:
                self._send(400, {"error": "invalid_request", "detail": "one Content-Length and no Transfer-Encoding required"})
                return
            try:
                length_text = lengths[0].strip()
                if not length_text.isascii() or not length_text.isdecimal():
                    raise ValueError("Content-Length must be ASCII digits")
                length = int(length_text)
            except ValueError:
                self._send(400, {"error": "invalid_request", "detail": "bad Content-Length"})
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(413, {"error": "payload_too_large"})
                return
            raw = self.rfile.read(length)
            # Body fully read: stop the absolute read deadline so a legitimate
            # request's generation time is not cut off by it.
            self._cancel_read_watchdog()
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


def serve(service, host="127.0.0.1", port=8000, max_connections=32):
    httpd = BoundedHTTPServer((host, port), make_handler(service), max_connections=max_connections)
    logging.getLogger("serve").info(
        json.dumps({"event": "listening", "host": host, "port": port}))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
