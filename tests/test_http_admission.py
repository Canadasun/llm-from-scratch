"""Real-socket regressions for connection bounds and HTTP framing."""

import socket
import threading
from contextlib import contextmanager

import pytest

from serve.http_server import BoundedHTTPServer, make_handler


@contextmanager
def live_server():
    started, released = threading.Event(), threading.Event()
    class Service:
        health_calls = 0
        generate_calls = 0
        def health(self):
            self.health_calls += 1
            return {"status": "ok"}
        def handle_generate(self, payload, client="-"):
            self.generate_calls += 1
            return 200, {"completion": "ok"}
    service = Service()
    base = make_handler(service)
    class Handler(base):
        def setup(self):
            super().setup()
            started.set()
    class Server(BoundedHTTPServer):
        def process_request_thread(self, *args):
            try:
                super().process_request_thread(*args)
            finally:
                released.set()
    server = Server(("127.0.0.1", 0), Handler, max_connections=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address, service, started, released
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def _request(address, request):
    with socket.create_connection(address, timeout=2) as client:
        client.sendall(request)
        data = bytearray()
        while True:
            part = client.recv(4096)
            if not part:
                return bytes(data)
            data.extend(part)


def test_idle_connection_consumes_slot_and_overflow_gets_503_then_recovers():
    with live_server() as (address, service, started, released):
        idle = socket.create_connection(address, timeout=2)
        try:
            assert started.wait(2)
            response = _request(address, b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
            assert b"503 Service Unavailable" in response
            assert b"Retry-After: 1" in response
            assert service.health_calls == 0
        finally:
            idle.close()
        assert released.wait(2)
        response = _request(address, b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        assert b"200 OK" in response
        assert service.health_calls == 1


@pytest.mark.parametrize("headers", [
    b"Content-Length: 2\r\nContent-Length: 2\r\n",
    b"Transfer-Encoding: chunked\r\nContent-Length: 2\r\n",
    b"Content-Length: invalid\r\n",
    b"Content-Length: +2\r\n",
    b"Content-Length: 2,2\r\n",
])
def test_bad_framing_closes_connection_without_processing_pipelined_request(headers):
    with live_server() as (address, service, _, released):
        response = _request(address, b"POST /generate HTTP/1.1\r\nHost: localhost\r\n" + headers
                            + b"\r\n{}GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
        assert b"400 Bad Request" in response
        assert b"Connection: close" in response
        assert response.count(b"HTTP/1.1") == 1
        assert released.wait(2)
        assert service.generate_calls == service.health_calls == 0


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_connection_limits_fail_before_binding(limit):
    with pytest.raises(ValueError):
        BoundedHTTPServer(("127.0.0.1", 0), None, max_connections=limit)


@contextmanager
def live_server_fast_read_deadline(seconds):
    """A server whose absolute request-READ deadline is `seconds` (so a slow/
    trickle client is torn down quickly for the test)."""
    class Service:
        def health(self):
            return {"status": "ok"}
        def handle_generate(self, payload, client="-"):
            return 200, {"completion": "ok"}
    base = make_handler(Service())
    class Handler(base):
        max_request_read_seconds = seconds
    server = BoundedHTTPServer(("127.0.0.1", 0), Handler, max_connections=4)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_absolute_read_deadline_tears_down_a_trickling_client():
    import time
    with live_server_fast_read_deadline(1) as address:
        # Open a connection and send an INCOMPLETE request slowly: headers never
        # finish, but a byte dribbles out just often enough to beat a per-read
        # inactivity timeout. The absolute read deadline must still close it.
        client = socket.create_connection(address, timeout=10)
        client.sendall(b"POST /generate HTTP/1.1\r\nHost: x\r\n")
        start = time.monotonic()
        client.settimeout(5)
        closed_within_deadline = False
        try:
            # A trickle that never completes the request.
            for _ in range(8):
                client.sendall(b"X-Pad: y\r\n")
                data = client.recv(4096)
                if data == b"":       # server shut the socket down
                    closed_within_deadline = True
                    break
                time.sleep(0.5)
        except OSError:
            closed_within_deadline = True   # reset/broken pipe also means torn down
        finally:
            elapsed = time.monotonic() - start
            client.close()
        assert closed_within_deadline, "trickling client was not torn down"
        assert elapsed < 4, f"took too long to enforce the read deadline ({elapsed:.1f}s)"


def test_normal_request_is_not_cut_off_by_the_read_deadline():
    # A complete request well within the deadline still succeeds.
    with live_server_fast_read_deadline(5) as address:
        body = b'{"prompt":"hi","max_new_tokens":4}'
        req = (b"POST /generate HTTP/1.1\r\nHost: x\r\nContent-Length: "
               + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
        resp = _request(address, req)
        assert b"200" in resp.split(b"\r\n", 1)[0] and b'"completion"' in resp
