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
