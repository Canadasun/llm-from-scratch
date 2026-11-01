"""Fail-closed writer exclusion using atomic directory creation.

There is deliberately no expiry/stealing: a paused writer cannot be fenced by a
clock. After a crash an operator must stop all holders before removing the lock.
This supplements SQLite's normal locking; it does not make nolock/network SQLite
safe. A legacy file lock is treated as occupied and must be recovered offline.
"""
import os
import socket
import threading
import time
import uuid


class WriterConflict(RuntimeError):
    """Another writer owns the lock, or ownership cannot be established."""


class WriterLease:
    def __init__(self, path, *, ttl=60.0, clock=time.time, holder=None):
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self.path = path
        self.ttl = float(ttl)  # compatibility only; locks never expire
        self._clock = clock
        self.holder = holder or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        self._token = uuid.uuid4().hex
        self._held = False
        self._pid = os.getpid()
        self._mutex = threading.RLock()

    def _owns(self):
        if not self._held or os.getpid() != self._pid:
            return False
        try:
            with open(os.path.join(self.path, "owner"), encoding="ascii") as f:
                return f.read() == self._token
        except OSError:
            return False

    def ensure(self):
        with self._mutex:
            if self._held:
                if self._owns():
                    return
                self._held = False
                raise WriterConflict("writer ownership lost; stop writes and recover offline")
            try:
                os.mkdir(self.path, 0o700)  # create-if-absent: never overwrite a holder
            except FileExistsError as exc:
                raise WriterConflict("writer lock occupied; no automatic expiry or stealing") from exc
            # If publishing ownership fails, leave the directory occupied. Safety
            # takes precedence over availability until offline recovery.
            with open(os.path.join(self.path, "owner"), "x", encoding="ascii") as f:
                f.write(self._token)
                f.flush()
                os.fsync(f.fileno())
            self._pid = os.getpid()
            self._held = True

    def held(self):
        with self._mutex:
            return self._owns()

    def release(self):
        with self._mutex:
            if self._owns():
                os.unlink(os.path.join(self.path, "owner"))
                os.rmdir(self.path)
            self._held = False


def lease_for_key_db(key_db, *, ttl=60.0, clock=time.time):
    if not isinstance(key_db, str) or not key_db.startswith("file:"):
        return None
    from urllib.parse import unquote
    fs_path = unquote(key_db[len("file:"):].split("?", 1)[0])
    if not fs_path or fs_path == ":memory:":
        return None
    return WriterLease(fs_path + ".writer.lock", ttl=ttl, clock=clock)
