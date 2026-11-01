"""Application-enforced single-writer lease for the shared key store.

The key store is SQLite opened with `nolock=1` on an Azure Files (SMB) share,
which disables the byte-range locks SQLite would normally use for cross-process
exclusion. Prior deployments relied on a written *instruction* ("do not change
keys during overlapping revisions") that nothing enforced (audit finding:
conditional persistence risk). This lease ENFORCES it: a writer must hold an
exclusive, time-limited lease file before it may initialise or write the store.

This is defense-in-depth for the current design, NOT a substitute for a real
transactional database. A managed Postgres with DB-enforced exclusion remains the
supported end state; this makes the interim safe instead of merely documented.

Mechanics (SMB-friendly, no byte-range locks required):
  - The lease is a small JSON file: {"holder": "<host:pid:uuid>", "ts": <epoch>}.
  - To claim, a writer atomically renames a unique temp file over the lease path
    (os.replace is atomic on POSIX and on SMB), then RE-READS it: only the writer
    whose holder id survived actually won, so a simultaneous race resolves to one
    winner and every loser raises WriterConflict.
  - An active writer refreshes `ts` on every write (ensure()). A lease older than
    `ttl` is considered abandoned (e.g. a deactivated revision that never
    released) and may be stolen. Keep ttl well above the write cadence.
"""

import json
import os
import socket
import time
import uuid


class WriterConflict(RuntimeError):
    """Raised when another live writer holds the lease."""


class WriterLease:
    def __init__(self, path, *, ttl=60.0, clock=time.time, holder=None):
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self.path = path
        self.ttl = float(ttl)
        self._clock = clock
        self.holder = holder or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._held = False

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _write_atomic(self, data):
        tmp = f"{self.path}.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)       # atomic publish (POSIX and SMB)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _claim(self, now):
        self._write_atomic({"holder": self.holder, "ts": now})
        # Resolve a simultaneous race: last atomic replace wins; everyone re-reads
        # and only the surviving holder considers itself the writer.
        current = self._read()
        if not current or current.get("holder") != self.holder:
            self._held = False
            raise WriterConflict("lost a lease race to another writer")
        self._held = True

    def ensure(self):
        """Acquire the lease, renew it if already ours, or steal it if the current
        holder's lease has expired. Raise WriterConflict if a live writer holds
        it. Call before every store write; it also serves as the heartbeat."""
        now = self._clock()
        current = self._read()
        if current is None:
            self._claim(now)                 # free
        elif current.get("holder") == self.holder:
            self._claim(now)                 # renew (heartbeat)
        elif now - float(current.get("ts", 0)) > self.ttl:
            self._claim(now)                 # abandoned -> steal
        else:
            self._held = False
            raise WriterConflict(f"another writer holds the lease: {current.get('holder')!r}")

    def held(self):
        return self._held

    def release(self):
        """Give up the lease if we still hold it (best effort)."""
        if self._held and (self._read() or {}).get("holder") == self.holder:
            try:
                os.unlink(self.path)
            except OSError:
                pass
        self._held = False


def lease_for_key_db(key_db, *, ttl=60.0, clock=time.time):
    """Build a WriterLease beside a persistent key DB, or None for non-persistent
    stores (":memory:" / plain local paths keep today's lease-free behaviour).
    Only a `file:` URI store (the Azure Files / shared-share deployment) gets a
    lease, since that is where nolock removes SQLite's own exclusion."""
    if not isinstance(key_db, str) or not key_db.startswith("file:"):
        return None
    fs_path = key_db[len("file:"):].split("?", 1)[0]
    if not fs_path:
        return None
    return WriterLease(fs_path + ".writer.lock", ttl=ttl, clock=clock)
