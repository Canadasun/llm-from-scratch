"""
A SQLite-backed store with the SAME tenant-isolation invariants as the in-memory
`Store`, but PERSISTENT across restarts. It is a drop-in replacement: the gateway,
key manager and secured service use the identical method surface, so nothing else
changes.

Why this exists: the in-memory store loses every API key when the process
restarts, so `--secured` keys were per-replica and ephemeral. Backing the store
with SQLite makes keys survive restarts.

The isolation boundary is enforced in SQL: every tenant-scoped lookup carries a
`WHERE ... AND tenant_id = ?` clause, the row-level-security analog. There is
deliberately no query that returns a resource/key by id while ignoring the tenant
— that absence is the guarantee, exactly as in the in-memory reference.

Durability scope (honest): a SQLite file persists across process restarts and is
correct for a single-node / single-replica deployment. To ALSO survive replica
recycles on a cloud host the file must live on shared storage (an Azure Files
mount). SQLite over SMB/CIFS, though, cannot take its usual byte-range locks, so
the path must be opened as a `file:...?nolock=1` URI (see __init__) and the app
pinned to ONE writer (max-replicas=1); that is the single-writer, network-durable
configuration used live. For TRUE concurrent multi-replica writes this class
should instead be re-pointed at a networked database (Postgres with real RLS
policies) — same interface and invariants, only the connection changes.
"""

import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from enterprise.keys import hash_key, verify_key
from enterprise.models import ApiKey, Principal, Resource, Tenant

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS principals (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, roles TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash TEXT PRIMARY KEY, id TEXT NOT NULL, tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL, scopes TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL DEFAULT 0, expires_at REAL NOT NULL DEFAULT 0,
    revoked INTEGER NOT NULL DEFAULT 0, replaced_by TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS resources (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'document',
    owner_id TEXT NOT NULL DEFAULT '', attributes TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_keys_id ON api_keys(id, tenant_id);
CREATE INDEX IF NOT EXISTS idx_res_tenant ON resources(tenant_id);
"""


class SqlStore:
    def __init__(self, path=":memory:", clock=time.time, writer_lease=None):
        # A `file:...` path is opened as a SQLite URI, which lets the caller pass
        # query parameters such as `?nolock=1`. That matters on a network share
        # (Azure Files / SMB): CIFS does not provide the byte-range locks SQLite
        # normally takes, so without nolock even the first CREATE TABLE fails with
        # "database is locked". Disabling locking is safe HERE because the store
        # is used single-writer (max-replicas=1) and already serialises every
        # access through self._lock below. A plain path or ":memory:" is opened
        # exactly as before (locking on), so local/default behaviour is unchanged.
        uri = path.startswith("file:")
        # check_same_thread=False + a lock: the HTTP server is threaded, and a
        # single serialized connection is simplest and correct for this load.
        self._db = sqlite3.connect(path, uri=uri, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._clock = clock
        # Optional enforced single-writer lease (see enterprise/writer_lease.py).
        # None -> today's behaviour (local/in-memory stores). When set, every
        # write (including the schema init below) must hold the lease, so two
        # overlapping revisions sharing a nolock DB cannot both write.
        self._writer_lease = writer_lease
        with self._lock:
            self._require_writer()
            self._db.executescript(_SCHEMA)
            self._commit()

    def _require_writer(self):
        """Guard every write path: acquire/renew the single-writer lease, or raise
        WriterConflict if another live writer holds it. No-op when no lease is
        configured. Reads never call this."""
        if self._writer_lease is not None:
            self._writer_lease.ensure()

    def close(self):
        if self._writer_lease is not None:
            self._writer_lease.release()
        self._db.close()

    def _commit(self):
        if not self._transaction_depth:
            self._db.commit()

    @contextmanager
    def key_transaction(self):
        """Atomic lifecycle operation, including across connections to a local DB.

        SQLite locking must be enabled for cross-process exclusion. A nolock
        network-share deployment still requires an externally enforced writer.
        """
        with self._lock:
            outer = self._transaction_depth == 0
            if outer:
                self._require_writer()
                self._db.execute("BEGIN IMMEDIATE")
            self._transaction_depth += 1
            try:
                yield
                if outer:
                    self._db.commit()
            except BaseException:
                if outer:
                    self._db.rollback()
                raise
            finally:
                self._transaction_depth -= 1

    # --- row <-> dataclass ---
    @staticmethod
    def _principal(row):
        return Principal(row["id"], row["tenant_id"], tuple(json.loads(row["roles"])))

    @staticmethod
    def _apikey(row):
        return ApiKey(row["id"], row["key_hash"], row["tenant_id"], row["principal_id"],
                      tuple(json.loads(row["scopes"])), created_at=row["created_at"],
                      expires_at=row["expires_at"], revoked=bool(row["revoked"]),
                      replaced_by=row["replaced_by"])

    @staticmethod
    def _resource(row):
        return Resource(row["id"], row["tenant_id"], row["kind"], row["owner_id"],
                        json.loads(row["attributes"]))

    # --- registration ---
    def add_tenant(self, tenant):
        with self._lock:
            self._require_writer()
            self._db.execute("INSERT OR REPLACE INTO tenants(id,name) VALUES(?,?)",
                             (tenant.id, tenant.name))
            self._commit()
        return tenant

    def add_principal(self, principal):
        with self._lock:
            self._require_writer()
            if not self._db.execute("SELECT 1 FROM tenants WHERE id=?",
                                    (principal.tenant_id,)).fetchone():
                raise ValueError(f"unknown tenant {principal.tenant_id}")
            self._db.execute(
                "INSERT OR REPLACE INTO principals(id,tenant_id,roles) VALUES(?,?,?)",
                (principal.id, principal.tenant_id, json.dumps(list(principal.roles))))
            self._commit()
        return principal

    def add_api_key(self, api_key):
        with self._lock:
            self._require_writer()
            if not self._db.execute("SELECT 1 FROM principals WHERE id=?",
                                    (api_key.principal_id,)).fetchone():
                raise ValueError(f"unknown principal {api_key.principal_id}")
            self._db.execute(
                "INSERT OR REPLACE INTO api_keys(key_hash,id,tenant_id,principal_id,"
                "scopes,created_at,expires_at,revoked,replaced_by) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (api_key.key_hash, api_key.id, api_key.tenant_id, api_key.principal_id,
                 json.dumps(list(api_key.scopes)), api_key.created_at,
                 api_key.expires_at, int(api_key.revoked), api_key.replaced_by))
            self._commit()
        return api_key

    def add_resource(self, resource):
        with self._lock:
            self._require_writer()
            if not self._db.execute("SELECT 1 FROM tenants WHERE id=?",
                                    (resource.tenant_id,)).fetchone():
                raise ValueError(f"unknown tenant {resource.tenant_id}")
            self._db.execute(
                "INSERT OR REPLACE INTO resources(id,tenant_id,kind,owner_id,attributes)"
                " VALUES(?,?,?,?,?)",
                (resource.id, resource.tenant_id, resource.kind, resource.owner_id,
                 json.dumps(resource.attributes)))
            self._commit()
        return resource

    def get_principal(self, principal_id):
        with self._lock:
            row = self._db.execute("SELECT * FROM principals WHERE id=?",
                                   (principal_id,)).fetchone()
        return self._principal(row) if row else None

    # --- authentication ---
    def authenticate(self, presented_key):
        with self._lock:
            row = self._db.execute("SELECT * FROM api_keys WHERE key_hash=?",
                                   (hash_key(presented_key),)).fetchone()
        if row is None:
            return None, None
        key = self._apikey(row)
        if not verify_key(presented_key, key.key_hash):   # defense in depth
            return None, None
        if not key.is_active(self._clock()):              # revoked or expired
            return None, None
        principal = self.get_principal(key.principal_id)
        # Tenant consistency: a key must only authenticate as a principal in the
        # key's OWN tenant. If the principal id was re-created in another tenant,
        # the stale key must not cross into it.
        if principal is None or principal.tenant_id != key.tenant_id:
            return None, None
        return principal, key

    # --- tenant-scoped API-key management (isolation enforced in SQL) ---
    def get_api_key(self, principal, key_id):
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM api_keys WHERE id=? AND tenant_id=?",
                (key_id, principal.tenant_id)).fetchone()
        return self._apikey(row) if row else None

    def list_api_keys(self, principal, principal_id=None):
        # Literal queries per branch (never build SQL text from a variable), so
        # tenant_id/principal_id only ever reach the DB as bound parameters.
        with self._lock:
            if principal_id is None:
                rows = self._db.execute(
                    "SELECT * FROM api_keys WHERE tenant_id=?",
                    (principal.tenant_id,)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM api_keys WHERE tenant_id=? AND principal_id=?",
                    (principal.tenant_id, principal_id)).fetchall()
        return [self._apikey(r) for r in rows]

    def update_api_key(self, api_key):
        with self._lock:
            self._require_writer()
            cur = self._db.execute(
                "UPDATE api_keys SET scopes=?,created_at=?,expires_at=?,revoked=?,"
                "replaced_by=? WHERE key_hash=?",
                (json.dumps(list(api_key.scopes)), api_key.created_at,
                 api_key.expires_at, int(api_key.revoked), api_key.replaced_by,
                 api_key.key_hash))
            if cur.rowcount == 0:
                raise KeyError("unknown api key")
            self._commit()
        return api_key

    # --- tenant-scoped resource access (THE isolation boundary, in SQL) ---
    def get_resource(self, principal, resource_id):
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM resources WHERE id=? AND tenant_id=?",
                (resource_id, principal.tenant_id)).fetchone()
        return self._resource(row) if row else None

    def list_resources(self, principal, kind=None):
        with self._lock:
            if kind is None:
                rows = self._db.execute(
                    "SELECT * FROM resources WHERE tenant_id=?",
                    (principal.tenant_id,)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM resources WHERE tenant_id=? AND kind=?",
                    (principal.tenant_id, kind)).fetchall()
        return [self._resource(r) for r in rows]
