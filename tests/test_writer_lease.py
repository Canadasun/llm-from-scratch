"""
Tests for the enforced single-writer lease (audit: conditional persistence risk).

The shared key store runs SQLite with nolock on an SMB share, which gives no
cross-process exclusion. The lease enforces one writer at the application layer:
a second writer is refused, an abandoned lease is reclaimed after its TTL, and
SqlStore writes are gated on holding it.
"""

import pytest

from enterprise.writer_lease import WriterLease, WriterConflict, lease_for_key_db
from enterprise.sqlstore import SqlStore
from enterprise.models import Tenant


class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, dt): self.t += dt


def test_second_writer_is_refused(tmp_path):
    path = str(tmp_path / "keys.db.writer.lock")
    a = WriterLease(path, ttl=60, holder="writer-A")
    b = WriterLease(path, ttl=60, holder="writer-B")
    a.ensure()
    assert a.held()
    with pytest.raises(WriterConflict):
        b.ensure()
    assert not b.held()


def test_holder_can_renew_repeatedly(tmp_path):
    clk = _Clock()
    a = WriterLease(str(tmp_path / "l.lock"), ttl=60, clock=clk, holder="A")
    a.ensure()
    clk.advance(30)
    a.ensure()  # renew; no conflict
    assert a.held()


def test_abandoned_lease_is_reclaimed_after_ttl(tmp_path):
    clk = _Clock()
    path = str(tmp_path / "l.lock")
    dead = WriterLease(path, ttl=60, clock=clk, holder="dead")
    dead.ensure()
    # Holder stops heart-beating (e.g. a deactivated revision). Before TTL a new
    # writer is refused; after TTL it may steal the lease.
    clk.advance(30)
    fresh = WriterLease(path, ttl=60, clock=clk, holder="fresh")
    with pytest.raises(WriterConflict):
        fresh.ensure()
    clk.advance(31)  # now 61s since the dead holder's last beat
    fresh.ensure()
    assert fresh.held()


def test_release_frees_the_lease(tmp_path):
    path = str(tmp_path / "l.lock")
    a = WriterLease(path, ttl=60, holder="A")
    a.ensure()
    a.release()
    assert not a.held()
    b = WriterLease(path, ttl=60, holder="B")
    b.ensure()  # free now
    assert b.held()


def test_lease_for_key_db_only_for_file_uris(tmp_path):
    assert lease_for_key_db(":memory:") is None
    assert lease_for_key_db(str(tmp_path / "keys.db")) is None   # plain local path
    lease = lease_for_key_db("file:/data/keys.db?nolock=1")
    assert lease is not None and lease.path == "/data/keys.db.writer.lock"


def test_sqlstore_write_blocked_without_lease_when_held_by_another(tmp_path):
    db = f"file:{tmp_path}/keys.db?nolock=1"
    lock = str(tmp_path / "keys.db.writer.lock")
    # A different holder already owns the lease.
    other = WriterLease(lock, ttl=60, holder="other-revision")
    other.ensure()
    mine = WriterLease(lock, ttl=60, holder="this-revision")
    with pytest.raises(WriterConflict):
        # Constructor performs the schema-init WRITE -> must hold the lease.
        SqlStore(db, writer_lease=mine)


def test_sqlstore_writes_ok_while_holding_lease(tmp_path):
    db = f"file:{tmp_path}/keys.db?nolock=1"
    lease = lease_for_key_db(db)
    store = SqlStore(db, writer_lease=lease)
    store.add_tenant(Tenant("t1", "Tenant One"))   # a guarded write
    assert lease.held()
    store.close()
    assert not lease.held()


def test_sqlstore_without_lease_is_unchanged(tmp_path):
    store = SqlStore(":memory:")          # no lease -> today's behaviour
    store.add_tenant(Tenant("t1", "Tenant One"))
    store.close()
