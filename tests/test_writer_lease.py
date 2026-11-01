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


def test_expired_holder_is_never_stolen(tmp_path):
    clk = _Clock()
    path = str(tmp_path / "l.lock")
    first = WriterLease(path, clock=clk)
    first.ensure()
    clk.advance(10000)
    with pytest.raises(WriterConflict):
        WriterLease(path, clock=clk).ensure()
    first.ensure()
    assert first.held()


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
    lease = lease_for_key_db("file:/data/keys.db")
    assert lease is not None and lease.path == "/data/keys.db.writer.lock"


def test_sqlstore_write_blocked_without_lease_when_held_by_another(tmp_path):
    db = f"file:{tmp_path}/keys.db"
    lock = str(tmp_path / "keys.db.writer.lock")
    # A different holder already owns the lease.
    other = WriterLease(lock, ttl=60, holder="other-revision")
    other.ensure()
    mine = WriterLease(lock, ttl=60, holder="this-revision")
    with pytest.raises(WriterConflict):
        # Constructor performs the schema-init WRITE -> must hold the lease.
        SqlStore(db, writer_lease=mine)


def test_sqlstore_writes_ok_while_holding_lease(tmp_path):
    db = f"file:{tmp_path}/keys.db"
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


def test_concurrent_acquisition_has_exactly_one_winner(tmp_path):
    import concurrent.futures
    import threading
    barrier = threading.Barrier(16)
    leases = [WriterLease(str(tmp_path / "lock")) for _ in range(16)]
    def claim(lease):
        barrier.wait()
        try:
            lease.ensure()
            return True
        except WriterConflict:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        assert sum(pool.map(claim, leases)) == 1
    assert sum(lease.held() for lease in leases) == 1


def test_legacy_lock_is_not_overwritten(tmp_path):
    path = tmp_path / "lock"
    path.write_text('{"holder": "legacy"}')
    with pytest.raises(WriterConflict):
        WriterLease(str(path)).ensure()
    assert path.read_text() == '{"holder": "legacy"}'


@pytest.mark.parametrize("option", ["nolock=1", "nolock=true", "nolock=on", "nolock=yes", "nolock="])
def test_disabled_sqlite_locking_is_refused_before_open(tmp_path, option):
    path = tmp_path / "keys.db"
    with pytest.raises(ValueError, match="nolock"):
        SqlStore(f"file:{path}?{option}")
    assert not path.exists()
