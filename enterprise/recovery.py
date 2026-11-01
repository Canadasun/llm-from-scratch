"""Consistent local SQLite backups and a local restore/isolation exercise.

No live cloud DB is read or changed. A successful local exercise does not prove
that production backups exist, are retained/encrypted, or can meet an RPO/RTO.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from evals.evidence import file_sha256


def backup(store, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('refusing to overwrite a backup')
    fd, tmp = tempfile.mkstemp(prefix='.backup-', dir=destination.parent)
    os.close(fd)
    try:
        with store._lock:
            if store._transaction_depth:
                raise ValueError('backup must run outside an active key transaction')
            target = sqlite3.connect(tmp)
            try:
                store._db.backup(target)
                if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('backup failed integrity check')
            finally:
                target.close()
        with open(tmp, 'rb') as f:
            os.fsync(f.fileno())
        # Atomic create-if-absent, never overwrite a concurrently published backup.
        os.link(tmp, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {'sha256': file_sha256(destination), 'bytes': destination.stat().st_size}
    finally:
        os.unlink(tmp)


def exercise():
    from enterprise.sqlstore import SqlStore
    from enterprise.models import Tenant, Principal, ApiKey, Resource
    from enterprise.keys import generate_key
    checks = {}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='llm_restore_exercise_') as tmp:
        source = SqlStore(str(Path(tmp, 'source.db')))
        restored = None
        try:
            for tenant in ('A', 'B'):
                source.add_tenant(Tenant(tenant))
                source.add_principal(Principal('user-' + tenant, tenant, ('member',)))
                source.add_resource(Resource('file-' + tenant, tenant, 'document', 'user-' + tenant))
            plain, hashed = generate_key()
            source.add_api_key(ApiKey('test-key', hashed, 'A', 'user-A'))
            snapshot = Path(tmp, 'backup.db')
            manifest = backup(source, snapshot)
            restored = SqlStore(str(snapshot))
            a = restored.get_principal('user-A')
            b = restored.get_principal('user-B')
            checks['identity_restored'] = restored.authenticate(plain)[0] == a
            checks['same_tenant_read'] = restored.get_resource(a, 'file-A') is not None
            checks['cross_tenant_read_denied'] = restored.get_resource(b, 'file-A') is None
            checks['cross_tenant_list_denied'] = all(r.tenant_id == 'B' for r in restored.list_resources(b))
            checks['plaintext_key_absent'] = plain.encode() not in snapshot.read_bytes()
            checks['backup_permissions_private'] = snapshot.stat().st_mode & 0o077 == 0
        finally:
            if restored:
                restored.close()
            source.close()
    return {'schema': 1, 'scope': 'temporary local synthetic database only',
            'live_backup_qualified': False, 'passed': all(checks.values()),
            'checks': checks, 'snapshot': manifest, 'elapsed_s': round(time.monotonic() - started, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', required=True)
    args = ap.parse_args()
    report = exercise()
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
