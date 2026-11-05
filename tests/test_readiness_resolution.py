"""Independent regressions for readiness evidence, identity and secure fixes."""
import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from enterprise.models import Tenant, Principal
from enterprise.store import Store
from enterprise.oidc import OrganizationIdentity


@pytest.fixture(scope='module')
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def organization(tmp_path, signing_key):
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    public.update(kid='current', alg='RS256', use='sig')
    policy = {'issuer': 'https://identity.example.test/tenant', 'audience': 'llm-app',
              'jwks': {'keys': [public]}, 'required_amr': ['mfa'],
              'bindings': {'subject': {'enabled': True, 'principal_id': 'pA', 'tenant_id': 'A'}}}
    path = tmp_path / 'identity.json'
    path.write_text(json.dumps(policy))
    store = Store()
    store.add_tenant(Tenant('A'))
    store.add_tenant(Tenant('B'))
    store.add_principal(Principal('pA', 'A', ('member',)))
    now = int(time.time())
    claims = {'iss': policy['issuer'], 'aud': policy['audience'], 'sub': 'subject',
              'iat': now, 'nbf': now, 'exp': now + 300, 'amr': ['pwd', 'mfa']}
    return OrganizationIdentity(path), store, path, policy, claims


def token(key, claims, **headers):
    return jwt.encode(claims, key, algorithm='RS256', headers={'kid': 'current', **headers})


def test_identity_requires_valid_signature_and_local_binding(organization, signing_key):
    identity, store, path, policy, claims = organization
    principal, key = identity.authenticate(token(signing_key, claims), store)
    assert principal == store.get_principal('pA') and key.scopes == ('model:infer',)
    policy['bindings']['subject']['enabled'] = False
    path.write_text(json.dumps(policy))
    assert identity.authenticate(token(signing_key, claims), store) == (None, None)


@pytest.mark.parametrize('change', [
    {'iss': 'https://other.example.test'}, {'aud': 'another-app'}, {'amr': ['pwd']},
    {'sub': 'unprovisioned'}, {'exp': 1}, {'nbf': 9999999999}, {'iat': 9999999999},
    {'exp': 9999999999}, {'aud': ['llm-app', 'other']}, {'exp': float('nan')},
])
def test_identity_rejects_claim_confusion(organization, signing_key, change):
    identity, store, path, policy, claims = organization
    claims.update(change)
    assert identity.authenticate(token(signing_key, claims), store) == (None, None)


def test_identity_rejects_unknown_key_and_token_key_urls(organization, signing_key):
    identity, store, path, policy, claims = organization
    assert identity.authenticate(token(signing_key, claims, kid='unknown'), store) == (None, None)
    assert identity.authenticate(token(signing_key, claims, jku='https://attacker.invalid'), store) == (None, None)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert identity.authenticate(token(other, claims), store) == (None, None)
    unsigned = jwt.encode(claims, '', algorithm='none', headers={'kid': 'current'})
    assert identity.authenticate(unsigned, store) == (None, None)


def test_identity_ignores_roles_and_blocks_tenant_rebinding(organization, signing_key):
    identity, store, path, policy, claims = organization
    claims.update(roles=['admin'], tid='B')
    principal, _ = identity.authenticate(token(signing_key, claims), store)
    assert principal.roles == ('member',) and principal.tenant_id == 'A'
    policy['bindings']['subject']['tenant_id'] = 'B'
    path.write_text(json.dumps(policy))
    assert identity.authenticate(token(signing_key, claims), store) == (None, None)


def test_identity_rejects_writable_policy(organization):
    identity, store, path, policy, claims = organization
    path.chmod(0o666)
    with pytest.raises(ValueError):
        OrganizationIdentity(path)


def test_organization_mode_does_not_accept_legacy_keys(organization):
    from enterprise.keys import generate_key
    from enterprise.models import ApiKey
    from enterprise.audit import AuditLog
    from serve.secured import SecuredInferenceService
    identity, store, path, policy, claims = organization
    plain, hashed = generate_key()
    store.add_api_key(ApiKey('legacy', hashed, 'A', 'pA'))
    class Inner:
        def handle_generate(self, *args, **kwargs):
            raise AssertionError('unauthorized generation ran')
    audit = AuditLog()
    service = SecuredInferenceService(Inner(), store, audit, organization_identity=identity)
    assert service.handle_generate({'prompt': 'x'}, api_key=plain)[0] == 401
    assert len(audit.events()) == 1


def test_oidc_is_wired_into_authorized_generation(organization, signing_key):
    from enterprise.audit import AuditLog
    from serve.secured import SecuredInferenceService
    identity, store, path, policy, claims = organization
    class Inner:
        def handle_generate(self, payload, client):
            return 200, {'completion': 'ok', 'request_id': 'test'}
    service = SecuredInferenceService(Inner(), store, AuditLog(), organization_identity=identity)
    status, body = service.handle_generate({'prompt': 'x'}, api_key=token(signing_key, claims))
    assert status == 200 and body['tenant_id'] == 'A'


def test_cloud_metadata_cannot_certify_enterprise():
    from enterprise.readiness import assess
    report = assess({'deployment': {'images': ['repo:latest']}, 'authentication': {}})
    assert not report['ready']
    assert next(c for c in report['checks'] if c['control'] == 'platform_authentication')['status'] == 'blocked'


def test_readiness_collector_uses_allowlisted_management_reads(monkeypatch):
    from enterprise import readiness
    calls = []
    def read(args):
        calls.append(args)
        return {}
    monkeypatch.setattr(readiness, 'az_json', read)
    evidence = readiness.collect()
    assert not evidence['mutations'] and not evidence['application_traffic']
    assert [c[:3] for c in calls] == [['containerapp', 'show', '-n'], ['containerapp', 'auth', 'show']]
    assert all('secret' not in c[:3] for c in calls)


def test_existing_deployment_requires_go_live(tmp_path):
    fake = tmp_path / 'az'
    fake.write_text('#!/bin/sh\necho CLOUD_CALLED\nexit 90\n')
    fake.chmod(0o755)
    import os
    result = subprocess.run(['bash', 'deploy/containerapp.sh'],
        env={**os.environ, 'PATH': str(tmp_path) + ':' + os.environ['PATH']}, capture_output=True, text=True)
    assert result.returncode == 1 and 'go-live approval' in result.stderr
    assert 'CLOUD_CALLED' not in result.stdout


def test_local_restore_exercise_preserves_tenant_boundary():
    from enterprise.recovery import exercise
    result = exercise()
    assert result['passed'] and not result['live_backup_qualified']


def test_backup_refuses_overwrite_and_active_transactions(tmp_path):
    from enterprise.recovery import backup
    from enterprise.sqlstore import SqlStore
    store = SqlStore()
    try:
        with store.key_transaction():
            with pytest.raises(ValueError, match='active'):
                backup(store, tmp_path / 'active.db')
        assert not (tmp_path / 'active.db').exists()
        dest = tmp_path / 'backup.db'
        backup(store, dest)
        with pytest.raises(ValueError, match='overwrite'):
            backup(store, dest)
    finally:
        store.close()


def test_protocol_hash_is_never_rewritten():
    from codereview.fixes import propose_fixes
    source = 'import hashlib\nhashlib.md5(value, usedforsecurity=False)\n'
    assert propose_fixes(source) == []


@pytest.mark.parametrize('source', [
    'import ssl\ndef context():\n    return ssl._create_unverified_context()\n',
    'import yaml\ndef parse(doc):\n    return yaml.load(doc, Loader=yaml.Loader)\n',
])
def test_extended_security_witnesses_are_independent(source, tmp_path):
    from codereview.security_tests import reproduce
    from codereview.fixes import propose_fixes, apply_fixes
    from evals.code_executor import sandbox_preflight
    if not sandbox_preflight()['available']:
        import os
        if os.environ.get('LLM_REQUIRE_SANDBOX_TESTS') == '1':
            pytest.fail('required sandbox unavailable')
        pytest.skip('sandbox unavailable')
    path = tmp_path / 'module.py'
    path.write_text(source)
    props = propose_fixes(source)
    assert len(props) == 1
    before = reproduce(path, props[0].line)
    assert before['detail'] == 'case 1 output mismatch'
    patched = apply_fixes(source, props)
    ast.parse(patched)
    path.write_text(patched)
    assert reproduce(path, props[0].line)['status'] == 'passed'


def test_image_protocol_never_sends_expected_answers():
    from evals.serving_image import ImageWorker
    worker = ImageWorker.__new__(ImageWorker)
    class Proc:
        stdin = io.BytesIO()
    worker.proc = Proc()
    worker.read = lambda timeout: {'completion': 'x'}
    worker.query({'type': 'generation', 'prompt': 'hello', 'target': 'secret target',
                  'answer': 4, 'test': 'private test'}, 10)
    payload = json.loads(worker.proc.stdin.getvalue())
    assert not {'target', 'answer', 'test'}.intersection(payload)


def test_training_handoff_has_no_cloud_or_promotion():
    from training.free_gpu import plan
    spec = plan('/tmp/test-llm-candidate', steps=1)
    assert not spec['azure_operations'] and not spec['paid_fallback'] and not spec['automatic_promotion']
    assert '--device' in spec['commands'][1] and 'cuda' in spec['commands'][1]
    with pytest.raises(ValueError):
        plan('checkpoints')


def test_free_gpu_runner_never_falls_back_to_cpu(tmp_path, monkeypatch):
    from training import free_gpu
    out = tmp_path / 'candidate'
    monkeypatch.setattr(sys, 'argv', ['free-gpu', '--output', str(out)])
    monkeypatch.setattr(free_gpu.subprocess, 'run', lambda *args, **kwargs:
                        pytest.fail('training must not start without a confirmed free GPU'))
    with pytest.raises(SystemExit) as stopped:
        free_gpu.main()
    assert stopped.value.code == 1 and not out.exists()


@pytest.mark.parametrize('missing', ['exp', 'iat', 'nbf', 'sub', 'iss', 'aud'])
def test_identity_rejects_missing_required_claims(organization, signing_key, missing):
    identity, store, path, policy, claims = organization
    del claims[missing]
    assert identity.authenticate(token(signing_key, claims), store) == (None, None)


def test_yaml_nested_edits_are_not_overlapped():
    from codereview.fixes import propose_fixes, apply_fixes
    source = 'import yaml\nyaml.load(fetch(verify=False))\n'
    props = propose_fixes(source)
    # TLS keyword edit can be proposed; the containing YAML call must not
    # overwrite that edit with a replacement built from the old argument AST.
    assert not any(p.rule_id == 'PY-INSECURE-DESERIALIZATION' for p in props)
    ast.parse(apply_fixes(source, props))
