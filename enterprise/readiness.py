"""Read-only cloud evidence and fail-closed enterprise release assessment.

Only Azure management GET/show commands, with field allowlists. Never contacts
application ingress (which could wake a replica), lists secret values, provisions
resources, changes tenant settings, or claims budgets hard-limit expenditure.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

APP_QUERY = ('{name:name,state:properties.provisioningState,'
    'ingress:properties.configuration.ingress.{external:external,allowInsecure:allowInsecure,fqdn:fqdn},'
    'revisionMode:properties.configuration.activeRevisionsMode,'
    'traffic:properties.configuration.ingress.traffic,'
    'images:properties.template.containers[].image,scale:properties.template.scale,'
    'envNames:properties.template.containers[].env[].name,'
    'secretRefs:properties.template.containers[].env[].secretRef,'
    'volumes:properties.template.volumes,identityType:identity.type}')
AUTH_QUERY = ('{platformEnabled:platform.enabled,'
    'requireAuthentication:globalValidation.requireAuthentication,'
    'unauthenticatedAction:globalValidation.unauthenticatedClientAction,'
    'entraEnabled:identityProviders.azureActiveDirectory.enabled}')


def az_json(args):
    result = subprocess.run(['az', *args, '-o', 'json'], check=True,
                            capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


def collect(app='llm-serve', group='llm-from-scratch-rg'):
    # Each command is explicitly read-only. Collectors with insufficient
    # permissions record unavailable evidence, not an inferred passing result.
    evidence = {'schema': 1, 'collected_at': datetime.now(timezone.utc).isoformat(),
                'app': app, 'resource_group': group, 'mutations': False,
                'application_traffic': False, 'azure_budget_usd': 0}
    for field, args in (
        ('deployment', ['containerapp', 'show', '-n', app, '-g', group, '--query', APP_QUERY]),
        ('authentication', ['containerapp', 'auth', 'show', '-n', app, '-g', group, '--query', AUTH_QUERY]),
    ):
        try:
            evidence[field] = az_json(args)
        except (subprocess.SubprocessError, ValueError) as exc:
            evidence[field] = None
            evidence.setdefault('unavailable', []).append({'control': field, 'reason': type(exc).__name__})
    return evidence


def assess(snapshot):
    app = snapshot.get('deployment') or {}
    auth = snapshot.get('authentication') or {}
    images = app.get('images') or []
    ingress = app.get('ingress') or {}
    checks = []
    def add(name, passed, detail):
        checks.append({'control': name, 'status': 'unmeasured' if passed is None else ('pass' if passed else 'blocked'), 'detail': detail})
    add('immutable_deployment', bool(images) and all('@sha256:' in x for x in images),
        'Every deployed image must be pinned by digest, not tag.')
    add('platform_authentication', auth.get('platformEnabled') is True and
        auth.get('requireAuthentication') is True and auth.get('entraEnabled') is True,
        'Require enabled organizational authentication; no configured settings were returned by this snapshot.')
    add('tls_ingress', ingress.get('allowInsecure') is False,
        'Require explicit rejection of insecure ingress.')
    add('service_identity', app.get('identityType') not in (None, '', 'None'),
        'Managed service identity must be configured and its least-privilege grants reviewed.')
    add('supported_persistence', bool(app.get('volumes')) and not any(
        volume.get('storageType') == 'AzureFile' for volume in app.get('volumes', [])),
        'Azure Files SQLite is not qualified. A supported database and tenant-isolation evidence are required.')
    # Passing the config checks above does NOT establish these behavioral and
    # organizational controls. They require evidence not present in ARM metadata.
    for name in ('identity_offboarding_mfa', 'tenant_isolation', 'secret_lifecycle',
                 'audit_retention', 'backup_restore', 'load_and_recovery',
                 'incident_and_rollback', 'model_capability', 'serving_image_qualification'):
        add(name, None, 'Not established by this management snapshot; requires independent execution evidence.')
    return {'schema': 1, 'ready': all(c['status'] == 'pass' for c in checks), 'checks': checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['collect', 'assess'])
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--report')
    ap.add_argument('--app', default='llm-serve')
    ap.add_argument('--group', default='llm-from-scratch-rg')
    args = ap.parse_args()
    if args.action == 'collect':
        snapshot = collect(args.app, args.group)
        Path(args.snapshot).write_text(json.dumps(snapshot, indent=2) + '\n')
    else:
        snapshot = json.loads(Path(args.snapshot).read_text())
    report = assess(snapshot)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
