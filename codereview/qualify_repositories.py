"""Reproducible real-source scans and controlled security mutation qualification.

Read-only upstream checkouts, pinned commits, no upstream imports or host tests.
Controlled injected defects are explicitly separate from natural findings. The
independent behavioral witnesses run in the JSON sandbox. This is mutation
coverage for supported grammars, NOT a recall claim over unknown vulnerabilities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from codereview.fixes import apply_fixes, propose_fixes
from codereview.scanner import scan_path
from codereview.security_tests import reproduce
from evals.code_executor import sandbox_preflight

REPOSITORIES = [
    {'name': 'requests', 'url': 'https://github.com/psf/requests',
     'commit': '3c7baaa05ab4aa49cc819ce3a819d260de19c157', 'file': 'src/requests/auth.py'},
    {'name': 'flask', 'url': 'https://github.com/pallets/flask',
     'commit': '49943b5a94413a530761350f04e896f76e8d620e', 'file': 'src/flask/sessions.py'},
    {'name': 'pyyaml', 'url': 'https://github.com/yaml/pyyaml',
     'commit': 'f5e4504a92f169f65350a1047dccb23bc97028be', 'file': 'lib/yaml/__init__.py'},
]
MUTATIONS = [
    {'kind': 'digest-algorithm', 'rule': 'PY-WEAK-HASH',
     'source': 'import hashlib\ndef llm_qualification_probe(value):\n    return hashlib.md5(value).hexdigest()\n'},
    {'kind': 'tls-context', 'rule': 'PY-TLS-VERIFY-DISABLED',
     'source': 'import ssl\ndef llm_qualification_probe():\n    return ssl._create_unverified_context()\n'},
    {'kind': 'yaml-construction', 'rule': 'PY-INSECURE-DESERIALIZATION',
     'source': 'import yaml\ndef llm_qualification_probe(value):\n    return yaml.load(value, Loader=yaml.Loader)\n'},
]


def qualify(root, output):
    sandbox = sandbox_preflight()
    if not sandbox['available']:
        raise ValueError('independent sandbox required')
    report = {'schema': 1, 'scope': 'real-source scans + controlled injected mutations',
              'not_established': ['natural-vulnerability precision/recall', 'upstream full test suites',
                                  'general remediation', 'model coding capability'],
              'sandbox': sandbox, 'repositories': []}
    for repo in REPOSITORIES:
        path = Path(root) / repo['name']
        commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != repo['commit']:
            raise ValueError(f'checkout mismatch: {repo["name"]}')
        if subprocess.check_output(['git', '-C', str(path), '-c', 'core.fsmonitor=false',
                                    'status', '--porcelain'], text=True).strip():
            raise ValueError('checkout must be clean')
        scan = scan_path(str(path))
        entry = {**repo, 'files_scanned': scan['files_scanned'], 'skipped': scan['skipped'],
                 'natural_findings': [{**f.to_dict(), 'file': str(Path(f.file).relative_to(path))}
                                      for f in scan['findings']], 'mutations': []}
        source = (path / repo['file']).read_text()
        entry['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        for mutation in MUTATIONS:
            injected = source + '\n\n' + mutation['source']
            start = len((source + '\n\n').splitlines()) + 1
            with tempfile.TemporaryDirectory(prefix='llm_security_mutation_') as tmp:
                target = Path(tmp) / 'subject.py'
                target.write_text(injected)
                props = [p for p in propose_fixes(injected, str(target))
                         if p.line >= start and p.rule_id == mutation['rule']]
                result = {'kind': mutation['kind'], 'rule': mutation['rule'], 'injected': True,
                          'detected_and_fixable': len(props) == 1, 'passed': False}
                if len(props) == 1:
                    before = reproduce(target, props[0].line)
                    target.write_text(apply_fixes(injected, props))
                    after = reproduce(target, props[0].line)
                    result.update(before={k: before.get(k) for k in ('status', 'detail', 'cases_passed', 'cases_total')},
                                  after={k: after.get(k) for k in ('status', 'detail', 'cases_passed', 'cases_total')})
                    result['passed'] = (before['status'] == 'failed' and
                        before['detail'] == 'case 1 output mismatch' and after['status'] == 'passed')
                entry['mutations'].append(result)
        report['repositories'].append(entry)
    report['mutations_total'] = sum(len(r['mutations']) for r in report['repositories'])
    report['mutations_passed'] = sum(m['passed'] for r in report['repositories'] for m in r['mutations'])
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    report = qualify(args.root, args.json)
    print(f"{report['mutations_passed']}/{report['mutations_total']} controlled security mutations verified")
    return 0 if report['mutations_passed'] == report['mutations_total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
