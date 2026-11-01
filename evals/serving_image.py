"""Measure a cached serving image without installing/replacing code inside it.

Model inference and MC likelihood run in that immutable image. Expected answers,
code execution and verdicts stay in the host controller. No Azure endpoint is
contacted; Docker uses --pull=never, no network, and bounded non-root execution.
This also works with legacy images that predate our current evaluator CLI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time
import uuid

from evals.evidence import CODE_EVALUATOR, SCORING_VERSION, suite_sha256
from evals.report import summarize
from evals.tasks import default_task_files, load_tasks
from evals import scorers
from evals.code_executor import evaluate_function, task_spec, sandbox_preflight

BRIDGE = r'''
import hashlib, json, sys
from pathlib import Path
import torch
torch.set_num_threads(2)
from serve.engine import InferenceEngine
from evals.scoring import choose_by_likelihood
engine = InferenceEngine.from_checkpoint('checkpoints/ckpt.pt', device='cpu', quantize=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
paths = [Path('model.py'), Path('tokenization.py')]
for folder in ('evals', 'serve'): paths.extend(sorted(Path(folder).rglob('*.py')))
h = hashlib.sha256()
for path in paths:
    h.update(str(path).encode() + bytes([0])); h.update(path.read_bytes())
print(json.dumps({'ready': True, 'checkpoint_sha256': sha('checkpoints/ckpt.pt'),
  'tokenizer_sha256': hashlib.sha256(json.dumps(engine.tokenizer.to_dict(), sort_keys=True).encode()).hexdigest(),
  'image_implementation_sha256': h.hexdigest(), 'quantized': engine.quantized,
  'torch_version': str(torch.__version__), 'block_size': engine.block_size}), flush=True)
for line in sys.stdin:
    try:
        q = json.loads(line)
        if q['type'] == 'multiple_choice':
            best, _ = choose_by_likelihood(engine.model, engine.tokenizer, q['prompt'], q['choices'], 'cpu')
            result = {'chosen': best}
        else:
            result = engine.generate(q['prompt'], max_new_tokens=q['max_new_tokens'],
                temperature=q['temperature'], top_k=q['top_k'], seed=1337)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__}), flush=True)
'''


class ImageWorker:
    def __init__(self, image):
        from deploy.release import IMAGE_PATTERN
        import re
        if not re.fullmatch(IMAGE_PATTERN, image):
            raise ValueError('immutable image digest required')
        self.name = 'llm-local-qualification-' + uuid.uuid4().hex[:12]
        self.stderr = tempfile.TemporaryFile()
        self.buffer = b''
        self.proc = subprocess.Popen([
            'docker', 'run', '--rm', '--pull=never', '--name', self.name,
            '--network=none', '--read-only', '--user=65534:65534', '--cap-drop=ALL',
            '--security-opt=no-new-privileges', '--memory=2g', '--pids-limit=64',
            '--cpus=2', '--log-driver=none', '--tmpfs=/tmp:rw,nosuid,nodev,size=128m',
            '-i', '--entrypoint=python', image, '-u', '-c', BRIDGE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr)

    def read(self, timeout):
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.proc.stdout], [], [], remaining)[0]:
                raise TimeoutError('serving image task deadline exceeded')
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('serving image worker exited')
            self.buffer += chunk
            if len(self.buffer) > 1024 * 1024:
                raise ValueError('oversized image response')
        line, self.buffer = self.buffer.split(b'\n', 1)
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError('invalid image response')
        return value

    def query(self, task, timeout):
        # Deliberately exclude expected answers, scorer targets and code cases.
        q = {key: task[key] for key in ('type', 'prompt', 'choices') if key in task}
        q.update(max_new_tokens=task.get('max_new_tokens', 256 if task['type'] == 'code' else 64),
                 temperature=task.get('temperature', .4 if task['type'] == 'code' else .6),
                 top_k=task.get('top_k', 40))
        self.proc.stdin.write(json.dumps(q).encode() + b'\n')
        self.proc.stdin.flush()
        return self.read(timeout)

    def close(self):
        subprocess.run(['docker', 'rm', '-f', self.name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=20)
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.communicate(timeout=10)
        self.stderr.close()


def score(task, response):
    base = {k: task[k] for k in ('id', 'type', 'category')}
    if 'error' in response:
        return {**base, 'correct': False, 'detail': 'image error: ' + response['error']}
    if task['type'] == 'multiple_choice':
        best = response.get('chosen')
        return {**base, 'correct': type(best) is int and best == task['answer'],
                'chosen': best, 'answer': task['answer'], 'n_choices': len(task['choices'])}
    text = response.get('completion')
    if not isinstance(text, str):
        raise ValueError('completion is required')
    if task['type'] == 'code':
        spec = task_spec(task)
        if not spec:
            return {**base, 'correct': False, 'execution_status': 'unavailable'}
        verdict = evaluate_function(scorers.extract_code(text), spec['function'], spec['cases'],
                                    timeout=task.get('timeout', 10))
        return {**base, 'correct': verdict.passed, 'execution_status': verdict.status,
                'detail': verdict.detail, 'cases_passed': verdict.cases_passed,
                'cases_total': verdict.cases_total, 'completion': text}
    name = task.get('scorer', 'contains')
    fn = {'exact': scorers.exact_match, 'contains': scorers.contains,
          'contains_any': scorers.contains_any, 'contains_all': scorers.contains_all,
          'regex': scorers.regex_match}[name]
    ok = fn(text, task['target']) and scorers.lacks_forbidden(text, task.get('forbidden'))
    return {**base, 'correct': ok, 'scorer': name,
            'verification': 'exact' if name == 'exact' else 'heuristic', 'completion': text}


def evaluate(image, output, timeout=120, limit=None):
    tasks = load_tasks(default_task_files())
    selected = tasks if limit is None else tasks[:limit]
    sandbox = sandbox_preflight()
    if not sandbox['available']:
        raise ValueError('independent controller sandbox unavailable: ' + sandbox['detail'])
    worker = ImageWorker(image)
    results = []
    run = {'results': results, 'meta': {'image': image, 'complete': False, 'suite': 'tasks',
        'limit': limit, 'suite_sha256': suite_sha256(tasks), 'scoring_version': SCORING_VERSION,
        'code_evaluator': CODE_EVALUATOR, 'sandbox': sandbox,
        'evaluator_transport': 'immutable-image-json-v1', 'tasks_total': len(tasks),
        'live_digest_attested': False}}
    try:
        run['meta'].update(worker.read(120))
        run['meta']['device'] = 'cpu'
        for index, task in enumerate(selected, 1):
            started = time.monotonic()
            result = score(task, worker.query(task, timeout))
            result['duration_s'] = round(time.monotonic() - started, 3)
            results.append(result)
            if index % 10 == 0 or task['type'] == 'code':
                print(f'{index}/{len(selected)} {task["id"]}: {result["correct"]}', flush=True)
            run['summary'] = summarize(results)
            Path(output).write_text(json.dumps(run, indent=2) + '\n')
        run['meta']['task_ids'] = [t['id'] for t in selected]
        run['meta']['capability_complete'] = limit is None
        return run
    finally:
        worker.close()
        run['summary'] = summarize(results)
        Path(output).write_text(json.dumps(run, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--task-timeout', type=float, default=120)
    args = ap.parse_args()
    try:
        out = evaluate(args.image, args.json, args.task_timeout, args.limit)
        print(json.dumps(out['summary'], indent=2))
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        ap.exit(1, f'IMAGE QUALIFICATION FAILED: {exc}\n')


if __name__ == '__main__':
    main()
