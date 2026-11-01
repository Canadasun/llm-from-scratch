"""Explicit handoff for an operator-provided FREE GPU; never provisions compute.

--plan is safe on CPU. Execution requires CUDA AND an explicit free-runtime
confirmation; there is no paid fallback, Azure API, deployment or promotion.
This is a training experiment, not a guarantee of passing coding capability.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def plan(output, steps=8000, batch_size=4, max_seconds=10800):
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1 or max_seconds < 1:
        raise ValueError('positive training limits required')
    path = Path(output).resolve()
    root = Path(__file__).resolve().parents[1]
    if path == root / 'checkpoints' or path == root / 'checkpoints/instruction-candidate':
        raise ValueError('use a NEW candidate directory, never serving checkpoints')
    return {'schema': 1, 'experiment': 'code-first gpu_xl foundation training',
        'requires': 'operator-provided free CUDA runtime', 'azure_operations': False,
        'paid_fallback': False, 'automatic_promotion': False,
        'steps': steps, 'batch_size': batch_size, 'max_seconds': max_seconds,
        'out_dir': str(path), 'commands': [
            [sys.executable, '-m', 'data.build_training_corpus', '--prose', '',
             '--out', str(path / 'corpus.txt')],
            [sys.executable, 'train.py', '--preset', 'gpu_xl', '--device', 'cuda',
             '--corpus', str(path / 'corpus.txt'), '--out-dir', str(path),
             '--max-iters', str(steps), '--batch-size', str(batch_size), '--checkpoint-interval', '200']]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-seconds', type=int, default=10800)
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--confirm-free-runtime', action='store_true')
    args = ap.parse_args()
    spec = plan(args.output, args.steps, args.batch_size, args.max_seconds)
    print(json.dumps(spec, indent=2), flush=True)
    if args.plan:
        return 0
    import torch
    if not args.confirm_free_runtime or not torch.cuda.is_available():
        ap.exit(1, 'TRAINING BLOCKED: an operator-confirmed free CUDA runtime is required; no CPU/paid fallback.\n')
    out = Path(spec['out_dir'])
    if out.exists():
        ap.exit(1, 'TRAINING BLOCKED: candidate directory exists; choose a fresh experiment directory.\n')
    out.mkdir(parents=True)
    (out / 'experiment.json').write_text(json.dumps(spec, indent=2) + '\n')
    deadline = time.monotonic() + spec['max_seconds']
    try:
        for command in spec['commands']:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True,
                           timeout=max(0.001, deadline - time.monotonic()))
    except (subprocess.SubprocessError, OSError) as exc:
        ap.exit(1, f'Training stopped without promotion: {type(exc).__name__}; preserve periodic checkpoints.\n')
    print('Candidate saved. Run independent CPU/image qualification; no release claim or promotion was made.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
