"""Explicit handoff for an operator-provided FREE GPU; never provisions compute.

--plan is safe on CPU. Execution requires CUDA AND an explicit free-runtime
confirmation; there is no paid fallback, Azure API, deployment or promotion.
Each session trains until its deadline, writes a resumable checkpoint, and the
next session continues with --resume. This is a training experiment, not a
guarantee of passing coding capability.

    python -m training.free_gpu --output RUN1 --data-dir DATA --expect-data-sha256 H \
        --steps 30000 --max-seconds 30600 --confirm-free-runtime
    python -m training.free_gpu --output RUN2 --data-dir DATA --expect-data-sha256 H \
        --resume RUN1/ckpt.pt --steps 30000 --max-seconds 30600 --confirm-free-runtime
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def plan(output, steps=8000, batch_size=4, max_seconds=10800, data_dir=None, resume=None,
         preset='gpu_xl', grad_accum_steps=None, lr_decay_iters=None, expect_data_sha256=None,
         reserve_seconds=600, checkpoint_minutes=15):
    for value in (steps, batch_size):
        if type(value) is not int or value < 1:
            raise ValueError('positive training limits required')
    if max_seconds < 1 or reserve_seconds < 0 or max_seconds <= reserve_seconds:
        raise ValueError('positive training limits required; the session must outlast the save reserve')
    path = Path(output).resolve()
    root = Path(__file__).resolve().parents[1]
    if path == root / 'checkpoints' or path == root / 'checkpoints/instruction-candidate':
        raise ValueError('use a NEW candidate directory, never serving checkpoints')
    if resume is not None and Path(resume).resolve().parent == path:
        raise ValueError('write each session to a new directory; never overwrite the checkpoint being resumed')
    train = [sys.executable, 'train.py', '--device', 'cuda', '--out-dir', str(path),
             '--max-iters', str(steps), '--checkpoint-interval', '200',
             '--checkpoint-minutes', str(checkpoint_minutes),
             '--deadline-seconds', str(max_seconds - reserve_seconds)]
    if resume is None:
        train[2:2] = ['--preset', preset]
        train += ['--batch-size', str(batch_size)]
        if grad_accum_steps:
            train += ['--grad-accum-steps', str(grad_accum_steps)]
        if lr_decay_iters:
            train += ['--lr-decay-iters', str(lr_decay_iters)]
    else:
        train += ['--resume', str(Path(resume).resolve())]
    if data_dir is not None:
        data = Path(data_dir).resolve()
        prepare = [sys.executable, '-m', 'data.shards', 'verify', str(data)]
        if expect_data_sha256:
            prepare += ['--expect-sha256', expect_data_sha256]
            train += ['--expect-data-sha256', expect_data_sha256]
        train += ['--data-dir', str(data), '--skip-shard-hash']
    else:
        prepare = [sys.executable, '-m', 'data.build_training_corpus', '--prose', '',
                   '--out', str(path / 'corpus.txt')]
        train += ['--corpus', str(path / 'corpus.txt')]
    return {'schema': 2, 'experiment': f'code-first {preset} foundation training',
            'requires': 'operator-provided free CUDA runtime', 'azure_operations': False,
            'paid_fallback': False, 'automatic_promotion': False,
            'steps': steps, 'batch_size': batch_size, 'max_seconds': max_seconds,
            'reserve_seconds': reserve_seconds, 'resume': str(resume) if resume else None,
            'data_dir': str(data_dir) if data_dir else None, 'expect_data_sha256': expect_data_sha256,
            'out_dir': str(path), 'commands': [prepare, train]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', required=True)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--grad-accum-steps', type=int)
    ap.add_argument('--lr-decay-iters', type=int)
    ap.add_argument('--preset', default='gpu_xl')
    ap.add_argument('--max-seconds', type=int, default=10800)
    ap.add_argument('--reserve-seconds', type=int, default=600)
    ap.add_argument('--data-dir')
    ap.add_argument('--expect-data-sha256')
    ap.add_argument('--resume')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--confirm-free-runtime', action='store_true')
    args = ap.parse_args()
    spec = plan(args.output, args.steps, args.batch_size, args.max_seconds, data_dir=args.data_dir,
                resume=args.resume, preset=args.preset, grad_accum_steps=args.grad_accum_steps,
                lr_decay_iters=args.lr_decay_iters, expect_data_sha256=args.expect_data_sha256,
                reserve_seconds=args.reserve_seconds)
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
    status_path = out / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'state': 'unknown'}
    print(json.dumps(status, indent=2))
    if status.get('state') == 'completed':
        print('Candidate saved. Run independent CPU/image qualification; no release claim or promotion was made.')
    else:
        print(f"Session ended ({status.get('state')}); continue with --resume {out / 'ckpt.pt'} in a new directory.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
