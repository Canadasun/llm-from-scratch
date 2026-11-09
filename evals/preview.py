"""Research-preview baseline: frozen hashes, fixed targets, and reproducible measurements.

    python -m evals.preview check --bench-dir DIR                # CI: rescore committed programs
    python -m evals.preview check --bench-dir DIR --print-only   # show scores without comparing

The preview baseline is separate from evals/baselines/production.json and never
changes it. Targets were set before Phase 4 training and must not move.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.bench import FROZEN  # noqa: E402
from evals.bench_run import BEHAVIOUR, score_all  # noqa: E402

PREVIEW = ROOT / "evals" / "baselines" / "preview.json"
PRODUCTION = ROOT / "evals" / "baselines" / "production.json"
PROGRAMS = ROOT / "evals" / "results" / "preview"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hashes():
    return {"bench_manifest": sha256(FROZEN), "behaviour_suite": sha256(BEHAVIOUR),
            "production_baseline": sha256(PRODUCTION)}


def bench_numbers(scored):
    out = {s: {"passed": v["passed"], "tasks": v["tasks"]} for s, v in scored["summary"].items()}
    out["parses_as_python"] = scored["parses_as_python"]
    out["behaviour"] = {k: {"correct": v["correct"], "n": v["n"]} for k, v in scored["behaviour"].items()}
    return out


def check(bench_dir, print_only=False):
    preview = json.loads(PREVIEW.read_text())
    problems = []
    current = frozen_hashes()
    for key, value in preview["frozen"].items():
        if current.get(key) != value:
            problems.append(f"frozen hash changed: {key}")
    measured = {}
    for name, entry in preview["measured"].items():
        path = PROGRAMS / f"programs-{name}.json"
        generated = json.loads(path.read_text())
        if generated["meta"]["checkpoint_sha256"] != entry["checkpoint_sha256"]:
            problems.append(f"{name}: programs were generated from a different checkpoint")
        if sha256(path) != entry["programs_sha256"]:
            problems.append(f"{name}: programs file changed")
        measured[name] = bench_numbers(score_all(generated, bench_dir))
        if entry.get("benchmarks") is None:
            print(f"{name}: benchmarks not yet recorded", file=sys.stderr)
        elif not print_only and entry["benchmarks"] != measured[name]:
            problems.append(f"{name}: rescored benchmarks differ from the recorded baseline")
    return measured, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--bench-dir", required=True)
    c.add_argument("--print-only", action="store_true")
    sub.add_parser("hashes")
    args = ap.parse_args(argv)
    if args.cmd == "hashes":
        print(json.dumps(frozen_hashes(), indent=2))
        return 0
    measured, problems = check(args.bench_dir, args.print_only)
    print(json.dumps(measured, indent=2, sort_keys=True))
    for p in problems:
        print(f"PROBLEM: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
