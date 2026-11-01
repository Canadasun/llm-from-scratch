"""Offline release verification and local qualification of immutable images.

No cloud calls. Qualification runs the image with no network, read-only root,
non-root UID, bounded memory/PIDs/CPU, and no host credentials. A host unable to
run the independent code sandbox fails qualification. Records are trusted CI
artifacts, not signatures: protect their publisher and release policy in CI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from evals.evidence import CODE_EVALUATOR, SCORING_VERSION, file_sha256, suite_sha256
from evals.tasks import default_task_files, load_tasks
from evals.run import implementation_sha256, HELDOUT
from evals.review import validate_results
from evals.regression import check

POLICY = Path(__file__).resolve().parents[1] / "evals/baselines/production.json"
IMAGE_PATTERN = r"[a-zA-Z0-9][a-zA-Z0-9./:_-]*@sha256:[0-9a-f]{64}"


def verify(record, image, policy_path=POLICY):
    if not re.fullmatch(IMAGE_PATTERN, image):
        raise ValueError("an immutable repository@sha256 image is required")
    if record.get("schema") != 1 or record.get("image") != image:
        raise ValueError("release record does not bind this image")
    policy = json.loads(Path(policy_path).read_text())
    if record.get("policy_sha256") != file_sha256(policy_path):
        raise ValueError("release policy changed; requalify")
    run = record.get("evaluation", {})
    validate_results(run)
    meta = run.get("meta", {})
    tasks = load_tasks(default_task_files())
    expected = {"complete": True, "quantized": True, "device": "cpu",
                "scoring_version": SCORING_VERSION, "code_evaluator": CODE_EVALUATOR,
                "suite_sha256": suite_sha256(tasks), "image": image,
                "task_ids": [t["id"] for t in tasks],
                "implementation_sha256": implementation_sha256(),
                "heldout_sha256": file_sha256(HELDOUT), "suite": "all", "limit": None}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"missing or mismatched artifact evidence: {key}")
    if [r.get("id") for r in run.get("results", [])] != expected["task_ids"]:
        raise ValueError("full frozen suite required")
    for task, result in zip(tasks, run["results"]):
        if (result.get("type"), result.get("category")) != (task["type"], task["category"]):
            raise ValueError("result task identity mismatch")
        if task["type"] == "generation" and task.get("scorer") != "exact":
            raise ValueError("frozen generation tasks still use heuristic scoring; replace with reviewed independent tests")
    for name in ("checkpoint_sha256", "tokenizer_sha256", "implementation_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(meta.get(name, ""))):
            raise ValueError(f"missing {name}")
    # Keyword/regex hits are exploratory measurements, never production proof.
    if any(r.get("type") == "generation" and r.get("verification") != "exact"
           for r in run.get("results", [])):
        raise ValueError("heuristic generation scores cannot qualify a production release")
    ok, checks = check(run, policy)
    if not ok:
        failed = [c["metric"] for c in checks if not c["ok"]]
        raise ValueError("production capability gate failed: " + ", ".join(failed))
    return record


def qualify(image, output):
    if not re.fullmatch(IMAGE_PATTERN, image):
        raise ValueError("qualify a digest, never a mutable tag")
    with tempfile.TemporaryDirectory(prefix="llm_qualification_") as tmp:
        os.chmod(tmp, 0o777)  # only this empty result directory is writable by worker
        subprocess.run([
            "docker", "run", "--rm", "--network=none", "--read-only",
            "--user=65534:65534", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--memory=2g", "--pids-limit=64", "--cpus=2",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=128m", "-v", f"{tmp}:/results:rw",
            "-e", f"RELEASE_IMAGE={image}", "--entrypoint=python", image,
            "-m", "evals.run", "--ckpt", "checkpoints/ckpt.pt", "--device=cpu",
            "--quantize", "--threads=2", "--task-timeout=120", "--json=/results/run.json"
        ], check=True, timeout=24000)
        evaluation = json.loads(Path(tmp, "run.json").read_text())
    record = {"schema": 1, "image": image, "policy_sha256": file_sha256(POLICY),
              "evaluation": evaluation}
    # Preserve failed measurements too; no success record is emitted silently.
    Path(output).write_text(json.dumps(record, indent=2) + "\n")
    verify(record, image)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["qualify", "verify"])
    ap.add_argument("--image", required=True)
    ap.add_argument("--record", required=True)
    args = ap.parse_args()
    try:
        if args.action == "qualify":
            qualify(args.image, args.record)
        else:
            verify(json.loads(Path(args.record).read_text()), args.image)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        ap.exit(1, f"RELEASE BLOCKED: {exc}\n")


if __name__ == "__main__":
    main()
