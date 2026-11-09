"""Local driver for Kaggle free-GPU sessions. Uses the official kaggle CLI and the
owner's ~/.kaggle credential; never prints or uploads that credential.

    python -m training.kaggle.driver code --ref HEAD
    python -m training.kaggle.driver dataset --slug llmfs-smoke-shards --dir SHARDS
    python -m training.kaggle.driver push --name llmfs-phase1-a --config session.json
    python -m training.kaggle.driver wait --name llmfs-phase1-a
    python -m training.kaggle.driver fetch --name llmfs-phase1-a --out OUT
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SESSION = Path(__file__).with_name("session.py")
ACCELERATOR = "NvidiaTeslaT4"


def username():
    with open(Path.home() / ".kaggle" / "kaggle.json") as fh:
        return json.load(fh)["username"]


def kaggle(*args, check=True):
    proc = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args[:2])} failed: {proc.stdout.strip()} {proc.stderr.strip()}")
    return proc.stdout


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def dataset_exists(ref):
    proc = subprocess.run(["kaggle", "datasets", "status", ref], capture_output=True, text=True)
    out = proc.stdout.lower()
    return proc.returncode == 0 and ("ready" in out or "pending" in out)


def wait_dataset(ref, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = kaggle("datasets", "status", ref, check=False).strip().lower()
        if "ready" in out:
            return
        if "error" in out:
            raise SystemExit(f"dataset {ref} failed processing: {out}")
        time.sleep(15)
    raise SystemExit(f"dataset {ref} not ready after {timeout}s")


def publish_dataset(slug, folder, title, message):
    ref = f"{username()}/{slug}"
    meta = {"title": title, "id": ref, "licenses": [{"name": "other"}]}
    (Path(folder) / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    if dataset_exists(ref):
        kaggle("datasets", "version", "-p", str(folder), "-m", message, "--dir-mode", "skip")
    else:
        kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "skip")
    wait_dataset(ref)
    return ref


def cmd_code(args):
    commit = subprocess.run(["git", "rev-parse", args.ref], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.strip()
    stage = Path(tempfile.mkdtemp(prefix="llmfs-code-"))
    try:
        archive = stage / "code.tar.pinned"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), commit], cwd=ROOT, check=True)
        digest = sha256_file(archive)
        ref = publish_dataset(args.slug, stage, args.slug, f"code {commit}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps({"dataset": ref, "commit": commit, "code_tar_sha256": digest}, indent=2))


def cmd_dataset(args):
    src = Path(args.dir)
    stage = Path(tempfile.mkdtemp(prefix="llmfs-data-"))
    try:
        for item in src.iterdir():
            if item.is_file() and item.name != "dataset-metadata.json":
                try:
                    os.link(item, stage / item.name)
                except OSError:
                    shutil.copy2(item, stage / item.name)
        hashes = {p.name: sha256_file(p) for p in sorted(stage.iterdir())}
        ref = publish_dataset(args.slug, stage, args.slug, args.message or f"files {len(hashes)}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps({"dataset": ref, "files": hashes}, indent=2))


def cmd_push(args):
    config = json.loads(Path(args.config).read_text())
    user = username()
    sources = [f"{user}/{config['code_slug']}"]
    for key in ("data_slug", "state_slug"):
        if config.get(key):
            sources.append(f"{user}/{config[key]}")
    for slug, _ in config.get("extra_inputs", {}).values():
        sources.append(f"{user}/{slug}")
    stage = Path(tempfile.mkdtemp(prefix="llmfs-kernel-"))
    try:
        script = SESSION.read_text().replace("__LLMFS_CONFIG__", json.dumps(config))
        (stage / "session.py").write_text(script)
        meta = {"id": f"{user}/{args.name}", "title": args.name, "code_file": "session.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": sources, "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
        (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        out = kaggle("kernels", "push", "-p", str(stage), "--accelerator", args.accelerator)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(out.strip())


def kernel_status(ref):
    return kaggle("kernels", "status", ref, check=False).strip()


def cmd_wait(args):
    ref = f"{username()}/{args.name}"
    deadline = time.monotonic() + args.timeout
    last = None
    while time.monotonic() < deadline:
        status = kernel_status(ref)
        if status != last:
            print(time.strftime("%H:%M:%S"), status, flush=True)
            last = status
        low = status.lower()
        if any(s in low for s in ("complete", "error", "cancel")):
            return 0 if "complete" in low else 1
        time.sleep(args.interval)
    print("timed out waiting", flush=True)
    return 2


def cmd_fetch(args):
    ref = f"{username()}/{args.name}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", ref, "-p", str(out), "-o")
    report = {"kernel": ref, "verified": []}
    for status_path in sorted(out.rglob("status.json")):
        status = json.loads(status_path.read_text())
        for key, name in (("ckpt_sha256", "ckpt.pt"), ("best_sha256", "best.pt")):
            path = status_path.parent / name
            if key in status and path.exists():
                ok = sha256_file(path) == status[key]
                report["verified"].append({"file": str(path.relative_to(out)), "sha256_matches_status": ok})
                if not ok:
                    raise SystemExit(f"{path} does not match the hash recorded in status.json")
    print(json.dumps(report, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("code")
    c.add_argument("--ref", default="HEAD")
    c.add_argument("--slug", default="llmfs-code")
    d = sub.add_parser("dataset")
    d.add_argument("--slug", required=True)
    d.add_argument("--dir", required=True)
    d.add_argument("--message")
    p = sub.add_parser("push")
    p.add_argument("--name", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--accelerator", default=ACCELERATOR)
    w = sub.add_parser("wait")
    w.add_argument("--name", required=True)
    w.add_argument("--timeout", type=int, default=6 * 3600)
    w.add_argument("--interval", type=int, default=60)
    f = sub.add_parser("fetch")
    f.add_argument("--name", required=True)
    f.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"code": cmd_code, "dataset": cmd_dataset, "push": cmd_push, "wait": cmd_wait,
            "fetch": cmd_fetch}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
