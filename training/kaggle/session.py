"""Kaggle kernel entry point. The driver embeds CONFIG before each push.

Runs on Kaggle only: finds the pinned code archive among the attached private
datasets, verifies its hash, extracts it outside /kaggle/working, and runs the
configured steps, writing logs and a summary to /kaggle/working. It never needs
internet access or any GitHub or cloud credential.
"""
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time

CONFIG = json.loads(r'''__LLMFS_CONFIG__''')

WORK = "/kaggle/working"
SRC = "/tmp/llmfs-src"
INPUT = "/kaggle/input"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def find_dir(marker, slug=None):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", marker), recursive=True))
    if slug:
        hits = [h for h in hits if slug in h]
    if not hits:
        raise SystemExit(f"attached datasets contain no {marker}" + (f" for {slug}" if slug else ""))
    return os.path.dirname(hits[0])


def extract_code():
    archive = os.path.join(find_dir("code.tar.pinned", CONFIG.get("code_slug")), "code.tar.pinned")
    digest = sha256_file(archive)
    if digest != CONFIG["code_tar_sha256"]:
        raise SystemExit(f"code archive sha256 {digest} does not match the pinned {CONFIG['code_tar_sha256']}")
    os.makedirs(SRC, exist_ok=True)
    with tarfile.open(archive) as tar:
        root = os.path.realpath(SRC)
        for member in tar.getmembers():
            target = os.path.realpath(os.path.join(SRC, member.name))
            if not (member.isfile() or member.isdir()) or os.path.commonpath([root, target]) != root:
                raise SystemExit(f"unsafe archive member: {member.name}")
        tar.extractall(SRC)
    return digest


def environment():
    info = {"python": platform.python_version(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch
        info["torch"] = str(torch.__version__)
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        info["torch_error"] = repr(exc)
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                             "--format=csv,noheader"], capture_output=True, text=True,
                                            timeout=30).stdout.strip()
    except Exception as exc:
        info["nvidia_smi_error"] = repr(exc)
    return info


def substitute(arg, paths):
    for key, value in paths.items():
        arg = arg.replace("{" + key + "}", value)
    return arg


def main():
    os.makedirs(WORK, exist_ok=True)
    summary = {"config": {k: v for k, v in CONFIG.items() if k != "steps"}, "steps": []}
    summary["environment"] = environment()
    summary["code_tar_sha256"] = extract_code()
    paths = {"work": WORK, "src": SRC}
    if CONFIG.get("data_slug"):
        paths["data"] = find_dir("manifest.json", CONFIG["data_slug"])
    if CONFIG.get("state_slug"):
        paths["state"] = find_dir("ckpt.pt", CONFIG["state_slug"])
        if CONFIG.get("state_ckpt_sha256"):
            digest = sha256_file(os.path.join(paths["state"], "ckpt.pt"))
            if digest != CONFIG["state_ckpt_sha256"]:
                raise SystemExit(f"resume checkpoint sha256 {digest} does not match the pinned value")
            summary["state_ckpt_sha256"] = digest
    for name, (slug, marker) in CONFIG.get("extra_inputs", {}).items():
        paths[name] = find_dir(marker, slug)
    summary["paths"] = paths
    failed = False
    for step in CONFIG["steps"]:
        args = [substitute(a, paths) for a in step["args"]]
        log_path = os.path.join(WORK, f"{step['name']}.log")
        began = time.monotonic()
        with open(log_path, "w") as log:
            proc = subprocess.run([sys.executable, *args], cwd=SRC, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=step.get("timeout", 3600))
        record = {"name": step["name"], "returncode": proc.returncode,
                  "seconds": round(time.monotonic() - began, 1), "log": os.path.basename(log_path)}
        summary["steps"].append(record)
        print(json.dumps(record), flush=True)
        with open(log_path) as fh:
            print("".join(fh.readlines()[-25:]), flush=True)
        if proc.returncode != 0:
            failed = True
            if not step.get("continue_on_error"):
                break
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(WORK, "session-summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if failed:
        raise SystemExit("one or more steps failed; see session-summary.json")


if __name__ == "__main__":
    main()
