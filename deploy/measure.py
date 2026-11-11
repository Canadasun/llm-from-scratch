import argparse
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

PROMPTS = (
    "def parse_config(path):\n    \"\"\"Read a JSON config file and return a dict.\"\"\"\n",
    "import hashlib\n\n\ndef file_digest(path):\n",
    "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n",
    "class LRUCache:\n    def __init__(self, capacity):\n",
    "def safe_join(base, *parts):\n",
)
LIMITS = {"tokens_per_second": 5.0, "peak_memory_bytes": int(1.8 * 1024 ** 3), "cold_start_seconds": 90.0}
UNITS = {"b": 1, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3, "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3}


def _bytes(text):
    match = re.match(r"([\d.]+)\s*([A-Za-z]+)", text.strip())
    return int(float(match.group(1)) * UNITS[match.group(2).lower()]) if match else 0


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post(url, body, timeout=300):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class MemoryWatch(threading.Thread):
    def __init__(self, name):
        super().__init__(daemon=True)
        self.name_, self.peak, self.stop = name, 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", self.name_],
                                 capture_output=True, text=True).stdout
            if out.strip():
                self.peak = max(self.peak, _bytes(out.split("/")[0]))
            time.sleep(0.5)


def measure(image, expect_checkpoint, port=18080, requests=len(PROMPTS), max_new_tokens=128):
    name = f"llmfs-measure-{int(time.time())}"
    began = time.monotonic()
    subprocess.run(["docker", "run", "-d", "--name", name, "--cpus=1", "--memory=2g", "--memory-swap=2g",
                    "--read-only", "--tmpfs=/tmp:rw,nosuid,nodev,size=64m", "--cap-drop=ALL",
                    "--security-opt=no-new-privileges", "--pids-limit=64", "-e", "QUANTIZE=1",
                    "-p", f"127.0.0.1:{port}:8080", image], check=True, capture_output=True)
    watch = MemoryWatch(name)
    watch.start()
    try:
        health = None
        while time.monotonic() - began < 300:
            try:
                health = _get(f"http://127.0.0.1:{port}/healthz")
                break
            except OSError:
                time.sleep(0.25)
        cold = round(time.monotonic() - began, 2)
        if health is None:
            raise RuntimeError("server never became healthy")
        if health.get("checkpoint_sha256") != expect_checkpoint:
            raise RuntimeError(f"served checkpoint {health.get('checkpoint_sha256')} is not {expect_checkpoint}")
        runs = []
        for i in range(requests):
            out = _post(f"http://127.0.0.1:{port}/generate",
                        {"prompt": PROMPTS[i % len(PROMPTS)], "max_new_tokens": max_new_tokens, "seed": i + 1})
            runs.append({"completion_tokens": out["completion_tokens"], "latency_ms": out["latency_ms"]})
            time.sleep(1.1)
        tokens = sum(r["completion_tokens"] for r in runs)
        seconds = sum(r["latency_ms"] for r in runs) / 1000
        cgroup = subprocess.run(["docker", "exec", name, "cat", "/sys/fs/cgroup/memory.peak"],
                                capture_output=True, text=True).stdout.strip()
    finally:
        watch.stop.set()
        watch.join(2)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    peak = int(cgroup) if cgroup.isdigit() else watch.peak
    result = {"image": image, "limits": {"cpus": 1, "memory": "2g"}, "health": health,
              "cold_start_seconds": cold, "requests": runs, "tokens_per_second": round(tokens / seconds, 2),
              "peak_memory_bytes": peak, "peak_memory_source": "cgroup memory.peak" if cgroup.isdigit()
              else "docker stats samples"}
    result["checks"] = {
        "tokens_per_second": result["tokens_per_second"] >= LIMITS["tokens_per_second"],
        "peak_memory_bytes": peak < LIMITS["peak_memory_bytes"],
        "cold_start_seconds": cold < LIMITS["cold_start_seconds"],
    }
    result["thresholds"] = LIMITS
    result["passed"] = all(result["checks"].values())
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--expect-checkpoint", required=True)
    ap.add_argument("--pull-seconds", type=float)
    ap.add_argument("--image-bytes", type=int)
    ap.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}", args.image):
        ap.exit(1, "an immutable repository@sha256 image is required\n")
    result = measure(args.image, args.expect_checkpoint)
    result["pull_seconds"] = args.pull_seconds
    result["image_bytes"] = args.image_bytes
    Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("cold_start_seconds", "tokens_per_second", "peak_memory_bytes",
                                             "checks", "passed")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
