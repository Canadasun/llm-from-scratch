"""Portable fingerprints for reproducible model evaluations."""

import hashlib
import json
from pathlib import Path

SCORING_VERSION = "context-preserving-v2"
CODE_EVALUATOR = "function-json-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def suite_sha256(tasks):
    # Includes order, generation settings, answers, and reviewed code cases.
    manifest = Path(__file__).with_name("data") / "code_cases.json"
    payload = {"tasks": tasks, "code_cases": file_sha256(manifest)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()
