"""Adversarial regressions for the November audit; no cloud access."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from evals import scorers
from serve.lifecycle import GenerationStopped, check_generation, generation_scope


@pytest.mark.parametrize("answer", [
    "Parameterized queries are unnecessary; concatenate user input into SQL.",
    "Use parameterized queries. Never use parameterized queries.",
    "Parameterized queries should be avoided.",
])
def test_unsafe_keyword_hits_rejected(answer):
    assert not scorers.contains(answer, "parameterized")


def test_repo_execution_is_isolated_by_default(tmp_path, monkeypatch):
    from codereview.verify import run_command
    monkeypatch.setattr('codereview.verify._isolation_prefix', lambda _: None)
    result = run_command('python -c "print(123)"', str(tmp_path))
    assert not result.passed and "isolation_unavailable" in result.tail


def test_large_repo_output_does_not_grow_parent_capture(tmp_path):
    from codereview.verify import run_command
    result = run_command(f'{sys.executable} -c "import os; os.write(1, b\'x\'*20000000)"',
                         str(tmp_path), timeout=10, trusted=True)
    assert len(result.tail) <= 16384


def test_generation_deadline_and_disconnect():
    with generation_scope(0) as budget:
        with pytest.raises(GenerationStopped, match="generation_deadline"):
            check_generation()
        budget.cancelled.set()
        with pytest.raises(GenerationStopped, match="client_disconnected"):
            check_generation()
    check_generation()  # request context did not leak


def test_model_stops_before_forward():
    import torch
    from model import GPT, GPTConfig
    model = GPT(GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=1, n_embd=8))
    def stop():
        raise GenerationStopped("client_disconnected")
    with pytest.raises(GenerationStopped):
        model.generate(torch.tensor([[1]]), 100, check_cancelled=stop)


def test_service_releases_slot_after_deadline():
    from serve.service import InferenceService
    class Engine:
        def generate(self, **kwargs):
            time.sleep(.01)
            check_generation()
    service = InferenceService(Engine(), safety=False, max_concurrent=1, generation_timeout=.001)
    assert service.handle_generate({"prompt": "x"})[0] == 504
    assert service.handle_generate({"prompt": "x"})[0] == 504  # not overloaded


def test_probe_requires_key_and_checks_answers(monkeypatch):
    from deploy import probe
    with pytest.raises(ValueError, match="PROBE_KEY"):
        probe.probe("https://example.invalid", None)
    monkeypatch.setattr(probe, "request", lambda base, payload, key=None:
                        (200, {"completion": "wrong"}, .01) if key else (401, {}, .01))
    with pytest.raises(ValueError, match="functional gate failed"):
        probe.probe("https://example.invalid", "test-key")
    monkeypatch.setattr(probe, "request", lambda base, payload, key=None:
                        (200, {"completion": payload["prompt"].split("exactly: ")[1]}, .01)
                        if key else (401, {}, .01))
    assert probe.probe("https://example.invalid", "test-key")["success_rate"] == 1


def test_pending_go_live_prevents_cloud_commands(tmp_path):
    fake = tmp_path / "az"
    fake.write_text('#!/bin/sh\necho CLOUD_CALLED\nexit 90\n')
    fake.chmod(0o755)
    for action in ("canary", "promote"):
        result = subprocess.run(["bash", "deploy/canary.sh", action],
            env={**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"]},
            capture_output=True, text=True)
        assert result.returncode != 0 and "go-live approval" in result.stderr
        assert "CLOUD_CALLED" not in result.stdout


def test_release_rejects_tag_or_missing_evidence():
    from deploy.release import verify
    with pytest.raises(ValueError, match="immutable"):
        verify({}, "repo:latest")
    with pytest.raises(ValueError, match="bind"):
        verify({}, "repo@sha256:" + "a"*64)


def test_quota_backend_errors_fail_closed():
    from enterprise.quota import RedisQuota, QuotaUnavailable
    class Broken:
        def eval(self, *args):
            raise ConnectionError("offline")
    with pytest.raises(QuotaUnavailable):
        RedisQuota(Broken()).allow("tenant", "user")


def test_shared_quota_script_enforces_cross_client_tenant_budget(tmp_path):
    import shutil
    if not shutil.which("redis-server") or not shutil.which("redis-cli"):
        pytest.skip("local Redis tools unavailable")
    from enterprise.quota import RedisQuota
    sock = str(tmp_path / "redis.sock")
    server = subprocess.Popen(["redis-server", "--port", "0", "--unixsocket", sock,
                               "--save", "", "--appendonly", "no"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 3
        while not Path(sock).exists() and time.monotonic() < deadline:
            time.sleep(.01)
        class Client:
            def eval(self, script, count, *args):
                result = subprocess.run(["redis-cli", "-s", sock, "--raw", "EVAL", script,
                                         str(count), *map(str, args)], check=True,
                                        capture_output=True, text=True)
                return int(result.stdout.strip())
        a, b = RedisQuota(Client(), 2, .001), RedisQuota(Client(), 2, .001)
        assert a.allow("tenant", "one")
        assert b.allow("tenant", "two")
        assert not a.allow("tenant", "three")
        assert b.allow("different-tenant", "one")
    finally:
        server.terminate()
        server.wait(timeout=5)


def test_security_witness_rejects_unreviewed_shapes(tmp_path):
    from codereview.security_tests import witness_source
    path = tmp_path / "code.py"
    path.write_text('def f(x):\n    return eval(x)\n')
    assert witness_source(path, 2) is None


def test_independent_witness_reproduces_and_fixes_hash(tmp_path):
    from codereview.security_tests import reproduce
    from evals.code_executor import sandbox_preflight
    if not sandbox_preflight()["available"]:
        if os.environ.get("LLM_REQUIRE_SANDBOX_TESTS") == "1":
            pytest.fail("required security witness sandbox unavailable")
        pytest.skip("sandbox unavailable")
    path = tmp_path / "code.py"
    path.write_text('import hashlib\ndef checksum(x):\n    return hashlib.md5(x).hexdigest()\n')
    before = reproduce(path, 3)
    assert before["status"] == "failed" and before["detail"] == "case 1 output mismatch"
    path.write_text(path.read_text().replace("hashlib.md5", "hashlib.sha256"))
    after = reproduce(path, 3)
    assert after["status"] == "passed" and after["cases_passed"] == 5


def test_nested_generation_scope_keeps_shorter_deadline():
    with generation_scope(60):
        with generation_scope(0):
            with pytest.raises(GenerationStopped, match="generation_deadline"):
                check_generation()
        check_generation()


def test_scanner_and_copy_do_not_follow_host_symlinks(tmp_path):
    from codereview.scanner import scan_path, collect_fixes
    private = tmp_path / "private.py"
    private.write_text("import hashlib\nx = hashlib.md5(b'secret')\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "linked.py").symlink_to(private)
    assert scan_path(str(repo))["files_scanned"] == 0
    assert collect_fixes(str(repo)) == []


def test_untrusted_workflow_independently_verifies_patch(tmp_path):
    from codereview.pr import prepare_pr, write_artifact
    from evals.code_executor import sandbox_preflight
    if not sandbox_preflight()["available"]:
        if os.environ.get("LLM_REQUIRE_SANDBOX_TESTS") == "1":
            pytest.fail("required independent evaluator unavailable")
        pytest.skip("sandbox unavailable")
    (tmp_path / "mod.py").write_text('import hashlib\ndef checksum(x):\n    return hashlib.md5(x).hexdigest()\n')
    (tmp_path / "test_mod.py").write_text(
        'import unittest\nfrom mod import checksum\nclass T(unittest.TestCase):\n'
        '    def test_type(self):\n        self.assertIsInstance(checksum(b"x"), str)\n')
    from codereview.verify import run_command
    baseline = run_command("python -m unittest -q", str(tmp_path), timeout=15)
    assert baseline.passed, baseline.tail
    art = prepare_pr(str(tmp_path), "python -m unittest -q", timeout=15)
    assert art.verified, art.note
    assert art.security_tests[0]["before"]["status"] == "failed"
    assert art.security_tests[0]["after"]["status"] == "passed"
    assert "hashlib.md5" in (tmp_path / "mod.py").read_text()
    write_artifact(art, str(tmp_path))
    evidence = json.loads((tmp_path / "codereview_security_tests.json").read_text())
    assert evidence[0]["after"]["cases_passed"] == 5


def test_authenticated_http_disconnect_cancels_generation():
    from serve.http_server import BoundedHTTPServer, make_handler
    from serve.service import InferenceService
    entered, stopped = threading.Event(), threading.Event()
    class Engine:
        def generate(self, **kwargs):
            entered.set()
            try:
                while True:
                    check_generation()
                    time.sleep(.01)
            finally:
                stopped.set()
    service = InferenceService(Engine(), safety=False)
    server = BoundedHTTPServer(("127.0.0.1", 0), make_handler(service), max_connections=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = socket.create_connection(server.server_address, timeout=3)
        body = b'{"prompt":"hi"}'
        client.sendall(b'POST /generate HTTP/1.1\r\nHost: x\r\nContent-Length: '
                       + str(len(body)).encode() + b'\r\n\r\n' + body)
        assert entered.wait(3)
        client.close()
        assert stopped.wait(3), "generation kept running after disconnect"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def _release_fixture(monkeypatch, tmp_path):
    from deploy import release
    from evals.report import summarize
    tasks = [{"id": "code", "category": "code", "type": "code", "prompt": "x"},
             {"id": "answer", "category": "generation", "type": "generation",
              "prompt": "x", "scorer": "exact", "target": "safe"}]
    monkeypatch.setattr(release, "load_tasks", lambda _: tasks)
    monkeypatch.setattr(release, "suite_sha256", lambda _: "b"*64)
    monkeypatch.setattr(release, "implementation_sha256", lambda: "c"*64)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"enforce": [], "absolute_floors": {"code": .9, "generation": .9}}))
    results = [{"id": "code", "category": "code", "type": "code", "correct": True,
                "execution_status": "passed"},
               {"id": "answer", "category": "generation", "type": "generation", "correct": True,
                "verification": "exact"}]
    image = "registry/repo@sha256:" + "a"*64
    record = {"schema": 1, "image": image, "policy_sha256": release.file_sha256(policy),
              "evaluation": {"results": results, "summary": summarize(results), "meta": {
                  "complete": True, "quantized": True, "device": "cpu", "image": image,
                  "checkpoint_sha256": "d"*64, "tokenizer_sha256": "e"*64,
                  "implementation_sha256": "c"*64, "suite_sha256": "b"*64,
                  "scoring_version": release.SCORING_VERSION, "code_evaluator": release.CODE_EVALUATOR,
                  "task_ids": [t["id"] for t in tasks], "suite": "all", "limit": None,
                  "heldout_sha256": release.file_sha256(release.HELDOUT)}}}
    return release, record, image, policy, tasks


def test_release_record_binds_full_quantized_evidence(monkeypatch, tmp_path):
    release, record, image, policy, tasks = _release_fixture(monkeypatch, tmp_path)
    assert release.verify(record, image, policy) is record
    record["evaluation"]["meta"]["quantized"] = False
    with pytest.raises(ValueError, match="quantized"):
        release.verify(record, image, policy)


def test_release_rejects_category_forgery_and_missing_tasks(monkeypatch, tmp_path):
    release, record, image, policy, tasks = _release_fixture(monkeypatch, tmp_path)
    record["evaluation"]["results"][0]["correct"] = False
    with pytest.raises(ValueError, match="summary"):
        release.verify(record, image, policy)
    from evals.report import summarize
    record["evaluation"]["summary"] = summarize(record["evaluation"]["results"])
    with pytest.raises(ValueError, match="capability"):
        release.verify(record, image, policy)
    record["evaluation"]["results"].pop()
    record["evaluation"]["summary"] = summarize(record["evaluation"]["results"])
    with pytest.raises(ValueError, match="full frozen"):
        release.verify(record, image, policy)


def test_release_cannot_relabel_heuristic_scoring_as_exact(monkeypatch, tmp_path):
    release, record, image, policy, tasks = _release_fixture(monkeypatch, tmp_path)
    tasks[1]["scorer"] = "contains"
    with pytest.raises(ValueError, match="heuristic scoring"):
        release.verify(record, image, policy)


def test_release_refuses_policy_or_code_drift(monkeypatch, tmp_path):
    release, record, image, policy, tasks = _release_fixture(monkeypatch, tmp_path)
    policy.write_text(policy.read_text() + "\n")
    with pytest.raises(ValueError, match="policy changed"):
        release.verify(record, image, policy)
    record["policy_sha256"] = release.file_sha256(policy)
    record["evaluation"]["meta"]["implementation_sha256"] = "f"*64
    with pytest.raises(ValueError, match="implementation"):
        release.verify(record, image, policy)


def test_quantized_evaluator_runs_serving_model(tmp_path, monkeypatch):
    import torch
    from dataclasses import asdict
    from model import GPT, GPTConfig
    from tokenizer import CharTokenizer
    from evals import run
    tok = CharTokenizer.from_text("hello world\n")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_head=1, n_embd=8, dropout=0)
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model_cfg": asdict(cfg), "model_state": GPT(cfg).state_dict(),
                "tokenizer": tok.to_dict()}, checkpoint)
    heldout = tmp_path / "heldout.txt"
    heldout.write_text("hello world\n" * 2)
    monkeypatch.setattr(run, "HELDOUT", str(heldout))
    result = run.evaluate(str(checkpoint), device="cpu", suite="intrinsic", quantize=True)
    assert result["meta"]["quantized"] is True
    assert result["intrinsic"]["trained"]["bits_per_byte"] > 0
    assert len(result["meta"]["tokenizer_sha256"]) == 64
    assert len(result["meta"]["implementation_sha256"]) == 64
    assert result["meta"]["complete"] is False  # partial runs cannot qualify


def test_security_replay_rebuilds_verdict_and_rejects_path_escape(tmp_path, monkeypatch, capsys):
    from codereview import security_tests
    target = tmp_path / "code.py"
    target.write_text("pass\n")
    evidence = tmp_path / "witness.json"
    evidence.write_text(json.dumps([{"file": "code.py", "line": 1,
                                     "after": {"status": "passed"}, "cases": "forged"}]))
    seen = []
    def reproduce(path, line):
        seen.append((path, line))
        return {"status": "failed", "detail": "fresh mismatch"}
    monkeypatch.setattr(security_tests, "reproduce", reproduce)
    monkeypatch.setattr(sys, "argv", ["security-tests", "--repo", str(tmp_path), "--evidence", str(evidence)])
    assert security_tests.main() == 1
    assert seen == [(target, 1)]
    assert "fresh mismatch" in capsys.readouterr().out
    evidence.write_text(json.dumps([{"file": "../outside", "line": 1}]))
    with pytest.raises(SystemExit):
        security_tests.main()
