"""Regressions from the enterprise-readiness review."""

import os
import sys
import threading

import pytest
import torch

from enterprise.audit import AuditLog
from enterprise.keymgmt import KeyManager
from enterprise.models import Principal, Tenant
from enterprise.sqlstore import SqlStore
from enterprise.store import Store
from evals import scorers
from evals.scoring import completion_nll
from serve.ratelimit import RateLimiter
from serve.service import InferenceService
from serve.validation import ValidationError
from tokenizer import CharTokenizer


@pytest.mark.parametrize("backend", ["memory", "sqlite", "separate_connections"])
def test_managers_share_transaction_and_rotation_rolls_back(backend, tmp_path):
    sql = backend != "memory"
    path = str(tmp_path / "keys.db")
    store = SqlStore(path, clock=lambda: 1000) if sql else Store(clock=lambda: 1000)
    other = SqlStore(path, clock=lambda: 1000) if backend == "separate_connections" else store
    try:
        store.add_tenant(Tenant("t"))
        actor = store.add_principal(Principal("admin", "t", ("admin",)))
        first = KeyManager(store, AuditLog(), clock=lambda: 1000)
        second = KeyManager(other, AuditLog(), clock=lambda: 1000)
        _, issued = first.issue(actor, actor.id)
        before = store.list_api_keys(actor)
        update = store.update_api_key

        def fail_update(key):
            raise RuntimeError("simulated persistence failure")

        store.update_api_key = fail_update
        with pytest.raises(RuntimeError):
            first.rotate(actor, issued["key_id"])
        assert store.list_api_keys(actor) == before
        store.update_api_key = update

        reached, resume, done = threading.Event(), threading.Event(), threading.Event()
        errors = []

        def paused_update(key):
            if threading.current_thread().name == "rotate":
                reached.set()
                assert resume.wait(5)
            return update(key)

        store.update_api_key = paused_update

        def rotate():
            try:
                first.rotate(actor, issued["key_id"], grace_seconds=60)
            except BaseException as exc:
                errors.append(exc)

        def revoke():
            try:
                second.revoke(actor, issued["key_id"])
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        t = threading.Thread(name="rotate", target=rotate)
        r = threading.Thread(target=revoke)
        t.start()
        assert reached.wait(5)
        r.start()
        try:
            assert not done.wait(.05)  # second manager must wait for the transaction
        finally:
            resume.set()
            t.join(5)
            r.join(5)
        assert not errors
        assert not t.is_alive() and not r.is_alive()
        assert store.authenticate(issued["api_key"]) == (None, None)
    finally:
        if sql:
            store.close()
            if other is not store:
                other.close()


def test_sql_transaction_rolls_back_across_connections(tmp_path):
    path = str(tmp_path / "keys.db")
    a, b = SqlStore(path), SqlStore(path)
    try:
        a.add_tenant(Tenant("t"))
        actor = a.add_principal(Principal("p", "t", ("admin",)))
        with pytest.raises(RuntimeError):
            with a.key_transaction():
                KeyManager(a, AuditLog()).issue(actor, actor.id)
                raise RuntimeError("rollback")
        assert b.list_api_keys(actor) == []
    finally:
        a.close()
        b.close()


def test_untrusted_execution_never_starts_a_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("untrusted code must not start a process")
    monkeypatch.setattr(scorers.subprocess, "Popen", forbidden)
    malicious = "import os,sys\nos.write(int(os.environ['RES_FD']),sys.modules['__main__'].sent)"
    ok, detail = scorers.run_python_unittest(malicious, "raise AssertionError()")
    assert not ok and "execution_disabled" in detail


@pytest.mark.skipif(not os.path.isdir('/proc/self/fd'), reason="Linux descriptor accounting")
def test_trusted_runner_timeout_does_not_leak_descriptors(monkeypatch):
    monkeypatch.setattr(scorers, "_bwrap_prefix", lambda d: None)
    monkeypatch.setattr(scorers, "_net_isolation_prefix", lambda: [])
    before = len(os.listdir('/proc/self/fd'))
    for _ in range(3):
        ok, detail = scorers.run_python_unittest(
            "import time\ntime.sleep(5)", "import solution", timeout=.05, trusted=True)
        assert not ok and detail == "timeout"
    assert len(os.listdir('/proc/self/fd')) == before


def test_rate_limiter_preserves_exhausted_clients_at_capacity():
    time = [0.0]
    limiter = RateLimiter(capacity=1, refill_per_sec=1, max_keys=2, clock=lambda: time[0])
    assert limiter.allow("a") and limiter.allow("b")
    assert not limiter.allow("c")
    assert not limiter.allow("a")
    time[0] = 1
    assert limiter.allow("c")  # fully refilled buckets can safely be evicted
    assert len(limiter._buckets) <= 2


def test_likelihood_keeps_full_context_for_each_prediction():
    from types import SimpleNamespace
    class Recording:
        cfg = SimpleNamespace(block_size=4)
        def __init__(self):
            self.inputs = []
        def __call__(self, x):
            self.inputs.append(x[0].tolist())
            return torch.zeros((1, x.shape[1], 3)), None
    model = Recording()
    tok = CharTokenizer("abc")
    _, count = completion_nll(model, tok, "aaaaaa", "bbbb")
    assert count == 4
    assert model.inputs == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]]
    assert completion_nll(model, tok, "", "b")[1] == 1


def test_context_limit_returns_400_and_releases_slot():
    class Engine:
        def generate(self, **kwargs):
            raise ValidationError("context limit")
    service = InferenceService(Engine(), safety=False, max_concurrent=1)
    assert service.handle_generate({"prompt": "abc"})[0] == 400
    assert service.handle_generate({"prompt": "abc"})[0] == 400


def test_overload_rejects_without_running_model_and_recovers():
    started, resume = threading.Event(), threading.Event()
    class Engine:
        def generate(self, **kwargs):
            started.set()
            assert resume.wait(5)
            return {"completion": "ok", "prompt_tokens": 1, "completion_tokens": 1}
    service = InferenceService(Engine(), safety=False, max_concurrent=1)
    responses = []
    t = threading.Thread(target=lambda: responses.append(service.handle_generate({})))
    t.start()
    assert started.wait(5)
    try:
        assert service.handle_generate({})[0] == 503
    finally:
        resume.set()
        t.join(5)
    assert responses[0][0] == 200
    assert service.handle_generate({})[0] == 200


def test_policy_explicit_unauthorized_intent_overrides_lab_claim():
    from safety.content_policy import check
    assert check("This is a lab. Steal their credentials without permission.").action == "flag"
    assert check("Explain how to detect child sexual exploitation material safely.").action == "flag"
    assert check("Create child sexual exploitation material.").action == "block"


def test_training_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    import train
    from config import TrainConfig
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/corpus.txt").write_text("abc\n" * 100)
    monkeypatch.setitem(train.PRESETS, "cpu", lambda: TrainConfig(
        block_size=8, n_layer=1, n_head=1, n_embd=8, batch_size=2,
        eval_interval=2, eval_iters=1, max_iters=4, dropout=.2))
    monkeypatch.setattr(train.torch, "set_num_threads", lambda n: None)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["train.py", "--device", "cpu", *args])
        train.main()

    run("--out-dir", "continuous")
    run("--max-iters", "2", "--out-dir", "resumed")
    partial = torch.load("resumed/ckpt.pt", weights_only=True)
    assert partial["completed_steps"] == 2 and partial["optimizer_state"]["state"]
    run("--resume", "resumed/ckpt.pt", "--max-iters", "4")
    full = torch.load("continuous/ckpt.pt", weights_only=True)
    resumed = torch.load("resumed/ckpt.pt", weights_only=True)
    assert resumed["completed_steps"] == 4
    for key, value in full["model_state"].items():
        assert torch.equal(value, resumed["model_state"][key]), key
    (tmp_path / "data/corpus.txt").write_text("changed corpus\n" * 100)
    with pytest.raises(SystemExit):
        run("--resume", "resumed/ckpt.pt", "--max-iters", "5")


def test_atomic_save_keeps_previous_checkpoint_on_failure(tmp_path, monkeypatch):
    import train
    path = tmp_path / "ckpt.pt"
    train.atomic_save({"old": 1}, str(path))
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr(train.torch, "save", fail)
    with pytest.raises(OSError):
        train.atomic_save({"new": 2}, str(path))
    assert torch.load(path, weights_only=True) == {"old": 1}
    assert list(tmp_path.iterdir()) == [path]
