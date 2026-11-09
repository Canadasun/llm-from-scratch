import json

from data import build_sft as S
from data.sft_authored import ABSTAIN, REFUSE, SECURE, authored_rows

MODULE = '''import os
import json as j
from pathlib import Path
from .local import helper

LIMIT = 10


def read_settings(path, default=None):
    """Return the parsed JSON settings stored at the given path, or the default."""
    target = Path(path)
    if not os.path.exists(target):
        return default
    with open(target, encoding="utf-8") as handle:
        return j.load(handle)


def uses_global(value):
    """Clamp the value so it never exceeds the configured module limit."""
    if value > LIMIT:
        return LIMIT
    return value + 0 * len(str(value))


def uses_relative(value):
    """Pass the value through the helper imported from a sibling module."""
    result = helper(value)
    result = result or value
    return result


def short_doc(x):
    """Todo."""
    y = x + 1
    return y * 2 + x
'''


def test_function_candidates_keep_only_self_contained_documented_functions():
    cands = {c["name"]: c for c in S.function_candidates(MODULE)}
    assert set(cands) == {"read_settings"}
    c = cands["read_settings"]
    assert c["imports"] == ["from pathlib import Path", "import json as j", "import os"]
    assert c["signature"] == "read_settings(path, default=None)"
    assert c["summary"].startswith("Return the parsed JSON settings")
    assert '"""' not in c["bare"] and "def read_settings" in c["bare"] and "j.load" in c["bare"]
    compile(c["bare"], "<bare>", "exec")


def test_free_names_handles_scopes():
    import ast
    src = ("def f(a, *rest, k=1, **kw):\n    import re\n    try:\n        pass\n    except E as err:\n"
           "        return err\n    total = [x for x in rest]\n    def g():\n        return f\n    return re, g, total, z\n")
    assert S.free_names(ast.parse(src).body[0]) == {"E", "z"}
    assert S.free_names(ast.parse("def f():\n    global x\n    x = 1\n").body[0]) is None


def test_review_rules_hit_real_patterns_and_skip_safe_ones():
    def rules(src):
        return {c["rule"] for c in S.review_candidates(src)}
    assert rules("import yaml\n\ndef load(p):\n    with open(p) as f:\n        return yaml.load(f)\n") == {"yaml-unsafe-load"}
    assert rules("import yaml\n\ndef load(p):\n    with open(p) as f:\n        return yaml.load(f, Loader=yaml.SafeLoader)\n") == set()
    assert rules("import subprocess\n\ndef run(name):\n    cmd = 1\n    subprocess.call('ls ' + name, shell=True)\n") == {"shell-command-injection"}
    assert rules("import subprocess\n\ndef run(name):\n    cmd = 1\n    subprocess.call(['ls', name])\n") == set()
    assert rules("def q(cur, uid):\n    x = 1\n    cur.execute(f\"SELECT * FROM t WHERE id={uid}\")\n") == {"sql-injection"}
    assert rules("def q(cur, uid):\n    x = 1\n    cur.execute(\"SELECT * FROM t WHERE id=%s\", (uid,))\n") == set()
    assert rules("import requests\n\ndef get(u):\n    x = 1\n    return requests.get(u, verify=False)\n") == {"tls-verification-disabled"}
    assert rules("import random\n\ndef make_token():\n    x = 1\n    return random.choice('abc')\n") == {"insecure-random-secret"}
    assert rules("import random\n\ndef shuffle_deck():\n    x = 1\n    return random.choice('abc')\n") == set()
    assert rules("import tarfile\n\ndef unpack(p, d):\n    with tarfile.open(p) as t:\n        t.extractall(d)\n") == {"tar-path-traversal"}
    assert rules("import tarfile\n\ndef unpack(p, d):\n    with tarfile.open(p) as t:\n        t.extractall(d, filter='data')\n") == set()
    assert rules("def drop(cur, table):\n    x = 1\n    cur.execute(\"DELETE FROM \" + table)\n") == set()
    assert rules("import hashlib\n\ndef pwned(password):\n    h = hashlib.sha1(password.encode()).hexdigest()\n"
                 "    return 'https://api.pwnedpasswords.com/range/' + h[:5]\n") == set()
    assert rules("import hashlib\n\ndef store(password):\n    x = 1\n    return hashlib.md5(password.encode())\n") == {"fast-hash-for-password"}
    assert rules("def ev(s):\n    x = 1\n    return eval(s)\n") == {"eval-of-dynamic-input"}


def test_review_row_quotes_line_and_scopes_claim():
    src = "import pickle\n\nclass C:\n    def load(self, blob):\n        data = blob.strip()\n        return pickle.loads(data)\n"
    (cand,) = S.review_candidates(src)
    assert cand["source"].startswith("def load(self, blob):") and cand["line_number"] == 3
    _, prompt, response = S.review_row(cand, "0" * 64)
    assert "pickle.loads(data)" in response and S.SCOPE_NOTE in response and "def load" in prompt


def test_authored_rows_are_distinct_and_split_by_family():
    items = SECURE + REFUSE + ABSTAIN
    assert len({f for f, _, _ in items}) == len(items)
    assert len({" ".join(p.split()) for _, p, _ in items}) == len(items)
    rows = authored_rows(0.15)
    assert 0 < len(rows["val"]) < len(rows["train"])
    assert not {r["family"] for r in rows["train"]} & {r["family"] for r in rows["val"]}


def test_build_end_to_end_inherits_repo_splits(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for split, name in S.SPLIT_FILES.items():
        lines = []
        for i in range(6):
            text = MODULE.replace("read_settings", f"read_settings_{split}_{i}").replace(
                "settings stored", f"settings number {i} stored in {split}") + (
                f"\n\nimport yaml\n\ndef load_{i}(p):\n    with open(p) as f:\n        return yaml.load(f)\n")
            text = text.replace("j.load(handle)", f"j.load(handle) or {{'n': {i}, 's': '{split}'}}")
            lines.append(json.dumps({"text": text, "source": "synthetic", "group": f"{split}/repo{i}",
                                     "licences": ["MIT"]}))
        (corpus / name).write_text("\n".join(lines) + "\n")
    out = tmp_path / "out"
    S.main(["--corpus", str(corpus), "--out", str(out), "--sample-mod", "1", "--workers", "1",
            "--eval-dir", str(tmp_path / "none")])
    manifest = json.loads((out / "manifest.json").read_text())
    groups = {}
    for split, name in S.OUT_FILES.items():
        for line in (out / name).read_text().splitlines():
            row = json.loads(line)
            assert row["prompt"] and row["response"]
            groups.setdefault(row["group"], set()).add(split)
            if row["source"] == "synthetic":
                assert row["group"].startswith(split + "/")
    assert all(len(s) == 1 for s in groups.values())
    assert manifest["review_rules"].get("yaml-unsafe-load", 0) >= 1
    assert manifest["rows"]["train"] > 0
