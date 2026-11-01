"""
Scanner coverage for import-aliased and directly-imported dangerous calls
(audit: reproduced missed findings for aliased subprocess.run(..., shell=True)
and directly imported pickle.loads(...)).
"""

from codereview.rules import analyze_source


def _rule_ids(src):
    return {f.rule_id for f in analyze_source(src)}


def test_aliased_subprocess_module_shell_true_is_flagged():
    src = "import subprocess as sp\nsp.run(cmd, shell=True)\n"
    assert "PY-SHELL-INJECTION" in _rule_ids(src)


def test_from_import_aliased_run_shell_true_is_flagged():
    src = "from subprocess import run as r\nr(cmd, shell=True)\n"
    assert "PY-SHELL-INJECTION" in _rule_ids(src)


def test_directly_imported_pickle_loads_is_flagged():
    src = "from pickle import loads\nobj = loads(data)\n"
    assert "PY-INSECURE-DESERIALIZATION" in _rule_ids(src)


def test_aliased_pickle_module_loads_is_flagged():
    src = "import pickle as p\np.loads(data)\n"
    assert "PY-INSECURE-DESERIALIZATION" in _rule_ids(src)


def test_from_import_aliased_eval_is_flagged():
    src = "from builtins import eval as e\ne(user_input)\n"
    assert "PY-DANGEROUS-EVAL" in _rule_ids(src)


def test_unrelated_run_is_not_a_false_positive():
    # A local run() unrelated to subprocess must not be flagged.
    src = "def run(x, shell=True):\n    return x\nrun(cmd, shell=True)\n"
    assert "PY-SHELL-INJECTION" not in _rule_ids(src)


def test_plain_subprocess_still_flagged_and_safe_still_clean():
    assert "PY-SHELL-INJECTION" in _rule_ids("import subprocess\nsubprocess.run(cmd, shell=True)\n")
    assert analyze_source("import subprocess\nsubprocess.run(['ls'])\n") == []
