"""Build the Phase 2 instruction set from the Phase 2 corpus.

    python -m data.build_sft --corpus ~/llmfs-data/corpus-v2 --out ~/llmfs-data/sft-v1

Rows come from three places:
  - real, self-contained functions with docstrings from licence-allowlisted
    files (implement-from-spec and summarise tasks), keeping repo and licence;
  - real functions that match a narrow insecure-pattern rule, answered with a
    finding that quotes the offending line and names the fix;
  - project-authored secure-coding, refusal and missing-information rows in
    data/sft_authored.py.

Mined rows inherit the corpus repository split, so no repository appears in
more than one split. Every row is screened against evaluation strings.
Source files are parsed with ast only; nothing is executed.
"""

import argparse
import ast
import builtins
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from data import pipeline as P  # noqa: E402
from data.sft_authored import authored_rows  # noqa: E402

BUILTINS = frozenset(dir(builtins)) | {"__name__", "__file__", "__doc__"}
SPLIT_FILES = {"train": "train.jsonl", "val": "val.jsonl", "test": "holdout_test.jsonl"}
OUT_FILES = {"train": "train.jsonl", "val": "validation.jsonl", "test": "holdout_test.jsonl"}
MAX_FUNCTION_CHARS = 2000
MAX_LINE = 120
_BAD_DOC = re.compile(r"\b(?:todo|fixme|xxx|deprecated|docstring|placeholder|hack)\b|^\s*[:>@]|https?://",
                      re.IGNORECASE)


def _hash(*parts):
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8", errors="replace")).hexdigest()


def _import_bindings(tree):
    bindings = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                bindings[bound] = ast.unparse(ast.Import(names=[alias]))
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module and node.module != "__future__":
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = ast.unparse(
                        ast.ImportFrom(module=node.module, names=[alias], level=0))
    return bindings


def free_names(fn):
    stored, loaded = {fn.name}, set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return None
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else stored).add(node.id)
        elif isinstance(node, ast.arg):
            stored.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stored.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            stored.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            stored.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            stored.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            stored.add(node.rest)
    return loaded - stored - BUILTINS


def _lines_ok(source):
    return len(source) <= MAX_FUNCTION_CHARS and all(len(line) <= MAX_LINE for line in source.splitlines())


def _signature(fn):
    returns = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    prefix = "async " if isinstance(fn, ast.AsyncFunctionDef) else ""
    return f"{prefix}{fn.name}({ast.unparse(fn.args)}){returns}"


def _doc_summary(doc):
    first = re.split(r"\n\s*\n", doc.strip(), maxsplit=1)[0]
    first = " ".join(first.split())
    if not 25 <= len(first) <= 300 or len(first.split()) < 5 or _BAD_DOC.search(first):
        return None
    if not first[0].isalpha() or sum(c.isascii() for c in first) / len(first) < 0.97:
        return None
    if re.search(r"\s:|^\w+\s*:|[{}\[\]=<>|]", first) or len(first.split()) < 6:
        return None
    first = first[0].upper() + first[1:]
    return first if first.endswith((".", "!", "?")) else first + "."


def function_candidates(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    imports = _import_bindings(tree)
    lines = text.splitlines(keepends=True)
    out = []
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.decorator_list:
            continue
        if fn.name.startswith(("test", "__")) or fn.name in {"main", "setup", "teardown"}:
            continue
        doc_node = fn.body[0] if fn.body else None
        doc = ast.get_docstring(fn, clean=True)
        if not doc or len(fn.body) < 2 or doc_node.lineno <= fn.lineno:
            continue
        summary = _doc_summary(doc)
        if summary is None:
            continue
        free = free_names(fn)
        if free is None or not free <= imports.keys():
            continue
        source = "".join(lines[fn.lineno - 1:fn.end_lineno]).rstrip() + "\n"
        start, stop = doc_node.lineno - fn.lineno, doc_node.end_lineno - fn.lineno + 1
        fn_lines = source.splitlines(keepends=True)
        bare = "".join(fn_lines[:start] + fn_lines[stop:])
        body_lines = [ln for ln in fn_lines[stop:] if ln.strip() and not ln.strip().startswith("#")]
        if not 3 <= len(body_lines) <= 40 or not _lines_ok(source) or "[REDACTED_" in source:
            continue
        if sum(1 for _ in ast.walk(fn)) < 20:
            continue
        try:
            ast.parse(bare)
        except SyntaxError:
            continue
        needed = sorted(imports[n] for n in free)
        header = "".join(fn_lines[:start]).rstrip()
        out.append({"name": fn.name, "signature": _signature(fn), "summary": summary, "imports": needed,
                    "source": source, "bare": bare, "header": header})
    return out


def _call_name(node):
    func = node.func
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _kw(node, name):
    return next((k.value for k in node.keywords if k.arg == name), None)


def _is_const_str(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))


def _dynamic_str(node):
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return _is_const_str(node.left) or _dynamic_str(node.left) or _is_const_str(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return _is_const_str(node.func.value)
    return False


_SECRET_NAME = re.compile(r"(?:token|secret|passw(?:or)?d|salt|nonce|otp|api_?key|session_?id|reset_?code)",
                          re.IGNORECASE)
_PASSWORD_NAME = re.compile(r"passw(?:or)?d|pwd|passwd", re.IGNORECASE)


def _rule_yaml(call, fn):
    if _call_name(call) != "yaml.load":
        return None
    loader = _kw(call, "Loader")
    if loader is None and len(call.args) > 1:
        loader = call.args[1]
    if loader is not None and ast.unparse(loader).split(".")[-1] in {"SafeLoader", "CSafeLoader", "BaseLoader"}:
        return None
    return "yaml-unsafe-load"


def _rule_shell(call, fn):
    name = _call_name(call)
    if name in {"os.system", "os.popen"} and call.args and not _is_const_str(call.args[0]):
        return "shell-command-injection"
    if name.startswith("subprocess.") and call.args:
        shell = _kw(call, "shell")
        if isinstance(shell, ast.Constant) and shell.value is True and (
                _dynamic_str(call.args[0]) or isinstance(call.args[0], ast.Name)):
            return "shell-command-injection"
    return None


def _rule_pickle(call, fn):
    if _call_name(call) in {"pickle.loads", "pickle.load", "cPickle.loads", "cPickle.load", "marshal.loads",
                            "dill.loads", "dill.load"}:
        return "unsafe-deserialization"
    return None


def _rule_sql(call, fn):
    if isinstance(call.func, ast.Attribute) and call.func.attr in {"execute", "executemany"} and call.args:
        prefix = _string_prefix(call.args[0])
        if (_dynamic_str(call.args[0]) and re.match(r"\s*(?:select|insert|update|delete)\b", prefix, re.IGNORECASE)
                and not re.search(r"\b(?:from|into|update|table|join)\s*[`\"']?$", prefix.rstrip(), re.IGNORECASE)):
            return "sql-injection"
    return None


def _string_prefix(node):
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant))[:40]
    if isinstance(node, ast.BinOp):
        return _string_prefix(node.left)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return _string_prefix(node.func.value)
    if _is_const_str(node):
        return node.value if isinstance(node.value, str) else ""
    return ""


def _rule_tls(call, fn):
    verify = _kw(call, "verify")
    name = _call_name(call)
    if (name.split(".")[-1] in {"get", "post", "put", "patch", "delete", "head", "request", "Client",
                                "AsyncClient"} and isinstance(verify, ast.Constant) and verify.value is False):
        return "tls-verification-disabled"
    if name == "ssl._create_unverified_context":
        return "tls-verification-disabled"
    return None


def _rule_weak_hash(call, fn):
    if _call_name(call) in {"hashlib.md5", "hashlib.sha1", "hashlib.sha256"} and call.args:
        arg = ast.unparse(call.args[0])
        if _PASSWORD_NAME.search(arg) and not re.search(r"pwnedpasswords|range/|k-?anonymity", ast.unparse(fn),
                                                         re.IGNORECASE):
            return "fast-hash-for-password"
    return None


def _rule_random_secret(call, fn):
    name = _call_name(call)
    if not name.startswith("random.") or name.split(".")[-1] not in {"choice", "choices", "randint", "random",
                                                                     "getrandbits", "sample", "randrange"}:
        return None
    if _SECRET_NAME.search(fn.name):
        return "insecure-random-secret"
    return None


def _rule_mktemp(call, fn):
    return "insecure-temp-file" if _call_name(call) == "tempfile.mktemp" else None


def _rule_eval(call, fn):
    if isinstance(call.func, ast.Name) and call.func.id in {"eval", "exec"} and call.args:
        if not _is_const_str(call.args[0]):
            return "eval-of-dynamic-input"
    return None


def _rule_tar(call, fn):
    if isinstance(call.func, ast.Attribute) and call.func.attr == "extractall" and _kw(call, "filter") is None:
        if re.search(r"\btarfile\b|\.tar\b|tar\.open|TarFile", ast.unparse(fn)):
            return "tar-path-traversal"
    return None


def _rule_debug(call, fn):
    debug = _kw(call, "debug")
    if (isinstance(call.func, ast.Attribute) and call.func.attr == "run" and isinstance(debug, ast.Constant)
            and debug.value is True and _kw(call, "host") is not None):
        host = _kw(call, "host")
        if isinstance(host, ast.Constant) and host.value == "0.0.0.0":
            return "debug-server-exposed"
    return None


RULE_CHECKS = (_rule_yaml, _rule_shell, _rule_pickle, _rule_sql, _rule_tls, _rule_weak_hash, _rule_random_secret,
               _rule_mktemp, _rule_eval, _rule_tar, _rule_debug)
_RULE_PREFILTER = re.compile(r"yaml\.load|os\.system|os\.popen|shell\s*=\s*True|pickle\.load|marshal\.loads|"
                             r"dill\.load|\.execute(?:many)?\(|verify\s*=\s*False|_create_unverified_context|"
                             r"hashlib\.(?:md5|sha1|sha256)|random\.|mktemp|\beval\(|\bexec\(|extractall|debug\s*=\s*True")


def review_candidates(text):
    if not _RULE_PREFILTER.search(text):
        return []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    lines = text.splitlines()
    out, seen = [], set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        source = "\n".join(lines[fn.lineno - 1:fn.end_lineno]) + "\n"
        first = min([d.lineno for d in fn.decorator_list] + [fn.lineno])
        if first != fn.lineno or not _lines_ok(source) or "[REDACTED_" in source or source.count("\n") < 3:
            continue
        indent = len(lines[fn.lineno - 1]) - len(lines[fn.lineno - 1].lstrip())
        source = "\n".join(line[indent:] for line in source.splitlines()) + "\n"
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call):
                continue
            for check in RULE_CHECKS:
                rule = check(call, fn)
                if rule and rule not in seen and call.lineno == call.end_lineno:
                    seen.add(rule)
                    out.append({"rule": rule, "name": fn.name, "source": source,
                                "line": lines[call.lineno - 1].strip(), "line_number": call.lineno - fn.lineno + 1,
                                "call": ast.unparse(call)})
    return out


IMPLEMENT_PROMPTS = (
    "Write a Python function with the signature `{signature}`.\n\nSpecification: {summary}",
    "Implement `{signature}` in Python. {summary}",
    "{summary}\n\nWrite this as a Python function named `{name}`.",
    "Complete this Python function.\n\n```python\n{header}\n    \"\"\"{summary}\"\"\"\n```",
)
SUMMARY_PROMPTS = (
    "What does this Python function do? Answer in one or two sentences.\n\n```python\n{bare}```",
    "Summarise the purpose of this function.\n\n```python\n{bare}```",
    "Write a one-line docstring for this function.\n\n```python\n{bare}```",
)
REVIEW_PROMPTS = (
    "Review this Python function for security problems.\n\n```python\n{source}```",
    "Is there a security issue in this code?\n\n```python\n{source}```",
    "Check this function before I merge it. Anything unsafe?\n\n```python\n{source}```",
)
RULE_TEXT = {
    "yaml-unsafe-load": ("`yaml.load` without `SafeLoader` can construct arbitrary Python objects, so a crafted "
                         "YAML document can run code.", "Use `yaml.safe_load(...)`, or pass `Loader=yaml.SafeLoader`."),
    "shell-command-injection": ("The command string is built from variable data and run through a shell. If any of that "
                                "data can come from users, files or the network, shell metacharacters (`;`, `&&`, "
                                "`$(...)`) in it would run extra commands. Even with trusted values today, an argument "
                                "list avoids quoting bugs.",
                                "Call `subprocess.run([...], check=True)` with an argument list and no shell, and "
                                "validate any user-controlled argument."),
    "unsafe-deserialization": ("Unpickling runs code chosen by whoever produced the bytes. This is only safe for "
                               "data that never crosses a trust boundary.",
                               "Use a data-only format such as JSON for anything external, or authenticate the "
                               "payload with `hmac` before loading it."),
    "sql-injection": ("The SQL text is built with string formatting. If the value can come from outside the "
                      "program, a quote in it can change the query (SQL injection); even trusted values can break "
                      "on quoting.", "Use the driver's placeholders and pass the values separately, for example "
                      "`cursor.execute(\"... WHERE id = %s\", (value,))` (the placeholder style depends on the driver)."),
    "tls-verification-disabled": ("Certificate verification is turned off, so anyone on the network path can "
                                  "impersonate the server and read or change the traffic.",
                                  "Remove `verify=False` (or the unverified context). For a private CA, pass the CA "
                                  "bundle path to `verify=` instead."),
    "fast-hash-for-password": ("A single fast hash of a password is cheap to brute-force offline if the hashes leak.",
                               "Use a slow, salted password hash: `hashlib.scrypt`, or a maintained library for "
                               "Argon2 or bcrypt, and compare with `hmac.compare_digest`."),
    "insecure-random-secret": ("The `random` module is predictable and not meant for security values such as "
                               "tokens, passwords or salts.", "Use the `secrets` module, for example "
                               "`secrets.token_urlsafe(32)` or `secrets.choice(alphabet)`."),
    "insecure-temp-file": ("`tempfile.mktemp` returns a name without creating the file, so another process can "
                           "create it first (a race that can lead to overwriting or reading other data).",
                           "Use `tempfile.NamedTemporaryFile(delete=False)` or `tempfile.mkstemp()`, which create "
                           "the file atomically."),
    "eval-of-dynamic-input": ("`eval`/`exec` on a non-constant string runs whatever code that string contains, so anyone "
                              "who can influence the string can run code.",
                              "Parse the data instead: `ast.literal_eval` for Python literals, `json.loads` for JSON, "
                              "or an explicit dispatch table for commands."),
    "tar-path-traversal": ("`extractall` on an untrusted archive can write outside the target directory through "
                           "`../` paths, absolute paths or links.",
                           "On Python 3.12+ pass `filter=\"data\"`; on older versions check each member's resolved "
                           "path stays inside the destination before extracting."),
    "debug-server-exposed": ("The development server runs with debug mode on every network interface. The "
                             "interactive debugger allows code execution for anyone who can reach it.",
                             "Disable debug outside local development and bind to `127.0.0.1`, or serve with a "
                             "production WSGI server."),
}
SCOPE_NOTE = "This covers the issue above only; it is not a full security review of the function."


def _pick(options, key):
    return options[int(key[:8], 16) % len(options)]


def implement_row(cand, key):
    imports = "\n".join(cand["imports"])
    response = (imports + "\n\n\n" if imports else "") + cand["source"]
    if int(key[8:10], 16) % 4 == 0:
        prompt = _pick(SUMMARY_PROMPTS, key).format(**cand)
        response = cand["summary"] + "\n"
        return "summarise", prompt, response
    return "implement", _pick(IMPLEMENT_PROMPTS, key).format(**cand), response


def review_row(cand, key):
    problem, fix = RULE_TEXT[cand["rule"]]
    prompt = _pick(REVIEW_PROMPTS, key).format(**cand)
    response = (f"Line {cand['line_number']}: `{cand['line']}`\n\n{problem}\n\nFix: {fix}\n\n{SCOPE_NOTE}\n")
    return "secure-review", prompt, response


def _scan(job):
    path, start, stop, split, sample_mod = job
    funcs, reviews = [], []
    with open(path, "rb") as fh:
        fh.seek(start)
        while fh.tell() < stop:
            raw = fh.readline()
            if not raw:
                break
            row = json.loads(raw)
            if row["source"] == "rfc-nist-prose":
                continue
            text = row["text"]
            meta = {"group": row["group"], "licences": row["licences"], "origin": row["source"], "split": split}
            for cand in review_candidates(text):
                reviews.append({**cand, **meta})
            if int(_hash(row["group"], len(text))[:8], 16) % sample_mod == 0:
                for cand in function_candidates(text):
                    funcs.append({**cand, **meta})
    return funcs, reviews


def _jobs(path, split, sample_mod, chunk_bytes=32 << 20):
    size = Path(path).stat().st_size
    jobs, start = [], 0
    with open(path, "rb") as fh:
        while start < size:
            fh.seek(min(start + chunk_bytes, size))
            fh.readline()
            stop = min(fh.tell(), size)
            jobs.append((str(path), start, stop, split, sample_mod))
            start = stop
    return jobs


def select(candidates, kind_fn, caps, per_group, near_dup, exact, rule_key=None):
    by_split = defaultdict(list)
    for cand in candidates:
        by_split[cand["split"]].append(cand)
    out = defaultdict(list)
    stats = Counter()
    for split, cands in by_split.items():
        cands.sort(key=lambda c: _hash(c["group"], c["name"], c.get("rule", ""), c["source"]))
        groups, rules = Counter(), Counter()
        for cand in cands:
            key = _hash(cand["group"], cand["name"], cand.get("rule", ""), cand["source"])
            bucket = cand[rule_key] if rule_key else "all"
            if rules[bucket] >= caps[split] or groups[cand["group"]] >= per_group:
                stats["capped"] += 1
                continue
            ckey = P.content_key(cand["source"])
            if ckey in exact:
                stats["exact-duplicate"] += 1
                continue
            exact.add(ckey)
            if not near_dup.add_if_new(cand["source"])[0]:
                stats["near-duplicate"] += 1
                continue
            kind, prompt, response = kind_fn(cand, key)
            out[split].append({"id": f"{kind}-{key[:16]}", "family": f"{kind}-{key[:16]}", "kind": kind,
                               "prompt": prompt, "response": response, "group": cand["group"],
                               "licences": cand["licences"], "source": cand["origin"],
                               **({"rule": cand["rule"]} if rule_key else {})})
            groups[cand["group"]] += 1
            rules[bucket] += 1
    return out, stats


def _write_jsonl(path, rows):
    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    Path(path).write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build(args):
    started = time.time()
    corpus = Path(args.corpus).expanduser()
    out = Path(args.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, name in SPLIT_FILES.items():
        path = corpus / name
        if path.exists():
            jobs += _jobs(path, split, args.sample_mod if split == "train" else 1, args.chunk_bytes)
    funcs, reviews = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (f, r) in enumerate(pool.map(_scan, jobs)):
            funcs += f
            reviews += r
            if (i + 1) % 10 == 0:
                print(f"  scanned {i + 1}/{len(jobs)} chunks: {len(funcs)} functions, {len(reviews)} reviews",
                      flush=True)
    near_dup, exact = P.NearDuplicateIndex(), set()
    fcaps = {"train": args.functions, "val": args.functions_eval, "test": args.functions_eval}
    rcaps = {"train": args.per_rule, "val": args.per_rule_eval, "test": args.per_rule_eval}
    mined_f, fstats = select(funcs, implement_row, fcaps, args.per_group, near_dup, exact)
    mined_r, rstats = select(reviews, review_row, rcaps, args.per_group, near_dup, exact, rule_key="rule")
    authored = authored_rows(args.authored_val_fraction)
    contamination = P.ContaminationIndex([]) if args.eval_dir is None else _contamination(args.eval_dir)
    splits, dropped, prompts = {"train": [], "val": [], "test": []}, Counter(), {}
    for split in splits:
        for row in mined_f.get(split, []) + mined_r.get(split, []) + authored.get(split, []):
            if contamination.hits(row["prompt"] + "\n" + row["response"]):
                dropped["eval-contamination"] += 1
                continue
            norm = " ".join(row["prompt"].split())
            if norm in prompts:
                dropped["duplicate-prompt"] += 1
                continue
            prompts[norm] = split
            splits[split].append(row)
    hashes = {split: _write_jsonl(out / OUT_FILES[split], rows) for split, rows in splits.items()}
    kinds = {split: dict(sorted(Counter(r["kind"] for r in rows).items())) for split, rows in splits.items()}
    rules = dict(sorted(Counter(r["rule"] for rows in splits.values() for r in rows if "rule" in r).items()))
    licences = Counter(lic for rows in splits.values() for r in rows for lic in r.get("licences", []))
    manifest = {
        "schema": 1, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_manifest_sha256": hashlib.sha256((corpus / "corpus-manifest.json").read_bytes()).hexdigest()
        if (corpus / "corpus-manifest.json").exists() else None,
        "rows": {split: len(rows) for split, rows in splits.items()}, "kinds": kinds, "review_rules": rules,
        "licences": dict(sorted(licences.items())),
        "authored": "data/sft_authored.py, project-authored, MIT",
        "split": "mined rows inherit the corpus repository split; authored rows split by family",
        "candidates": {"functions": len(funcs), "reviews": len(reviews)},
        "dropped": {"functions": dict(fstats), "reviews": dict(rstats), "final": dict(dropped)},
        "contamination": {"method": "13-gram overlap with evals/data", "eval_ngrams": len(contamination.grams)},
        "sha256": hashes, "elapsed_seconds": round(time.time() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "kinds", "review_rules", "dropped")}, indent=2), flush=True)
    return manifest


def _contamination(eval_dir):
    from data.build_phase2 import contamination_index
    return contamination_index(eval_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--functions", type=int, default=9000)
    ap.add_argument("--functions-eval", type=int, default=400)
    ap.add_argument("--per-rule", type=int, default=150)
    ap.add_argument("--per-rule-eval", type=int, default=15)
    ap.add_argument("--per-group", type=int, default=8)
    ap.add_argument("--sample-mod", type=int, default=6)
    ap.add_argument("--authored-val-fraction", type=float, default=0.15)
    ap.add_argument("--chunk-bytes", type=int, default=32 << 20)
    ap.add_argument("--eval-dir", default=str(HERE.parent / "evals" / "data"))
    ap.add_argument("--workers", type=int, default=8)
    build(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
