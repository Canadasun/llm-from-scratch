"""
AST-based vulnerability detectors for Python source.

Each rule inspects the parsed syntax tree (not regex over raw text), which keeps
false positives down: we can tell a literal from a runtime-built string, a
keyword argument from a coincidence. Every rule maps to a CWE and ships with
ground-truth fixtures under testdata/ so its FP/FN behaviour is measured by the
test suite.

A rule is `def rule(tree, lines, filename) -> list[Finding]`. Add a rule here and
a vulnerable+safe fixture pair, and the tests enforce it.
"""

import ast
import re

from codereview.findings import Finding

# ---------------- helpers ----------------

def _dotted(node):
    """Dotted name for an attribute/name chain, else None.
    subprocess.run -> 'subprocess.run'; os.system -> 'os.system'."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _attr(node):
    """The final attribute/name of a call target: subprocess.run -> 'run'."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _import_map(tree):
    """Resolve import aliases so a call's REAL target is known regardless of how
    it was imported. Returns (modules, symbols):
      modules: local name -> canonical module  (import subprocess as sp -> sp:subprocess)
      symbols: local name -> canonical dotted   (from subprocess import run as r -> r:subprocess.run)
    Without this, `sp.run(shell=True)` or `from pickle import loads; loads(x)`
    slip past rules that only match the literal `subprocess.`/`pickle.` prefix."""
    modules, symbols = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                modules[bound] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                symbols[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return modules, symbols


def _canonical(func, modules, symbols):
    """Canonical dotted target of a call node, resolving import aliases.
    r -> 'subprocess.run'; sp.run -> 'subprocess.run'; subprocess.run -> same."""
    if isinstance(func, ast.Name):
        return symbols.get(func.id, func.id)
    parts = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(modules.get(node.id, node.id))
        return ".".join(reversed(parts))
    return None


def _kw(call, name):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _is_true(node):
    return isinstance(node, ast.Constant) and node.value is True


def _is_literal_str(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _is_dynamic_str(node):
    """A string expression that incorporates runtime data (so it could carry
    attacker input): concatenation, %-format, f-string with substitutions,
    .format(...), or a bare variable/attribute/index."""
    if node is None or isinstance(node, ast.Constant):
        return False
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "format":
        return True
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        return True
    return False


def _snippet(lines, lineno):
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _mk(rule_id, cwe, sev, conf, msg, filename, node, lines, remediation=""):
    return Finding(rule_id=rule_id, cwe=cwe, severity=sev, confidence=conf,
                   message=msg, file=filename, line=getattr(node, "lineno", 0),
                   snippet=_snippet(lines, getattr(node, "lineno", 0)),
                   remediation=remediation)


# ---------------- rules ----------------

_SUBPROCESS_FUNCS = {"run", "call", "Popen", "check_output", "check_call"}


_SUBPROCESS_TARGETS = {f"subprocess.{f}" for f in _SUBPROCESS_FUNCS}


def rule_shell_injection(tree, lines, filename):
    out = []
    modules, symbols = _import_map(tree)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        canonical = _canonical(n.func, modules, symbols)
        dotted = canonical          # canonical resolves import aliases
        # subprocess.* with shell=True (even when imported/aliased)
        shell = _kw(n, "shell")
        is_subprocess = canonical in _SUBPROCESS_TARGETS
        if is_subprocess and _is_true(shell):
            cmd = n.args[0] if n.args else None
            dyn = _is_dynamic_str(cmd)
            out.append(_mk(
                "PY-SHELL-INJECTION", "CWE-78", "high",
                "high" if dyn else "medium",
                "subprocess called with shell=True"
                + (" and a runtime-built command (command injection)" if dyn
                   else " (avoid shell=True)"),
                filename, n, lines,
                "Pass an argument list and shell=False; never interpolate input "
                "into a shell string."))
        # os.system / os.popen with a dynamic argument
        if dotted in ("os.system", "os.popen") and n.args and _is_dynamic_str(n.args[0]):
            out.append(_mk(
                "PY-SHELL-INJECTION", "CWE-78", "high", "high",
                f"{dotted} with a runtime-built command (command injection)",
                filename, n, lines,
                "Use subprocess with an argument list and shell=False."))
    return out


def rule_eval_exec(tree, lines, filename):
    out = []
    modules, symbols = _import_map(tree)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        canonical = _canonical(n.func, modules, symbols)
        callee = canonical.split(".")[-1] if canonical else None
        if canonical in ("eval", "exec", "builtins.eval", "builtins.exec"):
            arg = n.args[0] if n.args else None
            if _is_dynamic_str(arg) or isinstance(arg, (ast.Name, ast.Attribute)):
                out.append(_mk(
                    "PY-DANGEROUS-EVAL", "CWE-95", "critical", "high",
                    f"{callee}() on a runtime value executes arbitrary code",
                    filename, n, lines,
                    "Avoid eval/exec on dynamic input; use ast.literal_eval or a "
                    "explicit dispatch table."))
    return out


def rule_insecure_deserialization(tree, lines, filename):
    out = []
    modules, symbols = _import_map(tree)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        dotted = _canonical(n.func, modules, symbols)   # resolves import aliases
        if dotted in ("pickle.load", "pickle.loads", "cPickle.load",
                      "cPickle.loads", "marshal.load", "marshal.loads"):
            out.append(_mk(
                "PY-INSECURE-DESERIALIZATION", "CWE-502", "high", "high",
                f"{dotted} can execute arbitrary code on untrusted data",
                filename, n, lines,
                "Do not unpickle untrusted data; use a safe format (JSON) or sign "
                "and verify the payload."))
        if dotted == "yaml.load":
            # yaml.load is unsafe unless an explicit safe Loader is given.
            loader = _kw(n, "Loader")
            safe = loader is not None and _attr(loader) in ("SafeLoader", "CSafeLoader")
            if not safe:
                out.append(_mk(
                    "PY-INSECURE-DESERIALIZATION", "CWE-502", "high", "high",
                    "yaml.load without SafeLoader can construct arbitrary objects",
                    filename, n, lines,
                    "Use yaml.safe_load(...) or yaml.load(..., Loader=SafeLoader)."))
    return out


def rule_weak_hash(tree, lines, filename):
    out = []
    weak = {"md5", "sha1"}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        dotted, attr = _dotted(n.func), _attr(n.func)
        hit = None
        if dotted in ("hashlib.md5", "hashlib.sha1"):
            hit = attr
        elif dotted == "hashlib.new" and n.args and _is_literal_str(n.args[0]) \
                and n.args[0].value.lower() in weak:
            hit = n.args[0].value.lower()
        if hit:
            out.append(_mk(
                "PY-WEAK-HASH", "CWE-327", "medium", "medium",
                f"weak hash {hit} (unsuitable for passwords/signatures; ok only "
                f"for non-security checksums)",
                filename, n, lines,
                "Use SHA-256+ for integrity, or bcrypt/scrypt/Argon2 for passwords."))
    return out


def rule_tls_verification_disabled(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            dotted = _dotted(n.func)
            v = _kw(n, "verify")
            if isinstance(v, ast.Constant) and v.value is False:
                out.append(_mk(
                    "PY-TLS-VERIFY-DISABLED", "CWE-295", "high", "high",
                    "TLS certificate verification disabled (verify=False)",
                    filename, n, lines,
                    "Leave verification on; pin a CA bundle if needed."))
            if dotted == "ssl._create_unverified_context":
                out.append(_mk(
                    "PY-TLS-VERIFY-DISABLED", "CWE-295", "high", "high",
                    "ssl._create_unverified_context disables certificate checks",
                    filename, n, lines,
                    "Use ssl.create_default_context()."))
    return out


# Whole-word secret indicators (matched on name word-parts, NOT as substrings —
# so 'tokenizer' does not match 'token'). "key" only counts alongside a
# qualifier like api/secret/access/private/aws to avoid flagging e.g. 'sort_key'.
_SECRET_WORDS = {"password", "passwd", "secret", "token", "credential",
                 "credentials", "apikey", "passphrase"}
_KEY_QUALIFIERS = {"api", "secret", "access", "private", "aws", "signing", "encryption"}
_PLACEHOLDERS = ("changeme", "change_me", "xxx", "todo", "none", "null",
                 "your_", "<", "{", "example", "placeholder", "redacted")


def _name_words(name):
    """Lowercase word parts of a variable name, splitting snake_case AND camelCase.
    'aws_secret_access_key' -> ['aws','secret','access','key']; 'apiKey' -> ['api','key']."""
    words = []
    for part in re.split(r"[_\W]+", name):
        words += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part)
    return [w.lower() for w in words if w]


def _is_secret_name(name):
    words = set(_name_words(name))
    if words & _SECRET_WORDS:
        return True
    # "key" is only a secret with a qualifier (api key, secret key, …)
    if "key" in words and (words & _KEY_QUALIFIERS):
        return True
    return False


def _looks_placeholder(s):
    low = s.strip().lower()
    if len(low) < 4:
        return True
    return any(low.startswith(p) for p in _PLACEHOLDERS)


def rule_hardcoded_secret(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
            value = n.value
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            targets = [n.target]
            value = n.value
        else:
            continue
        if not (_is_literal_str(value) and not _looks_placeholder(value.value)):
            continue
        for t in targets:
            name = t.id if isinstance(t, ast.Name) else (
                t.attr if isinstance(t, ast.Attribute) else None)
            if name and _is_secret_name(name):
                out.append(_mk(
                    "PY-HARDCODED-SECRET", "CWE-798", "high", "medium",
                    f"possible hardcoded secret in '{name}'",
                    filename, n, lines,
                    "Load secrets from env/secret manager, not source; rotate if "
                    "this was real and committed."))
                break
    return out


def rule_flask_debug(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and _attr(n.func) == "run" \
                and _is_true(_kw(n, "debug")):
            out.append(_mk(
                "PY-FLASK-DEBUG", "CWE-489", "medium", "medium",
                "debug=True exposes the interactive debugger / code execution",
                filename, n, lines,
                "Never run with debug=True in production."))
    return out


def rule_sql_injection(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr in ("execute", "executemany"):
            arg = n.args[0] if n.args else None
            if _is_dynamic_str(arg):
                out.append(_mk(
                    "PY-SQL-INJECTION", "CWE-89", "high", "high",
                    "SQL built from a runtime string passed to .execute()",
                    filename, n, lines,
                    "Use parameterised queries: execute(sql, params), never string "
                    "formatting/concatenation."))
    return out


def rule_insecure_tempfile(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and _dotted(n.func) == "tempfile.mktemp":
            out.append(_mk(
                "PY-INSECURE-TEMPFILE", "CWE-377", "medium", "high",
                "tempfile.mktemp is race-prone (the name can be hijacked "
                "before you open it)",
                filename, n, lines,
                "Use tempfile.mkstemp()/NamedTemporaryFile(), which create the "
                "file atomically."))
    return out


_SENSITIVE_EXTRA = {"nonce", "salt", "otp", "csrf"}
# module-level random.* funcs use a non-cryptographic PRNG (Mersenne Twister)
_INSECURE_RANDOM = {"random", "randint", "randrange", "choice", "choices",
                    "getrandbits", "uniform", "sample", "shuffle", "randbytes"}


def _is_sensitive_name(name):
    if _is_secret_name(name):
        return True
    return bool(set(_name_words(name)) & _SENSITIVE_EXTRA)


def rule_weak_random_for_secret(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        value = targets = None
        if isinstance(n, ast.Assign):
            targets, value = n.targets, n.value
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            targets, value = [n.target], n.value
        else:
            continue
        if not (isinstance(value, ast.Call)):
            continue
        dotted = _dotted(value.func)
        if not (dotted and dotted.startswith("random.")
                and dotted.split(".", 1)[1] in _INSECURE_RANDOM):
            continue
        for t in targets:
            name = t.id if isinstance(t, ast.Name) else (
                t.attr if isinstance(t, ast.Attribute) else None)
            if name and _is_sensitive_name(name):
                out.append(_mk(
                    "PY-WEAK-RANDOM", "CWE-330", "high", "medium",
                    f"non-cryptographic random used for a secret value '{name}'",
                    filename, n, lines,
                    "Use the `secrets` module (secrets.token_bytes/token_hex/"
                    "randbelow) for security-sensitive randomness."))
                break
    return out


def rule_django_debug(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "DEBUG" and _is_true(n.value):
            out.append(_mk(
                "PY-DJANGO-DEBUG", "CWE-489", "medium", "medium",
                "DEBUG = True must not ship to production (leaks tracebacks/"
                "settings)",
                filename, n, lines,
                "Drive DEBUG from an environment variable, defaulting to False."))
    return out


def rule_tarfile_extractall(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and _attr(n.func) == "extractall" \
                and _kw(n, "filter") is None and _kw(n, "members") is None:
            out.append(_mk(
                "PY-TARFILE-TRAVERSAL", "CWE-22", "medium", "medium",
                "extractall() without a filter can write outside the target dir "
                "(path traversal / 'tarbomb')",
                filename, n, lines,
                "Pass filter='data' (3.12+) or validate each member path before "
                "extracting."))
    return out


def rule_jinja_autoescape_off(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            a = _kw(n, "autoescape")
            if isinstance(a, ast.Constant) and a.value is False:
                out.append(_mk(
                    "PY-JINJA-AUTOESCAPE-OFF", "CWE-79", "high", "medium",
                    "autoescape=False enables cross-site scripting via templates",
                    filename, n, lines,
                    "Leave autoescape on (select_autoescape) and mark trusted "
                    "output explicitly."))
    return out


def rule_bind_all_interfaces(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        host = _kw(n, "host")
        hit = isinstance(host, ast.Constant) and host.value == "0.0.0.0"
        if hit:
            out.append(_mk(
                "PY-BIND-ALL-INTERFACES", "CWE-668", "low", "low",
                "binding to 0.0.0.0 exposes the service on all interfaces "
                "(fine in a container; review if not intended)",
                filename, n, lines,
                "Bind to 127.0.0.1 unless external exposure is intended and "
                "protected."))
    return out


RULES = [
    rule_shell_injection,
    rule_eval_exec,
    rule_insecure_deserialization,
    rule_weak_hash,
    rule_tls_verification_disabled,
    rule_hardcoded_secret,
    rule_flask_debug,
    rule_sql_injection,
    rule_insecure_tempfile,
    rule_weak_random_for_secret,
    rule_django_debug,
    rule_tarfile_extractall,
    rule_jinja_autoescape_off,
    rule_bind_all_interfaces,
]


def analyze_source(source, filename="<string>"):
    """Parse one Python source string and run every rule. Returns a list of
    Finding. A syntax error yields a single info finding (we never execute)."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        return [Finding("PY-PARSE-ERROR", "CWE-0", "info", "high",
                        f"could not parse: {e.msg}", filename, e.lineno or 0,
                        "", "")]
    lines = source.splitlines()
    findings = []
    for rule in RULES:
        findings.extend(rule(tree, lines, filename))
    return findings
