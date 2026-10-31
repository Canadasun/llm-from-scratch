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


def rule_shell_injection(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        dotted, attr = _dotted(n.func), _attr(n.func)
        # subprocess.* with shell=True
        shell = _kw(n, "shell")
        is_subprocess = (dotted and dotted.startswith("subprocess.")) or \
                        (attr in _SUBPROCESS_FUNCS and dotted and "subprocess" in dotted)
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
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id in ("eval", "exec"):
            arg = n.args[0] if n.args else None
            if _is_dynamic_str(arg) or isinstance(arg, (ast.Name, ast.Attribute)):
                out.append(_mk(
                    "PY-DANGEROUS-EVAL", "CWE-95", "critical", "high",
                    f"{n.func.id}() on a runtime value executes arbitrary code",
                    filename, n, lines,
                    "Avoid eval/exec on dynamic input; use ast.literal_eval or a "
                    "explicit dispatch table."))
    return out


def rule_insecure_deserialization(tree, lines, filename):
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        dotted = _dotted(n.func)
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


RULES = [
    rule_shell_injection,
    rule_eval_exec,
    rule_insecure_deserialization,
    rule_weak_hash,
    rule_tls_verification_disabled,
    rule_hardcoded_secret,
    rule_flask_debug,
    rule_sql_injection,
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
