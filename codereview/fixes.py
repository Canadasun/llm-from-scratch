"""
Patch proposals for the secure code-review engine (Stage 2, increment 1).

For the SUBSET of findings that have a safe, deterministic, behaviour-preserving
fix, this proposes a concrete unified diff for a human to review. It NEVER applies
anything on its own — proposing a patch and applying it are different authority
levels (see the platform directive: writes need explicit authorization + review).

Auto-fixable (minimal, local edits):
  - weak hash         hashlib.md5/sha1(...)  -> hashlib.sha256(...)
  - insecure yaml     yaml.load(x)           -> yaml.safe_load(x)   (no Loader kwarg)
  - TLS verify off    verify=False           -> verify=True
  - flask debug       debug=True             -> debug=False

Deliberately NOT auto-fixed (need real restructuring; left as manual findings):
  shell injection, SQL injection, eval/exec, hardcoded secrets.

Each proposal carries a caveat where the change can alter behaviour (e.g. changing
a hash algorithm invalidates previously stored digests), so the reviewer decides.
"""

import ast
import difflib
from dataclasses import dataclass

from codereview.rules import _attr, _dotted, _is_literal_str, _kw


@dataclass
class FixProposal:
    rule_id: str
    cwe: str
    line: int
    description: str
    confidence: str        # confidence the fix is correct & safe
    caveat: str
    diff: str              # unified diff vs the original file
    start: int             # absolute char offsets of the edit (for applying)
    end: int
    replacement: str
    file: str = ""


def _line_index(source):
    """Return (lines, starts): the source split with line endings kept, and the
    absolute CODEPOINT offset where each line begins."""
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    return lines, starts


def _pos(lines, starts, lineno, byte_col):
    # ast col_offset is a UTF-8 BYTE offset into the line, but `source` is indexed
    # by codepoint. Convert by decoding the first `byte_col` bytes of the line, so
    # a non-ASCII character earlier on the line can't shift the splice point (which
    # previously produced corrupt, invalid Python).
    line = lines[lineno - 1]
    char_col = len(line.encode("utf-8")[:byte_col].decode("utf-8", "ignore"))
    return starts[lineno - 1] + char_col


def _segment_is_ascii(source, start, end):
    # col_offset is a UTF-8 byte offset; our splicing assumes str indices match,
    # which holds for ASCII. If the edited span isn't ASCII, skip (propose nothing)
    # rather than risk a corrupt edit.
    return source[start:end].isascii()


def _edits_for_tree(tree, source):
    """Yield (node_to_replace, replacement, rule_id, cwe, desc, conf, caveat)."""
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        dotted, attr = _dotted(n.func), _attr(n.func)

        # weak hash -> sha256
        if dotted in ("hashlib.md5", "hashlib.sha1"):
            yield (n.func, "hashlib.sha256", "PY-WEAK-HASH", "CWE-327",
                   f"replace {dotted} with hashlib.sha256", "medium",
                   "changes the digest; if this verifies previously stored "
                   "hashes they must be recomputed. For passwords use "
                   "bcrypt/scrypt/Argon2, not a bare hash.")
        elif dotted == "hashlib.new" and n.args and _is_literal_str(n.args[0]) \
                and n.args[0].value.lower() in ("md5", "sha1"):
            yield (n.args[0], '"sha256"', "PY-WEAK-HASH", "CWE-327",
                   "use sha256 in hashlib.new", "medium",
                   "changes the digest; recompute any stored hashes.")

        # yaml.load -> yaml.safe_load (only the plain positional form)
        if dotted == "yaml.load" and _kw(n, "Loader") is None:
            yield (n.func, "yaml.safe_load", "PY-INSECURE-DESERIALIZATION",
                   "CWE-502", "replace yaml.load with yaml.safe_load", "high",
                   "safe_load refuses to construct arbitrary Python objects; if "
                   "the data legitimately relied on custom tags, a safe custom "
                   "loader is needed instead.")

        # verify=False -> verify=True
        v = _kw(n, "verify")
        if isinstance(v, ast.Constant) and v.value is False:
            yield (v, "True", "PY-TLS-VERIFY-DISABLED", "CWE-295",
                   "enable TLS verification (verify=True)", "high",
                   "if the endpoint uses a private CA, point verify at its CA "
                   "bundle path instead of disabling it.")

        # debug=True -> debug=False on a .run(...) call
        d = _kw(n, "debug")
        if attr == "run" and isinstance(d, ast.Constant) and d.value is True:
            yield (d, "False", "PY-FLASK-DEBUG", "CWE-489",
                   "disable the debugger (debug=False)", "high",
                   "use a proper logging/reloader setup for local development "
                   "instead of debug=True.")


def propose_fixes(source, filename="<string>"):
    """Return a list of FixProposal for one source string. Empty on parse error."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []
    src_lines, ls = _line_index(source)
    proposals = []
    for node, repl, rule_id, cwe, desc, conf, caveat in _edits_for_tree(tree, source):
        start = _pos(src_lines, ls, node.lineno, node.col_offset)
        end = _pos(src_lines, ls, node.end_lineno, node.end_col_offset)
        if not _segment_is_ascii(source, start, end):
            continue
        new_source = source[:start] + repl + source[end:]
        diff = "".join(difflib.unified_diff(
            src_lines, new_source.splitlines(keepends=True),
            fromfile=f"a/{filename}", tofile=f"b/{filename}", n=2))
        proposals.append(FixProposal(
            rule_id=rule_id, cwe=cwe, line=node.lineno, description=desc,
            confidence=conf, caveat=caveat, diff=diff,
            start=start, end=end, replacement=repl, file=filename))
    proposals.sort(key=lambda p: p.line)
    return proposals


def apply_fixes(source, proposals):
    """Apply all proposals to the source and return the patched text. Used only
    behind an explicit, human-approved action — never automatically."""
    edits = sorted(proposals, key=lambda p: p.start, reverse=True)
    out = source
    for p in edits:
        out = out[:p.start] + p.replacement + out[p.end:]
    return out
