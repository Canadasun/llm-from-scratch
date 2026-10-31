"""
Finding types for the secure code-review engine.

A Finding is a single, evidence-backed result: which rule fired, where, the exact
source line, a CWE id, a severity and a confidence. Confidence is explicit so the
report can separate high-confidence issues from heuristic ones (which is how we
keep false positives honest rather than hidden).
"""

from dataclasses import dataclass, asdict, field

# Severity: impact if the issue is real.
SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}

# Confidence: how sure the rule is that this is a true positive (not how bad it is).
CONFIDENCE_ORDER = {"high": 2, "medium": 1, "low": 0}


@dataclass(frozen=True)
class Finding:
    rule_id: str          # e.g. "PY-SHELL-INJECTION"
    cwe: str              # e.g. "CWE-78"
    severity: str         # critical | high | medium | low | info
    confidence: str       # high | medium | low
    message: str          # what and why, in one line
    file: str
    line: int
    snippet: str = ""     # the offending source line (evidence)
    remediation: str = "" # a concrete, human-reviewable suggestion

    def __post_init__(self):
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"bad severity: {self.severity}")
        if self.confidence not in CONFIDENCE_ORDER:
            raise ValueError(f"bad confidence: {self.confidence}")

    def sort_key(self):
        # Most severe first, then most confident, then stable by location.
        return (-SEVERITY_ORDER[self.severity],
                -CONFIDENCE_ORDER[self.confidence],
                self.file, self.line, self.rule_id)

    def to_dict(self):
        return asdict(self)
