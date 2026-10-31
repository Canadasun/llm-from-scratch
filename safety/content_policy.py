"""
Use-safety policy for a cybersecurity / software-dev / management assistant.

The hard design constraint: the intended users are **security professionals,
developers and engineering managers**. A policy that refuses "hacking",
"exploits", "malware" or "payloads" would make the model useless for its actual
job — pentesting, vulnerability research, detection engineering and secure
development all require that vocabulary. So this layer is **dual-use aware**:

  - It does NOT flag security *topics*. Exploits, reverse engineering, red-team
    tooling, CVEs and attack techniques are in scope for this model.
  - It flags text that signals *operational wrongdoing against a specific,
    non-consenting target*, or clearly malicious intent with no legitimate
    framing — e.g. instructions to actually deploy ransomware against a named
    victim, to exfiltrate a specific organisation's data, or to build something
    whose only purpose is harm (CSAM, etc.).

Honesty about scope: at the current model scale this is scaffolding, not a
guarantee. It is a transparent, auditable first layer — a policy you can read,
test and extend — put in place from the beginning so the project has a safety
seam to strengthen as the model grows. It returns a decision and the reason, and
never silently changes content.

The policy returns one of:
  - "allow"  : no concern
  - "flag"   : proceed, but surface a caution (dual-use / authorization reminder)
  - "block"  : refuse; clearly malicious with no legitimate reading
"""

import re
from dataclasses import dataclass


@dataclass
class Decision:
    action: str          # "allow" | "flag" | "block"
    reason: str = ""
    category: str = ""

    @property
    def allowed(self):
        return self.action != "block"


# Categories with no legitimate assistant use: refuse regardless of framing.
_BLOCK = [
    ("csam", re.compile(r"(?i)\b(child(?:\s|-)?(?:porn|sexual)|csam|minors?\b.{0,20}\bsexual)\b")),
    ("bioweapon", re.compile(r"(?i)\b(synthesi[sz]e|culture|weaponi[sz]e)\b.{0,30}\b(pathogen|nerve agent|anthrax|smallpox|sarin)\b")),
    ("mass_harm", re.compile(r"(?i)\b(build|make|construct)\b.{0,20}\b(bomb|explosive device|ied)\b.{0,30}\b(school|crowd|public|people)\b")),
]

# Dual-use markers: legitimate for pros, but worth a caution when paired with
# signals of acting against a specific target without authorization.
_WRONGDOING = re.compile(
    r"(?i)\b(ransom(?:ware)?|exfiltrate|steal|breach|take\s*down|ddos|dox|extort)\b")
_TARGETING = re.compile(
    r"(?i)\b(against|target(?:ing)?|victim|their|the company|my (?:ex|employer|boss)|without (?:their )?(?:consent|permission|authorization))\b")

# Phrases that signal a legitimate, authorized, or educational context. Their
# presence downgrades a flag toward allow.
_LEGIT = re.compile(
    r"(?i)\b(authori[sz]ed|with permission|penetration test|pentest|red team|"
    r"ctf|lab|homelab|my own|our own|defen[sc]e|detect|mitigat|harden|"
    r"vulnerability (?:research|assessment)|responsible disclosure|educational|"
    r"blue team|incident response|for a client\b)\b")

_EXPLICIT_UNAUTHORIZED = re.compile(
    r"(?i)\bwithout (?:their )?(?:consent|permission|authorization)\b")
_PROTECTIVE = re.compile(
    r"(?i)\b(?:detect|report|prevent|investigate|identify)\b.{0,60}"
    r"\b(?:child sexual|csam|exploitation)\b")
_HARMFUL_PRODUCTION = re.compile(r"(?i)\b(?:produce|create|generate|obtain|distribute)\b")


def check(text):
    """Classify a prompt or a generation. Returns a Decision."""
    if not text or not text.strip():
        return Decision("allow")

    for category, pat in _BLOCK:
        if pat.search(text):
            if category == "csam" and _PROTECTIVE.search(text) and not _HARMFUL_PRODUCTION.search(text):
                return Decision("flag", "sensitive protective investigation; follow approved procedures",
                                "sensitive_investigation")
            return Decision("block",
                            reason="matches a category with no legitimate use",
                            category=category)

    wrongdoing = bool(_WRONGDOING.search(text))
    targeting = bool(_TARGETING.search(text))
    legit = bool(_LEGIT.search(text)) and not _EXPLICIT_UNAUTHORIZED.search(text)

    if wrongdoing and targeting and not legit:
        return Decision(
            "flag",
            reason=("mentions offensive action against a specific target with no "
                    "authorization/defensive context — confirm this is authorized "
                    "(pentest, CTF, your own system) before proceeding"),
            category="possible_unauthorized_action")

    if wrongdoing and not legit:
        return Decision(
            "flag",
            reason="dual-use security content — ensure use is lawful and authorized",
            category="dual_use")

    return Decision("allow")


CAUTION_BANNER = (
    "[safety] {reason}. This model is for authorized, lawful and educational "
    "security and engineering work only. See docs/RESPONSIBLE_USE.md."
)


def caution_text(decision):
    return CAUTION_BANNER.format(reason=decision.reason)


if __name__ == "__main__":
    import sys
    d = check(" ".join(sys.argv[1:]))
    print(f"{d.action.upper()}: {d.reason or 'no concern'} [{d.category or '-'}]")
