# Responsible use

This model is built to help **cybersecurity professionals, software developers
and engineering managers** do their work. Security work is inherently dual-use:
the same knowledge that defends a system can attack one. This policy sets how
the project handles that, honestly.

## Intended users and uses

- Security engineers, pentesters, red/blue teams, vulnerability researchers.
- Software developers and architects.
- Engineering and security managers.

Intended uses include: explaining attacks and defenses, writing and reviewing
code, secure-development guidance, detection and incident-response help,
reasoning about CVEs and threat models, and management/communication support.

**Security topics are in scope.** Exploitation techniques, reverse engineering,
offensive tooling and attack theory are legitimate and are *not* refused. A tool
that cannot discuss them is useless to its users.

## Prohibited uses

Use against people or systems you are not authorized to test, or to cause harm:

- Attacking, intruding on, or disrupting systems without explicit authorization.
- Deploying malware/ransomware, exfiltrating data, or extorting a real target.
- Harassment, stalking, doxxing, or generating CSAM or other illegal content.
- Developing weapons of mass harm.

The dividing line is **authorization and intent**, not the topic. "How does this
exploit work / how do I test my own system / how do I detect this" is in scope.
"Help me break into *that* organization" is not.

## How the project enforces this

- **`safety/content_policy.py`** classifies prompts and generations as
  allow / flag / block. It allows security topics, *flags* dual-use content that
  signals action against a specific target without an authorization or defensive
  context (surfacing an authorization reminder), and *blocks* categories with no
  legitimate use (e.g. CSAM).
- **`safety/scrub.py`** keeps secrets and PII out of the training data.
- Honesty about limits: at the current model scale these are a transparent first
  layer, meant to be read, tested (`tests/test_safety.py`) and strengthened as
  capability grows — not a guarantee of safe output.

## Your responsibility

You are responsible for using this model lawfully and only within the scope of
systems and engagements you are authorized to work on.
