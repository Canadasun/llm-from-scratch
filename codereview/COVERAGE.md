# Scanner coverage and explicit limitations

`codereview/` is a deterministic AST scanner for a fixed set of Python security
rules. It is a useful first pass, **not** comprehensive security analysis. This
document states plainly what it does and does not do (audit follow-up: declare
coverage limitations rather than imply completeness).

## What it does

- Parses each file with `ast`; it never imports or executes scanned code.
- Resolves **import aliases** before matching call targets, so a dangerous call
  is caught regardless of how it was imported:
  - `import subprocess as sp; sp.run(cmd, shell=True)`
  - `from subprocess import run as r; r(cmd, shell=True)`
  - `from pickle import loads; loads(data)` / `import pickle as p; p.loads(data)`
  - `from builtins import eval as e; e(user_input)`
  (See `_import_map` / `_canonical` in `rules.py`.)
- Flags: shell injection (`subprocess … shell=True`, `os.system/os.popen` on
  dynamic input), `eval`/`exec` on runtime values, insecure deserialization
  (`pickle`/`marshal`/unsafe `yaml.load`), weak hashes, disabled TLS
  verification, hardcoded secrets, Flask/Django debug, insecure tempfiles, weak
  randomness for secrets, `tarfile.extractall`, Jinja autoescape off, binding to
  all interfaces.

## What it does NOT do (known gaps)

- **No taint / source-to-sink analysis.** It reasons about a single call site and
  shallow expression shapes, not data flow across functions, modules, or files.
  A tainted value laundered through a helper will be missed.
- **No cross-procedure or cross-module resolution** beyond import aliasing. A
  function assigned to another name at runtime (`f = subprocess.run; f(...)`),
  attributes fetched via `getattr`, or indirection through a dict/list are not
  resolved.
- **No dependency / supply-chain analysis.** It does not inspect third-party
  packages, versions, or advisories.
- **Rule set is finite.** Absence of findings is **not** proof of security; new
  vulnerability classes and framework-specific issues are out of scope until a
  rule is added.
- **No type information.** It cannot distinguish a safe same-named method from a
  dangerous one when the receiver's type is unknown.

## Consequence

Treat findings as leads for human review, and treat a clean scan as "no issues
from the current rules," never as an assurance. Executable and semantic checks
(the verification workflow and independent evaluation) are what establish that a
change is actually safe.
