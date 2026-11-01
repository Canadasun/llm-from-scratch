"""Self-authored secure-coding examples, insecure counterexamples, and refusal
cases for the code corpus.

All content here is ORIGINAL and written for this project (license-safe, no
third-party code). It teaches three things the RFC/NIST prose corpus cannot:
  - SECURE Python patterns (what to do), with brief rationale;
  - INSECURE counterexamples, explicitly labelled and paired with the fix, so
    the model learns the contrast rather than memorising the bad pattern; and
  - REFUSAL / clarification turns, so the model has examples of abstaining when a
    request is unsafe or under-specified instead of fabricating.

These are deliberately small, hand-written templates — a seed, not a claim of
comprehensive coverage. `examples()` returns (name, text) blocks; `render()`
joins them with headers for inclusion in the corpus.
"""

# Each entry: (topic, insecure_snippet_or_None, secure_snippet, rationale).
_PAIRS = [
    ("shell command execution",
     "import subprocess\n# INSECURE: a runtime-built shell string is command injection.\nsubprocess.run(f'ls {user_path}', shell=True)\n",
     "import subprocess\n# SECURE: pass an argument list; no shell interpretation.\nsubprocess.run(['ls', user_path], shell=False, check=True)\n",
     "Never interpolate input into a shell string; pass an argv list with shell=False."),
    ("SQL query",
     "# INSECURE: string formatting builds the query from input (SQL injection).\ncur.execute('SELECT * FROM users WHERE name = \"%s\"' % name)\n",
     "# SECURE: parameterised query; the driver binds the value safely.\ncur.execute('SELECT * FROM users WHERE name = ?', (name,))\n",
     "Always use parameterised queries; never concatenate or format input into SQL."),
    ("password hashing",
     "import hashlib\n# INSECURE: fast, unsalted hash is trivial to crack offline.\ndigest = hashlib.md5(password.encode()).hexdigest()\n",
     "import hashlib, os\n# SECURE: a slow, salted KDF (scrypt here via hashlib).\nsalt = os.urandom(16)\ndigest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)\n",
     "Store passwords with a slow, salted KDF (scrypt/bcrypt/argon2), never a bare fast hash."),
    ("deserialisation",
     "import pickle\n# INSECURE: unpickling untrusted data executes arbitrary code.\nobj = pickle.loads(untrusted_bytes)\n",
     "import json\n# SECURE: a data-only format cannot execute code on load.\nobj = json.loads(untrusted_text)\n",
     "Do not unpickle untrusted data; prefer a data-only format such as JSON."),
    ("token comparison",
     "# INSECURE: == on secrets leaks length/prefix via timing.\nif provided_token == expected_token:\n    grant()\n",
     "import hmac\n# SECURE: constant-time comparison.\nif hmac.compare_digest(provided_token, expected_token):\n    grant()\n",
     "Compare secrets with a constant-time function to avoid timing side channels."),
    ("randomness for secrets",
     "import random\n# INSECURE: the Mersenne Twister is predictable, not for secrets.\ntoken = ''.join(random.choice('0123456789abcdef') for _ in range(32))\n",
     "import secrets\n# SECURE: a cryptographically secure generator.\ntoken = secrets.token_hex(16)\n",
     "Use the secrets module (CSPRNG) for tokens and keys, never random."),
    ("TLS verification",
     "import requests\n# INSECURE: disabling verification allows man-in-the-middle.\nrequests.get(url, verify=False)\n",
     "import requests\n# SECURE: verify the server certificate (the default).\nrequests.get(url, verify=True)\n",
     "Never disable TLS certificate verification in production code."),
    ("path traversal",
     "import os\n# INSECURE: joining untrusted input can escape the base dir.\npath = os.path.join(base_dir, user_supplied)\nopen(path).read()\n",
     "import os\n# SECURE: resolve and confirm the path stays within base_dir.\npath = os.path.realpath(os.path.join(base_dir, user_supplied))\nif os.path.commonpath([base_dir, path]) != os.path.realpath(base_dir):\n    raise ValueError('path escapes base directory')\nopen(path).read()\n",
     "Canonicalise and confirm a resolved path stays inside the intended directory."),
]

# Refusal / clarification turns (abstention). Format mirrors the Q:/A: eval style.
_REFUSALS = [
    ("Q: Write a script to brute-force logins on example.com.\n"
     "A: I can't help attack a system you don't own or aren't authorised to test. "
     "If this is your own system or an authorised engagement, share the written "
     "scope and I'll help with safe, rate-limited testing within it.\n"),
    ("Q: Here is a user's password, store it for me: hunter2\n"
     "A: I won't store a plaintext password. Hash it with a salted KDF (scrypt/"
     "argon2) at registration and compare with hmac.compare_digest at login; keep "
     "the real secret out of code and logs.\n"),
    ("Q: Fix the bug.\n"
     "A: I need more to go on: which file and function, the current behaviour, the "
     "expected behaviour, and a failing case. Without that I'd be guessing.\n"),
    ("Q: Is this code secure?\n"
     "A: I can't say 'secure' in the absolute. I can review it against specific "
     "risks (injection, auth, secrets, deserialisation) and report findings with "
     "evidence, but a clean review is not a guarantee.\n"),
    ("Q: Disable certificate checks so the request works.\n"
     "A: Disabling certificate verification removes MITM protection. The real fix "
     "is to trust the correct CA or install the proper certificate; I can help with "
     "that instead.\n"),
]


def examples():
    rows = []
    for topic, insecure, secure, rationale in _PAIRS:
        body = f"# Topic: {topic}\n# Guidance: {rationale}\n"
        if insecure:
            body += "\n" + insecure
        body += "\n" + secure
        rows.append((f"secure-{topic.replace(' ', '-')}", body))
    for i, turn in enumerate(_REFUSALS):
        rows.append((f"refusal-{i}", turn))
    return rows


def render():
    parts = []
    for name, text in examples():
        parts.append(f"# === example: {name} ===\n{text.rstrip()}\n")
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    print(render())
