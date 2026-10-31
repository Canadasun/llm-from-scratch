"""
API key generation and verification.

Only a hash of a key is ever stored. A key is high-entropy (unlike a human
password), so a single SHA-256 is the right primitive here — the slow
password hashes (bcrypt/Argon2) exist to resist brute force over LOW-entropy
secrets, which does not apply to a 256-bit random token. Verification is
constant-time.
"""

import hashlib
import hmac
import secrets

_PREFIX = "sk"


def generate_key(prefix=_PREFIX):
    """Return (plaintext, key_hash). The plaintext is shown to the caller ONCE;
    only the hash is stored."""
    raw = secrets.token_urlsafe(32)          # 256 bits of entropy
    plaintext = f"{prefix}_{raw}"
    return plaintext, hash_key(plaintext)


def hash_key(plaintext):
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def verify_key(plaintext, key_hash):
    """Constant-time comparison so timing does not leak the hash."""
    return hmac.compare_digest(hash_key(plaintext), key_hash)
