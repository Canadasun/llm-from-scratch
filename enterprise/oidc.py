"""Offline-pinned organization JWT verification with immediate local offboarding.

Trusted policy supplies issuer, audience, JWKS and explicit subject→principal
bindings. Token URLs, roles, tenant and embedded keys never confer authority.
No discovery/JWKS network fetch, resource creation or cloud mutation occurs.
Pinned public keys/policy must be refreshed by the identity owner for rotation.
"""
import json
import math
from pathlib import Path
import stat
import time

from enterprise.models import ApiKey


class OrganizationIdentity:
    def __init__(self, policy_path):
        import jwt  # fail at startup when optional identity dependencies are missing
        self.path = Path(policy_path)
        self._policy()  # fail at startup on bad configuration

    def _policy(self):
        if self.path.is_symlink() or not stat.S_ISREG(self.path.stat().st_mode) or self.path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError('identity policy must be an administrator-controlled regular file')
        with self.path.open('rb') as f:
            raw = f.read(65537)
        if len(raw) > 65536:
            raise ValueError('identity policy too large')
        policy = json.loads(raw)
        if not isinstance(policy.get('issuer'), str) or not policy['issuer'].startswith('https://'):
            raise ValueError('HTTPS issuer required')
        if not isinstance(policy.get('audience'), str) or not policy['audience']:
            raise ValueError('audience required')
        if not isinstance(policy.get('bindings'), dict):
            raise ValueError('explicit subject bindings required')
        amr = policy.get('required_amr', ['mfa'])
        if not isinstance(amr, list) or not amr or any(not isinstance(x, str) or not x for x in amr):
            raise ValueError('nonempty required_amr required')
        age = policy.get('max_token_age_seconds', 3600)
        if type(age) is not int or not 30 <= age <= 3600:
            raise ValueError('max token age must be 30..3600 seconds')
        keys = policy.get('jwks', {}).get('keys')
        if not isinstance(keys, list) or not 1 <= len(keys) <= 10:
            raise ValueError('one to ten pinned public keys required')
        return policy

    def authenticate(self, token, store):
        # Dependencies are optional for local API-key use; identity mode fails
        # startup/import clearly unless requirements-identity.txt is installed.
        import jwt
        try:
            if not isinstance(token, str) or len(token) > 16384:
                return None, None
            policy = self._policy()  # no binding/token cache survives offboarding
            header = jwt.get_unverified_header(token)
            if header.get('alg') != 'RS256' or header.get('typ') not in ('JWT', 'at+jwt'):
                return None, None
            if any(header.get(x) for x in ('jku', 'x5u', 'jwk', 'crit')):
                return None, None
            keys = [key for key in policy['jwks']['keys'] if key.get('kid') == header.get('kid')]
            if len(keys) != 1 or not header.get('kid'):
                return None, None
            key = keys[0]
            if any(x in key for x in ('d', 'p', 'q', 'dp', 'dq', 'qi')):
                return None, None
            if key.get('kty') != 'RSA' or key.get('use', 'sig') != 'sig' or key.get('alg', 'RS256') != 'RS256':
                return None, None
            public = jwt.PyJWK.from_dict(key, algorithm='RS256').key
            if public.key_size < 2048:
                return None, None
            claims = jwt.decode(token, public, algorithms=['RS256'], audience=policy['audience'],
                issuer=policy['issuer'], options={'require': ['iss', 'aud', 'sub', 'exp', 'iat', 'nbf'],
                                                'strict_aud': True})
            if any(type(claims.get(x)) not in (int, float) or not math.isfinite(claims[x])
                   for x in ('exp', 'iat', 'nbf')):
                return None, None
            now = time.time()
            age = policy.get('max_token_age_seconds', 3600)
            if now - claims['iat'] > age or claims['exp'] - claims['iat'] > age or claims['exp'] <= claims['iat']:
                return None, None
            amr = claims.get('amr')
            if not isinstance(amr, list) or not set(policy.get('required_amr', ['mfa'])).issubset(amr):
                return None, None
            binding = policy['bindings'].get(claims['sub'])
            if not isinstance(binding, dict) or binding.get('enabled') is not True:
                return None, None
            principal = store.get_principal(binding.get('principal_id'))
            if principal is None or principal.tenant_id != binding.get('tenant_id'):
                return None, None
            # Roles and tenant come from the local authorized principal, never
            # token roles/groups/tid. OIDC tokens only grant this scope ceiling.
            return principal, ApiKey('oidc', '', principal.tenant_id, principal.id, ('model:infer',))
        except (jwt.PyJWTError, ValueError, TypeError, KeyError, OSError, AttributeError, RecursionError):
            return None, None
