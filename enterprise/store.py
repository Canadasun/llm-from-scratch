"""
In-memory store with the tenant-isolation invariant enforced at the boundary.

This is the reference for what a production database must do with row-level
security: **every resource access is scoped to the caller's tenant**, so a
principal can never read, or even learn the existence of, another tenant's
resource. In production this is a DB with RLS; the invariant is identical.

The single rule that matters: resource lookups take a Principal and filter by
`principal.tenant_id`. There is deliberately no "get by id" that ignores the
tenant — that absence is the isolation guarantee.
"""

import time

from enterprise.keys import verify_key


class Store:
    def __init__(self, clock=time.time):
        self._tenants = {}        # id -> Tenant
        self._principals = {}     # id -> Principal
        self._keys = {}           # key_hash -> ApiKey
        self._resources = {}      # id -> Resource
        self._clock = clock       # injectable for deterministic expiry tests

    # --- registration ---
    def add_tenant(self, tenant):
        self._tenants[tenant.id] = tenant
        return tenant

    def add_principal(self, principal):
        if principal.tenant_id not in self._tenants:
            raise ValueError(f"unknown tenant {principal.tenant_id}")
        self._principals[principal.id] = principal
        return principal

    def add_api_key(self, api_key):
        if api_key.principal_id not in self._principals:
            raise ValueError(f"unknown principal {api_key.principal_id}")
        self._keys[api_key.key_hash] = api_key
        return api_key

    def add_resource(self, resource):
        if resource.tenant_id not in self._tenants:
            raise ValueError(f"unknown tenant {resource.tenant_id}")
        self._resources[resource.id] = resource
        return resource

    def get_principal(self, principal_id):
        """Look up a principal by id (no tenant scoping — callers that act on a
        principal, e.g. key management, must themselves check it is in-tenant)."""
        return self._principals.get(principal_id)

    # --- authentication ---
    def authenticate(self, presented_key):
        """Return (Principal, ApiKey) for a valid key, else (None, None).
        Looks up by hash, verifies in constant time, and rejects a key that is
        revoked or past its expiry (so rotation/revocation take effect here)."""
        from enterprise.keys import hash_key
        api_key = self._keys.get(hash_key(presented_key))
        if api_key is None:
            return None, None
        if not verify_key(presented_key, api_key.key_hash):   # defense in depth
            return None, None
        if not api_key.is_active(self._clock()):              # revoked or expired
            return None, None
        principal = self._principals.get(api_key.principal_id)
        # Tenant consistency: if the principal was deleted and re-created under
        # the same id in ANOTHER tenant, the old key must NOT authenticate as it
        # (that would cross the tenant boundary). The key carries its own tenant.
        if principal is None or principal.tenant_id != api_key.tenant_id:
            return None, None
        return principal, api_key

    # --- tenant-scoped API-key management (same isolation rule as resources) ---
    def get_api_key(self, principal, key_id):
        """Return the key with this id ONLY if it belongs to the principal's
        tenant, else None — a key in another tenant is indistinguishable from
        one that does not exist, exactly like get_resource."""
        for k in self._keys.values():
            if k.id == key_id and k.tenant_id == principal.tenant_id:
                return k
        return None

    def list_api_keys(self, principal, principal_id=None):
        """Keys in the caller's tenant (optionally for one principal)."""
        return [k for k in self._keys.values()
                if k.tenant_id == principal.tenant_id
                and (principal_id is None or k.principal_id == principal_id)]

    def update_api_key(self, api_key):
        """Replace a stored key record. The hash is the storage slot and never
        changes on an update (revoke/expire/rotate-link only), so a lifecycle
        change can never alter which secret a record verifies."""
        if api_key.key_hash not in self._keys:
            raise KeyError("unknown api key")
        self._keys[api_key.key_hash] = api_key
        return api_key

    # --- tenant-scoped resource access (THE isolation boundary) ---
    def get_resource(self, principal, resource_id):
        """Return the resource ONLY if it belongs to the principal's tenant.
        Returns None for both 'not found' and 'other tenant' — callers must not
        distinguish the two, so existence never leaks across tenants."""
        r = self._resources.get(resource_id)
        if r is None or r.tenant_id != principal.tenant_id:
            return None
        return r

    def list_resources(self, principal, kind=None):
        return [r for r in self._resources.values()
                if r.tenant_id == principal.tenant_id
                and (kind is None or r.kind == kind)]
