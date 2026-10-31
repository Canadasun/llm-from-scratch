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

from enterprise.keys import verify_key


class Store:
    def __init__(self):
        self._tenants = {}        # id -> Tenant
        self._principals = {}     # id -> Principal
        self._keys = {}           # key_hash -> ApiKey
        self._resources = {}      # id -> Resource

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

    # --- authentication ---
    def authenticate(self, presented_key):
        """Return (Principal, ApiKey) for a valid key, else (None, None).
        Looks up by hash and verifies in constant time."""
        from enterprise.keys import hash_key
        api_key = self._keys.get(hash_key(presented_key))
        if api_key is None:
            return None, None
        if not verify_key(presented_key, api_key.key_hash):   # defense in depth
            return None, None
        return self._principals.get(api_key.principal_id), api_key

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
