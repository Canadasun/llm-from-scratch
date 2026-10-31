"""
Authorization: decided in software, never by model output.

Two gates, both must pass:
  1. Tenant isolation — if the action targets a resource, it must belong to the
     principal's tenant (defense in depth; the store already scopes lookups).
  2. RBAC — the principal's role(s) must grant the action, and if the API key is
     scoped, the key's scopes must also allow it (a key can only ever NARROW,
     never widen, what its principal may do).

The permission sets are small and explicit so they can be read and tested. ABAC
can extend `authorize` using resource.attributes without changing callers.
"""

from dataclasses import dataclass

# role -> permitted actions
ROLE_PERMISSIONS = {
    "viewer": {"model:infer", "resource:read"},
    "member": {"model:infer", "resource:read", "resource:write"},
    "admin": {"model:infer", "resource:read", "resource:write",
              "resource:delete", "tenant:manage", "apikey:manage"},
}

ALL_ACTIONS = set().union(*ROLE_PERMISSIONS.values())


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str = ""


def permissions_for(roles):
    granted = set()
    for r in roles:
        granted |= ROLE_PERMISSIONS.get(r, set())
    return granted


def effective_permissions(principal, scopes=None):
    """What a principal can ACTUALLY do right now: its role permissions,
    narrowed by the presented key's scopes if the key is scoped. This is the
    ceiling an admin may grant when issuing a key — a key can only ever narrow,
    never widen, the authority of the principal that mints it."""
    perms = permissions_for(principal.roles)
    if scopes:
        perms &= set(scopes)
    return perms


def authorize(principal, action, resource=None, scopes=None):
    """Return a Decision. `scopes`, if non-empty, is the API key's ceiling."""
    # 1. tenant isolation
    if resource is not None and resource.tenant_id != principal.tenant_id:
        return Decision(False, "cross-tenant access denied")
    # 2. RBAC
    if action not in permissions_for(principal.roles):
        return Decision(False, f"role {list(principal.roles)} lacks '{action}'")
    # 3. key scope ceiling (empty scopes = no extra restriction)
    if scopes:
        if action not in set(scopes):
            return Decision(False, f"api key scope does not include '{action}'")
    return Decision(True, "authorized")
