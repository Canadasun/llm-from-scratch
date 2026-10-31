"""
Authorized API-key issuance, rotation and revocation.

Keys are the credentials the serving gateway authenticates, so their lifecycle
is itself a privileged, tenant-scoped, audited operation — not a bare helper.
Every method here:

- is **authorized in software**: the actor must hold `apikey:manage` (admin),
  and that authority is narrowed by the actor's own key scope (a scoped key can
  manage keys only if its scope includes `apikey:manage`);
- is **tenant-scoped**: an actor can only mint, rotate or revoke keys for
  principals and keys **in its own tenant** — a key in another tenant is
  indistinguishable from one that does not exist (the same isolation rule as
  resources, enforced by the store);
- **can only narrow, never widen**: an issued key's scopes must be within what
  the actor is itself allowed to do right now (`effective_permissions`);
- **shows the plaintext exactly once** (only the hash is ever stored);
- **emits exactly one audit event**, allow or deny, with no secret in the trail.

Rotation issues a fresh key for the same principal and scopes, links the old
record to the new one (`replaced_by`), and either revokes the old key at once or
lets it expire after a grace window so a running client can swap over without a
gap. Authentication rejects a revoked or expired key in the store, so rotation
and revocation take effect the moment they are recorded.
"""

import time
import uuid

from enterprise.authz import ALL_ACTIONS, authorize, effective_permissions
from enterprise.keys import generate_key
from enterprise.models import ApiKey

MANAGE_ACTION = "apikey:manage"


class KeyManager:
    def __init__(self, store, audit, clock=time.time, authorizer=authorize):
        self.store = store
        self.audit = audit
        self.clock = clock
        self.authorizer = authorizer

    # -- internal: authorize the actor for apikey:manage ----------------------
    def _authorize_manage(self, actor, actor_scopes, rid, detail=""):
        d = self.authorizer(actor, MANAGE_ACTION, None, scopes=actor_scopes)
        if not d.allow:
            self.audit.record(rid, MANAGE_ACTION, "deny", "forbidden",
                              tenant_id=actor.tenant_id, principal_id=actor.id,
                              detail=d.reason)
        return d.allow

    # -- issuance -------------------------------------------------------------
    def issue(self, actor, target_principal_id, scopes=(), ttl_seconds=None,
              actor_scopes=(), request_id=None):
        """Mint a key for a principal in the actor's tenant. Returns
        (status, body); on success body['api_key'] is the plaintext, shown
        once. `scopes` must be within the actor's own effective permissions."""
        rid = request_id or uuid.uuid4().hex[:12]
        tid = actor.tenant_id

        if not self._authorize_manage(actor, actor_scopes, rid):
            return 403, {"error": "forbidden", "request_id": rid}

        # target principal must exist AND be in the actor's tenant
        target = self.store.get_principal(target_principal_id)
        if target is None or target.tenant_id != tid:
            self.audit.record(rid, MANAGE_ACTION, "deny", "not_found_or_forbidden",
                              tenant_id=tid, principal_id=actor.id,
                              detail=f"target={target_principal_id}")
            return 404, {"error": "not_found", "request_id": rid}

        # a key can only narrow: the issued key's authority must stay within what
        # the actor can itself do right now (`grantable`).
        scopes = tuple(scopes)
        grantable = effective_permissions(actor, actor_scopes)
        if scopes:
            over = set(scopes) - grantable
            if over:
                self.audit.record(rid, MANAGE_ACTION, "deny", "scope_exceeds_grantor",
                                  tenant_id=tid, principal_id=actor.id,
                                  detail=f"over={sorted(over)}")
                return 403, {"error": "forbidden",
                             "reason": f"cannot grant scopes beyond your own: {sorted(over)}",
                             "request_id": rid}
        elif actor_scopes:
            # An empty request means an UNSCOPED key (the target's full role
            # permissions). If the actor is itself scope-limited, that would hand
            # out more than the actor may wield, so bind the issued key to exactly
            # the actor's grantable set. (An unscoped actor keeps empty=unscoped.)
            # Without this, a key scoped to only `apikey:manage` could mint an
            # unrestricted key — a privilege-escalation hole.
            scopes = tuple(sorted(grantable))

        now = self.clock()
        plaintext, key_hash = generate_key()
        key_id = "key_" + uuid.uuid4().hex[:16]
        expires_at = (now + ttl_seconds) if ttl_seconds else 0.0
        record = ApiKey(key_id, key_hash, tid, target_principal_id, scopes,
                        created_at=now, expires_at=expires_at)
        self.store.add_api_key(record)

        self.audit.record(rid, MANAGE_ACTION, "allow", "issued",
                          tenant_id=tid, principal_id=actor.id,
                          detail=f"key_id={key_id} for={target_principal_id} "
                                 f"scopes={list(scopes)}")
        body = {"request_id": rid, "key_id": key_id, "api_key": plaintext,
                "principal_id": target_principal_id, "scopes": list(scopes)}
        if expires_at:
            body["expires_at"] = expires_at
        return 200, body

    # -- revocation -----------------------------------------------------------
    def revoke(self, actor, key_id, actor_scopes=(), request_id=None):
        """Revoke a key in the actor's tenant. Idempotent-safe: revoking an
        already-revoked key still returns ok."""
        rid = request_id or uuid.uuid4().hex[:12]
        tid = actor.tenant_id

        if not self._authorize_manage(actor, actor_scopes, rid):
            return 403, {"error": "forbidden", "request_id": rid}

        key = self.store.get_api_key(actor, key_id)   # tenant-scoped
        if key is None:
            self.audit.record(rid, MANAGE_ACTION, "deny", "not_found_or_forbidden",
                              tenant_id=tid, principal_id=actor.id,
                              detail=f"key_id={key_id}")
            return 404, {"error": "not_found", "request_id": rid}

        if not key.revoked:
            import dataclasses
            self.store.update_api_key(dataclasses.replace(key, revoked=True))
        self.audit.record(rid, MANAGE_ACTION, "allow", "revoked",
                          tenant_id=tid, principal_id=actor.id,
                          detail=f"key_id={key_id}")
        return 200, {"request_id": rid, "key_id": key_id, "revoked": True}

    # -- rotation -------------------------------------------------------------
    def rotate(self, actor, key_id, grace_seconds=0, actor_scopes=(),
               request_id=None):
        """Issue a new key for the same principal and scopes as `key_id`, link
        the old record to it, and retire the old key: revoked immediately if
        grace_seconds is 0, otherwise set to expire after the grace window so a
        client can swap over without an outage. Returns the new plaintext once."""
        rid = request_id or uuid.uuid4().hex[:16]
        tid = actor.tenant_id

        if not self._authorize_manage(actor, actor_scopes, rid):
            return 403, {"error": "forbidden", "request_id": rid}

        old = self.store.get_api_key(actor, key_id)    # tenant-scoped
        if old is None:
            self.audit.record(rid, MANAGE_ACTION, "deny", "not_found_or_forbidden",
                              tenant_id=tid, principal_id=actor.id,
                              detail=f"key_id={key_id}")
            return 404, {"error": "not_found", "request_id": rid}

        # Ceiling check (was missing): rotation re-issues the old key's authority
        # and hands the actor the new secret, so the actor must be allowed to
        # grant that authority right now. Otherwise a key scoped only to
        # `apikey:manage` could rotate a powerful key and obtain its replacement.
        grantable = effective_permissions(actor, actor_scopes)
        target = self.store.get_principal(old.principal_id)
        old_authority = (effective_permissions(target, old.scopes) if target
                         else set(ALL_ACTIONS))
        over = set(old_authority) - grantable
        if over:
            self.audit.record(rid, MANAGE_ACTION, "deny", "scope_exceeds_grantor",
                              tenant_id=tid, principal_id=actor.id,
                              detail=f"old={key_id} over={sorted(over)}")
            return 403, {"error": "forbidden",
                         "reason": f"cannot rotate a key beyond your own authority: {sorted(over)}",
                         "request_id": rid}

        now = self.clock()
        plaintext, key_hash = generate_key()
        new_id = "key_" + uuid.uuid4().hex[:16]
        # inherit the old key's scopes and its original ttl span if it had one
        ttl = (old.expires_at - old.created_at) if old.expires_at else 0.0
        new_record = ApiKey(new_id, key_hash, tid, old.principal_id, old.scopes,
                            created_at=now, expires_at=(now + ttl) if ttl else 0.0)
        self.store.add_api_key(new_record)

        # retire the old key: immediate revoke, or a grace expiry
        import dataclasses
        if grace_seconds > 0:
            retired = dataclasses.replace(old, replaced_by=new_id,
                                          expires_at=now + grace_seconds)
            outcome = "rotated_grace"
        else:
            retired = dataclasses.replace(old, replaced_by=new_id, revoked=True)
            outcome = "rotated"
        self.store.update_api_key(retired)

        self.audit.record(rid, MANAGE_ACTION, "allow", outcome,
                          tenant_id=tid, principal_id=actor.id,
                          detail=f"old={key_id} new={new_id} grace={grace_seconds}s")
        body = {"request_id": rid, "old_key_id": key_id, "key_id": new_id,
                "api_key": plaintext, "principal_id": old.principal_id,
                "scopes": list(old.scopes)}
        if grace_seconds > 0:
            body["old_key_expires_at"] = now + grace_seconds
        return 200, body
