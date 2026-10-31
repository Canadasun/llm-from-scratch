# Enterprise security (Stage 3)

Making the platform multi-tenant and trustworthy. This is the start of Stage 3 of
the plan, and it targets the **P0 release gate directly: zero cross-tenant access
in tests.**

The reference implementation is in-memory (`enterprise/`) so the invariants can be
*enforced by tests*. In production the store is a database with row-level
security and the gateway sits behind SSO/MFA and TLS — but the invariants here are
exactly the ones that must hold there.

## Principles (from the platform directive)

- **Authorization is decided in software, never by model output.** The model only
  produces text; whether an action is allowed is a code decision.
- **Tenant isolation is the top invariant.** A principal from tenant A can never
  read, act on, or even learn the existence of tenant B's resources.
- **API keys are stored hashed**, never in plaintext.
- **Every decision is audited**, with secrets/PII redacted from the trail.
- **Keys can only narrow**, never widen, what a principal may do (scoped keys).

## The pieces

| module | role |
|---|---|
| `models.py` | `Tenant`, `Principal` (roles), `ApiKey` (hash + scopes), `Resource` (tenant-owned) |
| `keys.py` | generate a key (plaintext shown once), store only a SHA-256 hash, constant-time verify |
| `store.py` | in-memory store; **resource lookups are tenant-scoped** — the isolation boundary |
| `authz.py` | RBAC (`viewer`/`member`/`admin`) + tenant check + key-scope ceiling |
| `audit.py` | structured, **redacted** audit events, queryable by tenant/decision |
| `gateway.py` | the request pipeline: authenticate → resolve-in-tenant → authorize → act → audit |

## The request pipeline

```
api key → authenticate → resolve resource WITHIN tenant → authorize (RBAC + tenant
          + key scope) → perform → audit
```

Returns `(status, body)` like the serving layer. Key behaviours:
- **Auth failure → 401**; **forbidden → 403**; **missing OR cross-tenant resource
  → the same 404** (so existence never leaks across tenants).
- **Every path emits exactly one audit event**, including denials.

## The P0 gate, in a test

`tests/test_enterprise.py::test_gateway_denies_all_cross_tenant_access` takes a
tenant-A principal and attempts **every** action against a tenant-B resource:

- every attempt returns `404` (never `200`),
- it is indistinguishable from a resource that doesn't exist,
- and every attempt is recorded as a denial.

Isolation is checked in three places (defense in depth): the store never returns a
cross-tenant row, the authorizer denies a cross-tenant resource, and the gateway
turns that into a non-leaking 404.

## Wired in front of `/generate`

The security core is now composed around the serving layer (`serve/secured.py`,
`SecuredInferenceService`), so inference is a multi-tenant, authenticated, audited
capability:

```
API key (Bearer)  ->  authenticate  ->  authorize("model:infer", RBAC + key scope)
                  ->  InferenceService (validate -> rate-limit -> use-safety ->
                      generate -> use-safety)  ->  audit the outcome
```

- **Identity comes from the API key, never the socket.** A missing or bad token
  is a `401` and the model never runs.
- **Infer is a tenant-scoped capability, not a per-resource action** — authorized
  by RBAC plus the key-scope ceiling (a key scoped away from `model:infer` gets a
  `403`, even if the role would allow it). The cross-tenant resource isolation is
  unchanged and still enforced in the store/authz/gateway.
- **Rate limiting is keyed by the authenticated principal**, not the connecting
  IP — real identity is throttled, not whatever address it came from.
- **Every path emits exactly one audit event**: auth failure, authz denial, a
  use-safety block, an internal error, or success (with the tenant attached).
- **Health stays open** (liveness probe) and advertises `"auth": "required"`.

Run it end to end with a demo tenant + key:

```bash
python -m serve --secured          # prints a one-time demo API key at startup
curl -s localhost:8000/generate -H 'authorization: Bearer sk_...' \
     -H 'content-type: application/json' -d '{"prompt":"TLS ","seed":1}'
```

The `--secured` key store is an **in-memory demo identity**, not production key
management (see below).

## Honest scope / what's next

This is the security *core* plus its wiring to the endpoint, not a finished
enterprise deployment. Still ahead for Stage 3: real persistence with DB row-level
security, an admin flow to issue/rotate keys (the `--secured` store is in-memory
and demo-only), SSO (OIDC/SAML) and MFA at the edge, TLS/encryption at rest, and a
shared (not in-process) rate-limit + quota store. No SOC 2/ISO claims are made;
these are building blocks mapped to OWASP LLM Top 10 / NIST AI RMF and the
directive's P5.
