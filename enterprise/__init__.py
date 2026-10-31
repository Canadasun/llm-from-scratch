"""
Enterprise security layer (Stage 3 of the platform plan).

The platform must be multi-tenant and trustworthy: a principal from one tenant
must NEVER reach another tenant's data, authorization is decided in software
(never by model output), API keys are stored hashed, and every decision is
audited with secrets redacted.

This package is the testable reference for that. In production the store is a
database with row-level security and the gateway sits behind SSO/MFA and TLS;
here the SAME invariants are implemented in-memory so they can be enforced by
tests — above all the P0 gate: **zero cross-tenant access**.

  models.py   Tenant, Principal, ApiKey, Resource
  keys.py     generate/verify API keys (only a hash is ever stored)
  store.py    in-memory store; the tenant-isolation invariant lives here
  authz.py    RBAC + tenant check: authorize(principal, action, resource)
  audit.py    structured, redacted audit log of every decision
  gateway.py  authenticate -> authorize -> act -> audit (the request pipeline)
"""
