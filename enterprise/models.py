"""Core identity + resource types for the enterprise layer."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tenant:
    id: str
    name: str = ""


@dataclass(frozen=True)
class Principal:
    id: str
    tenant_id: str
    roles: tuple = ()          # e.g. ("member",) or ("admin",)


@dataclass(frozen=True)
class ApiKey:
    id: str
    key_hash: str              # ONLY the hash is stored, never the plaintext
    tenant_id: str
    principal_id: str
    scopes: tuple = ()         # optional ceiling on actions this key may do


@dataclass(frozen=True)
class Resource:
    id: str
    tenant_id: str             # the owner tenant — the isolation boundary
    kind: str = "document"
    owner_id: str = ""
    attributes: dict = field(default_factory=dict)   # for ABAC extensions
