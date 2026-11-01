"""Atomic tenant + principal quotas for an EXISTING Redis-compatible service.

No service is provisioned. The default deployment remains local/$0; selecting
this adapter requires an operator-provided endpoint. Errors fail closed.
"""
import hashlib
import math


class QuotaUnavailable(RuntimeError):
    pass


SCRIPT = '''
local now = redis.call('TIME')
local t = tonumber(now[1]) + tonumber(now[2]) / 1000000
local cap = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local values = {}
for i, key in ipairs(KEYS) do
  local state = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(state[1]) or cap
  local last = tonumber(state[2]) or t
  values[i] = math.min(cap, tokens + math.max(0, t-last)*rate)
end
local allowed = 1
for i, tokens in ipairs(values) do
  if tokens < 1 then allowed = 0 end
end
for i, key in ipairs(KEYS) do
  redis.call('HSET', key, 'tokens', values[i]-allowed, 'ts', t)
  redis.call('EXPIRE', key, math.ceil(cap/rate)+1)
end
return allowed
'''


class RedisQuota:
    def __init__(self, client, capacity=10, refill_per_sec=1):
        if not math.isfinite(capacity) or not math.isfinite(refill_per_sec) or capacity < 1 or refill_per_sec <= 0:
            raise ValueError("invalid quota limits")
        self.client = client
        self.capacity = capacity
        self.refill = refill_per_sec

    def allow(self, tenant_id, principal_id):
        tenant = hashlib.sha256(tenant_id.encode()).hexdigest()
        principal = hashlib.sha256(principal_id.encode()).hexdigest()
        # Hash tag keeps the atomic script on one Redis Cluster slot.
        prefix = f"llm-quota:{{{tenant}}}"
        try:
            verdict = self.client.eval(SCRIPT, 2, prefix + ":tenant",
                                       prefix + ":" + principal, self.capacity, self.refill)
            if type(verdict) is not int or verdict not in (0, 1):
                raise ValueError("invalid quota response")
            return bool(verdict)
        except Exception as exc:
            raise QuotaUnavailable("shared quota service unavailable") from exc


def from_url(url, capacity=10, refill_per_sec=1):
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if parsed.scheme not in ("rediss", "unix") and not (
            parsed.scheme == "redis" and parsed.hostname in ("localhost", "127.0.0.1", "::1")):
        raise ValueError("remote quota endpoints require TLS (rediss)")
    try:
        import redis  # optional: install on the prepared worker, never auto-provision
    except ImportError as exc:
        raise ValueError("Redis quota adapter requires the optional redis client") from exc
    return RedisQuota(redis.Redis.from_url(url, socket_connect_timeout=2,
                                          socket_timeout=2), capacity, refill_per_sec)
