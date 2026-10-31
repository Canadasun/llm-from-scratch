"""
A simple in-process token-bucket rate limiter, keyed per client.

Not a distributed limiter — for a single-process server it stops one client from
monopolising the (CPU-bound) model. Thread-safe. The clock is injectable so the
behaviour is deterministic in tests.
"""

import threading
import time


class RateLimiter:
    def __init__(self, capacity=10, refill_per_sec=1.0, clock=time.monotonic,
                 max_keys=100_000):
        """capacity: max burst; refill_per_sec: sustained requests/second.
        max_keys bounds the per-client bucket table so a stream of distinct
        clients (principals/IPs) can't grow memory without limit."""
        if capacity <= 0 or refill_per_sec <= 0:
            raise ValueError("capacity and refill_per_sec must be positive")
        if isinstance(max_keys, bool) or not isinstance(max_keys, int) or max_keys < 1:
            raise ValueError("max_keys must be a positive integer")
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self._clock = clock
        self._max_keys = max_keys
        self._lock = threading.Lock()
        self._buckets = {}  # key -> (tokens, last_timestamp)

    def _evict(self, now):
        """Only forget buckets that have fully refilled; preserve spent budgets."""
        for k in [k for k, (t, _) in self._buckets.items()
                  if t + (now - self._buckets[k][1]) * self.refill >= self.capacity]:
            del self._buckets[k]

    def allow(self, key):
        """Return True if a request from `key` is permitted now, consuming one
        token; False if the bucket is empty (client should be throttled)."""
        now = self._clock()
        with self._lock:
            if key not in self._buckets and len(self._buckets) >= self._max_keys:
                self._evict(now)
                if len(self._buckets) >= self._max_keys:
                    return False
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens >= 1.0:
                tokens -= 1.0
                self._buckets[key] = (tokens, now)
                return True
            self._buckets[key] = (tokens, now)
            return False
