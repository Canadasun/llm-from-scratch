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
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self._clock = clock
        self._max_keys = max_keys
        self._lock = threading.Lock()
        self._buckets = {}  # key -> (tokens, last_timestamp)

    def _evict(self):
        """Bound the table. A bucket at full capacity is indistinguishable from a
        never-seen key, so dropping it is lossless; if still over, drop the
        stalest entry (it just gets a fresh full bucket next time)."""
        for k in [k for k, (t, _) in self._buckets.items()
                  if t >= self.capacity]:
            del self._buckets[k]
        while len(self._buckets) >= self._max_keys:
            del self._buckets[min(self._buckets,
                                  key=lambda k: self._buckets[k][1])]

    def allow(self, key):
        """Return True if a request from `key` is permitted now, consuming one
        token; False if the bucket is empty (client should be throttled)."""
        now = self._clock()
        with self._lock:
            if key not in self._buckets and len(self._buckets) >= self._max_keys:
                self._evict()
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens >= 1.0:
                tokens -= 1.0
                self._buckets[key] = (tokens, now)
                return True
            self._buckets[key] = (tokens, now)
            return False
