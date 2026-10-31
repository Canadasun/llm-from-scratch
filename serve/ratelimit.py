"""
A simple in-process token-bucket rate limiter, keyed per client.

Not a distributed limiter — for a single-process server it stops one client from
monopolising the (CPU-bound) model. Thread-safe. The clock is injectable so the
behaviour is deterministic in tests.
"""

import threading
import time


class RateLimiter:
    def __init__(self, capacity=10, refill_per_sec=1.0, clock=time.monotonic):
        """capacity: max burst; refill_per_sec: sustained requests/second."""
        if capacity <= 0 or refill_per_sec <= 0:
            raise ValueError("capacity and refill_per_sec must be positive")
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets = {}  # key -> [tokens, last_timestamp]

    def allow(self, key):
        """Return True if a request from `key` is permitted now, consuming one
        token; False if the bucket is empty (client should be throttled)."""
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens >= 1.0:
                tokens -= 1.0
                self._buckets[key] = (tokens, now)
                return True
            self._buckets[key] = (tokens, now)
            return False
