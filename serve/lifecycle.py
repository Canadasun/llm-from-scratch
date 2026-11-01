"""Cooperative per-request cancellation and wall-clock generation budgets.

Checks occur before each token/forward pass. A native operation already running
cannot be preempted; process-level worker deadlines remain an operational layer.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time


class GenerationStopped(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass
class GenerationBudget:
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)

    def check(self):
        if self.cancelled.is_set():
            raise GenerationStopped("client_disconnected")
        if time.monotonic() >= self.deadline:
            raise GenerationStopped("generation_deadline")


_current = ContextVar("generation_budget", default=None)


def check_generation():
    budget = _current.get()
    if budget is not None:
        budget.check()


@contextmanager
def generation_scope(seconds=60):
    existing = _current.get()
    deadline = time.monotonic() + seconds
    budget = (GenerationBudget(min(deadline, existing.deadline), existing.cancelled)
              if existing else GenerationBudget(deadline))
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)
