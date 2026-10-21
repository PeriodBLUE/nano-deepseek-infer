"""Bounded admission, FIFO waiting, cancellation/deadlines and token timestamps.

One owner calls step(); the HTTP adapter serializes access with a lock. An
injected monotonic clock makes timeout and latency arithmetic testable.
"""
from dataclasses import dataclass, field
from time import perf_counter
import math

from .engine import Request
from .model import encode, decode


class BusyError(Exception):
    pass


@dataclass
class Entry:
    request: Request
    arrival: float
    deadline: float
    status: str = "queued"
    admitted: float | None = None
    finished: float | None = None
    token_times: list = field(default_factory=list)
    error: str | None = None


class Service:
    def __init__(self, engine, max_pending=16, history=64, clock=perf_counter):
        if max_pending < 1 or history < max_pending:
            raise ValueError("history must cover positive pending capacity")
        self.engine, self.max_pending, self.history, self.clock = engine, max_pending, history, clock
        self.entries, self.counter = {}, 0

    def submit(self, prompt, max_tokens=64, timeout=30.):
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative integer")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be text")
        req = Request(str(self.counter), encode(prompt), max_tokens, temperature=0)
        if self.engine.pool is not None and self.engine.required_blocks(req) > len(self.engine.pool.refs):
            raise ValueError("request exceeds total KV capacity; shorten prompt/output or increase blocks")
        if sum(e.status in ("queued", "running") for e in self.entries.values()) >= self.max_pending:
            raise BusyError("pending request capacity reached")
        while len(self.entries) >= self.history:
            oldest = next(k for k, e in self.entries.items() if e.status not in ("queued", "running"))
            del self.entries[oldest]
        now = self.clock()
        self.entries[req.request_id] = Entry(req, now, now + timeout)
        self.counter += 1
        return req.request_id

    def _stop(self, key, status, now):
        entry = self.entries[key]
        self.engine.cancel(key)
        self.engine.requests.pop(key, None)
        entry.status, entry.finished = status, now

    def cancel(self, key):
        entry = self.entries[key]
        if entry.status in ("queued", "running"):
            self._stop(key, "cancelled", self.clock())
        return self.snapshot(key)

    def step(self):
        now = self.clock()
        for key, entry in self.entries.items():
            if entry.status in ("queued", "running") and now >= entry.deadline:
                self._stop(key, "timed_out", now)
        # FIFO admission. Memory is reserved for each request's full lifetime.
        for key, entry in self.entries.items():
            if entry.status != "queued":
                continue
            try:
                self.engine.add_request(entry.request)
            except MemoryError:
                break
            entry.status, entry.admitted = "running", self.clock()
            if entry.request.max_tokens == 0:
                self._stop(key, "completed", self.clock())
        try:
            rows = self.engine.step()
        except Exception as exc:
            for key, entry in self.entries.items():
                if entry.status == "running":
                    entry.error = type(exc).__name__ + ": " + str(exc)
                    self._stop(key, "failed", self.clock())
            self.engine.run([])  # worker has stopped; reset failed execution state
            return []
        now = self.clock()
        for row in rows:
            key = row["request"]
            entry = self.entries[key]
            entry.token_times.extend([now] * len(row["emitted_tokens"]))
            if now >= entry.deadline:
                self._stop(key, "timed_out", now)
            elif row["done"]:
                self._stop(key, "completed", now)
        # Keep observability bounded for a long-lived service.
        self.engine.events[:] = self.engine.events[-self.history:]
        self.engine.batches[:] = self.engine.batches[-self.history:]
        return rows

    def snapshot(self, key):
        e = self.entries[key]
        times = e.token_times
        intervals = [b - a for a, b in zip(times, times[1:])]
        return dict(id=key, status=e.status, text=decode(e.request.output), tokens=e.request.output.copy(),
                    error=e.error, metrics=dict(
                        queue_seconds=None if e.admitted is None else e.admitted - e.arrival,
                        ttft_seconds=None if not times else times[0] - e.arrival,
                        itl_seconds=intervals,
                        tpot_seconds=None if not intervals else sum(intervals) / len(intervals),
                        latency_seconds=None if e.finished is None else e.finished - e.arrival))
