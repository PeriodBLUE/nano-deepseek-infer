"""Round-robin continuous batching with chunked prefill and optional CPU lookahead.

One worker executes NumPy. While it runs, the scheduler prepares a batch of OTHER
ready requests. No future sampled token is guessed. This is a small, conservative
overlap pattern, not DeepInfer's same-request zero-overhead scheduling protocol.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .decoding import sample


@dataclass
class Request:
    request_id: str
    prompt: list[int]
    max_tokens: int = 12
    seed: int = 0
    temperature: float = 1.0
    eos: int | None = None
    output: list[int] = field(default_factory=list, init=False)
    cursor: int = field(default=0, init=False)


@dataclass(frozen=True)
class Work:
    request_id: str
    phase: str
    tokens: tuple[int, ...]
    prompt_end: int


class Engine:
    def __init__(self, model, token_budget=8, max_batch=3, prefill_chunk=4):
        if min(token_budget, max_batch, prefill_chunk) < 1:
            raise ValueError("all scheduler limits must be positive")
        self.model = model
        self.token_budget = token_budget
        self.max_batch = max_batch
        self.prefill_chunk = prefill_chunk
        self.events = []
        self.batches = []
        self.caches = {}

    def _plan(self, ready, requests, inflight):
        start = perf_counter()
        plan, remaining = [], self.token_budget
        while ready and remaining and len(plan) < self.max_batch:
            key = ready.popleft()
            req = requests[key]
            if req.cursor < len(req.prompt):
                end = req.cursor + min(self.prefill_chunk, remaining, len(req.prompt) - req.cursor)
                item = Work(key, "prefill", tuple(req.prompt[req.cursor:end]), end)
            else:
                item = Work(key, "decode", (req.output[-1],), req.cursor)
            remaining -= len(item.tokens)
            plan.append(item)
        self.events.append(dict(kind="schedule", start=start, end=perf_counter(),
                                inflight=inflight, requests=[w.request_id for w in plan]))
        return plan

    def _execute(self, plan):
        start = perf_counter()
        outputs = [self.model.forward(list(w.tokens), self.caches[w.request_id])[-1] for w in plan]
        return outputs, dict(kind="execute", start=start, end=perf_counter(),
                             requests=[w.request_id for w in plan])

    def _finish(self, plan, result, requests, rngs, ready):
        logits, event = result
        self.events.append(event)
        batch = []
        for work, values in zip(plan, logits):
            req = requests[work.request_id]
            req.cursor = work.prompt_end
            emitted = None
            if req.cursor == len(req.prompt):
                emitted = sample(values, rngs[req.request_id], req.temperature)
                req.output.append(emitted)
            done = len(req.output) >= req.max_tokens or (
                emitted is not None and emitted == req.eos)
            batch.append(dict(request=req.request_id, phase=work.phase,
                              input_tokens=len(work.tokens), emitted=emitted, done=done))
            if done:
                del self.caches[req.request_id]
            else:
                ready.append(req.request_id)
        self.batches.append(batch)

    def run(self, items, pipelined=False):
        keys = [r.request_id for r in items]
        if len(set(keys)) != len(keys):
            raise ValueError("request IDs must be unique")
        for req in items:
            if not req.prompt or req.max_tokens < 0 or req.output or req.cursor:
                raise ValueError("requests must be fresh, with nonempty prompt and nonnegative limit")
            if not np.isfinite(req.temperature) or req.temperature < 0:
                raise ValueError("invalid temperature")
        requests = {r.request_id: r for r in items}
        rngs = {r.request_id: np.random.default_rng(r.seed) for r in items}
        ready = deque(r.request_id for r in items if r.max_tokens)
        self.caches = {key: self.model.cache() for key in ready}
        self.events, self.batches = [], []
        plan = self._plan(ready, requests, False)
        try:
            if pipelined:
                with ThreadPoolExecutor(max_workers=1) as worker:
                    while plan:
                        future = worker.submit(self._execute, plan)
                        # In-flight requests have been removed from ready.
                        next_plan = self._plan(ready, requests, True)
                        self._finish(plan, future.result(), requests, rngs, ready)
                        plan = next_plan or self._plan(ready, requests, False)
            else:
                while plan:
                    self._finish(plan, self._execute(plan), requests, rngs, ready)
                    plan = self._plan(ready, requests, False)
        finally:
            self.caches.clear()
        return {r.request_id: r.output for r in items}
