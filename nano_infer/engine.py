"""One request lifecycle: admit -> plan -> execute -> commit -> release.

step() exposes the loop for live arrivals. run() is its offline convenience API.
A single worker executes requests individually, optionally while the CPU plans
OTHER ready requests. Ranks/GPU batches and same-request lookahead are not implied.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .decoding import BigramDraft, ConfidenceBudget, greedy_prefix, sample


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
    def __init__(self, model, token_budget=8, max_batch=3, prefill_chunk=4,
                 max_draft=0, adaptive=False):
        if min(token_budget, max_batch, prefill_chunk) < 1 or max_draft < 0:
            raise ValueError("positive scheduler limits and nonnegative max_draft required")
        self.model = model
        self.token_budget, self.max_batch = token_budget, max_batch
        self.prefill_chunk, self.max_draft, self.adaptive = prefill_chunk, max_draft, adaptive
        self._reset()

    def _reset(self):
        self.ready = deque()
        self.requests, self.rngs, self.caches = {}, {}, {}
        self.drafts, self.budgets = {}, {}
        self.events, self.batches = [], []
        self.failed = False

    def add_request(self, req):
        """Admit a fresh request between step() calls; IDs live for this run."""
        if self.failed:
            raise RuntimeError("engine failed; start a new run() before admitting requests")
        if req.request_id in self.requests:
            raise ValueError("request IDs must be unique")
        if not req.prompt or req.max_tokens < 0 or req.output or req.cursor:
            raise ValueError("requests must be fresh, with nonempty prompt and nonnegative limit")
        if not np.isfinite(req.temperature) or req.temperature < 0:
            raise ValueError("invalid temperature")
        if self.max_draft and req.temperature != 0:
            raise ValueError("speculative requests require temperature=0 (greedy)")
        # Validate before changing engine state, including zero-output requests.
        ids = np.asarray(req.prompt)
        if (not np.issubdtype(ids.dtype, np.integer) or
                np.any(ids < 0) or np.any(ids >= len(self.model.embedding))):
            raise ValueError("prompt must contain valid integer token IDs")
        if req.eos is not None and not 0 <= req.eos < len(self.model.embedding):
            raise ValueError("invalid EOS token")
        rng = np.random.default_rng(req.seed)
        self.requests[req.request_id] = req
        if req.max_tokens:
            self.rngs[req.request_id] = rng
            self.caches[req.request_id] = self.model.cache()
            if self.max_draft:
                self.drafts[req.request_id] = BigramDraft()
                self.drafts[req.request_id].observe(req.prompt)
                self.budgets[req.request_id] = ConfidenceBudget(self.max_draft)
            self.ready.append(req.request_id)

    def is_finished(self):
        return not self.caches

    def _plan(self, inflight=False):
        start = perf_counter()
        plan, remaining = [], self.token_budget
        while self.ready and remaining and len(plan) < self.max_batch:
            key = self.ready.popleft()
            req = self.requests[key]
            if req.cursor < len(req.prompt):
                end = req.cursor + min(self.prefill_chunk, remaining, len(req.prompt) - req.cursor)
                item = Work(key, "prefill", tuple(req.prompt[req.cursor:end]), end)
            elif self.max_draft:
                # Both pending and drafts consume the SAME scheduler token budget.
                output_left = req.max_tokens - len(req.output)
                desired = (self.budgets[key].choose(output_left) if self.adaptive
                           else self.max_draft)
                count = min(desired, remaining - 1, output_left - 1)
                proposals = self.drafts[key].propose(req.output[-1], count)
                item = Work(key, "verify", (req.output[-1], *proposals), req.cursor)
            else:
                item = Work(key, "decode", (req.output[-1],), req.cursor)
            remaining -= len(item.tokens)
            plan.append(item)
        self.events.append(dict(kind="schedule", start=start, end=perf_counter(),
                                inflight=inflight, requests=[w.request_id for w in plan]))
        return plan

    def _execute(self, plan):
        start = perf_counter()
        outputs = [self.model.forward(list(w.tokens), self.caches[w.request_id]) for w in plan]
        return outputs, dict(kind="execute", start=start, end=perf_counter(),
                             requests=[w.request_id for w in plan])

    def _finish(self, plan, result):
        logits, event = result
        self.events.append(event)
        batch = []
        for work, values in zip(plan, logits):
            key = work.request_id
            req, cache = self.requests[key], self.caches[key]
            before = len(cache) - len(work.tokens)
            req.cursor = work.prompt_end
            emitted, accepted = [], 0
            proposed = len(work.tokens) - 1 if work.phase == "verify" else 0
            if work.phase == "verify":
                emitted, accepted = greedy_prefix(values, work.tokens[1:])
            elif req.cursor == len(req.prompt):
                emitted = [sample(values[-1], self.rngs[key], req.temperature)]
            # EOS can occur INSIDE an accepted draft prefix, not just at its end.
            if req.eos in emitted:
                emitted = emitted[:emitted.index(req.eos) + 1]
            if work.phase == "verify":
                # Keep the history before the final emitted (pending) token.
                cache.truncate(before + len(emitted))
                accepted = min(accepted, len(emitted))
                self.budgets[key].update(proposed, accepted)
            if emitted and self.max_draft:
                self.drafts[key].observe([work.tokens[0] if work.phase == "verify"
                                          else req.prompt[-1]] + emitted)
            req.output.extend(emitted)
            done = len(req.output) >= req.max_tokens or (
                bool(emitted) and emitted[-1] == req.eos)
            batch.append(dict(request=key, phase=work.phase, input_tokens=len(work.tokens),
                              emitted=emitted[-1] if emitted else None,
                              emitted_tokens=emitted, proposed=proposed, accepted=accepted,
                              rolled_back=before + len(work.tokens) - len(cache),
                              cache_before=before, cache_after=len(cache), done=done))
            if done:
                for state in (self.caches, self.rngs, self.drafts, self.budgets):
                    state.pop(key, None)
            else:
                self.ready.append(key)
        self.batches.append(batch)
        return batch

    def _abort(self):
        # The worker must have stopped before this runs; never resume partial KV.
        self.ready.clear()
        for state in (self.caches, self.rngs, self.drafts, self.budgets):
            state.clear()
        self.failed = True

    def step(self):
        """Execute one synchronous batch; callers may admit new requests afterward."""
        if self.failed:
            raise RuntimeError("cannot resume a failed step; start a fresh run()")
        if self.is_finished():
            return []
        try:
            plan = self._plan()
            return self._finish(plan, self._execute(plan))
        except Exception:
            self._abort()
            raise

    def run(self, items, pipelined=False):
        """Offline convenience loop. Fresh requests; all arrive before execution."""
        if not self.is_finished():
            raise ValueError("finish active requests with step() before starting a new run")
        self._reset()
        items = list(items)
        try:
            for req in items:
                self.add_request(req)
            if pipelined:
                with ThreadPoolExecutor(max_workers=1) as worker:
                    plan = self._plan()
                    while plan:
                        future = worker.submit(self._execute, plan)
                        # In-flight requests have been removed from ready.
                        next_plan = self._plan(inflight=True)
                        self._finish(plan, future.result())
                        plan = next_plan or self._plan()
            else:
                while not self.is_finished():
                    self.step()
        except Exception:
            self._abort()
            raise
        return {r.request_id: r.output for r in items}
