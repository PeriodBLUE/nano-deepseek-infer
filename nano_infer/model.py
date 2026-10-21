"""One causal attention layer + top-2 MoE, with explicit transactional KV state.

Weights can be trained with training.py and restored from a native checkpoint.
Float64 makes numerical comparisons readable.
"""
from dataclasses import dataclass, field

import numpy as np

from .parallel import moe_forward

VOCAB = " abcdefghijklmnopqrstuvwxyz0123456789.,!?:{}\""


def encode(text):
    if not text:
        raise ValueError("prompt must not be empty")
    if any(c not in VOCAB for c in text):
        raise ValueError(f"toy tokenizer supports only: {VOCAB!r}")
    return [VOCAB.index(c) for c in text]


def decode(tokens):
    return "".join(VOCAB[t] for t in tokens)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class KVCache:
    width: int
    offset: int = 0
    k: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)

    def __post_init__(self):
        self.k = np.empty((0, self.width))
        self.v = np.empty((0, self.width))

    def __len__(self):
        return len(self.k)

    @property
    def end(self):
        return self.offset + len(self)

    def truncate(self, length):
        """Discard rejected draft suffix; length is relative to this cache."""
        if not 0 <= length <= len(self):
            raise ValueError("invalid cache rollback")
        self.k = self.k[:length].copy()
        self.v = self.v[:length].copy()

    def fork(self):
        """Copy a prefix into an independently mutable request cache."""
        child = KVCache(self.width, self.offset)
        child.k, child.v = self.k.copy(), self.v.copy()
        return child


class TinyModel:
    def __init__(self, seed=7, width=24, window=None):
        if window is not None and window < 1:
            raise ValueError("window must be positive")
        rng = np.random.default_rng(seed)
        weight = lambda a, b: rng.normal(0, a**-0.5, (a, b))
        self.width, self.window = width, window
        self.embedding = rng.normal(0, 0.5, (len(VOCAB), width))
        self.wq, self.wk, self.wv, self.wo = [weight(width, width) for _ in range(4)]
        self.gate = weight(width, 4)
        self.experts = [(weight(width, width * 2), weight(width * 2, width)) for _ in range(4)]
        self.head = weight(width, len(VOCAB))
        self.calls = self.projected_tokens = self.score_elements = 0

    def cache(self, offset=0):
        return KVCache(self.width, offset)

    def forward(self, tokens, cache, trace=None):
        """Append a chunk and return next-token logits at *each* chunk position.

        Q is new-token-only; K/V include the prefix. Causal masking makes a block
        verification equivalent to sequential decode (up to floating error).
        """
        if not tokens:
            raise ValueError("forward requires at least one token")
        ids = np.asarray(tokens, dtype=int)
        if np.any(ids < 0) or np.any(ids >= len(VOCAB)):
            raise ValueError("token outside vocabulary")
        positions = np.arange(cache.end, cache.end + len(ids))
        frequencies = 10000.0 ** (-np.arange(self.width) / self.width)
        x = self.embedding[ids] + 0.1 * np.sin(positions[:, None] * frequencies)
        q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
        cache.k = np.concatenate([cache.k, k])
        cache.v = np.concatenate([cache.v, v])
        keys = np.arange(cache.offset, cache.end)
        visible = keys[None, :] <= positions[:, None]
        if self.window is not None:
            visible &= keys[None, :] > positions[:, None] - self.window
        scores = q @ cache.k.T / self.width**0.5
        scores = np.where(visible, scores, -np.inf)
        attended = (softmax(scores) @ cache.v) @ self.wo
        h = x + attended
        routed = moe_forward(h, self.gate, self.experts)[0]
        logits = (h + routed) @ self.head
        if trace is not None:
            for name, tensor in (("embedding", x), ("query", q), ("key_cache", cache.k),
                                 ("value_cache", cache.v), ("attention_scores", scores),
                                 ("causal_visibility", visible), ("attention_output", attended),
                                 ("moe_output", routed), ("logits", logits)):
                trace.append(dict(stage=name, shape=list(tensor.shape)))
        self.calls += 1
        self.projected_tokens += len(ids)
        # Allocated dense score entries, including entries later masked out.
        self.score_elements += scores.size
        return logits


def greedy(model, prompt, max_tokens, cached=True):
    if not prompt or max_tokens < 0:
        raise ValueError("nonempty prompt and nonnegative max_tokens required")
    if max_tokens == 0:
        return []
    cache = model.cache()
    logits = model.forward(prompt, cache)[-1]
    output = []
    for i in range(max_tokens):
        token = int(np.argmax(logits))
        output.append(token)
        if i + 1 < max_tokens:
            logits = (model.forward([token], cache) if cached else
                      model.forward(prompt + output, model.cache()))[-1]
    return output


def bounded_replay(model, prompt, backtrack=0):
    """Rebuild a single-layer SWA cache, retaining window + rollback allowance.

    This is valid because this model's K/V depend only on embeddings + absolute
    positions. It is NOT a general multi-layer transformer/YoCo truncation rule.
    """
    if model.window is None or backtrack < 0:
        raise ValueError("bounded replay needs SWA and nonnegative backtrack")
    keep = model.window + backtrack
    start = max(0, len(prompt) - keep)
    cache = model.cache(offset=start)
    return model.forward(prompt[start:], cache), cache
