"""Independent control-flow reference for the toy equations, plus shape tours.

The oracle shares supplied weights and NumPy, but never calls forward(), the
optimized MoE dispatcher, or the production softmax. It is not official parity.
"""
import hashlib
import math
import platform
from pathlib import Path

import numpy as np

from .model import TinyModel, encode


def reference_logits(model, tokens, offset=0, fault=None):
    """Slow token-wise oracle; deliberate faults make verification falsifiable."""
    if not tokens or fault not in (None, "future_attention", "zero_position", "drop_moe"):
        raise ValueError("nonempty tokens and known fault required")
    embeddings = []
    for i, token in enumerate(tokens):
        pos = 0 if fault == "zero_position" else offset + i
        positional = np.array([0.1 * math.sin(pos * 10000.0**(-j / model.width))
                               for j in range(model.width)])
        embeddings.append(model.embedding[token] + positional)
    keys = [row @ model.wk for row in embeddings]
    values = [row @ model.wv for row in embeddings]
    result = []
    for i, row in enumerate(embeddings):
        query = row @ model.wq
        start = 0 if model.window is None else max(0, i - model.window + 1)
        end = len(tokens) if fault == "future_attention" else i + 1
        scores = [float(np.dot(query, keys[j])) / math.sqrt(model.width) for j in range(start, end)]
        weights = [math.exp(score - max(scores)) for score in scores]
        denominator = sum(weights)
        context = np.zeros(model.width)
        for j, weight in zip(range(start, end), weights):
            context += (weight / denominator) * values[j]
        h = row + context @ model.wo
        router = h @ model.gate
        selected = sorted(range(len(model.experts)), key=lambda e: (-router[e], e))[:2]
        probs = [math.exp(float(router[e] - max(router[selected]))) for e in selected]
        mixture = np.zeros(model.width)
        if fault != "drop_moe":
            for expert, weight in zip(selected, probs):
                w1, w2 = model.experts[expert]
                mixture += (weight / sum(probs)) * (np.tanh(h @ w1) @ w2)
        result.append((h + mixture) @ model.head)
    return np.asarray(result)


def verification_report():
    """Check full/cached paths against the oracle and record falsification probes."""
    cases = []
    tokens = [1, 4, 2, 7, 9, 3, 6, 8]
    for seed in (0, 7):
        for window in (None, 4):
            model = TinyModel(seed=seed, width=12, window=window)
            reference = reference_logits(model, tokens)
            full = model.forward(tokens, model.cache())
            cache = model.cache()
            chunked = np.concatenate([model.forward(tokens[:3], cache),
                                      model.forward(tokens[3:4], cache),
                                      model.forward(tokens[4:], cache)])
            np.testing.assert_allclose(full, reference, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(chunked, reference, atol=1e-12, rtol=1e-12)
            faults = {}
            for fault in ("future_attention", "zero_position", "drop_moe"):
                difference = float(np.max(np.abs(reference_logits(model, tokens, fault=fault) - full)))
                if difference <= 1e-10:
                    raise AssertionError(f"fixture failed to detect {fault}")
                faults[fault] = dict(detected=True, max_error=difference)
            cases.append(dict(seed=seed, width=12, window=window, tokens=tokens,
                              full_max_error=float(np.max(np.abs(full - reference))),
                              cached_max_error=float(np.max(np.abs(chunked - reference))),
                              faults=faults))
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(__file__).parent.glob("*.py"))}
    return dict(passed=True, python=platform.python_version(), numpy=np.__version__,
                tolerance=dict(atol=1e-12, rtol=1e-12), cases=cases,
                source_sha256=sources,
                scope="toy equations, shared weights, independent control flow; no official model parity")


def shape_tour():
    model = TinyModel()
    cache = model.cache()
    prefill, step = [], []
    logits = model.forward(encode("nano"), cache, trace=prefill)
    token = int(np.argmax(logits[-1]))
    model.forward([token], cache, trace=step)
    return dict(width=model.width, prefill=prefill, decode=step,
                cache_tokens=len(cache), scope="shapes observed during real NumPy execution")
