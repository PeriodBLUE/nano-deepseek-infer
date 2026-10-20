"""Reuse one explicitly supplied shared prefix across independent continuations.

Inspired by prefix caching in serving engines. This copies arrays per branch:
there is no hash lookup, paging, eviction or copy-on-write block manager.
"""
import numpy as np


def generate_shared_prefix(model, prefix, suffixes, max_tokens):
    if not prefix or max_tokens < 0:
        raise ValueError("nonempty prefix and nonnegative max_tokens required")
    if max_tokens == 0 or not suffixes:
        return [[] for _ in suffixes]
    snapshot = model.cache()
    prefix_logits = model.forward(prefix, snapshot)[-1]
    outputs = []
    for suffix in suffixes:
        cache = snapshot.fork()
        logits = model.forward(suffix, cache)[-1] if suffix else prefix_logits
        output = []
        for index in range(max_tokens):
            token = int(np.argmax(logits))
            output.append(token)
            if index + 1 < max_tokens:
                logits = model.forward([token], cache)[-1]
        outputs.append(output)
    return outputs
