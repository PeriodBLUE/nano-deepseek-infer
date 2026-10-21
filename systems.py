"""Run MLA, physical KV sharing and ragged batch experiments on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np

from nano_infer.model import TinyModel, encode
from nano_infer.mla import mla_lab
from nano_infer.paging import BlockPool, PagedCache, PrefixStore


def experiments():
    model = TinyModel()
    pool = BlockPool(16, 4, model.width)
    store = PrefixStore(pool)
    prefix = encode("nano ")
    cache = PagedCache(pool)
    model.forward(prefix, cache)
    store.put(prefix, cache)
    cache.release()
    branches = [store.get(prefix + encode(s))[0] for s in ("cache", "model")]
    shared = pool.snapshot()
    errors = []
    for branch, suffix in zip(branches, ("cache", "model")):
        logits = model.forward(encode(suffix), branch)
        expected = model.forward(prefix + encode(suffix), model.cache())[-len(suffix):]
        errors.append(float(np.max(abs(logits - expected))))
    after_write = pool.snapshot()
    for branch in branches: branch.release()
    store.clear()
    assert max(errors) < 1e-12 and not np.any(pool.refs)
    chunks = [encode("a"), encode("hello"), encode("cache")]
    caches = [model.cache() for _ in chunks]
    expected = [model.forward(c, model.cache()) for c in chunks]
    calls, projections = model.calls, model.projected_tokens
    actual = model.forward_batch(chunks, caches)
    error = max(float(np.max(abs(a - b))) for a, b in zip(actual, expected))
    assert error < 1e-12
    return dict(mla=mla_lab(), paging=dict(shared=shared, after_branch_writes=after_write,
                final=pool.snapshot(), max_abs_error=max(errors)),
                batching=dict(query_lengths=list(map(len, chunks)), live_tokens=sum(map(len, chunks)),
                projected_tokens_including_padding=model.projected_tokens - projections,
                forward_calls=model.calls - calls, max_abs_error=error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    text = json.dumps(experiments(), indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8", newline="\n")
