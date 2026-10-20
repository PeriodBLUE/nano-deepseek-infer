"""Executable lessons. Every advertised equivalence is checked before reporting."""
import json

import numpy as np

from .decoding import ChoiceGrammar, sample, speculative
from .engine import Engine, Request
from .model import VOCAB, TinyModel, bounded_replay, decode, encode, greedy, softmax
from .parallel import moe_forward, rebalance, split_attention


def cache_lab():
    prompt = encode("deepseek inference")
    cached, uncached = TinyModel(), TinyModel()
    a = greedy(cached, prompt, 24)
    b = greedy(uncached, prompt, 24, cached=False)
    assert a == b
    return dict(equal_tokens=True, cached_projected_tokens=cached.projected_tokens,
                uncached_projected_tokens=uncached.projected_tokens,
                cached_score_elements=cached.score_elements,
                uncached_score_elements=uncached.score_elements)


def speculative_lab():
    prompt = encode("nano inference")
    baseline, target = TinyModel(), TinyModel()
    reference = greedy(baseline, prompt, 48)
    output, trace = speculative(target, prompt, 48)
    assert output == reference
    proposed = sum(t["proposed"] for t in trace)
    accepted = sum(t["accepted"] for t in trace)
    fixed_model = TinyModel()
    fixed_output, fixed_trace = speculative(fixed_model, prompt, 48, adaptive=False)
    assert fixed_output == reference
    return dict(equal_tokens=True, output=decode(output), target_calls=target.calls,
                baseline_calls=baseline.calls, target_projected_tokens=target.projected_tokens,
                baseline_projected_tokens=baseline.projected_tokens,
                proposed=proposed, accepted=accepted,
                acceptance_rate=accepted / proposed if proposed else None, rounds=trace,
                fixed_budget=dict(target_calls=fixed_model.calls,
                                  projected_tokens=fixed_model.projected_tokens,
                                  proposed=sum(t["proposed"] for t in fixed_trace),
                                  accepted=sum(t["accepted"] for t in fixed_trace),
                                  rounds=fixed_trace))


def moe_lab():
    model = TinyModel()
    x = np.random.default_rng(2).normal(size=(64, model.width))
    baseline, counts, loads = moe_forward(x, model.gate, model.experts, [[0, 1], [2, 3]])
    placement = rebalance(counts, ranks=2, slots_per_rank=3)
    balanced, _, new_loads = moe_forward(x, model.gate, model.experts, placement)
    np.testing.assert_allclose(baseline, balanced, atol=1e-12, rtol=1e-12)
    hot = np.array([100, 10, 10, 10])
    hot_placement = rebalance(hot)
    replicas = np.bincount(sum(hot_placement, []), minlength=4)
    estimated = [sum(hot[e] / replicas[e] for e in row) for row in hot_placement]
    return dict(max_error=float(np.max(np.abs(baseline - balanced))),
                logical_routes=counts.tolist(), baseline_rank_routes=loads.tolist(),
                balanced_rank_routes=new_loads.tolist(), placement=placement,
                synthetic_hot_loads=hot.tolist(), synthetic_placement=hot_placement,
                synthetic_estimated_rank_loads=estimated)


def context_lab():
    rng = np.random.default_rng(3)
    q, k, v = rng.normal(size=12), rng.normal(size=(31, 12)), rng.normal(size=(31, 12))
    full = softmax(k @ q / len(q)**0.5) @ v
    partitioned = split_attention(q, k, v, parts=4)
    np.testing.assert_allclose(full, partitioned, atol=1e-12, rtol=1e-12)
    return dict(parts=4, keys=31, max_error=float(np.max(np.abs(full - partitioned))))


def replay_lab():
    prompt = encode("the quick brown fox jumps over the lazy dog")
    full_model, compact_model = TinyModel(window=8), TinyModel(window=8)
    full_cache = full_model.cache()
    full = full_model.forward(prompt, full_cache)
    compact, compact_cache = bounded_replay(compact_model, prompt, backtrack=3)
    # Check the current position AND the furthest permitted rollback position.
    errors = []
    for backtrack in (0, 3):
        errors.append(float(np.max(np.abs(full[-1 - backtrack] - compact[-1 - backtrack]))))
    np.testing.assert_allclose(errors, 0, atol=1e-12)
    for token in encode(" next"):
        a = full_model.forward([token], full_cache)
        b = compact_model.forward([token], compact_cache)
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
    return dict(window=8, backtrack_allowance=3, replayed_tokens=len(compact),
                original_tokens=len(prompt), max_error=max(errors),
                scope="single-layer SWA, absolute positions preserved")


def make_requests():
    return [Request("short", encode("hi"), 9, 41),
            Request("long", encode("continuous batching"), 5, 42),
            Request("middle", encode("nano"), 13, 43),
            Request("fourth", encode("cache"), 4, 44)]


def scheduling_lab():
    serial = Engine(TinyModel(), token_budget=6, max_batch=2, prefill_chunk=3)
    overlap = Engine(TinyModel(), token_budget=6, max_batch=2, prefill_chunk=3)
    a, b = serial.run(make_requests()), overlap.run(make_requests(), pipelined=True)
    assert a == b
    alone = {req.request_id: Engine(TinyModel()).run([req])[req.request_id] for req in make_requests()}
    assert a == alone
    return dict(equal_serial_pipeline_alone=True, caches_released=not overlap.caches,
                outputs={k: decode(v) for k, v in b.items()}, batches=overlap.batches,
                prepared_batches_while_inflight=sum(bool(e["requests"]) for e in overlap.events
                                                    if e["kind"] == "schedule" and e["inflight"]),
                timing_claim="none; CPU lookahead demo, not a GPU throughput benchmark")


def structured_lab():
    choices = ['{"ok":true}', '{"ok":false}']
    grammar = ChoiceGrammar(choices, VOCAB)
    model, rng = TinyModel(), np.random.default_rng(13)
    cache = model.cache()
    logits = model.forward(encode("json"), cache)[-1]
    text = ""
    while not grammar.done(text):
        token = sample(logits, rng, allowed=grammar.allowed(text))
        text += VOCAB[token]
        logits = model.forward([token], cache)[-1]
    assert isinstance(json.loads(text)["ok"], bool)
    return dict(output=text, valid_json=True, scope="finite choice grammar only")


LABS = dict(cache=cache_lab, speculative=speculative_lab, moe=moe_lab,
            context=context_lab, replay=replay_lab, scheduling=scheduling_lab,
            structured=structured_lab)
