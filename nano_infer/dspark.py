"""Executable DSpark head equations and two-generation verification scheduler.

Equation sources: DeepInfer 774ce244, models/mtp/dflash.py (sample_drafts),
layers/tilelang_ops/dflash.py (confidence_head_fused_ref), engine/mtp_varlen.hpp.
Matrices use NumPy [input, output] convention. Hidden states enter before norm.
Run: python -m nano_infer.dspark --json examples/dspark.json
"""
from dataclasses import dataclass
import argparse
import json
from pathlib import Path

import numpy as np


def rmsnorm(x, weight, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def project_features(layers, projection, norm_weight):
    """Concatenate selected target-layer features, project, then normalize."""
    return rmsnorm(np.concatenate(layers, axis=-1) @ projection, norm_weight)


def draft_head(hidden, norm_weight, lm_head, markov_embedding, markov_out,
               confidence_weight, golden_token, temperatures, eps=1e-6):
    """Greedy evaluation of the staged DSpark head; return logits and confidence.

    Hidden shape [steps, width]. Each sampled token becomes the next Markov
    input. Confidence consumes PRE-norm hidden + the same Markov embedding.
    """
    hidden = np.asarray(hidden, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    if hidden.ndim != 2 or temperatures.shape != (len(hidden),) or not np.isfinite(temperatures).all():
        raise ValueError("one finite confidence temperature per draft step required")
    base = rmsnorm(hidden, norm_weight, eps) @ lm_head
    previous = golden_token
    logits, tokens, confidence = [], [], []
    for step, h in enumerate(hidden):
        embedding = markov_embedding[previous]
        scores = base[step] + embedding @ markov_out
        z = np.concatenate([h, embedding]) @ confidence_weight
        z = np.clip(z / max(float(temperatures[step]), 1e-5), -60., 60.)
        confidence.append(float(1 / (1 + np.exp(-z))))
        previous = int(np.argmax(scores))
        tokens.append(previous)
        logits.append(scores)
    return dict(tokens=tokens, logits=np.asarray(logits), confidence=np.asarray(confidence))


@dataclass(frozen=True)
class DraftHistory:
    drafts: int
    rates: tuple = ()
    previous_rates: tuple = ()
    schedule_count: int = 0
    process_count: int = 0
    has_draft: bool = True


def _rates(history, rates):
    if history.drafts < 0 or not history.process_count <= history.schedule_count <= history.process_count + 1:
        raise ValueError("invalid draft count or generation counters")
    if not rates:
        has = history.has_draft or history.schedule_count == history.process_count + 1
        return [float(has)] * history.drafts
    if len(rates) != history.drafts or any(not np.isfinite(r) or not 0 <= r <= 1 for r in rates):
        raise ValueError("rates must contain one probability per draft")
    return rates


def schedule(histories, curve, slope_ms=0., sampler_ms=0.):
    """Choose total Q using N-2, allocate using latest available rates.

    Curve = increasing (graph capacity, startup milliseconds). Rates are the
    scheduler's marginal acceptance estimates, consumed directly as upstream.
    Runtime estimators are explicit inputs so every decision is inspectable.
    """
    n = len(histories)
    if not n:
        raise ValueError("at least one request required")
    if not np.isfinite(slope_ms) or not np.isfinite(sampler_ms) or sampler_ms < 0:
        raise ValueError("invalid runtime estimates")
    old, latest = [], []
    for i, h in enumerate(histories):
        paired = h.schedule_count == h.process_count
        old.extend((float(r), i) for r in _rates(h, h.previous_rates if paired else h.rates))
        latest.extend((float(r), i) for r in _rates(h, h.rates))
    if not curve:
        return dict(total_q=n + sum(h.drafts for h in histories),
                    seqlens_q=[1 + h.drafts for h in histories], candidates=[])
    if any(type(g) is not int or g < 1 or not np.isfinite(t) or t <= 0 for g, t in curve):
        raise ValueError("positive graph capacities and timings required")
    if any(a[0] >= b[0] for a, b in zip(curve, curve[1:])) or curve[-1][0] < n:
        raise ValueError("curve capacities must increase and cover golden tokens")
    old.sort(key=lambda pair: (-pair[0], pair[1]))
    latest.sort(key=lambda pair: (-pair[0], pair[1]))
    expected, best, chosen = float(n), -np.inf, n
    candidates = []
    for added in range(len(old) + 1):
        if added:
            expected += old[added - 1][0]
        total = n + added
        point = next(((g, t) for g, t in curve if g >= total), None)
        if point is None:
            break
        graph, baseline = point
        time_ms = baseline + max(0., slope_ms) * graph + sampler_ms
        throughput = expected / time_ms
        candidates.append(dict(total_q=total, graph=graph, expected_tokens=expected,
                               time_ms=time_ms, tokens_per_ms=throughput))
        # Strict > retains the smaller Q on equal throughput, as in upstream.
        if throughput > best:
            best, chosen = throughput, total
    lengths = [1] * n
    for _, request in latest[:chosen - n]:
        lengths[request] += 1
    return dict(total_q=chosen, seqlens_q=lengths, candidates=candidates)


def main():
    from .dspark_reference import conformance_report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = conformance_report()
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
