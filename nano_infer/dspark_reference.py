"""Scalar head equations and exhaustive subset scheduler oracle.

The reference never calls the production head, norm, or rate selection helpers.
Small exhaustive cases make scheduling mistakes visible without a GPU.
"""
from itertools import combinations
import math

import numpy as np

from .dspark import DraftHistory, draft_head, project_features, schedule


def scalar_head(h, norm, head, emb, out, confidence, golden, temps, eps=1e-6):
    token, logits, tokens, rates = golden, [], [], []
    for s in range(len(h)):
        scale = math.sqrt(sum(float(x)**2 for x in h[s]) / len(h[s]) + eps)
        row = [sum(float(h[s, j]) / scale * norm[j] * head[j, v] for j in range(len(norm)))
               + sum(emb[token, j] * out[j, v] for j in range(out.shape[0]))
               for v in range(head.shape[1])]
        features = list(h[s]) + list(emb[token])
        z = sum(float(x) * float(w) for x, w in zip(features, confidence)) / max(temps[s], 1e-5)
        rates.append(1 / (1 + math.exp(-max(-60., min(60., z)))))
        token = max(range(len(row)), key=row.__getitem__)
        tokens.append(token)
        logits.append(row)
    return dict(tokens=tokens, logits=np.asarray(logits), confidence=np.asarray(rates))


def exhaustive_schedule(histories, curve, slope=0., sampler=0.):
    n = len(histories)
    if not curve:
        return n + sum(h.drafts for h in histories), [h.drafts + 1 for h in histories]
    old, new = [], []
    for i, h in enumerate(histories):
        prior = h.previous_rates if h.schedule_count == h.process_count else h.rates
        fallback = float(h.has_draft or h.schedule_count > h.process_count)
        old.extend(prior or [fallback] * h.drafts)
        new.extend((r, i, j) for j, r in enumerate(h.rates or [fallback] * h.drafts))
    # Enumerate every subset at each Q, independently of production's sorted queue.
    best, selected = -1., 0
    for k in range(len(old) + 1):
        matching = [(g, t) for g, t in curve if g >= n + k]
        if not matching:
            continue
        g, t = matching[0]
        expectation = n + max(sum(old[j] for j in subset) for subset in combinations(range(len(old)), k))
        throughput = expectation / (t + max(0., slope) * g + sampler)
        if throughput > best + 1e-14:
            best, selected = throughput, k
    # Exhaustive allocation; lexicographic token identities resolve equal scores.
    preferred = sorted(range(len(new)), key=lambda j: (-new[j][0], new[j][1], new[j][2]))
    rank = {j: i for i, j in enumerate(preferred)}
    subsets = list(combinations(range(len(new)), selected))
    subset = min(subsets, key=lambda ids: (-sum(new[j][0] for j in ids), tuple(sorted(rank[j] for j in ids))))
    lengths = [1] * n
    for j in subset:
        lengths[new[j][1]] += 1
    return n + selected, lengths


def conformance_report():
    rng = np.random.default_rng(2026)
    max_logits = max_confidence = max_projection = 0.
    for case in range(24):
        steps, width, vocab, rank = 3, 4, 7, case % 3
        args = [rng.normal(size=(steps, width)), rng.normal(size=width),
                rng.normal(size=(width, vocab)), rng.normal(size=(vocab, rank)),
                rng.normal(size=(rank, vocab)), rng.normal(size=width + rank),
                case % vocab, np.array([.3, 1., 2.])]
        actual, expected = draft_head(*args), scalar_head(*args)
        assert actual["tokens"] == expected["tokens"]
        max_logits = max(max_logits, float(np.max(abs(actual["logits"] - expected["logits"]))))
        max_confidence = max(max_confidence, float(np.max(abs(actual["confidence"] - expected["confidence"]))))
        layers = [rng.normal(size=(2, width)) for _ in range(2)]
        projection, norm = rng.normal(size=(2 * width, width)), rng.normal(size=width)
        rows = []
        for row in range(2):
            features = list(layers[0][row]) + list(layers[1][row])
            projected = [sum(features[j] * projection[j, k] for j in range(2 * width)) for k in range(width)]
            scale = math.sqrt(sum(x*x for x in projected) / width + 1e-6)
            rows.append([x / scale * norm[k] for k, x in enumerate(projected)])
        max_projection = max(max_projection, float(np.max(abs(project_features(layers, projection, norm) - rows))))
    for case in range(80):
        histories = []
        for i in range(2):
            nd = int(rng.integers(0, 4))
            rates = tuple(rng.uniform(.05, .95, nd)) if case % 5 else ()
            previous = tuple(rng.uniform(.05, .95, nd)) if case % 7 else ()
            histories.append(DraftHistory(nd, rates, previous, 3 + (case + i) % 2, 3, bool(case % 3)))
        curve = [(2, 1.), (4, 1.2), (8, 2.1)] if case % 11 else []
        slope, sampler = (-.1 if case % 2 else .03), .2
        actual = schedule(histories, curve, slope, sampler)
        total, lengths = exhaustive_schedule(histories, curve, slope, sampler)
        assert (actual["total_q"], actual["seqlens_q"]) == (total, lengths)
    assert max(max_logits, max_confidence, max_projection) < 1e-12
    example = schedule([DraftHistory(3, (.9, .8, .4), (.8, .6, .2)),
                        DraftHistory(3, (.7, .3, .1), (.7, .5, .1))], [(2, 1.), (4, 1.1), (8, 2.5)])
    return dict(upstream_commit="774ce244b87cc0eb8d4b94ab9f5de2ab22049f01",
                head_cases=24, exhaustive_schedule_cases=80, max_logit_error=max_logits,
                max_confidence_error=max_confidence, max_projection_error=max_projection,
                example=example)
