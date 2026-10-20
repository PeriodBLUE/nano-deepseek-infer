"""Numerical MoE dispatch/combine and single-process parallelism simulations."""
import numpy as np


def rebalance(loads, ranks=2, slots_per_rank=3):
    """Replicate hot logical experts, then greedily place physical copies.

    Simplification: one aggregate load vector, no multi-step optimization or
    migration costs. Return physical-to-logical expert IDs grouped by rank.
    """
    loads = np.asarray(loads, dtype=float)
    if loads.ndim != 1 or not len(loads) or np.any(loads < 0):
        raise ValueError("loads must be a nonempty nonnegative vector")
    if ranks < 1 or slots_per_rank < 1 or ranks * slots_per_rank < len(loads):
        raise ValueError("not enough physical slots")
    copies = np.ones(len(loads), dtype=int)
    for _ in range(ranks * slots_per_rank - len(loads)):
        copies[np.argmax(loads / copies)] += 1
    jobs = [(float(loads[e] / copies[e]), e) for e in range(len(loads)) for _ in range(copies[e])]
    jobs.sort(key=lambda job: (-job[0], job[1]))
    placement = [[] for _ in range(ranks)]
    rank_loads = np.zeros(ranks)
    for load, expert in jobs:
        eligible = [r for r in range(ranks) if len(placement[r]) < slots_per_rank]
        rank = min(eligible, key=lambda r: (rank_loads[r], r))
        placement[rank].append(expert)
        rank_loads[rank] += load
    return placement


def moe_forward(x, gate, experts, placement=None):
    """Top-2 softmax router -> physical dispatch -> expert -> weighted combine.

    Replicas use exactly the same expert weights. Logical selection and combine
    order do not change when physical placement changes.
    """
    routes = np.argsort(-(x @ gate), axis=-1, kind="stable")[:, :2]
    selected = np.take_along_axis(x @ gate, routes, axis=1)
    weights = np.exp(selected - selected.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    if placement is None:
        placement = [list(range(len(experts)))]
    copies = {e: [] for e in range(len(experts))}
    for rank, slots in enumerate(placement):
        for slot, expert in enumerate(slots):
            if expert not in copies:
                raise ValueError("unknown expert in placement")
            copies[expert].append((rank, slot))
    if any(not slots for slots in copies.values()):
        raise ValueError("every logical expert needs a physical copy")
    jobs = {}
    count = np.zeros(len(experts), dtype=int)
    rank_loads = np.zeros(len(placement), dtype=int)
    for token, route in enumerate(routes):
        for branch, expert in enumerate(route):
            physical = copies[expert][count[expert] % len(copies[expert])]
            count[expert] += 1
            rank_loads[physical[0]] += 1
            jobs.setdefault((physical, int(expert)), []).append((token, branch))
    contributions = np.zeros((len(x), 2, x.shape[1]))
    for (_, expert), entries in jobs.items():
        rows = [i for i, _ in entries]
        w1, w2 = experts[expert]
        values = np.tanh(x[rows] @ w1) @ w2
        for value, (token, branch) in zip(values, entries):
            contributions[token, branch] = weights[token, branch] * value
    return contributions.sum(axis=1), count, rank_loads


def split_attention(q, k, v, parts=2):
    """Split K/V; merge local softmax sufficient statistics exactly in algebra.

    Numerical explanation of context sharding, not DeepInfer balanced CP code.
    q is a single query with all keys already visible.
    """
    if parts < 1 or not len(k):
        raise ValueError("positive parts and nonempty keys required")
    stats = []
    for indices in np.array_split(np.arange(len(k)), parts):
        if not len(indices):
            continue
        scores = k[indices] @ q / len(q)**0.5
        maximum = float(scores.max())
        weights = np.exp(scores - maximum)
        stats.append((maximum, weights.sum(), weights @ v[indices]))
    maximum = max(m for m, _, _ in stats)
    denominator = sum(np.exp(m - maximum) * z for m, z, _ in stats)
    numerator = sum(np.exp(m - maximum) * n for m, _, n in stats)
    return numerator / denominator
