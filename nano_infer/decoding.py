"""Greedy speculative decoding, historical confidence budgeting, local sampling."""
from collections import Counter, defaultdict

import numpy as np

from .model import softmax


class BigramDraft:
    """Cheap online draft: learn transitions from verified tokens, not target calls."""
    def __init__(self):
        self.transitions = defaultdict(Counter)

    def observe(self, tokens):
        for a, b in zip(tokens, tokens[1:]):
            self.transitions[a][b] += 1

    def propose(self, last, count):
        result = []
        for _ in range(count):
            votes = self.transitions[last]
            last = min(votes, key=lambda t: (-votes[t], t)) if votes else last
            result.append(last)
        return result


class ConfidenceBudget:
    """Choose k from previous rounds' prefix survival rates / an illustrative cost.

    Current proposals are not inspected. DeepInfer additionally uses confidence
    heads, lagged per-request histories and measured runtime curves.
    """
    def __init__(self, maximum=5):
        if maximum < 0:
            raise ValueError("maximum must be nonnegative")
        self.maximum = maximum
        self.trials = np.zeros(maximum)
        self.survived = np.zeros(maximum)

    def choose(self, remaining):
        limit = min(self.maximum, max(0, remaining - 1))
        rates = (self.survived + 1) / (self.trials + 2)
        rates = np.minimum.accumulate(rates)  # prefix survival is non-increasing
        expected = np.concatenate([[1.0], 1 + np.cumsum(rates[:limit])])
        # Arbitrary cost units, NOT milliseconds or a measured GPU model.
        cost = 1 + 0.25 * np.arange(limit + 1)
        return int(np.argmax(expected / cost))

    def update(self, proposed, accepted):
        self.trials[:proposed] += 1
        self.survived[:accepted] += 1


def greedy_prefix(logits, proposals):
    """Return the verified prefix plus correction/bonus; no cache mutation.

    Row i predicts proposal i, provided every earlier proposal was accepted.
    Both the single-request lesson and the scheduler use this same rule.
    """
    if len(logits) != len(proposals) + 1:
        raise ValueError("verification needs pending + proposal logits")
    predicted = np.argmax(logits, axis=-1).tolist()
    accepted = 0
    for proposal, target in zip(proposals, predicted):
        if proposal != target:
            break
        accepted += 1
    return list(proposals[:accepted]) + [predicted[accepted]], accepted


def speculative(model, prompt, max_tokens, draft=None, max_draft=5, adaptive=True):
    """Target-equivalent GREEDY decoding with a pending-token cache invariant.

    Before each round the cache contains history[:-1]; pending == history[-1].
    Verify [pending] + drafts in one causal forward. Keep accepted KV, discard
    rejected suffix, and leave the correction/bonus token pending for next round.
    No rejection-sampling implementation for temperature > 0 is claimed here.
    """
    if not prompt or max_tokens < 0 or max_draft < 0:
        raise ValueError("invalid prompt, max_tokens or max_draft")
    if max_tokens == 0:
        return [], []
    draft = draft if draft is not None else BigramDraft()
    draft.observe(prompt)
    cache = model.cache()
    if len(prompt) > 1:
        model.forward(prompt[:-1], cache)
    pending = prompt[-1]
    budget = ConfidenceBudget(max_draft)
    output, trace = [], []
    while len(output) < max_tokens:
        remaining = max_tokens - len(output)
        count = budget.choose(remaining) if adaptive else min(max_draft, remaining - 1)
        proposals = draft.propose(pending, count)
        before = len(cache)
        logits = model.forward([pending] + proposals, cache)
        committed, accepted = greedy_prefix(logits, proposals)
        cache.truncate(before + 1 + accepted)
        draft.observe([pending] + committed)
        output.extend(committed)
        trace.append(dict(proposed=count, accepted=accepted,
                          rolled_back=count - accepted, emitted=len(committed),
                          cache_tokens=len(cache)))
        pending = committed[-1]
        budget.update(count, accepted)
    return output, trace


def sample(logits, rng, temperature=1.0, allowed=None):
    """Request-local RNG; optional finite-vocabulary mask applied before sampling."""
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    values = np.array(logits, dtype=float, copy=True)
    if allowed is not None:
        mask = np.zeros(len(values), dtype=bool)
        for token in allowed:
            if not 0 <= token < len(values):
                raise ValueError("allowed token outside vocabulary")
            mask[token] = True
        values[~mask] = -np.inf
    if not np.isfinite(values).any():
        raise ValueError("no legal token")
    if temperature == 0:
        return int(np.argmax(values))
    return int(rng.choice(len(values), p=softmax(values / temperature)))


class ChoiceGrammar:
    """Trie-like character constraint over a finite list of complete strings.

    Demonstrates constrained logits; not a JSON Schema/regex/EBNF compiler.
    """
    def __init__(self, choices, vocabulary):
        if not choices or any(not c for c in choices):
            raise ValueError("nonempty choices required")
        if any(any(c not in vocabulary for c in choice) for choice in choices):
            raise ValueError("choice outside vocabulary")
        if any(a != b and b.startswith(a) for a in choices for b in choices):
            raise ValueError("choices must be prefix-free (no EOS token in toy grammar)")
        self.choices, self.vocabulary = choices, vocabulary

    def allowed(self, prefix):
        return {self.vocabulary.index(c[len(prefix)]) for c in self.choices
                if c.startswith(prefix) and len(c) > len(prefix)}

    def done(self, prefix):
        return prefix in self.choices
