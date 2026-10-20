import json
import unittest

import numpy as np

from nano_infer.decoding import ChoiceGrammar, ConfidenceBudget, sample, speculative
from nano_infer.engine import Engine, Request
from nano_infer.labs import LABS, make_requests
from nano_infer.model import VOCAB, TinyModel, bounded_replay, encode, greedy
from nano_infer.parallel import moe_forward, rebalance, split_attention


class CoreTests(unittest.TestCase):
    def test_every_lesson(self):
        for name, lab in LABS.items():
            with self.subTest(lesson=name):
                json.dumps(lab())

    def test_chunk_prefill_matches_full(self):
        for window in (None, 4):
            for size in (1, 3, 7):
                with self.subTest(window=window, chunk=size):
                    model = TinyModel(window=window)
                    tokens = encode("attention cache")
                    full = model.forward(tokens, model.cache())
                    cache = model.cache()
                    chunks = [model.forward(tokens[i:i + size], cache) for i in range(0, len(tokens), size)]
                    np.testing.assert_allclose(full, np.concatenate(chunks), atol=1e-12, rtol=1e-12)

    def test_causal_block_does_not_read_future_tokens(self):
        model = TinyModel()
        a = model.forward(encode("abcd"), model.cache())
        b = model.forward(encode("abzz"), model.cache())
        np.testing.assert_array_equal(a[:2], b[:2])

    def test_cached_matches_uncached_across_seeds(self):
        for seed in range(4):
            self.assertEqual(greedy(TinyModel(seed), [1, 2, 3], 16),
                             greedy(TinyModel(seed), [1, 2, 3], 16, cached=False))

    def test_speculative_rejection_and_rollback(self):
        for seed in range(5):
            for prompt in ([1], [2, 3, 4]):
                for length in (0, 1, 2, 17):
                    for adaptive in (False, True):
                        with self.subTest(seed=seed, prompt=prompt, length=length, adaptive=adaptive):
                            expected = greedy(TinyModel(seed), prompt, length)
                            actual, trace = speculative(TinyModel(seed), prompt, length, adaptive=adaptive)
                            self.assertEqual(expected, actual)
                            emitted = 0
                            for step in trace:
                                emitted += step["emitted"]
                                self.assertEqual(step["cache_tokens"], len(prompt) + emitted - 1)

    def test_forced_acceptance_and_rejection_paths(self):
        prompt = encode("proof")
        expected = greedy(TinyModel(), prompt, 20)

        class ScriptedDraft:
            # Test fixture knows the oracle continuation; production BigramDraft doesn't.
            def __init__(self, wrong):
                self.position = 0
                self.wrong = wrong
                self.initial = True

            def observe(self, tokens):
                if self.initial:
                    self.initial = False
                else:
                    self.position += len(tokens) - 1

            def propose(self, last, count):
                result = expected[self.position:self.position + count]
                if result and self.wrong:
                    result[0] = (result[0] + 1) % len(VOCAB)
                return result

        for wrong in (False, True):
            actual, trace = speculative(TinyModel(), prompt, 20, ScriptedDraft(wrong), adaptive=False)
            self.assertEqual(expected, actual)
            if wrong:
                self.assertTrue(any(t["rolled_back"] > 0 for t in trace))
                self.assertEqual(sum(t["accepted"] for t in trace), 0)
            else:
                self.assertTrue(any(t["accepted"] > 0 for t in trace))
                self.assertTrue(all(t["rolled_back"] == 0 for t in trace))

    def test_zero_draft_is_autoregressive(self):
        a, trace = speculative(TinyModel(), [1], 9, max_draft=0)
        self.assertEqual(a, greedy(TinyModel(), [1], 9))
        self.assertTrue(all(t["proposed"] == 0 for t in trace))

    def test_historical_confidence_changes_budget(self):
        good, bad = ConfidenceBudget(), ConfidenceBudget()
        for _ in range(30):
            good.update(5, 5)
            bad.update(5, 0)
        self.assertGreater(good.choose(20), bad.choose(20))
        self.assertEqual(bad.choose(20), 0)

    def test_moe_replicas_preserve_function(self):
        model = TinyModel()
        x = np.random.default_rng(5).normal(size=(37, model.width))
        a, counts, _ = moe_forward(x, model.gate, model.experts)
        for ranks, capacity in ((2, 2), (2, 4), (4, 2)):
            layout = rebalance(counts, ranks, capacity)
            b, _, _ = moe_forward(x, model.gate, model.experts, layout)
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
            self.assertTrue(all(len(row) == capacity for row in layout))

    def test_context_shards_uneven_and_empty(self):
        rng = np.random.default_rng(8)
        q, k, v = rng.normal(size=4), rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
        for parts in (1, 2, 9):
            np.testing.assert_allclose(split_attention(q, k, v, 1),
                                       split_attention(q, k, v, parts), atol=1e-12)

    def test_replay_at_backtrack_boundary(self):
        for length in (1, 5, 29):
            tokens = [i % len(VOCAB) for i in range(length)]
            for backtrack in (0, 3):
                model = TinyModel(window=5)
                full = model.forward(tokens, model.cache())
                compact, _ = bounded_replay(model, tokens, backtrack)
                for delta in range(min(backtrack + 1, length)):
                    np.testing.assert_allclose(full[-1 - delta], compact[-1 - delta], atol=1e-12)

    def test_request_order_batch_size_and_pipeline(self):
        baseline = Engine(TinyModel()).run(make_requests())
        for pipeline in (False, True):
            for batch in (1, 2, 4):
                engine = Engine(TinyModel(), token_budget=3, max_batch=batch, prefill_chunk=2)
                result = engine.run(list(reversed(make_requests())), pipelined=pipeline)
                self.assertEqual(baseline, result)
                self.assertFalse(engine.caches)
                for step in engine.batches:
                    self.assertLessEqual(sum(w["input_tokens"] for w in step), 3)
                    self.assertLessEqual(len(step), batch)

    def test_eos_empty_run_zero_limit_and_duplicate_ids(self):
        eos = greedy(TinyModel(), [1], 1)[0]
        engine = Engine(TinyModel())
        result = engine.run([Request("eos", [1], 10, temperature=0, eos=eos),
                             Request("zero", [2], 0)], pipelined=True)
        self.assertEqual(result, {"eos": [eos], "zero": []})
        self.assertEqual(engine.run([]), {})
        with self.assertRaises(ValueError):
            engine.run([Request("x", [1]), Request("x", [2])])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            encode("")
        with self.assertRaises(ValueError):
            Engine(TinyModel(), token_budget=0)
        with self.assertRaises(ValueError):
            TinyModel(window=0)
        with self.assertRaises(ValueError):
            TinyModel().cache().truncate(1)
        with self.assertRaises(ValueError):
            rebalance([1, 2, 3], ranks=1, slots_per_rank=2)
        with self.assertRaises(ValueError):
            bounded_replay(TinyModel(), [1])
        with self.assertRaises(ValueError):
            sample([1, 2], np.random.default_rng(0), allowed=set())
        with self.assertRaises(ValueError):
            ChoiceGrammar(["a", "ab"], VOCAB)


if __name__ == "__main__":
    unittest.main()
