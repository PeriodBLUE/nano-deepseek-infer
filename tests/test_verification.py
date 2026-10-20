import unittest

import numpy as np

from nano_infer.model import TinyModel, encode, greedy
from nano_infer.prefix import generate_shared_prefix
from nano_infer.verification import reference_logits, shape_tour, verification_report


class VerificationTests(unittest.TestCase):
    def test_oracle_agrees_across_positions_windows_and_seeds(self):
        for seed in range(3):
            for window in (None, 1, 5):
                for offset in (0, 17):
                    model = TinyModel(seed, width=12, window=window)
                    tokens = encode("independent")
                    expected = reference_logits(model, tokens, offset=offset)
                    cache = model.cache(offset=offset)
                    actual = np.concatenate([model.forward(tokens[:2], cache),
                                             model.forward(tokens[2:6], cache),
                                             model.forward(tokens[6:], cache)])
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_verifier_detects_mutations_and_records_provenance(self):
        report = verification_report()
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["cases"]), 4)
        for case in report["cases"]:
            self.assertTrue(all(x["detected"] for x in case["faults"].values()))
        self.assertIn("model.py", report["source_sha256"])
        self.assertIn("verification.py", report["source_sha256"])
        self.assertTrue(all(len(digest) == 64 for digest in report["source_sha256"].values()))

    def test_trace_observes_shapes_without_changing_results(self):
        tour = shape_tour()
        prefill = {x["stage"]: x["shape"] for x in tour["prefill"]}
        decode = {x["stage"]: x["shape"] for x in tour["decode"]}
        self.assertEqual(prefill["attention_scores"], [4, 4])
        self.assertEqual(decode["attention_scores"], [1, 5])
        self.assertEqual(decode["key_cache"], [5, 24])
        model = TinyModel()
        trace = []
        a = model.forward([1, 2, 3], model.cache(), trace=trace)
        b = model.forward([1, 2, 3], model.cache())
        np.testing.assert_array_equal(a, b)

    def test_prefix_forks_are_independent_on_append_and_rollback(self):
        model = TinyModel()
        parent = model.cache(offset=10)
        model.forward([1, 2, 3], parent)
        left, right = parent.fork(), parent.fork()
        self.assertFalse(np.shares_memory(left.k, parent.k))
        self.assertFalse(np.shares_memory(left.v, right.v))
        model.forward([4, 5], left)
        left.truncate(1)
        self.assertEqual((len(parent), len(right), len(left)), (3, 3, 1))
        self.assertEqual((parent.offset, right.offset, left.offset), (10, 10, 10))
        np.testing.assert_array_equal(parent.k, right.k)

    def test_shared_prefix_matches_full_requests_and_saves_work(self):
        for window in (None, 4):
            for count in (0, 1, 8):
                model, baseline = TinyModel(window=window), TinyModel(window=window)
                prefix, suffixes = [1, 2, 3], [[], [4], [5, 6]]
                a = generate_shared_prefix(model, prefix, suffixes, count)
                b = [greedy(baseline, prefix + suffix, count) for suffix in suffixes]
                self.assertEqual(a, b)
                if count:
                    self.assertEqual(baseline.projected_tokens - model.projected_tokens, 6)
                else:
                    self.assertEqual(model.calls, 0)

    def test_prefix_empty_batch_and_validation(self):
        model = TinyModel()
        self.assertEqual(generate_shared_prefix(model, [1], [], 3), [])
        self.assertEqual(model.calls, 0)
        with self.assertRaises(ValueError):
            generate_shared_prefix(model, [], [[1]], 3)
        with self.assertRaises(ValueError):
            generate_shared_prefix(model, [1], [[2]], -1)


if __name__ == "__main__":
    unittest.main()
