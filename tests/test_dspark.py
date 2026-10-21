from dataclasses import replace
import unittest

import numpy as np

from nano_infer.dspark import DraftHistory, draft_head, schedule
from nano_infer.dspark_reference import conformance_report, scalar_head


class DSparkTests(unittest.TestCase):
    def test_scalar_equations_and_exhaustive_allocation(self):
        report = conformance_report()
        self.assertLess(report["max_logit_error"], 1e-12)

    def test_generation_separation_and_pipeline(self):
        curve = [(2, 1.), (4, 1.1), (8, 5.)]
        histories = [DraftHistory(3, (.9, .8, .1), (.8, .6, .2), 4, 4),
                     DraftHistory(3, (.7, .3, .1), (.7, .5, .1), 4, 4)]
        base = schedule(histories, curve)
        changed = [replace(h, rates=tuple(reversed(h.rates))) for h in histories]
        self.assertEqual(schedule(changed, curve)["total_q"], base["total_q"])
        # Advancing schedule without process means rates now contain N-2.
        pending = [replace(h, schedule_count=5, rates=h.previous_rates) for h in histories]
        self.assertEqual(schedule(pending, curve)["total_q"], base["total_q"])
        switched = [replace(histories[0], rates=(0., 0., 0.)),
                    replace(histories[1], rates=(1., 1., 1.))]
        self.assertEqual(schedule(switched, curve)["seqlens_q"], [1, 3])

    def test_head_causality_and_mutation_sensitivity(self):
        rng = np.random.default_rng(42)
        args = [rng.normal(size=(4, 6)), np.ones(6), rng.normal(size=(6, 9)),
                rng.normal(size=(9, 3)), rng.normal(size=(3, 9)), rng.normal(size=9),
                2, np.array([.4, .8, 1.2, 1.6])]
        base = draft_head(*args)
        future = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
        future[0][2:] *= 10
        future[7][2:] *= 10
        changed = draft_head(*future)
        np.testing.assert_array_equal(base["logits"][:2], changed["logits"][:2])
        np.testing.assert_array_equal(base["confidence"][:2], changed["confidence"][:2])
        # Three plausible wrong implementations must disagree with scalar oracle.
        expected = scalar_head(*args)
        no_markov = list(args)
        no_markov[4] = np.zeros_like(args[4])
        self.assertGreater(np.max(abs(draft_head(*no_markov)["logits"] - expected["logits"])), .1)
        normalized = list(args)
        normalized[0] = args[0] / np.sqrt(np.mean(args[0]**2, axis=-1, keepdims=True) + 1e-6)
        self.assertGreater(np.max(abs(draft_head(*normalized)["confidence"] - expected["confidence"])), .01)
        flat_temp = list(args)
        flat_temp[7] = np.ones(4)
        self.assertGreater(np.max(abs(draft_head(*flat_temp)["confidence"] - expected["confidence"])), .01)
        floor = list(args)
        floor[7] = np.array([0., -1., 1e-9, 1e-5])
        np.testing.assert_allclose(draft_head(*floor)["confidence"], scalar_head(*floor)["confidence"], atol=1e-14)

    def test_fallback_ties_and_invalid_history(self):
        self.assertEqual(schedule([DraftHistory(2)], [ (1, 1.), (4, 1.) ])["total_q"], 3)
        self.assertEqual(schedule([DraftHistory(2, has_draft=False)], [(1, 1.), (4, 1.)])["total_q"], 1)
        self.assertEqual(schedule([DraftHistory(2)], [])["seqlens_q"], [3])
        with self.assertRaises(ValueError):
            schedule([DraftHistory(2, schedule_count=3)], [(4, 1.)])
        with self.assertRaises(ValueError):
            schedule([DraftHistory(2, rates=(float("nan"), .5))], [(4, 1.)])


if __name__ == "__main__":
    unittest.main()
