import unittest

from bench import benchmark
from nano_infer.engine import Engine, Request
from nano_infer.model import VOCAB, TinyModel, encode, greedy


class EngineTests(unittest.TestCase):
    def test_integrated_speculation_across_budgets_windows_and_pipeline(self):
        prompts = [[1], encode("nano"), encode("longer prompt")]
        for seed in (0, 7):
            for window in (None, 4):
                expected = {str(i): greedy(TinyModel(seed, window=window), p, 13, cached=False)
                            for i, p in enumerate(prompts)}
                for budget in (1, 2, 7):
                    for adaptive in (False, True):
                        for pipeline in (False, True):
                            with self.subTest(seed=seed, window=window, budget=budget,
                                              adaptive=adaptive, pipeline=pipeline):
                                engine = Engine(TinyModel(seed, window=window), token_budget=budget,
                                                prefill_chunk=2, max_draft=4, adaptive=adaptive)
                                reqs = [Request(str(i), p, 13, temperature=0)
                                        for i, p in reversed(list(enumerate(prompts)))]
                                self.assertEqual(engine.run(reqs, pipeline), expected)
                                emitted = {str(i): 0 for i in range(len(prompts))}
                                cursors = {str(i): 0 for i in range(len(prompts))}
                                for batch in engine.batches:
                                    self.assertLessEqual(sum(r["input_tokens"] for r in batch), budget)
                                    self.assertEqual(len({r["request"] for r in batch}), len(batch))
                                    for row in batch:
                                        key = row["request"]
                                        emitted[key] += len(row["emitted_tokens"])
                                        if row["phase"] == "prefill":
                                            cursors[key] += row["input_tokens"]
                                        self.assertEqual(row["cache_after"],
                                                         cursors[key] + max(0, emitted[key] - 1))
                                        self.assertEqual(row["cache_after"], row["cache_before"] +
                                                         row["input_tokens"] - row["rolled_back"])
                                self.assertTrue(engine.is_finished())
                                self.assertFalse(engine.rngs or engine.drafts or engine.budgets)

    def test_zero_and_short_output_limits(self):
        for length in (0, 1, 2, 5):
            for pipeline in (False, True):
                engine = Engine(TinyModel(), max_draft=8)
                req = Request("a", [1], length, temperature=0)
                self.assertEqual(engine.run([req], pipeline)["a"], greedy(TinyModel(), [1], length))
                self.assertFalse(engine.caches)

    def test_forced_prefix_acceptance_rejection_and_internal_eos(self):
        prompt = encode("proof")
        seed = next(s for s in range(20)
                    if (out := greedy(TinyModel(s), prompt, 12))[2] not in out[:2])
        expected = greedy(TinyModel(seed), prompt, 12)

        class ScriptedDraft:
            # Test-only oracle. Runtime drafts have no access to expected outputs.
            def __init__(self, wrong):
                self.position, self.wrong = 1, wrong

            def propose(self, last, count):
                tokens = expected[self.position:self.position + count]
                if tokens and self.wrong:
                    tokens[0] = (tokens[0] + 1) % len(VOCAB)
                return tokens

            def observe(self, tokens):
                self.position += len(tokens) - 1

        for wrong, eos in ((False, None), (True, None), (False, expected[2])):
            engine = Engine(TinyModel(seed), token_budget=16, prefill_chunk=16, max_draft=4)
            req = Request("a", prompt, 12, temperature=0, eos=eos)
            engine.add_request(req)
            engine.step()  # prefill emits expected[0]
            engine.drafts["a"] = ScriptedDraft(wrong)
            while not engine.is_finished():
                engine.step()
            self.assertEqual(req.output, expected if eos is None else expected[:3])
            rows = [r for batch in engine.batches for r in batch if r["phase"] == "verify"]
            self.assertTrue(rows)
            if eos is not None:
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["emitted_tokens"], expected[1:3])
                self.assertEqual(rows[0]["cache_after"], len(prompt) + 2)
                self.assertGreater(rows[0]["rolled_back"], 0)
            elif wrong:
                self.assertTrue(all(r["accepted"] == 0 for r in rows))
            else:
                self.assertTrue(all(r["rolled_back"] == 0 for r in rows))

    def test_live_arrival_progress_and_run_guard(self):
        engine = Engine(TinyModel(), token_budget=4, prefill_chunk=2, max_draft=3)
        long = Request("long", encode("a long prompt " * 4), 8, temperature=0)
        short = Request("short", [1], 2, temperature=0)
        engine.add_request(long)
        engine.step()
        engine.add_request(short)
        with self.assertRaises(ValueError):
            engine.run([])  # must not discard already admitted work
        completions = []
        while not engine.is_finished():
            completions.extend(r["request"] for r in engine.step() if r["done"])
        self.assertEqual(completions, ["short", "long"])
        for req in (long, short):
            self.assertEqual(req.output, greedy(TinyModel(), req.prompt, req.max_tokens, cached=False))
        self.assertEqual(engine.step(), [])
        engine.add_request(Request("later", [2], 1, temperature=0))
        self.assertEqual(len(engine.step()[0]["emitted_tokens"]), 1)

    def test_admission_validation_is_atomic(self):
        engine = Engine(TinyModel(), max_draft=4)
        for req in (Request("bad", [1], temperature=1), Request("bad", [-1], temperature=0),
                    Request("bad", [1.5], temperature=0),
                    Request("bad", [1], seed=-1, temperature=0)):
            with self.assertRaises(ValueError):
                engine.add_request(req)
            self.assertFalse(engine.requests or engine.ready or engine.caches)
        engine.add_request(Request("ok", [1], temperature=0))
        with self.assertRaises(ValueError):
            engine.add_request(Request("ok", [2], temperature=0))
        self.assertEqual(len(engine.caches), 1)

    def test_execution_failure_clears_state_and_cannot_resume(self):
        class BrokenModel(TinyModel):
            def forward(self, tokens, cache, trace=None):
                super().forward(tokens, cache, trace)
                raise RuntimeError("failure after KV mutation")

        for pipeline in (False, True):
            engine = Engine(BrokenModel(), max_draft=4)
            with self.assertRaises(RuntimeError):
                engine.run([Request("a", [1], temperature=0),
                            Request("b", [2], temperature=0)], pipeline)
            self.assertFalse(engine.caches or engine.ready or engine.drafts or engine.rngs)
            with self.assertRaises(RuntimeError):
                engine.step()
            engine.model = TinyModel()
            self.assertEqual(engine.run([Request("fresh", [1], 1, temperature=0)])["fresh"],
                             greedy(TinyModel(), [1], 1))

    def test_benchmark_uses_one_workload_and_checks_all_policies(self):
        report = benchmark(repeats=1, max_tokens=4, seed=0)
        self.assertEqual(len(report["results"]), 4)
        self.assertEqual(sum(map(len, report["oracle_outputs"].values())), 16)
        for result in report["results"].values():
            self.assertTrue(result["equal_full_recompute"])
            self.assertEqual(len(result["samples"]), 1)
            self.assertGreater(result["median_seconds"], 0)
        # No relative speed assertions: noise and overhead are legitimate outcomes.


if __name__ == "__main__":
    unittest.main()
