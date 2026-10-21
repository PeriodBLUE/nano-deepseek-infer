import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request as HTTPRequest, urlopen

import numpy as np

from nano_infer.engine import Engine, Request
from nano_infer.model import TinyModel, encode, greedy
from nano_infer.mla import MLA, mla_lab
from nano_infer.paging import BlockPool, PagedCache, PrefixStore
from nano_infer.service import BusyError, Service
from serve import start_server


class SystemsTests(unittest.TestCase):
    def test_paged_fork_rollback_cow_and_atomic_oom(self):
        pool = BlockPool(4, 3, 2)
        parent = PagedCache(pool)
        x = np.arange(10.).reshape(5, 2)
        parent.append(x, x + 100)
        child = parent.fork()
        self.assertEqual(pool.snapshot()["used_blocks"], 2)
        child.truncate(4)
        child.append(np.ones((2, 2)) * -1, np.ones((2, 2)))
        np.testing.assert_array_equal(parent.k, x)
        self.assertEqual(pool.copies, 1)
        before, blocks = child.k, child.blocks.copy()
        with self.assertRaises(MemoryError):
            child.append(np.ones((7, 2)), np.ones((7, 2)))
        np.testing.assert_array_equal(child.k, before)
        self.assertEqual(child.blocks, blocks)
        child.release()
        parent.release()
        self.assertEqual(len(pool.free), 4)
        self.assertEqual(len(set(pool.free)), 4)

    def test_prefix_lru_and_live_branch_survives_eviction(self):
        model = TinyModel()
        pool = BlockPool(12, 4, model.width)
        store = PrefixStore(pool, capacity=1)
        prefix = encode("nano ")
        cache = PagedCache(pool)
        model.forward(prefix, cache)
        store.put(prefix, cache)
        cache.release()
        branch, length = store.get(prefix + encode("cache"))
        self.assertEqual(length, len(prefix))
        snapshot = branch.k.copy()
        other = PagedCache(pool)
        model.forward(encode("a"), other)
        store.put(encode("a"), other)  # evict snapshot while branch is alive
        np.testing.assert_array_equal(branch.k, snapshot)
        logits = model.forward(encode("cache"), branch)
        full = model.forward(prefix + encode("cache"), model.cache())
        np.testing.assert_allclose(logits, full[-5:], atol=1e-12)
        branch.release(); other.release(); store.clear()
        self.assertFalse(np.any(pool.refs))

    def test_ragged_batch_masks_positions_and_paged_engine(self):
        for window in (None, 3):
            model = TinyModel(window=window)
            prefixes = [encode("abc"), encode("longer prefix"), [1]]
            chunks = [[2], [3, 4, 5], [6, 7]]
            caches = [model.cache(offset=i) for i in range(3)]
            for prefix, cache in zip(prefixes, caches):
                model.forward(prefix, cache)
            refs = [c.fork() for c in caches]
            expected = [model.forward(chunk, c) for chunk, c in zip(chunks, refs)]
            actual = model.forward_batch(chunks, caches)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, atol=1e-12)
            for batched in (False, True):
                pool = BlockPool(100, 4, model.width)
                engine = Engine(model, batched=batched, pool=pool, max_draft=4, token_budget=7)
                requests = [Request(str(i), p, 19, temperature=0) for i, p in enumerate(prefixes)]
                engine.run(requests, pipelined=True)
                for req in requests:
                    self.assertEqual(req.output, greedy(TinyModel(window=window), req.prompt, 19, cached=False))
                self.assertFalse(np.any(pool.refs))
                self.assertFalse(engine.reservations)

    def test_mla_absorption_incremental_and_causal(self):
        self.assertLess(mla_lab()["max_abs_error"], 1e-12)
        for seed in range(3):
            model = MLA(seed=seed)
            x = np.random.default_rng(seed).normal(size=(11, 24))
            base = model.forward(x, model.cache(), absorbed=False)
            changed = x.copy(); changed[6:] *= 100
            np.testing.assert_allclose(base[:6], model.forward(changed, model.cache())[:6], atol=1e-12)
            cache = model.cache()
            split = np.concatenate([model.forward(x[:4], cache), model.forward(x[4:], cache)])
            np.testing.assert_allclose(split, base, atol=1e-12)

    def test_service_queue_cancel_deadline_metrics_and_backpressure(self):
        now = [0.]
        class TimedModel(TinyModel):
            def forward_batch(self, chunks, caches):
                result = super().forward_batch(chunks, caches)
                now[0] += .1
                return result
        model = TimedModel()
        pool = BlockPool(3, 4, model.width)
        service = Service(Engine(model, batched=True, pool=pool), max_pending=2, clock=lambda: now[0])
        a = service.submit("nano", max_tokens=6)
        b = service.submit("nano", max_tokens=6)
        with self.assertRaises(BusyError): service.submit("nano", max_tokens=6)
        service.step()
        self.assertEqual(service.snapshot(b)["status"], "queued")
        self.assertAlmostEqual(service.snapshot(a)["metrics"]["ttft_seconds"], .1)
        service.cancel(a)
        service.step()
        self.assertEqual(service.snapshot(b)["status"], "running")
        self.assertAlmostEqual(service.snapshot(b)["metrics"]["queue_seconds"], .1)
        while not service.engine.is_finished(): service.step()
        metrics = service.snapshot(b)["metrics"]
        self.assertAlmostEqual(metrics["tpot_seconds"], .1)
        c = service.submit("nano", max_tokens=6, timeout=.01)
        now[0] += 1.
        service.step()
        self.assertEqual(service.snapshot(c)["status"], "timed_out")
        self.assertIsNone(service.snapshot(c)["metrics"]["ttft_seconds"])
        self.assertFalse(np.any(pool.refs))
        d = service.submit("nano", max_tokens=6, timeout=.01)
        service.step()  # deadline passes during a forward; cleanup at commit boundary
        self.assertEqual(service.snapshot(d)["status"], "timed_out")
        self.assertFalse(np.any(pool.refs) or service.engine.reservations)
        zero = service.submit("nano", max_tokens=0)
        service.step()
        self.assertEqual(service.snapshot(zero)["status"], "completed")
        self.assertIsNone(service.snapshot(zero)["metrics"]["ttft_seconds"])
        with self.assertRaises(ValueError): service.submit("nano", max_tokens=1000)

    def test_service_failure_releases_pool_and_recovers(self):
        class Broken(TinyModel):
            def forward_batch(self, chunks, caches):
                super().forward_batch(chunks, caches)
                raise RuntimeError("injected execution failure")
        pool = BlockPool(32, 4, 24)
        service = Service(Engine(Broken(), batched=True, pool=pool))
        key = service.submit("nano", 4)
        service.step()
        self.assertEqual(service.snapshot(key)["status"], "failed")
        self.assertFalse(np.any(pool.refs))
        service.engine.model = TinyModel()
        key = service.submit("nano", 4)
        for _ in range(4): service.step()
        self.assertEqual(service.snapshot(key)["status"], "completed")

    def test_http_submit_poll_cancel_and_invalid_input(self):
        model = TinyModel()
        service = Service(Engine(model, batched=True, pool=BlockPool(64, 4, 24)))
        server, close = start_server(service, port=0)
        base = f"http://127.0.0.1:{server.server_port}"
        def call(path, method="GET", body=None):
            data = None if body is None else json.dumps(body).encode()
            with urlopen(HTTPRequest(base + path, data=data, method=method), timeout=5) as response:
                return json.load(response)
        try:
            key = call("/requests", "POST", dict(prompt="nano", max_tokens=8))["id"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                result = call("/requests/" + key)
                if result["status"] == "completed": break
                time.sleep(.005)
            self.assertEqual(result["tokens"], greedy(TinyModel(), encode("nano"), 8))
            self.assertEqual(call("/health")["pool"]["used_blocks"], 0)
            self.assertEqual(call("/requests/" + key, "DELETE")["status"], "completed")
            with self.assertRaises(HTTPError) as error:
                call("/requests", "POST", dict(prompt="nano", max_tokens=-1))
            self.assertEqual(error.exception.code, 400)
            key = call("/requests", "POST", dict(prompt="nano", max_tokens=200))["id"]
            self.assertIn(call("/requests/" + key, "DELETE")["status"], ("cancelled", "completed"))
        finally:
            close()
        self.assertFalse(np.any(service.engine.pool.refs))


if __name__ == "__main__":
    unittest.main()
