import tempfile
import unittest
from pathlib import Path

import numpy as np

from nano_infer.engine import Engine, Request
from nano_infer.model import TinyModel, encode, greedy
from nano_infer.training import Adam, load_checkpoint, loss_and_grad, parameters, save_checkpoint
from bench import benchmark


class TrainingTests(unittest.TestCase):
    def test_batch_matches_serving_and_all_parameter_gradients(self):
        model = TinyModel(width=6, seed=11)
        x = np.array([encode("cache"), encode("model")])
        y = np.roll(x, -1, axis=1)
        _, grads, logits = loss_and_grad(model, x, y)
        for row, expected in zip(x, logits):
            np.testing.assert_allclose(model.forward(row.tolist(), model.cache()), expected, atol=1e-12)
        # Largest-magnitude entry in every tensor exercises every weight family.
        # Top-2 IDs remain locally constant around this finite-difference fixture.
        for name, value in parameters(model).items():
            index = np.unravel_index(np.abs(grads[name]).argmax(), value.shape)
            old, eps = value[index], 1e-5
            value[index] = old + eps
            plus = loss_and_grad(model, x, y)[0]
            value[index] = old - eps
            minus = loss_and_grad(model, x, y)[0]
            value[index] = old
            self.assertAlmostEqual((plus - minus) / (2 * eps), grads[name][index], places=6, msg=name)

    def test_training_checkpoint_and_serving(self):
        model = TinyModel(width=12)
        x = np.array([encode("the cache stores keys. ")])
        y = np.roll(x, -1, axis=1)
        initial = loss_and_grad(model, x, y)[0]
        optimizer = Adam(model, lr=.01)
        for _ in range(100):
            _, grads, _ = loss_and_grad(model, x, y)
            optimizer.update(grads)
        self.assertLess(loss_and_grad(model, x, y)[0], initial * .3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_checkpoint(model, path)
            loaded = load_checkpoint(path)
            for name, value in parameters(model).items():
                np.testing.assert_array_equal(parameters(loaded)[name], value)
            prompt = encode("the cache ")
            expected = greedy(model, prompt, 30, cached=False)
            for draft in (0, 4):
                request = Request("trained", prompt, 30, temperature=0)
                engine = Engine(loaded, max_draft=draft)
                engine.run([request])
                self.assertEqual(request.output, expected)
                self.assertFalse(engine.caches)
            measured = benchmark(repeats=1, max_tokens=4, checkpoint=path)
            self.assertIsNotNone(measured["workload"]["checkpoint_sha256"])
            self.assertTrue(all(r["equal_full_recompute"] for r in measured["results"].values()))
            tensors = parameters(model)
            tensors["head"][0, 0] = np.nan
            save_checkpoint(model, path)
            with self.assertRaisesRegex(ValueError, "invalid checkpoint tensor"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
