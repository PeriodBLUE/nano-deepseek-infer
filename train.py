"""Train the NumPy attention + MoE model, save weights, reload, and generate."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nano_infer.model import TinyModel, decode, encode, greedy
from nano_infer.training import Adam, load_checkpoint, loss_and_grad, parameters, save_checkpoint


def read_tokens(path):
    return np.array(encode(" ".join(Path(path).read_text(encoding="utf-8").split())), dtype=int)


def batch(tokens, rng, size, context):
    if len(tokens) <= context:
        raise ValueError("text must be longer than context")
    starts = rng.integers(0, len(tokens) - context, size=size)
    rows = starts[:, None] + np.arange(context + 1)
    return tokens[rows[:, :-1]], tokens[rows[:, 1:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--context", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data", type=Path, default=Path("data/inference.txt"))
    parser.add_argument("--validation", type=Path, default=Path("data/validation.txt"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/tiny-trained.npz"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/training.json"))
    args = parser.parse_args()
    if min(args.steps, args.width, args.context, args.batch_size) < 1:
        parser.error("steps, width, context and batch size must be positive")
    train, val = read_tokens(args.data), read_tokens(args.validation)
    rng = np.random.default_rng(args.seed)
    model = TinyModel(seed=args.seed, width=args.width)
    optimizer = Adam(model, args.learning_rate)
    # Fixed evaluation batches; evaluation never changes the training RNG.
    eval_rng = np.random.default_rng(2026)
    tx, ty = batch(train, eval_rng, 16, args.context)
    vx, vy = batch(val, eval_rng, 16, args.context)
    initial = dict(train=loss_and_grad(model, tx, ty)[0], validation=loss_and_grad(model, vx, vy)[0])
    history = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = batch(train, rng, args.batch_size, args.context)
        loss, grads, _ = loss_and_grad(model, x, y)
        norm = optimizer.update(grads)
        if step == 1 or step % 100 == 0 or step == args.steps:
            history.append(dict(step=step, loss=loss, gradient_norm=norm))
            print(f"step {step:5} loss {loss:.4f} grad {norm:.4f}", flush=True)
    elapsed = time.perf_counter() - start
    final = dict(train=loss_and_grad(model, tx, ty)[0], validation=loss_and_grad(model, vx, vy)[0])
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata = dict(config=config, initial_loss=initial, final_loss=final,
                    data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    validation_sha256=hashlib.sha256(args.validation.read_bytes()).hexdigest())
    save_checkpoint(model, args.out, metadata)
    restored = load_checkpoint(args.out)
    prompts = ["the cache ", "the model ", "the draft "]
    samples = []
    for prompt in prompts:
        generated = greedy(restored, encode(prompt), 100)
        if generated != greedy(model, encode(prompt), 100):
            raise AssertionError("checkpoint round trip changed generation")
        samples.append(dict(prompt=prompt, generated=decode(generated)))
        print(prompt + decode(generated))
    report = metadata | dict(parameters=sum(v.size for v in parameters(model).values()),
                             elapsed_seconds=elapsed, history=history, samples=samples,
                             checkpoint_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                             checkpoint_roundtrip_equal=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Loss {initial} -> {final}; checkpoint: {args.out}")


if __name__ == "__main__":
    main()
