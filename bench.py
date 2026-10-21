"""Compare one fixed workload under four execution policies, with correctness gates.

    python bench.py --repeats 5 --json benchmark.json

CPU wall time includes engine admission, planning, execution and commit. Model/
request construction and the full-recompute oracle are outside the timed region.
No GPU claims and no timing thresholds: a policy is allowed to be slower.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from nano_infer.engine import Engine, Request
from nano_infer.model import VOCAB, TinyModel, encode, greedy
from nano_infer.training import load_checkpoint
from nano_infer.paging import BlockPool


def benchmark(repeats=3, max_tokens=24, seed=7, checkpoint=None, batched=False):
    if repeats < 1 or max_tokens < 1:
        raise ValueError("positive repeats and output length required")
    rng = np.random.default_rng(2026)
    prompts = [rng.integers(0, len(VOCAB), size=n).tolist() for n in (4, 13, 29, 53)]
    if checkpoint:
        prompts = [encode(p) for p in ("the model ", "the cache ", "the draft ", "the scheduler ")]
    make_model = lambda: load_checkpoint(checkpoint) if checkpoint else TinyModel(seed)
    oracle = {f"r{i}": greedy(make_model(), p, max_tokens, cached=False)
              for i, p in enumerate(prompts)}
    policies = dict(cached=dict(max_draft=0),
                    fixed_draft=dict(max_draft=4),
                    adaptive_draft=dict(max_draft=4, adaptive=True),
                    pipelined_draft=dict(max_draft=4, pipelined=True))
    samples = {name: [] for name in policies}

    def trial(name):
        policy = policies[name].copy()
        pipelined = policy.pop("pipelined", False)
        model = make_model()
        engine = Engine(model, token_budget=12, max_batch=3, prefill_chunk=4, batched=batched,
                        pool=BlockPool(128, 8, model.width) if batched else None, **policy)
        requests = [Request(f"r{i}", p, max_tokens, temperature=0) for i, p in enumerate(prompts)]
        start = perf_counter()
        outputs = engine.run(requests, pipelined=pipelined)
        seconds = perf_counter() - start
        if outputs != oracle or engine.caches:
            raise AssertionError(f"{name}: output mismatch or cache leak")
        rows = [row for batch in engine.batches for row in batch]
        return dict(seconds=seconds, calls=model.calls, projected_tokens=model.projected_tokens,
                    score_elements=model.score_elements,
                    proposed=sum(r["proposed"] for r in rows),
                    accepted=sum(r["accepted"] for r in rows),
                    rolled_back=sum(r["rolled_back"] for r in rows))

    for name in policies:
        trial(name)  # one untimed warmup per policy (not included in samples)
    names = list(policies)
    for repeat in range(repeats):
        # Rotate order to reduce a fixed first/last-policy bias.
        for name in names[repeat % len(names):] + names[:repeat % len(names)]:
            samples[name].append(trial(name))
    output_tokens = len(prompts) * max_tokens
    results = {}
    for name in names:
        durations = [s["seconds"] for s in samples[name]]
        median = float(np.median(durations))
        results[name] = dict(policy=policies[name], median_seconds=median,
                             p95_seconds=float(np.percentile(durations, 95)),
                             output_tokens_per_second=output_tokens / median,
                             samples=samples[name], equal_full_recompute=True)
    root = Path(__file__).resolve().parent
    sources = [root / "bench.py", *sorted((root / "nano_infer").glob("*.py"))]
    blas = StringIO()
    with redirect_stdout(blas):
        np.show_config()
    return dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 platform=platform.platform(), machine=platform.machine(),
                                 processor=platform.processor(), cpu_count=os.cpu_count(),
                                 numpy_build=blas.getvalue(),
                                 thread_env={k: os.environ.get(k) for k in
                                             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}),
                workload=dict(seed=seed, prompts=prompts, max_tokens=max_tokens,
                              batched=batched, paged=batched,
                              checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
                              output_tokens=output_tokens, token_budget=12, max_batch=3,
                              prefill_chunk=4, repeats=repeats, warmups_per_policy=1),
                source_sha256={str(p.relative_to(root)).replace("\\", "/"):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                oracle_outputs=oracle, results=results,
                scope="CPU engine wall time; no model construction, oracle or network; all requests arrive at start")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batched", action="store_true", help="batched attention with paged KV")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        report = benchmark(args.repeats, args.max_tokens, args.seed, args.checkpoint, args.batched)
    except ValueError as exc:
        parser.error(str(exc))
    print("CPU only | identical workload | full-recompute outputs checked | warmup excluded")
    print("policy              median ms    p95 ms  output tok/s   calls  projected")
    for name, row in report["results"].items():
        work = row["samples"][0]
        print(f"{name:19} {row['median_seconds'] * 1000:9.2f}"
              f" {row['p95_seconds'] * 1000:9.2f} {row['output_tokens_per_second']:13.1f}"
              f" {work['calls']:7} {work['projected_tokens']:10}")
    print("Timings vary with CPU/BLAS/load; fewer calls need not be faster.")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
