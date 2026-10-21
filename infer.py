"""Read this first: live requests -> scheduler -> model -> KV commit -> output.

    python infer.py
    python infer.py --prompt "nano" --prompt "cache" --draft-tokens 4

Every run checks its outputs against full-prefix greedy recomputation.
"""
import argparse
import json
from pathlib import Path

from nano_infer.engine import Engine, Request
from nano_infer.model import TinyModel, decode, encode, greedy
from nano_infer.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", action="append", help="repeat for multiple requests")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--draft-tokens", type=int, default=4, help="0 disables speculation")
    parser.add_argument("--adaptive", action="store_true", help="use historical draft budgets")
    parser.add_argument("--token-budget", type=int, default=8)
    parser.add_argument("--prefill-chunk", type=int, default=4)
    parser.add_argument("--arrival-gap", type=int, default=2, help="logical steps between arrivals")
    parser.add_argument("--seed", type=int, default=7, help="model weight seed")
    parser.add_argument("--checkpoint", type=Path, help="load trained NumPy weights")
    parser.add_argument("--json", type=Path, help="save inputs, outputs and all state transitions")
    args = parser.parse_args()
    try:
        if args.arrival_gap < 0 or args.max_tokens < 0:
            raise ValueError("arrival gap and output length must be nonnegative")
        prompts = args.prompt or ["nano", "deepseek inference"]
        requests = [Request(f"r{i}", encode(text), args.max_tokens, temperature=0)
                    for i, text in enumerate(prompts)]
        make_model = lambda: load_checkpoint(args.checkpoint) if args.checkpoint else TinyModel(seed=args.seed)
        model = make_model()
        engine = Engine(model, token_budget=args.token_budget, prefill_chunk=args.prefill_chunk,
                        max_draft=args.draft_tokens, adaptive=args.adaptive)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Model: {args.checkpoint or 'seeded initialization'}")
    print("step request phase    input draft accept rollback cache     output")
    tick, next_request = 0, 0
    arrivals, trace = [], []
    while next_request < len(requests) or not engine.is_finished():
        if engine.is_finished() and next_request < len(requests):
            tick = max(tick, next_request * args.arrival_gap)
        while next_request < len(requests) and next_request * args.arrival_gap <= tick:
            req = requests[next_request]
            engine.add_request(req)
            arrivals.append(dict(step=tick, request=req.request_id))
            print(f"  + {req.request_id}: {prompts[next_request]!r}")
            next_request += 1
        # This is the complete driving loop. Try changing WHEN requests arrive.
        batch = engine.step()
        trace.append(dict(step=tick, batch=batch))
        for row in batch:
            cache = f"{row['cache_before']}->{row['cache_after']}"
            print(f"{tick:4} {row['request']:7} {row['phase']:8} {row['input_tokens']:5}"
                  f" {row['proposed']:5} {row['accepted']:6} {row['rolled_back']:8}"
                  f" {cache:9} {decode(row['emitted_tokens'])!r}"
                  f"{' done' if row['done'] else ''}")
        tick += 1

    # Deliberately slower, uncached baseline; outside any performance measurement.
    for req, prompt in zip(requests, prompts):
        expected = greedy(make_model(), req.prompt, req.max_tokens, cached=False)
        if req.output != expected:
            raise AssertionError(f"{req.request_id}: output differs from full recomputation")
        print(f"{req.request_id}: {prompt!r} -> {decode(req.output)!r}")
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  requests=[dict(id=r.request_id, prompt=r.prompt, output=r.output) for r in requests],
                  arrivals=arrivals, trace=trace, equal_full_recompute=True,
                  caches_released=not engine.caches,
                  work=dict(calls=model.calls, projected_tokens=model.projected_tokens,
                            score_elements=model.score_elements))
    print(f"PASS: all outputs match full recomputation; caches released={not engine.caches}")
    print(f"Work: {model.calls} target calls, {model.projected_tokens} projected tokens")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
