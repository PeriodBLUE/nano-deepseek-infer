"""Submit staggered HTTP requests and report server TTFT/ITL and client throughput.

Without --url, starts and closes an ephemeral localhost server using the bundled
checkpoint. Request outputs are checked against full-prefix recomputation.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import numpy as np

from nano_infer.engine import Engine
from nano_infer.model import encode, greedy
from nano_infer.paging import BlockPool
from nano_infer.service import Service
from nano_infer.training import load_checkpoint
from serve import start_server


def load_test(url=None, requests=8, max_tokens=32, arrival_ms=5., checkpoint="checkpoints/tiny-trained.npz"):
    if requests < 1 or max_tokens < 1 or arrival_ms < 0:
        raise ValueError("positive requests/tokens and nonnegative arrival gap required")
    close, service = None, None
    if url is None:
        model = load_checkpoint(checkpoint)
        pool = BlockPool(16, 8, model.width)
        service = Service(Engine(model, token_budget=16, max_batch=4, max_draft=4, batched=True, pool=pool),
                          max_pending=max(16, requests), history=max(64, requests))
        server, close = start_server(service, port=0)
        url = f"http://127.0.0.1:{server.server_port}"
    def call(path, data=None):
        req = Request(url + path, data=None if data is None else json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as response: return json.load(response)
    pending, results, admissions = {}, [], []
    start = perf_counter()
    try:
        for i in range(requests):
            intended = i * arrival_ms / 1000
            sleep(max(0, start + intended - perf_counter()))
            prompt = ("the model ", "the cache ", "the draft ")[i % 3]
            sent = perf_counter() - start
            try:
                key = call("/requests", dict(prompt=prompt, max_tokens=max_tokens, timeout=30.))["id"]
                pending[key] = prompt
                admissions.append(dict(id=key, scheduled_seconds=intended, sent_seconds=sent))
            except HTTPError as exc:
                results.append(dict(status="rejected", http_status=exc.code, error=json.load(exc)))
        deadline = perf_counter() + 35
        while pending:
            if perf_counter() >= deadline: raise TimeoutError("load test polling deadline")
            for key in list(pending):
                row = call("/requests/" + key)
                if row["status"] not in ("queued", "running"):
                    row["prompt"] = pending.pop(key)
                    results.append(row)
            if pending: sleep(.002)
        elapsed = perf_counter() - start
        completed = [r for r in results if r["status"] == "completed"]
        if service is not None:
            if len(completed) != requests:
                raise AssertionError("local workload did not complete every request; inspect capacity/deadlines")
            for row in completed:
                expected = greedy(load_checkpoint(checkpoint), encode(row["prompt"]), max_tokens, cached=False)
                if row["tokens"] != expected: raise AssertionError("HTTP output differs from full recomputation")
        def percentiles(values):
            return dict(zip(("p50", "p95", "p99"), map(float, np.percentile(values, [50, 95, 99])))) if values else None
        return dict(requests=requests, completed=len(completed), client_elapsed_seconds=elapsed,
                    client_output_tokens_per_second=sum(len(r["tokens"]) for r in completed) / elapsed,
                    ttft_seconds=percentiles([r["metrics"]["ttft_seconds"] for r in completed]),
                    queue_seconds=percentiles([r["metrics"]["queue_seconds"] for r in completed]),
                    itl_seconds=percentiles([v for r in completed for v in r["metrics"]["itl_seconds"]]),
                    checked_full_recompute=service is not None,
                    scope="TTFT/ITL use server commit timestamps; throughput includes HTTP and staggered arrivals",
                    admissions=admissions, results=results,
                    pool=service.engine.pool.snapshot() if service is not None else None)
    finally:
        if close: close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--arrival-ms", type=float, default=5.)
    parser.add_argument("--checkpoint", default="checkpoints/tiny-trained.npz")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = load_test(args.url, args.requests, args.max_tokens, args.arrival_ms, args.checkpoint)
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "admissions")}, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
