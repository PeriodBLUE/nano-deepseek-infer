# nano-deepseek-infer

A CPU-sized educational companion to [DeepInfer](https://github.com/deepseek-ai/DeepInfer). It uses a randomly initialized, one-layer causal attention + top-2 MoE model to make inference-system invariants executable.

This is an independent project, not an official DeepSeek implementation. It does not load DeepSeek checkpoints, reproduce DSpark/DFlash, or make production speed claims.

```bash
python -m pip install -e .
python -m nano_infer demo all
python -m unittest discover -s tests -v
```

Seven lessons cover KV reuse, greedy speculative verification and rollback, historical draft budgeting, expert replication and placement, context-sharded softmax merging, bounded suffix replay, request-local sampling, constrained output, and scheduling. Scheduling and structured output are separate lessons; confidence budgeting is included in the speculative lesson.

The only runtime dependency is NumPy. No downloads occur after installation. Text output is intentionally meaningless because weights are random.

## Scope

- Attention, expert matrices, cached forward passes and speculative verification are real numerical computations.
- Expert/context parallel ranks are simulated in one process.
- The scheduling engine executes requests individually and optionally prepares other ready requests while a single CPU worker executes.
- Bounded replay is proven only for the provided one-layer sliding-window model.
- Speculative decoding is greedy-only. Ordinary sampling uses independent per-request RNGs.
- Structured output uses a finite-choice grammar, not a general schema compiler.

See the [Chinese README](../README.md), [design walkthrough](design.md), and [pinned upstream source map](upstream-map.md). License: MIT.
