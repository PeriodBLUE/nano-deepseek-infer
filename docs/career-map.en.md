# From project evidence to AI Infra interviews

[中文](career-map.md) · [README](../README.md)

Completing this project should let you independently **generate text from trained weights, explain request/cache state at every step, and evaluate an inference optimization through controlled experiments.** Build your interview explanation around reproducible results and code you can modify.

## Official job descriptions and project evidence

Reviewed **2026-10-21**. Requirements below are paraphrased from official company postings; the mapping to project work is our analysis. Follow the links for current availability.

| Company and official role | Requirements emphasized | Evidence you can build here | Interview question |
|---|---|---|---|
| [Anthropic — Performance Engineer, Inference Engine](https://job-boards.greenhouse.io/anthropic/jobs/5418323008) | Batching, cached state, memory, observability, determinism and testing | Late arrivals, KV commit/rollback/release; EOS, failure and reorder tests | Who owns each state, and when can it be reused or released? |
| [OpenAI — Software Engineer, Inference - Performance Optimization](https://openai.com/careers/software-engineer-inference-performance-optimization-san-francisco/) | End-to-end analysis, microbenchmarks, latency/capacity/cost models | Fixed workloads, warmups and raw timing samples; DSpark benefit/cost objective | Why can fewer calls be slower, and what evidence identifies the bottleneck? |
| [NVIDIA — Inference Performance Engineer, Agent Driven Inference Optimization, JR2021689](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Inference-Performance-Engineer--AI-Inference-Configuration-Optimization_JR2021689) | Continuous batching, KV, MoE, speculation; controlled experiments and correctness | Draft budgets, replica placement, cache work counters and exhaustive scheduling reference | How do throughput, verification cost and output correctness interact? |
| [Together AI — Machine Learning Engineer - Inference](https://job-boards.greenhouse.io/togetherai/jobs/4385540007) | Inference runtimes, reliability, testing, developer documentation; speculation | Training→checkpoint→generation→engine, failure cleanup and independent references | How do you integrate a model into a maintainable inference runtime? |

These roles also require experience at their respective levels, plus accelerator, systems-programming or distributed-systems expertise. Use the following table to distinguish directly executable goals from concrete extensions you can build in this repository.

## Learning acceptance checklist

✓ means an existing executable path; ✗ means the extension still needs implementation. Personal mastery means independently changing, running and explaining it.

| Skill goal | Direct practice here | Personal deliverable |
|---|:---:|---|
| Derive attention, MoE and next-token loss | ✓ | Forward/backward shapes and gradient-check explanation |
| Train, save, reload and generate | ✓ | Training configuration, held-out loss, checkpoint digest and samples |
| Prefill/decode, continuous scheduling, KV lifecycle | ✓ | Changed arrivals/budgets and annotated cache lengths |
| Speculation, rollback and DSpark equations | ✓ | Hand-worked accepted prefix and N−2 budget/N−1 allocation counterexample |
| Expert placement and sharded softmax | ✓ | Changed hotspot distribution, load measurements and numerical error |
| Performance experiments and cost reasoning | ✓ | Identical inputs/outputs, work and time measurements, one negative result |
| Reproducible tests and debugging | ✓ | A state-bug regression that fails before a fix and passes afterward |
| CUDA/Triton and GPU profiling | ✗ | Port one attention/MoE operator; record profiles and numerical errors |
| Multi-GPU communication | ✗ | Replace simulated ranks with processes; measure bytes, waiting and hotspots |
| Online request lifecycle and latency | ✓ | HTTP submit/poll/cancel, deadline/backpressure tests and `load_test.py` TTFT/ITL percentiles |
| Batched execution and physical KV ownership | ✓ | Ragged mask comparison, COW/eviction tests and memory accounting from `systems.py` |
| Compressed KV equations | ✓ | V2 MLA expanded/absorbed agreement and actual latent-cache bytes |

## Build an interview project package

1. **Model:** run `train.py`, explain train versus validation loss, and prove checkpoint reload preserves predictions.
2. **State:** run `infer.py --checkpoint checkpoints/tiny-trained.npz --prompt "the model " --max-tokens 64 --json my-trace.json`; calculate cache length after a rejected draft.
3. **Equations:** run `python -m nano_infer.dspark`; calculate a graph-bucket boundary and alter latest confidence to distinguish total-budget selection from allocation.
4. **Performance:** run `bench.py --repeats 5 --json my-benchmark.json`; explain workload, timing scope, reference outputs and variance.
5. **Contribution:** change one implemented admission, batching or cache policy, preserving a failure case, fix, tests and measurements.

A factual project description can cover a trainable NumPy attention/MoE model, checkpoint inference, token-budget scheduling, KV rollback, independently checked DSpark equations and fixed-workload speculation measurements. Add your own measured results and be ready to derive every claim.

Practice follow-ups: Why does a weight update invalidate old KV? Where is top-2 routing nondifferentiable? Why does confidence use pre-norm hidden? Why must future confidence not determine current total work? Why is a CPU-run p95 different from an online-service p95? Ground each answer in code, equations and your own experiment.
