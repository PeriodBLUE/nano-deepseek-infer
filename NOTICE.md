# Attribution

This independent educational project is inspired by the public design and source of:

- DeepSeek-AI, **DeepInfer: High-performance, large-scale inference framework for DeepSeek models**.
- Repository: https://github.com/deepseek-ai/DeepInfer
- Reviewed revision: `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`
- Upstream license: Apache License 2.0.

The upstream repository credits vLLM, SGLang and TensorRT-LLM. This repository links to upstream sources for study; it does not vendor upstream code, kernels, model weights or branding assets. Its small NumPy examples and documentation were written independently and are licensed under MIT.

Related projects informed the teaching and verification improvements:

- [Alexander-Qiu/nano-deepinfer](https://github.com/Alexander-Qiu/nano-deepinfer): separate numerical reference checks.
- [singzhou/nano-deepseek-v4-inference](https://github.com/singzhou/nano-deepseek-v4-inference): explicit tensor-shape explanations.
- [hebo1221/nano-deepseek-v4](https://github.com/hebo1221/nano-deepseek-v4): independent oracles, guided tours and recorded verification evidence.
- [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): cross-request prefix reuse.

The corresponding NumPy oracle, instrumentation and prefix-fork examples were
written independently. No code from those projects is included. Pinned revisions
and the precise scope of the comparison are listed in [docs/comparison.md](docs/comparison.md).

DeepSeek and DeepInfer names identify the subject of the lessons. No affiliation or endorsement is implied.
