# What makes this a nano inference engine?

[中文](#设计取舍) · [English](#design-decisions) · [README](../README.md)

## 设计取舍

本项目的完成标准是：**读者能运行一条完整链路，找到核心循环，改变一个策略，在相同答案的前提下测量它的代价。** 代码行数只是阅读成本的一部分；把状态藏进抽象层，或者删掉关键行为，都不能自动让项目更适合学习。

### 阅读和运行应走同一条路径

```text
infer.py
  add_request()                    请求可在轮次之间到达
  while ...:
      step()
        _plan()                    prefill / decode / verify 共用预算
        _execute()                 TinyModel.forward 追加 KV
        _finish()                  采样或验证；提交、截断、释放
  full-prefix greedy comparison    最终答案必须与慢路径一致
```

`Engine` 保存请求身份、就绪队列、KV、RNG、draft 与历史预算；`Work` 只描述一次执行。未完成的计划不会被再次调度。草稿验证和独立实验共用 `greedy_prefix()`，使接受规则只有一份实现。

`infer.py` 的到达时间是**逻辑 step**，用于研究交错与状态，不是网络负载发生器。`run(pipelined=True)` 的所有请求在开始时提交；动态到达演示使用同步 `add_request()` / `step()`，不提供线程安全的并发接入。

### 成功要有明确定义

正确性由同输入下的全前缀重算、独立方程参考和状态边界测试约束。性能则交给 `bench.py`：

- 固定四个 prompt、模型种子、输出长度及调度限制。
- 每种策略 warmup 后重复执行，轮换顺序，保留每次耗时。
- 计时包括 admission、schedule、execute、commit；不包括模型/请求构造、参考结果计算、打印或写文件。
- 输出 median、样本 p95、吞吐、调用数与投影工作量；p95 是这些运行总耗时的样本分位数，不是线上单请求延迟 SLO。
- 输出必须一致，缓存必须释放；不要求草稿或线程模式一定更快。

[完整推理记录](../examples/inference.json)展示具体状态变化；[CPU 测量样例](../examples/cpu-benchmark.json)展示记录格式。CPU/BLAS、系统负载和线程设置都会影响耗时，应在自己的机器上重跑。不同调度策略可能改变有效草稿长度和投影总量，这是策略成本的一部分。

### 保持模型小，保持问题真实

随机单层 attention + MoE 足以让因果性、KV 生命周期、草稿回滚、专家副本和分片归一化产生真实的数值约束。它让入门实验无需模型下载，但不提供语言能力；因此本项目的第一结果是可解释的状态轨迹和正确性检查，而不是聊天演示。

完整 V4 架构、训练、预训练权重加载、分页缓存、设备 kernel 和分布式通信，各自都有独立复杂度。这里通过[上游源码地图](upstream-map.md)连接这些方向，避免把近似实现标成完整复现。当前 worker 仍逐请求计算；调度批次不等于向量化模型 batch。

### 当前检查过的参考项目

以下是 2026-10-21 的 GitHub API 和固定提交源码快照。Star 用于说明选择参考对象的背景，不能证明项目质量或当前增速，也不是“最热门”排名。nanoGPT 的 README 已注明弃用并引导到 nanochat；它仍是阅读简洁核心循环的重要参考。

| 项目 | 当日 Star | 固定版本与检查文件 | 适合研究的问题 |
|---|---:|---|---|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 63,949 | [`3adf61e`](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291)：README、`model.py`、`train.py`、`sample.py` | 模型定义、训练循环、验证与采样如何组成可读的小系统 |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | 58,797 | [`92d63d4`](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd)：README、`runs/speedrun.sh`、`nanochat/engine.py` | 完整入口如何连接训练、评估与使用；推理状态如何显式保存 |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 16,017 | [`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)：`example.py`、`bench.py`、engine、scheduler | 一个真实推理引擎如何组织 add/step/generate、调度与设备执行 |
| [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | 5,962 | [`4ea6b93`](https://github.com/KellerJordan/modded-nanogpt/tree/4ea6b937337a4889b8cfe3f38a93d120048d8f71)：README、`train_gpt.py` | 如何用固定任务、质量门槛和计时范围判断优化；算法与 kernel 如何分层阅读 |

这些项目的训练或 GPU 性能未在本机复测，也不与本项目做速度排名。这里研究的是可读实现与实验方法；代码独立编写，来源见 [NOTICE](../NOTICE.md)。

## Design decisions

The completion criterion is: **run a complete path, locate its loop, change a policy, and measure its cost while preserving answers.** Fewer lines alone do not establish that; hidden state or missing behavior can make a short implementation harder to understand.

Start at `infer.py`. Requests arrive between `step()` calls. Planning, execution and commit share one engine; prefill, decode and verification share one token budget. `Work` describes an immutable execution plan, while the engine owns request-local caches, RNGs, drafts and budget histories. Both the engine and the isolated speculation lab use `greedy_prefix()`.

Arrivals in this script use logical steps, not network time. Dynamic admission is synchronous between calls; the pipelined offline `run()` admits all requests before execution. There is no thread-safe concurrent admission API.

`bench.py` defines a fixed workload and checks every measured output against full-prefix recomputation. It warms up each policy, rotates order, retains raw samples and records environment/source metadata. Timing includes admission through output commit but excludes model/request construction, the reference, printing and file writes. Its p95 is a sample quantile of complete CPU runs, not a production per-request latency SLO. Threading or speculation may lose; no speed threshold is imposed.

The model stays small enough to inspect. Random weights provide real numerical constraints for cache state, causal verification and parallel reductions, while avoiding downloads. They do not provide language capability. Training, real checkpoints, V4-specific architecture, paging, device kernels and distributed communication are further areas connected through the [upstream map](upstream-map.md). The local worker still computes requests individually.

The table above records the reviewed projects, revisions and observed GitHub stars on 2026-10-21. nanoGPT is now marked deprecated in favor of nanochat, but remains useful for studying an explicit core loop. nanochat exposes an end-to-end run; nano-vLLM exposes a real engine lifecycle; modded-nanogpt defines a fixed quality target and timing scope. These are source reviews, not locally reproduced training results or cross-project performance rankings.

Inspect the [inference trace](../examples/inference.json) and [CPU measurement example](../examples/cpu-benchmark.json), then rerun on your own machine. Exact timing depends on CPU, BLAS and load; the output-equivalence requirement is fixed.
