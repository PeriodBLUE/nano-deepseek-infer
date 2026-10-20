# 相关项目比较：范围、版本与证据

核对日期：2026-10-21。通过 GitHub 仓库搜索和网页搜索定位相关项目，阅读 README、依赖声明及下表列出的部分实现／测试。比较对象包括直接以 DeepInfer 为灵感的小型引擎、V4 架构教学实现和常被用作参考的 nano-vLLM。

这是有范围限制的源码与文档核对，不是穷尽式调查，也没有在相同硬件上安装、运行或测量这些项目。表中能力是所列版本的实现／文档证据，不表示本项目独立验证过它们的全部运行结果。未检查到某个功能不能据此断言它不存在。

## 固定版本

| 项目 | 核对版本 | README 之外检查的证据 |
|---|---|---|
| Alexander-Qiu/nano-deepinfer | [`2684bd8`](https://github.com/Alexander-Qiu/nano-deepinfer/tree/2684bd8a8d98cb855dc88687723a40874acf5f71) | [依赖](https://github.com/Alexander-Qiu/nano-deepinfer/blob/2684bd8a8d98cb855dc88687723a40874acf5f71/pyproject.toml)、[scheduler](https://github.com/Alexander-Qiu/nano-deepinfer/blob/2684bd8a8d98cb855dc88687723a40874acf5f71/nano_deepinfer/engine/scheduler.py)、[引擎测试](https://github.com/Alexander-Qiu/nano-deepinfer/blob/2684bd8a8d98cb855dc88687723a40874acf5f71/tests/test_engine.py)、[Transformers 对照](https://github.com/Alexander-Qiu/nano-deepinfer/blob/2684bd8a8d98cb855dc88687723a40874acf5f71/tests/test_hf.py) |
| singzhou/nano-deepseek-v4-inference | [`58c5989`](https://github.com/singzhou/nano-deepseek-v4-inference/tree/58c5989369a31069b636c3ba2343349243858cf9) | [依赖](https://github.com/singzhou/nano-deepseek-v4-inference/blob/58c5989369a31069b636c3ba2343349243858cf9/requirements.txt)、[推理与结构测试](https://github.com/singzhou/nano-deepseek-v4-inference/blob/58c5989369a31069b636c3ba2343349243858cf9/tests/test_inference.py) |
| hebo1221/nano-deepseek-v4 | [`d25710f`](https://github.com/hebo1221/nano-deepseek-v4/tree/d25710f6d35e9344762f8e8af3c7367ba7a5d384) | [依赖](https://github.com/hebo1221/nano-deepseek-v4/blob/d25710f6d35e9344762f8e8af3c7367ba7a5d384/pyproject.toml)、[DSpark 调度实现](https://github.com/hebo1221/nano-deepseek-v4/blob/d25710f6d35e9344762f8e8af3c7367ba7a5d384/nano_deepseek_v4/dspark_scheduler.py)、[DSpark 调度测试](https://github.com/hebo1221/nano-deepseek-v4/blob/d25710f6d35e9344762f8e8af3c7367ba7a5d384/tests/test_dspark_scheduler.py) |
| GeeeekExplorer/nano-vllm | [`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8) | [依赖](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/pyproject.toml)、[GPU model runner](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py)、[缓存块管理](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py) |
| 本项目 | [`1dd1f8f`](https://github.com/PeriodBLUE/nano-deepseek-infer/tree/1dd1f8f3647d85919d3eb6751fb7894b1cb75056) | 本次比较前的实现版本：[实验](https://github.com/PeriodBLUE/nano-deepseek-infer/blob/1dd1f8f3647d85919d3eb6751fb7894b1cb75056/nano_infer/labs.py)、[测试](https://github.com/PeriodBLUE/nano-deepseek-infer/blob/1dd1f8f3647d85919d3eb6751fb7894b1cb75056/tests/test_core.py)、[记录](https://github.com/PeriodBLUE/nano-deepseek-infer/blob/1dd1f8f3647d85919d3eb6751fb7894b1cb75056/examples/results.json) |

## 对差异的判断

### 与 Alexander-Qiu/nano-deepinfer

它是最直接的同类项目：CPU、分块 prefill、连续批处理、增量 KV 和采样都与本项目有交集。其默认示例加载真实的 SmolLM2 小模型；测试另有随机 tiny Llama、ragged batch、slot 复用和 Transformers 对照。

本项目的教学中心转向草稿回滚、物理专家放置、分片 softmax 和有条件的尾部重放，并用 NumPy 随机模型统一驱动。前者更适合理解真实模型接入；本项目的示例可以在不下载权重的情况下研究这些系统机制。不能将前者描述为没有测试或只有 demo。

### 与 singzhou/nano-deepseek-v4-inference

它围绕 V4 的混合 attention、压缩 KV、mHC 和 MoE 展开，测试覆盖 prefill/decode 一致、缓存长度与环形顺序、Sinkhorn 约束和因果性。它的 tiny 模型也能在 CPU 上以随机权重运行。

本项目只用单层 attention 保持模型依赖易于分析，重点比较执行方式与状态变化。完整 V4 结构和压缩缓存不是本项目的优势；读者关心这些内容时，应进一步阅读该类架构实现。

### 与 hebo1221/nano-deepseek-v4

它同时覆盖模型结构与验证方法。所核对版本的 README 明确提供无下载 CPU conformance、独立方程参考、DSpark 语义向量与调度检查；调度测试还包括穷举 oracle 和随机小例对照。

因此不能声称“只有本项目验证正确性、其他项目只实现架构”，也不能将我们的简单历史接受率预算说成更忠实的 DSpark 实现。我们的差异在依赖、模型规模和学习路线：用一个 NumPy 小模型，把 greedy 块验证的真实接受／拒绝、KV 提交／回滚和工作量计数连成一个可运行实验，再与其他系统机制并列阅读。

### 与 GeeeekExplorer/nano-vllm

其依赖声明包含 PyTorch、Transformers、FlashAttention；model runner 使用 CUDA，缓存块管理服务于实际模型推理。它的 prefix caching、tensor parallelism 和 CUDA Graph 是理解 GPU 推理实现的重要入口。

本项目的 EP/CP 是单进程模拟，调度 worker 内也是逐请求执行，没有与其比较吞吐量的基础。这里保留的是正确性对照、状态可见性和最少运行依赖，不能据此推出速度优势。

## 本项目的差异如何被检查

| 定位 | 现有证据 | 不能推出的结论 |
|---|---|---|
| 只需 NumPy 的运行路径 | [pyproject.toml](../pyproject.toml)；随机模型直接构造权重 | 其他项目都需要 GPU 或下载 |
| 执行方式改变后的不变量 | [test_core.py](../tests/test_core.py)：缓存、推测回滚、专家副本、CP 合并、SWA 边界、调度重排 | 与官方模型或生产 kernel 数值等价 |
| 可观察的中间状态 | [results.json](../examples/results.json)：草稿轮次和调度批次 | 对 GPU stream 或真实通信的追踪 |
| 保留收益与反例 | 固定草稿调用更少但投影更多；自适应控制器冷启动后关闭草稿 | 墙钟性能排名或普遍最优策略 |
| 可回读上游的教程 | [upstream-map.md](upstream-map.md)：固定提交、文件、符号和简化边界 | 与上游功能或模型架构完整对应 |

这些是项目目前已有的组合与取舍，不是“首个”“唯一”或“全面领先”的声明。以后扩展功能时，应同步更新这一比较的版本和证据，避免把旧结论当作其他项目的现状。
