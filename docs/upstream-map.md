# 上游源码导读

固定版本：[deepseek-ai/DeepInfer @ 774ce244](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01)。核对日期：2026-10-21。

这里记录阅读过的文件与对应概念。nano 源码是独立编写的教学实现；“对应”表示解释同一个问题，不表示逐行移植、API 兼容或性能等价。

| 上游文件／入口 | 阅读时关注什么 | nano 入口 |
|---|---|---|
| [README.md](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md) | 发布范围、硬件限制、哪些服务组件不包含 | README 的边界表 |
| [llm_engine.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/engine/llm_engine.py)：`step`、`_execute_model_async_step`、`post_process` | schedule / execute / process-output 如何拆开 | `engine.Engine` |
| [scheduler.hpp](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp) | 请求生命周期、调度状态与 token 预算 | `_plan`、`_finish` |
| [mtp_varlen.hpp](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/mtp_varlen.hpp)：`MTPVarlen::schedule` | 历史 confidence 隔离、吞吐率目标、graph/runtime 成本曲线 | `ConfidenceBudget`，大幅简化 |
| [dflash.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/mtp/dflash.py)：`ConfidenceHead`、`DFlash` | 草稿和接受概率来自专门的模型部件 | `BigramDraft` 仅作廉价替身 |
| [eplb/policy.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/eplb/policy.py)：`rebalance_experts_global`、`_batch_greedy_indices` | 先分配冗余副本，再利用多个历史步负载放置 | `parallel.rebalance` 仅用聚合向量 |
| [v4_modules/moe.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/moe.py)：`TopKGate`、`FusedExperts` | logical / physical / local 专家数量如何区分 | `parallel.moe_forward` |
| [input_metadata/yoco.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/input_metadata/yoco.py)：`YocoMetadataMixin` | mask 与 compact 视图、保留绝对位置、CP 重分布 | `bounded_replay` 只解释局部依赖条件 |
| [v4_modules/yoco.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/yoco.py) | producer / consumer 与共享 cache，说明为何不能任意截断 | 设计文档中的限制 |
| [layers/sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/sampler.py) | 采样运行时如何同模型执行连接 | `sample` 与 `ChoiceGrammar` |

## 三处容易误读的设计

**EPLB 不是单纯地把 token 均匀切给不同 GPU。** 模型路由先决定逻辑专家；系统再利用物理专家副本减轻热点。改变 gate 选择会改变模型本身，复制同权重专家则可保持函数不变。

**置信度不是看到当前验证结果后再挑好看的 token。** `mtp_varlen.hpp` 中专门维护相邻历史轮次和 schedule/process 计数，隔离不同用途的统计。nano 只使用此前轮次的整体统计，未实现上游调度协议。

**compact 不是“任意删掉旧 token”。** 上游根据模型结构生成 consumer 视图、位置与可见域，并处理 CP 后布局；nano 只在可以直接证明的一层 SWA 情况下重算后缀。

## 继续读上游

完成 nano 实验后，再把 `TinyModel.forward` 替换成“由多层、不同 attention 和专门 kernel 组成的 worker”，沿 `LLMEngine.step` 看元数据如何生成和传递。随后读专家权重放置、MTP 元数据与 sampler。这样每个 tensor 都能对应到一个已经运行过的状态或不变量。
