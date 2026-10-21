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
| 本项目 | 与本文件同一提交 | [训练与权重](training.md)、[DSpark 方程](dspark.md)、 [实验](../nano_infer/labs.py)、[独立参考与导览](../nano_infer/verification.py)、[前缀复用](../nano_infer/prefix.py)、[新增测试](../tests/test_verification.py)、[带源码摘要的验证记录](../examples/verification.json) |

## README 勾叉表的判定口径

中英文 README 使用相同的行、列、符号和脚注。每一列只表达一种含义；资源条件与算法能力拆为两张表。最后一行加粗只是定位本项目，不是综合排名。

- **CPU tiny 路径**：可以在 CPU 执行 tiny 模型。nano-vLLM 的所查 runner 初始化 CUDA，因此该版本不满足该列。
- **仅 NumPy 运行时**：核心安装和实验只需要 NumPy 与标准库；四个比较对象的依赖中均含 PyTorch，因此记为 ✗，不表示它们依赖设计有问题。
- **默认 demo 不下载权重**：指 README 默认演示路线。Alexander-Qiu 的示例会下载 SmolLM2，而它的随机模型测试可以离线，二者不混为一谈。
- **训练后权重文本生成**：包括现有预训练模型或项目自身训练并保存的模型。hebo 的 ✓\* 指 native bundle；本项目提供全参数 NumPy 训练、原生 checkpoint 与引擎内生成路径，见 [训练记录](../examples/training.json)。
- **V4 架构组件**：指针对 V4 的混合 attention、压缩 cache、mHC 等组合，不将普通 attention + MoE 计作 V4。本项目明确为 ✗。
- **独立数值参考**：Alexander-Qiu 有 Transformers 对照，hebo 有独立 oracle；本项目提供独立控制流参考，但仍共享 NumPy 与权重，因此加脚注。仅 self full/cache 对比不足以据此打勾。
- **DSpark 方程验证**：本项目的 ✓* 对应独立的 [DSpark 方程与双代际调度实验](dspark.md)，含标量和穷举参考；在线 bigram 草稿路径单独测试。singzhou 将 MTP speculative decoding 列为未实现范围；hebo 有对应的方程／调度向量验证。
- **跨请求前缀复用**：一个请求内部的增量 KV 不算此项。nano-vLLM 有 prefix caching；本项目支持显式公共前缀复用，以数组复制隔离分支，故为 ✓\*。
- **专家副本放置实验**：仅有 MoE 路由不足以打勾。本项目有物理副本、放置和分发合并实验；未专门核实的其他项目为 —。

表中 **— 不等于 ✗**。勾代表所核对版本的实现或文档证据；待办或未来计划不算作支持。

## 对差异的判断

### 与 Alexander-Qiu/nano-deepinfer

它是最直接的同类项目：CPU、分块 prefill、连续批处理、增量 KV 和采样都与本项目有交集。其默认示例加载真实的 SmolLM2 小模型；测试另有随机 tiny Llama、ragged batch、slot 复用和 Transformers 对照。

本项目的教学中心转向草稿回滚、物理专家放置、分片 softmax 和有条件的尾部重放，并用可训练的 NumPy 模型统一驱动，提供 loss、权重文件、文本生成和完整请求 trace。

### 与 singzhou/nano-deepseek-v4-inference

它围绕 V4 的混合 attention、压缩 KV、mHC 和 MoE 展开，测试覆盖 prefill/decode 一致、缓存长度与环形顺序、Sinkhorn 约束和因果性。它的 tiny 模型也能在 CPU 上以随机权重运行。

本项目只用单层 attention 保持模型依赖易于分析，重点比较执行方式与状态变化。训练和推理共用模型定义，数组级计算、梯度与状态均可直接阅读。

### 与 hebo1221/nano-deepseek-v4

它同时覆盖模型结构与验证方法。所核对版本的 README 明确提供无下载 CPU conformance、独立方程参考、DSpark 语义向量与调度检查；调度测试还包括穷举 oracle 和随机小例对照。

本项目同样提供独立 DSpark head 和双代际调度方程检查。组合优势在依赖、模型规模和完整学习闭环：用一个可训练的 NumPy 小模型连接参数更新、权重重载、greedy 块验证、KV 提交／回滚与成本测量。

### 与 GeeeekExplorer/nano-vllm

其依赖声明包含 PyTorch、Transformers、FlashAttention；model runner 使用 CUDA，缓存块管理服务于实际模型推理。它的 prefix caching、tensor parallelism 和 CUDA Graph 是理解 GPU 推理实现的重要入口。

本项目的 EP/CP 是单进程模拟，调度 worker 内也是逐请求执行，没有与其比较吞吐量的基础。这里保留的是正确性对照、状态可见性和最少运行依赖，不能据此推出速度优势。

## 本项目的差异如何被检查

| 定位 | 现有证据 | 不能推出的结论 |
|---|---|---|
| 只需 NumPy 的运行路径 | [pyproject.toml](../pyproject.toml)；原生训练与约 662 KiB 的随仓库权重 | 其他项目都需要 GPU 或下载 |
| 执行方式改变后的不变量 | [test_core.py](../tests/test_core.py)：缓存、推测回滚、专家副本、CP 合并、SWA 边界、调度重排 | 与官方模型或生产 kernel 数值等价 |
| 可观察的中间状态 | [results.json](../examples/results.json)：草稿轮次和调度批次 | 对 GPU stream 或真实通信的追踪 |
| 保留收益与反例 | 固定草稿调用更少但投影更多；自适应控制器冷启动后关闭草稿 | 墙钟性能排名或普遍最优策略 |
| 可回读上游的教程 | [upstream-map.md](upstream-map.md)：固定提交、文件、符号和简化边界 | 与上游功能或模型架构完整对应 |
| 独立参考验证 | [verification.py](../nano_infer/verification.py)：逐 token oracle、四组 fixture、故障变体、源码摘要 | 独立数学规范或第三方推理后端；oracle 仍可能与实现共享规格错误 |
| 实际 shape 导览 | [shape-tour.json](../examples/shape-tour.json)：实际 prefill/decode 中间 shape | 完整 V4 多层架构导览 |
| 跨请求前缀复用 | [prefix.py](../nano_infer/prefix.py)：公共 KV 一次计算、分支复制；[测试](../tests/test_verification.py)检查隔离 | 自动 prefix 匹配、分页共享、copy-on-write 或网络缓存 |

这些是项目目前已有的组合与取舍，不是“首个”“唯一”或“全面领先”的声明。以后扩展功能时，应同步更新这一比较的版本和证据，避免把旧结论当作其他项目的现状。
