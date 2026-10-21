# 从项目成果到 AI Infra 面试

[English](career-map.en.md) · [README](../README.md)

学完这个项目，你应该能独立完成三件事：**从训练权重跑出文本；解释每轮请求与缓存状态；用对照实验评估一个推理优化。** 面试中的自信来自这些可复现的成果，以及你对代码和结果的解释。

## 官方岗位要求与项目证据

核对日期：**2026-10-21**。下表概括公司官方 JD，并将要求映射到本项目的练习与产物；这是项目作者的能力映射。岗位是否仍在招聘，以链接页面为准。

| 公司与岗位（官方 JD） | JD 中的重点 | 通过本项目可以拿出的证据 | 面试时可以讲清的问题 |
|---|---|---|---|
| [Anthropic — Performance Engineer, Inference Engine](https://job-boards.greenhouse.io/anthropic/jobs/5418323008) | 请求 batching、状态复用、内存管理、可观测性、确定性与测试 | `infer.py` 的迟到请求、KV 提交/回滚/释放；`test_engine.py` 的 EOS、失败与重排检查 | 哪些状态属于请求？什么时刻可以复用、提交或释放？ |
| [OpenAI — Software Engineer, Inference - Performance Optimization](https://openai.com/careers/software-engineer-inference-performance-optimization-san-francisco/) | 端到端性能分析、microbenchmark、延迟/容量/成本建模 | `bench.py` 的固定工作负载、warmup、原始样本；DSpark 的收益/耗时目标 | 为什么调用更少可能更慢？怎样证明优化针对了真正的瓶颈？ |
| [NVIDIA — Inference Performance Engineer, Agent Driven Inference Optimization，JR2021689](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Inference-Performance-Engineer--AI-Inference-Configuration-Optimization_JR2021689) | continuous batching、KV、MoE、推测解码；受控实验和模型正确性 | 草稿预算对比、专家副本放置、缓存工作量、DSpark 穷举参考 | 如何在吞吐、验证成本和模型输出之间做有证据的取舍？ |
| [Together AI — Machine Learning Engineer - Inference](https://job-boards.greenhouse.io/togetherai/jobs/4385540007) | 推理 runtime、可靠性、测试、开发文档；推测解码 | 训练→checkpoint→生成→引擎的完整路径；错误清理、独立参考与双语文档 | 如何让一个模型接入可维护的推理引擎？怎样验证优化及异常路径？ |

这些岗位同时要求各自层级的工程经验，以及 GPU、系统编程或分布式系统能力。下面把本项目直接覆盖的目标与可继续在本仓库实践的目标分开，帮助你安排准备顺序。

## 学习目标验收表

✓ 表示本仓库已有运行路径；✗ 表示需要完成表中的扩展任务。勾表示可以练习并出示证据，个人掌握程度由你能否独立修改、运行和解释来检验。

| 能力目标 | 本仓库直接实践 | 应提交的个人成果 |
|---|:---:|---|
| 从张量推导 attention、MoE 与 next-token loss | ✓ | 画出 forward/backward shape；解释所有参数的梯度检查 |
| 训练、保存、重载并生成文本 | ✓ | 自己的训练配置、留出集 loss、checkpoint 摘要与生成样本 |
| prefill/decode、连续调度、KV 生命周期 | ✓ | 修改到达间隔和 token budget；逐轮标注 cache 长度 |
| 推测解码、拒绝回滚与 DSpark 方程 | ✓ | 手算一轮接受前缀；解释 N−2 总预算与 N−1 分配；构造反例 |
| MoE 放置、负载与 softmax 分片合并 | ✓ | 改热点分布，报告负载变化与数值误差 |
| 性能实验与成本判断 | ✓ | 固定输入输出，分别报告工作量与墙钟耗时，解释一次负收益 |
| 可复现测试与故障定位 | ✓ | 新增一个能检出状态错误的测试，展示修复前失败、修复后通过 |
| CUDA/Triton kernel 与 GPU profiling | ✗ | 将一个 attention 或 MoE 算子移植到 GPU；记录 profiler 与数值误差 |
| 多卡通信与分布式运行 | ✗ | 把模拟 rank 改为进程，测量通信量、等待与热点 |
| 在线服务与 SLO | ✗ | 增加 API、取消/超时、背压和负载生成器；测 TTFT/ITL 的 p50/p95/p99 |

## 一份能带去面试的项目包

1. **模型证据**：运行 `train.py`，解释 train/validation loss 为什么不同；加载同一 checkpoint，证明保存不会改变预测。
2. **状态证据**：运行 `infer.py --checkpoint checkpoints/tiny-trained.npz --prompt "the model " --max-tokens 64 --json my-trace.json`，手算一轮草稿拒绝后的 cache 长度。
3. **方程证据**：运行 `python -m nano_infer.dspark`，挑一个 graph bucket 边界，计算期望收益和耗时；修改最新置信度，观察总预算与请求分配如何变化。
4. **性能证据**：运行 `bench.py --repeats 5 --json my-benchmark.json`，说明计时范围、工作负载、参考答案和误差来源。
5. **个人贡献**：完成验收表里一项扩展，保留失败案例、修复、测试和测量记录。

简历可以据实组织成：“实现可训练的 NumPy attention + MoE 与 checkpoint 推理；实现带 token budget 的请求调度、KV 回滚与释放；用独立标量/穷举参考验证 DSpark head 和预算分配，并用固定工作负载分析推测解码的收益与成本。”写入你自己测得的数字，并准备解释每一项。

面试追问练习：权重变了为什么旧 KV 不能复用？top-2 路由的梯度在哪里不连续？为什么 confidence 读取 norm 前 hidden？为什么当前轮的未来置信度不能反过来决定当前总预算？为什么 CPU p95 不能直接作为在线服务的 p95？答案应能回到代码、公式与自己的实验。
