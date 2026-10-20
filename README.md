# nano-deepseek-infer

[中文](README.md) | [English](README.en.md)

**用一台普通电脑，从零跑通推理引擎的关键机制，沿源码讲清 DeepInfer 的设计与取舍。**

[![Numerical contracts](https://github.com/PeriodBLUE/nano-deepseek-infer/actions/workflows/test.yml/badge.svg)](https://github.com/PeriodBLUE/nano-deepseek-infer/actions/workflows/test.yml)

面向初学者与推理工程求职者 · 中英双语 · 仅 NumPy · 无需 GPU 或模型权重

为什么生成一个 token 要读历史 KV？为什么多猜几个 token 有时反而更慢？为什么复制专家可以缓解负载，却不能随意更改路由结果？

从 [`infer.py`](infer.py) 开始：**请求陆续到达 → 分块 prefill → 调度草稿验证 → KV 提交/回滚 → 输出与释放。** 整条路径只依赖 NumPy，输出自动与全前缀重算对照。再用八个专题实验拆解细节，用 [`bench.py`](bench.py) 测量同一工作负载下的收益与成本，最后回到 DeepInfer 源码。

[开始运行](#三分钟运行) · [系统设计与面试导读](docs/interview-guide.md) · [DeepInfer 源码地图](docs/upstream-map.md) · [相关项目对比](#能力与资源矩阵)

> 独立教学项目，非 DeepSeek 官方项目。模型是随机初始化的单层 attention + top-2 MoE，输出没有学习过的语言含义。本项目复现可验证的机制与不变量，不加载 DeepSeek 权重，也不宣称复现其性能。

## 三分钟运行

Python 3.10+。Windows、Linux、macOS 均可使用，运行时依赖只有 NumPy。

```bash
git clone https://github.com/PeriodBLUE/nano-deepseek-infer.git
cd nano-deepseek-infer
python -m venv .venv
```

激活环境，任选对应的一行：

```bash
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -e .
python infer.py
```

如果 PowerShell 禁止激活脚本，可以直接使用 `.\.venv\Scripts\python.exe` 代替 `python`，无需修改系统执行策略。

默认运行的一段输出如下；`r1` 在 `r0` 已经开始 decode 后到达。`input` 包括 pending 与草稿，`rollback` 表示本轮丢弃的 KV 条目。

```text
step request phase    input draft accept rollback cache     output
   1 r0      verify       5     4      0        4 4->5      '6'
  + r1: 'deepseek inference'
   2 r0      verify       5     4      0        4 5->6      'u'
   2 r1      prefill      3     0      0        0 0->3      ''
   3 r0      verify       5     4      4        0 6->11     'uuuuu'
   3 r1      prefill      3     0      0        0 3->6      ''
```

每次运行结束都检查输出等价与缓存释放。字符没有语言含义，读者要观察的是状态怎样变化。

**改一个参数，验证一个判断：**

```bash
python infer.py --draft-tokens 0
python infer.py --draft-tokens 4 --token-budget 3 --prefill-chunk 2
python infer.py --prompt "nano" --prompt "cache" --arrival-gap 1
python bench.py --repeats 5 --json benchmark.json
```

`bench.py` 固定输入、模型种子和输出长度，比较普通缓存、固定草稿、自适应草稿与流水草稿。每种策略先 warmup，再轮换顺序重复测量；报告 CPU 墙钟耗时、吞吐和工作量，保存原始样本、环境与源码摘要。计时包含引擎执行，不包含模型构造和正确性参考；**没有“优化必须更快”的断言**。

深入观察与验证：

```bash
python -m nano_infer tour
python -m nano_infer verify
python -m unittest discover -s tests -v
python -m nano_infer demo speculative --trace
python -m nano_infer demo scheduling --trace
python -m nano_infer demo all --json lesson-results.json
python -m nano_infer verify --json verification.json
python -m nano_infer tour --json shape-tour.json
python -m nano_infer generate --prompt "nano" --max-tokens 24
```

`generate` 的字符 tokenizer 仅支持小写英文字母、数字、空格及少量标点。中文说明不意味着这个随机小模型支持中文生成。安装后也可使用 `nano-infer` 命令。

## 这个项目能帮你掌握什么

| 学习优势 | 你能具体做到 | 从哪里开始 |
|---|---|---|
| **一条完整的请求主线** | 改动请求到达时机与草稿预算，观察同一调度器中的 prefill、验证、回滚和结束 | [`infer.py`](infer.py) · [`Engine`](nano_infer/engine.py) |
| **每个概念都能看见** | 对照 prefill / decode 的实际 shape，逐轮检查草稿接受、KV 回滚与批次变化 | `tour` · `demo speculative --trace` · `demo scheduling --trace` |
| **优化有可检验的理由** | 说明改变执行方式后哪些输出必须不变，定位因果 mask、位置编码、缓存分支的错误 | `verify` · [正确性检查](docs/interview-guide.md#7-如何证明优化没有改变答案) |
| **性能判断有收益，也有成本** | 比较同输入、同输出下的 CPU 耗时、调用次数与投影工作，判断优化何时不划算 | [`bench.py`](bench.py) · [性能分析](docs/interview-guide.md#8-面试中的性能问题怎么回答) |
| **能从小模型走回生产源码** | 找到 DeepInfer 的调度、EPLB、DSpark、YoCo 和 sampler 入口，指出教学机制到生产实现还差什么 | [源码地图](docs/upstream-map.md) · [进阶路线](docs/interview-guide.md#9-走向-deepinfer-与完整模型实现) |

## 从第一次运行到能讲清设计

不要求先读完整个仓库。按下面顺序，每一步都产出一份自己可以解释的证据。

| 阶段 | 运行与阅读 | 你应该能独立回答 |
|---|---|---|
| 1. 看懂一次前向 | `tour`、`demo cache`；读 `model.py` | 为什么 prefill 的 score 是 `4×4`，下一步 decode 是 `1×5`？KV 保存了什么？ |
| 2. 看懂请求状态 | `python infer.py`；读 `engine.py`；再运行 `demo prefix` | 晚到请求怎样进入正在运行的引擎？token budget 如何约束草稿？哪些状态属于请求？ |
| 3. 讲清执行优化 | `demo speculative --trace`、`demo moe`、`demo context`、`demo replay` | 每项优化省什么、增加什么、依赖什么前提？被拒绝的草稿为何不能留在 KV 中？ |
| 4. 拿出验证证据 | `python bench.py`、`verify`、测试与[动手练习](docs/exercises.md) | 正确性通过后，优化是否真的降低了耗时？哪些结论仍需 GPU 测量？ |
| 5. 组织系统设计回答 | [面试导读](docs/interview-guide.md) → [固定版本上游源码](docs/upstream-map.md) | 如何把模型、引擎、kernel 和服务层连起来，并指出各自负责的问题？ |

## 八个实验，八个具体问题

| 实验 | 要回答的问题 | 本地实现与检查 |
|---|---|---|
| `cache` | 为什么 decode 不应重新计算整个前缀？ | 真实 Q/K/V 与因果 attention；有／无缓存生成 token 一致 |
| `prefix` | 不同请求的公共前缀能否只算一次？ | 复用前缀 KV，复制独立分支；与各请求完整重算一致 |
| `speculative` | 草稿错了怎么恢复？什么时候值得多猜几个？ | 一次块前向验证多个草稿；事务式回滚；固定预算与历史接受率预算对比 |
| `moe` | 热点专家导致的卡间失衡怎么缓解？ | top-2 路由、真实专家矩阵计算、复制／放置／分发／合并；结果数值一致 |
| `context` | 分开计算 softmax 的结果可以直接平均吗？ | 分片计算最大值、归一化常数与加权值，再合并；与完整 attention 比较 |
| `replay` | 只重放尾部有哪些严格前提？ | 单层 SWA，保留绝对位置及回滚余量；当前／回退位置和后续 decode 一致 |
| `scheduling` | 长 prompt 和短请求如何共同推进？ | token budget、chunked prefill、请求结束后退出批次、CPU lookahead、独立随机数流 |
| `structured` | 怎样让生成遵循格式约束？ | 对合法下一字符的 logits 做 mask；实际生成两种 JSON 候选之一 |

每个实验先执行正确性断言，再输出结果。记录：[实验结果](examples/results.json) · [实际 shape](examples/shape-tour.json) · [独立参考验证](examples/verification.json)。浮点误差的末位可能随 NumPy／BLAS 版本变化。验证记录包含 Python/NumPy 版本、源码摘要、四组输入及故障检出结果；这些摘要用于定位版本，不是签名或官方一致性认证。

这套默认配置的一次运行得到：

- **KV cache**：投影 token 数从 `708` 降到 `41`，输出 token 一致。这是工作量计数，不是墙钟加速倍数。
- **共享前缀**：三个请求共享 `29` 个 token 的前缀，投影 token 数从 `149` 降到 `91`，节省 `58` 次 token 投影；分支复制仍消耗内存和带宽。
- **固定长度草稿**：目标模型调用从 `48` 次降到 `26` 次，但投影 token 数从 `61` 增到 `156`。调用更少不代表 CPU 一定更快。
- **冷启动自适应草稿**：早期草稿质量差，预算降为零，目标调用 `49` 次。这一反例说明低质量草稿和不完善的控制器会增加开销。
- **MoE**：这批数据的两组模拟 rank 路由数从 `[80, 48]` 变为 `[66, 62]`，输出误差小于 `1e-12`。负载改善依赖输入分布。
- **尾部重放**：窗口 `8`、回滚余量 `3` 时，将这个单层模型的 prompt 重放从 `43` 个 token 缩至 `11` 个。

## 代码如何连起来

```mermaid
flowchart LR
    R[请求与独立 RNG] --> S[token budget / chunked prefill]
    S --> P[Work: prefill / decode / verify]
    D[请求独立的 bigram draft] --> P
    P --> W[CPU worker]
    W --> A[causal attention + KV cache]
    A --> M[top-2 MoE dispatch / combine]
    M --> L[logits]
    L --> O[采样或验证 / KV回滚 / 提交输出]
    O --> S
    O --> F[结束时释放状态]
```

`Engine(max_draft=4)` 将 greedy 草稿验证纳入调度批次，草稿也占用 token budget；每个请求独立保存 draft、历史预算与 KV。`add_request()` / `step()` 允许轮次间提交新请求，`run()` 提供离线与 CPU 流水入口。单请求 `speculative()` 保留为专题实验，与引擎共用 `greedy_prefix()` 验证规则。

建议按以下顺序阅读：

1. [infer.py](infer.py)：完整驱动循环，输入与输出都在这里。
2. [engine.py](nano_infer/engine.py)：入队 → 规划 → 执行 → 提交/回滚 → 释放。
3. [model.py](nano_infer/model.py) 与 [decoding.py](nano_infer/decoding.py)：数值前向和草稿验证。
4. [bench.py](bench.py) 与 [引擎测试](tests/test_engine.py)：固定工作负载、真实耗时与状态边界。
5. [parallel.py](nano_infer/parallel.py)、[labs.py](nano_infer/labs.py) 与 [prefix.py](nano_infer/prefix.py)：并行与前缀共享专题。
6. [verification.py](nano_infer/verification.py)：独立参考和实际 shape。[设计取舍与参考项目](docs/nano-design.md)解释为什么保留这些边界。

## 能力与资源矩阵

选择学习项目时，先看自己要补哪一层能力。本项目的特点是 **用同一个 NumPy 小模型，把机制、状态、正确性和成本连成一条学习路线**：从数组与执行轨迹出发，再进入生产代码。

| 你接下来的学习目标 | 可优先阅读的项目 |
|---|---|
| **推理机制与状态推演；用实验支撑系统设计回答** | **本项目：八个可运行实验、逐轮 trace、独立参考与上游源码地图** |
| 真实小模型接入与 Transformers 对照 | [Alexander-Qiu/nano-deepinfer](https://github.com/Alexander-Qiu/nano-deepinfer) |
| V4 模型结构、混合 attention 与压缩 KV | [singzhou/nano-deepseek-v4-inference](https://github.com/singzhou/nano-deepseek-v4-inference) |
| V4 方程、独立参考与 DSpark 调度检查 | [hebo1221/nano-deepseek-v4](https://github.com/hebo1221/nano-deepseek-v4) |
| GPU 模型执行、分页缓存与 CUDA Graph | [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) |

**符号：✓ 已有实现／明确文档；✗ 明确不支持或不满足该列条件；— 本次未核实；✓\* 有条件支持，见脚注。** 不把“未核实”当“不支持”，也不将每个勾视为优势。对其他项目的判断来自 2026-10-21 的固定版本源码／文档核对，未做横向运行或性能测量。[逐项依据](docs/comparison.md)

### 运行条件与模型用途

| 项目 | CPU tiny 路径 | 仅 NumPy 运行时 | 默认 demo 不下载权重 | 训练后权重文本生成 |
|---|:---:|:---:|:---:|:---:|
| [Alexander-Qiu/nano-deepinfer](https://github.com/Alexander-Qiu/nano-deepinfer) | ✓ | ✗ | ✗¹ | ✓ |
| [singzhou/nano-deepseek-v4-inference](https://github.com/singzhou/nano-deepseek-v4-inference) | ✓ | ✗ | ✓ | ✗ |
| [hebo1221/nano-deepseek-v4](https://github.com/hebo1221/nano-deepseek-v4) | ✓ | ✗ | ✓ | ✓\*² |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | ✗ | ✗ | ✗ | ✓ |
| **本项目** | **✓** | **✓** | **✓** | **✗** |

¹ 默认示例下载 SmolLM2-135M；它另有不下载权重的 tiny 测试。² 指其自训练 native bundle 的文本生成，不表示加载了官方 DSpark checkpoint。

### 机制与验证范围

| 项目 | V4 架构组件 | 独立数值参考 | DSpark 方程验证 | 跨请求前缀复用 | 专家副本放置实验 |
|---|:---:|:---:|:---:|:---:|:---:|
| Alexander-Qiu/nano-deepinfer | ✗ | ✓ | — | — | — |
| singzhou/nano-deepseek-v4-inference | ✓\*³ | — | ✗ | — | — |
| hebo1221/nano-deepseek-v4 | ✓\*³ | ✓ | ✓\*⁴ | — | — |
| GeeeekExplorer/nano-vllm | — | — | — | ✓ | — |
| **本项目** | **✗** | **✓\*⁵** | **✗** | **✓\*⁶** | **✓\*⁷** |

³ tiny 参考结构，不代表官方权重／生产 kernel 对齐。⁴ CPU 方程与调度向量检查，不代表官方权重推测运行时。⁵ 独立控制流的 toy oracle，共享权重与 NumPy；不是第三方推理后端对照。⁶ 显式公共前缀、分支数组复制；无分页、自动匹配或驱逐。⁷ 单进程物理专家副本模拟，无真实多卡通信。

## 与 DeepInfer 的对应和边界

源码核对基于上游提交 [`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01)，核对日期为 2026-10-21。逐项文件／符号对应见 [源码导读](docs/upstream-map.md)。

| 上游设计 | nano 中保留的部分 | 明确省略的部分 |
|---|---|---|
| 优化 kernel、模型 runtime | 真正的 NumPy attention 和 MoE 数值计算 | MLA、稀疏 attention、FP8、融合 kernel、CUDA Graph |
| EP + EPLB | 逻辑／物理专家映射、热点复制、分发与合并 | 多进程、网络通信、在线权重迁移、多历史步优化 |
| balanced CP | softmax 跨分片的数值合并原理 | 上游 zigzag/mirror query 划分及真实 collective |
| DSpark / DFlash 与置信度调度 | greedy 草稿验证、回滚、历史预算控制 | 训练的 draft/confidence head、上游代际协议、随机推测采样 |
| SWA bounded replay | 单层局部依赖的尾部重放证明 | 完整 YoCo、跨层共享 cache、压缩与稀疏状态 |
| 异步调度 | worker 执行时准备其他就绪请求 | 同一请求提前一轮调度、GPU stream / event |
| batch invariance | 同一请求独立计算、独立 RNG；改变批次顺序仍输出一致 | 高效批量矩阵乘的跨 batch 位级一致性 kernel |
| structured output | 有限候选字符串的字符约束 | 通用 JSON Schema、regex、EBNF 编译器 |

这里的 “batch” 是调度批次，worker 内逐请求执行。EP／CP 的 rank 是单进程模拟。没有 HTTP 服务、分布式 KV cache 服务或多机部署入口。

进一步阅读：[系统设计与面试导读](docs/interview-guide.md) · [设计推导](docs/design.md) · [动手练习](docs/exercises.md) · [完整英文 README](README.en.md)

## 测试与许可

27 项测试覆盖数值等价与引擎状态，包括草稿中的 EOS、极小 token budget、晚到请求、执行失败后清理、跨流水模式的输出一致，以及独立 oracle。GitHub Actions 配置 Windows/Linux × Python 3.10/3.12，并上传完整推理轨迹、CPU 基准、专题实验、shape 和验证结果。

本项目原创教学代码使用 [MIT License](LICENSE)。DeepInfer 的上游代码使用 Apache-2.0；此仓库通过链接提供来源，不包含其源码或权重。参见 [来源与致谢](NOTICE.md)。
