# nano-deepseek-infer

**把 DeepInfer 风格的系统优化，拆成七个可运行、可对照、可追踪的 NumPy 实验。**

[![Numerical contracts](https://github.com/PeriodBLUE/nano-deepseek-infer/actions/workflows/test.yml/badge.svg)](https://github.com/PeriodBLUE/nano-deepseek-infer/actions/workflows/test.yml)

中文教程 · Python + NumPy · 无 GPU · 无权重下载 · 七个可执行实验

> 独立教学项目，非 DeepSeek 官方项目。模型是随机初始化的单层 attention + top-2 MoE，输出没有学习过的语言含义。本项目复现可验证的机制与不变量，不加载 DeepSeek 权重，也不宣称复现其性能。

## 与其他 nano 推理项目有什么不同

这个项目围绕一个学习目标组织代码：**验证改变缓存、草稿、专家放置和请求调度之后，哪些结果必须保持不变，以及额外工作量出现在哪里。**

CPU 可运行、无权重下载和数值测试在其他项目中也已存在。这里的定位是将 **NumPy 小模型、系统机制对照、状态轨迹和优化反例** 放在同一套短教程中。下面是 2026-10-21 对相关项目 README、依赖及部分实现／测试的核对；这是学习路线对比，没有运行它们做横向性能排名。[固定版本与核对依据](docs/comparison.md)

| 项目 | 主要学习对象 | 运行路径与已有能力 | 更适合的学习目标 |
|---|---|---|---|
| [Alexander-Qiu/nano-deepinfer](https://github.com/Alexander-Qiu/nano-deepinfer) | `LLM → Scheduler → ModelRunner → Llama + KVCache` 的小型引擎 | CPU / PyTorch；默认示例加载 SmolLM2-135M；已有分块 prefill、连续批处理和 Transformers 数值对照 | 从真实小模型生成文本，理解引擎职责划分与模型接入 |
| [singzhou/nano-deepseek-v4-inference](https://github.com/singzhou/nano-deepseek-v4-inference) | V4 模型内部的张量与缓存路径 | CPU tiny / PyTorch；SWA、CSA、HCA、mHC、MoE，以及 prefill/decode 等价测试 | 逐算子理解 V4 架构和压缩 KV；这方面比本项目更完整 |
| [hebo1221/nano-deepseek-v4](https://github.com/hebo1221/nano-deepseek-v4) | V4 架构参考与一致性验证 | CPU / PyTorch + NumPy；无下载 demo、独立参考对照、DSpark 方程与调度验证 | 深入架构方程及 DSpark 验证；其验证范围不能简化为“只实现模型” |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 面向实际模型的精简 GPU 推理引擎 | CUDA / PyTorch / FlashAttention；模型权重、prefix caching、tensor parallelism、CUDA Graph | 阅读实际 GPU 执行、缓存块管理和性能优化 |
| **本项目** | **DeepInfer 风格系统机制的对照实验** | **仅 NumPy；随机单层模型；七个实验共用同一个数值测试载体** | **在 CPU 上逐项检查状态变更、结果等价和工作量代价，再回读 DeepInfer** |

### 在这里可以直接观察的差异

1. **围绕同一个小模型比较执行方式。** 缓存／重算、逐 token／块验证、专家原始放置／复制重排、完整／分片 attention 都有参考路径。模型保持刻意简单，读者可以把注意力放在系统状态上。[实现入口](nano_infer/labs.py)
2. **把接受、回滚和调度过程作为实验输出。** `--trace` 展示每轮 `proposed`、`accepted`、`rolled_back`、`cache_tokens`，以及每批请求的阶段、输入 token 数和完成状态；不只展示最终文本。[完整示例轨迹](examples/results.json)
3. **同时展示优化收益和失败情况。** 默认固定草稿预算把目标调用从 `48` 次降到 `26` 次，却把投影 token 从 `61` 增到 `156`；简单自适应预算冷启动后关闭草稿，调用反而变成 `49` 次。它们解释不同成本如何变化，不构成加速比承诺。[控制器与局限](docs/design.md#3-为什么不能只最大化接受率)
4. **将每个教学机制接回固定版本的上游源码。** 可以从本地测试追到 DeepInfer 的 scheduler、EPLB、MTP 和 YoCo 入口，同时看到哪些生产细节被省略。[源码对应表](docs/upstream-map.md)

选择本项目意味着接受明确的取舍：没有训练好的语言能力、完整 V4 架构、真实多卡通信或高性能 batch kernel。需要这些能力时，上表提供了相应的后续学习入口。“更小”和“依赖更少”描述的是本项目的教学取向，不代表对其他项目的性能或正确性优势。

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
python -m nano_infer demo all
python -m unittest discover -s tests -v
```

如果 PowerShell 禁止激活脚本，可以直接使用 `.\.venv\Scripts\python.exe` 代替 `python`，无需修改系统执行策略。

单独观察草稿接受／回滚和调度轨迹：

```bash
python -m nano_infer demo speculative --trace
python -m nano_infer demo scheduling --trace
python -m nano_infer demo all --json lesson-results.json
python -m nano_infer generate --prompt "nano" --max-tokens 24
```

`generate` 的字符 tokenizer 仅支持小写英文字母、数字、空格及少量标点。中文说明不意味着这个随机小模型支持中文生成。安装后也可使用 `nano-infer` 命令。

## 七个实验，七个具体问题

| 实验 | 要回答的问题 | 本地实现与检查 |
|---|---|---|
| `cache` | 为什么 decode 不应重新计算整个前缀？ | 真实 Q/K/V 与因果 attention；有／无缓存生成 token 一致 |
| `speculative` | 草稿错了怎么恢复？什么时候值得多猜几个？ | 一次块前向验证多个草稿；事务式回滚；固定预算与历史接受率预算对比 |
| `moe` | 热点专家导致的卡间失衡怎么缓解？ | top-2 路由、真实专家矩阵计算、复制／放置／分发／合并；结果数值一致 |
| `context` | 分开计算 softmax 的结果可以直接平均吗？ | 分片计算最大值、归一化常数与加权值，再合并；与完整 attention 比较 |
| `replay` | 只重放尾部有哪些严格前提？ | 单层 SWA，保留绝对位置及回滚余量；当前／回退位置和后续 decode 一致 |
| `scheduling` | 长 prompt 和短请求如何共同推进？ | token budget、chunked prefill、请求结束后退出批次、CPU lookahead、独立随机数流 |
| `structured` | 怎样让生成遵循格式约束？ | 对合法下一字符的 logits 做 mask；实际生成两种 JSON 候选之一 |

每个实验先执行正确性断言，再输出结果。完整记录见 [examples/results.json](examples/results.json)；浮点误差的末位可能随 NumPy／BLAS 版本变化。

这套默认配置的一次运行得到：

- **KV cache**：投影 token 数从 `708` 降到 `41`，输出 token 一致。这是工作量计数，不是墙钟加速倍数。
- **固定长度草稿**：目标模型调用从 `48` 次降到 `26` 次，但投影 token 数增加。调用更少不代表 CPU 一定更快。
- **冷启动自适应草稿**：早期草稿质量差，预算降为零，目标调用 `49` 次。这一反例说明低质量草稿和不完善的控制器会增加开销。
- **MoE**：这批数据的两组模拟 rank 路由数从 `[80, 48]` 变为 `[66, 62]`，输出误差小于 `1e-12`。负载改善依赖输入分布。
- **尾部重放**：窗口 `8`、回滚余量 `3` 时，将这个单层模型的 prompt 重放从 `43` 个 token 缩至 `11` 个。

## 代码如何连起来

```mermaid
flowchart LR
    R[请求与独立 RNG] --> S[token budget / chunked prefill]
    S --> W[CPU worker]
    W --> A[causal attention + KV cache]
    A --> M[top-2 MoE dispatch / combine]
    M --> L[logits]
    L --> O[采样 / 输出 / 释放缓存]
    O --> S
    D[廉价 bigram draft] --> V[块验证与 KV 回滚]
    V --> A
```

调度引擎和单请求推测解码是两个独立教学入口，共用 `TinyModel`。没有把推测解码强行塞进批次状态机，便于先分别证明正确性。

建议按以下顺序阅读：

1. [model.py](nano_infer/model.py)：一次前向怎样追加 KV，以及 prefill 和 decode 为什么是同一接口。
2. [decoding.py](nano_infer/decoding.py)：`pending` token、不接受后缀的回滚、请求独立采样。
3. [parallel.py](nano_infer/parallel.py)：逻辑专家与物理副本、分片 softmax 的合并。
4. [engine.py](nano_infer/engine.py)：等待队列、不可变工作计划、执行、提交输出。
5. [labs.py](nano_infer/labs.py) 与 [tests/test_core.py](tests/test_core.py)：每项设计能被什么实验推翻。

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

进一步阅读：[设计推导](docs/design.md) · [动手练习](docs/exercises.md) · [英文简介](docs/english.md)

## 测试与许可

测试覆盖缓存／块计算等价、全接受／全拒绝、生成长度边界、回滚、热点专家副本、上下文分片、SWA 回退边界、批次顺序／大小／流水模式、EOS 和缓存释放。GitHub Actions 配置 Windows/Linux × Python 3.10/3.12。

本项目原创教学代码使用 [MIT License](LICENSE)。DeepInfer 的上游代码使用 Apache-2.0；此仓库通过链接提供来源，不包含其源码或权重。参见 [来源与致谢](NOTICE.md)。
