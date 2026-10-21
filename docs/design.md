# 从不变量理解推理引擎

本教程关注“改变执行方式后，什么必须保持不变”。建议一边阅读一边运行对应实验。所有数值计算都执行真实 attention + MoE 前向。`train.py` 可训练同一模型，`infer.py --checkpoint` 加载权重并生成文本；专题实验固定种子便于数值对照。

## 1. Prefill 与 decode：改变的是查询长度

单层 attention：

```text
X = embedding(token) + absolute_position
Q = X Wq, K = X Wk, V = X Wv
H = X + softmax(Q K^T / sqrt(d) + causal_mask) V Wo
logits = (H + MoE(H)) Whead
```

`TinyModel.forward(new_tokens, cache)` 只为新 token 计算 Q/K/V，将新 K/V 追加到缓存。每个 query 可以看见已有前缀和当前块中不晚于自己的 token。

- Prefill：一次输入多个 prompt token。
- Decode：通常一次输入一个最新 token。
- 草稿验证：一次输入 pending token 和多个候选 token。

三者共用一个接口，是理解后续设计的起点。缓存必须保持顺序与位置一致；随机层和跨请求状态会破坏比较，因此这个参考实现没有 dropout，也没有共享采样 RNG。

`cache` 实验比较有缓存与每步全前缀重算。计数器 `projected_tokens` 统计重复做了多少 token 投影，`score_elements` 统计实际分配的稠密 attention score 元素，包含随后被 mask 的元素。它们不等于 FLOPs，更不等于耗时。

## 2. 推测解码：将 KV 当作可提交、可回滚的状态

每轮开始前的不变量是：

```text
完整已知历史 = cache 对应的 token + pending
cache 尚未包含 pending
```

草稿提出 `[d0, d1, d2]` 后，一次目标前向输入 `[pending, d0, d1, d2]`：

```text
logits[0] 验证 d0
logits[1] 验证 d1（仅当 d0 被接受时才有意义）
logits[2] 验证 d2（仅当 d0、d1 均被接受时才有意义）
logits[3] 给出全部接受时的 bonus token
```

如果 `d1` 不匹配目标 argmax，只输出 `d0` 和 `argmax(logits[1])`，丢弃 d1、d2 的 KV。修正 token 成为下一轮 pending。全部接受时输出草稿及 bonus token，bonus 同样不提前写入 KV。

因此，若接受了 `a` 个草稿：

```text
新缓存长度 = 旧缓存长度 + 1 + a
本轮输出长度 = a + 1
```

未接受的草稿只能增加计算，不能改变输出。这在 `tests/test_core.py` 中通过强制全接受、全拒绝和多个随机种子测试。测试用的 oracle draft 只负责覆盖分支，运行时的 `BigramDraft` 不调用目标模型，也不偷看未来答案。

这份算法仅支持 greedy。随机推测采样需要依据目标分布 p 和提议分布 q 做接受／残差采样，不能简单比较两次抽样结果。本项目的随机采样在普通 `Engine` 中单独演示。

## 3. 为什么不能只最大化接受率

验证长度为 k 时，多验证一些位置，可能减少模型启动次数，也会增加计算。教学控制器使用：

```text
收益(k) = 1 + sum_i P(前 i 个草稿均被接受)
成本(k) = 1 + 0.25 k
选择使 收益 / 成本 最大的 k
```

成本单位是人为设定的单位，不是 GPU 毫秒。概率来自以前轮次的前缀接受统计，绝不读当前轮的目标验证结果。

默认实验同时展示固定 k=5 和自适应预算。廉价 bigram 草稿在早期几乎不可用，自适应控制器很快关闭草稿；由于没有探索机制，它也无法发现后续 bigram 已学到一些转移。这是有意保留的最小控制器局限，不能把它叫作 DSpark 复现。

上游使用训练的置信度 head、历史轮次隔离、运行时间曲线及整个批次的预算分配；见源码导读。若要扩展 nano，可加入周期性探索或滑动统计窗口，同时保持 greedy 输出等价。

## 4. MoE：专家身份与设备位置分离

路由器选择的是 **逻辑专家**。EPLB 调整的是这些专家的 **物理副本及位置**。副本必须具有相同权重，否则移动后会改变模型。

本实现执行完整链条：

```text
router top-2 → logical expert
             → choose physical copy
             → gather token rows
             → expert MLP
             → scatter weighted contributions
             → combine in fixed logical order
```

`rebalance()` 用聚合历史负载分配热点副本，再用受槽位容量约束的贪心算法放置。真正的 expert MLP 使用相同权重，但物理分发被模拟在一个 Python 进程中，rank 计数代表路由工作量，不是 GPU 利用率。

实际生产系统还需要考虑多历史步热点变化、all-to-all 通信、复制显存成本和权重迁移。即使负载下降，增加副本也可能不划算。

## 5. CP：softmax 不是各分片输出的平均值

把一个 query 的 K/V 切成分片 j。在每片计算：

```text
m_j = max(scores_j)
z_j = sum(exp(scores_j - m_j))
n_j = sum(exp(scores_j - m_j) * V_j)
```

用全局最大值 `m = max_j m_j` 合并：

```text
output = sum_j(exp(m_j - m) * n_j) / sum_j(exp(m_j - m) * z_j)
```

稳定的归一化与加权统计才是正确合并的基础。这里展示 K/V 分片的数学，不是上游 balanced CP 的完整复刻。上游还会平衡不同位置 query 的因果 attention 工作量，并维护 mirror/zigzag 布局。

## 6. 尾部重放：必须写清可见域与位置

本模型只有一层，K/V 直接来自 token embedding 与绝对位置；窗口 W 内的最后一个位置不依赖窗口外的 K/V。若允许回退 B 步，保留 W+B 个 token 可覆盖最早回退位置所需的窗口。

`bounded_replay()` 将起始绝对位置保存在 `cache.offset`，然后重算后缀。不保留位置，位置编码就会变化。测试不仅比较最后一行，也比较回退边界和后续 decode。

**不能将这个规则直接用于任意多层 Transformer。** 多层模型中的后层 K/V 可能依赖更早的隐藏状态，完整 YoCo 又有共享 cache、consumer 层、压缩／稀疏 attention 等结构。本实验只证明显式限定的一层局部依赖。

## 7. 调度与采样：请求身份不能随批次变化

`Engine` 中每个请求有独立 prompt cursor、输出、KV cache 和 RNG；启用推测解码时还拥有独立 draft 和历史预算。每轮受 `token_budget`、`max_batch`、`prefill_chunk` 限制。长 prompt 被分块，短请求可持续 decode，完成后立即退出队列并释放 cache、RNG 与 draft 状态。

`add_request()` / `step()` 是可直接阅读的执行入口；[infer.py](../infer.py) 演示轮次间加入新请求。`max_draft>0` 时，`_plan()` 将 pending 与草稿一起计入 token budget，并为修正/bonus 保留输出长度余量。`_finish()` 通过共享的 `greedy_prefix()` 验证，提交后把 KV 截断到 `旧长度 + 实际输出数`。若 EOS 位于接受前缀内部，只输出到 EOS，并丢弃余下临时 KV。推测请求必须使用 `temperature=0`。

流水模式使用一个 CPU worker。当它执行批次 A 时，主线程为尚未在 A 中的请求准备批次 B。工作计划是不可变的，A 的请求在提交结果前不能再次被调度。这避免读取尚未产生的 token。

此实现的限制：

- `run()` 中的请求在开始时提交；动态到达通过同步 `add_request()` / `step()` 演示，没有网络接入或线程安全的并发提交接口。
- 同一请求不会在前一轮未完成时再次预调度。
- worker 内逐请求算数，未实现向量化 batch。
- 没有真实 GPU 调度；线程开销可能让 CPU 版本更慢。

执行异常后，队列和活动状态全部清理，禁止继续使用已经部分变更的 KV；可以用新请求开启一次新的 `run()`。[bench.py](../bench.py)在正确性通过后测量普通缓存、固定/自适应草稿与 CPU 流水策略的真实成本。

随机数按请求种子构造，因此重新排列请求或改变 batch 大小不会改变该请求的 RNG 流。数学／数值层面的 batch invariance 与采样层面的确定性是两个条件；仅设置随机种子无法修复浮点归约路径改变导致的差异。本实现采用逐请求固定路径，未解决高性能 GPU kernel 中的该问题。

`structured` 实验进一步在采样前屏蔽非法下一字符。这里的合法语言只是 `{"ok":true}` 与 `{"ok":false}` 两个完整候选；没有实现通用语法编译器。

## 8. 独立参考、张量导览与前缀复用

`python -m nano_infer verify` 将优化路径同另一份控制流独立的实现比较。参考实现逐 query、逐 expert 求值，不调用 `TinyModel.forward`、`moe_forward` 或运行时 `softmax`。它共享模型权重和 NumPy，验证对象仅为本文 toy 方程，不能证明官方模型一致性，也不能排除两份实现共有的规格错误。

四组 fixture 覆盖两个模型种子和完整／滑动窗口注意力；另有测试覆盖非零位置和更多窗口。参考路径中的三种故障变体分别允许读取未来 token、丢弃绝对位置和遗漏 MoE。验证要求这些错误引起可检测的差异。输出附有 Python/NumPy 版本、容差、输入及 Python 源文件 SHA-256，方便核对“结果来自哪份代码”。摘要没有签名或安全认证含义。

`python -m nano_infer tour` 在真正的 forward 中记录 shape。以默认 prompt `nano` 为例，prefill score 是 `[4, 4]`，一步 decode 是 `[1, 5]`：查询只有一个新 token，KV 已增长到五个。这使导览与实际执行使用同一条路径。

`python -m nano_infer demo prefix` 演示跨请求复用。先为公共前缀构建一次 KV 与最后一行 logits，再调用 `KVCache.fork()` 为每个后缀复制独立数组。一个分支追加或回滚不会污染另一个分支。生成结果与各请求完整重算逐 token 一致。

对于 R 个请求、P 个公共前缀 token，节省的 token 投影为 `(R - 1) × P`。示例中 R=3、P=29，所以节省 58。这个实现仍会复制 O(Pd) 的 K/V 到每个分支；没有分页、引用计数、自动前缀匹配、驱逐或 copy-on-write，不能把投影节省等同于内存节省。它是独立函数入口，暂未接入调度器的全局缓存管理。
