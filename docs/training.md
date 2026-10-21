# 从 loss 到文本 / From loss to text

[中文 README](../README.zh-CN.md) · [English README](../README.md)

## 运行完整链路

```bash
python train.py
python -m nano_infer generate --checkpoint artifacts/tiny-trained.npz --prompt "the model " --max-tokens 64
python infer.py --checkpoint artifacts/tiny-trained.npz --prompt "the model " --max-tokens 64 --json artifacts/trace.json
```

默认配置为 `width=64, context=96, batch_size=8, steps=6000, learning_rate=0.002, seed=7`。模型 **87,936 个参数**，全部参与训练。仓库附带 [checkpoint](../checkpoints/tiny-trained.npz) 与 [训练记录](../examples/training.json)，可以先运行生成，再自己训练。记录中的一次 CPU 训练约 395 秒；你的耗时取决于 CPU、BLAS 和线程配置。

训练文件位于 [data/inference.txt](../data/inference.txt)，留出文件位于 [data/validation.txt](../data/validation.txt)。它们是本项目编写的短篇英文推理知识文本，采用相同字符词表。每次从训练序列取随机窗口，输入 `tokens[:-1]`，标签 `tokens[1:]`；每个窗口的位置从零开始。

## 前向、反向与服务保持一致

| 阶段 | 张量与计算 | 学习重点 |
|---|---|---|
| 输入 | `[B,T]` token → `[B,T,D]` embedding + position | 字符词表、位置与 teacher forcing |
| Attention | Q/K/V → `[B,T,T]` causal scores → weighted V → residual | softmax Jacobian、mask、矩阵乘反向 |
| MoE | `[BT,4]` router scores → top-2 → tanh expert → weighted sum | 路由选择与选中分数的梯度 |
| Output | `[B,T,V]` logits → 平均 next-character cross entropy | 稳定 log-sum-exp、softmax−onehot |
| Update | 全参数梯度 → global norm clip → Adam | 一阶/二阶矩、bias correction |
| Serve | 保存参数 → 重载 → `TinyModel.forward` → `Engine` | 权重与请求 KV 分开管理 |

`loss_and_grad()` 的训练前向使用批量数组，`TinyModel.forward()` 的服务路径支持增量 KV。测试逐位置比较二者 logits，并对 **每个参数张量** 的梯度做中心差分。top-2 索引在局部视为常量，梯度经过选中分数的 softmax；专家排序发生切换的边界不可微。

checkpoint 使用原生 NPZ，包含模型配置、词表和所有参数；读取时关闭 pickle，并检查名称、shape 和有限值。训练输出同时记录数据、留出集和 checkpoint 的 SHA-256。它是生成用模型权重，优化器状态不在 checkpoint 内；重新运行 `train.py` 从指定种子开始训练。

## 读懂结果

本次固定评估窗口的平均交叉熵（自然对数单位）：

| 指标 | 训练前 | 训练后 |
|---|---:|---:|
| 训练文本 loss | 3.8947 | 0.1499 |
| 留出文本 loss | 3.9257 | 3.2790 |

用 prompt `the model `，greedy 生成的开头为：

```text
the model learns from text. training updates the weights to predict the next character.
```

这段延续来自训练语料，展示模型确实学会了序列；训练与留出 loss 的明显差距也展示了过拟合。完整记录保留另外两个 prompt 的不流畅输出及长生成的退化，便于分析效果，而不是只看这一段。可以尝试扩大语料、改变宽度/训练步数，比较相同留出集上的变化。生成超过训练窗口时还会遇到位置外推问题。

服务验证比较普通 greedy、完整前缀重算、引擎内块验证三条路径；输出相同且结束时释放 KV。请用同一个 checkpoint 比较执行策略，避免把不同权重的输出差异误判为调度错误。

## English walkthrough

Run the three commands above to train, save, reload and serve. Defaults are width 64, context 96, batch size 8, 6,000 steps, learning rate 0.002 and seed 7. All **87,936 parameters** receive gradients. The checked-in checkpoint and training receipt allow immediate generation; the recorded CPU run took approximately 395 seconds, depending on hardware and BLAS configuration.

The original training and held-out English texts live in `data/`. Random windows produce next-character inputs/labels; positions start at zero in each window. Batched training executes embedding/position → causal attention → residual → top-2 tanh MoE → output head → mean cross entropy. Analytic backprop differentiates selected router scores while holding the local top-2 IDs fixed. Global-norm clipping and bias-corrected Adam update every tensor.

Tests compare training logits with serving logits at every position and check a central finite difference in every parameter tensor. Native NPZ checkpoints preserve configuration, vocabulary and weights; loading disables pickle and checks names, shapes and finite values. Receipts record data and checkpoint digests. These are inference checkpoints; rerunning training starts from the seed, rather than resuming optimizer state.

The table above reports natural-log cross entropy on fixed evaluation windows. Training loss falls from 3.8947 to 0.1499; held-out loss falls from 3.9257 to 3.2790. The displayed continuation is learned from the training text. The gap illustrates overfitting; the complete receipt retains less fluent prompts and long-generation degeneration. Try more varied text and compare the same held-out set. Generating beyond the training context also introduces position extrapolation.

Loaded weights pass through the existing cache/scheduler/speculation engine. Tests compare cached greedy, full-prefix recomputation and block verification, including cache release. Always hold the checkpoint fixed when measuring execution policies.
