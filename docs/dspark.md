# DSpark：从方程到调度决策 / From equations to scheduling

[中文 README](../README.zh-CN.md) · [English README](../README.md)

运行 `python -m nano_infer.dspark --json dspark.json`。实现位于 [dspark.py](../nano_infer/dspark.py)，独立标量与穷举参考位于 [dspark_reference.py](../nano_infer/dspark_reference.py)。

## 1. 预测头的计算

选中主模型的多层特征，拼接后投影并做 RMSNorm：

$$f=\operatorname{RMSNorm}([h^{l_1};\ldots;h^{l_m}]P).$$

给定 draft decoder 的第 $i$ 个 **归一化前** hidden $u_i$，先用前一个已采样 token 查 Markov embedding，再计算 logits 和置信度：

$$e_i=E[t_{i-1}],\quad z_i=\operatorname{RMSNorm}(u_i)W_{lm}+e_iW_M.$$

$$c_i=\sigma\left(\operatorname{clip}\left(\frac{[u_i;e_i]w_c}{\max(T_i,10^{-5})},-60,60\right)\right).$$

本实验用 $t_i=\arg\max z_i$，再把 $t_i$ 作为下一步 Markov 输入。`norm_weight`、`lm_head`、`markov_embedding`、`markov_out`、`confidence_weight` 都是显式数组。`rank=0` 覆盖关闭 Markov 的分支。修改未来的 hidden 或温度，先前步骤的结果必须保持不变。

为什么 confidence 不能随手用 norm 后 hidden？因为它学习的输入尺度不同；测试会故意替换该输入，并检查独立参考能发现差异。温度控制 sigmoid 的尖锐程度，它不是 token sampling 的温度。

## 2. 预算与分配读取不同代际

先给每个请求分配一个 golden token。对于候选总验证长度 $Q$，将 **N−2** 轮 marginal acceptance estimates 从大到小排序，得到 $r_j$：

$$\mathbb{E}[\mathrm{tokens}(Q)]=n+\sum_{j=1}^{Q-n}r_j.$$

取能容纳 $Q$ 的最小 graph bucket $g(Q)$，用启动曲线、运行时斜率和 sampler 耗时估计成本：

$$t(Q)=t_{startup}(g(Q))+\max(0,s)g(Q)+t_{sampler},\quad Q^*=\arg\max_Q\frac{\mathbb{E}[\mathrm{tokens}(Q)]}{t(Q)}.$$

最后按最新可用置信度分配 $Q^*-n$ 个 draft 槽。普通配对执行读取 N−1；lookahead 尚未 post-process 时，最新值仍属于 N−2。`schedule_count` 与 `process_count` 显式表达这个差异。不能把本轮候选后部的置信度用于决定本轮总预算。

`schedule()` 接收调度器使用的 marginal rates，直接求和；不要再自行乘一次 cumulative product。启动曲线为空时验证全部已有草稿；历史缺失时按是否已有 draft 填 1 或 0；吞吐相同时保留较小 Q。所有候选的 graph、预期 token、耗时和吞吐都写入 JSON。

## 3. 怎样证明实现正确

| 检查 | 可展示的证据 |
|---|---|
| 24 组 projection、RMSNorm、Markov、confidence 方程 | 标量求和参考与数组实现比较，包含 rank 0/1/2 |
| 80 组调度场景 | 穷举候选子集寻找最大预期收益，独立验证总预算与分配 |
| 代际隔离 | 改 N−1 数据不能改变由 N−2 决定的总预算；能改变请求分配 |
| 错误检出 | 去掉 Markov、混用 norm 前后 hidden、把逐步温度改成常数都会失败 |
| 边界 | 空曲线、缺失历史、无 draft、相同吞吐、负 runtime 斜率与非法 counter |

实际数值结果：[examples/dspark.json](../examples/dspark.json)。`Engine` 的在线草稿仍来自 `BigramDraft`；这里独立运行 DSpark head 与调度方程，输入是可控张量和历史。下一步可以实现一个适配器，将训练后的 draft decoder hidden 与这些 head、历史结构连接起来。

## English

Run `python -m nano_infer.dspark --json dspark.json`. The equations above are executable: target feature concatenation/projection/RMSNorm; post-norm LM logits plus a previous-token Markov residual; pre-norm confidence with a per-step sigmoid temperature. Greedy selection feeds the selected token into the following Markov step. The temperature floor is `1e-5`; the confidence logit clamps to `[-60, 60]`.

The scheduler chooses total verification work using N−2 estimates, then allocates slots using the latest available rates. Paired execution has N−1 rates; one-step lookahead may still have N−2. Graph capacity rounds Q upward; runtime is startup time plus a nonnegative slope correction and sampler cost. Missing history, no startup curve and strict-throughput ties have explicit rules. Rates are already scheduler marginal estimates, so the implementation sums them directly.

The independent reference uses scalar arithmetic for heads and exhaustive subsets for small scheduling cases. The suite checks 24 equation fixtures and 80 scheduling fixtures, temporal isolation, future-step causality and plausible implementation faults. [Tests](../tests/test_dspark.py) show each assertion. The existing `Engine` uses `BigramDraft`; this laboratory executes the DSpark head and scheduler on controlled tensors and histories. Connecting a trained draft decoder is a concrete extension exercise.

## 固定源码 / Pinned sources

All equations are mapped to DeepInfer commit `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`:

- [DFlash feature projection and sample_drafts](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/mtp/dflash.py)
- [Confidence head reference equation](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/dflash.py)
- [MTPVarlen::schedule](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/mtp_varlen.hpp)
- [SAMPLING_EPS](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/sampling_params.py)
