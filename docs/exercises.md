# 动手练习

每个改动都先保留参考路径，再增加可推翻自己的测试。默认命令：

```bash
python -m unittest discover -s tests -v
```

## 1. 画出 KV cache 的工作量曲线

改变 prompt 长度和生成长度，用 `TinyModel.projected_tokens` 比较 cached/uncached。先推导生成 N 个 token 时分别处理多少 token（最后一个输出不必再前向），再和程序计数核对。不要用这个比值冒充端到端加速比。

## 2. 让自适应预算恢复探索

默认控制器可能在早期失败后永久选择零草稿。加入每隔若干轮的探索，或采用最近若干轮的统计。要求：

- 多个 prompt、seed 和输出长度都与 greedy 一致。
- 报告总候选数、实际接受数、目标调用数和投影 token 数。
- 同时报告收益和额外工作，不能只挑接受率高的轮次。

进一步挑战是实现随机推测采样；它需要 p/q 接受与残差分布。正确性目标是输出分布，而不是与独立随机抽样路径逐 token 相同。

## 3. 给 EPLB 加入变化的热点

将单个负载向量改成 `[历史步, 专家]`。构造一个总负载相近但每步峰值不同的数据集，比较聚合贪心与按步峰值目标。再加入每个副本的显存成本和迁移成本。

数值不变量：同一逻辑专家的副本权重相同，输出误差仍在规定容差内。

## 4. 故意破坏 bounded replay

先删掉 `cache.offset`，观察绝对位置错位导致的失败。再增加第二层 attention，试验简单的 W+B 后缀规则为何不足；把每层的信息依赖画出来。不要把增加第二层后的失败“修复”为更宽的数值容差。

## 5. 复核请求延迟与真实 batch

直接运行 `python load_test.py --requests 8 --json service-load.json`。它启动 HTTP 服务、错峰提交请求、等待结果并对照完整重算。检查 queue time、TTFT、ITL 与实际发送时间；取消、超时、背压和故障恢复已在 `tests/test_systems.py` 中实现并测试。

运行 `python systems.py`，检查 `[1,5,5]` 的 11 个有效 query 为什么产生 15 行投影。再运行 `bench.py --batched` 与默认路径，比较相同输出下的计时和 padding 工作。分页块引用、COW、LRU 驱逐、MLA 压缩均有同一命令可执行的数值案例，见[系统导读](systems.md)。
