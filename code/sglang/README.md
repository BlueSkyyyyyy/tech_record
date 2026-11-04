# SGLang 研读配套代码

这五个脚本只依赖 Python 标准库，用几百行把专题文章里的机制跑给你看。全 CPU、
秒级、确定性，不需要装 SGLang，也不需要 GPU。它们不是 SGLang 的复刻，也不是压测
工具——真实实现里对应的是 RadixAttention 的树操作、CUDA Graph 和 NCCL 上的
all-to-all，这里只保留「账」。

| 脚本 | 对应文章 | 做的事 |
|---|---|---|
| `radix_tree_sim.py` | 第 2 篇 基数树 / 第 12 篇 与 vLLM 对照 | 树 vs 块对拍：同一前缀池上比命中率、重算 token、分裂与淘汰 |
| `mem_budget.py` | 第 3 篇 KV 池与显存账本 | 从模型结构算 KV/token、`mem_fraction_static` 反推、vLLM 块数 |
| `scheduler_sim.py` | 第 4 篇 调度器 | decode 优先 + chunked prefill 的装箱，retract 与 preempt 的 victim 选择 |
| `overlap_sim.py` | 第 5 篇 两级 overlap | 批间流水与 TBO 的时间线账本：谁在等谁、省下的是哪一段 |
| `jump_forward.py` | 第 7 篇 结构化输出 | 跳词能省多少 forward/采样/掩码（**上界估计**，主路径未接线） |

## 运行

```bash
cd code/sglang
python3 radix_tree_sim.py --test
python3 mem_budget.py --test
python3 scheduler_sim.py --test
python3 overlap_sim.py --test
python3 jump_forward.py --test

# 默认场景（打印账本表，overlap_sim 还会画前 4 步的 ASCII 时间线）
python3 radix_tree_sim.py
python3 mem_budget.py
python3 scheduler_sim.py
python3 overlap_sim.py
python3 jump_forward.py
```

`--test` 只做断言，不打印中间过程；五个脚本的测试都是确定性的，不需要 GPU。

## 几个可改的参数

```bash
# 更多请求 / 更小的前缀池：看块粒度多吃掉多少重算
python3 radix_tree_sim.py --requests 200 --capacity 60000

# 40 GB 卡、TP=2：看 reserved_mem 与 mem_fraction_static 怎么变
python3 mem_budget.py --gpu 40 --tp 2

# 收紧 token 预算与 KV，逼出 retract / preempt
python3 scheduler_sim.py --requests 200 --budget 1024 --kv 12000

# 671B / EP16 的量级：CPU 3 ms、GPU 45 ms，通信占比仍按 18%
python3 overlap_sim.py --cpu-ms 3 --gpu-ms 45 --comm-frac 0.18

# 关掉通信，看 TBO 只剩启动开销
python3 overlap_sim.py --comm-frac 0

# 短骨架 + 小词表：掩码字节数跟着词表走，步数比不变
python3 jump_forward.py --keys 6 --vocab 32000
```

## 与真实实现的差距

- **没有 GPU**：`overlap_sim.py` 里的毫秒、`jump_forward.py` 里的 forward 次数都是
  模型算出来的，不是实测。文章里的毫秒数来自显存带宽/算力的量级估算，脚本只保证
  比例关系正确；真实数字受 kernel、batch、并行度影响，可能差一倍以上。
- **radix / 块都做了简化**：没有模拟 page 对齐（真实 vLLM 的块是 16 token，SGLang
  的 `req_to_token` 也有对齐）、没有 hybrid / SWA 池（mamba 与 sliding-window 的
  多池结构在 `kv_cache_configurator.py` 里，脚本只有一个池），也没有 CUDA Graph
  对 `bs` 的桶约束。
- **`scheduler_sim.py` 的 lpm 排序没有建模**：只对拍了「KV 不够时牺牲谁」这一处
  差别（SGLang 丢生成最少的，vLLM 丢已算 KV 最多的）。真实的最长前缀匹配排序、
  priority / 到达顺序、chunked prefill 的切块上限都只在文章里讲。
- **`jump_forward.py` 是上界估计，而且当前主路径未接线**：基线 commit `1ef3335` 上
  `try_jump_forward` 在 `srt/managers/`、`srt/speculative/` 下没有调用点，
  `outlines_backend.py:160-161` 还把 `jump_forward_map` 置成 `None`，所以线上真实
  收益是 0。脚本假设 key 与 value 各占一个 token，跳词步数比真实情况更乐观。
- **`mem_budget.py` 忽略了固定池与页对齐**：`req_to_token` 表、mamba/swa 池、
  expert slot、以及块对齐造成的尾部浪费都没有计入，所以算出来的是 KV 池的容量
  上界，真实可用 token 数会低几个百分点。
- **`overlap_sim.py` 的 TBO 只到「账」这一层**：真实 TBO 按层拆 stage、用
  `YieldOperation` 错位，还要过模型家族、a2a backend、DP 一致性三道门；脚本把
  通信当成一段可被另一半算力覆盖的时间，没有 SM 争用（通信会占 SM，净收益要求
  `r > n/(N-n)`）。
