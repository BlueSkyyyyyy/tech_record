# vLLM 研读配套代码

这四个脚本只依赖 Python 标准库，用几十到两百行把专题文章里的公式和机制跑给你看。
它们不是 vLLM 的复刻，也不是性能测试工具——vLLM 里对应的是 Triton kernel、CUDA Graph
和真实 GPU 上的调度，这里只保留"账"。

| 脚本 | 对应文章 | 做的事 |
|---|---|---|
| `kv_budget.py` | 第 2 篇 PagedAttention | 从模型结构算每 token/每块 KV 字节、块数、不同序列长度下的并发上限 |
| `block_alloc_sim.py` | 第 2 篇 PagedAttention | 分页 vs 连续预留的容量对比；块池的分配/释放/前缀共享/淘汰 |
| `prefix_cache_sim.py` | 第 3 篇 前缀缓存 | 链式哈希、命中、引用计数、LRU 淘汰、`cache_salt` 隔离 |
| `scheduler_sim.py` | 第 4 篇 调度器 | token budget、chunked prefill、TTFT 与单步工作量的权衡、抢占 |

## 运行

```bash
cd code/vllm
python3 kv_budget.py --test
python3 block_alloc_sim.py --test
python3 prefix_cache_sim.py --test
python3 scheduler_sim.py --test

# 默认场景
python3 kv_budget.py
python3 block_alloc_sim.py
python3 prefix_cache_sim.py
python3 scheduler_sim.py
```

`--test` 只做断言，不打印中间过程；四个脚本的测试都是确定性的，不需要 GPU。

## 几个可改的参数

```bash
# DeepSeek-V3 式的 MLA：61 层、每层 576 维 latent、fp8 权重按 2 字节算
python3 kv_budget.py --layers 61 --kv-heads 1 --head-dim 576

# 更大的块池（54 GiB）跑 2000 个请求
python3 block_alloc_sim.py --blocks 27648 --requests 2000

# 系统提示更长、请求更多，看命中率与淘汰次数
python3 prefix_cache_sim.py --requests 500 --sys-len 8000

# 只按首 chunk 准入，观察抢占
python3 scheduler_sim.py --no-reserve-isl --kv-blocks 256
```

## 与真实 vLLM 的差距

- **没有 GPU**：所有"时间"都是步数或 token 工作量，不是毫秒。文章里的毫秒数来自
  显存带宽/算力的量级估算，脚本只保证比例关系正确。
- **块大小固定 16**：真实 vLLM 的块大小由后端/平台决定（MLA、FA3 可能偏好 64）。
- **没有 swap**：抢占一律丢弃 KV 重算，这与 V1 一致；但真实实现还要处理异步调度、
  KV connector、投机解码的 placeholder，脚本都没有。
- **前缀缓存只模拟到块粒度**：`prefix_match_unit` 的 partial hit、EAGLE 丢弃最后一块、
  `prompt_logprobs` 禁用缓存这些边界，只在文章里讲。
