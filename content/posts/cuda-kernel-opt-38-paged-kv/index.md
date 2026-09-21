---
title: "CUDA 算子调优（三十八）：Paged KV-cache / flash-decoding 推理注意力——从 46% 打到 93% HBM"
date: 2026-09-21
draft: false
weight: 38
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "Attention", "decode", "PagedAttention", "KV-cache", "flash-decoding", "GQA", "Qwen3", "mma", "tensor-core", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」里推理方向的第一篇。前面几篇都在做 **prefill**（算力受限）：

- [第 16 篇]({{< relref "cuda-kernel-opt-16-mla-fused" >}})：MLA 单 kernel 融合（online softmax 进 QK epilogue、`C→A` 零 shuffle）；
- [第 19 篇]({{< relref "cuda-kernel-opt-19-dsa-sparse-attn" >}})：DSA 稀疏 MLA 消费端（gather top-k + 掩码 + 双缓冲）；
- [第 30 / 34 篇]({{< relref "cuda-kernel-opt-34-fused-moe-fp8" >}})：MoE expert FFN 的 fused / FP8，**prefill 是权重带宽受限**。

而**推理 decode** 正好反过来：`q_len = 1`，KV 是**分页**的、且要一遍遍从 HBM 重读——
这是一个**纯访存**算子，门槛不在 FLOPs，而在「能不能把 KV 读到 HBM 的 90%+」。

先给结论（Qwen3-8B，B=64，变长 32k 上下文，KV 读取 8.22 GB）：

| 版本 | 手段 | 耗时 | KV 带宽 | 占峰值 |
|---|---|---|---|---|
| v0 naive | 1 warp/(token,head)，KV 直读（读 5 遍） | 10.87 ms | 756 GB/s | 22.6% |
| v1 shared | 1 CTA/(token,kv_head)，协作搬 smem，KV 读 1 遍 | 7.27 ms | 1130 GB/s | 33.7% |
| v2 scalar + flash-decoding | v1 + 沿序列 split + 向量化 | 5.46 ms | 1506 GB/s | 44.9% |
| **v3 tensor-core（split=2×NW4）** | `mma.m16n8k16` 一次算完全部 GQA head | **2.64 ms** | **3111 GB/s** | **92.8%** |

ncu 佐证：v2 是 **issue 受限**（`Compute (SM)` 85%、IPC 3.44、DRAM 仅 44.7%）；
v3 换成 tensor core 后变成 **DRAM 受限**（`DRAM Throughput` **93.35%**、L2 90.2%、Compute 15%）——

> 核心一句话：**decode attention 的胜负手是「每个 KV 字节要摊多少条指令」。**
> 标量实现把 `G=5` 个 query head 各算一遍 dot + `__shfl` 归约，指令数压不下去；
> 用 `mma` 把 G 个 head 打包进 M 维，一次算完，指令数掉一个数量级，带宽立刻贴顶。

---

## 一、decode attention 的 roofline：为什么它只考访存

decode 阶段每个请求只有 1 个 query token，但 KV 是历史全部 key/value。对 GQA（`G = Hq/Hkv` 个
query head 共享同一份 KV）的单个 (token, kv_head)：

$$
\text{FLOPs} = 4\,G\,L\,D,\qquad \text{bytes} = 2\,L\,D\cdot 2\ \text{(K+V, bf16)}
$$

（第一个 4 = QK 与 PV 各 2 FLOP/MAC；注意 KV 只读一遍，因为 G 个 head 共享。）

于是算术强度：

$$
\text{AI} = \frac{4GLD}{4LD} = G
$$

Qwen3-8B 的 `G = Hq/Hkv = 40/8 = 5`。H100 的 bf16 tensor-core ridge 约
`989e12 / 3.352e12 ≈ 295 FLOP/byte`，而 decode 只有 **5**——**差 60 倍**，
所以只要写对，它一定卡在 HBM，而不是算力。判据很简单：**看 ncu 的 `DRAM Throughput`，
不是 `Memory Throughput`，更不是 TFLOPS。**

真实参数取自 `/ssd/models/qwen3-8B/config.json`：

```
num_attention_heads   (Hq)  = 40
num_key_value_heads   (Hkv) = 8      -> G = 5
head_dim              (D)   = 128
hidden_size                 = 5120
```

KV-cache 布局（vLLM / FlashInfer 口径）：

```
k_cache / v_cache : [num_blocks, block_size(P), n_kv_heads, head_dim]
block_table       : [batch, max_blocks]     # request -> 物理页号
seqlen            : [batch]
```

一个「页」存 `P=16` 个 key。**同一页内，某个 kv_head 的数据不是连续的**：第 p 行在
`((page*P + p)*Hkv + kv)*D`，行间跨 `Hkv*D*2 = 2 KB`——这是后面 gather 访存的第一个坑。

---

## 二、标量路线：三次尝试，卡在「指令发射」

### v0 naive —— 每 warp 一个 (token, q_head)

最直白的实现：一个 warp 负责一个 query head，lane 各管 `D/32 = 4` 个维度，
每读一个 key 就 `dot4` + 5 步 `__shfl` 归约出 score，online softmax，再 `PV`。
问题是 **G=5 个 head 各读一遍 KV**：流量 ×5。

```
KV 流量 = 5 × B·Hkv·L·D·2·2
```

实测 10.87 ms / 756 GB/s，按它自己 5× 的流量算其实有 ~3.8 TB/s 的等效带宽——
但**总量就是 5 倍**，没救。教训：**先算流量，再谈带宽。**

### v1 shared —— 一个 CTA 一个 (token, kv_head)，KV 搬进 smem

把 `G` 个 warp 放进同一个 CTA，KV tile 由所有线程协作搬进 smem（只从 HBM 读一遍），
每个 warp 从 smem 读自己 head 的 dot。16B 向量化（`float4` 一次搬 8 个 bf16）。

```
for blk in pages:
    cooperative_load(ks, vs)      # 全 CTA 协作，K/V 各一遍
    __syncthreads()
    for p in 0..P-1:              # 每个 key
        score = warp_sum(dot4(q, ks[p]))
        online_softmax_update(...)
    __syncthreads()
```

到 1130 GB/s（33.7%）。还不够。

### v2 flash-decoding —— 沿序列 split

长上下文下单个 (token, kv_head) 的 KV 太长、并行度不够。**flash-decoding** 把序列切成
`nsplit` 段，每段独立算 partial `(O, m, l)`，再用一个 combine kernel 合并：

$$
P_{\text{final}}[d] = \frac{\sum_s e^{m_s - m^\star}\, l_s\, O_s[d]}{\sum_s e^{m_s - m^\star}\, l_s},
\qquad m^\star = \max_s m_s
$$

扫 `nsplit = 1…32`，5.46 ms / **1506 GB/s（44.9%）** 到顶。这时 ncu 给了一记闷棍：

```
Compute (SM) Throughput   85.0%     <- 不是访存！
DRAM Throughput           44.7%
Executed Ipc Active        3.44
Issue Slots Busy          85.0%
Registers Per Thread         40
Achieved Occupancy        66.6%
```

**它是「指令发射」受限，不是内存受限。** 拆开看每个 key 每个 warp 的指令数：

```
dot4            4 FMA
warp_sum        5 SHFL + 5 FADD
softmax         2 exp + 2 FMUL/FADD
PV              4 FMA + 2 bf16 解包
```

再加上地址计算、循环、两次 `__syncthreads`——ncu 实测 `sm__inst_executed ≈ 4.9e9` 条
warp 指令、`sm__inst_executed_pipe_fma ≈ 4.0e9`，而真正有用的 FMA 只有 ~6.7e8。
**标量路线在 Hopper 上打不满，是因为 SM 的发射口被 `__shfl`/地址算/标量 FMA 占满了。**

结论：**要贴 HBM，必须把指令数砍掉一个数量级——上 tensor core。**

---

## 三、v3 tensor-core：一次 mma 算完全部 GQA head

关键观察：`G=5` 个 query head 共享同一份 KV，这正是可以打包进 `mma` 的 M 维。

```
S[G,P]  = Q[G,D] · K[P,D]^T        # QK^T: M=G(补齐16), N=P, K=D
O[G,D] += P[G,P] · V[P,D]          # PV  : M=G(补齐16), N=D, K=P
```

用 `mma.sync.aligned.m16n8k16`（bf16 输入 / fp32 累加），M 维取 16（G=5 补齐，11 行浪费无所谓，
反正 tensor core 闲置很多）。每个「页」（P=16 个 key）只需要：

- QKᵀ：`8 个 k-step × ldmatrix.x4` + **16 条 mma**
- softmax：行 = head（`lane/4`），列 = key
- PV：`8 个 dn-step × ldmatrix.x4.trans` + **16 条 mma**

相比标量路线的几百条指令，**每个页 ~32 条 mma**，指令数直接掉一个数量级。

### 布局与两个复用

沿用 [第 16 篇]({{< relref "cuda-kernel-opt-16-mla-fused" >}}) 那套已经验证过的片段布局：

- **K 存成 `[p][d]`（行主序）**，非转置 `ldmatrix` 直接得到 mma 要的 B（`.col`：lane 行通道 = n、列通道 = k）；
- **V 存成 `[p][d]`**，PV 要 B 是 `[k=p][n=d]`，用 `ldmatrix.x4.trans` 取；
- **`S` 的累加器布局恰好等于 PV 的 A 片段布局**：softmax 后把相邻两个 fp32 `pack2` 成 bf16x2，
  **零 shuffle** 就喂给 PV（16 篇的免费午餐）。

```
         ks[p][d]  (n=p, k=d)                vs[p][d]  (k=p, n=d)
            │                                    │
   ldmatrix (non-trans)                 ldmatrix.trans
            ▼                                    ▼
      B(n8k16)  ──mma──►  S[G,P] ──softmax──► P[G,P] ──pack2──► A(m16k16)
                            (行=head)                                │
                                                          mma ◄──────┘
```

### warp 内并行：一个 warp 管一组页，无跨 warp 同步

一个 CTA = 一个 `(token, kv_head)` × 一个 split；CTA 内 `NW` 个 warp **各管一组页**
（`blk = blk0 + warp; blk += NW`）。每个 warp 有自己的 smem tile（`ksv[warp][2][P][D+8]`），
**只在 warp 内 `__syncwarp()`，没有任何 `__syncthreads()`**。Q 只在 CTA 开始时搬一次（G 行有效、其余置 0）。

每个 warp 写自己的 partial，`combine` 再跨 `nsplit × NW` 合并。

代码见 `code/kernel-opt/38-paged-kv/paged_decode.cu:219`（`paged_decode_mma`）。
其中尾页掩码（`code/.../paged_decode.cu:293` 附近）很关键：分页的最后一块可能不足 P 个 key，
必须把 `key >= valid` 的列在 softmax 前置 `-inf`，否则会把页尾的垃圾 key 算进去。

---

## 四、实测：从 44.9% 到 92.8%

主实验：`B=64`、变长上下文（91%~100% of 32768）、KV 读取 **8.22 GB**，
`scripts/run.sh 38-paged-kv/paged_decode.cu 64 32768 2 all`。

```
v0 naive                   10.8711 ms   756.1 GB/s   (22.6%)     <- 自己读了 5 遍
v1 shared                   7.2705 ms  1130.5 GB/s  (33.7%)
v2 split=1                  7.2785 ms  1129.3 GB/s  (33.7%)
v2 split=16                 5.5539 ms  1480.0 GB/s  (44.1%)
v2 split=32                 5.4561 ms  1506.5 GB/s  (44.9%)     <- 标量天花板
v3 split=1x4                2.9147 ms  2820.0 GB/s  (84.1%)
v3 split=2x4                2.6423 ms  3110.8 GB/s  (92.8%)     <- 最佳
v3 split=8x4                2.6651 ms  3084.2 GB/s  (92.0%)
v3 split=2x8                2.9514 ms  2785.0 GB/s  (83.1%)
v3 split=2x16               2.9900 ms  2749.0 GB/s  (82.0%)
dense（无 page-table）      7.2797 ms  1129.1 GB/s  (33.7%)     <- 标量版对照
```

**v3 相对标量 v2 快 2.07×，带宽 ×2.07。** 逐项 ncu 对照（同一 shape）：

| 指标 | v2 scalar（split=16） | v3 mma（split=2×4） |
|---|---|---|
| Duration | 5.50 ms | **2.63 ms** |
| `DRAM Throughput` | 44.7% | **93.35%** |
| `L2 Cache Throughput` | 47.8% | 90.2% |
| `Compute (SM)` | **85.0%** | 15.1% |
| IPC | 3.44 | 0.60 |
| Registers/thread | 40 | 164 |
| Achieved occupancy | 66.6% | 17.2% |
| 瓶颈 | **发射口** | **HBM** |

`DRAM 93.35%` / `L2 90.2%` 说明**已经贴到 HBM 屋顶**（ncu 的 duration 2.63 ms 对应 3.13 TB/s）。
occupancy 只有 17% 也没关系——访存受限的 kernel，occupancy 不是第一优先级，
**足够的在飞请求（MLP）+ 每个字节摊的指令数**才是。

### 扫参：split × warp 数

| 配置 | split=1 | split=2 | split=4 | split=8 | split=16 |
|---|---|---|---|---|---|
| NW=4 | 2820 | **3111** | 3029 | 3084 | 3017 |
| NW=8 | — | 2785 | 2766 | — | — |
| NW=16 | — | 2749 | 2673 | — | — |

**NW=4 完胜 NW=8/16**：NW=4 时动态 smem 只有 `4×2×16×136×2 = 34.8 KB`（+4 KB 静态），
每 SM 能放 3 个 CTA；NW=8/16 时 smem 翻倍，锁到 2/1 CTA，在飞请求反而变少。
这也是全系列反复出现的教训：**别默认「更多 warp / 更大分块 = 更快」，先算 smem/寄存器账。**

### 上下文长度 & batch 规模

上下文越短越难满带宽（总请求少、尾效应重）；batch 越大越接近屋顶（并行度足够）：

```
# L=4096:  v3 split=2x4 = 2925.7 GB/s (87.3%)
# L=8192, v3 split=2x4:
#   B=1   354 GB/s (10.6%)   B=8  2023 GB/s (60.3%)
#   B=32 2694 GB/s (80.4%)   B=64 3034 GB/s (90.5%)   B=128 2987 GB/s (89.1%)
```

B 小的时候并行度只有 `B·Hkv·nsplit·NW` 个 warp，填不满 132 个 SM；
这时要**把 split 拉大**：B=1、L=32768 从 split=2（9.1%）一路加到 split=16（48.7%）才勉强起来。
生产里通常靠 continuous batching 把 batch 顶上去，正是这个道理。

---

## 五、对标 SOTA：和 torch SDPA / flash-attn / 纯读天花板比

`paged_ref.py` 把分页 KV gather 成 dense（这一步 torch 的 `view/reshape` 是零拷贝），
再跑 torch SDPA（`enable_gqa`）与 `flash_attn_func` 的 decode 路径，并测一个纯 KV 读带宽上限。
为了同口径，手写 kernel 用 `UNIFORM=1` 让所有序列等长（KV 恰好 8.59 GB）：

```
raw KV read          2.772 ms   3098.8 GB/s     <- 纯读，HBM 实际可达
torch SDPA (gqa)     2.687 ms   3196.9 GB/s     <- 稠密 KV（无分页）的上界
flash_attn decode    3.278 ms   2620.2 GB/s
ours  paged decode   2.735 ms   3140.9 GB/s     <- 92.5%~93.7% HBM
```

| | 耗时 | KV 带宽 | 占比 |
|---|---|---|---|
| 本系列 v3 paged decode | 2.735 ms | 3141 GB/s | 100% |
| torch SDPA（稠密，无分页）| 2.687 ms | 3197 GB/s | **1.018×** |
| flash_attn decode | 3.278 ms | 2620 GB/s | 1.20× |
| 纯 KV 读（HBM 可达）| 2.772 ms | 3099 GB/s | 1.014× |

- 手写 kernel **达到 torch SDPA 的 98.2%**（差 1.8%），且仅比「纯读 KV」慢 ~1%；
- 比 `flash_attn` 的 decode 路径**快 1.20×**（它的通用 varlen kernel 对这个 shape 不是最优）；
- **分页的代价几乎为零**：把物理页号全局打乱（`RANDOM=1`）后 2.771 ms vs 顺序 2.735 ms，只慢 **1.3%**。
  因为每个页内仍是 16 行 × 256 B 的规整访问，页号随机只影响 DRAM row 局部性，H100 的 HBM 完全吃得下。

正确性：`.cu` 内置一个**结构独立**的参考（`gather` 到 dense + 两趟非 online softmax，`paged_decode.cu:422`），
各版本对拍 `max_abs_err ~ 3–6e-4`（bf16 正常量化误差）；torch 侧 `flash_attn vs SDPA` 也一致到 `1.2e-4`。

---

## 六、小结

- **decode attention 是纯访存算子**：AI = G ≈ 5，比 H100 的 tensor-core ridge 低 60 倍，
  **先看 ncu `DRAM Throughput`**；
- **标量路线在 Hopper 上必然卡发射口**：v2 的 `Compute (SM)` 85%、IPC 3.44、DRAM 只有 44.7%，
  `__shfl` 归约 + 地址算 + 标量 FMA 把发射槽占满；
- **tensor core 是分水岭**：把 G 个 GQA head 打包进 `mma` 的 M 维，一次算完 QKᵀ 与 PV，
  每页仅 ~32 条 mma，**44.9% → 92.8%**；
- **复用 16 篇的片段布局是免费的**：`S` 累加器 = PV 的 A 片段，`pack2` 后零 shuffle；
  K/V 在 smem 里都存 `[p][d]` 行主序，一个非转置、一个转置 `ldmatrix`；
- **NW=4 打赢 NW=8/16**：小 smem 换高 occupancy，在飞请求更多；
- **分页本身几乎不花钱**：打乱物理页只掉 1.3%；真正贵的是「标量实现对每个 KV 字节要发的指令」。

**下一篇预告**：把同一套分页框架接到 **MLA**（DeepSeek-V4 / Kimi 的 `kv_lora_rank + rope` 吸收形式，
`Hkv=1` 的 MQA）与 **decode 的 FP8 KV-cache**，看吸收式 MLA 在 decode 下能否同样贴顶；
以及把这个 kernel 和 [第 25 篇]({{< relref "cuda-kernel-opt-25-moe-grouped-gemm" >}}) 的 continuous batching
一起接回端到端 token 吞吐。

---

**配套代码**：`code/kernel-opt/38-paged-kv/`（`paged_decode.cu`、`paged_ref.py`；
`sweep_B64_L32768.out.txt`、`sweep_B64_L4096.out.txt`、`batch_scan_L8192.out.txt`、
`smallbatch_scan.out.txt`、`uniform_random.out.txt`、`paged_ref_B64_L32768.out.txt`、
`ncu_v3_2x4.out.txt`、`ncu_scalar_v2.out.txt`）。

运行：

```bash
cd code/kernel-opt
scripts/run.sh 38-paged-kv/paged_decode.cu 64 32768 2 all
scripts/ncu.sh 38-paged-kv/paged_decode.cu --set detailed \
    --kernel-name regex:paged_decode_mma --launch-count 1 -- 64 32768 2 v3b
docker exec kernel_lab python3 38-paged-kv/paged_ref.py 64 32768
```
