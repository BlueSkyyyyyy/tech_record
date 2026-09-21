---
title: "CUDA 算子调优（二十七）：MoE 的 router、top-k 与 token 置换 —— 搬运才是大头，置换已贴 HBM 天花板"
date: 2026-09-21
draft: false
weight: 27
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "router", "top-k", "token permutation", "DeepSeek", "DeepSeek-V4", "sqrtsoftplus", "noaux_tc", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[上一篇]({{< relref "cuda-kernel-opt-26-moe-cluster-multicast" >}})把 MoE 的 expert FFN 做成了 grouped GEMM，[第 25 篇]({{< relref "cuda-kernel-opt-25-moe-grouped-gemm" >}})把 384 个 expert 折叠进一个 kernel。但那两个 kernel 的输入是**已经按 expert 归拢好的 permuted 激活**——这篇补上它的"前门"：**MoE router 的完整链路**。

**问题**：DeepSeek-V4 / Kimi-K2.6 每个 token 要从 **384 个 routed expert** 里选 **top-6**，权重还要做 `sqrtsoftplus` 打分、加 `e_score_correction_bias`（noaux_tc）、归一化、乘 `routed_scaling_factor`；选完还要把 token 按 expert **置换（permute）**好喂给 grouped GEMM，算完再 **反置换（unpermute）**加权求和。

我按 `/ssd/models/DeepSeek-V4-Pro/config.json` 的真实参数把这七个 kernel 全部手写并实测。结论先行：

- **gate GEMM** 我们做到 **204→257 TFLOPS**，但它只是 cuBLAS 的 **27.7%**（`738.2`）——这是这条链唯一明显落后 SOTA 的地方，根因是 `mma.sync` 路径的发射/延迟墙（ncu：`L2 60.7%`、`Compute 35.3%`、`No Eligible 57.6%`、occupancy 20%）。
- **router top-k + 元数据**（直方图/前缀和/游标 scatter）只要 **0.07 ms@16k**，把直方图融进 top-k 还省 15%。
- **置换才是大头**：`permute + unpermute` 占端到端 **69%**，而且**已经贴住 HBM 天花板**——permute `81%`、unpermute `91.7%` HBM。其中 permute v2（一个 block 读一次 `x[t]`、写 6 行）比 v1 快 **1.53×**。
- 对标 PyTorch：**permute 快 1.5×、unpermute 快 17.8×、端到端快 6.6×**（torch 的 `index_add` 反向加权只有 173 GB/s）。

| 阶段 | M=16384 实测 | 占端到端 | 效率 | 对标 |
|---|---|---|---|---|
| gate GEMM | 0.439 ms | 27% | 204 TFLOPS（20.7% 峰值） | cuBLAS 738（**27.7%**） |
| router top-k + meta（融合） | 0.071 ms | 4% | compute 62.6%、occ 84.7% | — |
| permute（v2/token） | 0.605 ms | 37% | **2717 GB/s（81.0% HBM）** | torch index_select 0.910 ms（**1.50×**） |
| unpermute（fused 加权） | 0.535 ms | 32% | **3075 GB/s（91.7% HBM）** | torch index_add 9.49 ms（**17.8×**） |
| **合计** | **1.651 ms** | — | — | torch 10.95 ms（**6.6×**） |

环境：H100 SXM 80GB（132 SM，实测 HBM **3352.3 GB/s**，bf16 TC dense 989 TFLOPS），CUDA 13.2。
数字来自 `code/kernel-opt/27-moe-router/moe_M{4096,8192,16384,32768}.out.txt`、`ncu_*.out.txt`、`torch_ref_M16384.out.txt`。

---

## 0. 算法：DeepSeek-V4 的 noaux_tc routing

参数取自 `/ssd/models/DeepSeek-V4-Pro/config.json`（Kimi-K2.6 同构，只有 `scoring_func`/`topk` 不同）：

```jsonc
"hidden_size": 7168,              // H
"n_routed_experts": 384,          // E
"num_experts_per_tok": 6,         // K (top-k)
"scoring_func": "sqrtsoftplus",   // V3/Kimi 是 sigmoid
"topk_method": "noaux_tc",        // 有一个只用于“选”的 bias
"norm_topk_prob": true,           // 选完的权重归一化
"routed_scaling_factor": 2.5      // 再乘 2.5
```

routing 的六步（与 vLLM 的 `_torch_topk_softplus_sqrt` 逐项对齐）：

$$
\begin{aligned}
\text{logits} &= x\,W_g^{\top}, &\quad x\in\mathbb{R}^{M\times H},\ W_g\in\mathbb{R}^{E\times H}\\
s &= \sqrt{\mathrm{softplus}(\text{logits})}, &\quad \mathrm{softplus}(z)=\ln(1+e^{z})\\
\text{choice} &= s + b, &\quad b = \texttt{e\_score\_correction\_bias}\\
\text{ids} &= \mathrm{topk}(\text{choice},\,K) &\quad \text{（按 choice 选，用的是 bias 后的分）}\\
w &= \frac{s[\text{ids}]}{\sum_j s[\text{ids}_j]}\times \gamma &\quad \text{（权重用**未加 bias** 的 }s\text{）}
\end{aligned}
$$

关键点：**bias 只参与"选谁"，不参与"权重多少"**（`noaux_tc` = no auxiliary loss + 一个可学习的 correction bias）。选出来的 6 个里 `s` 会被归一化再乘 `γ=2.5`。

然后置换。设每个 token $t$ 选中的专家集合为 $I_t$（$|I_t|=K$），定义置换 $\pi$：把所有 $(t,j)$（$t$ 的 slot $j$）按专家号归拢成连续区间：

```
 token:      0        1        2          ...        M-1
 ids:     [3,7,1] [7,2,0] [3,5,3]  ...            [ ... ]
          │ │ │   │ │ │   │ │ │                     │
          ▼ ▼ ▼   ▼ ▼ ▼   ▼ ▼ ▼                     ▼
 permute: 按 expert 分桶，相同 expert 的 (t,j) 连续排列
          expert 0: (1,2) (5,0) ...
          expert 1: (0,2) ...
          expert 3: (0,0) (2,0) (2,2) ...          ← expert_offsets 给出每段起点
          ...
 布局:    [permuted_x] 行数 = M*K，行 p 的 token = permuted_token[p]
 逆置换:  pos[t*K+j] = p    （t 的第 j 个 slot 去哪了）
```

grouped GEMM 消费 `permuted_x`，算完的 `permuted_y` 由 unpermute 反向加权：
$$
y_t = \sum_{j=0}^{K-1} w[t,j]\cdot \text{permuted\_y}[\text{pos}[tK+j]]
$$
因为同一个 token 的 K 份输入完全相同，所以有一个漂亮的不变量可以自检：**unpermute 的输出必须等于 $x_t \cdot \sum_j w[t,j]$**（置换若写错，这个等式立刻崩）。

---

## 1. gate GEMM：`logits = X @ Wgᵀ`

形状 `[M,7168] × [7168,384] → [M,384]`。N=384 很小（只有 3 个 128 列 tile），是典型 **tall-skinny / small-N** GEMM。我用[第 13 篇]({{< relref "cuda-kernel-opt-13-tensor-core" >}})的 `mma.m16n8k16 + ldmatrix + cp.async` 双缓冲，把 `Wg` 预先转置成 `WgT[H,E]` 复用标准 `B[K,N]` 布局（`moe_router.cu:175` 的 `gate_mma_pipe_t`）。

先把 helper 重构成模板 `<BMt,WMt,WNt>`，扫两种 M-tile：

| 配置 | 线程 | 寄存器 | CTA/SM | M=4096 | M=8192 | M=16384 | M=32768 |
|---|---|---|---|---|---|---|---|
| **BM=128**（默认） | 256 | 126 | 2 | 132.9 | 198.3 | **204.3** | **256.6** |
| BM=256 | 512 | 127 | 1 | 90.8 | 178.3 | 182.2 | 234.2 |

（单位 TFLOPS。）**BM=256 全线更慢（−10%）**，这是一次有价值的负结果：BM=256 本该把 `WgT` 的 L2 重读减半，但 512 线程 ×127 寄存器 = 65024，只剩 **1 CTA/SM**；而 BM=128 有 **2 CTA/SM**。占用率从 2 掉到 1 的损失 > 省下的 L2 字节（呼应[第 23/24 篇]({{< relref "cuda-kernel-opt-23-fp8-gemm-tma" >}})的同一教训）。所以我用 BM=128。

**ncu（BM=128, M=16384）**：

```
Compute (SM) Throughput   35.3%
L2 Cache Throughput       60.7%     ← 最高
DRAM Throughput           17.6%
L1/TEX                    52.0%
Registers/Thread          126
Theoretical/Achieved Occ  25% / 20.1%
No Eligible               57.6%     ← 每周期只有 0.63 个 warp 可发射
```

瓶颈有三层：① **L2 60.7%**——每个 m-tile 都要把整块 `WgT`（5.5 MB）从 L2 拉进 smem，`M/128` 个 CTA 就是 ~700 MB 的 L2 读，加上 A 的 3 次重读；② **`Compute` 只有 35%**，`No Eligible 57.6%`、warp 每发射一条要等 **7.59 cycle**（其中 2.5 cycle 是固定延迟依赖）——`mma.sync` 是同步指令，发射槽被占死；③ occupancy 20%，没有足够的 warp 去藏延迟。

这正是 `mma` 路径在 Hopper 上的天花板（第 13 篇 221 TFLOPS、这里 257）。**要追 cuBLAS 的 738 TFLOPS，必须换 `wgmma`（异步、SS 直读 smem，省发射槽）甚至 TMA + warp specialization**（第 22/23 篇的路线）。这是本系列的下一步（见文末）。

---

## 2. router top-k：1 warp 一个 token

384 路的 top-6 不需要排序，也不需要 radix。每行只有 384 个元素，**一个 warp 处理一个 token** 最直接（`moe_router.cu:219`）：

```
每 lane 处理 12 个 expert：s=sqrt(softplus(logit)), choice=s+bias → 写 smem(sc[], bd[])
重复 6 次：
   lane 本地扫 12 个取最大 → 5 步 shfl_down 归约出 (max_val, max_id)
   lane0 把 bd[max_id] 置 -inf（保证下轮不重复选）
lane0：sum = Σ sc[sel[j]]；w[j] = sc[sel[j]]/sum*2.5
```

第一版是 **1 warp/block**：`M` 个 block、每块 32 线程——`WgT` 的占用率只有 50%（H100 每 SM 最多 32 个 block，32×32=1024 线程）。改成 **8 warp/block**（每 warp 一个 token，smem 按 warp 分片）后，占用率上到 84.7%：

| 版本 | 线程/block | M=4096 | M=8192 | M=16384 | M=32768 |
|---|---|---|---|---|---|
| 1 warp/block | 32 | 0.0115 | 0.0197 | 0.0356 | 0.0817 |
| **8 warp/block** | 256 | **0.0116** | **0.0173** | **0.0312** | **0.0610** |

（单位 ms。）大 batch 上 **+13~25%**。ncu（M=16384）：`Compute 62.6%`、`DRAM 21.4%`、`Achieved Occ 84.7%`——已经转到 compute-bound（`expf` + 6×5 次 shuffle），不再是瓶颈。

**正确性**：对 8 个随机 token 用 CPU 重算 `sqrtsoftplus + bias + topk + renorm`，top-6 专家集合 **8/8 完全一致**，权重最大误差 **5.96e-8**（`check_M16384.out.txt`）。

---

## 3. 路由元数据：直方图 → 前缀和 → scatter

要把 token 归拢，需要三步：

1. `route_count_kernel`（`:280`）：对 `ids[M*K]` 做 expert 直方图（全局 `atomicAdd`）；
2. `route_scan_kernel`（`:286`）：单 block Hillis–Steele，把 384 个计数变成 **exclusive prefix sum** `expert_offsets[E+1]`；
3. `route_scatter_kernel`（`:302`）：每个 entry 用 `atomicAdd(&cursor[exp],1)` 抢到一个 permuted 位置 `p`，写 `pos[t*K+j]=p`、`permuted_token[p]=t`、`permuted_weights[p]=w`、`inv[p]=t*K+j`。

**优化：把直方图融进 top-k**。top-k 的 lane0 本来就知道 6 个 `sel[j]`，顺手 `atomicAdd(&cnt[sel[j]],1)` 即可，省掉一次 kernel 启动和一遍 `ids` 读取：

| M | 独立 count+scan+scatter | **topk(融合直方图)+scan+scatter** | 省 |
|---|---|---|---|
| 8192 | 0.0312 ms | **0.0447 ms**（含 topk） | — |
| 16384 | 0.0523 ms | **0.0710 ms**（含 topk） | vs 0.0356+0.0523=0.0879 → **−19%** |
| 32768 | 0.0929 ms | **0.1271 ms**（含 topk） | vs 0.0817+0.0929=0.175 → **−27%** |

**正确性**：`Σ expert_counts = M*K`；`offsets[e+1]-offsets[e] = counts[e]`；且每个 permuted 段 `[offsets[e],offsets[e+1])` 里的 token 的 `ids` 确实包含 `e`（`meta check ... offsets=OK ptok membership=OK`）。

---

## 4. 置换：两种写法，差 1.53 倍

置换要写 `M*K` 行、每行 7168 个 bf16。最直观的写法 **v1（一个 block 一个 permuted 行）**：block `p` 读 `x[permuted_token[p]]` 写 `px[p]`（`moe_router.cu:320`）。问题：每个 token 被它自己的 6 个 slot 各读一次，`x` 被读 **6 遍**——流量 `(K+1)M H·2`。

**v2（一个 block 一个 token）**：block `t` 只读一次 `x[t]`，然后写它的 6 个目标行（`:333`）：

```cuda
for (int i = threadIdx.x; i < n8; i += blockDim.x) {
  const uint4 v = src[i];                 // 读一次
  #pragma unroll
  for (int j = 0; j < TOPK; ++j)
    reinterpret_cast<uint4*>(px + (size_t)p[j] * H)[i] = v;   // 写 K 次
}
```

流量降到 `M H·2 + K M H·2`（读 1 次 + 写 K 次），**1.64 GB vs v1 的 2.82 GB**：

| M | v1/行 (GB/s, %HBM) | **v2/token** | v3/token + `stcs` | v2 加速 |
|---|---|---|---|---|
| 4096 | 0.2161 ms (3260, 97.3%) | 0.1540 (2669, 79.6%) | 0.1531 | 1.40× |
| 8192 | 0.4516 (3121, 93.1%) | 0.3040 (2704, 80.7%) | 0.3022 | 1.49× |
| 16384 | 0.9241 (3050, 91.0%) | **0.6052 (2717, 81.0%)** | 0.6013 | **1.53×** |
| 32768 | 1.8687 (3017, 90.0%) | 1.2058 (2727, 81.4%) | 1.1980 | 1.55× |

有意思的权衡：v1 的 HBM 效率**更高**（单读单写、两条顺序流），但流量大 1.72×；v2 效率降到 81%（写侧是 6 条散落的目标行，DRAM row 局部性变差），但**净时间赢 1.53×**。v3 用 `st.global.cs`（streaming store，别污染 L2）只快 **0.6%**（噪声级）——因为写出去的行马上要被 unpermute 读，L2 缓存本来也没坏处。

**峰值占比**：v2 的 81% 和 v1 的 91% 都说明置换是**纯 HBM 带宽问题**。ncu（v2, M=16384）：`DRAM 80.7%`、`L2 72.1%`、`Compute 7.0%`、`Achieved Occ 86.1%`——教科书级的带宽受限。

---

## 5. unpermute：加权求和，91.7% HBM

反向（`moe_router.cu:369`）：一个 block 一个 token，对 6 个 slot 做 `y[t] += w[t,j] * px[pos[t,j]]`，直接 bf16 累加到 fp32：

```cuda
for (int i = threadIdx.x; i < n8; i += blockDim.x) {
  float a[8] = {0};
  #pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    const int p = pos[t*TOPK + j];
    const float w = wgt[t*TOPK + j];
    uint4 v = ((const uint4*)(px + (size_t)p * H))[i];   // 读 6 行
    // a[q] += w * bf16(v[q])
  }
  // 写 y[t]
}
```

这个 kernel 天然是最小的：读 `K·M·H·2`、写 `M·H·2`，没有冗余。实测：

| M | ms | GB/s | %HBM |
|---|---|---|---|
| 4096 | 0.1379 | 2981 | 88.9% |
| 8192 | 0.2697 | 3048 | 90.9% |
| 16384 | **0.5347** | **3075** | **91.7%** |
| 32768 | 1.0645 | 3089 | 92.1% |

ncu：`DRAM 91.9%`、`3.08 TB/s`、`Compute 13.4%`。**这是一条真正贴顶的 kernel。**

---

## 6. 端到端：搬运占 69%，对标 PyTorch

M=16384 的分解（融合版 top-k+meta）：

```
gate GEMM        ████████████ 0.439 ms  (27%)   204 TFLOPS / cuBLAS 27.7%
route+topk       ██ 0.071 ms            ( 4%)
permute (v2)     █████████████████ 0.605 ms (37%)  81.0% HBM
unpermute        ██████████████ 0.535 ms   (32%)  91.7% HBM
                 ───────────────────────────
                 合计 1.651 ms     ← permute+unpermute 占 69%
```

对标同机 PyTorch（`moe_router_ref.py`，cuBLAS + `torch.topk` + `index_select` + `index_add`）：

| 阶段 | 手写 | PyTorch | 我们 |
|---|---|---|---|
| gate GEMM | 204 TFLOPS | **738 TFLOPS**（cuBLAS） | **0.28×** |
| permute | 0.605 ms（1.64 GB） | 0.910 ms（2.82 GB） | **1.50×** |
| unpermute | **0.535 ms** | 9.49 ms（173 GB/s！） | **17.8×** |
| 端到端 | **1.651 ms** | 10.95 ms | **6.6×** |

两个故事：

- **torch 输在反向置换的算法**：`index_add_` 配 fp32 广播 + 物化 `px.float()`，只有 **173 GB/s（5.2% HBM）**，一次 unpermute 要 9.5 ms。手写融合加权、直接 bf16 累加，**17.8×**。
- **torch 的置换其实也贴带宽**：`index_select` 真实流量 `2·P·H·2 = 2.82 GB`、0.910 ms = **3100 GB/s（92% HBM）**——它只是比我们多读了 5 遍 `x`。我们赢的 1.5× 完全来自「读一次、写 K 次」。
- **我们唯一落后的是 gate GEMM**：`204 vs 738`。整条 router 里 gate 占 27%，把它追到 cuBLAS 水平，端到端能从 1.65 ms 压到 ~1.33 ms。

---

## 7. 小结

- **DeepSeek-V4 的 noaux_tc routing 手写复现**：`sqrtsoftplus → +bias 选专家 → top-6 → 用未加 bias 的分归一化 ×2.5`；top-6 集合与 CPU 逐 token 一致，权重误差 6e-8。
- **gate GEMM 是 `mma` 路径的墙**：204→257 TFLOPS，ncu 显示 `L2 60.7% / Compute 35.3% / No Eligible 57.6%`；BM=256 想省 L2 但掉 occupancy（−10%）。**下一站是 `wgmma`/TMA。**
- **top-k 用「1 warp/token + 6 轮 argmax」最省事**，把直方图融进去省 19–27%；8 warp/block 比 1 warp/block 快 13–25%。
- **置换是一道纯带宽题**：`permute+unpermute` 占端到端 69%，其中 unpermute 到了 **91.7% HBM**；permute 的关键优化是**让 `x[t]` 只读一次**（1.53×），代价是写侧 6 条散流把效率从 91% 拉到 81%——但时间赢。
- **对标 SOTA**：置换/反置换大幅领先 PyTorch（1.5× / 17.8×），端到端 6.6×；gate GEMM 是可量化的短板（27.7%）。

**下一篇**预告：把 gate GEMM 换成 **`wgmma` + TMA + warp specialization**（复用第 23 篇的基础设施），目标把 gate 从 204 推向 500+ TFLOPS，让整条 router 端到端压进 1 ms；以及把 permute 直接**融进 grouped GEMM**（gather 版，省掉物化 `permuted_x` 的 1.41 GB 写）。

> 复现：`code/kernel-opt/27-moe-router/`
> `../scripts/run.sh 27-moe-router/moe_router.cu 16384 all`（自测 + 计时 + 正确性）
> `../scripts/ncu.sh 27-moe-router/moe_router.cu --set full --kernel-name regex:permute_copy_tok -- 16384 all`
