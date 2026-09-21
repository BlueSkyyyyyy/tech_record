---
title: "CUDA 算子调优（五十四）：融合 cross-entropy —— 把 2.12 GB 的 logits 从显存里省掉，和那个卡在 L2 上的 LM head"
date: 2026-09-22T07:00:00+08:00
draft: false
weight: 54
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "cross-entropy", "LM head", "融合算子", "DeepSeek-V4", "TMA", "wgmma", "logsumexp", "L2", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 32 篇]({{< relref "cuda-kernel-opt-32-fused-norm" >}})把 RMSNorm 推到 87% HBM，
[第 53 篇]({{< relref "cuda-kernel-opt-53-fused-rope" >}})把 RoPE 贴到访问模式的天花板。
这一篇做 Transformer 尾巴上最大的一个「隐性中间张量」：**LM head 的 logits**。

训练里最后一步是 `lm_head` 的 $[M,H]\times[V,H]^\top$ GEMM，紧接着对 logits 做 cross-entropy：

$$
\mathcal{L} = \frac{1}{M}\sum_{m}\Big[\underbrace{\log\sum_{v} e^{z_{m,v}}}_{\text{logsumexp}} - z_{m,t_m}\Big],\qquad z_{m,v}=x_m\cdot w_v .
$$

问题出在中间那个 $[M,V]$ 的 `logits`。以 DeepSeek-V4-Pro 为例（`hidden_size=7168`、`vocab_size=129280`），
$M=8192$ 个 token 时：

| 张量 | 形状 | bf16 | fp32 |
|---|---|---|---|
| `logits` | `[8192, 129280]` | **2.12 GB** | **4.24 GB** |
| 权重 `W` | `[129280, 7168]` | 1.85 GB | — |

一个只在两个 kernel 之间传手的中间量，比模型权重还大。**融合 cross-entropy**（Liger-Kernel 的
招牌算子）就是把这个张量彻底干掉：GEMM 的累加器还在寄存器里时，就地做 tile 级 logsumexp，
只把每个 token 的 `(max, sumexp)` 部分和写出去。

本篇把它实现出来并测清楚三件事：

1. **怎么融合**：split-N 的非对称 GEMM + epilogue online logsumexp + 跨 tile 合并；
2. **一个关键工程点**：朴素 schedule 下 LM head 的 DRAM 流量高达 **60 GB**，
   一个 L2 友好的 superblock swizzle 把它压到 **17.7 GB（3.4×）**；
3. **这个算子的真相**：在 $M=8192$ 这种 compute-bound 形状上，省掉 logits 只值 **~2.7 GB 流量 / ~1.3 ms**，
   而 GEMM 本身占 20 ms——融合的收益是**显存**，不是吞吐。我们手写的 wgmma GEMM 比 cuBLAS
   慢 **19%**，正好把省的这点吃回去。

结论先放这：融合版 **25.07 ms（605.7 TFLOPS，61.2% bf16 峰值）**，对标 cuBLAS+CE 的两 kernel
基线 **24.53 ms**（0.98×）、PyTorch eager 端到端 **23.04 ms**（0.92×）、
而「fp32 logits」基线 **29.04 ms（1.16×）**。logits 显存占用从 2.12/4.24 GB 降到 **66 MB**。

---

## 一、这个算子为什么值得融合

先把账算清楚。GEMM 本身是 $2MHK$ FLOP，而 cross-entropy 只是对 $[M,V]$ 做一遍行归约。
唯二会「变慢」的地方是：

- **物化 logits**：GEMM 要多写 $MV$ 个元素，CE 再多读一遍。流量 $2\cdot MV\cdot \text{sizeof}$。
- **logsumexp 的读写趟数**：PyTorch 的 `log_softmax` 习惯先把 logits 升到 fp32，于是又多出一份
  $[M,V]$ fp32（4.24 GB）的写+读。

以本机 H100 的 3.35 TB/s 算，`logits` 的 bf16 往返（写 2.12 + 读 2.12）值 **1.27 ms**，
fp32 版本再翻倍。但 GEMM 要 20 ms——所以**融合省下的是几个百分点的时间，和几个 GB 的显存**。

这里有个可以量化的判据。GEMM 的 DRAM/片上流量（2D tiling、无 L2 复用时）是
$\text{tile traffic}=M V K\cdot 2\cdot(1/B_M+1/B_N)$，logits 往返是 $2MV\cdot2$，两者之比

$$
\frac{\text{logits 往返}}{\text{GEMM 流量}} = \frac{2}{K\,(1/B_M+1/B_N)} .
$$

$K=7168$、$B_M=256,B_N=128$ 时只有 **2.4%**。也就是说：**只要 $K$ 够大（LM head 的 hidden 一般都不小），
融合 CE 在吞吐上注定是小赢或平手**——它的核心价值是把峰值显存从「几个 GB」降到「几十 MB」，
这正是长上下文 / 大词表训练里训练不 OOM 的关键。Liger 的动机也是这个。

---

## 二、融合设计

### 2.1 计算图

```
              ┌───────────────── 一个 CTA = (m_tile, n_tile) ─────────────────┐
x[M,H] ──┐    │   TMA ──► smem SW128 ──► wgmma.m64n128k16 (SS)              │
         ├─►  │        acc[BM][BN] (fp32, 寄存器)                             │
W[V,H] ──┘    │            │                                                 │
              │            ├─ 行 max (4-lane shfl 归约) ──► partial_max[m][nt]│
              │            ├─ Σ exp(acc-max)          ──► partial_sum[m][nt]│
              │            └─ target 落在本 tile 时     ──► target_logit[m]   │
              └─────────────────────────────────────────────────────────────┘
                                          │
                        combine_kernel：跨 T=V/BN 个 tile 合并
                        lse[m] = max_t pm + log Σ_t ps·exp(pm - max_t pm)
                        loss   = mean_m (lse[m] - target_logit[m])
```

关键决定：**沿 N（词表）切，不沿 K 切**。每个 `(m_tile, n_tile)` CTA 完整吃掉 $K$，
输出 tile 的 online logsumexp；跨 $T=V/B_N$ 个 tile 的部分和用第二个小 kernel 合并。
好处：

- **没有 split-K 的 `atomicAdd` 归约税**（K 方向不拆）；
- **target logit 就地取**：`target[m]` 落在哪个 n-tile，就由那个 CTA 直接写 `target_logit[m]`，
  不需要单独再算一次 $x_m\cdot w_{t_m}$；
- 部分和只有 $M\times T\times 2$ 个 float（$B_N=128$ 时 $8192\times1010\times2\times4=$ **66 MB**），
  相比 2.12 GB 的 logits 可以忽略。

### 2.2 epilogue：tile 级 online logsumexp

`wgmma.m64n128k16` 的累加器 `acc[64]` 每个线程拿 16 组 `(q0,q1,q2,q3)`：
`q0,q1` 属于行 `row0`、列 `col,col+1`；`q2,q3` 属于行 `row0+8`。
一个 warp 内，同一行的 4 个 `tig=lane&3` 线程恰好覆盖这个 n8 块的 8 列——于是**行归约完全在 warp 内**，
用两次 `__shfl_xor_sync` 即可，不需要跨 warp 的 smem 归约。

```cpp
// fused_ce.cu:207 —— tile 级 max，4-lane 组内 all-reduce
float rmax0 = -INFINITY, rmax1 = -INFINITY;
for (int jn=0; jn<NSPLIT; ++jn)
  for (int j=0; j<16; ++j) {
    rmax0 = fmaxf(rmax0, fmaxf(acc[jn][j*4+0], acc[jn][j*4+1]));
    rmax1 = fmaxf(rmax1, fmaxf(acc[jn][j*4+2], acc[jn][j*4+3]));
  }
rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));   // 对 rmax1 同理
// 再一趟 Σ__expf(acc-rmax)，同样两次 shfl 归约，tig==0 写出 partial
```

`target` 提取（`fused_ce.cu:242`）：判断 `target[r]` 是否落在 `[block_col, block_col+BN)`，
是则让持有该列的线程直接写 `target_logit[r]=acc`。因为词表被 n-tile 恰好无重叠覆盖，
一个 target 只会被一个 CTA 命中、一把写。

### 2.3 GEMM 骨架

Q/K 的 TMA + mbarrier + warp specialization + `wgmma` SW128 骨架直接复用
[第 35 篇]({{< relref "cuda-kernel-opt-35-dsa-compressor" >}})（`proj_ws_kernel`）：
`A[M,K]`、`W[V,K]` 都是 K-major，权重天然免转置；TMA box 内维 128 B（=64 个 bf16），
4 级 stage，1 个 producer warp + `BM/64` 个 consumer warpgroup。

---

## 三、把 DRAM 流量从 60 GB 压到 17.7 GB

### 3.1 问题：朴素 schedule 的重复读

一个 CTA 读它自己的 `A` tile（$B_M\times K$）和 `W` tile（$B_N\times K$）。
不借助 L2 复用，总流量是 $MVK\cdot2\cdot(1/B_M+1/B_N)\approx$ **178 GB**——
即便只看张量核，$B_M=256,B_N=128$ 的工作量对应 15.2 TFLOP，理论上 53 ms 就耗在喂数上。
实测朴素 schedule（`grid=(V/BN, M/BM)`，`blockIdx.x` 最快）的 DRAM 流量是 **60.2 GB**
（硬件 L2 已经帮忙挡了一部分），已经变成 DRAM 瓶颈。

### 3.2 解法：superblock swizzle

把 $G_M\times G_N$ 个 tile 编成一组，**组内的 `A`/`W` 面板**（$(G_M B_M+G_N B_N)K\cdot2$ 字节）
小到能住进 50 MB 的 L2；相邻 CTA 共享同一块 `W`（m 变化最快）或同一块 `A`。
这样每一份 `A`/`W` 只从 DRAM 取一次，组间顺次扫描：

```cpp
// fused_ce.cu:123 —— 1D 网格 + superblock 解码
int id = blockIdx.x;
int sb = id / (GM*GN), in = id % (GM*GN);
int ms = sb / GNs, ns = sb % GNs;
int mi = in % GM, ni = in / GM;        // m 变化最快 → 相邻 CTA 共享 W tile
int mt = ms*GM + mi, nt = ns*GN + ni;
if (mt >= Mt || nt >= Nt) return;      // padding 出界
```

实测（ncu，`--metrics dram__bytes`）：

| schedule | DRAM 流量 | DRAM% | L2% | tensor% | ncu 时长 |
|---|---|---|---|---|---|
| 朴素 `256×128 s4` | **60.21 GB** | 70.3% | 78.2% | 68.9% | 25.56 ms |
| `superblock 4×8` | **17.72 GB** | 23.7% | 82.2% | **78.8%** | 22.04 ms |

流量降到 **1/3.4**，瓶颈从 DRAM（70%）搬到 **L2 带宽（82%）+ 张量核（79%）**。
剩余 L2 流量是每个 CTA 仍要发自己的全部 tile load（$178$ GB 的 L2→SM 读），
想再降只能上更大的 tile 或 cluster multicast——下面「负结果」会说为什么这两条都堵死了。

> **踩坑**：swizzle 的第一版直接复用 `grid=(N/BN, M/BM)` 做块内解码，
> 当 $N_t/B_N$ 不能整除 $G_N$ 时（$1010/8$）会**漏算 tile**：部分 `(mt,nt)` 永远不被任何
> `blockIdx` 生成，`partial` 留着上一次的脏值，loss 悄悄错到 11.764（正确 11.7698）。
> 改成「1D 网格 = `ceil(Mt/GM)·ceil(Nt/GN)·GM·GN`，越界 `return`」后才严格双射。
> 这种 bug 不报错、只在非整除网格上现形，必须用**对拍 loss** 而不是肉眼看 kernel 正常退出。

---

## 四、实测

环境：H100 SXM（132 SM，3.35 TB/s，bf16 TC 989 TFLOPS），`kernel_lab` 容器。
shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：`M=8192, H=7168, V=129280`。
`fused_ce.cu` 内 `bench_ms(warmup=5, iters=20)`。

### 4.1 基线

| 基线 | GEMM | CE | 端到端 | 说明 |
|---|---|---|---|---|
| cuBLAS+CE（同进程，本文件） | 20.33 ms / **746.8 TFLOPS** | 1.90 ms | **24.53 ms** | 物化 bf16 logits |
| PyTorch eager `x@W.t() + F.cross_entropy` | 20.52 ms / 667 TFLOPS | 2.71 ms | **23.04 ms** | `torch_eager.py`，峰值显存 12.6 GB |
| PyTorch `CE(logits.float())` | 20.33 ms | 8.71 ms | ~29.0 ms | 升 fp32 logits（4.24 GB）再算 |

### 4.2 融合版扫参

| 配置 | GEMM-only（写 logits） | **融合** | 峰值占比 | loss |
|---|---|---|---|---|
| `128×128 s4` | 45.91 ms | 44.51 ms | 34.5% | 11.769797 |
| `128×256 s3` | 43.34 ms | 41.56 ms | 36.9% | 11.769794 |
| `256×128 s4`（朴素） | 30.09 ms | 28.73 ms | 53.4% | 11.769794 |
| `sw 1×16` | 29.85 ms | 28.51 ms | 53.8% | 11.769796 |
| `sw 2×8` | 26.64 ms | 26.27 ms | 58.4% | 11.769796 |
| `sw 2×16` | 26.10 ms | 26.36 ms | 58.2% | 11.769794 |
| **`sw 4×8`** | 25.05 ms | **25.07 ms** | **61.2%** | 11.769797 |
| `sw 4×16` | 25.14 ms | 25.17 ms | 61.0% | 11.769796 |
| `sw 4×8 s3` | 26.88 ms | 26.34 ms | 58.3% | 11.769796 |
| `sw 4×8 s2` | 27.51 ms | 26.62 ms | 57.7% | 11.769795 |
| `128×256 s3 sw` | 27.27 ms | 25.65 ms | 59.8% | 11.769797 |
| `256×256 s2` | 206.2 ms | 200.1 ms | 7.7% | 11.769795 |

正确性：所有融合配置的 loss 与 cuBLAS+CE 基线（`11.769796`）在 6 位小数内一致。

### 4.3 结论

- **融合 ↔ 两 kernel 基线：0.98×**。融合 kernel（25.07）与「cuBLAS+CE」（24.53）几乎打平，
  靠的是 GEMM 只慢 19% 的前提下省掉 logits。
- **vs PyTorch eager：0.92×**（25.07 vs 23.04）。torch 的 `x@W.t()` 用 cuBLAS 物化 bf16 logits，
  GEMM 比我们快 5 ms，而 logits 往返只值 ~1.3 ms，所以整体 torch 还略快。
- **vs fp32-logits 基线：1.16×**（25.07 vs ~29.0）。
- **显存**：logits 从 2.12 GB（bf16）/ 4.24 GB（fp32）降到部分和 **66 MB**。

一句话：**在 compute-bound 的大词表 LM head 上，融合 CE 是「显存优化」而非「吞吐优化」**。
把它当吞吐银弹会失望；当训练显存瓶颈时它立竿见影。

---

## 五、ncu：墙在哪

融合 `sw 4×8`（`--set full`）：

| 指标 | 值 |
|---|---|
| DRAM Throughput | 23.7%（17.72 GB） |
| **L2 Cache Throughput** | **82.2%** |
| **Compute (SM) / Tensor** | **78.8%** |
| Achieved Occupancy | 26.4%（16.9 warps/SM） |
| Registers | 90 / thread |
| Block Limit | Registers **1**、Shared Mem **1**（197 KB smem） |
| 主导 stall | `long_scoreboard` 9.1/15.6 cycles（58.3%，等 smem 里的 wgmma 操作数） |

两个硬约束：

1. **96 KB+ smem 锁 1 CTA/SM**：$B_M=256,B_N=128$、4 stage 的 smem = `4×(256+128)×128 = 197 KB`，
   再加上 `acc[64]` 的 90 寄存器也把 2 CTA/SM 挡掉（要 ≤60 regs 才够 544 线程 ×2）。
2. **L2 带宽到顶**：DRAM 已经只有 24%，但每个 CTA 仍要发 $178$ GB 的 L2→SM 读，
   L2 82% 成为新瓶颈。这就是 19% 差距的来源。

---

## 六、负结果

- **`256×256` tile 崩掉**：想用更大 tile 把 L2 流量从 178 GB 砍到 119 GB，
  但 `acc[2][64]=128` 寄存器 + wgmma 双累加器让 ptxas 报
  `C7511 wgmma.mma_async instructions are serialized due to insufficient register resources`，
  性能崩到 **75.9 TFLOPS**（`256×128` 的 1/8）。Hopper 单 CTA 几何下，tile 面积就这么大。
- **`128×128` 想换 2 CTA/SM**：寄存器/ smem 确实能塞 2 CTA，但 $B$ 重读翻倍
  （$1/B_M+1/B_N$ 从 0.0117 涨到 0.0156），L2 流量涨 33%，反而只有 **34.5% 峰值**。
- **`s2/s3` 更省 smem**：stage 变浅，喂数延迟藏不住，全线慢 5~13%。
- **只省「一次 pass」不够**：`FUSE=false`（只把 logits 写回、不做 lse）与融合版几乎一样快，
  说明 epilogue 的 logsumexp 是**免费**的（被 GEMM 藏住）——真正的开销是 GEMM 本身。

---

## 七、小结

- 融合 cross-entropy = **split-N GEMM + tile 级 online logsumexp + 跨 tile 合并 + 就地取 target**，
  把 $[M,V]$ 的 logits 从显存里彻底省掉（2.12/4.24 GB → 66 MB 部分和）。
- **L2 友好的 superblock swizzle** 是让它在带宽上不崩的关键：DRAM 流量 60.2 GB → 17.7 GB（3.4×），
  瓶颈从 DRAM 搬到 L2。踩坑：非整除网格下 swizzle 会漏算 tile，必须用 loss 对拍验证。
- 在 compute-bound 的大词表形状上，融合相对 cuBLAS+CE **0.98×**、相对 eager **0.92×**、
  相对 fp32-logits 基线 **1.16×**；**收益在显存不在吞吐**。差距的根因是手写 wgmma GEMM 比 cuBLAS
  慢 19%（L2 82% + 1 CTA/SM），而不是融合本身。
- 下一个能真正提速的方向：把权重（V4 的 LM head 本就是 FP8）换成 **FP8 wgmma**，
  权重字节减半、L2 流量减 17%，同时张量核吞吐上限翻倍——这是下一篇要试的。

---

## 复现

```bash
cd code/kernel-opt
# 主实验（基线 + 扫参）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda -lcublas" \
  scripts/run.sh 54-fused-ce/fused_ce.cu 8192 all
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda -lcublas" \
  scripts/ncu.sh 54-fused-ce/fused_ce.cu --set full --kernel-name regex:fce -- 8192 prof
# PyTorch eager 基线
python3 54-fused-ce/torch_eager.py
```

代码：`code/kernel-opt/54-fused-ce/fused_ce.cu`（融合 kernel `fce_kernel` `:102`、
superblock swizzle `:123`、epilogue `:207`、`combine_kernel` `:275`）；实测
`final_sweep.out.txt`、`torch_eager.out.txt`；ncu `ncu_fused_sw4x8.out.txt`、
`ncu_fused_nosw.txt`、`ncu_plain_sw4x8.txt`。
