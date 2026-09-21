---
title: "CUDA 算子调优（十八）：DSA 稀疏注意力（一）— lightning indexer 与 exact top-k"
date: 2026-09-21
draft: false
weight: 18
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "DSA", "稀疏注意力", "DeepSeek", "lightning indexer", "top-k", "radix select", "Tensor Core", "系列"]
categories: ["算子开发"]
---

前十七篇把 MLA 注意力做成了单 kernel 融合（`f4s` 170 TFLOPS），也把 MuonClip 的 NS 正交化拆成了 GEMM 链。这一篇进入 **DeepSeek Sparse Attention（DSA）**——DeepSeek-V3.2-Exp / V4 用来把长上下文 prefill 从 $O(S^2)$ 稠密 attention 里救出来的稀疏注意力。

DSA 把 attention 拆成两步：

1. **lightning indexer**：用少量 index head 给每个 query 对每个历史 key 打一个「索引分」，再取 **top-k**（V4-Pro：$H^I=64$、$d=128$、$k=1024$）；
2. **稀疏 attention**：只对被选中的 $k$ 个 key 做真正的 MLA。

这篇先啃前一步——**indexer 打分**与**精确 top-k**，因为它们是 DSA 特有的、也最容易成为新瓶颈的两个算子。先给结论（H100 80GB HBM3，bf16 dense 峰值 989 TFLOPS；真实 shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：`index_n_heads=64`、`index_head_dim=128`、`index_topk=1024`）：

| 算子 | 实现 | S=16384 耗时 | 算力/带宽 | 峰值占比 |
|---|---|---|---|---|
| indexer 打分 | 标量 FFMA（基线） | 4712 ms（按 S=4096 的 294.5 ms × 16 外推） | 0.93 TFLOPS | 0.09% |
| indexer 打分 | TC `mma` + head 循环 | 19.70 ms | 222 TFLOPS | 22.5% |
| **indexer 打分** | **TC + HG=2 + BN=128** | **15.32 ms** | **287–311 TFLOPS** | **≈31%** |
| exact top-k | 单直方图 radix-select（S=32768） | 26.7 ms | — | — |
| **exact top-k** | **per-warp 直方图 + 流式** | **6.75 ms** | 3.2 TB/s 等效 | 95% HBM |

把三者拼进 DSA 预算（indexer + top-k + 稀疏 attention），相对稠密 MLA 的端到端加速：$S{=}4k$ **2.9×**、$S{=}16k$ **9.7×**、$S{=}32k$ **14.8×**、$S{=}64k$ **18.6×**，理论上限约 **30×**（被 indexer 的 $O(S^2)$ 封顶）。

配套代码 [`code/kernel-opt/18-dsa-sparse/dsa.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/18-dsa-sparse/dsa.cu)，原始输出见同目录 `dsa_S*.out.txt` / `dsa_sweep.out.txt` / `indexer_ncu.out.txt` / `topk_ncu.out.txt` / `dense_mla_f4s.out.txt`。

---

## 1. DSA 的第一步：indexer 的数学与 shape

对 query 位置 $t$ 和它之前的所有 key 位置 $s$，lightning indexer 给出索引分

$$
I_{t,s} \;=\; \sum_{j=1}^{H^I} w_{t,j}\;\mathrm{ReLU}\!\left(\mathbf{q}^{I}_{t,j}\cdot \mathbf{k}^{I}_{s}\right),
$$

其中 $\mathbf{q}^I_{t,j}\in\mathbb{R}^{d}$ 是第 $j$ 个 index head 的 query 投影，$\mathbf{k}^I_s\in\mathbb{R}^{d}$ 是**所有 head 共享**的 index key，$w_{t,j}$ 是每个 (query, head) 的标量权重。拿到 $I_{t,\cdot}\in\mathbb{R}^{S}$ 后取 top-$k$ 个 key，交给后面的 MLA。

DeepSeek-V4-Pro 的真实参数（`config.json`）：$H^I=64$、$d=128$、$k=1024$；V4.1-Flash 是 $H^I=32$、$k=512$。生产里 indexer 用 FP8 以省算力/带宽，这一篇用 bf16 做等价 benchmark（结论对 FP8 同样成立，只是峰值换算不同）。

```
        query t ──► [ H^I=64 个 index head, 每个 d=128 ]
                          │
   key s ─────────────►  q_j · k_s   ──ReLU──► × w_{t,j} ──Σ_j──►  I_{t,s}
   (128 dim, 全 head 共享)                 │
                                           └─ 64 个点积共享同一个 k_s

   I[t, :] ∈ R^S ──top-k(1024)──► indices ──► 稀疏 MLA (下一篇)
```

**和普通 attention 的关键区别**：$k_s$ 不属于某个 head，而是被 $H^I$ 个 head 共用。因此 indexer 是「$H^I$ 个小 GEMM 共享同一个 B 操作数、结果按 head 加权求和」——这个结构决定了后面所有的优化。

### 1.1 它是算力还是访存受限？

单看计算量：$\text{FLOPs}=2S^2 H^I d = 2S^2\cdot 8192$。在 $S{=}16384$ 时是 **4.40 TFLOP**；同样 $S$ 的稠密 MLA（$H{=}128$、$d_{c}{+}d_{r}{+}d_{v}{=}1088$）是 **74.8 TFLOP**。也就是说 **indexer 只有稠密 attention 的 5.9% 算力**。这正是 DSA 能大幅省算力的来源：用很便宜的打分换掉大部分 key 的 attention。

访存上，输出 $I\in\mathbb{R}^{S\times S}$ 是唯一的 $O(S^2)$ 张量（fp32，$S{=}16384$ 时 1.07 GB）；输入 $\mathbf{q}^I$ 是 $S\cdot H^I d = 268$ MB。理想算术强度 $I=\text{FLOPs}/\text{Bytes}\approx 2S^2 H^I d / (4S^2)=H^I d/2=4096$ FLOP/byte——**远远算力受限**。下面 `ncu` 也证实：DRAM 只有 **2.9%**。

---

## 2. indexer 打分：从 0.9 到 311 TFLOPS

### 2.1 标量基线：0.93 TFLOPS

最直白的写法是「一个线程一个 $(t,s)$ 对」，内层循环 $H^I\cdot d=8192$ 次 FFMA：

```cuda
for (int j = 0; j < IHI; ++j) {
  float dot = 0.f;
  for (int d = 0; d < ID; ++d)
    dot += q[t][j][d] * k[s][d];
  acc += w[t][j] * fmaxf(dot, 0.f);   // ReLU
}
```

实测 **0.93 TFLOPS**（$S{=}4096$：294.5 ms）。$\mathrm{ReLU}$ 把求和拆开，无法把 64 个 head 折叠成一个更大的 GEMM，于是每个 $(t,s)$ 都要做 8192 次 FMA。FP32 FFMA 的理论脊线只有 $\sim 66.9/2\approx 33$ FLOP/byte 对应的量级，标量实现天花板很低。

### 2.2 Tensor Core + head 循环

把每个 head 的点积写成 $m16n8k16$ 的 `mma`：A = query tile $[BM,d]$，B = key tile $[BN,d]$（用非转置 `ldmatrix` 取 `.col` 片段，和 16 篇 MLA 的 QKᵀ 一模一样）。每个 head 的 $d=128$ 走 8 个 `k16`，累加到 fp32 的 `sacc`，epilogue 里 `ReLU` + 乘 $w$，再累加到跨 head 的 `acc`：

```
block 负责 [BM=64 query] × [BN key]，K tile 常驻 smem:
  for j in 0..HI:
     载入 Q_j tile (BM×128) -> smem
     sacc = 0
     for kx in 0..8:  A = ldmatrix(Q_j)        // 每 head 一次
        for nb in 0..NT: B = ldmatrix(K)       // ★ 每 head 重读一遍整个 K tile
           mma(sacc, A, B)
     acc += w_j * ReLU(sacc)                   // 加权求和
```

一个关键工程细节：**block 的网格顺序**。把 $S\times S$ 的输出 tile 按 `grid.x = KV块`、`grid.y = query块` 排，让同一个 query block 的所有 KV 块连续调度，Q tile（1 MB）才能留在 50 MB 的 L2 里。一开始我写成 `grid.x = query`，$S{=}32768$ 时 Q 被反复从 DRAM 拉，indexer 只有 **69.7 TFLOPS**；交换网格后立刻回到 **214 TFLOPS**（3.1×）。这是一条很典型的「$O(S^2)$ kernel 的 L2 工作集」陷阱。

### 2.3 杠杆一：head 合并（HG）——消掉 K 的重复读

第一次 `ncu`（$S{=}16384$）暴露真正的瓶颈：

```
DRAM Throughput       2.0 %      ← 完全不是访存
L1/TEX Cache Throughput 85.2 %   ← 卡在 L1 / ldmatrix
Compute (SM)          38.3 %
```

原因就在上面伪代码的 ★ 处：$K$ tile 对**所有 head 都是同一个 B 操作数**，但默认每个 head 都要用 `ldmatrix` 把整个 K tile 重读一遍。$H^I=64$ 个 head 就是 64 倍重复。把 **HG 个 head 一起算**，每个 `(kx, nb)` 只加载一次 B，喂给 HG 个 A：

```
for j0 in 0..HI step HG:                 # HG=2
  载入 HG 个 Q tile 到 smem
  sacc[HG] = 0
  for kx in 0..8:
     for h in 0..HG: A[h] = ldmatrix(Q_{j0+h})
     for nb in 0..NT:
        B = ldmatrix(K)                  # ★ 一次
        for h in 0..HG: mma(sacc[h], A[h], B)
  acc += Σ_h w_{j0+h} * ReLU(sacc[h])
```

$S{=}16384$、BN=64 下：HG=1 → **222 TFLOPS**，HG=2 → **252 TFLOPS**（+13%），HG=4 → 236（M 寄存器吃紧、occupancy 掉）。

### 2.4 杠杆二：加宽 N tile（BN）——摊薄 A 的重复读

A（Q）也有一份重复：每个 block 的每个 warp 都要重新读自己那 $16\times d$ 的 Q 片段，而 block 数 $\propto 1/(BM\cdot BN)$。**把 BN 从 64 加到 128**，block 数减半，A 的 L1 流量减半；同时 B 的复用也随 BN 放大。$S{=}16384$：

| 配置 | 耗时 | TFLOPS | %峰值 | 寄存器 |
|---|---|---|---|---|
| HG=1, BN=64 | 19.80 ms | 222 | 22.5% | 96 |
| HG=2, BN=64 | 17.42 ms | 252 | 25.5% | 161 |
| HG=1, BN=128 | 17.54 ms | 251 | 25.4% | 96 |
| **HG=2, BN=128** | **15.32 ms** | **287** | **29.0%** | 255 |
| HG=4, BN=128 | 63.34 ms | 69 | 7.0% | 255+spill |

注意最后一行的**反例**：HG=4 配上 BN=128 直接把寄存器顶到 255 并溢出到 local（`spill stores`），算力从 287 崩到 **69 TFLOPS**——和 10 篇、11 篇反复出现的「寄存器墙」是同一个坑。**memorize：先 `-Xptxas -v` 看寄存器，再看 TFLOPS。** 最优落在 HG=2/BN=128（29–31% 峰值），此时 `ncu` 已经比较均衡：

```
L1/TEX  62.5%   Compute 46.9%   DRAM 2.9%   occupancy 12.5%
```

L1 仍是主要压力但已不再一家独大，再往上要动寄存器/流水线，留作后话。

### 2.5 扫 S：indexer 的算力随规模稳定

| S | HG=1/BN=64 | HG=2/BN=64 | HG=2/BN=128 |
|---|---|---|---|
| 1024 | 145.7 | 183.8 | 190.2 |
| 4096 | 199.0 | 227.8 | 264.8 |
| 8192 | 216.3 | 246.8 | 303.6 |
| 16384 | 222.2 | 252.4 | 287.1 |
| 32768 | 215.2 | 245.5 | **311.0** |
| 65536 | 218.2 | 247.4 | 305.8 |

（单位 TFLOPS；原始输出 `dsa_S*.out.txt`。）HG=2/BN=128 在 $S\ge 8192$ 稳定在 **288–311 TFLOPS（29–31% 峰值）**，相对标量基线 **~330×**。小 S（1024）受 launch/尾效应限制，只有 190。

---

## 3. exact top-k：radix-select 与原子竞争的较量

拿到 $I$ 之后要取每行 top-1024。`S=65536` 时每行 65536 个 fp32，而 $S\times S$ 只要不物化就好——但生产实现会选择「边算边选」。这里先做**精确**的 token 级 top-k：对每行取第 $k$ 大的阈值，再收集所有 $>$ 阈值的元素、用 $=$ 阈值的补齐。

### 3.1 为什么不用「全排序 + 取前 k」

全排序是 $O(S\log S)$、还要处理 $S$ 个并行的段；而我们只要第 $k$ 大的值。**radix-select** 只用 $O(S)$ 每趟、几趟就能定位阈值：

1. 把 float 映射成**单调的 uint32**（保序变换）：$o(u)=u\oplus\big((\mathrm{sgn}(u)\text{?}\mathtt{0xffffffff}{:}\mathtt{0x80000000})\big)$；
2. 从最高 8 bit 开始：统计当前前缀下每个 digit（256 桶）的计数直方图；
3. 从高 digit 往低扫，找到累计计数跨过 $k$ 的 digit $d$，把它固定进前缀、扣掉计数；
4. 四趟（32/8）后，前缀就是**第 $k$ 大的有序值**（阈值）；
5. 收集所有 $>$ 阈值 的元素，再用 $=$ 阈值的补齐到 $k$ 个。

```
I[t, :]  ──order_float──►  uint32 序列
   pass0: [b31..b24]  256桶直方图 → digit d0
   pass1: [b23..b16]  (前缀=d0)  → digit d1
   pass2: [b15..b8 ]  ...
   pass3: [b7 ..b0 ]  ...  ⇒ prefix = 第 k 大的有序值
   collect: >prefix 全部 + =prefix 补齐 k 个  ⇒ indices
```

保序变换的逆不是「再 XOR 一次」（mask 依赖符号位，不自逆），而是：高位置位时取低 31 位、否则取反。这个坑值得记：我第一版逆变换写错，`out_val` 全错、对拍直接 `FAIL`。

### 3.2 瓶颈：shared atomicAdd 的竞争

第一版只有**一个** block 内共享直方图，所有线程对一个 256 桶数组 `atomicAdd`。$S{=}32768$ 时 top-k 要 **26.7 ms**（流式）/ **35.2 ms**（整行塞 smem）——非常慢。

`ncu` 立刻指出问题：`Issue Slots Busy 83%`、`Compute (SM) 83%`、IPC 3.35（接近 4 上限）、occupancy 99%。**它根本不是带宽受限，是指令/原子受限**：每元素每趟都要做保序变换 + 移位比较 + 一次共享原子加，5 趟下来 ~40 条指令/元素，共享原子在 Hopper 上又是串行/低吞吐的。

### 3.3 杠杆：per-warp 私有直方图

把直方图拆成 `NH = warps` 份（每 warp 私有），竞争立刻降一个数量级；4 趟后再规约。$S{=}32768$：

| 实现 | 耗时 | 说明 |
|---|---|---|
| 单直方图（流式，5 趟） | 26.7 ms | 原子竞争 |
| **per-warp 直方图（流式）** | **6.75 ms** | **3.96×** |
| 单直方图（整行 smem） | 35.2 ms | 1 block/SM，occupancy 死 |
| **per-warp（整行 smem）** | **8.23 ms** | 仍 1 block/SM |

**4.0× 提升全部来自消除原子竞争。** 这里还有一条反直觉的结论：**「把整行塞进 smem」在长序列上反而更慢**（8.23 vs 6.75 ms）。$S{=}32768$ 时每行 128 KB，动态 smem 把 occupancy 锁成 **1 block/SM**，4 趟片内扫描的延迟无法被其他 block 掩盖；而流式 6.75 ms 虽然要多读 4 遍 global（共 21.5 GB），却因高并发达到 **3.2 TB/s、95% HBM**，反而更快。**smem 缓存不是免费的：先算 occupancy。**

短序列则相反——$S{=}4096$ 时 smem 版 0.605 ms vs 流式 0.538 ms 已接近；$S{=}1024$ 时两者都被 launch/归约的固定开销支配（0.17/0.15 ms），占比不到 10%。

| S | topk smem | topk 流式 | 单遍等效带宽（流式） |
|---|---|---|---|
| 1024 | 0.171 ms | 0.148 ms | 28.4 GB/s（固定开销） |
| 4096 | 0.605 | 0.538 | 124.8 GB/s |
| 8192 | 1.231 | 1.115 | 240.7 GB/s |
| 16384 | 2.784 | 2.569 | 417.9 GB/s |
| 32768 | 8.294 | 6.754 | 635.9 GB/s |
| 65536 | —（行 289 KB > smem） | 32.968 | 521.1 GB/s |

正确性：对随机分数与 CPU `nth_element` 精确解逐行比对，选出值集合完全一致（`selected set vs CPU exact top-1024: OK`，各行 `dsa_S*.out.txt`）。

---

## 4. DSA 预算：相对稠密 MLA 能快多少

把上面测出的三个算子与稠密 MLA（16 篇 `f4s`，$H{=}128$、$d_c{+}d_r{+}d_v{=}1088$）放在同一张表。稀疏 attention 的 FLOPs 是 $2S\cdot k\cdot H(d_c{+}d_r{+}d_v)$，这里按稠密 MLA **实测到的每 FLOP 吞吐**折算（下一篇会用真正的稀疏 kernel 替换这个估算，并计入 gather 开销）：

| S | 稠密 MLA 实测 | indexer (HG2/BN128) | top-k | 稀疏 MLA（估） | DSA 合计 | 加速 |
|---|---|---|---|---|---|---|
| 4096 | 24.35 ms | 1.04 ms | 0.54 ms | 6.83 ms | 8.41 ms | **2.9×** |
| 16384 | 437.5 ms | 15.32 ms | 2.57 ms | 27.33 ms | 45.22 ms | **9.7×** |
| 32768 | 1737.6 ms | 56.56 ms | 6.75 ms | 54.27 ms | 117.6 ms | **14.8×** |
| 65536 | 6914.8 ms | 230.1 ms | 32.97 ms | 108.0 ms | 371.1 ms | **18.6×** |

（稠密 MLA 原始输出 `dense_mla_f4s.out.txt`。）

几个观察：

- **DSA 的加速随 $S$ 单调上升**：短序列（4096 及以下）稀疏带来的收益还盖不住 indexer 的固定开销，稠密甚至更划算；要到 16k 以上才明显。
- **indexer 逐渐接棒成为瓶颈**：$S{=}65536$ 时 indexer（230 ms）已是 top-k 的 7 倍、稀疏 attention 估算值的 2 倍。因为 indexer 也是 $O(S^2)$，而稀疏 attention 是 $O(Sk)$。
- **理论天花板**：当 $S$ 极大、稀疏 attention 与 top-k 相对可忽略时，
  $$
  \text{speedup}_{\max}=\frac{H(d_c+d_r+d_v)}{H^I d}\cdot\frac{\eta_{\text{indexer}}}{\eta_{\text{dense}}}
  =\frac{128\times1088}{64\times128}\times\frac{305}{173}\approx \mathbf{30\times}.
  $$
  18.6×（64k）已经逼近这个上限，说明后续要么把 indexer 做更快（提 $\eta_{\text{indexer}}$），要么引入更粗粒度的稀疏（如 block 级选择）来打破 $O(S^2)$。

---

## 5. 小结

- **DSA 的 indexer 是纯算力受限的 GEMM**（DRAM 2.9%），但它有特殊结构：$H^I$ 个小 GEMM 共享同一个 key 操作数。标量 0.93 → TC 222（HG=1）→ **287–311 TFLOPS**（HG=2 + BN=128），约 330×。
- **两条杠杆**：head 合并（HG）消掉 K 的 `ldmatrix` 重复读（L1 85%→62%）；加宽 BN 摊薄 Q 的重复读。两者都要盯着寄存器——HG=4/BN=128 溢出后掉到 69 TFLOPS。
- **网格顺序决定 L2 工作集**：`grid.x=KV` 让 Q tile 常驻 L2，$S{=}32768$ 上 3.1× 差异。
- **exact top-k 用 radix-select**：4 趟定位阈值 + 1 趟收集，不需要全排序；**独占瓶颈是 shared atomicAdd**，per-warp 私有直方图带来 **4.0×**（26.7→6.75 ms）。
- **整行塞 smem 不是万灵药**：长序列下 1 block/SM 的 occupancy 反而输给高并发的流式（6.75 vs 8.23 ms）。
- **DSA 相对稠密 MLA 的加速随 $S$ 上升**：4k→2.9×、16k→9.7×、32k→14.8×、64k→18.6×，上限约 30×，由 indexer 的 $O(S^2)$ 封顶。

**下一篇**：DSA 稀疏注意力（二）——真正写出**稀疏 MLA 消费端**（按 top-k 索引 gather KV、online softmax、TC 累加），把上面的估算换成实测，并对手写的 indexer 与 `~/github/flashinfer` 的 MSA / DSA 实现做同口径对标。
