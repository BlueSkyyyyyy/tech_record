---
title: "CUDA 算子调优（十九）：DSA 稀疏注意力（二）— 稀疏 MLA 消费端（gather top-k + cp.async 流水）"
date: 2026-09-21
draft: false
weight: 19
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "DSA", "稀疏注意力", "DeepSeek", "MLA", "top-k", "gather", "cp.async", "Tensor Core", "系列"]
categories: ["算子开发"]
---

上一章（[DSA 稀疏注意力（一）]({{< relref "cuda-kernel-opt-18-dsa-indexer-topk" >}})）把 DeepSeek Sparse Attention 的前半段做完了：**lightning indexer** 打分（287–311 TFLOPS）与 **exact top-k**（radix-select，6.75 ms @32k）。但真正的稀疏 attention——「只对选中的 $k$ 个 key 做 MLA」——当时只是按稠密吞吐折算的**估算**。

这一章把它做成真的 kernel：**稀疏 MLA 消费端**。给定每个 query 的 top-$k$ 索引，按索引 gather KV、做完 online softmax 的 MLA，并把显存里的 gather 延迟用 `cp.async` 双缓冲藏起来。

先给结论（H100 80GB HBM3，bf16 dense 峰值 989 TFLOPS；真实 shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：`num_attention_heads=128`、`head_dim=512`、`qk_rope_head_dim=64`、`index_topk=1024`，MLA 吸收口径 $d_c{=}512,d_r{=}64,d_v{=}512$）：

| 量 | 结果 |
|---|---|
| 稀疏 MLA（1 个 token × 64 head，gather $k{=}1024$） | **132–137 TFLOPS**（13–14% 峰值） |
| 相对稠密融合 MLA `f4s`（16 篇）的有效率 | **~75%**（做 $1/k$ 的活，效率只掉 1/4） |
| 长上下文 gather 延迟（`long_scoreboard`） | **4.56 → 1.46（cp.async 双缓冲，3.1×）** |
| 稀疏 attention 相对稠密 attention 的加速 | $S_k{=}4k$ **3.0×** → $S_k{=}64k$ **48.6×** |
| DSA 端到端（indexer + top-k + 实测稀疏 attn） | 4k **2.5×** → 64k **17.3×** |
| 距 SOTA（FlashMLA sm90 sparse prefill 640 TFLOPS，H800） | **~4.8×** |

配套代码 [`code/kernel-opt/19-dsa-sparse-attn/sparse_mla.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/19-dsa-sparse-attn/sparse_mla.cu)，原始输出见同目录 `sparse_S32768.out.txt` / `sweep_sk.out.txt` / `sparse_S32768_K512.out.txt` / `dense_mla_f4s_1024.out.txt` / `ncu_sk8192.out.txt` / `ncu_sk65536.out.txt`。

---

## 1. 稀疏 attention 长什么样：从「稠密 tile」到「gather 列表」

DSA 的第二步是：query 位置 $t$ 拿到 indexer 算出的 top-$k$ 索引 $I_t\subset\{0,\dots,t\}$（$|I_t|=k$，V4-Pro 是 1024），只对这 $k$ 个 key 做 MLA。所有 $H$ 个 query head **共享同一份** $I_t$——这正是能省的根源。

难点在于：**每个 query 的 key 集合都不一样**，无法像 16 篇那样把 KV 切成连续 tile，在 block 内多个 query 之间复用。工业实现（FlashMLA 的 `sm90/prefill/sparse/phase1.cuh`）采取的策略是：

- **一个 CTA = 一个 query token × 一个 head block**（$B_H$ 个 head）。同一 token 的所有 head 共享 $I_t$，于是 gather 一次喂给整个 head block；
- top-$k$ 按 **block 粒度**（$B_{\text{topk}}$ 个 key 一块）组织，块内用 `is_kv_valid` 掩码掉无效位置；变长序列用 `topk_length`。
- 计算仍是标准的 `QKᵀ → online softmax(掩码) → PV`，只是 $N$ 维从「全部 $S_k$」变成「$k$ 个选中 key」。

本篇采用同一骨架，但做**逐 key 的精确 gather**（更贴近 DSA 的 exact top-k，不做 block 粗化）。CTA 的数据流：

```
  一个 query 位置 t，一个 head block（BH=64 个 head）
  ┌────────────────────────────────────────────────────────────┐
  │ Q[BH][576] ──┐                                             │
  │              ├─► QKᵀ(mma) ─► mask(-inf) ─► online softmax ─► PV(mma) ─► O[BH][512]
  │ gather:      │                                             │
  │   idx = I_t[0..k)                                          │
  │   c_kv[idx][512] + k_rope[idx][64] ──► smem tile (KT 个 key)│
  └────────────────────────────────────────────────────────────┘
```

和 16 篇单 kernel 融合版相比，只多了两件事：**gather 访存** 和 **无效列掩码**。公式与 16 篇完全一致：对每个 head $h$ 与每个选中 key $s\in I_t$，

$$
S_{h,s}=\sum_{d=1}^{d_c} q^{\text{nope}}_{h,d}\,c_{s,d}+\sum_{r=1}^{d_r} q^{\text{rope}}_{h,r}\,k^{\text{rope}}_{s,r},
\qquad
O_h=\frac{\sum_{s\in I_t} e^{S_{h,s}-m_h}\,c_{s,:}}{\sum_{s\in I_t} e^{S_{h,s}-m_h}},
$$

其中 $m_h=\max_s S_{h,s}$，$c_s$ 同时充当 K 与 V（吸收形式）。

### 1.1 shape 与「谁共享谁」

| 参数 | 值（V4-Pro） | 来源 |
|---|---|---|
| $H$（query heads，CTA 的 $M$ 维） | 128 | `num_attention_heads` |
| $d_c$（NoPE / kv_lora） | 512 | `head_dim` |
| $d_r$（decoupled RoPE） | 64 | `qk_rope_head_dim` |
| $d_v$（输出维） | 512 | 吸收后 = $d_c$ |
| $k$（top-k） | 1024 | `index_topk` |

关键点：**$M$ 维是 head，不是 query 位置**。一个 CTA 里的 64 个 head 共享同一份 gather 出来的 KV——如果反过来（CTA 处理多个 query 位置），每个位置的 $I_t$ 不同，KV 无法共享。所以 DSA prefill 的并行度来自 $S_q\times(H/B_H)$，$S_q=1024$、$H{=}128$、$B_H{=}64$ 时有 2048 个 CTA，足够填满 132 个 SM。

计算量：稀疏 attention 是 $O(S_q\cdot k)$，稠密（causal）是 $O(S_q\cdot S_k)$，**FLOPs 比就是 $S_k/k$**。$S_k{=}64\text{k}$、$k{=}1024$ 时理论省 64×。

---

## 2. kernel 实现：在 16 篇 `mla_shared_kernel` 上加 gather + mask

代码主体沿用 16 篇的 `mla_shared_kernel`（smem 共享 P、消除 QK 重复、$C\!\to\!A$ 零 shuffle），改动集中在 gather 与掩码两处。

### 2.1 gather：行内合并、行间随机

一个 KV tile 有 `KT` 个 key，每行是 $c_{kv}[idx]$ 的 512 个 bf16（1 KB）。加载时让「同一行的连续线程读同一行的连续 16 B」：

```cuda
// sparse_mla.cu:145
for (int i = tid; i < KT * DC / 8; i += T) {
  const int row = i / (DC / 8), c8 = (i % (DC / 8)) * 8;
  const int j = k0 + row;
  const int idx = (j < valid) ? topk_idx[(size_t)t * K + j] : -1;
  kvflag[row] = (idx >= 0 && idx < Sk) ? idx : -1;
  uint4 v = make_uint4(0, 0, 0, 0);
  if (idx >= 0 && idx < Sk) v = *reinterpret_cast<const uint4*>(&ckv[(size_t)idx * DC + c8]);
  *reinterpret_cast<uint4*>(&ks[row][c8]) = v;
}
```

- **行内**：$c_8$ 连续 → 一个 warp 的 32 个线程在**同一行**上读连续 16 B，硬件合并成少而宽的请求；
- **行间**：索引随机，但每行 1 KB 会整行用满，没有浪费的 cache line。

这就是 gather 类访存的通用姿势：**牺牲行间连续性，保住行内连续与整行利用**。同一行的线程还会重复读 `topk_idx`，但同址广播、走 L1 命中，代价很低。

### 2.2 掩码：causal 不足 $k$ 个时的无效列

query $t$ 的全局位置是 $p=S_k-S_q+t$，causal 可见集是 $[0,p]$，所以有效个数 `valid = min(k, p+1)`。尾部 tile 有不足 `KT` 个有效 key，需要把无效列置 $-\infty$。做法是在 QK 之后、softmax 之前按列判断：

```cuda
// sparse_mla.cu:424（流水版）
if (k0 + KT > valid) {
  for (int i = 0; i < NT; ++i) {
    const int c0 = k0 + nbase + i * 8 + (lane & 3) * 2;
    if (c0 >= valid)     { S[i][0] = -INFINITY; S[i][2] = -INFINITY; }
    if (c0 + 1 >= valid) { S[i][1] = -INFINITY; S[i][3] = -INFINITY; }
  }
}
```

注意外层加了 `if (k0 + KT > valid)`：**只有最后一个 tile 需要掩码**，前面 $\lfloor valid/KT\rfloor$ 个 tile 全有效，直接跳过。这一条把掩码的 smem load + 分支从 16/17 的 tile 里省掉，实测把拷贝前的 `s1` 从 113 拉回 133 TFLOPS。

### 2.3 $C\!\to\!A$ 零 shuffle 与共享 P（16 篇遗产）

softmax 后的 $P$ 不需要物化：`mma.m16n8k16` 的 fp32 累加器 $(c_0,c_1)$ 与 PV 的 A 片段 $(a_0,a_1)$ 坐标天然一致，只要 `pack2` 做了 f32→bf16 打包就能直接喂给 PV。DV 被切成 `DVGRP` 份分给不同 warp（$O[16,d_v]$ 的寄存器墙），同一 row 组的 warp 通过 smem 共享 $P$，把 QK 的重复算降到 1×。这些细节见 16 篇，本篇不重复。

---

## 3. 正确性：对 CPU 精确解

对随机数据，抽 3 个 `(head, token)`，用 CPU 双精度按 §1 的公式逐行算参考，与 kernel 输出比对。各配置、各上下文长度全部通过：

```
  [s1    ] max_abs_err=1.192e-04 (ref~0.155) OK
  [s4    ] max_abs_err=1.192e-04 (ref~0.155) OK
  [p2    ] max_abs_err=1.245e-04 (ref~0.155) OK
```

（原始输出 `sparse_S32768.out.txt`。误差量级 $10^{-4}$，来自 bf16 与 fp32 在线 softmax 的舍入，相对误差 < 0.1%。）

---

## 4. 配置扫参与「长上下文 gather 延迟」的发现

$S_q{=}1024$、$H{=}128$、$k{=}1024$，扫 $S_k$。配置命名：`s*` = 同步加载版（16 篇骨架），`p*` = cp.async 双缓冲流水版；`BH=16\cdot ROWG`、`DVGRP` 决定 $d_v$ 切分、`KT` 是每 tile 的 key 数。

单位 TFLOPS（原始输出 `sweep_sk.out.txt`）：

| 配置 (BH,DVGRP,KT) | warps | $S_k{=}4096$ | $8192$ | $16384$ | $32768$ | $65536$ |
|---|---|---|---|---|---|---|
| s1 (64,2,64) | 8 | 133.5 | 132.1 | 127.6 | 119.3 | 114.8 |
| s2 (64,2,32) | 8 | 117.5 | 111.3 | 105.2 | 102.2 |
| s3 (32,4,64) | 8 | 84.9 | 78.3 | 69.8 | 66.3 |
| **s4 (64,4,64)** | **16** | **137.0** | 132.7 | 128.2 | 123.1 |
| **p2 (64,2,32)** | 8 | 136.3 | 129.9 | **133.3** | **132.4** | **127.9** |

（`s2` 的 $S_k{=}4096$ 与 `s1` 的 $8192$ 见 `sparse_S32768.out.txt` / `ncu_sk8192.out.txt`。）

读表得到三条结论：

1. **`KT` 比 `DVGRP` 重要**。s1（KT=64）全面优于 s2（KT=32）：tile 越大，`__syncthreads` 与 online-softmax 的固定开销被摊薄得越少。`s3`（BH=32）最差，因为 $B_H{=}32$ 意味着 $H/B_H{=}4$ 个 head block 各自把同一份 $I_t$ 的 KV 又 gather 一遍，**4× 的重复访存**。
2. **s4 用 16 warp 换 occupancy**：`DVGRP=4` 把每 warp 的 $O$ 累加器从 128 压到 64 个寄存器（128 regs vs 254），warps/SM 从 8 升到 16（12.5%→24.6%），在短上下文最好；代价是 Q 被 4 个 DV 份各 `ldmatrix` 一遍，`short_scoreboard` 升高。
3. **长上下文靠 `cp.async` 双缓冲**。`p2` 在 $S_k{=}64\text{k}$ 反超 s1（127.9 vs 114.8）：因为 $S_k$ 越大，$c_{kv}$ 总量（$64\text{k}\times576\times2{=}75$ MB）越超出 50 MB L2，gather 要打到 DRAM，同步加载的延迟直接暴露。

### 4.1 用 ncu 看清「同步 vs 流水」

$S_k{=}65536$ 时三种配置的 warp stall 指纹（`ncu_sk65536.out.txt`）：

| 配置 | 耗时 | DRAM% | L1/TEX% | tensor% | `long_scoreboard` | `short_scoreboard` |
|---|---|---|---|---|---|---|
| s1（同步） | 2.54 ms | 14.8 | 43.2 | 17.7 | **4.56** | 2.59 |
| s4（16 warp） | 2.38 ms | 16.2 | 55.6 | 18.9 | 5.33 | 5.89 |
| **p2（cp.async 双缓冲）** | **2.28 ms** | 16.5 | 51.6 | 19.7 | **1.46** | 2.31 |

`long_scoreboard`（全局访存依赖）从 **4.56 降到 1.46（3.1×）**——这正是 `cp.async` 的作用：把下一个 KV tile 的 gather 与当前 tile 的计算重叠。s4 虽然 occupancy 翻倍，但因为 Q 的重复 `ldmatrix`，`short_scoreboard`（共享内存依赖）升到 5.89，两者抵消。

$S_k{=}8192$（$c_{kv}$ 8 MB 全在 L2）时三者的 `long_scoreboard` 分别是 3.01 / 3.95 / 1.34，`p2` 依然最低（`ncu_sk8192.out.txt`）。可见**哪怕工作集在 L2，异步拷贝也能把 L2 延迟藏起来**。

### 4.2 流水实现要点

```cuda
// sparse_mla.cu:368——issue 一个 tile 的 gather 到指定 stage
__pipeline_memcpy_async(&buf[row][c8], &ckv[(size_t)idx * DC + c8], 16);
...
__pipeline_commit();

// 主循环：先发下一 tile，再等当前 tile
if (has_next) issue(k0 + KT, stage ^ 1);
__pipeline_wait_prior(has_next ? 1 : 0);   // 留 1 个 in-flight
__syncthreads();
// ... 用 kbuf 做 QK/softmax/PV ...
```

两个坑：

- **`cp.async` 的 16 B 对齐**：smem 行距要用 $d_k{+}8$ 个 bf16（$1168$ B，被 16 整除），且每个 16 B 块地址合法。$\text{DK}{=}576$ 的行距 $1152$ B 也整除，所以加 8 个 bf16 的 padding 同时满足「消 `ldmatrix` bank conflict」与「`cp.async` 对齐」，一石二鸟。
- **双缓冲的 smem 预算**：`p1`（KT=64 双缓冲）需要 234 KB > 227 KB 上限，直接弃用；只有 KT=32 的 `p2` 放得下。这也解释了为什么流水版不得不配小 tile——**smem 是硬约束**。

---

## 5. 稀疏 attention vs 稠密 MLA：实测对比

用 16 篇的稠密融合 `f4s`（同样 $S_q{=}1024$、$H{=}128$，非 causal 看全部 $S_k$）作对照。注意 DSA 是 causal 的，但 $S_q{\ll}S_k$ 时 causal 与 full 的 FLOPs 几乎相同（query 都在序列末尾），所以用 full 稠密的实测时间作上界是合理的、甚至偏保守。

| $S_k$ | 稠密 `f4s`（ms） | 稠密 TFLOPS | 稀疏 `p2`（ms） | 稀疏 TFLOPS | 稀疏/稠密效率 | attention 加速 |
|---|---|---|---|---|---|---|
| 4096 | 6.357 | 183.8 | 2.14 | 136.3 | 74% | **3.0×** |
| 8192 | 13.382 | 174.6 | 2.25 | 129.9 | 74% | **5.9×** |
| 16384 | 28.226 | 165.6 | 2.19 | 133.3 | 81% | **12.9×** |
| 32768 | 55.682 | 167.8 | 2.21 | 132.4 | 79% | **25.2×** |
| 65536 | 110.867 | 168.6 | 2.28 | 127.9 | 76% | **48.6×** |

（`dense_mla_f4s_1024.out.txt`、`sweep_sk.out.txt`。稀疏 FLOPs $=2H(d_c{+}d_r{+}d_v)\sum_t \text{valid}_t{=}0.292$ TFLOP。）

**稀疏核的效率保持在稠密融合版的 ~75%**——考虑到它多做了随机 gather、无效列掩码、且无法在 query 间复用 KV tile，只掉 1/4 是相当划算的。而它做的 FLOPs 是稠密的 $1/(S_k/k)$，于是 $S_k{\ge}8192$ 后 wall-time 加速迅速拉开。

---

## 6. DSA 端到端：把 18 篇的估算换成实测

DSA 做一次 prefill 的总时间是三段之和：

1. **indexer**：$2S_q S_k H^I d$ FLOPs，实测效率 ~305 TFLOPS（18 篇）；
2. **exact top-k**：每行 $S_k$ 个分数上做 radix-select，带宽受限（18 篇：$S_k{=}32768$ 单行 6.75 ms / $S^2$ 规模）；
3. **稀疏 attention**：本篇实测。

indexer 与 top-k 的成本都 $\propto S_q S_k$，把 18 篇在 $S_q{=}S_k$ 方阵上的实测按 $S_q/S_k$ 折算到本节的矩形 shape（标注为「估」），得到：

| $S_k$ | 稠密 attention | 稀疏 attention（实测） | indexer（估） | top-k（估） | DSA 合计 | **端到端加速** |
|---|---|---|---|---|---|---|
| 4096 | 6.36 ms | 2.14 | 0.26 | 0.14 | 2.54 | **2.50×** |
| 8192 | 13.38 | 2.25 | 0.45 | 0.14 | 2.84 | **4.71×** |
| 16384 | 28.23 | 2.19 | 0.96 | 0.16 | 3.31 | **8.53×** |
| 32768 | 55.68 | 2.21 | 1.77 | 0.21 | 4.19 | **13.3×** |
| 65536 | 110.87 | 2.28 | 3.60 | 0.52 | 6.39 | **17.3×** |

几个观察：

- **加速随 $S_k$ 单调上升**：4k 时稀疏 attention 本身的固定开销（gather + 掩码）还盖不住收益，只有 2.5×；到 64k 达到 17.3×。
- **瓶颈随规模转移**：短上下文是**稀疏 attention 核本身**（占了总时间的 84%）；长上下文又变回 **indexer 的 $O(S_q S_k)$**（64k 时 3.6 ms，超过 attention 的 2.28 ms）。18 篇算出的理论上限 ~30×（$\frac{H(d_c+d_r+d_v)}{H^I d}\cdot\frac{\eta_{\text{idx}}}{\eta_{\text{dense}}}\approx 31$）依然成立，17.3× 已逼近它。
- 想要进一步突破，要么把 indexer 做得更快，要么引入**更粗粒度的稀疏**（block 级选择）来打破 indexer 的 $O(S_qS_k)$。

### 6.1 与 SOTA 的差距

FlashMLA 的 sm90 sparse prefill（`phase1.cuh`，$d_{qk}{=}576$、$B_{\text{topk}}{=}64$）报告约 **640 TFLOPS**（H800）。本篇的 132–137 TFLOPS 约为其 **1/4.8**。差距来源清晰：

- FlashMLA 用 `wgmma` + TMA + warp specialization + `cp.reduce.async` 做跨 wg 归约；我们用的是 `mma.m16n8k16 + ldmatrix + cp.async`（16 篇同款基础设施）；
- 它按 block 粒度 top-k、统一 tile，避免了逐 key 的索引计算与分支；
- 它跑在 SM90 的 Tensor Memory 加速指令上，我们没碰。

把 16 篇遗留的 **SW128 swizzle + wgmma**（路线图 16b/31/39）接进来，是这个核从 13% 峰值往 25%+ 冲的下一步。

---

## 7. 小结

- **稀疏 MLA 消费端 = 16 篇融合核 + gather + 掩码**：CTA 取「一个 query token × 一个 head block」，$H$ 个 head 共享同一份 top-$k$ 索引，gather 一次喂满整个 block。
- **gather 访存的姿势**：行内连续线程读同一行的连续 16 B（合并 + 整行利用），行间随机无妨；`topk_idx` 同址广播，走 L1。
- **掩码只做尾 tile**：`valid` 已知，前 $\lfloor valid/KT\rfloor$ 个 tile 全有效，跳过掩码，实测把效率从 113 拉回 133 TFLOPS。
- **`KT` 比 `DVGRP` 重要**，$B_H$ 缩小会让多个 head block 重复 gather 同一份索引（s3 掉到 66 TFLOPS）。
- **长上下文靠 `cp.async` 双缓冲**：`long_scoreboard` 4.56→1.46（3.1×），$S_k{=}64\text{k}$ 时反超同步版 11%。smem 是硬约束，双缓冲只能配 KT=32。
- **效率 ~75% 稠密、工作量 $1/(S_k/k)$**：$S_k{=}64\text{k}$ 时 sparse attention 比稠密快 **48.6×**；DSA 端到端（含 indexer+top-k）**17.3×**，逼近 18 篇给出的 ~30× 上限。
- **距 FlashMLA sparse prefill（640 TFLOPS）约 4.8×**，出路是 `wgmma` + TMA + swizzle。

**下一篇**：把 16b/31 的 **SW128 swizzle + wgmma** 接进 MLA/稀疏 MLA（路线图 16b/39），以及 20/21 的 **MoE router / grouped GEMM**——第五/六部分的模型场景算子还很长。
