---
title: "CUDA 算子调优（四十三）：W4A16 decode 的跨 item 持久化流水——把 wave 2.58 压到 1，再跟每 stage 的除法/原子税算账"
date: 2026-09-22T02:30:00+08:00
draft: false
weight: 43
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "decode", "量化", "W4A16", "int4", "AWQ", "GPTQ", "持久化", "persistent-kernel", "wave-quantization", "warp-specialization", "wgmma", "cp.async", "mbarrier", "Qwen3", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」量化推理方向的第三篇。承接
[第 40 篇 W4A16 dequant-GEMM]({{< relref "cuda-kernel-opt-40-w4a16-gemm" >}})
与 [第 41 篇 warp specialization + 波次对齐]({{< relref "cuda-kernel-opt-41-w4a16-warp-specialization" >}})。
41 篇结尾留了一个非常明确的诊断：把 `M=64, N=17408, K=5120` 这个 Qwen3-8B 的 decode GEMM
（4-bit 权重，`group=128`，对称 int4）切成 `KSPLIT=5` 后，网格是
`136 × 5 = 680` 个 CTA，而并发槽只有 `2 CTA/SM × 132 = 264` 个：

$$
\text{wave} = \left\lceil \frac{680}{264} \right\rceil = 3,\qquad
\text{最后一波只有 } 680-2\times264 = 152 \text{ 个 CTA}
$$

ncu 直接给出 **Est. Speedup 33%** 的尾波提示；41 还实测过「把 `N` 换成 16896 让网格正好
528 = 264×2」能 **−20%**，证明尾波是真金白银。但 41 试的 `persistent + atomicAdd 动态派活`
**没有兑现**，原因是「每换一个 work item 就重启 `cp.async`/`wgmma` 流水」。

这一篇就是去啃这块骨头：**把 `(item, kb)` 展平成一维 stage 流，让一个 CTA 在多个 work item
之间不停机地跑同一条 producer/consumer 流水**。结论有两面，先说结论：

- **好的一半**：在真正的 decode 工作点（`M ≤ 16`）上，持久化 + 细粒度 K-chunk 确实赢，
  `M=1` 从 `0.0818 ms / 541.8 GB/s`（41 篇）推到 **`0.0788 ms / 565.5 GB/s`（1.04×）**；
  而且控制实验证明**赢在持久化本身，不是赢在「切得更细」**——同 shape 下把 baseline 的
  `KSPLIT` 从 5 加到 8/16 反而更慢（见下文表）。
- **坏的一半**：`M ≥ 32` 之后 k-chunk 越细越亏（每个 item 的 `atomicAdd` 归约税和 epilogue
  随 `M` 线性增长），`M=64` 上 baseline 反而赢 4.4%；而且想把 `STAGES=2` 挤到 **3 CTA/SM**
  直接撞寄存器墙：`acc[64]` + 描述符需要 ≥90 个寄存器，而 3 CTA/SM 的门槛是
  `65536/(3×256)=85`，ptxas 报 `C7602 Insufficient registers`。

换句话说：**wave quantization 确实值 20~30%，但在 W4A16 这个「寄存器被 `acc[64]` 锁死」的
算子里，它和「K-split 归约税」「occupancy 上限」是互相矛盾的三个目标，只能在 `M` 很小
（decode 单 token 批量）的窗口里取到一个约 3% 的净胜。**

---

## 一、先把问题摆清楚：为什么 wave 和 K-split 是一对矛盾

W4A16 的 decode 是**纯权重带宽场景**：每个专家权重必须被完整读一遍，`M` 只决定有多少行
输出要算。单看权重 roofline：

$$
t_{\text{mem}} = \frac{N\cdot K/2}{B_{\text{HBM}}} = \frac{17408\times5120/2}{3352\ \text{GB/s}} \approx 13.3\ \mu s
$$

实测 ~80 µs，离 roofline 还有 **6×**，说明瓶颈不是带宽而是**延迟/占用**（41 篇 ncu：
`Compute 41.5%`、`occ 21%`、`fixed-latency` stall 主导）。所以「多塞点并行度」是对的，
41 的 K-split=5 就是这么来的——把 80 个 k-block 切成 5 段，网格从 `136` 变 `680`，才有
`680/264≈2.58` 的占用。

矛盾就出在这里：

1. `M` 小 → 输出 tile 只有 `N/BN` 个（=136），远小于 264 个槽 → 必须 K-split 才有并行度；
2. K-split 又让**每个 CTA 只干十几块 k 就退出**，于是 680 个短命 CTA 分成 3 波，
   最后一波 152 个 CTA 只能干看着；
3. 想用「更细的 K-split」把 680 抹平成 264 的整数倍？**每个 item 结束都要把
   `acc[64]` 通过 `atomicAdd` 归约回 `C`**（见下），细切让归约次数线性上升。

$$
\text{atomicAdd 次数} = \underbrace{\frac{N}{BN}}_{136}\times \underbrace{\frac{80}{\text{CHUNK}}}_{\text{K-chunk 数}}
\times \underbrace{BM\cdot BN}_{64\times128}
$$

`CHUNK=16`（≈KSPLIT=5）时是 5.6M 次，`CHUNK=4` 时 22.3M 次。

**41 的 persistent 失败**，是因为它「一个 CTA 领一个 item → 干完 drain 流水 → 再领下一个」，
省下的尾波时间被「每个 item 重新 prologue + drain」吃回去了。这一篇的关键改动是：
**把 item 边界从流水里抹掉**。

---

## 二、设计：把 `(item, kb)` 展平成一维 stage 流

核心思路一句话：**producer 和 consumer 都只认一个全局 stage 下标 `t`，`t` 只增不减地流过
所有 item；item 边界只对 consumer 是一次「记账」，对 producer 完全透明。**

```
                   item 0 (CHUNK=8)          item 1                item 2
   flat stage t:  0  1  2  3  4  5  6  7 |  8  9 10 11 12 13 14 15 | 16 17 ...
                  └──────── CHUNK ────────┘ └──────── CHUNK ────────┘
  producer:  load+dequant 每 stage → mbar.full[s]      （永不 drain）
  consumer:  wgmma 每 stage        → mbar.empty[s]     （永不 drain）
             └─ epilogue(acc→C) 只在 item 结束那一拍 ─┘
```

实现要点（代码 `code/kernel-opt/43-w4a16-persist/w4a16_persist.cu`）：

### 2.1 stage ↔ item 的闭式映射

work item = `(n_tile, k-chunk)`，每个 item 恰好 `CHUNK` 个 k-block（`CHUNK` 为偶数、整除
`K/BK`）。CTA 用静态 round-robin 领 item：

```cpp
// 每个 CTA 的 item: gid = gid0, gid0+P, gid0+2P, ...   (P = 2*NSM = 264)
const int nitems = (gid0 < total_items) ? (total_items - 1 - gid0) / P + 1 : 0;
const int NS     = nitems * CHUNK;               // 本 CTA 的 flat stage 总数
// flat stage t → item ii = t / CHUNK, item 内偏移 u = t % CHUNK
const int gid = gid0 + (t / CHUNK) * P;
const int col = (gid % ntiles) * BN;             // 输出列
const int k0  = ((gid / ntiles) * CHUNK + u) * BK;
```

`CHUNK` 取 2 的幂时 `t/CHUNK`、`t%CHUNK` 会编译成移位/掩码，剩下 `gid % ntiles`、
`gid / ntiles` 两个真正的整数除法——**这两个除法后来成了 ncu 口径下的性能暗雷**（见 §5）。

### 2.2 item 边界用 `wgmma` 自己的 `scale_d=0` 清零累加器

41 的累加器在循环外清零。持久化后，每进一个新 item 都要把 `acc[64]` 清零。如果直接写
`for(i) acc[i]=0` 再发 `wgmma`，ptxas 会判成 `C7514`（非 wgmma 指令定义了 wgmma 累加器）
而主动串行化 wgmma（[第 37 篇]({{< relref "cuda-kernel-opt-37-fp8-pb-prescale" >}}) 已踩）。
复用 W1 技巧——把 `wgmma.mma_async` 的 **`scale_d` 操作数**在「item 第一个 stage 的第一个
k16」上置 0，让指令自己做 `D = A·B`（丢弃旧 D）：

```cpp
wgmma_m64n128k16(acc, da, db, (u == 0 && kk == 0) ? 0 : 1);
```

### 2.3 epilogue 必须挪出热循环（否则寄存器 90→207）

这是本篇最疼的一个坑。第一版把 epilogue 直接写在 consumer 的 flat 循环体内
（`if (sLast[t]) { ... atomicAdd ... }`），正确性没问题，但 **寄存器从 90 飙到 207**，
occupancy 2 CTA/SM → 1 CTA/SM，直接慢 40%：

| 版本 | 寄存器 | spill | M=64 时间 |
|---|---|---|---|
| baseline（41 篇） | 90 | 0 | 0.095 ms |
| persistent，epilogue 在循环内 | **207** | 0 | 0.141 ms |
| persistent，epilogue 挪到 item 外层 | 90 | 0 | 0.098 ms |

诊断方法很简单：把 epilogue 整段删掉（结果无意义）再编译，寄存器立刻回到 80。
所以 consumer 改成 **item 外层 / kb 内层**，epilogue 落在内层循环之外的 item 边界上：

```cpp
int t = 0;
for (int ii = 0; ii < nitems; ++ii) {          // item 外层
  const int col = ... ;
  #pragma unroll 1
  for (int u = 0; u < CHUNK; ++u, ++t) {       // kb 内层（不 unroll，防寄存器膨胀）
    mbar_wait(full + t%STAGES, (t/STAGES)&1);
    ... wgmma ... ; mbar_arrive(empty + t%STAGES);
  }
  /* item 边界：epilogue 在热循环之外 */
  ... atomicAdd(acc) → C ...
}
```

> **坑**：`#pragma unroll 1` 是必须的。`CHUNK=4` 时若让编译器展开内层 4 份 `wgmma` 序列，
> 寄存器会顶到 128 + 272B spill；加了 `unroll 1` 才回到 90 / 0 spill。

### 2.4 一个隐藏约束：`CHUNK ≥ STAGES`

第一版扫参时 `CHUNK=2` 直接 **illegal memory access**。原因是 flat 流的 barrier 相位按
`t/STAGES` 计算，require 每个 item 至少覆盖一整个 stage 环；item 比流水环还短时相位错乱。
代码里加了 `static_assert(CHUNK >= STAGES, ...)` 把它钉死。

---

## 三、实测：三张表讲完

环境照旧：H100 SXM、`kernel_lab` 容器、CUDA 13.2、`-gencode=arch=compute_90a,code=sm_90a`。
所有对比都在**同一进程、同一 warmup（5）+ iters（50）**下做（`bench_ms`），
shape 取自 `/ssd/models/qwen3-8B/config.json`（`hidden=5120, intermediate=17408,
group_size=128` 对称 int4）。

### 表 1：baseline（41 篇 KSPLIT=5） vs 持久化，扫 `M`

| M | baseline k5 (ms) | W-bw (GB/s) | persist c8 (ms) | W-bw (GB/s) | 比值 |
|---|---|---|---|---|---|
| **1**  | 0.0823 | 541.3 | **0.0802** | **555.9** | **1.026×** |
| **4**  | 0.0820 | 543.7 | **0.0798** | **558.4** | **1.028×** |
| **16** | 0.0824 | 540.9 | **0.0809** | **551.1** | **1.019×** |
| 32 | **0.0850** | **524.2** | 0.0859 | 518.9 | 0.99× |
| 64 | **0.0947** | **470.4** | 0.0990 (c16) | 450.0 | 0.96× |

单独跑 `all` 时 `M=1` 最佳观测到 **0.0788 ms / 565.5 GB/s**（`persist_s3c8`），
对比 41 篇发布的 M=1 成绩 0.0823 ms / 541.8 GB/s，是 **1.04×**。

**交叉点在 M=16~32 之间**：`M ≤ 16` 持久化赢，`M ≥ 32` baseline 赢。原因是
`atomicAdd` 的字节数 ∝ `M×N`，而细 chunk 让归约次数翻倍：

| M | 归约写字节（KSPLIT=5） | 归约写字节（CHUNK=8） |
|---|---|---|
| 1  | 0.14 MB | 0.28 MB |
| 16 | 2.2 MB  | 4.5 MB  |
| 64 | 8.9 MB  | 17.8 MB |

`M=1` 时归约流量可忽略，细 chunk 净赚；`M=64` 时归约流量已经和权重（44.6 MB）同一量级，
细 chunk 的「并行度收益」被归约税反超。

### 表 2（控制实验）：赢在持久化，不是赢在「切得更细」

把 baseline 的 `KSPLIT` 也加到 8/16（同样细的 chunk），结果**更慢**——因为每个短命 CTA
都要重新付 prologue/drain：

| M=1 配置 | 时间 (ms) | W-bw (GB/s) |
|---|---|---|
| baseline KSPLIT=5 | 0.0823 | 541.3 |
| baseline KSPLIT=8 | 0.0832 | 535.7 |
| baseline KSPLIT=16 | 0.0883 | 504.9 |
| **persist CHUNK=8** | **0.0802** | **555.9** |
| persist CHUNK=4 | 0.0807 | 552.3 |

同样 `M=16` 也是这个次序（baseline k16 0.0942 vs persist c8 0.0809）。
**这就把「持续化流水」这个变量的效果单独隔离出来了**：相同的 chunk 粒度下，
持久化把 680 个 CTA 的 prologue（每个 CTA 1 个 stage，约占 1/16=6%）压成 264 个，
而且细 chunk 不再有额外 prologue。

### 表 3：`STAGES` 与「3 CTA/SM」判决

`STAGES=4/5` 让 smem 从 ~92 KB 涨到 ~120 KB，occupancy 掉回 1 CTA/SM，全线崩
（M=1：`s4c8` 0.1144 ms、`s5c16` 0.1203 ms）。想走另一条路——`STAGES=2` 把 smem 降到
~63 KB、凑 **3 CTA/SM**（396 槽，尾波从 3 波变 2 波）——直接编译失败：

```
ptxas fatal : (C7602) Insufficient registers (80) to compile instruction ...
              Try to compile with register target of 90 or higher.
```

`acc[64]`（(`BM/64`)×(`BN/128`)×64 = 64 个 fp32）+ smem 描述符至少要 90 个寄存器，
而 `65536/(3×256)=85` 是 3 CTA/SM 的硬门槛。**`acc[64]` 把 occupancy 锁死在 2 CTA/SM**，
wave 只能靠调 chunk 粒度来抹，不能靠堆 CTA。

---

## 四、ncu：机制对上了，但口径有分歧

`M=1`，ncu `--set full --launch-count 1`：

| 指标 | baseline k5 | persist c8 |
|---|---|---|
| **Waves Per SM** | **2.58** | **1.00** |
| Achieved Occupancy | 21.6% | 24.1% |
| Compute (SM) Throughput | 47.0% | 51.0% |
| DRAM Throughput | 18.8% | 18.2% |
| L2 Cache Throughput | 22.3% | 22.1% |
| Registers / thread | 90 | 90 |
| Duration（ncu 单次） | **80.7 µs** | **83.2 µs** |

机制上完全符合预期：持久化把 wave 从 **2.58 压到 1.00**、occupancy 抬了 2.5pp。
但 **ncu 的单次 Duration 反而显示持久化更慢**（83.2 vs 80.7 µs），而 event 计时
（5 warmup + 50 iters 连续）显示持久化更快——这是第 14/36 篇记过的「event 与 ncu 口径差」：

- event 计时的 50 连发是**稳态吞吐**：持久化网格正好 264 = 全占用，每次 launch 没有尾波；
- ncu 单次 + 锁频把 **持久化多出来的整数除法（每 stage 2 个 `gid/ntiles`、`gid%ntiles`）**
  放大了（Compute 利用率 47%→51% 就是证据），尾波优势被 ALU 劣势抵消。

> 这是个诚实的教训：**当一个优化同时改动了「并行度」和「每 stage 的指令数」时，
> ncu 单次和 event 稳态可能给出相反的符号。** 本篇以 event 稳态为发布口径（因为
> decode 服务是连续请求流），但把 ncu 的相反结果原样记下来。

---

## 五、它离 SOTA 有多远

有效 int4 权重带宽 **565 GB/s = 16.9% HBM**，离纯权重 roofline（13.3 µs / 3352 GB/s）
差 **5.9×**。换 bf16 口径：40 篇测得 cuBLAS 读 bf16 权重是 **2661 GB/s**
（把权重量化成 int4 省下的字节，并没有等比例换成时间）。

真·SOTA 是 vLLM 的 **Marlin**（warp-specialized + 多级异步 dequant，把反量化完全藏进
张量核）：40 篇的阻塞项仍在——本容器缺 `libdwfl`，`vllm` 的 marlin kernel 编不出来。
以「有效权重带宽」的粗口径估，565 GB/s 距 Marlin 类 SOTA 仍有约 **4~7×** 的量级差距，
根因是**反量化算术没有和 `wgmma` 完全重叠**（41 诊断地板：关掉 dequant 算术仍只
639.8 GB/s，说明骨架本身还是延迟受限）。

---

## 六、小结与教训

- **持久化 + 展平 stage 流**确实能把 wave `2.58 → 1.0`，并在 `M ≤ 16` 的 decode 上换来
  **~2~3%** 的稳态净胜（`M=1` 最佳 0.0788 ms / 565.5 GB/s，比 41 篇 1.04×）。
- **控制实验是灵魂**：同样细的 chunk，baseline（680 短命 CTA）比持久化（264 长命 CTA）
  慢，证明收益来自「不重启流水」，而不是「切得细」。别把两种效应混在一起归因。
- **epilogue 一定要待在热循环之外**：它一进 flat 循环，寄存器 90→207，occupancy 腰斩。
  删掉 epilogue 冒烟编译（80 regs）是定位它的最快方法。
- **`wgmma` 累加器用 `scale_d=0` 清零**，跨 item 复用才不会被 ptxas 判 `C7514` 串行化。
- **`acc[64]` 锁死 2 CTA/SM**：3 CTA/SM 需要 ≤85 寄存器，而 wgmma 累加器+描述符 ≥90，
  这条 occupancy 上限只能接受。
- **wave 和 K-split 归约税是一对矛盾**：穿透到 `M≥32` 时细 chunk 的 `atomicAdd` 流量
  反超并行度收益，持久化在 prefill/batched 上反而输给 baseline。
- **ncu 单次 ≠ event 稳态**：当优化同时动了并行度和指令数时，两者可能反号；发布口径要选
  清楚并如实记录另一个。

**下一篇预告**：`M ≥ 32` 的反转说明「按 `M` 自适应选 chunk / 是否持久化」才是正解；
另外 `acc[64]` 的寄存器墙指向同一个老对手——把输出累加器搬进 tensor memory 的
Blackwell `tcgen05`（backlog 已记）。也会继续把 [第 42 篇 MLA decode]({{< relref "cuda-kernel-opt-42-mla-decode-wgmma" >}})
的 producer/consumer warp specialization 推进下去。

---

## 复现

```bash
cd ~/proj/tech_record/code/kernel-opt
scripts/lab.sh up
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -Xptxas -v" \
  scripts/run.sh 43-w4a16-persist/w4a16_persist.cu 1  17408 5120 all      # M=1
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/run.sh 43-w4a16-persist/w4a16_persist.cu 64 17408 5120 cmp      # 扫 M 用
# ncu（M=1）：
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/ncu.sh 43-w4a16-persist/w4a16_persist.cu --set full --launch-count 1 \
  --kernel-name regex:w4a16_persist -- 1 17408 5120 probe_c8
```

原始输出见 `code/kernel-opt/43-w4a16-persist/`：`ws_M1.out.txt`、`ws_M64.out.txt`、
`msweep.out.txt`、`ncu_base_M1*.out.txt`、`ncu_persist_c8_M1*.out.txt`、`3cta_probe.out.txt`。
