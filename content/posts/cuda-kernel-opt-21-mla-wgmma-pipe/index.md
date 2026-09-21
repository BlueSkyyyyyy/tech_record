---
title: "CUDA 算子调优（二十一）：MLA 极限冲刺（二）— 修好 V 转置访存 + cp.async 预取 K（105 → 193 TFLOPS）"
date: 2026-09-21
draft: false
weight: 21
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MLA", "wgmma", "Hopper", "cp.async", "访存合并", "Tensor Core", "DeepSeek", "系列"]
categories: ["算子开发"]
---

[第二十篇]({{< relref "cuda-kernel-opt-20-mla-wgmma-sw128" >}}) 我们把 MLA 的 `wgmma` 操作数布局从无 swizzle 的 `INTERLEAVE` 换成了 **K-major SW128**，smem store bank conflict 从 2.7e8 降到理论下限，`wgmma` 版从 84.6 涨到 **105.3 TFLOPS**——但仍只有 [第十六篇]({{< relref "cuda-kernel-opt-16-mla-fused" >}}) `mma+ldmatrix` 版 `f4s`（170）的 ~62%。20 篇结尾把矛头指向两个结构瓶颈：**1 CTA/SM + 单缓冲 + 无 producer/consumer**。

这一篇先啃其中**最要命、也最容易被忽略的一个**：`wgmma`（SS 形式）要求 B 操作数 **K-major**，于是 MLA 的 PV 必须把全局的 `c_kv`（`[Sk][DC]`，`d_v` 连续）**转置**成 `[d_v][kt]` 再写 SW128。20 篇的转置是**逐元素、地址完全碎片化**的 global gather，ncu 里它一个人吃掉了 73% 的 L1/TEX 吞吐、`long_scoreboard` 高达 4.81。本篇做两件事：

1. **把 V 转置的 global 读改合并**（迭代顺序 `dv` 最快）——105.3 → **142.3 / 159.4 TFLOPS**；
2. **用 `cp.async` 把下一块 K 的搬运藏进 softmax/PV**（单缓冲即可，安全性有严格论证）——再涨到 **159.8 / 193.2 / 198.8 TFLOPS（Sk=1k/4k/8k）**，**在长上下文下反超 `f4s`**。

| 版本 | 手段 | TFLOPS @Sk=1024 | @Sk=4096 | @Sk=8192 |
|---|---|---|---|---|
| 20 篇 `wgmma` SW128 | 无 swizzle 转置 + 单缓冲 | 105.1 | 119.8 | — |
| 本篇 v1 `mla_pipe` | ① V 转置访存合并 | 142.3 | 159.4 | 151.6 |
| **本篇 v2 `mla_pipe3`** | ① + ② `cp.async` 预取 K | **159.8** | **193.2** | **198.8** |
| 16 篇 `f4s`（mma+ldmatrix） | 同系列最好 mma 版 | 169.3 | 183.8 | 173.9 |
| FlashMLA sm90（sparse prefill, H800） | CUTLASS / TMA | ~640 | ~640 | — |

真实 shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json` 的 MLA 吸收口径：$H{=}128$、$d_c{=}512$、$d_r{=}64$、$d_v{=}512$（bf16 dense 峰值 989 TFLOPS @H100）。代码在 `code/kernel-opt/21-mla-wgmma-pipe/`（`sweep.out.txt`、`ncu_*.out.txt` 为原始输出）。

---

## 1. 现象：73% 的 L1/TEX 到底花在哪

20 篇的融合 kernel 每个 KV 分块做三件事：装载 K（`[kt][dk]`，K-major SW128）、装载 V（转置成 `[dv][kt]`）、然后 QK 与 PV。V 的转置代码是：

```cuda
// 20-mla-wgmma-sw128/mla_wgmma_sw.cu:91（20 篇原版）
for (int i = tid; i < DV * (KT / 8); i += T) {
  const int dv = i / (KT / 8), kb = i % (KT / 8);      // ← kb 最快
  uint4 v; unsigned short* s = reinterpret_cast<unsigned short*>(&v);
  for (int j = 0; j < 8; ++j)
    s[j] = ckv[(size_t)(k0 + kb * 8 + j) * DC + dv];   // ← 逐元素 2B 读
  sw128_store16(vs, dv, kb * 8, KT, v);
}
```

问题出在**线程编号顺序**。`i` 里 `kb` 是低位，于是一个 warp 里相邻线程拿到的 `kb` 是连续的（0..7 循环），`dv` 反而每 8 个线程才 +1。对固定 `j` 看全局地址：

```
addr(t) = ckv[(k0 + kb(t)*8 + j)*DC + dv(t)]
相邻线程：kb += 1  →  addr += 8*DC*2 = 8192 字节
```

**相邻线程地址间隔 8 KB**——每个线程独占一个 sector，global 读完全碎片化。ncu 对着 20 篇版（Sk=4096）实测：

```
l1tex__throughput            73.4 %      ← L1/TEX 几乎打满
sm__pipe_tensor_cycles_active 18.2 %      ← 张量单元反而很闲
smsp__..._stalled_long_scoreboard  4.81   ← 大量 warp 在等 global 访存
```

`long_scoreboard` 就是「等全局/L2 数据回来」的 stall 指纹。访存碎 + 只有 8 个 warp（1 CTA/SM），延迟根本藏不住。

## 2. 修法①：让 V 转置的 global 读合并

交换 `i` 的高低位：**`dv` 最快、`kb` 慢**：

```cuda
// 21-mla-wgmma-pipe/mla_pipe.cu:93
for (int i = tid; i < (KT / 8) * DV; i += T) {
  const int kb = i / DV, dv = i % DV;                 // ← dv 最快
  uint4 v; unsigned short* s = reinterpret_cast<unsigned short*>(&v);
  for (int j = 0; j < 8; ++j)
    s[j] = ckv[(size_t)(k0 + kb * 8 + j) * DC + dv];  // 相邻线程读相邻 dv
  sw128_store16(vs, dv, kb * 8, KT, v);
}
```

现在相邻线程 `dv += 1`、地址 += 2 字节，**一个 warp（32 lane）覆盖连续 64 字节 = 2 个 sector**，与理想合并度只差向量化的倍数。写入侧仍是 SW128 的 16B 散布，bank conflict ≈ 4-way（16B 存储的理论下限就是 4.0，20 篇已验证），**没有变差**。

一个容易被忽略的对偶效应：`i` 的映射变了，但**每个线程要读的 8 个元素（8 个 kt 行）没变**，所以指令数不变、寄存器不变——纯粹把访存从「碎片」变成「合并」。这就是「免费的午餐」：**同样多的 load，只是换了个顺序，105.3 → 142.3 TFLOPS（+35%）**。

ncu 前后对比（Sk=4096）：

| 指标 | 20 篇（碎片） | v1（合并） |
|---|---|---|
| `l1tex__throughput` | 73.4% | **45.8%** |
| `sm__pipe_tensor_cycles_active` | 18.2% | **24.4%** |
| `stalled_long_scoreboard` | 4.81 | **2.50** |
| TFLOPS | 119.8 | **159.4** |

## 3. 修法②：`cp.async` 把下一块 K 藏进 softmax/PV

v1 的 `long_scoreboard` 还有 2.50。此时每个 KV 分块内部的顺序是**同步**的：

```
装载 K,V → syncthreads → QK → softmax → 写 P → syncthreads → PV → syncthreads
             ↑ 装载必须等完                  ↑                 ↑
```

global 装载的几百周期延迟全程暴露。理想情况是用 `cp.async` 异步预取下一块、与当前块的计算重叠。但 smem 只有 221 KB（1 CTA/SM），**K/V 没法双缓冲**。关键洞察是：**K 其实不需要双缓冲**。

> **K 的生命周期**：K 只被 **QK 的 `wgmma`** 读；QK 一旦 `wgmma.wait_group 0` 返回，K 就不再被任何人使用。而 PV 只读 P 和 V，**不碰 K**。
>
> 所以：**可以在 PV 运行期间，用 `cp.async` 把下一块 K 覆盖写进 `ks`**——同一块 buffer，不冲突。

重构后的每块流程（`mla_pipe3.cu:105` 起）：

```
prologue: 装 K[0], V[0]; sync
loop:
  QK(K) ; wgmma_wait0                 // ks 用完
  softmax ; 写 P
  if has_next: cp.async K[next] → ks ; cp.async.commit_group   // 与 softmax/P/PV 重叠
  syncthreads                          // P 可见
  PV(P, V) ; wgmma_wait0(V); cp.async.wait_group 0   // 等 K[next] 落地
  if has_next: 装 V[next] → vs         // PV 读完 vs，才能覆盖
  syncthreads
```

`cp.async.cg.shared.global` 支持 16B 粒度，而 SW128 的 K tile 每行正好是 8 个 bf16 的 16B chunk，**目标地址算一下 `sw128_off` 就能直接异步拷**，不占寄存器：

```cuda
// 21-mla-wgmma-pipe/mla_pipe3.cu:32
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
```

为什么这样是**正确**的（而不是凭感觉）？三处约束逐一满足：

1. `cp.async` 写 `ks`，而 `ks` 上一使用者是 QK，`wgmma_wait0()` 已保证 QK 完成（`wgmma.wait_group` 会等矩阵乘读完操作数）；
2. `cp.async` 与并行进行的 softmax/P/PV 无地址冲突（分别写 `ks`、读 `ps`/`vs`）；
3. `cp.async.wait_group 0` 是 **per-thread** 语义，配合循环末尾的 `__syncthreads()` 保证所有线程的拷贝都落地、对下一轮 QK 可见。

**注意 V 仍然必须在 PV 之后才能覆盖**（PV 要读 `vs`），所以 V 的延迟没被藏住——这是单缓冲的硬约束，也是下一步 TMA/双缓冲要解决的。

ncu 前后对比（Sk=4096）：

| 指标 | v1（合并） | v2（+cp.async K） |
|---|---|---|
| `stalled_long_scoreboard` | 2.50 | **1.42** |
| `lts__throughput`（L2） | 19.4% | 33.6% |
| `sm__pipe_tensor_cycles_active` | 24.4% | **30.0%** |
| Duration（ncu） | 7.34 ms | **6.02 ms** |
| TFLOPS | 159.4 | **193.2** |

## 4. 结果：长上下文反超 `f4s`

`mla_pipe3.cu` 完整扫参（`sweep.out.txt`，全部 `max_abs_err` 通过）：

| Sk | v2 TFLOPS | f4s TFLOPS | 相对 f4s |
|---|---|---|---|
| 1024 | 159.8 | 169.3 | 0.94× |
| 2048 | 178.2 | 185.3 | 0.96× |
| 4096 | **193.2** | 183.8 | **1.05×** |
| 8192 | **198.8** | 173.9 | **1.14×** |

短上下文（Sk=1024）还差 ~6%，因为此时 K 预取能藏的比例小、而 V 转置的固定开销占比高；**上下文越长，`cp.async` 预取摊得越薄、优势越明显**，Sk≥4096 起稳定反超 16 篇的 `f4s`。距离 FlashMLA sm90（sparse prefill，H800，~640）约 **3.3×**（口径不同仅作量级参考），比 20 篇的 6.1× 收窄了接近一半。

## 5. 走不通的岔路：直接消灭 V 转置（MN-major）

既然转置是万恶之源，最直接的想法是**别转置**：让 `wgmma` 的 B 操作数直接吃 **MN-major**（$d_v$ 连续）的 `c_kv`。理论上 CUTLASS 的 GMMA 描述符支持 `layout_type` 区分 K-major / MN-major，我按 `mma_traits_sm90_gmma.hpp` 的 canonical 布局推导了一版 `mn128.cuh`（`[kt/8][dv/64][8][64]` + `c'=c^r` swizzle），并用小 GEMM 定点对拍（`mn_test2.cu`）。

结论是**没走通**，记下来当坑：

- 用 cute 的 `make_gmma_desc<Major::MN>` 打印出来的描述符，LBO/SBO 与我推导的**正好互换**（cute 把 n-group 记 `leading`、k-group 记 `stride`）；
- 用 cute 的值直接给 `wgmma` 会 **out-of-range 非法访问**；把 LBO/SBO 对调后不崩，但**结果盐值错乱**（`D[m][n]` 读出的是 `n%8` 之类的重排），说明**数据物理布局与硬件对该描述符的解释仍有偏差**；
- 这个偏差不是「多试几次」能蒙对的，盲试描述符字段既费时又不可靠。

所以本篇选择**保留转置、但把转置做到合并**——收益 35%，确定且已验证；MN-major 的精确定义留给后续用 CUTLASS CuTe 的 `make_tiled_mma` 直接生成描述符来对齐。

## 6. 小结 & 下一步

- **一个换顺序的免费午餐**：V 转置的 global 读从「kb 最快」改成「dv 最快」，相邻线程地址从差 8 KB 变成差 2 B，`l1tex__throughput` 73%→46%，**+35%**。写任何「列访问/转置」的 kernel，先看 ncu 的 `l1tex__throughput` 和 sectors/req。
- **`cp.async` 未必需要双缓冲**：只要某个操作数（这里是 K）的生命周期严格早于并行执行的计算（PV），**同一块 buffer 就能边算边覆盖**。这需要在代码里把「谁读谁、什么时候读完」想清楚，再用 `wgmma.wait_group` / `cp.async.wait_group` 把顺序钉死。
- **瓶颈迁移**：`long_scoreboard` 4.81 → 1.42，tensor pipe 18.2% → 30.0%，L1/TEX 46% 和 `barrier` 1.17 / `wait` 1.01 现在是下一道墙。
- **下一步（文章 22）**：① 剩下的 V 预取需要双缓冲 → 上 **TMA + mbarrier** 做真正的 producer/consumer；② **消掉 QK 的 2× 重复**（两个 warpgroup 现在各算一份完整 S），或改用 `wgmma` RS 让 P 从寄存器直接喂 PV，省掉 P 的 smem 往返与一次 `__syncthreads`；③ 冲 **250+ TFLOPS**，把与 FlashMLA 的差距压到 ≤2.6×。

代码与原始输出：`code/kernel-opt/21-mla-wgmma-pipe/`（`mla_pipe.cu`、`mla_pipe3.cu`、`mn128.cuh`、`mn_test2.cu`、`sweep.out.txt`、`ncu_20baseline.out.txt`、`ncu_pipe.out.txt`、`ncu_pipe3.out.txt`）。
