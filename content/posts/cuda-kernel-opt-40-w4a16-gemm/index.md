---
title: "CUDA 算子调优（四十）：W4A16 dequant-GEMM——权重 4-bit 化在 decode 赚的那 1.4×，以及反量化税"
date: 2026-09-22T01:00:00+08:00
draft: false
weight: 40
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "量化", "W4A16", "AWQ", "GPTQ", "dequant", "GEMM", "wgmma", "TMA", "tensor-core", "H100", "Hopper", "Qwen3", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」量化推理方向的一篇。前面 [第 37 篇]({{< relref "cuda-kernel-opt-37-fp8-pb-prescale" >}})
把 DeepSeek-V4 的 **FP8 per-block** 折成 per-tensor，[第 34 篇]({{< relref "cuda-kernel-opt-34-fused-moe-fp8" >}})
把 MoE 专家 FFN 整体 FP8 化。这一篇换到 **4-bit 权重**（W4A16，AWQ/GPTQ 风格）：激活仍是 bf16，
权重按 `int4` 打包 + `group_size=128` 的 per-group 对称 scale。

和 FP8 的动机不同：FP8 是「算力 + 带宽」双收益（Hopper 上 FP8 张量核是 bf16 的 2×）；
而 **int4 只减权重字节（4×），不减 FLOPs，也没有 int4 张量核**。所以它的价值场景很明确——
**decode**：`M` 很小（一个 batch 的 token），整个算子卡在「必须把 `N×K` 个权重读一遍」，
算术强度低到 tensor core 大量空转，此时把权重从 16 bit 压到 4 bit 就是直接的 4× 字节节省。

问题在于：**4-bit 权重不能直接喂张量核，必须反量化（dequant）回 bf16**。这篇就把这个「反量化税」
量化出来：它在什么工作点划算、在我们这个实现里吃掉多少、以及为什么它随 `M` 线性放大。

真实 shape 取自 `/ssd/models/qwen3-8B/config.json`（`hidden_size=5120, intermediate_size=17408`），
即 MLP 的 `up_proj`/`gate_proj`：`M×K @ N×K`，`K=5120, N=17408`，`group_size=128`，`M∈{64,128,256}`。

先把结论摆在这（`M=64` = 64 路并发 decode，H100 实测）：

| 版本 | 手段 | 耗时 | 有效权重带宽 | 相对 v0 |
|---|---|---|---|---|
| **v0 total** | `dequant` 物化 bf16 W + cuBLAS bf16 GEMM | **0.1536 ms** | 1160 GB/s（按 bf16 字节） | 1.00× |
| v0-a | 只是 `dequant`（读 44.6 MB int4，写 178.3 MB bf16） | 0.0866 ms | 2575 GB/s | — |
| v0-b | 只是 cuBLAS bf16 GEMM（读 178.3 MB bf16） | 0.0670 ms | 2661 GB/s | — |
| **v1 fused** | `cp.async` 流水 + 主循环内 dequant→SW128 + `wgmma` + split-K | **0.1114 ms** | **400 GB/s**（按 int4 字节） | **1.38×** |
| v1′ 诊断 | 同上但**关掉 dequant 算术**（孤立天花板，结果无意义） | 0.0613 ms | 728 GB/s | 2.51× |

四个结论：

1. **在 `M=64` 的 decode 工作点，融合版比「物化 + GEMM」快 1.38×**——字节确实从 178 MB 降到 45 MB。
2. 但它**远没到带宽顶**：int4 权重的理论 roofline 是 `44.6 MB / 3352 GB/s = 13.3 µs`，我们跑了 111 µs，
   **差 8.4×**。连「去掉 dequant 算术」的天花板（61 µs）也才 728 GB/s（22% HBM）。
3. **反量化税 = 0.051 ms，占融合 kernel 的 46%**。ncu 定位到根因：occupancy 只有 **6.5%**（shared-mem 锁 2 CTA/SM），
   主导 stall 是 **fixed-latency execution dependency（37.7%）**——就是那串反量化 ALU 依赖链没被藏住。
4. **融合版的 dequant 工作量 ∝ `M/BM`**：权重被每个 m-tile 重新反量化一遍。所以 `M=128` 就掉到 0.93×、
   `M=256` 只剩 0.55×。**W4A16 融合只在 `M ≤ BM`（单 m-tile）的纯 decode 工作点才赢**——
   恰好就是它的目标场景，这个边界本身是文章最有价值的部分。

## 一、背景：W4A16 到底在优化什么

### 1.1 工作点：decode 是权重带宽题

对一个线性层 `Y[M,N] = X[M,K] @ W[N,K]^T`：

$$
\text{FLOPs}=2MNK,\qquad \text{bytes}\approx \underbrace{2MK}_{\text{激活}}+\underbrace{w\,NK}_{\text{权重}}+\underbrace{c\,MN}_{\text{输出}}
$$

其中 `w` 是权重每元素字节数（bf16 为 2，int4 为 0.5），`c` 是输出每元素字节数（fp32 为 4）。
算术强度随 `M` 缩放：

$$\text{AI}(M)=\frac{2MNK}{2MK + wNK + cMN}\xrightarrow[M\to 0]{}\frac{2M}{w}$$

Hopper bf16 张量核的 ridge 点约 `989 TFLOPS / 3352 GB/s ≈ 295 FLOP/byte`。代入 `w=0.5`，
`M=64` 时 `AI ≈ 256`，已经低于 ridge——**就算是 4-bit，`M=64` 也才刚到算力/带宽交叉点附近**；
`w=2`（bf16）时 `M=64` 的 `AI≈64`，**深在带宽墙里**。

这解释了两件事：

- cuBLAS bf16 在 `M=64` 时只跑到 **170 TFLOPS（17% 峰值）**，但权重带宽 **2661 GB/s（79% HBM）**——纯带宽受限；
- 4-bit 把权重字节砍到 1/4，理论上能把这 79% 的带宽占用换成 ~4× 的时间节省，**前提是 dequant 几乎免费**。

### 1.2 两个必须跑一遍的基线

朴素部署路径（我们叫 v0）是两步：

1. **物化 dequant**：把 `[N, K/2]` 的 packed int4 展开成 `[N, K]` 的 bf16，写回显存；
2. **GEMM**：拿 bf16 权重跑 cuBLAS。

它的代价是**额外一趟「写 178 MB + 读 178 MB」**。在 decode 里这意味着每来一个 token，
权重被读写两遍：`44.6(读 int4) + 178(写) + 178(读) = 400 MB`，而融合路径只需 `44.6 MB`。

### 1.3 量化格式（AWQ/GPTQ 风格）

`group_size=128`：每 128 个连续 K 元素共享一个 scale。对称 int4，`q∈[-8,7]`：

$$
W[n,k]\approx q[n,k]\cdot s\!\left[n,\left\lfloor k/128\right\rfloor\right],\qquad
q=\mathrm{clamp}\!\left(\mathrm{round}(W/s),-8,7\right)
$$

打包：两个 `q` 挤进一个字节（`q+8` 存成 4 bit，低 nibble 是偶数 `k`，高 nibble 是奇数 `k`）：
`[N, K] → [N, K/2]` 的 `uint8`。scale 存 `[N, K/128]` 的 fp32。

## 二、v1 融合 kernel 的设计

融合的关键：**权重从不物化**，直接在 mainloop 里把 int4 反量化进 smem 的 SW128 布局，再喂 `wgmma`。

### 2.1 计算与搬运

一 CTA = 一个 m-tile（`BM=64`，1 个 warpgroup = 128 线程） × 一个 n-tile（`BN=128`）：

```
        global                    smem(per stage)                     compute
  A[M,K] bf16 ──cp.async──▶  As[64][64]  (SW128 K-major) ─┐
  Wp[N,K/2] u8 ─cp.async──▶ Wps[128][32] (compact)         │
                                   │ dequant (uint4 写)    ├─▶ wgmma.m64n128k16 ×4 → acc[64]
                                   ▼                        │
                            sW[128][64] bf16 (SW128 K-major)┘
```

- **A**：`bf16 [64,64]`，`cp.async` 16B 直接写进 SW128 布局（`sw128_off`）。
- **Wp**：每行 `BK/2 = 32` 字节紧凑排布，`cp.async` 2×16B 搬进来。
- **dequant**：每线程处理一个 `uint32`（4 packed 字节 = 8 个 `k`），读出来 → 8 个 `q` → `q·s` → 8 个 bf16
  → 打包成 **一个 `uint4`（16B）** 写进 `sW`。这一步是文章的性能核心，后面细说。
- **wgmma**：`m64n128k16` 走 `BK/16=4` 步，A、B 描述符都是 K-major SW128（同 20/22 篇）。

### 2.2 为什么是 SW128 且 `BK=64`

`wgmma` 的 SS 操作数从 smem 描述符直读，bf16 的 SW128 atom 是「8 行 × 64 元素 = 128B」，
所以 **`BK` 锁死 64**。反量化的输出必须**按 SW128 的物理偏移**写：
元素 `(row,k)` 的字节偏移是

$$
\text{off}(r,k)=\big(r/8\big)\cdot\big(K/64\big)\cdot1024 \;+\; \big(r\%8\big)\cdot128 \;+\; \big((k/8)\oplus(r\%8)\big)\cdot16 \;+\; (k\%8)\cdot2
$$

因为一个 `uint32` 覆盖的 8 个连续 `k` 落在同一个 8-元素组里，它们的 16B 正好是**同一个 16B 列**，
所以能一次 `uint4` 写完——这是让 dequant 写侧不爆炸的关键（早期按 2 字节标量写会撞 bank）。

### 2.3 用 split-K 买并行度

`M=64` 只有 `N/BN=136` 个 n-tile，即 **136 个 CTA**，每个 CTA 又只有 4 个 warp。
在 132 个 SM 上这等于「每 SM 一个 CTA、每 SM 4 个 warp」——**占用率低到藏不住任何延迟**。
沿 K 切 `KSPLIT` 份（z 维），把网格放大到 `136×KSPLIT`，用 `atomicAdd` 归约。实测 `KSPLIT=4` 把
`0.136 → 0.112 ms`（+21%），但 `4→8` 不再涨（atomic 与 extra C 流量抵消）。

## 三、实测结果

### 3.1 `M=64`（decode）

命令：

```bash
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcublas" \
  scripts/run.sh 40-w4a16-gemm/w4a16_gemm.cu 64 17408 5120 all
```

| 版本 | 配置 | 耗时 | TFLOPS | bf16 峰值% | W-bw |
|---|---|---|---|---|---|
| v0 dequant | 向量化 int4→bf16（写 178 MB） | 0.0866 ms | — | — | 2575 GB/s（223 MB 口径） |
| v0 cuBLAS | bf16 权重 GEMM | 0.0670 ms | 170.3 | 17.2% | 2661 GB/s |
| **v0 TOTAL** | 两步 | **0.1536 ms** | 74.3 | 7.5% | 1160 GB/s |
| fused | `64×128 s3 k1` | 0.1361 ms | 83.8 | 8.5% | 328 GB/s |
| fused | `64×128 s3 k2` | 0.1327 ms | 86.0 | 8.7% | 336 GB/s |
| **fused** | **`64×128 s3 k4`** | **0.1119 ms** | 102.0 | 10.3% | **398 GB/s** |
| fused | `64×128 s3 k8` | 0.1114 ms | 102.4 | 10.4% | 400 GB/s |
| fused | `64×128 s2 k4` | 0.1124 ms | 101.5 | 10.3% | 397 GB/s |
| fused | `64×128 s4 k4` | 0.1256 ms | 90.8 | 9.2% | 355 GB/s |
| fused | `64×256 s3 k4` | 0.1544 ms | 73.9 | 7.5% | 289 GB/s |
| *nodq* | *`64×128 s3 k4`，关 dequant 算术* | *0.0613 ms* | *186.3* | *18.8%* | *728 GB/s* |

`fused 0.112 ms` vs `v0 total 0.154 ms` = **1.38×**。但 `v0 cuBLAS` 单项就 0.067 ms——
也就是说**融合版（0.112）比「假设权重已经是 bf16、直接跑 cuBLAS」还慢**。
这正说明：int4 省下的字节，被反量化税吃掉了大半。

### 3.2 反量化税有多大：一个「关掉算术」的对照

要判断钱花在哪，把 kernel 里 `dequant_stage` 的算术**短路**（`if constexpr(!DQ) return;`）跑一遍——
B 全没写，结果无意义，但能测出「`cp.async` 搬运 + 屏障 + wgmma」的地板：

$$
T_{\text{dequant tax}} = 0.1119 - 0.0613 = 0.0506\ \text{ms}\approx 46\%
$$

即便去掉 dequant，**61 µs 也才 728 GB/s（22% HBM）**。所以瓶颈有两层：dequant 算术（46%）+
骨架本身的低占用/延迟（54%），后者才是更大的问题。

### 3.3 ncu：6.5% occupancy，卡在 ALU 依赖链

对最佳配置（`fused64x128s3k4`，`M=64`）跑 `ncu --set full`：

| 指标 | 值 |
|---|---|
| DRAM Throughput | **11.0%** |
| Compute (SM) Throughput | 25.9% |
| L1/TEX / L2 Throughput | 21.6% / 13.9% |
| Executed IPC (active) | 1.43 |
| Registers / thread | 156 |
| Dynamic smem / block | 86.0 KB |
| Theoretical / Achieved Occupancy | **12.5% / 6.5%** |
| 主导 stall | **fixed-latency execution dependency 37.7%** |

`DRAM` 只有 11%——**根本不是带宽受限**，而是**占用率太低、ALU 依赖链没被藏住**。
`86 KB` smem 把每 SM 锁在 2 个 CTA（理论 8 warp），实际只跑出 ~4 warp；
每个线程的反量化是「读 uint32 → 解 8 个 nibble → 8 次 int→float→bf16 → 打包 uint4」的一条长依赖链，
4 个 warp 轮不过来。

### 3.4 致命的结构性问题：dequant 随 `M` 线性放大

融合 kernel 里，**每个 `(m-tile, n-tile)` CTA 都要把它的 `BN×K` 权重反量化一遍**。
权重反量化的总工作量是

$$
\text{Work}_{\text{dequant}} \propto \frac{M}{BM}\cdot N\cdot K
$$

**它是 `M` 的线性函数**。而 cuBLAS 路径把权重物化一次后，对任意 `M` 复用同一份 bf16。
于是随着 `M` 增大，天平迅速倒向「物化 + GEMM」：

| M | v0 TOTAL | fused 最佳 | fused/v0 |
|---|---|---|---|
| **64** | 0.1536 ms | **0.1114 ms**（s3 k8） | **1.38×** ✅ |
| 128 | 0.1571 ms | 0.1688 ms（s2 k4） | 0.93× ❌ |
| 256 | 0.1658 ms | 0.3001 ms（s2 k4） | 0.55× ❌ |

`M=256` 时 cuBLAS bf16 已经到 **580 TFLOPS（58.7%）**，权重带宽退居次要；而我们的 dequant
被重复了 4 遍，直接把 kernel 拖到 0.30 ms。

```
融合版相对 v0 的收益
 1.4x |*
 1.2x | *
 1.0x |  *
 0.8x |   *
 0.6x |    *
 0.4x |     *
      +--------------
       64  128  256   M
       (赢)  (平)  (输)   ← 分界线就是 M = BM = 64
```

**结论：W4A16 融合的合理工作点是 `M ≤ BM`（单 m-tile）的纯 decode。**
这与「dequant 工作量与 `M` 无关」的直觉相反，是本文最该记住的一条。

## 四、和 SOTA 的差距

W4A16 的事实标准是 vLLM 的 **Marlin**（`~/github/vllm/csrc/quantization/marlin/`，也是
AWQ/GPTQ 的后端）。它的做法是：**`m16` 小 tile** 专门吃 decode 的 `M=1..16`、**把 dequant 与
`mma` 深度流水**、配合 weight-only 的特定 smem 布局。本机没法直接编 vLLM（要整套 build），
所以给一个可量化的上界对比：

| 口径 | 值 |
|---|---|
| 我们的 fused 有效权重带宽（`M=64`） | **400 GB/s** |
| cuBLAS bf16 权重带宽（对照，说明 DRAM 能跑多快） | 2661 GB/s（79% HBM） |
| int4 权重理论 roofline | 3352 GB/s |
| **差距** | **≈ 6.7×**（vs cuBLAS 实测口径）/ **8.4×**（vs 峰值） |

也就是说，Marlin 那类 kernel 靠 m16 tile + 流水化 dequant，能把这 400 GB/s 推到接近 2 TB/s 级；
**我们缺的正是「让反量化算术与张量核真正重叠」的流水**。本文的 `nodq` 诊断给出了天花板：
一旦 dequant 被藏住，就能到 728 GB/s；再叠上更好的占用率/流水，才有资格谈「逼近 roofline」。

## 五、把技巧记下来

1. **int4 的 `ldmatrix` 式技巧：一个 `uint32` → 一个 `uint4`**。SW128 布局下，同一个 16B 列里正好是
   8 个连续的 `k`（`(k%8)` 只动低 3 位、`(k/8)` 决定列），所以「读 4 个 packed 字节、写 1 个 16B」
   是最小 bank 冲突的粒度。用 2 字节标量写会有一堆 bank conflict。
2. **反量化不要 `__float2bfloat16(float(q)*s)` 三次运算**——那是一条长依赖链。可优化方向是
   `int→bf16` 后 `__hmul`，或每行预置 16 项 LUT。
3. **split-K 是低并行度（小 `M`）的救命稻草**，但 `KSPLIT>4` 后 `atomicAdd` 与 C 的读写流量会吃掉收益。
4. **融和 dequant 有「`M` 墙」**：`Work ∝ (M/BM)NK`。要么 `BM` 吃满整个 `M`，要么老老实实物化。
5. **别被「省了 4× 字节」骗了**：`M=64` 融合版的有效带宽只有 400 GB/s，比 cuBLAS 读 bf16 的
   2661 GB/s 低一个数量级——**字节省了，但吞吐掉了更多**。判断融合值不值，永远要算「省字节」和
   「多出来的算术/依赖」两笔账。

## 六、小结与下一篇

- 真实 shape（Qwen3-8B `K=5120,N=17408`，`group=128`）上，W4A16 融合在 `M=64` decode 达到
  **0.1114 ms vs 朴素两步 0.1536 ms（1.38×）**；`M=128/256` 反而输（0.93×/0.55×）。
- ncu 证明它**不是带宽受限**（DRAM 11%）：`86 KB` smem 把 occupancy 锁在 6.5%，
  反量化 ALU 依赖链（fixed-latency stall 37.7%）没被藏住，**占 46% 时间**。
- 与 Marlin 类 SOTA 的差距约 **6.7×**（按有效权重带宽），出路是 **dequant 与 wgmma 的 warp specialization
  / 多级流水**，以及 **`m16` 小 tile 打真正的 `M=1` decode**。
- 配套代码：`code/kernel-opt/40-w4a16-gemm/`（`w4a16_gemm.cu` + `w4a16_M{64,128,256}.out.txt` +
  `ncu_w4a16_fused.out.txt`）。

下一篇候选：回到 [ROADMAP](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/ROADMAP.md)
里搁置的 **MLA decode 的 `wgmma` 化**（主题 25c，解第 39 篇的 `ldmatrix`/寄存器墙），
或者给这篇的 W4A16 补上 **warp-specialized dequant**（把反量化税压下去）。
