---
title: "CUDA 算子调优（二十五）：MoE grouped GEMM — 一个 kernel 吃掉 384 个 expert"
date: 2026-09-21
draft: false
weight: 25
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "grouped GEMM", "FP8", "e4m3", "wgmma", "TMA", "warp specialization", "DeepSeek", "Hopper", "系列"]
categories: ["算子开发"]
---

[上一篇]({{< relref "cuda-kernel-opt-24-fp8-gemm-pb" >}})把 DeepSeek-V4 形状的 FP8 GEMM 用 TMA + wgmma 推到了 1217 TFLOPS。那些都是**单个大方阵**。这一篇换到 MoE 里最贵的那个算子：**expert FFN 的第二个 GEMM**——$G=384$ 个形状相同、M 各不相同的「小 GEMM」。

先给结论：

- 天真做法是**循环 384 次、每个 expert 发一个 GEMM**。实测它被 SM 空转和尾部拖死：不管 token 多少，总耗时都卡在 **~13–14 ms**（每个 expert 的 GEMM 只有几十个 CTA，132 个 SM 大部分时间在等）。
- 把 384 个 expert 塞进**一个 kernel**（B 的第 1 维按 `group*N+n` 折叠成行坐标，CTA 自己算用哪个 expert），直接降到 **3.7–9.6 ms**：小 batch **3.6×**、大 batch **1.5×**。
- 它还**反超了 cuBLAS 的 per-expert 循环**（CUDA graph 录 384 次 `_scaled_mm`）约 **1.17×**。
- decode（masked）布局是**纯权重带宽**场景：读一遍 8.45 GB 权重，实测 **2.8 TB/s（83.6% HBM）**，比 cuBLAS 的 padded batched 快 **1.53×**。

| 场景 | 布局 | 基线（per-expert loop） | **grouped（本篇）** | 加速 |
|---|---|---|---|---|
| prefill @8192 tokens | contiguous | 13.18 ms | **3.68 ms**（128×128 s3） | **3.59×** |
| prefill @16384 tokens | contiguous | 13.19 ms | **5.59 ms**（128×128 s3） | **2.36×** |
| prefill @32768 tokens | contiguous | 14.08 ms | **9.58 ms**（128×256 s3） | **1.47×** |
| decode @max_m=128 | masked | 4.62 ms（cuBLAS padded） | **3.02 ms**（128×128 s3） | **1.53×** |

环境：H100 SXM 80GB（132 SM，HBM 3352 GB/s，FP8 dense 峰值 1978 TFLOPS），CUDA 13.2。
所有数字来自 `25-moe-grouped-gemm/moe_S{8192,16384,32768}.out.txt`、`cublas_moe.out.txt`，ncu 见 `ncu_grp128x256s3.out.txt`。

---

## 1. MoE 的第二个 GEMM 长什么样

DeepSeek-V4-Pro 的配置（`/ssd/models/DeepSeek-V4-Pro/config.json`）：

```json
"hidden_size": 7168, "moe_intermediate_size": 3072,
"n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6
```

一层 MoE 的前向：

```
tokens ──router──▶ 每个 token 选 top-6 expert
        │
        ▼  搬运（permute）：把 token 按 expert 分组
   第 1 个 GEMM：  [M_total, K=7168] × [K, N]        ← gate/up
        ▼  SwiGLU
   第 2 个 GEMM：  [M_total, N=3072] × [N, K]        ← down
        ▼  搬回（unpermute）
```

其中 $M_{\text{total}} = \text{tokens}\times 6$，被分给 384 个 expert。这是一个 **grouped GEMM**：N、K 共享，每个 group 有自己的 M 和权重 $B_g\in\mathbb{R}^{N\times K}$。

两种数据布局，对应两种场景：

| 布局 | A 的形状 | 每组 M | 场景 | 语义 |
|---|---|---|---|---|
| **contiguous** | `(M_total, K)`，按 expert 连续拼接 | `gl[row]` 给出 row 属于哪个 expert（padding 行为 -1） | prefill / training | 每个 CTA 的 BM 行要么属于一个 expert，要么是 padding |
| **masked** | `(G, max_m, K)` | `masked_m[g]` 给出有效行数 | decode | 小 M 组里超出 `masked_m[g]` 的 tile 直接退出 |

关键约束：contiguous 布局要求**每组的 M 对齐到 BM**（本文 BM=128）。因为「一个 BM 行 tile 只能属于一个 expert」——否则同一 tile 的不同行要用不同权重。这个对齐会带来**padding 浪费**，后面会看到它随 batch 变化。

---

## 2. 基线：为什么「循环 384 次」不行

最直接的做法：对每个 expert 发一个普通的 GEMM（就是 23 篇那个 TMA+wgmma kernel），共 384 次 launch。

问题在**每个 expert 的 M 太小**。设平均每 expert 128 行（tokens=8192），一个 expert 的 GEMM 是 $128\times3072\times7168$：

```
grid = (N/BN, m_g/BM) = (24, 1) = 24 个 CTA
```

**24 个 CTA 铺在 132 个 SM 上**，每个 SM 连一个 CTA 都分不到。每个 launch 内部又只有 56 个 k 块，pipeline 刚热起来就结束了。于是：

```
tokens= 8192: loop 13.18 ms   (每 expert M≈128)
tokens=16384: loop 13.19 ms   (每 expert M≈256)
tokens=32768: loop 14.08 ms   (每 expert M≈512)
```

耗时几乎不随工作量翻倍——典型的「固定开销 / 并行度不足」指纹，不是算力受限。384 次 launch 的 host 派发只是小头，**真正的税是每次 launch 都只用了 1/5 的 GPU**。

正确的做法不是把每个 GEMM 做快，而是**让所有 expert 的 tile 同时在场**。

---

## 3. 设计：一个 kernel 覆盖全部 expert

核心问题：不同 CTA 要用不同 expert 的 B，怎么在一张 kernel 里表达？

最直接的答案：**把 B 的第 0 维按 `expert*N + n` 折叠**。`B` 在显存里本来就是 `(G, N, K)` 连续存放，等价于 `(G*N, K)` 的一个 2D 矩阵。于是一个 `(m_tile, n_tile)` 的 CTA 只要知道自己的 expert，就能算出 B tile 的行坐标：

```
b_row = group * N + blockIdx.x * BN      // B 的行
a_row = blockIdx.y * BM                  // contiguous：A 的全局行
```

`group` 从哪来？contiguous 布局下每个 m-tile 读一次 `grouped_layout[m_tile*BM]` 即可（整个 tile 同属一个 expert）：

```cpp
group = (gl != nullptr) ? gl[block_row] : 0;
if (group < 0) group = 0;
a_row = block_row;
b_row = group * N + block_col;
```

然后 producer warp 发两条 TMA：

```cpp
tma_load_2d(&tmA, As + st*BM*BK, kb*BK, a_row, full + st);
tma_load_2d(&tmB, Bs + st*BN*BK, kb*BK, b_row, full + st);
```

grid 就是所有 expert 的 tile 并集：

```
grid = (N/BN, M_total/BM, 1)
```

masked 布局只是把 expert 放到 `blockIdx.z`，并在行越界时整块早退：

```cpp
if constexpr (MASKED) {
  group = blockIdx.z;
  block_row = blockIdx.y * BM;
  if (block_row >= gl[group]) return;      // 整个 tile 无效，直接退出
  a_row = group * max_m + block_row;
  b_row = group * N + block_col;
  mlim  = group * max_m + gl[group];       // 写回时只写有效行
}
```

其余部分（TMA + mbarrier + warp specialization + wgmma）原样复用 [23 篇]({{< relref "cuda-kernel-opt-23-fp8-gemm-tma" >}})：

```
consumer warpgroups：wgmma.m64n128k32（SS，直读 smem 的 SW128 描述符）
producer warp      ：mbarrier.arrive.expect_tx + cp.async.bulk.tensor.2d
empty barrier      ：NCONS 个消费者线程各自 arrive
```

**相比 per-expert loop 省掉了什么**：

1. 384 次 launch → 1 次，kernel 内所有 tile 混合调度，132 个 SM 一直有活干；
2. 384×2 张 `CUtensorMap` → 2 张；
3. 每个 expert 的尾部空转被别的 expert 的 tile 填满。

### 3.1 一个把 kernel 挂死的坑：TMA 的 box 必须匹配 BN

我一开始只想在 BN=128 上验证，sweep 里 BN=256 是复用**同一张** B tensor map：

```cpp
CUtensorMap tmB = make_tmap_2d(B, K, G*N, /*boxK=*/128, /*boxR=*/128);  // ← 写死 128
...
launch_moe<128, 256, ...>(tmA, tmB, ...);                                // ← 但 kernel 要 256 行
```

结果是 **BN=256 的配置全部死锁**：producer 用 `expect_tx(BM*BK + BN*BK)` 声明要等 `128*128 + 256*128` 字节，而 TMA 只搬了 `128*128` 字节的 B tile——mbarrier 的 transaction 数永远补不齐，消费者永远卡在 `mbarrier.try_wait.parity` 上。

```
BN=128：expect_tx = 16384 + 16384 = 32768，TMA 搬 32768  ✓
BN=256：expect_tx = 16384 + 32768 = 49152，TMA 只搬 16384+16384 = 32768  ✗ 死锁
```

修法：tensor map 必须跟着 config 一起建，`boxR=BN`。这是个「不报错、只是 hang」的坑，`printf` 到设备端才定位到 producer 没走完。

> 顺便：我第一版想用 **3D TMA**（`dims={K,N,G}`，第三维坐标选 expert）来表达分组，想法更「正统」，但 2D 折叠成行坐标在同一套 SW128 描述符下更简单、少一次维度推理，性能也一样。**能用现成的 2D 坐标解决就不要动 3D**——尤其当排错要面对「TMA 不完成 → mbarrier 死锁」这类无声故障时。

---

## 4. prefill（contiguous）：实测

固定 `K=7168, N=3072, G=384, topk=6`，扫 token 数与 kernel 配置。所有结果都通过 CPU 参考对拍（`max_abs_err/ref < 3%`）。

### 4.1 扫描结果（tokens=16384，平均每 expert 256 行）

| 配置 | ms | TFLOPS（aligned） | 峰值占比 | B-read GB/s |
|---|---|---|---|---|
| per-expert loop（384 launch） | 13.19 | 406.5 | 20.6% | 641 |
| grp 128×128 s3 | **5.59** | **958.3** | **48.5%** | 1512 |
| grp 128×128 s4 | 6.83 | 785.1 | 39.7% | 1238 |
| grp 128×128 s5 | 6.83 | 784.9 | 39.7% | 1238 |
| grp 128×256 s2 | 5.82 | 921.0 | 46.6% | 1453 |
| grp 128×256 s3 | 6.03 | 889.2 | 45.0% | 1403 |
| grp 128×256 s4 | 6.61 | 810.9 | 41.0% | 1279 |

**分组 kernel 把 13.19 ms 打到 5.59 ms（2.36×）**。最佳是 `128×128 s3`，不是 23 篇里更大的 128×256——因为这里的瓶颈已经从「张量管线喂不饱」换成「B 从 L2 频繁重读」，更小的 BN 反而减少单块 smem、提高驻留并行度。

### 4.2 token 数 / padding 浪费的权衡

contiguous 要求每组 M 对齐到 128，padding 比例随 batch 增大而下降：

| tokens | 期望 M/expert | M_total（aligned） | 实际行数 | **padding 浪费** | loop | grouped 最佳 | 加速 |
|---|---|---|---|---|---|---|---|
| 8192 | 128 | 72192 | 48728 | **32.5%** | 13.18 | 3.68 (128×128 s3) | 3.59× |
| 16384 | 256 | 121728 | 97651 | **19.8%** | 13.19 | 5.59 (128×128 s3) | 2.36× |
| 32768 | 512 | 219264 | 195475 | **10.8%** | 14.08 | 9.58 (128×256 s3) | 1.47× |

两个值得记住的事实：

1. **小 batch 时 aligned M 可能是实际行数的 1.5 倍**（32.5% 浪费）——对齐的代价在 prefill 小 batch 下真实存在。想要更省，要么让 router 尽量把每 expert 的分到 128 的整数倍，要么用 mask 风格避免对齐（但那样 tile/权重映射会变复杂）。
2. **loop 的耗时对 token 数几乎不敏感**（13–14 ms），而 grouped 是随工作量走的（3.7→9.6 ms）。所以 batch 越小，grouped 的相对收益越大。

### 4.3 vs cuBLAS per-expert loop

把 384 次 `torch._scaled_mm` 用 **CUDA graph** 录下来回放（去掉 host 派发），tokens=16384：

| 实现 | ms | 折算 TFLOPS | 相对 |
|---|---|---|---|
| cuBLAS eager loop | 6.95 | 774.6 | 基线 |
| cuBLAS **graph** loop | 6.54 | 822.8 | 1.06× |
| **本篇 grouped**（归一化到同 M_total） | **5.62** | ~980 | **1.17× vs graph / 1.24× vs eager** |

即：**grouped kernel 不只赢自己的 per-expert loop，还赢了 cuBLAS 的 per-expert 循环**。后者的 384 个 GEMM 每个也只有一两个 CTA，cuBLAS 再快也补不回 GPU 的空转。

> 关于 SOTA：DeepGEMM 的 `m_grouped_fp8_gemm_nt_contiguous` 是这个算子的强基线，它用 persistent scheduler + TMA multicast + per-block scale 布局。本地 `~/github/DeepGEMM` 的 `_C` 扩展需要 `libdwfl`（容器缺），未能编译，因此这里用 cuBLAS per-expert loop 作为可复现的参照；与 DeepGEMM 的差距留给后续（见 backlog）。

---

## 5. decode（masked）：权重带宽说了算

decode 时每 expert 只有几个到几十个 token（这里平均 32），
$M_{\text{useful}} \ll 128$。但**权重必须全读**：$384\times3072\times7168 = 8.45$ GB。所以这是**权重带宽受限**的场景，算力几乎免费。

masked kernel 的 grid 是 `(N/BN, max_m/BM, G)`，`block_row >= masked_m[g]` 的 tile 早退——省的是**无用行的计算**（以及它们的 A 读和 D 写），但每个 expert 的 B 仍读一遍。

| 配置 | ms | TFLOPS（useful） | **B-read** | vs HBM |
|---|---|---|---|---|
| masked 128×128 s3 | **3.02** | 178.8 | **2802 GB/s** | **83.6%** |
| masked 128×128 s4 | 3.32 | 162.3 | 2544 GB/s | 75.9% |
| masked 128×256 s3 | 3.25 | 166.0 | 2601 GB/s | 77.6% |
| cuBLAS padded batched（graph） | 4.62 | 468.6（padded） | 1830 GB/s | 54.6% |

- 最好的 masked 版本把权重读推到 **2.8 TB/s（83.6% HBM）**；算上 D 写（0.6 GB）有效带宽约 **89% HBM**——基本贴着带宽墙。
- cuBLAS 的 batched 口径必须按 `max_m=128` 全算（它不知道 `masked_m`），**计算量是实际有用量的 4 倍**、且少了早退；即便如此我们的 early-exit 版本在墙钟上仍快 **1.53×**，且是把 8.45 GB 权重从 DRAM 拖出来的硬下限之上。
- 这个场景**堆配置没用**（s4、BN=256 都更慢）：一旦 B 带宽饱和，多出来的 occupancy 只会加剧 L2 竞争。

---

## 6. ncu：瓶颈在哪，下一步往哪打

对 prefill 最佳大 batch 配置 `grp 128×256 s3 @tokens=32768` 做剖析：

```
DRAM Throughput        52.0%
L2 Cache Throughput    81.9%     ← 最高
Compute (SM)           61.3%     ← tensor pipe 61.3%
Achieved Occupancy     13.9%     ← 1 CTA/SM（288 线程）
L2 Hit Rate            63.2%
Warp Cycles/Issued     12.60
  其中 long_scoreboard  ≈ 47.8%（等 L1TEX：global load / smem）
```

读法：

- **L2 81.9% 是最高项**，DRAM 只有 52% → **不是 HBM 受限，是 L2→SM 的带宽受限**。
- B 的 L2 流量被放大了：每个 `(m_tile, n_tile)` CTA 读一整块 `BN×K` 的 B，所以一个 expert 的 B 被它的**每个 m-tile 都重读一遍**。总 B 流量 $\approx (\text{m\_tiles})\times N\times K$：

$$
\text{B traffic} = \frac{M_{\text{total}}}{\text{BM}}\times N\times K
= \frac{219264}{128}\times 3072\times 7168 \approx 37.7\ \text{GB}
$$

  而理想只需读 8.45 GB——**放大约 4.5×**。小 batch 时更差（8192 tokens 时放大约 12×）。

- 47.8% 的 `long_scoreboard` 正是等这些 L2 回包。

**下一步（backlog）**：把 B 的 load 换成 **TMA cluster multicast**。同一 expert 的相邻 m-tile 放同一个 cluster，B tile 从 L2 只取一次、广播给多个 SM，理论上能把 B 的 L2 流量按 cluster 大小整除，直接顶到 tensor pipe 而不是 L2。DeepGEMM 的 `Scheduler` 里那套 `kNumTMAMulticast` / `is_peer_cta_alive`（处理变长 group 无法两两配对时的动态关闭）就是干这个的。另一个方向是 per-block scale 布局（DeepSeek 真实的 `128×128` 量化），会把 24 篇的寄存器墙再叠加进来。

---

## 7. 复现

```bash
cd code/kernel-opt
scripts/lab.sh up

# 编译 + 全量对比（默认 tokens=16384）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/run.sh 25-moe-grouped-gemm/moe_grouped.cu all 16384

# 单跑某个配置（mode ∈ loop | grp… | masked | all）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/run.sh 25-moe-grouped-gemm/moe_grouped.cu grp128x256s3 32768

# cuBLAS per-expert 参照（含 CUDA graph）
python3 25-moe-grouped-gemm/cublas_moe_ref.py --tokens 16384
# ncu
scripts/ncu.sh 25-moe-grouped-gemm/moe_grouped.cu \
  --kernel-name regex:moe_kernel --launch-count 1 --set full -- grp128x256s3 32768
```

代码结构：

- `moe_grouped.cu`：wgmma/TMA/mbarrier helpers + `moe_kernel<BM,BN,BK,STAGES,MASKED>`；
- `moe_grouped_launch.h`：构造真实 MoE 分布（`make_dist`）、tensor map、bench、CPU 对拍；
- `cublas_moe_ref.py`：cuBLAS per-expert（eager + CUDA graph）参照。

原始输出：`moe_S8192.out.txt` / `moe_S16384.out.txt` / `moe_S32768.out.txt` / `cublas_moe.out.txt` / `ncu_grp128x256s3.out.txt`。

---

## 8. 小结

- **MoE grouped GEMM 的第一性原理是并行度**：per-expert loop 每个 GEMM 只有几十个 CTA，132 个 SM 大部分时间空转，耗时对 token 数几乎不敏感（13–14 ms）。
- **把 expert 编码进 B 的行坐标**（`b_row = group*N + n`），一张 kernel 同时调度全部 expert 的 tile，再叠加 23 篇的 TMA + warp specialization：
  - prefill 8192/16384/32768 tokens：**3.59× / 2.36× / 1.47×**，最佳 958–1008 TFLOPS（aligned，峰值 48–51%）；
  - 反超 cuBLAS per-expert 循环 **1.17×**。
- **contiguous 布局的对齐 padding 是真金白银**：小 batch 下浪费 32.5%，随 batch 增大降到 10.8%。
- **decode（masked）是权重带宽场景**：8.45 GB 权重必须全读，实测 **2.8 TB/s（83.6% HBM）**，早退比 cuBLAS padded batched 快 **1.53×**。
- **ncu 指出下一堵墙是 L2（81.9%），不是 DRAM（52%）**：专家 B 被每个 m-tile 重读，L2 流量放大 4.5×。解法是 **TMA cluster multicast**（DeepGEMM 的 scheduler 正是这么做的）。
- 踩坑：TMA 的 `boxR` 必须跟着 BN 一起建，否则 `expect_tx` 永远补不齐 → **不报错、只是死锁**；3D TMA 能表达分组，但 2D 行折叠更省事、同等性能。

**下一篇预告**：把 multicast 补上（B 的 L2 放大会是最直接的增益），或回到 [19 篇]({{< relref "cuda-kernel-opt-19-dsa-sparse-attn" >}})欠下的 **DSA compressor**（ratio ∈ {4,128,0} 的 KV 压缩），把 DeepSeek-V4 稀疏注意力那条线收尾。
