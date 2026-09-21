---
title: "CUDA 算子调优（三十一）：MoE grouped GEMM 叠上 DeepSeek 的 per-block FP8 缩放 —— 寄存器墙在分组场景下更贵还是更便宜？"
date: 2026-09-21
draft: false
weight: 31
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "grouped GEMM", "FP8", "per-block scaling", "DeepSeek", "DeepSeek-V4", "wgmma", "TMA", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 25 篇]({{< relref "cuda-kernel-opt-25-moe-grouped-gemm" >}})把 384 个 expert 的 FFN GEMM 揉进
**一个 kernel**（把 expert 编码进 B 的行坐标 `b_row = group*N + n`，一个 2D TMA 描述符覆盖全部权重），
per-tensor FP8 在 32k token 上跑到 **1008 TFLOPS**；[第 22–24 篇]({{< relref "cuda-kernel-opt-24-fp8-gemm-pb" >}})
则把 DeepSeek-V4 真实的 `e4m3 + weight_block=128×128` 缩放做进**单个稠密 GEMM**，结论是 per-block 的
代价不来自多算的 FLOPs，而是每线程多一个跨 k 保留的 fp32 累加器 `fin` 把寄存器从 90 顶到 154，
**occupancy 2 CTA/SM → 1 CTA/SM**。

这一篇把两者合起来，回答一个很实际的问题：**当 GEMM 变成「384 个 expert、分组调度」的形态后，
per-block 缩放还会不会那么贵？** 答案有点反直觉——**更便宜**，而且便宜的机制正好是 24 篇指出的那一条。

shape 全部取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：
`hidden=7168, moe_intermediate_size=3072, n_routed_experts=384, num_experts_per_tok=6,
quantization_config={fmt: e4m3, scale_fmt: ue8m0, weight_block_size: [128,128]}`。

先给结论（H100 SXM，FP8，实测；数字均为 aligned 口径 `2·M_total·N·K`，即含 padding 行）：

| 配置 | tokens=8192 | 16384 | 32768 |
|---|---|---|---|
| per-block grouped 最佳 | **707.7**（128×128 s4） | **790.4**（128×128 s4） | **806.4**（128×128 s3） |
| per-tensor grouped 同二进制 | 777.4 | 960.5 | 980.5 |
| per-block / per-tensor | 91.0% | 82.3% | 82.2% |
| 稠密单 GEMM（24 篇，M=4096） | — | — | per-block 936.5 / per-tensor 1209.6 = **77.4%** |

即：**在分组场景下 per-block 的相对损失（~18%）比稠密单 GEMM（~23%）小**，而 decode（masked）
这个纯权重带宽场景里 per-block 的折算几乎免费（B-read 2.86 TB/s ≈ 85% HBM，与 per-tensor 的 25 篇持平）。

---

## 1. 算子与缩放布局

expert FFN 的第二个 GEMM：

$$
C[m,n] = \sum_{k=0}^{K-1} A[m,k]\cdot B[\text{expert}(m),n,k]\cdot \text{sa}[m,\lfloor k/128\rfloor]\cdot \text{sb}[\text{expert}(m),\lfloor n/128\rfloor,\lfloor k/128\rfloor]\cdot s_{\text{tensor}}
$$

- `A` 是 `(M_total, K)` 的 fp8，按 expert 连续拼接；`B` 在显存里是 `(G, N, K)`，就是 `(G·N, K)`。
- 激活 `sa` 是 **per 行 per 128-k 块**的标量（DeepSeek 的 dynamic 1×128）；权重 `sb` 是
  **128×128 块**标量，行坐标 = `group·(N/128) + n/128`。
- `s_tensor` 是全局缩放（模拟 ue8m0 的整张量 exponent），三处独立相乘。

kernel 骨架原样复用 25 篇：TMA `cp.async.bulk.tensor.2d`（A/B 各一张 2D 描述符，SW128）+ mbarrier
+ warp specialization（1 producer warp 发 TMA，`BM/64` 个 consumer warpgroup 只做
`wgmma.m64n128k32`）。唯一的改动是在 wgmma 主循环里插一段 **per-block 折算**：

```cpp
// 每个 128-k 块：先排空本块 wgmma，再把 acc 折进跨块保留的 fin
wgmma_wait0();
const float sa0 = (r0g < mlim) ? sa[r0g * KBLK + kb] : 0.f;   // 本线程负责的两行
const float sa1 = (r1g < mlim) ? sa[r1g * KBLK + kb] : 0.f;
const float sbv = sb[(group * (N/128) + block_col/128) * KBLK + kb];  // 整块一个标量
for (int j = 0; j < 16; ++j) {
  fin[j*4+0] += sa0*sbv*acc[j*4+0];  fin[j*4+1] += sa0*sbv*acc[j*4+1];
  fin[j*4+2] += sa1*sbv*acc[j*4+2];  fin[j*4+3] += sa1*sbv*acc[j*4+3];
}
for (int i = 0; i < 64; ++i) acc[i] = 0.f;
```

`acc[j*4+0/1]` 对应行 `r0`、`acc[j*4+2/3]` 对应行 `r0+8`，两行的 `sa` 不同——这就是 24 篇 K5 坑说的
「相邻列各有 scale」，只不过这里是「相邻行各有 scale」；`sb` 则因为 `BN=128` 恰好对齐 `weight_block`，
整块退化成**一个标量**，由全 block 广播。

> **per-tensor 对照**：同一份二进制里 `PERBLOCK=false` 时跳过折叠、`fin` 不改、主循环用 25 篇的
> `wgmma_wait_group<STAGES-2>` 延迟排空。这样两组数字的差异**只**来自「折叠 + 寄存器」。

---

## 2. 实测：越到小 batch，per-block 越疼

`tokens` 扫 8192 / 16384 / 32768（tokens×top6/G = 每 expert 平均 128 / 256 / 512 行，
对齐到 `BM=128` 后的 padding 分别为 32.5% / 19.8% / 10.8%）：

| 配置 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|
| `pb 128×128 s2` | 615.96 | 681.97 | 715.06 |
| `pb 128×128 s3` | 688.24 | 764.01 | **806.42** |
| `pb 128×128 s4` | **707.70** | **790.42** | 761.25 |
| `pb 64×128 s3` | 587.07 | 655.65 | 648.14 |
| `pb 64×128 s4` | 590.88 | 551.99 | 672.69 |
| `pb 64×128 s3`（`minBlocks=2`） | 581.56 | 587.68 | 601.36 |
| `pb 64×128 s4`（`minBlocks=2`） | 466.95 | 642.27 | 656.73 |
| `pt 128×128 s2` | 688.86 | **960.45** | **980.51** |
| `pt 128×128 s3` | **777.44** | 952.35 | 885.72 |
| `pt 128×128 s4` | 680.99 | 772.35 | 814.61 |
| `pt 64×128 s3` | 612.77 | 575.91 | 689.67 |

（单位 TFLOPS；`pb` = per-block，`pt` = 同二进制 per-tensor；`sN` = TMA 流水级数。）

几个观察：

1. **大 batch（32k）**：per-block 最佳 806.4（s3），per-tensor 980.5，比值 **82.2%**。
2. **16k**：per-block 790.4（s4）vs per-tensor 960.5，比值 82.3%。
3. **8k**：per-block 707.7（s4）vs per-tensor 777.4，比值 **91.0%**——小 batch 反而**损失更小**。
   原因是 8k 时 padding 32.5%、有效行少，整个 kernel 更偏 L2/DRAM 受限，张量管线的空转占比被摊薄。
4. **`s4` 的异常**：`pt s4` 在 32k 只有 814.6（不如 s2/s3），而 `pb s4` 在 8k/16k 是最好的。
   `s4` 时 per-tensor 也是 1 CTA/SM（90 regs 却因 197KB smem 只能 1 个），此时 per-block 和
   per-tensor 的 occupancy 一样，**折叠几乎免费**：16k `pb s4` 790.4 > `pt s4` 772.4（+2.3%）。
   这直接印证了 24 篇的结论——**per-block 的代价是 occupancy，不是折算指令本身**。

---

## 3. 根因：ncu 把「寄存器 → occupancy → tensor 空转」链条钉死

对 32k 的最佳配置 `128×128 s3` 做 ncu（`--launch-count 1`，同一份二进制）：

| 指标 | per-tensor | per-block | 变化 |
|---|---:|---:|---|
| 寄存器 / 线程 | **90** | **155** | +72% |
| Achieved occupancy | **27.9%**（2 CTA/SM） | **13.9%**（1 CTA/SM） | 减半 |
| `sm__pipe_tensor_op_hmma_cycles_active` | **62.4%** | **46.1%** | **−16.3pp** |
| `Compute (SM) Throughput` | 62.2% | 45.6% | −16.6pp |
| `L2 Cache Throughput` | 88.5% | 74.8% | −13.7pp |
| `DRAM Throughput` | 53.2% | 48.7% | −4.5pp |
| `L1/TEX Throughput` | 53.6% | 44.8% | −8.8pp |
| stalls `long_scoreboard` | **12.61** | **2.03** | −84% |
| stalls `barrier` | 0.12 | **1.16** | +9.7× |
| stalls `wait` | 1.41 | 0.79 | — |
| shared bank conflict | 0 | 0 | — |
| ncu duration | 9.11 ms | 11.15 ms | **+22.4%** |

链条非常清楚：`fin[64]` 让寄存器 90→155，越过 2-CTA 门槛（`65536/(288·2)=113`），occupancy 减半；
per-tensor 里占主导的 `long_scoreboard`（12.61，等下一块 TMA/wgmma 操作数）在 per-block 里降到 2.03
——因为每块都要 `wgmma_wait0` 把张量管线排空后折叠，**根本来不及攒下「等操作数」的深度**；代价直接
体现在 `barrier`（1.16，mbarrier 的等待）和只有 46% 的 tensor pipe 活跃度上。L2 吞吐也随并发 CTA 数
一起下降（88.5%→74.8%），因为 1 CTA/SM 时同时飞行的 TMA 请求更少。

从 roofline 看，32k 的最佳 per-block = 806.4 TFLOPS = **40.8% FP8 峰值（1978）**，而 per-tensor 是 49.6%。
一句话：**分组调度的收益（并行度 + B 行坐标折叠）和 per-block 的损失（occupancy）是正交的两件事**，
可以叠加，损失量级约 18%。

### 我试过但没用的（负结果，避免重走）

- **`BM=64` 想让寄存器够 2 CTA**：`BM=64/BN=128` 的每线程 `acc[64]+fin[64]=128` regs，160 线程时
  2 CTA 只需 ≤204 regs，157 确实塞得下——但实测全线更慢（8k 587 / 16k 656 / 32k 648）。
  因为 `BM` 减半让 B 被重读的次数 `M_total/BM` 翻倍，L2 直接压垮，省下的 occupancy 得不偿失。
  即便显式 `minBlocks=2` 也没救回来。
- **`STAGES` 加大**：`s4` 在 32k 比 `s3` 慢（761 vs 806）。smem 已经到极限（1 CTA/SM），
  多一级 TMA 只是把有限的 smem 占满，对折叠造成的 tensor 空转没有帮助。

---

## 4. decode（masked）：per-block 折算免费

decode 用 25 篇的 masked 分支（`A=(G,max_m,K)`，超出 `masked_m[g]` 的 tile 早退），`max_m=128`、
每 expert 平均 32 个 token。这时 kernel 是**纯权重带宽**场景：

| 配置 | 耗时 | useful TFLOPS | B-read |
|---|---:|---:|---:|
| `maskedPB 128×128 s3` @16384 | 3.35 ms | 161.2 | 2527.6 GB/s（75.4% HBM） |
| `maskedPB 64×128 s3` @16384 | 2.95 ms | 182.7 | 2863.5 GB/s（85.4% HBM） |
| `maskedPB 128×128 s3` @32768 | 3.26 ms | 165.4 | 2592.3 GB/s（77.3% HBM） |
| `maskedPB 64×128 s3` @32768 | 2.95 ms | 182.6 | 2862.0 GB/s（85.4% HBM） |

（`B` = 8.45 GB 专家权重；per-tensor 的 25 篇实测 2.8 TB/s。`BM=64` 在 decode 反而更快，因为
masked 场景下每个 m-tile 只有很少有效行，`BM=64` 的 padding/早退粒度更细。）

这里 `BM=64` 是赢家：**读 8.45 GB 权重、2.86 TB/s、85% HBM**——per-block 的折叠完全没有成为瓶颈，
因为整个 kernel 卡在显存带宽上，多余的 FMA 和 `wait0` 都被 DRAM 延迟藏住了。
**这就是为什么「per-block 是否贵」不能脱离工作点回答**：prefill 是算力受限（折叠暴露），
decode 是带宽受限（折叠免费）。

---

## 5. 对标 SOTA：这一档缺公开对照

- **per-tensor grouped（我们的上界）**：同二进制 per-tensor 980.5 @32k；25 篇的 per-tensor
  best 1007.9，cuBLAS per-expert graph loop 1028.8（32k）。
- **per-block grouped 的 SOTA 是 DeepGEMM 的 block-scale grouped GEMM**，但本机 `kernel_lab`
  容器缺 `elfutils/libdwfl-dev`，`import deep_gemm` 的 `_C` 扩展编不出来（见路线图「阻塞」），
  暂时给不出精确差距。
- **cuBLAS 侧也没有 block-scale 入口**：实测 `torch._scaled_mm` 只接受 TensorWise / RowWise
  缩放，传 `(M, K/128)` 的 2D block scale 会报 `Invalid scaling configuration`；
  `torch._scaled_grouped_mm` 同理。所以能给的**上界**就是「per-tensor grouped」：
  **我们 806.4 达到它的 82.2%**。

参考量级：22–24 篇在同一台机器、同一个 `M=4096,N=3072,K=7168` 的稠密 GEMM 上，
per-tensor TMA 版 1217（cuBLAS FP8 1381 的 88%），per-block 936.5。把 per-block 做到分组场景，
**损失从 23% 收窄到 18%**，且 decode 上免费。

---

## 6. 小结

- **把 DeepSeek 真实的 `e4m3 + 128×128 weight_block` 缩放叠进 25 篇的 grouped 骨架**：一个 CTA
  仍只靠 `b_row = group*N+n` 选 expert，per-block 折算就地加进 wgmma 主循环。
- **prefill**：per-block 最佳 **806.4 TFLOPS @32k（128×128 s3）**，达同二进制 per-tensor 的
  **82.2%**；小 batch 损失更小（8k 91%）。
- **根因（ncu 实测）**：`fin` 让寄存器 90→155 → occupancy 2→1 CTA/SM → tensor pipe 62.4%→46.1%，
  主导 stall 从 `long_scoreboard` 变成 `barrier`（每块 `wait0` 排空）。**代价是 occupancy，不是 FLOPs**。
- **分组让 per-block 更便宜**（损失 18% < 稠密 23%），因为 grouped 天生更偏 L2/带宽，张量空转占比更小；
  decode 这种纯权重带宽场景里折算**完全免费**（2.86 TB/s，85% HBM）。
- **负结果**：`BM=64` 省寄存器但 B 重读翻倍，全线更慢；`STAGES=4` 在 32k 反而更慢。
- **对标**：per-block grouped SOTA（DeepGEMM）因容器缺 `libdwfl` 无法编译，cuBLAS 无 block-scale
  入口；给出的上界是同口径 per-tensor grouped 的 82.2%。

### 下一篇

回到 [21 篇]({{< relref "cuda-kernel-opt-21-mla-wgmma-pipe" >}})没做完的 MLA 单 kernel：把 V 的
global 装载也用 TMA + mbarrier 做成 producer/consumer（双缓冲、不再单缓冲），争取把 Sk 长上下文下
残留的 `long_scoreboard` 彻底消掉，逼近 FlashMLA。

---

**配套代码**：[`code/kernel-opt/31-moe-grouped-pb/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/kernel-opt/31-moe-grouped-pb)
（`moe_pb.cu` + `moe_pb_launch.h`；`moe_pb_S{8192,16384,32768}.out.txt`、`ncu_pb_s3.out.txt`、
`ncu_pt_s3.out.txt`、`ptxas.txt`）。复现：

```bash
cd code/kernel-opt
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/run.sh 31-moe-grouped-pb/moe_pb.cu all 32768
```
