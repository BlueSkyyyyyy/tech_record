---
title: "CUDA 算子调优（二十四）：FP8 GEMM（三）— per-block 缩放的两堵墙：寄存器与 ptxas 序列化"
date: 2026-09-21
draft: false
weight: 24
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "FP8", "e4m3", "GEMM", "Tensor Core", "wgmma", "TMA", "per-block scaling", "Hopper", "DeepSeek", "量化", "系列"]
categories: ["算子开发"]
---

[上一篇]({{< relref "cuda-kernel-opt-23-fp8-gemm-tma" >}})用 TMA + mbarrier + warp specialization 把 DeepSeek-V4 形状（M=4096, N=3072, K=7168）的 FP8 per-tensor GEMM 推到 **1217 TFLOPS（cuBLAS 的 88.1%）**，但同一篇里 **per-block 缩放**（e4m3 + ue8m0 + `weight_block=128×128`，DeepSeek-V4 的真实量化方案）只到 **906 TFLOPS**。

这一篇专门回答一个问题：**为什么 per-block 追不上 per-tensor？差的那 25% 到底花在哪了？**

结论先放这里：差的不是「折算」的算力，而是**累加器的寄存器**。

- per-tensor 的 fp32 累加器每线程 64 个寄存器，128×128 配置下 ptxas 只用 **90 寄存器**，**2 CTA/SM**；
- per-block 必须额外维护一个跨 k 块的 fp32 累加和（`fin`），每线程 **+64 寄存器 → 154**，**掉到 1 CTA/SM**；
- 能救场的几何配置（BM=256，1 CTA）需要每线程 **128 个累加器寄存器**，而 544 线程的 1-CTA 预算只有 **120**，必然 spill → 实测崩到 **224 TFLOPS**；
- 想用「双累加器 ping-pong」把折算和下一块 `wgmma` 重叠？ptxas 会因为「主循环里读了还在飞的 wgmma 累加器」而 **主动把 wgmma 串行化**（C7514 / C7511），实测反而掉到 289–408。

一句话：**在 sm_90 的 wgmma 上，per-block 缩放的代价主要不是 FLOPs，而是它把 occupancy 从 2 CTA 打到 1 CTA，并且堵死了唯一的补偿路径。**

| 版本 | 手段 | TFLOPS @4096×3072×7168 | 峰值占比 | 寄存器 | occupancy |
|---|---|---|---|---|---|
| 23：per-tensor 128×128 s3 | `wait_group<STAGES-2>` | **1090.8** | 55.1% | 90 | 2 CTA/SM |
| 23：per-tensor 256×128 s4 | 同上 | **1209.6** | 61.2% | 96 | 2 CTA/SM |
| **24：per-block 128×128 s4（最佳）** | 单累加器 + 每块 `wait0` | **936.5** | 47.3% | 154 | 1 CTA/SM |
| 24：per-block 256×128 s3 | 同上 | 224.0 | 11.3% | 96 + 608B spill | 2 CTA/SM（溢出） |
| 24：per-block ping-pong | 双累加器重叠折算 | 358.4 | 18.1% | 168 + spill | 1 CTA/SM（串行化） |
| cuBLAS FP8 per-tensor | cuBLASLt | 1379.5 | 69.7% | — | — |
| cuBLAS FP8 per-row | cuBLASLt | 1335.9 | 67.5% | — | — |

环境：H100 SXM 80GB（132 SM，HBM 3352 GB/s，FP8 dense 峰值 1978 TFLOPS），CUDA 13.2，`ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda -Xptxas -v"`。所有数字来自 `24-fp8-gemm-pb/fp8_pb.out.txt`；`pt_*` 是**同一份代码**里关掉折算的 per-tensor 对照（MODE=3）。

---

## 1. per-block 缩放的数学，以及它为什么贵

DeepSeek-V4 的量化（`/ssd/models/DeepSeek-V4-Pro/config.json`）：

```json
"quantization_config": {
  "fmt": "e4m3", "scale_fmt": "ue8m0", "activation_scheme": "dynamic",
  "weight_block_size": [128, 128]
}
```

激活按 `1×128`（每 token、每 128 个 k）量化，权重按 `128×128` 量化。于是 GEMM 不再是单一的 $C=A B^\top$，而是

$$
C[i,j] = \sum_{kb} \underbrace{s_a[i,kb]}_{\text{逐 token,逐 k 块}}\; \underbrace{s_b[nb,kb]}_{\text{逐 n 块,逐 k 块}}\; \underbrace{\sum_{k\in kb} A[i,k]\,B[j,k]}_{P_{kb}[i,j]}
$$

per-tensor 只需把累加器乘一个标量再输出；per-block 则必须在**每一个 128 宽的 k 块**上给部分积 $P_{kb}$ 分别乘 $s_a s_b$，再加进一个跨 k 块保留的 fp32 累加和。

在 wgmma 的寄存器语义下，这意味着：

```
per-tensor:   acc[64]                  ← 只有一个跨 k 的累加器
per-block:    acc[64]  +  fin[64]      ← 还要一个跨 k 保留的「已折算和」
              └─每次 k 块：wait 到 acc 就绪 → fin += sa*sb*acc → 清空 acc
```

每线程多出的 64 个 fp32 寄存器，就是这篇故事的起点。

---

## 2. 先量出「折算」到底值多少：同 config、同二进制对照

要把「折算的代价」和「装载/几何的代价」分开，最干净的做法是**在同一份代码里关掉折算**。我在同一 kernel 上加了 `MODE=3`：仍然读 per-block 的 `sa/sb`（保证访存行为一致），但**不折算**、允许 `wgmma` 用 `wait_group<STAGES-2>` 跨块流水，最后一次性输出。这就等价于把 per-tensor 路径搬进同一个 kernel。

| 配置 | per-tensor（MODE=3） | per-block（MODE=0） | 差 |
|---|---|---|---|
| 128×128 s3 | 1090.8 | 913.3 | **−16.3%** |
| 128×128 s4 | 938.7 | 936.5 | −0.2% |
| 128×256 s3 | 1162.8 | （不可行） | — |
| 256×128 s4 | 1209.6 | （不可行） | — |

两条关键信息：

1. per-block 的**绝对上限**就是 936（s4），而不是 per-tensor 的 1209；
2. 好用的 per-tensor 配置（128×256、256×128）在 per-block 下**直接不可行**。

先解释第 2 条——这是寄存器墙。

---

## 3. 墙一：寄存器与 occupancy

### 3.1 一笔手算的寄存器账

BM=128、BN=128 时，消费者是 2 个 warpgroup（256 线程）+ 1 个 producer warp（32 线程）= **288 线程**。每个线程负责 $128\times128/256 = 64$ 个输出元素。

| 项 | per-tensor | per-block |
|---|---|---|
| 当前 k 块的部分积 `acc` | 64 | 64 |
| 跨 k 块的已折算和 `fin` | 0 | **64** |
| 地址/描述符/循环等开销 | ~26 | ~26 |
| **合计** | **~90** | **~154** |

而 288 线程下：

```
1 CTA/SM 预算：65536 / 288      ≈ 227 registers/thread
2 CTA/SM 预算：65536 / (2×288)  ≈ 113 registers/thread
```

154 > 113 → per-block **只能 1 CTA/SM**；90 < 113 → per-tensor **2 CTA/SM**。ptxas 实测完全对上：

```
per-tensor 128×128 s3:  Used 90 registers  → Theoretical Occupancy 26.6%（2 CTA）
per-block  128×128 s4:  Used 154 registers → Theoretical Occupancy 14.1%（1 CTA）
```

### 3.2 ncu 对上了

```
               Compute(SM)  L2     DRAM   Achieved Occ  Regs   No Eligible
per-block s4      49.7%     64.8%  30.5%    13.7%        154     64.1%     ← 1 CTA/SM
per-tensor 256x128 67.7%    57.2%  26.8%    25.4%         96     76.2%     ← 2 CTA/SM
```

per-block 的 `Compute` 只有 49.7%（tensor pipe 没喂饱），而 per-tensor 是 67.7%。这就是 1 CTA vs 2 CTA 的差距：只有 2 个 warpgroup 在发 `wgmma` 时，张量管线在折算/等待期间没有另一个 CTA 的 wgmma 来填。

### 3.3 那能不能用 BM=256？

per-tensor 最好的配置是 **256×128 s4 = 1209.6**（96 寄存器、2 CTA/SM）。直觉上 per-block 也想要 BM=256：A tile 的复用翻倍、CTA 数减半、4 个 warpgroup 发 wgmma 更密。

但 BM=256、BN=128 时，消费者是 4 个 warpgroup（512 线程）+ 32 = **544 线程**，每线程输出 $256\times128/512=64$ 个元素：

```
per-block 累加器 = acc[64] + fin[64] = 128 registers/thread
1 CTA/SM 预算：65536 / 544 ≈ 120 registers/thread
128 > 120  →  必然 spill
```

实测（ptxas）：

```
per-block 256×128 s3:  Used 96 registers, 608 bytes spill stores  → 224.0 TFLOPS
per-block 256×128 s4:  Used 96 registers, 608 bytes spill stores  → 220.8 TFLOPS
```

**BM=256 的 per-block 累加器恰好等于整个寄存器文件**（$512 \times 128 = 65536$），连开销都没地方放，所以无论怎么调都是 spill。折中试了 BM=192（3 个 warpgroup，预算 157）：

```
per-block 192×128 s3:  Used 128 registers, 52 bytes spill  → 683.4 TFLOPS
per-block 192×128 s3 + scale 预取 smem            → 699.4 TFLOPS
```

比 128×128 还差（部分 tile 不满 + 3 个 warpgroup 的 A 复用不如预期）。

**结论**：per-block 在 sm_90 上，2 CTA 的几何（128×128）寄存器不够，1 CTA 的好几何（256×128）寄存器更不够。这个两难是所有下文的根。

---

## 4. 墙二：ptxas 会主动串行化「边算边读累加器」

第 3 节的 936 是「老实」写法：每个 k 块 `wgmma_wait0` → 折算 → 下一块。理论上可以把折算折进下一块 wgmma 的阴影里：

```
issue wgmma(block kb)        ─┐
wait_group<1>  ← 等 kb-1      │  kb 的 wgmma 继续在 tensor core 上跑
fin += sa*sb*acc[kb-1]       ─┘  ALU 折算与 kb 的 wgmma 重叠
```

用双累加器（ping-pong）实现，寄存器 +64（acc 变 128 + fin 64 = 192），实测：

```
ping-pong 128×128 s3:  Used 168 registers, 776B spill  → 358.4 TFLOPS
pair-drain 128×128 s3: Used 168 registers, 736B spill  → 397.3 TFLOPS   ← 两块共用一次 wait0
```

比老实的 913 慢 2.3×。原因不是（只是）寄存器：ptxas 直接告诉我们它把 wgmma 串行化了——

```
ptxas info: (C7514) Potential Performance Loss: wgmma.mma_async instructions are
serialized due to non wgmma instructions reading accumulator registers of a wgmma
between start and end of the pipeline stage
```

也就是说：**只要主循环里出现「读 wgmma 累加器」的普通指令，而同一时刻还有 wgmma 在飞，ptxas 就会把整段 wgmma 串行化**，于是「重叠」变成「排队」。想绕过它只有两个方向（本系列后续再试）：

- 用 CUTLASS/CuTe 的 `warpgroup_fence_operand` + `warpgroup_wait<0>` 结构，把累加器读与 wgmma 严格分开（DeepGEMM 就是 `wait<0>` 后折算）；
- 换 **1 个 warpgroup、248 寄存器/线程**（128 线程预算 ~500 → 上限 255）的布局，让累加器不再是瓶颈——但那仍是 1 CTA/SM。

顺带排除一个常见猜测：**不是 scale 的全局访存慢**。把 `sa/sb` 预取进 smem 后反而略慢，说明 scale 读不是瓶颈：

```
per-block 128×128 s3 + scale 全局读 : 913.3
per-block 128×128 s3 + scale 预取 smem: 871.4   ← 更慢
```

也排除了「靠 `__launch_bounds__` 硬塞 2 CTA」：强行把 154 压到 96，只会换回 608B 的 spill：

```
per-block 128×128 s3, launch_bounds(288,2): 219.7 TFLOPS
```

---

## 5. 最终结果与差距

| 实现 | TFLOPS | 峰值 | 对标 |
|---|---|---|---|
| per-block 128×128 s4（**本篇最佳**） | **936.5** | 47.3% | 23 篇 906.2 → **+3.3%** |
| 同二进制 per-tensor 128×128 s3 | 1090.8 | 55.1% | — |
| 同二进制 per-tensor 256×128 s4 | 1209.6 | 61.2% | 本系列 per-tensor 上限 |
| cuBLAS FP8 per-tensor | 1379.5 | 69.7% | 本篇 per-block 的 1.47× |
| cuBLAS FP8 per-row（逐行/逐列 scale） | 1335.9 | 67.5% | 注意：这是**便宜得多**的算子 |

**差距百分比**：本篇 per-block 936.5 是 cuBLAS **per-tensor**（1379.5，上界）的 **67.9%**，是同二进制 per-tensor（1209.6）的 **77.4%**。它也已经略超 23 篇的 per-block 记录（906.2）。

需要强调口径：`torch._scaled_mm` 的 per-row 是 $C[i,j]=s_a[i]s_b[j]\sum_k A B$——scale 只在最后施加一次，**没有「每 128-k 块折算」**。DeepSeek 的 `128×128` block scale 是更贵的算子，所以直接拿它与 cuBLAS per-row 比会低估难度；这里把它列出来仅为给出上界参考。

---

## 6. 复现

```bash
cd code/kernel-opt
scripts/lab.sh up
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda -Xptxas -v" \
  scripts/run.sh 24-fp8-gemm-pb/fp8_gemm_pb.cu          # 全表
# 只跑某一个配置（M N K which）：
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/run.sh 24-fp8-gemm-pb/fp8_gemm_pb.cu 4096 3072 7168 base_gl_128x128s4
```

kernel 里用 `MODE` 模板参数区分各条路径：`0`=per-block 单累加器（23 结构）、`1`=ping-pong、`2`=pair-drain、`3`=per-tensor 对照；`PRELOAD` 控制 scale 是否预取 smem；`MINB` 对应 `__launch_bounds__` 的 minBlocks。

原始输出：`24-fp8-gemm-pb/fp8_pb.out.txt`、`ncu_pb_s4.out.txt`、`ncu_pt_s4.out.txt`。

---

## 7. 小结

- **per-block 缩放的代价主要不是 FLOPs，是寄存器。** 多出来的 64 个 fp32 `fin` 寄存器把 128×128 从 2 CTA/SM 打到 1 CTA/SM，Compute 利用率从 67.7% 掉到 49.7%。
- **补偿路径被堵死。** per-tensor 的王牌 256×128 需要 128 累加器寄存器/线程，而 544 线程的 1-CTA 预算只有 120，必然 spill（实测 224）。
- **想重叠折算，ptxas 不让。** 主循环里读飞行中的 wgmma 累加器会触发 C7514/C7511，wgmma 被串行化，ping-pong/pair-drain 反而慢 2.3×。
- **可用的窗口很窄**：`128×128 s4` 单累加器 + 每块 `wait0` → 936.5 TFLOPS（47.3% 峰值，同配置 per-tensor 的 99.8%）。
- 出路（后续验证）：CUTLASS 式 `fence_operand` + `wait<0>`、1-warpgroup/248 寄存器的 DeepGEMM 布局，或换到 Blackwell 的 **tcgen05**（累加器放 tensor memory，不吃寄存器）。

**下一篇预告**：既然单卡 per-block 的窗口这么窄，转去做**模型级真正省时间的地方**——MoE grouped GEMM（384 experts top-6 的变长分组），以及把 DSA 的稀疏注意力补完。FP8 per-block 的「1 warpgroup / 248 寄存器」布局留作 backlog。
