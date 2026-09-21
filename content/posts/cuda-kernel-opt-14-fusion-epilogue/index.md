---
title: "CUDA 算子调优（十四）：融合与 epilogue — 给 GEMM 加上 bias 和激活"
date: 2026-09-21
draft: false
weight: 14
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "epilogue", "融合", "GEMM", "GELU", "ReLU", "bias", "Tensor Core", "split-K", "系列"]
categories: ["算子开发"]
---

上一篇（{{< relref "cuda-kernel-opt-13-tensor-core" >}}）我们把 BF16 Tensor Core GEMM 做到了 **220.98 TFLOPS（22.3% 峰值）**，并留下一个伏笔：真实模型里的 GEMM 几乎从不单独出现，后面总跟着一个 **bias + 激活**。这一篇就来做 **epilogue（尾声）融合**——把 GEMM 的结果在还躺在寄存器里的时候，直接加 bias、过激活函数，再落盘。

配套代码 [`code/kernel-opt/14-fusion-epilogue/fusion_epilogue.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/14-fusion-epilogue/fusion_epilogue.cu)，原始输出、ncu 数据同目录。硬件仍是 H100 80GB HBM3（132 SM，BF16 TC dense 峰值 989 TFLOPS）。

要回答的问题：

1. 「GEMM + bias + 激活」如果拆成两个 kernel，到底贵在哪？
2. 在 mma 的累加器上直接做 epilogue，会不会拖慢 GEMM 本身？
3. 这个收益和矩阵形状（尤其是 K）有什么关系？

核心结论（M=N=2048，K 见下表，BF16 输入 / FP32 累加 / 输出 FP32）：

| 版本 | 做法 | K=2048 | K=512 | K=256 |
|---|---|---|---|---|
| `base` | 纯 GEMM，直接写 C | 0.0778 ms | 0.0336 ms | 0.0260 ms |
| `sep` | GEMM 落盘 + 独立 epilogue kernel 读改写 | 0.0874 ms | 0.0427 ms | 0.0349 ms |
| `fused` | GEMM 寄存器里融合 bias+GELU | **0.0786 ms** | **0.0338 ms** | **0.0262 ms** |
| `fused_relu` | 融合 bias+ReLU | 0.0764 ms | 0.0319 ms | 0.0247 ms |

**融合版比两 kernel 版快 11%（K=2048）、26%（K=512）、33%（K=256）**，而相比不融合的纯 GEMM 只多了约 1% 的耗时——epilogue 几乎是白送的。

---

## 1. 背景：GEMM 之后为什么要跟一个 epilogue

一个完整的线性层（忽略 dropout 之类）是：

```
Y = act(X · W + b)
```

在框架里，这通常被拆成：

1. 调 GEMM 库（cuBLAS/cuBLASLt）算出 `X·W`，写成 `Y`；
2. 启一个 elementwise kernel，读 `Y`、加 `b`、过激活、写回 `Y`。

第 2 步看似廉价，但它对 `Y` 做了 **一趟读 + 一趟写**。对于 2048×2048 的 FP32 输出，一趟就是 16.8 MB，读写加起来 **33.6 MB** 的显存流量——而 GEMM 本身的输入 `A`、`B` 加起来才 16.8 MB。**epilogue 的访存量竟然和主计算相当。**

融合（fusion）的动机就很直白：**既然 GEMM 的结果本来就要写一遍，那就在写之前顺手把 bias 和激活做了，省掉「再读一遍、再写一遍」。** 这正是 FlashAttention、cuBLASLt 的 `CUBLASLT_EPILOGUE_*`、以及 CUTLASS 的 `Epilogue` 在做的事。

---

## 2. 数据流：两 kernel vs 融合

```
两 kernel（sep）：
   A,B ──► [GEMM] ──写C──► DRAM ──读C──► [bias+act] ──写C──► DRAM
                        16.8MB      16.8MB          16.8MB
                        └────────── 额外的 33.6MB 往返 ──────────┘

融合（fused）：
   A,B ──► [GEMM + bias + act] ──写C──► DRAM
                                   16.8MB
             累加器在寄存器里，bias/激活在读出来之前就做完了
```

关键点：在 mma 版本里，一个 block 的 128×128 输出**全程驻留在寄存器**（`acc[2][8][4]`，每线程 64 个 fp32），直到最后 `store_acc` 才写回。融合就是把这个 `store_acc` 改造成「先 bias、再激活、再写」。

---

## 3. 实现：给 `store_acc` 加一段可插拔的 epilogue

第 13 篇的 `store_acc` 把寄存器累加器展开成全局地址写回。这里把它模板化，插入一个 `apply_epilogue`（[`fusion_epilogue.cu:138`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/14-fusion-epilogue/fusion_epilogue.cu)）：

```cpp
enum EpiMode { EPI_NONE, EPI_BIAS, EPI_BIAS_GELU, EPI_BIAS_RELU };

template <int MODE>
__device__ __forceinline__ float apply_epilogue(float v, float b) {
  if (MODE == EPI_NONE) return v;          // 纯 GEMM
  const float x = v + b;                   // bias 广播：同一列共享
  if (MODE == EPI_BIAS) return x;
  if (MODE == EPI_BIAS_GELU) return gelu_tanh(x);
  return fmaxf(x, 0.0f);                   // ReLU
}
```

`store_acc_epi` 在展开每一行时，用全局列号 `c` 取 `bias[c]`，套上 epilogue 再写：

```cpp
const float bv = (MODE == EPI_NONE) ? 0.f : bias[c];
C[(size_t)r * N + c] = apply_epilogue<MODE>(acc[i][j][q], bv);
```

GELU 用 tanh 近似（和 PyTorch `nn.GELU(approximate='tanh')` 一致），CPU 参考实现用同一个公式对拍：

```cpp
__host__ __device__ float gelu_tanh(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
```

其中 `0.7978845608 = sqrt(2/π)`，与 PyTorch 的 tanh 近似逐项一致。

**为什么 bias 的读取不贵？** 一个 block 里，`bias[c]` 被同一列的所有行复用，且这些地址会被 L1/L2 缓存。ncu 显示 fused 版的 `DRAM Throughput` 7.37%，与纯 GEMM 的 7.33% 几乎完全一致——bias 流量（每 block 128 个 float）淹没在噪声里。

### 对照用的独立 epilogue kernel

为了让「两 kernel」不是稻草人，独立 epilogue 用 **float4 向量化 + grid-stride** 写（[`fusion_epilogue.cu:211`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/14-fusion-epilogue/fusion_epilogue.cu)）：

```cpp
float4 v = reinterpret_cast<float4*>(C)[idx];        // 一次搬 4 个连续列
const float4 b = *reinterpret_cast<const float4*>(&bias[c]);
v.x = apply_epilogue<MODE>(v.x, b.x);  // ... y/z/w
reinterpret_cast<float4*>(C)[idx] = v;
```

单独测这个 kernel：**0.0084 ms，约 3979 GB/s 的有效带宽**（33.6 MB 流量，部分命中 L2）——已经跑得很快了，但再快也得花掉这 8.4 µs。

---

## 4. 实测：融合省下的正是那 8.4 µs

`scripts/run.sh 14-fusion-epilogue/fusion_epilogue.cu`，M=N=K=2048（17.18 GFLOP），原始输出见 `fusion_epilogue.out.txt`：

| 版本 | 耗时 | 算力 | %bf16 峰值 | 寄存器 | smem/block |
|---|---|---|---|---|---|
| `base` | 0.0778 ms | 220.88 TFLOPS | 22.3% | 128 | 37.9 KB |
| `sep` | 0.0874 ms | 196.49 TFLOPS | 19.9% | 128（GEMM） | 37.9 KB |
| `fused` | 0.0786 ms | 218.61 TFLOPS | 22.1% | 128 | 37.9 KB |
| `fused_relu` | 0.0764 ms | 224.78 TFLOPS | 22.7% | 128 | 37.9 KB |
| `epi_only` | 0.0084 ms | 3979 GB/s | — | 32 | 0 |

读法：

- **`fused` 只比 `base` 慢 0.8 µs（约 1%）**。epilogue 在寄存器上多算一条 `fma` + 一个 `tanh`，而 GEMM 本身卡在 L2/Tensor Core 供给上，这点标量运算完全被藏住。
- **`sep - fused = 8.8 µs`，恰好等于 `epi_only` 的 8.4 µs 加一次 kernel launch 开销**。这就是那「多出来的一趟显存往返」的价钱，没有别的。
- **`fused_relu` 甚至比 `base` 还快一点点**。ReLU 只有一条 `max`，比 base 只多了 bias 加法；差异在测量噪声级别，但足以说明 epilogue 本身不是负担。

`nvcc -Xptxas -v` 确认 **三个 GEMM 实例（MODE=0/2/3）寄存器数都是 128、0 spill**（`ptxas_regs.out.txt`）——融合没有增加寄存器压力，occupancy 不变（ncu 实测 23.2%）。

---

## 5. 收益随 K 缩小而放大：算术强度视角

GEMM 的计算量 ∝ `M·N·K`，而 epilogue 的额外访存 ∝ `M·N`（与 K 无关）。所以 **K 越小，epilogue 占比越大**。固定 M=N=2048，扫 K：

| K | FLOPs | `base` | `sep` | `fused` | `fused_relu` | sep/fused 慢多少 |
|---|---|---|---|---|---|---|
| 2048 | 17.18 GFLOP | 0.0778 | 0.0874 | 0.0786 | 0.0764 | **+11%** |
| 512 | 4.29 GFLOP | 0.0336 | 0.0427 | 0.0338 | 0.0319 | **+26%** |
| 256 | 2.15 GFLOP | 0.0260 | 0.0349 | 0.0262 | 0.0247 | **+33%** |

（原始输出 `k512.out.txt` / `k256.out.txt`，单位 ms。）

这条趋势很重要：decode 阶段的小 batch / 小 K 场景里，GEMM 越来越「瘦」，那个固定 8.4 µs 的 epilogue 就从「11%」涨成「33%」。**越是低算术强度的 GEMM，融合 epilogue 越值钱。**

---

## 6. ncu 证据：融合不改瓶颈

`ncu --set full`，取关键 SOL 指标（`ncu_base.out.txt` / `ncu_fused.out.txt` / `ncu_epilogue.out.txt`）：

| 指标 | `base` (MODE=0) | `fused` (MODE=2) | `epilogue_kernel` |
|---|---|---|---|
| Duration | 78.53 µs | 78.91 µs | 11.26 µs |
| Compute (SM) Throughput | 37.08% | 40.41% | 37.38% |
| L2 Cache Throughput | **70.34%** | **68.62%** | 72.69% |
| DRAM Throughput | 7.33% | 7.37% | 44.97% |
| L1/TEX Cache Throughput | 55.48% | 55.07% | 26.53% |
| Registers / thread | 128 | 128 | 32 |
| Achieved Occupancy | 23.26% | 23.18% | 79.03% |

- 融合前后，GEMM 的**瓶颈指标几乎逐项不变**：Compute 微升（37.1%→40.4%），L2 微降，DRAM 持平。这说明 epilogue 只是「搭了段顺风车」，没有引入新的访存或占用。
- `epilogue_kernel` 单独看是典型的 **memory-bound**：DRAM 45%、occupancy 79%、L2 72.7%，`Duration` 11.26 µs（ncu 口径，比 event 口径的 8.4 µs 略长，含 replay 开销）。它本身没有优化空间了——**省掉它才是解法**。
- 融合后 GEMM 的 `L2 Cache Throughput` 68.6% 仍是头号瓶颈，延续了第 13 篇的结论：要继续提升得靠更大分块 / `wgmma` / TMA，而不是 epilogue。

---

## 7. 顺带：split-K 与「生产形态」串讲

**split-K** 是另一个 epilogue 相关的话题。当 K 很大而 M、N 很小时，grid 只有 `M/BM × N/BN` 个 block，喂不满 132 个 SM。split-K 把 K 切段，每个 block 只算一段 K 的部分和，最后再用一个 reduction 把部分和加起来。结构上是：

```
              K
   ┌───────────┬───────────┬───────────┐
   │  block(0) │  block(1) │  block(2) │   ← 每个 block 算一段 K 的部分和
   └─────┬─────┴─────┬─────┴─────┬─────┘
         │ 部分和      │ 部分和      │ 部分和
         └────────► [reduce + epilogue] ◄────────┘
                          │
                          ▼
                        C[M,N]
```

注意这里的 reduction 本身就是一种「epilogue」：如果让它和 bias/激活一起在最后一个 kernel 里完成，就又是一次融合。生产库（CUTLASS 的 `splitk_serial`/`parallel`、cuBLAS 的 `CUBLAS_GEMM_*_SPLITK`）都是这个套路。

把整个系列串起来，一个生产级 GEMM 算子的完整形态是：

| 阶段 | 本篇之前学过的点 | 对应文章 |
|---|---|---|
| 搬 A/B 进 smem | coalescing、cp.async、多级流水 | 04 / 12 |
| 喂 Tensor Core | `ldmatrix`、bank conflict padding | 13 |
| 主循环累加 | 寄存器分块、thread tile、occupancy | 09 / 11 |
| **epilogue** | **bias + 激活 + 可能的 split-K reduce** | **本篇** |
| 落盘 | 合并写、避免 bank/coalescing 问题 | 04 |

---

## 小结

- **epilogue 融合省下的是「GEMM 结果的一次额外读 + 一次额外写」**。2048² FP32 输出就是 33.6 MB 的多余流量，独立 kernel 实测要 8.4 µs（≈3979 GB/s，已接近跑满）。
- **融合几乎免费**：`fused` 相对 `base` 只慢约 1%，三个 GEMM 实例寄存器都是 128、0 spill，ncu 的瓶颈指标逐项不变。
- **收益随算术强度下降而放大**：M=N=2048 时，融合比两 kernel 快 **11%（K=2048）/ 26%（K=512）/ 33%（K=256）**。小 K 场景（decode、小 batch）最值得做。
- **epilogue 可插拔**：一个模板参数就切换 bias-only / bias+GELU / bias+ReLU，正确性由 CPU 同公式参考对拍保证。
- **split-K 的 reduction 是 epilogue 的近亲**，生产库把它和 bias/激活合并成最后一个 kernel，思路一脉相承。
- 最佳融合版 **0.0786 ms / 218.61 TFLOPS（22.1% 峰值）**，在完全保留第 13 篇性能的前提下多算了 bias+激活。

> 下一篇：CUDA 算子调优（十五）· FlashAttention 串讲——把 online softmax（第 07 篇）、融合 epilogue（本篇）、异步流水线（第 12 篇）拼成一个完整的 attention 算子。
