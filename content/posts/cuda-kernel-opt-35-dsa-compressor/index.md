---
title: "CUDA 算子调优（三十五）：DSA Compressor —— 把 KV 用「门控池化」压成更少的 key，以及一个被 L2 卡住的合并投影"
date: 2026-09-21
draft: false
weight: 35
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "DeepSeek", "DeepSeek-V4", "DSA", "稀疏注意力", "MLA", "compressor", "KV压缩", "wgmma", "TMA", "warp specialization", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

前面 [第 18 篇]({{< relref "cuda-kernel-opt-18-dsa-indexer-topk" >}}) 和
[第 19 篇]({{< relref "cuda-kernel-opt-19-dsa-sparse-attn" >}}) 把 DeepSeek-V4 的 DSA 稀疏注意力
拆成了「lightning indexer 打分 + exact top-k + 稀疏 MLA 消费端」三段。但 DSA 还有一块拼图我们一直
没碰：**indexer 和稀疏 attention 要消费的那些「压缩 KV」是怎么来的？** 答案就是这篇的主角 ——
`Compressor`，一个把连续 `compress_ratio` 个 token 的 KV 用**学习到的门控 softmax 池化**压成一个
key 的算子。

这一篇不猜、不简化：直接读 DeepSeek-V4 官方推理实现
`/ssd/models/DeepSeek-V4-Pro/inference/model.py`（`class Compressor`，`model.py:279`，`forward`
在 `model.py:316`），把 prefill 语义逐行还原，自己写两个 kernel（合并投影 + 融合池化）跑出来，
并用 `/ssd/models/DeepSeek-V4-Pro/config.json` 的真实 shape 实测。

结论先放这儿：

- 整个 compressor 拆成 **一个合并投影 GEMM**（把 `wkv` 和 `wgate` 拼成一个 `[2C, D]` 的权重，A 只读一遍）
  **+ 一个融合池化 kernel**（online softmax + RMSNorm + RoPE 三合一）；端到端相对官方 eager 参考 **2.0~2.4×**。
- 投影 GEMM 自研 **650~663 TFLOPS（bf16 峰值的 65~67%）**，是同 shape cuBLAS 合并 bf16 GEMM（803~816）的 **~80%**。
- 池化 kernel 是纯访存，实测 **83~84% HBM**（ncu `DRAM Throughput` 87%）。
- 一个反直觉的发现：**这个 GEMM 不是算力受限，而是 L2 带宽受限**（ncu `L2 Cache Throughput` 85~88%、
  `Compute` 只有 57~59%）。原因是权重 `[2C,D]` 只有 29~58 MB、却要被 `M/BM` 个 m-tile 重读，A 又被
  `2C/BN` 个 n-tile 重读——L2 请求率成了墙。

> 编号说明：原计划的第 33 篇（MLA 极限冲刺三 · producer warp 专门化）因 Hopper 上的 smem/寄存器硬阻塞
> 暂缓（见路线图「阻塞」）；本篇先补上第五部分 DSA 的最后一块（主题 17 剩余）。

## 一、Compressor 到底在算什么

### 1.1 从稀疏注意力的需求说起

DeepSeek-V4 每一层的 attention 有三种 KV：

1. 滑动窗口里的原始 KV（`window_size=128`）；
2. **压缩后的 KV**：每层有一个 `compress_ratio`（V4-Pro 的 `compress_ratios` 在 `128` 和 `4` 之间交替，
   末层为 `0` 表示不压缩，见 `config.json`）。compressor 把每 `ratio` 个 token 的 latent KV 平均成一个；
3. indexer 自己那份压缩 KV（用 `GetIndexer` 的 `Compressor`，`ratio=4`、`head_dim=128`、带 Hadamard 旋转）。

稀疏 attention 只在前 `index_topk=1024` 个压缩 KV 上算——所以 **compressor 的压缩质量/速度直接决定
DSA 能省多少**。这篇只做 attention 主路径的那个 compressor（`head_dim=512`、`coff = 1 + (ratio==4)`）。

### 1.2 参考实现的 prefill 语义（逐行还原）

```python
# /ssd/models/DeepSeek-V4-Pro/inference/model.py:323-342（start_pos == 0，prefill）
kv    = self.wkv(x)            # [S, C]，C = coff * head_dim
score = self.wgate(x)          # [S, C]
kv    = kv.unflatten(1, (-1, ratio))        # [S/r, r, C]
score = score.unflatten(1, (-1, ratio)) + self.ape   # ape: [r, C]，加在窗口内每个位置上
if ratio == 4:                             # overlap=True，用「重叠窗」让边界更平滑
    kv    = self.overlap_transform(kv, 0)
    score = self.overlap_transform(score, -inf)
out = (kv * score.softmax(dim=2)).sum(dim=2)          # [S/r, d]
out = self.norm(out)                                  # RMSNorm(d=512)
apply_rotary_emb(out[..., -64:], freqs)               # 末 64 维 RoPE
```

把它画成图（`ratio=4`、overlap 版本）：

```
token:    x0  x1  x2  x3   x4  x5  x6  x7   x8  x9  x10 x11
          └──── 窗 t-1 ───┘  └──── 窗 t ──────┘  └──── 窗 t+1 ────┘
proj:     kv[0..3] | kv[4..7] | kv[8..11] ...
          ┌ 前半 dim (0..d-1) 来自上一窗
compressed[t] = Σ over 8 slots  softmax(score + ape) ⊙ kv      （overlap=2r 个 slot）
          └ 后半 dim (d..2d-1) 来自本窗

non-overlap（ratio=128，coff=1）：
compressed[t] = Σ_{i=0..127} softmax_i(score[t*128+i] + ape[i]) ⊙ kv[t*128+i]
```

两个关键点：

- **softmax 是在「窗口内 token 维」上做的，且每个输出列 `j∈[0,d)` 各做一次**（列与列之间完全独立）。
  这意味着池化可以逐列并行，天然适合一个线程管一列。
- **投影 `wkv`/`wgate` 只差一个权重矩阵，输入 A 相同**——所以可以把它们合并成一次
  `X[M, D] @ Wm[2C, D]^T`，A 只读一遍。

### 1.3 Roofline：谁是大头？

以 `M=32768`、`D=7168`、`head_dim=512` 为例：

| ratio | coff | C | N=2C | 投影 FLOPs | 中间张量 Y 字节 | 池化读 Y 字节 |
|---|---|---|---|---|---|---|
| 128 | 1 | 512 | 1024 | 481 GFLOP | 134 MB | 134 MB |
| 4 | 2 | 1024 | 2048 | 962 GFLOP | 268 MB | 268 MB |

`AI = 2·M·N·D / (M·N·4) = D/2 = 3584 FLOP/byte`，远高于 bf16 TC 的 ridge（~295），
**是算力受限**。池化读 134 MB 只要 ~40 µs，投影要 ~0.6~1.5 ms——**投影 GEMM 就是全部**。
所以优化的重心是：把合并投影 GEMM 做快，池化顺手融掉后面三小步。

## 二、实现

代码：`code/kernel-opt/35-dsa-compressor/compressor.cu`。两个 kernel：

### 2.1 `proj_ws_kernel`：合并投影 + TMA + warp specialization

复用 23/28 篇的 Hopper 全家桶：TMA `cp.async.bulk.tensor.2d`（`SWIZZLE_128B`）+ mbarrier 生产者/消费者
+ `wgmma.m64n128k16` SS（K-major SW128 描述符）。bf16 的 SW128 atom 是 8 行 × 64 元素（128B），
所以 `BK` 锁 64。权重 `Wm[2C, D]` 的 D 连续、天然 K-major，B 不用转置。

唯一 compressor 专属的点：**把 `wkv`/`wgate` 拼成一个权重**，GEMM 输出 `Y[M, N]`，其中
`Y[:, :C]=kv`、`Y[:, C:]=score`。这样 A 只被 TMA 搬一遍（对比 eager 里两次 `F.linear` 搬两遍），
而且一个 kernel 就把两个投影都算了。

扫配置（`M=32768`，单独进程各跑一次避免 DVFS 串扰）：

| config (BM×BN×BK, stages) | ratio=128 TFLOPS | ratio=4 TFLOPS |
|---|---|---|
| 128×128×64 s2 | 540.3 | 538.2 |
| 128×128×64 s3 | 529.3 | 528.2 |
| 128×256×64 s3 | 616.6 | 622.9 |
| 256×128×64 s2 | 569.4 | 573.2 |
| 256×128×64 s3 | 593.3 | 599.2 |
| **256×128×64 s4** | **649.8（65.7%）** | **655.2（66.3%）** |

规律和 28/29 篇一致：`BM=256` 把 A 的复用翻倍、`s4`（1 CTA/SM）用更深的流水把 L2 延迟藏住，赢过
`s2/s3` 的高 occupancy。`192×256` 那种几何会触发 ptxas `C7511`（wgmma 因寄存器不足被串行化），
直接崩到 ~147 TFLOPS，所以要避开。

### 2.2 `pool_kernel`：online softmax + RMSNorm + RoPE 三合一

一个 block = 一个输出窗，`HD=512` 个线程，**线程 j 负责输出第 j 列**。每列独立做一次
「对窗口内 slot 的 softmax 加权」：

```
for each slot (kv_v, sc_v):        # sc_v 已加 ape；overlap 时首窗的前半是 -inf 被跳过
    mn = max(m, sc_v);  a = exp(m-mn);  p = exp(sc_v-mn)
    l  = l*a + p
    acc= acc*a + p*kv_v
    m  = mn
out = acc / l
```

线上访存：score 与 kv 在 Y 里的偏移是 `C+j` 和 `j`，线程 j 连续 → 每次读一个 warp 覆盖连续 128B，
完全合并。`ratio=128` 每列读 128 个 slot（256 次 load），`ratio=4` 每列读 8 个 slot。

之后**同一个 kernel 里**接着做：

- RMSNorm（窗内 512 列）：warp shuffle 归约 + `__shared__ float sred[16]` 二级归约；
- RoPE（末 64 维）：经 `srow[512]` smem 交换成对元素，用 YaRN 预计算的 cos/sin（`compress_rope_theta=160000`、
  `factor=16`、`beta_fast/slow=32/1`、`original_max_position=65536`）旋转；
- 写回 bf16。

对比 eager：参考实现里池化是 `softmax` + `mul` + `sum` + `RMSNorm` + `apply_rotary` **五个独立 kernel、
外加物化 `[S/r, r, 2C]` 的中张量**；我们一个 kernel、零中张量。

## 三、正确性

`compressor.cu` 里带一份 **CPU 全流程参考**（用同一份 bf16 权重、fp64 累加，含 YaRN RoPE），
在 `M=2·ratio`（2 个输出窗）上与 GPU 结果对拍：

```
ratio=128: correctness (M=256): max_abs_err=7.619e-03 (ref~3.093) OK
ratio=4  : correctness (M=8):   max_abs_err=7.794e-03 (ref~3.398) OK
```

相对误差 ~0.2%，来自 bf16 输入 + `__expf`。两种模式（含 overlap 的 8-slot 语义）都对上了。

## 四、实测

`comp_ratio{128,4}_M{8192,16384,32768}.out.txt`（H100 SXM，bf16 峰值 989 TFLOPS）。

### 4.1 端到端 vs 官方 eager 参考

eager = `x.float()` + 两次 fp32 `linear` + 上面的池化/Norm/RoPE（PyTorch，RoPE 在 GPU 上）。
`torch_ref.out.txt`。

| ratio | M | eager fp32 | 两次 fp32 matmul | cuBLAS 合并 bf16 | **自研 proj** | 自研 pool | **自研 e2e** | e2e 加速 |
|---|---|---|---|---|---|---|---|---|
| 128 | 8192  | 0.468 | 0.307 | 0.150 (803) | 0.181 (663) | 0.014 | **0.195** | **2.39×** |
| 128 | 16384 | 0.856 | 0.614 | 0.299 (806) | 0.372 (647) | 0.030 | **0.402** | **2.13×** |
| 128 | 32768 | 1.577 | 1.262 | 0.595 (809) | 0.742 (648) | 0.048 | **0.790** | **2.00×** |
| 4 | 8192  | 0.845 | 0.583 | 0.296 (812) | 0.367 (656) | 0.028 | **0.395** | **2.14×** |
| 4 | 16384 | 1.635 | 1.160 | 0.589 (816) | 0.737 (653) | 0.051 | **0.787** | **2.08×** |
| 4 | 32768 | 3.220 | 2.603 | 1.185 (814) | 1.480 (650) | 0.096 | **1.577** | **2.04×** |

单位 ms；括号内为 TFLOPS。拆开看（`M=32768`）：

- **投影**：eager 的两次 fp32 GEMM 1.262 ms → bf16 合并 0.742 ms（**1.70×**）。其中合并本身也有收益：
  cuBLAS 两次独立 bf16 GEMM 0.630 ms vs 合并 0.595 ms（ratio=128，**5.5%**；ratio=4 更是 1.398→1.185，**15%**）——
  A 只读一遍。
- **池化+Norm+RoPE**：eager ≈ 1.577−1.262 = 0.315 ms → 融合 0.048 ms（**~6.6×**）。

### 4.2 对标 SOTA：同 shape cuBLAS 合并 bf16

自研投影是同 shape cuBLAS 合并 bf16（803~816 TFLOPS）的 **~80%**；端到端（cuBLAS 投影 + 我们的池化）
相比我们全自研是 0.595+0.048=0.643 vs 0.790 → 我们达 **81%**（ratio=128, M=32768）。
考虑到后面看到的 L2 墙，这个差距主要来自 cuBLAS 更激进的 L2 调度策略。

### 4.3 ncu：投影被 L2 卡住，池化贴 HBM

`ncu_proj_256x128s4.out.txt` / `ncu_proj_r4_256x128s4.out.txt` / `ncu_pool_r*.out.txt`：

| kernel | L2 | Compute(SM) | DRAM | occupancy | No Eligible |
|---|---|---|---|---|---|
| proj ratio=128 M=32768 | **85.3%** | 58.8% | 26.3% | 27.8% | 85.6% |
| proj ratio=4 M=32768 | **87.7%** | 56.5% | 28.9% | 27.9% | 86.3% |
| pool ratio=128 M=32768 | 79.1% | 27.0% | **84.0%** | 47.8% | 71.3% |
| pool ratio=4 M=32768 | 83.0% | 40.0% | **87.2%** | 80.5% | 58.3% |

投影的 `L2 Cache Throughput` 高达 85~88%，而 `DRAM` 只有 26~29%、`Compute` 只有 57~59%——
**它是 L2 带宽受限，不是算力受限**。算一笔 L2 流量的账（`M=32768, N=1024, K=7168, BM=256, BN=128`）：

$$
\text{L2 读} \;=\; \underbrace{\frac{N}{BN}\cdot M K}_{\text{A 被 }N/BN{=}8\text{ 个 n-tile 重读}}
       + \underbrace{\frac{M}{BM}\cdot N K}_{\text{B 被 }M/BM{=}128\text{ 个 m-tile 重读}}
       \;\propto\; 8\cdot 470\text{MB} + 128\cdot 29\text{MB} \approx 7.5\text{ GB}
$$

在 H100 的 L2 带宽（~7~10 TB/s）下正好是几百 µs 量级，与 742 µs 的耗时吻合。**B 的重读（128 次）
是 A（8 次）的 4 倍**——这正是 25/29 篇反复出现的「小 N GEMM 的 B 重读墙」。

## 五、负结果：把中间张量 Y 降成 bf16

既然投影写 134 MB、池化又读 134 MB，把 Y 改成 bf16 不是能省一半字节吗？实测（`comp_*.out.txt` 末行）：

| ratio | M | fp32-Y e2e | **bf16-Y e2e** | pool 单看 |
|---|---|---|---|---|
| 128 | 32768 | 0.790 | 0.787 | 0.048 → 0.038 |
| 4 | 16384 | 0.787 | **0.772** | 0.051 → 0.038 |
| 4 | 32768 | 1.577 | 1.580 | 0.096 → 0.074 |

**池化确实快了（读减半），但投影变慢了**——`wgmma` 累加器从 fp32 转 bf16 再存，转换指令与
更窄的 store 把省下的字节又吃了回去；而且 GEMM 本来就是 L2/算力受限，**输出字节根本不是瓶颈**。
净效果是「打平、略负」。教训：**融合/降精度的收益只在「被优化的那一级真的是瓶颈」时才兑现**——
和 [32 篇]({{< relref "cuda-kernel-opt-32-fused-norm" >}})「FP8 量化融合不划算」是同一个道理。

## 六、小结

- **Compressor 是 DSA 的最后一块拼图**：门控池化把每 `ratio` 个 token 的 KV 压成一个，
  `ratio=4` 用重叠窗（8 个 slot）让边界平滑。官方 prefill 语义已逐行还原。
- 工程上拆成 **合并投影 GEMM + 融合池化 kernel**：A 只读一遍，池化一步到位（省掉 5 个 kernel 与
  一个 `[S/r, r, 2C]` 中张量）。端到端 **2.0~2.4×**，池化环节 **~6.6×**。
- 投影自研 **650~663 TFLOPS（65~67% 峰值）**，cuBLAS 合并 bf16 的 **~80%**；
  ncu 判决它是 **L2 带宽受限**（L2 85~88%、Compute 57~59%），根因是权重 B 被 `M/BM=128` 个 m-tile 重读。
- 池化是纯访存，**83~84% HBM**。
- 负结果：把中间张量降成 bf16 打平略负——瓶颈在 L2/算力，不在输出字节。

**下一步的方向**（见路线图）：① 用 TMA cluster multicast 沿 M 广播权重 B，把 128 次重读砍下去
（但要先过 26/29 篇「`lts__throughput` 不等于字节」那一关）；② indexer 的 compressor
（`ratio=4, head_dim=128, rotate=True` 的 Hadamard + FP4 量化）还没做；③ 把 compressor 与
18/19 篇的 indexer/top-k/稀疏 MLA 串成真正的 DSA 端到端。

## 附：复现

```bash
cd ~/proj/tech_record/code/kernel-opt
scripts/lab.sh up
# 正确性 + 性能（ratio, M, which）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/run.sh 35-dsa-compressor/compressor.cu 128 32768 all
# 配置扫描（单 config）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/run.sh 35-dsa-compressor/compressor.cu 128 32768 256x128s4
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/ncu.sh 35-dsa-compressor/compressor.cu --set full --kernel-name regex:proj_ws -- 128 32768 256x128s4 1
# eager / cuBLAS 对照
python3 35-dsa-compressor/compressor_ref.py 128 32768
```
