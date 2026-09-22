# fp16 反向实现分析（P1-1 ~ P1-5）

> 代码：
> - **单文件**：`src/fp16/fa_bwd_fp16_onefile.cu`（自包含：preprocess + main kernel + launcher + 自测 main）
> - **两文件**（P1-4）：`src/fp16/fa_bwd_fp16_kernels.cuh`（device：三个 kernel）+ `src/fp16/fa_bwd_fp16_main.cu`（host：npy/launcher/自测）
>
> 实测原始输出：单文件 `fa_bwd_fp16_onefile.out.txt`（S=512）、`..._s4096.out.txt`（S=4096）、
> `..._ncu_main.out.txt`（ncu `--set full`）、`..._refbench.out.txt`（FA/TE 基线）；
> 两文件 `fa_bwd_fp16_main_s512.out.txt`、`..._s4096.out.txt`、`..._ncu_main.out.txt`。
> 对照：`docs/00-fa-bwd-optimization-catalog.md`、`../ROADMAP.md`。

---

## 1. 实现结构（FA2 风格三段式）

| 段 | 函数 | 职责 |
|---|---|---|
| preprocess | `preprocess_kernel` | 逐行求 `LSE=logsumexp(scale·QKᵀ)` 与 `D=rowsum(dO∘O)` |
| main | `fa_bwd_fp16_kernel` | 每个 Q 块固定，遍历 K/V 块（1colblock），recompute S/P，累加 dQ/dK/dV |
| convert | `convert_kernel` | fp32 累加缓冲 `dq/dk/dv` → fp16 写回 |

### 1.1 为什么 preprocess 要重算 LSE
FA 的前向会把 `LSE` 存下来给反向用。本仓库的 dump 里只有 `O/dO`，没有现成的 `LSE`，
所以 preprocess 里用 online-softmax 单趟把每行的 `LSE` 算出来（数学上就等于前向的 `LSE`），
`P = exp(scale·QKᵀ − LSE)` 一步恢复。这与 `ref_impl.py:flash_attn_bwd` 的“`exp(S−LSE)` 恢复 P”一致。
`D` 的预计算让主 kernel 不必同时驻留 `O` 与 `dO`（只多读一行 `delta`），也是 FA2 的做法。

### 1.2 分块与线程映射
- `BM=64`（Q 行/块）、`BN=32`（K/V 行/块）、`THREADS=128`（4 warps）、head_dim=128。
- 页面/行方向的操作（S、P、dP、dS）用「一个 warp 负责 `BM/4=16` 个 Q 行，lane 对应一列」。
- 列方向输出（dV、dK）用「一个 warp 负责 `BN/4=8` 个 K 行，lane 覆盖 4 个 head_dim 槽」。
- dQ 用「warp 负责 16 个 Q 行，lane 覆盖 4 个 head_dim 槽」，跨 K tile 累加在 smem `dQs`。
- causal：整块跳过（`ncols = min(S, m0+BM)`），对角 tile 逐元素 `jg > qi → P=0`。

---

## 2. 优化手段（本版本所用，逐条）

| 手段 | 在本文件的体现 | 作用 |
|---|---|---|
| **recompute P** | 主循环每 tile 用 `Ss` 重算 `QKᵀ`，不物化 N×N 的 P | 省 O(N²) 显存 |
| **预计算 D=rowsum(dO∘O)** | `preprocess_kernel` 的 `sh_delta` 归约 | 主 kernel 不驻留 O/dO |
| **LSE 一步恢复 P** | `P = expf(dot − lse[row])` | 不重跑 online-softmax |
| **smem 复用 Q/K/V/dO** | `Qs/Ks/Vs/dOs` 存 fp16 | 全局访存只读一遍 |
| **causal 模板化（运行期）** | `ncols` 截断 + 逐元素 mask | 省对角以外一半计算 |
| **fp32 累加** | S/P/dS/dQ/dK/dV 全 fp32 | 降误差 |
| **dQ smem 累加（无 atomic）** | 一个 CTA 独占其 Q 行 | 避免 dQ 原子竞争 |
| **dK/dV fp32 全局缓冲 atomicAdd** | 多个 Q 块贡献同一 K/V 行 | 非确定性但实现简单（FA2 非确定模式同款） |
| **动态 smem** | 96KB > 48KB 静态上限，`cudaFuncSetAttribute` | 放得下分块 |

**尚未做**（留给后续优化）：张量核、smem padding 消 bank conflict、多 CTA/SM、
异步拷贝/流水、warp specialization、`dQ_accum` 确定性拆分、GQA/变长。

---

## 3. 数值对拍（vs fp32 ref，O 取 `ref_o.npy`）

容差口径：fp16 ~1e-3。`max_rel = max |a−b|/(|b|+1e-3)`（小量用 1e-3 兜底，否则近零元素会虚高）。

### S=512, B=1, H=16, D=128, causal（case `b1_s512_h16_d128_causal_fp16`）

| | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|
| **ours** | **1.671e-3** | **1.680e-3** | **1.899e-3** |
| FA 2.7.4 | 1.679e-3 | 1.684e-3 | 1.899e-3 |
| TE 2.14 | 1.716e-3 | 2.287e-3 | 1.899e-3 |

### S=4096, B=1, H=16, D=128, causal

| | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|
| **ours** | **1.499e-3** | **1.572e-3** | **2.225e-3** |
| FA 2.7.4 | 1.883e-3 | 1.734e-3 | 1.966e-3 |
| TE 2.14 | 1.883e-3 | 1.858e-3 | 1.966e-3 |

**结论**：我们的 fp16 实现与 ref 的偏差和 FA/TE 处于**同一个量级（fp16 量化噪声）**，
没有系统性误差，正确性达标。

---

## 4. ncu 剖析（main kernel，S=512 causal，`--set full`）

```
DRAM Throughput          %    0.21      <- HBM 几乎空闲
L2  Cache Throughput     %    0.67
L1/TEX Cache Throughput  %   53.53      <- 当前最高
Compute (SM) Throughput  %    8.45
Issue Slots Busy         %    8.45
Executed Ipc Active            0.62
Theoretical Occupancy    %   12.50      <- 受 shared memory 限制
Achieved Occupancy       %    6.25
Waves Per SM                   0.48      <- grid 只有 128 个 block
Registers Per Thread     54
Dynamic Shared Memory    98.30 KB/block
excessive shared wavefronts 141,557,760 (75%)
```

### bound 结论
1. **不是 HBM bound**：DRAM 仅 0.21%，L2 0.67% —— 数据规模太小、且都在 smem/L2。
2. **不是算力 bound**：Compute 8.45%，IPC 0.62。
3. **L1/TEX（shared memory）是当前最高单元（53.5%）**，且 **75% 的 shared wavefronts 是多余的
   （bank conflict）**：Q/K/V/dO 以 fp16 行主序存 smem，lane 连续读 half → 2-way conflict；
   更严重的是 `Ss/Ps` 的按列访问。
4. **occupancy 被 96KB 动态 smem 卡死在 1 CTA/SM（理论 12.5%）**，加上 grid 只有 128 个 block
   （< 132 SM），尾波/并行度都不足。
5. 综合：这是**访存延迟 / 发射受限**的标量实现，瓶颈是「smem 访问 + 低 occupancy」，
   不是带宽也不是算力。后续优化优先级：**消 bank conflict → 降 smem/提 occupancy → 上张量核**。

---

## 5. 性能对标（CUPTI 纯 device 时间，FA/TE 由 `fa_bwd_bench.py bench` 测得）

FLOPs 口径与 harness 一致：`4·B·S²·H·D`（causal 实际约一半，尚未折算）。
H100 峰值：FP16 Tensor Core dense ≈ 989 TFLOPS。

| shape | ours (total) | FA 2.7.4 | TE 2.14 | ours 峰值占比 |
|---|---|---|---|---|
| B1 S512 H16 D128 causal | 3.593 ms / **0.60 TF** | 0.0699 ms / 30.71 TF | 0.0455 ms / 47.23 TF | 0.06% |
| B1 S4096 H16 D128 causal | 138.97 ms / **0.99 TF** | 1.0613 ms / 129.50 TF | 0.5918 ms / 232.22 TF | 0.10% |

分段时间（ours）：

| shape | preprocess | main | convert |
|---|---|---|---|
| S=512 | 1.199 ms | 2.378 ms | 0.016 ms |
| S=4096 | 69.70 ms | 69.03 ms | 0.239 ms |

**结论**：本版本是**正确性优先的标量（CUDA-core）实现**，性能只有 FA 的 ~0.8%、TE 的 ~0.4%。
preprocess 重算 LSE 与 main 同量级开销（FA 的 LSE 来自前向、反向不额外付这笔），
后续若做端到端对标需要把 LSE 并入前向或摊薄。**性能优化是后续任务（见 ROADMAP backlog）。**

---

## 6. 复现命令

```bash
cd code/flash-attention/fa-bwd
# 编译运行 + 数值对拍（默认 case b1_s512_h16_d128_causal_fp16）
scripts/run.sh src/fp16/fa_bwd_fp16_onefile.cu

# S=4096 case（需先 dump）
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python "$PWD/harness/fa_bwd_bench.py" dump \
    --dtype fp16 --shape 1 4096 16 128 causal
scripts/run.sh src/fp16/fa_bwd_fp16_onefile.cu \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp16 --iters=10

# ncu（main kernel）
scripts/ncu.sh src/fp16/fa_bwd_fp16_onefile.cu --set full --launch-count 1 \
    --kernel-name regex:fa_bwd_fp16_kernel -- --iters=1

# FA/TE 基线
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python "$PWD/harness/fa_bwd_bench.py" bench \
    --dtype fp16 --shape 1 4096 16 128 causal
```

## 7. 两文件版（P1-4）

把单文件拆分为 **device 头 + host 源**，保持 kernel 代码**逐字未改**，行为与单文件**逐位一致**：

| 文件 | 内容 |
|---|---|
| `fa_bwd_fp16_kernels.cuh` | 编译期常量（`BM/BN/THREADS/SMEM_BYTES`）、`preprocess_kernel`、`fa_bwd_fp16_kernel`、`convert_kernel` |
| `fa_bwd_fp16_main.cu` | `#include "fa_bwd_fp16_kernels.cuh"` + `CUDA_CHECK`、npy 读取、`diff_stat`、`main`（launcher/计时/对拍） |

**一致性验证**（同一 case，逐项对比单文件 vs 两文件）：

| 指标 | 单文件 | 两文件 |
|---|---|---|
| S=512 dq/dk/dv max_abs | 1.671e-3 / 1.680e-3 / 1.899e-3 | 1.671e-3 / 1.680e-3 / 1.899e-3 |
| S=4096 dq/dk/dv max_abs | 1.499e-3 / 1.572e-3 / 2.225e-3 | 1.499e-3 / 1.572e-3 / 2.225e-3 |
| S=512 total / TFLOPS | 3.5925 ms / 0.60 | 3.5994 ms / 0.60 |
| ncu DRAM / L1TEX / Compute | 0.21% / 53.52% / 8.45% | 0.21% / 53.52% / 8.45% |
| ncu Achieved Occupancy | 6.25% | 6.25% |

> 计时/对拍数字完全一致（±跑机噪声），ncu 逐指标一致；两文件仅改变了代码组织，
> 未触碰 kernel 逻辑。后续 bf16/fp8 的 dtype 参数化会以 `_kernels.cuh` 为模板。

复现：`scripts/run.sh src/fp16/fa_bwd_fp16_main.cu`（编译的是 host 源，`.cuh` 随 `#include` 编入）。

---

## 8. GQA / MQA 支持（P5-1）

生产形状里 Q 头数 `H` 常远大于 KV 头数 `Hkv`（GQA，如 Qwen3），甚至 `Hkv=1`（MQA，如 DSA
indexer）。ff16 反向加一层 Q 头 → KV 头映射即可，**算法/线程映射完全不变**：

- **映射口径**：与 `ref_attn`（`harness/fa_bwd_bench.py`）的 `repeat_interleave` 对齐——
  第 `h` 个 Q 头用 KV 头 `hkv = h / (H / Hkv)`（要求 `H % Hkv == 0`）。
- **改动点（两文件 + 单文件同步，device 代码逐字一致）**：
  1. `preprocess_kernel` / `fa_bwd_fp16_kernel` 增加入参 `int Hkv`，K/V 的行索引用
     `((b*S+j)*Hkv + hkv)*D`（Q/O/dO/LSE/delta 仍按 `H`）；
  2. `dk_acc/dv_acc` 及 `dk/dv` 输出按 `B*S*Hkv*D` 分配，`convert_kernel` 改为接收
     `n_q`（dq）与 `n_kv`（dk/dv）两个长度；
  3. host 从 `k.npy` 的 shape[2] 读出 `Hkv`（默认 MHA 时 `Hkv==H`，与旧行为逐位一致）。
- **`Hkv==H` 时所有索引退化为原式**，MHA 回归数字与 P1 记录完全相同（S=512 1.671/1.680/1.899e-3；
  S=4096 1.499/1.572/2.225e-3），确认无副作用。
- 反向里 dK/dV 本就跨 Q 块 `atomicAdd`，GQA 下多个 Q 头同时累加到同一 KV 头行，语义天然成立
  （非确定性，与 MHA 路径同一口径）。

### 8.1 数值对拍（四个 dump 的 GQA/MQA case，ours-vs-ref，fp16 causal）

| case (B1 S1024 D128 causal) | dq max_abs | dk max_abs | dv max_abs | FA dk/dv | TE dk/dv |
|---|---|---|---|---|---|
| h32 kv4 (Qwen3-30B) | 2.134e-3 | 3.078e-3 | 3.963e-3 | 3.32e-3 / 5.11e-3 | 3.18e-3 / 5.11e-3 |
| h40 kv8 (Qwen3-8B) | 1.580e-3 | 2.380e-3 | 3.999e-3 | 3.00e-3 / 4.33e-3 | 2.89e-3 / 4.33e-3 |
| h64 kv4 (Qwen3-235B) | 1.974e-3 | 5.704e-3 | 4.938e-3 | 4.78e-3 / 5.65e-3 | 3.82e-3 / 5.65e-3 |
| h64 kv1 (DSA MQA) | 1.780e-3 | 7.586e-3 | 7.517e-3 | 7.59e-3 / 1.06e-2 | 6.45e-3 / 1.06e-2 |

**结论**：全部在 fp16 噪声量级，且 **ours 的 dk/dv 与 FA/TE 同量级或更小**，无系统误差。
原始输出：`src/fp16/fa_bwd_fp16_main_p51_gqa.out.txt`（两文件）、
`src/fp16/fa_bwd_fp16_onefile_p51_gqa.out.txt`（单文件，与两文件逐位一致）。

### 8.2 ncu（main kernel，h32 kv4 S1024，`--set full`）

```
Duration                 ms   10.73       L1/TEX Cache Throughput  %  74.96  <- 最高
DRAM Throughput          %    0.07        L2 Cache Throughput      %   0.97
Compute (SM) Throughput  %   15.08       No Eligible              %  78.16
Theoretical Occupancy    %   12.50        Achieved Occupancy       %  11.50
Waves Per SM                  1.94        Registers Per Thread            54
Block Limit Shared Mem   block  2
```

bound 与 MHA 版一致：**smem 访问（fp16 标量读的 bank conflict，L1/TEX 75%）+ 被 96KB 动态 smem
卡住的 1 CTA/SM 低 occupancy**，非 DRAM 也非算力。原始输出
`src/fp16/fa_bwd_fp16_main_p51_ncu_gqa_kv4.out.txt`。

### 8.3 性能对标（CUPTI，FA/TE 由 `fa_bwd_bench.py bench --requested --dtype fp16` 测得）

FLOPs 口径与 harness 一致：`4·B·S·H·S·(D+Dv)`；H100 FP16 峰值 ≈ 989 TFLOPS。

| shape | ours total | FA 2.7.4 | TE 2.14 | ours 峰值占比 |
|---|---|---|---|---|
| h32 kv4 S1024 | 19.32 ms / 0.89 TF | 0.2216 ms / 155.03 TF | 0.1430 ms / 240.25 TF | 0.09% |
| h40 kv8 S1024 | 23.38 ms / 0.92 TF | 0.2608 ms / 164.66 TF | 0.1689 ms / 254.28 TF | 0.09% |
| h64 kv4 S1024 | 34.85 ms / 0.99 TF | 0.3664 ms / 187.54 TF | 0.2460 ms / 279.30 TF | 0.10% |
| h64 kv1 S1024 | 34.91 ms / 0.98 TF | 0.3663 ms / 187.60 TF | 0.2664 ms / 257.96 TF | 0.10% |

**结论**：GQA/MQA 是纯索引改造，性能与 MHA 标量版同量级（~1 TF，峰值 0.1%）；相比 FA 差
~170×、TE 差 ~260×——瓶颈是「标量 CUDA-core + 低 occupancy」，不是 GQA 本身。后续 bf16/fp8 复用
同一改造（P5-3），fp8 走已优化的张量核路径。原始输出
`src/fa_bwd_bench_requested_fp16_p51.out.txt`。

复现：

```bash
cd code/flash-attention/fa-bwd
scripts/run.sh src/fp16/fa_bwd_fp16_main.cu \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h32_d128_kv4_causal_fp16 --iters=20
scripts/run.sh src/fp16/fa_bwd_fp16_onefile.cu \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h64_d128_kv1_causal_fp16 --iters=20
scripts/ncu.sh src/fp16/fa_bwd_fp16_main.cu --set full --kernel-name regex:fa_bwd_fp16_kernel \
    --launch-count 1 -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h32_d128_kv4_causal_fp16 --iters=1
docker exec kernel_lab python "$PWD/harness/fa_bwd_bench.py" bench --requested --dtype fp16 --repeat 30
```

---

## 9. MLA head_dim=512 支持（P5-2）

生产 MLA 的**主注意力** head_dim 远大于 128（这里按 `REQUESTED_SHAPES` 用 `D=512` 建模；
`FA2/FA3` 与 `TE` 的训练反向都只支持到 `head_dim≤256`，所以这三个 case **只有 fp32 ref 可对**，
性能数字也只能由 ours 提供）。把 `fa_bwd_fp16` 的 `head_dim` 从编译期常量改成**模板参数**即可，
算法、数学口径、线程映射全部不变。

### 9.1 改动：`HD` 模板化 + `BM` 随容量选择

- 新增 `template <int HD, int BM> struct BwdTraits`：`WM_ROWS=BM/4`、`WN_ROWS=BN/4`、
  `NCH=HD/32`（lane 覆盖 head_dim 的分段数），以及 `smem_bytes`（随 `HD`、`BM` 编译期求出）。
- `fa_bwd_fp16_kernel` 改为 `template <int HD, int BM>`；`preprocess_kernel` 增加运行时 `HD` 入参；
  三处 `for (int kk = 0; kk < 4; ++kk)` 的 head-dim 分段循环统一改成 `kk < NCH`、`acc[4]→acc[NCH]`。
- **容量约束决定 `BM`**：`dQs[BM*HD]` 是 fp32，占 smem 大头。`HD=512` 时若沿用 `BM=64`，
  仅 `dQs` 就 128KB、总 smem 336KB > 227KB 上限。故：
  - **`HD=128` → `BM=64`**：与 P1/P5-1 的实例**逐字等价**，MHA/GQA 回归逐位不变；
  - **`HD=512` → `BM=16`**：`Qs/dOs` 各 16KB、`Ks/Vs` 各 32KB、`dQs` 32KB、`Ss/Ps` 各 2KB，
    总 **135.17KB**（`BM=16,BN=32`），1 CTA/SM。
- host 用 `launch_bwd_main<HD,BM>` 分派（`D==128` / `D==512`，其余报错），并对每个模板实例
  分别 `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`；单文件与两文件 device 代码同源。

### 9.2 数值对拍（三个 MLA dump case，ours-vs-ref，fp16 causal）

FA/TE 无反向（`fa=NA`、`te=NA`），只能对 fp32 ref：

| case (B1 D=512 causal) | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|
| (1,256,2,512) | 1.638e-3 | 1.582e-3 | 1.753e-3 |
| (1,512,4,512) | 2.324e-3 | 2.916e-3 | 1.724e-3 |
| (1,1024,2,512) | 1.250e-3 | 1.454e-3 | 2.058e-3 |

**结论**：全部在 fp16 噪声量级（~1–3e-3），与 D=128 路径同量级、无系统误差。
单文件与两文件结果**逐位一致**（如 (1,1024,2,512) 均为 1.250/1.454/2.058e-3）。原始输出：
`src/fp16/fa_bwd_fp16_main_p52_mla_{s256,s512h4,s1024h2}.out.txt`、
`src/fp16/fa_bwd_fp16_onefile_p52_mla_*.out.txt`。

MHA `D=128` 回归（两文件/单文件）：S=512 `1.671/1.680/1.899e-3`、S=4096 `1.499/1.572/2.225e-3`，
与 P1 记录**逐位相同**，证明模板化未改动 D=128 路径。

### 9.3 ncu（main kernel，(1,1024,2,512) D=512，`--set full`）

```
Duration                 ms    5.06       L1/TEX Cache Throughput  %  53.21  <- 最高
DRAM Throughput          %    0.10       L2 Cache Throughput      %   1.74
Compute (SM) Throughput  %    7.16       No Eligible              %  85.50
Theoretical Occupancy    %    6.25       Achieved Occupancy       %   6.25
Waves Per SM                  0.97       Registers Per Thread            48
Dynamic Shared Mem/block  Kbyte  135.17  Block Limit Shared Mem   block  1
```

stall 表：`MIO scoreboard ≈36%`（等 smem）、shared load **76.5% 多余 wavefront**（bank conflict）。
bound 与 D=128 标量版完全同类：**smem 访问（bank conflict）+ 被 135KB smem 卡住的 1 CTA/SM 低
occupancy**，DRAM/算力都不是墙。原始输出
`src/fp16/fa_bwd_fp16_main_p52_ncu_main_s1024h2.out.txt`。

### 9.4 性能（CUPTI/CUDA-event，H100 FP16 峰值 ≈989 TFLOPS）

FLOPs 口径 `4·B·S·H·S·(D+Dv)`，`D=Dv=512`。MLA 反向无 FA/TE 基线可比：

| shape | preprocess | main | total | TFLOPS | 峰值占比 |
|---|---|---|---|---|---|
| (1,256,2,512) | 0.211 ms | 1.058 ms | 1.278 ms | 0.21 | 0.02% |
| (1,512,4,512) | 1.291 ms | 2.080 ms | 3.491 ms | 0.62 | 0.06% |
| (1,1024,2,512) | 2.463 ms | 4.143 ms | 6.789 ms | 0.63 | 0.06% |

同进程 GQA/MQA FP16 基线（`fa_bwd_bench.py bench --requested --dtype fp16`）：
FA 155–187 TF、TE 240–278 TF（MLA 三行均 `NA`）。ours 是**标量 CUDA-core + 1 CTA/SM**，
仅为峰值的 ~0.06%，符合预期；后续 fp8 张量核路径（P5-3/P5-4）才是性能目标。原始输出
`src/fa_bwd_bench_requested_fp16_p52.out.txt`。

### 9.5 复现

```bash
cd code/flash-attention/fa-bwd
scripts/run.sh src/fp16/fa_bwd_fp16_main.cu \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp16 --iters=50
scripts/run.sh src/fp16/fa_bwd_fp16_onefile.cu \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h4_d512_causal_fp16 --iters=50
scripts/ncu.sh src/fp16/fa_bwd_fp16_main.cu --set full --kernel-name regex:fa_bwd_fp16_kernel \
    --launch-count 1 -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp16 --iters=1
docker exec kernel_lab python "$PWD/harness/fa_bwd_bench.py" bench --requested --dtype fp16
```

---

## 10. 下一步

见 `../ROADMAP.md`：P1~P4 已完成；P5-1（fp16 GQA/MQA）与 **P5-2（本节，MLA head_dim=512）** 完成；
其后 **P5-3 bf16/fp8 复用 GQA/MLA 改造**、P5-4 fp8 GQA/MQA 对拍；MLA 的性能优化（更小 smem
冲 2 CTA/SM、张量核）与其余消 bank conflict / 提 occupancy 一并列入 backlog。
