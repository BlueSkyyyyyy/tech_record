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

## 10. fp16 张量核反向（O5，main 11.8–14.9×）

前面第 1–9 节的 fp16 反向都是**正确性优先的标量 golden**（CUDA-core FFMA），
用来把数学/对拍/接口先跑通；真正的性能杠杆是**张量核**。本节（ROADMAP 的 **O5**）把 5 个
GEMM 全部换成 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` + `ldmatrix`，
与 fp8 张量核版（P3-4，`docs/03`）同构，但**没有 rowwise scale**（fp16 无量化）。

- 前置：`src/fp16/fa_bwd_fp16_mma_smoke.cu` 已用最小 GEMM 验证三种操作数布局
  （A=[M][K]+`ldmatrix.x4`、B=[N][K]+`ldmatrix.x2`、B=[K][N]+`ldmatrix.x2.trans`）与 CPU 参考一致（PASS）。
- 交付：单文件 `src/fp16/fa_bwd_fp16_mma_onefile.cu` + 两文件
  `fa_bwd_fp16_mma_kernels.cuh` / `fa_bwd_fp16_mma_main.cu`。本版只做 **head_dim=128**（MHA/GQA）；
  MLA(HD=512) 的张量核留 backlog（标量 MLA 见 §9）。

### 10.1 改动：5 个 GEMM 的 mma 布局（HD=128, BM=64, BN=32，4 warp 2×2）

| GEMM | 输出 | A 操作数 | B 操作数（布局） |
|---|---|---|---|
| ① S=scale·QKᵀ | [BM][BN] | Qs[BM][HD] | Ks[BN][HD]（`[N][K]`，`ldmatrix.x2`）|
| ② dP=dO·Vᵀ | [BM][BN] | dOs[BM][HD] | Vs[BN][HD]（`[N][K]`）|
| ③ dV=Pᵀ·dO | [BN][HD] | PsT[BN][BM]（P 转置） | dOs[BM][HD]（`[K][N]`，`ldmatrix.x2.trans`）|
| ④ dK=scale·dSᵀ·Q | [BN][HD] | dSsT[BN][BM]（dS 转置） | Qs[BM][HD]（`[K][N]`）|
| ⑤ dQ=scale·dS·K | [BM][HD] | dSs[BM][BN] | Ks[BN][HD]（`[K][N]`）|

要点与逐位等价性：

- **P/dS 就地转 half**：① 的 epilogue 把 `P=exp(S−LSE)` 写成 `PsT[c][r]`（转置，供 ③ 的 A），
  同时用寄存器 `pval[2][2][4]` 把 fp32 P 带到 ②；② 用同一 (r,c) 映射算 `dS=P∘(dP−D)`，
  写 `dSs[r][c]`（供 ⑤）与 `dSsT[c][r]`（供 ④）。这与 FA2「P/dS 转 half 进张量核、D/LSE 用 fp32」一致。
- **dQ 寄存器累加**：本版没有 split-K（grid=`S/BM×H×B`），每个 Q 块由**唯一 CTA** 独占，
  故 ⑤ 的累加器 `dqacc[2][8][4]` 沿 nt 在寄存器里累加、循环后**直接写** `dq_acc`（无需跨 CTA atomic），
  等价于 fp8 的 O7。
- **dK/dV 跨 CTA 归约**：③④ 仍用 `atomicAdd(float2*)`（O4c 口径，相邻两列打包）。
- **smem**：`Qs/dOs/Ks/Vs`（half）+ `PsT/dSs/dSsT`（half），行距分别 `HD+8`/`BM+8`/`BN+8`
  （+8 个 half = +16B 消 `ldmatrix` bank conflict）；**共 66.56KB**，168 寄存器、0 spill，
  **3 CTA/SM**（theoretical occ 18.75%）。不存 `[BM][HD]` 的 fp32 dQ 累加器，省 32KB。

### 10.2 数值对拍（ours-vs-ref / FA / TE，fp16 causal）

MHA 与 GQA/MQA（S512/S4096、S1024 各 KV 头数）**全部与 ref/FA/TE 同量级（~1e-3）**，无系统误差：

| case (B1 causal) | dq max_abs | dk max_abs | dv max_abs | FA dk | TE dk |
|---|---|---|---|---|---|
| (1,512,16,128) MHA | 1.671e-3 | 1.771e-3 | 1.899e-3 | 1.684e-3 | 2.287e-3 |
| (1,4096,16,128) MHA | 1.883e-3 | 1.734e-3 | 1.966e-3 | 1.734e-3 | 1.858e-3 |
| (1,1024,32,128) kv4 | 2.134e-3 | 3.305e-3 | 3.850e-3 | 3.321e-3 | 3.175e-3 |
| (1,1024,40,128) kv8 | 2.008e-3 | 2.931e-3 | 3.891e-3 | 2.996e-3 | 2.893e-3 |
| (1,1024,64,128) kv4 | 2.348e-3 | 5.704e-3 | 3.893e-3 | 4.778e-3 | 3.818e-3 |
| (1,1024,64,128) kv1 MQA | 2.292e-3 | 7.934e-3 | 7.517e-3 | 7.586e-3 | 6.452e-3 |

单文件与两文件**逐指标一致**（168 regs、0 spill，数值逐位相同）。原始输出
`src/fp16/fa_bwd_fp16_mma_onefile_o5_*.out.txt`、`src/fp16/fa_bwd_fp16_mma_main_o5_sweep.out.txt`。

### 10.3 性能（CUDA-event；FLOPs 口径 `4·B·S·H·S·(D+Dv)`，峰值 989 TFLOPS）

同 session A/B（标量 golden vs 本节张量核，只比 main）：

| shape | scalar main | mma main | 加速 | mma main TFLOPS | 占峰值 |
|---|---|---|---|---|---|
| (1,512,16,128) | 2.279 ms | **0.192 ms** | **11.8×** | 22.4 | 2.3% |
| (1,4096,16,128) | 67.55 ms | **4.555 ms** | **14.9×** | 60.4 | 6.1% |

端到端（preprocess + main + convert）：

| shape | ours total | preprocess | main | total TFLOPS (占峰值) |
|---|---|---|---|---|
| (1,512,16,128) | 1.440 ms | 1.209 ms | 0.195 ms | 2.98 (0.30%) |
| (1,1024,32,128) kv4 | 9.459 ms | 8.802 ms | 0.634 ms | 3.63 (0.37%) |
| (1,1024,40,128) kv8 | 11.85 ms | 10.97 ms | 0.868 ms | 3.63 (0.37%) |
| (1,1024,64,128) kv4 | 18.74 ms | 17.62 ms | 1.113 ms | 3.67 (0.37%) |
| (1,1024,64,128) kv1 | 18.57 ms | 17.46 ms | 1.015 ms | 3.70 (0.37%) |
| (1,4096,16,128) | 73.34 ms | 68.70 ms | 4.555 ms | 3.75 (0.38%) |

同 session **纯反向** FA3/TE/FA2 基线（`harness/fa_vs_te_bwd_only.py fp16`，本机 FA3 3.0.0 SM90）：
GQA/MQA S1024 为 FA3 356–438 TF、TE 306–354 TF、FA2 218–259 TF；MHA S4096 为 FA3 848 / TE 624 / FA2 378 TF。
即 **ours main 约 FA3 的 7–16%、端到端约 0.8–1.7%**。

> **关键结论**：张量核把 **main 提高 12–15×**，但**端到端只提高 ~1.8×**——因为 fp16 的
> **preprocess（LSE+D）仍是标量 O(S²)**，S=4096 时 68.7ms > main 4.56ms，占比 94%。
> 下一步优先级：**O8（preprocess 的 mma 分块 LSE，对齐 fp8 的 O1）** 才是端到端最大杠杆。

### 10.4 ncu 剖析（main kernel，`--set full`）

S=4096（1,4096,16,128），3 CTA/SM：

```
Duration                 ms    4.55       DRAM Throughput        %   1.41
L1/TEX Cache Throughput  %   33.32       L2 Cache Throughput    %  24.86
Compute (SM) Throughput  %   17.44       Issued Ipc Active  inst/cyc 0.91
Theoretical Occupancy    %   18.75       Achieved Occupancy     %  16.58
Waves Per SM                  2.59       Registers Per Thread         168
Dynamic Shared Mem/block Kbyte 66.56     Block Limit Shared Mem block   3
```

stall：**`long_scoreboard`（等 L1TEX/全局读）占平均 11.64 cycle 的 63.4%**
（`No Eligible` 77.2%）。S=512（grid=128<132 SM、`Waves 0.32`、`Achieved occ 6.25%`）是纯 grid-bound。

**bound = 全局访存延迟**（K/V/dO/Q 每 tile 同步 global→smem，无 `cp.async`/预取覆盖），
DRAM/算力/L2 都很闲。下一步 **O3/O6（寄存器预取或 `cp.async` 双缓冲）** 可把 K/V 载入藏进 5 个 GEMM。
原始输出 `src/fp16/fa_bwd_fp16_mma_onefile_o5_ncu_main_{s512,s4096}.out.txt`。

---

## 11. preprocess 的 mma 分块 LSE（O8，端到端 4.4–13.2×）

§10 把 main 换成张量核后（12–15×），端到端却被**标量 preprocess（LSE + D）**拖住：
它是 O(S²) 的逐行 QK 点积（每 (s,h) 行一个 block、128 线程标量扫 K），S=4096 时 68.7ms
≈ main 的 15×，占端到端 94%。O8 照搬 fp8 的 O1：把 LSE 交给张量核、D 拆成独立 kernel。

### 11.1 改动

- **`lse_mma_kernel<HD>`**（替代旧 `preprocess_kernel` 的 LSE 部分）：
  grid=`(S/LBM × H × B)`，每 CTA 吃 **LBM=64 行 Q**、沿 N 一次吃 **LBN=64 列 K**；4 个 warp
  各 16 行（`wm=wid`），`mma.m16n8k16` 算 `QKᵀ`（f16×f16→fp32，与 main GEMM1 同一
  `mma_block_f16` 封装、同一 k-loop 分块顺序 ⇒ 与 main 的 P 自洽）。**P 不物化**：
  fp32 累加器直接做 online-softmax（每个线程 2 个 row-slot），行 max/sum 在同 row 的
  4 个 lane（`lane&3`）间 `shfl_xor` 归约，仅 `lane&3==0` 写 LSE。causal mask 与
  boundary（`qi<S/jg<S`）同旧版。smem = `(LBM+LBN)·(HD+8)` 个 half = **34816 B**。
- **`delta_kernel<HD>`**（D=`rowsum(dO∘O)`）：纯 O(S·H·D) 的逐行归约（`pg=(S,H,B)`），
  与 LSE 解耦。fp16 无量化，直接 `__half2float` 乘加（同旧版）。
- 单/两文件同步：`fa_bwd_fp16_mma_{kernels.cuh,main.cu}` 与 `fa_bwd_fp16_mma_onefile.cu`
  的 device O8 代码块**逐字一致**（脚本核对 `device O8 block identical: True`）。
  旧 `preprocess_kernel` 被删除（标量 golden `fa_bwd_fp16_{kernels.cuh,onefile.cu}` 保留）。

### 11.2 数值（与 O5 **逐位相同**，证明只换算法数据流、未改数学口径）

| case (B1 causal) | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|
| (1,512,16,128) MHA | 1.671e-3 | 1.771e-3 | 1.899e-3 |
| (1,4096,16,128) MHA | 1.883e-3 | 1.734e-3 | 1.966e-3 |
| (1,1024,32,128) kv4 | 2.134e-3 | 3.305e-3 | 3.850e-3 |
| (1,1024,40,128) kv8 | 2.008e-3 | 2.931e-3 | 3.891e-3 |
| (1,1024,64,128) kv4 | 2.348e-3 | 5.704e-3 | 3.893e-3 |
| (1,1024,64,128) kv1 MQA | 2.292e-3 | 7.934e-3 | 7.517e-3 |

与 §10.2 的 O5 表**逐位相同**，且与 ref/FA/TE 同量级。原始输出
`src/fp16/fa_bwd_fp16_mma_main_o8_*.out.txt`、`..._onefile_o8_s512.out.txt`。

### 11.3 性能（CUDA-event，同 session）

| shape | preprocess 旧→新 | 加速 | total 旧→新 | total 加速 | total TFLOPS (占 989) |
|---|---|---|---|---|---|
| (1,512,16,128) | 1.209→**0.071 ms** | **17.0×** | 1.440→**0.326 ms** | **4.42×** | 6.59 (0.67%) |
| (1,1024,32,128) kv4 | 8.802→**0.186 ms** | **47.4×** | 9.459→**0.871 ms** | **10.9×** | 19.72 (1.99%) |
| (1,1024,40,128) kv8 | 10.97→**0.213 ms** | **51.5×** | 11.85→**1.106 ms** | **10.7×** | 19.41 (1.96%) |
| (1,1024,64,128) kv4 | 17.62→**0.305 ms** | **57.7×** | 18.74→**1.467 ms** | **12.8×** | 23.42 (2.37%) |
| (1,1024,64,128) kv1 | 17.46→**0.307 ms** | **56.9×** | 18.57→**1.415 ms** | **13.1×** | 24.28 (2.46%) |
| (1,4096,16,128) | 68.70→**0.986 ms** | **69.7×** | 73.34→**5.581 ms** | **13.1×** | 24.63 (2.49%) |

同 session **纯反向** FA3/TE/FA2 基线（`harness/fa_vs_te_bwd_only.py fp16`）：
S4096 MHA FA3 **0.3246ms/847TF**、TE 0.4441/619、FA2 0.7284/377；GQA kv4 S1024 FA3
0.0824/417、TE 0.1124/306。即 **ours total 现为 FA3 的 2.9–4.7%（按 TFLOPS），
时间比 10.6–17.2×**（O5 时端到端只有 FA3 的 ~0.4%）。S512 MHA（单测）FA3 0.0265ms/162TF、
TE 0.032/134：ours total 0.326ms，**时间比 12.3×**（TF 4.1%）。

> **结论**：O8 把端到端从「preprocess 主导」翻转为「**main 主导**」：
> S=4096 preprocess 68.7→0.99ms，每 CTA 只剩 LBM=64 行 Q 的 QKᵀ；total 13.1×。
> 剩余空间在 **main（S4096 4.47ms，占 80%）**、S512 时 **convert（0.064ms）** 已和
> preprocess 同量级。

### 11.4 ncu（`--set full`，S=4096）

`lse_mma_kernel<128>`（grid 64×16×1）：

```
Duration                 ms    1.03       DRAM Throughput        %   1.06
L1/TEX Cache Throughput  %   23.42       L2 Cache Throughput    %   5.23
Compute (SM) Throughput  %   39.53       No Eligible            %  42.88
Theoretical Occupancy    %   37.50       Achieved Occupancy     %  28.49
Waves Per SM                  1.29       Registers Per Thread         80
Block Limit Shared Mem   block     6
```

stall（同 session 定向采集）：`long_scoreboard 2.17` + `wait 1.34` + `short_scoreboard 0.76`
+ `barrier 0.21` ⇒ **bound = 全局访存延迟 + 低 occupancy**（80 regs 把理论 occ 卡在 37.5%，
且 grid=1024 只有 1.29 波、尾波明显），与 fp8 O1 的结论一致；DRAM/L2/张量核都很闲。

`delta_kernel<128>`：Duration **42.8µs**、DRAM 24.8% / L1TEX 73.5% / Compute 71.9% /
Achieved occ 71.9%（17 regs）、Waves 31.0 ⇒ 访存/算力均衡的轻量归约，仅占端到端 <1%。
原始输出 `src/fp16/fa_bwd_fp16_mma_main_o8_ncu_lse_s4096.out.txt`、
`..._o8_stall_lse_s4096.out.txt`、`..._o8_ncu_delta_s4096.out.txt`。

---

## 12. O6：main 的 `cp.async` 双缓冲（消 63% `long_scoreboard`，main 2.26–2.41×）

### 12.1 动机

O5（§10）把 fp16 main 换成 `mma.m16n8k16` 后，ncu 结论是 **`long_scoreboard` 占
11.64 cycle 的 63.4%**——即每个 tile 先把 K/V 从全局**同步**读进 smem
（`for i = tid; i < BN*HD; i += THREADS` 标量 `LDG`），`__syncthreads` 后才能开算，
这段全局访存延迟完全暴露。O8（§11）把 preprocess 从 68.7ms 打到 0.99ms 后，
端到端瓶颈落回 main（S=4096 main 4.47ms，占 80%）——**O6 就是 main 的下一刀**。

### 12.2 改动（单/两文件 device 代码逐字一致）

1. **`cp.async.cg` + 双缓冲 K/V**：新增 `cp_async16`（`cp.async.cg.shared.global …
   ,16`，走 L2-only，不占寄存器）与 `kv_issue_async<HD,BN>`——把一个 K/V 列块按
   **8 个 half（16B）为最小单位**分成 `BN*HD/8 = 512` 个 unit，128 线程每线程 4 个，
   行越界（`jg>=S`）改用普通 smem 写 0（`cp.async` 无谓词）；发完 `commit_group`。
2. **两阶段流水**：`Ks/Vs` 各扩成 2 份（`KVSB=2*KVL`），`stage = nt & 1`。
   prologue 先发 tile0；循环里 `cp.async.wait_group 0` → `__syncthreads`（等本 tile
   落地，并保证上一 tile 的 GEMM5 已读完那个 stage）→ 发下一 tile 进另一 stage →
   做 5 个 GEMM。**下一 tile 的全局读延迟被本轮 5 个 GEMM（+ 1 次 mid barrier）覆盖**。
3. **顺带省掉 1 次 barrier/tile**：原版每 tile 有 3 个 `__syncthreads`
   （载入后 / GEMM1-2 后 / 收尾），双缓冲后收尾 barrier 由「下一轮循环首的 barrier」
   承担（写的又是另一 stage），降到 **2 个/tile**（对齐 fp8 的 O4a 思路）。
4. 用模板参数 `PIPE`（`false`=原版、`true`=O6）在同一 kernel 里共存，便于**同 session
   A/B**；`__launch_bounds__(THREADS, PIPE?2:3)`。

**代价**：smem `66.56KB → 83.97KB`（K/V 双缓冲 +17.4KB）⇒ 从 **3 CTA/SM 降到 2 CTA/SM**
（182 regs，S=4096 block limit shared=2）。fp16 张量核只有 128 线程/CTA，2 CTA=8 warp/SM；
但 O6 的目的正是「不靠堆 warp、靠异步拷贝把访存延迟从 dependency scoreboard 里拿出去」。

### 12.3 实测（同 session A/B，CUDA event，main-only）

| shape | nopipe (ms / TF) | **O6 pipe (ms / TF)** | 加速 |
|---|---|---|---|
| MHA S=512 | 0.1912 / 11.23 | **0.0847 / 25.34** | **2.26×** |
| MHA S=4096 | 4.5097 / 30.48 | **1.8730 / 73.38** | **2.41×** |
| GQA q32/kv4 S=1024 | 0.6349 / 27.06 | **0.3749 / 45.82** | **1.69×** |

端到端（preprocess + main + convert，CUDA event）：S=512 **0.1868ms（11.50 TF）**、
S=4096 **3.043ms（45.16 TF）**、GQA kv4 **0.597ms（28.78 TF）**。
S=4096 的 `preprocess` 现在是 1.007ms、`main` 1.873ms、`convert` 0.098ms——
**main 仍占 62%**，但已从 4.47ms 的绝对墙降到 1.87ms。
单文件 `fa_bwd_fp16_mma_onefile.cu` 与两文件逐指标相同（S512 pipe 0.0846ms、
三个 max_abs 逐位一致）。

### 12.4 数值（与 O5/O8**逐位相同**）

| shape | dq | dk | dv | 对拍 |
|---|---|---|---|---|
| MHA S=512 | 1.671e-3 | 1.771e-3 | 1.899e-3 | 与 O5 记录逐位一致 |
| MHA S=4096 | 1.883e-3 | 1.734e-3 | 1.966e-3 | 与 O5 记录逐位一致 |
| GQA q32/kv4 S=1024 | 2.134e-3 | 3.305e-3 | 3.850e-3 | 与 O5 记录逐位一致 |

O6 只改**搬运动作**、不改数学与 GEMM 顺序 ⇒ 结果 bitwise 不变（这既是正确性锚点，
也说明 A/B 的差异纯来自性能）。

### 12.5 ncu（main, S=4096, PIPE）

```
Duration                 ms    1.95       DRAM Throughput        %   3.31
L1/TEX Cache Throughput  %   65.19       L2 Cache Throughput    %  60.11
Compute (SM) Throughput  %   27.07       Registers Per Thread        182
Theoretical Occupancy    %   12.50       Achieved Occupancy     %  11.83
Waves Per SM                  3.88       Block Limit Shared Mem  block  2
```

stall（定向采集，per issue active）：**`wait 1.95`（fixed-latency 依赖）+ `long_scoreboard
1.12` + `short_scoreboard 0.79` + not_selected 0.17 + barrier 0.10 + mio 0.12**。

**对照 O5**：`long_scoreboard 7.35 → 1.12`（**全局访存延迟被 cp.async 流水吃掉**），
Duration 4.55→1.95ms，L1/TEX 33.3→65.2%、L2 24.9→60.1%（张量核/搬运变密）、
Compute 17.4→27.1%、occ 18.75%(3 CTA)→11.8%(2 CTA)。**新墙 = fixed-latency 依赖
（`wait`）+ `short_scoreboard`（smem→`ldmatrix` 依赖）+ L1/TEX 65%**，不再是全局访存延迟；
barrier 已接近 0，说明每 tile 2 个 barrier 不再是问题。

### 12.6 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

FA3（SM90, 3.0.0）MHA S4096 **0.3253ms / 845 TF**、TE2.14 0.4441/619、FA2.7.4 0.7339/375。
ours total 3.043ms / 45.2 TF ⇒ **FA3 的 5.4%（O8 时 2.9–4.7%）、时间比 9.3×**
（O8 时 10.6–17.2×）；main-only 73.4 TF 是 FA3 的 8.7%。
端到端剩余空间：main 62%（已 latency-hidden，需 O9 wgmma/TMA 或降 smem 回 3 CTA）、
preprocess 33%（LSE 尾波/occupancy，O8b）。

### 12.7 原始输出

`src/fp16/fa_bwd_fp16_mma_main_o6_{s512,s4096,gqa_kv4}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o6_s512.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o6_ncu_s4096.out.txt`、
`..._o6_stall_s4096.out.txt`、`src/fp16/fa_bwd_fp16_o6_fa3_te_baseline.out.txt`。

## 12b. O6b：K/V 双缓冲压回 3 CTA/SM + `ldmatrix.x4.trans` 消转置副本（main 同 session +3–6%）

### 12b.1 动机

O6（§12）把全局访存延迟打掉后，ncu 新墙 = `wait`(fixed-latency) + `short_scoreboard`
（smem→`ldmatrix`）+ L1/TEX，且**代价是 smem 83.97KB → 只能 2 CTA/SM**（理论 occ 12.5%）。
要再进一步，得在不牺牲流水的前提下**把 smem 降回 3 CTA/SM 预算（≤ 77.8KB）**，同时减少
smem 访存量（L1/TEX 已 65%）。

### 12b.2 两个改动（单/两文件 device 代码逐字一致）

1. **A 操作数改 `ldmatrix.x4.trans`，删掉 `PsT/dSsT` 两份转置副本**。
   GEMM3/GEMM4 需要 A=`Pᵀ/dSᵀ`，原来把 P/dS（天然按 `[BM][BN]` 算出）**额外复制一份
   转置**存成 `[BN][BM]`。新增 smoke `fa_bwd_fp16_atrans_smoke.cu` 证明：把 A 存成
   `[K][M]` 行主序、用 `ldmatrix.x4.trans`（地址的 bit3/bit4 互换）读到的 A 片段与
   `A[M][K]`+`ldmatrix.x4` **逐位一致**（`bitwise_diff=0/128`，PASS）。于是 P/dS 各只存
   一份 `[BM][BN]`（`LDS=BN+8`），**免掉 `PsT`、`dSsT` 的写与读**。`mma_block_f16` 加
   `ATRANS` 模板参数。
2. **只双缓冲 K，V 单缓冲且在 GEMM2 之后预取**。V 只在 GEMM2 被读、GEMM2 在 tile 前段：
   `cp.async` 的 K 仍走双缓冲（循环首预取），V 发进**同一个单缓冲**——发起点放在
   GEMM2 的消费者 barrier 之后，延迟由随后的 GEMM3/4/5 盖住。省下一整个 V 缓冲。

**smem**：`83.97KB → 71.17KB`（Q/dO 2×17.4 + K 双 2×8.5 + V 单 8.5 + Ps/dSs 2×5.1），
Block Limit Shared Mem **2 → 3**，理论 occ 12.5%→**18.75%**（168 regs，0 spill）。

### 12b.3 自动档（O6b 不是无脑更快）

O6b 的 V 预取窗口比 O6 短：**网格受限**（S=512 grid=128 < 132 SM、单波）时 O6 反而更快，
大网格时 O6b 赢。host 按网格大小自动选：`grid = (S/64)×H×B ≥ 396`（3 CTA/SM 的一个满波）
用 **O6b(2)**，否则用 **O6(1)**；`--nopipe/--pipe/--pipe2` 可强制。

### 12b.4 实测（同 session A/B，CUDA event，main-only）

| shape | nopipe (ms) | **O6 (ms)** | **O6b (ms)** | O6/O6b |
|---|---|---|---|---|
| MHA S=512 (grid=128, 自动选 O6) | 0.1883 | **0.0832** | 0.0866 | 0.96× |
| MHA S=4096 (grid=1024) | 4.4936 | 1.9000 | **1.8597** | 1.02× |
| GQA q32/kv4 S=1024 (grid=512) | 0.6373 | 0.3802 | **0.3571** | 1.06× |

端到端（preprocess+main+convert）：S=512 **0.1847ms（11.63 TF，走 O6）**、
S=4096 **2.9517ms（46.56 TF，走 O6b）**、GQA kv4 **0.5720ms（30.04 TF，走 O6b）**。
单文件与两文件**逐指标相同**（S512 0.0842、S4096 1.8635、GQA 0.3535 main；max_abs 逐位一致）。

### 12b.5 数值（与 O5/O8/O6 **逐位相同**）

| shape | dq | dk | dv |
|---|---|---|---|
| MHA S=512 | 1.671e-3 | 1.771e-3 | 1.899e-3 |
| MHA S=4096 | 1.883e-3 | 1.734e-3 | 1.966e-3 |
| GQA q32/kv4 S=1024 | 2.134e-3 | 3.305e-3 | 3.850e-3 |

A 转置读只是换个读取方式（smoke 已证逐位）、K/V 只是换个流水时点 ⇒ 数学与 GEMM 顺序不变。

### 12b.6 ncu（main, S=4096；`--set full` + 定向 stall）

```
              Duration  smem/block  occ(achieved)  L1/TEX  L2    Compute  DRAM  BlockLimitSMem
O6  (PIPE=1)   1.95ms     83.97KB      11.83%       65.2%  60.1%  27.1%    3.31%   2
O6b (PIPE=2)   1.86ms     71.17KB      16.90%       71.9%  63.1%  29.8%    3.47%   3
```

stall（per issue active）：O6b `wait 1.88 + long_scoreboard 1.79 + short 0.81 +
not_selected 0.37 + mio 0.28 + barrier 0.17`；O6 `wait 1.94 + long 1.09 + short 0.78 +
not_selected 0.17 + barrier 0.10`。**结论**：O6b 把 occupancy 从 11.8% 提到 16.9%、
Duration 降 ~5%，但 V 预取晚 ⇒ `long_scoreboard` 1.09→1.79、warps 多 ⇒ `not_selected`
0.17→0.37；墙仍是 **L1/TEX 72% + L2 63% + fixed-latency(`wait`)**，属于**吞吐受限**，
加 warp 收益有限。真正下一刀是**降 L1/L2 流量**（O7 去 atomic / O4b 式配对）或 **O9 wgmma/TMA**。

### 12b.7 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

FA3（SM90）MHA S4096 **0.3248ms / 846 TF**、TE 0.4452/617、FA2 0.7279/378；
GQA q32/kv4 S1024 FA3 **0.0823ms / 417 TF**、TE 0.1126/305。
ours（O6b）total S4096 2.952ms/46.6 TF ⇒ **FA3 的 5.5%（时间比 9.1×）**；
GQA kv4 total 0.572ms/30.0 TF ⇒ **FA3 的 7.2%（时间比 6.9×）**。

### 12b.8 原始输出

`src/fp16/fa_bwd_fp16_atrans_smoke.out.txt`、`src/fp16/fa_bwd_fp16_mma_main_o6b_{s512,s4096,gqa_kv4}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o6b_{s512,s4096}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o6b_ncu_main_s4096.out.txt`、`..._o6b_stall_s4096.out.txt`；
`src/fa_bwd_o6b_fa3_te_baseline.out.txt`。

---

## 13. O8b：LSE 预处理负载均衡 + `cp.async` 双缓冲（fp16，端到端 1.27×）

### 13.1 动机：preprocess 变成端到端 34% 的第一瓶颈

O6/O6b（§12/§12b）把 main 从 4.5ms 打到 1.86ms 后，**preprocess 仍停在 ~1.0ms**，
占端到端 `2.95ms` 的 **34%**。对 `lse_mma_kernel`（S=4096, causal）做定向 ncu：

- Duration **1.03ms**、Compute 39.5%、L1/TEX 23.4%、L2 5.2%、DRAM **1.06%**；
- stall `long_scoreboard 2.19 + wait 1.34`（**延迟受限**，每 tile 同步标量读 K→sync→mma）；
- grid=(64,16,1)、Waves **1.29**，ncu 报「尾波可达 **50%**」——因为因果下第 `mblk` 个 CTA
  要做 `mblk+1` 个 K tile（0→63 线性增长），而 GPU 按 blockIdx 递增把重块全排在最后。

两处改动（用模板 `PIPE` 分开以便消融）：

### 13.2 改动（单/两文件 device 代码逐字一致）

1. **镜像配对负载均衡**。每 CTA 同时处理**配对的两个 m 块** `m = blockIdx.x` 与
   `m' = nblk-1-m`：工作量 = `(m+1)+(nblk-m) = nblk+1` 个 tile（**完美均衡**），
   grid.x 从 `nblk` 减半到 `ceil(nblk/2)`（S=4096 时 64→32，整块 grid 512）。
   两个 m 块共用同一个 `Qs` 缓冲（逐块加载），`nvcc` 展开不变。
2. **K 的 `cp.async.cg` 16B 双缓冲**（`PIPE=1`）。每 tile 的 K 从「同步标量读 8 个 half
   →`__syncthreads`→mma」改成：prologue 发 tile0，循环首 `wait_group 0`+`sync` 后**把下一
   tile 发进另一 stage**，再做 `QKᵀ`+online-softmax；复用 main 的 `cp_async16`。`PIPE=0`
   退回同步标量读，用于同 session 消融。softmax 的 mask 由 `!(causal&&jg>qi)` 写成
   `jg<=qi`（该 kernel 仅 causal 使用），其余 online-softmax / 4-lane `shfl_xor` 归约逐字不变。

smem：`Qs[LBM*LD] + 2×Ks[LBN*LD]`（PIPE=1）= **52.2KB**（PIPE=0 与 O8 同为 34.8KB）。
host：非 causal 各块工作量相同、无需配对，仍走 O8 原版；causal 走 `lse_mma_kernel_bal<128,1>`。

### 13.3 消融（同 session A/B，CUDA event，lse-only）

| shape | O8 原版 | 镜像配对(单缓冲) | **镜像配对+cp.async(双缓冲)** |
|---|---|---|---|
| MHA S=4096 | 0.9853ms | 0.4457ms（2.20×） | **0.3484ms（2.81×）** |
| MHA S=512 | 0.0633ms | 0.0560ms（1.13×） | **0.0458ms（1.38×）** |
| GQA q32/kv4 S=1024 | 0.1620ms | 0.1014ms（1.60×） | **0.0774ms（2.09×）** |

⇒ **主收益来自镜像配对（负载均衡）**，`cp.async` 双缓冲再叠加 **1.28×**（S=4096）；
S=512 网格小、本来就接近单波，收益只有 1.38×。

### 13.4 性能（同 session，端到端 preprocess+main+convert）

| shape | preprocess O8→O8b | total O6b→O8b | TFLOPS |
|---|---|---|---|
| MHA S=512 | 0.0536→**0.0481ms** | 0.1603→**0.1583ms** | 13.6 |
| MHA S=4096 | 1.0094→**0.3870ms（2.6×）** | 2.9517→**2.3364ms（1.26×）** | **58.8** |
| GQA q32/kv4 S=1024 | 0.1040→**0.0892ms** | 0.5720→**0.5183ms** | 33.4 |

端到端瓶颈重新回落 **main**（S=4096：main 1.84ms 占 79%、preprocess 0.39ms 占 17%）。
单文件与两文件**逐指标相同**（S=4096 total 2.3322 vs 2.3364，main 1.8435 vs 1.8372）。

### 13.5 数值（与 O5/O8/O6/O6b **逐位相同**）

| shape | dq | dk | dv |
|---|---|---|---|
| MHA S=512 | 1.671e-3 | 1.771e-3 | 1.899e-3 |
| MHA S=4096 | 1.883e-3 | 1.734e-3 | 1.966e-3 |
| GQA q32/kv4 S=1024 | 2.134e-3 | 3.305e-3 | 3.850e-3 |

镜像配对只改「哪个 CTA 做哪块」、`cp.async` 只改「何时搬 K」，QKᵀ 的 k-loop 顺序与
online-softmax 的归约顺序完全不变 ⇒ LSE 与 main 的 P 仍自洽，数值逐位不变。

### 13.6 ncu（`lse_mma_kernel_bal<128,1>`，S=4096）

```
              Duration  grid      waves  occ(achieved)  Compute  L1/TEX  L2    DRAM
O8  原版       1.03ms  (64,16,1)  1.29    28.49%        39.5%   23.4%   5.2%  1.06%
O8b bal+async  354µs   (32,16,1)  0.97    22.89%        60.4%   32.4%  27.0%  3.07%
```

stall（per issue active）：O8b `wait 1.57 + short_scoreboard 1.33 + not_selected 0.79 +
long_scoreboard 0.34`（O8：`long 2.19 + wait 1.34`）。**结论**：`long_scoreboard` 2.19→0.34
（全局读延迟被 `cp.async` 消掉），Waves 1.29→**0.97**（尾波消除），吞吐从延迟受限变为
**Compute 60%（张量核+softmax）+ smem 依赖**受限。occ 22.9%（52.2KB smem → 4 CTA/SM）。

### 13.7 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

FA3（SM90）MHA S4096 **0.3241ms / 848 TF**、TE 0.4454/617、FA2 0.7267/378；
GQA q32/kv4 S1024 FA3 **0.0826ms / 416 TF**、TE 0.1121/307。
ours（O8b）total S4096 **2.336ms / 58.8 TF ⇒ FA3 的 6.9%（时间比 7.2×）**，
O6b 时为 5.5%（时间比 9.1×）；GQA kv4 total 0.518ms/33.4 TF ⇒ **FA3 的 8.0%**（O6b 7.2%）。

### 13.8 原始输出

`src/fp16/fa_bwd_fp16_mma_main_o8b_{s512,s4096,gqa_kv4}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o8b_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o8b_ncu_lse_bal_s4096.out.txt`、
`..._o8b_stall_lse_bal_s4096.out.txt`、`src/fp16/fa_bwd_fp16_o8b_fa3_te_baseline.out.txt`。

---

## 13b. O6c：主 kernel 的 tile 几何参数化 + 小网格并行度自适应（fp16，main 最多 1.13×）

### 13b.1 动机与一次重要的「负结果」

O6b 的 ncu 显示 `fa_bwd_fp16_mma_kernel`（S=4096）有 **Waves 2.59、尾波最多占 33%**，
且 SM active cycles 最大比均值高 22.8%、最小低 26.5%。据此先做了一个「不改数学、只重排
`blockIdx.x → mblk` 双射」的负载均衡（`sched=1` 交错 `0,n-1,1,n-2,...` / `sched=2` 逆序）：
因果下第 `m` 个 Q 块要算 `m+1` 个 K/V tile，把轻/重块混进每个波。**实测只有 0~2%、且不同
session 正负号不稳定**（`[O6c A/B]` 行）——说明 GPU 的 block 调度本就是动态的，静态重排
拿不到收益。该开关保留（默认 `sched=0`）但不作为优化。

真正的杠杆来自另一个观察：**小网格时 kernel 是「每 SM 一个 CTA」的**。S=512/H16 时
`grid=(512/64)·16=128 < 132 SM`，每个 SM 只有 1 个 CTA（4 warp）；此时把 `BM` 从 64 减半到
32 会让 `grid` 翻倍到 256（每 SM 2 个 CTA），并行度直接翻倍。代价是 dK/dV 的跨 CTA 原子量
随 `BM` 反比翻倍，所以只在 `S` 较小（S≤1024）时才划算。

### 13b.2 改动（单/两文件 device 代码**逐字一致**）

1. **tile 几何参数化**：`fa_bwd_fp16_mma_kernel<HD,BM,BN,PIPE>` 原本把 `mma_block_f16`
   的 `WARP_M/N` 和所有 `16/32/64` 字面量写死为 `BM=64,BN=32`。改成由 `(BM,BN)` 派生的
   2×2 warp 网格几何：
   ```
   GM1/ GN1 = BM/2, BN/2   // GEMM1/2（S / dP）每 warp 的 M×N
   GMV/GNV  = BN/2, HD/2   // GEMM3/4（dV/dK）→ 2 warp 铺 BN、2 warp 铺 HD
   GMQ/GNQ  = BM/2, HD/2   // GEMM5（dQ）
   MT*      = (WARP_M/16, WARP_N/8)  // 每个 warp 的 m16/n8 tile 数
   ```
   `pval/dqacc/acc` 全部按 `MT*` 定义；`static_assert` 钉死整除关系。`HD=128` 下
   `BM=64,BN=32` 展开后与 O6b **完全同构**（因此 `BM=64,BN=32` 回归逐位不变）。
2. **host 自动档**（`launch_cfg` + CLI `--bm/--bn/--pipe` 覆盖）：
   - `grid < 132 且 S ≤ 1024` ⇒ `(BM=32, BN=32, PIPE=1)`（提并行度）；
   - 否则 `BM=64`，`grid ≥ 396` 用 `PIPE=2`（O6b），小网格用 `PIPE=1`（O6）；
   - `S ≥ 4096` 用 `BN=64`（见下）。
3. **`BN=64` 配置**（大 S）：`S=4096` 时 `BN=64` 把 Q/dO 的 `ldmatrix` 复用翻倍，
   L1/TEX 从 71.9%→57.3%，但 smem 105KB 使占用从 3→2 CTA/SM；净 +1.5~2%。
4. 单文件 `fa_bwd_fp16_mma_onefile.cu` 由两文件 device 段 + host 段重新拼接
   （脚本 `diff` 核对 `device region identical: True`）。

### 13b.3 配置 A/B（同 session，CUDA event，main-only，单位 ms）

| shape (grid) | (64,32,2) | (64,64,2) | (32,32,1) | (32,32,2) | best/原 |
|---|---|---|---|---|---|
| S=512 H16 (128)  | 0.0858 | 0.0814 | **0.0776** | 0.0800 | **1.106×** |
| S=1024 H32 kv4 (512) | **0.3576** | 0.3849 | 0.4562 | 0.4527 | 1.000× |
| S=1024 H40 kv8 (640) | **0.4305** | 0.4307 | 0.5319 | 0.5322 | 1.000× |
| S=4096 H16 (1024) | 1.8535 | **1.8220** | 2.6352 | 2.6255 | 1.017× |

自动档选中的配置：S=512⇒(32,32,1)、S=1024⇒(64,32,2)、S=4096⇒(64,64,2)。**端到端**：
S=512 `preprocess+main+convert` 0.1598→**0.1504ms（1.06×）**、S=4096 2.3472→**2.3495ms
（持平，main 1.02× 被 session 噪声吃掉）**；GQA/MQA 形状 grid 已 ≥512，配置不变。

### 13b.4 数值（与 O5/O8/O6/O6b/O8b **逐位相同**）

`BM=64,BN=32` 路径展开后与 O6b 同构；`BM=32`/`BN=64` 只改 tile 划分、不改数学口径。
S=512 dq/dk/dv max_abs = 1.671/1.771/1.899e-3；S=4096 = 1.883/1.734/1.966e-3；
S=1024 kv4 = 2.134/3.305/3.850e-3；kv8 = 2.008/2.931/3.891e-3（与 §10–§13 记录一致）。
单/两文件逐指标一致。

### 13b.5 ncu（main）

- **S=512**：原 `(64,32,1)` Duration **99.3µs** / achieved occ **6.24%** / L1TEX 26.3% / L2 24.3%；
  新 `(32,32,1)` Duration **83.9µs（−15.5%）** / achieved occ **10.99%** / L1TEX 43.5% / L2 47.1%。
  机制 = **并行度**：grid 128→256，每 SM 1→2 个 CTA，把 SM 从「单 CTA 等延迟」里救出来。
- **S=4096 `BN=64`**：Duration 1.87ms、**L1/TEX 71.87%→57.27%**、L2 62.65%、Compute 26.9%、
  achieved occ 18.75%→**12.5%（2 CTA/SM，smem 105KB）**。**结论：`BN=64` 确实降了 L1/TEX
  访存压力（Q/dO 复用翻倍），但被 occupancy 掉档抵消，净收益只有 ~1.5%；要同时拿到两者
  必须先把 smem 压到 `≤77.7KB`（BN=64 需砍 ~28KB），留待 O9。**

### 13b.6 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

S=4096 MHA：FA2 0.7291ms/377TF、**FA3 0.3240ms/848TF**、TE 0.4440/619。ours total
2.3495ms/58.5TF ⇒ **FA3 的 6.9%（时间 7.25×）**（O8b 6.9%，基本持平；本项主要赢在小 shape）。
S=512 MHA：FA2 0.0442/49、**FA3 0.0265/81**、TE 0.0321/67；ours total 0.1504/14.3 ⇒ FA3 的
17.6%（时间 5.7×），比 O8b 的 12.3× 明显拉近。

### 13b.7 原始输出

`src/fp16/fa_bwd_fp16_mma_main_o6c_{s512_h16,s4096_h16,s1024_h32_kv4,s1024_h40_kv8}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o6c_{s512_h16,s4096_h16}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o6c_ncu_{s512_main,s512_main_bm64,s4096_main}.out.txt`、
`src/fp16/fa_bwd_o6c_fa3_te_baseline.out.txt`、`src/fp16/fa_bwd_o6c_fa3_te_s512.out.txt`。

---

## 14. O7c：LSE/D 预装寄存器 + dK/dV float4 归约试错（fp16，main 1.14–1.19×）

O6c 之后 main 的墙是 `wait` + `long/short_scoreboard` + L1/TEX，`No Eligible` 66–72%、
occupancy 只有 11.8%（`BN=64`）/16.9%（`BN=32`）。本节沿两条独立杠杆各做了一个「同 session
A/B」实验：**① dK/dV 归约宽度 float2→float4（负结果）**；**② LSE/D 预装寄存器（正结果）**。
实现都放进 `fa_bwd_fp16_mma_kernel` 的两个模板开关 `R4`（默认 false，float2）与
`PREL`（默认 true，预装），host 用 `--r4=` / `--prel=` 及一个 2×2 A/B 段挑选。

### 14.1 负结果：dK/dV 归约从 float2 提升到 float4 反而更慢

`mma.m16n8` 的累加器里，一个 quad（`lane&3=0..3`）的 `c2=(lane&3)*2` 正好是同 row 的连续
8 列。于是用两次 `__shfl_down_sync(...,1)` 把「本 lane 的 float2」和「下一 lane 的 float2」
拼成两个 `float4`，让列 0–3 由 lane0、列 4–7 由 lane2 各发一次 `red.global.add.v4.f32`
（地址天然 16B 对齐）。SASS 证实确实生成了 `REDG.E.ADD.F32x4`（相对 float2 的
`F32x2` 事务数减半），但**四个几何全部变慢（−1% ~ −7%）**：

| 几何（main-only，ms） | base（float2） | float4 | 变化 |
|---|---|---|---|
| (64,32,2) S=4096 | 1.7990 | 1.9317 | **−6.9%** |
| (64,64,2) S=4096 | 1.8464 | 1.8954 | −2.6% |
| (32,32,1) S=512 | 0.0750 | 0.0802 | −6.4% |
| (64,32,2) S=512 | 0.0852 | 0.0914 | −6.8% |

**结论**：此配置下 dK/dV 归约**不是事务数 bound**——`red.global` 的 RMW 已经在 L1/L2 内部
高效合并，`float4` 省下的事务被 `shfl` 指令 + 「只有一半 lane 发 store」的并行度损失吃掉。
所以 O7 的「去原子/reduce 事务数」在这条路上收益有限；真正的墙是 **occupancy/延迟**
（见 §14.4），需靠 O9（wgmma+TMA，降 smem/提并行）而非抠归约宽度。

### 14.2 正结果：LSE/D 预装寄存器（PREL）

ncu 的 `Memory Workload Analysis` 报「global load 平均每 thread 只用 4.4/32B」。来源是
**每个 K tile 的 GEMM1/GEMM2 epilogue 都按 `qi` 去 global 读 `lse[qi]` 与 `delta[qi]`**——
而这两个量**只依赖 CTA 自己的 Q 行（`qi=m0+r`），与 K tile 完全无关**，本可只读一次。

改法：在 `nt` 循环前，把本线程需要的 `MTM1×2` 个 row 的 `lse`/`delta` 一次性装进寄存器
（`lse_r[MTM1][2]`、`del_r[MTM1][2]`，行号 `r=wr*GM1+i*16+g+(s?8:0)` 与 epilogue 的
`(i, q>=2)` 一一对应，越界写 0），循环内 epilogue 改读寄存器、零 global 读。数学**逐位等价**。

同 session A/B（CUDA event，main-only）：

| 几何 | base（f2，无 PREL） | f2+PREL | 提升 |
|---|---|---|---|
| (64,64,2) S=4096 | 1.8464 ms / 74.4 TF | **1.5576 ms / 88.2 TF** | **+18.5%** |
| (64,32,2) S=4096 | 1.7990 / 76.4 | **1.6081 / 85.5** | +11.9% |
| (64,32,2) S=512 | 0.0852 / 25.2 | **0.0747 / 28.7** | +13.9% |
| (64,64,2) S=512 | 0.0809 / 26.5 | **0.0698 / 30.8** | +16.0% |
| (32,32,1) S=512 | 0.0750 / 28.6 | 0.0714 / 30.1 | +5.0% |
| GQA kv4 S=1024 (64,32,2) | 0.3627 / 47.4 | **0.3044 / 56.4** | +19.1% |
| GQA kv4 S=1024 (64,64,2) | 0.3801 / 45.2 | **0.3219 / 53.4** | +18.1% |

（`(32,32,*)` 只 +4–5%：小 S 是 grid-bound，省下来的 LDG 不是主因。）

端到端（自动档 `(BN=64,PIPE=2)`，S=4096）：`main 1.8819→1.6029 ms`、total
`2.3762→2.0935 ms`（约 1.13×）；S=512：`main 0.0796→0.0763`、total `0.1504→0.1495 ms`。

### 14.3 单/两文件与数值

单文件 `fa_bwd_fp16_mma_onefile.cu` 与两文件 `fa_bwd_fp16_mma_kernels.cuh` 的 device 段
**逐字一致**（脚本核对 `identical: True`）。数值与 O5/O8/O6/O6b/O8b/O6c **逐位相同**
（PREL 只换读寄存器、float4 默认关）：

| shape | dq | dk | dv |
|---|---|---|---|
| S=512 MHA | 1.671e-3 | 1.771e-3 | 1.899e-3 |
| S=4096 MHA | 1.883e-3 | 1.734e-3 | 1.966e-3 |
| S=1024 GQA kv4 | 2.134e-3 | 3.305e-3 | 3.850e-3 |

### 14.4 ncu（main，S=4096，(64,64,2)）——墙从 L1/TEX 移到 L2

| 指标 | O6c | O7c(PREL) |
|---|---|---|
| Duration | 1.99 ms | **1.61 ms** |
| L1/TEX | 57.5% | 55.7% |
| **L2** | 58.7% | **71.6%（新墙）** |
| Compute (SM) | 27.0% | 23.4% |
| DRAM | 3.2% | 4.0% |
| occupancy（theor/achieved） | 12.5/11.8% | 12.5/11.8% |
| regs / smem | 242 / 105.47KB（2 CTA/SM） | 250 / 105.47KB |
| stall | long 1.79 / wait 1.88 / short 0.81 / mio 0.28 | long **1.38** / wait **2.00** / short 0.88 / mio 0.49 |

`long_scoreboard` 1.79→1.38（消掉每 tile 的 LSE/D 全局读），墙前移到 **L2 71.6%**（残余
dK/dV 的 `red` + Q/dO 重读）。**但 float4 实验已证明「减 red 事务数」不划算**，且 occupancy
被 105KB smem / 250 regs 锁死在 2 CTA/SM——**要同时拿低 L1/L2 与高 occupancy 必须靠 O9
（TMA+wgmma 的更低 smem 数据通路）**。

### 14.5 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

S=4096 MHA：FA2 0.7289ms/377TF、**FA3 0.3235ms/850TF**、TE 0.4438/619。ours total
2.0935ms（时间 **6.47×** FA3；真反向 FLOPs 口径 131 TF ≈ FA3 的 15.4%）。S=512 MHA：
FA3 0.0265ms/81TF，ours total 0.1495ms（时间 5.64×）。GQA kv4 S=1024：FA3 0.0828ms/415TF，
ours total 0.4344ms（5.25×）。

> 口径提醒：kernel 内打印的 TFLOPS 用 `4·B·S²·H·D`，而反向真 FLOPs 是
> `4·B·S²·H·(D+Dv)`；当 `Dv=D` 时打印值恰是真值的一半（`harness/` 用的是真值口径）。

### 14.6 原始输出

`src/fp16/fa_bwd_fp16_mma_main_o7c_{s512,s4096,gqa_kv4}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o7c_{s512,s4096}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o7c_ncu_s4096.out.txt`、
`..._o7c_stall_s4096.out.txt`、`src/fp16/fa_bwd_fp16_o7c_fa3_te_baseline.out.txt`；
以及改动前基线 `src/fp16/fa_bwd_fp16_mma_main_o7c_base_{s512,s4096}.out.txt`。
bf16 同款见 `01b` §6k。

---

## 14b. MLA（head_dim=512）张量核反向（O5c，main 3.2–5.5×）

P5-2 的 MLA 反向是**标量 golden**（CUDA-core FFMA，135KB smem、1 CTA/SM）；本节把 O5/O6/
O6b/O6c/O7c 的张量核路径扩到 **head_dim=512**，与 fp8 的 MLA（`docs/03` §14）同构。

### 14b.1 改动（单/两文件 device 代码逐字同源）

`fa_bwd_fp16_mma_kernel<HD,BM,BN,PIPE,...>` 原先 `static_assert(HD==128)`，现支持 128/512：

- **GEMM1/2（S=QKᵀ、dP=dO·Vᵀ）**：归约维是 HD ⇒ 只是 k-loop 从 `HD/16=8` 步变 `32` 步，
  A/B 布局与累加器完全不变。
- **GEMM3/4/5（dV/dK/dQ）**：输出 N 维是 HD ⇒ 加一层 **N-tile 循环**（`NTW=WN*64=128` 列/遍，
  共 `NDT=HD/NTW` 遍）。B 操作数（dO/Q/K 的 `[K][HD]` 转置布局）列方向连续 ⇒ 基址 `+hd0`
  即选中本遍列、写回列号 `+hd0`；`hd0=0` 时与 O5c **逐字等价**。
- **dQ 累加**：HD=128 时 dQ 的 N 一次铺满，用「寄存器沿 nt 累加、结束统一写回」。HD=512 时
  `[BM][HD]` 放不进寄存器 ⇒ 在 GEMM5 epilogue 里**直接全局累加**（`base[0/1] += ...`）。因为每个
  `(qi, 列)` 由**唯一线程**拥有（唯一 CTA + 唯一 warp + 唯一 N-tile），这是**非原子 RMW、无竞争**；
  `dq_acc` 由 host `memset(0)`（原本 run_all 就有）。
- host：`D∈{128,512}` 均可；LSE/`delta` 按 `D` 分派 `<128>/<512>` 实例并各自设动态 smem；
  MLA 自动档 `BM=32,BN=32,PIPE=1`（BM=64 的 K/V 双缓冲会超 232KB）。

### 14b.2 数值对拍（ours vs fp32 ref，fp16 causal，D=Dv=512）

| case | dq (max_abs) | dk | dv |
|---|---|---|---|
| S=256 H=2 | 1.638e-3 | 1.582e-3 | 1.753e-3 |
| S=512 H=4 | 2.516e-3 | 2.916e-3 | 1.724e-3 |
| S=1024 H=2 | 1.987e-3 | 1.712e-3 | 1.848e-3 |

均为 fp16 噪声量级（与 P5-2 标量版同量级；S=256/S=512 的 dk/dv 与标量版**逐位相同**，
dQ 因「寄存器累加→全局累加」次序略变而与标量版有 ~2e-3 差异，仍在噪声内）。
**FA3/TE 反向不支持 head_dim=512**（FA 限 ≤256、TE 训练 bwd 限 256），只有 fp32 ref 可对。
**MHA D=128 回归逐位不变**：S=512 1.671/1.771/1.899e-3、S=4096 1.883/1.734/1.966e-3（与
O5/O6/O6b/O6c/O7c 记录相同），单/两文件逐指标一致。

### 14b.3 性能（同 session CUDA event，preprocess/main/total，ms）

| case | 标量 main | **张量核 main** | main 加速 | 标量 total | **张量核 total** | total 加速 |
|---|---|---|---|---|---|---|
| S=256 H=2 | 1.0651 | **0.2037** | **5.23×** | 1.2877 | **0.3003** | 4.29× |
| S=512 H=4 | 2.1200 | **0.3847** | **5.51×** | 3.5065 | **0.5355** | 6.55× |
| S=1024 H=2 | 4.2140 | **0.7248** | **5.81×** | 6.8166 | **0.9393** | 7.26× |

配置 A/B（S=512 H4，main-only）：`(32,32,PIPE=1)` **0.3806ms**（5.64 TF 打印口径，真值 ~11.3 TF）
< `(32,32,PIPE=0)` 0.6371ms < `(64,32,PIPE=0)` 0.9029ms ⇒ **K 双缓冲 + 小 BM 提高并行度**
在 MLA 上值 **1.67×**。print 的 TFLOPS 用 `4BS²HD`，MLA 的 Dv=D ⇒ 真反向 FLOPs 是它的 **2×**。
峰值（fp16 ~989 TF）占比 ~0.8–1.2%。

### 14b.4 ncu（main，S=1024 H2 D=512，`(32,32,PIPE=1)`）

Duration **769µs**（标量 main 4.21ms ⇒ 5.5×）；DRAM 0.83% / **L1/TEX 40.09%** / L2 13.87% /
Compute 2.22%；168 regs、**207.36KB smem、1 CTA/SM**、理论=achieved occ **6.25%**、
**Waves 0.48**、No Eligible 91.6%；stall **`long_scoreboard` 7.68** + `wait` 2.23 + short 0.55、
barrier 0.05；smem bank conflict `op_ld` 仅 1.13M（对比标量 MLA 的 76.5% 多余 wavefront）。

⇒ **bound = 全局访存延迟 + 低并行度**（grid=`S/BM·H`=64 < 132 SM ⇒ Waves 0.48；且 207KB smem
把 occupancy 锁在 1 CTA/SM）。不再是标量版的 smem bank conflict。要进一步提速需**降 smem 冲
2 CTA/SM**（消 `Qp/dOp/Kp` 等副本）或 **split-KV 提高 grid**，留 backlog。

### 14b.5 原始输出

`src/fp16/fa_bwd_fp16_mma_main_p5mla_{s256h2,s512h4,s1024h2}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_p5mla_s512h4.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_p5mla_reg_d128_s512.out.txt`、
`src/fp16/fa_bwd_fp16_main_p5mla_{s256h2,s512h4,s1024h2}.out.txt`（标量基线）、
`src/fp16/fa_bwd_fp16_mma_main_p5mla_ncu_{sol,stall}_s1024h2.out.txt`。
bf16 同款见 `01b` §6l。

---

## 14c. O10：Q/dO 载入向量化 + `cp.async` 重叠 + 打包写回（fp16，main/total 1.03–1.32×）

### 14c.1 动机（ncu 证据）

O7c 把墙移到 **L2 71.6% + `wait` + 低 occupancy** 后，逐条看 ncu 的 `Source Counters`，
两处仍是「标量化的全局访问」：

- **主 kernel prologue 的 Q/dO**：`for i = tid; i < BM*HD; i += THREADS` 逐元素
  `LDG.U16 + STS.U16`（每线程 S=4096 时约 128 次标量 load + 128 次标量 store，且每元素都做
  `i/HD`、`i%HD` 整数除模）。ncu 报 **global loads 平均仅用满 26.4/32 B/sector**，且这串
  标量 load 的延迟**串在** K/V 的 `cp.async` 之前。
- **`lse_mma_kernel_bal` 的 Q** 同样是逐元素标量读（K 早已是 16B `cp.async`）。
- **dQ 写回**：`base[0]=…; base[1]=…` 两次 4B 写；ncu 报 global stores 平均仅 16/32 B/sector。

### 14c.2 改动（单/两文件 device 代码逐字一致，脚本核对 `identical: True`）

1. **新增 `qdo_issue_async<HD,BM>`**：把整块 Q/dO（BM×HD）用 16B `cp.async.cg` 发进 smem，
   行越界写 0；与 `kv_issue_async` 同构。主 kernel `PIPE>=1` 的 prologue 改用它
   （PIPE==0 保持同步标量读，因无流水语义）。Q/dO 与 K/V 各提交一个 `commit_group`，
   循环首的 `cp.async.wait_group 0` 一并等待 ⇒ Q/dO 的全局延迟与 K/V **重叠**。
2. **`lse_mma_kernel_bal` 的 Q** 同样改走 `issue_q`（PIPE=1 用 `cp.async`，PIPE=0 标量）。
3. **dQ 写回**（含 HD>128 的直接累加路径）相邻两列打包成一次 `float2`（8B）读/写
   （列号 `c` 恒为偶数 ⇒ 自然 8B 对齐）。

**数值与 O5/O5b/O8/O6/O6b/O8b/O6c/O7c 逐位相同**（搬的是同样的 half、乘的同样的 scale），
单/两文件逐指标一致。

### 14c.3 性能（同 session A/B：改动前二进制 vs 改动后，CUDA event，ms）

| case | total base → O10 | total 加速 | main base → O10 | main 加速 | preprocess base → O10 |
|---|---|---|---|---|---|
| S=512 H16 d128 | 0.1485 → **0.1256** | **1.18×** | 0.0770 → **0.0731** | 1.05× | 0.0536 → **0.0405** |
| S=4096 H16 d128 | 2.0965 → **2.0044** | 1.05× | 1.5916 → **1.5350** | 1.04× | 0.3915 → 0.3653 |
| S=1024 H32 kv4 | 0.4360 → **0.3767** | **1.16×** | 0.3062 → **0.2620** | **1.17×** | 0.1048 → 0.0874 |
| S=256 H2 d512 | 0.2990 → **0.2272** | **1.32×** | 0.2030 → **0.1662** | **1.22×** | 0.0837 → 0.0470 |
| S=512 H4 d512 | 0.5309 → **0.4357** | **1.22×** | 0.3833 → **0.3368** | 1.14× | 0.1154 → 0.0780 |
| S=1024 H2 d512 | 0.9417 → **0.8230** | **1.14×** | 0.7190 → **0.6657** | 1.08× | 0.1756 → 0.1381 |
| S=1024 H40 kv8 | 0.5159 → **0.4604** | 1.12× | 0.3657 → 0.3214 | 1.14× | 0.1220 → 0.1052 |
| S=1024 H64 kv1 | 0.7104 → **0.6424** | 1.11× | 0.5140 → 0.4586 | 1.12× | 0.1574 → 0.1380 |

收益来自两处：**LSE 的 Q 向量化**（S=512 时 preprocess 0.0536→0.0405，占端到端大头）与
**主 kernel Q/dO 的 `cp.async` 重叠**（GQA/MQA 与 MLA 的 main 1.1–1.2×，小网格尤其明显）。
端到端 total 1.05–1.32×，**大 S（S=4096）收益最小（1.05×）**，因 prologue 只占 kernel 的一小段。

### 14c.4 ncu（main，S=4096，`(64,64,2)`）

| 指标 | O7c | **O10** |
|---|---|---|
| Duration | 1.61 ms | **1.48 ms** |
| Executed Instructions | 350,867,456 | **342,052,864（−2.5%）** |
| `long_scoreboard` | 1.38 | **0.89** |
| `mio_throttle` | 0.49 | 0.70 |
| `wait` / `short` / `barrier` | 2.00 / 0.88 / 0.11 | 2.04 / 0.87 / 0.14 |
| shared load 多余 wavefront | 15.36% | **12.2%** |
| L1/TEX / L2 / Compute | 55.7 / 71.6 / 23.4 % | 57.0 / **76.9** / 23.7 % |
| regs / smem / occ | 250 / 105.47KB / 11.8% | 250 / 105.47KB / 11.8% |

`long_scoreboard`（全局读延迟）被压下，代价是 `mio_throttle` 略升（更多异步请求同时飞行）；
**墙仍是 `wait`（fixed-latency mma 依赖）+ L2 吞吐 + 2 CTA/SM 的低 occupancy**，只有 O9
（更低 smem 的 wgmma/TMA 数据通路）能同时解。

### 14c.5 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

MHA S=4096：FA3 0.3242ms/848TF、TE 0.4413/623、FA2 0.7287/377。ours total O10 **2.0044ms**
（68.6 TF，O7c 2.0935ms）⇒ **ours total 时间 6.18×（O7c 6.47×）**，为 FA3 的 ~8%（端到端口径）。
GQA kv4 S=1024：FA3 0.0821ms/418TF，ours total 0.3767ms ⇒ 4.59×

### 14c.6 原始输出

`src/fp16/fa_bwd_fp16_mma_main_o10_{ab,allshapes,ncu_s4096,ncu_s512,stall_s4096}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o10_*.out.txt`、
`src/fp16/fa_bwd_fp16_o10_fa3_te_baseline.out.txt`。

---

## 14d. O11：快速 exp/log（fp16，preprocess 8%，total ~1.8%）

**动机**：softmax 的 `expf`/`logf` 在主 kernel 与 LSE 预处理里都是**每元素**要算的热点，
而它们默认走 libdevice 的精确软件实现（~10 多条指令）。FA2/FA3 用的是硬件 `exp2f`（MUFU.EX2）。

**改动**（单/两文件 device 代码逐字一致）：新增 `fexp`/`flog` 内联（`FAST_EXP` 宏，默认 1）把
`expf`/`logf` 换成 `__expf`/`__logf`（MUFU.EX2/LG2，相对误差 ~2^-21），覆盖 `lse_mma_kernel`、
`lse_mma_kernel_bal` 与主 kernel 的 P 计算共 9 处；`-DFAST_EXP=0` 可退回精确版做 A/B。

**A/B（同文件顺序编译，fp16 MHA S=4096，CUDA event）**：`total` 2.0053→**1.9697ms（1.8%）**、
`preprocess` 0.3712→**0.3439ms（8.0%）**、`main` 1.5429→1.5400（~不变）。**数值逐位不变**
（dq/dk/dv vs ref 仍 1.883/1.734/1.966e-3）。结论：主 kernel 的墙**不在 exp**（是 L2/occupancy），
但 LSE 白赚 8%。原始输出 `src/fp16/fa_bwd_fp16_mma_main_o11_ab_fastexp.out.txt`。

> 同轮尝试并**证伪**的两条：① 主 kernel `(BM=32,BN=64,PIPE=2)`（更小 Q tile、更低寄存器、
> grid 翻倍）——dK/dV 跨 CTA 原子量翻倍，`S=4096` main 1.49→2.61ms（**慢 1.75×**），不可用；
> ② `cp.async` 加 `.L2::256B` 预取提示——main 无变化（L2 命中已 97%，不是扇区利用率问题）。

---

## 14e. O9a：LSE 预处理上 Hopper `wgmma`（冒烟 + 集成，fp16/bf16，单/两文件）

### 14e.1 动机

O8b 把 LSE 从 0.99ms 打到 0.30ms 后，它仍是端到端 ~15% 的一块，且 ncu 报
`Compute (SM) 60%` 而张量核利用率很低——因为 LSE 的 QKᵀ 走的是 **SM80 兼容的
`mma.m16n8k16 + ldmatrix`**：每个 k-step 都要 `ldmatrix`（ncu L1/TEX 32–38%），
且 mma 是同步指令、发射后要等结果，`Executed Ipc` 只有 2.36。

O9 的目标是把反向整体推到 Hopper 原生 `wgmma + TMA`。作为**风险最小的第一步**
（O9a），本节点先把 **LSE 的单个 QKᵀ GEMM** 换成 `wgmma.m64n64k16` SS：
它只有 1 个 GEMM、没有转置操作数、epilogue 只是 online-softmax，最适合先验证
「SW128 布局 + 描述符 + 累加器映射」这套机器。

### 14e.2 冒烟：`fa_bwd_fp16_wgmma_smoke.cu`

上集成前先证明两个映射（否则在完整 kernel 里 debug 布局极贵）：

* **SW128（128B swizzle）K-major smem 布局**：`[row/8][k/64][8 行][64 元素]` atom 1024B，
  atom 内 16B chunk `c' = c ^ r`；描述符 `SBO=(K/64)*1024`、`LBO=1`、k16 步进
  `floor(s/4)*1024 + (s%4)*32`。
* **`wgmma.mma_async.m64n64k16.f32.f16.f16` 的累加器布局**：warp w 持行 `[16w,16w+16)`，
  warp 内 `d[j*4+q]` ↔ `row=16*wid+g+(q>=2?8:0)`、`col=j*8+2*(lane%4)+(q&1)`——
  **与 `mma.m16n8` 的 `acc[i][j][q]` 逐字同构**，所以现有 epilogue 可以直接复用。

实测：`wgmma.m64n64k16 SW128 vs CPU: max_abs=0.000e+00`（`PASS`）。
注：CUDA 13 的 nvcc 必须用 `-gencode=arch=compute_90a,code=sm_90a`（`-arch=sm_90` 会静默
退化成 `sm_90`、ptxas 拒绝 wgmma）；本轮把 `scripts/run.sh`/`ncu.sh` 改成「`ARCH` 为空串时
只用 `NVCC_FLAGS` 里的 gencode」，并在 kernel 里用 `__CUDA_ARCH_FEAT_SM90_ALL` 把 wgmma asm
包起来，使默认 `sm_90` 构建仍可用（wgmma 路径退化为空实现，只有显式开启且用 sm_90a 才走）。
原始输出 `src/fp16/fa_bwd_fp16_wgmma_smoke.out.txt`。

### 14e.3 集成：`lse_mma_kernel_bal_wgmma<HD, PIPE>`（仅 HD=128、causal）

在 O8b 的镜像配对 + `cp.async` 双缓冲骨架上：

* Q/K 改成 **SW128 tile**：`cp.async` 的目标地址由 `sw128_off(row, c8*8, HD)` 给出
  （16B chunk 置换是双射，所以 16B 的 `cp.async` 可以直接落进 swizzled 布局）；行越界写 0。
* 单个 64×64×128 的 QKᵀ 用 **8 条 `wgmma.m64n64k16`** 完成（取代「4 warp × m16n64 ×
  8 k-step」的 mma + 每步 ldmatrix）。累加器按 §16.2 的映射直接喂给原有 online-softmax。
* smem 从 52.22KB 降到 **50.18KB**（Q+2×K 的 SW128 tile = 3×16KB + 1KB 对齐余量）。
* 单/两文件 device 代码逐字一致；默认走原 mma 版，`--lsewgm=1` 切 wgmma 版（需 sm_90a 构建）。

### 14e.4 性能（同 session A/B，CUDA event，LSE-only，ms）

| shape | O8（mma 单缓冲） | O8b bal+cpasync | **O9a wgmma+SW128** | vs O8b | vs O8 |
|---|---|---|---|---|---|
| fp16 S=512 MHA | 0.0641 | 0.0338 | **0.0330** | 1.024× | 1.942× |
| fp16 S=4096 MHA | 0.8597 | 0.3004 | **0.2871** | 1.046× | 3.022× |
| fp16 S=1024 GQA kv4 | 0.1515 | 0.0666 | **0.0642** | 1.037× | 2.360× |
| bf16 S=4096 MHA | 0.8749 | 0.3006 | **0.2856** | 1.054× | 3.073× |

端到端（fp16 S=4096）：preprocess 0.3251ms、main 1.5205ms、convert 0.1254ms、
**total 1.953ms（70.1 TF）**——`preprocess` 比 O11 的 0.344ms 再低 ~5%，total 与 O10/O11
基本持平（LSE 只占 ~15%，且 1.05× 的增益不足以改变总时间）。

### 14e.5 ncu（LSE，S=4096，同 session 对照）

| 指标 | O8b `lse_mma_kernel_bal` | **O9a wgmma** |
|---|---|---|
| Duration | 303.62 µs | **286.18 µs** |
| Compute (SM) | 57.04% | 60.66% |
| **L1/TEX** | **37.62%** | **18.05%**（ldmatrix 消失） |
| L2 | 31.62% | 20.25% |
| Executed Ipc Active | 2.36 | **2.52** |
| regs / smem | 62 / 52.22KB | 62 / 50.18KB |
| occupancy / Waves | 23.21% / 0.97 | 23.02% / 0.97 |

**结论**：wgmma 把 LSE 的 **L1/TEX 打掉一半**（37.6%→18.1%，`ldmatrix` 被 SS 直读取代）、
L2 也降（20.3%），`Executed Ipc` 升到 2.52；但 Duration 只快 6%——因为 LSE 的墙本来就不是
张量吞吐，而是 **softmax epilogue + 发射（Compute 60%、Waves 0.97、网格不足一个波）**。
所以 wgmma 在「小 GEMM + 重 epilogue」的 LSE 上收益有限（这也解释了 O8b 之后它的占比一直降不下来）。
原始输出 `src/fp16/fa_bwd_fp16_mma_main_o9_ncu_lse_{wgmma,mma}_s4096.out.txt`。

### 14e.6 数值（与 O8b **逐位相同**）

wgmma 的 k 归约顺序与 mma 的 k-step 顺序一致、都是 fp32 累加，所以 LSE 逐位相同，最终
dq/dk/dv 与 O8b/O10/O11 完全一致：fp16 S=4096 1.883/1.734/1.966e-3、S=512
1.671/1.771/1.899e-3、GQA kv4 2.134/3.305/3.850e-3、bf16 S=4096 1.510/1.340/1.631e-2。
单/两文件逐指标一致。

### 14e.7 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py`）

FA3 MHA S=4096 fp16 0.3240ms/848TF、TE 0.4406/624；bf16 FA3 0.3194/861、TE 0.4359/631。
ours total 1.95ms ⇒ 仍为 FA3 的 ~6.0×（与 O8b/O10 持平；LSE 不是差距的主因）。

### 14e.8 原始输出

* `src/fp16/fa_bwd_fp16_wgmma_smoke.out.txt`（冒烟 PASS）
* `src/fp16/fa_bwd_fp16_mma_{main,onefile}_o9_{s512,s4096}.out.txt`、`..._o9_gqa_kv4.out.txt`
* `src/bf16/fa_bwd_bf16_mma_{main,onefile}_o9_s4096.out.txt`
* `src/fp16/fa_bwd_fp16_mma_main_o9_ncu_lse_{wgmma,mma}_s4096.out.txt`
* `src/fp16/fa_bwd_fp16_o9_fa3_te_baseline.out.txt`、`src/bf16/fa_bwd_bf16_o9_fa3_te_baseline.out.txt`

**下一步（O9b）**：把这套 `wgmma + SW128` 机器推到 **主 kernel 的 5 个 GEMM**——那里 ncu 是
`wait`（mma 依赖）+ L2（dK/dV 原子）双墙，wgmma 的异步 mma 允许把 tile i+1 的 mma 和 tile i 的
epilogue 重叠、并免掉 ldmatrix；转置操作数（dK/dV 的 `B`）需要按 FA3 的 `dKV_swapAB` 思路处理。

## 14f. O13：主 kernel auto tile 重新标定 + fix 端到端 memset/convert 冗余（fp16/bf16）

### 动机

O6c 给主 kernel 加的自动 tile 选择是**基于当时的实验**（还没有 O7c-PREL）：
> `grid<132 且 S≤1024` → `(BM=32,BN=32,PIPE=1)`「保并行度」，否则 `BM=64`、`S≥4096→BN=64`。

但 O7c 的 LSE/D 预装寄存器、O10 的 Q/dO 向量化之后，**每个 CTA 的固定开销占比下降、BM=64 的
每 CTA 效率反超 BM=32**。本轮复测发现：S=512 MHA 上 `(64,64,2)` 的主 kernel 比自动档
`(32,32,1)` **快 1.23×**（O5c A/B：0.0695→0.0566ms），说明旧启发式已过时。

另外两处端到端冗余：
1. **dQ 的 memset 在 HD=128 时完全多余**——dQ 由主 kernel 寄存器累加后**覆盖写**（不是 RMW），
   只有 MLA（HD=512）的 GEMM5 才是全局 RMW 累加、需要清零。
2. `convert_kernel` 逐元素 `LDG.32 + STG.16`，改成 **`float4` 读 + `half2` 写**（4 元素/次）。

### 改动（单/两文件 device 与 host 逐字一致）

- **auto tile 重标定**：
  - 取消 `BM=32` 分支（实测 BM=64 全面更优；`--bm=32` 仍可手动覆盖）。
  - `BN=64` 的判据改为 `S≥4096 || grid≤256 || grid>600`；其中 `256<grid≤600` 保留 `BN=32`：
    该区间 `BN=64` 的 105KB smem 把 occupancy 压到 2 CTA/SM、恰好落进「2 个波」的坏量化点
    （实测 `grid=512` 的 S1024/GQA-kv4：`BN=32` 0.2534 vs `BN=64` 0.2706ms）。
  - `auto_pipe` 仅对 `BN=32` 路径生效；`BN=64` 路径固定走 PIPE=2（只双缓冲 K）。
- **dQ memset**：`cudaMemset(d_dq_acc,…)` 加 `if (D==512)` 守卫（HD=128 时省一趟 `n` 个 float）。
- **convert 向量化**：`float4` 读 d*_acc、`half2` 打包写 dq/dk/dv（基址 256B 对齐），尾元素标量兜底。

### 数值（全部逐位不变）

fp16 S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、GQA kv8 2.008/2.931/3.891e-3、
MLA S512H4 2.516/2.916/1.724e-3；bf16 S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3。
只改「选哪个 tile」与「搬运方式」，不改数学 ⇒ 与 O5~O10 逐位相同；单/两文件一致。

### 性能（同 session A/B，CUDA event）

| shape | 配置(旧→新) | main 旧→新 | total 旧→新 |
|---|---|---|---|
| S512 H16 (fp16) | (32,32,1)→(64,64,2) | 0.0721→**0.0574 (1.26×)** | 0.1255→**0.1121 (1.12×)** |
| S512 H16 (bf16) | 同上 | 0.0727→**0.0572 (1.27×)** | 0.1267→**0.1112 (1.14×)** |
| S1024 h40 kv8 | (64,32,2)→(64,64,2) | 0.3203→**0.3060 (1.05×)** | 0.4555→**0.4414 (1.03×)** |
| S1024 h64 kv1 | (64,32,2)→(64,64,2) | 0.4611→**0.4362 (1.06×)** | 0.6354→**0.6109 (1.04×)** |
| S1024 h64 kv4 | (64,32,2)→(64,64,2) | 0.4921→**0.4525 (1.09×)** | 0.6635→**0.6261 (1.06×)** |
| S1024 h32 kv4 | 保持 (64,32,2) | 0.2635（不变） | 0.3800（不变） |
| S4096 H16 | 保持 (64,64,2) | 1.5247（不变） | 1.9614（不变） |

`convert` 桶（含 3 个 memset）：S512 0.0157→**0.0132ms**、S4096 0.1037→**0.0939ms**。

同 session 纯反向基线（`harness/fa_vs_te_bwd_only.py`）：S4096 MHA FP16 FA3 0.3255ms/845TF、
TE 0.4442/619 ⇒ ours total 时间 **6.03×**（O10 6.18×）；GQA kv8 FA3 0.1217/353 ⇒ 3.63×。

### ncu（main, S=512, 旧 `(32,32,1)` vs 新 `(64,64,2)`）

| 指标 | 旧 `(32,32,1)` | 新 `(64,64,2)` |
|---|---|---|
| Duration | 83.9 µs | **59.7 µs（−29%）** |
| L1/TEX | 43.5% | **20.2%** |
| L2 | — | 36.1% |
| Compute | 11.9% | 10.7% |
| achieved occ | 10.99% | 6.23%（grid=128<132 SM，1 CTA/SM） |
| stall | — | `wait 2.00 + long 1.01 + short 0.46` |

S=512 的 `grid=128` 不足 132 SM，是**尾波/grid-bound**；`(64,64,2)` 用更大的每-CTA tile 把
L1/TEX 砍半，故仍更快。S=4096（`grid=1024`）：Duration 1.54ms、occ 11.85%（2 CTA/SM）、
L1/TEX 46.6%、**L2 73.7%**、Compute 22.5% —— 与 O7c 一致，**墙仍是 L2（dK/dV 原子）+ `wait`**。

原始输出：`src/fp16/fa_bwd_fp16_mma_main_o13_{s512,s4096,gqa_kv8}.out.txt`、
`..._mma_onefile_o13_s512.out.txt`、`..._o13_ncu_main_{s512,s4096}.out.txt`、`src/fa_bwd_o13_fa3_te_baseline.out.txt`；
bf16 同构（`src/bf16/..._o13_*`）。

---

## 14g. O9b：主 kernel GEMM1/GEMM2 上 Hopper `wgmma`（SW128 + `ldmatrix` 桥接，fp16）

### 14g.1 动机

O13 之后 fp16 main 的墙是 **`wait`（`mma.m16n8k16` 的依赖）+ L2（dK/dV 跨 CTA 原子）+ 2 CTA/SM**。
O9a 已建立 Hopper `wgmma.m64n64k16 + SW128` 数据通路并在 LSE 上验证，但 LSE 是 epilogue bound。
O9b 把 wgmma 推到主 kernel 的 GEMM1/2（`S=QKᵀ`、`dP=dO·Vᵀ`，A/B 都 K-major、可直接 SS），
这是主 kernel 两个最大、最热的 GEMM。

### 14g.2 冒烟：`fa_bwd_fp16_wgmma_main_smoke.cu`（逐位 PASS）

难点在于 GEMM3/4/5 的 B（dO/Q/K）在主 kernel 里是**转置读**，与 wgmma 的 K-major 布局冲突。
冒烟把三件事拼在一起、与 CPU fp32 参考逐位比对（整数输入，`max_abs=0.000e+00`）：

1. `wgmma.m64n64k16` 的 `S=Q·Kᵀ`（Q/K 存 SW128 K-major）；
2. `dV=Pᵀ·dO`：A=P（`ldmatrix.x4.trans`，ATRANS）＋ B=dO 从 **SW128 tile** 转置读；
3. `dQ=P·K`：A=P（普通 `ldmatrix.x4`）＋ B=K 从 **SW128 tile** 转置读。

关键结论：**`ldmatrix` 能从 SW128 tile 直接读「转置」数据**——SW128 只在 **16B 粒度**做
`c' = c ^ r` 置换，每个 16B chunk（8 个 fp16）完好，`ldmatrix` 每个 lane 只需一个 16B 地址；
用 `sw128_off(row=token, k=hd, HD)` 算出地址喂 `ldmatrix.x2.trans`，与行主序 `[K][N]` 转置读
**逐位一致**。于是 Q/K/V/dO 只需要一份 SW128 布局，GEMM1/2 给 wgmma、GEMM3/4/5 给 mma。

### 14g.3 实现（单/两文件 device 代码逐字一致，`diff` 核对 `WGMMA DEVICE BLOCK IDENTICAL`）

新增 `fa_bwd_fp16_wgmma_kernel<HD>`（仅 HD=128 / BM=BN=64 / fp16），由 `#ifdef FA_WGMMA` 包裹
（默认 `sm_90` 构建不编译、行为不变；wgmma 构建需 `-gencode=arch=compute_90a,code=sm_90a`）：

- **smem**：Q/dO/K/V 全部 **SW128 K-major**（各 16KB；K 双缓冲、V 单缓冲后段预取），P/dS 仍
  `[BM][BN]` 行主序 +8 行距（GEMM3/4 用 ATRANS）。总 ≈ **101KB / 2 CTA/SM**（与 O13 的
  `(64,64,2)` 同 occupancy）。SW128 描述符要求 tile 1024B 对齐 ⇒ 手动对齐动态 smem 基址
  （与 O9a 的 LSE wgmma 同做法）。
- **GEMM1/2**：`wgmma_mn64_issue` 把两组异步 mma 一起发出（8 步 k16 各一条），最后统一
  `wgmma.wait_group 0` ⇒ **两条 wgmma 重叠**（相比每次内部 wait 的初版，S512 A/B 1.043×→1.09×）。
- **GEMM3/4/5**：仍 `mma.m16n8k16`，但 B 用新 helper `mma_block_swb` 从 SW128 tile
  `ldmatrix.x2.trans` 读；A（P/dS）用 `ldmatrix.x4` / `x4.trans`（O6b）。
- **epilogue**：wgmma m64n64 的累加器布局与 `mma.m16n8` 同构（`d[j*4+q]` ↔ 行 `wid*16+g+(q≥2?8:0)`、
  列 `j*8+2*(lane%4)+(q&1)`），softmax 的 `p=exp(scale·S−LSE)`、`dS=P∘(dP−D)` 直接沿用；
  LSE/D 仍是 O7c 的预装寄存器版。
- host 加 `--wgmma=0/1`，自动档 `sel=(BM=64,BN=64)` 且 D=128 时走 wgmma，否则回退原 mma 路径
  （GQA 的 `BN=32` 自动回退，回归不变）。

### 14g.4 数值（与 O13 的 mma 路径**逐位相同**）

| shape | dq / dk / dv max_abs（ours vs ref） | FA vs ref | TE vs ref |
|---|---|---|---|
| MHA S=512 | 1.671 / 1.771 / 1.899e-3 | 1.679 / 1.684 / 1.899e-3 | 1.716 / 2.287 / 1.899e-3 |
| MHA S=4096 | 1.883 / 1.734 / 1.966e-3 | 1.883 / 1.734 / 1.966e-3 | 1.883 / 1.858 / 1.966e-3 |
| GQA q32/kv4 S=1024（回退 mma） | 2.134 / 3.305 / 3.850e-3 | 1.727 / 3.321 / 5.107e-3 | 2.075 / 3.175 / 5.107e-3 |

与 O5~O13 记录**逐位一致**：只换数据通路（SW128 + wgmma），不改数学口径。

### 14g.5 性能（同 session A/B，CUDA event，main-only，ms）

| shape | mma `(64,64,2)` | **wgmma** | 加速 |
|---|---|---|---|
| MHA S=4096 | 1.5103 | **1.4435** | **1.046×** |
| MHA S=512 | 0.0570 | **0.0522** | **1.092×** |

端到端（两文件，`--wgmma=1`）：S=512 total 0.1091ms、main 0.0527ms；S=4096 total **1.8822ms
（真反向 FLOPs 口径 ≈146 TF）**、main 1.4477ms。同 session 纯反向 FA3 S=4096 **0.3255ms/844TF**、
TE 0.4449/618 ⇒ ours total **5.8×** FA3（O13 6.03×）、真反向 TF 约 FA3 的 **17%**。

### 14g.6 ncu（main，S=4096，wgmma，`--launch-count 1`）

| 指标 | O13 mma `(64,64,2)` | O9b wgmma |
|---|---|---|
| Duration | 1.54 ms | **1.43 ms** |
| L1/TEX | 46.6% | 55.1% |
| **L2** | 73.7% | **74.4%（仍为墙）** |
| Compute | 22.5% | 26.5% |
| DRAM | 4.0% | 4.6% |
| regs / smem | 250 / 105.5KB | 242 / **101.4KB** |
| CTA/SM（occ） | 2（11.85%） | 2（**11.88%**） |
| stall | `wait 1.94 + long 1.45 + short 0.46` | **`wait 1.50 + long 1.25 + short 0.56`** |

**结论**：wgmma 把 GEMM1/2 的 `wait` 与 ldmatrix 打掉（`wait` 1.94→1.50），但 **GEMM3/4/5
仍是 mma、dK/dV 仍是跨 CTA 原子**，L2 74% 依旧封顶、occupancy 仍被 101KB smem 锁在 2 CTA/SM。
故收益是**真实但有限**（main 4.6–9.2%）。要再进一步必须 **O9b-2**：把 GEMM3/4/5 也上 wgmma
（A=P/dS 得进 smem/SW128；dKV 转置 B 按 FA3 `dKV_swapAB` 处理），并 **TMA 化 K/V**、双缓冲
P/dS 让 tile nt+1 的 GEMM1/2 与 tile nt 的 GEMM3/4/5 重叠；以及 O7b 的 dK/dV 去原子。

### 14g.7 原始输出

`src/fp16/fa_bwd_fp16_wgmma_main_smoke.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o9b_{s512,s4096}.out.txt`（两文件 wgmma）、
`src/fp16/fa_bwd_fp16_mma_onefile_o9b_{s512,s4096}.out.txt`（单文件 + `[O9b A/B]`）、
`src/fp16/fa_bwd_fp16_mma_main_o9b_ncu_s4096.out.txt`、`src/fa_bwd_o9b_fa3_te_baseline.out.txt`。
bf16 的 O9b 留作下一步（本版只做 fp16）。

---

## 14h. O9b-2（第一步）：主 kernel GEMM3/4/5 也上 `wgmma`（MN-major 转置读，fp16；数值逐位正确、性能中性）

### 14h.1 动机

O9b（§14g）只把 GEMM1/2 换成 wgmma，收尾结论是「新墙 = `wait`（GEMM3/4/5 的 `mma`）+
L2（dK/dV 原子）+ 2 CTA/SM」。O9b-2 要把 GEMM3/4/5（`dV=Pᵀ·dO`、`dK=dSᵀ·Q`、`dQ=dS·K`）
也上 wgmma。难点：这三个 GEMM 的 A/B 在 1colblock 数据流里是**转置读**，而 P/dS/dO/Q/K
又同时要被 wgmma 当 K-major 操作数。

### 14h.2 关键手段：`Major::MN` 描述符 = 对 K-major SW128 tile 的「转置读」

FA3 的 `dKV_swapAB` 就是这么做的：**同一份 K-major SW128 tile，用 MN-major 描述符 +
`trans=1` 读，等价于读它的转置**。对本实现的 `sw128_off` 布局可精确推导（冒烟逐位验证）：

- 描述符：`layout_type=B128`、**LBO=64**（相邻 64 列组的 u128 步长）、
  **SBO=(W/64)*64**（相邻 8 行组的 u128 步长）；`W` = tile 的连续维宽度（元素）。
- 转置操作数第 `s` 个 k16 slab（K=行，每次前进 16 行 = 2 个行组）地址 =
  `base + s*2*SBO*16` 字节（新增 `trans_k16_addr`）。
- wgmma 指令尾部立即数 `..., scaleA, scaleB, tnspA, tnspB`，K-major 时 `tnsp=0`、
  MN-major 时 `tnsp=1`（CUTLASS `MMA_64x64x16_F32F16F16_SS` 同形）。

前置冒烟 `src/fp16/fa_bwd_fp16_wgmma_bwd_smoke.cu`（整数输入，CPU fp32 参考）逐位通过：

```
wgmma S=QKᵀ        vs CPU: max_abs=0.000e+00
wgmma dV=PᵀdO(trans) vs CPU: max_abs=0.000e+00
wgmma dK=dSᵀQ(trans) vs CPU: max_abs=0.000e+00
wgmma dQ=dS·K(B trans) vs CPU: max_abs=0.000e+00
PASS
```

于是 Q/K/V/dO 仍是**一份** SW128 K-major（GEMM1/2 直接读；GEMM3/4/5 转置读）；P/dS 也改成
**一份** SW128 K-major（GEMM3/4 用 MN-major 读，GEMM5 的 A=dS 用 K-major 读）。

### 14h.3 实现（单/两文件 device 代码逐字一致，`scripts/sync_onefile_device.py` 核对 `identical: True`）

`fa_bwd_fp16_wgmma_kernel<HD>`（`#ifdef FA_WGMMA`）改动：

- **P/dS 改 SW128 K-major**（`[BM][BN]`，各 8KB），epilogue 用**相邻两列打包成 4B 直写**
  `sw128_off`（`pds_store_sw128`；最初用 quad `shfl` 拼 16B，S=4096 反而慢 3%，改 4B 直写后反超）。
- **5 个 GEMM 全 wgmma**：GEMM1/2 沿用 `wgmma_mn64_issue`；GEMM3/4/5 新增
  `wgmma_m64n64k16_t<TA,TB>`，按 N 半（`nh=0/1`）分两遍，每遍三条一起发、统一 `wait0`。
- **dQ 寄存器累加重映射**：wgmma m64n64 的累加器里 warp `wid` 固定持行 `[16w,16w+16)`，
  故 `dqacc[nh][j][q]`（64 个 fp32/线程）直接对应两条 N 半的寄存器，nt 循环后一次写出。
- **smem** `101.4→97.3KB`（去掉行主序 P/dS 及其 +8 行距），仍 **2 CTA/SM**；**230 regs、0 spill**。

### 14h.4 数值（与 O5~O13 的 mma / O9b **逐位相同**）

| shape | dq / dk / dv max_abs（ours vs ref） | FA vs ref | TE vs ref |
|---|---|---|---|
| MHA S=512 | 1.671 / 1.771 / 1.899e-3 | 1.679 / 1.684 / 1.899e-3 | 1.716 / 2.287 / 1.899e-3 |
| MHA S=4096 | 1.883 / 1.734 / 1.966e-3 | 1.883 / 1.734 / 1.966e-3 | 1.883 / 1.858 / 1.966e-3 |
| GQA q32/kv4 S=1024 | 2.134 / 3.305 / 3.850e-3 | 1.727 / 3.321 / 5.107e-3 | 2.075 / 3.175 / 5.107e-3 |

单文件与两文件逐位一致；只换了数据通路（MN-major 描述符 + 全 wgmma），数学口径未动。

### 14h.5 性能（同 session A/B，CUDA event，main-only，ms）—— **中性偏负**

| shape | mma 最优 | O9b（仅 GEMM1/2 wgmma） | **O9b-2 全 wgmma** | 结论 |
|---|---|---|---|---|
| MHA S=512 | 0.0584 `(64,64,2)` | 0.0522* | 0.0604 | 比 mma 慢 ~3% |
| MHA S=4096 | 1.4708–1.4840 `(64,64,2)` | 1.4435* | 1.4996–1.5227 | 与 mma 持平/略慢 |
| GQA q32/kv4 S=1024 | 0.2482 `(64,32,2)` | —（回退 mma） | 0.2664 | 比 mma 慢 ~7% |

（带 `*` 为 §14g 当时 session 的数，非本轮同 session。）同一 session 内直接对比：全 wgmma
**没有跑赢 mma 最优档**，也略慢于 O9b。端到端 S=4096 total ≈ **1.91–1.97ms（~72 TF）**。

### 14h.6 ncu（main，`--wgmma=1`，`--launch-count 1`）

| 指标 | O9b wgmma（§14g） | **O9b-2 全 wgmma** |
|---|---|---|
| Duration（S=4096） | 1.43 ms | 1.49–1.54 ms |
| L1/TEX | 55.1% | 38.6–40.3% |
| **L2** | **74.4%（墙）** | **69.1–71.4%（仍为墙）** |
| Compute | 26.5% | 22.5–23.5% |
| Tensor pipe active | — | 11.6–12.1% |
| regs / smem | 242 / 101.4KB | 230 / **99.33KB** |
| CTA/SM（occ） | 2（11.88%） | 2（**11.83%**） |
| stall（S=4096） | `wait 1.50 + long 1.25 + short 0.56` | **`wait 1.43 + long 1.96 + short 0.29`** |
| stall（S=512） | — | `wait 1.41 + long 0.86 + short 0.28`（grid=128 单波） |

**结论**：全 wgmma 把 GEMM3/4/5 的 `ldmatrix` 也打掉（`short_scoreboard` 0.56→**0.29**），
但**墙没有移动**：L2 仍 ~70%（dK/dV 跨 CTA `red.global` 原子）、P/dS 的 4B+MN-major 读写
与跨-tile 串行让 `long_scoreboard` 反升到 ~1.96、occupancy 仍被 ~99KB smem + 230 regs 锁在
**2 CTA/SM**。即 **O9b-2 的「GEMM3/4/5 上 wgmma」本身不是瓶颈所在**——真正的墙是
**dK/dV 原子归约 + occupancy**。因此本步收益中性，但其价值是**建立了「MN-major 转置读」的
数据通路**（后续 TMA 化 + P/dS 双缓冲 + 跨-tile 流水、以及 O7b 去原子都要用它）。

### 14h.7 原始输出

`src/fp16/fa_bwd_fp16_wgmma_bwd_smoke.out.txt`（冒烟 PASS）、
`src/fp16/fa_bwd_fp16_mma_main_o9b2_{s512,s4096,gqa_kv4}.out.txt`（两文件 wgmma）、
`src/fp16/fa_bwd_fp16_mma_onefile_o9b2_{s512,s4096,gqa_kv4}.out.txt`（单文件）、
`src/fp16/fa_bwd_fp16_mma_main_o9b2_ncu_{s512,s4096}.out.txt`、
`src/fp16/fa_bwd_fp16_o9b2_fa3_te_baseline.out.txt`（同 session 纯反向 FA3/TE）。

---

## 14i. O15a/O16：TMA 数据通路冒烟 + epilogue 重叠（负结果）+ 把墙钉在 L2 原子

### 14i.1 先量化：main 的墙到底是哪一级（ncu `--set full`，S=4096 causal）

causal S=4096、H=16、D=128、主 kernel `fa_bwd_fp16_wgmma_kernel<128>` 单次 launch：

| 指标 | 值 |
|---|---|
| Duration | 1.54 ms |
| **L2 Cache Throughput** | **68.83%** ← 第一 |
| L1/TEX Cache Throughput | 48.00% |
| Compute (SM) Throughput | 23.67% |
| DRAM Throughput | 4.19% |
| Achieved / Theoretical Occupancy | 11.90% / 12.50%（230 regs、99.33KB smem、Block Limit Shared Mem=2）|
| Waves Per SM | 3.88 |

把 L2 的扇区按操作拆开（`lts__t_sectors_op_*`）：

| L2 操作 | 扇区数 | 占比 |
|---|---|---|
| **`red`（dK/dV 的 `atomicAdd`）** | **102,236,160** | **73.1%** |
| `read` | 35,829,121 | 25.6% |
| `write` | 1,576,021 | 1.1% |
| 合计 | 139,841,608 | 100% |

`L2 Hit Rate = 96.2%`、DRAM 仅 4.2% ⇒ **所谓「L2 墙」的真身是 dK/dV 的跨 CTA
原子归约吞吐，不是数据带宽**。stall：`long_scoreboard 2.01 + wait 1.43 + barrier 0.84
+ short 0.29 + not_selected 0.21`（per issue active），Executed Instructions 331.1M。

> 结论：主 kernel 是 **L2 原子字节数 bound**。任何只改「搬运方式/等待时机」的优化都动不了它；
> 只有**减少 dK/dV 的原子字节数**（= 减少每个 KV 元素被多少个 CTA 贡献）才是真杠杆。

### 14i.2 O15a：TMA（`cp.async.bulk.tensor`）+ SW128 数据通路冒烟（PASS）

`src/fp16/fa_bwd_fp16_tma_smoke.cu`：用 `cuTensorMapEncodeTiled(SWIZZLE_128B)` 建
[64][128] fp16 矩阵的 tensormap，kernel 里用 2D TMA + mbarrier 把一个 [64][64] 的 box
搬进 smem，然后：

1. **逐字节**比对 TMA 写出的 smem vs kernel 里 `sw128_off()` 描述的 K-major SW128；
2. 用 `wgmma.m64n64k16` 直接消费这块 smem 算 `Q·Kᵀ`，与 CPU 对拍。

实测：

```
TMA layout byte-mismatch = 0 (expect 0)
TMA->wgmma QKt vs CPU: max_abs=0.000e+00
PASS
```

原始输出 `src/fp16/fa_bwd_fp16_tma_smoke.out.txt`。

**关键发现（可直接复用）**：一个 TMA box 的内维是 128B（SWIZZLE_128B），对 fp16 就是
**64 个元素**；所以 **HD=128 的 K-major tile 必须拆成 2 个 k-chunk（{0..63}、{64..127}）**，
每个是独立的 8KB SW128 区块，wgmma 侧用两个 `SBO=1024` 的描述符读（而不是一个
`SBO=2048` 的交织布局）。原因是 SW128 canonical 布局是 `[rg][kg][rr][kk]`（kg 夹在 rg
的 atom 之间），而 TMA 只能把 box **连续**写进 smem，无法在一维 box 里产生这种交织。
这与 kernel-opt 23/28 篇「bf16 TMA 的 `BK` 必须锁 64」是同一件事。

### 14i.3 O16：用分段 `wgmma.wait_group` 重叠 epilogue（负结果，中性）

动机：stall 里 `wait 1.43`（fixed-latency，wgmma 累加器依赖）排第二。想法是**只改等待时机**
（数值逐位不变）：S=QKᵀ 与 dP=dO·Vᵀ 两条 wgmma 发出后，先 `wait_group<1>` 只等前者、把
P 的 exp/写 smem 与仍在飞的 dP 重叠；GEMM3/4/5 三条也按 `wait_group<2>/<1>/wait0` 逐个收，
让 dV/dK 的 `red` 与前一条 wgmma 重叠。模板开关 `OW` + CLI `--ow=`（默认关）。

同 session A/B（`--iters=100`，main-only，ms）：

| 会话 | `wait0`（原） | `wait_group`（重叠） | 比 |
|---|---|---|---|
| S=4096 第一次 | 1.5124 | 1.5064 | 1.004× |
| S=4096 第二次 | 1.5062 | 1.5193 | 0.991× |

数值：`max|diff| dq=0`（逐位相同），`dk/dv ~9e-5~1.4e-4`（只是 `atomicAdd` 次序变了）。

**结论**：中性。`wait` 只是**症状**——wgmma 本身不是瓶颈时，把等待错开也拿不到收益；
真正的墙是 §14i.1 的 L2 原子。故 `--ow=` 默认关、保留 A/B（`src/fp16/fa_bwd_fp16_mma_main_o16_s4096.out.txt`）。

### 14i.4 同 session 对标（纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

| shape | FA2.7.4 | **FA3（SM90）** | TE2.14 |
|---|---|---|---|
| (1,4096,16,128) MHA | 0.7208ms / 381 TF | **0.3230ms / 851 TF** | 0.4383ms / 627 TF |
| (1,1024,64,128) kv1 MQA | 0.2640 / 260 | **0.1560 / 440** | 0.2135 / 322 |
| (1,1024,32,128) kv4 GQA | 0.1571 / 219 | **0.0823 / 417** | 0.1116 / 308 |

ours 端到端（fp16，S=4096，mma 默认档）≈ **1.93ms**（preprocess 0.34 + main 1.50 + convert 0.09）
⇒ 时间约 FA3 整条反向的 **6.0×**、TFLOPS ~2.4%；瓶颈全在 main 的 L2 原子（§14i.1）。
原始输出 `src/fp16/fa_bwd_o15_fa3_te_baseline_fp16.out.txt`。

### 14i.5 这一轮把「还能怎么抠 fp16 main」的路线收敛为

| 路线 | 机制 | 为什么（不）成立 |
|---|---|---|
| 分段 wait / epilogue 重叠（O16） | 改等待时机 | **中性**（§14i.3）：墙不在 wgmma |
| dK/dV 归约 float4（O7c） | 减 red 事务数 | **更慢 1–7%**：墙在字节不在事务数 |
| TMA 化 operand（O15a 已建通路） | 减 load 指令/地址 | 只动 L1/long_scoreboard，动不了 L2 red |
| **跨 warpgroup 归约（BM=128）** | 一个 KV 元素被 **nblk/2** 个 CTA 贡献 | **唯一真杠杆**：直接把 red 字节砍半；代价是 smem 176KB→1 CTA/SM + smem 合并 |
| KV-outer 重排（FA2 式） | dK/dV 存 smem 顺序累加、改 dQ 原子 | MHA 下原子字节减半，但 **GQA/MQA 会放大**（dQ 是 H 维、dK/dV 是 Hkv 维），不通用 |

## 14j. O17：跨 warpgroup 归约（BM=128、2 warpgroups），dK/dV 的 red 字节砍半

### 14j.1 动机（§14i 已把墙钉死）

§14i 用 ncu 拆开 S=4096 主 kernel 的 L2 扇区：**`red`（dK/dV 跨 CTA `atomicAdd`）
= 102,236,160 / 139,841,608 = 73.1%**、`read` 25.6%、`write` 1.1%，DRAM 仅 4.2% ⇒ main 是
**L2 原子字节数 bound**。O16（错开 `wait_group`）与 O7c（float4 归约）都已证明「动等待 / 动事务数」
无效。**唯一杠杆**是让每个 KV 元素被更少的 CTA 贡献：`1colblock` 下每个 CTA 覆盖 BM 行 Q，
dK/dV 沿 M 维归约 ⇒ 一个 KV 元素被 `nblk = S/BM` 个 CTA 各贡献一次。把 **BM 从 64 翻到 128**，
贡献它的 CTA 数直接减半 ⇒ red 字节砍半。

难点是 BM=128 后寄存器/并行度：GEMM1/2 的 S/dP 累加器、GEMM5 的 dQ 累加器都随 BM 翻倍。
解法是 **2 个 warpgroup（256 线程），各算自己 64 行 Q**，把 BM=128 摊到两组上；dK/dV 的跨 wg
合并则用一个技巧（下节）。

### 14j.2 关键手段：dK/dV 让**一个** warpgroup 对全 BM 归约

若两组各自算 `dV_wg = P_wgᵀ·dO_wg`（[BN][HD]），再把两组偏和合并，就要一次 smem 暂存。
更省的做法是：**只让 wg0 做 GEMM3/4，并把两个 m64 半（Q 行 0–63 与 64–127，即 s=0..7 共 128 行）
连续喂给同一个 `wgmma.m64n64k16` 累加器**——两次 m64 的输出形状都是 `[BN][HD]`，累加器天然相加，
得到的**就是整个 BM 的 dV/dK**，于是每个 KV 元素只 `red` 一次。wg1 在并行做自己的 GEMM5（dQ 只依赖
本 wg 的 Q 行，互不干扰）。

这套「对 [128][*] 的 K-major SW128 tile，用 MN-major 描述符按 `s=0..7` 读转置」是新的、也是本项
唯一的数据通路风险，故先写最小冒烟 **`fa_bwd_fp16_wgmma2_smoke.cu`** 逐位验证：
- `dV=PᵀdO`（A=P[128][BN] 转置、B=dO[128][HD] 转置，K=BM=128）；
- `dK=dSᵀQ` 同构；
- `dQ=dS·K`（两个 m64 半，A=dS 的对应 64 行块）。
  实测三项 **max_abs=0.000e+00（逐位 PASS）**，确认 MN-major 描述符的 `trans_k16_addr`
  （每 k16 slab 前进 16 行 = 2 rowgroup）对 128 行依然成立。

### 14j.3 实现（单/两文件 device 代码逐字一致，`scripts/sync_onefile_device.py` 核对 `identical: True`）

新增 `fa_bwd_fp16_wgmma2_kernel<HD>`（`#ifdef FA_WGMMA`，仅 HD=128）：
- **2×2→2 wg**：`wg = tid>>7`，`wid = (tid>>5)&3`；本 wg 的 Q/dO/P/dS tile 基址按「64 行 =
  8 rowgroup」偏移（`wg * 8 * (HD/64) * 1024` / `wg * 8 * (BN/64) * 1024`）；
- GEMM1/2 各 wg 用 `wgmma_mn64_issue`（m64n64）算自己的 64×BN 的 S/dP；P/dS 按 SW128
  `pds_store_sw128` 写进**共享**的 `Ps/dSs[128][64]`（wg 写自己 64 行）；
- 中段 `__syncthreads` 后 **wg0** 做 GEMM3/4：`for nh: for s in 0..7` 用
  `desc_k16_mn(Ps/dSs, s, BN)`、`desc_k16_mn(dO/Q + nh*1024, s, HD)`，全 BM 归约 + `red_add2`；
  **wg1** 只做 GEMM5；
- GEMM5 各 wg 算自己 64 行（A=本 wg 的 dS 行块 `desc_k16_k(dSw,s,BN)`、B=K 转置
  `desc_k16_mn(Kt+nh*1024,s,HD)`），寄存器累加、循环后一次 `float2` 写回（每 Q 块唯一 CTA）；
- K/V 用 **256 线程** `cp.async`（`kv_issue_async_sw`/`qdo_issue_async_sw` 加模板参数
  `NT=256`；K 双缓冲、V 单缓冲后段预取，同 O6b/O9b）；Q/dO 也 256 线程一次发。
- smem：Q 32KB + dO 32KB + K 2×16KB + V 16KB + P 16KB + dS 16KB ≈ **149.5KB → 1 CTA/SM**
  （256 线程 = 8 warps/SM，与 O9b 的 2 CTA/SM × 4 warps 相同）。寄存器 **200**、0 spill。

单/两文件：`launch_bwd_wgmma2<HD>` + CLI `--wg2`；`run_main` 优先走 wg2；`[O17 A/B]` 段同 session
对比 mma 最优档 / O9b wgmma / O17 wgmma2，并打印 `max|diff|`。**没有**改默认行为（需 `--wg2`）。

### 14j.4 数值（ours-vs-ref，fp16 causal，max_abs）

| shape | dq | dk | dv | `max\|diff\|` wg2-vs-mma (dq/dk/dv) |
|---|---|---|---|---|
| MHA S=512 | 1.671e-3 | 1.771e-3 | 1.899e-3 | 0 / 8.8e-5 / 1.6e-4 |
| MHA S=4096 | 1.883e-3 | 1.734e-3 | 1.966e-3 | 0 / 3.6e-4 / 4.3e-4 |
| GQA h32kv4 S=1024 | 2.134e-3 | 3.305e-3 | 3.850e-3 | 3.6e-7 / 6.1e-4 / 1.1e-3 |
| MQA h64kv1 S=1024 | 2.292e-3 | 7.934e-3 | 7.517e-3 | 0 / 5.1e-3 / 5.9e-3 |

**vs ref 与历史逐位一致**（fp16 误差由舍入主导，非原子次序）；`dq` 在 MHA 下 **逐位相同**
（dQ 无跨 CTA 原子），dk/dv 只差 atomic 累加次序。单/两文件逐指标一致。

### 14j.5 性能（同 session A/B，CUDA event，main-only，ms）

| shape | mma 最优档 | O9b wgmma | **O17 wgmma2(BM128)** | vs mma |
|---|---|---|---|---|
| MHA S=512 (64,64,1) | 0.0573 | 0.0591 | **0.0517** | **1.11×** |
| MHA S=4096 (64,64,2) | 1.5310 | 1.5165 | **0.9784** | **1.57×** |
| GQA kv4 S=1024 (64,32,2) | 0.2636 | 0.2671 | **0.1789** | **1.47×** |
| MQA kv1 S=1024 (64,64,2) | 0.4326 | 0.4217 | **0.2812** | **1.54×** |

端到端（preprocess+main+convert）fp16：S=4096 **1.4268ms（96.3 TF，`4BS²HD` 口径）**、
S=512 0.1071ms、GQA kv4 0.2925ms、MQA kv1 0.4555ms。**main-only S=4096 140.5 TF**（O9b 90.2 TF）。

### 14j.6 ncu（main，S=4096，同 session，O9b vs O17）

| 指标 | O9b wgmma (BM=64) | **O17 wgmma2 (BM=128)** |
|---|---|---|
| `lts__t_sectors_op_red` | 102,236,160 | **51,904,512（0.508×）** |
| `lts__t_sectors_op_read` | 35,887,831 | 17,952,433（0.50×，Q/dO 少读一遍） |
| `lts__t_sectors_op_write` | 1,576,222 | 1,575,847 |
| Duration | 1.48 ms | **0.996 ms** |
| L2 Throughput | 71.76% | **54.72%** |
| L1/TEX | 40.40% | 36.85% |
| DRAM | 4.36% | 6.47% |
| Compute (SM) | 23.58% | 27.48% |
| regs / smem | 230 / 100.35KB | 200 / 149.5KB |
| achieved occ | 11.89%（2 CTA/SM） | 12.41%（1 CTA/SM） |

**red 字节精确减半、read 也减半**（BM=128 让 Q/dO 只读一遍），L2 从 71.8% 降到 54.7%、
Duration 1.48→1.00ms——**O17 的机制假设被 ncu 完全证实**。shared bank conflict 仍为 **0**
（SW128 + padding）。**新墙仍是 L2**（red 51.9M 占 L2 扇区 ~72.6%），故进一步路线是
**BM=256 / 4 warpgroups**（再砍半）或 `dK/dV` 分块累加（O7b）；TMA 化 Q/K/V 仍排在后面。

### 14j.7 原始输出

`src/fp16/fa_bwd_fp16_wgmma2_smoke.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o17_{s4096}.out.txt`、`..._mma_onefile_o17_s512.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o17_ncu_wg2_s4096.out.txt`、`..._o17_ncu_o9b_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_o17_fa3_te_baseline.out.txt`。

---

## 14k. O17b：BM=256 / 4 warpgroups（跨 wg 归约再砍半）—— **负结果 + 资源墙**

### 14k.1 动机

O17（§14j）把 BM 从 64 提到 128、用 2 个 warpgroup 把 dK/dV 的跨 CTA `red` 精确砍半
（ncu `lts__t_sectors_op_red` 102.2M→51.9M），但仍占 L2 扇区 ~72.6% ⇒ main 仍是 L2 原子
字节数 bound。自然下一步是把 **BM 再翻倍到 256**：每个 KV 元素只被 `nblk/4` 个 CTA 贡献
⇒ red 预期再砍半。O17b = `fa_bwd_fp16_wgmma4_kernel<HD,SEQ>`（4 个 warpgroup = 512 线程）。

### 14k.2 先做资源账（本项的关键，结论是墙）

| 资源 | 预算 | O17b 需求 | 结果 |
|---|---|---|---|
| 寄存器 | 512 线程 @1 CTA/SM ⇒ `65536/512 = **128 regs/线程**` | `dqacc[2][8][4]=64` + `sacc/dpacc` 各 32（m64n64 累加器，必须同时存活到 wait0）| **128 碰顶 + spill** |
| smem | 本卡上限 227KB | Q 64KB + dO 64KB + K 16KB + V 16KB + P 32KB + dS 32KB = **224KB（K/V 只能单缓冲）** | 1 CTA/SM，丢掉 K/V 流水 |

* **寄存器墙是硬约束**：`dqacc`（dQ 寄存器累加，O7 起的必要手段，砍掉它会让 dQ 的 red
  按 ntile 爆炸）就占 64；GEMM1/2 的两条 wgmma 累加器在 `wait0` 后都要读，占 64；
  两者相加已经等于 128 的上限，寻址/循环变量必然溢出到 local memory。
* **smem 墙**：BM=256 要求 Q/dO 各 64KB、P/dS 各 32KB；K/T 的 SW128 tile 只 16KB，
  但 K/V 双缓冲 +16KB 就 240KB 超限 ⇒ 只能单缓冲，靠「V 在 GEMM2 后、K 在 GEMM5 后
  各自 cp.async 预取下一 tile」部分弥补流水。

### 14k.3 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

新增 `fa_bwd_fp16_wgmma4_kernel<HD,SEQ>`（`#ifdef FA_WGMMA`，仅 HD=128）：
* 4 个 wg 各持自己 64 行 Q/dO，各自算 GEMM1/2 并把 P/dS 写进共享 `[256][BN=64]` SW128 tile；
* **GEMM3（dV）只由 wg0、GEMM4（dK）只由 wg1 对全 BM=256 归约**（s=0..15 进同一累加器）；
  GEMM5（dQ）4 个 wg 各自寄存器累加、无跨 CTA 原子；K/V 单缓冲 + 后段 cp.async 预取；
* 两档：**ovlp**（默认，GEMM1/2 一起发、`sacc/dpacc` 同时存活）与 **SEQ**（串行 GEMM1/2、
  把 half P 写 smem 后**读回**再算 dS，省掉 `sacc`/`dpacc` 的共存）。SEQ 的读回是
  `__half` 精度 ⇒ 会让 dS 多一层 fp16 舍入（见 §14k.4），**默认用 ovlp**。
* host 加 `--wg4` / `--wg4seq=1` 与 `[O17b A/B]`（同 session 对比 mma/O17/O17b-ovlp/O17b-seq）。
  默认路径完全不变。

### 14k.4 数值（ours-vs-ref，fp16 causal）

* **ovlp 版与 O17/O9b 在 fp16 噪声内**：`max|diff|` wg4-vs-wg2 dq/dk/dv =
  0 / 6.1e-5 / 8.8e-5（S=4096；仅 atomic 次序），vs mma 0 / 6.5e-5 / 9.5e-5。
* **SEQ 版有系统性偏差**：wg4seq-vs-wg4 dk ≈ **2.6e-2～7.3e-2**（S=512/S4096），
  dq ≈ 2.0e-3 —— 因为 dS 改用了「读回的 half P」。**结论：SEQ 不能作为正确路径**。
* 默认（mma）路径 `vs ref` 逐位不变（S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3）。

### 14k.5 性能（同 session A/B，CUDA event，main-only，ms）

| shape | O17 (BM128) | O17b ovlp | O17b seq | vs O17 |
|---|---|---|---|---|
| MHA S=512 | 0.0522 | 0.0958 | 0.0851 | **0.55× / 0.61×** |
| GQA kv4 S1024 | 0.1779 | 0.1919 | 0.1710 | 0.93× / **1.04×** |
| MQA kv1 S1024 | 0.2869 | 0.3429 | 0.3046 | 0.84× / 0.94× |
| MHA S=4096 | 0.9946 | 1.1956 | 1.0528 | 0.83× / 0.95× |

**O17b 在所有 shape 上都没有跑赢 O17**（S=512 因 grid 减半最差；大 S 上 red 虽减半，
但 spill/单缓冲把收益吃光）。唯一 >1 的是 SEQ 在 GQA kv4 的 1.04×，但它有 §14k.4 的精度损失。

### 14k.6 ncu（main，S=4096，同 session，O17 vs O17b）

| 指标 | O17 (wgmma2) | O17b merged | O17b seq |
|---|---|---|---|
| Duration | 0.997 ms | 1.20 ms | 1.05 ms |
| **`lts__t_sectors_op_red`** | 51.90 M | **26.74 M（0.515×）** | **26.74 M（0.515×）** |
| `..._op_read` | 18.03 M | 35.14 M | 28.70 M |
| `..._op_write` | 1.57 M | **34.04 M（21.6×）** | 14.53 M |
| `sm__inst_executed` | 260.5 M | 231.2 M | 229.6 M |
| L2 Cache Throughput | 54.66% | 40.74% | — |
| L1/TEX Throughput | 48.78% | 51.76% | — |
| Compute (SM) | 27.48% | 20.29% | — |
| regs / smem | 200 / 149.5KB | 128 / 224KB | 128 / 224KB |
| achieved occ | 12.41% (2CTA/SM) | 24.88% (1CTA/SM) | — |
| Waves | 3.88 | 1.94 | — |

**机制假设被证实、代价也被量化**：
1. `red` **精确减半**（51.90M→26.74M = 0.515×），与「每个 KV 元素贡献 CTA 数减半」完全一致；
2. 但 **`write` 扇区从 1.57M 暴涨到 14.5–34M** —— ptxas 报 `fa_bwd_fp16_wgmma4_kernel`
   用了 **128 regs + 溢出**（merged 468B/376B、seq 148B/156B spill stores/loads），
   ncu 显示 **local memory 占 L2 扇区 ~48%**（每次 spill 只用到 1/32 B/sector，极不划算）；
3. `pds_load_sw128`（seq）虽把静态 spill 从 468B 降到 148B，但**动态** spill 流量仍高
   （write 14.5M），说明 `dqacc`/累加器在循环里的活跃压力本身就撑爆 128-reg 预算；
4. 叠加 K/V 单缓冲与「GEMM3/4 只有 2 个 wg 在算」的欠并行，净效果是 Duration 变长。

### 14k.7 结论与下一步

**O17b（BM=256/4wg）在本卡上是负结果，根因是寄存器文件墙**：512 线程把每线程上限压到
128 regs，而 dQ 寄存器累加器（64）+ 两条 GEMM 的 wgmma 累加器（64）已经吃满，必然 spill，
spill 的 local 流量（uncoalesced）反而比省下的 red 更贵。**继续沿「放大 BM」这条路走不通**，
除非先解决「dQ 累加器不占寄存器」或「用 TMA/更深的算子融合把 Q/dO 与 P/dS 的驻留量压下来」。

据此把 fp16/bf16 main 的下一步调整为：
1. **O7b（dK/dV 确定性分块累加）**：把跨 CTA `red` 换成「CTA 局部累加 + 非原子写 + 二次归约」，
   直接消掉 L2 原子而不是靠放大 BM；顺带拿到确定性反向。red 字节不减，但原子 RMW 换成
   普通写，且可让每次只写一次。
2. fp8 侧同构的跨 wg 归约（fp8 main 的 L2 red 亦是墙，且 fp8 是 1 字节 operand，smem 更省，
   4wg 的寄存器压力也小一档）。
3. 也可回到 O17（BM=128）做**微优化**：`BN=128` 减半 tile 数/barrier（smem 恰好 224KB、2 wg
   寄存器够用），收益待测。

### 14k.8 原始输出

`src/fp16/fa_bwd_fp16_o17b_sweep.out.txt`（单/两文件 ×5 shape 计时+逐元素对拍）、
`src/fp16/fa_bwd_fp16_main_o17_ncu_s4096_samesession.out.txt`、
`src/fp16/fa_bwd_fp16_main_o17b_ncu_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_main_o17_o17b_ncu_red_s4096.out.txt`。

---

## 14l. O17-2：把 O17 的 GEMM3/GEMM4 拆分到两个 warpgroup（负载再平衡）

### 14l.1 动机（O17 的张量工作量是 3:1 失衡的）

O17（§14j）用 BM=128 + 2 warpgroups 把 dK/dV 的跨 CTA `red` 砍半，但它的 phase B 里
**只有 wg0 串行做 GEMM3(dV) 和 GEMM4(dK)**（各 `nh=0,1` 两遍、每遍 `commit/wait0` + 一次
`red_add2` epilogue，共 4 条串行 red 链），而 **wg1 在 phase B 只做自己的 GEMM5(dQ)**。
按每 tile 的 wgmma 条数记：wg0 = GEMM3(2×8) + GEMM4(2×8) + GEMM5(2×8) = **48**，
wg1 = GEMM5 一路 = **16** ⇒ **张量工作量 3:1**，wg1 在 GEMM3/GEMM4 期间基本空等（ncu
`barrier` stall 高）。这正是 O17b（BM=256）里已经采用、但 BM=128 版没做的「dV/dK 分给两个 wg」。

### 14l.2 关键手段：按输出拆分，归约语义不变

dV 与 dK 是**两个独立的输出**（不同张量），但都沿 m（Q 行）方向对全 BM=128 归约。
把 **GEMM3→wg0、GEMM4→wg1**，各自仍把两个 m64 半（`s=0..7`）连续喂进**同一个**
`wgmma.m64n64k16` 累加器 ⇒ **每个 KV 元素仍只 `red` 一次**（red 字节不变），只是发射的
warpgroup 不同。GEMM5（dQ）仍由每个 wg 算自己 64 行（A=本 wg 的 dS 行块），无法也不需拆分。

- **正确性**：dV/dK 的 wgmma 归约次序（`s=0..7` 的顺序、累加器布局）与 O17 完全相同，
  只是换了发射方；`red_add2` 的 (r,c) 映射与 O17 逐字相同。故**结果值逐位相同**，
  仅跨 CTA `atomicAdd` 的交错次序可能微变（`max|diff|` dk/dv ~3e-5~9e-4、dq = 0）。
- **实现**：`fa_bwd_fp16_wgmma2_kernel<HD, bool SPLIT=true>`，`if constexpr (SPLIT)` 分支里
  `if (wg==0) {dK...} else {dV...}`；`SPLIT=false` 保留 O17 原版用于同 session A/B。host 加
  `--wg2split=0/1`（默认 1）。单/两文件 device 代码逐字一致（`sync_onefile_device.py` 核对
  `identical: True`），bf16 逐字 dtype 同构。

### 14l.3 性能（同 session A/B，CUDA event，main-only，ms）

| shape | O17（wg0 串行 dV+dK） | **O17-2（wg0=dV,wg1=dK）** | 比 |
|---|---|---|---|
| MHA S=512 | 0.0520 | **0.0518** | 1.004× |
| MHA S=4096 | 1.0026 | **0.9895** | **1.013×**（138.9 TF） |
| GQA kv4 S1024 | 0.1800 | **0.1764** | **1.021×**（97.4 TF） |
| GQA kv8 S1024 | 0.2050 | **0.2035** | 1.007× |
| MQA kv1 S1024 | 0.2856 | **0.2843** | 1.005× |

端到端（O17-2）：S=512 **0.113 ms（19.0 TF）**、S=4096 **1.946 ms（70.6 TF）**。
数值与 O5~O17 历史值**逐位一致**（S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、
GQA kv4 2.134/3.305/3.850e-3、kv8 2.008/2.931/3.891e-3、MQA kv1 2.292/7.934/7.517e-3）。

### 14l.4 ncu（main，S=4096，同 binary `--wg2split` 0/1）

| 指标 | O17（no-split） | **O17-2（split）** |
|---|---|---|
| `gpu__time_duration` | 997.7 µs | **982.4 µs** |
| `lts__t_sectors_op_red` | 51,904,512 | **51,904,512（不变）** |
| `read` 扇区 | 17,967,585 | 18,014,141 |
| stall `barrier` | **1.61** | **0.46（−3.5×）** |
| stall `wait` | 1.18 | 1.19 |
| stall `long_scoreboard` | 0.40 | 0.75 |
| stall `short_scoreboard` | 0.26 | 0.40 |
| regs / smem / occ / Waves | 200 / 149.5KB / 12.4% / 3.88 | 200 / 148.5KB / 12.5% / 3.88 |

**机制结论**：`red` 字节完全不变（51.9M）——说明收益**不是**来自减少原子，而是来自
**把 wg0 的 4 条串行 red 链拆到两个 wg、消掉 wg1 在 GEMM3/GEMM4 期间的 barrier 空等**
（`barrier` 1.61→0.46）。代价是两 wg 并发发 red 让 `long/short_scoreboard` 略升，净收益
1.3%（S4096）。墙仍是 **L2 red（~55%）+ `wait`**，与本项正交。

### 14l.5 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

FA3 MHA S=4096 **0.3247ms/846TF**、TE 0.4451/618、FA2 0.7275/378；GQA kv4 FA3 0.0826/416。
ours 端到端 S=4096 1.946ms ⇒ 时间为 FA3 的 **5.99×**（O17 5.97×，ncu 单 kernel 1.5% 已在
端到端被 preprocess/convert 稀释）。

### 14l.6 原始输出

`src/fp16/fa_bwd_fp16_mma_main_o17_2_sweep.out.txt`（两文件 ×5 shape A/B + 对拍）、
`src/fp16/fa_bwd_fp16_mma_onefile_o17_2_s4096.out.txt`（单文件）、
`src/fp16/fa_bwd_fp16_mma_main_o17_2_ncu_wg2_s4096.out.txt`（`--set full`）、
`src/fp16/fa_bwd_fp16_mma_main_o17_2_ncu_red_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o17_nosplit_ncu_red_s4096.out.txt`（A/B 基线）、
`src/fa_bwd_o17_2_fa3_te_baseline_fp16.out.txt`。

---

## 14m. O18：BN=128 版 wgmma2（tile 数减半，MHA main 1.029×；GQA/MQA 中性）

### 14m.1 动机（§14k.7 列出的「BN=128 微优化」）

O17（§14j）/O17-2（§14l）后，`fa_bwd_fp16_wgmma2_kernel`（BM=128、BN=64、2 warpgroups）把 dK/dV
的跨 CTA `red` 砍半（102.2M→51.9M 扇区），但 ncu 仍显示 **1 CTA/SM（8 warps、occupancy 12.5%）、
`sms__` 无饱和资源、`wait`/`barrier` 主导 ⇒ 延迟受限**；`red` 仍占 L2 扇区 ~72.6%（字节未再减）。
「放大 BM」（O17b）已被寄存器墙证伪。§14k.7 item 3 留下的实验是 **BN 64→128**：

* 每个 CTA 的 KV-tile 数**减半** ⇒ `__syncthreads` / `cp.async.wait_group` / wgmma
  `commit_group`+`wait0` 的**固定序列减半**（延迟受限下这直接摊薄每单位工作的串行开销）；
* GEMM1/2 从 `m64n64k16` 换成 **`m64n128k16`**（一条指令算两倍），发射/依赖链减半；
* 代价是 smem 145→**225KB**（P/dS 各 32KB、K/V 各 32KB），仍 **1 CTA/SM**；寄存器 200→230（0 spill）。
* **注意（预期管理）**：BN 不改变 dK/dV 的归约结构 ⇒ **`red` 字节完全不变**，故这不是「打掉
  L2 原子墙」的杠杆，而是把每-tile 的固定开销摊薄。

### 14m.2 先冒烟（新增 `fa_bwd_fp16_wgmma2b_smoke.cu`，逐位 PASS）

`m64n128k16` 的累加器布局（每线程 64 个 fp32：行 `wid*16+lane/4 (+8)`、列 `j*8+(lane&3)*2 (+1)`、
`j=0..15`）与转置描述符是新的风险点，先写最小冒烟逐位验证四条 GEMM：

| 冒烟项 | GEMM | 描述符 | `max_abs` |
|---|---|---|---|
| (1) `dV=Pᵀ·dO` | m64n128（M=BN=128 分 2 个 m64 半） | A=P¹转置(MN)、B=dO 转置(MN) | **0.000e+00** |
| (2) `dK=dSᵀ·Q` | 同上 | A=dS 转置(MN)、B=Q 转置(MN) | **0.000e+00** |
| (3) `dQ=dS·K` | m64n128（每 wg 64 行） | A=dS K-major、B=K 转置(MN) | **0.000e+00** |
| (4) `S=Q·Kᵀ` | m64n128 | A/B 均 K-major | **0.000e+00** |

关键推导：GEMM3/4 的输出 M = **BN（KV 行）= 存储列**，故 m64 半 `mh=1` 的转置描述符基址偏移是
**存储列 64 的 SW128 字节（W=128 时为 `kg=1` 的 atom，`+1024B`）**，而不是行方向的
`mh*(64/8)*(BN/64)*1024`（后者是 GEMM5 的 K-major A 用）。冒烟确认 `+1024B` 正确。
原始输出 `src/fp16/fa_bwd_fp16_wgmma2b_smoke.out.txt`。

### 14m.3 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

新增 `fa_bwd_fp16_wgmma2b_kernel<HD=128, SPLIT>`（`#ifdef FA_WGMMA`，仅 HD=128、256 线程）：

* 新增 `wgmma_m64n128k16_t<TA,TB>`（64 累加器）与 `wgmma_mn128_issue`（GEMM1/2，A/B K-major）；
* P/dS tile 变 `[128][128]`，softmax epilogue 列组 `j=0..15`（`pval[16][4]`）；
* GEMM3(dV)→wg0、GEMM4(dK)→wg1（同 O17-2），各自对全 BM=128 归约（`s=0..7`），每个输出
  `[BN=128][HD=128]` 分 2 个 m64 半 `mh`（描述符 `+mh*1024`），每 KV 元素仍只 `red` 一次；
* GEMM5(dQ) 每 wg 一条 `m64n128`，寄存器累加 `dqacc[16][4]`；
* K 双缓冲、V 单缓冲后段预取（同 O17）；host 加 `--wg2bn[=0/1]` 与 `[O18 A/B]`（含 `max|diff|`）。

`ptxas`：**230 regs、0 spill**；动态 smem **230400B（225KB）**，1 CTA/SM。默认路径不变（需 `--wg2bn`）。

### 14m.4 数值（ours-vs-ref，fp16 causal，max_abs）

| shape | dq | dk | dv | `max\|diff\|` wg2b-vs-wg2 (dq/dk/dv) |
|---|---|---|---|---|
| MHA S=512 | 1.671e-3 | 1.771e-3 | 1.899e-3 | 2.98e-7 / 2.10e-5 / 3.05e-5 |
| MHA S=4096 | 1.883e-3 | 1.734e-3 | 1.966e-3 | 2.38e-7 / 3.05e-5 / 3.05e-5 |
| GQA h32kv4 S=1024 | 2.134e-3 | 3.305e-3 | 3.850e-3 | 2.98e-7 / 5.34e-5 / 1.07e-4 |
| MQA h64kv1 S=1024 | 2.292e-3 | 7.934e-3 | 7.517e-3 | 3.58e-7 / 1.37e-4 / 2.44e-4 |

**vs ref 与 O5~O17-2 历史逐位一致**；`dq` 近逐位（dQ 无跨 CTA 原子），dk/dv 只差 atomic 次序。
单/两文件逐指标一致。

### 14m.5 性能（同 session A/B，CUDA event，main-only，ms）

| shape | O17 wg2(BN64) | **O18 wg2b(BN128,split)** | vs O17 | wg2b 串行 |
|---|---|---|---|---|
| MHA S=512 | 0.0513（41.8 TF） | **0.0499（43.0 TF）** | **1.028×** | 0.0505（1.017×） |
| MHA S=4096 | 0.9895（138.9 TF） | **0.9618（142.9 TF）** | **1.029×** | 0.9888（1.001×） |
| GQA h32kv4 S=1024 | 0.1773（96.9 TF） | 0.1782（96.4 TF） | 0.995× | 0.1819（0.975×） |
| MQA h64kv1 S=1024 | 0.2797（122.9 TF） | 0.2812（122.2 TF） | 0.994× | 0.2853（0.980×） |

⇒ **MHA（S=512/4096）稳定 +2.8~2.9%**；GQA/MQA 略负（-0.5~-0.6%，小网格、Hkv 广播下
tile 变大反而降低并行度余量）。单文件版本 S=4096 同测 **1.021×**（session 噪声内一致）。
**结论：BN=128 是 MHA 的小杠杆，GQA/MQA 保持 O17。**

### 14m.6 ncu（main，S=4096，同 session A/B：O17-2 vs O18）

| 指标 | O17-2 (BN=64) | **O18 (BN=128)** |
|---|---|---|
| Duration | 982.4 µs | **951.1 µs** |
| `lts__t_sectors_op_red` | 51,904,512 | **51,904,512（不变）** |
| `lts__t_sectors_op_read` | 18,012,681 | 18,281,020 |
| `lts__t_sectors_op_write` | 1,576,020 | 1,576,566 |
| L2 Cache Throughput | 55.44% | **57.21%** |
| L1/TEX Throughput | 49.46% | **44.34%** |
| DRAM | 6.57% | 6.77% |
| Compute (SM) | 27.75% | 23.60% |
| regs / smem | 200 / 148.48KB | **230 / 224.0KB** |
| achieved occ | 12.48%（1 CTA/SM） | **12.50%（1 CTA/SM）** |
| stall（wait/barrier/long/short） | 1.19 / 0.46 / 0.75 / 0.40 | **1.25 / 0.93 / 0.60 / 0.43** |
| No Eligible | 63.0% | 68.9% |

**机制核对**：`red` **逐字节不变**（证实 BN 不改归约结构）；Duration −3.2% 来自 tile 数减半（L1/TEX
44.3%↓、Compute 23.6%↓），而 **`barrier` 反而 0.46→0.93**（tile 内 two-wg 同步的相对权重变大）。
墙仍是 **L2（57.2%，red 占 ~72%）+ `wait` + 1 CTA/SM 低 occupancy** ⇒ **真正的杠杆只有
「减 red 字节」或「提 occupancy」**，二者分别对应 O7b/跨 wg 再合（BM 受限）与 TMA/降 smem。

### 14m.7 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

| shape | FA2.7.4 | **FA3（SM90）** | TE2.14 |
|---|---|---|---|
| MHA S=4096 | 0.7295ms / 377 TF | **0.3253ms / 845 TF** | 0.4457ms / 617 TF |
| GQA kv4 S=1024 | 0.1584 / 217 | **0.0825 / 416** | 0.1127 / 305 |
| MQA kv1 S=1024 | 0.2666 / 258 | **0.1566 / 439** | 0.2149 / 320 |

O18 main S=4096 = **0.9618ms / 142.9 TF**；端到端（preprocess 0.343 + main 0.962 + convert
0.108 ≈ 1.41ms）≈ FA3 整条反向的 **4.35×**（时间；O17-2 时 4.4×）。

### 14m.8 原始输出

`src/fp16/fa_bwd_fp16_wgmma2b_smoke.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o18_s512.out.txt`、`..._o18_sweep.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o18_s512.out.txt`、`..._mma_onefile_o18_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o18_ncu_wg2b_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_o18_fa3_te_baseline.out.txt`。

---

## 14n. O23：Hopper 路径默认化（主 kernel wgmma2/wgmma2b + LSE wgmma，端到端 1.08–1.43×）

### 14n.1 动机（最便宜的一步：最优路径一直是 opt-in）

O9a/O9b/O17/O18 把 fp16 反向的 Hopper 快路做到了 main S=4096 **142.9 TF**（§14g/§14j/§14m），
但它一直是**显式开关**——必须传 `--wg2`/`--wg2bn` 才生效；不传就退回慢 1.5× 的 `mma` 路径
（`run_main` 里 `wg2_sel` 默认 0）。fp8 侧早在 **O22** 就把 `-DFA_WGMMA` 构建下的 LSE + 主 kernel
GEMM1/2 的 wgmma **默认打开**（ROADMAP 第六十二轮），fp16/bf16 一直没做这一步。本项补齐，
让「用 Hopper 快路」成为默认行为，`--wg2=0 --wg2bn=0` 仍可退回 mma 做同 session A/B。

### 14n.2 改动（单/两文件 host 逐字一致；device 代码未动）

`fa_bwd_fp16_mma_{main.cu, onefile.cu}`（bf16 同构）：

- 把 `--wg2`/`--wg2bn`/`--lsewgm` 的解析加一个 `wg_forced`/`lse_forced` 标记（用户显式指定则
  尊重），并在 `run_main` 前加自动段：

```cpp
#ifdef FA_WGMMA
  if (!wg_forced && D == 128) { if (S >= 4096) wg2bn_sel = 1; else wg2_sel = 1; }
  if (!lse_forced && D == 128 && causal) lse_wgm = 1;   // LSE 也走 wgmma
#endif
```

  * `S>=4096` 选 **BN=128（O18）**，其余选 **BN=64（O17）**——与 O18 实测一致（wg2b 的明确收益
    只在大 S；GQA/MQA 中性，保持 O17 不引入回归）。
  * `lse_wgm` 只在 causal 生效（`run_pre` 里 `causal && lse_wgm`）；非 causal 自动落回 O8 原版。
  * **非 `FA_WGMMA`（纯 `sm_90`）构建完全不变**：`wg2_sel/wg2bn_sel` 恒 0、`lse_wgm` 恒 0，
    `--wg2=0 --wg2bn=0`、`--lsewgm=0` 提供回归对照。
  * 新增 `[O23] main backend = … | lse = …` 打印，便于日志确认默认档。

### 14n.3 数值（ours-vs-ref，fp16 causal，max_abs）—— 与历史逐位一致

S512 1.671/1.771/1.899e-3；S4096 1.883/1.734/1.966e-3；GQA kv4 S1024 2.134/3.305/3.850e-3；
MQA kv1 S1024 2.292/7.934/7.517e-3；MLA S512H4 2.516/2.916/1.724e-3。单文件与两文件**逐指标一致**
（S4096 total 1.3609 vs 1.3641ms）。`--wg2=0 --wg2bn=0` 的 mma 结果与历史 mma 逐位相同。

### 14n.4 性能（同 session 端到端 total，CUDA event，ms）

| shape | 旧默认（mma） | **新默认** | × | 主 kernel 后端 | LSE |
|---|---|---|---|---|---|
| MHA S512 | 0.1132 | **0.1045** | 1.08 | wgmma2(BN=64) | wgmma |
| GQA kv4 S1024 | 0.3751 | **0.2864** | 1.31 | wgmma2 | wgmma |
| MQA kv1 S1024 | 0.6024 | **0.4425** | 1.36 | wgmma2 | wgmma |
| MHA S4096 | 1.9450 | **1.3641** | **1.43** | wgmma2b(BN=128) | wgmma |
| MLA S512 D512 | 0.4367 | 0.4349 | 1.00 | mma（D=512 无 wgmma2） | mma |

main-only 的 A/B（程序内 `[O17 A/B]`/`[O18 A/B]`）：S4096 mma 1.5111 vs O17 0.9959（1.52×）vs
O18 0.9601（**144.1 TF**）；S512 mma 0.0568 vs O17 0.0506（1.12×）；GQA kv4 mma 0.2656 vs
O17 0.1806（1.47×）；MQA kv1 mma 0.4337 vs O17 0.2831（1.53×）。LSE wgmma 再叠加约 1.3%
（S4096 preprocess 0.3446→0.3270）。

### 14n.5 ncu（默认路径 = `fa_bwd_fp16_wgmma2b_kernel<128,1>`，S=4096，`-c 1`）

`lts__t_sectors_op_red=51,904,512`（与 O18 逐字节相同）、255 regs、231.42KB smem、achieved occ
12.48%、L2 56.58%、L1/TEX 33.41%、Compute 23.55%、DRAM 6.70%；stall `wait 1.25 +
long_scoreboard 0.60 + barrier 0.93`。即 **默认档就是 O17/O18 的墙：L2 的 dK/dV 跨 CTA `red`
（占 L2 扇区 ~72%）+ 1 CTA/SM**，与 §14j/§14m 结论一致——本项只改「默认选谁」，未改机制。

### 14n.6 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`，FA2/FA3/TE 三列）

| shape | FA2.7.4 | **FA3（SM90）** | TE2.14 | ours total | ours/FA3 | ours/TE |
|---|---|---|---|---|---|---|
| MHA S=4096 | 0.7301ms / 376 | **0.3251 / 846** | 0.4444 / 619 | 1.3641ms | **4.20×** | 3.07× |
| GQA kv4 S=1024 | 0.1600 / 215 | **0.0831 / 413** | 0.1124 / 306 | 0.2864 | 3.45× | 2.55× |
| MQA kv1 S=1024 | 0.2659 / 258 | **0.1566 / 439** | 0.2150 / 320 | 0.4425 | 2.83× | 2.06× |

（ours total 的 TFLOPS 用 harness 的 `4BS²H(D+Dv)` 口径：S4096 201.5 TF、GQA 120 TF、MQA 155 TF
⇒ 为 FA3 的 23.8%/29.0%/35.4%。）O18 时 ours/FA3 是 4.30×，本项默认化后为 **4.20×**（同时把
「不传 flag 的用户」从 593%（mma）直接带到默认快路）。

### 14n.7 原始输出

`src/fp16/fa_bwd_fp16_mma_main_o23_{s4096,shapes}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o23b_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o23b_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o23_ncu_default_s4096.out.txt`、
`src/fa_bwd_o23_default_ab.out.txt`（`--wg2=0 --wg2bn=0` 的旧默认端到端）、
`src/fa_bwd_o23_shapes_final.out.txt`、`src/fa_bwd_o23_fa3_te_baseline_{fp16,bf16}.out.txt`。

---

## 14o. O7b：确定性 dK/dV（partial + 二次归约）——机制成立、数值可复现，但净负（S4096 0.81×）

### 14o.1 动机

§14i/§14j 把 fp16 main 的墙钉在 **dK/dV 的跨 CTA `atomicAdd`**：O17 后 `lts__t_sectors_op_red`
仍 = 51,904,512（S=4096，占余下 L2 扇区 ~72.6%）。atomicAdd 在 L2 是 **read-modify-write**；
O17b（放大 BM）与 O7c（float4 归约）都已证伪「减 red 事务数/放大归约宽度」。O7b 换一条路：
**完全不用原子**——每个 (Q 块, KV 行) 的贡献写进带 `mblk` 下标的独立 partial 缓冲（每元素只被
本 CTA 写一次，非原子覆盖写），再由一个 reduce kernel 按 `mblk` 升序求和。副作用：结果与执行
顺序无关 ⇒ **确定性反向**（同输入同输出逐位可复现）。

### 14o.2 实现（单/两文件 device 代码逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

- `fa_bwd_fp16_wgmma2b_kernel<HD, SPLIT, DET=false>` 加 `DET` 模板开关与 `dk_part/dv_part/nblk`
  参数（默认 `nullptr`，`DET=false` 时行为不变）。`DET=true` 时四处 dK/dV epilogue 从
  `red_add2(...)` 改为 `dkv_det_store(part, a, b)`（一次 `float2` 覆盖写）：
  `part[((b*H + h)*nblk + mblk)*S*HD + jg*HD + c]`。
- 新增 `dkv_reduce_kernel<HD>`：对每个输出元素 `(b,hkv,jg,c)` 求和 `mblk ∈ [jg/128, nblk)`
  与 **Q 头广播组** `h ∈ [hkv*G, (hkv+1)*G)`（`G=H/Hkv`），写入 `dk_acc/dv_acc`。
- **踩坑（GQA race）**：partial 初版按 `hkv` 索引；GQA/MQA 下同一 KV 头有 `H/Hkv` 个 Q 头，
  同 `(mblk,hkv)` 的多个 `h` 会互相覆盖 ⇒ S=4096（MHA）对、**GQA kv4 实测 `max|diff|=6.3`（错）**。
  改成按 Q 头 `h` 分片、reduce 里对广播组求和后修复（MHA 时 `G=1` 退化，逐位不变）。
- host：`--det=1` 开 A/B（仅 `#ifdef FA_WGMMA && D==128`）；分配 `B*H*nblk*S*D` 的 partial（S4096
  ≈ 2.15GB，跑完释放），不 memset dk/dv（reduce 全覆盖），只计时 main+reduce。默认路径不受影响。

### 14o.3 数值（fp16 causal，ours-vs-ref max_abs；DET 可复现性）

| shape | ours-vs-ref dq/dk/dv | DET 跑两遍 bitwise-diff dk/dv | DET-vs-atomic dk/dv |
|---|---|---|---|
| S512 MHA | 1.671/1.771/1.899e-3 | **0 / 0** | 2.38e-7 / 4.77e-7 |
| S1024 GQA kv4 | 2.134/3.305/3.850e-3 | **0 / 0** | 2.86e-6 / 3.81e-6 |
| S4096 MHA | 1.883/1.734/1.966e-3 | **0 / 0** | 7.15e-7 / 9.54e-7 |

- **DET 逐位可复现**（两次运行差 = 0），与 atomic 路径只差浮点加法次序（≤3e-6），与历史对拍值一致。
- 默认路径（DET 关）数值不变（S4096 1.883/1.734/1.966e-3）。

### 14o.4 性能（同 session A/B，CUDA event，main[+reduce] ms）

| shape | atomic（默认） | DET（main+reduce） | 比值 |
|---|---|---|---|
| S512 MHA | 0.0551 | **0.0538** | **1.023×** |
| S1024 GQA kv4 | 0.1865 | 0.2100 | 0.888× |
| S4096 MHA | 0.9598（143.2 TF） | 1.1849 | **0.810×** |

**机制成立但净负**：S=512 时 partial（~67MB）基本落在 L2，DET 反而略快；S=4096 partial 达
1.66GB，二次归约变成一趟纯 DRAM 带宽扫描，把收益吃回去。

### 14o.5 ncu（S=4096，同 session，`-c 1`）

| 指标 | atomic main | DET main | reduce |
|---|---|---|---|
| `lts__t_sectors_op_red` | 51,904,512 | **0** | — |
| `lts__t_sectors_op_read` | 18.36 M | 18.83 M | 51.91 M |
| `lts__t_sectors_op_write` | 1.574 M | 53.48 M | 3.15 M |
| Duration | 958.6 µs | **802.0 µs** | 384.9 µs |
| DRAM Throughput | — | — | **90.4%（带宽上限）** |
| occupancy | 12.48% | 12.47% | — |

- **主 kernel 因为去掉 atomic RMW 反而快 1.20×**（958.6→802.0µs；red 51.9M→0，write 1.57M→53.5M）。
- **reduce 是纯 DRAM 带宽 bound（90.4%）**，384.9µs；`main+reduce=1.185ms > atomic 0.960ms` ⇒ 净负。
- 结论：**O7b 的确定性值得保留（`--det=1` opt-in），但作为性能杠杆无效**——它把 L2 原子墙换成
  一趟 DRAM 归约墙。要真正消 red 只能靠 **减少每个 KV 元素的贡献 CTA 数**（BM 放大已证伪 §14k）
  或 **cluster 分布式归约**（不落全局内存），转 backlog。

### 14o.6 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

MHA S4096：FA3 **0.3249ms/846TF**、TE 0.4449/618、FA2 0.7292/377；GQA kv4 S1024 FA3 0.0825/416。
默认路径端到端仍为 FA3 的 ~4.2×（O23）；O7b（DET）在主 kernel 上更接近（main 802µs），
但被二次归约拖回。**默认路径不受影响**。

### 14o.7 原始输出

`src/fp16/fa_bwd_fp16_main_o7b_det_{s512,s4096,gqa_kv4}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o7b_det_{s512,s4096}.out.txt`、
`src/fp16/fa_bwd_fp16_main_o7b_ncu_{atomic_s4096,det_s4096,reduce_s4096}.out.txt`、
`src/fa_bwd_o7b_fa3_te_baseline_fp16.out.txt`。

## 14p. O24：preprocess `delta` 向量化 + dQ 直写 fp16（消 convert 的 dQ 一趟）

### 14p.1 动机（main 已是硬墙，回头清「非 main」的固定开销）

O23 把 Hopper 快路默认化后，fp16 MHA S=4096 端到端 = **preprocess 0.327 + main 0.943 +
convert 0.096 ≈ 1.366 ms**。main 的墙是 L2 的 dK/dV 跨 CTA `red`（~72%，见 §14j/§14k），
已无「搬运/等待」类杠杆；但 **preprocess（24%）与 convert（7%）里还有两处纯工程浪费**：

1. **`delta_kernel`（D = rowsum(dO∘O)）**：旧实现**每个 (s,h,b) 一行一个 128 线程 CTA**，
   用 `__shared__` 树归约 + `log2(THREADS)` 次 `__syncthreads`。对 HD=128 每行只有 128 个乘加，
   block/同步开销远大于计算（ncu：S=4096 **Compute 72% / L1TEX 74% / DRAM 25%、42.5µs**）。
2. **`convert_kernel` 的 dQ 一趟**：D=128 时 dQ 由主 kernel **寄存器累加后唯一拥有**
   （无跨 CTA 原子，见 §10），却仍先写 fp32 `dq_acc`、再由 convert 读回转 fp16 写 `dq`——
   多一趟 `n` 个 float 的读 + `n` 个 half 的写。

### 14p.2 改动（单/两文件 device 与 host 逐字一致）

* **`delta_warp_kernel<HD>`**（对齐 fp8 O14 的 `quantize_row_warp_kernel` 思路）：
  **每 warp 一行**，lane 沿 HD 以 `__half2`（4B）coalesced 读（每步 warp 读 32×4=128B），
  `__shfl_xor_sync` 五级树归约，**无 smem / 无 barrier**；grid-stride 覆盖任意行数。
  `delta_warp_sel`（`--deltawarp=0/1`，默认 1）做同 binary A/B；HD=512（MLA）同样走新版。
* **dQ 直写 fp16**：给 `fa_bwd_fp16_wgmma2_kernel` / `wgmma2b_kernel` 加 `__half* dq_h=nullptr`；
  非空时 dQ epilogue 直接 `__floats2half2_rn` 写 `dq`（与 convert 的 `__float2half` 同为 RN，
  **数值逐位不变**）。host 在 D==128 且选中 wgmma2/wgmma2b 时传 `dq` 并令 `convert` 的 `n_q=0`
  （`--dqdirect=0` 关）。其余路径（mma/wgmma/wgmma4/MLA RMW）仍写 `fp32 dq_acc`、convert 照旧。
* 单文件由 `sync_onefile_device.py` 同步 device、host 段与两文件同步重建。
* **数值**：新旧 `dq/dk/dv` vs ref **逐位相同**（S512 1.671/1.771/1.899e-3；S4096
  1.883/1.734/1.966e-3；GQA kv4 2.134/3.305/3.850e-3；MQA kv1 2.292/7.934/7.517e-3），
  因为只改「delta 的求和次序」与「dQ 的写入位置/宽度」，数学口径不变。

### 14p.3 性能（同 session A/B，CUDA event）

| shape | 版本 | total | preprocess | main | convert |
|---|---|---|---|---|---|
| MHA S=4096 | base(`--deltawarp=0 --dqdirect=0`) | 1.3802 | 0.3260 | 0.9441 | 0.1101 |
| MHA S=4096 | **O24** | **1.3309 (1.037×)** | **0.3002** | 0.9521 | **0.0786** |
| MHA S=512 | base | 0.1052 | 0.0407 | 0.0515 | 0.0130 |
| MHA S=512 | **O24** | **0.1001 (1.051×)** | **0.0362** | 0.0523 | **0.0117** |
| GQA kv4 S=1024 | base | 0.2892 | 0.0876 | 0.1797 | 0.0220 |
| GQA kv4 S=1024 | **O24** | **0.2712 (1.066×)** | **0.0717** | 0.1790 | 0.0205 |
| MQA kv1 S=1024 | base | 0.4436 | 0.1235 | 0.2833 | 0.0369 |
| MQA kv1 S=1024 | **O24** | **0.4119 (1.077×)** | **0.0945** | 0.2858 | 0.0316 |
| MLA D=512 S=1024H2 | base | 0.8263 | 0.1360 | 0.6496 | 0.0407 |
| MLA D=512 S=1024H2 | **O24** | 0.8211 (1.006×) | 0.1356 | 0.6639 | 0.0217 |

`delta` 单项（同 session）：S=4096 0.0425→**0.0127ms（3.35×）**、S=512 0.0075→**0.0035（2.1×）**、
GQA kv4 0.0223→0.0072（3.1×）；**收益主要来自 `delta`，dQ 直写再叠加 convert 的一小段**。

### 14p.4 ncu（S=4096，`--set full`，`-c 1`）

| 指标 | 旧 `delta_kernel` | 新 `delta_warp_kernel` |
|---|---|---|
| Duration | 42.53 µs | **14.50 µs** |
| DRAM Throughput | 24.94 % | **72.87 %（带宽 bound）** |
| L1/TEX | 73.68 % | 25.84 % |
| Compute (SM) | 72.16 % | 29.42 % |
| Executed Instructions | 23,199,744 | **4,194,304（−82%）** |
| Waves/SM | 31.03 | 7.76 |

⇒ 墙从 **smem 归约 + 标量加载**（Compute/L1TEX 双高）移到 **DRAM 带宽**（elementwise 上限），
与 fp8 O14 的 `quantize_row_warp_kernel` 结论一致。

### 14p.5 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`，FA2/FA3/TE 三列）

MHA S=4096 FA3 **0.3246ms/847TF**、TE 0.4405/624、FA2 0.7293/377 ⇒ ours total 1.3309ms =
**FA3 的 4.10×**（时间；O23 4.20×）。GQA kv4 S=1024 FA3 0.0827/416 ⇒ 3.28×（O23 3.45×）。

### 14p.6 原始输出

`src/fp16/fa_bwd_fp16_main_o24_sweep.out.txt`（两文件 ×5 shape × base/O24）、
`src/fp16/fa_bwd_fp16_mma_onefile_o24_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_main_o24_ncu_delta_{old,warp}_s4096.out.txt`、
`src/fa_bwd_o24_fa3_te_baseline_fp16.out.txt`。

## 14q. O25：cluster 分布式归约（Hopper thread block cluster）——机制成立、数值正确，但净负（S4096 main 0.13×）

### 14q.1 动机

O7b（§14o）把 dK/dV 的跨 CTA `atomicAdd` 换成「partial 覆盖写 + 二次归约 kernel」，`red`
51.9M→0、主 kernel 快 1.20×，但二次归约是一整趟 DRAM 扫描（90.4% 带宽 bound、384.9µs），
`main+reduce` 净负。其结论明确写下：「要真正消 red 只能上 **cluster 分布式归约**——在 SM 间
smem 内合并偏和、不落全局内存」。O25 就是这条路的实现与判决。目标：**不落全局 partial、
不做二次扫描**，用 thread block cluster 把「相邻 mblk 的 CTA 对同一 KV 行的偏和」在 leader
的 smem 里合并，leader 每 tile 只发一次全局 `red_add2` ⇒ 跨 CTA red 字节砍半。

### 14q.2 机制冒烟（先验证 DSM 原语再合入）

`src/fp16/fa_bwd_fp16_cluster_reduce_smoke.cu`：cluster=2（`__cluster_dims__(2,1,1)`），
每个 rank 把自己的偏和用 `red.shared::cluster.add.f32`（地址经 `mapa.shared::cluster` 映射到
leader rank 0 的 smem）推进 leader 的累加器；`barrier.cluster.arrive/wait` 同步后 leader
flush 到全局。多 tile 复用累加器（flush→barrier→清零→barrier）。实测
`ref-vs-expected max_abs=1.2e-06, cluster-vs-ref max_abs=1.4e-06`（仅 fp32 加法次序）⇒ **PASS**。

### 14q.3 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

给 `fa_bwd_fp16_wgmma2_kernel` 增加模板参数 `int CL = 1`（默认 1 ⇒ 行为逐字不变）：
- **cluster 沿 bx**（`__cluster_ctarank`，配对相邻两个 mblk 2c / 2c+1）；
- leader 的 smem 里加两个 `[BN][HD]` fp32 合并累加器（`dvacc/dkacc`，+64KB，214KB 仍 1 CTA/SM）；
- 6 处 dK/dV red（SPLIT 与非 SPLIT 各 2/4 处）经 lambda `dkv_red` 分流：`CL==1` 走原
  `red_add2`，`CL>1` 走 `red.shared::cluster.add.f32`（非 leader 远程、leader 本地）；
- crate 高的（rank1）恒多做 `BM/BN=2` 个 KV tile，故低 mblk 的 rank0 把循环延到 rank1 的
  tile 数（多出的 tile 因 causal mask 使 P=0、偏和为 0，只参与 barrier），两 CTA 锁步；
- 每 tile 尾：`cluster_sync` → leader flush（一次全局 `red_add2` 并清零）→ `cluster_sync`；
- host 新增 `--cluster[=2]`（`cudaLaunchKernelEx` + `cudaLaunchAttributeClusterDimension`），
  开启时强制 `wgmma2(BN=64)`（BN=128 的 2b 放不下合并累加器），`nblk%2!=0` 自动回退。

### 14q.4 数值（ours-vs-ref，fp16 causal，max_abs）—— 与历史逐位一致

S=512 `1.671/1.771/1.899e-3`；S=4096 `1.883/1.734/1.966e-3`——**与 O17/O18/O23/O24 完全相同**，
单/两文件逐指标一致。证明 cluster 只改归约次序，不改数学（`red` 合并的是同一批偏和）。

### 14q.5 性能（同 session A/B，CUDA event，ms）—— 显著净负

| shape | main 默认 | main +cluster2 | 时间比 | 端到端 total 默认 | total +cluster2 |
|---|---|---|---|---|---|
| S=512 | 0.0528 | 0.3443 | **6.5×** | 0.1002 | 0.3924 |
| S=4096 | 0.9855 | 7.3178 | **7.4×** | 1.3597 | 7.8881 |

同 session 纯反向基线：FA3 MHA S4096 `0.3238ms/849TF`、TE `0.4407/624`、FA2 `0.7253/379`
⇒ 默认 ours total 为 FA3 的 4.2×，cluster 版退化到 24×。

### 14q.6 ncu（main，S=4096，同 session、同 binary，`-c 1`）

| 指标 | wg2 默认 | wg2 +cluster2 |
|---|---|---|
| `lts__t_sectors_op_red` | 51,904,512 | **26,738,688（0.515×，精确减半）** |
| `lts__t_sectors_op_read` | 18,158,517 | 11,717,044（0.645×） |
| Duration | 989.7µs | **7.64ms（7.7×）** |
| stall long_scoreboard | 0.75 | **5.83（7.8×）** |
| stall short_scoreboard | 0.41 | **4.19（10×）** |
| warps_active | 12.46% | 12.50% |

**机制假设被 ncu 完全证实**（`red` 精确减半），**但代价是灾难性的**：`red.shared::cluster.add.f32`
是**逐元素远程原子**（每个 CTA 每 tile 8192 个 dV + 8192 个 dK），跨 SM 互连延迟极高，
`long_scoreboard` 从 0.75 飙到 5.83；同时每 tile 的 leader flush 是**串行段**（另一 SM 在
`cluster_sync` 空等），把原本分散在 512 个 CTA 上、与计算重叠的全局 red 集中到 leader 的
non-overlapped epilogue。二者叠加 ⇒ 7.4×。

### 14q.7 结论

**cluster 分布式归约在 fp16 wgmma2 上不是可行杠杆**：机制正确（数值逐位一致、red 精确减半），
但「逐元素远程 smem 原子 + 每 tile leader 串行 flush」的成本远超省下的全局 red。
非原子 DSM（`st.async.shared::cluster` + leader 求和）可避免远程原子延迟，但需要
`CL×[BN][HD]` 的 smem（CL=2 时 +128KB，214→342KB）**放不下**；把每元素偏和先在 CTA 内
`red` 合并再做 cluster 又会退回 O7b 的 partial/reduce 结构。因此**保留 `--cluster` 为
opt-in A/B（默认关）**，O7b 的确定性 `--det=1` 仍是唯一可用选项；fp16/bf16 main 的 L2 red 墙
在本卡上暂无便宜的解法（与 §14k 的寄存器墙、§14o 的 reduce 带宽墙合起来，三条路都被证伪）。

### 14q.8 原始输出

`src/fp16/fa_bwd_fp16_cluster_reduce_smoke.out.txt`、
`src/fp16/fa_bwd_fp16_o25_s512_ab.out.txt`、`src/fp16/fa_bwd_fp16_o25_s4096_ab.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o25_s512.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o25_ncu_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_o25_fa3_te_baseline.out.txt`。

## 14r. O30：LSE 预处理改用 4D-TMA 载入 Q/K（唯一能压指令数的搬运杠杆，1.30–1.36×；数值逐位不变）

### 14r.1 动机

O15a（第 51 轮，§14i）已用 `fa_bwd_fp16_tma_smoke.cu` 证明 `cuTensorMapEncodeTiled`
(`SWIZZLE_128B`) 写出的 smem 布局与 kernel 的 `sw128_off` **逐字节相同**，并指出
**HD=128 的 K-major tile 必须拆成 2 个 K=64 chunk**（TMA box 内维 128B = 64 个 fp16），
wgmma 侧用两个 `SBO=1024` 描述符读。但那条通路一直只停在冒烟，没落进真 kernel。
O23 把 fp16/bf16 的 Hopper 主 kernel + LSE 默认化后，端到端里剩下最大的、还没上 TMA 的
搬运就是 **LSE 的 Q/K 载入**：它每 tile 用「逐 16B `cp.async` + 一手地址运算」搬
`LBM×HD = 64×128` 个 half（= 8 个 unit/线程/tile），S=4096 时最多 64 个 K tile/CTA。
本项把 LSE 的 Q/K 换成 **4D TMA**（坐标 `{k0,row,head,batch}`），一条 bulk 指令搬一个
8KB chunk，省掉 load 指令与地址运算，并复用同一份 SW128 tile 供 `wgmma.m64n64k16` 直读。
**数学与 `lse_mma_kernel_bal_wgmma` 完全一致**（镜像配对、online-softmax、4-lane `shfl`），
只换搬运方式 ⇒ 数值应逐位相同。

### 14r.2 实现（单/两文件 device 代码逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

device 侧（`fa_bwd_fp16_mma_kernels.cuh`，紧随 `lse_mma_kernel_bal_wgmma` 之后）：

* 新增 TMA 原语 `mbar_init`/`mbar_arrive_expect`/`mbar_wait`/
  `tma_load_4d`（`cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes`）
  与 `wgmma_qkt64_tma`（两个 K=64 chunk、各自 `SBO=1024` 的 4+4 条 `wgmma.m64n64k16`）。
  全部用 `FA_HAS_WGMMA` 包裹 ⇒ `sm_90` 构建下退化为空实现，行为不变。
* 新增 `lse_mma_kernel_bal_tma<HD,PIPE=1>`：smem = `Q(16KB) + 2×K(各 16KB)` + 3 个
  mbarrier；`issue_q`/`issue_k` 各发 2 条 4D TMA（chunk 0/1），K 双缓冲用两个 barrier +
  逐 barrier 的相位计数；prologue 发 Q + tile0，循环内 `mbar_wait → __syncthreads → 发下一
  tile(st^1) → wgmma`，与 `lse_mma_kernel_bal_wgmma` 的流水结构对应。

host 侧（`fa_bwd_fp16_mma_main.cu`）：新增 `make_lse_map(ptr,H,S,D,B)`（4D 描述符，
`dims={D,S,H,B}`、stride `{H*D*2, D*2, S*H*D*2}`、box `{64,64,1,1}`）与 CLI `--lsetma=0/1`；
TMA 路径需驱动 API，故整体用 **`-DFA_TMA`** 编译开关（隐含 `-DFA_WGMMA`）包裹：
`-DFA_WGMMA -DFA_TMA -lcuda` 构建下，D==128/causal 默认开 TMA（`--lsetma=0` 可退回 wgmma）；
仅 `-DFA_WGMMA` 或纯 `sm_90` 构建**完全不编译/不引用驱动符号**，无需 `-lcuda`。

### 14r.3 数值（ours-vs-ref，fp16 causal，max_abs）—— 与历史逐位一致

TMA-vs-wgmma 的 LSE **`max_abs = 0.000e+00`（逐位相同）**，5 个 shape × 单/两文件全部如此。
最终 `dq/dk/dv vs ref` 与 O5–O24 历史值一致：

| shape | dq | dk | dv |
|---|---|---|---|
| MHA S=512 | 1.671e-3 | 1.771e-3 | 1.899e-3 |
| MHA S=4096 | 1.883e-3 | 1.734e-3 | 1.966e-3 |
| GQA kv4 S=1024 | 2.134e-3 | 3.305e-3 | 3.850e-3 |
| GQA kv8 S=1024 | 2.008e-3 | 2.931e-3 | 3.891e-3 |
| MQA kv1 S=1024 | 2.292e-3 | 7.934e-3 | 7.517e-3 |

### 14r.4 性能（同 session A/B，CUDA event，单/两文件）

LSE-only（`[O30 A/B]`）与端到端 total：

| shape | lse wgmma (ms) | lse tma (ms) | 加速 | total (tma) | TFLOPS |
|---|---|---|---|---|---|
| MHA S=512 | 0.0328 | **0.0249** | **1.32×** | 0.0942 | 22.8 |
| MHA S=4096 | 0.2878 | **0.2128** | **1.35×** | 1.2553 | 109.5 |
| GQA kv4 S=1024 | 0.0640 | **0.0488** | **1.31×** | 0.2575 | 66.7 |
| GQA kv8 S=1024 | 0.0684 | **0.0528** | **1.30×** | 0.2905 | 73.9 |
| MQA kv1 S=1024 | 0.0777 | **0.0590** | **1.32×** | 0.3945 | 87.1 |

单文件与两文件逐指标一致（差异 <1% session 噪声）。端到端 S=4096 **1.2553ms（O24 1.3309ms，
1.06×）**；preprocess S=4096 0.2271ms（O24 0.3002ms）。

### 14r.5 ncu（lse，S=4096，同 session，`-c 1`）

| 指标 | wgmma+cp.async | **4D TMA** |
|---|---|---|
| Duration | 286.30 µs | **214.56 µs（1.33×）** |
| Executed Instructions | 165.30 M | **117.87 M（−28.7%）** |
| registers/thread | 62 | **58** |
| shared mem/block | 50.18 KB | 50.24 KB |
| achieved occupancy | 23.02 % | 23.11 % |
| DRAM / L1TEX / L2 | 3.77 / 17.43 / 20.27 % | 5.05 / 17.69 / 26.77 % |
| Compute (SM) | 60.93 % | 58.30 % |
| stall `wait` / `short` / `long` | 2.26 / 0.77 / 0.04 | 2.33 / 0.86 / 0.17 |

结论：收益来源是 **指令数 −28.7%**（8 个 `cp.async`+地址运算/tile/线程 → 1 条 bulk 指令/lane），
Duration 随之 1.33×；**墙没有变**——仍是 **Compute ~58% + `wait`（softmax/mma 固定延迟）**，
与 O8b/O9 的判断一致。即 TMA 只把「搬运那半」压掉，**LSE 的真正天花板是每元素 softmax epilogue**。

### 14r.6 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`，FA2/FA3/TE 三列）

FA3 MHA S4096 **0.3244ms/847TF**、GQA kv4 0.0825/417、kv8 0.1212/354、MQA kv1 0.1562/440；
TE MHA 0.4406/624。ours total 时间比 FA3：MHA S4096 **3.87×**（O24 4.10×）、GQA kv4 **3.12×**
（O24 3.28×）、kv8 2.40×、MQA kv1 2.53×。**全 shape 小幅改善**。

### 14r.7 复现

```bash
# 构建（TMA 需驱动 API：-DFA_TMA -lcuda；不带 FA_TMA 则只需 -DFA_WGMMA）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp16 --iters=50
# A/B：--lsetma=0 退回 wgmma+cp.async（程序内 [O30 A/B] 同时计时两者 + max_abs 对拍）
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/ncu.sh src/fp16/fa_bwd_fp16_mma_main.cu -c 1 \
  --kernel-name regex:lse_mma_kernel_bal_tma --set full \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp16 --iters=1
```

### 14r.8 原始输出

`src/fp16/fa_bwd_fp16_o30_lse_tma_sweep.out.txt`（单/两文件 ×5 shape × `[O30 A/B]` + 对拍）、
`src/fp16/fa_bwd_fp16_lse_tma_ncu_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_lse_wgmma_ncu_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_o30_fa3_te_baseline.out.txt`。

## 14s. O33：主 kernel 的 Q/K/V/dO 改用 4D-TMA 载入（逐 atom，布局逐字节不变；main 1.04×）

### 14s.1 动机

O30–O32（§14r / `docs/01b` §6x / `docs/03` §35）已把 **LSE 的 Q/K 载入**从逐 16B `cp.async`
换成 4D-TMA。主 kernel（`fa_bwd_fp16_wgmma2b_kernel`，S≥4096 的默认快路）的 Q/dO（prologue）
与 K/V（每个 KV tile）仍用「逐 16B `cp.async` + `sw128_off` 地址运算」，是本轮要收掉的搬运开销。

难点：wgmma 的 5 个 GEMM 都从同一块 **SW128 K-major** tile 直读（描述符地址与 atom 交织次序
绑定），而 TMA 一个 box 内维只有 128B（fp16 = 64 元素）。O15a 已证「HD=128 的 K-major tile 的
交织布局无法用一个 2D box 复现」。本项走**第三条路**：

* 一个 `[8 行][64 列]` 的 2D box 恰好 = 一个 1024B SW128 atom；
* **逐 atom 发 TMA**（每个 `(rg,kg)` 一条），dst = `sw128_off(rg*8, kg*64, HD)`
  = `(rg*(HD/64)+kg)*1024`，于是**原样复现交织布局**（atom 次序 `rg*(HD/64)+kg`）；
* ⇒ **所有 wgmma 描述符完全不用改**，TMA 搬的是与 `cp.async` 逐字节相同的 smem。

代价：Q/dO 各 32 atom、K/V 各 32 atom，但每条 TMA 只占一条指令、**不占寄存器、不记
scoreboard**，且由 tid0 串行发射（异步）；比 4096 条 `cp.async` 摊到 256 线程更省发射与地址运算。
OOB 行由 TMA 自动补 0，与 `cp.async` 的显式零写一致。

### 14s.2 实现（单/两文件 device 代码逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

device（`fa_bwd_fp16_mma_kernels.cuh`）：

* 新增 `tma_fill_sw128<R,HD>`：对 `(rg,kg)` 逐 atom 发 `tma_load_4d`（复用 O30 的原语）。
* 新增 `fa_bwd_fp16_wgmma2b_tma_kernel<HD,SPLIT>`：与 `fa_bwd_fp16_wgmma2b_kernel` **几何/
  数据流/描述符逐字相同**，只把载入与同步换成 TMA + mbarrier——
  * prologue：`mbar_init` 4 个 barrier（`qbar,kbar0,kbar1,vbar`），tid0 发 Q/dO（expect
    `2*QTILE`）与 K0/V0（各 `KTILE`）；
  * 循环：`mbar_wait(Q 首轮) → mbar_wait(kbar[st]) → mbar_wait(vbar) → __syncthreads →
    发 K[nt+1](st^1) → GEMM1/2 → P/dS → __syncthreads → 发 V[nt+1] → GEMM3/4/5`；
    K 双缓冲两个 barrier（相位每 `nt` 翻一次），V 单缓冲后段预取（同 O6b）。
  * smem = `1024(对齐) + Q32K + dO32K + K 2×32K + V32K + P32K + dS32K + 128(barrier)`，
    仍 1 CTA/SM。
* TMA asm / `__grid_constant__` 由 `FA_HAS_WGMMA` / `FA_WGMMA+FA_TMA` 包裹，`sm_90` 构建不变。

host（`fa_bwd_fp16_mma_main.cu`）：新增 `make_main_map(ptr,H,S,D,B)`（box `{64,8,1,1}`）、
`launch_bwd_wgmma2b_tma<HD,SPLIT>` 与 CLI `--maintma[=0/1]`（默认 0，**opt-in**，与 cp.async
版同 binary A/B；程序内 `[O33 A/B]` 同时计时两者并给 `max|diff|`）。仅 `wgmma2b` 几何
（`S≥4096` 自动选，或 `--wg2bn=1`）生效。

### 14s.3 数值（ours-vs-ref，fp16 causal，max_abs）—— 与 O5–O32 历史一致

| shape | dq | dk | dv |
|---|---|---|---|
| MHA S=4096 | 1.883e-3 | 1.734e-3 | 1.966e-3 |

`[O33 A/B] max|diff| tma-vs-cp.async`：**dq `0.00e+00`（逐位）**、dk/dv ~`2e-5`——
差异**仅来自 dK/dV 的跨 CTA `atomicAdd` 次序**（TMA 改变发射时序，其它 A/B 如 O18 同样有
`1e-5` 量级差异），smem 字节与 wgmma 完全一致 ⇒ 布局正确。vs-ref 与历史**数值相同**。

### 14s.4 性能（同 session A/B，CUDA event，main-only，S=4096）

| 版本 | main ms | TFLOPS |
|---|---|---|
| wg2b + cp.async | 0.9638 | 142.6 |
| **wg2b + TMA** | **0.9249** | **148.6** (1.042×) |

单文件版同量级（0.9647→0.9237，1.044×）。端到端 total **1.2289ms / 111.8 TF**
（cp.async 1.2553ms）——main 占端到端 ~75%，故 total ~1.021×。
限制：`--maintma` 只挂到 BN=128 的 `wgmma2b`（S≥4096 默认快路）；BN=64 的 `wgmma2` 未加
TMA，小 S/GQA 不受影响。

### 14s.5 ncu（main，S=4096，同 session，`-c 1`）

| 指标 | cp.async | **TMA** |
|---|---|---|
| Duration | 968.96 µs | **923.87 µs** |
| L1/TEX | 33.27% | 34.78% |
| L2 | 56.99% | 60.13% |
| `lts__t_sectors_op_red` | 51,904,512 | **51,904,512（逐字节不变）** |
| `read` / `write` | 18.71M / 1.57M | 19.09M / 2.03M |
| `long_scoreboard` / `wait` / `barrier` | 0.61 / 1.23 / 0.94 | 1.37 / 1.33 / 1.47 |
| regs / smem / occupancy | 255 / 230.5KB / 12.5% | 255 / 230.5KB / 12.5% |

结论：**TMA 不改 red（仍占 L2 ~72%）、不改 occupancy**，与预期一致；它省的是「搬运那半」的
发射/地址运算，ncu Duration 与 event 同向 **~1.05×**。stall 计数（long_scoreboard/wait/barrier）
反而升，但那是**更少 warp 在更短时间里等同一批 wgmma/barrier** 的归一化假象——净 duration
下降。**main 的第一墙仍是 dK/dV 的 L2 `red`（三条消 red 路已在 §14k/§14o/§14q 证伪），
TMA 动不了它**。

### 14s.6 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`，FA2/FA3/TE 三列）

FA3 MHA S4096 **0.3263ms/842TF**、TE **0.4443/619**；ours total 1.2289ms ⇒ **为 FA3 的
3.77×**（O30 时 3.87×）、main-only 148.6 vs 842 = 5.66×。

### 14s.7 复现

```bash
# 构建（TMA 需 -DFA_WGMMA -DFA_TMA -lcuda）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp16 --iters=50 --maintma
# A/B：--maintma=0 退回 cp.async（程序内 [O33 A/B]）
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/ncu.sh src/fp16/fa_bwd_fp16_mma_main.cu --kernel-name regex:wgmma2b_tma -c 1 --set full \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp16 --maintma --iters=1
```

### 14s.8 原始输出

`src/fp16/fa_bwd_fp16_o33_maintma{0,1}_fa_bwd_fp16_mma_main_s4096.out.txt`（A/B + 对拍）、
`src/fp16/fa_bwd_fp16_o33_onefile_s4096.out.txt`（单文件 `[O33 A/B]`）、
`src/fp16/fa_bwd_fp16_o33_ncu_cpasync_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_o33_ncu_maintma_metrics_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_o33_ncu_maintma_s4096.out.txt`（`--set full`）、
`src/fp16/fa_bwd_fp16_o33_fa3_te_baseline.out.txt`。

## 14t. O35：BN=64 的 `wgmma2` 主 kernel 也改用逐 atom 4D-TMA（补全 O33 的几何；小 S 中性、大 S 1.02×）

### 14t.1 动机

O33（§14s，fp16）/ O34（`docs/01b` §6y，bf16）只把 **BN=128** 的 `wgmma2b` 主 kernel 的
Q/K/V/dO 换成 4D-TMA；而 **O23 的默认档在 S<4096 与 GQA/MQA 走的是 BN=64 的 `wgmma2`**
（`-DFA_WGMMA` 构建下自动选）。这条更常用的路仍用逐 16B `cp.async` + `sw128_off` 地址运算。
本项把 O33 的「逐 atom TMA、原样复现 SW128 交织布局、wgmma 描述符零改动」搬到 BN=64 几何，
补全 backlog 里「BN=64 的 `wgmma2` 几何待做」这一条，并量化它在不同 S 下的收益。

### 14t.2 实现（单/两文件 device 代码逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

device（`fa_bwd_fp16_mma_kernels.cuh`）：

* 新增 `fa_bwd_fp16_wgmma2_tma_kernel<HD,SPLIT>`：与 `fa_bwd_fp16_wgmma2_kernel` 的
  **几何/数据流/描述符逐字相同**（BM=128、BN=64、2 warpgroup，GEMM1/2 `wgmma_mn64_issue`、
  GEMM3/4/5 `wgmma_m64n64k16_t` + MN-major 转置描述符、GEMM5 的 `dqacc[2][8][4]` 寄存器累加），
  只把载入与同步换成 TMA + mbarrier：
  * prologue：`mbar_init` 4 个 barrier（`qbar,kbar0,kbar1,vbar`），tid0 发 Q/dO
    （`tma_fill_sw128<128,HD>`，expect `2*QTILE`）与 K0/V0（`tma_fill_sw128<64,HD>`，各 `KTILE`）；
  * 循环：`mbar_wait(qbar 首轮) → mbar_wait(kbar[st]) → mbar_wait(vbar) → __syncthreads →
    发 K[nt+1](st^1) → GEMM1/2 + P/dS → __syncthreads → 发 V[nt+1] → GEMM3/4/5`；
  * 一个 `[8 行][64 列]`（128B 内维）的 box = 一个 1024B SW128 atom，`tma_fill_sw128<64,HD>`
    共 8×2=16 atom；smem 与 cp.async 版同（+4 个 mbarrier），仍 **1 CTA/SM**。

host（`fa_bwd_fp16_mma_main.cu`）：新增 `launch_bwd_wgmma2_tma<HD,SPLIT>`，并让
`--maintma` 在 **BN=64 的 `wg2` 分支**也生效（此前只在 `wg2b`/BN=128 生效）；新增
`[O35 A/B]`（mode 9）与 cp.async 版（mode 2）做**同 session head-to-head + `max|diff|`**。

### 14t.3 数值（ours-vs-ref，fp16 causal，max_abs）—— 与 O5–O34 历史逐位一致

| shape | dq | dk | dv | `max|diff|`(TMA-vs-cp.async) dq/dk/dv |
|---|---|---|---|
| S=512 H16 | 1.671e-03 | 1.771e-03 | 1.899e-03 | `0.00e+00 / 6.10e-05 / 7.63e-05` |
| S=1024 H32 kv4 (GQA) | 2.134e-03 | 3.305e-03 | 3.850e-03 | `0.00e+00 / 1.22e-04 / 2.44e-04` |
| S=4096 H16 | 1.883e-03 | 1.734e-03 | 1.966e-03 | `0.00e+00 / 6.10e-05 / 7.63e-05` |

**dq 逐位相同**；dk/dv 的差异只来自跨 CTA `atomicAdd` 的次序（搬的是与 `cp.async` 逐字节
相同的 smem），不构成精度问题。

### 14t.4 性能（同 session A/B，CUDA event，main-only）—— 小 S 中性、大 S 1.02×

| shape | wg2(BN64, cp.async) | wg2(BN64)+TMA | 比 |
|---|---|---|---|
| S=512 H16（两文件） | 0.0508 ms (42.2 TF) | 0.0512 ms (41.9 TF) | **0.992×** |
| S=512 H16（单文件） | 0.0508 ms (42.3 TF) | 0.0514 ms (41.8 TF) | **0.987×** |
| S=1024 H32 kv4 | 0.1797 ms (95.6 TF) | 0.1803 ms (95.3 TF) | **0.996×** |
| S=4096 H16（强制 BN=64） | 0.9884 ms (139.1 TF) | 0.9685 ms (141.9 TF) | **1.021×** |

**原因（见 ncu）**：TMA 把 main 的指令数砍 ~24%、寄存器 200→184，但 **BN=64 几何在两个小
shape 上是「延迟/grid bound」而非「发射 bound」**——S=512 时 `grid=128 < 132 SM`（Waves 0.48、
achieved occ 12.4%）、S=1024 GQA 时 `grid=512`（~1.94 wave），少发指令换不到时间；只有
S=4096（强制 BN=64，`grid=512`、tile 数翻 4 倍）才让「省发射」体现在 Duration 上。

> 注：S≥4096 的默认档是 **BN=128 的 `wgmma2b`**（O33，见 §14s），O35 只在用户显式
> `--wg2=1 --maintma=1` 或用 BN=64 几何时生效；故它主要价值是**补全几何覆盖、确认机制**，
> 不改变默认端到端数字。

### 14t.5 ncu（main，同 session、同 binary，`-c 1`）

| 指标 | cp.async（BN64） | **+TMA** |
|---|---|---|
| Executed Instructions | 5,259,008 | **3,976,256（−24.4%）** |
| Registers/thread | 200 | **184** |
| L1/TEX Throughput | 47.5% | 44.7% |
| Duration（S=512） | 53.02 µs | **52.96 µs（持平）** |
| Waves Per SM / occ | 0.48 / 12.27% | 0.48 / 12.39% |

S=4096（强制 BN=64）：cp.async 989.95µs / inst 260.7M / regs 200 / L2 55.1% / SM 27.8%
→ **TMA 959.20µs（1.032×）/ inst 200.8M（−23.0%）/ regs 184 / L2 57.4% / SM 22.0%**。
结论：**TMA 只省「搬运的指令/地址运算」，动不了延迟/grid 墙**；这与 O33（大 S 上 1.04×）
完全一致，只是在 BN=64 的小 S 场景墙更靠「延迟」。

### 14t.6 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`，FA2/FA3/TE 三列）

S=4096 MHA：FA2 0.7270ms/378TF、**FA3 0.3233ms/850TF**、TE 0.4435ms/620TF；GQA kv4 S=1024：
FA3 0.0829ms/414TF、TE 0.1125ms/305TF。默认端到端 ours S=4096 total 1.2566ms（109.4 TF）
⇒ **FA3/ours = 3.89×**（与 O33/O34 的 3.77–3.79× 同量级，session 噪声）。

### 14t.7 复现

```bash
# 编译（wgmma + TMA）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu --iters=50
# GQA
... --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h32_d128_kv4_causal_fp16
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/ncu.sh src/fp16/fa_bwd_fp16_mma_main.cu --set full --launch-count 1 \
  --kernel-name regex:wgmma2_tma -- --iters=5
```

### 14t.8 原始输出

`src/fp16/fa_bwd_fp16_mma_main_o35_{s512,gqa_kv4,s4096}.out.txt`（两文件 A/B + 对拍）、
`src/fp16/fa_bwd_fp16_mma_onefile_o35_s512.out.txt`（单文件 `[O35 A/B]`）、
`src/fp16/fa_bwd_fp16_o35_ncu_main_{tma,cpasync}_s512.out.txt`（`--set full`）、
`src/fp16/fa_bwd_fp16_o35_ncu_main_{tma,cpasync}_s4096.out.txt`（关键指标）、
`src/fa_bwd_o35_fa3_te_baseline_fp16.out.txt`。

## 14u. O38-fp16：LSE 的 K 维 split + 二次归约（第 85 轮）—— **正结果，已设为默认 auto**

> 承接 fp8 **O38**（`docs/03` §41，第八十四轮）：LSE 的镜像配对虽然把每个 CTA 的工作量压成
> 常数 `nblk+1` 个 K tile，但**每 CTA 仍要串行扫完自己那段**；小 S / 低 H 时 `grid=ceil(nblk/2)·H`
> 远小于 SM 数（S512/H16 只有 64 CTA），并行度/临界路径就是墙。fp8 用「把 K 维按 tile 切片、
> 跨 CTA 各扫一段、再二次归约」解决了它（LSE 最多 2.0×）。本轮把同一套做法移植到 fp16（及
> bf16，见 `docs/01b` §6ac）的 **TMA LSE**（`lse_mma_kernel_bal_tma`，D=128/causal 的默认快路）。

### 14u.1 实现（单/两文件 device 代码逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

* **kernel**（`src/fp16/fa_bwd_fp16_mma_kernels.cuh`）：`lse_mma_kernel_bal_tma<HD,PIPE>` 加
  `float* lse_part, int ksplit`；grid 变为 `(pairs, H, B*ksplit)`，`b=blockIdx.z/ksplit`、
  `ksp=blockIdx.z%ksplit`；每个 `(pair,ksp)` 只扫本 m 块 K tile 的**连续切片**
  `[nt0,nt1)=[ntiles·ksp/ksplit, ntiles·(ksp+1)/ksplit)`。流水 stage 用**切片内相对下标**
  `rnt & 1`（而非全局 `nt & 1`），避免 `nt0` 奇偶错位；prologue 发 `issue_k(0, nt0*LBN)`。
  `ksplit==1` 时切片即整段、直接写 `lse`，**逐位退化为 O30 原路径**。
* **merge**：新增 `lse_split_merge_kernel`（`[row][ks]->(m,l)` 沿 `ks` 做 online-softmax 合并，
  `lse=m+log(l)`），数学上与「单 CTA 顺序扫全部 K tile」完全等价，只差 fp32 求和次序。
* **host**（`fa_bwd_fp16_mma_main.cu` / 单文件同名 host 段）：新增 `--lsesplit=N`（`0=auto`）、
  `d_lse_part` 缓冲、`[O38 A/B]` sweep。默认 `auto` 目标 **`grid*split ≈ 528`**（= 4 CTA/SM ×
  132 SM，即「填满一个波」；LSE TMA smem ~50KB ⇒ 4 CTA/SM），上限 8；grid 已达一个波则退回 1。
  *注*：fp8 的 auto 目标是 `≈2048`（fp8 LSE 并行度更低）；fp16/bf16 实测大 S 时 512 CTA 已铺满
  一个波（S4096 `split>1` 反而慢 3–7%），故以「一波」为准。

### 14u.2 数值（ours-vs-ref，fp16 causal，max_abs）—— 全 shape 与 O5–O36 历史逐位一致

| case | dq | dk | dv | `max_abs(split-vs-split1)` |
|---|---|---|---|---|
| MHA S512 | 1.671e-3 | 1.771e-3 | 1.899e-3 | 9.5e-7 |
| MHA S4096 | 1.883e-3 | 1.734e-3 | 1.966e-3 | 1.9e-6 |
| GQA q32/kv4 S1024 | 2.134e-3 | 3.305e-3 | 3.850e-3 | 9.5e-7 |
| GQA q40/kv8 S1024 | 2.008e-3 | 2.931e-3 | 3.891e-3 | 9.5e-7 |
| MQA q64/kv1 S1024 | 2.292e-3 | 7.934e-3 | 7.517e-3 | 9.5e-7 |

`max_abs(split-vs-split1)` ≤ 2e-6（纯 fp32 求和次序），**auto 路径对 ref 与历史逐位相同**；
单/两文件逐指标一致。

### 14u.3 性能（同 session `[O38 A/B]`，CUDA event；LSE-only 与端到端）

LSE-only（`auto` 选中档）：

| case | split1 | auto | auto 倍数 | 备注 |
|---|---|---|---|---|
| MHA S512 | 0.0259 ms | **0.0150 ms**（split=8） | **1.72×** | grid 64 → 512 |
| GQA q32/kv4 S1024 | 0.0506 ms | **0.0370 ms**（split=2） | **1.37×** | grid 256 → 512 |
| GQA q40/kv8 S1024 | 0.0535 ms | 0.0535 ms（split=1） | 1.00× | grid 320；手动 split4 1.035× |
| MQA q64/kv1 S1024 | 0.0596 ms | 0.0596 ms（split=1） | 1.00× | grid 512 |
| MHA S4096 | 0.2064 ms | 0.2064 ms（split=1） | 1.00× | grid 512，已满一波 |

端到端 total（3 kernel，另 session；preprocess 含 LSE+delta+merge）：

| case | split1 total | auto total | 倍数 |
|---|---|---|---|
| MHA S512 | 0.0935–0.0953 ms | **0.0831 ms** | **1.13–1.15×** |
| GQA q32/kv4 S1024 | 0.2572–0.2598 ms | **0.2429 ms** | **1.06–1.07×** |
| MHA S4096 | 1.260–1.266 ms | 1.261 ms | 1.00× |
| GQA q40/kv8 S1024 | 0.289 ms | 0.289 ms | 1.00× |
| MQA q64/kv1 S1024 | 0.396 ms | 0.396 ms | 1.00× |
| GQA kv4 S1024（bf16，参照） | 0.258–0.259 ms | **0.2451 ms** | **1.06×** |

merge kernel（S512 split8）仅 **4.16µs**（LSE 的 ~28%，占端到端 <5%），可忽略。

### 14u.4 ncu（LSE 主 kernel，S=512，同 binary `--lsesplit=1` vs 默认 split=8）

| 指标 | split=1 | split=8 |
|---|---|---|
| Duration | 27.74 µs | **12.64 µs（2.19×）** |
| Waves Per SM | 0.12 | **0.97** |
| Achieved Occupancy | 6.25% | **20.61%** |
| Compute (SM) | 8.22% | **26.89%** |
| Executed Ipc Active | 0.75 | **1.50** |
| No Eligible | 81.27% | 61.32% |
| regs / smem carveout | 58 / 233.47KB | 同 |

⇒ **墙不是指令/访存，而是并行度 / 临界路径**（split1 只有 0.12 个波、Ipc 0.75、No Eligible 81%），
split 把 grid 从 64 抬到 512、占满一个波后 Compute/Ipc 翻倍——与 fp8 O38 的结论一致。
merge kernel：Duration 4.16µs、Compute 1.77%、Waves 0.03（纯带宽/归约，无压力）。

### 14u.5 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`，FA2/FA3/TE 三列）

| shape | ours total (ms) | FA3 (ms/TF) | TE (ms/TF) | ours/FA3 时间 | O30 时 |
|---|---|---|---|---|---|
| MHA S4096 | 1.261 | 0.3250 / 846 | 0.4440 / 619 | 3.88× | 3.87× |
| GQA q32/kv4 S1024 | 0.2429 | 0.0823 / 417 | 0.1124 / 306 | **2.95×** | 3.12× |
| GQA q40/kv8 S1024 | 0.2894 | 0.1210 / 355 | 0.1320 / 325 | **2.39×** | 2.40× |
| MQA q64/kv1 S1024 | 0.3957 | 0.1567 / 439 | 0.2148 / 320 | **2.53×** | 2.53× |

（ours total 打印口径 `4BS²HD`；对标口径 `4BS²H(D+Dv)=8BS²HD`，故 ours 的真反向 TF 为打印值 ×2。
S4096 auto=1 与 O30 持平；GQA/MQA 因 LSE split 时间比小幅改善。）

### 14u.6 复现

```bash
F='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
# 两文件（默认 auto；单文件把 main 换成 fa_bwd_fp16_mma_onefile.cu）
ARCH="" NVCC_FLAGS="$F" scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu --iters=50 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp16
# split1 基线
... --lsesplit=1 --dir=...
# ncu（第一个匹配即 run_all 的 auto 档）
ARCH="" NVCC_FLAGS="$F" scripts/ncu.sh src/fp16/fa_bwd_fp16_mma_main.cu --set full -c 1 \
  --kernel-name regex:lse_mma_kernel_bal_tma -- --iters=20
```

### 14u.7 原始输出

`src/fp16/fa_bwd_fp16_mma_main_o38_{b1_s512_h16_d128_causal_fp16,b1_s4096_h16_d128_causal_fp16,b1_s1024_h32_d128_kv4_causal_fp16,b1_s1024_h40_d128_kv8_causal_fp16,b1_s1024_h64_d128_kv1_causal_fp16}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o38_s512.out.txt`、
`src/fp16/fa_bwd_fp16_o38_ncu_lse_split{1,8}_s512.out.txt`、`..._o38_ncu_merge_s512.out.txt`、
`src/fa_bwd_o38_split1_baseline.out.txt`、`src/fa_bwd_o38_fa3_te_baseline_fp16_bf16.out.txt`。
（bf16 见 `docs/01b` §6ac；fp8 见 `docs/03` §41。）

## 14v. O39-fp16：MLA（head_dim=512）LSE 的 K 维 split + 二次归约（第八十六轮）—— **正结果，默认 auto**

### 14v.1 动机

O38-fp16（§14u）只把 K 维 split 做进 **D=128/TMA** 的 LSE。fp16 的 **MLA（D=512）反向**
causal LSE 走 **mma 版** `lse_mma_kernel_bal<512,1>`（HD>128 无 SW128/TMA 快路）：S1024H2
时 `nblk=16,pairs=8`，grid=`8×2×1=16` CTA，ncu **Waves 0.12 / Achieved Occupancy 6.25% /
Compute 2.71%** —— 纯并行度墙。O39 把切片 + merge 机制移植到 mma 版（fp8 O39 的 fp16 版）。

### 14v.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

`lse_mma_kernel_bal<HD,PIPE>` 加尾部默认参数 `float* lse_part=nullptr, int ksplit=1`：
`grid.z: B→B*ksplit`、`b=z/ksplit, ksp=z%ksplit`；每 `(pair,ksp)` 只扫 K tile 切片
`[nt0,nt1)`（**流水 stage 用切片内相对下标 `rnt&1`**）；`ksplit==1` 逐位退化为 O8b；
部分 `(m,l)` 写 `lse_part`，`lse_split_merge_kernel` 沿 `ksp` online-softmax 合并。
host `launch_lse_bal` 加 `lse_part/ksplit`，`--lsesplit=N`（`0=auto`）：**D=512 目标
`grid*split≈132`**（fp16 LSE smem ~200KB ⇒ **1 CTA/SM**）、上限 16，再按 `nblk=ceil(S/64)`
封顶；D=128 TMA 维持 O38 的 `≈528/8`。另把 D=128 的**非 TMA mma 回退路径**也接上 split。

### 14v.3 数值（vs fp32 ref，fp16 causal，max_abs dq/dk/dv）

| case (B1 D=Dv=512) | dq | dk | dv | max_abs(split-auto vs split1) |
|---|---|---|---|---|
| S256 H2 | 1.638e-3 | 1.582e-3 | 1.753e-3 | 4.8e-7 |
| S512 H4 | 2.516e-3 | 2.916e-3 | 1.724e-3 | 4.8e-7 |
| S1024 H2 | 1.987e-3 | 1.712e-3 | 1.848e-3 | 9.5e-7 |

与 O5c（§14b）历史逐位一致；单/两文件逐指标一致；D=128 MHA 回归逐位不变
（S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、GQA kv4 2.134/3.305/3.850e-3）；
varlen MLA（§16.9）回归不变。

### 14v.4 性能（CUDA event；同 session `--lsesplit=1` vs auto）

| case | LSE split1 (ms) | LSE auto (ms) | LSE 倍数 | total split1 (ms) | total auto (ms) | total 倍数 |
|---|---|---|---|---|---|---|
| MLA S256H2 | 0.0350 | **0.0200** (split4) | **1.75×** | 0.2182 | **0.2014** | **1.08×** |
| MLA S512H4 | 0.0583 | **0.0216** (split8) | **2.70×** | 0.4146 | **0.3785** | **1.10×** |
| MLA S1024H2 | 0.1018 | **0.0282** (split8) | **3.62×** | 0.7846 | **0.7154** | **1.10×** |

auto 命中 per-shape 最优（split 全扫见原始输出）；main 在 MLA 端到端中占 ~85–90%，
故 total 增益（1.08–1.10×）小于 LSE-only（1.75–3.62×）。

### 14v.5 ncu（LSE 主 kernel，`--launch-count 1`，同 binary split 1 vs 8，S1024H2）

| | split1 | split8 |
|---|---|---|
| Duration | 108.80 µs | **25.66 µs** |
| Waves Per SM | 0.12 | **0.97** |
| Achieved Occupancy | 6.25% | 6.25% |
| Compute (SM) | 2.71% | **16.64%** |
| L1/TEX / L2 | 28.82% / 3.34% | 22.19% / 24.08% |
| Registers / Dynamic smem | 64 / 199.68 KB（Block Limit Shared Mem = 1） | 同 |

**结论：墙 = 网格不足（Waves 0.12 → 0.97）；split 把并发槽填满后回到 1 CTA/SM 的
`mma wait` + smem 依赖**（与 fp8 O39 §42.5 一致）。

### 14v.6 复现 / 原始输出

```bash
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp16 --o=ref_o --iters=100
ARCH="" ... scripts/run.sh src/fp16/fa_bwd_fp16_mma_onefile.cu --dir=... --lsesplit=8
```

原始输出：`src/fp16/fa_bwd_fp16_main_o39_b1_s{256_h2,512_h4,1024_h2}_d512_causal.out.txt`、
`..._o39_split1_*`、`..._o39_reg_*`、`..._o39_mmafallback_s512.out.txt`、
`..._o39_ncu_lse_split{1,8}_s1024h2.out.txt`、`src/fp16/fa_bwd_fp16_mma_onefile_o39_*`、
`..._o39_varlen_*`。

## 14w. O40-fp16：非 TMA wgmma LSE 的 K 维 split + varlen 接入（第八十七轮）—— **正结果，默认 auto**

### 14w.1 动机

O38-fp16（§14u）把 split 做进 **D=128 的 TMA LSE**，O39-fp16（§14v）做进 **mma 版 LSE**
（D=512 MLA）。但 **非 TMA 的 wgmma 版 `lse_mma_kernel_bal_wgmma`**（O9a/O9b）从未有 split，
而它是：① 定长 D=128/causal 的 `--lsetma=0` 回退（S512 grid=64、Waves 0.12、occ 6.25%）；
② **VARLEN D=128/causal 的默认 LSE**（`run_varlen` 直接 launch wgmma 版）。O40 把 O39 的切片
+ merge 机制移植到它，并把 split 接进 `run_varlen`（含 D=512 的 mma LSE）。fp8 同轮见
`docs/03` §43。

### 14w.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

- `lse_mma_kernel_bal_wgmma<HD,PIPE>` 加尾部默认参数 `float* lse_part = nullptr, int ksplit = 1`；
  `b = blockIdx.z/ksplit`、`ksp = blockIdx.z%ksplit`；每 `(pair,ksp)` 只扫本 m 块 K tile 的连续切片
  `[nt0,nt1)`，流水 stage 用切片内相对下标 `rnt&1`；`ksplit==1` **逐位退化为 O9b 原路径**。
- 部分 `(m,l)` 写 `lse_part[(row*ksplit+ksp)*2+{0,1}]`，由 `lse_split_merge_kernel`（O38 已有）
  合并。**varlen 的 merge 行数是 `T*H`**（`d_lse` 只分配 `T*H`），故 host 加 `merge_rows` 参数。
- `run_varlen` 加 `--lsesplit=N`（0=auto）：D=128 目标 `grid*split≈528`（= 4 CTA/SM × 132 = 一个波）、
  D=512 `≈132`，上限 8/16，再按最大序列 `nblk` 封顶；D=128 causal 走 wgmma 版、D=512 causal 走
  O39 的 mma 版；各新增 `d_lse_part`（`T*H*16*2` fp32）缓冲。定长 `--lsetma=0` 回退路径接入
  `lse_split_eff`（沿用 O38 的 auto）。单文件 host 同步。

### 14w.3 数值（ours-vs-ref，fp16 causal，max_abs）

| case | dq | dk | dv | max_abs(split vs split1) |
|---|---|---|---|---|
| MHA S512 H16（定长 `--lsetma=0`） | 1.671e-3 | 1.771e-3 | 1.899e-3 | 0（auto=8） |
| varlen b4_t3840_h16 D128 | 3.163e-3 | 2.158e-3 | 1.966e-3 | 0（auto=1） |
| varlen b4_t4096_h16 D128 | — | — | — | 0（auto=1） |
| varlen b1_t512_h2 D512 | 1.303e-3 | 1.537e-3 | 1.557e-3 | 0（auto=8） |
| varlen b3_t1792_h2 D512 | 2.415e-3 | 1.834e-3 | 1.856e-3 | 0（auto=4） |

全形状与 O5–O39 历史**逐位一致到打印精度**；`max_abs(split vs split1)=0`。单/两文件逐指标一致；
D=128 MHA/GQA 默认 TMA 路径回归不受影响（S512 1.671/1.771/1.899e-3、D512 1.987/1.712/1.848e-3）。

### 14w.4 性能（CUDA event；same-session `--lsesplit=1` vs auto）

| case | split1 | auto | 倍数 |
|---|---|---|---|
| MHA S512（定长 `--lsetma=0`，preprocess） | 0.0364 ms | **0.0211 ms** | **1.73×** |
| MHA S512（端到端 total） | 0.1011 ms | **0.0847 ms** | **1.19×** |
| varlen b1_t512_h2 D512（total） | 0.4098 ms | **0.3737 ms** | **1.10×** |
| varlen b3_t1792_h2 D512（total） | 0.9112 ms | **0.8607 ms** | **1.06×** |

fp16 的 D=128 varlen case（b4_t3840 / b4_t4096 / b5_t3968 / b8_t2904）base 已 ≥ 目标 528，auto=1、
**中性**（与大 S 的 D=128 TMA 一致）；D=512 的 MLA varlen（base=8/48，严重 under-filled）受益
1.06–1.10×。单文件 total 同量级（S512 0.0857 vs 两文件 0.0847）。

### 14w.5 ncu（LSE `lse_mma_kernel_bal_wgmma`，`--launch-count 1`，定长 S512）

| | split1 | split8 |
|---|---|---|
| Duration | 33.66 µs | **14.85 µs** |
| Waves Per SM | 0.12 | **0.97** |
| Achieved Occupancy | 6.25% | **19.27%** |
| Compute (SM) | 9.32% | **36.67%** |
| Memory Throughput | 3.85% | 18.10% |
| No Eligible | 79.1% | 48.6% |

**墙 = 网格不足一个波**（Waves 0.12）；split 填满一波后回到 LSE 固有 `mma wait` + smem 依赖。

### 14w.6 复现 / 原始输出

```bash
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp16 --lsetma=0 --lsesplit=0
ARCH="" ... scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu --varlen \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b1_t512_h2_d512_causal_fp16 --lsesplit=0
ARCH="" ... scripts/ncu.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --kernel-name regex:lse_mma_kernel_bal_wgmma --launch-count 1 --set full -- --lsetma=0 --lsesplit=8
```

原始输出：`src/fp16/fa_bwd_fp16_mma_main_o40_fixed_s512_split{1,0}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o40_varlen_{b1_t512_h2_d512,b3_t1792_h2_d512,b4_t3840_h16_d128,
b4_t4096_h16_d128}_split{1,0}.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o40_ncu_lse_wgmma_split{1,8}_s512.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o40_fixed_s512_auto.out.txt`。

## 16. VARLEN：fp16 反向支持变长 / `cu_seqlens`（单/两文件）

> O37 之后 ROADMAP「可选·变长」的 fp16 补全（fp8 已在第 77 轮完成，`docs/03` §37）。
> 目标：让 fp16 反向直接吃 **packed** `[T,H,D]` + `cu_seqlens`，逐序列只算 `len_b` 的因果
> 注意力，避免 padding 到 `max_b len_b`。

### 16.1 动机与口径

* `q`/`dO`/`dQ`：`[T,H,D]`；`k`/`v`/`dK`/`dV`：`[T,Hkv,D]`；`cu_seqlens`：`B+1` 个 int 前缀和。
* kernel 用 `qbase=cu_seqlens[b]`、`len=cu_seqlens[b+1]-qbase` 取代定长 `b*S`/`S`；
  `S` 参数传 `maxlen`（仅用于 grid / 镜像配对计数），每个 `(b,h,mblk)` 只处理本序列内 tile。
* ref 的 `ref_dq/dk/dv.npy` 也是 packed，逐元素比对。FA/TE 变长在本机不可用 ⇒ 只对 fp32 ref。

### 16.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

改动收敛在 **3 处 device 代码**（5 个 GEMM 的数学/数据流完全不动）：

1. **4 个搬运 helper** 加一个**默认参数** `int qbase = -1`：
   `kv_issue_async`/`qdo_issue_async`/`kv_issue_async_sw`/`qdo_issue_async_sw` 内部把
   `b*S` 换成 `(qbase>=0 ? qbase : b*S)`。定长调用不传该参 ⇒ **逐位不变**；变长传
   `cu_seqlens[b]`、`S` 传 `len`。这样不改动约 30 处既有调用点。
2. **`lse_mma_kernel_bal_wgmma`**（默认 Hopper LSE）加 `const int* cu_seqlens`：
   `qbase/len`、`nblk=(len+LBM-1)/LBM`、越界判定 `<len`、地址 `qbase+…`；并加
   `if (pair >= (nblk+1)/2) return;` 让短序列多余的镜像配对 CTA 直接退出（定长恒不触发）。
3. **`fa_bwd_fp16_wgmma2_kernel`**（默认 D=128 主 kernel）加 `const int* cu_seqlens`：
   同样 `qbase/len`，`qdo/kv_issue_async_sw` 传 `qbase`，P mask/LSE/D 读/dK/dV red/dQ 写回
   的长度与地址全改 `len`/`qbase`；`nblk` 仅 cluster/DET 用（变长不走），保持定长语义。

`delta_warp_kernel` 本就是「每 warp 一行」的扁平实现，packed 天然可用，无需改。
host 新增 `--varlen` 分支 `run_varlen`（读 packed 输入 + `cu_seqlens.npy`，LSE 用
`lse_mma_kernel_bal_wgmma`、主 kernel 用 `launch_bwd_wgmma2<128,true,1>`，均传 `d_cu`）。

### 16.3 数值（ours vs fp32 ref，fp16 causal；max_abs）

| case | lengths | dq | dk | dv | total (ms) | TFLOPS（Σ_b 4HL²D） |
|---|---|---|---|---|---|---|
| b4_t3840 不齐 | `[512,1024,2048,256]` | 3.163e-3 | 2.158e-3 | 1.966e-3 | 0.7764 | 58.77 |
| b4_t4096 等长 | `[1024]×4` | 2.112e-3 | 2.252e-3 | 1.915e-3 | 0.4497 | 76.40 |
| b5_t3968 GQA kv8 | `[128,256,512,1024,2048]` | 2.438e-3 | 3.433e-3 | 3.843e-3 | 1.4388 | 63.62 |
| b8_t2904 强倾斜 | `[2048,512,…,8]` | 2.624e-3 | 2.158e-3 | 2.139e-3 | 0.5839 | 62.96 |

全部 fp16 噪声量级（~2–4e-3），**无 system error、无 padding 泄漏**；单文件与两文件**逐位相同**
（b4_t3840：3.163/2.158/1.966e-3）。
定长回归（`nullptr`）**逐位不变**：S512 `1.671/1.771/1.899e-3`。

### 16.4 性能与对标

等长 case `[1024]×4` 等价定长 B4 S1024 H16 D128，同口径（`4BS²H(D+Dv)` = `8BS²HD`）对标：

| 实现 | ms | TFLOPS |
|---|---|---|
| **ours total（varlen 等长）** | **0.4497** | **152.8** |
| FA2.7.4（定长） | 0.2502 | 274.7 |
| **FA3（SM90，定长）** | 0.1457 | **471.5** |
| TE2.14（定长） | 0.1761 | 390.2 |

ours 时间 = FA3 的 **3.09×**、TFLOPS 为 FA3 的 **32%**（同 shape S4096 时约 26%，小 S 相对好）。

### 16.5 ncu（主 kernel，b4_t3840，`--set full --launch-count 1`）

| 指标 | 值 |
|---|---|
| Duration | 549.44 µs |
| DRAM / L1/TEX / L2 | 10.05% / **48.74% / 63.18%** |
| Compute (SM) | 31.03% |
| regs / Block Limit Shared Mem | 202 / 1 |
| Achieved / Theoretical Occupancy | 12.43% / 12.50% |
| Waves Per SM / No Eligible | 7.76 / 63.49% |
| Issued Ipc Active / Warp Cycles Per Issued Inst | 1.46 / 5.44 |

bound 与定长 fp16 wgmma2 完全一致：**L2（dK/dV 跨 CTA red）+ L1/TEX 吞吐 + 1 CTA/SM 的
延迟受限**，DRAM 仅 10% ⇒ 非带宽 bound。

### 16.6 复现

```bash
# dump fp16 varlen cases
python harness/fa_bwd_bench.py dump --varlen-all --dtype fp16
# 两文件（wgmma2 + wgmma LSE）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu --varlen --iters=50 \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b4_t3840_h16_d128_causal_fp16
# 单文件
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp16/fa_bwd_fp16_mma_onefile.cu --varlen --iters=50 --dir=...
```

原始输出：`src/fp16/fa_bwd_fp16_varlen_b{4_t3840,4_t4096,5_t3968,8_t2904}*.out.txt`
（两文件）、`..._onefile_b4_t3840.out.txt`（单文件）、`..._ncu_main_b4_t3840.out.txt`（ncu）、
`src/fa_bwd_varlen_fa3_te_baseline_fp16.out.txt`（FA2/FA3/TE 基线）。

### 16.7 限制 / 后续

* 只做 **fp16 / HD=128 / MHA+GQA**；非 causal 见 §16.8；MLA、负载均衡留后续。
* 用 **非 TMA** 主 kernel（wgmma2 + cp.async）；TMA 需为 packed 布局重建描述符。
* 短序列的 CTA 早退，强倾斜 case 并行度浪费；可引入按 `cu_seqlens` 的均衡分块。

### 16.8 非 causal（full attention）——第 79 轮

非 causal 与 causal 的区别：**每个 m 块的 K 列数恒为 `len`**（不再随 `mblk` 递增），工作天然
均衡、不需要镜像配对。**device 改动只 1 处**（单/两文件逐字一致，`sync_onefile_device.py`
核对 `identical: True`）：`lse_mma_kernel<HD>`（O8 通用 mma 版）加默认参数
`const int* cu_seqlens = nullptr`——`qbase/len`，Q/K 行下标与边界由 `b*S`/`S` 改为
`qbase`/`len`，加 `if (m0 >= len) return;`。`nullptr` 时逐式退化，**定长逐位不变**。
host `run_varlen` 去掉「只做 causal」限制：causal 走 `lse_mma_kernel_bal_wgmma`，非 causal
走 `lse_mma_kernel<128>`（`grid.x=nblk`，传 `d_cu`）；主 kernel `fa_bwd_fp16_wgmma2_kernel`
本就带 `causal` 参数，无需改。

**数值（ours vs fp32 ref，fp16 full，max_abs dq/dk/dv）**：b4_t3840 不齐
`5.603/6.841/2.338e-4`；b4_t4096 等长 `4.094/4.953/1.234e-4`；b5_t3968 GQA kv8
`7.341/7.906/4.880e-4`；b8_t2904 强倾斜 `1.486/1.555/1.582e-3` —— 全 fp16 噪声、无 padding
泄漏；单/两文件逐位一致。**定长回归逐位不变**（causal S512 `1.671/1.771/1.899e-3`；
fixed full S1024 `3.27/2.52/1.23e-4`）。

**性能（total，event，`Σ_b 4HL²D` 口径）**：b4_t3840 1.2870ms/35.46TF、b4_t4096（等长）
0.9090ms/37.80TF、b5_t3968 GQA 2.4577ms/37.25TF、b8_t2904 1.0616ms/34.63TF。非 causal 的
总量约为 causal 的 2×，故时间是 causal 的 ~2×（等长 0.909 vs causal 0.450）。按定长对标口径
`4BS²H(D+Dv)` 乘 2（`D=Dv=128`）⇒ 等长 **75.6 TF**；同 session TE fp16 定长 full
`0.2805ms/245.0TF`、FA2.7.4 `0.4740ms/145.0TF` ⇒ ours 为 TE 的 3.24×。

**ncu（main，b4_t3840，`--set full -c 1`）**：Duration **705.4µs**、DRAM 7.89% /
L1/TEX 49.03% / **L2 70.31%** / Compute 35.79%、202 regs、**Block Limit Shared Mem=1
（occ 12.45%）**、Waves 7.76、No Eligible 63.35% ⇒ bound 与定长 wgmma2 一致：
**L2（dK/dV 跨 CTA red）+ 1 CTA/SM 延迟受限**，非带宽。

**复现**：`python harness/fa_bwd_bench.py dump --varlen-all --dtype fp16 --full`；
`ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" scripts/run.sh
src/fp16/fa_bwd_fp16_mma_main.cu --varlen=1 --full --iters=50 --dir=...full_fp16`。
原始输出 `src/fp16/fa_bwd_fp16_varlen_full_sweep.out.txt`、`..._varlen_full_onefile.out.txt`、
`..._varlen_full_ncu_main_s3840.out.txt`；`src/fa_bwd_varlen_full_regression.out.txt`。

### 16.9 VARLEN 的 MLA（head_dim=512）——第 81 轮

第 77–79 轮的 varlen 覆盖 fp8/fp16/bf16 × causal/full，但都限 **HD=128（MHA/GQA）**；
第 80 轮把 fp8 扩到 **MLA（HD=512）**（`docs/03` §39）。本轮把同一能力补到 **fp16/bf16**。

**device 改动 2 处**（单/两文件逐字一致，`sync_onefile_device.py` 核对 `identical: True`）：

1. **`lse_mma_kernel_bal<HD>`**（O8b 非 wgmma 的 mma 镜像配对 LSE）加默认参数
   `const int* cu_seqlens = nullptr`：`qbase/len`，`nblk` 用 `len`，`if (pair >= (nblk+1)/2)
   return;` 让短序列多余的配对 CTA 退出，Q/K 下标与边界全改 `qbase`/`len`。D=512 的 causal
   无 wgmma/SW128 快路（tile 过大），故走此 mma 版（与 fp8 第 80 轮同构）。
2. **`fa_bwd_fp16_mma_kernel`**（HD=128/512 通用 mma 主 kernel）加 `const int* cu_seqlens`：
   `qbase/len` 传进 `qdo_issue_async`/`kv_issue_async`（helper 的 `qbase` 形参早就有），
   `ncols`/P mask/LSE/D 读/dV·dK red/dQ 写回的长度与地址全改 `len`/`qbase`；`sched` 仍只
   定长 causal 用（varlen 传 0）。`nullptr` 时逐式退化 ⇒ **定长路径逐位不变**。
   `lse_mma_kernel<HD>`（非 causal）早在第 79 轮已带 `cu_seqlens`，无需改；`delta_warp_kernel`
   是扁平每 warp 一行，packed 天然可用。

host `run_varlen` 按 D 分派：`D==128` 仍走 wgmma2（causal 用 `lse_mma_kernel_bal_wgmma`）；
`D==512` 走 `lse_mma_kernel_bal<512,1>`/`lse_mma_kernel<512>` + `launch_bwd_mma<512,32,32,1,
false,true>`（与定长 D=512 同几何），dQ 已在 fp32 累加缓冲（NDT>1）⇒ 先 memset、`convert`
转全部 `nq/nkv`。两文件与单文件 host 由 `splice` 同步。

**数值（ours vs fp32 ref，max_abs dq/dk/dv）**：b3_t1792 `[256,512,1024]` H2 D512 causal
fp16 `2.415/1.834/1.856e-3`、bf16 `1.267/1.217/1.796e-2`；full fp16 `5.516/4.451/2.385e-4`、
bf16 `3.100/3.526/2.316e-3` —— 同 dtype 噪声量级、无 padding 泄漏。**单/两文件逐位一致**
（@index 也相同）。**定长 D=512 回归逐位不变**：S1024H2 causal fp16 `1.987/1.712/1.848e-3`、
bf16 `5.838/9.519/1.568e-2`（与历史文件相同）；HD=128 varlen full 回归 b4_t3840 fp16
`5.603/6.841/2.338e-4` 与第 79 轮**逐位相同**。

**性能（ours total，event；MLA 的 `D=Dv=512`，`Σ_b 4HL²D` 口径）**：

| varlen case | causal ms/TF | full ms/TF | 同 shape 定长 causal |
|---|---|---|---|
| b3_t1792 `[256,512,1024]` H2 D512（fp16） | 0.9382 / 6.01 | 1.4652 / 3.85 | — |
| b3_t1792（bf16） | 0.9347 / 6.03 | 1.4690 / 3.84 | — |
| b1_t512 `[512]` H2 D512（fp16） | 0.4264 / 2.52 | 0.5511 / 1.95 | 0.4232 / 2.54 |
| b1_t512（bf16） | 0.4258 / 2.52 | 0.5471 / 1.96 | 0.4247 / 2.53 |

**varlen b1 与同 shape 定长仅差 +0.8%（causal）** ⇒ varlen kernel 无额外固定开销。
MLA 反向 FA3/TE 均不支持（FA forward 限 hdim≤256、TE 组合同样非法），故只有 ours 数字。

**ncu（fp16，b3_t1792 causal，`--set full -c 1`）**：

| kernel | Duration | DRAM | L1/TEX | L2 | Compute | regs | Occ | Waves | No Eligible |
|---|---|---|---|---|---|---|---|---|---|
| main | **784.1 µs** | 1.61% | 38.74% | 25.80% | 4.40% | 168 | 6.25% | 1.45 | **89.04%** |
| lse | 136.4 µs | 1.62% | 22.30% | 3.59% | 2.86% | 56 | 6.25% | **0.36** | 81.59% |

main bound = **1 CTA/SM（BM=32 下 smem ~202KB）+ smem→mma 依赖延迟**（No Eligible 89%、
DRAM/Compute 都低），与 fp8 MLA（`docs/03` §39）一致；LSE 是 **并行度 bound**（grid=48 < 132 SM，
Waves 0.36）。这两条（降 smem 冲 2 CTA/SM、均衡分块）是后续候选。

**复现**：`python harness/fa_bwd_bench.py dump --lengths 256 512 1024 --H 2 --D 512 --kv 2
--Dv 512 --dtypes fp16 bf16 [--full]`；
`ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda"
scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu --varlen=1 [--full] --dir=...`。
原始输出 `src/fp16/fa_bwd_fp16_varlen_mla_sweep.out.txt`、`..._varlen_mla_perf.out.txt`、
`..._varlen_mla_ncu_{main,lse}_b3.out.txt`。

### 16.10 VARLEN 主 kernel 的 TMA 化 —— 第 83 轮（判决：中性 / 偏负，opt-in）

第 82 轮把「按 `cu_seqlens` 均衡分块」判为负结果后，剩下两条 varlen 候选是 **LSE 的 K 维
split** 和 **varlen TMA 化**。本轮先把后者做完（fp16 先行；`docs/03` §40）。

**动机/直觉**：定长的 fp16/bf16 主 kernel 已全面 TMA 化（O33–O36）：**逐 atom 4D-TMA 复现
SW128 交织布局**（一个 `[8 行][64 列]` box = 一个 1024B atom，描述符零改动），把每 tile 的
上千条 `cp.async` 合成几十条 TMA，指令数 **−24%**。varlen 的 packed 布局 `[T,H,D]` 只是少了
batch 维，**描述符按 `dims={D,T,H,1}`（`S=T, B=1`）建、kernel 用行坐标 `cu_seqlens[b]+row`、
batch 坐标 0** 即可复用同一套 TMA 循环——看起来是 O33/O35 的直接延伸。

**实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）**：

* `fa_bwd_fp16_wgmma2_tma_kernel<HD,SPLIT>`（BN=64，O35 的 kernel）加默认参数
  `const int* cu_seqlens = nullptr`：新增 `qbase/len` 与 `rowbase/tbatch`（`nullptr` 时
  `rowbase=0,tbatch=b` ⇒ **定长坐标 `(row,b)` 逐字不变**）；全局下标 `b*S→qbase`、边界
  `S→len`；4 处 `tma_fill_sw128`（Q/dO、K0/V0、K 预取、V 预取）的行坐标改 `rowbase+…`、
  batch 改 `tbatch`。
* host `run_varlen`：`--varlentma[=0/1]`（**默认 0，opt-in**，同 binary A/B）；建 4 张
  varlen 描述符复用 `make_main_map(d_q,H,T,D,1)`；`D==128` 且开启时走 TMA 壳。
  定长默认档（cp.async）路径**逐位不变**。

**数值（ours vs fp32 ref，fp16，max_abs dq/dk/dv）—— TMA 与 cp.async 完全一致**
（TMA 只换搬运；本例跨 CTA `atomicAdd` 次序也恰好未变）：

| case | cp.async | TMA |
|---|---|---|
| b4_t3840 causal | 3.163/2.158/1.966e-3 | 3.163/2.158/1.966e-3 |
| b4_t4096 等长 causal | 2.112/2.252/1.915e-3 | 2.112/2.252/1.915e-3 |
| b8_t2904 强倾斜 causal | 2.624/2.158/2.139e-3 | 2.624/2.158/2.139e-3 |
| b5_t3968 GQA kv8 causal | 2.438/3.433/3.843e-3 | 2.438/3.433/3.843e-3 |
| b4_t3840 full | 5.603/6.841/2.338e-4 | 5.603/6.841/2.338e-4 |

**性能（同 binary A/B，event，`Σ_b 4HL²D` 口径，iters=200）—— 中性，强倾斜反而变慢**：

| case | cp.async ms | TMA ms | TMA/cp |
|---|---|---|---|
| b4_t3840 causal | 0.7761 | **0.7641** | 1.016× |
| b4_t4096 等长 causal | 0.4484 | 0.4461 | 1.005× |
| b5_t3968 GQA kv8 causal | 1.4102 | 1.4085 | 1.001× |
| b4_t3840 full | 1.2822 | 1.2746 | 1.006× |
| **b8_t2904 强倾斜 causal** | 0.5785 | **0.6073** | **0.953×（−4.7%）** |

强倾斜变慢的机制：TMA 的 box 行数固定（Q/dO 各 16 个 8 行 atom），**越界行也被 TMA「搬」
（只是不写 smem）**，而长序列少的 case（len=8/16/32…）里 `cp.async` 版对越界行是**直接不发**。
即 TMA 省的是长序列的发射指令，短序列反而多了固定发射开销。

**ncu（main，b4_t3840 causal，`--set full -c 1`，同 binary）**：

| backend | Duration | Executed Inst | L1/TEX | L2 | Compute | Occupancy | Waves |
|---|---|---|---|---|---|---|---|
| cp.async | 546.2 µs | 163.7 M | 48.8% | 63.5% | 31.0% | 12.43% | 7.76 |
| **TMA** | **543.6 µs** | **125.9 M（−23.1%）** | 46.8% | 65.2% | 24.4% | 12.44% | 7.76 |

指令数确实按预期降了 23%，但 **Duration 不变**：kernel 是 **1 CTA/SM（动/静态 smem ~148KB）
+ 延迟受限**（achieved occupancy 12.4%，Warp Cycles/Issued 7.06 vs 5.45——指令少了但每条
等得更久）。这与 O35（BN=64 定长 TMA 也基本中性）的结论一致：**BN=64 的 varlen 主 kernel
不是发射/访存指令 bound，TMA 打不动真正的墙**。

**对标（等长 `[1024]×4` 落回定长 B=4,S=1024,H=16,D=128；纯反向 CUPTI，三列）**：
ours varlen total **0.446 ms / 真反向 ~153 TF** vs **FA3 0.1457ms/471.7TF、TE 0.1760/390.6、
FA2 0.2507/274.1** ⇒ ours 为 FA3 的 **3.06×**（时间）、TE 的 2.53×、FA2 的 1.78×（与第 78 轮
的 3.09× 一致）。full：FA3 0.1953/351.9、TE 0.2143/320.7、FA2 0.3294/208.6。

**结论**：varlen TMA 化**判为中性/偏负**——保留 `--varlentma` opt-in 复现，**不作为默认**。
与第 82 轮的「均衡分块」一样，varlen 的真正墙是 **L1/L2 吞吐 + 低 occupancy 的延迟隐藏**，
而非发射指令；剩下的 varlen 杠杆只有 **LSE 的 K 维 split + 二次归约**（降镜像配对的
`nblk+1` 串行临界路径）与 **并行度/occupancy**。

**复现**：
```
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu --varlen --varlentma=1 \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b4_t3840_h16_d128_causal_fp16 --iters=200
```
原始输出 `src/fp16/fa_bwd_fp16_varlen_tma_ab.out.txt`、
`..._varlen_{tma,cpasync}_ncu_main_b4_t3840.out.txt`、
`..._varlen_fa3_te_baseline.out.txt`。

## 14x. O43-fp16：wgmma2 主 kernel 的 N 方向 split-K（第九十轮）—— **正结果，默认 auto（仅小 grid）**

### 14x.1 动机

O23 默认档下 D=128 的小 S 走 `wgmma2`（BM=128/BN=64、2 warpgroups、256 线程、1 CTA/SM）。
其 grid = `ceil(S/128) × H × B`：**S=512 MHA（H=16,B=1）= 64 CTA < 132 SM**，即 ncu
`Waves Per SM = 0.48`——**一半 SM 空转**（O17 之后 S≥1024 已满波，故只有 S≤512 受影响；
fp8 的 mma 主 kernel 早有 `ksplit`，fp16/bf16 的 wgmma2 一直没有）。

fp16/bf16 的 wgmma2 之前不做 split-K 的**直接障碍**是 O24：dQ 由本 CTA 唯一拥有 ⇒ 主 kernel
**直接写 fp16 `dq`**、convert 跳过 dQ。切 K 后同一 Q 行由多个 CTA 贡献 ⇒ dQ 必须跨 CTA 原子累加。

### 14x.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

`fa_bwd_fp16_wgmma2_kernel` 加运行时参数 `int ksplit = 1`（只走 cp.async 路径；cluster / TMA
分支的生效值恒 1，不参与）：
- **网格分解**：`ksp = bx % ksplit`、`mblk = bx / ksplit`（`ksplit==1` 时逐式退回 `mblk=bx`）；
- **KV tile 切片**：`nt_begin = ntiles*ksp/ksplit`、`nt_end = ntiles*(ksp+1)/ksplit`，
  切片为空则整 CTA 早退；K/V 预取从 `nt_begin*BN` 起，循环用全局列偏移 `(nt_begin+nt)*BN`；
- **dQ 归约**：`ksplit>1` 时末段 epilogue 改 `red_add2(dq_acc, ...)`（跨 CTA `atomicAdd`），
  host 传 `dq_h=nullptr` 并 `memset(dq_acc)`、`convert` 补回 dQ 一趟。

**默认 auto**（`--wg2ksplit=-1`）：仅 `D==128 && wgmma2(BN=64) && 非 cluster/TMA` 且未切块
`base = ceil(S/128)*H*B < 132` 时，取「填满一个波」的 2 的幂（cap 8、按 `ceil(S/64)` 封顶）；
其它 shape（S≥1024、GQA/MQA、S≥4096 的 wgmma2b）恒 1 ⇒ 逐位回归。`--wg2ksplit=N` 可强制/关闭。

### 14x.3 数值（ours-vs-fp32-ref，fp16 causal，max_abs dq/dk/dv）

S=512 MHA：ksplit=1/2/4 三者 **完全相同** `1.671/1.771/1.899e-3`（与 O5 以来历史逐位一致；
切 K 只改 fp32 原子求和次序，落在 fp16 噪声内）。回归：S=1024 full `3.27/2.52/1.23e-4`、
S=4096 causal `1.88/1.73/1.97e-3` 均与 ksplit=1 一致。**单/两文件逐位一致**。

### 14x.4 性能（CUDA event；同 binary、同 session）

| S=512 MHA fp16 | main (ms) | total (ms) |
|---|---|---|
| ksplit=1（旧） | 0.0529 | 0.0871 |
| **ksplit=2（auto）** | **0.0317（1.67×）** | **0.0687（1.27×）** |
| ksplit=4 | 0.0342 | 0.0732 |

即端到端 0.0871→**0.0687ms / 31.3 TF**。同 session 纯反向对标（`fa_vs_te_bwd_only.py`）：
FA2 0.0437ms/98 TF、**FA3 0.0261ms/164 TF**、TE 0.0320ms/134 TF ⇒ ours/FA3 时间比
**3.34×→2.63×**。S=1024 / S=4096 因 `base≥132`、走 wgmma2b，auto=1、数值与时间均不变。

### 14x.5 ncu（`fa_bwd_fp16_wgmma2_kernel`，`--set full --launch-count 1`，S=512）

| 指标 | ksplit=1 | ksplit=2 |
|---|---|---|
| Duration | 54.08 µs | **33.50 µs（1.61×）** |
| Waves Per SM | **0.48** | **0.97** |
| Achieved Occupancy | 12.47% | 12.46% |
| Executed Ipc Elapsed | 0.42 | **0.77** |
| DRAM / L2 Throughput | 9.4% / 25.6% | 18.8% / 49.8% |

**结论：墙 = grid 不足一个波（Waves 0.48，half-SM 空转），与 per-SM occupancy（1 CTA/SM）无关。**
切 K=2 把 wave 填到 0.97、elapsed IPC 近翻倍；per-SM 仍 12.5%（smem/regs 卡 1 CTA/SM），
`No Eligible` 65%→69%——即**打掉的是「SM 空转」，不是「延迟隐藏」**。

### 14x.6 复现 / 原始输出

```
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp16 --iters=300 [--wg2ksplit=N]
```
原始输出：`src/fp16/fa_bwd_fp16_o43_sweep.out.txt`、
`..._o43_ncu_wg2_s512_ks{1,2}.out.txt`。

**VARLEN（负结果，opt-in）**：短序列 varlen（如 `[300,200]` H16，base≈96<132）切 K=2
实测 total 0.0684→0.0696ms（**−1.8%**）——Q/dO 重载 + 额外 convert 抵不过填 SM（mblk 少、
tile 切片过细）。**故 varlen 不做 auto，仅 `--wg2ksplit=N>=2` 显式开启**。

## 14y. O44-fp16：MLA（D=512）mma 主 kernel 的 N 方向 split-K（第九十一轮）—— **正结果，默认 auto**

### 14y.1 动机

O43（§14x）已给 fp16 默认档的 `wgmma2`（D=128）加了 N 方向 split-K，解决「S=512 MHA
grid=64 < 132 SM」。但 **MLA（head_dim=512）走的是 mma 主 kernel（`fa_bwd_fp16_mma_kernel`，
`BM=32/BN=32/PIPE=1`），它没有 ksplit**：`grid = ceil(S/32)·H·B`，S=1024H2 只有 **64 CTA**、
S=512H4 64、S=256H2 **16**，ncu 全部 `Waves 0.48`（半个波都不到）、achieved occ 6.25%
（207KB smem → 1 CTA/SM）、No Eligible 91.8%（§7.8）。即 **MLA 的墙是并行度不足、不是带宽/算力**。
本轮把 O43 的机制扩到 mma 主 kernel（从而覆盖 MLA），并保留 HD=128 fallback 的逐位退化。

### 14y.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

`fa_bwd_fp16_mma_kernel<HD,BM,BN,PIPE,...>` 新增运行时 `int ksplit=1`：

1. **Q 块与 K 片**：`ksp=bx%ksplit`、`mblk=bx/ksplit`（同一 mblk 的 ksplit 个 CTA 共享同一份
   Q/dO，均分 KV tile）；`nt_begin=ntiles*ksp/ksplit`、`nt_end=ntiles*(ksp+1)/ksplit`，循环从
   `nt_begin` 起、只发本切片的 K/V；空切片（causal 小 mblk）直接 `return`。
2. **pipeline stage 对齐（第一个坑）**：`stage = nt&1`，所以 prologue 必须把首 tile 发进
   `Ks/Vs + (nt_begin&1)*KVL`；只改数据偏移、忘改 stage 会让首个 tile 读空 buffer——症状是
   split>1 时 dq/dk/dv 误差 O(1)（本轮实测踩到、已修）。
3. **dQ 归约**：`ksplit==1` 时 HD>128 的 GEMM5 是「唯一 CTA + 唯一 warp + 唯一 N-tile」的非原子
   RMW；`ksplit>1` 时同一 `(qi,列)` 被 ksplit 个 CTA 各加一次 ⇒ 改 `red_add2`（`atomicAdd(float2*)`），
   `d_dq_acc` 由 host `memset(0)`。dK/dV 本来就是跨 CTA `red_add2`，不变。
4. **host**：`launch_bwd_mma` 加 `ksplit` 透传；`--mlaksplit=N`（`-1`=auto/`1`=关/`>=2`=强制）；
   auto 仅 `D==512`，目标 **`grid*sp ≈ 528`**（1 CTA/SM 的 4 个波）、上限 16，再按
   `nblk=ceil(S/BN)` 封顶。D!=512 恒 1 ⇒ D=128 路径（wgmma2/mma）逐位不变。

### 14y.3 数值（ours-vs-fp32-ref，fp16 causal，max_abs dq/dk/dv）

与 O5c/§7.8 **逐位一致**（split>1 只改 atomic 次序）：

| MLA case | ours-vs-ref（O44） | 历史（O5c） |
|---|---|---|
| (1,256,2,512) | 1.638 / 1.582 / 1.753e-3 | 同 |
| (1,512,4,512) | 2.516 / 2.916 / 1.724e-3 | 同 |
| (1,1024,2,512) | 1.987 / 1.712 / 1.848e-3 | 同 |

MHA D=128 回归**逐位不变**（S=512 1.671/1.771/1.899e-3，Hopper 构建下 `wgmma2` + O43 ksplit=2
仍为同值）；varlen MLA 走 `ksplit=1`、逐位不变。

### 14y.4 性能（CUDA event；Hopper 构建 `-DFA_WGMMA -DFA_TMA`，同一 binary/同 session）

**split sweep**（`[O44 A/B]`：同一 config `(32,32,1)` 的 ksplit=1/2/4，main-only）：

| MLA case | base grid | ksplit=1 | 2 | 4 | **auto（=8）** | main 比 | total 比 |
|---|---|---|---|---|---|---|---|
| (1,256,2,512) | 16 | 0.1854 | 0.0515 | 0.0281 | **0.0222 ms** | 8.4× | 5.3× |
| (1,512,4,512) | 64 | 0.3685 | 0.1063 | 0.0930 | **0.0840 ms** | 4.4× | 4.2× |
| (1,1024,2,512) | 64 | 0.7171 | 0.2102 | 0.1736 | **0.1524 ms** | 4.7× | 4.7× |

端到端 total：**0.0566 / 0.1284 / 0.2009 ms（4.74 / 16.73 / 21.38 TF）**，对比 O5c
0.300 / 0.536 / 0.939 ms ⇒ **4.2–5.3×**。**auto 比「只填一个波（132）」更激进是实测结论**：
S512H4/S1024H2 在 sp=8（≈512）优于 sp=2/4（§「下一步」候选②的 split-KV 就此落地）。
MLA 的 FA/TE 反向后端均不支持，无第三方对标。~~但 **fp16 MLA total 已比 fp8 MLA（§7.7
0.308/0.591/1.022ms）快 4.9–5.1×**（fp8 MLA 仍 1 CTA/SM 且没吃到这条 split）~~ ——
**本句已被第 92 轮（O45，`docs/03` §46）更正**：§7.7 是 P5-3 时代（第 21 轮）的旧数字，
fp8 MLA 早在 **O29（第 70 轮）** 就有 auto ksplit（`target=S/2`）。本轮实测 fp8 MLA total =
**0.0896 / 0.1996 / 0.2835 ms**，fp16 仅快 **1.4–1.9×**（main 1.3–2.0×），不是 4.9–5.1×。

### 14y.5 ncu（`fa_bwd_fp16_mma_kernel`，S=1024H2，`--set full --launch-count 1`）

| 指标 | ksplit=1 | ksplit=2 |
|---|---|---|
| Duration | 764.5 µs | **228.7 µs** |
| Waves Per SM | 0.48 | **0.97** |
| Issued Ipc Active | 0.33 | **0.57** |
| Achieved Occupancy | 6.25% | 6.23%（仍 1 CTA/SM） |
| Issue Slots Busy / No Eligible | 2.01% / 91.78% | **6.70% / 85.67%** |
| DRAM / L1TEX / L2 / Compute | 0.83 / 33.3 / 14.0 / 2.0 % | 2.78 / 52.7 / 45.9 / 6.7 % |
| 头号 stall | long scoreboard（7.4 cyc） | fixed-latency `wait`（2.2 cyc） |

⇒ **O44 打掉的是「SM 空转」**（Waves 0.48→0.97、每 SM 的 occ 不变），与 O43 同一机制；
填满波后墙回到 **mma 依赖延迟（`wait`）+ L1/L2 吞吐**。

### 14y.6 复现 / 原始输出

```
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp16 --iters=200 [--mlaksplit=N]
```
原始输出：`src/fp16/fa_bwd_fp16_o44_mla_sweep.out.txt`、
`..._o44_ncu_main_ks{1,2}_s1024h2.out.txt`、`..._o44_reg_s512_h16.out.txt`；
单文件 `src/fp16/fa_bwd_fp16_mma_onefile.cu` 同源。

## 14z. O46-fp16：MLA（D=512）mma 主 kernel 的「256 线程 / 8-warp 几何」（第九十三轮）—— **正结果，D=512 默认**

### 14z.1 动机

O45（第 92 轮）把 fp8 MLA 的墙复核后给出结论：MLA 主 kernel 是 **1 CTA/SM（smem ~207KB 锁死）
× 4 warp ⇒ 每个 scheduler 只有 1 个 warp**（ncu `Active Warps/Sched 1.00`、`No Eligible 85.7%`、
`long 2.33 + wait 1.54 + short 0.76`），墙 = **并行度 / 延迟**；`2 CTA/SM` 因 smem 不可达、
`bulkred`/`ILV`/`split` 均已证伪。**「256 线程 / 8-warp 几何」是唯一能提「每 scheduler warp 数」
的杠杆**（fp16/bf16 的 MLA mma 主 kernel 同样 4 warp/1 CTA ⇒ 同一问题）。

### 14z.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对）

把 `fa_bwd_fp16_mma_kernel<HD,BM,BN,PIPE,R4,PREL>` 的 warp 网格从写死的 2×2 改成由
**新模板参数 `NTH`（线程数）+ `NWAR`（N 方向 warp 数）派生**：

* `NWM = NTH/32/NWAR`（M 方向 warp 数）；`wr = wid/NWAR`、`wc = wid%NWAR`；
* warp tile：`GM1=BM/NWM, GN1=BN/NWAR, GMV=BN/NWM, GNV=NTW/NWAR, GMQ=BM/NWM, GNQ=NTW/NWAR`
  （`NTW` 固定 128，与 warp 数解耦）；`__launch_bounds__(NTH, …)`。
* `kv_issue_async`/`qdo_issue_async` 加默认模板参数 `NTH`（默认 `THREADS`），kernel 内
  所有 `THREADS` 换成 `NTH`。

**默认 `NTH=128/NWAR=2` 与 O5c 逐字等价**（`NWM=2`、`NTW=128`、几何全同）；MLA（D=512）用
`NTH=256/NWAR=4`（2×4 网格）⇒ 同 1 CTA/SM 下 **8 warp、每 scheduler 2 warp**。smem 与 grid
不变、dK/dV 归约字节不变。host 加 CLI `--mla8w=0/1`（D=512/BM=32/PIPE=1 默认 1）与
`LAUNCH_CFG_W`；`run_varlen` 保持 4-warp。

### 14z.3 数值（ours-vs-fp32-ref，fp16 causal，max_abs dq/dk/dv）

| shape | 4-warp（=历史） | 8-warp |
|---|---|---|
| S256H2 D512 | 1.638 / 1.582 / 1.753e-3 | **同**（1.638 / 1.582 / 1.753e-3） |
| S512H4 D512 | 2.516 / 2.916 / 1.724e-3 | **同** |
| S1024H2 D512 | 1.987 / 1.712 / 1.848e-3 | **同** |
| MHA S512 D128（未走 8w） | 1.671 / 1.771 / 1.899e-3 | 回归逐位不变 |
| MHA S4096 D128 | 1.883 / 1.734 / 1.966e-3 | 回归逐位不变 |
| GQA kv4 S1024 | 2.134 / 3.305 / 3.850e-3 | 回归逐位不变 |

⇒ 8-warp 档 vs ref 的 max_abs 与 4-warp 档**逐值一致**；D=128 默认路径（`NWAR=2`）与历史
**逐位不变**。

### 14z.4 性能（CUDA event；同 binary、同 session 的 `[O46 A/B]`，main-only，ms）

| shape | 4-warp | 8-warp | 加速 | total 4w→8w |
|---|---|---|---|---|
| S256H2 D512 | 0.0221 | **0.0204** | 1.083× | 0.0549→**0.0534** |
| S512H4 D512 | 0.0839 | **0.0757** | 1.108× | 0.1287→**0.1196** |
| S1024H2 D512 | 0.1512 | **0.1409** | 1.073× | 0.2044→**0.1898** |

main-only 12.2→13.2 TF（S256）/ 25.6→28.4（S512）/ 28.4→30.5（S1024）。**8-warp 实例
148–151 regs、0 spill；4-warp 是 168 regs + 152–296B spill**（每线程累加器更小）。

### 14z.5 ncu（`fa_bwd_fp16_mma_kernel`，S1024H2，`--set full --launch-count 1`）

| 指标 | 4-warp（`--mla8w=0`） | 8-warp（默认） |
|---|---|---|
| Duration | 155.1 µs | **145.4 µs** |
| Achieved Occupancy | 6.10% | **12.36%** |
| Issued Ipc Active | 0.57 | **0.65** |
| Registers Per Thread | 168（+spill） | **151（0 spill）** |
| Waves Per SM | 3.88 | 3.88 |
| DRAM / L1TEX / L2 / Compute | 4.1 / 48.1 / 70.6 / 11.5 % | 4.3 / 50.3 / **73.3** / 13.0 % |
| Warp Cycles / Issued | 6.84 | 12.26（每 warp 等更久，但总吞吐更高） |
| No Eligible | 85.41% | 83.51% |

⇒ **占用率翻倍（6.1→12.4%）、Ipc +14%**，墙仍是 **L2（dK/dV 跨 CTA red）+ 延迟**；提升来自
「每 scheduler 从 1 warp 到 2 warp」，与 O45 的诊断一致（不是带宽/算力）。

### 14z.6 复现 / 原始输出

```
scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp16 --causal --iters=30 [--mla8w=0]
scripts/ncu.sh src/fp16/fa_bwd_fp16_mma_main.cu --set full -c 1 \
  --kernel-name regex:fa_bwd_fp16_mma_kernel -- \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp16 --causal --iters=1 [--mla8w=0]
```
原始输出：`src/fp16/fa_bwd_fp16_o46_sweep.out.txt`、
`..._o46_ncu_mla{4,8}w_s1024h2.out.txt`；单文件 `src/fp16/fa_bwd_fp16_mma_onefile.cu` 同源。

## 14aa. O48-fp16：D=128 **mma fallback** 主 kernel 的「256 线程 / 8-warp 几何」（第九十五轮）—— **正结果仅在 grid ≤ SM 时；大 grid 负结果**

### 14aa.1 动机

O47 的「下一步候选 ①」：D=128 主 kernel 是否也能吃 8-warp。背景是 O46/O47 在 MLA（D=512）
1 CTA/SM 上证明「把每 scheduler 的 warp 数从 1 提到 2」是通用杠杆。对 D=128：
- **生产路径**（`-DFA_WGMMA`）走 `wgmma2`（已 256 线程、2 warpgroup），本项不动；
- **mma fallback**（纯 `sm_90` 构建、或 `--wg2=0`）是 128 线程 / 4 warp。它才是本项对象。

假设：当 `grid ≤ SM 数`（每 SM 只有 1 个 CTA）时，4-warp 的每 scheduler 仅 1 个 warp、
延迟藏不住；8-warp 把它翻到 2，应能加速。反之 grid 已够时 8-warp 只会把 occupancy 砍半。

### 14aa.2 实现（单/两文件 device 逐字一致）

`fa_bwd_fp16_mma_kernel<HD,BM,BN,PIPE,R4,PREL,NTH,NWAR>` 的 warp 网格派生是 O46 已经做好的
（`NWM=NTH/32/NWAR`、`GM1/GN1/GMV/GNV/GMQ/GNQ` 全派生），故本项只加**运行期开关**：

- CLI `--d128w=0/1`（默认 0，与历史 4-warp 逐字相同）。`launch_cfg` 在 D==128 分支里，
  当 `d128w` 时把 `(BM,BN,PIPE)` 映射到 `LAUNCH_CFG_W(...,256,4)`；否则吃原 `LAUNCH_CFG`。
- 主文件的 `[O48 A/B]` 段同 session 对比 `(64,32,1)` 的 4w/8w 并逐元素对拍；生产 auto
  （S=512 用 `(64,64,2)`）用 `--d128w=0/1` 两次运行对比（raw 输出里 `[timing] main`）。
- 单文件 `fa_bwd_fp16_mma_onefile.cu` 同步（bf16 见 `01b` §6ai）。**默认档行为、数值逐位不变**
  （回归见 §14aa.3）。

### 14aa.3 数值（ours-vs-fp32-ref，fp16 causal，max_abs dq/dk/dv）

| shape | 4-warp（历史） | 8-warp | `max_abs(8w-vs-4w)` |
|---|---|---|---|
| S=512 MHA | 1.671/1.771/1.899e-3 | 同量级 | 0 / 2.4e-7 / 4.8e-7 |
| S=4096 MHA | 1.883/1.734/1.966e-3 | 同量级 | 0 / 9.5e-7 / 1.9e-6 |
| S=1024 GQA kv4 | 2.134/3.305/3.850e-3 | 同量级 | 0 / 3.3e-6 / 3.8e-6 |

8w-vs-4w 的差异仅来自 **dK/dV 跨 CTA `atomicAdd` 的次序**（dQ 无 atomic ⇒ 逐位相等），
与 ref 的误差不变 ⇒ 只换 warp 网格、未改数学口径。

### 14aa.4 性能（同 session CUDA event，main-only，ms）

| shape（grid） | 4w(128/2) | 8w(256/4) | 比 |
|---|---|---|---|
| S=512 MHA（grid=128 < 132 SM） | 0.0622 | **0.0589** | **1.056×** |
| S=4096 MHA（grid=1024） | 1.5842 | 1.7972 | 0.881× |
| S=1024 GQA kv4（grid=512） | 0.2810 | 0.2744 | 1.024× |

生产 auto（S=512，`(64,64,2)`）：`--d128w=0` main 0.0568ms / total 0.0952ms；
`--d128w=1` main **0.0514ms**（1.105×）/ total **0.0884ms（1.077×，24.3 TF）**。
对 FA3（同 session 纯反向）S512 0.0264ms/163TF ⇒ ours total 时间比 3.6×→**3.35×**。

### 14aa.5 ncu（`fa_bwd_fp16_mma_kernel`，S=512，auto `(64,64,2)`，`--launch-count 1`）

| 指标 | 4w(128/2) | 8w(256/4) |
|---|---|---|
| regs/thread | 255 | 191 |
| Achieved Occupancy | 6.24% | 12.27% |
| Active Warps / Scheduler | **1.00** | **1.99** |
| Waves / SM | 0.48 | 0.97 |
| Duration | 58.27µs | **52.48µs** |

**机制**：S=512 grid=128 < 132 SM ⇒ 每 SM 只有 1 个 CTA；4-warp 时每 scheduler 只有 **1 个
warp**（`No Eligible 80%`），8-warp 把它翻到 2 ⇒ Duration −10%。S=4096 时 4-warp 已有
1.90 warps/scheduler（≈2 CTA/SM），8-warp 没便宜可占，且每 CTA 覆盖的 m 更多、L1 复用变差，
净 **0.88×**。⇒ **判据是「4-warp 的每 scheduler warp 数是否 <2」，即 grid 是否 ≤ 约 SM 数**。

### 14aa.6 结论 / 复现

- **正结果**（grid≤SM，如 S=512 MHA）：8-warp main 1.06–1.10×、端到端 1.08×；**大 grid 负结果**
  （0.88×）。默认保持 4-warp（opt-in），避免改动历史逐位值；若日后要默认化，推荐 auto 条件
  `D==128 && mma 路径 && grid < 132`（与 O43 的 split-K「填满一个波」同思路，但二者互斥）。
- 生产 `wgmma2`（FA_WGMMA 默认）已是 256 线程 + O43 split-K，故本项主要收益在 **纯 sm_90
  fallback 的可移植路径**。
- 复现：
  ```
  scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp16 --causal [--d128w=1]
  scripts/ncu.sh src/fp16/fa_bwd_fp16_mma_main.cu --kernel-name regex:fa_bwd_fp16_mma_kernel \
    --launch-count 1 --section Occupancy --section SchedulerStats -- \
    --dir=.../b1_s512_h16_d128_causal_fp16 --causal [--d128w=1]
  ```
  原始输出：`src/fp16/fa_bwd_fp16_o48_d128_{s512,s4096,kv4_s1024}.out.txt`、
  `..._o48_ncu_d128_{0,1}w_{s512,s4096}.out.txt`、`..._o48_onefile_s512.out.txt`；
  单文件 `src/fp16/fa_bwd_fp16_mma_onefile.cu` 同源。

## 14ab. O49-fp16：D=128 mma 路径 8-warp 几何的 **自动档默认化**（第九十六轮）

### 14ab.1 动机

O48（§14aa）实测：**grid ≤ SM 数**时（每 SM 仅 1 个 CTA，4-warp 的每 scheduler 只有 1 个
warp）8-warp 几何有 1.05–1.10× 收益；大 grid 负结果。但 O48 把开关留成 opt-in（`--d128w=1`），
理由是「会改 dK/dV 的 atomic 次序、破坏历史逐位值」。O49 落实 O48 §14aa.6 推荐的 auto 条件
`D==128 && mma 路径 && grid ≤ SM 数`，把它**默认化**。

**为什么可以接受非逐位**：8-warp 只改 warp 网格，dQ 无 atomic（逐位相等），差异全部来自
**dK/dV 跨 CTA `atomicAdd` 的次序**（本就是非确定性、run-to-run 都会变），量级 ≤ 5e-7（见
§14ab.3）。历史「逐位值」只是同一 grid 下的一个采样；`--d128w=0` 可随时取回 4-warp 逐位档。

### 14ab.2 改动（单/两文件 device 逐字一致，仅改 host）

- `d128w` 默认 `0` → **`-1`（自动）**：`d128w_eff = (d128w>0) ? true : (d128w<0 ? (D==128 &&
  grid <= sm_count) : false)`，`sm_count` 用 `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)`
  查询（本机 132）。`grid` 是 `(S+63)/64 * H * B`（fp16/bf16 的 mma 路径逻辑网格）。
- `launch_cfg` 增加 `w8` 形参：run_main 传 `d128w_eff`，所有 A/B 段（O5c/O7c/O6c 等）传
  `false` ⇒ 这些对照仍是 4-warp，不受自动档影响。BN=64 分支与 4-warp 路径一致固定 `PIPE=2`。
- 打印 `[O49] d128 8-warp = ...`；auto 触发时同时打印 `[O49]` 与 `[O6c]`。
- D=512（MLA）走 O46 的 `mla8w`，不经此路；`-DFA_WGMMA` 生产构建走 `wgmma2`（已 256 线程），
  `launch_cfg` 不被调用 ⇒ 自动档只在**纯 sm_90 的 mma 可移植路径**上生效。

### 14ab.3 数值（fp16 causal，ours-vs-fp32-ref max_abs dq/dk/dv）

| shape（grid） | 4-warp（`--d128w=0`） | auto（8-warp） | `max_abs(8w-vs-4w)` |
|---|---|---|---|
| S=512 MHA（128） | 1.671/1.771/1.899e-3 | 1.671/1.771/1.899e-3 | 0 / 2.4e-7 / 4.8e-7 |
| S=4096 MHA（1024，auto off） | 1.883/1.734/1.966e-3 | 同（逐位） | 0 / 0 / 0 |
| S=1024 GQA kv4（512，auto off） | 2.134/3.305/3.850e-3 | 同（逐位） | 0 / 0 / 0 |

与 ref 的误差量级不变；auto 只在 grid=128 的 S=512 MHA 上改 dK/dV 的求和次序。

### 14ab.4 性能（同 session CUDA event，plain sm_90 mma 路径，ms）

| 口径 | 4-warp | auto（8-warp） | 比 |
|---|---|---|---|
| S=512 MHA main | 0.0567 | **0.0509** | **1.114×** |
| S=512 MHA total（pre+main+convert） | 0.0960（22.37 TF） | **0.0880（24.40 TF）** | **1.091×** |
| S=4096 main / total | 1.5059 / 1.9288 | 同（auto off） | 1.00× |
| S=1024 GQA kv4 main / total | 0.2612 / 0.3457 | 同（auto off） | 1.00× |

单文件 `fa_bwd_fp16_mma_onefile.cu` auto：main **0.0515** / total **0.0899ms**（与两文件一致）。
同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`：S=512 MHA **FA3 0.0263ms/163TF**、
TE 0.0319/135、FA2 0.0439/98 ⇒ ours total 时间比 **3.65×→3.35×**。

### 14ab.5 ncu（`fa_bwd_fp16_mma_kernel`，S=512，auto `(64,64,2)`，`--launch-count 1`）

| 指标 | 4-warp | auto（8-warp） |
|---|---|---|
| Duration | 57.89µs | **52.96µs** |
| Active Warps / Scheduler | **1.00** | **1.99** |
| Achieved Occupancy | 6.24% | **12.40%** |
| No Eligible | 80.21% | **75.48%** |
| Compute (SM) Throughput | 10.91% | 13.38% |

机制同 O48：grid=128 < 132 SM ⇒ 每 SM 1 个 CTA，4-warp 每 scheduler 仅 1 个 warp；8-warp
把它翻到 2 ⇒ Duration −8.5%。大 grid（如 S=4096 grid=1024）4-warp 已有 ≈2 warps/scheduler，
8-warp 无便宜可占且每 CTA 覆盖的 m 更多、L1 复用变差，故**自动档用 `grid ≤ sm_count` 关门**。

### 14ab.6 结论 / 复现

- **正结果**：S=512 MHA（及任何 `grid ≤ 132` 的 D=128 mma 小网格）默认拿到 8-warp，main
  **1.11×**、端到端 **1.09×**；大 grid 逐位不变、零代价。默认 `-1`（auto），`--d128w=0/1`
  可强制回 4-warp / 强制 8-warp 做同 binary A/B。
- 复现：
  ```
  scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp16 --causal
  scripts/run.sh ... --causal --d128w=0        # 4-warp 对照
  ```
  原始输出：`src/fp16/fa_bwd_fp16_o49_{auto,4w}_s512.out.txt`、
  `..._o49_reg_{s4096,kv4}.out.txt`、`..._o49_onefile_s512.out.txt`、
  `..._o49_ncu_{8w,4w}_s512.out.txt`；单文件与两文件 device 区逐字一致。

## 15. 下一步

> **O23（§14n）已完成**：把 O17/O18 的主 kernel + O9a 的 LSE 在 `-DFA_WGMMA` 构建下**默认打开**
> （对齐 fp8 O22），端到端 **1.08–1.43×**（MHA S4096 1.945→1.364ms），数值与历史逐位一致；
> `--wg2=0 --wg2bn=0`/`--lsewgm=0` 保留 A/B。默认档的墙仍是 **L2 red + 1 CTA/SM**。
>
> **O7b（§14o）已完成（负结果 + 确定性能力）**：DET=partial+二次归约把 `red` 51.9M→**0**、
> 主 kernel 快 **1.20×**（958.6→802.0µs），且**逐位可复现**；但二次归约是纯 DRAM 带宽 bound
> （90.4%、384.9µs），`main+reduce` 在 S=4096 反而 **0.810×**（S=512 时 1.02×）。⇒ 保留
> `--det=1` opt-in，**不作为性能杠杆**；真正消 red 需 cluster 分布式归约（转 backlog）。

见 `../ROADMAP.md`：P1~P4/P5 已收口；**O5（§10）、O8（§11）、O6（§12）、O6b（§12b）、
O8b（§13）、O6c（§13b）、O7c（§14）、MLA 张量核（§14b）、O10（§14c）、O11（§14d）、
O9a（§14e）、O13（§14f）、O9b（§14g）、O9b-2 第一步（§14h）、O15a TMA 通路 + O16 负结果（§14i）、
**O17 跨 wg 归约（§14j）、O17-2 GEMM3/4 拆分再平衡（§14l）、O18 BN=128（§14m）** 完成。O7c 已把「减 red 事务数」这条杠杆**证伪**（float4 更慢），
O10 又把 Q/dO 的标量载入与 dQ 写回向量化（`long_scoreboard` 压下、指令数 −2.5%），
O13 修正了 O6c 的过时 auto tile（S=512 main 1.26×、端到端 1.12×），O9b 把主 kernel 的
GEMM1/2 换成 Hopper `wgmma`（main S512 1.09×/S4096 1.05×，数值逐位不变）。

**§14i（O15a/O16）把墙钉在 L2 的 dK/dV 跨 CTA 原子**；**§14j 的 O17 用 BM=128 + 2 warpgroups
把 red 字节精确砍半**（ncu 102.2M→51.9M），main S=4096 **1.57×**（140.5 TF）、
端到端为 FA3 的 **4.4×**（时间；O9b/O13 时 ~6.0×）。**下一步（按回报排序）**：
1. ~~**O17b（BM=256，4 warpgroups）**~~ **已做，负结果（§14k）**：red 确实精确减半
   （51.9M→26.7M），但 512 线程 ⇒ 每线程只有 128 regs，dQ 累加器 64 + 两条 wgmma 累加器 64
   吃满后必然 spill，local 流量占 L2 ~48%，净 Duration 反而 +21%（S4096）/+83%（S512）。
   ⇒ **「放大 BM」这条路在本卡走不通**，转为下面的 O7b。
2. ~~**O7b（去 dK/dV 原子）**~~ **已做（§14o）——负结果 + 确定性能力**：`DET` 模板把跨 CTA
   `atomicAdd` 换成 per-(b,h,mblk) partial 覆盖写 + `dkv_reduce_kernel` 二次归约；`red`
   51.9M→**0**、主 kernel 快 **1.20×**、**结果逐位可复现**，但二次归约是纯 DRAM 带宽 bound
   （90.4%、384.9µs）⇒ `main+reduce` S4096 **0.810×**（S512 1.02×）。保留 `--det=1` opt-in。
   - ~~**O25（cluster 分布式归约，§14q）**~~ **已做，负结果**：`red` 精确减半（51.9M→26.7M）
     且数值逐位一致，但逐元素远程 `red.shared::cluster.add.f32` 延迟 + 每 tile leader 串行
     flush ⇒ S4096 main **0.13×**（0.99→7.64ms）。保留 `--cluster` opt-in。**至此
     「放大 BM」（§14k）、partial/reduce（§14o）、cluster（§14q）三条消 red 路全部证伪。**
   - **O17-2（§14l）**（已做）：把 O17 的 GEMM3/GEMM4 拆分到两个 wg，消 wg1 的 barrier 空等：
     `barrier` 1.61→0.46、main S4096 1.013×、GQA kv4 1.021×，数值逐位不变。red 字节不变。
3. **fp8 侧的跨 wg 归约**：fp8 是 1 字节 operand、smem 更省，4wg 的寄存器压力比 fp16 小一档，
    同样对 fp8 main 的 L2 red 有效。
4. ~~**O17 的 `BN=128` 微优化**~~ **已做（§14m）**：MHA main **1.029×**（S=4096 0.9895→0.9618ms，
   142.9 TF），GQA/MQA 中性（保持 O17）；`red` 逐字节不变（证实 BN 不动归约结构），
   收益来自 tile 数减半摊薄 barrier/wgmma 固定开销。
5. **TMA 化 operand**：**LSE 的 Q/K 已落地（§14r，O30，第七十一轮）**——4D-TMA + 2×K=64 chunk，
   LSE-only **1.30–1.36×**、指令数 −28.7%、数值逐位不变；但墙不变（Compute ~58% + `wait`）。
   **主 kernel 的 Q/K/V/dO 也已落地（§14s，O33，第七十四轮）**——**逐 atom TMA 复现交织布局**
   （描述符零改动）：`--maintma` opt-in，main **1.04×**（142.6→148.6 TF），`red` 逐字节不变、
   occupancy 不变，端到端为 FA3 的 3.77×（O30 3.87×）。**剩余**：bf16/fp8 的对应 dtype 参数化
   （fp8 SW128 的 `k/16` atom 下标）与 BN=64 的 `wgmma2` 几何。
6. MLA 降 smem 冲 2 CTA/SM / split-KV 仍在列。

## 14ac. O50-fp16：wgmma2 的 GEMM1/2 等待拆分（中性偏正）+ MLA ksplit auto 重标定（正结果，第九十七轮）

> 第九十六轮（O49）把 D=128 的 8-warp 几何默认化后，ROADMAP「下一步候选」的剩余几条都以
> 「mma 依赖延迟 + occupancy 硬件锁死」为由列 backlog。本轮换两个**尚未试过**的、不触
> occupancy/red 结构的角度：① 把 wgmma2 里 GEMM1(S=QKᵀ)/GEMM2(dP=dO·Vᵀ) 的**统一 `wait0`
> 拆开**，用算 P 的 CUDA-core 工作掩盖 GEMM2；② 对 fp16/bf16 的 MLA 主 kernel，把 O44 的
> split-KV **auto 目标重标定**（O46 把几何换成 8-warp 后没重标）。

### 14ac.1 改动（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

- **`FA_WS1`（默认 1）**：`fa_bwd_fp16_wgmma2_kernel`（及 `wgmma2b`）里两条 wgmma 各自 commit
  后，原先是 `wgmma.wait_group 0`（等两条）再 fold；改成先 `wgmma_wait_group<1>()`（只等
  GEMM1 的 `sacc`）算 P、`pds_store_sw128(Pw,...)`，**再** `wgmma_wait0()` 取 GEMM2 的 `dpacc`
  算 dS。数学/数值**逐位不变**（同一批 wgmma、同一累加器、同一 fold 表达式，只改 wait 时机）。
  `-DFA_WS1=0` 退回原路径做 A/B。
- **MLA ksplit auto**：`D==512` 的自动档原为「`grid*sp≈528` 的 2 的幂、再用 `nblk` 封顶」。
  改为 `sp = max(528 目标, 2^k ≤ ceil(nblk/2))`，**去掉 `nblk` 封顶**（切得比 K tile 细只产生
  空切片，kernel 内 `early-exit`）。即「至少把每个 m 块的 K 范围切成 ≈2 份」。`--mlaksplit=N`
  仍可强制。

### 14ac.2 数值（ours-vs-fp32-ref，fp16 causal，max_abs dq/dk/dv）

完全不变：S512 MHA 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、GQA kv4
2.134/3.305/3.850e-3、MLA S1024H2 1.987/1.712/1.848e-3、MLA S256H2 1.638/1.582/1.753e-3、
MLA S512H4 2.516/2.916/1.724e-3。单/两文件逐值一致。

### 14ac.3 性能

**`FA_WS1` A/B（同 binary，300 iters，main-only ms）**：

| shape | ws0（原 wait0） | ws1（拆分） | 比 |
|---|---|---|---|
| S512 MHA | 0.0315 | 0.0315 | 1.00× |
| S4096 MHA | 0.9558 | **0.9543** | 1.002× |
| GQA kv4 S1024 | 0.1801 | **0.1782** | 1.011× |
| MQA kv1 S1024 | 0.2868 | 0.2865 | 1.001× |

⇒ 方向一致**非负**，但幅度 ≤1%（多数在噪声内）。ncu（S512）只见 `barrier` 0.39→0.25、
Duration 33.57→33.47µs；`red`/指令数逐字节不变。**故 fp16/bf16 默认开（小幅非负），fp8 默认关**
（见 `docs/03` §50）。

**MLA ksplit auto 重标定（main-only ms，`--mlaksplit` sweep）**：

| MLA shape | 旧 auto | 新 auto | k=8 | k=16 | 结果 |
|---|---|---|---|---|---|
| S1024H2 | 8 | **16** | 0.1411 | **0.1366** | **1.033×** |
| S512H4 | 8 | 8 | **0.0756** | 0.0801 | 不变 |
| S512H2 | 16 | 16 | 0.0465 | 0.0454 | 不变 |
| S256H2 | 8 | **16** | 0.0204 | **0.0192** | **1.063×** |

端到端 total：MLA S1024H2 0.1911→**0.1855ms（1.030×）**、S256H2 0.0544→**0.0533ms（1.021×）**。
根因：O46 把 MLA 几何从 4-warp 换成 8-warp（1 CTA/SM 下每 scheduler 2 warp），split-KV 的
最优粒度随之变大——旧的 `528`（4 个波）对 S1024H2 只切到 8，实测 16（≈8 个波）更快。

### 14ac.4 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py fp16`）

FA3 MHA S512 0.0265ms/162TF、S4096 0.3240/848、GQA kv4 0.0826/416（TE 见原始输出）。
ours total：S512 0.0684ms（**FA3 的 2.58×**，O43 后 2.63×）、S4096 1.2532ms（**3.87×**）、
GQA kv4 0.2446ms（2.96×）。原始输出 `src/fa_bwd_o50_fa3_te_fp16.out.txt`。

### 14ac.5 复现 / 原始输出

```bash
F='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
ARCH="" NVCC_FLAGS="$F" scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp16 --iters=300
ARCH="" NVCC_FLAGS="$F -DFA_WS1=0" scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu ...   # A/B
# MLA ksplit sweep：--mlaksplit=4/8/16/32
```

原始输出：`src/fp16/fa_bwd_fp16_mma_o50_*.out.txt`（6 shape 对拍+计时）、
`..._mma_onefile_o50_*.out.txt`（单文件）、`..._o50_mlaksplit_sweep.out.txt`、
`..._o50_ws_ab.out.txt`、`..._o50_ncu_wg2_s512.out.txt`、`src/fa_bwd_o50_fa3_te_fp16.out.txt`。

## 14ad. O52-fp16：把 MLA 的 8-warp 几何推广到 **fp16 varlen**（第九十九轮）—— **正结果，varlen MLA 默认**

### 14ad.1 动机 / 改动（host-only；device 逐字不变）

O46（fp16/bf16 MLA 主 kernel 的 256 线程 / 8-warp 几何）只落在**定长** `D=512` 路径；
`run_varlen` 的 MLA 主 kernel 仍是历史 `launch_bwd_mma<512,32,32,1,false,true>`（默认
`NTH=128,NWAR=2`）。本轮把 O46 的几何搬进 varlen：`run_varlen` 加 `mla8w` 入参，`D==512`
分支在 8-warp `<512,32,32,1,false,true,256,4>` 与历史 4-warp 之间按 `--mla8w` 选择；`main`
把已解析的 `--mla8w=` 透传；`run_varlen` 末尾加 `[O52 A/B]` 段（同 binary 计时 + 逐元素比对）。
**device 代码一行未改**（`fa_bwd_fp16_mma_kernel` 早由 O46 参数化），单/两文件 host 同步、
device 仍 `identical: True`。

### 14ad.2 数值（ours vs fp32 ref，fp16 causal varlen；max_abs dq/dk/dv）

与 §16.9 历史**同量级/逐值一致**；8-warp 只改 dK/dV 的 atomic 次序（`max_abs(8w-vs-4w)`
dq=0、dk/dv ≤1e-6）：

| case (D=Dv=512) | dq | dk | dv | main 4w→8w |
|---|---|---|---|---|
| b1_t512 causal | 1.303e-3 | 1.537e-3 | 1.557e-3 | 0.3885→0.2379ms (**1.63×**) |
| b3_t1792 causal | 2.415e-3 | 1.834e-3 | 1.856e-3 | 0.8495→0.5586ms (**1.52×**) |

**单文件与两文件逐指标一致**。D=128 varlen 回归**逐位不变**（只改 `D==512` 分支）。
（非 causal full 的 MLA varlen：**更正**——并非 HEAD 偏差，而是 §14ad 的对拍脚本漏传 `--full`
导致按 causal 跑（见 §14ae.2）；显式 `--full` 后 fp16 full 的 max_abs 仅 3.0e-4/4.4e-4/1.3e-4。）

### 14ad.3 性能（event，`Σ_b 4HL²D` 口径，同 session）

| case | O52 前（§16.9，4-warp） | O52（8-warp） | 加速 |
|---|---|---|---|
| b1_t512 causal total | 0.4264 ms / 2.52 TF | **0.2740 ms / 3.92 TF** | 1.56× |
| b3_t1792 causal total | 0.9382 ms / 6.01 TF | **0.6523 ms / 8.64 TF** | 1.44× |

MLA 反向 FA2/FA3/TE 均不支持 head_dim=512 ⇒ 无外部基线。墙仍是 1 CTA/SM（smem 硬约束）+
smem→mma 依赖；8-warp 把每 scheduler 的 warp 数 1→2（与 O46 同机制）。

### 14ad.4 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b1_t512_h2_d512_causal_fp16 --varlen --iters=30
... --mla8w=0    # A/B 退回 4-warp
```

原始输出：`src/fp16/fa_bwd_fp16_mma_main_o52_varlen.out.txt`、
`src/fp16/fa_bwd_fp16_mma_onefile_o52_varlen.out.txt`。

## 14ae. O53-fp16：把 MLA 的 **N 方向 split-K（split-KV）** 推广到 **fp16 varlen**（第 100 轮）—— **正结果，varlen MLA 默认 auto**

### 14ae.1 动机 / 改动（host-only；device 逐字不变）

O52 把 O46 的 8-warp 几何搬进 varlen 后，MLA varlen 主 kernel 只剩 **单个 CTA 做整条 K**：
`D=512`、`BM=32`，base grid = `ceil(maxlen/32)·H·B` 在 `b1_t512_h2` 只有 **16 CTA**
（`b3_t1792_h2` 也只 32×2×3=192），而 smem 207.36KB 锁死 **1 CTA/SM** ⇒ ncu `Waves 0.48`、
大量 SM 空转。定长侧 O44/O50 早已用「N 方向 split-K」解决（把每个 m 块的 KV tile 均分给
`ksplit` 个 CTA，dQ 改跨 CTA `red_add2`）；**varlen 的 `run_varlen` 一直传 `ksplit=1`**。

本轮把该机制搬进 varlen：`run_varlen` 加 `mlaksplit` 入参，`D==512` 时按定长同款 auto 选
`mla_ks_eff`（`--mlaksplit=N` 可强制/关），`mg.x *= mla_ks_eff`，并把 `mla_ks_eff` 传给两处
`launch_bwd_mma<512,32,32,1,...>`。auto 口径与定长完全一致：① target `base*sp≈528`（1 CTA/SM
的 4 个波）、cap 16；② 至少把每个 m 块的 K 范围切成 ≈2 份（`nt_cap/2`）。`main` 透传
`--mlaksplit=`。**device 代码一行未改**（O44 早已支持切片、空切片早退、跨 CTA dQ `red_add2`，
`ksplit==1` 逐式退化），单/两文件 host 同步。

顺带在 `run_varlen` 末尾加 `[O53 A/B]`（固定 8-warp 几何、main-only，sweep `ksplit=1/2/4/8/16`），
并把 varlen 的 grid 打印从「未乘 ksplit 的 base」改成实际 `mg`。

### 14ae.2 「HEAD 偏差」更正（非本改动引入）

O52 曾记「三 dtype 非 causal（full）MLA varlen 在 HEAD 已偏差（max_abs≈7）」——本轮定位为
**对拍脚本漏传 `--full`**（输出头 `causal=1`），即按 causal 去比 full 的 ref，误差自然 O(1)。
显式 `--full` 后 fp16 full 的 max_abs 仅 **3.0e-4 / 4.4e-4 / 1.3e-4**（b1_t512，
`[timing]` 头 `causal=0`），fp8 full ≈5e-2（与第 80/81 轮记录一致），bf16 full ≈1.7e-3。
**不是回归、不是 kernel bug**，无需修复。

### 14ae.3 数值（ours vs fp32 ref，fp16 varlen；max_abs dq/dk/dv）

| case (D=Dv=512) | dq | dk | dv |
|---|---|---|---|
| b1_t512 causal | 1.303e-3 | 1.537e-3 | 1.557e-3 |
| b3_t1792 causal | 2.415e-3 | 1.834e-3 | 1.856e-3 |
| b1_t512 full | 3.046e-4 | 4.449e-4 | 1.327e-4 |
| b3_t1792 full | 5.516e-4 | 4.451e-4 | 2.385e-4 |

均 fp16 噪声量级。**单文件与两文件逐指标一致**（上表两文件/单文件完全相同）；`max_abs(8w-vs-4w)`
dq ≤2e-7、dk/dv ≤1e-6（仅跨 CTA atomic 次序），`max_abs(ksplit=16-vs-1)` 同量级。MLA 反向
FA2/FA3/TE 均不支持 `head_dim=512` ⇒ 无外部基线；D=128 varlen 回归逐位不变。

### 14ae.4 性能（CUDA event，`Σ_b 4HL²D` 口径，同 session）

| case | O52（8-warp，ksplit=1） | O53（8-warp，auto split-KV） | 端到端加速 |
|---|---|---|---|
| b1_t512 causal total | 0.2740 ms | **0.0819 ms / 13.11 TF** | **3.35×** |
| b3_t1792 causal total | 0.6523 ms | **0.3518 ms / 16.02 TF** | **1.85×** |
| b1_t512 full total | — | 0.2853 ms / 3.76 TF | — |
| b3_t1792 full total | — | 1.0093 ms / 5.59 TF | — |

**主 kernel-only（同 session `[O53 A/B]`，8-warp）**：

| case | ksplit=1 | 2 | 4 | 8 | 16（auto） | split 加速 |
|---|---|---|---|---|---|---|
| b1_t512 causal | 0.2363 | 0.0833 | 0.0549 | 0.0464 | **0.0454** | **5.21×** |
| b3_t1792 causal | 0.5532 | 0.3007 | 0.2649 | 0.2530 | **0.2544** | 2.18× |
| b1_t512 full | 0.2401 | 0.0945 | **0.0665** | 0.0709 | 0.0743 | 3.61×（最优 k=4） |
| b3_t1792 full | 0.6049 | 0.3864 | 0.3898 | **0.3863** | 0.3903 | 1.57×（最优 k=8） |

**结论**：causal auto 即最优（b1 选 16、b3 选 16 与最优 8 差 0.6%）；**非 causal full 的 auto
偏大**（b1 最优 k=4、auto=16 落后 12%），因为 full 的每个 m 块都扫满 K、不像 causal 那样随
mblk 递增，故「把每块 K 切 ≈2 份」的 `sp_min` 启发式偏激进。full 是次要路径（性能主场是
causal），且相对 k=1 仍 3.2×/1.6×，本轮不改 auto，`--mlaksplit=4` 可选最优。
端到端 total 还含非 causal 的 LSE（`lse_mma_kernel<512>`，未做负载均衡，是 full 端到端的新瓶颈，
约占 b3 full total 的 60%）——留 backlog。

### 14ae.5 ncu（`fa_bwd_fp16_mma_kernel`，varlen b3_t1792 H2 D512 causal，`--launch-count 1`）

| 指标 | 值 |
|---|---|
| Duration | **259.4 µs**（split-KV 后；k=1 时每 CTA 串行扫整条 K，见 `[O53 A/B]` 的 0.553ms） |
| DRAM / L1TEX / **L2** / Compute | 4.69% / 52.74% / **80.90%** / 19.33% |
| regs / smem / occ（theoretical→achieved） | 151 / 207.36KB / 12.50%→**12.21%** |
| Block Limit | **Registers=1、Shared Mem=1**（1 CTA/SM）；Waves **23.27** |
| stall ratio | **long 2.73** + wait 2.28 + short 1.58 + barrier 0.14 |
| L2 `op_red` / `op_read` sectors | **18.41M / 12.91M**（red 占 L2 扇区 **58.8%**） |

bf16 同 shape **逐项一致**（Duration 259.1µs、L2 81.02%、occ 12.21%、red 18.407M）；
`Block Limit Registers=1`（151 regs）与 `Shared Mem=1`（207.36KB）双卡 1 CTA/SM。
**bound = L2（跨 CTA dQ `red_add2`，58.8% 扇区）+ 全局/共享访存延迟（long 2.73 + wait 2.28）**，
不再是 O52 的「grid 不足一个波」。这与定长 MLA 的 O44 结论（split-KV 后墙移到 L2 red + 延迟）
一致；下一步要么降红（更大 BM/跨 warpgroup 归约），要么降 smem 冲 2 CTA/SM。

### 14ae.6 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA'
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --varlen --causal /home/xieminglin/proj/output/fa-bwd/varlen_b3_t1792_h2_d512_causal_fp16
# --full 走非 causal；--mlaksplit=1 关 split / =4 强制
```

原始输出：`src/fp16/fa_bwd_fp16_o53_varlen.out.txt`（两文件 4 case + 单文件 2 case）、
`src/fp16/fa_bwd_fp16_o53_ncu_varlen_b3.out.txt`、`..._o53_ncu_stall_varlen_b3.out.txt`；
bf16 同构 `src/bf16/fa_bwd_bf16_o53_varlen.out.txt`、`..._o53_ncu_varlen_b3.out.txt`。

---

## 15. O54-fp16：非 causal（full）MLA varlen 的 LSE 走 K 维 split（第 101 轮）—— 正结果，full varlen 默认 auto

### 15.1 动机

O53 把 split-KV 搬进 varlen **主 kernel** 后，`b3_t1792` full 的端到端仍是 **1.013 ms**，其中
主 kernel-only 仅 0.392 ms —— **LSE 占 ~0.62 ms（60%）**。原因：非 causal 的 MLA LSE 一直走
O1 的 `lse_mma_kernel<512>`（**每个 m 块一个 CTA、无 K 维 split、标量 K 载入**），而 causal
早已用带 **镜像配对 + `cp.async` 双缓冲 + O40 split** 的 `lse_mma_kernel_bal<512,1>`。full 下
所有 m 块工作量相同（无需镜像配对），但**缺少 split/双缓冲**，`b3` 的 base grid 只有
`nblk0·H·B = 16·2·3 = 96 < 132 SM`、单 CTA 顺序扫 16 个 K tile ⇒ 并行度不足 + 延迟受限。

### 15.2 实现（单/两文件 device 逐字一致）

给 `lse_mma_kernel_bal` 加模板参数 **`bool FULL = false`**（`docs/01b`/`docs/03` 同款）：
* `FULL=true`：`grid.x=nblk`，**一个 CTA 一个 m 块**（不做镜像配对，`t==1` 用 `if constexpr` 删掉，
  `mblk=pair`）；`ncols = len`（不是 `min(len, m0+LBM)`）；掩码去掉 `jg<=qi`。
* `FULL=false`（默认）经 `if constexpr` 化简后**编译出与原版逐位相同的代码**，causal/定长不受影响。
* 复用同一套 `issue_q/issue_k`（`cp.async` 双缓冲）、online-softmax、4-lane `shfl` 归约、
  O40 的连续 K tile 切片 + `lse_split_merge_kernel` 二次归约。

host 侧（`run_varlen`，`D==512 && !causal`）：`lse_split_eff > 1` 时 launch
`lse_mma_kernel_bal<512,1,true>`，`grid=(nblk,H,B*split)`，再跑 merge；否则退回原
`lse_mma_kernel<512>`（历史路径）。auto：`base = nblk0·H·B`、目标 **≈3 个波（384）**、cap 16，
再按 `nblk0` 封顶——`b1` 得 8、`b3` 得 4（都是 sweep 最优）。末尾加 `[O54 A/B]`（LSE-only）。

### 15.3 数值（ours vs fp32 ref，fp16 full varlen，max_abs dq/dk/dv）

| case | dq / dk / dv | 与 old 路径 |
|---|---|---|
| b3_t1792 full | 5.516 / 4.451 / 2.385e-4 | **逐位相同**（仅 fp32 求和次序，split 亦然） |
| b1_t512 full | 3.046 / 4.449 / 1.327e-4 | 逐位相同 |

`[O54 A/B]` 的 `balFULL/split1` 与 old 值一致；split>1 只改 `lse_part` 的 fp32 归约次序，
max_abs 不变。单/两文件逐指标一致。

### 15.4 性能（CUDA event，LSE-only 同 session；total 端到端）

| case | old O8 LSE | split1 | split2 | **split4** | split8 | split16 | auto | total old→new |
|---|---|---|---|---|---|---|---|---|
| b3_t1792 full | 0.3791 | 0.0826 | 0.0468 | **0.0411** | 0.0474 | 0.0577 | 4 | **1.013→0.468 ms（2.17×）** |
| b1_t512 full | 0.1933 | 0.0445 | 0.0266 | 0.0168 | **0.0135** | 0.0236 | 8 | ~0.28→**0.102 ms** |

LSE 本身 **9.2×（b3）/ 14.3×（b1）**；`split1` 也已有 4.6× —— 主要来自 **`cp.async` 双缓冲 +
一个 CTA 一个 m 块**（old 是标量逐字节载入）。

### 15.5 ncu（`lse_mma_kernel_bal<512,1,true>`，b3 full，split4，`-c 1`）

Duration **40.99 µs**、DRAM 5.38% / L1TEX 28.52% / L2 24.00% / Compute 20.55%、
regs 63、smem **199.68 KB → 1 CTA/SM**、occ 6.25%、**Waves 2.91**、
No Eligible **74.1%**、Active Warps/Sched **1.00**、fixed-latency stall **37.3%**。
**bound = 低 occupancy（1 CTA/SM，smem 硬约束）+ fixed-latency 依赖**，非带宽/算力。
split 已把 `Waves` 从 old 的不足一波抬到 2.91；再往上要降 smem（去 `Qs`/单缓冲）才能冲 2 CTA/SM。

### 15.6 回归

* **causal b3（D=512）逐位不变**：dq/dk/dv = 2.415/1.834/1.856e-3（与 O53 完全一致），
  main 8-warp 0.2558 ms。
* **D=128 full varlen（`varlen_b4_t4096_h16_d128_full_fp16`）逐位不变**：total 0.9075 ms
  （O53 记 0.909），dq/dk/dv = 4.094/4.953/1.234e-4。

### 15.7 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --varlen --full /home/xieminglin/proj/output/fa-bwd/varlen_b3_t1792_h2_d512_full_fp16
# --lsesplit=1 关 split；[O54 A/B] 会打印 old 与 split1/2/4/8/16
```

原始输出：`src/fp16/fa_bwd_fp16_o54_varlen_full_b3.out.txt`（两文件）、
`..._o54_varlen_full_b3_onefile.out.txt`（单文件）、`..._o54_varlen_full_b1.out.txt`、
`src/fp16/fa_bwd_fp16_o54_ncu_lse_full_b3.out.txt`（ncu）。

## 15b. O55-fp16：varlen MLA full 的 split-KV auto 重新标定（第 102 轮）—— 正结果，full varlen 默认

### 15b.1 动机

O53 给 varlen MLA（D=512）主 kernel 加的 N 方向 split-KV，其 auto 对 **causal 与 full 用了同一个
目标**（`base*sp ≈ 528`，即 1 CTA/SM 的 4 个波）+ 「每 m 块 K 切 ≈2 份（`nt_cap/2`）」。但 `[O53 A/B]`
的 sweep 显示 **full 的最优 split 明显更小**：

| case | k=1 | k=2 | k=4 | k=8 | k=16 | O53 auto |
|---|---|---|---|---|---|---|
| b1_t512 causal | 0.2350 | 0.0835 | 0.0544 | 0.0463 | **0.0455** | 16 ✓ |
| b3_t1792 causal | 0.5543 | 0.3014 | 0.2642 | **0.2535** | 0.2543 | 16（≈最优） |
| b1_t512 **full** | 0.2394 | 0.0918 | **0.0647** | 0.0709 | 0.0748 | **16（偏大 1.16×）** |
| b3_t1792 **full** | 0.6022 | **0.3873** | 0.3903 | 0.3868 | 0.3909 | 16（≈最优） |

即 full b1 的 auto 给出 k=16（0.0748 ms），最优却是 k=4（0.0647 ms，**1.16×**）。原因：full 主 kernel
每 m 块工作量相同、base grid 已能靠少量切分铺到「一个波」（`base=32`，k=4 → 128 CTA ≈ 1×132 SM）；
再往 16 切只是在**重复读 Q/dO + 增加 dQ 跨 CTA atomic**，纯亏。

### 15b.2 实现（host-only；单/两文件 device 逐字一致）

`run_varlen` 的 O53 auto 分支：**causal 分支逐字保持 O53 口径**（`target=528` + `sp_min=(nt_cap+1)/2`）
以保逐位回归；**full 分支改用 `target=132`（1 个波）、`sp_min=2`**（b3 的 base=192 已超 132、
target 会给 1，实测 k=2 最优）：

```cpp
const int target = causal ? 528 : 132;
while (sp < 16 && base * (sp * 2) <= target) sp *= 2;
int sp_min = 1;
if (causal) while (sp_min < 16 && sp_min * 2 <= (nt_cap + 1) / 2) sp_min *= 2;
else        sp_min = 2;                 // full：至少 2 份
if (sp_min > sp) sp = sp_min;
```

`D==512` 时 `mg.x *= mla_ks_eff` 照旧。device 一行未改（O44 早已支持切片 + 空切片早退 +
dQ 跨 CTA `red_add2`，`ksplit==1` 逐式退化）。得到 **b1 full→4、b3 full→2**。

### 15b.3 数值（ours vs fp32 ref，fp16 varlen，max_abs dq/dk/dv）

| case | O55 auto | dq / dk / dv | 与 O53/O54 |
|---|---|---|---|
| b1_t512 full | 4 | 3.046 / 4.449 / 1.327e-4 | 逐位相同 |
| b3_t1792 full | 2 | 5.516 / 4.451 / 2.385e-4 | 逐位相同 |
| b1_t512 causal | 16 | 1.303 / 1.537 / 1.557e-3 | 逐位相同（causal 分支未动） |
| b3_t1792 causal | 16 | 2.415 / 1.834 / 1.856e-3 | 逐位相同 |

split 只改 `dQ` 跨 CTA atomic 次序（fp32 累加），max_abs 不变；单/两文件逐指标一致。

### 15b.4 性能（CUDA event，同 session；`[O53 A/B]` 是同一 binary 内的 A/B）

* **b1_t512 full**：main（auto 选中）**0.0745 → 0.0681 ms（1.094×）**；同 binary sweep
  k=4 **0.0647** vs k=16 0.0748（**1.16×**）；**端到端 total 0.1030 → 0.0958 ms（1.075×，11.20 TF）**。
* **b3_t1792 full**：main 0.3888 → **0.3873**（≈1.004×）；total 0.4680 → **0.4636 ms（1.009×）**。
* **causal b1/b3**：auto 仍 16、total 0.0820/0.3512 与 O53 持平（回归）。

### 15b.5 ncu（fp16 b1 full main，`-c 1`，同 session A/B）

| auto | grid | Waves | Duration | L1TEX | L2 | Compute | occ |
|---|---|---|---|---|---|---|---|
| O53 k=16 | 256×2×1 | 3.88 | 77.54 µs | 44.11% | 67.95% | 14.69% | 12.27% |
| **O55 k=4** | 64×2×1 | **0.97** | **68.00 µs（1.14×）** | 49.06% | 73.24% | 12.49% | 12.44% |

**读法**：k=4 恰好填满一个波（`Waves 0.97`），而 k=16 要跑 3.88 个波；虽然 k=4 的 L2% 更高
（每元素 dQ 归约次数少、Q/dO 重读少，单位时间 L2 占用更集中），Duration 反而短 14%。
**bound 仍是 L2（dK/dV 跨 CTA `red`）+ 1 CTA/SM 的低 occupancy**；本次是**去掉过切**而非改结构。
b3 full（auto=2，grid 64×2×3=384）Duration 387 µs、L2 83.79%、Waves 2.91、occ 12.46%。

### 15b.6 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA'
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp16/fa_bwd_fp16_mma_main.cu \
  --varlen --full /home/xieminglin/proj/output/fa-bwd/varlen_b1_t512_h2_d512_full_fp16
# 同 binary A/B：--mlaksplit=4 / 16；[O53 A/B] 打印 1/2/4/8/16
```

原始输出：`src/fp16/fa_bwd_fp16_o55_varlen.out.txt`（两文件，full+causal b1/b3）、
`..._o55_varlen_onefile.out.txt`（单文件）、`..._o55_ncu_varlen_full_b1_t512.out.txt`（ncu k=4）、
`..._o55_ncu_varlen_full_b1_ks16.out.txt`（ncu k=16 对照）、`..._o55_ncu_varlen_full_b3_t1792.out.txt`。
