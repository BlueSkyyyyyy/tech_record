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

---

## 13. 下一步

见 `../ROADMAP.md`：P1~P4/P5 已收口；**O5（§10）、O8（§11）、O6（§12）** 完成。
后续按回报排序：**O6b/降 smem**（fp16 main 现 2 CTA/SM，想办法把 K/V 双缓冲压回
3 CTA/SM，如只双缓冲 K 或更省的转置布局）→ **O9**（`wgmma`+TMA+warp specialization，
对标 FA3；或先把 `short_scoreboard`/`wait` 依赖压下去）→ **O7**（dK/dV 去 `atomicAdd`，
移植 fp8 的 O4c/O7）→ **O8b**（LSE 的 occupancy/尾波、convert 融合）→ MLA 张量核。
