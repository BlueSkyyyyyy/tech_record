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

## 15. 下一步

见 `../ROADMAP.md`：P1~P4/P5 已收口；**O5（§10）、O8（§11）、O6（§12）、O6b（§12b）、
O8b（§13）、O6c（§13b）、O7c（§14）、MLA 张量核（§14b）、O10（§14c）、O11（§14d）、
O9a（§14e）、O13（§14f）、O9b（§14g）、O9b-2 第一步（§14h）、O15a TMA 通路 + O16 负结果（§14i）、
**O17 跨 wg 归约（§14j）** 完成。O7c 已把「减 red 事务数」这条杠杆**证伪**（float4 更慢），
O10 又把 Q/dO 的标量载入与 dQ 写回向量化（`long_scoreboard` 压下、指令数 −2.5%），
O13 修正了 O6c 的过时 auto tile（S=512 main 1.26×、端到端 1.12×），O9b 把主 kernel 的
GEMM1/2 换成 Hopper `wgmma`（main S512 1.09×/S4096 1.05×，数值逐位不变）。

**§14i（O15a/O16）把墙钉在 L2 的 dK/dV 跨 CTA 原子**；**§14j 的 O17 用 BM=128 + 2 warpgroups
把 red 字节精确砍半**（ncu 102.2M→51.9M），main S=4096 **1.57×**（140.5 TF）、
端到端为 FA3 的 **4.4×**（时间；O9b/O13 时 ~6.0×）。**下一步（按回报排序）**：
1. **O17b（BM=256，4 warpgroups）**：同一机制再砍半（red → ~26M），但 smem ≈ 270KB 需
   先把 Q/dO 改「按 wg 只存自己 64 行」或 TMA 直供、或把 P/dS 用 8-bit 存；
   先做寄存器/smem 账再上。
2. **O7b（去 dK/dV 原子）**：分块 `*_accum` + convert（确定性）；字节不减、多一趟读回，
   但可与 O17 的「CTA 内归约」叠加。
3. **TMA 化 Q/K/V/dO（O15a 通路已就绪）**：需把 K-major HD=128 tile 改成 **2×K=64 chunk**
   （§14i.2），压 `long_scoreboard`/指令数；动不了 L2 red，排在后面。
4. **bf16 版**照搬（`__half`→`__bfloat16`，SW128/描述符/累加器映射逐字节同构）。
5. MLA 降 smem 冲 2 CTA/SM / split-KV 仍在列。
