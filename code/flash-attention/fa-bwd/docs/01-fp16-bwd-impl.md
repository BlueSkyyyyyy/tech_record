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

## 8. 下一步

见 `../ROADMAP.md`：P1-4 已完成（本节），随后 P2 bf16（复用 fp16 骨架做 dtype 参数化）、P3 fp8（重点）。
性能优化（消 bank conflict / 提 occupancy / 张量核）列入 backlog。
