# FP8 反向实现分析（P3-2，golden 版）

> 代码：`src/fp8/fa_bwd_fp8_onefile.cu`（单文件：quantize + preprocess + main + convert + 自测）
> 设计：`docs/02-fp8-bwd-design.md`（量化/scaling 布局）
> 原始输出：`src/fp8/fa_bwd_fp8_onefile_s512.out.txt`、`..._s1024h32.out.txt`、`..._s4096.out.txt`、
> `..._ncu_main.out.txt`（ncu `--set full`）、`..._refbench.out.txt`（TE FP8 基线）。
> 本轮只做 **P3-2（单文件 golden）**；mma/两文件/文档分册留待 P3-4/P3-5。

---

## 1. 实现结构（四段式）

| 段 | 函数 | 职责 |
|---|---|---|
| quantize | `quantize_row_kernel` | fp32 `q/k/v`→E4M3、`do`→E5M2，逐行算 `amax`+scale（模拟 TE 计时区外预量化） |
| preprocess | `preprocess_kernel` | 反量化 Q/K 重算 `LSE`；反量化 dO 与 fp32 O 算 `D=rowsum(dO∘O)` |
| main | `fa_bwd_fp8_kernel` | 1colblock，反量化载入 Q/K/V/dO，fp32 标量累加；P/dP 量化 |
| convert | `convert_kernel` | fp32 累加缓冲 → 输出（本版 fp32） |

`BM=64, BN=32, THREADS=128, head_dim=128`，与 fp16/bf16 版逐一对齐，便于横向比较。
动态 smem 68.35KB（fp8 主数据 1B/元素 + scale + fp32 累加）。

### 1.1 FP8 量化 / 反量化

- 转换必须走 `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3/E5M2)` 与
  `__nv_cvt_fp8_to_halfraw(...)`。**不能用 `float(fp8)`**：该 CUDA 工具链下它返回的是
  **原始位模式**而非数值（见 `agent_skills/kernel-opt.md` 32 篇），会得到永远对不上的误差。
- `scale = amax / FP8_MAX`（E4M3: 448，E5M2: 57344），fp32 任意值；反量化 `x' = float(q)*scale`。
- **标量 golden 在反量化时已乘回 scale**，所以等价于「真实 FP8 张量核 + `sa*sb` 正确折算」，
  无需在累加后再折（真实 mma 版才需要 epilogue 折 `sa*sb`，见 `docs/02` §3.2）。

### 1.2 量化对象

| 张量 | dtype | 粒度 | 实现位置 |
|---|---|---|---|
| Q/K/V | E4M3 | rowwise（每 `(b,s,h)` 行 over D） | `quantize_row_kernel` |
| dO | E5M2 | rowwise | `quantize_row_kernel` |
| P | E4M3 | scale=1（`P∈[0,1]`） | main kernel 内 `cvt_e4m3(p)` |
| dP | E5M2 | rowwise（warp `__shfl_xor` 求行内 amax） | main kernel 内 |
| dS / LSE / D / 累加器 | fp32 | — | main kernel 内 |

`dS = P'∘(dP'−D)` 中 `P'` 用反量化后的 E4M3 P、`dP'` 用 E5M2 反量化；mask 位置 `P=0`
量化后仍为 `0`（`cvt(0)=0`），causal 语义保持。

### 1.3 与 fp16/bf16 版的差异

- 新增 `quantize_row_kernel`，Q/K/V/dO 在全局内存是 **FP8(1B)+scale**，载入 smem 时逐元素反量化。
- S 的缩放是三级：`dot *= scale * qs[r] * ks[lane]`；dK/dQ 同理带 `qs/ks`。
- dV/dK/dQ 的 P/dO/Q/K 都带反量化因子，累加仍是 fp32。
- 输出保持 fp32（TE 返回 bf16），对比时注意口径差（见 §3）。

---

## 2. 数值对拍

容差参考：TE FP8 反向 vs fp32 ref 的 `max_abs` ≈ 0.3–0.9（`docs/02` §4），即 **O(1) 绝对误差**。
ref 梯度幅值：`amax(dq)=2.96, amax(dk)=3.24, amax(dv)=6.22`（S=512 case），
所以相对量级约 **8%–20%**。`max_rel`（除以 `|ref|+1e-3`）会被近零元素放大到 1e2–1e3，不作为判据。

### S=512, B=1, H=16, D=128, causal（case `b1_s512_h16_d128_causal_fp8`）

| | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|
| **ours vs fp32 ref** | **4.815e-1** | **6.426e-1** | **3.319e-1** |
| TE FP8 vs fp32 ref | 4.859e-1 | 4.038e-1 | 5.906e-1 |
| **ours vs TE FP8** | **5.384e-1** | **5.524e-1** | **8.775e-1** |

### S=4096, B=1, H=16, D=128, causal

| | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|
| **ours vs fp32 ref** | 4.568e-1 | 5.009e-1 | 3.786e-1 |
| TE FP8 vs fp32 ref | 3.763e-1 | 3.686e-1 | 6.688e-1 |
| **ours vs TE FP8** | 5.566e-1 | 6.200e-1 | 7.072e-1 |

**结论**：我们的 FP8 实现与 fp32 ref、与 TE FP8 的偏差都在 **同一量级（~0.3–0.9）**，
没有系统性误差；且「ours vs TE」与「ours/TE vs ref」同量级，说明实现口径与 TE 一致。
误差主要来自 FP8 量化本身（尾数 2–3 位），与我们 golden 的 fp32 累加无关。

> 口径差：我们的 O 取 `ref_o`（fp32 精确前向），TE 用其 FP8 前向输出作 O；
> 且我们输出 fp32、TE 输出 bf16。因此「ours vs TE」不可能为 0，~0.5 属于正常差异。

---

## 3. ncu 剖析（main kernel `fa_bwd_fp8_kernel`，S=512 causal，`--set full`）

```
Duration                     ms     6.01
DRAM Throughput              %      0.06      <- HBM 几乎空闲
L2 Cache Throughput          %      0.26
L1/TEX Cache Throughput      %     75.96      <- 当前最高单元
Compute (SM) Throughput      %      4.98
Memory Throughput            %     41.63
Executed Ipc Active                 0.36
Issue Slots Busy             %      4.98
Registers Per Thread         reg    40
Dynamic Shared Memory        KB     68.35
Theoretical Occupancy        %     18.75      <- 受 shared memory 限制
Achieved Occupancy           %      6.25
Waves Per SM                        0.32
Warp Cycles Per Issued Inst  cyc   11.02
  其中 69.2% 的 stall = MIO scoreboard（等 shared memory 依赖）
excessive shared wavefronts  90%（bank conflict）
```

### bound 结论
1. **不是 HBM bound**：DRAM 0.06%、L2 0.26%——数据都在 smem/寄存器，规模也小。
2. **不是算力 bound**：Compute 4.98%、IPC 0.36（标量 FFMA 本来也远低于峰值）。
3. **L1/TEX 76%，且 90% shared wavefronts 是多余的（bank conflict）**；warp 67% 时间
   卡在 **MIO scoreboard**（等 smem）。FP8 每读一个元素都要 `cvt`/`halfraw` 解码 + 乘 scale，
   同一条 smem 行上 lane 连续读 `uchar`，冲突比 fp16 更重。
4. **occupancy 被 68KB smem 卡在 1 CTA/SM（理论 18.75%，实测 6.25%）**，且 `Waves=0.32`
   （grid=128 < 132 SM）。fp8 主数据本可更省 smem，但 scale + fp32 `Ss/dQs` 仍占用较多。
5. 综合：**bound = shared memory 访问（bank conflict + 解码开销）+ 低 occupancy/并行度**，
   和 fp16 版同源但更严重。优化优先级：**消 bank conflict（padding / 向量化 fp8 读）→
   降 smem 提 occupancy → 上 mma 张量核（把解码从 CUDA core 移到 tensor core）**。

---

## 4. 性能对标（CUPTI 纯 device 时间）

FLOPs 口径 `4·B·S²·H·D`（causal 未折算，实际约一半）。H100 FP8 TC dense 峰值 **1978.8 TFLOPS**。
TE 由 `fa_bwd_bench.py bench --dtype fp8` 测得（`..._refbench.out.txt`）；FA 反向无 FP8。

| shape | ours total | ours main | TE FP8 | ours/峰值 | TE/峰值 |
|---|---|---|---|---|---|
| B1 S512 H16 D128 causal | 7.17 ms / **0.30 TF** | 5.90 ms | 0.0720 ms / 29.81 TF | 0.015% | 1.51% |
| B1 S1024 H32 D128 causal | 37.52 ms / **0.46 TF** | 29.16 ms | 0.1363 ms / 126.02 TF | 0.023% | 6.37% |
| B1 S4096 H16 D128 causal | 265.93 ms / **0.52 TF** | 198.48 ms | 0.4514 ms / 304.45 TF | 0.026% | 15.39% |

分段时间（ours）：

| shape | quant | preprocess | main | convert |
|---|---|---|---|---|
| S=512 | 0.031 ms | 1.20 ms | 5.90 ms | ~0.04 ms |
| S=1024 H32 | 0.093 ms | 8.94 ms | 29.16 ms | ~0.04 ms |
| S=4096 | 0.179 ms | 71.52 ms | 198.48 ms | ~0.05 ms |

**结论**：golden 版是正确性优先的标量实现，只有 TE FP8 的 ~0.3%，相对 FP8 峰值的
~0.02%。瓶颈与 fp16 版相同（smem/occupancy），但因逐元素 FP8 解码更慢。
**性能优化（mma + bank conflict + occupancy）是 P3-4 的任务。**

---

## 5. 复现命令

```bash
cd code/flash-attention/fa-bwd
# dump FP8 case（含 TE FP8 参考输出）：
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python harness/fa_bwd_bench.py dump \
    --dtype fp8 --shape 1 512 16 128 causal
# 编译运行 + 对拍（默认 case b1_s512_h16_d128_causal_fp8）：
scripts/run.sh src/fp8/fa_bwd_fp8_onefile.cu --iters=10
# S=4096：
scripts/run.sh src/fp8/fa_bwd_fp8_onefile.cu --iters=5 \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu（main）：
scripts/ncu.sh src/fp8/fa_bwd_fp8_onefile.cu --set full --launch-count 1 \
    --kernel-name regex:fa_bwd_fp8_kernel -- --iters=1
# TE FP8 基线：
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python harness/fa_bwd_bench.py bench \
    --dtype fp8 --shape 1 4096 16 128 causal
```

---

## 6. 下一步

- **P3-3**：把对拍脚本化（ours vs ref vs TE 汇总表），并补 S=1024/H32 等 shape。
  （本轮已在 kernel 内输出三项对拍，P3-3 可做正式 harness/报告。）
- **P3-4**：`mma.m16n8k32`（E5M2×E4M3）+ `ldmatrix` + smem padding，消 bank conflict、
  提 occupancy、把解码从 CUDA core 移到张量核；ncu 复测 bound。
- **P3-5**：两文件拆分 + 本文档拆分/补全。

---

## 7. P3-4：张量核版（`src/fp8/fa_bwd_fp8_mma_onefile.cu`）

> 目标：把 golden 的 5 个矩阵乘换成 `mma.sync.aligned.m16n8k32`（FP8），保持 rowwise 口径。
> 原始输出：`src/fp8/fa_bwd_fp8_mma_onefile_{s512,s1024h32,s4096,ncu_main}.out.txt`；
> 布局最小复现：`src/fp8/fa_bwd_fp8_mma_smoke.cu` + `..._smoke.out.txt`。

### 7.1 先做「最小 GEMM 复现」验证 mma 布局（P3-4a）

在合入反向之前，先用 `fa_bwd_fp8_mma_smoke.cu` 把 `m16n8k32` 的片段布局/`ldmatrix`/
rowwise scale 折回逐位验证，覆盖 **E4M3×E4M3、E5M2×E4M3、E4M3×E5M2、E5M2×E5M2** 四种组合
（反向 dV 恰是 E4M3×E5M2）：

```
[E4M3xE4M3] M=64 N=32 K=128  mma-vs-ref max_abs=4.77e-07
[E5M2xE4M3] M=64 N=32 K=128  mma-vs-ref max_abs=7.15e-07
[E4M3xE5M2] M=64 N=32 K=128  mma-vs-ref max_abs=4.77e-07   <- dV 用
[E5M2xE5M2] M=64 N=32 K=128  mma-vs-ref max_abs=1.19e-07
[E4M3xE5M2] M=128 N=64 K=128 mma-vs-ref max_abs=9.54e-07
```

误差 ~1e-6 即 fp32 舍入，说明 A/B 片段（`ldmatrix.x4`/`x2`）、累加器行列映射
（`c0/c1: lane>>2`、`c2/c3: +8`；列 `(lane&3)*2+(q&1)`）与 epilogue `sa[r]*sb[c]`
全部正确。布局公式沿用 `code/kernel-opt/22-fp8-gemm`（见 `agent_skills/kernel-opt.md`）。

### 7.2 反向里的量化/折算记账

5 个 GEMM 全部走 mma，累加器 fp32。**难点是 rowwise scale 的折算**：当某个操作数的
scale 恰好沿**归约维**变化时，不能简单在 epilogue 乘常数，必须把该 scale 折进**另一个**
操作数并重新量化。最终方案（`Ap/dS2/dS3` 三个折叠操作数）：

| GEMM | A（fp8） | B（原始 fp8） | epilogue 折算 | dtype |
|---|---|---|---|---|
| `S=scale·QKᵀ` | Q | K | `scale·qs[m]·ks[j]` | E4M3×E4M3 |
| `dP=dO·Vᵀ` | dO | V | `dos[m]·vs[j]` | E5M2×E4M3 |
| `dV=PᵀdO` | `Ap[j][m]=P[m][j]·dos[m]` | `do8`（raw） | `sA[j]` | E4M3×E5M2 |
| `dQ=scale·dS·K` | `dS2[m][j]=dS[m][j]·ks[j]` | `k8`（raw） | `scale·sds2[m]` | E5M2×E4M3 |
| `dK=scale·dSᵀ·Q` | `dS3[j][m]=dS[m][j]·qs[m]` | `q8`（raw） | `scale·sds3[j]` | E5M2×E4M3 |

以 `dV` 为例：真实值 `Σ_m P·dO = Σ_m (P·dos)·do8`，把 `dos[m]`（沿归约维 m 变化）折进
`Ap=P·dos` 并按 j 行 rowwise 量化，B 改用**未乘 scale 的原始 `do8`**，于是折算因子退化为
每输出行一个 `sA[j]`。`dQ/dK` 同理（分别折 `ks[j]`/`qs[m]`）。**与 golden 的差异**：
`dP` 不再量化（它只进 `dS=P∘(dP−D)` 的 fp32 计算，不进张量核），更接近「只量化真正进
张量核的操作数」的口径；`dS` 也由 fp32 经上述折叠后再量化。

### 7.3 实现要点

- `BM=64, BN=32, THREADS=128`（4 warp），`D=128`。5 个 mma 的 warp 平铺分别为
  `S/dP: 2×2→32×16`、`dQ: 2×2→32×64`、`dK/dV: 2×2→16×64`。
- smem 行距全部取 **16B 的整数倍 + padding**（`144/48/80/48`）以配合 `ldmatrix` 并消
  bank conflict；为 `dK/dV/dQ` 的 B 操作数额外存了转置副本 `Qt/dOt/Kt`（`[d][m]`/`[d][j]`）。
- **踩坑（已修）**：scale 数组个数数错导致 `sds2` 越过 `Ps`（`kNScale` 应为 `3*BM+4*BN`，
  不是 `2*BM+…`）。症状是 S=512 正确、S≥1024 时 dV 出现 ~O(1) 误差——因为越界写入的
  `sds2` 位置随 tile 数变化。定位靠「在 dV 里用同一份输入重算 P 并比对 `Ps`」的计数器。
- `dQ/dK/dV` 仍用 fp32 全局 `atomicAdd` 累加（与 golden 口径一致，便于对拍）。

### 7.4 数值对拍（max_abs，causal）

| shape | | dq | dk | dv |
|---|---|---|---|---|
| S=512 | ours vs ref | 2.43e-1 | 2.98e-1 | 3.74e-1 |
| | ours vs TE | 5.38e-1 | 4.43e-1 | 8.55e-1 |
| S=1024 H32 | ours vs ref | 2.40e-1 | 4.20e-1 | 3.54e-1 |
| | ours vs TE | 4.65e-1 | 5.70e-1 | 9.43e-1 |
| S=4096 | ours vs ref | 2.64e-1 | 2.64e-1 | 3.22e-1 |
| | ours vs TE | 4.53e-1 | 5.32e-1 | 6.81e-1 |

对比 golden（S=512：0.48/0.64/0.33；S=4096：0.46/0.50/0.38），mma 版**数值相当或更好**
（尤其 dq/dk），与 TE 同量级、无系统误差。S=4096 时 ours-vs-ref 甚至优于 TE-vs-ref
（TE：dq 0.376 / dk 0.369 / dv 0.669）。

### 7.5 ncu（main `fa_bwd_fp8_mma_kernel`，S=512，`--set full`）

```
Duration                  us    553.06
DRAM Throughput           %     0.92
L2 Cache Throughput       %    12.94
L1/TEX Cache Throughput   %    19.15     <- 从 golden 的 75.96% 大幅下降
Compute (SM) Throughput   %     4.76
Issue Slots Busy          %     4.76
No Eligible               %    91.66
Registers Per Thread      reg   128
Dynamic Shared Memory     KB    80.13     <- 1 CTA/SM
Achieved Occupancy        %     6.25
Waves Per SM                   0.48
top stall = scoreboard 依赖（~31.9%）
```

**bound 变化**：golden 的「smem bank conflict + FP8 逐元素解码」（L1/TEX 76%、90% 多余
wavefront、MIO scoreboard 69%）已经消失；现在是 **低 occupancy / 并行度不足**
（1 CTA/SM、4 warp/SM、`No Eligible` 91.7%、`Waves 0.48`）导致的延迟受限。下一步：
降寄存器/smem 提 occupancy、pipeline（`cp.async`/双缓冲）、把 dQ/dK/dV 的 atomicAdd 换成
`dQ_accum` 缓冲。

### 7.6 性能对标（CUPTI/event，FP8 峰值 1978.8 TFLOPS）

| shape | golden main | mma main | **main 加速** | mma total | TE FP8 | mma main/峰值 |
|---|---|---|---|---|---|---|
| S=512 H16 | 5.898 ms | **0.452 ms** | **13.1×** | 1.697 ms / 1.27 TF | 0.072 ms / 29.8 TF | 0.24% |
| S=1024 H32 | 29.165 ms | **1.802 ms** | **16.2×** | 10.78 ms / 1.59 TF | 0.136 ms / 126 TF | 0.48% |
| S=4096 H16 | 198.484 ms | **10.188 ms** | **19.5×** | 80.55 ms / 1.71 TF | 0.451 ms / 304 TF | 0.68% |

- main kernel 相对 golden 提速 **13–20×**；相对 TE FP8 的 main 仍只有 ~4–16%（因低 occupancy、
  无流水、每 tile 原子累加）。
- **端到端瓶颈已转移到 `preprocess`**（LSE 的 O(S²) 点积未分块）：S=4096 时 70.6 ms
  vs main 10.2 ms。下一步优先把 preprocess 分块/向量化（或并入前向），再谈 main 的提升。

### 7.7 复现

```bash
cd code/flash-attention/fa-bwd
# 布局最小复现（4 种 dtype 组合）
scripts/run.sh src/fp8/fa_bwd_fp8_mma_smoke.cu
# mma 反向：编译运行 + 对拍（默认 S=512）
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=10
# S=4096
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=3 \
    /home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --set full --launch-count 1 \
    --kernel-name regex:fa_bwd_fp8_mma -- /home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8 --iters=1
```

---

## 8. P3-5：两文件版（`src/fp8/fa_bwd_fp8_kernels.cuh` + `fa_bwd_fp8_main.cu`）

P3-4 的张量核版交付后，按 fp16/bf16 的同款约定把 FP8 反向也拆成两文件：

- `src/fp8/fa_bwd_fp8_kernels.cuh`（device 部分）：编译期常量 + smem 布局 + fp8 转换 helper
  + `mma`/`ldmatrix` 封装 + `quantize_row_kernel` + `preprocess_kernel` +
  `fa_bwd_fp8_mma_kernel` + `convert_kernel`，即单文件里除 host/自测外的**全部 device 代码**。
- `src/fp8/fa_bwd_fp8_main.cu`（host 部分）：`#include "fa_bwd_fp8_kernels.cuh"`，只保留
  npy 读取 / launcher / 计时 / 对拍自测。

拆分标准与 P1-4/P2-2 一致：**device 代码逐字未改**（只是移动到 `.cuh` 并用 include guard），
因此期望逐指标完全一致。选择以 **mma 单文件**（`fa_bwd_fp8_mma_onefile.cu`）为源，
而非 golden，因为 mma 版是 fp8 的最终形态，两文件版应对齐它。

### 8.1 数值核对（与 mma 单文件逐指标一致）

| shape | 指标 | mma 单文件 (P3-4) | 两文件版 (P3-5) |
|---|---|---|---|
| S=512 H16 | dq/dk/dv vs ref max_abs | 2.426 / 2.975 / 3.735e-1 | 2.426 / 2.975 / 3.735e-1 |
| S=512 H16 | main | 0.452 ms | 0.454 ms |
| S=1024 H32 | dq/dk/dv vs ref max_abs | 2.400 / 4.195 / 3.536e-1 | 2.400 / 4.195 / 3.536e-1 |
| S=1024 H32 | main | 1.802 ms | 1.814 ms |
| S=4096 H16 | dq/dk/dv vs ref max_abs | 2.635 / 2.643 / 3.216e-1 | 2.635 / 2.643 / 3.216e-1 |
| S=4096 H16 | main | 10.188 ms | 10.164 ms |

对拍误差**逐位相同**；main 时间在 event 计时噪声内一致。ncu 的 SASS 级统计也逐项相同
（`Executed Instructions = 25,543,552`、`Registers = 128`、`Dynamic Shared Memory = 80.13 KB`、
`DRAM 0.93% / L1TEX 19.0% / Compute 4.75% / occupancy 6.25% / Waves 0.48`），确认拆分无行为差异。
原始输出见 `src/fp8/fa_bwd_fp8_main_{s512,s1024h32,s4096,ncu_main}.out.txt`。

### 8.2 性能对标（未变）

两文件版与单文件共用同一 kernel SASS，性能口径同 §7.6：main 相对 FP8 峰值 0.24–0.68%、
相对 TE FP8 main 约 4–16%，端到端瓶颈仍是 `preprocess`（S=4096 时 70.8 ms >> main 10.2 ms）。
优化 backlog（pipeline / 提 occupancy / dQ 缓冲 / preprocess 分块）见 `../ROADMAP.md`。

### 8.3 复现

```bash
cd code/flash-attention/fa-bwd
# 两文件版：编译运行 + 对拍（默认 S=512）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=10
# S=1024 H32 / S=4096
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=10 \
    /home/xieminglin/proj/output/fa-bwd/b1_s1024_h32_d128_causal_fp8
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=3 \
    /home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --launch-count 1 \
    --kernel-name regex:fa_bwd_fp8_mma -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8 --iters=1
```

## 9. O1：preprocess 的 mma 分块 LSE（`lse_mma_kernel` + `delta_kernel`）

P4 之后端到端第一瓶颈是 `preprocess`：S=4096 时 **70.8 ms >> main 10.2 ms**。旧 `preprocess_kernel`
每个 `(s,h)` 行一个 block、128 线程，逐 `j` 标量扫 K，每对 `(i,j)` 反量化 `2×128` 个 e4m3 再 FFMA，
且 `grid=(S,H,B)=65536` 个「1 行」block 并行度看着高、实际每 block 只有 1 个 warp 在 N 上有活干、
K 行反复从 global 读。O1 把它换成与反向主 kernel 同源的 **tensor-core QK**，并把 D 的计算拆出去。

### 9.1 实现

- **`lse_mma_kernel`**（`grid=(S/64, H, B)`，128 线程=4 warp）：
  - 每 CTA 处理 `LBM=64` 行 Q，4 个 warp 各 16 行（`wm=wid`），沿 N 方向一次吃 `LBN=64` 列；
    Q/K 分块进 smem（`ASLD=144`，`mma_block<16,64,kHeadDim,E4E4>`）。
  - **P 不物化**：`mma.m16n8k32` E4M3×E4M3 的 fp32 累加器直接做 online-softmax（running max/l）。
    同一行的 4 个 lane（`lane&3`）在 warp 内 `shfl_xor`（±1、±2）归约，`c2==0` 的 lane 写 `lse`。
  - 数学与旧版逐项一致：`scale·qs[r]·ks[c]`、causal mask、`sv==-INF` 跳过保证 NaN 安全。
- **`delta_kernel`**（`grid=(S,H,B)`，128 线程）：`D = rowsum(dO∘O)` 原本和 LSE 挤在一个 kernel，
  现在独立——它只 O(S·H·D)，与 LSE 解耦后 LSE 可以纯粹地按 tile 并行。
- 两文件版（`fa_bwd_fp8_kernels.cuh` + `fa_bwd_fp8_main.cu`）与 mma 单文件（`fa_bwd_fp8_mma_onefile.cu`）
  同步修改，device 代码逐字一致。

### 9.2 数值核对（与 P3-5 逐位相同）

| shape | dq/dk/dv vs ref max_abs（P3-5 → O1） | vs TE max_abs（O1） |
|---|---|---|
| S=512 H16 | 2.426 / 2.975 / 3.735e-1 | 5.381e-1 / 4.429e-1 / 8.546e-1 |
| S=1024 H32 | 2.400 / 4.195 / 3.536e-1 | 4.652e-1 / 5.701e-1 / 9.431e-1 |
| S=4096 H16 | 2.635 / 2.643 / 3.216e-1 | 4.525e-1 / 5.324e-1 / 6.807e-1 |

新 LSE 与旧标量 LSE 算出的 `lse` 在 fp32 上一致到末位（下游 dq/dk/dv 的误差**逐位相同**），
确认只换了算法数据流、没改数学口径。

### 9.3 性能（event 计时，ms；FP8 峰值 = 1978.8 TFLOPS）

| shape | preprocess 旧 | preprocess 新 | 提速 | total 旧 | total 新 | total 提速 | total TFLOPS |
|---|---|---|---|---|---|---|---|
| S=512 H16 | 1.1991 | 0.0850 | **14.1×** | 1.7103 | 0.6049 | 2.83× | 3.55 |
| S=1024 H32 | 8.8033 | 0.2162 | **40.7×** | 10.7192 | 2.2138 | 4.84× | 7.76 |
| S=4096 H16 | 70.7674 | 1.1370 | **62.2×** | 80.8639 | 11.4702 | 7.05× | 11.98 |

`main` 未改（0.451 / 1.812 / 9.924 ms，在 event 噪声内）。同 session 重测 TE FP8 基线
（CUPTI）为 0.0723 / 0.1381 / 0.4537 ms，故端到端 ours/TE = 8.4× / 16.0× / 25.3×
（P3-5 时是 23.5× / 77.7× / 178×）；total 相对 FP8 峰值占比 0.18% / 0.39% / 0.61%。
S=4096 时端到端瓶颈已从 preprocess **回落到 main（9.92 ms，占 87%）**。

### 9.4 ncu 剖析

`lse_mma_kernel`（S=4096，`grid=(64,16,1)`，128 线程，17 passes）：

| 指标 | 值 | 指标 | 值 |
|---|---|---|---|
| Duration | 1.16 ms | DRAM Throughput | **0.46%** |
| Compute (SM) | 39.36% | L2 Throughput | 5.35% |
| L1/TEX | 20.63% | Registers | 72 |
| Achieved Occupancy | **29.48%**（理论 43.75%） | Waves Per SM | 1.11 |

- **bound = 延迟/并行度 + 尾波**，不是带宽、也不是 smem：
  `No Eligible 42.09%`（`long_scoreboard` 主导的访存等待）、`Waves=1.11`（1 个满波 + 100 个 block 的
  尾波，ncu 估计尾波最多占 50% 时间）。理论 occupancy 被 **72 寄存器**卡在 43.75%
  （`Block Limit Registers=7`，128 线程；降到 ≤64 可到 8 block/32 warp=50%）。
- S=512 时 `grid=128 < 132 SM`，`Waves=0.14`，是纯粹的「grid 太小」——但即便如此仍比旧的标量版快 14×。
- `delta_kernel`（S=4096）：43.6 µs，DRAM 30.65% / L1TEX 75.07% / Compute 75.36%，是 O(S·H·D) 的
  逐行归约，量级可忽略（含在 preprocess 里）。

### 9.5 复现

```bash
cd code/flash-attention/fa-bwd
# 两文件版（O1）：对拍 + 计时（S=512 / S=1024H32 / S=4096）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=10
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=10 \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h32_d128_causal_fp8
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=10 \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# 单文件版（与两文件逐位一致）
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=10
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --section SpeedOfLight --section Occupancy \
    --section SchedulerStats --section WarpStateStats --section LaunchStats \
    --kernel-name regex:lse_mma_kernel --launch-count 1 -- --iters=1 \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
```

## 10. O2：降 smem 提 occupancy（3 CTA/SM，main 1.16–1.19×）

O1 后台端到端瓶颈回落到 `main`（S=4096 占 87%）。O2 的目标是把 main 的 occupancy
从 2 CTA/SM 抬到 ≥2（实际做到 **3 CTA/SM**），手段是**降动态 smem**。

### 10.1 实现：把两个 per-tile 小缓冲折叠进「本 tile 内已死亡」的 Ks/Vs

| buffer | 原大小 | 原用途 | 只用在哪一步 | 处置 |
|---|---|---|---|---|
| `dS3` [BN][QTS] | 2560 B | dK 的 A（E5M2） | GEMM4 | 放进 `Ks`（[BN][ASLD]=4608 B）尾部 |
| `Ap`  [BN][QTS] | 2560 B | dV 的 A（E4M3） | GEMM3 | 放进 `Vs`（[BN][ASLD]=4608 B）尾部 |

依据：

- `Ks` 只在 **GEMM1（S=QKᵀ）** 被读，GEMM1 后有一个 `__syncthreads()`；
  `dS3` 是在其后的 fold 阶段才写入的，故不冲突。
- `Vs` 只在 **GEMM2（dP=dO·Vᵀ）** 被读，GEMM2 后同样有 sync；`Ap` 在 fold 阶段才写。
- `P`/`dS` 两块 fp32 不能合并（fold 阶段 `Ap` 读 P、`dS2/dS3` 读 dS，同时活跃），保留各一块。

smem 从 `80128 B (78.25 KB)` 降到 **`75008 B (73.25 KB)`**（kFp8Bytes 57344 + scales/P/S 17664）。
驱动据此自动选 `Shared Memory Configuration Size = 233.47 KB`（原先 167.94 KB）：

```
3 × (75008 + 1044 driver) = 228156 B ≤ 233472 B   ⇒  Block Limit Shared Mem = 3
```

**单文件 `fa_bwd_fp8_mma_onefile.cu` 与两文件 `fa_bwd_fp8_kernels.cuh` 同步修改**，device
代码逐字一致（改的是 smem 指针与 `kFp8Bytes/kSmemBytes` 常量）。

### 10.2 数值核对（与 P3-5/O1 **逐位相同**）

| shape | dq / dk / dv vs ref (max_abs) | main ms（单文件 / 两文件） |
|---|---|---|
| S=512 H16 | 2.426e-1 / 2.975e-1 / 3.735e-1 | 0.4610 / 0.4651 |
| S=1024 H32 | 2.400e-1 / 4.195e-1 / 3.536e-1 | 1.5255 / 1.5258 |
| S=4096 H16 | 2.635e-1 / 2.643e-1 / 3.216e-1 | 8.5623 / 8.6167 |

与 P3-4/P3-5 基线逐位一致；ncu `Executed Instructions = 25,543,552` 与改前相同
（只挪 smem，未增删指令）。原始输出 `src/fp8/fa_bwd_fp8_{main,mma_onefile}_o2_*.out.txt`。

### 10.3 性能（event 计时，ms）

| shape | main O1 | main O2 | 提速 | total O1 | total O2 | total TFLOPS |
|---|---|---|---|---|---|---|
| S=512 H16 | 0.4506 | 0.4610 | 1.00×（grid-bound） | 0.6049 | 0.6209 | 3.46 |
| S=1024 H32 | 1.8137 | **1.5255** | **1.19×** | 2.2138 | **1.9214** | 8.94 |
| S=4096 H16 | 9.9241 | **8.5623** | **1.16×** | 11.4702 | **10.067** | 13.65 |

同 session TE FP8 基线（CUPTI）0.0724 / 0.1371 / 0.4550 ms。main 相对 FP8 峰值
（1978.8 TFLOPS）S=4096 = 16.05 TF / 0.81%；端到端 ours/TE = 8.6× / 14.0× / 22.1×。

### 10.4 ncu（main `fa_bwd_fp8_mma_kernel`，S=4096）

| 指标 | O1 | **O2** | 说明 |
|---|---|---|---|
| Duration | 10.27 ms | **8.85 ms** | 1.16× |
| Dynamic smem/block | 80.13 KB | **75.01 KB** | -5.1 KB |
| smem config | 167.94 KB | **233.47 KB** | 自动选最大 carveout |
| Block Limit Shared Mem | 2 | **3** | 不再被 smem 卡在 2 |
| Theoretical Occupancy | 12.50% | **18.75%** | 3 CTA/SM |
| Achieved Occupancy | 11.80% | **16.85%** | |
| Active Warps / Scheduler | 1.91 | **2.69** | +41% |
| No Eligible | 82.83% | **79.64%** | |
| Waves Per SM | 3.88 | 2.59 | 尾波占比上升（233/396 partial） |
| DRAM / L1TEX / L2 / Compute | 0.72 / 43.29 / 39.58 / 13.81% | 1.25 / 51.99 / 46.49 / 15.95% | 各级利用率随 issue 提升 |

**bound 结论**：O2 把墙从「smem 容量锁 2 CTA/SM」推到「occupancy=3 下仍 latency-bound」
（`No Eligible 79.6%`，`long_scoreboard`/`barrier` 主导）。下一堵墙是：(a) 小 S 的
**grid 太小**（S=512 `grid=128 < 132 SM`，achieved 仍 6.25%）；(b) S=4096 的 **尾波**
（3 CTA 下 waves 2.59，partial wave 233 块）；(c) 想上 4 CTA/SM 需再砍 ~17 KB
（等价于消除 `Kt/Qt/dOt` 三个转置副本 → 需要 fp8 的 `ldmatrix.trans`，留 backlog）。

### 10.5 复现

```bash
cd code/flash-attention/fa-bwd
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=20 \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=20 \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma --launch-count 1 \
    --section LaunchStats --section Occupancy --section SpeedOfLight --section SchedulerStats -- \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=1
```

---

## 11. O3：K/V 向量化 + 寄存器预取流水（main 1.15–1.18×）

O2 后 main 仍是 latency-bound（`No Eligible 79.6%`，`long_scoreboard`/`barrier` 主导），
而 per-tile 的 K/V 全局读是「载入 → `__syncthreads()` → 算」，全局/L2 延迟完全暴露。
O3 把这段改成**向量化 4B 读 + 寄存器双缓冲预取**。

### 11.1 实现：寄存器双缓冲（不加 smem，保持 3 CTA/SM）

关键约束：**不能加 smem**。O2 后 smem=75008 B，3 CTA/SM 已用满 233.47 KB carveout，
再加 ~9 KB（K/V 双缓冲）就会掉回 2 CTA/SM。因此用**寄存器**而不是 smem 做双缓冲：

- 新增 `kv_prefetch` / `kv_commit`（`fa_bwd_fp8_kernels.cuh`，单文件同步）：
  每线程按 `(tid + e*THREADS)*4` 预取 `KVU = BN*kHeadDim/4/THREADS = 8` 个 `uint32`
  （= 4 个连续 fp8，行内 4B 对齐，故一次 4B 读）。相比原来的逐字节标量读（每线程 32 次），
  **读指令数降到 1/4**，且省掉每元素的 `cvt`/地址计算。落盘时同时写 Kt 转置副本（供 GEMM5 的 B）。
- 预取流水：`kv_prefetch(nt+1)` 发在本轮 5 个 GEMM **之前**，`kv_commit` 在本轮所有 GEMM
  读完 smem 之后（末尾 `__syncthreads()` 后），于是全局延迟被**一整轮计算**覆盖。
  barrier 数与原版相同（每 tile 5 个）。
- prologue 用同款 `kv_prefetch(0)+kv_commit` 载入 tile 0。
- 寄存器 128 → **168**（16 个给 `pk/pv`），无 spill；`65536/(3×128)=170` ⇒ 仍 3 CTA/SM。

### 11.2 数值核对（与 P3-5/O1/O2 **逐位相同**）

| shape | dq / dk / dv vs ref (max_abs) | 单文件 main | 两文件 main |
|---|---|---|---|
| S=512 H16 | 2.426e-1 / 2.975e-1 / 3.735e-1 | 0.3903 | 0.3934 |
| S=1024 H32 | 2.400e-1 / 4.195e-1 / 3.536e-1 | — | 1.3231 |
| S=4096 H16 | 2.635e-1 / 2.643e-1 / 3.216e-1 | 7.5002 | 7.5092 |

三次 shape 的 max_abs 与 P3-5/O1/O2 表逐位一致；只换数据搬运方式，数学口径未变。
单/两文件差异 <1%（run-to-run 噪声）。原始输出 `src/fp8/fa_bwd_fp8_{main,mma_onefile}_o3_*.out.txt`。

### 11.3 收益归因（消融：向量化 vs 流水）

为拆分「向量化」与「预取流水」两个变量，额外编了一个**只向量化、不跨迭代预取**
（每 tile 开头同步载入）的消融版：

| shape | O2 标量同步 | 向量化同步（消融） | O3 向量化+预取 | 向量化贡献 | 流水贡献 |
|---|---|---|---|---|---|
| S=512 H16 main | 0.4655 | 0.4007 | **0.3934** | 1.16× | 1.02× |
| S=4096 H16 main | 8.6697 | 7.6952 | **7.5092** | 1.13× | 1.03× |

**结论：O3 的收益主要来自把逐字节标量 K/V 读改成 4B 向量化读（少 3/4 读指令 + 少地址运算），
寄存器预取流水只再贡献 ~2–3%。** 消融原始输出 `src/fp8/fa_bwd_fp8_main_o3_ablation_*.out.txt`。

### 11.4 性能（event 计时，ms；FP8 峰值 = 1978.8 TFLOPS）

| shape | main O2 | main O3 | 提速 | total O2 | total O3 | total TFLOPS |
|---|---|---|---|---|---|---|
| S=512 H16 | 0.4655 | **0.3934** | **1.18×** | 0.6214 | **0.5402** | 3.98 |
| S=1024 H32 | 1.5255* | **1.3231** | **1.15×** | 1.9214* | **1.6978** | 10.12 |
| S=4096 H16 | 8.6697 | **7.5092** | **1.15×** | 10.086 | **9.060** | 15.17 |

（*S=1024H32 O2 值取自 O2 轮记录。）main-only TFLOPS = 5.46 / 12.98 / 18.30 TF
（峰值占比 0.28% / 0.66% / **0.93%**）。同 session TE FP8 基线（CUPTI）
0.0723 / 0.1377 / 0.4513 ms ⇒ main ours/TE = **18.4% / 10.4% / 6.0%**，端到端 13.4% / 8.1% / 5.0%。

### 11.5 ncu（main，S=4096；stall 为 per-issued-inst 平均周期）

| 指标 | O2 | **O3** | 说明 |
|---|---|---|---|
| Duration | 8.85 ms | **7.57 ms** | 1.16×（与 event 一致） |
| Registers / Thread | 128 | **168** | +16（pk/pv），无 spill |
| Theoretical / Achieved Occ | 18.75% / 16.85% | 18.75% / **16.71%** | 仍 3 CTA/SM |
| Waves Per SM | 2.59 | 2.59 | 不变 |
| DRAM / L1TEX / L2 / Compute | 1.25 / 51.99 / 46.49 / 15.95% | 1.48 / **65.59** / 53.67 / 13.79% | L1TEX 升（同字节下更多在飞请求） |
| **long_scoreboard** stall | 2.94 | **2.21** | **全局延迟停顿下降**（O3 目标达成） |
| short_scoreboard stall | 3.61 | **4.12** | smem → mma 依赖成为新主导 |
| barrier stall | 3.34 | **5.13** | 5 个 `__syncthreads`/tile 成为最大停顿 |
| No Eligible | 79.64% | 81.91% | issue 仍稀 |

S=512（`--set full`）：Duration 406 µs、Theoretical Occ 18.75%、**Achieved 6.25%**
（`grid=128 < 132 SM`，1 CTA/SM，grid-bound）、Waves 0.32、No Eligible 91.3%。

**bound 结论**：O3 成功把 `long_scoreboard`（全局延迟）从主导项压下去（2.94→2.21），
下一堵墙变成 **(a) CTA barrier（5.13，5 个 `__syncthreads`/tile）** 与
**(b) short_scoreboard（4.12，smem→mma 的 `ldmatrix` 延迟）**。后续方向：减少 per-tile
barrier 数（合并 GEMM/折叠阶段）、把 Kt 转置副本换成 `ldmatrix.trans`（既省 smem 又能省一次
smem 往返，O2c/O4 backlog），以及小 S 的 grid/尾波。

### 11.6 复现

```bash
cd code/flash-attention/fa-bwd
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8
# stall 分解
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma_kernel \
  --launch-count 1 \
  --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
```
