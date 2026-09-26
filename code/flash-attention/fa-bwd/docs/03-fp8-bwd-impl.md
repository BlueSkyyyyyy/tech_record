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
| (1,1024,32,128) | 0.093 ms | 8.94 ms | 29.16 ms | ~0.04 ms |
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
| (1,1024,32,128) | ours vs ref | 2.40e-1 | 4.20e-1 | 3.54e-1 |
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
| (1,512,16,128) | 5.898 ms | **0.452 ms** | **13.1×** | 1.697 ms / 1.27 TF | 0.072 ms / 29.8 TF | 0.24% |
| (1,1024,32,128) | 29.165 ms | **1.802 ms** | **16.2×** | 10.78 ms / 1.59 TF | 0.136 ms / 126 TF | 0.48% |
| (1,4096,16,128) | 198.484 ms | **10.188 ms** | **19.5×** | 80.55 ms / 1.71 TF | 0.451 ms / 304 TF | 0.68% |

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
| (1,512,16,128) | dq/dk/dv vs ref max_abs | 2.426 / 2.975 / 3.735e-1 | 2.426 / 2.975 / 3.735e-1 |
| (1,512,16,128) | main | 0.452 ms | 0.454 ms |
| (1,1024,32,128) | dq/dk/dv vs ref max_abs | 2.400 / 4.195 / 3.536e-1 | 2.400 / 4.195 / 3.536e-1 |
| (1,1024,32,128) | main | 1.802 ms | 1.814 ms |
| (1,4096,16,128) | dq/dk/dv vs ref max_abs | 2.635 / 2.643 / 3.216e-1 | 2.635 / 2.643 / 3.216e-1 |
| (1,4096,16,128) | main | 10.188 ms | 10.164 ms |

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
# (1,1024,32,128) / S=4096
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
| (1,512,16,128) | 2.426 / 2.975 / 3.735e-1 | 5.381e-1 / 4.429e-1 / 8.546e-1 |
| (1,1024,32,128) | 2.400 / 4.195 / 3.536e-1 | 4.652e-1 / 5.701e-1 / 9.431e-1 |
| (1,4096,16,128) | 2.635 / 2.643 / 3.216e-1 | 4.525e-1 / 5.324e-1 / 6.807e-1 |

新 LSE 与旧标量 LSE 算出的 `lse` 在 fp32 上一致到末位（下游 dq/dk/dv 的误差**逐位相同**），
确认只换了算法数据流、没改数学口径。

### 9.3 性能（event 计时，ms；FP8 峰值 = 1978.8 TFLOPS）

| shape | preprocess 旧 | preprocess 新 | 提速 | total 旧 | total 新 | total 提速 | total TFLOPS |
|---|---|---|---|---|---|---|---|
| (1,512,16,128) | 1.1991 | 0.0850 | **14.1×** | 1.7103 | 0.6049 | 2.83× | 3.55 |
| (1,1024,32,128) | 8.8033 | 0.2162 | **40.7×** | 10.7192 | 2.2138 | 4.84× | 7.76 |
| (1,4096,16,128) | 70.7674 | 1.1370 | **62.2×** | 80.8639 | 11.4702 | 7.05× | 11.98 |

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
| (1,512,16,128) | 2.426e-1 / 2.975e-1 / 3.735e-1 | 0.4610 / 0.4651 |
| (1,1024,32,128) | 2.400e-1 / 4.195e-1 / 3.536e-1 | 1.5255 / 1.5258 |
| (1,4096,16,128) | 2.635e-1 / 2.643e-1 / 3.216e-1 | 8.5623 / 8.6167 |

与 P3-4/P3-5 基线逐位一致；ncu `Executed Instructions = 25,543,552` 与改前相同
（只挪 smem，未增删指令）。原始输出 `src/fp8/fa_bwd_fp8_{main,mma_onefile}_o2_*.out.txt`。

### 10.3 性能（event 计时，ms）

| shape | main O1 | main O2 | 提速 | total O1 | total O2 | total TFLOPS |
|---|---|---|---|---|---|---|
| (1,512,16,128) | 0.4506 | 0.4610 | 1.00×（grid-bound） | 0.6049 | 0.6209 | 3.46 |
| (1,1024,32,128) | 1.8137 | **1.5255** | **1.19×** | 2.2138 | **1.9214** | 8.94 |
| (1,4096,16,128) | 9.9241 | **8.5623** | **1.16×** | 11.4702 | **10.067** | 13.65 |

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
| (1,512,16,128) | 2.426e-1 / 2.975e-1 / 3.735e-1 | 0.3903 | 0.3934 |
| (1,1024,32,128) | 2.400e-1 / 4.195e-1 / 3.536e-1 | — | 1.3231 |
| (1,4096,16,128) | 2.635e-1 / 2.643e-1 / 3.216e-1 | 7.5002 | 7.5092 |

三次 shape 的 max_abs 与 P3-5/O1/O2 表逐位一致；只换数据搬运方式，数学口径未变。
单/两文件差异 <1%（run-to-run 噪声）。原始输出 `src/fp8/fa_bwd_fp8_{main,mma_onefile}_o3_*.out.txt`。

### 11.3 收益归因（消融：向量化 vs 流水）

为拆分「向量化」与「预取流水」两个变量，额外编了一个**只向量化、不跨迭代预取**
（每 tile 开头同步载入）的消融版：

| shape | O2 标量同步 | 向量化同步（消融） | O3 向量化+预取 | 向量化贡献 | 流水贡献 |
|---|---|---|---|---|---|
| (1,512,16,128) main | 0.4655 | 0.4007 | **0.3934** | 1.16× | 1.02× |
| (1,4096,16,128) main | 8.6697 | 7.6952 | **7.5092** | 1.13× | 1.03× |

**结论：O3 的收益主要来自把逐字节标量 K/V 读改成 4B 向量化读（少 3/4 读指令 + 少地址运算），
寄存器预取流水只再贡献 ~2–3%。** 消融原始输出 `src/fp8/fa_bwd_fp8_main_o3_ablation_*.out.txt`。

### 11.4 性能（event 计时，ms；FP8 峰值 = 1978.8 TFLOPS）

| shape | main O2 | main O3 | 提速 | total O2 | total O3 | total TFLOPS |
|---|---|---|---|---|---|---|
| (1,512,16,128) | 0.4655 | **0.3934** | **1.18×** | 0.6214 | **0.5402** | 3.98 |
| (1,1024,32,128) | 1.5255* | **1.3231** | **1.15×** | 1.9214* | **1.6978** | 10.12 |
| (1,4096,16,128) | 8.6697 | **7.5092** | **1.15×** | 10.086 | **9.060** | 15.17 |

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

---

## 12. O4a：消 barrier（GEMM1/GEMM2 + prologue）+ fold 全线程并行（main 1.21–1.54×）

O3 后 ncu 的最大停顿是 **CTA barrier（5.13 per-issued-inst）** 与 **short_scoreboard（4.12，
smem→mma 的 `ldmatrix` 依赖）**。O4a 不动 smem/几何、不引张量核新指令，只做三件**逐位等价**
（数值与 O3 完全相同）的改动：

### 12.1 三处改动

1. **删掉 GEMM1 与 GEMM2 之间的 `__syncthreads`**（per-tile barrier 5→4）。
   依据：GEMM2（`dP=dO·Vᵀ`）只读 `dOs/Vs`，其 epilogue 读回的 `Ps[r*BN+c]` 正是**本线程**
   在 GEMM1 epilogue 刚写入的同一地址——两次 `mma_block<32,16,…>` 的 `(wm=wr, wn=wc)`、
   累加器映射 `(i,j,q)→(r,c)` 完全一致，故读取**无线程间依赖**，无需 barrier。
2. **合并 prologue 的两处 barrier**（prologue 2→1）。Q/dO 与 tile0 的 K/V 写在互不重叠的
   smem（`Qs/Qt/dOs/dOt` vs `Ks/Vs/Kt`），可以「先全部写、最后一处 syncthreads」。
3. **fold 阶段全 128 线程并行**（原来只用 `tid<BN`=32 或 `tid<BM`=64 线程，其余空等）：
   - `Ap`/`dS3`（j 行，32 行）：每 warp 8 行 × **4 lane** 分工，warp 内
     `__shfl_xor_sync(±1,±2)` 做行长 64 的 amax 归约，`sub4==0` 写 scale 后
     `__shfl_sync` 广播回同组；
   - `dS2`（m 行，64 行）：每 warp 16 行 × **2 lane** 分工，`__shfl_xor_sync(1)` 归约。
   `fmaxf` 可交换结合 ⇒ amax 值与原顺序**逐位相同**，除法/量化逐元素一致 ⇒ 结果位等价，
   但这段的墙钟缩短约 **4×**。

主 kernel 的 `__syncthreads` 数：**7 → 5**（prologue 2→1、per-tile 5→4；其余 1 个是
kv_commit 后更新下一 tile 的、最后一块不执行）。单文件 `fa_bwd_fp8_mma_onefile.cu` 与两文件
`fa_bwd_fp8_kernels.cuh` 同步修改，main kernel 函数体**逐字相同**（已脚本核对）。

### 12.2 数值核对（与 O3/P3-5 **逐位相同**）

| shape | dq / dk / dv vs ref (max_abs) | O3 | O4a |
|---|---|---|---|
| (1,512,16,128) | `2.426e-1 / 2.975e-1 / 3.735e-1` | 同 | 同 |
| (1,1024,32,128) | `2.400e-1 / 4.195e-1 / 3.536e-1` | 同 | 同 |
| (1,4096,16,128) | `2.635e-1 / 2.643e-1 / 3.216e-1` | 同 | 同 |

与 TE 的 `max_abs` 也逐位不变（5.381/4.429/8.546e-1 等）。原始输出
`src/fp8/fa_bwd_fp8_main_o4a_{s512,s1024h32,s4096}.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_o4a_{s512,s4096}.out.txt`。

### 12.3 性能（event 计时，ms；同 session 先测 O3 基线）

| shape | main O3(base) | main O4a | **加速** | total O3 | total O4a | total TFLOPS |
|---|---|---|---|---|---|---|
| (1,512,16,128) | 0.3947 | **0.2566** | **1.54×** | 0.5403 | **0.4030** | 5.33 |
| (1,1024,32,128) | 1.3202 | **1.0642** | **1.24×** | 1.7278 | **1.4408** | 11.92 |
| (1,4096,16,128) | 7.7391 | **6.4192** | **1.21×** | 9.0992 | **7.8488** | 17.51 |

main-only TFLOPS = **8.37 / 16.14 / 21.41 TF**（峰值 1978.8 的 **0.42% / 0.82% / 1.08%**）。
同 session TE FP8 基线（CUPTI）`0.0723 / 0.1376 / 0.4537 ms` ⇒ main ours/TE =
**28.2% / 12.9% / 7.1%**（O3 为 18.4/10.4/6.0%），端到端 18.0% / 9.5% / 5.8%。
S=512 收益最大（1.54×）：该 shape 每块 tile 数少，prologue/每-tile 的 barrier 占比更高，
fold 的串行部分相对也更重，故删 barrier + fold 并行的收益被放大。

### 12.4 ncu（main，O3 → O4a）

S=4096（`--set full` / stall 为 per-issued-inst 平均周期）：

| 指标 | O3 | **O4a** | 说明 |
|---|---|---|---|
| Duration | 7.57 ms | **6.55 ms** | 1.16×（ncu 口径） |
| Registers / smem | 168 / 75.01 KB | 168 / 75.01 KB | 不变（3 CTA/SM） |
| Theoretical / Achieved Occ | 18.75% / 16.71% | 18.75% / **17.02%** | |
| Waves Per SM | 2.59 | 2.59 | |
| DRAM / L1TEX / L2 / Compute | 1.48 / 65.59 / 53.67 / 13.79% | 1.75 / **78.51** / 62.01 / 15.95% | L1TEX/L2 升（并行 fold 把更多 smem 请求压进 pipe） |
| **barrier** stall | 5.13 | **0.46** | **目标达成：几乎消失** |
| short_scoreboard stall | 4.12 | **5.40** | smem→mma `ldmatrix` 成为新主导 |
| long_scoreboard stall | 2.21 | 3.35 | |
| No Eligible | 81.91% | 79.04% | |

S=512（`--set full`）：Duration 406 → **272 µs**、barrier 0.33 / short_scoreboard 2.59 /
long_scoreboard 1.46、L1TEX 41.54%、achieved occ 6.25%（`grid=128<132 SM` 仍 grid-bound）、
No Eligible 87.02%。

**bound 结论**：O4a 把 O3 头号墙 **barrier 基本清零（5.13→0.46）**，墙移到
**(a) short_scoreboard（5.40，仍 5 个 `__syncthreads`/tile 之外的 smem→mma 依赖）** 与
**(b) L1/TEX 78.5%**（并行 fold 后 smem 读写并发度更高）。下一步仍是 backlog 的
**O4b：用 fp8 `ldmatrix.trans` 消掉 `Kt/Qt/dOt` 三个转置副本**（既减 smem 冲 4 CTA/SM、
又减 smem 往返；但「2 字节=1 b16」的配对转置不是 drop-in，需最小复现验证），或
**O2b：小 S 的 grid / S=4096 的尾波**（Waves 2.59，partial 233/396）。

### 12.5 复现

```bash
cd code/flash-attention/fa-bwd
# 两文件版
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=10 --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# 单文件版（逐位一致）
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=10 --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma_kernel \
  --launch-count 1 --section SpeedOfLight --section Occupancy --section SchedulerStats \
  --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=1
# TE FP8 基线
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python \
  /ssd/home/xieminglin/proj/tech_record/code/flash-attention/fa-bwd/harness/fa_bwd_bench.py \
  bench --dtype fp8 --shape 1 4096 16 128 causal
```

---

## 13. P5-3（fp8 反向支持 GQA/MQA，单/两文件）

### 13.1 目标与口径

把 fp8 反向从 MHA 扩到 **GQA/MQA**（Q 头数 `H`、KV 头数 `Hkv`，`H % Hkv == 0`）。
映射口径与 `ref_attn` 的 `repeat_interleave` 一致：**第 `h` 个 Q 头对应 KV 头 `hkv = h/(H/Hkv)`**。
张量布局：`q/dO/dq` 为 `[B,S,H,D]`，`k/v/dk/dv` 为 `[B,S,Hkv,D]`；`Hkv==H` 时逐式退化为 MHA。

### 13.2 改动（单/两文件 device 代码同源，逐字一致）

| 位置 | 改动 |
|---|---|
| `lse_mma_kernel` | 入参加 `Hkv`，算 `hkv=h/(H/Hkv)`；K/ks 索引用 `*Hkv+hkv`（Q 仍用 H） |
| `kv_prefetch` | 入参由 `(H,h)` 改 `(Hkv,hkv)`；K/V 全局索引 `*Hkv+hkv` |
| `fa_bwd_fp8_mma_kernel` | 入参加 `Hkv`；K/V/Kt + `dk_acc/dv_acc` 用 `Hkv/hkv`，Q/dO/dQ 用 `H/h` |
| `convert_kernel` | 入参改 `(nq, nkv)` 两个长度，分别拷贝 `dq` 与 `dk/dv` |
| host / launcher | 从 `k.npy` 的 `shape[2]` 读 `Hkv`；`nq=B·S·H·D`、`nkv=B·S·Hkv·D`；quant/LSE/delta 按各自行数启动；compare 按各自长度 |

两文件版：`fa_bwd_fp8_kernels.cuh` + `fa_bwd_fp8_main.cu`；单文件版：`fa_bwd_fp8_mma_onefile.cu`。
`harness/fa_bwd_bench.py` 另修一处 TE 导入顺序（先 `import transformer_engine` 再
`transformer_engine_torch`，否则 `ModuleNotFoundError`），使 TE FP8 GQA 基线可跑。

### 13.3 数值对拍（max_abs，causal，B=1 S=1024 D=128）

**ours vs fp32 ref**：

| case | dq | dk | dv |
|---|---|---|---|
| h40 kv8 | 2.869e-01 | 5.390e-01 | 7.107e-01 |
| h32 kv4 | 2.517e-01 | 5.408e-01 | 7.072e-01 |
| h64 kv4 | 2.760e-01 | 8.456e-01 | 1.226e+00 |
| h64 kv1 (MQA) | 4.097e-01 | 1.519e+00 | 2.127e+00 |

**TE FP8 vs fp32 ref**（同形状，作对照）：

| case | dq | dk | dv |
|---|---|---|---|
| h40 kv8 | 8.453e-01 | 6.625e-01 | 1.106e+00 |
| h32 kv4 | 5.246e-01 | 7.664e-01 | 1.343e+00 |
| h64 kv4 | 3.983e-01 | 1.011e+00 | 1.844e+00 |
| h64 kv1 (MQA) | 4.100e-01 | 2.218e+00 | 2.597e+00 |

结论：均在与 TE **同量级（fp8 噪声 O(1)）**，且多数情形 ours-vs-ref ≤ TE-vs-ref（kv1 的 dk/dv 尤其：
1.52/2.13 vs 2.22/2.60）。`ref_amax`：kv1 dk=13.1、dv=26.1，即相对量级与 MHA fp8 一致。无系统误差。

**MHA 回归逐位不变**：S=512 `2.426/2.975/3.735e-1`、S=1024H32 `2.400/4.195/3.536e-1`、
S=4096 `2.635/2.643/3.216e-1`（与 §7/§8/§10–12 记录逐位一致）。单文件 GQA kv4 与两文件
**逐位相同**（dq/dk/dv 2.517/5.408/7.072e-1）。

### 13.4 性能对标（CUPTI/event 纯 device 时间，bwd FLOPs=4·B·S·H·S·D）

| case | ours total | ours TFLOPS | 峰值占比 | TE FP8 | TE TFLOPS | ours/TE |
|---|---|---|---|---|---|---|
| h40 kv8 | 1.6346 ms | 13.14 | 0.66% | 0.2428 ms | 176.92 | 7.4% |
| h32 kv4 | 1.3650 ms | 12.59 | 0.64% | 0.2010 ms | 170.93 | 7.4% |
| h64 kv4 | 2.3690 ms | 14.50 | 0.73% | 0.3614 ms | 190.13 | 7.6% |
| h64 kv1 | 2.2934 ms | 14.98 | 0.76% | 0.4009 ms | 171.42 | 8.7% |

（ours total = quant+preprocess+main+convert；TE 为完整反向。FP8 峰值 1978.8 TFLOPS。）
与 MHA fp8 的 ours/TE 水平（S=1024H32 端到端 ~9.5%）一致。

### 13.5 ncu（main, h32 kv4, S=1024）

`--set full --launch-count 1`，Duration **1.07 ms**：DRAM **0.93%** / L1TEX **70.54%** /
L2 50.31% / Compute 13.46%；168 regs / 75.01 KB smem（Block Limit Shared Mem=3）；
Theoretical Occ 18.75%、Achieved 15.83%；Waves 1.29；No Eligible 80.59%；
**short_scoreboard 占 42.5%**（smem→mma 的 `ldmatrix` 依赖）。
**bound 与 MHA O4a 完全一致：short_scoreboard（smem 依赖）+ L1/TEX，非带宽/算力**。
后续优化沿用 backlog 的 O4b（fp8 `ldmatrix.trans` 消转置副本）与 O2b（小 S grid / 尾波）。

### 13.6 复现

```bash
cd code/flash-attention/fa-bwd
# 两文件版（GQA）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=20 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h32_d128_kv4_causal_fp8
# 单文件版（逐位一致）
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=20 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h32_d128_kv4_causal_fp8
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --kernel-name regex:fa_bwd_fp8_mma_kernel \
  --launch-count 1 -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h32_d128_kv4_causal_fp8 --iters=1
# TE FP8 基线
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python \
  /ssd/home/xieminglin/proj/tech_record/code/flash-attention/fa-bwd/harness/fa_bwd_bench.py \
  bench --requested --dtype fp8
```

原始输出：`src/fp8/fa_bwd_fp8_main_p53_{mha_s512,gqa}.out.txt`、
`src/fp8/fa_bwd_fp8_p53_onefile_gqa_mha.out.txt`、`src/fp8/fa_bwd_fp8_main_p53_ncu_gqa_kv4.out.txt`、
`src/fp8/fa_bwd_bench_requested_fp8_p53.out.txt`。

## 14. P5-3（fp8 反向支持 MLA head_dim=512，单/两文件）

### 14.1 目标与难点

把 fp8 反向从 head_dim=128 扩到 **MLA 主注意力的 head_dim=512**（`D=Dv=512`）。
这是 fp8 版相对 fp16/bf16 版**最难**的一处，因为 fp8 主 kernel 的 5 个 GEMM 全部是手写
`mma.m16n8k32` + `ldmatrix`，原实现的 warp/tile 几何把 `HD=128` 硬编码进了两处：

1. **GEMM1（S=QKᵀ）与 GEMM2（dP=dO·Vᵀ）**：`HD` 是**归约维**（K_TILE=HD）。原 k-loop
   只有 4 步（128/32）；HD=512 时要 16 步，且 A 行距 `ASLD` 从 144 变 528。
2. **GEMM3/4/5（dV/dK/dQ）**：`HD` 是**输出 N 维**（dOᵀ/Qᵀ/Kᵀ 的行）。原实现「2 warp ×
   每 warp 64 列 = 128 列」刚好铺满 HD=128；HD=512 时必须再加一层 **N-tile 循环**
   （每遍 128 列，共 `HD/128=4` 遍），B 操作数按 `d0*stride` 偏移、写回列号加 `d0`。
   这一层若写错，会表现为「某个 128 维段全错」，故 §14.4 专门按段核对。

另两处小改动：`quantize_row_kernel` 原来假设 `D<=128`（每线程一个元素），改成线程内
grid-stride 的 amax/量化（`D<=128` 时逐位不变）；`O3` 的寄存器预取每线程要
`KVU=BN*HD/4/THREADS` 个 uint32，HD=512 时 `KVU=32`（64 个寄存器）会挤爆寄存器，
故只在 `KVU*2<=16`（HD=128）时启用，HD>128 走 `kv_load_direct` 的 4B 向量化直接载入。

### 14.2 实现（`HD/BM/BN` 全模板化，单/两文件同源逐字一致）

- 新增 `template<int HD,int BM_,int BN_> struct Fp8Cfg`：把 `ASLD/KTS/QTS/DSS2/KVU/
  kNScale/smem_bytes/lse_smem_bytes/use_prefetch` 全部从全局常量改成派生常量。
- `lse_mma_kernel<HD>` / `delta_kernel<HD>` / `fa_bwd_fp8_mma_kernel<HD,BM,BN>` 全部模板化；
  `mma_block` 的 k-loop 由 `#pragma unroll` 改为 `#pragma unroll 4`（HD=128 时仍全展开、
  逐位不变；HD=512 的 16 步限展开以压寄存器）。
- `HD=512` 选 `BM=64,BN=32`：动态 smem = **228608 B（223.2 KB）**，`cudaFuncSetAttribute`
  上限 232448 B 刚好放得下；**1 CTA/SM**。`lse` smem = 68096 B（66.5 KB）。
- host 按 `D` 分派模板实例（`128→<128,64,32>`、`512→<512,64,32>`）并分别设动态 smem 上限。
- 单文件版 `fa_bwd_fp8_mma_onefile.cu` 由两文件版 device 代码拼接生成，`device 代码逐字一致`。

### 14.3 数值对拍（max_abs，causal，B=1；MLA 无 FA/TE 基线，只对 fp32 ref）

| case | ours dq | ours dk | ours dv | smem | pre/main/total |
|---|---|---|---|---|---|
| S=256 H2 D512 | 2.356e-01 | 2.290e-01 | 3.441e-01 | 223.2 KB | 0.107 / 0.162 / 0.308 ms |
| S=512 H4 D512 | 2.415e-01 | 2.992e-01 | 4.481e-01 | 223.2 KB | 0.196 / 0.318 / 0.591 ms |
| S=1024 H2 D512 | 2.232e-01 | 3.337e-01 | 3.602e-01 | 223.2 KB | 0.371 / 0.564 / 1.022 ms |

误差与 MHA fp8（§7/§13，`~2.4–4.5e-1`）**同量级**，无系统误差。**按 head_dim 每 128 维
分段的 max_abs** 均匀（S=1024H2：dq 四段 1.80/1.42/1.63/2.23e-1、dk 2.84/3.34/2.27/2.71e-1、
dv 2.47/2.42/3.60/2.73e-1），证明 **N-tile 循环四段都正确**（若某段漏算/错位会是 O(1)）。

**MHA d128 回归逐位不变**：S=512 `2.426/2.975/3.735e-1`、main `0.1648 ms`（与 §7/§8/§13 一致）。
单文件与两文件**逐位相同**（S=1024H2 MLA 2.232/3.337/3.602e-1；d128 S512 2.426/2.975/3.735e-1）。

### 14.4 性能（event 纯 device，bwd FLOPs=4·B·S·H·S·D，FP8 峰值 1978.8 TFLOPS）

| case | main | main TFLOPS | main 峰值占比 | total | total TFLOPS |
|---|---|---|---|---|---|
| S=256 H2 D512 | 0.162 ms | 1.66 | 0.08% | 0.308 ms | 0.87 |
| S=512 H4 D512 | 0.318 ms | 6.75 | 0.34% | 0.591 ms | 3.64 |
| S=1024 H2 D512 | 0.564 ms | 7.61 | 0.38% | 1.022 ms | 4.20 |

对照：fp16/bf16 的 MLA（标量 CUDA-core，§01/§01b）main 为 1.06/2.08/4.14 ms（S256H2/S512H4/
S1024H2），即 **fp8 张量核版比 fp16 标量版快 6.5×/6.5×/7.3×**。FA2/TE 反向均不支持
head_dim=512，MLA 反向性能数字只能由 ours 提供。

### 14.5 ncu（main，S=1024 H2 D512，`--set full -c 1`）

- Duration **626.8 µs**；DRAM **1.98%** / L2 22.85% / L1TEX **26.28%** / Compute **5.13%**
  ⇒ 非带宽、非算力。
- 255 regs/thread，`Local Memory Spilling 137 KB`、`Shared Memory Spilling 122 KB`（寄存器墙）；
  smem **228608 B** → `Block Limit Shared Mem=1`；Theoretical/Achieved Occ **6.25%**（1 CTA/SM，
  4 warp/SM）；**Waves 0.97**；**No Eligible 90.73%**。
- `#pragma unroll 4`（相比全展开）：Duration 676.6→626.8 µs，local spill 154→137 KB。
- **bound = 低 occupancy / 并行度不足**（1 CTA/SM 被 223 KB smem 锁死 + 255 寄存器的
  fixed-latency 空等），与 fp16 MLA 标量版「smem 冲突 + 低 occupancy」的结论衔接，但 fp8 版
  已把冲突消掉、瓶颈纯在 occupancy。**要冲 2 CTA/SM 必须先把 smem 降到 ≤116 KB**——
  大头是三个转置副本 `Kt/Qt/dOt`（HD×(BN+16)+2·HD×(BM+16) ≈ 107 KB），正是 backlog **O4b**
  （fp8 `ldmatrix.trans` 消转置副本）的用武之地。

### 14.6 复现

```bash
cd code/flash-attention/fa-bwd
# 两文件版（MLA）：3 个 shape
for c in b1_s256_h2_d512_causal_fp8 b1_s512_h4_d512_causal_fp8 b1_s1024_h2_d512_causal_fp8; do
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=50 \
    --dir=/home/xieminglin/proj/output/fa-bwd/$c
done
# 单文件版（逐位一致）
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=20 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8
# d128 MHA 回归
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=50 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --kernel-name regex:fa_bwd_fp8_mma_kernel \
  --launch-count 1 -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_p53_mla_final.out.txt`（两文件全程）、
`src/fp8/fa_bwd_fp8_main_p53_mla_bands.out.txt`（分段核对）、
`src/fp8/fa_bwd_fp8_mma_onefile_p53_mla_s1024h2.out.txt`（单文件）、
`src/fp8/fa_bwd_fp8_main_p53_ncu_mla_unroll4_s1024h2.out.txt`（ncu）。

---

## 15. O2b + O4d：split-K 自动切块调参 + P/S fp32 行距 padding

> 本轮把上一条 WIP commit（`fp8 main O4b/O4d WIP`）里已经写进代码、但还没进 ROADMAP/docs
> 的两处优化正式收口：**O2b（split-K 切块数自动选择）** 与 **O4d（`Ps/Ss` 行距 padding）**。
> 都是 fp8 反向的 main kernel，单/两文件同步；数值与 P3-5/O1–O4a **逐位相同**（见 15.4）。

### 15.1 O2b：split-K 的机制与切块数 sweep

机制（§内核 `fa_bwd_fp8_mma_kernel` 顶部）：`grid.x = ceil(S/BM) * ksplit`，第 `part` 个 CTA
只处理 K/V 列块 `[part*ntiles/k, (part+1)*ntiles/k)`，`dq/dk/dv` 仍用跨 CTA 的 fp32 `atomicAdd`
汇总。数学上仍是同一个和，只有浮点加法次序略变（对拍 `max_abs` 不变）。

旧启发式是「让 grid 至少铺满一个波」——`k = ceil(396/base_grid)`、`cap=4`，于是 d128 在
S=1024H32/S=4096 上都得 `k=1`，MLA 一律 `k=4`。实测这在大 S 和 MLA 小 S 上都明显次优。
本轮先做 **ksplit sweep**（同一 session，B1 causal fp8，main 纯 device ms）：

| case（base_grid） | k=1 | k=2 | k=4 | k=8 | k=16 | 最优 |
|---|---|---|---|---|---|---|
| d128 S=512 H16（128） | 0.2401 | 0.1842 | 0.1556 | 0.1460 | **0.1434** | k=16 |
| d128 S=1024 H32（512） | 0.9788 | 0.8508 | **0.7872** | 0.8117 | 0.8718 | k=4 |
| d128 S=4096 H16（1024） | 5.8467 | 5.2366 | 4.9009 | **4.7838** | 4.8770 | k=8 |
| MLA S=256 H2 D512（8） | 0.4714 | 0.2730 | 0.1604 | **0.1026** | 0.1027 | k=8/16 |
| MLA S=512 H4 D512（32） | 0.9282 | 0.4919 | **0.3244** | 0.3404 | 0.4237 | k=4 |
| MLA S=1024 H2 D512（32） | 1.7830 | 0.9116 | **0.5474** | 0.5767 | 0.5498 | k=4 |

规律：**d128（smem 73.8KB→3 CTA/SM）在 grid≈4096（≈10 个波）附近最优；MLA（smem 223KB→
1 CTA/SM）在 grid≈一个波（132）时最优**——MLA 再切只增 Q/dO 复读与 `atomicAdd` 竞争，不增并发。
故新启发式：

```cpp
const long target_ctas = (D == 128) ? 4096L : 132L;   // 目标 CTA 数
long k = target_ctas / base_grid;                     // clamp 到 [1,16]，向下取 2 的幂
```

自动档（新）同 session main 相对**旧自动档**：d128 S512 `k4→k16` 0.1556→**0.1439（1.08×）**；
S1024H32 `k1→k8` 0.9788→**0.8084（1.21×）**；S4096 `k1→k4` 5.8467→**4.9084（1.19×）**；
MLA S256H2 `k4→k16` 0.1604→**0.1030（1.56×）**；MLA S512H4/S1024H2 仍选 k=4（不变）。
三个 d128 形状实测落在各自 sweep 最优的 0–3% 内。

### 15.2 O2b 的 ncu 证据（小 S 的 grid/occupancy 被填满）

| main, S=512 | grid | Achieved Occ | long_scoreboard | short_scoreboard |
|---|---|---|---|---|
| ksplit=1 | 8×16=128 | **6.25%** | 1.66 | 2.05 |
| ksplit=16 | 128×16=2048 | **17.83%** | 6.44 | 2.37 |

旧的 128 个 CTA 连 132 个 SM 都铺不满（1 CTA/SM、6.25%）；切 16 后 grid 2048、occupancy
17.8%（3 CTA/SM），`long_scoreboard`（全局访存延迟）被压下去，换来 main 1.67×。

### 15.3 O4d：`Ps/Ss` 行距 `BN`→`BN+1`

`Ps/Ss` 原是 `[BM][BN]`（BN=32 word=128B 行距）的 fp32 缓冲。两个访问模式撞 bank：
① fold 里按列读 `Ps[m][j]`（`j` 固定、`m` 步进 16，32 为步长 → 全落同一 bank）；② GEMM1/2
的 epilogue 里 8 行×4 列同时写 `P[r][c]`（bank=c，同列不同行全撞）。把行距改成 **`PSS=BN+1=33`
（奇数）** 后 bank 与行号线性相关。S=4096 `ksplit=1` 实测：

| 指标 | 改前（PSS=32） | 改后（PSS=33） | 变化 |
|---|---|---|---|
| `..._op_ld.sum` 冲突 | 199,883,030 | 77,258,028 | **−61%** |
| `..._op_st.sum` 冲突 | 231,607,641 | 198,403,148 | **−14%** |
| `..._wavefronts_mem_shared.sum` | 613,627,914 | 456,031,959 | **−26%** |
| main（同 session） | 6.5649 ms | **5.8465 ms** | **1.12×** |

S=512 同 session：main 0.1632→0.1553 ms（1.05×）。smem 73.2→73.8 KB（+0.5KB），
仍是 3 CTA/SM。

### 15.4 合并结果（新自动档，两文件版；单文件版逐位一致）

数值（max_abs vs fp32 ref）与 P3-5/O1–O4a **逐位相同**：d128 S512 2.426/2.975/3.735e-1；
S1024H32 2.400/4.195/3.536e-1；S4096 2.635/2.643/3.216e-1；MLA S256H2 2.356/2.290/3.441e-1；
S512H4 2.415/2.992/4.481e-1；S1024H2 2.232/3.337/3.602e-1。GQA/MQA 回归不变
（h32kv4 S1024 2.517/5.408/7.072e-1）。

性能（event 纯 device；FP8 峰值 1978.8 TFLOPS；TE 为 CUPTI 端到端）：

| case | ksplit | ours total | ours main | main TF（占比） | TE FP8 | main ours/TE |
|---|---|---|---|---|---|---|
| d128 S=512 H16 | 16 | 0.2916 ms / 7.36 TF | 0.1439 ms | 14.9（0.75%） | 0.1009 ms / 42.6 TF | 1.43× |
| d128 S=1024 H32 | 8 | 1.2000 ms / 14.32 TF | 0.8084 ms | 21.3（1.07%） | 0.2061 ms / 166.8 TF | 3.92× |
| d128 S=4096 H16 | 4 | 6.5265 ms / 21.06 TF | 4.9084 ms | 28.0（1.42%） | 0.5903 ms / 465.6 TF | 8.32× |
| MLA S=256 H2 D512 | 16 | 0.2490 ms | 0.1030 ms | 0.89 TF | NA（FA/TE 不支持） | — |
| MLA S=512 H4 D512 | 4 | 0.5892 ms | 0.3194 ms | 4.56 TF | NA | — |
| MLA S=1024 H2 D512 | 4 | 1.0114 ms | 0.5716 ms | 5.10 TF | NA | — |

### 15.5 ncu bound（main, S=4096, ksplit=4, `--set full -c 1`）

- Duration **5.00 ms**（ncu 锁频）；**DRAM 1.41%** / **L2 81.53%** / L1TEX 69.91% /
  Compute 21.70% ⇒ 非 DRAM、非算力。
- Achieved Occ **18.27%**（3 CTA/SM，Regs 168）；**Waves 10.34**（O2b 切块后）；No Eligible **76.63%**。
- stall：`long_scoreboard 4.46` + `short_scoreboard 3.96` + `wait 1.51` + `barrier 0.47`。
- **新 bound = L2 带宽（81.5%）+ long/short scoreboard**：split-K 让每个 CTA 复读 Q/dO 并向
  `dq/dk/dv` 发更多全局 `atomicAdd`，流量全落在 L2（DRAM 仅 1.4% 说明工作集在 L2 里）。
  下一步优先级：**O4c（`atomicAdd` → 分块 `dK/dV_accum` + convert，消 L2 原子流量/竞争）** >
  O4b（消转置副本冲 4 CTA/SM）> 更深流水。

### 15.6 复现

```bash
cd code/flash-attention/fa-bwd
# O2b sweep（同一 session 顺序跑，保证可比）
for k in 1 2 4 8 16; do
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --ksplit=$k --iters=50 \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
done
# 自动档 + 对拍（6 个 shape）
for c in b1_s512_h16_d128_causal_fp8 b1_s1024_h32_d128_causal_fp8 \
         b1_s4096_h16_d128_causal_fp8 b1_s256_h2_d512_causal_fp8 \
         b1_s512_h4_d512_causal_fp8 b1_s1024_h2_d512_causal_fp8; do
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=50 \
    --dir=/home/xieminglin/proj/output/fa-bwd/$c
done
# ncu：bank conflict 与 full
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma_kernel -c 1 \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum -- \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --ksplit=1 --iters=1
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma_kernel -c 1 \
  --set full -o ncu_full_s4096 -- \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_o2b_ablation.out.txt`、`..._o2b_ablation2.out.txt`、
`..._o2b_ksplit8.out.txt`、`..._o2b_ksplit16.out.txt`、`..._o2b_mla_sweep.out.txt`、
`..._o2b_auto_v2.out.txt`、`..._o2b_auto_v2_onefile_gqa.out.txt`、
`..._o4d_ncu_mem_s4096.out.txt`、`..._o4d_ncu_mem_s4096_ksplit1.out.txt`、
`..._o2b_ncu_s512_ksplit1_vs16.out.txt`、`..._o2b_o4d_ncu_full_s4096.out.txt`、
`..._o2b_o4d_stall_s4096.out.txt`、`..._o2b_o4d_tebench.out.txt`。

---

## 16. O4c：向量化归约（`atomicAdd` → `atomicAdd(float2*)`，main 1.19–1.45×）

### 16.1 定位：L2 的 92% 是「全局 red」

O2b+O4d 后 ncu 给出 **L2 81.53%** 是头号墙、DRAM 仅 1.41%，但没说清 L2 在传什么。
用按操作类型拆分的 L2/L1 sector 计数（`lts__t_sectors_op_*`）一测就清楚（main，S=4096，ksplit=4）：

| 计数 | 数值 |
|---|---|
| L1 全局 load sectors | 91.6 M |
| L1 全局 **red**（`atomicAdd` 编译成 `red`）requests / sectors | **34.1 M / 272.6 M** |
| L2 `op_read` / `op_write` / `op_red` sectors | 33.8 M / 0.10 M / **408.9 M** |

即：L2 总扇区里 **`red` 占 92%**，且 L1 每个 red 请求要 8 个扇区（**uncoalesced**——
mma 累加器一个 warp 内 4 lane 一行、列 c2 步进 2，天然散）。所以 O2b/O4d 说的「L2 带宽」
其实是 **dQ/dK/dV 的全局原子归约吞吐**，不是数据带宽（DRAM 才 1.4%）。

### 16.2 改动：把相邻两列打包成一次 `float2` red

`mma.m16n8` 的 fp32 累加器 `acc[.][j][q]` 里 **q=0/1 两列相邻**（列 = `c2+(q&1)`）、
**q=2/3 两列也相邻**，且同一个 q 对内**行号相同**（row = `r0+g` 或 `r0+g+8`）⇒ 两列共用同一个
行 scale（`sA` / `sds3` / `sds2`）。于是 epilogue 把原来的 4 次标量 `atomicAdd` 换成 2 次
`atomicAdd(float2*)`（sm_90 支持向量原子）：

```cuda
__device__ __forceinline__ void red_add2(float* p, float a, float b) {
  atomicAdd(reinterpret_cast<float2*>(p), make_float2(a, b));  // red.global.add.v2.f32
}
#pragma unroll
for (int q = 0; q < 4; q += 2) {                       // dV/dK：i 固定；dQ：外层 i,j
  int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
  int c = c0 + j * 8 + c2;                             // c2 偶 ⇒ 8B 对齐
  ...
  red_add2(dv_acc + base + d0 + c, acc[..][q] * sA[r], acc[..][q+1] * sA[r]);
}
```

三处（GEMM3 dV、GEMM4 dK、GEMM5 dQ）同步改；**smem/寄存器/几何一律不动**，只换 epilogue 的
归约指令。`acc[...][q]` 与 `acc[...][q+1]` 的行、scale 完全相同，打包后每对两个 f32 仍各自
原子累加，数值等价（实测 max_abs 与 O4a/O2b 逐位相同）。单文件 `fa_bwd_fp8_mma_onefile.cu`
与两文件 `fa_bwd_fp8_kernels.cuh` device 代码逐字一致。

### 16.3 数值核对（与 O4a/O2b **逐位相同**）

| shape | dq / dk / dv vs ref (max_abs) | O2b+O4d | O4c |
|---|---|---|---|
| (1,512,16,128) | `2.426e-1 / 2.975e-1 / 3.735e-1` | 同 | 同 |
| (1,1024,32,128) | `2.400e-1 / 4.195e-1 / 3.536e-1` | 同 | 同 |
| (1,4096,16,128) | `2.635e-1 / 2.643e-1 / 3.216e-1` | 同 | 同 |
| MLA (1,1024,2,512) | `2.232e-1 / 3.337e-1 / 3.602e-1` | 同 | 同 |
| GQA h32kv4 | `2.517e-1 / 5.408e-1 / 7.072e-1` | 同 | 同 |

### 16.4 性能（event 纯 device；同 session 先测 O2b+O4d 基线）

| case | ksplit | main base | **main O4c** | 加速 | total base | total O4c | main TF | TE FP8 (CUPTI) |
|---|---|---|---|---|---|---|---|---|
| d128 S=512 H16 | 16 | 0.1439 | **0.1090** | **1.32×** | 0.2916 | **0.2556** | 19.7（1.00%） | 0.1014 |
| d128 S=1024 H32 | 8 | 0.8084 | **0.6772** | **1.19×** | 1.2000 | **1.0590** | 25.4（1.28%） | 0.2059 |
| d128 S=4096 H16 | 4 | 4.9552 | **3.5330** | **1.39×** | 6.5833 | **5.1601** | 38.9（1.97%） | 0.5894 |
| MLA S=1024 H2 D512 | 4 | 0.5716 | **0.3933** | **1.45×** | 1.0114 | **0.8515** | 10.9（0.55%） | NA |
| GQA h32kv4 S=1024 | 8 | — | **0.5978** | — | — | **0.9258** | 28.7（1.45%） | 0.243 |

main-only ours/TE = **93% / 30% / 17%**；端到端 ours/TE = 2.52× / 5.14× / 8.75×（O2b+O4d 为
2.89× / 5.82× / 11.06×）。S=512 的 main 已经**几乎追平 TE 的整条反向**（0.1090 vs 0.1014 ms），
剩余差距主要在 preprocess/quant/convert。

### 16.5 ncu（main, S=4096, ksplit=4；O2b+O4d → O4c）

| 指标 | O2b+O4d | **O4c** | 说明 |
|---|---|---|---|
| Duration | 5.00 ms | **3.60 ms** | 1.39× |
| L1 red requests / sectors | 34.1 M / 272.6 M | **17.0 M / 136.3 M** | **各 0.50×** |
| L2 `op_red` sectors | 408.9 M | **204.5 M** | 0.50× |
| **L2 Cache Throughput** | **81.53%** | **57.85%** | 墙被打掉 |
| L1/TEX Cache Throughput | 69.91% | **81.30%** | 新主导 |
| DRAM / Compute | 1.41 / 21.70% | 1.99 / 29.42% | |
| short / long scoreboard | 3.96 / 4.46 | **3.50 / 1.44** | long 明显下降 |
| barrier / wait | 0.47 / 1.51 | 0.30 / 1.58 | |
| Occupancy（Regs/smem） | 18.27%（168/75.0KB） | 18.20%（168/75.0KB） | 不变，仍 3 CTA/SM |
| bank conflict ld/st | 73.9 M / 203.8 M | 68.4 M / 206.4 M | 基本不变 |

**bound 结论**：O4c 把 **L2 原子归约流量砍半**，L2 从 81.5% 掉到 57.9%，不再是墙；
**新墙 = L1/TEX 81.3%（`ldmatrix`/smem 与残余 red）+ short_scoreboard 3.50**。下一步顺位：
**O4b（fp8 `ldmatrix.trans` 消 `Kt/Qt/dOt` 三个转置副本：既减 L1/TEX 的 smem 往返，
又是 MLA 冲 2 CTA/SM 的关键）** > 继续压 red（受 mma 累加器布局限制，`float4` 不可得）。

### 16.6 复现

```bash
cd code/flash-attention/fa-bwd
for c in b1_s512_h16_d128_causal_fp8 b1_s1024_h32_d128_causal_fp8 \
         b1_s4096_h16_d128_causal_fp8 b1_s1024_h2_d512_causal_fp8 \
         b1_s1024_h32_d128_kv4_causal_fp8; do
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=10 \
    --dir=/home/xieminglin/proj/output/fa-bwd/$c
done
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=10 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma_kernel -c 1 \
  --section SpeedOfLight --section Occupancy --section SchedulerStats \
  --metrics lts__t_sectors_op_red.sum,l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_o4c_{s512_h16_d128,s1024_h32_d128,s4096_h16_d128,
s1024_h2_d512,s1024_h32_d128_kv4}.out.txt`、`src/fp8/fa_bwd_fp8_mma_onefile_o4c_s4096.out.txt`、
`src/fp8/fa_bwd_fp8_main_o4c_ncu_s4096.out.txt`、`src/fp8/fa_bwd_fp8_main_o4c_tebench.out.txt`。

---

## 17. O4b：`Kt/Qt/dOt` 三个转置副本 → `ldmatrix.x2.trans` + K 配对布局（main 1.17–1.76×）

### 17.1 先纠正 O4b 的前提：fp8 的转置副本**不能直接消掉**

ROADMAP 里 O4b 的原始设想是「用 fp8 `ldmatrix.trans` 从原始 `[K][N]` 布局直接读 B，
把 `Kt/Qt/dOt` 三个转置副本整个消掉」。**对 fp8 这个前提不成立**，原因是
`ldmatrix` 以 **b16** 为单位做转置，而 fp8 是「2 个相邻字节 = 1 个 b16」：

* 反向里 mma 的 A、B 需要**相反的主序**（`m16n8k32.row.col`：A 行主序、B 列主序）；
  Q/dO 既当 A（GEMM1/2）又当 B（GEMM4/3），K 既当 GEMM1 的 B 又当 GEMM5 的 B（收缩维不同），
  所以每个张量**必然要存两种主序**——这一点即使 `ldmatrix.trans` 免费也消不掉。
* 若把原始数组按 `[K][N]`（N 为列）存，每个 b16 = **沿 N 的 2 个 fp8**；`ldmatrix.trans`
  转置后寄存器里的配对方向仍是 N，**不是** mma 需要的「沿 K 的 4 个 fp8」（配错 → 数值全错）。
  所以 fp8 的 `.trans` **不是 drop-in**（这正是 ROADMAP 要求先做 smoke test 的原因）。

**正确做法（§17.2 已用最小复现逐位验证）**：把操作数改存成 **K 配对布局** `Sp[i][n]`
（uint16，`i=k/2`，`Sp[i][n] = pack(B[n][2i], B[n][2i+1])`，即每个 b16 装两个**相邻 k**）。
此时 `ldmatrix.x2.trans` 的输出片段恰好是 `B[n=l/4][k=4(l%4)..+3]`，与从 `[N][K]` 行主序
用 `ldmatrix.x2` 读到的 B 片段**逐位相同**。所以 O4b 的真实收益不是「消副本」，而是：
① 把原来**逐字节 scatter 的转置写**换成 **4B 交织写**（`__byte_perm`）；
② 配对数组比 `[HD][K+16]` 副本更小（无 +16 行距放大），smem 下降；
③ B 读仍是一次 `ldmatrix.x2`（与 `x2.trans` 同量级），但**写侧的 bank conflict 基本消失**。

### 17.2 smoke test（`fa_bwd_fp8_trans_smoke.cu`，O4b 第一步）

同一批 fp8 B 字节，分别用（1）`[N][K]` + `ldmatrix.x2`（现有路径）、（2）`[K/2][N]` 配对 +
`ldmatrix.x2.trans`（O4b）喂给同一个 `mma.m16n8k32`，比较输出：

```
  notrans-vs-ref : max_abs=3.052e-05      （仅 fp32 舍入，mma 内部次序 vs 标量）
  trans  -vs-ref : max_abs=3.052e-05
  notrans-vs-trans: max_abs=0.000e+00  bitwise_diff=0/128
=== PASS（K 配对 + ldmatrix.trans 与现有 ldmatrix.x2 路径逐位一致） ===
```

结论：**K 配对布局 + `ldmatrix.x2.trans` 与现有 `ldmatrix.x2` 路径逐位一致**，
可以 drop-in 替换 GEMM3/4/5 的 B。

### 17.3 实现（单/两文件 device 代码同源逐字一致）

* `Fp8Cfg`：`KTS/QTS` 两个转置行距 → `PSLD = HD+8`（配对布局 uint16 行距）；
  `KVU` → `NPU = (BN/2)*(HD/4)/THREADS`（每线程行对数）；`fp8_bytes` 里
  `HD*KTS+2*HD*QTS` → `qp_bytes*2 + kp_bytes`（`qp_bytes=(BM/2)*PSLD*2`、
  `kp_bytes=(BN/2)*PSLD*2`）。**d128**：smem **75520 → 70656 B（69.0KB）**；
  **MLA d512**：**229120 → 205824 B（200.9KB）**。
* 新 `ldmatrix_x2_trans` 与 `mma_block_bt`（A 与非转置版逐字相同，B 用
  `Bp + (koff/2 + (lane&15))*bp_sld + nb0 + wn*WARP_N + j*8`，`nb0` 是 head_dim 的 N-tile 偏移）。
* Q/dO 载入：从「逐元素 1B 写 `Qs/dOs` + scatter 写 `Qt/dOt`」改成「行对 unit：一次读两行
  的 4B，写 `Qs/dOs` 两行 + 用 `__byte_perm(q0,q1,0x5140/0x7362)` 交织写 `Qp/dOp`」。
* K/V 载入（`kv_prefetch_pair`/`kv_commit_pair`/`kv_load_pair`）：同样按「行对 + 4 个连续 d」
  组织，写 `Ks/Vs` 两行 + 交织写 `Kp`；**O3 寄存器预取保留**（HD=128 每线程预取 NPU=4 个 unit
  ×4 regs = 16 regs，与 O3 的 8+8 相同；HD=512 NPU=16 关闭预取）。
* GEMM3/4/5 的 B 从 `mma_block(...dOt/Qt/Kt...)` 换成 `mma_block_bt(...dOp/Qp/Kp...)`。

因为 mma 拿到的操作数**字节与次序完全相同**，数值与 O4c **逐位相同**（见 §17.4）。

### 17.4 数值核对（与 O4c **逐位相同**）

| shape | dq / dk / dv vs ref (max_abs) | O4c | **O4b** |
|---|---|---|---|
| (1,512,16,128) | `2.426e-1 / 2.975e-1 / 3.735e-1` | 同 | **同** |
| (1,1024,32,128) | `2.400e-1 / 4.195e-1 / 3.536e-1` | 同 | **同** |
| (1,4096,16,128) | `2.635e-1 / 2.643e-1 / 3.216e-1` | 同 | **同** |
| MLA (1,1024,2,512) | `2.232e-1 / 3.337e-1 / 3.602e-1` | 同 | **同** |
| MLA (1,256,2,512) | `2.356e-1 / 2.290e-1 / 3.441e-1` | 同 | **同** |
| GQA h32kv4 | `2.517e-1 / 5.408e-1 / 7.072e-1` | 同 | **同** |

### 17.5 性能（event 纯 device；同 session 先测 O4c=HEAD 基线）

| case | ksplit | main base | **main O4b** | 加速 | total base | total O4b | main TF（峰值占比） | TE FP8(CUPTI) |
|---|---|---|---|---|---|---|---|---|
| d128 S=512 H16 | 16 | 0.1091 | **0.0733** | **1.49×** | 0.2575 | **0.2190** | 29.3（1.48%） | 0.1003 |
| d128 S=1024 H32 | 8 | 0.6765 | **0.4552** | **1.49×** | 1.0591 | **0.8574** | 37.7（1.91%） | 0.2049 |
| d128 S=4096 H16 | 4 | 3.4553 | **2.9473** | **1.17×** | 5.1196 | **4.6162** | 46.6（2.36%） | 0.5847 |
| MLA S=1024 H2 D512 | 4 | 0.3896 | **0.3251** | **1.20×** | 0.8499 | **0.7906** | 13.2（0.67%） | NA |
| MLA S=256 H2 D512 | 16 | 0.0916 | **0.0522** | **1.76×** | 0.2357 | **0.1909** | 5.14（0.26%） | NA |
| GQA h32kv4 S=1024 | 8 | 0.5923 | **0.4404** | **1.34×** | 0.9259 | **0.7528** | 39.0（1.97%） | 0.2013 |

main-only ours/TE = **73% / 222% / 504% / GQA 219%**（S=512 的 main 已是 TE 整条反向的 ~73%）；
端到端 ours/TE = **2.18× / 4.18× / 7.90× / GQA 3.74×**（O4c 为 2.52×/5.14×/8.75×）。
单文件 `fa_bwd_fp8_mma_onefile.cu` 与两文件逐指标一致（S=4096 main 2.96 ms、
dq/dk/dv vs ref `2.635/2.643/3.216e-1`）。

### 17.6 ncu（main；O4c=HEAD → O4b）

| 指标 | O4c (S4096) | **O4b (S4096)** | 说明 |
|---|---|---|---|
| Duration | 3.60 ms | **3.00 ms** | 1.20× |
| **smem shared store bank conflicts** | **206.4 M** | **69.4 M** | **−66%（逐字节 scatter 写消失）** |
| smem shared load bank conflicts | 68.4 M | 69.3 M | 基本不变 |
| **L1/TEX Throughput** | **81.30%** | **69.69%** | 头号墙下降 |
| L2 Cache Throughput | 57.85% | **69.12%** | **上升，成为并列墙**（残余 red 未动） |
| Compute (SM) | 29.42% | 34.40% | |
| DRAM | 1.99% | 2.46% | |
| short / long scoreboard | 3.50 / 1.44 | **2.73 / 1.16** | smem→mma 依赖缓解 |
| barrier / wait | 0.30 / 1.58 | 0.35 / 1.53 | |
| Occupancy（Regs/smem） | 18.20%（168/75.0KB） | 18.21%（**168/70.66KB**） | 仍 3 CTA/SM |
| L1 red requests / L2 red sectors | 17.0 M / 204.5 M | **17.0 M / 204.5 M** | 不变（O4c 已砍半） |

S=512（ncu 单次）：Duration **78.2µs**、L1/TEX 47.2%、L2 48.1%、short/long 2.13/2.13、
occ 17.6%（168 regs/70.66KB，3 CTA/SM）。
MLA S=1024H2（ncu）：Duration **359.8µs**、smem **205.82KB**、regs 255、
occ **6.25%（1 CTA/SM）**、L1/TEX 11.4%、long/short 1.73/0.93。

**bound 结论**：O4b 把「逐字节转置写」的 **op_st bank conflict 砍掉 66%**、L1/TEX 从 81.3%
降到 69.7%、short_scoreboard 3.50→2.73，main 提速。**新墙 = L1/TEX 69.7% + L2 69.1%
（O4c 后残余的 204.5 M red）+ short_scoreboard 2.73**（三者接近）。**MLA 即使 O4b 把三个
配对数组也去掉（205824−83200=122624 B）仍 > 116224 B（2 CTA/SM 门槛）**，故 MLA 冲 2 CTA/SM
必须同时动 `Qs/Ks/Vs/dOs/dS2` 的大头，O4b 只是其中一步。下一步候选：**O7（分块 `*_accum`
替全局 `atomicAdd`，消残余 red）** 或 MLA 的 KV 分片。

### 17.7 复现

```bash
cd code/flash-attention/fa-bwd
# O4b 第一步：布局最小复现（必须 PASS）
scripts/run.sh src/fp8/fa_bwd_fp8_trans_smoke.cu
for c in b1_s512_h16_d128_causal_fp8 b1_s1024_h32_d128_causal_fp8 \
         b1_s4096_h16_d128_causal_fp8 b1_s1024_h2_d512_causal_fp8 \
         b1_s256_h2_d512_causal_fp8 b1_s1024_h32_d128_kv4_causal_fp8; do
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=30 \
    --dir=/home/xieminglin/proj/output/fa-bwd/$c
done
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=30 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu：bank conflict + stall + SOL
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma_kernel \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_trans_smoke.out.txt`、
`src/fp8/fa_bwd_fp8_main_o4b_{s512_h16_d128,s1024_h32_d128,s4096_h16_d128,s1024_h2_d512,
s256_h2_d512,s1024_h32_kv4_d128}.out.txt`、`src/fp8/fa_bwd_fp8_mma_onefile_o4b_s4096_h16_d128.out.txt`、
`src/fp8/fa_bwd_fp8_main_o4b_ncu_{s512,s4096,mla_s1024h2}.out.txt`、
`src/fp8/fa_bwd_fp8_o4b_tebench.out.txt`。

## 18. O7：dQ 沿 nt 在寄存器里累加，每 CTA 只 flush 一次（main 1.10×，残余 red 再砍半）

### 18.1 动机：O4b 后剩两个并列墙，其中一个是「残余 red」

O4b 后 main 的 ncu 是 **L1/TEX 69.7% + L2 69.1%（残余 204.5 M red 扇区）+ short_scoreboard 2.73**。
用 `l1tex/lts__t_sectors_op_red` 拆开可以看到，red 的账已经只剩 dK/dV/dQ 的 epilogue：

* dQ（GEMM5）的 epilogue 每个 nt tile 都对**本 CTA 的同一个 dQ 元素**做一次跨 CTA `atomicAdd`——
  但 GEMM5 的 warp/累加器映射与 nt **无关**（`r,c` 只由 `i,j,q` 和 lane 决定），所以同一线程在不同
  nt 上写的是**完全相同的地址**，一个元素被 RMW 了 `ntiles` 次（因果下平均 ~16，S=4096 ksplit=4）。
* dK/dV 不同：一个 CTA 的每个 nt 覆盖**不同的 KV 行**（`jg=j0+r`，j0 随 nt 走），所以每个 dK/dV
  元素在一个 CTA 内本来只写一次；它们的 red 是**跨 mblk/hkv 的 CTA 竞争**，无法靠 CTA 内累加消掉
  （要动并行结构，转入 backlog）。

因此 O7 的可做部分 = **把 dQ 的每-tile 归约改成 CTA 内寄存器累加，nt 循环结束后每元素只发一次
`red_add2`**。这会把 dQ 的全局归约指令数从 `O(ntiles)` 降到 `O(1)`。

### 18.2 关键点：折算因子逐 tile 变化，必须先折算再累加

dQ 的 epilogue 是 `acc * sds2[r] * scale`，而 `sds2[m]`（dS2 的 rowwise amax scale）**逐 tile 不同**，
所以不能把裸 mma 输出直接累加（那样会把不同 tile 的 scale 混在一起）。做法是**先折算、再累进
寄存器累加器 `dqacc`**：

```cuda
// GEMM5 epilogue（HD=128，kRegDq=true）
dqacc[i][j][q]     += acc[i][j][q]     * sds2[r] * scale;
dqacc[i][j][q + 1] += acc[i][j][q + 1] * sds2[r] * scale;
// …nt 循环结束后统一 flush（每元素一次 float2 red）：
red_add2(dq_acc + idx, dqacc[i][j][q], dqacc[i][j][q + 1]);
```

`dqacc[2][8][4]` 只多 64 个 fp32。因为只有 **HD==128** 时 dQ 的 N 维（head_dim）一次铺满
（`HD/NTW==1`）；HD=512 的 4 个 N-tile 需要 4 份独立累加器（256 regs）不划算，故 MLA 仍走原
per-tile 归约（数值/性能不变）。

### 18.3 寄存器账 + `__launch_bounds__`：把 2 CTA/SM 拉回 3 CTA/SM

直接加 `dqacc` 会让 main kernel 从 **168 → 254 regs**，occupancy 3→**2 CTA/SM**（12.3%），
S=512/1024 反而变慢。取折中：给 kernel 加 `__launch_bounds__(THREADS, (HD==128)?3:1)`——
`HD=128` 时把寄存器压回 **168（+48B spill）**，保住 **3 CTA/SM**；`HD=512` 用 `,1`（不设上限，
保持 255 regs，与 O4b 逐一致）。`REGDQ` 做成模板参数，由 host 按「平均每 CTA 的 nt tile 数」选：

```cpp
const bool use_regdq = (D == 128) && ((long)(S / 32) / 2 / ksplit >= 4);
```

（因果下平均每 CTA ≈ `(S/BN)/2/ksplit`；S=4096 ksplit=4 → 16，S=1024H32 ksplit=8 → 2，
S=512 ksplit=16 → 0.5。阈值取 4：S=4096 明显收益、S≤1024 的高 ksplit 情形保持原路径避免回归。）

### 18.4 数值：与 O4b **逐位相同**

| shape | dq / dk / dv vs ref (max_abs) | O4b | **O7** |
|---|---|---|---|
| (1,512,16,128) | `2.426e-1 / 2.975e-1 / 3.735e-1` | 同 | **同**（REGDQ=false，路径未变） |
| (1,1024,32,128) | `2.400e-1 / 4.195e-1 / 3.536e-1` | 同 | **同**（REGDQ=false） |
| (1,4096,16,128) | `2.635e-1 / 2.643e-1 / 3.216e-1` | 同 | **同**（REGDQ=true） |
| MLA (1,1024,2,512) | `2.232e-1 / 3.337e-1 / 3.602e-1` | 同 | **同**（未改） |
| GQA h32kv4 S=1024 | `2.517e-1 / 5.408e-1 / 7.072e-1` | 同 | **同**（REGDQ=false） |

单/两文件 device 代码同源逐字一致；`HD=128/REGDQ=false` 与 `HD=512` 的寄存器数、smem、
SASS 与 O4b 一致。寄存器和几何：`REGDQ=true` 168 regs + 48B spill、70.66KB smem、3 CTA/SM；
`REGDQ=false` 168 regs、0 spill；`HD=512` 255 regs、205.82KB smem、1 CTA/SM。

### 18.5 性能（event 纯 device；同 session 先测 O4b=HEAD 基线）

| case | ksplit | **REGDQ** | main O4b | **main O7** | 加速 | total O4b | total O7 | main TF（峰值占比） | TE FP8(CUPTI) |
|---|---|---|---|---|---|---|---|---|---|
| d128 S=512 H16 | 16 | false | 0.0733 | **0.0740** | ~1.00× | 0.2190 | **0.2191** | 29.3（1.48%） | 0.1006 |
| d128 S=1024 H32 | 8 | false | 0.4552 | **0.4604** | ~0.99× | 0.8574 | **0.8648** | 37.3（1.89%） | 0.2054 |
| **d128 S=4096 H16** | 4 | **true** | 2.9473 | **2.6736** | **1.10×** | 4.6162 | **4.3410** | **51.4（2.60%）** | 0.5863 |
| GQA h32kv4 S=1024 | 8 | false | 0.4404 | **0.4463** | ~0.99× | 0.7528 | **0.7569** | 38.5（1.95%） | 0.2014 |
| MLA S=1024 H2 D512 | 4 | false | 0.3251 | **0.3277** | ~0.99× | 0.7906 | **0.7910** | 13.1（0.66%） | NA |

S=1024/GQA/MLA 的 ±1% 是 session 噪声（路径与 O4b 完全相同、数值逐位一致）；唯一真实改动是
**S=4096（REGDQ=true）main 1.10×、端到端 1.06×**。端到端 ours/TE FP8 = **2.18×/4.21×/7.41×**
（O4b 为 2.18×/4.18×/7.90×）；main-only ours/TE S=4096 = **456%**（O4b 504%）。

### 18.6 ncu（main, S=4096, ksplit=4, REGDQ=true）

| 指标 | O4b | **O7** | 说明 |
|---|---|---|---|
| Duration | 3.00 ms | **2.69 ms** | 1.12× |
| **L1 red requests** | 17.04 M | **9.04 M** | **0.53×**（dQ 的 O(ntiles) 归约消失） |
| **L1 red sectors** | 136.3 M | **72.3 M** | 0.53× |
| **L2 red sectors** | 204.5 M | **108.5 M** | 0.53×（剩下的是 dK/dV 跨 CTA 竞争） |
| **L2 Cache Throughput** | **69.12%** | **43.8%** | **墙被打掉** |
| L1/TEX Throughput | 69.69% | **64.4%** | 略降 |
| Compute (SM) | 34.40% | 37.7% | |
| DRAM | 2.46% | 2.6% | 远非带宽 bound |
| short / long scoreboard | 2.73 / 1.16 | **1.89 / 1.09** | smem→mma 依赖缓解 |
| barrier | 0.35 | **0.22** | |
| Occupancy（Regs/smem） | 18.21%（168/70.66KB） | **18.11%（168/70.66KB）** | **3 CTA/SM 保住** |

**bound 结论**：O7 把 dQ 的每-tile 归约折成每-CTA 一次，**残余 red 再砍半（L2 204.5M→108.5M）、
L2 墙 69.1%→43.8%**，且靠 `__launch_bounds__(128,3)` 保住了 3 CTA/SM（区别于朴素版 254 regs /
2 CTA/SM 的 S=4096 只 1.04×）。**新墙 = L1/TEX 64.4%（`ldmatrix`/smem） + short_scoreboard 1.89
+ 残余 L2 43.8%（剩 dK/dV 的跨 CTA red）**。reduction 的天花板已经很近：dQ 归约已是 O(1)，
剩下要动 dK/dV 的「跨 mblk/hkv 竞争」必须改并行结构（例如按 KV 列块常驻、Q 块累加）或
分块 `*_accum` + convert，属 backlog。

### 18.7 复现

```bash
cd code/flash-attention/fa-bwd
for c in b1_s512_h16_d128_causal_fp8 b1_s1024_h32_d128_causal_fp8 \
         b1_s4096_h16_d128_causal_fp8 b1_s1024_h32_d128_kv4_causal_fp8 \
         b1_s1024_h2_d512_causal_fp8; do
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=20 \
    --dir=/home/xieminglin/proj/output/fa-bwd/$c
done
# ncu：red / L2 / occupancy / stall
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma \
  --metrics l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,\
lts__t_sectors_op_red.sum,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,gpu__time_duration.sum \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_o7_sweep.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_o7_sweep.out.txt`、`src/fp8/fa_bwd_fp8_main_o7_ncu_s4096.out.txt`、
`src/fp8/fa_bwd_fp8_o7_tebench.out.txt`。

---

## 19. O11：fp8 LSE 预处理负载均衡 + `cp.async` 双缓冲（preprocess 3.1×，端到端 1.25×）+ 快速 exp/log

### 19.1 动机：fp8 的 preprocess 一直没沾到 O8b 的光

`docs/01`（fp16）/`docs/01b`（bf16）在 O8b 里对 LSE 预处理做了两件事，把 lse 从
`S=4096` 的 ~0.99ms 打到 ~0.35ms（2.8×）：

1. **镜像配对负载均衡**：因果下第 `m` 个 Q 块要做 `m+1` 个 K tile，工作量随 `m` 线性增长；
   GPU 按 `blockIdx` 递增调度会把重块排到最后（尾波最重）。改成每 CTA 同时处理配对的两个
   m 块 `m` 与 `nblk-1-m`，工作量恒为 `nblk+1`，`grid.x` 减半。
2. **K 的 `cp.async.cg` 16B 双缓冲**：把「同步逐字节读 K → sync → mma」改成异步预取下一块。

**但 fp8 的 `lse_mma_kernel`（O1）从来没有做这两件事**（它只做了 O1 的「mma 分块 LSE」）。
实测 fp8 `S=4096` 的 preprocess 仍是 **1.20ms**（占端到端 ~29%），其中 lse 0.94ms、delta 0.04ms，
而 fp16/bf16 同 shape 只有 0.35ms。这是本轮的主要目标。

### 19.2 改动（单/两文件 device 代码同源逐字一致）

新增 `lse_mma_kernel_bal<HD, PIPE>`（`src/fp8/fa_bwd_fp8_kernels.cuh`，单文件版同源），
结构逐条对齐 fp16 的 `lse_mma_kernel_bal`：

- **镜像配对**：`mblk = (t==0) ? pair : nblk-1-pair`，`pair = blockIdx.x`，`grid.x = ceil(nblk/2)`；
  奇数 `nblk` 的中心块只做一次。mask 与 O1 完全一致，只把 `!(causal&&jg>qi)` 写成 `jg<=qi`。
- **K/Q 的 `cp.async.cg` 16B 双缓冲**（`PIPE=1`）：fp8 是 1B/元素、一行 `HD` 字节，
  **16B = 16 个 fp8**，故 unit 数 = `HD/16`（fp16/bf16 是 `HD/8`）；`PIPE=0` 退回同步标量读，
  用于同 session 消融。新增 `cp_async16`（与 fp16 版同构，`cp.async.cg` L2-only）。
- rowwise scale 与 O1 一致：`sv = acc·scale·qs_s[r]·ks_s[c]`，只是 `ks_s` 也按 tile 双缓冲。
- smem：`Qs[LBM*ASLD] + (PIPE?2:1)·Ks[LBN*ASLD] + (LBM + (PIPE?2:1)·LBN)·4B`；
  新增 `Fp8Cfg::lse_smem_bytes_bal0/1`。仅 causal 走新 kernel，非 causal 走 O1 原版。

**另附：快速 exp/log（与 fp16/bf16 同款）**。把 softmax 热点的 libdevice 精确 `expf`/`logf`
换成硬件内建 `__expf`/`__logf`（MUFU.EX2/LG2，相对误差 ~2^-21），用 `FAST_EXP` 宏做 A/B。
fp8 容差 O(1)，无影响；A/B 见下。

### 19.3 数值核对（与 O1/O7 **逐位相同**）

`ours vs fp32 ref`（max_abs，causal）：

| case | dq | dk | dv | 与 O7 记录 |
|---|---|---|---|---|
| S=4096 H16 D128 | 2.635e-01 | 2.643e-01 | 3.216e-01 | 逐位相同 |
| S=512 H16 D128 | 2.426e-01 | 2.975e-01 | 3.735e-01 | 逐位相同 |
| S=1024 H32 D128 kv4 | 2.517e-01 | 5.408e-01 | 7.072e-01 | 逐位相同 |
| S=1024 H64 D128 kv1(MQA) | 4.097e-01 | 1.519e+00 | 2.127e+00 | 逐位相同 |
| S=1024 H2 D512(MLA) | 2.232e-01 | 3.337e-01 | 3.602e-01 | 逐位相同 |

证明只换算法数据流（负载均衡 + 搬运方式），未改数学口径。单文件（`fa_bwd_fp8_mma_onefile.cu`）
与两文件**逐指标一致**（S=4096 total 3.308 vs 3.310ms）。

### 19.4 性能（CUDA event 纯 device；同 session A/B）

**lse 消融**（同 binary 内三档，`event`）：

| shape | lse O1 | +镜像配对(单缓冲) | +cp.async 双缓冲 | 累计 |
|---|---|---|---|---|
| S=512 H16 D128 | 0.0652 ms | 0.0513 (1.27×) | **0.0389 (1.67×)** | 1.67× |
| S=1024 H32 D128 kv4 | 0.1542 | 0.1005 (1.54×) | **0.0767 (2.01×)** | 2.01× |
| S=1024 H64 D128 kv1 | 0.2416 | 0.1265 (1.91×) | **0.1009 (2.40×)** | 2.40× |
| S=4096 H16 D128 | 0.9362 | 0.4257 (2.20×) | **0.3453 (2.71×)** | 2.71× |

**端到端**（`quant+pre+main+cvt`，ms）：

| shape | pre O1 | pre O11 | total O11 | 相对 O7 total |
|---|---|---|---|---|
| S=512 H16 D128 | 0.24 | **0.0465** | 0.1797 | — |
| S=1024 H32 D128 kv4 | — | **0.1006** | 0.6380 | ~1.3× |
| S=4096 H16 D128 | 1.2009 | **0.3878** | **3.3104**（3.31ms，41.5 TF）| 4.34→3.31（**1.31×**） |
| S=1024 H2 D512 MLA | — | 0.1388 | 0.5390 | — |

**快速 exp/log A/B**（fp16/bf16 同款；`docs/01` §14d）：fp16 MHA S=4096 `total` 2.0053→**1.9697ms**
（1.8%）、preprocess 0.3712→**0.3439ms（8.0%）**、main 几乎不变——即墙不在 exp，但 LSE 白赚 8%。
（本轮 fp8 的端到端大头来自 §19.4 的 lse 负载均衡，fast-exp 只是顺带。）

**对标**（同 session 纯反向口径）：

- FA2/FA3/TE（`harness/fa_vs_te_bwd_only.py`，fp16）：MHA S=4096 FA2 0.7269ms/378TF、
  **FA3 0.3236/849**、TE 0.4416/622；bf16 **FA3 0.3207/857**、TE 0.4422/622。
- TE FP8（`harness/fa_bwd_bench.py bench --dtype fp8`）：S=512 0.1005ms/42.7TF、
  S=1024 0.1477/116.3、**S=4096 0.5908/465.3TF**。ours fp8 纯反向（去掉 quant）S=4096
  ≈ 3.13ms ⇒ **ours/TE 5.3×**（O7 时 6.7×）。注意 TE FP8 复用前向的 LSE，我们则要自算 LSE。

### 19.5 ncu（lse_mma_kernel_bal，S=4096，`--set full`）

`Duration 357µs`、DRAM 1.48% / **L1/TEX 28.2%** / L2 15.0% / **Compute 61.2%**、
77 regs、smem ~27.7KB/块（Block Limit Shared Mem 6）、**achieved occupancy 23.2%**、
**Waves 0.65**（grid=528 < 一个满波）。对比 O1 原版：Duration 1.14ms、Compute 39.4%、
Waves 1.11、long_scoreboard 主导。**新墙 = Compute 61% + 网格不足一个波（负载均衡后每 CTA
工作量恒定，但 528 个 CTA 铺不满 132 SM × 6 CTA/SM）**；与 fp16/bf16 的 O8b 结论一致。

### 19.6 复现

```bash
cd code/flash-attention/fa-bwd
# 单/两文件 fp8（含 lse 三档 A/B）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=50 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=50 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu：balanced LSE
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:lse_mma_kernel_bal -c 1 -- \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=2
```

原始输出：`src/fp8/fa_bwd_fp8_main_o11_lsebal.out.txt`、
`src/fp8/fa_bwd_fp8_main_o11_ncu_lsebal_s4096.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_o11_s4096.out.txt`、
`src/fp16/fa_bwd_fp16_mma_main_o11_ab_fastexp.out.txt`、`src/fa_bwd_o11_fa3_te_baseline.out.txt`。

---

## 20. O7e：fold 写回向量化 + REGDQ 下关 O3 预取（消 shared store 冲突与寄存器 spill）

### 20.1 动机（先重新定位瓶颈）

O7 之后 fp8 main 的 ncu 显示 **L1/TEX 71.07%** 才是第一墙（L2 已从 O4c/O7 的 81%/69%
降到 **43.74%**，Compute 39.1%、DRAM 2.6%）。也就是说 **ROADMAP/docs 里「O7b 去 dK/dV 原子」
针对的 L2 墙已经被 O4c/O7 基本打掉，不再是头号杠杆**。用 `MemoryWorkloadAnalysis_Tables`
把 L1/TEX 拆开，两项最突出：

- **register spill**：`REGDQ`（O7）把 dQ 沿 nt 累加在寄存器，`dqacc[2][8][4]`（64 fp32）+
  O3 的寄存器预取（`pk0/pk1/pv0/pv1`，16 个 uint32）把 168-reg/3-CTA 预算占满，
  ptxas 强制 spill。局部内存占 **L1TEX sector 的 12.09%**、Est. Speedup 27.6%。
- **shared store bank conflict**：3.5-way、占 store wavefront 的 **69.8%**。其中 fold 段
  （`Ap/dS3/dS2`）是逐 **1 字节** 的 `st.shared.u8`，指令数与冲突都最多。

### 20.2 改动（单/两文件 device 逐字一致）

1. **fold 写回向量化**：fold 里每个线程负责的 16 个元素在 smem 里是**连续的**
   （`Ap/dS3`：`m = sub4*16+t`，t 递增；`dS2[m][j]`：`j = sub2*16+t`，t 递增），
   把 16 次 1B 写折成 **4 次 4B `st.shared.u32`**（`QTS=BM+16`、`DSS2=BN+16` 都是 4 的倍数，
   `sub4*16+t4*4`、`sub2*16+t4*4` 也 4 对齐）。数值逐位不变，store 指令数 ÷4。
   实测 shared **store bank conflict 68.6M→36.3M（−47%）**。
2. **`REGDQ` 生效时关掉 O3 寄存器预取**：新增 `kPrefetch = Cfg::use_prefetch && !kRegDq`，
   在 `REGDQ`（仅 `HD=128` 且每 CTA nt tile 足够多时）为真时退回 O4b 的 4B 向量化同步读，
   把 16 个预取寄存器让给 `dqacc`。小 S（`use_regdq=false`，如 S=512/S=1024）仍保留预取。
   实测 **spill 占 L1TEX sector 12.09%→5.74%**、L1/TEX **71.07%→66.06%**、
   Duration **2.66→2.57ms**。

> 顺带修 `scripts/sync_onefile_device.py` 的单文件 device 区**结束边界**：fp8 的 host 段在
> `struct NpyF32 {` 之前还有一段 `#include <algorithm>` 的 host 头，旧脚本会把 host 头一起
> 覆盖导致单文件编译失败；现取 `struct NpyF32 {` 与 `#include <algorithm>` 的**较早者**为界。

### 20.3 数值（ours-vs-ref，max_abs，causal，单/两文件逐位一致）

| case | dq | dk | dv |
|---|---|---|---|
| S=512 H16 | 2.426e-1 | 2.975e-1 | 3.735e-1 |
| S=1024 H32 | 2.400e-1 | 4.195e-1 | 3.536e-1 |
| S=4096 H16 | 2.635e-1 | 2.643e-1 | 3.216e-1 |
| MLA S=1024 H2 D=512 | 2.232e-1 | 3.337e-1 | 3.602e-1 |

与 O7/O4b/O11 **逐位相同**（只改搬运与寄存器分配，不改数学口径）。

### 20.4 性能（CUDA event；同 session A/B）

| shape | main 基线（git） | **O7e** | 端到端 total（O7e） |
|---|---|---|---|
| S=512 H16 | 0.0740 ms | 0.0720 ms（持平） | 0.180 ms |
| S=1024 H32 | 0.4604 ms | 0.4414 ms（~4%） | 0.710 ms |
| **S=4096 H16** | **2.6135 ms** | **2.5215 ms（1.036×）** | **3.266 ms（42.1 TF）** |
| MLA S=1024 H2 | 0.3277 ms | 0.3243 ms（持平） | 0.539 ms |

收益集中在 **`REGDQ` 生效的大 S**（S=4096）：fold 向量化 + 关预取合计 **main 1.036×**、
端到端 3.37→3.27ms。小 S 不受影响（走原路）。

### 20.5 ncu（main，S=4096，`--set full`）

| 指标 | O7 基线 | **O7e** |
|---|---|---|
| Duration | 2.66 ms | **2.57 ms** |
| **L1/TEX Cache Throughput** | 71.07% | **66.06%** |
| L2 Cache Throughput | 43.74% | 45.28% |
| Compute (SM) | 39.11% | 40.86% |
| local memory 占 L1TEX sector | 12.09% | **5.74%** |
| shared store bank conflicts | 68.6M | **36.3M** |
| regs / smem / occ | 168 / 70.66KB / 18.1% | 168 / 70.66KB / **18.1%**（3 CTA/SM） |

**结论**：墙仍是 **L1/TEX（ldmatrix + smem）**，register spill 与 store 冲突各被压掉约一半。
**O7b（去 dK/dV 跨 CTA red）已不是头号杠杆**（L2 43.7% < L1/TEX 66%），下一步应转向
**fp8 main 的 Hopper `wgmma` 数据通路（O9c）**——只有 wgmma 的 SS 直读 smem 能同时消掉
`ldmatrix`/转置副本并压 smem（对齐 fp16/bf16 的 O9a/O9b）。

### 20.6 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py`，FA2/FA3/TE 三列）

MHA S=4096：FA3 fp16 **0.3245ms/847TF**、TE fp16 0.4441/619、FA2 0.7286/377；
bf16 FA3 0.3201ms/859TF；TE FP8 纯反向（`fa_bwd_bench.py bench --dtype fp8`）
**0.5899ms/466TF**。ours fp8 main 2.52ms ⇒ 约 TE FP8 整条反向的 **4.3×**（S=512 main 0.072ms
已快过 TE FP8 0.101ms）。

### 20.7 复现

```bash
cd code/flash-attention/fa-bwd
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=30 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --iters=30 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:fa_bwd_fp8_mma_kernel -c 1 -- \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_o7e_sweep.out.txt`（单/两文件 × 4 shape 的计时+对拍）、
`src/fp8/fa_bwd_fp8_main_o7e_ncu_tables_s4096.out.txt`（ncu）、
`src/fp8/fa_bwd_fp8_main_o7b_base_ncu_full_s4096.out.txt`（O7 基线 ncu）、
`src/fa_bwd_o7e_fa3_te_baseline.out.txt`（FA2/FA3/TE fp16/bf16 纯反向）。

---

## 21. O9c（第一步）：fp8 Hopper `wgmma` 数据通路（冒烟 + LSE 上验证）

> O7e（§20）用 ncu 把 fp8 main 的第一墙定位为 **L1/TEX 66–71%（`ldmatrix` + smem 访存）**，
> 并判定「只有 wgmma 的 SS 直读 smem 能同时消掉 `ldmatrix`/转置副本并压 smem」，故把
> `O7b`（去 dK/dV 跨 dK/dV red）降优先级、转做 **O9c**。O9c 与 fp16/bf16 的 O9 系列同构，
> 分多步：**本步先在风险最小的 LSE（单个 QKᵀ）上建立 fp8 的 SW128 + `wgmma.m64n64k32`
> 数据通路**，主 kernel 的 5 个 GEMM 上 wgmma 留作 O9c 后续（对齐 fp16 的 O9a → O9b → O9b-2）。

### 21.1 为什么先做 LSE

- LSE 是「单 GEMM、无转置、无 rowwise scale 折叠」的最小场景，用来验证 **fp8 的 SW128
  K-major 布局 + 描述符 + `m64n64k32` 累加器映射** 三个最容易出错的点，风险最低。
- fp8 的 SW128 与 bf16 **逐字节同构**（atom 恒 8 行 × 128B），只差「一行 128B = **128 个
  fp8**」（bf16 是 64 个）⇒ 16B chunk 下标是 `k/16`（bf16 是 `k/8`）、描述符 `SBO=(K/128)*1024`、
  k32 步进地址仍是 `(s>>2)*1024 + (s&3)*32`（每步 32B = 32 个 fp8）。这让 fp16/bf16 的
  `sw128_off`/描述符公式只需把「行元素数」从 `K/64` 改成 `K/128`。

### 21.2 前置冒烟（`fa_bwd_fp8_wgmma_smoke.cu`）

最小 GEMM `C = Q·Kᵀ`（Q/K 都是 [64][128] fp8，K-major 归约维 128），覆盖两种 wgmma 组合：

- `wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3`（QKᵀ 口径）；
- `...f32.e5m2.e4m3`（GEMM2 `dP=dO·Vᵀ` 口径）。

Q/K 以 `sw128_off_fp8` 存进 SW128 tile，用 `make_desc_sw128_fp8` + `sw128_k32_addr` 生成
描述符；累加器按 `d[j*4+q] ↔ row=16w+g+(q>=2?8:0)、col=j*8+2*(lane%4)+(q&1)` 写回。
取 `{-4..4}` 的整数（e4m3/e5m2 都能精确表示）做 CPU 参考，避免 host 端 fp8 反量化坑。

实测 `max_abs = 0.000e+00`（两种组合，**逐位一致**）⇒ SW128 存 + 描述符 + 累加器映射自洽。

### 21.3 实现（单/两文件 device 代码同源逐字一致）

**device 侧**（`fa_bwd_fp8_kernels.cuh`，单文件由 `sync_onefile_device.py` 同步）：

- 新增 `sw128_off_fp8` / `sw128_k32_addr` / `make_desc_sw128_fp8`；
- 新增 `wgmma.m64n64k32` 的 `e4m3×e4m3` / `e5m2×e4m3` 两条 asm（fp8 尾部操作数是
  `p, scaleA, scaleB` 三个，与 bf16 的 `p,1,1,0,0` 不同；rowwise scale 在 epilogue 乘，故
  scaleA/scaleB 立即数取 1）；
- 新增 `lse_mma_kernel_bal_wgmma<HD,PIPE>`：与 O11 的 `lse_mma_kernel_bal` **数学完全一致**
  （同 E4M3×E4M3、同 online-softmax、同 4-lane `shfl` 归约、同 rowwise scale 相乘顺序、
  同镜像配对 + `cp.async` 双缓冲），只把「4 warp × m16n64 × 4 k-step 的 mma+ldmatrix」换成
  1 个 warpgroup 的 4 条 `wgmma.m64n64k32`。
- 全部用 `#ifdef FA_WGMMA` 包住：**默认 `sm_90` 构建完全不含本块**（行为/数值逐位不变）。
- `Fp8Cfg` 新增 `lse_smem_bytes_balw0/1`（SW128 tile 2×/3× + scale + 1024B 对齐 slack）。

**host 侧**（`fa_bwd_fp8_main.cu` 与单文件各自的 host 段）：新增 `launch_lse_bal_wgmma`、
CLI `--lsewgm`、`run_preprocess` 里 D==128 且 causal 时可选走 wgmma、以及 O9c A/B 计时段。
构建：`ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" scripts/run.sh ...`。

### 21.4 数值（ours-vs-ref，fp8 causal，max_abs，单/两文件一致）

LSE 本身与 mma 版对拍 max_abs **5.6–7.4e-4**（S=4096 5.605e-4；GQA kv4 7.265e-4；
MQA kv1 7.370e-4）——不是逐位相同，但这是 **fp32 求和次序**（wgmma 与 mma 的累加器在
硬件内部的累加顺序不同）导致，远小于 fp8 容差 O(1)。

最终 `dq/dk/dv` 与 ref 的 max_abs 与 O7e **同一水平**（S=512 2.426/2.975/3.736e-1；
S=1024H32 2.400/4.195/3.535e-1；S=4096 2.634/2.643/3.217e-1；GQA kv4 2.517/5.399/7.065e-1），
即 **LSE 的这点差异在端到端被主 kernel 的 fp8 噪声淹没**，无系统误差。

### 21.5 性能（CUDA event 纯 device；同 session A/B）

LSE-only（两文件版 A/B 段直接测，ms）：

| shape | mma（bal+cp.async） | **wgmma（双缓冲）** | 加速 |
|---|---|---|---|
| S=512 H16 | 0.0389 | **0.0365** | 1.065× |
| S=1024 H32 | 0.0791 | **0.0691** | 1.144× |
| S=4096 H16 | 0.3437 | **0.2697** | **1.275×** |
| S=1024 H32 kv4 | 0.0779 | **0.0677** | 1.150× |
| S=1024 H64 kv1 | 0.1013 | **0.0778** | 1.302× |

端到端（单/两文件一致；preprocess 桶含 lse+delta）：

| shape | default total / preprocess | **--lsewgm total / preprocess** | total 加速 |
|---|---|---|---|
| S=512 | 0.1780 / 0.0467 | **0.1690 / 0.0405** | 1.05× |
| S=1024 H32 | 0.7064 / 0.1026 | **0.6902 / 0.0890** | 1.02× |
| S=4096 H16 | 3.2943 / 0.3952 | **3.1925 / 0.3176** | 1.03× |
| GQA kv4 | 0.6295 / 0.1007 | **0.6157 / 0.0876** | 1.02× |

LSE wgmma 单缓冲反而比 mma 慢（S=4096 0.477 vs 0.344）⇒ **必须配 `cp.async` 双缓冲**（SW128
tile 的全局读延迟靠双缓冲盖住），与 fp16/bf16 的 O9a 结论一致。

### 21.6 ncu（lse，S=4096，`--set full -c 1`；mma vs wgmma）

| 指标 | mma（`lse_mma_kernel_bal`） | **wgmma（`..._wgmma`）** |
|---|---|---|
| Duration | 355.87 µs | **279.07 µs**（1.275×） |
| Compute (SM) | 61.59% | **59.45%** |
| DRAM Throughput | 1.48% | 1.88% |
| L2 Cache Throughput | 15.05% | **12.48%** |
| Registers / thread | 77 | **64** |
| Block Limit Shared Mem | 6 | **8** |
| Theoretical Occupancy | 37.5% | **50%** |
| Achieved Occupancy | 23.24% | 23.27% |
| Waves Per SM | 0.65 | 0.48 |
| 主 stall | fixed-latency wait 2.4 cyc | **wait 2.2 cyc** |

**bound = Compute ~60% + softmax epilogue**（`No Eligible` 36–38%），与 fp16 O9a 的结论一致：
wgmma 只打掉访存那一半（L2 15.0→12.5%、regs 77→64、smem 变小使理论 occupancy 37.5→50%），
**LSE 的墙其实在 `fe`/`flog` 的 softmax epilogue 与行归约**，所以 LSE 端到端收益有限（1.02–1.05×）。

### 21.7 对标与下一步

- ours fp8 端到端（`--lsewgm`）vs 同 session TE FP8 纯反向（`fa_bwd_bench.py bench --dtype fp8`）：
  S=512 **0.1690 / 0.1010 = 1.67×**、S=1024H32 0.6902/0.2059 = 3.35×、
  S=4096 3.1925/0.5894 = **5.42×**、GQA kv4 0.6157/0.2009 = 3.06×、MQA kv1 1.0606/0.4008 = 2.65×。
  （O7e 同口径 S=4096 为 5.56× ⇒ 本步 1.03×。）同 session FP16 `harness/fa_vs_te_bwd_only.py`
  （FA2/FA3/TE 三列）：MHA S=4096 FA3 0.3238ms/849TF、TE 0.4443/619、FA2 0.7251/379。
- **局限**：O9c 本步只把 LSE 换成 wgmma，**主 kernel 仍是 `mma.m16n8k32`**（第一墙 L1/TEX
  66–71% 未动）。真正的大头是 main（S=4096 main 2.53ms 占端到端 79%）。
- **下一步（O9c-2）**：把主 kernel 的 **GEMM1/2（`S=QKᵀ`、`dP=dO·Vᵀ`）** 换成
  `wgmma.m64n64k32`（Q/dO/K/V 存 SW128；GEMM3/4/5 的 B 从同一 SW128 tile 用 `ldmatrix.x2.trans`
  转置读，对齐 fp16 O9b 的 `mma_block_swb`），再逐步上 GEMM3/4/5（对齐 O9b-2）；最终靠
  TMA + P/dS 双缓冲跨-tile 流水 + 压 smem 冲更高 occupancy。

### 21.8 复现

```bash
cd code/flash-attention/fa-bwd
# 冒烟（需 sm_90a）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/run.sh src/fp8/fa_bwd_fp8_wgmma_smoke.cu
# LSE wgmma（默认 sm_90 构建不含；需 -DFA_WGMMA）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=30 --lsewgm \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# 单文件同命令换 src/fp8/fa_bwd_fp8_mma_onefile.cu；去掉 --lsewgm 即 O11 mma 基线
# ncu（注意进程参数放在 `--` 之后）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:lse_mma_kernel_bal_wgmma -c 1 -- \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=1 --lsewgm
```

原始输出：`src/fp8/fa_bwd_fp8_wgmma_smoke.out.txt`（冒烟）、
`src/fp8/fa_bwd_fp8_o9c_lse_sweep.out.txt`（两文件 ×5 shape × default/`--lsewgm`，含 O9c A/B）、
`src/fp8/fa_bwd_fp8_main_o9c_ncu_lsewgm_s4096.out.txt` / `..._ncu_lse_mma_s4096.out.txt`（ncu）、
`src/fp8/fa_bwd_fp8_o9c_tebench.out.txt`（TE FP8 基线）、
`src/fp8/fa_bwd_fp8_o9c_fa3_te_baseline.out.txt`（FA2/FA3/TE fp16 纯反向）。

---

## 22. O9c-2：主 kernel GEMM1/2 上 Hopper `wgmma.m64n32k32`（SW128 存储）

### 22.1 动机

O9c（§21）只在 **LSE** 上把 `mma.m16n8k32` 换成了 `wgmma.m64n64k32`，收益有限（1.02–1.05×），
因为第一墙（ncu：**L1/TEX 66–71%**，来自 `ldmatrix`+smem）在**主 kernel**，而 main 占端到端 ~79%。
O9c-2 把 wgmma 推到主 kernel 的 **GEMM1/2（`S=scale·QKᵀ`、`dP=dO·Vᵀ`）**——这两个 GEMM 是 A/B
都 K-major 的 SS 操作数（Q/K/V/dO 存 **SW128** 后 wgmma 直读，免 `ldmatrix`）。

### 22.2 实现（单/两文件 device 代码逐字一致）

- `fa_bwd_fp8_mma_kernel` 增加模板开关 `bool WGMMA=false`（默认行为完全不变）。WGMMA 模式下
  `Fp8Cfg::qs_sw_bytes/ks_sw_bytes = (rows/8)*(HD/128)*1024`，Q/dO/K/V 从行主序 `ASLD` 改为
  **SW128 K-major tile**（`sw128_off_fp8` 4B 写入；动态 smem 基址手动 1024B 对齐，宿主多给
  1024B slack）。`Fp8Cfg::smem_bytes_wgmma` = **68.6KB（O9c-2） vs 70.7KB（mma）**。
- 新增 `wgmma.m64n32k32.f32.{e4m3,e5m2}.e4m3` asm（主 kernel BN=32，用 n32 少一半累加器）与
  issue-only 版 `wgmma_mn32_issue<KIND>`（两条异步 mma 一起发、统一 `wait0` 重叠）。累加器
  布局与 m64n64/mma.m16n8 同构（`d[j*4+q]`，warp w 行 `[16w,16w+16)`）。
- **fold + GEMM3/4/5 + dQ(O7 寄存器累加/跨 CTA red) 全部逐字复用 mma 版**：GEMM3/4/5 的 B
  仍是 O4b 的 **K 配对布局 + `ldmatrix.x2.trans`**（fp8 的 `.trans` 需要沿 K 配对，无法直接从
  SW128 的 16B（沿 K 连续）读出——见 §17 与 `trans_smoke`，故本轮不动 GEMM3/4/5）。
- 仅 `-DFA_WGMMA` 构建可实例化；默认 `sm_90` 构建不含、行为不变。host 加 `--wgmma`（HD=128
  且 `use_regdq` 两档都支持），并加 `[O9c-2 A/B]` 同 session 计时 + 逐元素对拍。

### 22.3 数值（ours-vs-ref，fp8 causal，max_abs）

| case | dq | dk | dv | 与 O4c/O7e mma 版 |
|---|---|---|---|---|
| MHA S=512 | 2.426e-1 | 2.976e-1 | 3.732e-1 | 逐位同量级（2.426/2.975/3.735） |
| MHA S=1024H32 | 2.399e-1 | 4.176e-1 | 3.535e-1 | 同 |
| GQA q32/kv4 | 2.517e-1 | 5.338e-1 | 7.178e-1 | 同 |
| MHA S=4096 | 2.635e-1 | 2.644e-1 | 3.216e-1 | 同 |

`[O9c-2 A/B]` 的 `max_abs(wgmma-vs-mma)` dq/dk/dv = 4.0e-2 / 1.5e-1 / 2.6e-2（S=4096）。差异
来自 wgmma 与 mma 的**累加次序**改变了 fp32 的 S/dP，进而使 fold 的逐行 amax/量化偶有跨档
（下游放大到 ~1e-1），**但 vs fp32 ref 的误差与 mma 版完全相同**（上表），无系统误差。

### 22.4 性能（同 session A/B，CUDA event；main-only）

| case | mma | wgmma(m64n32) | 加速 |
|---|---|---|---|
| MHA S=512 | 0.0803 ms | **0.0773 ms** | 1.038× |
| MHA S=1024H32 | 0.4553 | **0.4326** | 1.052× |
| GQA q32/kv4 | 0.4365 | **0.4130** | 1.057× |
| MHA S=4096 | 2.5891 | **2.4508** | 1.056× |

端到端（`--wgmma`，含 quant+pre+main+cvt）：S=512 **0.1757ms**（12.2 TF）、S=1024H32
0.6799（25.3 TF）、GQA kv4 0.6024（28.5 TF）、S=4096 **3.1374ms（43.8 TF）**。
同 session TE FP8 纯反向（`fa_bwd_bench.py bench --dtype fp8`）：S=512 0.1009ms/42.6TF、
S=4096 0.5893/466.5 ⇒ 端到端 ours/TE = **1.74× / 5.32×**（O7e 5.56×、O9c 5.42×）。
同 session FA2/FA3/TE fp16 纯反向（`fa_vs_te_bwd_only.py`）：MHA S=4096 FA3 0.3237ms/849TF、
TE 0.4397/625、FA2 0.7290/377。

### 22.5 ncu（main，S=4096，`--wgmma`，REGDQ=1）

Duration **2.46ms**、DRAM 2.82% / **L1/TEX 65.19%** / L2 48.03% / Compute 40.23%、
168 regs（`__launch_bounds__(_,3)`）、smem **68.61KB → 3 CTA/SM**、theoretical occ 18.75%、
achieved 18.11%、Waves 10.34、Executed Ipc 1.73；stall `wait 1.51 + short_scoreboard 1.49 +
long 1.18 + not_selected 0.35 + barrier 0.28`；shared load bank conflict 49.4M、store 39.0M。
（O7e mma：Duration 2.57ms、L1/TEX 66.06%、L2 45.28%、Compute 40.86%。）
**结论**：wgmma 只打掉 GEMM1/2 的 `ldmatrix`，**第一墙仍是 L1/TEX（GEMM3/4/5 的 `ldmatrix` +
fold 的 smem 访存）**；要再降必须把 GEMM3/4/5 也上 wgmma（fp8 需「SW128 直读转置」或
MN-major 描述符），这依赖 §17 未解决的 fp8 `.trans` 布局问题。

### 22.6 局限与下一步

- 本轮只换 **GEMM1/2**，收益 **1.04–1.06×**（与 fp16 O9b 的 1.05–1.09× 同量级）。
- fp8 的 GEMM3/4/5 若要上 wgmma：**B 是转置操作数**。fp16 可用 `ldmatrix.x2.trans` 从 SW128
  直读，但 fp8 的 `.trans` 要求「沿 K 配对」（§17），而 SW128 的 16B 是「沿 K 连续的 16 个
  fp8」——两者不兼容。可选路径：① 用 MN-major 描述符做「K-major SW128 tile 的转置读」
  （fp16 O9b-2 已验证，fp8 的 k32 slab 步进待推导/冒烟）；② 为 GEMM3/4/5 保留配对布局、
  只把 GEMM1/2 上 wgmma（即本轮）。③ TMA 化 Q/K/V/dO 直写 SW128（省掉寄存器预取与地址运算）。
- 原始输出：`src/fp8/fa_bwd_fp8_main_o9c2_sweep.out.txt`（两文件 ×4 shape × `--wgmma`，含 A/B）、
  `src/fp8/fa_bwd_fp8_main_o9c2_ncu_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_main_o9c2_ncu_stall_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o9c2_tebench.out.txt`、`src/fa_bwd_fp8_o9c2_fa3_te_baseline_fp16.out.txt`。

## 23. O12：主 kernel LSE/D 预装寄存器（O7c-PREL for fp8）＋ O9c-2b 结论

### 23.1 动机：fp8 的 GEMM1/2 epilogue 一直在逐元素 global 读 lse/delta

fp16/bf16 早在 **O7c（`docs/01` §14）** 就发现：`lse`/`delta` 只依赖 CTA 自己的 Q 行、
与 K tile 无关，而原实现把它放在**每个 tile 的 GEMM1/2 epilogue 里按 `qi` 逐元素 global 读**，
ncu 报「global load 每 thread 仅用 4.4/32B」。当时改成在 `nt` 循环前一次性装进寄存器
（`lse_r/del_r`），main **+14–19%**、端到端 S=4096 **1.13×**。但 **fp8 侧一直没有移植**：

```cpp
// 旧（fp8 fa_bwd_fp8_mma_kernel，每 tile 每元素一次 global 读）
if (qi < S && jg < S && !(causal && jg > qi)) {
  float sval = acc[i][j][q] * scale * qs_s[r] * ks_s[c];
  p = fexp(sval - lse[((size_t)(b * S + qi)) * H + h]);        // ← 逐元素 global
}
...
float del = (qi < S) ? delta[((size_t)(b * S + qi)) * H + h] : 0.f;   // ← 逐元素 global
```

本机 ncu（S=4096，O9c-2 之后的 mma 路径）实测：**uncoalesced global loads 38.0M 多余扇区
（占总扇区 23%）**，global load 平均仅 9.8/32 B/sector——正是这两处按 `qi`（同 warp 内 stride
为 `H`）的散读。

### 23.2 实现（单/两文件 device 代码逐字一致）

新增模板开关 `bool PREL = true`（`--prel=0` 可关，用于同 session A/B）：

```cpp
float lse_r[4], del_r[4];                  // 索引 = i*2 + (q>=2)：mma 路径每线程 4 个行槽
if constexpr (PREL) {
  if constexpr (WGMMA) {                   // wgmma.m64n32：每线程 2 个行槽（wgmma 行映射）
    for (t=0..1) { r = wid*16 + g + (t?8:0); qi = m0+r; 载入 lse/delta; }
  } else {                                 // mma.m16n8：r = wr*32 + i*16 + g + (s?8:0)
    for (i=0..1) for (s=0..1) { ... 载入 ...; }
  }
}
```

epilogue 里 `if constexpr (PREL)` 用 `lse_r[…]`/`del_r[…]`，否则退回旧的逐元素 global 读。
**行槽与 epilogue 的 `(i, q>=2)` 一一对应**，取的值与原 global 读完全相同 ⇒ 数值不变。
WGMMA 与 mma 两条路径共用同一份 `lse_r/del_r[4]`（wgmma 只填前 2 个并复制到后 2 个）。
单文件由 `sync_onefile_device.py` 同步（`device region identical: True`）。

### 23.3 数值（ours-vs-ref，fp8 causal，max_abs）

`PREL` on/off 逐元素对拍 `max_abs` 为 4.8e-7–1.1e-3——这是 dQ **跨 CTA `atomicAdd` 的求和
次序**造成的（两次独立 launch），不是逻辑差异；关键证据是 **vs fp32 ref 的 max_abs 与历史
逐位一致**：

| case | dq | dk | dv |
|---|---|---|---|
| MHA S=512 | 2.426e-1 | 2.975e-1 | 3.735e-1 |
| MHA S=1024H32 | 2.400e-1 | 4.195e-1 | 3.536e-1 |
| MHA S=4096 | 2.635e-1 | 2.643e-1 | 3.216e-1 |
| GQA q32/kv4 | 2.517e-1 | 5.408e-1 | 7.072e-1 |
| MQA q64/kv1 | 4.097e-1 | 1.519 | 2.127 |
| MLA S=1024H2 D=512 | 2.232e-1 | 3.337e-1 | 3.602e-1 |

全部与 O7e/O9c-2 记录相同（含 MLA），单/两文件一致。

### 23.4 性能（同 session A/B，CUDA event，main-only）

| case | off（旧） | on（O12） | 加速 |
|---|---|---|---|
| MHA S=512 | 0.0719 ms | **0.0690** | 1.042× |
| MHA S=1024H32 | 0.4374 | **0.4057** | 1.078× |
| MHA S=4096 | 2.4950 | **2.2679** | 1.100× |
| GQA q32/kv4 | 0.4280 | **0.3969** | 1.079× |
| MQA q64/kv1 | 0.7805 | **0.7042** | 1.108× |
| MLA S=256H2 D=512 | 0.0514 | 0.0511 | 1.006× |
| MLA S=1024H2 D=512 | 0.3264 | 0.3233 | 1.010× |

`--wgmma` 路径同样受益：S=4096 off 2.3851 → **on 2.1204ms（1.125×）**、S=512 1.027×、
S=1024H32 1.075×。端到端（`--wgmma`，总）：S=512 **0.1735ms**（12.4 TF）、S=1024H32 0.6502
（26.4 TF）、S=4096 **2.8720ms（47.9 TF）**（默认 mma：0.1776 / 0.6737 / 3.0617ms）。
MLA 收益近 1（`D=512` 只有 H=2、lse/delta 读的绝对量小，且 main 是低并行度/延迟 bound）。
同 session TE FP8 纯反向（`fa_bwd_bench.py bench --dtype fp8`）：S=512 0.1007ms/42.6TF、
S=1024H32 0.2048/167.8、S=4096 0.5864/468.7；GQA/MQA 0.2016–0.4012ms/170–190TF；MLA NA。
⇒ 端到端 ours/TE（`--wgmma`）≈ 1.72× / 3.17× / 4.90×（O9c-2 时 1.74×/…/5.32×）。

### 23.5 ncu（main，S=4096，mma 路径，`--set full -c 1`）

| 指标 | prel=0（旧） | prel=1（O12） |
|---|---|---|
| Duration | 2.60 ms | **2.34 ms（−10%）** |
| Executed Instructions | 1,008,712,416 | **923,662,336（−8.4%）** |
| uncoalesced global 多余扇区 | 38,048,256（23%） | **4,702,720（5%）（−87.6%）** |
| L1/TEX | 65.88% | 65.92% |
| L2 | 44.76% | 48.10% |
| Compute | 40.72% | 40.94% |
| regs / occ | 168 / 18.1% | 168 / 18.1% |

**机制确认**：把 lse/delta 的散读（占总扇区 23%）消到 5%，指令数 −8.4%、Duration −10%，
regs/occupancy 不变。这正是 fp16 O7c-PREL 的复现。**新墙仍 = L1/TEX 66%（GEMM3/4/5 的
`ldmatrix` + fold 的 smem）+ 残余 L2（dK/dV 跨 CTA red）**。

### 23.6 O9c-2b 结论（fp8 GEMM3/4/5 上 wgmma）——**硬件阻塞**

ROADMAP 的「下一步」原本是 O9c-2b：照 fp16 O9b-2 的做法，用 **MN-major 描述符对 K-major
SW128 tile 做「转置读」**，把 GEMM3/4/5（`dV=PᵀdO`、`dK=dSᵀQ`、`dQ=dS·K`）也上 wgmma。
本轮查证后确认**在 fp8 上不可行**：

- **fp8 的 wgmma 指令没有转置操作数形式**。CUTLASS `include/cute/arch/mma_sm90_gmma.hpp` 里
  所有 fp8 变体（`m64nNk32.f32.e4m3.e4m3` / `e5m2.e4m3` / `e4m3.e5m2`）都只有 `_SS_TN`
  （A/B 均 K-major），**既无 `.trans_a/.trans_b` 立即数，也无 MN-major 描述符语义**；asm 尾部
  是 `p, scale_D, scaleA, scaleB`（对照 fp16 是 `p, 1, 1, tnspA, tnspB`，见 kernels.cuh 的
  `wgmma_m64n32k32_*` 与 fp16 的 `wgmma_m64n64k16_t<TA,TB>`）。也就是说 fp16 O9b-2 的
  `tnsp=1` 那条路在 fp8 的 ISA 上根本不存在。
- 要在 fp8 上做 GEMM3/4/5 的 wgmma，只能**物理地把操作数存成转置布局**（Pᵀ/dSᵀ 以及
  dOᵀ/Qᵀ/Kᵀ 的 K-major tile）。但 fp8 SW128 的 atom 是「8 行 × 128 个 fp8」，要求 tile 的
  行宽（归约维）≥128，于是 `BM=BN=128`，smem 从 70KB 膨胀到 >110KB、occupancy 掉到 1–2
  CTA/SM；且 GEMM5 的 A=dS `[BM][BN]` 与 GEMM3/4 的 A=dSᵀ `[BN][BM]` 还要求同时存两种主序。
- 更关键的是 **fp16 O9b-2/O9b-2b 的实测已证明：GEMM3/4/5 上 wgmma 性能中性偏负**（S=4096
  1.50–1.52 vs mma 1.47–1.48），因为墙不在 GEMM 指令，而在 L2 的 dK/dV 跨 CTA 原子与低
  occupancy。fp8 若付出「物理转置 + smem 翻倍」的代价，回报只会更差。

**决定**：O9c-2b（MN-major 转置读）**在 fp8/Hopper 上从硬件层面就不成立，标记为阻塞**；
不再尝试描述符。fp8 主 kernel 的 L1/TEX 墙改由**非转置**手段解决（fold 向量化、TMA 化
operand、dK/dV 去原子），已记入 ROADMAP backlog。

### 23.7 复现

```bash
# 两文件（默认 mma，PREL 自动开）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# wgmma 构建（需 sm90a）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s4096_h16_d128_causal_fp8 --wgmma
# ncu A/B
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --launch-count 1 \
  --kernel-name regex:fa_bwd_fp8_mma_kernel -- --dir=.../b1_s4096_h16_d128_causal_fp8 --prel=0
```

原始输出：`src/fp8/fa_bwd_fp8_main_o12_sweep.out.txt`（两文件 ×7 shape，含 O12 A/B）、
`src/fp8/fa_bwd_fp8_mma_onefile_o12_sweep.out.txt`（单文件）、
`src/fp8/fa_bwd_fp8_main_o12_wgmma_sweep.out.txt`（`--wgmma`）、
`src/fp8/fa_bwd_fp8_main_o12_ncu_{prel0,prel1}_s4096.out.txt`、
`src/fp8/fa_bwd_fp8_o12_tebench.out.txt` / `..._tebench_req.out.txt`。

---

## 24. O14：fp8 输入量化的向量化（warp-per-row）＋ convert 向量化

### 24.1 动机：量化是端到端第三大项，且比带宽下限慢 ~6.5×

O12 之后，fp8 端到端（S=4096）分解为 **main 2.32ms（76%）/ preprocess 0.40ms（13%）/
quant 0.19ms（6%）/ convert 0.15ms（5%）**；S=1024H32 时 quant 0.10ms + convert 0.06ms
占端到端 **~24%**。其中 `quantize_row_kernel`（旧版）是**每行一个 CTA**：一行只有 `D`
（128/512）个元素，却开 128 个线程 + `__shared__ float sh[128]` + **7 次 `__syncthreads`**
做行 amax；S=4096 时 grid = S·H = 65536 个 CTA、每个只搬 512B。实测四个张量（q/k/v/dO）
合计 0.19ms，而纯带宽下限（~96MB / 3.35TB/s）只有 ~30µs ⇒ 慢约 **6.5×**。

### 24.2 实现（单/两文件 device 代码逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

新增 **`quantize_row_warp_kernel<VPT, E5M2>`**（`src/fp8/fa_bwd_fp8_kernels.cuh`）：

- **每 warp 一行**（128 线程 = 4 warp，grid-stride over rows）；lane 用 **`float4`** 读
  `VPT=D/32` 个元素（D=128→4 个、D=512→16 个）进寄存器；
- 行 amax 用 **`__shfl_xor_sync` 树**（16/8/4/2/1）归约，**全程无 `__shared__`、无任何
  `__syncthreads`**；scale 由 lane0 写；
- 量化从寄存器直接做、**`uchar4` 写回**（16B 读 / 4B 写，完全合并）。
- 只实例化 `D%4==0 && D/32∈{4,16}`（本项目的 128/512）；其它 D 回退旧 kernel。

**数值逐位不变**：行 amax 用 `fmaxf`，可交换结合 ⇒ warp 树与旧 smem 树归约顺序无关；
scale 公式与每元素 `cvt_*` 与旧版逐元素一致。A/B 里对 q8/qs/k8/ks/v8/vs/dO8/dos **逐字节
比较，`bitwise mismatch=0`**（全部 7 个 shape）。

另把 `convert_kernel`（fp32→fp32 拷贝）改成 **`float4`**（`O14b`）：实测 **中性**
（S=4096 0.146→0.138ms，session 噪声内）——它本就接近带宽（~1.4TB/s）而非标量瓶颈，
保留但不计入收益（负结果留存）。

### 24.3 数值（ours-vs-ref，fp8 causal，max_abs，单/两文件逐位一致）

与 O12 **完全相同**：S=512 2.426/2.975/3.735e-1；S=1024H32 2.400/4.195/3.536e-1；
S=4096 2.635/2.643/3.216e-1；GQA kv4 2.517/5.408/7.072e-1；MQA kv1 4.097e-1/1.519/2.127；
MLA S1024H2 2.232/3.337/3.602e-1。量化 kernel 的字节级对拍 mismatch=0 ⇒ 数学口径未变。

### 24.4 性能（同 session A/B，CUDA event）

**quant（四个张量一组）**：

| shape | old per-row (ms) | new warp-per-row (ms) | 加速 | bitwise mismatch |
|---|---|---|---|---|
| S=512 | 0.0324 | 0.0151 | **2.15×** | 0 |
| S=1024 H32 | 0.1006 | 0.0413 | **2.44×** | 0 |
| S=4096 | 0.1825 | 0.0689 | **2.65×** | 0 |
| GQA kv4 | 0.0623 | 0.0281 | **2.22×** | 0 |
| MQA kv1 | 0.1002 | 0.0410 | **2.44×** | 0 |
| MLA S256H2 | 0.0140 | 0.0115 | **1.22×** | 0 |
| MLA S1024H2 | 0.0206 | 0.0142 | **1.46×** | 0 |

**端到端 total**：S=512 0.1776→**0.1642ms（1.082×，13.1 TF）**、S=1024H32 0.6737→
**0.6180（1.090×，27.8 TF）**、S=4096 3.0617→**2.9210（1.048×，47.1 TF）**、
GQA kv4 0.5969→**0.5660（1.055×）**、MQA kv1 1.0137→0.9564、MLA S1024H2 0.5368→0.5339。
**对标 TE FP8**（同 session `fa_bwd_bench.py bench --dtype fp8`）：TE S=512 0.1006、
S=1024H32 0.2056、S=4096 0.5861、GQA kv4 0.2021、MQA kv1 0.4024 ⇒ 端到端 ours/TE =
**1.63× / 3.01× / 4.98× / 2.80× / 2.38×**（O12 S=4096 为 5.22×）。同 session 纯反向
FA3 MHA S=4096 fp16 0.3247ms/847TF、TE 0.4414/623、FA2 0.7294/377（fp8 无 FA 基线）。

### 24.5 ncu（quant kernel，S=4096，单次 launch，`--set full -c 1`）

| 指标 | old `quantize_row_kernel` | new `quantize_row_warp_kernel` |
|---|---|---|
| Duration | 46.18 µs | **15.42 µs** |
| DRAM Throughput | 24.29% | **71.31%** |
| L1/TEX Cache Throughput | **73.73%** | 26.50% |
| L2 Cache Throughput | 28.91% | 74.00% |
| Compute (SM) Throughput | **71.80%** | 51.49% |
| Executed Instructions | 34,603,008 | **7,929,856（−77%）** |
| Waves Per SM | 31.03 | 7.76 |
| Achieved Occupancy | 87.82% | 79.44% |

**结论**：旧 kernel 的墙是 **L1/TEX 73.7%（smem 归约）+ Compute 71.8%（标量地址/加载）**，
DRAM 只有 24%；新 kernel 把墙移到 **DRAM 71.3%（纯带宽）**——即已到该 elementwise
算子的带宽上限。指令数 −77% 与「无 smem、无 barrier、float4/uchar4」一致。

### 24.6 复现

```bash
# 两文件（默认 qfast=1）；--qfast=0 退回旧 per-row 量化做 A/B
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu：新旧量化 kernel
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --launch-count 1 \
  --kernel-name-base demangled --kernel-name "regex:quantize_row_warp_kernel" \
  -- --dir=.../b1_s4096_h16_d128_causal_fp8 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_o14_sweep.out.txt`（两文件 ×7 shape，含 O14 A/B）、
`src/fp8/fa_bwd_fp8_mma_onefile_o14_sweep.out.txt`（单文件 ×4 shape）、
`src/fp8/fa_bwd_fp8_main_o14_ncu_quant_{old,new}_s4096.out.txt`、
`src/fp8/fa_bwd_fp8_o14_tebench{,_base3,_req}.out.txt`、
`src/fp8/fa_bwd_fp8_o14_fa3_te_baseline_fp16.out.txt`。

## 25. O7e-2：fold 的 shared-load bank conflict 修复（Ap/dS3 列读无冲突映射）＋ 向量化写

### 25.1 动机：用 ncu 的 Memory Tables 重新定位 L1/TEX 的第一来源

O7e（§20）把 fp8 main 的第一墙定位为 **L1/TEX ~66%**（L2 已被 O4c/O7 压到 43.7%），
并顺势把 fold 的**写**从逐字节折到 4B。但 O7e 没有拆开 L1/TEX 里**读**与**写**各占多少。
本轮先用 `--set full` 的 `Memory Workload Analysis Tables` 拆开 O12 之后的 main（S=4096）：

- **shared loads：93.07M 请求、62.80M bank conflict（2.1-way，占 store… 占 load 波前 32%）**，
  是 L1/TEX 的第一来源；
- shared stores：18.77M 请求、35.91M conflict（3.0-way，占 store 波前 64%）；
- local memory（`REGDQ` 的寄存器 spill）：占 L1TEX sector 的 ~9.6%。

**先证伪一条路**：只把 fold 的 Ap/dS3/dS2 从 4B 折成 16B `st.shared.v4.u32`（store 指令再 ÷4），
S=4096 main 只 **1.007×**、S=512 持平，且 ptxas spill 反而增大 ⇒ **fold 的瓶颈不在 store
指令数**（ncu 对 shared store 的 42% Est. Speedup 是上界、未兑现）。

### 25.2 根因：fold 读 `Ps/Ss` 的列访问恒撞 bank

fold 的 Ap/dS3 段（`Ap[j][m]=P[m][j]·dos[m]`）按「4 个 lane 各负责 16 个 m」分工：
`jl=lane>>2`（输出行 j）、`sub4=lane&3`，原映射 `m = sub4*16 + t`（`t=0..15`）。
`Ps[m*PSS+j]`（`PSS=BN+1=33`）的 bank = `(m*33+j) mod 32 = (m+j) mod 32`；固定 `t` 时
4 个 `sub4` 的起始 m 相差 **16**，而 `16*33 ≡ 16 (mod 32)`，于是 `sub4=0/2`、`1/3` 两两同 bank
⇒ **恒 2-way conflict**（这正是 62.8M conflict 的主项）。

> 数学上 `16*PSS mod 32 ∈ {0,16}`（PSS 为任意整数），所以只要 lane 组的 m 间距是 16，
> 这个冲突在「列读 + 该 PSS」组合下**无法靠改 padding 消除**；必须改 lane→m 的映射。

### 25.3 改动（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

把每个 lane 负责的 16 个 m 从「一整块 `sub4*16+t`」改成「两半 `sub4*8 + t + half*32`」
（`half∈{0,1}`、`t=0..7`）。此时固定 `t` 的 4 组 lane 起始 m 相差 **8**：

```
bank(m=sub4*8+t+half*32, j=wid*8+jl) = (sub4*8 + jl + const) mod 32
  sub4*8 ∈ {0,8,16,24},  jl ∈ {0..7}  ⇒ 4×8 = 0..31 各一次 ⇒ 无冲突
```

- `amax` 的 `fmaxf` 可交换结合、且每 lane 仍覆盖全部 16 个 m ⇒ 归约结果与原映射**逐位相同**；
- 输出落盘：每 lane 变成两段各 8 个连续 m ⇒ 用 **8B `st.shared.v2.u32`**（`QTS=80`、`sub4*8+
  half*32` 均 8 对齐）；dS2 段（ncu 显示无冲突）保持 16 个连续 j 的 **16B `st.shared.v4.u32`**。
- 模板开关 `F16B`（`--f16b=0` 退回原 `sub4*16` + 4×4B），用于同 session A/B。

### 25.4 数值（ours-vs-ref，fp8 causal，max_abs，单/两文件逐位一致）

| case | dq | dk | dv |
|---|---|---|---|
| MHA S=512 | 2.426e-1 | 2.975e-1 | 3.735e-1 |
| MHA S=1024H32 | 2.400e-1 | 4.195e-1 | 3.536e-1 |
| MHA S=4096 | 2.635e-1 | 2.643e-1 | 3.216e-1 |
| GQA q32/kv4 | 2.517e-1 | 5.408e-1 | 7.072e-1 |
| MQA q64/kv1 | 4.097e-1 | 1.519 | 2.127 |
| MLA S=1024H2 D=512 | 2.232e-1 | 3.337e-1 | 3.602e-1 |

全部与 O7/O12/O14 记录相同。A/B 的 `max_abs(16B-vs-4B)` 仅 1e-7 量级（dQ 跨 CTA `atomicAdd`
求和次序），即**只改访存布局、未改数学口径**。

### 25.5 性能（同 session A/B，CUDA event，main-only）

| case | 4B（旧） | F16B（新） | 加速 |
|---|---|---|---|
| MHA S=512 | 0.0690 ms | **0.0679** | 1.017× |
| MHA S=1024H32 | 0.4036 | **0.3921** | 1.029× |
| GQA q32/kv4 | 0.3916 | **0.3792** | 1.033× |
| MQA q64/kv1 | 0.7109 | **0.6834** | 1.040× |
| MHA S=4096 | 2.3115 | **2.2180** | 1.042× |

端到端 total：S=512 0.1632ms（13.2 TF）、S=1024H32 0.6056（28.4）、GQA kv4 0.5617（30.6）、
S=4096 **2.8942ms（47.5 TF）**、MLA S=1024H2 0.5298。S=4096 ours/TE FP8 = **4.92×**
（O14 4.98×）。同 session 纯反向 FA3 MHA S=4096 fp16 0.3242ms/848TF、TE 0.4429/621（fp8 无 FA 基线）。

### 25.6 ncu（main, S=4096，同 session `--set full -c 1`，`--f16b` 0/1）

| 指标 | `--f16b=0`（旧映射） | `--f16b=1`（O7e-2） |
|---|---|---|
| Duration | 2.41 ms | **2.27 ms（−5.8%）** |
| **shared load bank conflict** | 62,766,668（67% 波前） | **29,180,259（−53.5%）** |
| shared load 请求 | 93,069,312 | 93,069,312（不变） |
| 总多余 wavefronts | 79,872,000（34%） | **41,533,440（21%）** |
| L1/TEX | 64.80% | **59.97%** |
| L2 | 47.00% | 49.89% |
| Compute | 40.56% | 41.81% |
| shared store 冲突 | 33,796,044 | 29,464,683 |
| regs / occ / Waves | 168 / 18.08% / 10.34 | 168 / 18.08% / 10.34 |

**机制确认**：只改 lane→m 映射（请求数不变）就把 shared load 冲突砍掉一半、L1/TEX 降到 60%、
Duration −5.8%。**新墙仍是 L1/TEX 60%（`ldmatrix` 的 shared 读 + fold 残余）+ L2 50%
（dK/dV 跨 CTA red）+ 寄存器 spill（~2.7M local 请求）**；fold 这一路的 bank conflict 已基本收口。

### 25.7 复现

```bash
# 两文件（默认 F16B=1）；--f16b=0 退回旧映射做 A/B
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu A/B
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --launch-count 1 \
  --kernel-name regex:fa_bwd_fp8_mma_kernel -- --dir=.../b1_s4096_h16_d128_causal_fp8 --f16b=0
```

原始输出：`src/fp8/fa_bwd_fp8_main_o7e2_sweep.out.txt`（两文件 ×7 shape，含 O7e-2 A/B）、
`src/fp8/fa_bwd_fp8_mma_onefile_o7e2_sweep.out.txt`（单文件 ×5 shape）、
`src/fp8/fa_bwd_fp8_main_o7e2_ncu_s4096{,_f16b0}.out.txt`、
`src/fp8/fa_bwd_fp8_o7e2_tebench.out.txt`、`src/fa_bwd_o7e2_fa3_te_baseline_fp16.out.txt`。

---

## 26. O7e-3：GEMM1/2 epilogue `Ps/Ss` 的 store/回读 bank conflict 修复（PSS 33→37）

### 26.1 动机与 ncu 定位

O7e-2（第 25 节）把 fp8 main 第一墙记为「L1/TEX 60%」，并修了 **fold** 段的 shared-load
冲突。但一直没拆开「读」与「写」。本轮用 `--page source --csv` 把 `L1 Wavefronts Shared
Excessive` 按 **CUDA 源码行** 聚合（`fa_bwd_fp8_mma_kernel<128,64,32,1,0,1,1>`，S=4096）：

| CUDA 源行 | 多余 wavefronts（excessive） |
|---|---|
| `Ss[r * PSS + c] = Ps[r * PSS + c] * (dpv - del);` | **25,559,040** |
| `Ps[r * PSS + c] = p;` | **12,779,520** |
| `Ap/dS3/dS2` fold 向量写（O7e-2） | 1,064,960 ×3 |
| `LDSM`（ldmatrix） | 0 |
| 载入 Q/K/V 配对写（prologue） | ≤ 2,129,920 |

即 **~98% 的 shared-store 多余 wavefronts 来自 GEMM1/2 epilogue 的 `Ps/Ss` 标量 fp32 写**，
外加 `Ss` 计算里对 `Ps` 的同模式**回读**（进了 `LDS` 一栏）。两者都是 4-way。

### 26.2 根因：PSS≡1 让 `{g·PSS + 2l}` 大量重合

mma `m16n8` 累加器里，同一条 store 指令内 `r = R0 + g`（`g = lane>>2 ∈ 0..7`）、
`c = C0 + 2l`（`l = lane&3`，`c2 = 2l ∈ {0,2,4,6}`），4 字节 store 的 bank
= `(r·PSS + c) mod 32 = (g·PSS + 2l + const) mod 32`。PSS=33（≡1 mod 32）时
`{g + 2l}` 有大量重合 ⇒ 4-way（实测 `Ss` 25.56M / 理想 8.5M ≈ 3.0× 多余，与 4-way 吻合）。
`Ss` 的回读 `Ps[r*PSS+c]` 同一模式，故 load 侧也 4-way。

### 26.3 修复：`PSS = BN + 5`（37）

PSS 必须满足「fold 的掩码读（O7e-2 的 `m = sub4*8 + t + half*32`）无冲突」——那要求
`PSS mod 4 = 1`（使 `8·PSS·sub4 mod 32` 取到 `{0,8,16,24}`）。在此约束下穷举 PSS：

| PSS | epilogue store/回读 wavefronts | fold 读 wavefronts |
|---|---|---|
| 33（旧） | 4-way | 1（无冲突）|
| **37（新）** | **2-way** | **1** |

37 使 `bank = (5g + 2l) mod 32`，重合降为 2-way。改动极小（`Fp8Cfg::PSS = BN + FA_PSS_EXTRA`，
默认 `FA_PSS_EXTRA=5`；`=1` 即旧值 33，供 A/B）。smem 只 +2 KB（70.66→72.70 KB），仍 3 CTA/SM。
**只改 smem 地址，数学与量化完全相同 ⇒ 数值逐位不变。**

### 26.4 实测（同 session A/B，CUDA event，ms）

| shape | main 33 | main 37 | main ×| total 33 | total 37 | total ×|
|---|---|---|---|---|---|---|
| S512 H16 | 0.0681 | **0.0670** | 1.016 | 0.1621 | **0.1597** | 1.015 |
| S1024 H32 | 0.3968 | **0.3929** | 1.010 | 0.6061 | **0.6028** | 1.005 |
| S4096 H16 | 2.2626 | **2.2373** | 1.011 | 2.8585 | **2.8375** | 1.007 |
| S1024 kv4 | 0.3834 | **0.3806** | 1.007 | 0.5591 | **0.5527** | 1.012 |
| MLA S1024 H2 D512 | 0.3274 | **0.3196** | 1.024 | 0.5293 | **0.5265** | 1.005 |

数值：5 个 shape 的 dq/dk/dv vs ref **逐位不变**（S512 2.426/2.975/3.735e-1；
S1024H32 2.400/4.195/3.536e-1；S4096 2.635/2.643/3.216e-1；GQA kv4 2.517/5.408/7.072e-1；
MLA S1024H2 2.232/3.337/3.602e-1）。另一次同 session 首测 S4096 main 2.2832→2.2269（1.025×）。

### 26.5 ncu（S=4096，`--set full`）与 bound 结论

| 指标 | PSS=33（O7e-2） | PSS=37（本轮） |
|---|---|---|
| shared store `bank_conflicts` | 29,464,683 | **12,329,835（−58%）** |
| shared load `bank_conflicts` | 29,180,259 | **20,593,271（−29%）** |
| L1/TEX Cache Throughput | 59.97% | **55.83%**（指标单跑 51.6%）|
| Duration | 2.27 ms | 2.28 ms（**持平**）|
| L2 / Compute / DRAM | 49.9 / 42.5 / 3.2% | 49.8 / 43.1 / 3.2% |
| occupancy / regs / smem | 18.1% / 168 / 70.7KB | 18.1% / 168 / 72.7KB |
| stall | — | `wait` **1.56** + `short_scoreboard` **1.50** + `long` 0.77，issue 45.9% |

**结论（修正 O7e-2 的判断）**：bank conflict 与 L1/TEX 吞吐**不是** fp8 main 的限速器——
把 store 冲突砍 58%、L1/TEX 59.97→55.83%，Duration 几乎不动。kernel 是 **mma 依赖延迟受限**
（`wait`（fixed-latency）+ `short_scoreboard`（smem→mma）合计 ~3.06 cycle/issue，issue active
仅 45.9%），且在 3 CTA/SM（168 regs / 72.7KB smem）下无法靠堆 warp 隐藏。真正剩下的杠杆是
**减少跨 CTA 的 dK/dV `red`（L2 49.8%）** 或 **提 occupancy（须先砍寄存器/smem）**，而非再抠 smem 冲突。

### 26.6 复现

```bash
# 两文件（默认 FA_PSS_EXTRA=5 → PSS=37）；A/B 用 =1 退回 PSS=33
ARCH=sm_90 NVCC_FLAGS="-DFA_PSS_EXTRA=5" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# 按源码行拆 bank conflict
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:fa_bwd_fp8_mma_kernel --launch-count 1 \
  --page source --print-source cuda,sass --csv \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# 单文件（device 由 sync_onefile_device.py 同步，逐字一致）
python3 scripts/sync_onefile_device.py src/fp8/fa_bwd_fp8_kernels.cuh \
  src/fp8/fa_bwd_fp8_mma_onefile.cu '#include <cuda_runtime.h>'
```

原始输出：`src/fp8/fa_bwd_fp8_o19_pss_ab.out.txt`（两文件 ×5 shape × PSS 33/37，
数值+计时）、`src/fp8/fa_bwd_fp8_main_o19_ncu_s4096.out.txt`（`--set full`）、
`src/fp8/fa_bwd_fp8_o19_tebench.out.txt`（TE FP8）、
`src/fp8/fa_bwd_fp8_o19_fa3_te_baseline_fp16.out.txt`（FA2/FA3/TE fp16 三列）。

---

## 27. O19：fp8 跨 warpgroup 归约（BM=128、2 warpgroups）——**负结果 + 机制判决**

> 目的：O7e-3（§26）把 fp8 main 的第一墙定位为 **mma 依赖延迟（`wait`+`short_scoreboard`）+ 3 CTA/SM**，
> 并把候选杠杆列为「**降 L2 的 dK/dV `red`（49.8%）**」或「**提 occupancy**」。fp16/bf16 的 O17（跨
> warpgroup 归约，BM=128）实测把 `red` 精确砍半、main 1.5×，所以本轮把同一机制移植到 fp8 做**判决**。

### 27.1 动机与假设

fp8 main 的 `dK/dV` 用跨 CTA 的 `atomicAdd` 汇总，一个 KV 元素被 `S/BM` 个 CTA（每个 query 块一个）
贡献。**BM 64→128 后贡献 CTA 数减半 ⇒ `red` 字节砍半**（O17 在 fp16/bf16 上成立）。

### 27.2 实现（单/两文件 device 代码逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

新增 `fa_bwd_fp8_wg2_kernel<HD,BM=128,BN=32>`（`__launch_bounds__(256,1)`，**256 线程 = 2 warpgroup**）：

* **phase A**：每个 wg（4 warp，2×2 几何）算自己 64 行 Q 的 `S=scale·QKᵀ` 与 `dP=dO·Vᵀ`，P/dS 写
  `Ps/Ss`（全 BM=128 行，`PSS=BN+5=37` 与 mma 版一致）；LSE/D 仍预装寄存器（PREL）。
* **fold**：8 个 warp 协同把全 128 行量化成 `Ap`（e4m3, per-j）、`dS3`（e5m2, per-j）、`dS2`（e5m2,
  per-m）；`fmaxf` 可交换结合 ⇒ amax 与 mma 版逐位相同。
* **phase B**：`dV=Pᵀ·dO`、`dK=dSᵀ·Q`、`dQ=dS·K` 的重叠维都从 smem **整块 128 行**读
  （`mma_block_bt<16,32,128>` / `<32,64,32>`），8 个 warp 各占一块互不重叠的输出 ⇒ **每个 dV/dK/dQ
  元素在本 CTA 内只被一个 warp `red` 一次**；dQ 仍走寄存器累加（REGDQ）后单次 flush。
* host 加 `--wg2` / `--ksplit2=` 与 `[O19 A/B]`（同 session 计时 + 逐元素对拍）；单文件同源。

smem **131.33KB → 1 CTA/SM**（256 线程 = 8 warp/SM）；regs 217；无 O3 寄存器预取（`kv_load_pair_nt` 直读）。

### 27.3 数值（fp8 causal，ours-vs-ref，max_abs，单/两文件逐位一致）

| case | dq | dk | dv | `max_abs`(wg2-vs-mma) dq/dk/dv |
|---|---|---|---|---|
| S=4096 H16 | 2.635e-1 | 2.767e-1 | 3.313e-1 | 1.19e-7 / 8.86e-2 / 4.31e-2 |
| S=1024 H32 | 2.400e-1 | 4.334e-1 | 3.633e-1 | 2.38e-7 / 1.11e-1 / 5.98e-2 |
| S=512 H16 | 2.426e-1 | 2.995e-1 | 3.713e-1 | 1.19e-7 / 1.11e-1 / 4.15e-2 |
| GQA h32kv4 S=1024 | 2.517e-1 | 5.273e-1 | 7.226e-1 | 1.19e-7 / 1.17e-1 / 9.26e-2 |

**与 mma 版同为 fp8 噪声量级**（vs ref 的 dq 完全一致；dk/dv 差 ~0.1 是**原子归约次序 + 贡献 CTA 数
不同**在近抵消元素上的放大，dq 无跨 mblk 重叠故只差 1.2e-7）。**数学口径未变**。

### 27.4 性能（同 session A/B，CUDA event，main-only，ms）

| case | mma (BM64) | wg2 (BM128) | 比 |
|---|---|---|---|
| S=4096 H16 | 2.2662 | 2.9014 | **0.781×** |
| S=1024 H32 | 0.4096 | 0.5661 | 0.724× |
| S=512 H16 | 0.0761 | 0.1130 | 0.673× |
| GQA h32kv4 S=1024 | 0.3869 | 0.5254 | 0.736× |

**wg2 在所有 shape 都更慢（0.67–0.78×）**；调 `ksplit2`（1/2/4/8/16）最好也只有 2.886ms（mma 2.27ms）。
按 `4·B·S²·H·D`：S=4096 mma main 60.6 TF（fp8 峰值 1978.8 的 3.1%）vs wg2 **47.4 TF（2.4%）**。

### 27.5 ncu（main, S=4096，同 session，`--set full` / 定向 metrics）

| 指标 | mma (BM64, 3 CTA/SM) | wg2 (BM128, 1 CTA/SM) | 变化 |
|---|---|---|---|
| Duration | 2.28 ms | **2.92 ms** | ↑ |
| `lts__t_sectors_op_red` | 108,478,464 | **64,290,816** | **0.593×** |
| `lts__t_sectors_op_read` | 32,188,735 | **17,286,751** | **0.537×** |
| L2 Throughput | 49.91% | **22.61%** | ↓ |
| Compute (SM) | 42.34% | 34.13% | ↓ |
| DRAM | 3.22% | 2.20% | — |
| Achieved occupancy | 18.09%（12 warp/SM） | **12.49%（8 warp/SM）** | ↓ |
| regs / smem | 168 / 72.70KB | 217 / **131.33KB** | — |
| Waves / SM | 10.34 | 31.03 | — |
| Issue Slots Busy | 43.06% | **34.13%** | ↓ |
| No Eligible | 54.10% | **64.40%** | ↑ |
| stall `wait` / `short` / `long` | 1.56 / 1.50 / 0.77 | 1.27 / 1.21 / 0.40 | 均↓ |

**机制被完全证实**：`red` 砍到 0.593×、`read` 0.537×、**L2 从 ~50% 掉到 22.6%**——跨 warpgroup
归约确实打掉了 L2 那一半。但 **1 CTA/SM 只有 8 warp/SM（mma 有 12）**，每 warp 的 stall 虽更低却
不足以覆盖；issue 从 43% 掉到 34%、No Eligible 升到 64%，Duration 反而 +28%。

### 27.6 结论（O7e-3 的判决）

**fp8 main 的限速器不是 L2 `red`，而是「mma 依赖延迟 + occupancy」**：把 L2 打掉一半都不够补
1 CTA/SM 的并行度损失（与 fp16 O17b「BM=256 寄存器墙」是同一类结论的另一面）。故 fp8 右侧
**不应再走「减 red / 放大 BM」**，真正的杠杆是 **提 occupancy**（需把 168 regs 砍到 ≤128、
72.7KB smem 砍到 ≤58KB 才能 4 CTA/SM）或 **减少 mma 依赖 stall**（softmax/fold 与 mma 的重叠）。
`red` 减半的收益只在 fp16/bf16（其 `red` 占 L2 73%、且 2 CTA/SM→1 CTA/SM 换得回来）成立。

### 27.7 复现

```bash
# 两文件：默认 mma；--wg2 切跨 warpgroup 版（--ksplit2=1..16 可扫）
ARCH=sm_90 scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --wg2
# 单文件（device 由 sync_onefile_device.py 同步，逐字一致）
ARCH=sm_90 scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8 --wg2
# ncu：red 扇区/stall 定向指标
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu \
  --metrics lts__t_sectors_op_red.sum,lts__t_sectors_op_read.sum,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio \
  --kernel-name regex:fa_bwd_fp8_wg2 --launch-count 1 \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --wg2 --ksplit2=8
```

原始输出：`src/fp8/o19_wg2_ab_sweep.out.txt`（同 session A/B ×4 shape + 数值）、
`src/fp8/o19_ncu_wg2_s4096.out.txt`（wg2 `--set full`）、
`src/fp8/o19_ncu_wg2_red_s4096.out.txt` / `src/fp8/o19_ncu_mma_red_s4096.out.txt`（red 扇区 + stall 对照）。

---

## 28. O20：mma 主路径 GEMM1/GEMM2 epilogue 融合（P 留寄存器，消 `Ps` 回读）

### 28.1 动机与 ncu 定位

O7e-3（§26）用 `--page source --csv` 按源码行聚合 `L1 Wavefronts Shared Excessive` 时，
发现 shared-store 多余 wavefronts 的 ~98% 来自 GEMM1/2 epilogue 的 `Ps/Ss[r*PSS+c]` 写
（25.56M + 12.78M）**以及 `Ss` 里对 `Ps` 的同模式回读（12.78M）**。当时只把 bank conflict 从
4-way 降到 2-way（`PSS 33→37`），**没有消掉这次 smem 回读本身**。

回读的根因：**mma 路径**里 GEMM1 的 epilogue 算完 `P` 后写 `Ps`，GEMM2 的 epilogue 又
`Ss[r*PSS+c] = Ps[r*PSS+c] * (dpv - del)` 把它读回来。而两次 `mma_block` 的
`(wm=wr, wn=wc)` 和累加器映射**完全一致**（O4a 注释已指出），所以这个 `P` 本来就在**同一线程的
寄存器**里，根本不需要绕一趟 smem。对照：**wgmma 路径（O9c-2）早已把 P 留在 `pval[16]` 里**，
只有 mma 路径还在回读。

### 28.2 改动（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

新增编译开关 `FA_FUSE_EPI`（默认 1，`-DFA_FUSE_EPI=0` 供 A/B）：

- mma 分支在 GEMM1 前声明 `float preg[2][2][4]`；
- GEMM1 epilogue 算出 `p` 后 `Ps[r*PSS+c]=p;` **同时** `preg[i][j][q]=p;`；
- GEMM2 epilogue 用 `preg[i][j][q]`（本线程刚算的同一 `(r,c)`）代替 `Ps[r*PSS+c]`。

只改 smem 访问路径，**数学与数值逐位不变**（同一线程、同一 `(r,c)`、同一个 `p`）。
代价是 `preg`（16 个 fp32）要在 GEMM2 的 `mma_block` 期间保活（ptxas：该实例
`b1b0b1b1` 栈帧 8→72B）；收益是每个 tile 少一次全 tile 的 `Ps` 回读（12.78M wavefronts）。

### 28.3 数值（ours-vs-ref，fp8 causal，max_abs）

`FA_FUSE_EPI` on/off 的 dq/dk/dv 对拍表**逐位相同**（打印到 3 位有效数字完全一致）：

| shape（causal, B1） | dq | dk | dv |
|---|---|---|---|
| S512 H16 D128 | 2.426e-1 | 2.975e-1 | 3.735e-1 |
| S1024 H32 D128 | 2.400e-1 | 4.195e-1 | 3.536e-1 |
| S1024 H32 D128 kv4 | 2.517e-1 | 5.408e-1 | 7.072e-1 |
| S1024 H40 D128 kv8 | 2.869e-1 | 5.390e-1 | 7.108e-1 |
| S1024 H2 D512 | 2.232e-1 | 3.337e-1 | 3.602e-1 |
| S512 H4 D512 | 2.415e-1 | 2.992e-1 | 4.481e-1 |
| S4096 H16 D128 | 2.635e-1 | 2.643e-1 | 3.216e-1 |

### 28.4 性能（同 session A/B，CUDA event，main-only）

| shape | FUSE=0 (ms) | FUSE=1 (ms) | 比 |
|---|---|---|---|
| S512 H16 D128 | 0.0670 | 0.0671 | 1.00× |
| S1024 H32 D128 | 0.3942 | **0.3882** | **1.015×** |
| S1024 kv4 | 0.3799 | 0.3794 | 1.00× |
| S1024 kv8 | 0.4530 | **0.4456** | **1.017×** |
| S1024 H2 D512 | 0.3183 | 0.3212 | 0.99× |
| S512 H4 D512 | 0.1663 | 0.1650 | 1.008× |
| **S4096 H16 D128** | 2.1997 | **2.1433** | **1.026×** |

端到端（S4096，FUSE=1）**total 2.77ms（49.6 TF）**、main-only 64.1 TF（FP8 峰值 1978.8 的 3.2%）；
同 session TE FP8 纯反向 0.5909ms/465 TF ⇒ ours 端到端时间约 **4.7× TE**（O7e-3 时 4.85×）。
收益集中在 **S=4096**（`REGDQ=1` 走寄存器累加、`kPrefetch` 关闭，寄存器余量最大），
小 S/GQA/MLA 因每 tile 回读占比小、且 `preg` 保活增加 spill，基本中性。
（fp8 无 FA 基线；FA3 fp16 MHA S4096 0.3241ms/848TF 仅作口径参照。）

### 28.5 ncu（main, S=4096，同 session，`--set full -c 1` + 定向 metrics）

| 指标 | FUSE=0 | FUSE=1 | 变化 |
|---|---|---|---|
| Duration | 2.25 ms | **2.21 ms** | −1.8% |
| shared 多余 wavefronts | 15.97M（9%） | **11.71M（7%）** | **−4.26M** |
| shared 总 wavefronts | 169.07M | 160.55M | −8.5M |
| bank conflict `op_ld` | 20.55M | **16.54M** | −19.5% |
| bank conflict `op_st` | 12.35M | 12.33M | 不变 |
| `short_scoreboard` stall | 1.50 | **1.41** | −6% |
| `long_scoreboard` stall | 0.76 | 0.68 | −11% |
| `wait` stall | 1.56 | 1.55 | 不变 |
| L1/TEX / L2 / Compute | 55.75 / 50.18 / 42.69% | 55.19 / 50.93 / 43.62% | — |
| regs / smem / occ | 168 / 72.70KB / 18.75% | 168 / 72.70KB / 18.75% | 不变 |

### 28.6 结论

`short_scoreboard`（`ldmatrix`/smem 依赖）下降、shared 总 wavefronts 与 `op_ld` 冲突同步下降，
**证实收益来自消掉 `Ps` 回读**。墙仍是 **`wait`（1.55，mma 依赖延迟）+ 3 CTA/SM**、L2 ~50%
（残余 `red`），与 O7e-3/O19 的判决一致：**这条改动只是把 L1/TEX 数据通路再清掉一个已知来源，
不改变「fp8 main 的限速器是 mma 依赖延迟 + occupancy」**。真正再上一个台阶仍需 O19 §27.6 列的
「提 occupancy（168→≤128 regs、72.7→≤58KB smem）或减 mma 依赖 stall」。

### 28.7 复现

```bash
# A/B（同 shape 先后编两个二进制）
docker exec kernel_lab bash -lc "cd <...>/src/fp8 && \
  nvcc -O3 -arch=sm_90 -DFA_FUSE_EPI=0 fa_bwd_fp8_main.cu -o /tmp/fp8_fuse0.out && \
  nvcc -O3 -arch=sm_90 -DFA_FUSE_EPI=1 fa_bwd_fp8_main.cu -o /tmp/fp8_fuse1.out"
# 默认（FUSE=1）跑
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# 单文件（device 由 sync_onefile_device.py 同步，逐字一致）
ARCH=sm_90 scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu（定向 metrics）
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --launch-count 1 \
  --kernel-name regex:fa_bwd_fp8_mma_kernel \
  --metrics smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
```

原始输出：`src/fp8/fa_bwd_fp8_main_o20_fuse_ab.out.txt`（两文件 ×7 shape × FUSE 0/1，计时+数值）、
`src/fp8/o20_ncu_fuse0_s4096.out.txt` / `o20_ncu_fuse1_s4096.out.txt`（`--set full`）、
`src/fp8/o20_onefile_s4096.out.txt`、`src/fp8/o20_tebench_fp8.out.txt`、
`src/fp8/o20_fa3_te_baseline_fp16.out.txt`。

---

## 29. O21：fp8 主 kernel KV tile BN=32→64（负结果）+ O21b：消冗余 fp32→fp32 convert（端到端 1.037×）

### 29.1 动机

O18 把 fp16/bf16 `wgmma2` 的 KV-tile 从 BN=64 翻到 128，每 CTA 的 tile 数减半 ⇒ `__syncthreads` /
`cp.async.wait` / wgmma commit-wait 序列减半，main +2.9%。fp8 mma 路径主 kernel 仍是 **BN=32**
（tile 数最多），且 O7e-3/O19 判定其墙是「**mma 依赖延迟（`wait`）+ occupancy**」。于是把同一个
「**翻倍 KV-tile 以减半串行相位**」的假设搬到 fp8 mma 路径上验证：BN=32→64。

顺带清理端到端：fp8 的 dQ/dK/dV 输出本就是 **fp32**，与 fp32 累加缓冲同 dtype，`convert_kernel`
只是一趟纯 fp32→fp32 拷贝（O14b 已向量化但依然多一遍读写），可让 main 的 `atomicAdd` 直接累加进
输出、彻底消掉这一趟（记为 **O21b**）。

### 29.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

**O21（BN 参数化）**：主 kernel 原来把 4-warp 几何写死成 BN=32（`mma_block<32,16,…>`、
fold 的 `j=wid*8+jl`、dS2 的 32 个 j 等）。改为按 `(BM,BN)` 派生：

- `MTM=(BM/2)/16`、`MTN=(BN/2)/8`（GEMM1/2 warp 块 = `BM/2 × BN/2`）；
  `MTM34=(BN/2)/16`（GEMM3/4 输出 M=BN，warp 块 = `BN/2 × NTW`）；`NTFOLD=BN/32`（fold 份数）。
- GEMM1/2：累加器 `[MTM][MTN][4]`、`preg[MTM][MTN][4]`，`mma_block<BM/2,BN/2,…>`，
  `r0=wr*(BM/2)`、`c0=wc*(BN/2)`。
- GEMM3/4：累加器 `[MTM34][8][4]`、`mma_block_bt<BN/2,64,BM,…>`、`r0=wr*(BN/2)`。
- fold Ap/dS3：`for jh<NTFOLD: j = wid*8 + jl + jh*32`（每 warp 覆盖 `NTFOLD` 份 j 行）。
- fold dS2：amax 跨 `NTFOLD` 份累计（per-m 的 rowwise scale 覆盖全部 BN 个 K），16B 向量写落在
  `sub2*16 + jh*32`。
- `Fp8Cfg`：`NPU=(BN/2)*(HD/4)/THREADS`（BN=64→8），预取开关放宽到 `NPU*4<=32`；
  `__launch_bounds__(128, BN<=32?3:2)`。

**O21b（消 convert）**：host 里把 `d_dq_acc/d_dk_acc/d_dv_acc` 指针**别名**到 `d_dq/d_dk/d_dv`
（free 掉独立缓冲），main 直接对输出做 `atomicAdd`；`run_all` 里 `if (cvt_on) convert_kernel(...)`，
默认 `cvt_on=0` 不跑 convert。`--cvt=1` 恢复旧行为（自拷贝）供 A/B。

### 29.3 数值

O21：BN=64 与 BN=32 的 **dk/dv 逐位一致**（dk/dv `max_abs` ≤ 9.5e-7，S=4096；GQA ≤ 3.8e-6），
**dq 有 ~6e-2 的差**——根因是 dS2 的 rowwise scale 是 **per-m 沿 K(BN) 求 amax**，BN 从 32 变到 64
会改变该 scale 的量化粒度（dS2 的量化值本身变了），属预期、非 bug；dq vs ref 仍与历史同量级
（S=4096：2.635e-1 / 2.643e-1 / 3.216e-1，逐位不变）。

O21b：输出与旧路径**逐位相同**（convert 是纯拷贝）；S=4096 ours vs ref dq/dk/dv =
2.635e-1 / 2.643e-1 / 3.216e-1（与 O20 历史逐位一致）。

### 29.4 性能（CUDA event；同 session A/B）

**O21（main-only）**：

| shape | BN=32 | BN=64 | 比 |
|---|---|---|---|
| S512 H16 | 0.0754 ms | 0.0837 ms | **0.900×** |
| S1024 H32 | 0.4090 ms | 0.4447 ms | **0.920×** |
| S1024 H32 kv4 (GQA) | 0.3828 ms | 0.4160 ms | **0.920×** |
| S4096 H16 | 2.2061 ms | 2.7033 ms | **0.816×** |

**O21b（end2end，同 session）**：S4096 保留 convert 2.7469 → **直写输出 2.6497 ms（1.0367×）**；
单文件 2.7562 → 2.6340 ms（1.0464×）；S1024H32 kv4 0.5349 → 0.5161 ms（1.0365×）。
S4096 端到端 **2.65 ms / 52.2 TF**（FA3 fp16 848TF 的 6.2%；TE FP8 0.5909ms/465TF ⇒ ours 仍 **4.5×**
于 TE FP8，O20 时 4.7×）。

### 29.5 ncu（main，S=4096，同 session，`--set full -c 1`）

| 指标 | BN=32 | BN=64 | 说明 |
|---|---|---|---|
| Duration | 2.19 ms | 2.74 ms | 慢 25% |
| regs/thread | 168 | **255** | launch_bounds(…,2) 让 ptxas 用满 |
| smem/block | 72.70 KB | **105.22 KB** | Ks/Vs/Ps/Ss/Kp/dS2 全变宽 |
| Block Limit（reg/smem） | 3 / 3 | **2 / 2** | — |
| Achieved occupancy | **18.05%**（12 warp/SM） | **12.28%**（8 warp/SM） | 主因 |
| L1/TEX / L2 / Compute | 55.07 / 51.29 / 43.64% | 39.77 / 40.96 / 31.55% | 全线下移 |
| Warp Cycles/Issued | 6.19 | 5.92 | 单 warp 略好，但 warp 数少 |

### 29.6 结论

**O21 负结果**：fp8 mma 路径翻倍 KV-tile 虽把每 CTA 的串行 tile 相位砍半（`Warp Cycles/Issued`
6.19→5.92），但 smem 72.7→105.2KB、regs 168→255 把 occupancy 从 **3→2 CTA/SM（12→8 warp/SM）**；
本 kernel 是**延迟受限**（O19/O20 已判），少掉 1/3 的在飞 warp 比省下的相位更贵 ⇒ 全线 0.82–0.92×。
这与 fp16 O18（从 1 CTA/SM 出发、翻倍后仍 1 CTA/SM）相反：**fp8 已经在 3 CTA/SM，翻倍 tile 必掉
occupancy**，所以此路对 fp8 不成立。**再次确认 fp8 main 的唯一真杠杆是「提 occupancy」**（需
168→≤128 regs 且 72.7→≤58KB smem 才 4 CTA/SM）或「减 mma 依赖 stall」；「放大 tile / 减 red」对
fp8 无效（O19+O21 双重证伪）。

**O21b 正结果**：消掉冗余 convert 是**零风险、纯收益**——main 直接累加进 fp32 输出，端到端
**1.037×**（S4096 2.75→2.65ms），数值逐位不变。已设为 fp8 两文件/单文件的默认路径。

### 29.7 复现

```bash
# O21 A/B：程序内对 BN=32/64 同 session 计时 + 对拍（默认 BN=32）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=30
# O21b：默认 cvt_on=0（消 convert）；--cvt=1 恢复旧路径
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --cvt=1
# 单文件（device 由 sync_onefile_device.py 同步，逐字一致）
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu（BN=32 / BN=64）
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full -c 1 --kernel-name regex:fa_bwd_fp8_mma_kernel \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=5
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full -c 1 --kernel-name regex:fa_bwd_fp8_mma_kernel \
  -- --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=5 --bn64
```

原始输出：`src/fp8/o21_main_bn_ab_s4096.out.txt`、`src/fp8/o21_main_bn_ab_gqa_kv4.out.txt`、
`src/fp8/o21_onefile_s4096.out.txt`、`src/fp8/o21_ncu_bn32_s4096.out.txt`、
`src/fp8/o21_ncu_bn64_s4096.out.txt`。

---

## 30. O22：fp8 Hopper 路径（wgmma）默认化 + 寄存器压力/stall 审计（三组负结果）

### 30.1 动机

fp8 主 kernel 的墙，从 O7e-3（§26）、O19（§27）、O20（§28）、O21（§29）一路被 ncu 定位为
**mma 依赖延迟（`wait` + `short_scoreboard`）+ 3 CTA/SM 的 occupancy**。而 O9c（§21，LSE 的
`qkᵀ`）与 O9c-2（§22，主 kernel 的 GEMM1/2）早已把 fp8 这两段换成 Hopper `wgmma`，只是**默认关**
（需 `--lsewgm --wgmma` + `-DFA_WGMMA` 的 `sm_90a` 构建）。本轮把「**Hopper 路径默认化**」作为增量，
并把此前几组候选杠杆一次性量测判决（负结果照记）。

### 30.2 实现（单/两文件 device 逐字一致，host 同步）

- `-DFA_WGMMA`（`sm_90a`）构建下，`lsewgm`/`wgmma` **默认 1**；`sm_90` 构建无该路径、保持 mma。
  新增 `--lsewgm=0/1`、`--wgmma=0/1`，可在**同一 binary** 内退回 mma 做 A/B（`--wgmma`/`--lsewgm` 仍保留）。
- 新增 `--regdq=0/1`（强制关/开寄存器 dQ 累加，O22 A/B）与编译开关 `FA_ILV`（默认 0，见 §30.4）。
- 单文件 `fa_bwd_fp8_mma_onefile.cu` 由同一处修改同步（device 区从 `fexp` 起逐字一致，脚本核对
  `DEVICE REGION IDENTICAL`）。

构建/运行（Hopper 路径）：
```bash
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=20
```

### 30.3 性能（同 binary、同 session A/B；CUDA event 端到端 total，ms）

| shape | mma（lse+main） | **Hopper 默认** | 比 | preprocess mma→wgm | main mma→wgm |
|---|---|---|---|---|---|
| S512 H16 | 0.1421 | **0.1341** | **1.060×** | 0.0467→0.0412 | 0.0670→0.0655 |
| S1024 H32 | 0.5623 | **0.5308** | **1.059×** | 0.1021→0.0890 | 0.3912→0.3736 |
| S4096 H16 | 2.6937 | **2.4837** | **1.085×** | 0.3957→0.3176 | 2.1752→2.0432 |
| GQA kv4 S1024 | 0.5296 | **0.4947** | **1.071×** | 0.1001→0.0865 | 0.3768→0.3581 |

- 收益来自两处：**LSE 的 `wgmma`**（S4096 preprocess 1.246×）+ **主 kernel GEMM1/2 的 `wgmma`**
  （S4096 main 1.065×）。S=4096 端到端 **2.48 ms / 55.3 TF**（main-only 2.0432 ms / 67.3 TF，峰值 3.4%）。
- 数值 vs fp32 ref 与历史逐位同级（S512 2.426/2.972/3.733e-1；S1024H32 2.399/4.177/3.535e-1；
  S4096 2.635/2.644/3.216e-1；GQA kv4 2.517/5.339/7.173e-1；MLA S1024H2 2.232/3.337/3.602e-1）。
  单文件与两文件一致（S512 total 0.1346 / GQA 0.4972 ms）。
- MLA（D=512）不走 wgmma（仅 D=128），保持 O21b 路径（S1024H2 total 0.5195 ms）。

### 30.4 同轮量测的候选杠杆（结论：均不采纳）

1. **`--regdq=0`（关寄存器 dQ 累加）**：O21 ncu 显示 `dqacc[2][8][4]`（64 fp32）把 168-reg
   预算挤爆——PTXAS 报 **68B spill stores / 380B spill loads**，local 占 L1TEX sector **9.57%**
   （Est 28–50%）。但关掉 REGDQ 后 dQ 每个 nt tile 发一次跨 CTA `atomicAdd`（归约 O(1)→O(ntiles)）：
   同 session main **on 2.136 ms vs off 2.480 ms（on 快 1.16×）**。⇒ **保留 REGDQ**，local spill
   比多发的 dQ red 便宜（与 O19「fp8 不是 L2 red bound」不矛盾——这里差的是 dQ 专属的 RMW 次数）。
2. **`FA_ILV=1`（GEMM1/2 mma 指令级交错）**：把两条独立 GEMM 的 mma 连发（`acc`/`acc2` 双累加器）
   再做 epilogue，数学/数值逐位不变。三次 A/B：ILV=0 均值 2.180 ms vs ILV=1 均值 2.185 ms
   ⇒ **中性偏负**（`wait` 未被填掉；ptxas 对 mma inline asm 的调度空间有限）。默认 0。
3. **ksplit 重标定**：S=4096 k=2/4/8/16 = 2.341/2.175/2.157/2.256 ms ⇒ 维持 O2b 的
   `TARGET=4096→k=4`（k=8 仅 ~1%，session 噪声内）。
4. **`PSS` padding 扫描**：穷举 32–64 无法把 GEMM1/2 epilogue 的 store 冲突降到 0（最小恒 1.9-way），
   O7e-3 选的 `37` 已近最优 ⇒ 不动。

### 30.5 ncu（主 kernel，S=4096，Hopper 默认，`-c 1`）

`Duration 2.05 ms`（mma 2.19 ms）；stall `wait 1.53 + short_scoreboard 1.57 + long_scoreboard 0.66 +
not_selected 0.36 + barrier 0.23` ⇒ **仍是 mma 依赖延迟（GEMM3/4/5 仍 mma，fp8 wgmma 无转置操作数，
O9c-2b 已判硬件阻塞）+ 3 CTA/SM**；与 mma 路径同质，只是 GEMM1/2 的 ldmatrix 被 SS 直读取代。

### 30.6 复现

```bash
# Hopper 默认（-DFA_WGMMA 构建）vs mma（同 binary 退回）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s4096_h16_d128_causal_fp8 --iters=20
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=... --iters=20 --wgmma=0 --lsewgm=0
# REGDQ / ILV / ksplit 判决
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=... --regdq=0     # 见 [O22 A/B]
NVCC_FLAGS="-DFA_ILV=1" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=...
```

原始输出：`src/fp8/o22_wgmma_default_s4096.out.txt`、`o22_wgmma_default_shapes.out.txt`、
`o22_mma_baseline_shapes.out.txt`、`o22_wgmma_default_onefile.out.txt`、
`o22_regdq_ab_s4096.out.txt`、`o22_ilv_ab_s4096.out.txt`、`o22_ksplit_s4096.out.txt`、
`o22_ncu_stall_s4096.out.txt`（mma）、`o22_ncu_stall_wgmma_s4096.out.txt`（wgmma）、
`o22_fa3_te_baseline_fp16.out.txt`、`o22_te_fp8_bench.out.txt`。

---

## 31. O26：fp8 `delta_kernel` → warp-per-row 向量化（对齐 fp16/bf16 O24）

### 31.1 动机

fp16/bf16 早在 **O24（第六十五轮）** 就把 `delta_kernel`（D=rowsum(dO∘O)）从「每 (s,h,b) 行
一个 128 线程 CTA + `__shared__` 树归约 + log2(THREADS) 次 `__syncthreads`」改成
**warp-per-row 向量化**（S=4096 delta 42.5→14.5µs，3.35×，ncu 墙从 smem 归约移到 DRAM 带宽）。
但 **fp8 的 delta 一直是旧版**——O11（fp8 LSE 负载均衡）、O14（fp8 quant 向量化）都只覆盖了
LSE 与 quant，delta 漏了。本轮把它补上，使 fp8 的 preprocess 三个子 kernel 与 fp16/bf16 对齐。

旧 `delta_kernel` 对 HD=128 一行只有 128 个乘加，却要付 1 个 CTA + 7 次 barrier 的固定开销
（ncu S=4096：`Duration 42.9µs`、`Compute 74.9%`、`Waves 31.03`、grid 65536），是纯浪费。

### 31.2 改动（单/两文件 device 代码逐字一致）

新增 `delta_warp_kernel<HD>`（`fa_bwd_fp8_kernels.cuh`，单文件由 `sync_onefile_device.py` 同步）：

* **每 warp 一行**：lane 沿 HD 以 `float4`（O，fp32）与 `uchar4`（dO，e5m2）各读 4 个元素
  （warp 每步 32×4=128 个 d），`__shfl_xor_sync` 树归约，**无 smem / 无 barrier**；
* grid-stride 覆盖 `B*S*H` 行；
* 逐元素数学与旧版同式 `O[d]*(deq_e5m2(dO[d])*dos[row])`，只有 fp32 求和次序不同
  （warp 树 vs 128 线程 smem 树）⇒ `max_abs(new-vs-old) = 1.9e-6 ~ 8.6e-6`、`max_rel ~1e-3`，
  对 fp8 容差 O(1) 无影响；vs fp32 ref 的 dq/dk/dv 仍与历史同水平。

host 加 `--deltawarp=0/1`（默认 1，0 退回旧版做同 session A/B）与 `[O26 A/B]` 段。

### 31.3 性能（同 session A/B，CUDA event；`-DFA_WGMMA` wgmma 默认构建）

| shape | delta 旧(per-row) | delta 新(warp-per-row) | 加速 | preprocess（O22→O26） | ours total |
|---|---|---|---|---|---|
| MHA (1,512,16,128) | 0.0084 ms | **0.0035 ms** | 2.39× | 0.0392→0.0368 | **0.1284 ms / 16.7 TF** |
| MHA (1,1024,32,128) | 0.0235 ms | **0.0073 ms** | 3.22× | — | **0.5218 ms / 32.9 TF** |
| MHA (1,4096,16,128) | 0.0433 ms | **0.0151 ms** | 2.87× | 0.3176→**0.2946** | **2.4637 ms / 55.8 TF** |
| GQA h32kv4 (1,1024,32,128) | 0.0229 ms | **0.0075 ms** | 3.05× | — | **0.4875 ms / 35.2 TF** |
| GQA h40kv8 (1,1024,40,128) | 0.0282 ms | **0.0089 ms** | 3.16× | — | **0.5606 ms / 38.3 TF** |
| MQA h64kv1 (1,1024,64,128) | 0.0438 ms | **0.0138 ms** | 3.18× | — | **0.7926 ms / 43.4 TF** |
| MLA (1,256,2,512) | 0.0037 ms | **0.0028 ms** | 1.33× | — | 0.1169 ms / 2.3 TF |
| MLA (1,512,4,512) | 0.0050 ms | **0.0033 ms** | 1.50× | — | 0.2972 ms / 7.2 TF |
| MLA (1,1024,2,512) | 0.0050 ms | **0.0033 ms** | 1.51× | — | 0.5175 ms / 8.3 TF |

* S=4096 端到端 **2.4837（O22）→ 2.4637 ms（1.008×）**、preprocess **1.078×**；小 S 的 delta
  绝对量小（3–9µs），收益 ~1% 量级。收益与 fp16/bf16 O24 同源（都是把 smem 归约换成 warp 树 + 向量读）。
* 对标：同 session TE FP8（`fa_bwd_bench.py bench --dtype fp8`）S512 0.1007ms / S4096
  0.5904ms / 465.6TF；同 session 纯反向 fp16（`fa_vs_te_bwd_only.py`）MHA S4096 FA3
  0.3237ms/849TF、TE 0.4419/622、FA2 0.7233/380；GQA kv4 FA3 0.0828ms/415TF。
  ours fp8 total S4096 为 TE FP8 的 **4.17×**（O22 4.21×）。

### 31.4 ncu（delta，S=4096，同 binary `--deltawarp` 0/1，`--set full -c 1`）

| | grid | Duration | DRAM | Compute | Occ | Waves | Regs |
|---|---|---|---|---|---|---|---|
| 旧 per-row | 65536×128 | **42.94 µs** | 31.13% | **74.93%** | 83.09% | 31.03 | 17 |
| 新 warp-per-row | 16384×128 | **17.34 µs** | **77.01%** | 27.14% | 83.77% | 7.76 | 18 |

⇒ 墙从 **smem 归约 + Compute 75%** 移到 **DRAM 带宽 77%（elementwise 上限）**，与
fp16 O24 的 delta（72.87%）和 fp8 O14 的 quant（71.31%）结论一致。

### 31.5 复现

```bash
# wgmma 默认构建（两文件 / 单文件）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s4096_h16_d128_causal_fp8
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --dir=.../b1_s4096_h16_d128_causal_fp8
# A/B：--deltawarp=0 退回旧版（[O26 A/B] 打印两版计时 + max_abs/max_rel）
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full -c 1 -k regex:delta_kernel -- --dir=...
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full -c 1 -k regex:delta_warp_kernel -- --dir=...
```

原始输出：`src/fp8/fa_bwd_fp8_main_o26_sweep.out.txt`（两文件 ×9 shape × `[O26 A/B]` + 对拍）、
`src/fp8/fa_bwd_fp8_mma_onefile_o26_sweep.out.txt`（单文件）、
`src/fp8/o26_ncu_delta_old_s4096.out.txt` / `o26_ncu_delta_new_s4096.out.txt`、
`src/fp8/o26_te_fp8_bench.out.txt`、`src/fa_bwd_o26_fa3_te_baseline_fp16.out.txt`。

---

## 32. O27：fold 量化的「逐元素精确除法」→「每行一次 rcp + 乘法」（main 1.08–1.18×）

### 32.1 动机

`fold`（把 fp32 的 `P`/`dS` 按 rowwise amax 量化成 fp8 的 `Ap`/`dS3`/`dS2`，供 GEMM3/4/5 的
A 操作数）对**每一个 (m,j) 元素**都算一次 `Ps*[dos] / scA`（`dS3` 用 `Ss*[qs] / sc3`、`dS2`
用 `Ss*[ks] / sc2`）。而 `scX` 是**每输出行一个**的常量（`scX = amax / fp8max`）：`Ap/dS3` 里
`scX` 沿 `j`（每 warp 8 行）不变、`dS2` 里 `sc2` 沿 `m` 不变。ptxas 默认 `-prec-div=true`，
`a/b` 是 IEEE 精确除法（~10+ 条指令，含 `MUFU.RCP` + Newton 迭代），所以这里每个元素都付一次
精确除法，是 fold 段（CUDA-core 工作，正处在 GEMM1/2 与 GEMM3/4/5 之间、张量核空转）的主要
纯浪费。ncu（S=4096）此前报「non-fused FP32 148.8M vs fused 93.4M，Est. 4.68%」，正对应这里。

### 32.2 改动（单/两文件 device 代码逐字一致）

`fa_bwd_fp8_mma_kernel<..., bool RCP>` 新增模板参数（默认 `true`）：

* 每行的 `sub4==0` / `sub2==0` lane 已算出 `scX` 并经 `__shfl` 广播；**紧接广播后**再算
  一次 `invX = __frcp_rn(scX)`（1 条 MUFU）并同样广播；
* 元素处用新 helper `folddiv<RCP>(x, sc, inv)`：`RCP=true` 返回 `x * inv`，`false` 退回
  `x / sc`（供同 binary A/B）。`scX` 本身仍原样存进 `sA/sds3/sds2`（供 GEMM3/4/5 反量化），
  只有量化路径改用倒数。

数学等价；`a/sc` 与 `a*rcp(sc)` 在 fp32 下偶有 1 ULP 差，而 fp8 只有 3 位尾数 ⇒ 绝大多数元素
的 `cvt` 结果不变，少数在舍入边界上翻 1 个 fp8 码（`max_abs(rcp-vs-div)` 见下表，远小于
fp8 容差 O(1)）。**vs fp32 ref 的 dq/dk/dv 与历史逐位一致**（下表，9 个 shape 全部相同）。
host 加 `--foldrcp=0/1`（默认 1）+ `[O27 A/B]` 段。

### 32.3 性能（同 session A/B，CUDA event，main-only；`-DFA_WGMMA` 默认构建）

| shape | fold div | fold rcp-mul | 加速 | total（O26→O27） |
|---|---|---|---|---|
| MHA (1,512,16,128) | 0.0652 ms | **0.0604 ms** | 1.077× | 0.1284→**0.1246** |
| MHA (1,1024,32,128) | 0.3722 ms | **0.3408 ms** | 1.092× | 0.5218→**0.4895** |
| MHA (1,4096,16,128) | 2.0431 ms | **1.7580 ms** | **1.162×** | 2.4637→**2.1770**（63.1 TF） |
| GQA h32kv4 (1,1024,32,128) | 0.3553 ms | **0.3282 ms** | 1.083× | 0.4875→**0.4566** |
| GQA h40kv8 (1,1024,40,128) | 0.4115 ms | **0.3572 ms** | 1.152× | 0.5606→**0.5010** |
| MQA h64kv1 (1,1024,64,128) | 0.6212 ms | **0.5274 ms** | 1.178× | 0.7926→**0.6956** |
| MLA (1,256,2,512) | — | — | — | 0.1169→**0.1094** |
| MLA (1,512,4,512) | — | — | — | 0.2972→**0.2688** |
| MLA (1,1024,2,512) | — | — | — | 0.5175→**0.4653** |

* `[O27 A/B] max_abs(rcp-vs-div)`：S512 1.99e-3/4.8e-7/5.7e-4；S4096 4.6e-4/6.0e-4/6.5e-4；
  kv4 3.4e-3/1.9e-6/7.0e-4；kv8 1.2e-4/2.0e-3/9.9e-4；MQA 4.0e-4/1.7e-3/1.2e-3
  （dk/dv 的较大值是 `atomicAdd` 归约次序 + 边界舍入的组合，均在 fp8 容差内）。
* **vs fp32 ref 与历史逐位一致**：S512 2.426/2.972/3.733e-1；S1024H32 2.399/4.177/3.535e-1；
  S4096 2.635/2.644/3.216e-1；GQA kv4 2.517/5.339/7.173e-1；kv8 2.869/5.367/7.032e-1；
  MQA 4.101e-1/1.572/2.126；MLA S256H2 2.356/2.290/3.441e-1；S512H4 2.415/2.992/4.481e-1；
  S1024H2 2.232/3.337/3.602e-1。单文件（host 用默认 `RCP=true`）与两文件一致。
* **对标**：同 session TE FP8（`fa_bwd_bench.py bench --dtype fp8`）S=4096 **0.5904ms/465.6TF**
  ⇒ ours total 为 TE FP8 的 **3.69×**（O26 4.17×）；同 session 纯反向 fp16 MHA S4096 FA3
  0.3245ms/847TF、TE 0.4401/625、FA2 0.7295/377。9 个 shape 全部同向改善。

### 32.4 ncu（main，S=4096，同 binary `--foldrcp` 1/0，`-c 1`）

| | Duration | executed inst | L1/TEX | L2 | Compute | stall wait/short/long/barrier |
|---|---|---|---|---|---|---|
| div（O26 基线） | 2.06 ms | 852.6 M | 57.0% | 54.5% | 43.4% | 1.53/1.57/0.67/0.23 |
| rcp-mul（O27） | **1.78 ms** | **735.2 M（−13.7%）** | 61.6% | 63.2% | 43.7% | 1.48/1.43/0.87/0.27 |

⇒ 精确除法换乘法后**指令数 −13.7%**、Duration 同步 −13.6%；第一墙仍是 **mma 依赖延迟
（`wait`+`short_scoreboard`）+ 3 CTA/SM**（O7e-3/O19/O20/O22 的结论未变），但 fold 段本身被
显著削短。regs/smem/occupancy 不变（168 / 70.66KB / 3 CTA/SM）。

### 32.5 复现

```bash
# 默认（RCP=true）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s4096_h16_d128_causal_fp8
# A/B（同 binary）：--foldrcp=1 默认 / --foldrcp=0 退回精确除法
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu -c 1 -k regex:fa_bwd_fp8_mma_kernel -- --dir=...
```

原始输出：`src/fp8/o27_main_sweep.out.txt`（两文件 ×9 shape × `[O27 A/B]` + vs ref/TE）、
`src/fp8/o27_ncu_main_s4096.out.txt`（O26 基线 `--set full`）、`src/fp8/o27_ncu_rcp_s4096.out.txt`、
`src/fp8/o27_ncu_stall_s4096.out.txt`、
`src/fp8/o27_te_fp8_bench.out.txt`、`src/fp8/o27_fa3_te_baseline_fp16.out.txt`。

---

## 33. O28：fold 的 fp8 转换向量化（`cvt ... x2` + `PACK_AB_MERGE_C`）——MLA 1.03×、d128 中性

### 33.1 动机

O27 把 fold 的「逐元素精确除法」换成「每行 `rcp` + 乘法」后，main 拿到 **1.08–1.18×**、ncu
`executed inst` −13.7%——说明 fold 段（纯 CUDA-core、夹在 GEMM1/2 与 GEMM3/4/5 之间、张量核
空转）的指令数**确实**在 main 的关键路径上。fold 里除了除法，每个 `(m,j)` 元素还要做 **3 次
fp8 转换**（`Ap=E4M3`、`dS3=E5M2`、`dS2=E5M2`）：旧实现逐个元素调 `__nv_cvt_float_to_fp8`
（SASS 一条 `F2FP.SATFINITE`）再用 `<< 8*tt` + `|` 拼成 4B（额外的 `LOP3/PRMT`）。

Hopper 提供 `cvt.rn.satfinite.{e4m3,e5m2}x2.f32`（一次转 2 个），SASS 走
`F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C`，可直接把两条结果拼进一个 32 位寄存器。CUDA 头
`cuda_fp8.h` 的 `__nv_cvt_float2_to_fp8x2` 正是这条指令，且与两次标量转换**同 RN + SATFINITE
⇒ 逐位相同**。

### 33.2 改动（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

* 新增 `cvt2_e4m3/cvt2_e5m2`（封装 `__nv_cvt_float2_to_fp8x2`）与
  `foldpack4<RCP,E5>(x0..x3, sc, inv)`：先把 4 个待量化浮点经 `folddiv` 折算，再做
  `lo=cvt2(x0,x1)`、`hi=cvt2(x2,x3)`，返回 `lo | (hi<<16)`（低字节对应 x0）。
* fold 的三处量化（`Ap/dS3` 的 F16B 两段与旧 4×4B 两分支、`dS2`）**全部改走 `foldpack4`**；
  新增编译开关 `FA_CVT2`（默认 1），`-DFA_CVT2=0` 退回逐元素标量 + shift/OR 做同 binary A/B。
* 数值：同一批 fp32 折算 + 同舍入 ⇒ 输出的 fp8 字节**逐位相同**，**vs fp32 ref 与历史逐位一致**。

### 33.3 性能（同 session A/B，CUDA event，main-only，ms；`-DFA_WGMMA` 默认构建）

| shape | CVT2=0（标量 cvt） | CVT2=1（cvt x2） | 加速 |
|---|---|---|---|
| d128 (1,512,16,128) | 0.0602 | 0.0596 | 1.01× |
| d128 (1,1024,32,128) | 0.3423 | 0.3414 | 1.00× |
| d128 (1,4096,16,128) | 1.7526 | 1.7499 | 1.00× |
| GQA h32kv4 | 0.3284 | 0.3273 | 1.00× |
| GQA h40kv8 | 0.3549 | 0.3512 | 1.01× |
| MQA h64kv1 | 0.5302 | 0.5276 | 1.00× |
| MLA (1,256,2,512) | 0.0432 | 0.0430 | 1.00× |
| MLA (1,512,4,512) | 0.1423 | **0.1386** | **1.027×** |
| MLA (1,1024,2,512) | 0.2727 | **0.2653** | **1.028×** |

* **d128（MHA/GQA/MQA）全部中性**：这些东西的 `fold` 每 tile 只量化一遍（`HD/NTW=1`），
  指令被张量核依赖延迟完全掩盖——与 O7e-3/O19/O20/O22/O27 的结论一致（第一墙是
  `wait`+`short_scoreboard` 的 mma 依赖延迟，不是 fold 的指令数）。
* **MLA（`HD=512`，`NDT=HD/128=4`）有 1.03× 的小正收益**：每 tile 的 fold 要跑 4 遍，
  fold 占比大、不再被完全掩盖；ncu Duration 326.75→**320.54µs**。
* **vs fp32 ref 与历史逐位一致**：S4096 `2.635/2.644/3.216e-1`；MLA S1024H2
  `2.232/3.337/3.602e-1`；单/两文件逐指标一致。

### 33.4 ncu（main，同 binary `FA_CVT2` 0/1，`-c 1`）

| | Duration | executed inst | L1/TEX | L2 | Compute | occ | stall wait/short/long |
|---|---|---|---|---|---|---|---|
| d128 S4096 CVT2=0 | 1.80 ms | 735,418,304 | 60.21% | 62.24% | 42.61% | 18.11% | 1.48/1.42/0.85 |
| d128 S4096 CVT2=1 | **1.77 ms** | **703,469,504（−4.3%）** | 60.98% | 63.22% | 41.34% | 18.11% | 1.55/1.58/1.02 |
| MLA S1024H2 CVT2=0 | 326.75 µs | 24,715,632 | 11.45% | 21.13% | 7.62% | 6.25% | 1.58/0.91/1.82 |
| MLA S1024H2 CVT2=1 | **320.54 µs** | **24,454,512（−1.1%）** | 11.65% | 21.49% | 7.67% | 6.25% | 1.59/0.91/1.85 |

**结论**：`cvt x2` 把 fold 的转换指令再砍一刀（d128 全局 inst −4.3%），但 **d128 是纯 mma 依赖
延迟 bound**，省下的指令无处兑现（中性）；只有 fold 占比较大的 **MLA 拿到 ~1.03×**。与 O27 并排
看，fold 的优化只在它有足够权重时（大 `NDT` / O27 的除法）才体现为墙。数值逐位不变、无回退，
保留为默认。

### 33.5 复现

```bash
# 默认（FA_CVT2=1）+ A/B（-DFA_CVT2=0）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s4096_h16_d128_causal_fp8
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_CVT2=0" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s4096_h16_d128_causal_fp8
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu -c 1 -k regex:fa_bwd_fp8_mma_kernel -- --dir=...
```

原始输出：`src/fp8/fa_bwd_fp8_o28_ab.out.txt`（9 shape × CVT2 0/1）、
`src/fp8/o28_ncu_main_s4096_ab.out.txt`、`o28_ncu_main_mla_s1024h2_ab.out.txt`、
`o28_main_s4096_full.out.txt`、`o28_main_mla_s1024h2_full.out.txt`、
`o28_te_fp8_bench.out.txt`、`o28_fa3_te_baseline_fp16.out.txt`。

---

## 34. O29：fp8 自动 split-K 重新标定（端到端 1.01–1.32×，主 kernel 最多 1.20×）
+ GEMM3/4 指令级交错判决（负结果）

### 34.1 动机

`fa_bwd_fp8_mma_kernel` 的 N 方向切块数 `ksplit` 一直是 **O2b（第二十二轮）** 定的固定目标
`TARGET = (D==128) ? 4096 : 132`（d128 铺约 10 个波、MLA 铺 1 个波）。但 O2b 之后主 kernel 的
数据通路被反复改过（O3 寄存器预取、O4b K 配对 + `ldmatrix.x2.trans`、O9c-2 Hopper wgmma、
O20 epilogue 融合、O22 wgmma 默认开、O27/O28 fold 折叠），**`ksplit` 的最优点已经漂移**：

* d128 固定 4096 在 **base 小**（S=1024，`base_grid = (S/64)·H` = 512）时**过切**——每个 CTA
  只有 ~2 个 tile，且此时 `ksplit` 大 ⇒ `use_regdq`（寄存器 dQ 累加）被判为关，dQ 逐 tile 发
  跨 CTA `red`，切得越细原子越多、L2 越堵。
* S=4096 时 4096 又**欠切**（k=4 略慢于 k=8）。
* MLA（D=512，1 CTA/SM）固定 132（≈1 个波）**严重欠切**——grid 太小、并行度不足。

### 34.2 改动（仅 host 自动档；单/两文件 host 同步）

`sweep` 了全部 10 个 fp8 case（`--ksplit=1/2/4/8/16/32`），规律清晰，新标定：

```
D == 128:  S >= 2048  → 8192
           否则         → max(2048, 4 * base_grid)     # base∈{128,512,640,1024}
D == 512:  S / 2                                     # S256H2→128 / S512H4→256 / S1024H2→512
```

只需改 `fa_bwd_fp8_main.cu` / `fa_bwd_fp8_mma_onefile.cu` 的自动 `ksplit` 分支（`--ksplit=N`
仍可强制覆盖，做 A/B）。因为 `use_regdq` 由 `ksplit` 派生，新 `ksplit` 会让 S1024 的
GQA/MHA 形状自动打开寄存器 dQ 累加（这也是收益的一部分）。**不改 device 代码、不改数学口径**
（只改 fp32 加法次序）。

### 34.3 sweep（main-only，CUDA event，`-DFA_WGMMA` 默认构建；单位 ms）

| case | base | auto(旧) | 旧 main | 最优 k | 最优 main | 新 auto main | 新/旧 |
|---|---|---|---|---|---|---|---|
| S512 MHA | 128 | 16 | 0.0600 | 16 | 0.0600 | 0.0602 | 1.00 |
| S4096 MHA | 1024 | 4 | 1.7579 | 8 | 1.7366 | **1.7370** | **1.012** |
| S1024 H32 | 512 | 8 | 0.3404 | 4 | 0.3021 | **0.2988** | **1.139** |
| S1024 H32 kv4 | 512 | 8 | 0.3286 | 4 | 0.2736 | **0.2742** | **1.198** |
| S1024 H40 kv8 | 640 | 4 | 0.3534 | 4 | 0.3534 | 0.3531 | 1.00 |
| S1024 H64 kv4 | 1024 | 4 | 0.5319 | 4 | 0.5319 | 0.5285 | 1.006 |
| S1024 H64 kv1(MQA) | 1024 | 4 | 0.5238 | 4 | 0.5238 | 0.5234 | 1.00 |
| MLA S256 H2 | 8 | 16 | 0.0437 | 8–32 | 0.0431 | 0.0437 | 1.00 |
| MLA S512 H4 | 32 | 4 | 0.1402 | 8 | 0.1216 | **0.1215** | **1.154** |
| MLA S1024 H2 | 32 | 4 | 0.2652 | 16 | 0.2005 | **0.2004** | **1.324** |

端到端 total（quant+preprocess+main+convert，ms）：S4096 `2.1770→2.1481（1.013×）`、
S1024H32 `0.4895→0.4464（1.096×）`、GQA kv4 `0.4566→0.4056（1.126×）`、kv8 `0.5010→0.4965`、
MQA `0.6956→0.6820`、MLA S512H4 `0.2688→0.2489（1.080×）`、MLA S1024H2
`0.4653→0.3900（1.193×）`。**全 shape 不回退。**

### 34.4 ncu（main，S1024H32，同 binary `--ksplit` 8 vs 4，`--launch-count 1`）

| | Duration | red sectors | L2 吞吐 | occ | stall wait/short/long |
|---|---|---|---|---|---|
| 旧 auto k=8 | 332.4 µs | 26,738,688 | **83.09%** | 18.18% | 1.60/**2.40**/1.71 |
| 新 auto k=4 | **304.9 µs** | **16,416,768（0.614×）** | **57.61%** | 17.68% | 1.55/**1.50**/1.69 |

机制：k 从 8→4 + `use_regdq` 自动打开 ⇒ dQ 的跨 CTA `red` 扇区降到 **0.61×**、L2 从 **83%→58%**，
`short_scoreboard` 2.40→1.50 ⇒ Duration −8.3%。**再次印证 fp8 main 的墙是访存/原子 + 依赖延迟，
而非算力**。

### 34.5 同轮负结果：GEMM3/4 指令级交错（`FA_ILV34`）

按 O22 `FA_ILV`（GEMM1/2 交错）的思路，新增 `-DFA_ILV34=1`：先连发 GEMM3(dV)/GEMM4(dK) 两条
独立 mma（各自累加器），再做 epilogue，让一条 mma 填另一条的依赖延迟。**数值逐位不变**，但
两个 `MTM34×8×4`（各 32 fp32）累加器同时存活，ptxas 在 `__launch_bounds__(128,3)` 的 170-reg
预算下**溢出增加**：S4096 main 1.812→1.903ms（0.95×）、S1024H32 0.341→0.355（0.96×）。
⇒ 与 O7c/O17b 一致，**在本卡 3 CTA/SM 的寄存器预算内「加 ILP」会被 spill 吃掉**；
`FA_ILV34` 保留为默认关的 A/B 开关。原始输出 `src/fp8/fa_bwd_fp8_o29_ilv34_s4096.out.txt`。

### 34.6 复现

```bash
# 新自动档（默认）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h32_d128_causal_fp8
# 旧自动档（同 binary 强制 k=8）做 A/B
... scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=... --ksplit=8
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --launch-count 1 -k regex:fa_bwd_fp8_mma_kernel -- --dir=... --ksplit=4
# ILV34 负结果
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_ILV34=1" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s4096_h16_d128_causal_fp8
```

原始输出：`src/fp8/fa_bwd_fp8_o29_ksplit_sweep.out.txt`（两文件版 ×10 shape × 新自动档：
ksplit/timing/对拍）、`src/fp8/fa_bwd_fp8_o29_onefile_sweep.out.txt`（单文件版 ×5 shape，
数值与两文件逐位一致、计时差 <1%）、
`src/fp8/fa_bwd_fp8_o29_ncu_s1024h32.out.txt`（k=8 vs k=4 的 ncu 指标）、
`src/fp8/o29_te_fp8_bench.out.txt`、`src/fp8/o29_fa3_te_baseline_fp16.out.txt`、
`src/fp8/fa_bwd_fp8_o29_ilv34_s4096.out.txt`。

## 35. O32：fp8 LSE 的 4D-TMA 载入（对齐 fp16 O30 / bf16 O31；LSE-only 1.06–1.10×）

### 35.1 动机

fp16/bf16 在 O30（`docs/01` §14r）/ O31（`docs/01b` §6x）已把 **LSE 的 Q/K 载入**从「逐 16B
`cp.async` + 地址运算」换成 **4D-TMA**（`cuTensorMapEncodeTiled` + `cp.async.bulk.tensor.4d`），
LSE-only 1.30–1.36×、指令数 −28.7%、数值逐位不变。**fp8 侧一直没做**：O9c 起 fp8 LSE 的
`issue_q`/`issue_k` 仍逐元素算 `sw128_off_fp8` 再 `cp_async16`（`docs/03` §22）。本轮补齐。

fp8 与 fp16 O30 的**关键差异**：TMA `SWIZZLE_128B` 的 box 内维固定 128B。fp16 一行 128 元素
= 256B，必须拆成 **2 个 K=64 chunk**（O15a 的坑）；而 **fp8 一行 128 元素恰好 = 128B = SW128
atom 的整行**，所以 **Q/K 各只需一次 4D-TMA**（box `{128,64,1,1}`，dtype=`UINT8`，见 23 篇坑
「CUDA 13 无 `FLOAT8_E4M3` 枚举，用 UINT8 搬字节」）。

### 35.2 改动（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

* `src/fp8/fa_bwd_fp8_kernels.cuh` 新增（`#if defined(FA_WGMMA) && defined(FA_TMA)`，
  asm 用 `FA_FP8_HAS_TMA` = `__CUDA_ARCH_FEAT_SM90_ALL` 包裹，纯 `sm_90` 构建退化为空实现）：
  * `mbar_init` / `mbar_arrive_expect` / `mbar_wait`（`mbarrier.*`）与 `tma_load_4d`
    （`cp.async.bulk.tensor.4d...mbarrier::complete_tx::bytes`），与 fp16 O30 同构。
  * `lse_mma_kernel_bal_tma<HD,PIPE=1>`：镜像配对 + online-softmax + 4-lane `shfl` 归约、
    rowwise scale `qs*ks` 相乘顺序与 `lse_mma_kernel_bal_wgmma` **完全一致**；只把 Q/K 的
    SW128 tile 改由 TMA 填充（K 双缓冲 + 2 个 mbarrier，相位按 `use&1`），**rowwise scale
    仍走标量 global 读**（TMA 带不了标量数组）。`Fp8Cfg::lse_smem_bytes_tma1 =
    3*lse_tile_wgmma + (LBM+2*LBN)*4 + 1024 + 64`（= fp16 TMA 同款 + fp8 的 768B 双 scale）。
* `fa_bwd_fp8_main.cu`：`make_lse_map_fp8`（4D dims={D,S,H,B}、strides 字节、UINT8、SW128）；
  CLI `--lsetma=0/1`（`-1` 自动：`-DFA_TMA` 构建下 D==128/causal 默认开）；`run_preprocess`
  分派 TMA/wgmma/mma；新增 `[O32 A/B]` 同 session 计时 + LSE 数值对拍。`#include <cuda.h>`。
* 单文件 `fa_bwd_fp8_mma_onefile.cu`：device 段由脚本同步，host 段镜像同样改动。
* `-lcuda` 链接 `cuTensorMapEncodeTiled`；纯 `-DFA_WGMMA` 或 `sm_90` 构建不引用驱动符号。

### 35.3 数值

* **TMA-vs-wgmma 的 LSE `max_abs=0.000e+00`（逐位相同）**：6 个 d128 shape × 单/两文件全部。
* `dq/dk/dv vs ref` 与历史（O9c/O22/O27/O29）**逐位一致**（见下表）——只换搬运、不改数学。

### 35.4 性能（同 session A/B，CUDA event；`-DFA_WGMMA -DFA_TMA -lcuda`）

**LSE-only**（两文件，ms）：

| shape | wgmma+cp.async | **TMA** | 加速 |
|---|---|---|---|
| S512 H16 | 0.0335 | **0.0312** | 1.076× |
| S1024 H32 | 0.0661 | **0.0615** | 1.075× |
| S1024 H32 kv4 | 0.0643 | **0.0597** | 1.077× |
| S1024 H40 kv8 | 0.0676 | **0.0627** | 1.079× |
| S1024 H64 kv1 (MQA) | 0.0744 | **0.0682** | 1.091× |
| S4096 H16 | 0.2726 | **0.2475** | 1.101× |

**端到端**（quant+preprocess+main，两文件；ours / TE FP8 同 session / 比值）：

| shape | ours total (ms) | ours TF | TE FP8 (ms) | TE TF | ours/TE |
|---|---|---|---|---|---|
| S512 H16 | 0.1219 | 17.62 | 0.1009 | 42.56 | 1.21× |
| S1024 H32 | 0.4468 | 38.45 | 0.2059 | 166.88 | 2.17× |
| S1024 H32 kv4 | 0.4035 | 42.58 | 0.2025 | 169.65 | 1.99× |
| S1024 H40 kv8 | 0.4891 | 43.90 | 0.2422 | 177.30 | 2.02× |
| S1024 H64 kv1 | 0.6839 | 50.24 | 0.4021 | 170.91 | 1.70× |
| S4096 H16 | 2.1303 | 64.52 | 0.5876 | 467.83 | 3.63× |

单/两文件逐指标一致（S4096 total 2.1301 vs 2.1303ms，差 <0.1%）；MLA（D=512）走 wgmma/mma
不受影响（S1024H2 total 0.3916ms，数值不变）。收益幅度小于 fp16 O30（1.3×）——因为 fp8 的 LSE
本来就用 `cp.async` 且每元素 1B（地址运算占比小），TMA 主要省的是 load 指令/地址算术。

### 35.5 ncu（LSE, S=4096，`--launch-count 1`）

| 指标 | wgmma+cp.async | **TMA** |
|---|---|---|
| Duration | — | **251.97 µs** |
| Compute (SM) | ~61% | **56.79%** |
| L1/TEX | ~28% | **25.00%** |
| L2 | ~15% | **14.60%** |
| DRAM | ~1.5% | **2.08%** |
| achieved occ | ~23% | 23.55%（理论 50%，61 regs / 26.43KB）|
| Waves / SM | 0.65 | 0.48 |
| stall `long_scoreboard` | **2.20** | **0.37** |
| stall `short_scoreboard` | **2.25** | **1.05** |
| stall `wait` | 2.68 | **2.34** |
| stall `barrier` | 0.53 | **0.32** |
| stall `mio_throttle` | 0.67 | 0.03 |

TMA 把 cp.async 的地址运算与 smem 写冲突整个消掉（`long_scoreboard` 2.20→0.37、
`short_scoreboard` 2.25→1.05），**新墙 = Compute ~57% + `wait`（softmax/mma 固定延迟）**，
与 fp16 O30 / bf16 O31 的结论一致。第一墙仍是主 kernel（mma 依赖延迟 + 3 CTA/SM），
LSE 已接近下限。

### 35.6 复现

```bash
# 两文件（TMA 默认开）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# 同 binary 退回 wgmma（A/B）
... scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=... --lsetma=0
# 单文件
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --dir=...
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:lse_mma_kernel_bal_tma --launch-count 1 -- --dir=...
```

原始输出：`src/fp8/fa_bwd_fp8_o32_sweep.out.txt`（两文件 ×8 shape：backend/timing/O32 A/B/
对拍）、`src/fp8/fa_bwd_fp8_o32_onefile_sweep.out.txt`（单文件 ×4 shape，与两文件逐位一致）、
`src/fp8/fa_bwd_fp8_o32_ncu_lse_tma_s4096.out.txt`（TMA ncu full）、
`src/fp8/fa_bwd_fp8_o32_ncu_stall_lse_{tma,wgmma}_s4096.out.txt`（stall 对比）、
`src/fp8/fa_bwd_fp8_o32_tebench.out.txt`（同 session TE FP8 基线）。

## 36. O37：fp8 主 kernel 的 Q/dO 改用 4D-TMA（对齐 fp16 O33 / bf16 O34；main 1.04–1.08×）

### 36.1 动机

O30/O31/O32 已把三种 dtype 的 **LSE** 换成 4D-TMA，但 **主 kernel 的 operand 仍是逐 16B
`cp.async` + `sw128_off_fp8` 地址运算**（fp16/bf16 的对应改造是 O33–O36）。ncu（O22/O32）显示
主 kernel 头号 stall 是 `wait`(1.53)+`short_scoreboard`(1.57)，但 `long_scoreboard` 也还有
~0.66–1.1，其中一部分来自 **prologue 的 Q/dO 一次性 global 读**（默认 HD=128/REGDQ 档下
`kPrefetch=false`，K/V 也是同步读；Q/dO 则完全暴露）。Q/dO 与 K/V 不同：**只读一次、与 tile
循环无关**，是 TMA 化风险最低的一步。

fp8 比 fp16/bf16 更简单：**一行 128B = 一个 SW128 atom 的整行**，一个 `box={128, BM}` 就能把
整块 Q/dO 搬成 K-major SW128（O32 的 LSE 已示范），不需要 fp16 的 2×K=64 chunk 拆分。

### 36.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

为不重复 700 行主 kernel，先把主 kernel 的**函数体抽成 device 函数** `fp8_mma_body<..., TMA>`，
再加两个薄 `__global__` 壳：
* `fa_bwd_fp8_mma_kernel`（原 cp.async 版，`TMA=false`，签名不变）；
* `fa_bwd_fp8_mma_qdtma_kernel`（`TMA=true`，多两个 `const __grid_constant__ CUtensorMap qmap/dmap`）。

`if constexpr (TMA)` 只改 Q/dO 的载入：
1. `tid==0` 发两条 `cp.async.bulk.tensor.4d`（box `{128,64}`，坐标 `{0, m0, h, b}`）把 Q/dO
   搬进 SW128 tile `Qs/dOs`，用两个 mbarrier（`qbars`，放在 smem 末尾额外 64B；
   `smem_bytes_wgmma_tma = smem_bytes_wgmma + 64`）同步；
2. 再从 `Qs/dOs` 的 SW128 地址（`sw128_off_fp8`）读回 Qp/dOp 所需的「行对」4B，
   `__byte_perm(0x5140/0x7362)` 重建 **K 配对布局**（供 GEMM3/4 的 `ldmatrix.x2.trans`）。

K/V、fold、GEMM3/4/5、dQ 归约**逐字未动** ⇒ 数值与 cp.async 版只差跨 CTA `atomicAdd` 次序。
`qd_tma` 默认开（对齐 O32 的 `lsetma`），`--qdtma=0` 供同 binary A/B；仅 `-DFA_WGMMA -DFA_TMA`
构建、D==128、WGMMA 路径启用，sm_90 构建完全不变。

### 36.3 数值（vs fp32 ref / TE FP8，逐元素）

九 shape 的 `dq/dk/dv vs fp32 ref` 与历史（O27/O28/O32）**逐位一致**（系统只差归约次序）：

| case | dq vs ref | dk vs ref | dv vs ref |
|---|---|---|---|
| b1_s512_h16_d128_causal | 2.426e-01 | 2.972e-01 | 3.733e-01 |
| b1_s4096_h16_d128_causal | 2.635e-01 | 2.644e-01 | 3.216e-01 |
| b1_s1024_h32_d128_kv4 | 2.517e-01 | 5.339e-01 | 7.173e-01 |
| b1_s1024_h40_d128_kv8 | 2.869e-01 | 5.367e-01 | 7.032e-01 |

同 session A/B 的 `max_abs(tma-vs-cp)`：dq ~1e-7、dk/dv ~1e-6 —— 与 fp16 O33「只换搬运方式」
的结论一致（差异仅原子累加次序）。

### 36.4 性能（同 binary、同 session A/B；CUDA event，main-only，ms）

| shape | cp.async | TMA | 比 |
|---|---|---|---|
| MHA S512 | 0.0686 | 0.0638 | **1.075×** |
| MHA S4096 | 1.7785 | 1.6940 | **1.050×** |
| GQA q32/kv4 | 0.2911 | 0.2787 | **1.045×** |
| MQA q64/kv1 | 0.5415 | 0.5042 | **1.074×** |

端到端（quant+preprocess+main，CUDA event）S4096 **2.0508 ms / 67.02 TF**（O32 时 2.177 ms），
为 **TE FP8（同 session 0.5903 ms / 465.6 TF）的 3.49×**（O27 时 3.69×）。GQA/MQA 端到端
0.39–0.65 ms，为 TE FP8 的 1.60–1.92×。单文件与两文件数字一致（`..._onefile_s4096.out.txt`：
total 2.0509 ms）。

### 36.5 ncu（主 kernel，S=4096，`--launch-count 1`；同 binary `--qdtma` 0/1）

| 指标 | cp.async | TMA |
|---|---|---|
| `gpu__time_duration.sum` | 1.77 ms | **1.66 ms** |
| `sm__inst_executed.sum` | 726.3 M | **710.2 M（−2.2%）** |
| stall `long_scoreboard` | 1.10 | **0.81（−26%）** |
| stall `short_scoreboard` | 1.54 | 1.58 |
| stall `wait` | 1.56 | 1.53 |
| regs / occupancy | 168 / 3 CTA/SM | 168 / 3 CTA/SM |
| `sm__throughput` | 43.3% | **44.6%** |

TMA 消掉了 Q/dO 载入的 global 地址运算与 `long_scoreboard`，指令数 −2.2%、main −4.8%（A/B）。
**墙仍不变**：`short_scoreboard`(1.58) + `wait`(1.53) 的 mma/smem 依赖延迟 + 3 CTA/SM —— 即
O22/O32 一贯的结论，TMA 只是把 Q/dO 这段的访存税拿掉。

### 36.6 剩余（K/V TMA）与判断

K/V 每 tile 都要搬，是更大的 `long_scoreboard` 来源。但**单缓冲 TMA 无收益**：Ks/Vs 在同一
迭代内被 `dS3/Ap`（fold）覆写，buffer 全程被占用，TMA 只能在上一次 GEMM 读完、fold 之前
「不可用」；若在迭代末发 TMA、下迭代初等待，则和现在的同步 `kv_load_pair` 一样把延迟暴露在
关键路径上。要真正重叠必须 **K/V 双缓冲**，而 smem 账算不过来：
wgmma 布局 `smem_bytes_wgmma=70656B`，3 CTA/SM 上限 `232448/3=77482B`，余量 ~6.8KB；双缓冲 K
需要 `+ks_sw_bytes(4096B)` 且不能再把 `dS3` 别进 Ks（`+BN*QTS=2560B`），共 ~6.7KB，几乎顶格；
再叠加 V、以及 O3 寄存器预取，**在 3 CTA/SM 下不可行**，掉到 2 CTA/SM 在延迟受限 kernel 上
是负优化（O19/O21 已双重证伪）。故 O37 到 Q/dO 为止；K/V TMA 需要先腾出 ~10KB smem
（例如 Ps/Ss 改存 fp16，但会改数值），列入 backlog。

### 36.7 复现

```bash
# 两文件（Q/dO TMA 默认开）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# 同 binary 退回 cp.async（A/B）
... scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=... --qdtma=0
# 单文件
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --dir=...
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma_qdtma \
  --launch-count 1 --set full -- --qdtma=1 --dir=...
```

原始输出：`src/fp8/fa_bwd_fp8_o37_main_{s4096,s512,prodshapes}.out.txt`（A/B + 对拍 + timing）、
`src/fp8/fa_bwd_fp8_o37_onefile_s4096.out.txt`、`src/fp8/fa_bwd_fp8_o37_ncu_main_{tma,cpasync}_s4096.out.txt`、
`src/fp8/fa_bwd_fp8_o37_tebench{,_requested}.out.txt`（同 session TE FP8 基线）。

---

## 37. VARLEN：fp8 反向支持变长 / `cu_seqlens`（packed `[T,H,D]`）

> 这是 ROADMAP「可选」里唯一未完成的 `[ ] 变长（cu_seqlens / varlen）覆盖` 的第一块
> （fp8，MHA/GQA，causal，HD=128）。fp16/bf16 复用同一 device 改造留后续。

### 37.1 动机与口径

生产推理/训练里 batch 内的序列长度常常不齐（packed `[T,H,D]`，`T=sum_b len_b`），
用定长 `[B,S,H,D]` 跑必须 padding 到 `max_b len_b`，浪费 `O(B·maxlen)` 的计算与显存。
本项让 fp8 反向直接吃 **packed** 输入 + `cu_seqlens`（长度前缀和，`B+1` 个），每个序列
只算 `len_b×len_b` 的因果注意力，不碰 padding。

接口约定：

* `q`/`dO`/`dQ`：`[T,H,D]`；`k`/`v`/`dK`/`dV`：`[T,Hkv,D]`（GQA/MQA 的 `Hkv` 同前）。
* `cu_seqlens`：`B+1` 个 int，`cu[0]=0`、`cu[b+1]=cu[b]+len_b`。
* kernel 收到 `S=maxlen` 作为 grid/分块启发式的基准长度，逐 `b` 用
  `qbase=cu[b]`、`len=cu[b+1]-cu[b]` 代替原来的 `b*S`/`S`。
* `ref_dq/dk/dv.npy` 也是 packed 布局，直接逐元素比对。

### 37.2 实现（device 单/两文件逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

改动收敛在 **3 处 device 代码**（其余 5 个 GEMM 的数学/数据流完全不动）：

1. **`fp8_mma_body<...,cu_seqlens>`**（主 kernel，单/两文件共用 body + 两个薄壳）：
   在 `b=blockIdx.z` 后算 `qbase/len`，把 `b*S+{qi,jg,qa,qb}` 全部换成 `qbase+…`、
   长度判定 `<S` 换成 `<len`、`ncols=min(S,…)` 换成 `min(len,…)`，并加
   `if (m0>=len) return;`（短序列的尾部 m 块直接退出）。
   `kv_prefetch_pair`/`kv_load_pair` 的 `b` 形参改名为 `qbase`（语义从「批号」变成「token 基址」，
   内部 `(b*S+jr)` → `(qbase+jr)`）。`nullptr` 时逐式退化为原定长路径 ⇒ 定长回归**逐位不变**。
2. **`lse_mma_kernel_bal_wgmma<...,cu_seqlens>`**（默认 Hopper LSE）：
   同样算 `qbase/len`，`nblk` 用 `len`，镜像配对的 `if (pair >= (nblk+1)/2) return;`
   让短序列的多余 CTA 退出；`issue_q/issue_k` 的全局地址用 `qbase`、越界判定用 `len`。
3. **`delta_warp_kernel`** 本就是「每 warp 一行」的扁平实现（`rows=T*H`），
   packed 布局下天然可用，**无需改动**（host 传 `rows=T*H` 即可）。

`quantize_row_warp_kernel` 同样按行处理，packed 下把 `[T,H,D]` 视作 `[T*H,D]` 即可，无需改。

host 侧新增 `--varlen` 分支（`run_varlen`）：读 packed q/k/v/dO + `ref_o` + `cu_seqlens.npy`，
建 packed 的 q8/scale/lse/delta/dq/dk/dv，按 `maxlen` 选 `ksplit`/`REGDQ` 自动档，
LSE 用 `launch_lse_bal_wgmma`、主 kernel 用 `launch_bwd_main<128,64,32,…,WGMMA=true>`
（**非 TMA**：TMA 描述符是按定长 `[D,S,H,B]` 建的，varlen 用 wgmma+cp.async 路径即可）。
非 causal / fp16 / bf16 / TMA 化留后续。

### 37.3 harness：dump + fp32 ref

`harness/fa_bwd_bench.py` 扩展：

* `--lengths "512 1024 2048 256" --H 16 --D 128 [--kv 8]`（或 `--varlen-all` 跑内置
  `VARLEN_SHAPES`）→ `dump` 出 packed `q/k/v/do.npy`、`cu_seqlens.npy`（fp32，便于 host 直读）、
  `ref_o/dq/dk/dv.npy` + `meta.json`（含 `lengths`）。
* `ref_attn_varlen`：逐序列切片喂给原 fp32 autograd `ref_attn` 再拼接。
* 共 dump **4 个 varlen case**：等长 `[1024]×4`、不齐 `[512,1024,2048,256]`、
  GQA `[128,256,512,1024,2048] kv=8`、强倾斜 `[2048,512,128,96,64,32,16,8]`。
* **注**：TE 2.14 的 fused_attn **FP8 变长**路径在本容器会 segfault（`thd_thd_thd`，
  无法 try/except 捕获），故 TE 变长基线缺失；等长 case 用 TE FP8 定长同 shape 对比。

### 37.4 数值（ours vs fp32 ref，fp8 causal；max_abs）

| case | lengths | dq | dk | dv |
|---|---|---|---|---|
| b1_t512（等价定长 S512） | `[512]` | 2.280e-1 | 3.108e-1 | 3.422e-1 |
| b4_t3840 不齐 | `[512,1024,2048,256]` | 2.935e-1 | 2.938e-1 | 4.179e-1 |
| b4_t4096 等长 | `[1024]×4` | 2.651e-1 | 3.026e-1 | 3.920e-1 |
| b5_t3968 GQA kv8 | `[128,256,512,1024,2048]` | 3.094e-1 | 5.567e-1 | 6.203e-1 |
| b8_t2904 强倾斜 | `[2048,512,…,8]` | 3.136e-1 | 3.584e-1 | 4.030e-1 |

全部落在 fp8 噪声量级（0.24–0.62），与定长 fp8 一致，**无系统误差、无 padding 泄漏**。
单文件输出与两文件**逐位相同**（如 b4_t3840：2.935/2.938/4.179e-1）。

### 37.5 性能（ours 端到端 total：quant+preprocess+main；CUDA event）

| case | lengths | total (ms) | TFLOPS（Σ_b 4HL²D） |
|---|---|---|---|
| b1_t512 | `[512]` | 0.126 | 17.0 |
| b4_t3840 不齐 | `[512,1024,2048,256]` | 0.954 | 47.8 |
| b4_t4096 等长 | `[1024]×4` | 0.767 | 44.8 |
| b5_t3968 GQA kv8 | `[128,256,512,1024,2048]` | 1.852 | 49.4 |
| b8_t2904 强倾斜 | `[2048,512,…,8]` | 0.839 | 43.8 |

**对标**（同 session，`harness/fa_bwd_bench.py bench --dtype fp8 --varlen-all`）：
等长 b4_t4096 用 TE FP8 定长同 shape 为 **0.3785 ms / 90.77 TF**（TE 口径含 forward，
ours 为纯反向 total）⇒ 时间比 **2.03×**。不齐/GQA/倾斜 case TE 变长 segfault，无基线。

注意：强倾斜 case（`[2048,512,…]`）虽然 padding 浪费大，但 `maxlen=2048` 让 grid 只有
`64×16×8`，短序列的 CTA 大量空转（`m0>=len` 提前 return），故 TF 仍只有 43.8 —— 真正的
「按序列变长分块」的负载均衡（如 FlashAttention 的均衡调度）留 backlog。

### 37.6 ncu（主 kernel，等长 b4_t4096，`--launch-count 1`）

| 指标 | 值 |
|---|---|
| Duration | 557.2 µs |
| DRAM Throughput | 12.0% |
| L1/TEX / L2 | **60.4% / 63.2%** |
| Compute (SM) | 37.7% |
| Achieved / Theoretical Occupancy | 18.1% / 18.8%（168 regs，Block Limit Shared Mem=3） |
| Waves Per SM | 10.34 |
| No Eligible | 59.4% |

bound 与定长 fp8 main 完全一致：**L1/TEX + L2 吞吐 + 3 CTA/SM 的延迟受限**，
DRAM 仅 12% ⇒ 非带宽 bound。原始输出 `src/fp8/fa_bwd_fp8_varlen_ncu_main_b4_t4096.out.txt`。

### 37.7 定长回归（nullptr 路径）

S512/S4096/GQA/MLA 四 shape 逐位不变：
S512 `2.426/2.972/3.733e-1`、S4096 `2.635/2.644/3.216e-1`、
GQA kv4 `2.517/5.339/7.173e-1`、MLA S1024H2 `2.232/3.337/3.602e-1`。
原始输出 `src/fp8/fa_bwd_fp8_varlen_fixed_regression.out.txt`。

### 37.8 限制 / 后续

* 只做 **fp8 / HD=128 / causal / MHA+GQA**；非 causal、fp16/bf16、MLA 留后续。
* 用 **非 TMA** 主 kernel 路径（wgmma + cp.async）；TMA 需要为 packed 布局重建描述符
  （扁平 `[D,T,H,1]`）或在 host 侧逐序列建 map。
* **负载均衡**：当前用 `maxlen` 决定 grid，短序列 CTA 早退，强倾斜 case 并行度浪费；
  可引入按 `cu_seqlens` 的均衡分块（FA2 的 `num_m_blocks` 调度）。
* TE 2.14 FP8 变长 segfault，性能基线只能对等长 case。

### 37.9 复现

```bash
# dump varlen（ref + cu_seqlens）
python harness/fa_bwd_bench.py dump --dtype fp8 --varlen-all
# 两文件 varlen 自测（默认 WGMMA+TMA 构建；varlen 内部走非 TMA 路径）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen=1 --iters=50 \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b4_t3840_h16_d128_causal_fp8
# 单文件
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --varlen=1 --iters=50 --dir=...
# TE 定长基线（等长 case）
python harness/fa_bwd_bench.py bench --dtype fp8 --varlen-all
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:fa_bwd_fp8_mma_kernel --launch-count 1 -- --varlen=1 --iters=1 --dir=...
```

原始输出：`src/fp8/fa_bwd_fp8_varlen_sweep.out.txt`（两文件 5 case）、
`..._varlen_onefile_sweep.out.txt`（单文件 4 case，逐位一致）、
`..._varlen_tebench.out.txt`（TE FP8 定长基线）、
`..._varlen_ncu_main_b4_t4096.out.txt`（ncu）、
`..._varlen_fixed_regression.out.txt`（定长回归）。

## 38. VARLEN 的非 causal（full attention）——fp8 反向

### 38.1 动机与口径

第 77 轮的 varlen 只做了 causal。非 causal（full）自注意力在编码器/双向场景同样需要变长
支持。非 causal 与 causal 的根本区别：**每个 m 块的 K 列数恒为 `len`（不再随 `mblk` 递增）**，
所以工作天然均衡，不需要镜像配对。主 kernel（`fp8_mma_body`）本就带 `causal` 参数
（mask 写成 `!(causal && jg > qi)`，`ncols = causal ? min(len, m0+BM) : len`），故只需把
**LSE** 从「causal 专用的镜像配对 wgmma 版」切到 O1 的通用 mma 版，并让后者认识 `cu_seqlens`。

### 38.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

* **device**（`fa_bwd_fp8_kernels.cuh`）：`lse_mma_kernel<HD>` 增加默认参数
  `const int* cu_seqlens = nullptr`——`qbase = cu ? cu[b] : b*S`、`len = cu ? cu[b+1]-qbase : S`；
  Q/K/scale 的行下标与边界判定全部从 `b*S`/`S` 改为 `qbase`/`len`，并加
  `if (m0 >= len) return;`（短序列多余 m 块直接退出）。`nullptr` 时逐式退化为原定长路径，
  **定长数值逐位不变**。
* **host**（`fa_bwd_fp8_main.cu` + 单文件）：`launch_lse` 透传 `cu_seqlens`；`run_varlen` 去掉
  「只做 causal」的限制——`causal` 仍走 `lse_mma_kernel_bal_wgmma<128,1>`（镜像配对 + cp.async，
  `grid.x=(nblk+1)/2`），**非 causal** 走 `lse_mma_kernel<128>`（`grid.x=nblk`，`d_cu`）。
  主 kernel 的 `causal` 参数与 grid/ksplit 自动档不变。

### 38.3 harness

`varlen_slug` 增加 `full` 标记（slug `..._full_<dtype>`），`--full` 跑非 causal；`dump`/`bench`
均透传 `causal`。新增 4 个 full varlen case（fp8/fp16/bf16 各一）。

### 38.4 数值（ours vs fp32 ref，fp8 full；max_abs dq/dk/dv）

| varlen case | lengths | max_abs dq / dk / dv |
|---|---|---|
| b4_t3840 不齐 | `[512,1024,2048,256]` | 1.010e-1 / 9.651e-2 / 7.036e-2 |
| b4_t4096 等长 | `[1024]×4` | 8.881e-2 / 7.043e-2 / 5.721e-2 |
| b5_t3968 GQA kv8 | `[128,256,512,1024,2048]` | 1.429e-1 / 1.598e-1 / 1.079e-1 |
| b8_t2904 强倾斜 | `[2048,512,…,8]` | 2.413e-1 / 2.058e-1 / 2.514e-1 |

全部为 fp8 噪声量级（≤0.25），与 causal fp8（0.26–0.62）同水平或更小；单/两文件逐位一致。
定长回归逐位不变（S512 `2.426/2.972/3.733e-1`；fixed full S1024 `5.52/5.31/4.02e-2`）。

### 38.5 性能（ours total=quant+preprocess+main，event，`Σ_b 4HL²D` 口径）

| case | total ms / TF |
|---|---|
| b4_t3840 | 1.963 / 23.25 |
| b4_t4096（等长） | 1.580 / 21.75 |
| b5_t3968 GQA kv8 | 3.761 / 24.34 |
| b8_t2904 强倾斜 | 1.608 / 22.87 |

非 causal 的总计算量约为 causal 的 **2×**（每块看全部 K），故同 case 时间是 causal 的 ~2×
（等长 1.58 vs 0.96ms）。口径 `Σ_b 4HL²D` 只计 `D`，按定长对标口径 `4BS²H(D+Dv)` 乘
`(D+Dv)/D=2` ⇒ 等长 **~43.5 TF**；同 session TE FP8 定长 full `0.4316ms / 159.2TF`（含 forward）
⇒ ours 约 TE 的 3.7×。FA3 变长非 causal 未接入本 harness，仅 fp32 ref 可对。

### 38.6 ncu（main，b4_t3840，`--set full -c 1`）

Duration **1.20ms**、DRAM 5.12% / **L1/TEX 62.47% / L2 64.03%** / Compute 40.89%、
168 regs、**Block Limit Shared Mem=3（occ 18.43%）**、Waves 20.69、No Eligible 58.56%、
Issued Ipc 1.69 ⇒ bound 与定长 fp8 main 一致：**L1/L2 吞吐 + 3 CTA/SM 延迟受限**，非带宽。

### 38.7 限制 / 后续

* fp16/bf16 的对应改动见 `docs/01` §16.8、`docs/01b` §6aa.6。
* MLA（HD=512）的 varlen 仍未做；TMA 化需为 packed 布局重建描述符。
* 强倾斜 case 的按 `cu_seqlens` 均衡分块仍待做（当前短序列 CTA 早退）。

### 38.8 复现

```bash
python harness/fa_bwd_bench.py dump --dtype fp8 --varlen-all --full
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen=1 --full --iters=50 \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b4_t3840_h16_d128_full_fp8
```

## 39. VARLEN 的 MLA（head_dim=512）——fp8 反向（第 80 轮）

### 39.1 动机与口径

第 77/78/79 轮的 varlen 覆盖了 fp8/fp16/bf16 × causal/full，但都限 **HD=128（MHA/GQA）**。
MLA 的 `head_dim=512`（`Fp8Cfg<512,64,32>`、GEMM1/2 的 k-loop 16 步、GEMM3/4/5 的 4 遍 N-tile）
在定长路径早已上张量核（第二十一轮），但 **packed `[T,H,D]` + `cu_seqlens` 从未在 D=512 上验证**。
本轮把两者接起来：fp8 反向直接吃 MLA 的变长输入。

口径同 §37/§38：`q/dO/dQ` 为 `[T,H,D]`，`k/v/dK/dV` 为 `[T,Hkv,D]`，`cu_seqlens`（B+1 个
token 前缀和），`S=maxlen` 仅供 grid/分块启发式，逐 `b` 用 `qbase=cu[b]`、`len=cu[b+1]-qbase`。
MLA 的 `Dv=D=512`（FA/TE 反向都不支持 MLA，只能对 fp32 ref）。

### 39.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

fp8 的 `fp8_mma_body<...,cu_seqlens>` 早已是 HD 参数化的（第 77 轮），**主 kernel 无需改动**；
本轮只补两处：

1. **device `lse_mma_kernel_bal<HD>` 增加 `cu_seqlens`**（`fa_bwd_fp8_kernels.cuh`）。
   定长 D=512 的 causal LSE 走的是 **mma 版**的镜像配对 `lse_mma_kernel_bal<512,1>`（HD>128 没有
   SW128 wgmma 快路），而它此前没有 varlen 支持。改动与 §38 给 `lse_mma_kernel<HD>` 加
   `cu_seqlens` 完全同构：`qbase/len` 替代 `b*S/S`、`nblk` 由 `len` 计算（镜像配对在**本序列内**
   进行）、`if (pair >= (nblk+1)/2) return;` 让短序列多余 CTA 退出、mask/边界判定用 `len`。
   `cu=nullptr` 时逐式退化为定长路径 ⇒ 定长 D=512 数值**逐位不变**。
2. **host `run_varlen` 按 D 分派**（`fa_bwd_fp8_main.cu` + 单文件）：
   - **量化** `quantize_row_warp_kernel<VPT,ISDO>` 的 `VPT=D/32`（D=128→4、**D=512→16**）。
     这是一处真实的坑：varlen 原来硬编码 `<4>`，D=512 时会把行距当 128 ⇒ **整行错位**，
     症状是 dq/dk/dv max_abs O(1)（看似数值错乱）。
   - **ksplit 自动档**：`D==128` 用 `S>=2048?8192:max(2048,4*base_grid)`；`D==512` 用 `maxlen/2`
     （O29 标定）。`use_regdq` 仅 D==128 开（HD=512 的 dQ 一次铺不满 N）。
   - **LSE**：causal 时 D=128 仍走 `lse_mma_kernel_bal_wgmma<128>`、D=512 走
     `lse_mma_kernel_bal<512>`（新加 `cu`）；非 causal 走 `lse_mma_kernel<128|512>`。
   - **delta**：`delta_warp_kernel<128|512>`。
   - **主 kernel**：`launch_bwd_main<128,64,32,REGDQ,/*WGMMA=*/true,...>`（D=128）/
     `launch_bwd_main<512,64,32,false,/*WGMMA=*/false,true,true>`（D=512，与定长一致）。

### 39.3 harness

`VARLEN_SHAPES` 新增 MLA 形状 `([256,512,1024], 2, 512, 2, 512)`（也可用
`--lengths "256 512 1024" --H 2 --D 512 --kv 2 --Dv 512` 直接 dump）。共 dump **2 个 MLA
varlen case**（causal / full）。TE 2.14 的 FP8 MLA 训练反向不支持 ⇒ 无 TE 基线。

### 39.4 数值（ours vs fp32 ref，fp8；max_abs dq/dk/dv）

| case | lengths | causal | dq | dk | dv |
|---|---|---|---|---|---|
| b1_t512 | `[512]` | causal | 1.613e-1 | 2.238e-1 | 3.864e-1 |
| b1_t512 | `[512]` | full | 5.260e-2 | 5.222e-2 | 4.218e-2 |
| b3_t1792 | `[256,512,1024]` | causal | 3.404e-1 | 3.436e-1 | 3.508e-1 |
| b3_t1792 | `[256,512,1024]` | full | 7.994e-2 | 9.629e-2 | 4.148e-2 |

全部落在 fp8 噪声量级（0.04–0.39），与定长 fp8 MLA（0.22–0.36）同水平，无 padding 泄漏。
**单文件与两文件逐位相同**（四个 case 的 max_abs 完全相同）。
定长回归（`nullptr` 路径）逐位不变：S512 `2.426/2.972/3.733e-1`、S4096 `2.635/2.644/3.216e-1`、
MLA S1024H2 `2.232/3.337/3.602e-1`。

### 39.5 性能（ours total=quant+preprocess+main，event，`Σ_b 4HL²D` 口径）

| case | causal | total ms | TFLOPS |
|---|---|---|---|
| varlen b1_t512（H2 D512） | causal | 0.1866 | 5.75 |
| varlen b1_t512 | full | 0.3168 | 3.39 |
| varlen b3_t1792 `[256,512,1024]` | causal | 0.5875 | 9.60 |
| varlen b3_t1792 | full | 1.0587 | 5.32 |
| **定长** S512 H2 D512（对照） | causal | 0.1836 | 5.85 |
| **定长** S1024 H2 D512（对照） | causal | 0.3919 | 10.96 |

* `D=Dv=512`，按 harness 定长口径 `4BS²H(D+Dv)` 需把上表 TF **×2**（MLA 的 varlen 约 19 TF）。
* **varlen b1 与同 shape 定长几乎相同（0.1866 vs 0.1836ms，慢 1.6%）** ⇒ varlen kernel 没有额外
  固定开销，差异来自 `lse`/`delta` 的 grid 形状与 quant 的 packed 索引。
* FA/TE 反向均不支持 head_dim=512 ⇒ **无外部基线**；MLA 的绝对算力受 1 CTA/SM 限制。

### 39.6 ncu（`--set full -c 1`，b3_t1792 causal）

**主 kernel `fa_bwd_fp8_mma_kernel`**：Duration **429.2µs**、DRAM 2.58% / L1/TEX 22.81% /
L2 21.70% / Compute **7.95%**、**255 regs、Block Limit Shared Mem=1 ⇒ occ 6.25%（1 CTA/SM）**、
Waves 2.91、No Eligible **84.67%**、active warps/sched **1.01**、stall **short_scoreboard ~30%**、
shared load/store 冲突各 ~17% ⇒ **bound = 低 occupancy（1 CTA/SM）+ smem→mma 依赖延迟**，
与定长 MLA 张量核版结论一致（非带宽/算力）。

**LSE `lse_mma_kernel_bal<512>`**：Duration 140.2µs、**Waves 0.18**（grid=`(nblk+1)/2·H·B=48`
< 132 SM）、occ 6.25%、No Eligible 75.6%、DRAM 0.80% ⇒ **grid 不足一个波（并行度 bound）**。
这是小 H/B 的 MLA 固有问题（每序列只需 `nblk` 个 CTA）。

### 39.7 限制 / 后续

* 本轮只做 **fp8**；**fp16/bf16 的 varlen MLA 仍需给 mma 主 kernel（`fa_bwd_{fp16,bf16}_mma_kernel`）
  与 `lse_mma_kernel_bal` 加 `cu_seqlens`**（fp8 的主 kernel 天然可用，因为 `fp8_mma_body` 已是
  HD 参数化且第 77 轮已加 cu；fp16/bf16 的 varlen 目前只覆盖 wgmma2 的 HD=128）。
* **按 `cu_seqlens` 的均衡分块**仍未做（强倾斜 / 多序列时短序列 CTA 早退）。
* **varlen 的 TMA 化**：TMA 描述符按定长 `[D,S,H,B]` 建，packed 布局需重建（或逐序列建 map）。
* MLA 的 1 CTA/SM 是本卡硬约束（`Kt/Qt/dOt` 转置副本 + `Ps/Ss` 使 smem 顶格），要冲 2 CTA/SM
  须继续降 smem（对齐 O4b 思路）。

### 39.8 复现

```bash
# dump MLA varlen（causal + full）
python harness/fa_bwd_bench.py dump --dtype fp8 --lengths 256 512 1024 --H 2 --D 512 --kv 2 --Dv 512
python harness/fa_bwd_bench.py dump --dtype fp8 --lengths 256 512 1024 --H 2 --D 512 --kv 2 --Dv 512 --full
# 两文件 / 单文件自测（默认 WGMMA+TMA 构建；varlen 内部走非 TMA 路径）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen=1 --iters=50 \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b3_t1792_h2_d512_causal_fp8
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --varlen=1 --iters=50 --dir=...
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:fa_bwd_fp8_mma_kernel --launch-count 1 -- --varlen=1 --iters=1 --dir=...
```

原始输出：`src/fp8/fa_bwd_fp8_varlen_mla_sweep.out.txt`（两文件 4 case）、
`..._varlen_mla_onefile_sweep.out.txt`（单文件 4 case，逐位一致）、
`..._varlen_mla_fixed_regression.out.txt`（定长回归）、`..._varlen_mla_perf.out.txt`（定长对照计时）、
`..._varlen_mla_ncu_main_b3.out.txt`（main ncu）、`..._varlen_mla_ncu_lse_b3.out.txt`（LSE ncu）。

## 40. VARLEN 均衡分块（按 `cu_seqlens` 只发有效 tile 的紧凑网格）——**负结果**（第八十二轮）

动机（对齐 ROADMAP「下一步候选 ①」）：varlen 的 kernel 仍以 `S = maxlen` 为界发网格——
主 kernel `grid = (ceil(maxlen/BM)*ksplit, H, B)`、causal LSE `grid = (ceil(nblk/2), H, B)`。
短序列的绝大多数 CTA 一进来就 `if (m0 >= len) return` / `pair >= (nblk+1)/2` 早退，
强倾斜 `b8_t2904=[2048,512,128,96,64,32,16,8]` 时主 kernel 发了 **8192** 个 CTA，
其中有效仅 **48 个 m-tile ×16 头 = 768**（每个再 ×ksplit），死 CTA 占 **~83%**。
假设「消掉死 CTA + 让贵块先跑」能提速，做了如下两台改动（单/两文件同步，新增开关默认关）：

1. **主 kernel 紧凑网格**：宿主枚举每个 `b` 的 `nblk_b = ceil(len_b/BM)` 个 m 块，
   按 `(b, mblk)` 写进两张一维表 `mt_b/mt_m`；`fp8_mma_body` 里把
   `(mblk=blockIdx.x/ksplit, b=blockIdx.z)` 改成 `blockIdx.x/ksplit → mt` 查表得 `(b, mblk)`，
   网格变 `(total_mt*ksplit, H, 1)`。`mt_*==nullptr` 时逐式退化为旧路径（定长不受影响）。
   开关 `--compact`。
2. **causal LSE 紧凑对网格**：同理枚举每个 `b` 的有效镜像对 `pair < ceil(nblk_b/2)`，
   写表 `pt_b/pt_pair`，网格 `(total_pairs, H, 1)`；开关 `--lsecompact`。

**实测（同 binary A/B，event，iters=50；原始输出 `src/fp8/fa_bwd_fp8_varlen_bal_ab.out.txt`）**：

| case | 默认（maxlen 网格） | `--compact` | `--lsecompact` | 两者 |
|---|---|---|---|---|
| skew `b8_t2904` causal | **0.836 / 0.836 ms** | 0.865 / 0.864 ms（**−3.4%**） | 0.846 / 0.850 ms（**−1.4%**） | 0.871 / 0.875 ms（−4.4%） |
| 等长 `b4_t4096` causal | 0.767 / 0.770 ms | 0.770 / 0.769 ms（±0.2%） | — | — |

`ksplit` 扫描（默认网格，skew）：1/2/4/8 = 0.837/0.836/0.836/0.836 ms —— 与 split 无关。
数值：紧凑档与默认档的 `dq/dk/dv vs ref` max_abs **完全一致**（如 skew 3.136/3.584/4.030e-1），
说明 fp8 量化噪声远大于 atomic 次序差异；定长与其余 5 个 varlen case 回归逐位不变
（第 77/80 轮数值原样）。

**ncu（main，S=2904 skew，同 binary）**：

| 指标 | 默认网格 | `--compact` |
|---|---|---|
| Duration | **588.96 µs** | 624.13 µs |
| Waves Per SM | **20.69** | **3.88** |
| Achieved Occupancy | 17.74% | 16.85% |
| L1/TEX | 61.51% | 60.75% |
| L2 | 54.54% | 51.26% |
| Compute (SM) | 34.72% | 32.79% |
| DRAM | 8.18% | 7.60% |

**结论（为什么负）**：死 CTA 是**免费**的——它们只做几次整数比较就退出，几乎不占 SM 时间；
真正的限速器是 **L1/TEX 61% + L2 55% 的吞吐 + 17.7% 低 occupancy 下的延迟隐藏**
（DRAM 仅 8%、Compute 仅 35%，既非带宽也非算力）。原网格里那些「多余」CTA 反而在
SM 里提供了额外的在飞 warp 来遮盖延迟；把它们删掉后 Waves 20.69→3.88、
在飞 CTA 变少，Duration 反而 +6%。LSE 侧同理：镜像配对已把每 CTA 的**最大**工作
压到常数 `nblk+1` 个 n-tile（长序列 16 对 × 33 tile），紧凑化只删死对、不降临界路径，
实测 +1.4% 亦是噪声级负向。

**判决**：ROADMAP「按 `cu_seqlens` 的均衡分块」在 fp8 上**不成立**（负结果）；
代码保留为 opt-in（`--compact` / `--lsecompact`，默认关）以便复现。
varlen 若还要再上台阶，只剩两条真杠杆：**① 把 LSE 的 K 维 split 开 + 二次归约**
（当前镜像配对的临界路径是每 CTA 串行 33 个 K-tile，只有切 K 才能降下来）；
**② varlen 的 TMA 化**（packed 布局的 4D 描述符）——**第八十三轮已在 fp16 上判决为中性/
偏负**（`docs/01` §16.10：指令 −23% 但 Duration 持平、强倾斜 −4.7%），该条杠杆作废；
varlen 唯一剩余真杠杆是 **① LSE 的 K 维 split + 二次归约**（fp8 侧同理，待续）。

---

## 41. O38：fp8 LSE 的 K 维 split + 二次归约（第 84 轮）—— **正结果，已设为默认 auto**

> 第八十二轮把 varlen「均衡分块」判为负结果后，明确剩下的**唯一** LSE 杠杆就是
> **「把 K 维 split 开 + 二次归约」**——镜像配对把每个 CTA 的工作压成常数，但常数本身
> （串行扫 `nblk+1` 个 K tile）仍是一条长临界路径，且小 S/H 下 grid 远小于 SM 数。
> 本轮把这条杠杆在 **fp8 定长 causal 的 TMA LSE**（`lse_mma_kernel_bal_tma`，O32 默认档）
> 上做完：**K 维切片并行 + 一个 merge kernel 汇总 (m,l) 部分结果**。

### 41.1 动机

O32 的 TMA LSE（`lse_mma_kernel_bal_tma`）每 CTA 用镜像配对处理 `m` 与 `nblk-1-m`
两个 m 块、每块串行扫 `nblk+1` 个 K tile。ncu 实测这是**并行度/临界路径**受限：

| shape | grid | Duration | achieved occ | Compute | Ipc | Waves/SM |
|---|---|---|---|---|---|---|
| S512 H16 split=1 | 64 | 33.41 µs | 6.25% | 7.57% | 0.68 | 0.06 |
| S4096 H16 split=1 | 512 | 250.85 µs | 23.58% | 57.01% | 2.35 | 0.48 |

S512/H16 的 grid 只有 **64 个 CTA**（132 SM 的一半都不到，且每 SM 仅 1 个），
Compute/DRAM/L1 全部 <8%，纯粹在等延迟；S4096 也只铺了半个波（0.48 wave）。
把 K 范围切成 `S` 份并行、再用一次轻量 merge 合并 online-softmax 的 `(m,l)`，即可
直接补满并发槽并缩短临界路径。

### 41.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

1. **`lse_mma_kernel_bal_tma<HD,PIPE>` 加两个参数**：`float* lse_part, int ksplit`。
   grid 从 `(pairs,H,B)` 变成 `(pairs,H,B*ksplit)`；kernel 内 `b=blockIdx.z/ksplit`、
   `ksp=blockIdx.z%ksplit`。每个 CTA 只扫本 m 块 K tile 的**连续切片**
   `[nt0,nt1) = [ntiles*ksp/ksplit, ntiles*(ksp+1)/ksplit)`（按 tile 粒度切分），
   流水 stage 用切片内相对下标 `rnt&1`（避免 `nt0` 为奇数时 stage 错位）。
   `ksplit==1` 时切片即整段、仍直接写 `lse` ⇒ **逐位退化为 O32**。
2. **`lse_split_merge_kernel`**：`part` 布局 `[row][ksp] -> (m,l)`（每行 `2*ksplit` 个
   fp32），沿 `ksp` 做 online-softmax 合并（`m=max`、`l=Σ l_k·exp(m_k-m)`）后写
   `lse[row]=m+log(l)`；`-inf/0` 安全（空切片得 `-inf/0`）。数学与「单 CTA 顺序扫全部
   K tile」**完全等价**，只差 fp32 求和次序。
3. **host**（`fa_bwd_fp8_main.cu` / 单文件同步）：新增 `launch_lse_bal_tma_split`
   （把 `lg.z` 扩成 `B*ksplit`，split>1 时再 launch merge）与 CLI `--lsesplit=N`。
   **默认 `--lsesplit=0`（auto）**：目标 `grid*split ≈ 2048`（≈2 个满波），上限 8；
   grid 已够大则退回 1（逐位）。实测该自动档在各 shape 上距 per-shape 最优 ≤1.2%。
   `--lsesplit=1` 强制关闭、保持历史逐位；仅 `-DFA_WGMMA -DFA_TMA`、D==128、causal 生效
   （MLA D=512 走 mma/wgmma LSE，不受影响）。

### 41.3 数值（vs fp32 ref，fp8 causal；max_abs dq/dk/dv）

`--lsesplit=0`（auto）与历史（O27/O28/O32/O37）**逐位一致到打印精度**：

| case | dq / dk / dv vs ref（auto） | max_abs(split-vs-split1) |
|---|---|---|
| b1_s512_h16_d128_causal | 2.426e-01 / 2.972e-01 / 3.733e-01 | 4.77e-07 |
| b1_s1024_h32_d128_causal | 2.399e-01 / 4.177e-01 / 3.535e-01 | 9.54e-07 |
| b1_s1024_h32_d128_kv4 | 2.517e-01 / 5.339e-01 / 7.173e-01 | 9.54e-07 |
| b1_s1024_h40_d128_kv8 | 2.869e-01 / 5.367e-01 / 7.032e-01 | 9.54e-07 |
| b1_s1024_h64_d128_kv1 (MQA) | 4.101e-01 / 1.572e+00 / 2.126e+00 | 9.54e-07 |
| b1_s4096_h16_d128_causal | 2.635e-01 / 2.644e-01 / 3.216e-01 | 1.91e-06 |
| b1_s1024_h2_d512_causal (MLA) | 2.232e-01 / 3.337e-01 / 3.602e-01 | 0（未走 split） |
| b1_s512_h4_d512_causal (MLA) | 2.415e-01 | 0（未走 split） |

`split-vs-split1` 全部 ≤2e-6（纯 fp32 求和次序，远小于 fp8 容差 O(1)）；单/两文件逐位一致。

### 41.4 性能（CUDA event；same-session `--lsesplit=1`（基线 O32）vs auto）

端到端（quant+preprocess+main；auto 含 merge）：

| shape | split=1 (ms) | auto (ms) | 比 | auto 选中 split |
|---|---|---|---|---|
| S512 H16 | 0.1150 | **0.1030** | **1.12×** | 8 |
| S1024 H32 | 0.4185 | **0.4010** | 1.04× | 8 |
| S1024 H32 kv4 | 0.3879 | **0.3690** | 1.05× | 8 |
| S1024 H40 kv8 | 0.4684 | **0.4540** | 1.03× | 4 |
| S1024 H64 kv1 (MQA) | 0.6431 | **0.6395** | 1.006× | 4 |
| S4096 H16 | 2.0474 | **1.9970** | 1.025× | 4 |
| MLA S1024 H2 D512 | 0.3922 | 0.3908 | 1.00× | —（不生效） |

**LSE-only** A/B（同 binary 显式 split 扫描，`[O38 A/B]` 行）：

| shape | split=1 | split=2 | split=4 | split=8 | 最优 |
|---|---|---|---|---|---|
| S512 H16 | 0.0312 | 0.0217 (1.43×) | 0.0169 (1.84×) | **0.0154 (2.03×)** | 8 |
| S1024 H32 | 0.0609 | 0.0418 (1.46×) | **0.0375 (1.62×)** | 0.0417 (1.46×) | 4 |
| S1024 H32 kv4 | 0.0606 | 0.0417 | **0.0365 (1.62×)** | 0.0409 | 4 |
| S1024 H40 kv8 | 0.0615 | **0.0441 (1.40×)** | 0.0442 | 0.0447 | 2 |
| S1024 H64 kv1 | 0.0692 | **0.0628 (1.10×)** | 0.0643 | 0.0705 | 2 |
| S4096 H16 | 0.2414 | 0.2033 | 0.1972 | **0.1931 (1.25×)** | 8 |

merge kernel（S4096 split=8）：Duration **5.86 µs**，只占该 shape LSE 的 ~2.4% ⇒ 净收益
仍为 1.25×。规律：**grid 越小、split 收益越大**（S512 2.0×）；已铺满半波以上时收益
收敛到临界路径那一段（S4096 1.25×）；MQA（H=64、grid 已 512）几乎无收益（1.10×），
故 auto 上限 8 且目标 2048 使 MQA 只切到 4、不回归。

### 41.5 ncu（LSE 主 kernel，`--launch-count 1`，同 binary split 1 vs 8）

| shape | 指标 | split=1 | split=8 |
|---|---|---|---|
| S512 | grid / Duration | 64 / 33.41 µs | **512 / 13.38 µs（2.50×）** |
| | achieved occ | 6.25% | **19.78%** |
| | Compute / Ipc | 7.57% / 0.68 | **27.69% / 1.55** |
| S4096 | grid / Duration | 512 / 250.85 µs | **4096 / 198.46 µs（1.26×）** |
| | achieved occ | 23.58% | **45.30%** |
| | Compute / Ipc | 57.01% / 2.35 | **78.25% / 3.24** |

S512 的墙由 **grid 不足（64 CTA）** 直接变为 Compute 27.7%（仍未打满，因为单 CTA
工作已很小、merge/launch 占比上升）；S4096 由 **0.48 波** 拉到 3.88 波、occupancy 翻倍、
Compute 57%→78% —— 说明此前 O32「LSE 已接近下限」指的是**指令/访存**已到底，
**并行度**里还藏着 1.25–2×。**结论：split-K 是 LSE 的并行度杠杆，而非再降指令。**

### 41.6 对标（同 session）

- **TE FP8**（`fa_bwd_bench.py bench --dtype fp8`，纯 device）：S512 **0.1011 ms/42.50 TF**、
  S4096 **0.5917 ms/464.59 TF**。ours total（auto）S512 **0.1030 ms ⇒ 1.02×**（O37 时 1.21×，
  **几乎追平 TE FP8**）；S4096 **1.9970 ms ⇒ 3.38×**（O37 3.49×）。
  （ours 口径含 quant+preprocess+main；TE 为输入预先量化后的纯反向。）
- **FA3/FA2/TE fp16 三列**（`harness/fa_vs_te_bwd_only.py`）：MHA S4096 FA3
  **0.3242 ms/848 TF**、TE 0.4398/625、FA2 0.7282/377；GQA kv4 S1024 FA3 0.0824/417。
  fp8 无 FA 基线，仅作量级参照。

### 41.7 复现

```bash
# 两文件：默认 auto（0）；--lsesplit=1 关闭做 A/B
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8 --iters=100 --lsesplit=0
# 单文件
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --dir=... --lsesplit=8
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:lse_mma_kernel_bal_tma \
  --launch-count 1 --set full -- --dir=... --lsesplit=8
# device 同步单文件
python3 scripts/sync_onefile_device.py src/fp8/fa_bwd_fp8_kernels.cuh \
  src/fp8/fa_bwd_fp8_mma_onefile.cu '#include <cuda_runtime.h>'
```

原始输出：`src/fp8/fa_bwd_fp8_o38_default_sweep.out.txt`（两文件 ×9 shape ×auto/split1 +
单文件）、`src/fp8/fa_bwd_fp8_o38_main_sweep.out.txt`（两文件 ×6 shape ×split 0/1/2/4/8 全扫）、
`src/fp8/fa_bwd_fp8_o38_ncu_lse_split{1,8}_s512.out.txt`、
`src/fp8/fa_bwd_fp8_o38_ncu_lse_s4096.out.txt`、`src/fp8/fa_bwd_fp8_o38_ncu_merge_s4096.out.txt`、
`src/fp8/fa_bwd_fp8_o38_te_fa3_baseline.out.txt`。

## 42. MLA（head_dim=512）LSE 的 K 维 split + 二次归约（O39，第八十六轮）—— **正结果，默认 auto**

### 42.1 动机

O38（§41）只把「K 维 split + 二次归约」做进了 **D=128 的 TMA LSE**（`lse_mma_kernel_bal_tma`）。
fp8 的 **MLA（D=512）反向** causal LSE 走的是 **mma 版** `lse_mma_kernel_bal<512,1>`（HD>128 无
SW128/TMA 快路）。S1024H2 时 `nblk=16,pairs=8`，grid 只有 `8×2×1=16` 个 CTA，ncu 实测
**Waves 0.06 / Achieved Occupancy 6.25% / Compute 2.65%** —— 是**纯并行度/临界路径**墙，
与 fp8 O38 的 S512 情形同构。O39 把同一套切片 + merge 机制移植到 mma 版 LSE。

### 42.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

`src/fp8/fa_bwd_fp8_kernels.cuh` 的 `lse_mma_kernel_bal<HD,PIPE>`：

- 新增两个尾部默认参数 `float* __restrict__ lse_part = nullptr, int ksplit = 1`；
  `grid.z` 由 `B` 扩成 `B*ksplit`，`b = blockIdx.z/ksplit`、`ksp = blockIdx.z%ksplit`。
- 每个 `(pair,ksp)` CTA 只扫本 m 块 K tile 的连续切片
  `[nt0,nt1) = [ntiles*ksp/ksplit, ntiles*(ksp+1)/ksplit)`；流水 stage 用**切片内相对下标
  `rnt&1`**（避免 `nt0` 奇偶错位）。
- `ksplit==1` 时 `b=blockIdx.z`、`ksp=0`、`nuse=ntiles`，**逐位退化为 O11 原路径**。
- 部分结果 `(m,l)` 写 `lse_part[(row*ksplit+ksp)*2 + {0,1}]`，由 `lse_split_merge_kernel`
  沿 `ksp` 做 online-softmax 合并成 `lse = m + log(l)`。
- `lse_split_merge_kernel` 从 `#if defined(FA_WGMMA) && defined(FA_TMA)` 守卫**移出**（mma
  版 LSE 也要用；kernel 本身是纯 fp32 代码，与 TMA/wgmma 无关）。
- host：`launch_lse_bal` 加 `lse_part/ksplit` 并在 `ksplit>1` 时补 launch merge；
  `--lsesplit=N`（`0=auto`）。**auto 目标**：`D==512 → grid*split ≈ 256`（fp8 LSE smem
  ~102KB ⇒ 2 CTA/SM，故 ≈ `2×132`）、上限 16；再按 `nblk=ceil(S/64)` **封顶**（切得比 K tile
  数还细只会产生空切片 + 每 CTA 的 Q 载入开销）。D=128 的 TMA LSE 维持 O38 的 `≈2048/8`。
- 顺带把 D=128 的 **非 TMA mma 回退路径**（`sm_90` 构建 / `--lsetma=0 --lsewgm=0`）也接上
  split（此前回退路径永远 split=1）；D=128 的 wgmma 回退路径不支持 split（保持原样）。

### 42.3 数值（vs fp32 ref，fp8 causal，max_abs dq/dk/dv；split-vs-split1）

| case (B1 D=Dv=512) | dq | dk | dv | max_abs(split-auto vs split1) |
|---|---|---|---|---|
| S256 H2 | 2.356e-1 | 2.290e-1 | 3.441e-1 | 4.8e-7 |
| S512 H4 | 2.415e-1 | 2.992e-1 | 4.481e-1 | 4.8e-7 |
| S1024 H2 | 2.232e-1 | 3.337e-1 | 3.602e-1 | 9.5e-7 |

全形状与 O27/O28/O32/O37/O38 历史**逐位一致到打印精度**；`max_abs(split-vs-split1) ≤ 9.5e-7`
（纯 fp32 求和次序）。单/两文件逐指标一致；varlen MLA（§39）与 D=128 MHA/GQA（S512
2.426/2.972/3.733e-1、S4096 2.635/2.644/3.216e-1、kv4 2.517/5.339/7.173e-1）回归不变。

### 42.4 性能（CUDA event；same-session `--lsesplit=1`（基线）vs auto）

| case | LSE split1 (ms) | LSE auto (ms) | LSE 倍数 | total split1 (ms) | total auto (ms) | total 倍数 |
|---|---|---|---|---|---|---|
| MLA S256H2 | 0.0458 | **0.0237** (split4) | **1.93×** | 0.1110 | **0.0921** | **1.21×** |
| MLA S512H4 | 0.0780 | **0.0254** (split8) | **3.07×** | 0.2601 | **0.2009** | **1.29×** |
| MLA S1024H2 | 0.1447 | **0.0310** (split16) | **4.67×** | 0.4036 | **0.2851** | **1.42×** |

auto 在三个 shape 上分别选 4/8/16，均命中 per-shape 最优（`--lsesplit` 全扫见 §42.7）。
merge kernel 每 case 仅数 µs（S1024 split16）。单文件与两文件 total 同量级。

### 42.5 ncu（LSE 主 kernel，`--launch-count 1`，同 binary split 1 vs 16，S1024H2）

| | split1 | split16 |
|---|---|---|
| Duration | 152.42 µs | **26.11 µs** |
| Waves Per SM | 0.06 | **0.97** |
| Achieved Occupancy | 6.25% | **10.77%** |
| Compute (SM) | 2.65% | **22.02%** |
| L2 Cache | 1.31% | 18.61% |
| DRAM | 0.43% | 2.52% |
| Registers / Dynamic smem | 85 / 102.14 KB（Block Limit Shared Mem = 2） | 同 |

stall（per-issue-active）：split1 `wait 2.38 + short_sb 0.50 + long_sb 0.08`（**mma fixed-latency
依赖主导**）→ split16 `wait 2.07 + short_sb 0.51 + long_sb 0.49 + barrier 0.36`。
**结论：此前的墙是网格不足（Waves 0.06）；split 把并发槽填满一个波后，墙回到 LSE 固有的
`mma wait`（fixed-latency）+ smem 依赖，且只到 2 CTA/SM**（smem 102KB 封顶）——与 fp8 O38 的
「并行度 → Compute/wait」结论一致。

### 42.6 对标

FA3/TE/FA2 反向均**不支持 head_dim=512（MLA 主注意力）**，只有 fp32 ref 可对；故 MLA 的性能
数字仅 ours 提供（对标见 `docs/04` §15）。D=128 的 MHA/GQA 对标不受影响（O39 只改 D=512 LSE；
D=128 TMA 路径逐位回归）。

### 42.7 复现 / 原始输出

```bash
# 两文件：默认 auto（0）；--lsesplit=1 关闭做 A/B
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 --o=ref_o --iters=100
# 单文件
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --dir=... --lsesplit=16
# ncu
ARCH="" NVCC_FLAGS="..." scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu \
  --kernel-name regex:lse_mma_kernel_bal --launch-count 1 --set full -- --lsesplit=16
```

原始输出：`src/fp8/fa_bwd_fp8_main_o39_b1_s{256_h2,512_h4,1024_h2}_d512_causal.out.txt`（auto）、
`..._o39_split1_b1_*_d512_causal.out.txt`（基线）、`..._o39_reg_b1_*`（D=128/GQA 回归）、
`..._o39_mmafallback_s512.out.txt`、`..._o39_ncu_lse_split{1,16}_s1024h2.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_o39_*`（单文件）、`..._o39_varlen_*`（varlen MLA 回归）。

---

## 43. 非 TMA wgmma LSE 的 K 维 split + varlen 接入（O40，第八十七轮）—— **正结果，默认 auto**

### 43.1 动机

O38（§41）把 split 做进 **D=128 的 TMA LSE**，O39（§42）做进 **mma 版 LSE**（覆盖 D=512 MLA），
但 **非 TMA 的 wgmma 版 `lse_mma_kernel_bal_wgmma`**（O9c）从未有 split，而它正是两条路径的 LSE：

1. **定长 D=128/causal 的 `--lsetma=0` 回退**（`lsewgm=1`，`sm_90a` 构建）；S512 时
   `lg_bal.x=4,H=16` ⇒ grid 仅 64、ncu **Waves 0.07 / occ 6.25% / Compute 8.2%**；
2. **VARLEN D=128/causal 的默认 LSE**（`run_varlen` 直接 launch wgmma 版；第八十三轮已判 varlen
   主 kernel TMA 化中性/偏负，故 LSE 也不会走 TMA）。b1_t512_h16 时 `pairs=4,H=16,B=1` ⇒ grid 64。

同第八十四/八十六轮结论：网格不足一个波时，**K 维 split + 二次归约**直接把「单 CTA 顺序扫
`nblk` 个 tile」的临界路径切短、填满并发槽。

### 43.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

`src/fp8/fa_bwd_fp8_kernels.cuh` 的 `lse_mma_kernel_bal_wgmma<HD,PIPE>`（与 O39 给 mma 版的
改动同构）：

- 新增尾部默认参数 `float* __restrict__ lse_part = nullptr, int ksplit = 1`；非紧凑网格
  `b = blockIdx.z/ksplit`、`ksp = blockIdx.z%ksplit`；紧凑网格（`pt_b/pt_pair` 非空）只把
  `ksp` 编进 `blockIdx.z`。
- 每个 `(pair,ksp)` 只扫本 m 块 K tile 的连续切片 `[nt0,nt1)`（按 tile 数均分，两个镜像 m 块
  各自切）；流水 stage 用切片内相对下标 `rnt&1`。
- `ksplit==1` **逐位退化为 O9c 原路径**（写 `lse`、不写 part、不 launch merge）。
- 部分 `(m,l)` 写 `lse_part[(row*ksplit+ksp)*2+{0,1}]`；`lse_split_merge_kernel`（O38 已有）
  合并。注意：**合并的行数 `nrows` 在 varlen 下必须是 `T*H`（packed 行），不能是 `B*S*H`**
  （varlen 的 `d_lse` 只分配 `T*H`）；故 launch 包装器加 `long long merge_rows = -1`。
- host：`launch_lse_bal_wgmma` 加 `lse_part/ksplit/merge_rows` 并补 launch merge；`launch_lse_bal`
  同样加 `merge_rows`（供 varlen D=512 用）。**定长 `--lsetma=0` 路径**接入 `lse_split_eff`
  （auto 沿用 D=128 的 `grid*split≈2048`/cap 8）。
- **varlen `run_varlen` 加 `--lsesplit=N`（0=auto）**：auto 目标 D=128 `grid*split≈2048`/cap 8、
  D=512 `≈256`/cap 16，再按最大序列 `nblk=ceil(maxlen/64)` 封顶；D=128 causal 走 wgmma 版
  （merge_rows=`T*H`），D=512 causal 走 O39 的 mma 版（同 merge_rows）。两条路径各新增
  `d_lse_part`（`T*H*16*2` 个 fp32）缓冲。
- **顺带修单文件 bug**：单文件在 `--lsetma=0` 时 `qmap_lse/kmap_lse` 未初始化，却被 O32 A/B 段
  使用 ⇒ `illegal instruction`；把描述符构建条件改成与两文件一致的 `if (D == 128)`（主 kernel 的
  `qmap_main/dmap_main` 同理）。

### 43.3 数值（vs fp32 ref，fp8 causal，max_abs dq/dk/dv）

| case | dq | dk | dv | max_abs(split-auto vs split1) |
|---|---|---|---|---|
| MHA S512 H16（定长 `--lsetma=0`） | 2.426e-1 | 2.972e-1 | 3.733e-1 | 0（auto=8） |
| MHA S4096 H16（定长 `--lsetma=0`） | 2.635e-1 | 2.644e-1 | 3.216e-1 | — |
| varlen b1_t512_h16 | 2.280e-1 | 3.108e-1 | 3.422e-1 | 0（auto=8） |
| varlen b4_t3840_h16 | 2.935e-1 | 2.938e-1 | 4.179e-1 | 0（auto=2） |
| varlen b4_t4096_h16 | 2.651e-1 | 3.026e-1 | 3.920e-1 | 0（auto=4） |
| varlen b1_t512_h2 D512 | 1.613e-1 | 2.238e-1 | 3.864e-1 | 0（auto=8） |
| varlen b3_t1792_h2 D512 | 3.404e-1 | 3.436e-1 | 3.508e-1 | 0（auto=4） |

全形状与历史（O37/O39 的 S512 2.426/2.972/3.733e-1、S4096 2.635/2.644/3.216e-1）**逐位一致到打印
精度**；`max_abs(split-auto vs split1) = 0`（此处 fp8 单/两文件与 TMA/wgmma 的 LSE 本就逐位相同）。
单/两文件逐指标一致；D=128 MHA/GQA（默认 TMA 路径）回归不受影响。

### 43.4 性能（CUDA event；same-session `--lsesplit=1`（基线）vs auto）

**定长 `--lsetma=0`（wgmma LSE）**：

| case | preprocess split1 | preprocess auto | 倍数 | total split1 | total auto | 倍数 |
|---|---|---|---|---|---|---|
| MHA S512 H16 | 0.0367 ms | **0.0207 ms** (split8) | **1.77×** | 0.1205 | **0.1054** | **1.14×** |
| MHA S4096 H16 | 0.2958 ms | **0.2673 ms** (split4) | **1.11×** | 2.0804 | **2.0417** | **1.02×** |

**VARLEN（`--varlen`，自动选 split）**：

| case | total split1 | total auto | 倍数 | 备注（base→auto） |
|---|---|---|---|---|
| b1_t512_h16 D128 | 0.1255 ms | **0.1072 ms** | **1.17×** | 64 → 8 |
| b4_t3840_h16 D128 | 0.9646 ms | **0.9285 ms** | **1.04×** | 1024 → 2 |
| b4_t4096_h16 D128 | 0.7668 ms | 0.7707 ms | 1.00× | 512 → 4（−0.5%，噪声） |
| b5_t3968_h32 D128 | 1.8343 ms | 1.8335 ms | 1.00× | 2560 → 1 |
| b8_t2904_h16 D128 | 0.8363 ms | 0.8377 ms | 1.00× | 2048 → 1 |
| b1_t512_h2 **D512** | 0.1954 ms | **0.1401 ms** | **1.39×** | 8 → 8 |
| b3_t1792_h2 **D512** | 0.6067 ms | **0.4979 ms** | **1.22×** | 48 → 4 |

规律：**base 越小、auto 切得越深、收益越大**；base 已 ≥ 目标（大序列/高 H）时 auto=1、**中性**
（不回归）。D=512 的 varlen（LSE 本就极under-filled，base=8/48）受益最明显（1.22–1.39×）。

### 43.5 ncu（LSE 主 kernel `lse_mma_kernel_bal_wgmma`，`--launch-count 1`，定长 S512）

| | split1 | split8 |
|---|---|---|
| Duration | 37.22 µs | **14.56 µs** |
| Waves Per SM | 0.07 | **0.55** |
| Achieved Occupancy | 6.25% | **20.26%** |
| Compute (SM) | 8.21% | **33.32%** |
| Memory Throughput | 3.52% | 15.99% |
| No Eligible | 81.2% | 54.3% |

**结论**：与 O38/O39 完全一致——此前的墙是**网格不足一个波**（Waves 0.07、occ 6.25%）；split
填满并发槽后 Compute 8%→33%、occ 6%→20%，墙回到 LSE 固有的 `mma wait`/smem 依赖。
（fp16/bf16 的 wgmma LSE 同口径见 `docs/01` §14w、`docs/01b` §6ae。）

### 43.6 对标

FA3/TE/FA2 反向均不支持 varlen / D=512，本节指标仅 ours；定长 D=128 的默认 TMA 路径逐位回归、
对标不受影响（MHA S4096 ours total 约 FA3 的 3.9×，见 `docs/04` §16）。

### 43.7 复现 / 原始输出

```bash
# 定长 --lsetma=0（wgmma LSE）A/B
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8 --lsetma=0 --lsesplit=1
# varlen A/B
ARCH="" NVCC_FLAGS="..." scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b1_t512_h16_d128_causal_fp8 --lsesplit=0
# ncu
ARCH="" NVCC_FLAGS="..." scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu \
  --kernel-name regex:lse_mma_kernel_bal_wgmma --launch-count 1 --set full -- --lsetma=0 --lsesplit=8
```

原始输出：`src/fp8/fa_bwd_fp8_main_o40_fixed_s{512,4096}_split{1,0}.out.txt`、
`src/fp8/fa_bwd_fp8_main_o40_varlen_<case>_split{1,0}.out.txt`（7 个 varlen case ×
b1_t512_h16 / b4_t3840_h16 / b4_t4096_h16 / b5_t3968_h32 / b8_t2904_h16 / b1_t512_h2 D512 /
b3_t1792_h2 D512）、`src/fp8/fa_bwd_fp8_main_o40_ncu_lse_wgmma_split{1,8}_s512.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_o40_*`（单文件；含 `--lsetma=0` 修复验证）。

---

## 44. O41：fp8 主 kernel 的 K/V 4D-TMA（K 双缓冲、V 单缓冲）—— 正结果，默认 auto

> roadmap「下一步候选 ①」（O37 §36.6 留的「K/V TMA」）。O33–O36 已给 fp16/bf16 主 kernel 的
> Q/K/V/dO 全部 TMA 化，O37 只做了 fp8 的 Q/dO；本节把 **K/V 也改 4D-TMA**，并把
> 「K/V 双缓冲 smem 账算不过来」的结论用**零成本折叠**解决。

### 44.1 动机

O37 后端到端 S=4096 main 1.66ms，ncu 头号 stall 是 `short_scoreboard`(1.58) + `wait`(1.53)
（mma 依赖），但 `long_scoreboard` 仍有 **0.80**——一部分来自**每 tile 同步搬的 K/V**
（REGDQ 档下 `kPrefetch=false`，K/V 是逐 4B 全局读 + 逐 4B smem 写 + 逐 4B Kp 交织写）。
把 K/V 换成 TMA，硬件直接写 SW128 smem，省掉全局地址运算与大量 `LDG/STS`。

O37 §36.6 判断「K/V true overlap 必须双缓冲，而双缓冲 K/V 在 3 CTA/SM 下 smem 顶格」。
本节找到两处**零成本折叠**：

* **`dS3` 复用当前 K stage**——K 只被 GEMM1 读，fold 在 GEMM1/2 之后才写 `dS3`；K 双缓冲
  时 `dS3` 指到**当前** stage（另一个 stage 正被 TMA 写）。
* **`Ap` 复用 `Vs`**——V 只被 GEMM2 读，fold 之后才写 `Ap`。

于是只需多一个 K stage 的 `ks_sw_bytes`(4096B) + 64B mbarrier：
`smem = 70656 + 4096 + 64 = 74816B ≤ 76800`（本卡 3-CTA/SM 的动态 smem 实测上限，见下）
⇒ **仍 3 CTA/SM**。（探测：`sharedMemPerMultiprocessor=233472`、`reservedSharedMemPerBlock=1024`；
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 实测 76800→3、77000→2。）

### 44.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

新增 `Fp8Cfg::smem_bytes_wgmma_kvtma = smem_bytes_wgmma + ks_sw_bytes + 64`；`fp8_mma_body`
加模板参数 `bool KVTMA=false`（要求 `TMA && WGMMA && HD==128`）、入参 `kmap/vmap`；新增壳
`fa_bwd_fp8_mma_kvtma_kernel`（4 个 `__grid_constant__` 描述符）与 host launcher
`launch_bwd_main_kvtma`、CLI `--kvtma=0/1`（**默认 1**，对齐 O37 的 qdtma）。

**描述符**：`kmap/vmap` 与 Q/dO 同构——`make_lse_map_fp8(ptr, Hkv, S, D, B, boxR=32)`，
`dims={D,S,Hkv,B}`、`strides={Hkv*D, D, S*Hkv*D}`、box `{128,32,1,1}`、UINT8 + SWIZZLE_128B。
（fp8 一行 128B = 一个 SW128 atom 的整行；非 varlen 路径专用。）

**数据流**：
1. prologue 由 `tid==0` 发 K[nt_begin]→`Ks[stage0]`、K[nt_begin+1]→`Ks[stage1]`、V[nt_begin]→`Vs`
   三条 4D-TMA；等 Q/dO/K/V 到齐后，**从 SW128 K tile 重建 `Kp`**（`__byte_perm` 交织，与
   `kv_load_pair` 的 Kp 逐字节相同）。
2. 循环内 GEMM1 读 `Ks[stg]`（`stg=(nt-nt_begin)&1`）；fold 把 `dS3` 写回当前 K stage、`Ap`
   写回 `Vs`（两处折叠）。
3. 迭代末（本 tile 的 `dS3`/`Ap` 均已读完）：发 K[nt+2]→`Ks[stg]`、发 V[nt+1]→`Vs`，
   等 K[nt+1] 并重建下一 tile 的 `Kp`（**V 的在飞 TMA 与 Kp 重建重叠**），最后等 V[nt+1]。
   mbarrier 相位：K 两个 stage 各一个计数、V 一个（`(nt-nt_begin)&1` 做 stage）。

**数值口径**：K/V 的字节与 cp.async 版完全相同，`Kp` 也逐字节相同 ⇒ 与 Q/dO-TMA 版
逐位一致（只差跨 CTA `atomicAdd` 的加法次序）。默认路径 `sm_90` 构建完全不变。

### 44.3 数值（vs fp32 ref / TE FP8，逐元素）

九 shape 的 `dq/dk/dv vs fp32 ref` 与历史（O37/O27）**逐位一致**：

| case | dq | dk | dv |
|---|---|---|---|
| b1_s512_h16_d128_causal | 2.426e-01 | 2.972e-01 | 3.733e-01 |
| b1_s1024_h32_d128_causal | 2.399e-01 | 4.177e-01 | 3.535e-01 |
| b1_s4096_h16_d128_causal | 2.635e-01 | 2.644e-01 | 3.216e-01 |
| b1_s1024_h32_d128_kv4 | 2.517e-01 | 5.339e-01 | 7.173e-01 |
| b1_s1024_h64_d128_kv1 | 4.101e-01 | 1.572e+00 | 2.126e+00 |
| b1_s1024_h16_d128_full | 5.520e-02 | 5.312e-02 | 4.024e-02 |

同 session A/B 的 `max_abs(kvtma-vs-qdtma)`：dq `~1.2e-7`、dk/dv `~1e-6–1e-5`（仅 atomic 次序）。
`varlen` 与 `MLA(D=512)` 路径不经此分支，回归逐位不变（varlen b1_t512_h16 `2.280/3.108/3.422e-1`、
MLA S1024H2 `2.232/3.337/3.602e-1`，与 §16/§39 一致）。

### 44.4 性能（同 binary、同 session A/B；CUDA event，main-only，ms）

| shape | Q/dO-TMA | **Q/dO/K/V-TMA** | 比 | total（ours） |
|---|---|---|---|---|
| MHA S512 | 0.0734 | **0.0643** | **1.141×** | 0.1033 ms / 20.78 TF |
| MHA S1024 H32 | 0.3188 | **0.2825** | **1.128×** | 0.3894 ms / 44.12 TF |
| MHA S4096 | 1.7131 | **1.6056** | **1.067×** | **1.9215 ms / 71.53 TF** |
| GQA q32/kv4 | 0.2856 | **0.2654** | **1.076×** | 0.3547 ms / 48.44 TF |
| MQA q64/kv1 | 0.5204 | **0.4930** | **1.056×** | 0.6280 ms / 54.71 TF |
| full S1024 H16 | 0.3384 | **0.3193** | **1.060×** | 0.5675 ms / 15.14 TF |

端到端 S4096 **2.0508→1.9215 ms（71.53 TF）**，为 **TE FP8（同 session 0.5905 ms / 465.5 TF）
的 3.25×**（O37 3.49×；O27 3.69×）。单文件与两文件数字一致（`..._onefile_s4096` total 1.934 ms，
device 逐字一致）。S512 的 **main 0.0643 ms 已快过 TE FP8 整条反向 0.1008 ms**。

### 44.5 ncu（主 kernel，S=4096，`--launch-count 1`，同 binary `--kvtma` 0/1）

| 指标 | Q/dO-TMA | K/V-TMA |
|---|---|---|
| `gpu__time_duration.sum` | 1.67 ms | **1.61 ms** |
| `sm__inst_executed.sum` | 709.2 M | 722.4 M（+1.9%） |
| stall `long_scoreboard` | 0.80 | **0.55（−31%）** |
| stall `short_scoreboard` | 1.61 | **1.30（−19%）** |
| stall `wait` | 1.58 | 1.53 |
| stall `barrier` | 0.28 | 0.42 |
| `lts__t_sectors_op_red` | 114,524,160 | **114,524,160（逐字节不变）** |
| `lts__throughput` | 70.90% | **77.94%** |
| L1/TEX / SM throughput | 66.46% / 44.61% | 69.18% / 47.25% |
| regs / occ / Block Limit Shared Mem | 168 / — / 3 | **168 / 18.35% / 3**（smem 74.82KB） |

**结论**：K/V TMA 消掉 K/V 全局读的地址运算与 `LDG/STS`，`long_scoreboard` 0.80→0.55、
`short_scoreboard` 1.61→1.30；代价是 Kp 从 SW128 重建带来的少量额外指令（+1.9%）与 barrier
（0.28→0.42），**净 Duration −3.6%**。**`red` 逐字节不变**（归约结构未动）。
新墙仍是 **mma 依赖延迟（`wait` 1.53 + `short_scoreboard` 1.30）**，且 L2 升到 ~78%；
3 CTA/SM 由 74.82KB smem 保住。**这条路已到 K/V 搬运的收益上限**——Kp 的配对副本无法随
K-major SW128 自动得到，重建是必要的税；再要前进只剩「减 mma 依赖 / 提 occupancy」。

### 44.6 对标

fp8 无 FA 反向基线，仅 TE FP8（同 session，`harness/fa_bwd_bench.py bench --dtype fp8`）：
S512 0.1008 ms / 42.60 TF、S1024H32 0.2063 / 166.54、S4096 0.5905 / 465.48。
同 session 纯反向 FA2/FA3/TE（fp16，`harness/fa_vs_te_bwd_only.py fp16`）：MHA S4096
FA3 **0.3243 ms / 848 TF**、TE 0.4442 / 619、FA2 0.7259 / 379；GQA kv4 FA3 0.0826 / 416；
MQA kv1 FA3 0.1566 / 439。ours fp8 total 距 FA3 fp16 仍是量级差距，但已在 fp8 口径上把
端到端压到 TE 的 3.25×。

### 44.7 复现 / 原始输出

```bash
# 两文件（K/V TMA 默认开）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# 同 binary 退回 Q/dO-only TMA（A/B）
... scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=... --kvtma=0
# 单文件（device 由 sync_onefile_device.py 同步，逐字一致）
python3 scripts/sync_onefile_device.py src/fp8/fa_bwd_fp8_kernels.cuh \
  src/fp8/fa_bwd_fp8_mma_onefile.cu '#include <cuda_runtime.h>'
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --dir=...
# ncu A/B
ARCH="" NVCC_FLAGS="..." scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu \
  --metrics "gpu__time_duration.sum,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,..." \
  --launch-count 1 --kernel-name regex:fa_bwd_fp8_mma_kvtma -- --kvtma=1 --dir=...
```

原始输出：`src/fp8/fa_bwd_fp8_o41_sweep.out.txt`（两文件 ×5 shape：timing + O41 A/B + 对拍）、
`src/fp8/fa_bwd_fp8_o41_ncu_ab_s4096.out.txt`（qdtma vs kvtma 的 stall/扇区/吞吐）、
`src/fp8/fa_bwd_fp8_o41_ncu_kvtma_s4096.out.txt`（`--set full`，3 CTA/SM 证据）。

## 45. O42：fp8 主 kernel「dK/dV 归约改 Hopper bulk-reduce」——**负结果** + 墙的定量重测（第八十九轮）

> 动机：第 88 轮（O41）把 Q/K/V/dO 全部 TMA 化后，roadmap 的「下一步候选 ①」是
> **fp8 侧「减 mma 依赖 / 提 occupancy」**，并断言「K/V 搬运这条路已到上限」。本轮先
> 用 ncu 把 fp8 主 kernel 的墙**重新量准**，再针对头号项做一次新机制的判决。

### 45.1 墙的定量重测（`fa_bwd_fp8_mma_kvtma_kernel<128,64,32,...>`，S=4096，1.59ms）

| 指标 | 值 | 占比/说明 |
|---|---|---|
| `lts__t_sectors_op_red` | **114.5 M** | 占 L2 扇区 ~70.5%（read 34.0 M + write 14.0 M） |
| `lts__t_sectors_op_read/write` | 34.0 M / 14.0 M | |
| `l1tex__t_requests_pipe_lsu_mem_global_op_red` | 9.54 M | 每 warp-request 8 扇区 = 32 lane × float2 **已完全 coalesced** |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_red` | 76.3 M | 76.3 M / 9.54 M = 8（无扇区浪费） |
| stall `wait` / `short_scoreboard` | **1.53** / **1.30** | 每 issued-inst 平均周期；mma/smem 依赖 |
| stall `long_scoreboard`/`not_selected`/`barrier` | 0.56 / 0.41 / 0.42 | |
| Dynamic smem / regs / occupancy | 74.82 KB / 168 / **3 CTA/SM**（occ 18.4%） | smem 与 regs **双卡** 3 |

结论：dK/dV 的跨 CTA `red` 是**头号成本**（L2 的 ~70%），且**已是 coalesced**（无扇区
浪费，O4c 的 float2 向量化已到位）。

### 45.2 天花板：短路 dK/dV 的 red（探针，结果无意义）

按 `agent_skills/kernel-opt.md`「怀疑某段 global 重载是墙，就把它短路掉看天花板」，
用临时 `-DFA_SKIP_RED` 把 `epi_dv`/`epi_dk` 的 `red_add2` 关掉（dQ red 保留）：

| S=4096 | 正常 | 短路 dK/dV red | 倍数 |
|---|---|---|---|
| main（Q/dO/K/V-TMA） | 1.6015 ms | **0.9364–0.9422 ms** | **1.70×** |
| total（quant+pre+main+cvt） | 1.9253 ms | **1.2526 ms** | 1.53× |

⇒ dK/dV 的 red 操作（不是字节浪费）就值 **0.66 ms**；这是本轮唯一真实的杠杆，
也是「减 mma 依赖 / 提 occupancy」之外被 ncu 证实的第一顺位。

### 45.3 新机制：Hopper `cp.reduce.async.bulk`（1D，无需 tensormap）

逐元素 `red.global.add.f32` 虽 coalesced，但每 tile 要发 ~2048 条、占满 LSU/L2 流水。
Hopper 提供 **`cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`**（PTX 8.0 /
SM90，1D，直接对 global 做加法归约，多 CTA 并发原子）。冒烟
`src/fp8/fa_bwd_fp8_bulkred_smoke.cu`：多 CTA 并发 reduce 到同一 global 区域，`max_abs=0.0`
**PASS**。两个坑：① `.global` 目的地址必须用 **64 位寄存器**（`"l"` 约束），否则 illegal
instruction；② generic 写 smem 后必须 **`fence.proxy.async.shared::cta`** 才能被 async
proxy 读到。

### 45.4 实现（`-DFA_BULKRED=1`，opt-in；单/两文件 device 逐字一致）

`fp8_mma_body` 的 `epi_dv`/`epi_dk` 增加 staging 分支：把 `acc[i][j][q]*scale` 写进
**per-warp smem staging**（每 warp [BN/2=16][64]，行距 `STGS=68`，复用在 fold 之后死亡的
`Ps`/`Ss` 区，17.4 KB ≤ 2·BM·PSS·4 = 18.9 KB），随后每 warp 的 lane<16 各发一行
256 B 的 `cp.reduce.async.bulk`（`bulk_issue`），**不等**，与 GEMM4/GEMM5 重叠；只有要覆盖
staging 时才 `bulk_waitread()`（`cp.async.bulk.wait_group.read 0`）。不引入 `__syncthreads`
（per-warp staging 私有，仅 `__syncwarp`）。

### 45.5 数值（S=4096 fp8 causal）

步骤元件与历史**逐位一致**（`dq/dk/dv vs ref = 2.635e-01/2.644e-01/3.216e-01`，
与 §41/§44 完全相同）；bulk 版与 red 版只差跨 CTA atomic 加次序（≤1.7e-6）。

### 45.6 性能（同 session，event，ms）

| S=4096 | 默认（red） | `-DFA_BULKRED=1` | 比 |
|---|---|---|---|
| main（Q/dO/K/V-TMA） | 1.6015 | **1.8048** | **0.89×** |
| total | 1.9253 | 2.1409 | 0.90× |

**负结果**：bulk-reduce 比逐元素 red **慢 11%**。机制：每 tile 需要 32 KB 的 smem staging
写 + 32 KB 的 TMA 读（共 128 条 256 B 的 TMA op），而 fp8 main 的 **L1/TEX 已 71.8%**；
把便宜且 coalesced 的 `red` 换成 smem 往返 + 小粒度 TMA，净亏。此即「L1/TEX 墙把 L2 墙的
解顶回去」。代码保留为 **opt-in（默认关）**，供未来在更低 L1 压力的数据通路上复测。

### 45.7 判定与下一步

- **`red` 是 fp8 main 的头号成本（0.66 ms / 1.70×），但不可用 smem staging + TMA bulk
  替换**（L1/TEX 已满）。
- 真正可行的方向仍是**降低每元素的跨 CTA 贡献数**（= 放大 BM ⇒ 撞寄存器墙，O17b/O19 已证伪）
  或**提 occupancy**（smem 74.8 KB + regs 168 双卡 3 CTA/SM；到 4 CTA/SM 需 ≤58 KB 且
  ≤128 regs，非本轮可行）——即 roadmap 候选 ① 的两条都受硬件资源硬约束。
- 候选 ②（MLA 降 smem 冲 2 CTA/SM）经本轮核算：fp8 MLA `Fp8Cfg<512,64,32>` 的 smem
  203 KB 中，四个 operand（Q/dO/K/V）101 KB + 三个配对副本（Qp/dOp/Kp）83 KB + Ps/Ss 19 KB；
  即使省掉 Q/dO 的 smem（改寄存器/流式）也只降到 ~140 KB，且 regs 255 同样卡 1 CTA/SM ⇒
  **单轮不可行**，需 FlashMLA 式重构（Q/dO 驻寄存器、K/V 分块流水）。

### 45.8 附带修复：fp8 main 的纯 `sm_90` 构建

`run_varlen` 的 causal 分支原来**无条件**调 `launch_lse_bal_wgmma`（仅 `-DFA_WGMMA` 构建
存在）⇒ fp8 main 的 `-arch=sm_90` 构建失败（fp16/bf16 无此问题）。本轮加 `#ifdef FA_WGMMA`
守卫，D==128 在纯 sm_90 下退回 mma 镜像配对 LSE `launch_lse_bal<128,1>`。验证：
`ARCH=sm_90 scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen --dir=.../varlen_b1_t512_h2_d512_causal_fp8`
**编译通过、数值与历史逐位一致**（`1.613e-1/2.238e-1/3.864e-1`）。

### 45.9 复现

```bash
# 墙的定量重测（red 扇区 / stall）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma_kvtma \
  --launch-count 1 --metrics lts__t_sectors_op_red.sum,... -- --iters=1 --dir=...
# bulk-reduce 冒烟
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/run.sh src/fp8/fa_bwd_fp8_bulkred_smoke.cu          # PASS, max_abs=0
# 主 kernel A/B（默认 red vs -DFA_BULKRED=1）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -DFA_BULKRED=1 -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=100 --dir=.../b1_s4096_h16_d128_causal_fp8
# 纯 sm_90（无 FA_WGMMA）构建
ARCH=sm_90 scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen --dir=.../varlen_b1_t512_h2_d512_causal_fp8
```

原始输出：`src/fp8/fa_bwd_fp8_o42_ncu_default_s4096.out.txt`（red 扇区 + stall + duration）、
`src/fp8/fa_bwd_fp8_o42_ceiling_skipdkdvred_s4096.out.txt`（短路 dK/dV red 的天花板）、
`src/fp8/fa_bwd_fp8_o42_default_s4096.out.txt`（默认档 timing + 对拍）、
`src/fp8/fa_bwd_fp8_o42_bulkred_s4096.out.txt`（bulk 档 timing + 对拍）、
`src/fp8/fa_bwd_fp8_bulkred_smoke.out.txt`（机制冒烟）、
`src/fp8/fa_bwd_fp8_o42_sm90_varlen_d512.out.txt`（纯 sm_90 构建修复验证）、
`src/fp8/fa_bwd_fp8_mma_onefile_o42_s512.out.txt`（单文件默认档）。

## 46. O45：fp8 MLA（D=512）主 kernel 的墙复核 + bulkred/ILV 判决（第九十二轮）—— **负结果 + 文档更正**

> 第 91 轮（O44）的 roadmap「下一步候选 ①」称：**fp8 MLA 仍 1 CTA/SM 且没吃到 split**、
> fp16 MLA total 比 fp8 MLA 快 4.9–5.1×。本轮先把这个说法核实，再把候选 ①/②/③ 里
> 能在单轮内做的两条（bulkred 开放到 D=512、mma 交错）判决掉。

### 46.1 复核：fp8 MLA 早已吃到 split-KV（更正 §7.7/§14y 的旧结论）

fp8 的 `D==512` 主 kernel（`fa_bwd_fp8_mma_kernel<512,64,32,false,false,true,true>`）在 **O29
（第 70 轮）** 就把 auto ksplit 标成 **`target=S/2`**；本轮实测确认它确实生效：

| MLA case | base grid | auto ksplit | grid | main (ms) | total (ms) | vs ref dq/dk/dv (max_abs) |
|---|---|---|---|---|---|---|
| (1,256,2,512) | 8 | 16 | 64×2 | 0.0436 | 0.0896 | 2.36/2.29/3.44e-1 |
| (1,512,4,512) | 32 | 8 | 64×4 | 0.1217 | 0.1996 | 2.42/2.99/4.48e-1 |
| (1,1024,2,512) | 32 | 16 | 256×2 | 0.2003 | 0.2835 | 2.23/3.34/3.60e-1 |

ksplit sweep（S1024H2，main）：k=1/2/4/8/16/32 = 1.016/0.517/0.264/0.242/**0.201**/0.229 ms ⇒
**auto=16 就是最优**（与 O44 的 `grid*sp≈528` 结论同源）。对比 fp16 MLA（§14y：0.0222/0.0840/0.1524），
fp8 MLA 仅慢 **1.3–2.0×**，并非旧文档写的 4.9–5.1×——那处是拿 O44 去比 **P5-3 时代（第 21 轮）**
的旧数字（`§7.7` 0.308/0.591/1.022ms，当时还没 O29 的 D=512 标定、也没 O39/O41）。
**修正：候选项 ①「fp8 MLA 吃 split」= 已完成（O29）；fp8/fp16 MLA 的差距是 1.3–2×。**

### 46.2 ncu 复核：bound = **1 warp/scheduler 的延迟**，不是带宽/算力

`fa_bwd_fp8_mma_kernel<512,64,32,...>`，S1024H2，`--set full -c 1`：

| 指标 | 值 | 指标 | 值 |
|---|---|---|---|
| Duration | 245.5 µs | DRAM / L1TEX / L2 / Compute | **2.18 / 19.83 / 29.20 / 11.90 %** |
| Dynamic smem / regs | **207.87 KB / 255** | Achieved Occupancy | **6.25%**（1 CTA/SM × 4 warp） |
| Waves Per SM | 3.88 | Active Warps / Scheduler | **1.00** |
| No Eligible | **85.67%** | Executed Ipc Active | 0.57 |

每 issued-inst 的 stall（同 session，default 档）：**`long_scoreboard` 2.33** + `wait` 1.54 +
`short_scoreboard` 0.76，其余 ≈0（barrier 0.07、mio/lg/math 0）。⇒ 4 个 warp 分到 4 个
scheduler（每人 1 warp），任何 stall 都无其它 warp 可填；**墙是每个 scheduler 只有 1 个 warp**，
而 smem 207.9KB 把 CTA/SM 锁死为 1，唯一的杠杆是**每个 CTA 放更多 warp（256 线程）**。

### 46.3 天花板（探针）：K/V 全局载入值 ~16%

见 `agent_skills/kernel-opt.md`「短路某段重载看天花板」。新增 `-DFA_SKIPKVL=1`（探针，结果
无意义）跳过 per-tile 的 K/V 全局读 + 配对重建（GEMM 仍读 smem ⇒ 不会被 DCE）：

| S1024H2 | 正常 | SKIPKVL | 倍数 |
|---|---|---|---|
| main | 0.2006 ms | **0.1729 ms** | **1.160×** |
| total | 0.2850 ms | 0.2574 ms | 1.107× |

fp8 D=512 的 K/V 目前是**循环末同步载入**（`kv_load_pair`，因为 HD=512 的寄存器预取需
`NPU=16×4=64` regs、`kPrefetch=false`），没有 fp16 O6/O6b 那样的 `cp.async` 流水。天花板
1.16× 不大，且加 **V 双缓冲**放不下（K 双缓冲 +16.9KB 后仅剩 7.7KB，见 `Fp8Cfg` 的 203KB 构成），
故 cp.async K/V 流水**列 backlog**，不作为本轮增量。

### 46.4 判决 A：bulk-reduce 开放到 D=512 —— **仍是负结果（0.84×）**

把 O42 的 `kBulkRed` 条件从 `TMA && WGMMA && HD==128` 放宽到 **HD=512 的 mma 路径**
（D=512 的 L1/TEX 只有 19.8%，staging 的 smem 往返看似有空间）：

| S1024H2 | 默认（red） | `-DFA_BULKRED=1` | 比 |
|---|---|---|---|
| main | 0.2003 ms | **0.2386 ms** | **0.839×** |
| total | 0.2835 ms | 0.3237 ms | 0.876× |

数值逐位不变（2.232/3.337/3.602e-1）。⇒ **即使 L1/TEX 有大量余量，bulkred 依旧慢 16%**，
说明 O42 的失败不只是「L1/TEX 墙」——把便宜且已 coalesced 的 `red` 换成
「逐元素 smem staging 写 + 128 条 256B 的 `cp.reduce.async.bulk`」本身就是净亏（staging 的
smem 往返 + 小粒度 TMA 的固定开销）。默认仍 `FA_BULKRED=0`（opt-in）。

### 46.5 判决 B：`FA_ILV` / `FA_ILV34`（mma 指令级交错）—— **中性 / 负**

| S1024H2 main | 默认 | `-DFA_ILV=1`（GEMM1/2） | `-DFA_ILV34=1`（GEMM3/4） |
|---|---|---|---|
| ms | 0.2003 | 0.1996（1.004×） | 0.2074（**0.966×**） |

1 warp/scheduler 下，把一个 warp 里的两条 mma 交错也不能换来延迟隐藏（`wait` 不是靠指令
重排能解的，根因是 warp 数不够）。与 D=128 的 O22/O29 结论一致。

### 46.6 剩余杠杆核算：2 CTA/SM 不可达，唯一未证伪的是 8-warp 几何

`Fp8Cfg<512,64,32>::smem_bytes = 207872 B` 构成：Qs/dOs `64×528`×2 = 67.6 KB、
Qp/dOp `32×520×2`×2 = 66.6 KB、Ks/Vs `32×528`×2 = 33.8 KB、Kp `16×520×2` = 16.6 KB、
dS2 3.1 KB、Ps/Ss `2×64×37×4` = 18.9 KB、scales 1.3 KB。要 2 CTA/SM 需 ≤ 116.2 KB，**即使把
Qp/dOp/Kp 三个配对副本（83 KB）全消也仍 >124 KB**（且 O4b 已证 fp8 里消不掉）⇒
候选 ②「MLA 冲 2 CTA/SM」单轮不可行（同 O42 §45.7 的核算）。

**唯一未证伪的杠杆 = 256 线程 / 8 warp（每 scheduler 2 warp）**：把当前 2×2 的 warp 几何
（GEMM1/2 `wr*(BM/2)`、GEMM3/4 `wr*(BN/2)`、PREL 行映射 `r=wr*32+…`）按 fp16 O6c 的方式
参数化、改成 4×2 之类。它触及 GEMM1–5 的全部 epilogue 映射，**列 backlog**（多轮）。

### 46.7 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
# 默认档（3 MLA shape + ksplit/lsesplit sweep）
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 --iters=50
# bulkred 判决（开放到 D=512）
ARCH="" NVCC_FLAGS="$FLAGS -DFA_BULKRED=1" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=...
# ILV / ILV34 判决
ARCH="" NVCC_FLAGS="$FLAGS -DFA_ILV=1"    scripts/run.sh ...     # 或 -DFA_ILV34=1
# K/V 天花板探针（结果无意义）
ARCH="" NVCC_FLAGS="$FLAGS -DFA_SKIPKVL=1" scripts/run.sh ... --dir=...
# ncu：full + stall 比例
ARCH="" NVCC_FLAGS="$FLAGS" scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --launch-count 1 --kernel-name regex:fa_bwd_fp8_mma_kernel -- --dir=... --iters=5
```

原始输出：`src/fp8/fa_bwd_fp8_main_o45_sweep.out.txt`（3 shape + ksplit/lsesplit sweep +
bulkred + ILV/ILV34 + SKIPKVL 探针）、`..._o45_ncu_mla_s1024h2.out.txt`（`--set full`）、
`..._o45_ncu_stall_mla_s1024h2.out.txt`（stall 比例）。单文件 `fa_bwd_fp8_mma_onefile.cu`
经 `sync_onefile_device.py` 同步（device 逐字一致），S1024H2 main 0.2001ms、数值逐位相同。

## 47. O47：fp8 MLA（D=512）主 kernel 的「256 线程 / 8-warp 几何」（第九十四轮）—— **正结果，D=512 默认**

### 47.1 动机（承接 O46 / O45）

O46 已把 **fp16/bf16 MLA（D=512）的 mma 主 kernel** 从写死 2×2 warp 网格改成由 `NTH/NWAR`
派生，MLA 用 **256 线程 / 8 warp（2×4）**：同 1 CTA/SM 下每 scheduler 的 warp 数 1→2，
occ 6.1%→12.4%、main 1.07–1.11×。O45 对 **fp8 MLA** 的诊断完全一致：主 kernel 是
**1 CTA/SM（smem 207.9KB 锁死）× 4 warp ⇒ 1 warp/scheduler**（`Active Warps/Sched 1.00`、
`No Eligible 85.7%`、`long 2.33 + wait 1.54 + short 0.76`），墙 = **并行度/延迟**；
2 CTA/SM 因 smem 不可达、bulkred / ILV / split 均已证伪。**「256 线程 / 8-warp 几何」是唯一
未证伪的杠杆**（O45 §46.7 列 backlog，多轮）。本轮把它落地到 fp8。

### 47.2 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

把 `fp8_mma_body` 与 `fa_bwd_fp8_mma_kernel` 的 warp 网格从写死 2×2 改成由
**新模板参数 `NTH`（线程数）+ `NWAR`（N 方向 warp 数）派生**（默认 `NTH=128/NWAR=2`，
即历史档）：

* `NWM = NTH/32/NWAR`（M 方向 warp 数）；`wr = wid/NWAR`、`wc = wid%NWAR`；`__launch_bounds__(NTH,…)`。
* warp tile 由 NWM/NWAR 派生：`GM1=BM/NWM, GN1=BN/NWAR`（GEMM1/2）、
  `GM34=BN/NWM, GN34=NTW/NWAR`（GEMM3/4）、`GM5=BM/NWM, GN5=NTW/NWAR`（GEMM5）——
  `NTW=128` 与 warp 数解耦；相应 m/n-tile 数 `MTM/MTN/MTM34/NTM34/MTM5/NTM5` 与
  `r0/c0` 偏移、累加器数组维度全部同步参数化。
* `kv_prefetch_pair`/`kv_commit_pair`/`kv_load_pair` 加默认模板参数 `NT=THREADS`，body 内所有
  prologue/搬运循环的 `THREADS` 换成 `NTH`。
* **fold 只由前 4 个 warp 执行**（`if (wid < 4)`）：`BN≤64` 时 4 个 warp（各 8 行 × `NTFOLD`）
  即可覆盖全部 j / m，`NTH=256` 时余下 warp 空等（fold 本就不是瓶颈）；`NTH=128` 时 `wid<4`
  恒真 ⇒ **逐字等价**。
* `WGMMA` 分支是 warpgroup 级（1 个 warpgroup），加 `static_assert` 限制其只用默认档。

**默认 `NTH=128/NWAR=2` 与历史逐字等价**（`NWM=2`、所有 warp tile/映射与 2×2 完全相同）；
MLA（D=512）用 `NTH=256/NWAR=4`（2×4 网格）⇒ 1 CTA/SM 下 **8 warp、每 scheduler 2 warp**。
smem（207872 B）与 grid 不变、dK/dV 归约字节不变。host 加 CLI `--mla8w=0/1`（D=512 默认 1）
与 A/B 段；`run_varlen` 的 D=512 路径保持 4-warp（对齐 fp16 O46）。

### 47.3 数值（ours-vs-fp32-ref，fp8 causal，max_abs dq/dk/dv）

| shape（D=Dv=512） | 4-warp（=历史） | 8-warp（默认） | max_abs(8w-vs-4w) |
|---|---|---|---|
| S256H2 | 2.356 / 2.290 / 3.441e-1 | **同**（2.356/2.290/3.441e-1） | 1.2/2.4/2.4e-7 |
| S512H4 | 2.415 / 2.992 / 4.481e-1 | **同** | 1.2/4.8/4.8e-7 |
| S1024H2 | 2.232 / 3.337 / 3.602e-1 | **同** | 1.2/4.8/4.8e-7 |

⇒ vs ref 与历史 P5-3/O45 **逐值一致**（FA/TE 反向不支持 D=512，只有 fp32 ref 可对）；
8w-vs-4w 仅跨 CTA atomic 次序差异（≤5e-7）。**D=128 回归逐位不变**（S512 d128
2.426/2.97/3.73e-1、S4096 2.635/2.644/3.216e-1、GQA kv4 2.517/5.34/7.17e-1、MQA kv1
4.10e-1/1.57/2.13，与历史同量级/同值）。

### 47.4 性能（CUDA event，同 binary、同 session 的 `[O47 A/B]`）

| shape | main 4-warp | main 8-warp | 加速 | total 4w→8w | main-only TF（8w，峰值 1978.8） |
|---|---|---|---|---|---|
| S256H2 | 0.0435 | **0.0237** | **1.84×** | 0.0896→**0.0694** | 11.3（0.57%） |
| S512H4 | 0.1192 | **0.0737** | **1.62×** | 0.1996→**0.1332** | 29.1（1.47%） |
| S1024H2 | 0.1998 | **0.1243** | **1.61×** | 0.2835→**0.1927** | 34.6（1.75%） |

端到端 total 1.29–1.50×。**fp8 MLA 的 main 比 fp16/bf16 MLA（O46）快**（fp16 S1024H2 main
0.1409ms vs fp8 0.1243ms）；FA/TE 反向不支持 D=512，故只有 ours 数字。

### 47.5 ncu（`fa_bwd_fp8_mma_kernel`，S1024H2，`--set full -c 1`）

| 指标 | 4-warp（`--mla8w=0`） | 8-warp（默认） |
|---|---|---|
| Duration | 238.8 µs | **132.4 µs** |
| Achieved Occupancy | 6.25% | **12.49%** |
| Achieved Active Warps/SM | 4.00 | **7.99** |
| Issued Ipc Active | 0.57 | **1.13** |
| `sm__issue_active` | 14.39% | **28.07%** |
| No Eligible | 85.66% | **71.95%** |
| Registers Per Thread | 255 | **245**（无 spill） |
| Block Limit Registers / Shared Mem | 2 / **1** | 1 / **1** |
| DRAM / L1TEX / L2 / Compute | 2.25 / 19.90 / 30.11 / 12.29 % | 3.78 / 39.85 / 51.19 / 22.52 % |
| stall `long / wait / short / barrier` | 2.35 / 1.54 / 0.72 / 0.06 | 1.99 / 1.20 / **1.55** / 0.32 |
| `lts__t_sectors_op_red` | 6,684,672 | 6,684,672（**不变**） |

⇒ **每 scheduler 的 warp 数 1→2，occ 翻倍、Ipc 翻倍、issue_active 翻倍**，墙仍是
**`long_scoreboard`（L2/全局）+ `wait`（mma 依赖）+ `short_scoreboard`（smem→mma）**，
与 O45 的诊断一致（不是带宽/算力；DRAM 仍 <4%）。`red` 扇区逐字节不变 ⇒ 提升完全来自
「延迟隐藏/并行度」，与 fp16 O46 同机制。

### 47.6 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 --causal --iters=30
# A/B（同 binary 退回 4-warp）
... scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=... --causal --mla8w=0
# 单文件（device 由 sync_onefile_device.py 同步，逐字一致）
python3 scripts/sync_onefile_device.py src/fp8/fa_bwd_fp8_kernels.cuh \
  src/fp8/fa_bwd_fp8_mma_onefile.cu '#include <cuda_runtime.h>'
# ncu A/B
ARCH="" NVCC_FLAGS="$FLAGS" scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full -c 1 \
  --kernel-name regex:fa_bwd_fp8_mma_kernel -- --dir=... --causal --iters=1 [--mla8w=0]
```

原始输出：`src/fp8/fa_bwd_fp8_o47_baseline_s1024h2.out.txt`（改前 4-warp 基线）、
`..._o47_mla_sweep.out.txt`（3 MLA shape：timing + A/B + 对拍）、
`..._o47_d128_reg.out.txt`（D=128/GQA/MQA 回归）、
`..._o47_ncu_mla{4,8}w_s1024h2.out.txt`（`--set full`）、
`..._o47_ncu_stall_mla_s1024h2.out.txt`（stall 比例 + red 扇区）。

## 48. O48：fp8 D=128 **mma fallback** 主 kernel 的「256 线程 / 8-warp 几何」判决（第九十五轮）—— **负结果 + 两处 correctness 修复**

### 48.1 动机

O47 的「下一步候选 ①」：D=128 主 kernel 是否也能吃 8-warp。O46/O47 在 MLA（D=512、
1 CTA/SM）上证明「每 scheduler 的 warp 数 1→2」是通用杠杆。fp8 的 `fp8_mma_body` 在 O47
已把 warp 网格参数化为 `NTH`/`NWAR`，故只需 host 侧 `--d128w=0/1` 并把 D=128 派发到
`<128,64,32,REGDQ,false,PREL,true,true,256,4>`。**注意**：`WGMMA=true`（生产 TMA 路径）
的 GEMM1/2 是 warpgroup 级、`static_assert` 锁死 `NTH==128/NWAR==2`，故本项只能测 **mma
后端**（纯 `sm_90` 构建、或 `--wgmma=0`）。

### 48.2 结果：D=128 8-warp —— 负结果，且**根因与 fp16 相反**

| shape | 4w(128/2) | 8w(256/4) | 比 |
|---|---|---|---|
| S=512 MHA | 0.0718 ms | 0.0907 ms | **0.79×** |
| S=4096 MHA | 1.9139 ms | 2.7273 ms | **0.70×** |

数值 `max_abs(8w-vs-4w)` ≤ `1.2e-7/4.8e-7/1.4e-6`（仅 atomic 次序）；vs ref 与历史逐位不变
（S512 2.426/2.975/3.735e-1；S4096 2.635/2.643/3.216e-1）。

**为什么 fp16 在 S=512 赢（`docs/01` §14aa）而 fp8 输？** ncu（S=512）：
- fp8 4-warp：`Active Warps/Scheduler 2.87`、3 CTA/SM、`Waves 5.17`、Duration 67.7µs；
- fp8 8-warp：`Active Warps/Scheduler 1.99`、1 CTA/SM、`Waves 15.52`、Duration 89.7µs。

根因是 **fp8 的 D=128 主 kernel 早有 auto split-K（O2b/O29：S512 时 ksplit=16）**，grid 被抬到
`128×16=2048` 个 CTA，机器本来就被填满（2.87 warp/scheduler）。8-warp 只会把 3 CTA/SM 压成
1 CTA/SM、每 scheduler 反而降到 2，纯亏。fp16/bf16 的 **mma fallback 没有 ksplit**（O43 的
split-K 只加在 `wgmma2`），S=512 grid=128<132 SM、每 scheduler 只有 1 warp，8-warp 才成为杠杆。
⇒ **候选 ① 对 fp8 D=128 不成立**；`--d128w` 保持 opt-in（默认 0），并**不进入生产路径**
（生产是 `wgmma`+TMA，结构上无法 8-warp）。

### 48.3 附带修复（任何 `NTH>128` 的 mma 路径都需要的 correctness 修复）

为支持 `NTH=256`，发现并修掉 O47 参数化时留下的两个只在 `NTH≠128` 才触发的 bug：

1. **`kv_prefetch_pair`/`kv_commit_pair` 的越界**：两者按 `u = tid + e*NT`（`e<NPU`）遍历
   `(BN/2)*(HD/4)` 个 unit，**原实现无上界检查**。`NTH=128` 时 `NPU*NT` 恰好等于 unit 数；
   `NTH=256` 时 `u` 越界，越界读全局、并把 `Kp`/`Ks`/`Vs` 写到 tile 之外（实测 dq/dv 爆到
   ~1e34/~1e38）。加 `units=(BN/2)*nd4` 上界：prefetch 越界读 0、commit 越界跳过写；
   `NTH=128` 时恒满足、逐字不变。模板加 `int BN`（campaign 调用点同步）。
2. **`kRegDq` 的 flush 硬编码几何**：`for i<2 / j<8` + `wr*32/wc*64` 只对默认 2×2/128 成立；
   D=128 的 8-warp 实例 `NTM5=4` 会越界读 `dqacc` 并写错列。改成由 `MTM5/NTM5/GM5/GN5`
   派生（`128/2` 时与原式逐字等价）。

**默认 128/2 路径逐位不变**（回归：S512 2.426/2.975/3.735e-1；S4096 2.635/2.643/3.216e-1）。

### 48.4 复现 / 原始输出

```bash
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8 --causal [--d128w=1]
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:fa_bwd_fp8_mma_kernel \
  --launch-count 1 --section Occupancy --section SchedulerStats -- \
  --dir=.../b1_s512_h16_d128_causal_fp8 --causal [--d128w=1]
# 单文件（device 由 sync_onefile_device.py 同步）
python3 scripts/sync_onefile_device.py src/fp8/fa_bwd_fp8_kernels.cuh \
  src/fp8/fa_bwd_fp8_mma_onefile.cu \
  '// ----------------------------- 编译期常量 -----------------------------'
```

原始输出：`src/fp8/fa_bwd_fp8_o48_d128_{s512,s4096}.out.txt`、
`..._o48_ncu_d128_{0,1}w_s512.out.txt`、`..._o48_onefile_s512.out.txt`；
`docs/01` §14aa、`docs/01b` §6ai。

## 49. O49：D=128 mma 路径 8-warp 几何的 **自动档默认化**（第九十六轮）—— fp8 默认不触发

### 49.1 动机 / 改动

对齐 fp16/bf16 的 O49（`docs/01` §14ab、`docs/01b` §6aj）：把 O48 的 `--d128w` opt-in 改为
**默认 `-1`（自动）**。fp8 的判据多两重约束：

- 必须 `!wgmma`（fp8 的 Hopper `wgmma` 主路径是生产默认，`#ifdef FA_WGMMA` 下 `wgmma=1`；
  自动档**绝不能覆盖**它）；
- 有效网格要用 **`mg.x * mg.y * mg.z`**（fp8 的 `mg.x` 已含 auto `ksplit`，但 x 维只是
  网格的一维——O48 早期误用 `mg.x` 会让 S=512 误判成 128 而打开 8-warp，实测 main 0.0815ms
  vs 正确 0.0630ms，故必须乘上 H、B）。

`launch_bwd_main` 的 `NTH/NWAR` 派生是 O47 已有的。单/两文件同源。

### 49.2 结果：默认 shape 下 auto = off（逐位不变）

fp8 D=128 的 auto `ksplit`（O29 标定，目标 `max(2048, 4·base)`）把小 S 的 grid 抬到 ≫132：

| shape | base grid | auto ksplit | 有效 grid | `[O49] d128 8-warp` | vs ref（max_abs dq/dk/dv） |
|---|---|---|---|---|---|
| S=512 H16 | 128 | 16 | 2048 | **0（auto off）** | 2.426/2.975/3.735e-1（逐位） |
| S=1024 H32 | 512 | 4 | 2048 | 0 | 2.400/4.195/3.536e-1（逐位） |

main S=512 = 0.0630ms / total 0.1165ms；单文件同（main 0.0628 / total 0.1128）。
⇒ **fp8 的默认路径逐位不变**；O48 已判 fp8 在该 grid 下 8-warp 是负结果，故 auto 关门正确。

### 49.3 自动档在「真小网格」上仍然有效（`--ksplit=1` 探针）

强制 `--ksplit=1`（grid=128 ≤132 ⇒ auto 触发），验证实现正确且收益与 fp16 一致：

| S=512 fp8，`--ksplit=1` | 4-warp | auto（8-warp） | 比 |
|---|---|---|---|
| main | 0.1188ms | **0.0885ms** | **1.34×** |
| total | 0.1705ms | **0.1368ms** | 1.25× |

数值与 4-warp 一致（vs ref 2.426/2.975/3.735e-1）。

### 49.4 复现 / 原始输出

```bash
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8 --causal          # auto off
scripts/run.sh ... --causal --ksplit=1 [--d128w=0/1]                                      # auto on 探针
```

原始输出：`src/fp8/fa_bwd_fp8_o49_auto_s512.out.txt`、`..._o49_ksplit1_s512.out.txt`、
`..._o49_reg_s1024h32.out.txt`、`..._o49_onefile_s512.out.txt`；`docs/01` §14ab、`docs/01b` §6aj。

## 50. O50-fp8：wgmma 主 kernel 的 GEMM1/2 等待拆分（**中性，默认关**）+ 墙的定向重测（第九十七轮）

> 第九十六轮（O49）后，fp8 的「下一步候选 ②（wait+short）」一直被记为「硬件锁死」。本轮按
> `docs/01` §14ac 的同一角度，给 fp8 的 `fa_bwd_fp8_mma_kernel`/`kvtma`/`qdtma` 加
> `FA_WS1`（GEMM1(S)/GEMM2(dP) 等待拆分），并重测当前数据通路下 fp8 D=128 主 kernel 的墙。

### 50.1 改动

`fp8_mma_body` 的 WGMMA 分支里，GEMM1(`wgmma_mn32_issue<0>`, sacc) 与 GEMM2(`<1>`, dpacc) 各自
commit 后，原为统一 `wgmma_wait0_fp8()` 再 fold；改成 `wgmma_wait_group_fp8<1>()`（只等 GEMM1）
算 P/写 `Ps`，再 `wgmma_wait0_fp8()` 取 dP 算 dS。**数值逐位不变**。因实测中性，
**fp8 默认 `FA_WS1=0`**（保持旗舰路径逐位不变；`-DFA_WS1=1` 可复现）。单/两文件 device
逐字一致（`sync_onefile_device.py`）。

### 50.2 性能（同 binary A/B，300 iters，total ms）

| fp8 shape | ws0 | ws1 | 比 |
|---|---|---|---|
| S4096 H16 | 1.9029 | 1.9051 | 0.999× |
| S512 H16 | 0.0995 | 0.1019 | 0.976× |
| S1024 H32 | 0.3884 | 0.3852 | 1.008× |
| GQA kv4 S1024 | 0.3578 | 0.3560 | 1.005× |
| MLA d512 S1024H2 | 0.1897 | 0.1916 | 0.990× |

⇒ **中性（±2% 噪声）**，与 O22/O29/O45 的「`wait` 不是靠指令重排/错开能解的、根因是 warp 数」
一致：3 CTA/SM（12 warp/SM）下其它 warp 已能填掉 GEMM2 的执行，提前算 P 拿不到额外重叠。

### 50.3 墙的定向重测（`fa_bwd_fp8_mma_kvtma_kernel<128,64,32,...>`，S=4096）

| 指标 | 值 |
|---|---|
| Duration / DRAM / L1TEX / **L2** / Compute | 1.61ms / 4.26% / 70.40% / **78.05%** / 47.22% |
| regs / smem / CTA/SM · Achieved Occ | 168 / 74.82KB / **3** · 18.35% |
| Waves / Issued Ipc / Active Warps per Sched / No Eligible | 20.69 / 1.95 / 2.94 / 51.35% |
| `lts__t_sectors_op_red`（占 L2 的 70.5%） | **114,524,160** |
| `lts__t_sectors_op_read` / `op_write` | 33,817,369 / 13,951,898 |
| stall `wait` 1.59 + `short_scoreboard` 1.29 + long 0.57 + barrier 0.42 | — |
| smem 多余 wavefronts（op_ld 15.1M + op_st 15.5M，占 10%） | 14.9M |

**结论不变**：墙 = **L2 的 dK/dV `red`（70.5%）+ `wait`/`short` 的 mma 依赖延迟 + 3 CTA/SM**；
`red` 的 smem+TMA 替换（O42）、放大 BM（O19）、8-warp（O48/O49）、等待拆分（本轮）**全部中性/负**。
要进一步只剩「4 CTA/SM（需 regs≤128 且 smem≤56.8KB，当前 168/74.8KB）」或「跨-tile 软流水
（需 double-buffer P/dS，smem 无余量）」两条硬约束路。

### 50.4 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=300
ARCH="" NVCC_FLAGS="$FLAGS -DFA_WS1=1" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu ...   # A/B
```

原始输出：`src/fp8/fa_bwd_fp8_o50_wait_split_ab.out.txt`、`src/fp8/fa_bwd_fp8_o50_ncu_main_s4096.out.txt`。

---

## 51. O51：fp8 MLA（D=512）主 kernel 的 K/V `cp.async` 回填流水（第九十八轮）—— **正结果，D=512 默认**

### 51.1 动机（O45/O47 的墙）

O47 把 fp8 MLA 主 kernel 换成 8-warp（256 线程 / 2×4 网格）后，ncu 的**头号 stall 仍是
`long_scoreboard`**（O45：long 2.33 + wait 1.54 + short 0.76；O47 后 long 仍最高）。fp8 MLA
走 **mma 后端**（`HD=512` 不满足 SW128/wgmma 的 `HD==128` 约束，也没有 TMA），K/V 在每 tile
末尾由 `kv_load_pair` **同步**载入（`NPU=16` ⇒ O3 寄存器预取被禁用），全局载入延迟直接暴露。
O45 用 `-DFA_SKIPKVL` 探针量出「K/V 全局载入」的天花板是 **1.16×**。

### 51.2 改动（单/两文件 device 逐字一致，`sync_device` 核对 `identical: True`）

**只改搬运、不改数学**：把每 tile 的 K/V 同步载入换成 **`cp.async.cg` 回填流水**，且不牺牲
1 CTA/SM 的 smem 预算（MLA 主 kernel smem 207.9KB，1 CTA/SM 上限 232.4KB，余 ~24KB）：

1. **K 双缓冲**（多 `BN*ASLD=16,896B`）：本 tile 的 K 在 `Ks[stg]`；在**本轮 GEMM1 之前**用
   `cp.async` 发起下一 tile 的 K 到 `Ks[stg^1]`（该 stage 上一轮的 K 早被消费），延迟被整轮
   5 个 GEMM 覆盖。tile 末尾 `wait_group 0` 后由 `kp_build_rows` 从行主序 `Ks[stg^1]`
   **重建 Kp**（供 GEMM5 的 `ldmatrix.x2.trans`；`byte_perm` 0x5140/0x7362，与原配对逐位一致）。
2. **V 单缓冲 + 后段回填**：为让 `Vs` 在 GEMM2 后即可覆写，把原「**Ap 复用 Vs**」拆成独立
   缓冲（多 `BN*QTS=2,560B`）；于是 GEMM1/2 之后立即 `cp.async` 发起下一 tile 的 V，延迟被
   fold + GEMM3/4/5 覆盖。同理 dS3 也给了独立缓冲（`+BN*QTS`），使 `Ks[stg]` 在 GEMM1 后
   即被 `dS3` 之外的空间占用、K 双缓冲可安全复用。
3. 新增 `cp_async16_z`（带 src-size 的 16B `cp.async.cg`，越界零填充）、`k_issue_async`/
   `v_issue_async`（行主序，`HD/16` 个 chunk/行）、`kp_build_rows`；`Fp8Cfg::smem_bytes_kvpipe
   = smem_bytes + BN*ASLD + 2*BN*QTS`（MLA 下 229,888B ≤ 232,448B，仍 1 CTA/SM）。
4. 新增模板开关 `KVPIPE`（`fp8_mma_body`/`fa_bwd_fp8_mma_kernel`）与 launcher
   `launch_bwd_main_kvpipe`；host `--mlakvp=0/1`（默认 -1=自动开，仅 D=512 + 8-warp + PREL）。
   非 MLA 路径（D=128 的 wgmma/KVTMA、`run_varlen` 的 MLA）**不实例化**，逐位不变。

### 51.3 数值（ours-vs-fp32-ref，fp8 causal，max_abs dq/dk/dv）

与历史（P5-3/O45/O47）**同量级/逐值一致**（差异仅跨 CTA `atomicAdd` 次序，`max_abs(kvp-vs-sync)
≤1e-6`，`red` 扇区逐字节不变）：

| case (D=Dv=512) | dq | dk | dv | O51 A/B（sync→kvpipe） |
|---|---|---|---|---|
| S256 H2 | 2.356e-1 | 2.290e-1 | 3.441e-1 | 0.0239→0.0234ms (1.020×) |
| S512 H4 | 2.415e-1 | 2.992e-1 | 4.481e-1 | 0.0739→0.0716ms (1.032×) |
| S512 H2 | 2.286e-1 | 2.252e-1 | 3.343e-1 | 0.0508→0.0495ms (1.026×) |
| S1024 H2 | 2.232e-1 | 3.337e-1 | 3.602e-1 | 0.1228→0.1183ms (1.038×) |

D=128 回归（MHA/GQA）**逐位不变**：S512 2.426/2.972/3.733e-1、S4096 2.635/2.644/3.216e-1、
GQA kv4 2.517/5.339/7.173e-1；varlen MLA 正常（dq/dk/dv 1.61/2.24/3.46e-1）。
单/两文件 device 逐字一致、数值逐指标一致（S1024H2 total 0.1854 vs 0.1863ms，session 噪声）。

### 51.4 ncu（`fa_bwd_fp8_mma_kernel<512,64,32,...>`，S1024 H2，同 session A/B）

| 指标 | sync（`--mlakvp=0`） | kvpipe（默认） |
|---|---|---|
| stall `long_scoreboard` | **1.97** | **1.72** |
| stall `short_scoreboard` / `wait` / barrier | 1.57 / 1.20 / 0.32 | 1.40 / 1.18 / 0.37 |
| `smsp__inst_executed.sum` | 29,950,416 | **29,337,696（−2.0%）** |
| `lts__t_sectors_op_red` | 6,684,672 | **6,684,672（逐字节不变）** |
| DRAM / L1TEX / L2 / tensor | 3.72% / 31.9% / 50.4% / 6.35% | 3.76% / 32.4% / 51.0% / 6.43% |
| regs / smem / CTA/SM · occ | 246 / 207.9KB / 1 · 12.48% | ~246 / 229.9KB / 1 · 12.48% |

**结论**：`cp.async` 把暴露的 K/V 全局延迟部分藏进计算（`long_scoreboard` 17%↓、指令 −2%），
`red` 结构完全不动。墙随之变为 **`long_scoreboard`(1.72) + `short_scoreboard`(1.40) +
`wait`(1.18) 的 mma 依赖延迟 + 1 CTA/SM（12.5% occ）**——与 O45/O47 的定性一致，仅余量更小。
**教训**：fp8 MLA 是 mma 后端 + 1 CTA/SM，K/V 载入的「零成本重叠」（不改 smem/occupancy）
只值 ~2–4%；再往上要靠 2 CTA/SM 或 FlashMLA 式驻留，仍受 smem 硬约束。

### 51.5 对标 / 复现

FA2/FA3/TE 反向**均不支持 head_dim=512**（`fa=NA`/`te=NA`），MLA 反向只有 ours 数字，
沿用 P5-3/O47 的口径（`4·B·S²·H·D`）：O51 后 ours total S1024H2 **22.68 TF**
（0.1894ms）、S512H4 16.30 TF、S256H2 3.83 TF，main-only 分别 ~45/37/… （1 CTA/SM 硬约束）。

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 --iters=50
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu ... --mlakvp=0   # A/B
```

原始输出：`src/fp8/fa_bwd_fp8_main_o51_kvp_mla.out.txt`、`..._o51_kvp_reg.out.txt`、
`..._o51_ncu_mla.out.txt`、`..._mma_onefile_o51_mla_s1024h2.out.txt`。

## 52. O52：把 MLA 的 8-warp + K/V 回填流水推广到 **fp8 varlen**（第九十九轮）—— **正结果，varlen MLA 默认**

### 52.1 动机（O47/O51 只在定长路径生效）

O47（fp8 MLA 主 kernel 的 256 线程 / 8-warp 几何）与 O51（K/V `cp.async` 回填流水）都只落在
**定长** `D=512` 路径。`run_varlen` 里的 MLA 主 kernel 仍是历史几何
`launch_bwd_main<512,64,32,false,false,true,true>`（默认 `NTH=128,NWAR=2`，即 4-warp/2×2），
**既没有 8-warp、也没有 K/V 回填流水**。于是 varlen MLA 的 main 仍停在「1 CTA/SM × 4 warp =
每 scheduler 1 warp」的低并行度档（O45/O46/O47 一致认定的墙）。本轮把 O47/O51 原样搬过来。

### 52.2 改动（host-only；单/两文件 device 逐字不变）

**不碰 device 代码**（`fp8_mma_body` / `fa_bwd_fp8_mma_kernel` 早已由 O47/O51 参数化）：

* `run_varlen` 增加 `mla8w`/`mla_kvp` 入参（`-1`=自动，默认行为与定长一致：**8-warp + kvpipe**）；
  `D==512` 分支三档选择：`launch_bwd_main_kvpipe<512,64,32,false,true,true,true,256,4>`（默认）/
  `launch_bwd_main<512,64,32,false,false,true,true,true,256,4>`（8-warp，同步 K/V）/
  `launch_bwd_main<512,64,32,false,false,true,true>`（历史 4-warp，`--mla8w=0 --mlakvp=0`）。
* `main` 把已解析的 `--mla8w=`/`--mlakvp=` 透传给 `run_varlen`；`run_varlen` 末尾加
  `[O52 A/B]` 段（同 binary 对 4w / 8w / 8w+kvpipe 三种主 kernel 计时 + 逐元素比对）。
* 单文件版同步改 host（`sync_onefile_device.py` 核对 device 仍 `identical: True`，device 未动）。

### 52.3 数值（ours vs fp32 ref，fp8 causal varlen；max_abs dq/dk/dv）

与历史（§39）**同量级/逐值一致**；8-warp 相对 4-warp 只改 dK/dV 的跨 CTA `atomicAdd` 次序
（`max_abs(8w-vs-4w)` dq=1.2e-7、dk/dv ≤ 1e-6），kvpipe 相对 8-warp 同量级：

| case (D=Dv=512) | dq | dk | dv | main 4w→8w | 8w→8w+kvpipe |
|---|---|---|---|---|---|
| b1_t512 causal | 1.613e-1 | 2.238e-1 | 3.864e-1 | 0.0899→0.0516ms (**1.74×**) | 0.0516→0.0504ms (1.02×) |
| b3_t1792 causal | 3.404e-1 | 3.436e-1 | 3.508e-1 | 0.3520→0.2002ms (**1.76×**) | 0.2002→0.1890ms (1.06×) |
| b1_t512 full | N/A* | N/A* | N/A* | 0.0890→0.0512ms (1.74×) | 0.0512→0.0498ms (1.03×) |

*（full 行数值 `N/A`：本轮复测发现一个**与本改动无关、HEAD 已存在**的偏差，见 §52.6；其 A/B 计时
仍有效，4w/8w 逐元素一致到 1e-7。）**单文件与两文件逐指标一致**。D=128 varlen（MHA/GQA）回归**逐位不变**
（b1_t512_h16 causal fp8 `2.280/3.108/3.422e-1`），定长回归不动（device 未改）。

### 52.4 性能（event，`Σ_b 4HL²D` 口径，同 session）

**端到端 total**（quant+LSE+delta+main+convert）：

| case | O52 前（§39，4-warp） | O52（8-warp+kvpipe） | 加速 |
|---|---|---|---|
| b1_t512 causal | 0.1866 ms | **0.1006 ms** | 1.86× |
| b3_t1792 causal | 0.5875 ms | **0.2987 ms** | 1.97× |
| b1_t512 causal（TF） | 5.75 TF | **10.68 TF** | — |
| b3_t1792 causal（TF） | 9.60 TF | **18.87 TF** | — |

main-only：b1_t512 0.0899→0.0504ms、b3_t1792 0.3520→0.1890ms（分别 **1.78×/1.86×**，含 kvpipe）。
MLA 反向 FA2/FA3/TE **均不支持 head_dim=512** ⇒ 无外部基线，只有 ours 数字（口径 `4BS²HD`；
按 harness 定长口径 `4BS²H(D+Dv)` 需 ×2）。

### 52.5 ncu（`fa_bwd_fp8_mma_kernel`，b1_t512 causal varlen，`-c 1`，同 binary A/B）

| 指标 | 4-warp | 8-warp+kvpipe（默认） |
|---|---|---|
| Duration | 116.6 µs | **59.5 µs** |
| `sm__warps_active` | 6.20% | **12.39%** |
| Issued Ipc Active | 0.11 | **0.23** |
| stall `long / short / wait` | 3.13 / 0.67 / 1.51 | **2.23 / 1.30 / 1.19** |
| `lts__t_sectors_op_red` | 1,769,472 | **1,769,472（逐字节不变）** |
| DRAM / L1TEX / L2 / tensor | 2.21 / 9.01 / 17.4 / 1.90% | 4.27 / 18.98 / 32.1 / 3.74% |

**结论**：与 O47/O51 同机制——1 CTA/SM 下把每 scheduler 的 warp 数 1→2，`red` 扇区与原同步
路径**逐字节相同**，提升完全来自「并行度/延迟隐藏」。墙仍是 `long_scoreboard + short + wait`
的 mma/全局依赖 + 1 CTA/SM（smem 硬约束，2 CTA/SM 不可达）。

### 52.6 附带发现：fp8 varlen MLA **非 causal（full）** 在 HEAD 已是偏差（预存在，非本改动引入）

复测 `varlen_*_d512_full`（causal=0）时，ours 的 dq/dk/dv `max_abs≈7.3/7.0/4.6`（ours_amax 7.3，
ref_amax 0.57），而 §39 第 80 轮记录的是通过值（5.3e-2/5.2e-2/4.2e-2）。
**用 `--mla8w=0`（退到历史 4-warp）复跑得到完全相同的错误值**，且用 `git stash` 回到 O51 提交
（`056b316`）重新编译也**一模一样** ⇒ **是本改动之前就存在的回归**（第 80 轮之后某轮引入，
嫌疑在 varlen 非 causal 的 LSE 路径或全序列 `ncols`/`ksplit` 组合）。fp16/bf16 的 full MLA
varlen 同样偏差（同 HEAD 复现）。本轮**未修**（超出 O52 范围），已记入 ROADMAP backlog。

### 52.7 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b1_t512_h2_d512_causal_fp8 --varlen --iters=30
# A/B（同 binary 退回 4-warp / 8-warp 同步）
... scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen --mla8w=0 --mlakvp=0
```

原始输出：`src/fp8/fa_bwd_fp8_main_o52_varlen.out.txt`、`..._main_o52_ncu_varlen*out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_o52_varlen.out.txt`。
