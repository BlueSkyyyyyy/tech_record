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
