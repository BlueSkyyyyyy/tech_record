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

*（full 行数值当时记 N/A：以为是 HEAD 偏差，**O53 已更正为「漏传 `--full`」的假警报**，见 §52.6；
 正确值应与 §39 一致 ~5e-2。其 A/B 计时仍有效，4w/8w 逐元素一致到 1e-7。）**单文件与两文件逐指标一致**。D=128 varlen（MHA/GQA）回归**逐位不变**
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

### 52.6 附带发现（第一百轮**更正为假警报**）：fp8 varlen MLA 非 causal（full）的偏差

> **更正（O53，第 100 轮）**：当时复测 `varlen_*_d512_full` 得到 `max_abs≈7.3/7.0/4.6`、
> 并以为「HEAD 已存在的回归」。实际原因是**那条复测命令漏传 `--full`**：`run_varlen` 的输出头
> 写的是 `causal=1`（可在 `src/fp8/fa_bwd_fp8_main_o52_varlen.out.txt` 第 27/39 行核对），
> 即**按 causal 去比 full 的 ref**，误差自然是 O(1)。本轮显式加 `--full` 复跑
> `varlen_b1_t512_h2_d512_full_fp8`：输出头 `causal=0`，ours-vs-ref `max_abs`
> **5.26/5.22/4.22e-2**，与第 80 轮 §39 记录（5.3e-2/5.2e-2/4.2e-2）一致。
> fp16/bf16 的 full 同样通过（fp16 3e-4–5.5e-4、bf16 1.6e-3–3.5e-3，见 `docs/01` §14ae、
> `docs/01b` §6am）。**结论：不是回归、不是 kernel bug、无需修复。教训：跑 full 用例先核对
> 输出头 `causal=` 字段。**

（原记录已作废，保留经过去：~~复测 `varlen_*_d512_full`（causal=0）时，ours 的 dq/dk/dv
`max_abs≈7.3/7.0/4.6`（ours_amax 7.3，ref_amax 0.57），而 §39 第 80 轮记录的是通过值
（5.3e-2/5.2e-2/4.2e-2）。用 `--mla8w=0` 复跑得到完全相同的错误值，且用 `git stash` 回到 O51
提交（`056b316`）重新编译也一模一样 ⇒ 以为是本改动之前就存在的回归。~~）

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

---

## 53. O54-fp8：非 causal（full）MLA varlen 的 LSE 走 K 维 split（第 101 轮）—— 正结果，full varlen 默认 auto

与 fp16 §15 / bf16 §6an 逐字同构：给 fp8 的 `lse_mma_kernel_bal<HD,PIPE,bool FULL=false>`
加 FULL 模式（一个 CTA 一个 m 块、`ncols=len`、无因果掩码），host `run_varlen` 的
`D==512 && !causal` 分支在 `lse_split_eff>1` 时走 `launch_lse_bal<512,1,true>`（内部把
`grid.z` 扩成 `B*split` 并跑 `lse_split_merge_kernel`）。fp8 的 auto 目标为 **768**（比
fp16/bf16 的 384 更大——实测 b3 最优 k=8、b1 k=8，故取 768 让 b3 也到 8）。device 代码同源、
单/两文件 device 逐字一致。

* **LSE-only（b3_t1792 full，`[O54 A/B]`）**：old(O1 mma) 0.3753 ms → split1 0.1291 /
  split2 0.0743 / split4 0.0429 / **split8 0.0374** / split16 0.0435（auto=8）⇒ **10.0×**。
* **端到端**：total **0.349→0.346→（old 约 0.72）** ⇒ 相对 old 路径 ~2.1×；
  现 total **0.346 ms / 16.30 TFLOPS**。
* **数值 vs fp32 ref**：dq/dk/dv = 7.994e-2 / 9.629e-2 / 4.148e-2（fp8 噪声，与 old 同量级）。
* **b1_t512 full**：old 0.1899 → auto=8 split8 **0.0156 ms**（12.2×）。
* **causal b3 回归**：total 0.2974 ms、数值 3.404/3.436/3.508e-1（与历史同量级，FULL=false
  路径逐位不变）。

原始输出：`src/fp8/fa_bwd_fp8_o54_varlen_full_b3.out.txt`（两文件）、
`..._o54_varlen_full_b3_onefile.out.txt`（单文件）、`..._o54_varlen_causal_reg_b3.out.txt`（回归）。

## 54. O58-fp8（第一百零五轮，**正结果，causal varlen 默认；full opt-in**）：fp8 MLA varlen LSE 的 2 CTA/SM / 8-warp 几何

### 54.1 动机与改动

O54 给 fp8 的 full MLA varlen 接了 `lse_mma_kernel_bal<512,1,true>`（FULL，K 维 split），
但 fp8 的 **causal** MLA varlen LSE 一直是 `<512,1>`（PIPE1/LBN64），且 fp8 侧**从未**做
O56/O57 的「8-warp / 2 CTA/SM」几何（O56 曾刻意不扩到 fp8，因 fp16/bf16 在 full 上判负）。
本轮把 fp16/bf16 的 **O58（§15e）** 同构到 fp8，并顺带补做 fp8 的 O56/O57。

**device（fp8 `lse_mma_kernel_bal`）**：模板由 `<HD,PIPE,FULL>` 参数化为
`<HD,PIPE,FULL,NTH=THREADS,LBN_=LBN>`（`LBM_=(NTH/32)*16`、`MTN=LBN_/8`、`KVL=LBN_*ASLD`；
`issue_q/issue_k` 步长用 `NTH`、`acc[1][MTN][4]`、`mma_block<16,LBN_,HD,E4E4>`），默认档与原版
**逐位等价**；单文件 device 由 `sync_onefile_device.py` 同步（`identical: True`）。

**host（`run_varlen` + `launch_lse_bal` 参数化 + `--lseocc/--lse8w`）**：
- **causal 默认改为 cfg6**（`<512,1,false,128,16>`，PIPE1/LBN16），`--lseocc=4` 退回旧默认、
  `5`=P0/LBN32、`--lse8w=1`=8-warp（NTH=256/LBM=128/LBN=32）；cfg5/6 的 split auto 目标
  由 256 提到 **1024**（b1→8 cap、b3→16，A/B 最优附近）。
- **full 仍默认 O54 旧路**（O56/O57 判该杠杆在 full 上为混合/小正），`--lseocc=6`/`--lse8w=1`
  opt-in；`[O58 A/B]` 在同 binary 内扫 legacy/cfg6/cfg5/8w × split（LSE-only）。

> **与 fp16/bf16 的关键差异**：fp8 一元素 1B ⇒ LSE smem 只有 fp16/bf16 的一半。
> **旧默认 P1/LBN64 已是 100,608B（< 116,224B）⇒ 本就 2 CTA/SM**；cfg6 的 49,920B 则给到
> **4 CTA/SM**。所以 fp8 的「2 CTA/SM」不是新门槛，cfg6 的收益来自**再翻一档 occupancy**。

### 54.2 结果（同 session event）

| case | 旧默认 P1/LBN64 | **新默认 cfg6** | 比值 | 8-warp | cfg5 |
|---|---|---|---|---|---|
| b1_t512 causal，LSE 最优 | split8 0.0251 | split16 **0.0198** | **1.27×** | 0.0224 | 0.0326 |
| b3_t1792 causal，LSE 最优 | split8 0.0366 | split16 **0.0319** | **1.15×** | 0.0338 | 0.0563 |
| b1 total | 0.1006 ms | **0.0965 ms** | **1.04×** | — | — |
| b3 total | 0.2997 ms | **0.2789 ms** | **1.07×** | — | — |
| b3 full total（O57-fp8，opt-in） | 0.3440 ms | `--lseocc=6` 0.3413 / `--lse8w=1` **0.3404** | 1.008–1.011× | — | — |

⇒ **causal 上 cfg6 是稳定净正**（端到端 1.04–1.07×），**causal 默认切 cfg6**；
full 上同 fp16/bf16 只小正（LSE 1.06–1.15×、端到端 ~1%），保持 **opt-in**。

### 54.3 数值（ours vs fp32 ref，max_abs dq/dk/dv）

b1 causal `1.613e-1/2.238e-1/3.864e-1`、b3 causal `3.404e-1/3.436e-1/3.508e-1`（与 O53 fp8
causal 历史**逐位相同**）；full b3 与 O54 同量级。单/两文件逐指标一致
（b3 onefile total 0.2781 vs 两文件 0.2789ms）。D=128 varlen 回归正常
（b1_t512_h16 causal total 0.1104ms、vs ref 2.28e-1/3.11e-1/3.42e-1）；
固定 MLA 回归 S1024H2 causal vs ref `2.232/3.337/3.602e-1` 不变。

### 54.4 ncu（fp8 b3 causal，同 split16，`-c 1`）

| | 旧默认 P1/LBN64 | **cfg6 P1/LBN16** |
|---|---|---|
| `gpu__time_duration` | 40.70 µs | **27.07 µs（1.50×）** |
| `sm__warps_active` | 11.14%（≈2 CTA） | **18.26%（≈4 CTA）** |
| `launch__shared_mem_per_block` | 103.17 KB | **52.10 KB** |
| `sm__throughput` | 34.84% | **47.17%** |
| `l1tex__throughput` | 17.76% | 28.19% |
| `short_scoreboard` / `wait` / `long` stall | 0.51 / 2.02 / 0.34 | 0.46 / **1.97** / 0.89 |

**结论**：fp8 旧默认本就 2 CTA/SM，cfg6 把 smem 砍到 ~50KB ⇒ **4 CTA/SM**，occupancy
11.14%→18.26%、Duration 1.50×；`wait`（mma/softmax 固定延迟）仍是头号，但被更多 CTA 摊薄。
**bound = compute/softmax + `wait` 的固定延迟 + occupancy**（不再是 smem 容量）。

### 54.5 复现 / 原始输出

```bash
F='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
ARCH="" NVCC_FLAGS="$F" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --varlen --causal --iters=50 /home/xieminglin/proj/output/fa-bwd/varlen_b3_t1792_h2_d512_causal_fp8
# A/B：--lseocc=4（旧默认）/ 6 / 5 / --lse8w=1；full 用 --full
```
原始输出：`src/fp8/fa_bwd_fp8_o58_varlen_causal_b{1,3}.out.txt`、
`..._b{1,3}_legacy.out.txt`、`..._b3_onefile.out.txt`、`..._o58_varlen_full_b3.out.txt`、
`..._o58_ncu_lse_{cfg6,legacy}_b3.out.txt`。

## 55. O59（第一百零六轮，**正结果，定长 causal MLA 默认；含一处 device 竞争修复**）：把 O58 的 causal MLA LSE `cfg6` 从 varlen 推广到**定长**路径

### 55.1 动机：O58 只改了 varlen，定长 causal MLA 的 LSE 一直走旧默认

O58（§54）把 causal MLA **varlen** 的 LSE 从旧默认 `<512,1>`（PIPE=1/LBN=64，1 CTA/SM）
切到 `cfg6 <512,1,false,128,16>`（PIPE=1/LBN=16，2 CTA/SM），LSE 1.18–1.22×。但 `cfg6` 只落在
`run_varlen` 的 `D==512 && causal` 分支里；**定长**（非 varlen）路径——`main()` 的 `run_pre`
`D==512` causal 分支——仍然只调 `launch_lse_bal<512,1>`。4 个定长 MLA 生产形状
（S256H2 / S512H4 / S1024H2，fp16/bf16/fp8）因此没吃到这条杠杆；fp8 定长 MLA 的 LSE 是
端到端 preprocess 的绝对大头（S1024H2：preprocess 0.0352ms / LSE ~0.031ms，占端到端 17%）。

### 55.2 发现并修复一处 latent 竞争（device，三 dtype）

把 cfg6 接进定长路径后，对拍发现 **LSE 出现 ~1e-2 量级的非确定性抖动**（同一 binary 两次跑
`cfg6 vs cfg6` 的 max_abs = 4.17e-2 / 3.16e-2 各不相同；`cfg6(sp=1) vs legacy(sp=1)` 却逐位为 0）。
根因在 `lse_mma_kernel_bal` 的 PIPE=1 镜像配对循环：

- 每进入一个 m 块先 `issue_q(Qs, m0)` 发 Q 的 `cp.async`；循环首 `cp.async.wait_group 0` 才 drain；
- 但当某切片 `nuse == 0`（`ksplit` 大于该 m 块的 K tile 数时会出现）时，循环体不执行、
  **本 m 块发出的 Q `cp.async` 从不被 wait**；随后 `t=1` 又 `issue_q(Qs, m0')` 写同一 `Qs`，
  两个异步拷贝竞争同一 smem 地址 ⇒ 结果取决于谁后落地。
- LBN=64（legacy）的空切片更少、且旧 timing 常掩盖它；LBN=16 + 大 `ksplit` 让空切片变多，
  竞争被放大成可观测的非确定抖动。这也解释了 O58 的 cfg6 在某些 varlen 形状下的隐患。

**修复**：在每个 m 块（`t` 循环体）末尾、切换下一 m 块之前，先 drain 本块仍在飞的 `cp.async`
再 `__syncthreads()`：

```cpp
if constexpr (PIPE) asm volatile("cp.async.wait_group 0;\n");
__syncthreads();
```

三 dtype 的 `lse_mma_kernel_bal` 与 `lse_mma_kernel_bal_wgmma`（fp8 的 wgmma 版也有同样结构）
同步加；单文件 device 经 `sync_onefile_device.py` 同步（`identical: True`）。修复后
`cfg6 vs cfg6` 逐位为 0、`cfg6 vs legacy` = **4.768e-7**（只剩 split 求和次序差异）。

### 55.3 改动（host-only + 上面的 device 修复）

- `fa_bwd_fp8_main.cu` / `fa_bwd_fp8_mma_onefile.cu` 定长 `run_pre` 的 `D==512` causal 分支：
  默认走 `launch_lse_bal<512,1,false,128,16>`（cfg6）；`--lseocc=4` 退回旧默认、
  `5`=PIPE0/LBN32；split auto 目标由 256 抬到 **1024**（cfg6 的 4 CTA/SM 把并发槽翻倍）。
- fp16 / bf16 同构（device 早由 O56 参数化；thost 补 `kLseSmemBal1_5/6`、`cudaFuncSetAttribute`
  与 `run_pre` 分派；单/两文件同步），拆分 auto 目标由 132 抬到 **528**（对齐 O58 varlen）。
- 定长不支持 8-warp（`--lse8w` 在定长忽略）；数学口径完全不变。

### 55.4 结果（同 session，同 binary A/B：`--lseocc=4` vs 默认 cfg6）

LSE-only（`[O59 A/B]`，三 dtype 一致，仅列 fp8）：

| shape | legacy(sp) | cfg6(sp) | 比 |
|---|---|---|---|
| S256H2 | 0.0236 ms (4) | **0.0197 ms (4)** | **1.195×** |
| S512H4 | 0.0256 ms (8) | **0.0212 ms (8)** | **1.206×** |
| S1024H2 | 0.0312 ms (16) | **0.0261 ms (16)** | **1.196×** |

端到端 total（三 dtype × 3 shape，`src/fa_bwd_o59_fixed_mla_shapes.out.txt`）：

| dtype | S256H2 | S512H4 | S1024H2 |
|---|---|---|---|
| fp8 | 0.0720→**0.0649（1.11×）** | 0.1340→**0.1280（1.05×）** | 0.1882→**0.1808（1.04×）** |
| fp16 | 0.0534→**0.0502（1.06×）** | 0.1180→**0.1146（1.03×）** | 0.1884→**0.1836（1.03×）** |
| bf16 | 0.0535→**0.0504（1.06×）** | 0.1188→**0.1162（1.02×）** | 0.1874→**0.1832（1.02×）** |

小 shape 收益最大（LSE 占比高、且 4 CTA/SM 消掉 2 CTA/SM 的半个波）；大 shape 收益收敛到 ~1.03×。

### 55.5 ncu（S512H4，`-c 1`，同 binary）

| | fp8 legacy `<512,1>` | fp8 cfg6 | fp16 legacy | fp16 cfg6 |
|---|---|---|---|---|
| `gpu__time_duration` | 23.62 µs | **18.98 µs（1.24×）** | 19.97 µs | **16.19 µs（1.23×）** |
| `launch__shared_mem_per_block` | 102.14 KB | **51.07 KB** | 199.68 KB | **99.84 KB** |
| `Block Limit Shared Mem`（CTA/SM） | 2 | **4** | 1 | **2** |
| theoretical occupancy | 12.50% | **25.0%** | 6.25% | **12.5%** |
| Waves Per SM | 0.48 | **0.24** | 0.97 | **0.48** |
| Compute (SM) | 12.53% | **17.36%** | 14.52% | **20.05%** |

**结论**：cfg6 把 LSE smem 精确减半（fp8 4 CTA/SM、fp16 2 CTA/SM），Duration 1.23–1.24×。
这是把 O58 在 varlen 上验证过的「**压 smem 换 CTA 级并行度**」杠杆复用到定长 causal MLA，
收益与 O58 同量级。bound 从「低 occupancy / 半个波」转向 **compute/softmax + `wave` 量化**。

### 55.6 数值（ours vs fp32 ref，max_abs dq/dk/dv）

- fp8：S256H2 `2.356e-1/2.290e-1/3.441e-1`、S512H4 `2.415e-1/2.992e-1/4.481e-1`、
  S1024H2 `2.232e-1/3.337e-1/3.602e-1`——与 P5-3/O58 历史**逐位一致**。
- fp16：S256H2 `1.638/1.582/1.753e-3`、S512H4 `2.516/2.916/1.724e-3`、
  S1024H2 `1.987/1.712/1.848e-3`；bf16 同量级（~1e-2）。
- `cfg6-vs-legacy` 的 LSE max_abs = **4.768e-7**（仅 split 求和次序）。
- **D=128 回归逐位不变**：fp16 S4096 `1.883/1.734/1.966e-3`、bf16 `15.10/13.40/16.31e-3`、
  fp8 `2.635/2.643/3.216e-1`（与 `docs/04` 表逐位一致）。
- 单/两文件逐指标一致（fp8 S512H4 0.1246 vs 0.1280 ms，数值同；O59 A/B 1.219 vs 1.206，session 噪声）。

### 55.7 复现 / 原始输出

```bash
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=20 --lseocc=4 \
  /home/xieminglin/proj/output/fa-bwd/b1_s512_h4_d512_causal_fp8   # 旧默认
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=20 \
  /home/xieminglin/proj/output/fa-bwd/b1_s512_h4_d512_causal_fp8   # cfg6（默认）
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:lse_mma_kernel_bal --launch-count 1 -- --lseocc=4 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h4_d512_causal_fp8
```

原始输出：`src/fp8/fa_bwd_fp8_o59_fixed_s1024h2.out.txt`、
`..._o59_onefile_s{512h4,1024h2}.out.txt`、`..._o59_ncu_lse_{cfg6,legacy}_s512h4.out.txt`；
`src/fp16/..._o59_*`、`src/bf16/..._o59_*`；汇总 `src/fa_bwd_o59_fixed_mla_shapes.out.txt`。

---

## 56. P3-4e（第一百一十八轮，**正结果，opt-in `--det`**）：fp8 反向的**确定性 dK/dV 归约**

### 56.1 动机（补齐 `docs/00` catalog §4.2 第 5 条）

`docs/00` 在「FP8 反向：为什么最难点」里明写第 5 条：**「确定性：FP8 kernel 一般非确定性
（原子累加）；对拍用容差而非位相等」**。fp16/bf16 早在 **O7b（第六十四轮）** 就给了
`--det=1`（partial + 二次归约，逐位可复现、代价量化，见 `docs/01` §14o）；fp8 一直没有。
本轮把这条补齐，并顺带量化「确定性模式的代价」（`ROADMAP` backlog 里的
「deterministic 模式的代价量化」）。

fp8 反向的非确定性来自跨 CTA 的 `atomicAdd`：dK/dV 的每个 KV 元素被 `(Q 块 × Q 头)`
个 CTA 贡献，dQ 在 `ksplit>1` 时也被多个 CTA 贡献；**浮点加法次序随 block 调度变化**，
末位会抖动。所谓「确定性」= 把每个 CTA 的贡献写进**独立的 partial 缓冲**（非原子覆盖写，
每元素只被一个 CTA 写），再由一个固定的归约 kernel **按固定次序求和**。

### 56.2 实现（单/两文件 device 逐字同源）

改动（`src/fp8/fa_bwd_fp8_kernels.cuh` + `fa_bwd_fp8_main.cu`，单文件由
`scripts/sync_onefile_device.py` 同步、`device region identical: True`）：

- **device**：`fp8_mma_body` / `fa_bwd_fp8_mma_kernel` 加模板参数 `bool DET=false` 与三个
  默认实参 `float* dk_part, float* dv_part, int nblk`；GEMM3(dV)/GEMM4(dK) 的 epilogue 在
  `DET` 下改成 `dkv_det_store(dv_part + (((b*H + h)*nblk + mblk)*S + jg)*HD + d0 + c, a, b)`
  （一次 `float2` 覆盖写），否则保持 O4c 的 `red_add2`。
  新增 `dkv_reduce_kernel<HD, BM>`：partial `[((b*H+h)*nblk+mblk)*S*HD]` → `dk_acc`，按
  `h`（GQA 广播组）升序、`mblk` 升序求和；causal 下 KV 行 `jg` 只被 `mblk ≥ jg/BM` 写，
  故从 `jg/BM` 起求和。**partial 按 Q 头 `h` 分片**（不是 KV 头）：GQA 下多个 Q 头共享同一
  KV 头，只按 hkv 分片会互相覆盖（race）。
- **host**：`launch_bwd_main_det<...>`（`DET=true` 的薄壳，ksplit 固定 1）；`--det` / `--det=1`
  触发一段**同 session A/B**：`atomicAdd`（ksplit=1）vs `DET(partial+reduce)`，计时 + 跑两遍
  DET 验证 `runs[1-2] bitwise-diff` + 与 atomic 比 `max|diff|`。默认关，常规路径一行未改。

**范围**：仅定长（非 varlen）、`HD=128` 的默认 mma 路径（`--det` A/B 内固定 `ksplit=1`；
varlen/MLA/TMA/wgmma 路径未接入）。partial 元素数 `B*H*nblk*S*HD`（S=4096/H16/BM64 时
2.15 GB/缓冲、共 4.3 GB）。

### 56.3 数值：逐位可复现 + 与 atomic 同量级（`runs[1-2]` 两次跑）

| case（fp8） | atomic (ksplit=1) | DET | 比值 | `runs[1-2]` dk/dv | `DET-vs-atomic` dk/dv |
|---|---|---|---|---|---|
| S=512 MHA causal（两文件） | 0.1657 ms | 0.1723 ms | 0.961× | **0 / 0** | 3.58e-7 / 4.77e-7 |
| S=512 MHA causal（单文件） | 0.1676 ms | 0.1744 ms | 0.961× | **0 / 0** | 2.38e-7 / 4.77e-7 |
| S=4096 MHA causal | 2.3649 ms | 2.8588 ms | 0.827× | **0 / 0** | 9.54e-7 / 2.38e-6 |
| S=1024 GQA kv4 causal | 0.3882 ms | 0.4611 ms | 0.842× | **0 / 0** | 2.38e-6 / 3.81e-6 |
| S=1024 MHA full（非 causal） | 0.4182 ms | 0.4671 ms | 0.895× | **0 / 0** | 8.94e-8 / 5.96e-8 |

- **`runs[1-2] bitwise-diff dk/dv = 0.00e+00`**（三种 shape/两种 causal/单两文件）：确定性达成。
- `DET-vs-atomic` 差 ~e-7–e-6，是 **fp32 归约次序不同**造成的末位舍入（atomic 本身不收敛到
  某个确定值），与 fp8 容差（O(0.3)）无关。**最终 `ours vs fp32 ref` 与历史逐位一致**
  （S512 `2.426/2.975/3.735e-1`、GQA kv4 `2.517/5.408/7.072e-1`、full `5.518/5.309/4.007e-2`）。
- 单/两文件 A/B 数值逐位一致（S512 `0.961×`、diff 同量级）。

### 56.4 代价（确定性模式慢多少）

- 与**同一 ksplit=1 的 atomic** 比：S=4096 **0.827×**（+21%）、GQA kv4 0.842×、full 0.895×、
  S=512 0.961×。**慢的绝对值 ≈ 0.5 ms（S=4096）**，主要落在 reduce（见 §56.5）。
- 与**调优后的默认档**比（S=4096 默认 ksplit=4、main 1.5945 ms Hopper / 1.8915 ms
  sm_90）：DET 2.86 ms = 默认 main 的 **1.5–1.8×**；因为 DET 额外要求 `ksplit=1`
  （每 `(h,mblk)` 恰好一个 CTA 写 partial），牺牲了 split-K 的并行度。

### 56.5 ncu：bound = **全局 partial 写/读的 DRAM 带宽**，不是算力

`dkv_reduce_kernel`（S=4096 causal，`--set full`）:

| 指标 | 值 |
|---|---|
| Duration / DRAM / L2 / L1TEX / Compute | **731.6 µs** / **91.47%** / 88.76% / 9.70% / 12.89% |
| 带宽 / Achieved Occupancy | **3.07 TB/s** / 71.87% |

DET 主 kernel（S=4096 causal，ksplit=1，`--set full`）vs 历史 atomic 主 kernel：

| 指标 | DET 主 kernel | atomic 主 kernel（P117） |
|---|---|---|
| Duration | 2.05 ms | ~1.6 ms |
| DRAM | **33.34%**（写 2.20 GB） | **4.26%** |
| L2 / L1TEX / Compute / occ | 36.22% / 63.20% / 33.08% / 16.72% | 78.63% / 68.23% / 47.19% / 18.25% |
| `lts__t_sectors_op_red` | **1.57 M**（仅 dQ） | **114.5 M** |
| `lts__t_sectors_op_write` | 102.4 M | — |
| stall | wait 1.56 + short 1.23 + long 1.02 + barrier 0.40 | wait 1.59 + short 1.29 + long 0.60 |

**结论**：DET 把 atomic 归约（114.5 M red 扇区）换成 **partial 覆盖写（102.4 M write 扇区、
2.20 GB DRAM 写）+ 一次纯带宽 bound 的二次归约（731.6 µs、DRAM 91.5%）**。L2 red 压力消失
（L2 78.6%→36.2%），但 DRAM 从 4.3% 抬到 33.3%，且 reduce 自身 ~0.73 ms 是新增成本 ——
这就是「确定性」在本工作点的售价。对**需要跨运行逐位复现**（CI 数值回归、调试）的场景可接受；
对纯性能优先的生产路径不建议默认开。

### 56.6 复现 / 原始输出

```bash
# A/B（atomic vs DET，单/两文件；打印 runs[1-2] bitwise-diff）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --det --iters=10            # S=512
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --det --iters=10
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --det --iters=10     # 单文件
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:dkv_reduce_kernel --launch-count 1 -- --det --iters=1 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --kernel-name-base demangled \
  --kernel-name "regex:fa_bwd_fp8_mma_kernel.*, \(bool\)1>" --launch-count 1 -- --det --iters=1 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
```

原始输出：`src/fp8/fa_bwd_fp8_p34e_det_s512.out.txt`、
`..._mma_onefile_p34e_det_s512.out.txt`、`..._p34e_det_s4096.out.txt`、
`..._p34e_det_gqa_kv4.out.txt`、`..._p34e_det_s1024_full.out.txt`、
`..._p34e_default_s512.out.txt`（回归）、`..._p34e_ncu_reduce_s4096.out.txt`、
`..._p34e_ncu_detmain_s4096.out.txt`、`..._p34e_ncu_detmain_stall_s4096.out.txt`。

---

## 57. P3-4f（第一百一十九轮，**正结果，opt-in `--detk>1`**）：`--det` 扩展到 split-K（dQ 也走 partial）

### 57.1 动机（落实上一轮「下一步候选 ①」）

P3-4e 的 `--det` 固定 `ksplit=1`，注释理由是「DET 要求每个 `(h,mblk)` 恰好一个 CTA 写它的
partial」。但仔细分解后 **dK/dV 的 partial 其实不需要 part 维**：`ksplit` 是对**同一 m 块的
K/V 列块（ntile）** 切分，而一个 `(mblk, jg)` 只对应一个 K tile `nt=jg/BN`，**只属于一个
part**；各 part 写的是不相交的 `jg`，最后由 `dkv_reduce_kernel` 按 `mblk` 固定次序求和即可。
真正需要按 part 分片的只有 **dQ**：同一个 `(row=b*S+qi, h, c)` 会被该 mblk 的 `ksplit` 个 part
各贡献一个偏和。所以「扩展到 ksplit>1」= 给 dQ 加一份 part 分片 partial + 一次固定次序归约。

这样 DET 就能重新吃 split-K 的并行度：小 S 补满并发槽、大 S 削尾波——直接把 P3-4e「DET 要求
ksplit=1、白扔 split-K」的代价（S=4096 比默认档慢 1.5×）收回来一大截。

### 57.2 实现（单/两文件 device 逐字同源）

改动（`src/fp8/fa_bwd_fp8_kernels.cuh` + `fa_bwd_fp8_main.cu`，单文件由
`scripts/sync_onefile_device.py` 同步，`device region identical: True`）：

- **device**：`fp8_mma_body` / `fa_bwd_fp8_mma_kernel` 再加一个默认实参 `float* dq_part`；
  在 O7 的寄存器 dQ flush 处，`DET && ksplit>1` 时把每个 `(row,h)` 的 `dqacc` 写成
  `dq_part[((row*H + h)*ksplit + part)*HD + c]`（一次 `float2` **非原子覆盖写**），
  `ksplit==1` 时仍走原无竞争 `red_add2`（逐位不变）。
  新增 `dq_reduce_kernel<HD>`：`grid=(B*S, H)`、`block=HD`，每线程一个 `(row,h,c)` 按
  `part=0..ksplit-1` **固定次序**求和写回 `dq_acc`。空 part（causal 小 mblk 被切空）写不到，
  故 host 每次先把 `dq_part` 清零。
- **host**：`launch_bwd_main_det<...>` 增参 `int ksplit, float* dq_part`；新增 `--detk=N`
  （默认 1）。A/B 段改为在同一 binary 内对 `ksplit ∈ {1,2,4,8}`（由 `--detk` 指定）做
  atomic vs DET 计时 + 两次跑 DET 验证 `dq/dk/dv` 逐位可复现 + DET-vs-atomic。
  另：`ksplit>1` 时**强制 `REGDQ=true`**（dQ 必须先寄存器累加再写 partial；逐 tile 写会与
  同 CTA 内其它 tile 竞争）。

**范围**：仅定长（非 varlen）、`HD=128`、默认 mma 路径。dK/dV partial 布局与大小不变
（`B*H*nblk*S*HD`，S=4096 时 2.15 GB/缓冲）；新增 dQ partial `B*S*H*ksplit*HD`（S=4096、
ksplit=4 时 134 MB）。varlen/MLA/TMA/wgmma 未接入。

### 57.3 数值：`dq/dk/dv` 全逐位可复现（`runs[1-2] = 0`），与 atomic 同量级

同 session ksplit sweep（两文件，`--iters=30`/`10`）：

| case | DET k=1 | DET k=2 | DET k=4 | DET k=8 | 最优比 k=1 |
|---|---|---|---|---|---|
| S=512 MHA causal | 0.1461 ms | 0.1116 | **0.1115** | 0.1215 | **1.31×** |
| S=4096 MHA causal | 2.8380 ms | 2.6256 | **2.5514** | 2.6475 | **1.11×** |

- **`runs[1-2] bitwise dq/dk/dv = 0.00e+00`**：所有 ksplit（含 dQ！）、单/两文件全部达成
  确定性。
- `DET-vs-atomic dq/dk/dv` 全部 ~e-7–e-6（fp32 归约次序的末位舍入），与 fp8 容差 O(0.3) 无关。
- 默认路径（无 `--det`）数值逐位不变：S512 `ours vs ref 2.426/2.975/3.735e-1`、
  S4096 `2.635/2.643/3.216e-1`，与历史一致。

### 57.4 代价（同 session，atomic vs DET，含 reduce）

| case | atomic k=1 / k=4 | DET k=1 / k=4 | DET/atomic(k=4) |
|---|---|---|---|
| S=512 | 0.1262 / 0.0752 ms | 0.1461 / 0.1115 | 0.675× |
| S=4096 | 2.3773 / 1.8911 ms | 2.8380 / 2.5514 | 0.741× |

- atomic 随 ksplit 增大单调变快（split-K 并行度）；DET 在 **k=4 触底**（S=512 的 k=2≈k=4，
  取 k=4 统一），再大（k=8）反而回升——因为 DET 的固定成本（partial 写 + reduce 读）与
  ksplit 无关，而主 kernel 收益递减。
- **相对 P3-4e 的 k=1 DET**：S=512 **1.31×**、S=4096 **1.11×**；即「确定性 + split-K 并行度」
  把上一轮白扔的并行度收回了一大半。

### 57.5 ncu：DET 主 kernel 把成本记在 partial 写（DRAM），bound 仍是 reduce 的 DRAM 带宽

DET 主 kernel（`--set full`，唯一模板实例 `...,(bool)1>`）：

| 指标 | S=512 k=4 | S=4096 k=4 |
|---|---|---|
| Duration | 61.6 µs | **1.72 ms** |
| DRAM / L2 / L1TEX / Compute | 20.0 / 30.1 / 52.6 / 23.5 % | **42.3** / 49.0 / **69.8** / 42.7 % |
| regs / CTA-per-SM / achieved occ / Waves | 168 / 3 / 15.7% / 1.29 | 168 / 3 / 18.1% / 10.34 |
| 主 stall | long_scoreboard 30.8%（2.3/7.33 cyc） | — |

归约 kernel（`--set full`，S=4096 causal）：

| 指标 | `dkv_reduce_kernel` | `dq_reduce_kernel` |
|---|---|---|
| Duration / DRAM / Compute | **730.7 µs** / **91.56%** / 12.87% | 55.1 µs / 87.61% / 33.21% |
| 带宽 / occ | **3.07 TB/s** / 71.97% | 2.93 TB/s / 83.00% |

**结论**：拆分到 ksplit>1 后，DET 的成本结构与 P3-4e 一致——**主 kernel 因为多写一份 partial
（+2.1 GB）把 DRAM 从 4.3% 抬到 42.3%，真正的墙是 `dkv_reduce_kernel` 的纯 DRAM 带宽
（730.7 µs、91.6%、3.07 TB/s）**；`dq_reduce_kernel` 只占 55 µs。S=512 时两个 reduce 合计仅
~26 µs，故 DET 代价主要体现在大 S。要再快只能**减 partial 字节**（更细粒度的分块归约 /
按 KV 行分片的跨 warpgroup 偏和），与 P3-4e 的阻塞一致。

### 57.6 复现 / 原始输出

```bash
# ksplit sweep（同一 binary 内 atomic vs DET；打印 runs[1-2] bitwise-diff）
for k in 1 2 4 8; do scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --det --detk=$k --iters=10; done
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --det --detk=4 --iters=10   # 单文件
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:reduce_kernel --launch-count 2 -- --det --detk=4 --iters=1
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --kernel-name-base mangled \
  --kernel-name "regex:fa_bwd_fp8_mma_kernelILi128ELi64ELi32ELb1ELb0ELb1ELb1ELb1ELi128ELi2ELb0ELb1E" \
  --launch-count 1 -- --det --detk=4 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_p34f_det_s512.out.txt`、`..._p34f_det_s4096.out.txt`、
`..._mma_onefile_p34f_det_s512.out.txt`、`..._p34f_detsweep.out.txt`、
`..._p34f_ncu_reduce_s512.out.txt`、`..._p34f_ncu_reduce_s4096.out.txt`、
`..._p34f_ncu_detmain_s512.out.txt`、`..._p34f_ncu_detmain_s4096.out.txt`。

---

## 58. P3-4g（第一百二十轮，**正结果，opt-in `--det`**）：`--det` 扩到 Hopper TMA 快路（Q/dO/K/V-4D-TMA）

### 58.1 动机（落实第一百一十九轮「下一步候选 ①」）

P3-4e/P3-4f 的 `--det` 只挂在**默认 mma 路径**（`launch_bwd_main_det` → `fa_bwd_fp8_mma_kernel`）。
而 `--hopper`（`-DFA_WGMMA -DFA_TMA`，P3-4d 入标准 harness）下的**主路径**是 Q/dO/K/V 全
4D-TMA 的 `fa_bwd_fp8_mma_kvtma_kernel`（O41），它的 dK/dV 仍走跨 CTA `atomicAdd` ⇒ **快路无法
做确定性复现**。本项把 deterministic dK/dV 的 partial + 固定次序归约接进 TMA 快路。

好消息：DET 的 epilogue 早就写在**共享的 `fp8_mma_body`** 里（P3-4e），TMA 只是它的一个后端
（`TMA=true/KVTMA=true`）。所以 device 侧**一行数学都没改**，只是把 `DET` 模板参数与
`dk_part/dv_part/nblk/dq_part` 实参从两个 TMA 壳（`qdtma`/`kvtma`）透传进 body。

### 58.2 实现（单/两文件 device 逐字同源）

- **device（`fa_bwd_fp8_kernels.cuh` + 同步进 `..._mma_onefile.cu`，`sync_onefile_device.py`
  核对 `device region identical: True`）**：`fa_bwd_fp8_mma_qdtma_kernel` /
  `fa_bwd_fp8_mma_kvtma_kernel` 各加 `bool DET=false` 模板参数与
  `dk_part/dv_part/nblk/dq_part` 尾部默认实参，转调
  `fp8_mma_body<..., DET>`。默认 `DET=false` ⇒ 既有 TMA 路径逐位不变。
- **host（`fa_bwd_fp8_main.cu` + `..._mma_onefile.cu`）**：新增 `launch_bwd_main_kvtma_det`
  （定长、HD=128、ksplit=1，`DET=true` 实例化）；在既有 TMA A/B 段（`--det` 时）加
  **P3-4g A/B**：同 `ksplit=1` 下 `atomic vs DET`（隔离 partial+reduce 净开销），跑两遍 DET
  验证逐位可复现，再与 atomic 比 `max|diff|`。

范围：定长、HD=128（MHA/GQA）、ksplit=1、`-DFA_WGMMA -DFA_TMA` 构建。ksplit>1 与
varlen/MLA/TMA 的进一步覆盖仍列 backlog。

### 58.3 数值：逐位可复现，与 atomic 同量级（`runs[1-2]`）

三 shape × 单/两文件，`runs[1-2] bitwise dk/dv = 0.00e+00`（完全可复现）：

| case | 单/两文件 | `DET-vs-atomic` dk / dv |
|---|---|---|
| S512 H16 D128 | 两文件 | 4.77e-07 / 4.77e-07 |
| S512 H16 D128 | 单文件 | 2.38e-07 / 7.15e-07 |
| S1024 H32 D128 (kv4 GQA) | 两文件 | 2.86e-06 / 4.77e-06 |
| S4096 H16 D128 | 两文件 | 9.54e-07 / 1.07e-06 |
| S4096 H16 D128 | 单文件 | 9.54e-07 / 1.43e-06 |

`DET-vs-atomic` ~e-7–e-6（fp32 归约次序末位）。**`ours vs ref` 与历史逐位一致**：
S512 `2.426/2.972/3.733e-1`、S1024H32 `2.399/4.177/3.535e-1`、S1024 kv4 `2.517/5.339/7.173e-1`、
S4096 `2.635/2.644/3.216e-1`；默认路径（不加 `--det`）数值逐位不变（`--no-run --ci` 73 case 全绿）。
GQA 的 partial **按 Q 头 `h` 分片**（非 KV 头），归约端按广播组 `G=H/Hkv` 求和 ⇒ 正确覆盖 GQA。

### 58.4 代价（同 session，同 binary A/B，ksplit=1，含 `dkv_reduce`）

| case | atomic | DET(partial+reduce) | 比值 |
|---|---|---|---|
| S512 H16 | 0.120 ms | 0.139 ms | 0.86× |
| S1024 H32 | 0.334 ms | 0.407 ms | 0.82× |
| S1024 kv4 | 0.334 ms | 0.402 ms | 0.84× |
| S4096 H16 | 2.00 ms | 2.57 ms | 0.78× |

与 P3-4e/f 的结论一致：**确定性的售价 = 把 L2 原子归约换成一次性 DRAM partial 写读**，大 S 更贵。

### 58.5 ncu：主 kernel 把 L2 red 换成 DRAM partial 写；墙仍是 `dkv_reduce` 的 DRAM 带宽

DET TMA 主 kernel（`fa_bwd_fp8_mma_kvtma_kernel<...,(bool)DET=1>`，`--set full`，S=4096）：

| 指标 | DET TMA (`--det`) | 对照：atomic TMA（`docs/04` §42 / P3-4d） |
|---|---|---|
| Duration | **1.80 ms** | ~1.72 ms（atomic，ksplit 默认档） |
| DRAM / L2 / L1TEX / Compute | **38.3** / 42.5 / 72.0 / 33.9 % | 4.26 / **78.6** / 68.2 / 47.2 % |
| regs / smem / achieved occ / Waves | 168 / 74.82KB / 16.8% / 2.59 | 168 / 74.8KB / 18.3% / 20.7 |

即 DET 把 **L2 的 `red`（114.5M 扇区、78.6%）换成 partial 的 DRAM 写（38.3%）**；
主 kernel 的其它指标几乎不动。归约 kernel（`dkv_reduce_kernel`，S=4096）：

| 指标 | 值 |
|---|---|
| Duration / DRAM / L2 / Compute | **729.8 µs** / **91.68%** / 88.89% / 12.90% |
| 带宽 / occ / regs | **3.07 TB/s** / 71.93% / 40 |

与 P3-4e 的 731.6 µs / 3.07 TB/s 逐项一致。**bound = `dkv_reduce` 的纯 DRAM 带宽**（非算力）。

### 58.6 复现 / 原始输出

```bash
# 定长 Hopper 快路 + --det（打印 [P3-4g A/B]）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --det --iters=10
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --det --iters=10
# ncu（DET 主 kernel 的 mangled 实例尾为 ...Lb1ELb1ELb1ELb1ELb1E）
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --kernel-name-base mangled \
  --kernel-name regex:kvtma_kernelILi128ELi64ELi32ELb1ELb1ELb1ELb1ELb1E \
  --launch-count 1 -- --det --iters=1
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:dkv_reduce_kernel --launch-count 1 -- --det --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_p34g_det_{s512,s4096,gqa}.out.txt`、
`..._mma_onefile_p34g_det_{s512,s4096}.out.txt`、
`..._p34g_ncu_detmain_s4096.out.txt`、`..._p34g_ncu_reduce_s4096.out.txt`。

## 59. P3-4h（第一百二十一轮，**正结果，opt-in `--det`**）：`--det` 扩到 MLA（HD=512）

### 59.1 动机（落实第 119/120 轮「下一步候选 ①」的 MLA 部分）

P3-4e/f/g 的 `--det` 覆盖了 D=128 的 mma 默认路径与 Hopper TMA 快路。但 **MLA（HD=512）**
路径的 dK/dV 仍是跨 CTA `atomicAdd`，无法做确定性复现。MLA 与 D=128 的关键差别在于 **dQ 的
累加方式**：

- D=128：`kRegDq = REGDQ && (HD/NTW == 1)`，`NTW = WN*64 = 128`、`HD/NTW = 1` ⇒ 可把 dQ
  沿 n-tile 累加在寄存器里，`ksplit>1` 时写 partial、再由 `dq_reduce_kernel` 固定次序求和；
- MLA：`NTW = 2*64 = 128`、`HD/NTW = 4` ⇒ `kRegDq` **恒 false**，dQ 逐 tile 走 `red_add2`
  写 `dq_acc`。`ksplit>1` 时同一个 `(row,h,c)` 会被多个 part 的 CTA 原子加 ⇒ 非确定。
  因此 MLA 的确定性只能取 **`ksplit=1`**：此时每个 `(row,h)` 只属于唯一的 m 块 CTA，
  `red_add2` 退化为单写者（无竞争）⇒ dQ 也确定。

好消息：`dkv_reduce_kernel<HD,BM>` 的 partial 布局与求和次序本来就对任意 HD 成立，body 的
DET 分支（`dkv_det_store`）也 HD 无关。**device 数学一行未改**，只是 host 需要把 DET 路径
实例化到 MLA 的 8-warp/256 几何。

### 59.2 实现（单/两文件 device 逐字同源，`sync_onefile_device.py` 核对 `identical: True`）

- **device（`fa_bwd_fp8_kernels.cuh` + 同步进 `..._mma_onefile.cu`）**：**一行未改**。
- **host（`fa_bwd_fp8_main.cu` + `..._mma_onefile.cu`）**：
  - `launch_bwd_main_det` 加模板参数 `int NTH = THREADS, int NWAR = WN`（默认与旧实例逐字
    等价 ⇒ D=128 路径不变），使其能把 DET 实例化到 MLA 的 `256/4` 几何；
  - 新增 **P3-4h A/B**（`if (det_ab && D == 512 && !varlen)`）：`ksplit=1` 下 `atomic(256/4)`
    vs `DET(256/4)` 同几何对比（仅 DET 一个变量），跑两遍 DET 验逐位，`dkv_reduce_kernel<512,64>`
    以 512 线程启动（`c = threadIdx.x ∈ [0,512)` 覆盖 HD）。

范围：定长、D=512（MLA，MHA/GQA 均可）、ksplit=1、默认 mma 后端。ksplit>1 的 MLA DET 与
varlen 的 DET 仍列 backlog。

### 59.3 数值：逐位可复现，dQ 与 atomic 逐位同值（`runs[1-2]`）

三 shape × 单/两文件，`runs[1-2] bitwise dq/dk/dv = 0.00e+00`（完全可复现）：

| case | 单/两文件 | `DET-vs-atomic` dq / dk / dv |
|---|---|---|
| S256 H2 D512 | 两文件 | 0.00e+00 / 2.38e-07 / 3.58e-07 |
| S512 H4 D512 | 两文件 | 0.00e+00 / 4.77e-07 / 9.54e-07 |
| S512 H4 D512 | 单文件 | 0.00e+00 / 2.38e-07 / 9.54e-07 |
| S1024 H2 D512 | 两文件 | 0.00e+00 / 4.77e-07 / 9.54e-07 |
| S1024 H2 D512 | 单文件 | 0.00e+00 / 4.77e-07 / 7.15e-07 |

**dQ 恒 `0.00e+00`**（ksplit=1 单写者 vs atomic 同值）、dk/dv ~e-7–e-6（fp32 归约次序末位）。
`ours vs ref` 与历史逐位一致：S256H2 `2.356/2.290/3.441e-1`、S512H4 `2.415/2.992/4.481e-1`、
S1024H2 `2.232/3.337/3.602e-1`。默认路径（不加 `--det`）数值逐位不变。

### 59.4 代价（同 session，同 binary A/B，同 256/4 几何、ksplit=1，含 `dkv_reduce`）

| case | atomic(256/4) | DET(256/4) | 比值 |
|---|---|---|---|
| S256 H2 D512 | 0.147 ms | 0.142 ms | **1.035×** |
| S512 H4 D512 | 0.285 ms | 0.307 ms | 0.923× |
| S1024 H2 D512 | 0.551 ms | 0.586 ms | 0.940× |

与 P3-4e/f/g 同结论：**确定性的售价 = 把 L2 原子归约换成一次性 DRAM partial 写读**。小 S
（grid 小、atomic 竞争相对不划算）DET 净赚；大 S 净亏。

### 59.5 ncu：主 kernel 把 L2 red 换成 DRAM partial 写；墙仍是 `dkv_reduce` 的 DRAM 带宽

S=1024 H2 D512（`--set full`，`--det`，`--iters=1`）：

| 指标 | DET 主 kernel（256/4，ksplit=1） | atomic 主 kernel（256/4，ksplit=1） |
|---|---|---|
| Duration | **541.4 µs** | 611.4 µs |
| DRAM / L2 / L1TEX / Compute | **3.53** / 10.24 / 57.96 / 3.78 % | 0.83 / 11.23 / 54.14 / 4.12 % |
| regs / achieved occ / Waves | 255 / 12.5% / 0.24 | 245 / 12.5% / 0.24 |

归约 kernel（`dkv_reduce_kernel<512,64>`，S=1024H2）：

| 指标 | 值 |
|---|---|
| Duration / DRAM / L2 / Compute | **29.8 µs** / **76.35%** / 74.70% / 24.22% |
| 带宽 / occ / regs / Waves | **2.56 TB/s** / 65.63% / 40 / 5.17 |

即 DET 把主 kernel 的 L2 原子归约换成 partial 的 DRAM 写；归约 kernel 是 **纯 DRAM 带宽
bound**（与 P3-4e/f/g 逐项一致）。注意 ncu 锁频单次下 DET 主 kernel 反而比 atomic 主 kernel
略快（541 vs 611µs，atomic 的 L2 `red` 在锁频下更贵），但含 reduce 的 event 稳态在 S512/S1024
上 DET 净慢（0.92–0.94×）——**锁频 ncu 读机制，event 稳态定发布口径**（同 `docs/01` 的教训）。

### 59.6 复现 / 原始输出

```bash
# 定长 MLA + --det（打印 [P3-4h A/B]）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 --det --iters=10
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 --det --iters=10
# ncu（DET MLA 主 kernel 的 mangled 实例尾为 ...ELi256ELi4ELb0ELb1E；atomic 为 ...ELb0ELb0E）
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --kernel-name-base mangled \
  --kernel-name regex:ILi512ELi64ELi32ELb0ELb0ELb1ELb1ELb1ELi256ELi4ELb0ELb1E \
  --launch-count 1 -- --dir=.../b1_s1024_h2_d512_causal_fp8 --det --iters=1
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:dkv_reduce_kernelILi512 --launch-count 1 -- \
  --dir=.../b1_s1024_h2_d512_causal_fp8 --det --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_p34h_det_b1_{s256_h2,s512_h4,s1024_h2}_d512_causal_fp8.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_p34h_det_b1_{s512_h4,s1024_h2}_d512_causal_fp8.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34h_ncu_{detmain,atomicmain,reduce}_s1024h2.out.txt`。

---

## 60. P3-4i：`--det` 扩到 varlen（第一百二十二轮）

### 60.1 动机（落实第一百二十一轮「下一步候选 ①」的最后一块）

第一百一十八~一百二十一轮把 fp8 反向的确定性模式 `--det` 依次扩到 **定长 → split-K → Hopper
TMA 快路 → MLA**，只剩 **varlen（packed `[T,H,D]` + `cu_seqlens`）** 未覆盖。varlen 的 dK/dV
同样跨 CTA `atomicAdd`（`fp8_mma_body` 的 `epi_dv/epi_dk`），调度一变末位就抖、无法位复现。
本项把同一 partial + 固定次序归约机制接上变长路径。

### 60.2 设计：partial 布局沿用定长式、归约按逐序列定界

关键观察：**body 的 DET 分支只依赖 `(b, h, mblk, jg, S, nblk)`，与定长/变长无关**。
varlen 的 main kernel 已经把 `S=maxlen`（packed 索引由 `qbase` 定界）传进 body，因此：

1. **device 数学一行未改**——body 的 `dkv_det_store(...(((b*H+h)*nblk+mblk)*S+jg)*HD+c...)`
   在 varlen 下只要 host 传 `S=maxlen`、`nblk=nblk_max=ceil(maxlen/BM)` 即成立（`jg<len_b`、
   `mblk<nblk_b` 恒在范围内）。
2. **只新增一个归约 kernel** `dkv_reduce_varlen_kernel<HD,BM>`：grid=`(B*Hkv, maxlen)`，
   按 `cu_seqlens` 解出 `len_b=qbase` 差、`nblk_b=ceil(len_b/BM)`，只对 `jg<len_b` 的行、
   `mblk∈[causal? jg/BM : 0, nblk_b)` 求和（`hh` 升序、`m` 升序 **固定次序**），输出按
   packed token `qbase+jg` 写 `[T,Hkv,D]`。
3. dQ 的确定性沿用 P3-4f：`ksplit==1` 时单写者 `red_add2`（确定）；`ksplit>1` 时 dQ 也走
   `dq_part`（按 `part` 分片）+ `dq_reduce_kernel`（它对 packed 全局行号天然成立）。故 A/B
   强制 `REGDQ=true`。
4. `launch_bwd_main_det` 加了尾部模板参数 `WGMMA=false` 与默认实参
   `cu_seqlens/mt_b/mt_m`：定长调用不传 ⇒ **逐字不变**；varlen D=128（构建恒 `-DFA_WGMMA`、
   默认主 kernel 走 WGMMA）传 `WGMMA=true` + `d_cu`。

partial 缓冲沿用 maxlen-strided 布局 `B*H*nblk_max*maxlen*HD`（本例最大 b5 H32 → 5.4GB/缓冲；
D=128 varlen 的 maxlen≤2048，可接受；MLA/D=512 varlen 未纳入本项）。

### 60.3 数值：两次跑逐位可复现；DET-vs-atomic 仅 fp32 次序末位

| case（fp8 causal） | ksplit | atomic | DET | 比 | runs[1-2] bitwise dq/dk/dv | DET-vs-atomic dq/dk/dv |
|---|---|---|---|---|---|---|
| b1_t512 MHA | 1 | 0.1263 | 0.1394 | 0.906× | 0 / 0 / 0 | 0 / 4.77e-7 / 4.77e-7 |
| b1_t512 MHA | 4 | 0.0748 | 0.1049 | 0.713× | 0 / 0 / 0 | 5.96e-8 / 4.77e-7 / 7.15e-7 |
| b4_t3840 MHA 不齐 | 4 | 0.7641 | 1.0389 | 0.736× | 0 / 0 / 0 | 1.19e-7 / 7.15e-7 / 7.15e-7 |
| b5_t3968 GQA q32/kv8 | 4 | 1.4242 | 1.9746 | 0.721× | 0 / 0 / 0 | 1.19e-7 / 1.43e-6 / 1.67e-6 |

- **`runs[1-2] bitwise = 0` 全部成立**（含 GQA 广播组、含不齐 length 的逐序列 `nblk_b`）。
- `DET-vs-atomic` 只有 fp32 归约次序的末位差（e-7–e-6），**数学口径一致**。
- `ksplit==1` 时 dQ 逐位等于 atomic（同一单写者 `red_add2`）。
- 恢复默认路（`run_all()`）后 `ours vs fp32 ref`：b1 `2.280/3.108/3.422e-1`、
  b4 `2.935/2.938/4.179e-1`、b5(GQA) `3.094/5.567/6.203e-1`——**全 fp8 噪声量级**。
- 单/两文件一致：onefile b1 k4 `0.0761/0.1055`、b4 k4 `0.7667/1.0367`（与两文件同量级）。

### 60.4 代价与 ncu：bound 仍是 reduce 的纯 DRAM 带宽

- **代价**（同 session 同 binary A/B，含 reduce）：DET/atomic = **0.71–0.91×**（ksplit=1 最轻，
  ksplit=4 因固定 partial 写读成本而 heavier）。与定长 P3-4e/f 同构——确定性 = 把 L2 原子归约
  换成「partial 的 DRAM 写 + 一趟 DRAM 读归约」。
- **ncu（reduce, b4_t3840, DET ksplit=4）**：`dkv_reduce_varlen_kernel<128,64>` **274.05µs**、
  **DRAM 87.40%** / L2 85.33% / L1TEX 12.09% / SM 26.32%、occ 56.37%、读 744.53MB/写 58.26MB
  ⇒ **纯 DRAM 带宽 bound**（与 P3-4e/f/g/h 逐项一致）。主 kernel 的 DET 路径与 P3-4e 共用同一
  body/epilogue，bound 结论沿用（见 §56）。

### 60.5 复现 / 原始输出

```bash
# varlen + --det（打印 [P3-4i A/B]）；varlen 构建需 sm90a + -DFA_WGMMA
ARCH= NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b4_t3840_h16_d128_causal_fp8 \
  --varlen --det --detk=4 --iters=30
# ncu（varlen reduce）
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:dkv_reduce_varlen \
  --launch-count 1 --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  -- --dir=.../varlen_b4_t3840_h16_d128_causal_fp8 --varlen --det --detk=4 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_p34i_det_varlen_{b1_t512,b1_t512_k4,b4_t3840_k4,b5_t3968_gqa_k4}.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_p34i_det_varlen_{b1_t512_k4,b4_t3840_k4}.out.txt`、
`src/fp8/fa_bwd_fp8_p34i_ncu_reduce_varlen_b4.out.txt`、`src/fa_bwd_p34i_fa_baseline_fp16.out.txt`。

**对标（纯反向，fp16 变长，`fa_vs_te_bwd_only.py fp16`）**：FA3 不齐 `[512,1024,2048,256]`
H16 **0.1621ms/282TF**、等长 4×1024 **0.1471/234**、GQA q32/kv8 **0.3766/243**、强倾斜
`[2048,512,…]` **0.1418/259**；TE2.14 反向不支持 ragged QKV（NA）。ours fp8 变长默认路
（含 quant+preprocess）b4_t3840 total 0.9274ms/49.2TF（docs §37 口径）。DET 为 opt-in 正确性
模式，代价见 §60.4。

---

## 61. P3-4j：`--det` 扩到 MLA（HD=512）varlen（第一百二十三轮）

### 61.1 动机（落实第一百二十二轮「下一步候选 ②」）

P3-4i 把 `--det` 扩到了 **D=128** 的变长，但 **MLA（HD=512）的变长**仍是跨 CTA
`atomicAdd`——调度一变末位就抖。P3-4h 已把 DET 扩到 **定长 MLA**，并指出关键约束：
MLA 的 dQ 无法用寄存器累加（`kRegDq = REGDQ && (HD/NTW==1)`，HD=512 时 `HD/NTW=4`
⇒ 恒 false），所以「确定 dQ」只能靠 **ksplit=1 的单写者 `red_add2`**。本项把同一结论
搬到**变长**：device 数学一行未改，只补 host A/B + 复用 `dkv_reduce_varlen_kernel<512,64>`。

### 61.2 设计：定长 MLA DET + varlen 归约

- **partial 布局**沿用定长式 `part[((b*H+h)*nblk_max+mblk)*maxlen + jg]`（body 传
  `S=maxlen`、`nblk=nblk_max`），归约端换成 `dkv_reduce_varlen_kernel<HD,BM>`（按
  `cu_seqlens` 的逐序列 `len_b/nblk_b` 定界、只对 `jg<len_b` 求和、输出按 packed token
  `qbase+jg` 定位）。**该 kernel 模板化在 HD 上，`<512,64>` 直接可用**——这是 P3-4i 把
  partial 布局抽象成「定长式参数化」的直接红利。
- **ksplit 锁 1**：与 P3-4h 定长 MLA 同因（dQ 不可寄存器累加 ⇒ partial 非原子覆盖写会在
  同 CTA 的 nt 循环里互相覆盖）。A/B 的 atomic 基线与 DET 同为 **O47/O51 的 8-warp/256
  线程几何**（`launch_bwd_main<512,64,32,false,false,true,true,true,256,4>` vs
  `launch_bwd_main_det<512,64,32,false,true,true,true,256,4>`），只差 DET 一个变量。
  注：默认 MLA varlen 主 kernel 走 kvpipe 版，DET 未接进 kvpipe（与 P3-4h 一致）。
- **改动范围**：仅 `fa_bwd_fp8_main.cu` / `fa_bwd_fp8_mma_onefile.cu` 的 `run_varlen`
  各 +76 行 host A/B；`fa_bwd_fp8_kernels.cuh` **一行未改**。

### 61.3 数值：两次跑逐位可复现；DET-vs-atomic 仅 fp32 次序末位

| case | 后端 | atomic(8w) ms | DET(8w) ms | 比 | runs[1-2] bitwise dq/dk/dv | DET-vs-atomic dq/dk/dv |
|---|---|---|---|---|---|---|
| b1_t512 causal | 两文件 | 0.2856 | 0.2810 | **1.016×** | 0 / 0 / 0 | 0 / 4.77e-7 / 4.77e-7 |
| b3_t1792 causal | 两文件 | 0.5795 | 0.6109 | **0.949×** | 0 / 0 / 0 | 0 / 4.77e-7 / 5.96e-7 |
| b1_t512 full | 两文件 | 0.2868 | 0.2996 | **0.957×** | 0 / 0 / 0 | 0 / 5.96e-8 / 5.96e-8 |
| b1_t512 causal | 单文件 | 0.2797 | 0.2789 | **1.003×** | 0 / 0 / 0 | 0 / 4.77e-7 / 4.77e-7 |
| b3_t1792 causal | 单文件 | 0.5794 | 0.6078 | **0.953×** | 0 / 0 / 0 | 0 / 7.15e-7 / 9.54e-7 |

- **`runs[1-2] bitwise dq/dk/dv = 0` 全部成立**（含不齐 length 的逐序列 `nblk_b`）。
- `DET-vs-atomic` **dq 恒 0**（ksplit=1 单写者，与 atomic 逐位同值）、dk/dv e-7–e-6
  （fp32 归约次序末位），**数学口径一致**。
- **恢复默认路（`run_all()`）后 `ours vs fp32 ref` 与历史逐位一致**：b1 causal
  `1.613e-1/2.238e-1/3.864e-1`、b3 causal `3.404e-1/3.436e-1/3.508e-1`、b1 full
  `5.260e-2/5.222e-2/4.218e-2`——全 fp8 噪声量级（D=512/MLA 的 FA3/TE 反向均不支持，
  仅 fp32 ref 可对）。
- **单/两文件一致**：b1 causal 两文件 1.613e-1/2.238e-1/3.864e-1、单文件同值；
  b3 causal 两文件 3.404e-1/3.436e-1/3.508e-1、单文件同值。

### 61.4 代价与 ncu：bound 仍是 reduce 的纯 DRAM/L2 带宽

- **代价（同 session 同 binary A/B，只差 DET 一个变量）**：DET/atomic = **0.95–1.02×**
  （main kernel 本体几乎免费——确定性只是把 dK/dV 的 L2 `red` 换成 partial 写）。
- **真正的代价在「ksplit 必锁 1」**：MLA varlen 的 auto ksplit 很大（b1 auto=16、b3 auto=4），
  强制 k=1 后主 kernel 并行度下降——同 session 默认 8w+kvpipe main-only：b1 **0.0502ms**
  vs DET(k=1) 0.281ms（**~5.6×**）、b3 0.188ms vs 0.611ms（**~3.25×**）。**这与 P3-4h
  定长 MLA 的结论一致**：确定性在 MLA 上不是「免费换归约方式」，而是「用 ksplit=1 的单写者
  换掉 split-K 并行度」。要恢复并行度需让 MLA 的 dQ 也能进 partial（见 §61.5）。
- **ncu（reduce, b3_t1792 causal, DET k=1）**：`dkv_reduce_varlen_kernel<512,64>`
  grid `(6,1024,1)` × 512、**Duration 48.03µs**、**DRAM 65.81%** / **L2 67.59%** /
  L1TEX 10.34% / SM 29.79% / **occ 42.71%**、读 95.43MB/写 10.45MB ⇒ **bound = reduce 的
  纯 DRAM/L2 带宽**（与 P3-4e/f/g/h/i 逐项一致）。主 kernel 的 DET 路径与 P3-4e 共用同一
  body/epilogue，bound 结论沿用 §56。

### 61.5 复现 / 原始输出 / 下一步

```bash
# MLA varlen + --det（打印 [P3-4j A/B]）；varlen 构建需 sm90a + -DFA_WGMMA
ARCH= NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b3_t1792_h2_d512_causal_fp8 \
  --varlen --det --iters=20            # full 加 --full
# ncu（varlen MLA reduce）
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:dkv_reduce_varlen \
  --launch-count 1 --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  -- --dir=.../varlen_b3_t1792_h2_d512_causal_fp8 --varlen --det --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_p34j_det_varlen_{b1_t512,b3_t1792,b1_t512_full}.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_p34j_det_varlen_{b1_t512,b3_t1792}.out.txt`、
`src/fp8/fa_bwd_fp8_p34j_ncu_reduce_varlen_b3.out.txt`。

**下一步候选**：① **让 MLA 的 dQ 也进 partial，从而支持 DET 的 ksplit>1**——观察：非
`kRegDq` 的 dQ epilogue 是对**唯一 CTA**（固定 `(mblk,part)`）的 `red_add2`，
若把目标从 `dq_acc` 换成 `dq_part[((row*H+h)*ksplit+part)*HD+c]`（host 预先清零），
则同一 partial 元素仍只被一个 CTA 写、CTA 内 nt 次序固定 ⇒ **确定性天然成立**，且
`dq_reduce_kernel` 已按 part 固定次序求和。这样 MLA DET 就能保留 split-K 并行度（代价
是 ksplit× 的 atomic 流量 + partial 读写）。② 长序列的 partial **compact per-sequence
布局**（现 maxlen-strided，强倾斜时浪费）；③ 其余性能仍受本卡寄存器/smem 硬墙锁定。

---

## 62. P3-4k（第一百二十四轮，**正结果，opt-in `--detk>1`**）：MLA（HD=512）的 dQ 进 partial ⇒ DET 支持 split-K

### 62.1 动机（落实上一轮「下一步候选 ①」）

P3-4h/P3-4j 把 `--det` 扩到了 MLA（定长 / varlen），但**锁死 `ksplit=1`**：MLA 的
`kRegDq = REGDQ && (HD/NTW==1)` 恒 false（`HD=512` 时 `HD/NTW=4`），dQ 不能像 D=128 那样先在
寄存器里跨 nt 累加再**一次性覆盖写** partial，只能逐 tile `red_add2`。若 `ksplit>1`，同一个
`(row,h,c)` 会被多个 part 的 CTA 原子相加 ⇒ 调度一变末位就抖。于是上一轮的结论是「确定性在
MLA 上 = 用单写者换掉 split-K 并行度」，主 kernel 相对 auto split 慢 3.3–5.6×。

**本轮的关键观察**（与 §61.5 候选 ① 一致）：非 `kRegDq` 的 dQ epilogue 虽然逐 tile 写，但每个
`(mblk, part)` 是**唯一一个 CTA**，把目标从 `dq_acc` 换成按 part 分片的
`dq_part`（`((row*H+h)*ksplit+part)*HD+c`）后：

- **同一 partial 元素只被一个 CTA 写**（固定 `(mblk,h,part)`）⇒ 无跨 CTA 竞争；
- **CTA 内**同一个 `(row,c)` 由同一线程按 `nt` **程序序**写（每个 nt tile 累加一次）⇒
  `red_add2` 虽是跨 CTA 设计，对本 CTA 私有区仍是**确定次序**；
- `dq_reduce_kernel`（P3-4f 已就位）按 `part=0..ksplit-1` 固定次序求和 ⇒ **跨 part 也确定**。

因此**确定性天然成立**，无需寄存器累加，MLA 的 DET 就能保留 split-K 并行度。

### 62.2 实现（单/两文件 device 逐字同源）

改动（`src/fp8/fa_bwd_fp8_kernels.cuh` + `fa_bwd_fp8_main.cu`，单文件由
`scripts/sync_onefile_device.py` 同步，`device region identical: True`）：

- **device**（`fp8_mma_body` 的 dQ epilogue，非 `kRegDq` 分支）：新增
  `if constexpr (DET)`，`ksplit>1` 时把目标从 `dq_acc + ((row*H+h)*HD + d0+c)` 改成
  `dq_part + (((row*H+h)*ksplit + part)*HD + d0+c)`（`row=qbase+qi`，仍是 `float2` 的
  `red_add2`）；`ksplit==1` 时保持原无竞争 `red_add2`（**逐位不变**）。`DET=false` 时整段与
  历史逐字相同。`dq_part` 早就是 body 的尾部默认实参（P3-4f 加的），**无需改签名**。
- **host**：`--detk=N` 现在对 MLA 也生效——把 P3-4h 的 A/B 段扩成与 P3-4f 同构：分配
  `dq_part`（`B*S*H*ksplit*HD`）、每次跑前清零（空 part 需为 0）、DET 跑完接
  `dkv_reduce_kernel<512,64>` + `dq_reduce_kernel<512>`；atomic 参照用**同 256/4 几何、同
  ksplit** 的非 kvpipe 主 kernel（只差 DET 一个变量）。新增 `DET k=1 → k=N` 的 split speedup
  打印。单/两文件 host 同步。

**范围**：定长、`HD=512`（MLA）、默认 mma 后端。`ksplit=1` 与 P3-4h 逐位一致；D=128
（P3-4f）路径不受影响（kRegDq 分支未动）。MLA varlen 的 P3-4j A/B 仍锁 `ksplit=1`（device
已支持，host 未接线），留 backlog。

### 62.3 数值：`ksplit>1` 下 dq/dk/dv 全逐位可复现；与 atomic 同量级

同 session（两文件 `--iters=20`，`--det --detk=4`）：

| case | atomic k=4 ms | DET k=4 ms | DET/atomic | DET k=1 ms | k1→k4 split 提速 | runs[1-2] bitwise dq/dk/dv | DET-vs-atomic dq/dk/dv |
|---|---|---|---|---|---|---|---|
| S=256 H=2 | 0.0465 | 0.0633 | 0.734× | 0.1683 | **2.657×** | 0 / 0 / 0 | 1.19e-7 / 2.38e-7 / 2.38e-7 |
| S=512 H=4 | 0.0960 | 0.1574 | 0.610× | 0.3767 | **2.393×** | 0 / 0 / 0 | 1.19e-7 / 2.38e-7 / 4.77e-7 |
| S=1024 H=2 | 0.1701 | 0.2589 | 0.657× | 0.7238 | **2.796×** | 0 / 0 / 0 | 1.19e-7 / 4.77e-7 / 7.15e-7 |

ksplit sweep（S=1024 H=2，两文件）：

| ksplit | atomic ms | DET ms | DET/atomic | DET split 提速 (vs k=1 0.7256) | runs[1-2] |
|---|---|---|---|---|---|
| 1 | 0.5522 | 0.7256 | 0.761× | 1.000× | 0 / 0 / 0 |
| 4 | 0.1701 | **0.2589** | 0.657× | **2.80×** | 0 / 0 / 0 |
| 8 | 0.1482 | 0.2650 | 0.559× | 2.744× | 0 / 0 / 0 |
| 16 | 0.1312 | 0.2738 | 0.479× | 2.639× | 0 / 0 / 0 |

- **`runs[1-2] bitwise dq/dk/dv = 0.00e+00` 全部成立**（含 dQ、含 ksplit>1），单/两文件一致
  （单文件 S1024H2：0.661×、2.807×、runs 0/0/0；S512H4：0.573×、2.413×、runs 0/0/0）。
- `DET-vs-atomic`：`ksplit=1` 时 **dq 恒 0**（单写者同值，与 §59 一致）、`ksplit>1` 时
  dq/dk/dv ~e-7（fp32 归约次序末位），与 fp8 容差 O(0.3) 无关。
- **默认路径（无 `--det`）数值逐位不变**：S256H2 `2.356/2.290/3.441e-1`、S512H4
  `2.415/2.992/4.481e-1`、S1024H2 `2.232/3.337/3.602e-1`，与 P3-4h 记录逐位一致；D=128
  回归（P3-4f A/B）`0.676× / runs 0 / e-7` 与历史一致，S512 `ours vs ref 2.426/2.975/3.735e-1`。

### 62.4 代价与 ncu：DET 在 k=4 触底；bound 仍是 reduce 的纯 DRAM 带宽

- **DET 在 k=4 触底**（S1024H2：k=4 0.2589 < k=8 0.2650 < k=16 0.2738）：DET 的固定成本
  （partial 写 + reduce 读）与 ksplit 无关，主 kernel 收益递减——与 P3-4f（D=128）同形。
- **相对 P3-4h 的「锁 k=1」**：S1024H2 DET 主链从 0.7256 → **0.2589 ms（2.80×）**，把上一轮
  白扔的 split-K 并行度收回一大截；但 DET 仍比 atomic 慢（0.61–0.73×，见上表），差在
  partial 写 + 两次 reduce。
- **ncu（S=1024H2, DET k=4）**：
  - DET 主 kernel：**Duration 133.34 µs**、DRAM 3.79% / **L2 51.55%** / L1TEX 40.60% /
    Compute 22.95%、249 regs（**0 spill**）、229.89KB 动态 smem、Block Limit 1、
    achieved occ **12.48%**、**Waves 3.88**、Issue Slots Busy 22.95% ⇒ **bound = L2 归约流量
    + 低 occupancy**（主 kernel 本体比 atomic 版更快，因为把 `red` 换成了普通 partial 写）。
  - `dkv_reduce_kernel<512,64>`：**29.60 µs / DRAM 77.06% / L2 75.76%** / L1TEX 11.29% /
    occ 65.30% / Waves 5.17 ⇒ **纯 DRAM/L2 带宽 bound**。
  - `dq_reduce_kernel<512>`：8.74 µs / DRAM 57.88% / L2 64.14% / occ 74.70%。
  - 即 DET 的增量 ~38 µs（reduce）+ partial 清零/读写；**真正的墙仍是 `dkv_reduce_kernel` 的
    DRAM 带宽**（与 P3-4e/f/g/h/i/j 逐项一致）。

### 62.5 对标

MLA（`head_dim=512`）的 **FA3/TE 反向均不支持**（FA 限 `head_dim≤256`、TE 训练 bwd 限 256），
只有 fp32 ref 可对，故本轮无 FA/TE 同 shape 对标；纯反向口径的 D=128 列见 `docs/04` §38。

### 62.6 复现 / 原始输出 / 下一步

```bash
# 定长 MLA + --det --detk=N（打印 [P3-4k A/B]；默认 mma 后端）
for k in 1 4 8 16; do scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 \
  --causal --det --detk=$k --iters=20; done
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 \
  --causal --det --detk=4 --iters=20                # 单文件
# ncu（DET 主 kernel / 两个 reduce）
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:fa_bwd_fp8_mma_kernel -c 1 -- --dir=... --causal --det --detk=4 --iters=1
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:dkv_reduce_kernel -c 1 -- --dir=... --causal --det --detk=4 --iters=1
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:dq_reduce_kernel -c 1 -- --dir=... --causal --det --detk=4 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_main_p34k_det_b1_{s1024_h2,s512_h4,s256_h2}_d512_causal_fp8.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34k_detk{1,8,16}_b1_s1024_h2_d512_causal_fp8.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_p34k_det_b1_{s1024_h2,s512_h4}_d512_causal_fp8.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34k_regr_d128_s512.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34k_ncu_{detmain,dkvreduce,dqreduce}_s1024h2.out.txt`。

**下一步候选**：① **把 MLA varlen 的 P3-4j A/B 也接上 `ksplit>1`**（device 已支持；host 补
`dq_part` 清零 + `dq_reduce_kernel<512>` 的 packed 定位即可）；② 减 partial 字节（按 KV 行
跨 warpgroup 偏和 / 更细分块）以压低 reduce 的 DRAM 墙；③ 性能（非确定性）仍受本卡寄存器/
smem 硬墙锁定，见 ROADMAP「阻塞」。

## 63. P3-4l（第一百二十五轮，**正结果，opt-in `--detk>1`**）：MLA（HD=512）varlen 的 DET 接上 split-K（候选 ① 收口）

### 63.1 动机（落实上一轮「下一步候选 ①」）

P3-4k 让 **定长** MLA 的 dQ 也进 partial，从而 `--det` 支持 `ksplit>1`；但 P3-4j 的
**MLA varlen** A/B 仍写死 `ksplit=1`（device 早已支持、host 未接线）。本轮的观察与
P3-4k 完全同构：

- varlen 的 body 的 DET 分支只依赖 `(b,h,mblk,jg,S,nblk)`，与定长/变长无关；
- dQ 的 per-part partial 用 `((qbase+qi)*H + h)*ksplit + part`，其中 `qbase+qi` 就是 packed
  全局 q token，与定长 `dq_reduce_kernel` 的 `row` 语义**逐字相同**；
- 故只需在 host 把 `dq_part` 分配/清零、把 `dq_reduce_kernel<512>`（`grid=(T,H)`、
  `block=512`）接到 `dkv_reduce_varlen_kernel<512,64>` 之后即可，**device 一行未改**。

### 63.2 实现（host-only；单/两文件 device 逐字同源）

改动只在 `run_varlen` 的 P3-4j A/B 段（`fa_bwd_fp8_main.cu` + `fa_bwd_fp8_mma_onefile.cu`
各 +~20 行），与 P3-4i/P3-4k 同构：

- 取 `ks = det_ksplit<1 ? 1 : det_ksplit`；`g = dim3(nblk_max*ks, H, B)`；
- `dq_part` 大小 `T*H*ks*HD`，`ks>1` 时分配、每跑前 `cudaMemset(...,0)`（空 part 需为 0）；
- atomic 参照与 DET 用**同 8-warp/256 几何、同 ksplit**的非 kvpipe 主 kernel（只差 DET）；
- DET 跑完接 `dkv_reduce_varlen_kernel<512,64>`，`ks>1` 时再接
  `dq_reduce_kernel<512><<<(T,H),512>>>` 按 `part` 固定次序求和。

`ksplit==1` 时走单写者 `red_add2`，与 P3-4j **逐位一致**。单文件由
`scripts/sync_onefile_device.py` 核对 `device region identical: True`（device 未动）。

### 63.3 数值：`ksplit>1` 下 dq/dk/dv 全逐位可复现

两文件 `--iters=30`，同 session：

| case | ksplit | atomic ms | DET ms | DET/atomic | runs[1-2] bitwise dq/dk/dv | DET-vs-atomic dq/dk/dv |
|---|---|---|---|---|---|---|
| b1_t512 causal | 1 | 0.2834 | 0.3424 | 0.828× | 0 / 0 / 0 | 0 / 4.77e-7 / 4.77e-7 |
| b1_t512 causal | 4 | 0.0805 | 0.1236 | 0.651× | 0 / 0 / 0 | 1.19e-7 / 2.38e-7 / 4.77e-7 |
| b1_t512 causal | 8 | 0.0556 | **0.0929** | 0.599× | 0 / 0 / 0 | 1.19e-7 / 2.38e-7 / 4.77e-7 |
| b1_t512 causal | 16 | 0.0585 | 0.1220 | 0.480× | 0 / 0 / 0 | 1.19e-7 / 4.77e-7 / 4.77e-7 |
| b3_t1792 causal | 1 | 0.5795 | 0.7533 | 0.769× | 0 / 0 / 0 | 0 / 4.77e-7 / 7.15e-7 |
| b3_t1792 causal | 4 | 0.2150 | 0.3649 | 0.589× | 0 / 0 / 0 | 1.19e-7 / 3.58e-7 / 7.15e-7 |
| b3_t1792 causal | 8 | 0.1868 | **0.3544** | 0.527× | 0 / 0 / 0 | 1.19e-7 / 2.38e-7 / 4.77e-7 |
| b3_t1792 causal | 16 | 0.1719 | 0.3898 | 0.441× | 0 / 0 / 0 | 1.19e-7 / 3.58e-7 / 4.77e-7 |
| b1_t512 full | 4 | 0.0845 | 0.1323 | 0.638× | 0 / 0 / 0 | 8.94e-8 / 5.96e-8 / 5.96e-8 |

- **`runs[1-2] bitwise dq/dk/dv = 0.00e+00` 全部成立**（含 dQ、含 ksplit>1），单/两文件一致
  （单文件 b1 k=4：0.653×、runs 0/0/0；b3 k=4：0.591×、runs 0/0/0）。
- `ksplit=1` 时 `DET-vs-atomic` **dq 恒 0**（单写者同值，与 P3-4j 一致）；`ksplit>1` 时 dq/dk/dv
  ~e-7（fp32 归约次序末位），与 fp8 容差 O(0.3) 无关。
- **恢复默认路后 `ours vs ref` 与历史逐位一致**：b1 causal `1.613e-1/2.238e-1/3.864e-1`、
  b3 causal `3.404e-1/3.436e-1/3.508e-1`、b1 full `5.260e-2/5.222e-2/4.218e-2`。全 fp8 噪声量级。

### 63.4 代价与 ncu：DET 在 k=8 触底；bound 仍是 reduce 的纯 DRAM 带宽

- **DET 触底点比定长 MLA（k=4）更靠后**：b1_t512 k=8（0.0929）< k=4（0.1236）< k=16（0.1220）、
  b3_t1792 k=8（0.3544）< k=4（0.3649）< k=16（0.3898）。varlen 的短序列在 k=4 时并行度仍不足，
  故 k=8 才吃饱；k=16 后固定 reduce 成本占优、反升。
- **相对 P3-4j「锁 k=1」**：b1_t512 DET 主链 0.3424 → **0.0929 ms（3.69×）**、b3_t1792
  0.7533 → **0.3544 ms（2.13×）**——把 P3-4j 白扔的 split-K 并行度收回。
- **ncu（b3_t1792 causal, DET k=8）**：
  - `dkv_reduce_varlen_kernel<512,64>`（grid `(6,1024,1)`×512）：**48.13 µs / DRAM 65.68% /
    L2 67.48%** / L1TEX 10.38% / SM 29.90% / occ 42.67%、读 95.43MB / 写 10.46MB ⇒ **纯 DRAM/L2
    带宽 bound**（与 P3-4j 的 48.03µs/65.81%/67.59% 逐项一致；k 越大 partial 总量不变、耗时恒定）。
  - `dq_reduce_kernel<512>`（grid `(1792,2,1)`×512）：**23.33 µs / DRAM 81.73% / L2 77.21%** /
    L1TEX 10.05% / SM 18.63% / occ 81.02%、读 58.73MB / 写 5.10MB ⇒ 同样是**纯 DRAM 带宽 bound**
    （与 P3-4k 的 8.74µs 定长版量级一致，varlen 的 T 更大故耗时更长）。
  - 即 DET 的增量 ~72 µs（两个 reduce）+ partial 清零/读写；**真正的墙仍是 reduce 的 DRAM 带宽**，
    与 P3-4e/f/g/h/i/j/k 逐项一致。

### 63.5 回归与对标

- `--no-run --ci`（73 case，默认路径）：全绿——单/两文件 worst fp16 `3.906e-3` / bf16 `7.812e-3` /
  fp8 `9.537e-6`（按 dtype gate 全 OK），`--check docs/04` OK（194 行，rtol=5e-3）。**默认路径数值
  逐位不变**。
- **对标**：MLA（`head_dim=512`）的 FA2/FA3/TE 反向**均不支持**（本机实测 `fa2/fa3/te failed
  MLA d512`），故本轮无 FA/TE 同 shape 对标；刷新纯反向 D=128 基线（`fa_vs_te_bwd_only.py fp16`）：
  FA3 MHA S4096 `0.3236ms/849TF`、TE `0.4403/624`、FA2 `0.7265/378`；varlen 等长 4×1024 causal
  FA3 `0.1477ms/233TF`、不齐 `0.1623/281`——与 §37/§38/§40 口径一致。

### 63.6 复现 / 原始输出

```bash
# MLA varlen + --det --detk=N（打印 [P3-4l A/B]；varlen 需 sm90a + -DFA_WGMMA）
for k in 1 4 8 16; do ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b1_t512_h2_d512_causal_fp8 \
  --varlen --det --detk=$k --iters=30; done
# full 需显式 --full；单文件同理换 fa_bwd_fp8_mma_onefile.cu
# ncu（两个 reduce）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name regex:dkv_reduce_varlen_kernel -c 1 \
  --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,... \
  -- --dir=... --varlen --det --detk=8 --iters=3
```

原始输出：`src/fp8/fa_bwd_fp8_main_p34l_det_varlen_b1_t512_h2_d512_causal_{1,4,8,16}.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34l_det_varlen_b3_t1792_{k1,k4,k8,k16}.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34l_det_varlen_b1_t512_full_k4.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_p34l_det_varlen_b1_t512_k4.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_p34l_det_varlen_b3_t1792_k4.out.txt`、
`src/fp8/fa_bwd_fp8_p34l_ncu_{dkvreduce,dqreduce}_varlen_b3.out.txt`、
`src/fa_bwd_p34l_fa_baseline_fp16.out.txt`。

**下一步候选**：① **减 partial 字节**（按 KV 行跨 warpgroup 偏和 / 更细分块）以压低 reduce 的
DRAM 墙——varlen 的 maxlen-strided partial 尤其浪费，可换 compact per-sequence offset；
② 把两个 reduce 融合进一个 kernel（省一趟 partial 读）；③ 性能（非确定性）仍受本卡寄存器/
smem 硬墙锁定，见 ROADMAP「阻塞」。

## 64. P3-4m（第一百二十六轮，**正结果，opt-in `--det`**）：把 `--det` 接进 MLA 的 K/V `cp.async` 回填流水（kvpipe）

### 64.1 动机：DET 一直走非 kvpipe 主 kernel，白扔 O51 的 1.78–1.86×

`--det` 的 dK/dV partial 与二次归约（P3-4e…P3-4l）在 device 侧早已完备，但 host 的两个
`launch_bwd_main_det`（定长 / varlen）**写死 `KVPIPE=false`**，于是 MLA（HD=512）的 DET 主
kernel 一直走「每 tile 末尾同步载入 K/V」的旧路，而默认（非 DET）MLA 主 kernel 早在 **O51**
就改用 K/V `cp.async` 回填流水（§O51/O52：8-warp 主 kernel `sync → kvpipe` 在 b1_t512/b3_t1792
上 **1.78×/1.86×**）。`fp8_mma_body` 的模板参数里 **`KVPIPE && DET` 本来就并存**（只有
`static_assert(!KVPIPE || (!WGMMA && !TMA && !KVTMA))` 限制后端），缺的只是 host 接线。

### 64.2 实现（host-only；单/两文件 device 逐字同源）

- `launch_bwd_main_det` 增加模板参数 `bool KVPIPE = false`（默认与历史逐字不变），
  `kSmem` 选择改为 `WGMMA ? smem_bytes_wgmma : (KVPIPE ? smem_bytes_kvpipe : smem_bytes)`，
  并把 `KVPIPE` 透传给 `fa_bwd_fp8_mma_kernel`。MLA（HD=512,BN=32）的
  `smem_bytes_kvpipe = 207872 + 16896 + 5120 = 229888B ≤ 232448`（1 CTA/SM，与 O51 一致）。
- P3-4k（定长）与 P3-4l（varlen）的 A/B 段各加一个 `run_dt_kv` 变体（`launch_bwd_main_det<
  512,64,32,false,true,true,true,256,4,false,true>`，即 `WGMMA=false, KVPIPE=true`），与旧
  `run_dt`（非 kvpipe）共用同一段 reduce；新增 `[P3-4m A/B]` 打印与逐位对拍。单/两文件同步。
- 顺带修 `scripts/sync_onefile_device.py` 的一个既有坑：fp8 单文件的 device 区起点 marker 是
  `#include <cuda_runtime.h>`，而 `#include "../fa_bwd_dump.h"` 原在其后 ⇒ 每次同步都会把它
  一并覆盖掉（单文件报 `fa_bwd_save_npy_f32 undefined`）。本轮把该 include **移到 marker 之前**
  （它只依赖标准库），从此同步不再误删。device 区仍 `identical: True`。

### 64.3 数值：kvpipe DET 与非 kvpipe DET **逐位相同**

同 session `--iters=30`（两文件；单文件 b3 复核）：

| case | ksplit | DET 非 kvpipe ms | DET kvpipe ms | 加速 | kvpipe runs[1-2] bitwise | kvpipe-vs-非kvpipe |
|---|---|---|---|---|---|---|
| MLA 定长 S256H2 | 4 | 0.0652 | 0.0640 | 1.018× | 0 / 0 / 0 | 0 / 0 / 0 |
| MLA 定长 S512H4 | 4 | 0.1578 | 0.1477 | 1.068× | 0 / 0 / 0 | 0 / 0 / 0 |
| MLA 定长 S1024H2 | 4 | 0.2594 | 0.2394 | 1.084× | 0 / 0 / 0 | 0 / 0 / 0 |
| MLA varlen b1_t512 | 8 | 0.0928 | 0.0887 | 1.046× | 0 / 0 / 0 | 0 / 0 / 0 |
| MLA varlen b3_t1792 | 8 | 0.3550 | **0.3258** | 1.090× | 0 / 0 / 0 | 0 / 0 / 0 |
| varlen b3_t1792（单文件） | 8 | 0.3556 | 0.3248 | 1.095× | 0 / 0 / 0 | 0 / 0 / 0 |

- **`kvpipe-vs-非kvpipe` 与 `runs[1-2]` 的 dq/dk/dv 全为 `0.00e+00`**——kvpipe 只改 K/V 搬运、
  不改任何归约次序，DET 的逐位可复现性原样保留，且与非 kvpipe DET 逐字节一致。
- 默认路 `ours vs ref` 逐位不变（S256H2 `2.356/2.290/3.441e-1`、S512H4 `2.415/2.992/4.481e-1`、
  S1024H2 `2.232/3.337/3.602e-1`；varlen b1 `1.613e-1/2.238e-1/3.864e-1`、b3 `3.404e-1/3.436e-1/
  3.508e-1`），与 P3-4k/l 历史逐位一致。
- **收益随主 kernel 占比增大而增大**（kvpipe 是把 K/V 全局读延迟藏进本轮计算）：S256H2 只有
  1.018×，S1024H2 / b3_t1792 到 1.08–1.09×（reduce 占 DET 总时 ~20%，故端到端加速略小于
  O52 的主 kernel-only 1.78–1.86×）。

### 64.4 ncu：kvpipe 把 `long_scoreboard` 3.93→3.25，主 kernel 243→220µs

同 session、同 binary、`--iters=1`、`-k regex:fa_bwd_fp8_mma_kernel`（b3_t1792 causal k=8；
DET 非 kvpipe = 模板 `<...,0,1>` / DET kvpipe = `<...,1,1>`）：

| 指标 | DET 非 kvpipe | DET kvpipe |
|---|---|---|
| Duration | 243.0 µs | **220.2 µs（1.104×）** |
| Dynamic smem | 207.87 KB | 229.89 KB |
| DRAM / L1TEX / L2 / Compute | 29.8 / 23.8 / 34.3 / 15.0 % | 33.0 / 26.8 / 38.4 / 16.5 % |
| regs / achieved occ | 255 / 12.49% | 255 / 12.49% |
| stall long_scoreboard | 3.93 | **3.25** |
| stall wait / short / barrier | 1.69 / 1.04 / 1.07 | 1.71 / 1.07 / 1.04 |

⇒ **机制与 O51 逐项一致**：kvpipe 用 `cp.async` 提前发下一 tile 的 K/V，把「同步全局读」的
`long_scoreboard` 从 3.93 压到 3.25，主 kernel 时间 −9.4%（DRAM/L1/L2 利用率同时上升，说明
更多访存被重叠进来而非减少）；`wait`/`short`/`barrier` 基本不变，occupancy 仍 12.5%（1 CTA/SM，
255 regs + 230KB smem）。墙仍是 **L2（残余 dK/dV red）+ 低 occupancy**，与 P3-4 系列结论一致。

### 64.5 回归 / 对标

- `python3 harness/fa_bwd_run.py --ci`（全量 73 case + gate + docs/04 校验）：全绿——单/两文件
  worst fp16 `1.953e-3` / bf16 `3.125e-2` / fp8 `7.629e-6`（按 dtype gate 全 OK），
  `--check docs/04` OK（194 行，rtol=5e-3）。**默认路径（无 `--det`）数值逐位不变**。
- MLA（head_dim=512）的 FA2/FA3/TE 反向均不支持，仍只有 fp32 ref；D=128 纯反向基线见
  `docs/04` §38/§44（本轮未重跑）。
- **意义**：`--det` 从「能用」推进到「用上当前最优的数据通路」——确定性模式下不再额外付出
  「非 kvpipe 主 kernel」这一层代价；`--det` 相对 atomic 的剩余差距（reduce 的两趟 DRAM partial）
  才是确定性真正的净成本。

### 64.6 复现 / 原始输出

```bash
# 定长 MLA（打印 [P3-4k]/[P3-4m]）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 --det --detk=4 --iters=30
# varlen MLA（需 sm90a + -DFA_WGMMA）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" scripts/run.sh \
  src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b3_t1792_h2_d512_causal_fp8 \
  --varlen --det --detk=8 --iters=30
# 单文件同理换 fa_bwd_fp8_mma_onefile.cu
# ncu：DET 非 kvpipe = launch-skip 8；DET kvpipe = launch-skip 13（--iters=1）
```

原始输出：`src/fp8/fa_bwd_fp8_main_p34m_det_varlen_{b1_t512,b3_t1792}_k8.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34m_det_mla_{s1024h2_k4,s256h2_s512h4_k4}.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_p34m_det_varlen_b3_t1792_k8.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34m_ncu_{detnonkv,detkv,metrics,stall}_varlen_b3.out.txt`、
`src/fp8/fa_bwd_p34m_ci.out.txt`。

**下一步候选**：① 减 partial 字节（按 KV 行跨 warpgroup 偏和 / varlen compact per-sequence
offset）以压低 reduce 的 DRAM 墙；② 把两个 reduce 融合进一个 kernel；③ 非确定性性能仍受本卡
寄存器/smem 硬墙锁定，见 ROADMAP「阻塞」。

---

## 65. P3-4n（第一百二十七轮，**正结果（小 S）/中性（大 S），默认开**）：把两个二次归约融合进一个 kernel

### 65.1 动机（落实第一百二十六轮「下一步候选 ②」）

`--det` 的二次归约是**两次 launch**：`dkv_reduce_kernel`（读 dk/dv partial，P3-4e）+
`dq_reduce_kernel`（读 dq partial，P3-4f；仅 `ksplit>1`）。两者数据不相交、互不依赖。
P3-4e/f 的 ncu 显示 `dkv_reduce` 730.7µs/DRAM 91.6%，而 `dq_reduce` 只有 55.1µs/DRAM 87.6%
（大 S）——第二次 launch 的**字节无法减少**（各读各的 partial），但 dq reduce 的**尾部**
（小 shape 下尤甚：S512 时 dq 仅 8.6µs、DRAM 58.7%，远没打满）可以藏进 dkv 的重块里，
同时省掉一次 launch。候选 ①（减字节）需要 `BM=128`，而 O19 已证伪 fp8 的「放大 BM/减 red」
（1 CTA/SM 的并行度损失 > red 收益），故先做候选 ②。

### 65.2 实现（单/两文件 device 逐字同源，host 逐字一致）

- **device**（`fa_bwd_fp8_kernels.cuh`）：新增两个融合 kernel——
  `dkv_dq_reduce_kernel<HD,BM>`（定长）与 `dkv_dq_reduce_varlen_kernel<HD,BM>`（变长）。
  grid 改成 **1D**：`dkv_blocks + dq_blocks`、block=HD；`blockIdx.x < dkv_blocks` 的块跑
  dK/dV（代码与旧 `dkv_reduce[_varlen]_kernel` 逐字相同），其余块跑 dQ（与旧
  `dq_reduce_kernel` 逐字相同，`row = qi/H, h = qi%H`）。`ksplit==1` 时 `dq_blocks=0`，
  退化成纯 dkv。**求和次序（hh 升序、m 升序 / part 升序）完全不变 ⇒ 融合版与分开版逐位相同。**
- **host**：`--nofusered`（默认 `fuse_reduce=1`）在 4 个 DET reduce 站点切换：
  ① D=128 定长（P3-4e/f）、② D=512 MLA 定长（P3-4h/k/m 的 `do_reduce`）、
  ③ D=128 varlen（P3-4i）、④ D=512 MLA varlen（P3-4l）。Hopper TMA 快路（D=128、锁
  `ksplit=1`）没有 dq reduce，无需融合。run_varlen 增 `fuse_reduce` 形参透传。
  另在 ①（定长）与 ③（varlen）各加一段 `[P3-4n A/B]`：**只测二次归约**（partial 由一次 DET
  主 kernel 预置），对比 2-launch vs 1-launch 并逐位对拍。
- 单文件由 `scripts/sync_onefile_device.py` 同步 device 区（`device region identical: True`），
  host 区与 `fa_bwd_fp8_main.cu` 逐字合并（`diff` 无差异）。**device 数学/数据流一行未改**。

### 65.3 数值：融合版与分开版 dq/dk/dv **逐位相同**

| case | ksplit | `fused-vs-sep` bitwise dq/dk/dv | `runs[1-2]` bitwise | ours vs ref（与历史逐位） |
|---|---|---|---|---|
| D128 定长 S512 | 4 | 0 / 0 / 0 | 0 / 0 / 0 | 2.426/2.975/3.735e-1 |
| D128 定长 S4096 | 4 | 0 / 0 / 0 | 0 / 0 / 0 | 2.635/2.643/3.216e-1 |
| D512 MLA 定长 S1024H2 | 4 | （经 `do_reduce`，`runs[1-2]`=0） | 0 / 0 / 0 | 2.232/3.337/3.602e-1 |
| D128 varlen b1_t512 | 8 | 0 / 0 / 0 | 0 / 0 / 0 | 2.280/3.108/3.422e-1 |
| D512 MLA varlen b3_t1792 | 8 | （经 `do_reduce`，`runs[1-2]`=0） | 0 / 0 / 0 | 3.404/3.436/3.508e-1 |

单文件（S512 定长、b3 varlen D512）与两文件逐指标一致；默认路（无 `--det`）数值逐位不变。

### 65.4 性能：reduce-only S512/varlen 小 shape 1.15–1.17×，大 S 中性

同 session、同 binary（`[P3-4n A/B]`，只测二次归约，ms）：

| case | separate（2 launch）| fused（1 launch）| 加速 |
|---|---|---|---|
| D128 定长 S512 | 0.0290 | **0.0253** | **1.149×** |
| D128 定长 S512（单文件）| 0.0292 | 0.0254 | 1.152× |
| D128 定长 S4096 | 0.7917 | 0.7867 | 1.006× |
| D128 varlen b1_t512 | 0.0362 | 0.0309 | **1.170×** |

- **机制**：ncu（S512，`--set full -c 1`）——分开版 `dkv_reduce_kernel` **17.31µs /
  DRAM 70.4% / occ 65.5% / waves 5.17**（grid 8192）+ `dq_reduce_kernel` **8.64µs /
  DRAM 58.7% / occ 74.2% / waves 3.88**（grid 8192），合计 25.95µs；融合版 **22.94µs /
  DRAM 79.7% / occ 86.1% / waves 7.76**（grid 16384）——**一个网格让 dq 的轻块与 dkv 的重块
  交错，把 dq 的低 DRAM 尾部（58.7%）填进 dkv 的发射口，整体 DRAM 从 70% 抬到 80%**。
- 大 S（S4096）reduce 已是纯 DRAM 带宽墙（91.6%），两次 launch 的字节不变 ⇒ 融合只剩省一次
  launch（~1µs），故 **1.006× 中性**。这是预期的边界：融合打的是「小/中 shape 的尾部与
  launch 开销」，不是带宽。
- 端到端（DET 含主 kernel）侧写：MLA S1024H2 DET 非 kvpipe 0.2594→**0.2549ms**、
  kvpipe 0.2394→**0.2366ms**（历史 P3-4m 对照）；b3_t1792 D512 varlen DET 0.3495ms。

### 65.5 回归

`python3 harness/fa_bwd_run.py --ci`（全量 73 case + gate + docs/04 校验）全绿——
gate[fp16] 3.906e-3 / gate[bf16] 1.562e-2 / gate[fp8] 7.629e-6（按 dtype 容差全 OK），
`--consistency` OK（worst 1.562e-2），`--check docs/04` OK（194 行，rtol=5e-3）。
**默认路径（无 `--det`）数值逐位不变。**

### 65.6 复现 / 原始输出

```bash
# 定长（打印 [P3-4f]/[P3-4n]）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8 --det --detk=4 --iters=30
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --det --detk=4 --iters=10
# MLA 定长（打印 [P3-4k]/[P3-4m]）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8 --det --detk=4 --iters=30
# varlen（需 sm90a + -DFA_WGMMA；打印 [P3-4i]/[P3-4n]）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" scripts/run.sh \
  src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b1_t512_h16_d128_causal_fp8 \
  --varlen --det --detk=8 --iters=30
# 单文件同理换 fa_bwd_fp8_mma_onefile.cu；--nofusered 退回两次 launch
# ncu：融合 = regex:dkv_dq_reduce_kernel；分开 = regex:dkv_reduce_kernel / regex:^dq_reduce_kernel
```

原始输出：`src/fp8/fa_bwd_fp8_main_p34n_d128_{s512,s4096}_detk4.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34n_mla_s1024h2_detk4.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34n_varlen_{b1_t512_d128,b3_t1792_d512}_k8.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_p34n_{d128_s512_detk4,varlen_b3_t1792_d512_k8}.out.txt`、
`src/fp8/fa_bwd_fp8_p34n_ncu_{fused,dkvsep,dqsep}_s512.out.txt`、`src/fp8/fa_bwd_p34n_ci.out.txt`。

**下一步候选**：① 减 partial 字节——需 `BM=128`（O19 已证伪 fp8 的放大 BM），或按 KV 行
跨 warpgroup 偏和 / varlen compact per-sequence offset；② 融合已收口（本轮），进一步的
「两次 reduce 融合 + partial 原地累加」需先解决字节问题；③ 非确定性性能仍受本卡寄存器/
smem 硬墙锁定，见 ROADMAP「阻塞」。

---

## 66. P3-4o（第一百二十八轮，**负结果（性能）/正结果（显存），opt-in `--partcompact`**）：varlen `--det` 的 partial 试换 compact per-sequence 布局

### 66.1 动机（落实第一百二十七轮「下一步候选 ①」）

第一百二十七轮候选 ① 是「**减 partial 字节**以压低 DET 二次归约的 DRAM 墙」，其中可选项之一是
「varlen 的 **maxlen-strided partial 尤其浪费，可换 compact per-sequence offset**」。动机的
定量背景：varlen 的 dK/dV partial 布局是 `part[((b*H+h)*nblk_max+mblk)*maxlen+jg]`，对长度
远小于 `maxlen` 的序列会留大片空洞——例如 `b8_t2904`（8 段、最长 2048）的 partial 缓冲要
**4.29GB**，而真正被写的只有约 0.58GB。compact 布局把每序列的 partial 收紧到
`H*nblk_b*len_b` 行，理论地址跨度缩小到 ~13%。本轮把这个选项落地并实测。

### 66.2 实现（单/两文件 device 逐字同源，host 逐字一致）

- **device** `fp8_mma_body`：新增尾参 `const int* part_base = nullptr`（行前缀和）。两个 DET
  dK/dV 写点（GEMM3 的 dV、GEMM4 的 dK）在 `part_base` 非空时用
  `row = part_base[b] + (h*nblk_seq + mblk)*len_b + jg`（`nblk_seq = ceil(len_b/BM)`），
  否则**逐字**走原 `((b*H+h)*nblk+mblk)*S+jg`；三个 `__global__` 壳（普通/dO-TMA/KV-TMA）透传。
- **device** `dkv_reduce_varlen_kernel` / `dkv_dq_reduce_varlen_kernel`：同样加 `part_base`
  尾参，非空时读 compact 地址（`(h0+hh)*nblk_b+m` 同式），空时逐式退化。**求和集合与次序均
  不变**（仅地址不同）⇒ 与旧布局**逐位相同**。
- **host**（`fa_bwd_fp8_main.cu` 与单文件 `fa_bwd_fp8_mma_onefile.cu`）：D=128 与 D=512 的
  varlen DET A/B 各计算 `part_base_h[b+1] = part_base_h[b] + H*nblk_b*len_b`、上传 device；
  `pb = part_compact ? d_part_base : nullptr`（**默认 nullptr = 旧布局**）；新增 `--partcompact`
  开关；同 binary 打印 `[P3-4o]`（两种缓冲大小 + 当前 layout）与 `[P3-4o A/B]`（两布局各跑
  一遍 main+reduce 的 event 计时 + 数值逐位对比）。

### 66.3 数值：compact-vs-legacy **逐位相同**（仅改地址）

varlen 各 case（ksplit=8）的 `[P3-4o A/B] compact-vs-legacy bitwise dq/dk/dv = 0.00e+00` 全部
成立；`runs[1-2]` 与 `DET-vs-atomic` 与历史一致。默认路径（`part_base=nullptr`）：
`b4_t3840` `ours vs ref` dq/dk/dv = 2.935e-1/2.938e-1/4.179e-1（与 P3-4i 记录逐位一致）；
`b3_t1792` D=512 = 3.404e-1/3.436e-1/3.508e-1（与 P3-4l 记录逐位一致）。

### 66.4 性能：**分配大幅缩小，但耗时中性偏负（0.944–0.975×）**

| case | compact 缓冲 / 旧缓冲 | 旧布局 (ms) | compact (ms) | 比 |
|---|---|---|---|---|
| D=128 b1_t512 (单段) | 33.6MB / 33.6MB (100%) | 0.1120 | 0.1151 | 0.973× |
| D=128 b4_t3840 | 713.0MB / 2147.5MB (33%) | 1.1504 | 1.1869 | 0.969× |
| D=128 b8_t2904 | 575.1MB / 4295.0MB (13%) | 1.0162 | 1.0485 | 0.969× |
| D=128 b5_t3968 H32 | 1430.3MB / 5368.7MB (27%) | 2.2434 | 2.3000 | 0.975× |
| D=512 b1_t512 | 16.8MB / 16.8MB (100%) | 0.0925 | 0.1005 | 0.920× |
| D=512 b3_t1792 | 88.1MB / 201.3MB (44%) | 0.3633 | 0.3835 | 0.947× |
| D=512 b3_t1792（单文件） | — | 0.3612 | 0.3828 | 0.944× |

**ncu（`dkv_dq_reduce_varlen_kernel<128,64>`，b4_t3840，融合版，`--launch-count 1`）**：
旧布局 **373.76µs / DRAM 86.89% / L2 81.96% / L1TEX 11.41% / SM 33.55%**；
compact **384.86µs / DRAM 84.38% / L2 81.29% / L1TEX 11.33% / SM 33.50%**。
两者都是 **reduce 的纯 DRAM/L2 带宽 bound**；compact 的 DRAM% 反而略低、耗时略长。

### 66.5 结论（为什么是负结果）

- reduce **读的字节数不变**：它本就只遍历被写过的 `(h,mblk,jg)` 条目（`m<nblk_b`、`jg<len_b`），
  所以 maxlen-strided 的「空洞」并不产生额外 DRAM 流量；compact 只缩小**地址跨度**。
- 缩小跨度没有提升 DRAM 效率（84–87% 两种布局相当），反而因为把不同序列的 partial 挤到更近的
  地址、破坏了跨序列的通道/页并行，**略慢 2.5–5.6%**。
- 因此 **compact 的性能价值为负**，唯一实打实的收益是**缓冲footprint**（最多降到 13%），
  这对被显存限制的目标卡有工程意义，故保留为 **opt-in `--partcompact`**，默认保持旧布局
  （不回归）。
- 真正的「减 partial **字节**」只能靠 **BM=128 的跨 warpgroup 偏和**（因果下 partial 条目数
  ∝ `nblk²/2`，BM 翻倍即减到 1/4），但 fp8 的放大 BM 已被 O19 证伪（smem 硬墙），见 ROADMAP
  「阻塞」。本轮的 compact 路线到此收口。

### 66.6 复现 / 原始输出

```bash
# 两文件（sm90a + -DFA_WGMMA）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" scripts/run.sh \
  src/fp8/fa_bwd_fp8_main.cu --varlen --det --detk=8 --iters=20 \
  /home/xieminglin/proj/output/fa-bwd/varlen_b4_t3840_h16_d128_causal_fp8
# 单文件；--partcompact 打开 compact（默认旧布局）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" scripts/run.sh \
  src/fp8/fa_bwd_fp8_mma_onefile.cu --varlen --det --detk=8 --iters=20 \
  /home/xieminglin/proj/output/fa-bwd/varlen_b3_t1792_h2_d512_causal_fp8
# ncu：regex:dkv_dq_reduce_varlen（加/不加 --partcompact 分别抓 compact/旧布局的首次 reduce）
```

原始输出：`src/fp8/fa_bwd_fp8_main_p34o_d128_b4t3840.out.txt`、
`src/fp8/fa_bwd_fp8_main_p34o_d512_mla.out.txt`、
`src/fp8/fa_bwd_fp8_mma_onefile_p34o_d512_b3.out.txt`、
`src/fp8/fa_bwd_fp8_p34o_ncu_reduce_b4.out.txt`、`src/fp8/fa_bwd_p34o_ci.out.txt`（CI）、
`src/fp8/fa_bwd_p34o_ci_full.out.txt`（全量 --ci）、`src/fp8/fa_bwd_p34o_ci_afterapply.out.txt`。

**下一步候选**：① **减 partial 字节**只剩「BM=128 跨 warpgroup 偏和」（fp8 撞 smem 硬墙，见
「阻塞」）或「partial 降精度存储（fp16/bf16，确定性保留但改数值口径）」；② 把 DET 的
reduce 做成 **L2 内偏和**（persistent CTA / cluster 分布式归约）以少一趟 DRAM；③ 非确定性
性能仍受本卡寄存器/smem 硬墙锁定，见 ROADMAP「阻塞」。

---

## 67. F1（第一百三十二轮，**正结果，默认化**）：fp8 主路径默认切到 Hopper（wgmma GEMM1/2 + Q/K/V/dO 4D-TMA）

### 67.1 动机与现状（fp8 专项冲刺第一步）

ROADMAP『fp8 专项冲刺：用 SASS/PTX 对标 TE』指出：TE 的 fp8 反向是
`..._flash_bprop_wgmma_f8_...`（**QGMMA + TMA + WARPGROUP**，64x64x128、384 线程、132 CTA，
S=4096 约 258µs），而 ours 的 `ours` 口径此前一直跑在 **`-arch=sm_90` 的 mma.sync** 上
（SASS 全 `HMMA`/`LDSM`，无 `QGMMA`/`TMA`）。但仓库里其实**早就有** Hopper 实现：
O9c-2 的主 kernel GEMM1/2 `wgmma`、O32/O9c 的 LSE `wgmma`/TMA、O37 的 Q/dO 4D-TMA、
O41 的 K/V 4D-TMA——只是它们**只在显式传 `--hopper` / 用 `-DFA_WGMMA -DFA_TMA` 构建时才生效**，
默认 harness 仍走 mma。F1 = 把这条已存在的快路**默认化**。

### 67.2 实现（host / harness-only，device 一行未改）

- **`scripts/run.sh` 默认不变**（仍 `-arch=sm_90`，供 smoke / 单 kernel 用），只改**标准入口**
  `harness/fa_bwd_run.py`：新增 `FP8_HOPPER_DEFAULT = True`，**fp8 定长 case 的构建环境切到
  `HOPPER_FLAGS`**（`-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda`）；
  varlen 与 fp16/bf16 保持原口径（`docs/04` 表稳定）。新增 `--mma` 让 fp8 退回旧 mma 构建做 A/B。
  `-lcuda` 是 TMA 描述符 `cuTensorMapEncodeTiled` 的链接项；`ARCH=""` 让 gencode 生效
  （CUDA 13 的 nvcc 不认 `-arch=sm_90a`）。
- **device 无改动**：`-DFA_WGMMA -DFA_TMA` 下 host 自动打印
  `O32: lse backend = tma`、`O37: main qd-tma = on`、`O41: main kv-tma = on`，主 kernel 选到
  `fa_bwd_fp8_mma_kvtma_kernel`（Q/dO/K/V 全 4D-TMA + GEMM1/2 wgmma）。

### 67.3 SASS 证据（`cuobjdump -sass`，与 TE 的 QGMMA+TMA 对标）

| kernel（活跃实例） | QGMMA | HMMA | LDSM | UTMA/UBLKCP |
|---|---|---|---|---|
| 旧默认 `fa_bwd_fp8_mma_kernel`（`sm_90`，REGDQ） | **0** | **160** | **78** | 0 |
| 新默认 `fa_bwd_fp8_mma_kvtma_kernel`（`sm90a`，REGDQ） | **8** | **96** | **46** | **7** |

即默认化后：**GEMM1/2（S=QKᵀ、dP=dO·Vᵀ）切到 `wgmma`（8 条 QGMMA），Q/K/V/dO 由 7 条 TMA
搬运**；**GEMM3/4/5 仍是 mma.sync（96 HMMA + 46 LDSM）**——这正是 backlog 的下一堵墙
（O9c-2b 已判 fp8 `wgmma` 无转置操作数、MN-major 描述符无效，见 ROADMAP「阻塞」）。
原始输出 `src/fp8/fa_bwd_fp8_p132_sass_counts.out.txt`。

### 67.4 数值：与历史在 fp8 容差内一致（默认路径换后端，不改数学口径）

fp8 定长 12 case（含 GQA/MQA/MLA）**ours vs ref** 与历史逐位同级：S512 `2.426/2.972/3.733e-1`、
S1024H32 `2.399/4.177/3.535e-1`、S4096 `2.635/2.644/3.216e-1`、GQA kv4 `2.517/5.339/7.173e-1`、
MLA S1024H2 `2.232/3.337/3.602e-1`；单/两文件一致性 gate `worst=1.049e-5`（容差 1e-4）OK。
`docs/04` auto 表已按新默认刷新（fp8 各行仅末位噪声级变化）。

### 67.5 性能（同 session，CUDA event，两文件 fp8）

| shape | mma 默认 total / main (ms) | **Hopper 默认 total / main (ms)** | 端到端比 | TE FP8 total (ms) | ours/TE |
|---|---|---|---|---|---|
| S512 H16 | 0.1139 / 0.0636 | **0.1004 / 0.0552** | 1.135× | 0.0355 | 2.83× |
| S1024 H32 | — / — | **0.3879 / 0.2641** | — | 0.0741 | 5.23× |
| S4096 H16 | 2.3990 / 1.8849 | **1.9459 / 1.6084** | 1.233× | 0.3027 | 6.43× |

默认化**端到端 1.14–1.23×**（S4096 主 kernel 1.17×、LSE 0.38→0.22ms）；相对同 session TE FP8
仍是 2.8–6.4×（sprint 起点，F2–F5 继续收）。

### 67.6 ncu：默认换到 Hopper 后**墙没有变**（验证 backlog 判断）

`fa_bwd_fp8_mma_kvtma_kernel`（S=4096，ksplit=8，grid 8192，128 线程）：
Duration **1.59ms**、**L2 77.9% / L1TEX 70.4% / Compute 47.1% / DRAM 4.3%**、
168 regs / 74.82KB smem → **3 CTA/SM（occ 18.4%）**、Waves 20.7；
stall **`wait 1.59` + `short_scoreboard 1.29` + `long 0.59` + `not_selected 0.40` + `barrier 0.42`**；
L2 `red` = **114.5M 扇区**（dK/dV 跨 CTA `atomicAdd`）。这与 ROADMAP「阻塞」里
「fp8 主 kernel 3 CTA/SM 下唯一能打的杠杆是跨-tile `P/dS` 双缓冲，但 smem 放不下」的核算
**逐项吻合**——默认 wgmma+TMA 只是把搬运/GEMM1-2 打快，**GEMM3/4/5 的 mma 依赖（wait+short）
与 L2 `red` 仍是墙**，是 F3/F4 的靶子。
原始输出 `src/fp8/fa_bwd_fp8_p132_ncu_main_s4096.out.txt`、`..._p132_stall_s4096.out.txt`。

### 67.7 复现 / 原始输出

```bash
# 新默认（fp8 定长走 Hopper）：一键跑 ours + 汇总 + 一致性 + docs/04 校验
python3 harness/fa_bwd_run.py --dtype fp8 --fixed-only
python3 harness/fa_bwd_run.py --no-run --ci
# A/B：--mma 退回旧 mma.sync 构建
python3 harness/fa_bwd_run.py --dtype fp8 --fixed-only --mma
# 直接跑（sm90a + wgmma + TMA）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
```

原始输出：`src/fp8/fa_bwd_fp8_p132_main_{s512,s1024_h32,s4096}*.out.txt`（Hopper）、
`src/fp8/fa_bwd_fp8_p132_mma_{s512,s4096}*.out.txt`（A/B mma）、
`src/fp8/fa_bwd_fp8_p132_sass_counts.out.txt`、`..._p132_ncu_main_s4096.out.txt`、
`..._p132_stall_s4096.out.txt`、`..._p132_te_baseline.out.txt`。

**下一步候选**：① 仍无法把 GEMM3/4/5 上 wgmma（无转置操作数，见「阻塞」）⇒ 转 **F3
（warp specialization / 更深 mbarrier 流水）**与 **F4（dK/dV 归约去 L2 `red`）**；
② 把 fp8 的 partial 降精度 + 扇区化（第 131 轮候选 ①，`fp8_mma_body` 两个写点）接上，
压 L2 `red`；③ preprocess（F5）S4096 已仅 0.22ms、非当前墙。

---

## 68. F4（第一百三十三轮，**正结果（DET 路径），opt-in `--det`**）：fp8 DET 的 dK/dV partial「降精度 + 写扇区化」

### 68.1 动机与现状（落实第 131 轮候选 ① / §67.7 下一步候选 ②）

`docs/03` §56 已定量：fp8 的确定性反向（`--det`）把跨 CTA `atomicAdd`（114.5M L2 `red` 扇区）
换成「按 `(Q 头, Q 块)` 分片的 partial 覆盖写 + 固定次序二次归约」。DET 的**二次归约是纯 DRAM
带宽 bound**（`dkv_reduce_kernel` 731.6µs、DRAM 91.5%），partial 写 2.20GB / 读 4.3GB（dk+dv）——
因为 partial 是 **fp32**。fp16/bf16 早在 **O60**（partial 降精度到 fp16/bf16）+ **O62**（写扇区化）
就把这条墙打掉（reduce 1.24–1.66×、DET 主 kernel 写侧 0.84→1.10–1.17×、`--det` 首次快过
非确定 atomic）；**fp8 一直没做**（`fp8_mma_body` 的两个 DET 写点仍是 fp32）。

**fp8 与 fp16/bf16 的关键差别**：fp8 的 mma `m16n8k32` 片段里，同一 quad 的 4 个 lane 的列
`c2=(lane&3)*2` 写 `float2`（8B）恰为连续 32B ⇒ **fp32 partial 已落满 L2 扇区**。若只把存储
改 fp16（每 lane 4B `__half2`），quad 只覆盖 16B、**落不满 32B 扇区** ⇒ store 扇区数不减、
主 kernel 写侧变碎（O60 在 fp16/bf16 上量到的正是此现象）。故 **降精度必须叠加 O62 的写扇区化**
才成立。

### 68.2 实现（单/两文件 device 逐字同源，`sync_onefile_device.py` 核对 `identical: True`）

- **device（`fa_bwd_fp8_kernels.cuh`）**：
  新增 `dkv_det_store_h4(float* base, size_t off, a0,a1,b0,b1)`（把相邻两列组 `j`、`j+1` 的两个
  `half2` 拼成一次 `uint2` 8B 写）与 `dkv_p16_perm(int c)`（16 列块内置换，公式与 fp16/bf16
  **逐字相同**：`c = j*8 + 2L + h` → `((j>>1)<<4)+(L<<2)+((j&1)<<1)+h`；`c0=wc*GN34` 是 16 的
  倍数，故全局列索引可直接套用）。`fp8_mma_body` 加模板参数 `bool DET_HALF=false`，`epi_dv`/
  `epi_dk` 在 `DET && DET_HALF` 时按 `(j>>1)*16+(lane&3)*4` 的置换列一次写 8B（每个 lane 的
  `uint2` 恰含 group `j`、`j+1` 两段，求和集合/次序不变）。`fa_bwd_fp8_mma_kernel`、`_qdtma`、
  `_kvtma` 三个壳加 `DET_HALF` 透传。
  `dkv_reduce_kernel<HD,BM>`、`dkv_dq_reduce_kernel<HD,BM>` 加 `bool P16=false`：`P16` 时按
  `dkv_p16_perm(c)` 读回 fp16 partial（`__half2float` 进 fp32 累加器，求和次序逐字不变）。
- **host（`fa_bwd_fp8_main.cu`，两文件；单文件同步 host 段）**：`launch_bwd_main_kvtma_det` 加
  `DET_HALF` 模板；P3-4g 的 `--det` A/B 段加 `run_dth`（额外分配 fp16 partial 缓冲、走
  `DET_HALF=true` 主 kernel + `dkv_reduce_kernel<128,64,true>`），打印 `[F4 A/B]` 三列时间 +
  `runs[1-2]` 逐位复现 + `fp16-vs-fp32`/`fp16-vs-atomic` 差。**默认路径一行未改**（`DET_HALF`
  默认 false；`--ci` gate 全绿、`--check docs/04` 194 行 OK）。

### 68.3 数值：确定性不变、只差 fp16 舍入（`runs[1-2]` 两次跑）

| case（fp8，`--det`，ksplit=1） | atomic ms | DET-fp32 ms | **DET-fp16(扇区化) ms** | `runs[1-2]` dk/dv | fp16-vs-fp32 dk/dv | ours-vs-ref（不变） |
|---|---|---|---|---|---|---|
| S=512 MHA causal（两文件） | 0.1205 | 0.1367 (0.882×) | **0.1290 (0.934×)** | **0 / 0** | 1.01e-3 / 1.81e-3 | 2.426/2.972/3.733e-1 |
| S=4096 MHA causal（两文件） | 2.0288 | 2.5974 (0.781×) | **2.2142 (0.916×)** | **0 / 0** | 9.66e-4 / 1.76e-3 | 2.635/2.644/3.216e-1 |
| S=1024 GQA kv4 causal（两文件） | 0.3320 | 0.4142 (0.802×) | **0.3679 (0.903×)** | **0 / 0** | 1.91e-3 / 2.36e-3 | 2.517/5.339/7.173e-1 |

- **`runs[1-2] bitwise dk/dv = 0`**（三种 shape）：确定性不变。
- **fp16-vs-fp32 ≈ 1e-3–2.4e-3**：partial 过一趟 fp16 舍入（dK/dV 梯度 amax ~3–6，相对 ~3–4e-4，
  与 O60 fp16/bf16 同量级），非结构误差；`ours vs fp32 ref` 与历史**逐位不变**。
- 单文件（S=512）与两文件逐指标一致（`DET-fp16 0.937×`、`fp16-vs-fp32 1.01e-3/1.81e-3`、runs=0）。

### 68.4 性能（同 session，CUDA event，`--det` A/B）

| case | atomic ms | DET-fp32（0.78–0.88×） | **DET-fp16** | fp16 vs fp32 | fp16 vs atomic |
|---|---|---|---|---|---|
| S=512 | 0.1205 | 0.1367 | **0.1290** | 1.06× | 0.934× |
| S=4096 | 2.0288 | 2.5974 | **2.2142** | **1.17×** | 0.916× |
| GQA kv4 | 0.3320 | 0.4142 | **0.3679** | 1.13× | 0.903× |

即 fp8 DET 相对 fp32 DET 端到端 **1.06–1.17×**（与 O60/O62 在 fp16/bf16 上的方向一致）。
但 **ksplit=1 下仍略慢于非确定 atomic**（0.90–0.93×）——剩余差距是「partial 多一趟写 + 一趟读」
的固有字节（atomic 在 L2 里 RMW，无额外 DRAM）；fp8 的 partial 即使 fp16 也有 1.1GB 写 + 1.1GB 读。
这与 O62 时 fp16/bf16「sector 减半后 DET 反超 atomic」不同：fp8 原 fp32 partial 已满扇区，降精度只
减字节、不再减扇区数（见 §68.5 的 store 扇区：68.2M→34.1M 是「字节减半 + 扇区也减半」，因为
fp16 后 8B/lane 恰好仍覆盖 32B；但 reduce 读扇区 102.2M→51.1M 同理）。**结论：fp8 DET 的确定性
售价从 +21%（fp32）降到 +9%（fp16）；要让 fp8 `--det` 端到端转正需再减 partial 字节
（fp8 partial？）或提 reduce 效率。**

### 68.5 ncu：写扇区/读扇区精确减半、reduce 1.75×

**DET 主 kernel**（`fa_bwd_fp8_mma_kvtma_kernel<128,64,32,1,1,1,1,1,{0,1}>`，S=4096，ksplit=1）：

| 指标 | DET-fp32 | **DET-fp16(扇区化)** |
|---|---|---|
| Duration | 1.93 ms | **1.78 ms** |
| DRAM 写 / 读 | 2.21 GB / 94.6 MB | **1.11 GB / 80.7 MB** |
| L1 store 扇区 | 68,157,440 | **34,078,720（−50%）** |
| L2 store 扇区 | 103,890,606 | **53,014,348（−49%）** |

即 **O62 扇区化在 fp8 上同样成立**：每 lane 8B、quad 32B 连续 ⇒ store 扇区精确减半（不是只减字节）。

**`dkv_reduce_kernel<128,64,{0,1}>`**（S=4096）：

| 指标 | P16=false | **P16=true** |
|---|---|---|
| Duration | 734.7 µs | **420.9 µs（1.746×）** |
| DRAM 读 | 2.18 GB | **1.09 GB** |
| L2 读扇区 | 102,243,661 | **51,126,082（−50%）** |
| DRAM 吞吐占比 | 91.1% | **81.5%** |

置换读零代价（扇区仍满），仍纯 DRAM 带宽 bound。

### 68.6 复现 / 原始输出

```bash
# A/B（atomic vs DET-fp32 vs DET-fp16）——走 F1 默认的 Hopper 快路（TMA+wgmma）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --det --iters=10 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu：DET-fp32 vs DET-fp16 主 kernel 的 store 扇区
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --kernel-name-base mangled \
  --kernel-name "regex:kvtma_kernelILi128ELi64ELi32ELb1ELb1ELb1ELb1ELb1ELb1E" --launch-count 1 \
  --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_op_write.sum \
  -- --det --iters=1 --dir=.../b1_s4096_h16_d128_causal_fp8
```

原始输出：`src/fp8/fa_bwd_fp8_p133_h_det_{s512,s4096,gqa_kv4}.out.txt`、
`..._p133_onefile_det_s512.out.txt`、`..._p133_ncu_main_{p32,p16}_s4096.out.txt`、
`..._p133_ncu_reduce_s4096.out.txt`、`src/fa_bwd_p133_ci.out.txt`。

**下一步候选**：① F4 的**默认路径**（非确定 atomic 的 L2 `red`，114.5M 扇区）仍被 O42 判为
「不可用 smem+TMA 替换、只剩放大 BM/提 occupancy 两条硬约束路」锁定，见 ROADMAP「阻塞」；
② fp8 DET 要把端到端转正需再减 partial 字节（fp8 partial / 只存 causal 非零块）或提 reduce
效率；③ 本改动可同样扩到 varlen/MLA 的 DET 路径（当前只在定长 D=128 kvtma 的 A/B 接线）。

---

## 69. F4-b（第一百三十四轮，**正结果（DET 路径），opt-in `--det`**）：把「fp16 partial + 写扇区化」扩到 fp8 的 MLA 与 varlen DET

### 69.1 动机（落实 §68.6 下一步候选 ③）

第 133 轮（`docs/03` §68）把 fp16/bf16 O60+O62 的「dK/dV partial 降精度 + 写扇区化」搬到了
fp8，但**只在定长 D=128 的 Hopper `kvtma` 快路**接线（`launch_bwd_main_kvtma_det` 的
`DET_HALF`）。fp8 的 DET 还有三条路径没吃到：**定长 MLA（HD=512）**、**varlen D=128**、
**varlen MLA（HD=512）**。本轮把这三条补齐，使 `--det` 的扇区化覆盖 fp8 的全部 DET 路径。

### 69.2 实现（单/两文件 device 逐字同源，`sync_onefile_device.py` 核对 `identical: True`）

- **device（`fa_bwd_fp8_kernels.cuh`）**：两个 varlen 归约内核加 `bool P16=false` 模板参数——
  `dkv_reduce_varlen_kernel<HD,BM,P16>` 与 `dkv_dq_reduce_varlen_kernel<HD,BM,P16>`：`P16` 时把
  原来的行地址 `row + c` 拆成 `row + dkv_p16_perm(c)`、按 `__half` 读回（`__half2float` 进 fp32
  累加器）。**求和集合与次序逐字不变 ⇒ 仍确定性，数值只差 fp16 舍入**。`dkv_p16_perm` 在
  §68 已定义、与 fp16/bf16 公式逐字相同。定长两内核（`dkv_reduce_kernel`/`dkv_dq_reduce_kernel`）
  §68 已支持 `P16`，本轮直接复用。
- **为什么 HD=512 也成立**：`dkv_p16_perm` 只在 **16 列块内**置换，而 GEMM3/4 的 warp n-tile
  起点 `c0=wc*GN34`：HD=128/8-warp 时 `GN34=64`、HD=512/4-N-tile 时 `GN34=32`（`NTW=128`、
  `NWAR=4`），**都是 16 的倍数** ⇒ 全局列索引可直接套用同一置换，与 D=128 同一布局（含 `nd`
  N-tile 循环的 `d0` 也是 128 的倍数）。`fp8_mma_body` 的 `DET_HALF` 分支本就同时支持
  `part_base`（compact per-sequence）与非 compact 布局，无需改动。
- **host（`fa_bwd_fp8_main.cu` 与单文件 `fa_bwd_fp8_mma_onefile.cu`）**：给三处 DET A/B 各加一个
  `run_dth` 变体（分配一份 fp16 partial 缓冲，主 kernel 走 `DET_HALF=true`、二次归约走 `P16`），
  打印新的 `[F4 A/B]` 行（DET-fp32 vs DET-fp16 时间 + `runs[1-2]` 逐位 + `fp16-vs-fp32`）：
  ① varlen D=128（`launch_bwd_main_det<128,...,kWgmVarlen,false,true>`）；
  ② varlen MLA（`launch_bwd_main_det<512,64,32,...,256,4,false,false,true>`）；
  ③ 定长 MLA（同 ②，非 varlen）。
  顺带修一个**单文件的历史遗漏**：`fa_bwd_fp8_mma_onefile.cu` 的 `launch_bwd_main_det` 此前缺
  `DET_HALF` 模板参数（§68 只补了 `kvtma_det`），本轮补齐，使单文件的默认-mma DET 路径与两文件
  完全一致。**默认路径一行未改**（`DET_HALF` 默认 false、`P16` 默认 false）。

### 69.3 数值：确定性不变、只差 fp16 舍入（`runs[1-2]` 两次跑）

| case（fp8，`--det`，ksplit=1） | DET-fp32 ms | **DET-fp16(扇区化) ms** | `runs[1-2]` dk/dv | fp16-vs-fp32 dk/dv | ours-vs-ref（不变） |
|---|---|---|---|---|---|
| 定长 MLA S=1024 H2 D512（两文件） | 0.7841 | **0.7210 (1.088×)** | **0 / 0** | 9.07e-4 / 1.79e-3 | 2.232/3.337/3.602e-1 |
| varlen D=128 b1_t512_h16（两文件） | 0.1333 | **0.1262 (1.057×)** | **0 / 0** | 1.01e-3 / 1.83e-3 | 2.280/3.108/3.422e-1 |
| MLA varlen b1_t512_h2 D512（两文件） | 0.3709 | **0.3351 (1.107×)** | **0 / 0** | 9.73e-4 / 1.67e-3 | 1.613/2.238/3.864e-1 |

- **`runs[1-2] bitwise dk/dv = 0`**（全部）：确定性保留。
- **fp16-vs-fp32 ≈ 1e-3–2.4e-3**：partial 过一趟 fp16 舍入（dK/dV 梯度 amax ~3–6，相对 ~3e-4），
  与 §68 的定长 D=128 同量级；`ours vs fp32 ref` 与历史**逐位不变**。
- **单文件与两文件逐指标一致**（三处 `fp16-vs-fp32` 完全相同：MLA `9.07e-4/1.79e-3`、
  varlen D128 `1.01e-3/1.83e-3`、MLA varlen `9.73e-4/1.67e-3`）。
- ksplit=4（含 causal/full、D128/MLA）同样 `runs[1-2]=0`、1.07–1.20×：大 case
  `varlen_b5_t3968_h32_d128` 达 **1.202×**（1.943→1.617ms）。

### 69.4 ncu：store 扇区在 varlen/MLA 上同样精确减半（O62 普适）

**定长 MLA**（`fa_bwd_fp8_mma_kernel<512,64,32,0,0,1,1,1,256,4,0,1,{0,1}>`，S=1024 H2）：

| 指标 | DET-fp32 | **DET-fp16(扇区化)** |
|---|---|---|
| L1 global store 扇区 | 2,228,224 | **1,114,112（−50%）** |
| L2 write 扇区 | 3,466,388 | **1,761,733（−49%）** |

**varlen D=128**（`fa_bwd_fp8_mma_kernel<128,64,32,1,1,1,1,1,128,2,0,1,{0,1}>`，b1_t512_h16）：

| 指标 | DET-fp32 | **DET-fp16(扇区化)** |
|---|---|---|
| L1 global store 扇区 | 1,179,648 | **589,824（−50%）** |
| L2 write 扇区 | 1,802,282 | **916,170（−49%）** |

即 **O62 的「相邻两列组拼 8B + 16 列块内置换」在 HD=512 与 varlen 上逐字节成立**：每 lane
写 8B、quad 覆盖连续 32B ⇒ store 扇区精确减半（不是只减字节）。

### 69.5 复现 / 原始输出

```bash
# 定长 MLA（默认 mma 构建）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --det --iters=8 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h2_d512_causal_fp8
# varlen D=128 / MLA varlen（Hopper 构建：sm90a + -DFA_WGMMA）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen --det --iters=8 \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b1_t512_h16_d128_causal_fp8
# ncu：DET-fp32 vs DET-fp16 主 kernel 的 store 扇区（按模板尾部分辨两个实例）
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_op_write.sum \
  --kernel-name regex:mma_kernel --csv -- target.out --det --iters=1 --dir=...
```

原始输出：`src/fp8/fa_bwd_fp8_p134_f4b_{mla_fixed,varlen_d128,mla_varlen,onefile_mla,
onefile_varlen}.out.txt`、`..._p134_ncu_{mla,varlen}_det_sectors.out.txt`；
全量回归 `--ci` 全绿（fp16 3.906e-3 / bf16 1.562e-2 / fp8 7.629e-6，`--check docs/04` 194 行 OK）。

**下一步候选**：① **F3（warp specialization / 更深 mbarrier 流水，对标 TE 384 线程/1 CTA/SM）**
——默认路径 `wait 1.59 + short_scoreboard 1.29` 的最大来源是 GEMM3/4/5 的 mma 依赖，WS 只能
*重叠*不能*减少*，先按 O48 的判据（被拆的工作是否「发射即返回」）评估；② F4 默认路径的
L2 `red`（114.5M 扇区）仍被 O42 双硬约束锁定，见「阻塞」；③ fp8 DET 要端到端转正还需再减
partial 字节（fp8 partial / 只存 causal 非零块）或提 reduce 效率。

---

## 70. F3-a（第一百三十五轮，**正结果（微优化）**）：fp8 主 kernel 去 local 化 + F3 可行性评估（对标 TE）

### 70.1 动机（F3 的「先评估再动手」）

『fp8 专项冲刺』F3 = **warp specialization / 更深 mbarrier 流水**，目标是默认路径的头号
stall `wait 1.59 + short_scoreboard 1.29`、以及 `L2 77.9%`。按 O48 的判据（被拆的工作是否
「发射即返回」），先做 SASS/SOL 级对标，再决定 WS 值不值得做。

### 70.2 对标（ncu，S=4096 causal，两实现同 shape）

| 指标 | **TE**（`cudnn...flash_bprop_wgmma_f8_...`） | **ours**（默认 `fa_bwd_fp8_mma_kvtma_kernel`） |
|---|---|---|
| 线程 / CTA 几何 | 384 线程（3 warpgroup）、132 CTA、**1 CTA/SM** | 128 线程、grid 8192（ksplit=8）、**3 CTA/SM** |
| smem / 寄存器 | **232.45 KB** / 168 regs | 74.82 KB / 168 regs |
| Duration | **258 µs** | 1610 µs |
| L2 / L1 / DRAM / Compute | 70.95% / 58.84% / 18.27% / 45.0% | 77.94% / 70.43% / 4.27% / 46.9% |
| Executed Instructions | 108.3 M | 719.2 M |
| L2 命中率 | 88.73% | — |

**两者都是 L2 bound，但 ours 搬的 L2 量约为 TE 的 ~6.9×**（`Duration × L2%`：1.61×77.9% vs
0.258×70.95%）。这解释了 6× 的时长差：TE 的 232KB smem + 1 CTA/SM 让它**每份数据只搬一次、
无 split-K 冗余**；ours 受 3 CTA/SM 的 74.8KB 限制，靠 ksplit=8 凑并行度 ⇒ Q/dO 被重读、
dK/dV 跨 CTA `red`（O42 定量：114.5M 扇区）。**结论：WS 只能*重叠***不能*减少*这笔 L2 流量
（O48 判据不满足），**F3 不是当前最优点**；真杠杆仍是 F4（去/减 L2 `red`）或改工作划分
（FA2 式 dK/dV-over-KV，roadmap backlog）。**更深 mbarrier 流水**同样被 3 CTA/SM 的
74.8KB↔77.5KB 硬间隙锁死（放不下额外 stage / 跨-tile `P/dS` 双缓冲，见「阻塞」）。

### 70.3 正结果：主 kernel 去 local 化（`int kuse[2]` → 两标量）

对标时 `ncu` 报默认 kernel **local memory 占 L1TEX 扇区 18.29%**（占 L2 扇区 7.61%）。
定位到 `fp8_mma_body` 里 K/V TMA 的 mbarrier 相位计数器 `int kuse[2] = {0,0}`——它按**运行期**
`kuse[stg ^ 1]` 动态下标 ⇒ ptxas 把它放进 **local memory**（栈帧 8B + 每 tile `LDL/STL`；
`-Xptxas -v` 原为 8B stack，与 kernel-opt 的「mbarrier 相位别用动态下标数组」坑一致）。
改成两个标量 `kuse0/kuse1` + 运行期三元选择（`sk2=stg^1; kc=sk2?kuse1:kuse0`），**逐位语义不变**。

- `-Xptxas -v`：默认实例 `<...,RDG=1,...>` 的 8B stack 消失；KVTMA 路径不再有动态下标数组。
- **ncu（<128,64,32,1,1,1,1,0,0>，S=4096）**：local `op_ld` 扇区 **7.33M→5.84M**、
  `op_st` **8.48M→5.67M**（−20%/−33%）；Duration 1.59→**1.58ms**。
  剩余的 local 流量是默认 `REGDQ` 实例的**寄存器 spill**（168 regs 下 48B stack、60B
  spill loads/stores）——`floor(65536/(3×128))=170`、但寄存器分配粒度 8 ⇒ 实际封顶 168，
  **属硬墙**（与「阻塞」里的寄存器约束一致）。
- **性能（同 binary、同 session A/B，main-only，event；6 次交替）**：
  base 均值 **1.588ms** → new 均值 **1.557ms（1.9–2.0%↑）**；端到端 S4096 total
  1.9285→**1.8969ms（1.6%↑，72.46 TF）**。逐 shape（main）：S512 0.0560→**0.0547**、
  GQA kv4 0.2539→**0.2494（1.8%）**、MLA d512 0.1201→**0.1194**、S1024H32 中性。
- **数值**：与历史**逐位相同**（S512 2.426/2.972/3.733e-1；S4096 2.635/2.644/3.216e-1；
  GQA kv4 2.517/5.339/7.173e-1；MLA 2.232/3.337/3.602e-1）；单/两文件一致性 gate
  `worst=6.676e-6`（tol 1e-4）OK、`--check docs/04` rc=0。

### 70.4 复现 / 原始输出

```bash
# A/B：base（int kuse[2]）vs new（标量），同一 session 交替
nvcc -O3 -gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda \
  src/fp8/fa_bwd_fp8_main.cu -o ab_new.out
./ab_new.out --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu local 扇区（默认 kvtma 实例）
ncu --kernel-name regex:fa_bwd_fp8_mma_kvtma --launch-skip 1 --launch-count 1 \
  --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum -- ./ab_new.out --dir=...
# TE 对标
ncu --set full --launch-skip 3 --launch-count 1 \
  --kernel-name regex:flash_bprop_wgmma_f8 --print-summary per-kernel \
  python3 harness/te_fp8_ncu.py '1 4096 16 128 causal'
```

原始输出：`src/fp8/fa_bwd_fp8_p135_ab.out.txt`（base/new 计时 + 对拍）、
`..._p135_ncu_local.out.txt`（local 扇区 A/B）、`..._f3_te_ncu_s4096.out.txt`（TE 全 set）、
`..._p132_ncu_main_s4096.out.txt`（ours 对照）。

**下一步候选**：① **F4（减 L2 `red`）才是真杠杆**——本轮对标证明 ours 的 L2 搬运量是 TE 的
~6.9×，WS/流水都改不了这个量；可走 FA2 式「dK/dV 按 KV 并行」的分块（backlog，工程量大）
或继续 DET 降字节；② F3 的 WS 在 3 CTA/SM 的资源约束下不成立（评估已收口）；
③ 默认 `REGDQ` 实例的 60B spill（168-reg 硬墙）是剩余 local 流量，除非改工作点（BM/occupancy）。

---

## 71. F5（第一百三十六轮，**正结果，默认化**）：fp8 LSE 的「tile 内两趟 softmax」+ mbarrier 相位去 local

### 71.1 动机与现状（fp8 专项冲刺第五步 = preprocess 提速）

『fp8 专项冲刺』F5 = preprocess 继续提速。F1–F4 把 main 默认切到 Hopper（wgmma+TMA）、
把 DET partial 降精度+扇区化后，S=4096 端到端的时间构成为
**main 1.55ms（81%）/ preprocess 0.22ms（12%）/ quant 0.069ms（3.6%）/ convert 0.05ms**。
main 的杠杆已被 F3-a 判为「3 CTA/SM 的 L2 流量锁死」（见「阻塞」），故 F5 转向 **preprocess**。

先用 ncu 定位 preprocess 的构成：D=128 定长 causal 的 LSE 是 **`lse_mma_kernel_bal_tma`**（O32/O38），
S=4096 时 **Duration 201µs / Executed Instructions 141.0M / Issue Slots Busy 73.6% / Ipc 3.09 /
occupancy 42%**——**纯指令发射（issue）受限**，DRAM 仅 2.7%、Compute-pipe 也非 mma（是 softmax 的
`fexp`/比较/归约）。对 SASS 做 opcode 直方图（`cuobjdump -sass`）看每 tile 的 epilogue：

| opcode | 每 tile 计数 | 说明 |
|---|---|---|
| `MUFU.EX2` | **64** | 每元素 2 个 `fexp`（= 2×FMUL+2×MUFU），32 元素/tile |
| `FMUL` | 64 | 其中一半是 `fexp` 的 `×log2e` |
| `FSETP.*` | **180** | 掩码 + `sv != -INF` + online max 三类比较 |
| `BSSY`/`BSYNC` | **107 / 107** | `if (sv != -INF) {…}` 逐元素分支 → 分支同步指令 |
| `LDS` | 316 | smem 读 |

即旧 epilogue 是「**逐元素 online-softmax**」：每个 (m,j) 元素都做
`mn=max(m,sv); l=l*exp(m-mn)+exp(sv-mn); m=mn`。因为 `mn` 必是 `m`、`sv` 之一，
**两次 `exp` 里恒有一次是 `exp(0)=1`（纯浪费）**；且 `if (sv != -INF)` 逐元素分支在 SASS 里
展开成大量 `BSSY/BSYNC`。这正是可动的地方。

### 71.2 实现（单/两文件 device 逐字同源，`sync_onefile_device.py` 核对 `identical: True`）

把 4 个 LSE kernel 的 epilogue 从「逐元素 online」改成 **tile 内两趟 softmax**：

1. **第一趟**：对本 lane 负责的 16 个列（每个 `s` 行槽 8 个 `j` × 2 个 `(q&1)`）先算掩码后的
   `sv`（掩码位 `-inf`）并顺手写回累加器数组，同时用 `fmaxf` 求本 lane 的**列 max `mloc`**；
2. `mn = max(mrow, mloc)`；**第二趟**：`add = Σ exp(sv - mn)`（掩码位 `exp(-inf)=0` 自动为 0），
   再 `l = l*exp(mrow-mn) + add`、`m = mn`。

与旧实现**数学等价**（max 是精确的、sum 只换了结合次序），但：
- `fexp` 从每 tile **64 → 34**（16 列 ×2 个 s + 2 个 rescale），砍掉那次恒为 `exp(0)` 的浪费；
- 删掉逐元素的 `if (sv != -INF)` 分支（改成第一趟的三元选择 + 第二趟的 `fexp(-inf)=0`）；
- 全掩码（`mn` 仍为 `-inf`）时用 `if (mn != -INF)` 跳过本 tile（每 tile 仅 2 次，而非每元素）；
- `qi < len`（每 `s` 一次）从内层提出；`jg < len && jg<=qi` 合并进各列的三元判断。

覆盖的 4 个 kernel：`lse_mma_kernel_bal_tma`（D=128 定长 causal，默认）、
`lse_mma_kernel_bal_wgmma`（varlen D=128 causal、`--lsetma=0` 回退）、
`lse_mma_kernel_bal`（D=512 MLA、`sm_90` 回退；支持 `FULL`/`MTN`）。

**顺带修一个既有坑（同 F3-a 在 main kernel 修的）**：`lse_mma_kernel_bal_tma` 里 mbarrier 相位
`int kuse[2]` 按运行期 `kuse[st]`（`st=rnt&1`）动态下标 → ptxas 落到 **local memory**
（ncu：local `op_ld/op_st` = 532,480 / 598,016 扇区）。改两标量 `kuse0/kuse1` + 运行期三元，
local 扇区归 **0**。

### 71.3 数值：与 vs-ref/TE 打印精度一致、单两文件 gate 全绿（只改 exp/求和次序）

SASS 的两趟化只改 LSE 的 fp32 求和次序 ⇒ LSE 差 ~1e-7，再经 fp8 量化取整边界会翻少数元素，
使 dq/dk/dv 相对旧实现偏移 **~1e-3**（远小于 fp8 本身 ~0.26 的对拍误差，且不改变 vs-ref 的
max_abs）。实测 **vs ref / vs TE 与历史打印位一致**：

| case（fp8 causal） | vs fp32 ref max_abs（OLD = NEW） |
|---|---|
| S512 H16 | 2.426e-1 / 2.972e-1 / 3.733e-1 |
| S1024 H32 | 2.399e-1 / 4.177e-1 / 3.535e-1 |
| S1024 H32 kv4 | 2.517e-1 / 5.339e-1 / 7.173e-1 |
| S4096 H16 | 2.635e-1 / 2.644e-1 / 3.216e-1 |
| MLA S1024 H2 D512 | 2.232e-1 / 3.337e-1 / 3.602e-1 |
| varlen b1_t512 D128 | 2.280e-1 / 3.108e-1 / 3.422e-1 |

单/两文件一致性 gate：**worst=1.335e-5（tol 1e-4）OK**；`--check docs/04` OK 194 行
（`docs/04` 内嵌表无需刷新）。旧/新逐元素差 `max|new-old|`：S4096 dq 4.29e-3 / dk 1.62e-3 /
dv 6.68e-4（S512 dq 8.35e-7 —— 小 shape 常整块命中，差异更小）。

### 71.4 性能（同 session，CUDA event，OLD vs NEW）

**preprocess（LSE+delta）**：

| case | OLD preprocess (ms) | **NEW preprocess (ms)** | 倍数 |
|---|---|---|---|
| S512 H16 | 0.0188 | **0.0156** | 1.21× |
| S1024 H32 | 0.0513 | **0.0396** | 1.30× |
| S1024 H32 kv4 | 0.0499 | **0.0371** | 1.34× |
| S4096 H16 | 0.2155 | **0.1378** | **1.56×** |

**端到端 total**：S512 0.1010→**0.0977**（1.034×）、S1024H32 0.3847→**0.3711**（1.037×）、
kv4 0.3524→**0.3381**（1.042×）、S4096 1.8876→**1.8172ms（75.63 TF，1.039×）**。
**varlen**：b1_t512 D128 0.1090→**0.1066**（1.023×）、b4_t3840 D128 0.9264→**0.8904（1.040×）**、
b3_t1792 D512(MLA) 0.2766→**0.2737**（1.011×）。main 不受影响（逐 shape 持平）。

**对标**（同 session 纯反向 `harness/fa_bwd_bench.py bench --dtype fp8`，CUPTI device）：
TE FP8 S512 **0.0357ms/120.3TF**、S1024H16 0.0552/311、S4096 **0.3003ms/915TF** ⇒ ours total
（含 quant+preprocess+main）S512 **2.74×**、S4096 **6.05×**（F1 时 2.83×/6.43×）。

### 71.5 ncu：LSE 1.67×、指令 −38.7%、local 归零

`lse_mma_kernel_bal_tma<128,1>`（S=4096，grid (32,16,4)，128 线程）：

| 指标 | OLD | **NEW** |
|---|---|---|
| Duration | 201.25 µs | **120.29 µs（1.67×）** |
| Executed Instructions | 140,994,592 | **86,441,386（−38.7%）** |
| local `op_ld`/`op_st` 扇区 | 532,480 / 598,016 | **0 / 0** |
| SM Throughput | 73.73% | 75.73% |
| registers / occupancy | 59 / 42.19% | 59 / 41.69% |

指令数 −38.7% 与 SASS 归因（`fexp` 64→34、去 214 条 BSSY/BSYNC、去 ~120 条 FSETP）吻合。
**结论：LSE 此前是纯 issue-bound（73.6%），该 epilogue 重写把「每元素额外一次废物 exp +
逐元素分支」这块纯开销拿掉，是本轮全部收益来源。**

### 71.6 复现 / 原始输出

```bash
# 两文件（Hopper：sm90a + wgmma + TMA）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=80
# A/B：OLD（pre-change）二进制 vs NEW
# ncu LSE
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --launch-count 1 \
  --kernel-name regex:lse_mma_kernel_bal_tma --set full -- --iters=1
# 单文件 device 同步
python3 scripts/sync_onefile_device.py src/fp8/fa_bwd_fp8_kernels.cuh \
  src/fp8/fa_bwd_fp8_mma_onefile.cu '#include <cuda_runtime.h>'
```

原始输出：`src/fp8/fa_bwd_fp8_f5_lse2pass_ab.out.txt`（8 shape×OLD/NEW 计时+对拍）、
`src/fp8/fa_bwd_fp8_f5_ncu_lse_s4096.out.txt`（LSE OLD/NEW 全指标）、
`src/fp8/fa_bwd_fp8_f5_te_baseline_fp8.out.txt`（同 session TE FP8 基线）。
全量 `python3 harness/fa_bwd_run.py --ci` 73 case 全绿（fp16 7.812e-3 / bf16 7.812e-3 /
fp8 1.335e-5，`--check docs/04` OK 194 行）。

**下一步候选**：① main kernel 仍是 81% 的墙（F3-a：L2 流量受 3 CTA/SM 锁死，真杠杆是 F4/改工作划分）；
② quant（0.069ms）已 DRAM ~72%，delta/convert 也近带宽，F5 至此收口；③ F4 默认路径的 L2 `red`
（114.5M 扇区）仍受本卡寄存器/smem 硬墙锁定，见「阻塞」。

---

## 72. F4-c（第一百三十七轮，**正结果（DET 路径），opt-in `--det`**）：fp16 DET partial 归约的读取向量化

### 72.1 动机与现状（落实 §70.4 / §71.6 的「补充任务」）

F4/F4-b 把 fp8 的确定性 `--det` 路径的 dK/dV partial 从 fp32 降到 **fp16 + 写扇区化**
（`dkv_det_store_h4` 每 lane 写 8B = 4 个连续物理列），reduce 端的 `P16` 却仍**逐列 2B 标量读**
（`dkv_p16_perm(c)` 反查）。ncu 实测这版 fp16 reduce：

| S=4096 causal | fp32 reduce（`P16=0`） | fp16 reduce（`P16=1`，F4-b） |
|---|---|---|
| grid / block | `(16, 4096)×128` | `(16, 4096)×128` |
| Duration | 734 µs | ~421 µs |
| **DRAM 吞吐** | **91.2%** | **~78%** |

即 fp16 partial 虽把字节减半，但 **2B 标量读把 HBM 效率从 91% 掉到 78%**——reduce 是纯带宽
bound（§56/§68），这部分效率亏空是可回收的。本轮把 `P16` 归约读取向量化为 8B。

**关键观察**：`dkv_det_store_h4` 写的 8B 单元在物理上就是 4 个连续列，而它对应的 4 个逻辑列
恰是 `{16·blk+2L, +1, 16·blk+8+2L, +1}`（`blk=j/2`、`L=lane&3`，见 `dkv_p16_perm` 推导）。
故 reduce 端只要**每线程按物理 4 列一组**读回，就能一次 `uint2` 拿 4 个元素；每个输出列的
求和集合/次序与标量读**逐字相同 ⇒ 结果逐位相同**。

### 72.2 实现（单/两文件 device 逐字同源，`sync_onefile_device.py` `identical: True`）

- 新增 `h4_to_f4(uint2, float&×4)`：把 8B 单元拆成 4 个 fp32。
- 四个归约 kernel 的 `P16` 分支改为**每 block 处理 `RW=4` 行**（`blockDim=HD`、每行 `HD/4` 个
  列组；`sub=threadIdx.x/(HD/4)`、`t=threadIdx.x%(HD/4)`），行号
  `jg = blockIdx.y*4 + sub`（定长/独立版）或 `(idx%rowblocks)*4+sub`（融合版）。
  每 (hh,m) 读一次 `uint2`；输出写 4 列。
  - `dkv_reduce_kernel` / `dkv_reduce_varlen_kernel`（独立版）；
  - `dkv_dq_reduce_kernel` / `dkv_dq_reduce_varlen_kernel`（融合版的 dK/dV 分支）。
  - `P16=false`（fp32）路径与 `dq` 分支**逐字未动**。
- host：4 处独立 P16 启动的 `grid.y = S→ceil(S/4)`（varlen 用 `ceil(maxlen/4)`）；3 处融合 P16
  的 `dkv_blocks = B*Hkv*S → B*Hkv*ceil(S/4)`（varlen 用 `ceil(maxlen/4)`）。**默认（非 `--det`）
  路径一行未改**。

### 72.3 数值：只改「哪个线程算哪列」，逐位不变

- 三处 `runs[1-2] bitwise dk/dv = 0`（确定性保留）。
- `fp16-vs-fp32`：S4096 `9.66e-4/1.76e-3`、MLA S1024H2 `9.07e-4/1.79e-3`、
  varlen D128 `1.01e-3/1.83e-3`、MLA S512H4 `9.26e-4/1.58e-3`——**与 F4/F4-b 打印逐位相同**。
- `ours-vs-ref` 与历史逐位不变（S4096 2.635/2.644/3.216e-1、varlen 2.280/3.108/3.422e-1、
  MLA S1024H2 2.232/3.337/3.602e-1）；单/两文件逐指标一致。

### 72.4 性能（同 session、同 binary 交替 A/B，`--det`）

S=4096 causal 的 `[F4 A/B]`（OLD=§68/§69 的标量 P16 读，NEW=本轮），3 次交替：

| | DET-fp16 均值 | DET/atomic |
|---|---|---|
| OLD（标量 P16 读） | **2.180 ms** | 0.899× |
| NEW（向量化 P16 读） | **2.147 ms（1.015×）** | **0.908×** |

端到端 DET 只动 1.4–2.5%（main 占 DET 的大头），**收益集中在 reduce 本体**：

| reduce（S4096，`--det`） | Duration | DRAM |
|---|---|---|
| fp32（`P16=0`，基线） | 734 µs | 91.2% |
| fp16 标量读（F4-b） | ~421 µs | ~78% |
| **fp16 向量化读（F4-c）** | **370 µs（1.14×）** | **92.5%** |

varlen D128 的 `[F4 A/B]` fp32→fp16 由 F4-b 的 1.057× 提到 **1.088×**；MLA S1024H2 1.088× /
S512H4 1.107×（main 占比大，故总比变化小）。

### 72.5 ncu（`dkv_reduce_kernel<128,64,1>`，S4096）

网格 `(16, 1024)×128`、Duration **369–370 µs**、**`dram__throughput` 92.5–92.7%**、
`l1tex 11.4%`。即向量化后 fp16 reduce 的 HBM 效率与 fp32 版持平（91%），**把「降精度减半字节」
的收益完整落袋**（此前被 2B 标量读的 78% 效率吃掉一半）。

### 72.6 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=100 --det
# reduce ncu（P16 实例）
ncu --launch-count 12 --kernel-name regex:dkv_reduce_kernel \
  --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  ./fp8.out --dir=... --iters=1 --det
# 单文件 device 同步
python3 scripts/sync_onefile_device.py src/fp8/fa_bwd_fp8_kernels.cuh \
  src/fp8/fa_bwd_fp8_mma_onefile.cu '// ----------------------------- 编译期常量 -----------------------------'
```

原始输出：`src/fp8/fa_bwd_fp8_p137_f4c_ab.out.txt`（同 session OLD/NEW×3）、
`..._p137_ncu_reduce16.out.txt`（fp32 vs fp16 P16 全指标）、
`..._p137_f4c_varlen_mla.out.txt`（varlen/MLA DET A/B）。

**下一步候选**：① F4 默认路径的 L2 `red`（114.5M 扇区）仍受本卡寄存器/smem 硬墙锁定（见
ROADMAP「阻塞」）；② main 仍是端到端 ~86%；③ DET 仅 opt-in，默认（非确定）路径本次未动。

## 73. 第 138 轮：F3/F4-default/GEMM3·5-wgmma 收口（fp8 main 墙的定量复核 + fp8 wgmma ISA 验证 + 负结果）＋ LSE split auto 大 S 档微调

### 73.1 动机

『fp8 专项冲刺』F1→F5 后，`下一步候选` 反复收敛到同一句：**默认（非确定）路径的 dK/dV/dQ
跨 CTA `red`（114.5M 扇区）是唯一真杠杆，但受本卡寄存器（168 regs @3 CTA/SM）与 smem
（74.8KB @3 CTA/SM）双硬墙锁定**。本轮不臆断，直接把这条结论**定量复核**一遍，并逐条排除
几个「看起来能做」的小改动；同时把 §23.6 那条『fp8 wgmma 无转置操作数』的**阻塞判据**拿
真实 CUTLASS 头文件二次核对（它是 GEMM3/4/5 能否上 wgmma 的唯一前提）。

### 73.2 默认 fp8 main 的墙（ncu，S=4096 causal，`fa_bwd_fp8_mma_kvtma_kernel<128,64,32,...>`）

| 指标 | 值 |
|---|---|
| Duration | **1.59 ms**（grid 512×16、ksplit=8、3 CTA/SM、128 线程/CTA） |
| **L2 Cache Throughput** | **76.96%**（新墙/并列第一） |
| L1/TEX Cache Throughput | 71.75% |
| Compute (SM) Throughput | 47.89% |
| DRAM Throughput | ~4.3%（L2 Hit Rate **97.08%**） |
| L2 扇区总计 | **154.0 M**（`red` **114.5 M = 74.3%**、read 30.9 M、write 8.4 M） |
| `lts__t_sectors_op_red` 占峰值 | **52.79%** |
| stall（per issue-active inst） | **`wait` 1.61 + `short_scoreboard` 1.27** + barrier 0.41 + long 0.40 |
| Scheduler | No Eligible 50.12%、Active Warps/Scheduler **2.94** |

结论与 F3-a/O42 一致：**L2 流量（其中 `red` 占 74%）是墙**；`wait`+`short_scoreboard`
合计 2.88/5.88≈49% 是**症状**（3 warp/scheduler 不足以同时隐藏 wgmma 依赖与 smem→mma 依赖），
不是能靠「重排指令」消掉的发射序问题（O50/O29 已证）。

**ksplit 复核**（同 session 交替）：k=1/2/4/8 → main **1.952/1.701/1.591/1.578 ms**，
即 auto 选的 k=8 正确（切 K 提并行度的收益 > dQ red ×8 的代价）。

### 73.3 fp8 wgmma ISA 验证（GEMM3/4/5 上 wgmma 的唯一前提）

直接读容器内 CUTLASS `cute/arch/mma_sm90_gmma.hpp`：

- fp8 `MMA_64x64x32_F32E4M3E4M3_SS_TN` 的 asm 尾操作数为 **`p, scaleA, scaleB`**（3 个）；
- fp16 `MMA_64x64x16_F32F16F16_SS` 的 asm 尾操作数为
  **`p, scaleA, scaleB, tnspA, tnspB`**（5 个）。

⇒ **fp8 `wgmma` 没有运行时转置立即数**，只能读固定的 TN 布局（A/B 的归约维连续）。反向的
GEMM3(dV=PᵀdO)/GEMM4(dK=dSᵀQ)/GEMM5(dQ=dS2·K) 的 **B 操作数都是 N（head_dim）连续**，
要上 wgmma 必须为 Q/dO/K 各物理存一份**转置 SW128 tile**（+20KB smem，且转置写是 O4b 已
证昂贵的 scatter）。**§23.6 的阻塞判据成立、and 给出精确 ISA 证据**（原始：
`src/fp8/fa_bwd_fp8_p138_wgmma_isa.out.txt`）。

### 73.4 负结果（逐条排除，均同 session 交替 / 同 binary A/B）

1. **累加缓冲清零旁路 stream 重叠**：把 dQ/dK/dV 三个 ~100MB `cudaMemset` 放非阻塞 stream、
   与 quant+preprocess 重叠（事件同步）。S=4096：**total 1.8072 vs 1.8096 ms（≈0.1%，噪声内）**
   ——mem 与 compute 争 SM/带宽，重叠不赚。`--zeroov` A/B 开关已回退（不入库）。
2. **量化 grid 上限 cap**（靠 kernel 自带 grid-stride 多吃几轮）：quant **0.0686 vs 0.0687 ms**
   ——quant 实测单 kernel 15.5µs、DRAM 70.8%，是**真带宽 bound**（非块调度），cap 无效。

⇒ 非 main 的三块（quant 0.069 / preprocess 0.139 / 残余 ~0.03ms）都已是各自机理下的
近最优，端到端固定开销没有可捡的余量。

### 73.5 正结果（小）：LSE K 维 split auto 的大 S 档重标定

D=128 TMA LSE 的 auto（§41 O38，目标 `grid*split≈2048`）在 **S≥2048 时比实测最优多切一档**：
S=4096 H16 的 `lg_grid=512`，目标 2048 → split=4；同 binary 交替实测
**split=1 / 2 / 4 / 8 = 0.1219 / 0.1185 / 0.1189 / 0.1272 ms**（best=2，+2.9% vs split1）。
改 `target`：**`S≥2048 → 1024`**（仍 ≈2 个满波量级），`S<2048` 维持 2048。S512 / S1024H32
实测两档相同（无回归）。端到端在噪声内（1.8086 vs 1.8083 ms），但使 auto 贴合 per-shape 最优。
单/两文件同步（`fa_bwd_fp8_main.cu` / `fa_bwd_fp8_mma_onefile.cu`）。

### 73.6 数值 / 回归

- `ours vs ref` 与历史**逐位相同**：S512 2.426/2.972/3.733e-1、S4096 2.635/2.644/3.216e-1、
  GQA kv4 2.517/5.339/7.173e-1、MLA S1024H2 2.232/3.337/3.602e-1。
- `python3 harness/fa_bwd_run.py --ci --dtype fp8 --fixed-only --hopper`：**rc=0**，
  单/两文件 gate `worst=7.629e-06（tol 1e-4）OK`、`--check docs/04` OK 194 行。

### 73.7 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --iters=300
# ncu（默认 main）
ARCH="" NVCC_FLAGS="$FLAGS" scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu \
  --launch-count 1 --kernel-name regex:kvtma --section SpeedOfLight --section SchedulerStats \
  --section WarpStateStats --metrics lts__t_sectors.sum,... -- \
  --dir=.../b1_s4096_h16_d128_causal_fp8 --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_p138_s4096.out.txt`（默认 + O38 split A/B + 全 A/B）、
`..._p138_ncu_main_s4096.out.txt`（L2 扇区/SOL/stall）、
`..._p138_wgmma_isa.out.txt`（fp8 vs fp16 SS asm 操作数）、
`..._p138_ci_fp8.out.txt`（CI 全绿）。

**下一步候选**：① 默认路径 L2 `red` 仍是唯一真杠杆——只有三条路：**跨 CTA 分块偏和**
（FA2 式 `dK/dV-over-KV`，工程量大）、**增大 BM**（O17b 寄存器墙）、**换卡（目标卡 smem/寄存器更大）**；
② GEMM3/4/5 的 wgmma 需物理转置 SW128 B（+20KB smem + scatter），O4b 已判净负，**ISA 层锁死**；
③ DET 仅 opt-in，非目标。

---

## 74. 第 139 轮：fp8 主 kernel「唯一真杠杆」定位——dK/dV 跨 CTA `red` 的定量归因 + 三条候选判决

### 74.1 动机

第 133–138 轮（F1→F5 + 收口）后，`下一步候选` 反复收敛到「默认 fp8 main 的 L2 `red` 是唯一真
杠杆」，并列了三条路：① 跨 CTA 分块偏和（FA2 式 `dK/dV-over-KV`）、② 放大 BM、③ 换卡。本轮
**不再猜测**，用 ncu 把 `red` 拆到「哪个梯度（dK/dV vs dQ）、哪一维（m-block / part）」并逐条
判决三条候选，避免对一条流量中性的路做一次大规模重写。

### 74.2 `red` 的定量归因（`fa_bwd_fp8_mma_kvtma_kernel<128,64,32,...>`，S=4096 causal）

| ksplit | L2 `op_red` 扇区 | L2 `op_read` | L1 `op_red` 扇区 | L2 吞吐 | Duration |
|---|---|---|---|---|---|
| **1**（dQ 无原子） | **103.8M** | 24.3M | 69.2M | 56.9% | 1.92 ms |
| **8**（默认 auto） | **114.5M** | 30.9M | 76.3M | 77.0% | 1.57 ms |

- **dK/dV 的跨 CTA `red` 与 ksplit 无关**：每个 K tile 只属于一个 part（K 切片不相交），故
  `red` 的 dK/dV 项恒为 ~104M 扇区。
- **ksplit 8→1 只让 `red` 掉了 ~10.7M（−9.3%）**，这部分就是 **dQ 的跨 part `red_add2`**
  （REGDQ 路径：每 part 覆盖同一 `dQ[BM][HD]` 的完整寄存器累加，再跨 part 原子加）。
  ⇒ **`red` 的 ~90% 是 dK/dV**，dQ 只占 ~10%。
- 结论：**「把 dQ 改 partial + reduce 消掉 dQ 原子」最多省 ~10% `red`，还要付 `dq_reduce`**
  （DET 实测 0.908× 已佐证），不值。

### 74.3 候选①「dK/dV-over-KV」判决：**流量中性（负杠杆），关闭**

设 causal、Q 块数 `nblk = S/BM`、`BM=BN`。

- **当前 Q 主序**：`dK/dV[j]` 被覆盖它的每个 m 块原子加一次 ⇒ 贡献数 `nblk - j`；
  总跨 CTA 元素加次数 `∝ Σ_{j=0}^{nblk-1} (nblk-j) · BN·HD = [nblk(nblk+1)/2] · BN·HD`。
  dQ 寄存器累加（ksplit=1 时无原子）。
- **KV 主序**：`dK/dV[j]` 单写者（`red → 0`），但 `dQ[m]` 被覆盖它的每个 K tile 原子加一次 ⇒
  贡献数 `m+1`；总跨 CTA 元素加次数 `∝ Σ_{m=0}^{nblk-1} (m+1) · BM·HD = [nblk(nblk+1)/2] · BM·HD`。
- `BM=BN` 时**两者逐项相等** ⇒ 只是把原子负担从 dK/dV 搬到 dQ，**总流量不变**。
- **GQA（`H > Hkv`）时更差**：dQ 是 `[S,H,HD]` 而 dK/dV 是 `[S,Hkv,HD]`，把原子搬到更大的 dQ
  上会让 red 放大 `H/Hkv` 倍。⇒ **该候选应关闭**（此前 ROADMAP 把它列为「唯一路之一」是乐观的）。

### 74.4 候选②「放大 BM 到 128」：**单独不够，必须配 TMA+wgmma**

现有 `fa_bwd_fp8_wg2_kernel<128,128,32>`（BM=128、2 warpgroup、mma + `cp.async`、无 TMA/wgmma）：

| kernel | BM | L2 `op_red` | L2 吞吐 | L1TEX | SM | warps active | Duration |
|---|---|---|---|---|---|---|---|
| `kvtma<...,64,32>` | 64 | 114.5M | **77.0%** | 71.8% | 47.9% | — | **1.57 ms** |
| `wg2<128,...>` | **128** | **58.2M（≈½）** | **20.4%** | 45.5% | 33.1% | 12.5% | **2.96 ms** |

- BM 64→128 确实把 `red` **砍半**（114.5M→58.2M），但 `wg2` **根本不是 L2-bound**（L2 仅 20.4%），
  而是 **1 CTA/SM（12.5% warps）的延迟/occupancy bound**（无 TMA、无 wgmma、每 tile 两次
  `__syncthreads`、单缓冲 `cp.async`、smem ~131KB 锁 1 CTA/SM）⇒ 慢 0.53×。
- ⇒ **「BM=128」本身不足以赢**；必须是 **TMA + wgmma + 双 warpgroup 的 BM=128**，才能「把 red
  砍半」的同时保持 kvtma 的高 L2 利用率。当前 `fp8_mma_body` 用
  `static_assert(WGMMA==false || (NTH==THREADS && NWAR==WN))` **锁死单 warpgroup**，故需新写
  双 warpgroup 的 body（对标 fp16/bf16 的 `fa_bwd_fp16_wgmma2_kernel`）。**这是唯一路径，工程量大。**

### 74.5 候选③ ksplit auto 复核（S=4096 causal，event iters=50）

`ksplit` = 1/2/4/8/12/16/24/32 → main **1.92/1.69/1.58/1.53/1.57/1.62/2.30/2.31 ms**
⇒ **auto=8 就是最优点**，无调参回归空间。

### 74.6 非 main 复核（F5 后 LSE）

`lse_mma_kernel_bal_tma<128,1>`（S4096 split=2）：Duration **121.7µs**、`sm__throughput`
**73.3%**、`issue_active` **77.6%**、warps active 36.0% ⇒ **仍是 issue-bound、已接近指令侧下限**
（F5 已 1.67×），无新的低垂果实。

### 74.7 结论 / 下一步

默认 fp8 main 的 `red`（**dK/dV，~104M L2 扇区，占 `red` 90% / L2 流量 ~67%**）是唯一真杠杆。
三条候选里只有「**双 warpgroup + TMA + wgmma 的 BM=128**」能从根上把 `red` 砍半并保持 L2 利用率；
其余（dQ partial、KV 主序、ksplit 调参）要么收益 <10%、要么流量中性甚至更差。
→ 新立 **F6**（见 ROADMAP「fp8 专项冲刺」）。

### 74.8 复现 / 原始输出

```bash
FLAGS='-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda'
# ksplit sweep（event）
for k in 1 2 4 8 12 16 24 32; do
  ARCH="" NVCC_FLAGS="$FLAGS" scripts/run.sh src/fp8/fa_bwd_fp8_main.cu \
    --dir=.../b1_s4096_h16_d128_causal_fp8 --iters=50 --ksplit=$k
done
# ncu：默认 main / ksplit=1 / wg2 / LSE
ARCH="" NVCC_FLAGS="$FLAGS" scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu \
  --kernel-name regex:kvtma -c 1 --metrics lts__t_sectors_op_red.sum,... -- ...
```

原始输出：`src/fp8/fa_bwd_fp8_p139_ksplit_sweep.out.txt`、
`..._p139_ncu_main_ks8_s4096.out.txt`、`..._p139_ncu_main_ks1_s4096.out.txt`、
`..._p139_ncu_wg2_bm128_s4096.out.txt`、`..._p139_ncu_lse_s4096.out.txt`。

---

## 75. 第 140 轮：F6 第一步——双 warpgroup（256 线程）fp8 wgmma 冒烟（BM=128 几何钉死）

### 75.1 动机（落实 §74.7 / ROADMAP『fp8 专项冲刺』F6）

第 139 轮把默认 fp8 main 的「唯一真杠杆」钉死为 **dK/dV 跨 CTA 的 L2 `red`**（~104M 扇区，
占 `red` 90%、L2 流量 ~67%），且判决只有 **BM 64→128**（让每个 KV 元素被一半 CTA 贡献）
能从根上砍半。已有 `fa_bwd_fp8_wg2_kernel<128>`（BM=128、2 warpgroup、mma + `cp.async`）
**确实把 `red` 砍半（58.2M）**，却是 **217 regs / 131.33KB smem → 1 CTA/SM**、L2 仅 20.4%、
慢 0.53×（延迟/occupancy bound）。ROADMAP 判定的解法是「**双 warpgroup + TMA + wgmma**」：
用 Hopper 原语把寄存器/指令压下来。F6 是一条大改，本轮先把其中最不确定、也最不可复用的一块
**单独冒烟钉死**：在 **256 线程（2 个 warpgroup）** 的 CTA 里，用 K-major SW128 描述符跑
fp8 `wgmma.m64n64k32`，每个 warpgroup 各算 BM=128 的一半（64 行）。

### 75.2 冒烟内容（`src/fp8/fa_bwd_fp8_wgmma2_smoke.cu`）

最小 GEMM，输入取 `{-3..3}` 整数（e4m3 的 3 位尾数、e5m2 的 2 位尾数都能**精确**表示，
CPU 参考无需 host 反量化，避开 kernel-opt 32 篇的 fp8 转换坑）：

- **GEMM1** `S = Q·Kᵀ`：`Q[128][128]` e4m3、`K[64][128]` e4m3，K 归约维 = HD = 128 = 4×k32；
- **GEMM2** `dP = dO·Vᵀ`：`dO[128][128]` e5m2、`V[64][128]` e4m3；
- 2 个 warpgroup 各发自己的 `wgmma`：第 2 个 WG 的 A 描述符基址按 `(64 行/8)×atom(1024B)
  = 8192B` 偏移；accumulator epilogue 用 `row = wg*64 + wl*16 + g + (q≥2?8:0)`、
  `col = j*8 + 2*(lane%4) + (q&1)`。

### 75.3 结果（原始输出 `src/fp8/fa_bwd_fp8_wgmma2_smoke.out.txt`）

```
wgmma2(2 WG) GEMM1 S=QKᵀ  e4m3×e4m3  BM=128 vs CPU: max_abs=0.000e+00
wgmma2(2 WG) GEMM2 dP=dO·Vᵀ e5m2×e4m3 BM=128 vs CPU: max_abs=0.000e+00
PASS
```

SASS（`..._sass.out.txt`，`cuobjdump -sass` opcode 直方图）：**4×`QGMMA.64x64x32.F32.E4M3.E4M3`
+ 4×`QGMMA.64x64x32.F32.E5M2.E4M3`，无 HMMA / 无 LDSM**——两个 warpgroup 各自完整走完归约维
（128/32 = 4 步），且都落在 Hopper `wgmma` 指令上。

ncu（`..._ncu.out.txt`，`--set full`，单 CTA）：**90 regs**、smem 49.15KB、`Block Limit
Registers=2`、`Block Limit Shared Mem=2`、achieved occ 12.24%。⇒ **wgmma 版 GEMM1/2 的寄存器
压力远低于 wg2 的 217**（关键 F6 论据：wgmma 省掉 `ldmatrix` 与全局地址寄存器）。

### 75.4 F6 的剩余工作量与资源账（下一步）

冒烟只证明「双 WG wgmma 几何成立」，真正的 F6 kernel 还要：
1. **Q/K/V/dO 走 4D-TMA + SW128**（复用 O32/O37/O41 的描述符与 mbarrier 基建，已在
   `lse_mma_kernel_bal_tma` / `fp8_mma_body<...,TMA>` 验证），去掉 wg2 的逐元素全局读与
   `Qp/dOp/Kp` 的寄存器构造；
2. **GEMM1/2 换 wgmma（K-major）**；3/4/5 仍用 `mma.m16n8k32`（fp8 `SS_TN` asm 无
   `tnspA/tnspB`，见 §73 阻塞，转置读不可用）；
3. **smem 必须压到 ≤116,224B 且 regs ≤128** 才能 2 CTA/SM：wg2 现 131.33KB（超 15.1KB）、
   217 regs（超 89）。冒烟显示「wgmma GEMM1/2」本身只要 90 regs，缺口主要在
   `dqacc[2][8][4]`(64) + fold/pair 构造 + 3/4/5 的 mma epilogue；TMA 化 Q/dO（去掉
   `Qp/dOp` 的寄存器构造）与「Ps/Ss 二选一驻留」是两条待试的具体路径。

`red` 的账不变：BM=128 ⇒ 每 KV 元素贡献 CTA 数减半 ⇒ `red` 从 114.5M 砍到 ~58M，
若同时靠 TMA+wgmma 把 occupancy 从 1 拉到 2 CTA/SM、L2 从 20.4% 拉回 ~50%，即可兑现 F6。

### 75.5 复现 / 原始输出

```bash
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/run.sh src/fp8/fa_bwd_fp8_wgmma2_smoke.cu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/ncu.sh src/fp8/fa_bwd_fp8_wgmma2_smoke.cu --set full \
  --kernel-name regex:wgmma2_smoke --launch-count 1
```

原始输出：`src/fp8/fa_bwd_fp8_wgmma2_smoke.out.txt`、`..._sass.out.txt`、`..._ncu.out.txt`。
回归：`harness/fa_bwd_run.py --ci --dtype fp8 --fixed-only` **rc=0**（gate worst=7.629e-06、
`--check docs/04` OK 194 行），默认路径一行未改（原始输出 `src/fp8/fa_bwd_fp8_p140_ci_fixed.out.txt`）。

## 76. 第 141 轮：F6 第二步——双 warpgroup 主 kernel 的 GEMM1/2 wgmma 化（`--wg2wgmma`，**正确但中性**）

### 76.1 动机与实现

第 140 轮冒烟（§75）证明「256 线程 / 2 个 warpgroup 下用 K-major SW128 跑 fp8
`wgmma.m64n32k32`、每 WG 算 BM=128 的一半」几何成立且只有 90 regs。本轮把它**落进主 kernel**：
新增 `src/fp8/fa_bwd_fp8_kernels.cuh` 的
`fa_bwd_fp8_wgmma2_kernel<HD=128, BM=128, BN=32>`（`#ifdef FA_WGMMA`），它 =
`fa_bwd_fp8_wg2_kernel`（§3b，BM=128、2 warpgroup、mma）**把 GEMM1/2 换成 wgmma**：

- **Q/dO/K/V 存 SW128 K-major**（wgmma 描述符直读 smem），Qp/dOp/Kp 由 SW128 用
  `__byte_perm` 重建（供 GEMM3/4/5 的 `ldmatrix.x2.trans`）；**fold 与 GEMM3/4/5 逐字沿用 wg2**。
- **GEMM1/2**：每个 warpgroup 各发 `wgmma.m64n32k32`（A 描述符按 WG `+ (64/8)*1024 = 8192B`），
  累加器映射 `row = wg*64 + wl*16 + g + (q>=2?8:0)`、`col = j*8 + c2 + (q&1)`，epilogue 直接
  写 `Ps/Ss`；rowwise scale 仍在 epilogue 折算（口径与 wg2 逐项一致，仅 fp 归约次序略变）。
- **Ap/dS3 从 wg2 的「别名 Ks/Vs」改为独立缓冲**：SW128 的 Ks/Vs 各只有 4096B，放不下
  `BN×QTS = 32×144 = 4608B`，且 Vs 在 GEMM2 时仍被读。

host 加 `--wg2wgmma`（`--wg2` 的 wgmma 版，复用同一 `mg2`/ksplit2 自动档）；单/两文件同步
（`sync_onefile_device.py`），默认路径一行未改。

### 76.2 数值（vs fp32 ref，causal；单/两文件逐位一致）

| shape | 默认 kvtma（历史） | **wg2wgmma** |
|---|---|---|
| S=512 | 2.426 / 2.972 / 3.733e-1 | **2.426 / 2.996 / 3.713e-1** |
| S=4096 | 2.635 / 2.644 / 3.216e-1 | **2.635 / 2.760 / 3.325e-1** |

均 fp8 噪声量级、无系统误差。`[F6 A/B]` 的 `wg2wg-vs-wg2` dq/dk/dv：S512
`4.5e-2/5.4e-2/5.7e-2`、S4096 `4.0e-2/1.5e-1/2.6e-2`（与既有 `[O9c-2 A/B]` 的
wgmma-vs-mma 同量级——来自 GEMM1/2 的累加/归约次序，不是 bug）。

### 76.3 性能（同 session，CUDA event，main-only）

| shape | 默认 kvtma<128,64,32> | wg2（mma GEMM1/2） | **wg2wgmma** | vs wg2 | vs 默认 |
|---|---|---|---|---|---|
| S=512 | 0.0717 ms | 0.1101 ms | **0.1078 ms** | **1.021×** | 0.66× |
| S=4096 | 1.911 ms | 2.9311 ms | **2.8081 ms** | **1.044×** | 0.68× |

⇒ **GEMM1/2 wgmma 化相对 wg2 只快 2–4%**，整 kernel 仍远慢于默认 BM=64（0.66–0.68×）。

端到端（含 quant/preprocess/convert）wg2wgmma：S512 **0.1410ms / 14.73 TF**、S4096
**3.0599ms / 44.92 TF**，相对 FP8 峰值（1978.8 TF）为 **0.74% / 2.27%**（仅参考；该路是
F6 优化探针、非发布口径，默认 kvtma 档的 TE 对标见 `docs/04`）。

### 76.4 ncu（wg2wgmma，S=512，`--set full`）

**212 regs / smem 136.4KB**（wg2 是 217 / 131.3KB）、`Block Limit Registers=1`、
`Block Limit Shared Mem=1`、theoretical = achieved **12.50%**（**1 CTA/SM**）、Duration 109µs、
DRAM 4.67% / L1/TEX 35.5% / L2 29.2% / Compute 25.0%、No Eligible 70.5%、stall
`wait 1.67 + long_scoreboard 1.49 + short 0.76`。
SASS（`cuobjdump -sass` opcode 直方图）：**`16×QGMMA.64x32x32.F32.E4M3.E4M3` +
`16×QGMMA.64x32x32.F32.E5M2.E4M3`**（2 WG × 4 k-step × 2 GEMM）+ `1568×HMMA` + `766×LDSM`
（GEMM3/4/5 仍 `mma.sync`）⇒ GEMM1/2 确实走 wgmma。

### 76.5 结论 / 下一步

- **GEMM1/2 wgmma 化本身是正的、但太小**（1.02–1.04× over wg2）：第 140 轮冒烟的 90 regs
  只在「只有 GEMM1/2」时成立；一旦叠上 `dqacc[2][8][4]`(64) + fold + GEMM3/4/5 的 mma，
  整 kernel 仍 **212 regs / 136KB → 1 CTA/SM**，延迟/occupancy bound 未变 ⇒ 不敌 BM=64 的
  3 CTA/SM 默认档。且本轮为 Ap/dS3 独立缓冲把 smem **推高**（131→136KB），方向相反。
- **F6 的必要条件是「TMA + 压 smem ≤116,224B」**：唯一能把 smem 压到 2 CTA/SM 预算
  （≤116,224B）的路径是**去掉 Qp/dOp（34,816B）**——从 SW128 的 Q/dO tile 里直接 `ldmatrix`
  读出 GEMM3/4 的 B（kernel-opt 42 篇已证 SW128 的 16B chunk 可 `ldmatrix` 转置读），
  或改 3/4/5 的 warp 几何。这是 F6 主体下一小步，见 ROADMAP backlog/「下一步」。
- 默认路径一行未改；`harness/fa_bwd_run.py --ci --dtype fp8 --fixed-only` **rc=0**
  （gate worst=1.049e-5、`--check docs/04` OK 194 行）。

### 76.6 复现 / 原始输出

```bash
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8 --wg2wgmma --iters=20
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --kernel-name regex:fa_bwd_fp8_wgmma2_kernel \
  --launch-count 1 -- --dir=.../b1_s512_h16_d128_causal_fp8 --wg2wgmma --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_p141_wg2wgmma_{s512,s4096}.out.txt`、
`..._p141_onefile_wg2wgmma_s512.out.txt`（单文件，数值逐位一致）、
`..._p141_ncu_wg2wgmma_s512.out.txt`、`..._p141_ci_fixed.out.txt`。

## 77. 第 142 轮：F6 判定——「去 Qp/dOp（SW128 直读 B）」在 fp8/`.row.col` 下**不可行** + 资源账收口

### 77.1 动机（落实 §76.5「下一步候选 ①」）

第 141 轮把 F6（BM=128 双 warpgroup）要冲 2 CTA/SM 的**最大障碍**定位为 `Qp/dOp`（两份
「K 配对」B 副本，共 34,816B），并判断「可去掉它——从 SW128 的 Q/dO tile 直接 `ldmatrix`
读出 GEMM3/4 的 B（kernel-opt 42 篇已证 SW128 16B chunk 可转置读）」。本轮不臆断，先用
**ISA 探针 + 最小复现 + 资源账**三条证据把这一前提判决清楚。

### 77.2 判决 1（ISA）：fp8 `mma.m16n8k32` **只支持 `.row.col`**

`src/fp8/fa_bwd_fp8_mma_variant_probe.cu` 编译唯一合法布局 `.row.col`；对
`.col.row` / `.row.row` / `.col.col` 分别单编，ptxas 全部拒绝（原始输出
`fa_bwd_fp8_mma_variant_probe.illegal.out.txt`）：

```
----- .col.row -----  Illegal alayout '.col' for instruction 'mma'
                      Illegal blayout '.row' for instruction 'mma'
----- .row.row -----  Illegal blayout '.row' for instruction 'mma'
----- .col.col -----  Illegal alayout '.col' for instruction 'mma'
```

⇒ B 操作数**必须是 col-major**（逻辑 `B[K][N]` 存成 `[N][K]`、K 连续）⇒ 反向的
GEMM3/4/5（B=`Qᵀ`/`dOᵀ`/`Kᵀ`）**必须转置**，不能沿用 GEMM1/2 的 K-major（N 连续）tile。
这与第 138 轮 fp8 wgmma「无 `tnspA/tnspB`」是同一根因（转置不可免）。

### 77.3 判决 2（pairing）：fp8 `.trans` 的配对轴是 tile 的连续轴（N），不是 K

`ldmatrix` 以 **b16** 为单位转置，而 **1 个 b16 = 2 个相邻 fp8**。K-major tile（Q/dO 的
`[m][d]`，`d`=N 连续）里每个「16B 行」的 2-fp8 配对**沿 N**；`.trans` 只交换 8×8 矩阵的
行列、**不改变 b16 内部的配对方向** ⇒ 寄存器里得到的仍是「沿 N 的 4 个 fp8」，而不是 mma
需要的「沿 K 的 4 个 fp8」。O4b 的 `fa_bwd_fp8_trans_smoke.cu`（§17）已给出同一结论并据此
改用「K 配对布局」；本轮把它显式化。

### 77.4 判决 3（穷举）：K-major tile 的 `x4.trans` 寄存器任意 2 个都拼不出 B 片段

新增 `src/fp8/fa_bwd_fp8_f6_directb_smoke.cu`：同一批 fp8 B 字节，一路从 col-major
`[N][K]` 用 `ldmatrix.x2`（已知正确的 mma B 路径）取片段，另一路从 K-major `[K][N]`
（即 Q/dO 原布局）用 `ldmatrix.x4.trans` 取全部 4 个寄存器，然后**穷举** 4 个寄存器里任意
有序 2 个，看能否逐位复现前者的片段。实测：

```
  lane 级片段匹配（x4.trans 任选 2 reg vs col-major x2）：0/32
      lane0  ref = 4f359c20 4e26b2c7
      lane0  att = b420b731 2f313731 24b12db6 b1aa1299
=== FAIL：K-major (N 连续) tile 无法用 ldmatrix.trans 拼出 mma 的 B 片段 ⇒ F6『去 Qp/dOp』不可行 ===
```

⇒ **不存在可用的地址/寄存器重排方案**。F6「从 SW128 直读 B」在 fp8 上不成立。

### 77.5 资源账：F6 要 2 CTA/SM 的缺口与可行项

`cuobjdump --dump-resource-usage`（sm90a 构建）：

| kernel | REG | 动态 smem | CTA/SM |
|---|---|---|---|
| 默认 `fa_bwd_fp8_mma_kvtma_kernel<128,64,32,...>` | **164–168** | 74.8KB | **3** |
| `fa_bwd_fp8_wg2_kernel<128,128,32>`（BM=128, mma） | 217 | 131.3KB | 1 |
| `fa_bwd_fp8_wgmma2_kernel<128,128,32>`（F6） | **212** | **135,424B（132.25KB）** | **1** |

wgmma2 的 smem 明细（代码常量）：`Qs/dOs` SW128 2×16384、`Ks/Vs` 2×4096、`Qp/dOp`
2×17408、`Kp` 4352、`dS2` 6144、`scales` 2048、`Ps/Ss` 2×18944、`Ap/dS3` 2×4608
（+1024 对齐余量）。2 CTA/SM 的上限 = `232448/2 = 116,224B`，**缺口 19,200B**。

- **去 Qp/dOp（34,816B）**：本是最直接的解法，但被 §77.2–77.4 三条证据判**不可行**。
- **可行但代价高**：`Ps/Ss` fp32→fp16（−18,944B）+ 去全部 padding（约 −10KB）刚压到
  ~105KB，但 `Ps` 是 `[0,1]` 的 P、`Ss` 改半精度会**改数值口径**；且 regs 212 要降到
  ≤128 才能真 2 CTA/SM（`__launch_bounds__(256,2)`），**−84 寄存器必然大 spill**。
  - ⇒ 即便绕过 Qp/dOp，F6 也**只能停在 1 CTA/SM**；而 1 CTA/SM 下 212 regs/8 warp 的延迟
  隐藏正是它只有默认档 0.66–0.68× 的原因（§76）。**F6「冲 2 CTA/SM」在本卡不成立。**

本轮 ncu（`fa_bwd_fp8_wgmma2_kernel`，S=512，`--set full`）复核：**212 regs**、
**Dynamic Shared Memory 136.45KB**、`Block Limit Registers=1` / `Block Limit Shared Mem=1`、
**Theoretical/Achieved Occupancy 12.50%/12.45%**、DRAM 4.67% / L2 29.18% / Compute 25.09%、
`No Eligible 70.45%`（原始输出 `fa_bwd_fp8_p142_ncu_wgmma2_s512.out.txt`）⇒ 与 §76 逐项一致，
确认「寄存器 + smem 双限 1 CTA/SM」是 F6 的硬墙。

### 77.6 性能复核（同 session，CUDA event，main-only）

```
[O41 A/B] main Q/dO-TMA 1.7160 ms | Q/dO/K/V-TMA 1.5691 ms   ← 默认 BM=64 档
[F6 A/B]  main wg2(mma) 2.9225 ms | wg2wgmma 2.8083 ms (1.041×)  ← BM=128 档
```

S=4096 causal：默认 main **1.569ms** / total ≈1.92ms；F6（wgmma2）main **2.808ms**（默认的
0.56×）。同 session TE FP8 纯反向基线 S=4096 = **0.3003ms/915.25TF**（`fa_bwd_fp8_f5_te_
baseline_fp8.out.txt`）⇒ 默认 ours main/TE ≈5.2×、F6 ≈9.4×。**F6 当前不是前进方向。**

### 77.7 结论 / 下一步

1. **F6 的「去 Qp/dOp 冲 2 CTA/SM」在 fp8/`.row.col` 下不可行**（ISA + pairing + 穷举三证）。
2. F6 若继续，唯一理论路 = **物理转置 SW128 B**（`[N][K]` K 连续，+16KB smem + scatter），
   O4b（§17）已判净负；或**换卡**（目标卡 smem/寄存器更大）。**转 backlog**。
3. 默认 BM=64 `kvtma` 档仍是主力，其墙仍是 **L2 `red`（dK/dV 跨 CTA 原子）**，见「阻塞」；
   F6 不构成对它的替代。默认路径一行未改。

### 77.8 复现 / 原始输出

```bash
scripts/run.sh src/fp8/fa_bwd_fp8_mma_variant_probe.cu
scripts/run.sh src/fp8/fa_bwd_fp8_f6_directb_smoke.cu
docker exec kernel_lab bash -lc "cd .../src/fp8 && nvcc -O3 -gencode=arch=compute_90a,code=sm_90a \
  -DFA_WGMMA -DFA_TMA -lcuda fa_bwd_fp8_main.cu -o /tmp/fp8wg.out && \
  cuobjdump --dump-resource-usage /tmp/fp8wg.out | grep -A1 wgmma2"
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8 --wg2wgmma --iters=20
```

原始输出：`src/fp8/fa_bwd_fp8_f6_directb_smoke.out.txt`、`fa_bwd_fp8_mma_variant_probe.out.txt`、
`fa_bwd_fp8_mma_variant_probe.illegal.out.txt`、`fa_bwd_fp8_p142_f6judge_s4096.out.txt`。

## 78. O64（第一百四十四轮，**正结果，默认化**）：fp8 输入量化 + 累加缓冲清零融合成单 launch

### 78.1 动机

F1→F6 收口后，fp8 默认 main 受本卡寄存器/smem 硬墙锁定（见 ROADMAP「阻塞」），改从**非 main 的
固定开销**找余量。用 nsys 逐 kernel 拆 fp8 默认路径（S1024H32 causal）发现：默认路径 per-call 有
**4 个 `quantize_row_warp_kernel`（q/dO/k/v）+ 3 个 `cudaMemset`（dQ/dK/dV）** 共 7 次串行小 launch；
ncu 显示每个 quant kernel 只到 **DRAM 58% / Compute 43%**（8.7µs/个，尾波 + launch 间隙截断），
清零又与量化串行。合计 quant ~40.8µs + 尾/清零残留 ~31.4µs ≈ **端到端 19%**（S1024H32）。

### 78.2 实现

新增 `quantize_zero_warp_kernel<VPT>`（`src/fp8/fa_bwd_fp8_kernels.cuh`）：每 warp 一行，单一 1D grid
按任务序号覆盖「量化 Q/dO/K/V + 清零 dQ/dK/dV」7 类任务；量化逻辑抽成 `quant_row_warp<VPT,E5M2>`
（与 O14 **逐字相同**：float4 读、lane amax、warp `shfl_xor` 树、scale、`cvt_*`、`uchar4` 写），
清零用 `zero_row_warp<VPT>`（float4 写 0）。默认路径（定长 + varlen）的 `run_all` 改走它；
`--qfuse=0` 退回旧「4 quant + 3 cudaMemset」做同 binary A/B。**数值逐位不变**（每行 amax/scale/cvt
与 O14 完全一致；清零只是写 0）。fp16/bf16 无量化步，故本项仅 fp8 适用。

### 78.3 结果（同 session 同 binary `[O64 A/B]`，CUDA event）

| shape | 形态 | fused (ms) | unfused (ms) | 加速 |
|---|---|---|---|---|
| b1_s512_h16_d128 causal | 定长 | 0.0848 | 0.0990 | **1.168×** |
| b1_s1024_h32_d128 causal | 定长 | 0.3550 | 0.3714 | **1.046×** |
| b1_s1024_h32 kv4 | 定长 GQA | 0.3232 | 0.3388 | **1.048×** |
| b1_s4096_h16_d128 causal | 定长 | 1.7974 | 1.8256 | **1.016×** |
| varlen b1_t512_h2_d512 causal | varlen MLA | 0.0793 | 0.0933 | **1.177×** |
| varlen b4_t3840_h16_d128 causal | varlen | 0.8777 | 0.8951 | **1.020×** |

数值：`max_abs(fused-vs-unfused) dq/dk/dv = 1.2e-7 / 4.8e-7 / 7.2e-7`（仅跨 CTA `atomicAdd` 次序）；
对 fp32 ref / TE 的 max_abs 与 O4b/O7 历史一致。`--ci --dtype fp8`（定长 + varlen）**全绿**
（一致性 gate worst 定长 1.144e-05 / varlen 2.861e-06，tol 1e-4）、`--check docs/04` OK 194 行。
单/两文件同源（device 由 `sync_onefile_device.py` 核对 `identical: True`）。

**ncu**（融合 kernel，S1024H32）：`Duration 45.7µs、DRAM 73.1%`（旧单 kernel 58%）、`L2 82.6%`、
`Compute 61.2%`、`27 regs`、`occ 61.5%`。**nsys**：默认 per-call kernel 数 **10→5**
（`quantZERO` 44.5µs 一次、无 memset；其余 LSE 27.4 / merge 2.8 / delta 10.9 / main 257.9µs），
per-call 间隙残留 ~31→~12µs。

### 78.4 结论 / 复现

收益来自「把 7 次串行小 launch（含 3 次 memset）并成 1 次、让 DRAM 流水连续」——**小/中 shape 收益
最大**（launch 与尾波占比高），S4096 因主 kernel 占比高仅 ~1%。零风险默认化。

```
# 定长（Hopper 快路）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s1024_h32_d128_causal_fp8 --iters=50
# varlen（wgmma 构建）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen --dir=.../varlen_b1_t512_h2_d512_causal_fp8
# 同 binary A/B：加 --qfuse=0
```

原始输出 `src/fp8/fa_bwd_fp8_o64_{fixed_s512,fixed_s1024h32,fixed_gqa_kv4,fixed_s4096,onefile_s512,
varlen_b1t512,onefile_varlen_b1t512,varlen_b4t3840}.out.txt`、`..._o64_ncu_{quantzero,quantold}_s1024h32.out.txt`、
`..._o64_nsys_s1024h32.out.txt`。

---

## 79. O66（第一百四十六轮，**正结果，默认化**）：fp8 把 delta 融进「量化 + 清零」单 launch

### 79.1 动机与现状

O64（§78）已把「4 次输入量化 + 3 次累加缓冲清零」并成 1 个 `quantize_zero_warp_kernel`，但
**delta（`D = rowsum(dO∘O)`）仍是紧接着的一次独立 launch**（`delta_warp_kernel`，O26）。
delta 恰好只依赖 **dO 与 O** 两项、且是**逐 query 行的归约**——与 dO 的 rowwise 量化同一数据域、
同一 warp-per-row 几何。背景：F1→F6 收口后默认 fp8 main 受本卡寄存器/smem 硬墙锁定（见 ROADMAP
「阻塞」），本轮的既定路线是继续清**非 main 固定开销**（继 O64 后的下一小步）。

### 79.2 实现（单/两文件 device 逐字同源，`sync_onefile_device.py` 核对 `identical: True`）

- **device（`fa_bwd_fp8_kernels.cuh`）**：新增 `quant_delta_row_warp<VPT>`——把 O26 的
  `delta_warp_kernel` 的「读 O(fp32)+dO(e5m2)、行点积、`__shfl_xor_sync` 树」逻辑**并进 dO 的
  量化任务**：量化时 `v[]` 已在寄存器里，量化后直接用 `deq_e5m2(cvt_e5m2(v/s))·s` 与同步读入的
  `O` 行做点积，一次 `scale[row]`/`delta[row]` 写回。新增 `quantize_zero_delta_warp_kernel<VPT>`
  = O64 版任务序**逐字相同**，只把 dO 那一档换成 `quant_delta_row_warp`（多 2 个入参 `o`/`delta`）。
- **为什么必须耦合、不能「追加 delta 任务」**：同一 launch 内各 warp 任务无跨 warp 同步，若把 delta
  作为**独立任务**读 `do8`，会与 dO 量化任务产生读后写竞争。耦合在同一 warp 内（量化后立即用它
  寄存器里的值算 delta）天然无竞争。
- **数值逐位不变的证明**：lane 的累加顺序同为「`t` 外层（元素 `t*128+lane*4`）、float4 内
  `x/y/z/w` 内层」，随后同一 `__shfl_xor_sync` 树；`s == dos[row]`、`cvt/deq` 与量化/delta 现用
  函数逐字一致。A/B 实测 `max_abs(delta fused-vs-separate) = 0.000e+00`（所有 shape）。
- **host（`fa_bwd_fp8_main.cu` / 单文件 host）**：新增 `--dfuse=`（默认 1，需 `qfuse=1` 且
  `delta_warp_sel`）；定长 `run_all` 与 varlen `run_all` 在融合时调 `quant_zero_delta()`、并让
  `run_preprocess(!fused_delta)` 跳过独立 delta launch；新增 `[O66 A/B]`（同 binary 端到端 +
  delta 缓冲逐位校验）。`--dfuse=0` 退回 O64+独立 delta 做同 session A/B。

### 79.3 数值：与历史**逐位相同**

`ours vs fp32 ref`（fp8，causal）与 O4b/O7/O64 历史完全一致：S512 `2.426/2.972/3.733e-1`、
S1024H32 `2.399/4.177/3.535e-1`、S4096 `2.635/2.644/3.216e-1`、MLA S1024H2 `2.232/3.337/3.602e-1`；
varlen D128 b1_t512 `2.280/3.108/3.422e-1`、varlen MLA b3 `3.404/3.436/3.508e-1`。
`--ci --dtype fp8`（定长 + varlen）**全绿**：一致性 gate `worst = 9.537e-06`（tol 1e-4）、
`--check docs/04` OK 194 行；`delta fused-vs-separate` 全部 `0.000e+00`。

### 79.4 性能（同 binary `[O66 A/B]`，CUDA event）

| shape（fp8 causal） | delta 独立 (ms) | **delta 融合 (ms)** | 加速 | delta 逐位差 |
|---|---|---|---|---|
| S512 MHA（两文件） | 0.0847 | **0.0809** | **1.047×** | 0 |
| S512 MHA（单文件） | 0.0849 | **0.0809** | **1.049×** | 0 |
| S1024H32 | 0.3519 | **0.3434** | **1.025×** | 0 |
| S4096 | 1.7872 | **1.7828** | **1.003×** | 0 |
| MLA S1024H2（D=512） | 0.1642 | **0.1608** | **1.021×** | 0 |
| varlen D128 b1_t512（`--dfuse=1/0`） | 0.0934 | **0.0902** | **1.036×** | 0 |

即端到端 **1.003–1.049×**，**小/中 shape 收益最大**（launch 与尾波占比高）、大 S 边际——与
O64/O65 的规律一致：**融合只值「省一次小 launch」**。`[timing]` 分解显示 delta 的字节被并进
quant（quant 略升、preprocess 等量下降），净省的是 `delta_warp_kernel` 的 launch 与尾波。

### 79.5 ncu（融合 kernel，S1024H32）

`quantize_zero_delta_warp_kernel<4>`：**Duration 50.85µs、DRAM 76.3%（旧单 quant 58%）、L2 84.7%、
SM 61.8%、31 regs**（O64 版 27 regs）⇒ 仍是 **DRAM/L2 带宽 bound**，融合后 delta 的 O 读被
DRAM 流水吸收、未新增明显尾延迟。

### 79.6 结论 / 复现

**fp8 非 main 固定开销的「小 launch 融合」至此收口**：7 类任务 + delta 现在是**一次** launch。
零风险默认化（数值逐位不变）。剩余墙仍是默认 fp8 main 的 L2 `red`（受本卡寄存器/smem 硬墙锁定，
见 ROADMAP「阻塞」）。

```bash
# 定长（Hopper 快路）一键 + 汇总 + 一致性 + docs/04 校验
python3 harness/fa_bwd_run.py --dtype fp8 --fixed-only
python3 harness/fa_bwd_run.py --no-run --ci
# 同 binary A/B：加 --dfuse=0
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s1024_h32_d128_causal_fp8 --iters=50 [--dfuse=0]
# varlen（wgmma 构建）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen --dir=.../varlen_b1_t512_h16_d128_causal_fp8
```

原始输出 `src/fp8/fa_bwd_fp8_o66_{main_s512,main_s1024_h32,main_s4096,main_mla_s1024h2,onefile_s512,
varlen_b1_t512,varlen_b1_t512_dfuse0,varlen_b3_d512}.out.txt`、`..._o66_ncu_quant_s1024h32.out.txt`、
`..._o66_ci_{fixed,full}.out.txt`。

## 80. O67（第一百四十七轮）：dK/dV 归约 float4（`FA_R4`）——负结果；并用 ncu 重测 TE 定位真差距

> 落实 ROADMAP「fp8 专项冲刺」/「阻塞」里默认 fp8 main 的唯一真杠杆 **L2 `red`**。默认
> `kvtma` main（S4096 causal）ncu 复核：**`lts__t_sectors_op_red = 114,524,160`**（占 L2 总扇区
> 154M 的 74%）、read 30.9M、write 8.4M、`l1tex__t_requests_..._op_red = 9,543,680`（L1→L2
> 展开 ~12×）。本轮做了两件事：① 把 `red_add2`（8B `red.global.add.v2.f32`）提升到
> **`red_add4`（16B `.v4.f32`）**，期待请求/扇区再减半；② 用 `harness/te_fp8_ncu.py` 重测
> TE 的 fp8 反向并看 SASS，把「差距在哪」钉死。

### 80.1 O67 改动（`FA_R4`，默认 0）

`red_add2`（O4c）已把「相邻两列」打包成一次 `atomicAdd(float2*)`。本轮新增 `red_add4` 与
编译开关 `FA_R4`（默认 0）：fp8 `mma.m16n8k32` 的累加器里一个 quad（`lane&3`=0..3）的
`c2=(lane&3)*2` 恰是 **同 row 的连续 8 列**，用 `__shfl_down_sync(...,1)` 把 quad 的 float2
拼成两个 float4（列 0-3 由 lane0 写、列 4-7 由 lane2 写），`red.global.add.v4.f32` 的请求数
相对 float2 再减半。`epi_dv`/`epi_dk` 的非 DET/BULKRED 分支同步改；shfl 放在 `jg<len` guard
之外（全 warp 参与），仅偶 lane 落 st（地址 `c2∈{0,4}` 天然 16B 对齐）。单/两文件 device 同源
（`sync_onefile_device.py` 核对 `identical: True`）。默认 0 ⇒ 旗舰路径逐位/逐字节不变。

### 80.2 实测：负结果（同 shape 交替 A/B，iters=100，event，main-only ms）

| shape | `FA_R4=0` | `FA_R4=1` | 比 |
|---|---|---|---|
| S512 H16 | 0.0548 | 0.0550 | 0.996× |
| S1024 H32 | 0.2580 | 0.2588 | 0.997× |
| S1024 H32 GQA kv4 | 0.2493 | 0.2514 | 0.992× |
| S4096 H16 | 1.5615 | 1.5694 | 0.995× |

SASS 确认 `FA_R4=1` 确实生成了 `REDG.E.ADD.F32x4`（`kvtma` kernel：base 320×`F32x2` →
R4 256×`F32x2`+64×`F32x4`）。**但 ncu 的 red 计数一字不变**：base 与 R4 都是
`l1tex requests 9,543,680`、`lts requests/sectors 114,524,160`、read 30.9M。即
**把请求从 v2 提到 v4 并未减少 L2 侧的 red 请求/扇区数**（硬件对 red 的扇区计数不随
「同 warp 内合并列」而变），所以 main 持平偏负（-0.4~-0.8%）。与 O7c 在 fp16 上的结论一致
（`docs/01` §14q），本轮把「fp8 也如此」补齐。数值 vs ref 与历史逐位一致
（S4096 `2.635/2.644/3.216e-1`），CI 不受影响（默认关）。

### 80.3 TE ncu/SASS 重测：真差距是**工作划分**，不是归约宽度

TE（`cudnn_generated_..._flash_bprop_wgmma_f8_knob_26_64x64x128_1x4x1_cga1x1x1`，
grid=132、384 线程、tile **64×64×128**，即 **BM=64 与 ours 相同**）S4096 causal：

| 指标 | TE | ours（默认 `kvtma`） | 比 |
|---|---|---|---|
| Duration | **260.7 µs** | 1560 µs | 5.98× |
| `lts__t_sectors_op_red` | **25,957,088** | 114,524,160 | **4.41×** |
| `lts__t_sectors_op_read` | 10,202,429 | 30,934,323 | 3.03× |
| `lts__t_sectors_op_write` | 789,522 | 8,401,388 | 10.6× |
| L2 总量（red+read+write） | **~36.9M** | ~153.9M | **4.17×** |
| `sm__throughput` | 45.0% | 48.2% | — |

TE 的 SASS 里 dK/dV 归约是 **`REDG.4D.ADD`**（128-bit red，与我们的 `F32x4` 同级），但红量
只有 ours 的 **4.4×↓**、L2 总量 4.2×↓。**关键**：TE 的 `BM=64` 与 ours 完全相同，却把 red 降
到 1/4 ⇒ 差距**不在「每一笔归约多宽」**（O67 已证加宽无效），而在**「每个 KV 元素被多少个
CTA 贡献」= 工作划分 / tile 调度**：TE grid 只有 **132**（1 CTA/SM、persistent），ours 默认
grid 8192（ksplit=8）。这**修正/补充**了第一百三十九轮把 red 归因为「只能靠放大 BM（撞寄存器
墙）」的结论——**同 BM 下仍有 ~4× 的 red 可压**，路径是 **dK/dV-over-KV（每个 KV 元素单一
owner）+ TMA store-reduce / persistent 调度**，而非放大 BM。O42 的 `cp.reduce.async.bulk`
失败是因为**没换工作划分**（归约次数没降、只多了 staging）。

### 80.4 结论 / 下一步

- O67：**fp8 dK/dV 归约加宽到 v4 = 负结果**（red 计数不变、main -0.4~-0.8%）。`FA_R4` 保留
  为 opt-in A/B 工具（默认 0）。
- **新定位（F7）**：默认 fp8 main 的 L2 `red` 真差距是**工作划分**——TE 在同 BM=64 下 red 仅
  1/4.4×、L2 总量 1/4.2×、时间 1/6×。下一杠杆 = **dK/dV-over-KV 单一 owner 的 persistent
  调度 + TMA store-reduce**（对标 TE 的 `REDG.4D.ADD` + 132 CTA），工程量大但方向已被 ncu 钉死。
  见 ROADMAP「阻塞」更新。

```bash
# 同 binary A/B（R4）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda [-DFA_R4=1]" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s4096_h16_d128_causal_fp8 --iters=100
# TE 侧 ncu / SASS
python3 harness/te_fp8_ncu.py "1 4096 16 128 causal"   # 配合 ncu --kernel-name regex:flash_bprop
```

原始输出 `src/fp8/fa_bwd_fp8_p147_r4_ab.out.txt`、`..._p147_ncu_red.out.txt`、
`..._p147_te_ncu.out.txt`。

## 81. F7 第一步（第一百四十八轮）：dK/dV 的「KV-owner 单一 owner」划分 smoke —— 正结果（机制）

> 落实 ROADMAP「fp8 专项冲刺」F7：O67（§80）把默认 fp8 main 的 L2 `red`（S4096 114.5M 扇区、
> 占 L2 74%）真差距钉到**工作划分**——TE 在同 **BM=64** 下 `red` 仅 ours 的 **1/4.4×**、L2 总量
> 1/4.2×、时间 1/6×（TE grid=132 persistent vs ours ksplit=8→8192）。默认 ours 是 **Q-owner**：
> 每 CTA 拥有一个 query 行块、遍历 KV 块，dK/dV 用**跨 CTA `atomicAdd`**（每个 KV 元素被
> ~mblk 个 CTA 各加一次）。F7 = 换成 **KV-owner**：每 CTA 拥有一个 KV 行块、遍历所有 query 块，
> dK/dV 在本地累加后**一次写**。本轮是 F7 的**前置 de-risk**（同 F6 第一步、O42 smoke 的做法）：
> 用一个**独立、自包含的最小 kernel**证明「换划分」在数学上正确、且能把 `red` 打到 **0**。

### 81.1 smoke 设计（`src/fp8/fa_bwd_fp8_kvowner_smoke.cu`）

- 自包含：host 造随机 Q/K/V/dO（S=512,H=8,HD=128,causal,scale=1/√HD），用 **double 版 host 参考**
  （独立算 LSE、delta、dK、dV）作基准；**fp32 标量** device kernel（隔离「划分」这一变量；
  真实 kernel 用 fp8 mma，但划分机制与 dtype 无关）。
- 两个 device kernel 对同一批输入：
  - `mowner_kernel`（现有 ours 划分）：grid=`(S/BM,H)`，每 CTA 一块 query，遍历 KV（causal 裁到
    `jmax=m0+BM-1`），逐 `(m,kv)` tile 对 dK/dV `atomicAdd` —— **red 大**。
  - `kvowner_kernel`（F7 划分）：grid=`(S/BN,H)`，每 CTA 一块 KV `[j0,j0+BN)`，从含 `j0` 的 query
    块开始遍历到 S，把 dK/dV 在 smem `dKa/dVa[BN][HD]` 里**本地累加**，循环结束后**一次 plain
    store** 写回 —— **red = 0**。

### 81.2 数值（S=512 H=8 HD=128 causal，fp32 标量）

| 实现 | dk max_abs / rel | dv max_abs / rel |
|---|---|---|
| Q-owner（atomic） vs double ref | 3.64e-05 / 9.60e-07 | 2.44e-06 / 3.76e-07 |
| **KV-owner（single store） vs double ref** | **3.64e-05 / 9.60e-07** | **2.28e-06 / 3.51e-07** |

`KV-owner vs Q-owner`：dk max_abs **5.72e-06**、dv max_abs **7.15e-07**（仅 fp32 加和次序）。
⇒ **KV-owner 划分数学正确**，与现有 Q-owner 在 fp32 舍入内一致。

### 81.3 ncu：red 归零（机制成立）

`lts__t_sectors`（同 shape，单 kernel）：

| kernel | `op_red` | `op_read` | `op_write` | `l1tex_...op_red`(请求) |
|---|---|---|---|---|
| Q-owner（atomic） | **1,671,168** | 1,504,317 | 16,252 | 278,528 |
| **KV-owner** | **0** | 44,081,332 | 200,208 | **0** |

⇒ F7 的**核心目标达成**：KV-owner 把跨 CTA 原子归约 **完全消除**（red 0、L1 red 请求 0），
dK/dV 每个元素只被唯一 owner 写一次（write 200K 扇区 ≈ 2 份输出的 coalesced 写）。

### 81.4 代价与结论（为什么 F7 主体需要 persistent + smem/TMA staging）

smoke 里 KV-owner 的 `op_read` 反而涨到 **44M**（Q-owner 仅 1.5M）、标量耗时 **0.64×**。原因：
这个**最小原型没有做 operand staging**——Q-owner 的 K/V tile 被大量并发 query CTA 复用、L2 命中
极高；KV-owner 每个 KV CTA 反复从 global 重读 Q/dO（且 `dot` 循环里 `kat(j,d)` 沿 lane 跨行、
未合并），读放大把省下的 red 又吃回去。**这正是 F7 必须「persistent 调度 + TMA/smem staging」
而非「简单翻转 grid」的原因**：

- **结论 1（正）**：KV-owner 单一 owner 划分**正确**、能把 dK/dV 的 `red` 从 O(mblk·元素) 打成 **0**
  ——机制成立，与 TE 的低 red 一致。
- **结论 2（下一步判据）**：不能只翻转 grid。F7 主体必须让每个 persistent CTA **拥有 KV 块并把
  Q/dO/K/V 经 smem/TMA staging 复用**（对标 TE grid=132 persistent + 本卡已建好的 TMA 数据通路），
  否则读放大会抵消 red 收益。**prize 仍由 O42 钉死**：短路 dK/dV 的 red ⇒ main 1.60→0.94ms
  （**1.70×**）、total 1.93→1.25ms（1.53×）。

```bash
# 运行 + ncu（red 归零）
scripts/run.sh src/fp8/fa_bwd_fp8_kvowner_smoke.cu
scripts/ncu.sh src/fp8/fa_bwd_fp8_kvowner_smoke.cu \
  --metrics lts__t_sectors_op_red.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum \
  --kernel-name regex:kvowner --launch-count 1 --   # red=0
```

原始输出 `src/fp8/fa_bwd_fp8_kvowner_smoke.out.txt`、`..._kvowner_ncu_red.out.txt`。

## 82. F7 第二步（第一百四十九届）：persistent KV-owner + smem/cp.async 暂存 —— 读放大消除（机制）

> 承接 §81 的「结论 2（下一步判据）」：KV-owner 把 `red` 打成 0，但**没做 operand staging** ⇒
> `op_read` 从 Q-owner 的 1.5M 暴涨到 **44.03M（29×）**、标量耗时 0.64×。本轮落实这一判据：
> 在**同一 smoke 文件**里新增 `kvowner_stage_kernel`，验证「**persistent CTA 拥有 KV 块 +
> Q/dO 经 smem/cp.async staging**」能把读放大消除、且 `red` 仍为 0。这是 F7 主体的最小机制原型
> （真实 fp8 主 kernel 仍走 mma/TMA，但 staging 与划分机制同构）。

### 82.1 设计（`kvowner_stage_kernel<BM=32,BN=32,HD=128>`）

与 §81 的 `kvowner_kernel` 同一划分（grid=`(S/BN,H)`，每 CTA 一块 KV、遍历 query、dK/dV 本地
累加后一次 plain store），区别在 **operand 全部 staging 到 smem**：

- **拥有的 K/V 行块只从 global 读一次**：`Ks/Vs[BN][HD]` 在 kernel 开头 vectorized `float4`
  载入后常驻（原版每个 m0 迭代都重读 K/V）。
- **Q/dO 用 `cp.async.cg` 16B 双缓冲流水搬入**：`Qs/dOs[2][BM][HD]`，prologue 发 stage0，
  每次迭代先 `issue(st^1, m0+BM)` 再 `cp.async.wait_group`（`has_next` 时 `<1>`、末轮 `<0>`）；
  越界行补 0。
- phase1（S=QKᵀ、P、dP=dO·Vᵀ、dS）与 phase2（dK=dSᵀQ、dV=PᵀdO）**全部从 smem 读**。
- 动态 smem：`4·BN·HD + 2·BM·BN + 4·BM·HD = 34,816` floats = **139,264 B**（>48KB 静态上限 ⇒
  `extern __shared__` + `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`）。

### 82.2 数值：与 KV-owner **逐位相同**（只换 operand 来源，不换数学次序）

| 实现 | dk max_abs / rel | dv max_abs / rel |
|---|---|---|
| Q-owner（atomic） vs double ref | 3.64e-05 / 9.60e-07 | 2.68e-06 / 4.13e-07 |
| KV-owner（single store） vs double ref | 3.64e-05 / 9.60e-07 | 2.28e-06 / 3.51e-07 |
| **KV-owner+stage vs double ref** | **3.64e-05 / 9.60e-07** | **2.28e-06 / 3.51e-07** |

`KV-owner+stage vs Q-owner`：dk 7.63e-06、dv 4.77e-07；`KV-owner+stage vs KV-owner` = **0**
（逐位相同，符合预期）。原始输出 `src/fp8/fa_bwd_fp8_kvowner_stage_run.out.txt`。

### 82.3 ncu：`red` 仍为 0，`op_read` 回落（读放大消除）

`lts__t_sectors` / `l1tex` 请求（同 shape，单 kernel，`-c 1`）：

| kernel | `op_red` | `op_read` | `op_write` | global ld 请求 | L1TEX% / SM% |
|---|---|---|---|---|---|
| Q-owner（atomic） | 1,671,168 | 1,502,165 | 9,862 | 26,808,320 | 32.5 / 3.95 |
| KV-owner（无 staging） | **0** | **44,066,794** | 199,907 | 26,808,320 | 20.8 / 2.55 |
| **KV-owner+stage** | **0** | **1,382,731** | 199,821 | **147,456** | 39.0 / 10.08 |

⇒ 判据全部达成：① `red = 0`（跨 CTA 原子归约消除，同 §81）；② `op_read` 从 **44.07M → 1.38M
（31.9×）**，甚至略低于 Q-owner 的 1.50M；③ global 载入请求从 26.8M → **147K（182×）**
（staging 把「逐元素 global 读」换成「每 tile 一次 16B`cp.async`」）。

### 82.4 性能（event，iters=20，同 binary A/B）

| 实现 | 时间 (ms) | vs Q-owner | vs KV-owner(无 stage) |
|---|---|---|---|
| Q-owner（atomic） | 3.46 | 1.00× | — |
| KV-owner（无 staging） | 5.41 | **0.64×** | 1.00× |
| **KV-owner+stage** | **0.876** | **3.95×** | **6.17×** |

ncu（stage kernel）：`Duration 880µs`、**32 regs**、`139.26KB` smem → **1 CTA/SM**（occ 6.25%）、
stall `short_scoreboard 4.49 + wait 1.67 + long_scoreboard 0.21`、barrier 0.01。
即 staging 后墙回到 **smem→计算的依赖（short_scoreboard）+ 低 occupancy**（标量 dot 的 smem 读），
**不再是 global 读放大，也不是原子归约**。

### 82.5 结论 / 下一步

- **F7 机制两根支柱均已 de-risk**：① 单一 owner 划分把 dK/dV 的 `red` 打成 0（§81）；
  ② persistent + smem/cp.async staging 把读放大消除（本节）。
- **F7 主体 = 把这两点落进真实 fp8 主 kernel**：persistent CTA 拥有 KV 块、Q/dO/K/V 走本卡
  已建好的 **4D-TMA** 通路暂存 smem、dK/dV 在 smem/寄存器本地累加后经 **TMA store（或
  `cp.reduce.async.bulk`）一次写出**（对标 TE grid=132 persistent）。**prize 由 O42 钉死**：
  短路 dK/dV 的 red ⇒ main **1.70×**、total **1.53×**。
- 本节的标量原型（0.876ms）只为证明机制与量级；真实 fp8 主 kernel 的绝对性能需在 F7 主体里
  用 mma/wgmma + TMA 重做（标量 `dot` 的 `short_scoreboard 4.49` 是标量特有，不代表最终形态）。

```bash
# 运行 + ncu（red=0 且 op_read 回落）
scripts/run.sh src/fp8/fa_bwd_fp8_kvowner_smoke.cu
scripts/ncu.sh src/fp8/fa_bwd_fp8_kvowner_smoke.cu -c 3 \
  --metrics lts__t_sectors_op_red.sum,lts__t_sectors_op_read.sum, \
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum --kernel-name regex:owner
```

原始输出 `src/fp8/fa_bwd_fp8_kvowner_stage_run.out.txt`、`src/fp8/fa_bwd_fp8_kvowner_stage_ncu.out.txt`。

## 83. F7 第三步（第一百五十轮）：KV-owner 落进真实 fp8 张量核（dK/dV 的 mma 原型）

> 承接 §82「结论 / 下一步」：F7 的两根机制支柱（单一 owner 消 `red` + staging 消读放大）已在
> **fp32 标量** smoke 上 de-risk，但**真实 fp8 主 kernel 走 mma/TMA**，必须把同一划分落进真实
> 数据通路才能判断它是否真的成立。本轮是 F7 主体的**第一步**：`src/fp8/fa_bwd_fp8_kvowner_mma.cu`
> 用 **E4M3/E5M2 + rowwise scale + `mma.m16n8k32`** 实现 **KV-owner 的 dK/dV**（暂不含 dQ）。
> **默认路径一行未改**（纯新增独立文件）。

### 83.1 设计（`fp8_kvowner_dkv_kernel<HD=128,BM=64,BN=32>`，128 线程）

- **划分**：grid=`(ceil(S/BN), H)`，每 CTA **拥有 KV 行块 `[j0,j0+BN)`**，沿 query 块
  `m0 = floor(j0/BM)*BM … S`（causal 裁剪）遍历；dK/dV 在**寄存器**本地累加，循环外**一次
  plain store** ⇒ 每元素仅 owner 写一次、**无跨 CTA 原子 / 无 `red`**。
- **K/V 常驻 smem**：拥有的 K/V 行块只从 global 读一次（`float4`? 4B 向量化），供 GEMM1 的
  B=`Ks`、GEMM2 的 B=`Vs`；`ks_s/vs_s` rowwise scale 也一次装好。
- **Q/dO staging**：每个 m0 把 Q/dO 行块搬进 `Qs/dOs`（行主序 ASLD）+ 配对布局 `Qp/dOp`
  （uint16，`ldmatrix.x2.trans` 读 B）。
- **4 个 GEMM**（省掉 dQ 的 GEMM4；记账与 Q-owner 主 kernel **完全一致**）：
  | GEMM | 算式 | A×B | 输出 | 处理 |
  |---|---|---|---|---|
  | 1 | `S = scale·QKᵀ` | E4M3×E4M3 | `[BM][BN]` | epilogue `P=exp(S−LSE)` |
  | 2 | `dP = dO·Vᵀ` | E5M2×E4M3 | `[BM][BN]` | epilogue `dS=P∘(dP−D)` |
  | 3 | `dV += Ap·dOᵀ` | E4M3×E5M2 | `[BN][HD]` | 乘 `sA[j]` 累进 `dVacc` 寄存器 |
  | 5 | `dK += scale·dS3·Qᵀ` | E5M2×E4M3 | `[BN][HD]` | 乘 `sds3[j]·scale` 累进 `dKacc` 寄存器 |
- **fold**：`Ap[j][m]=P[m][j]·dos[m]`（E4M3, per-j）、`dS3[j][m]=dS[m][j]·qs[m]`（E5M2, per-j），
  与 Q-owner 主 kernel **同款**（全 128 线程均衡分工 + 4-lane `shfl` 归约 amax + `foldpack4`）。
- smem = 70,144 B（K/V 常驻 + Q/dO staging + P/dS + 折叠操作数）。

### 83.2 数值：与既有 fp8 ours **逐位同量级**（只差跨 CTA 加法次序）

| shape（causal, MHA, fp8） | ours KV-owner mma dk / dv (vs ref) | 既有 ours Q-owner（已知值） |
|---|---|---|
| S=512  H16 | **2.975e-1 / 3.735e-1** | dk/dv 2.975e-1 / 3.735e-1 |
| S=4096 H16 | **2.643e-1 / 3.216e-1** | dk/dv 2.643e-1 / 3.216e-1 |

两个 shape 的 `max_abs` 与既有 Q-owner ours **完全相同**（同一 fp8 量化、同一 rowwise 折算、
同一 mma 布局），`ours(KV) vs ours(Q)` 仅 **3.9e-2/5.6e-2（S512）**、**1.50e-1/2.57e-2（S4096）**
——纯 fp32 加法次序差（Q-owner 逐 tile 原子、KV-owner 逐 query 块寄存器累加）。与 ref/TE 同量级，
无系统误差。原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_{s512,s4096}.out.txt`。

### 83.3 ncu：真实 mma 路径下 `red` 精确为 0（S4096 H16 causal，同 session）

| kernel | Duration | regs | `op_red` | `op_read` | `op_write` | occ（CTA/SM） |
|---|---|---|---|---|---|---|
| Q-owner 主 kernel（mma 路径，5 GEMM + dQ） | **1.91 ms** | 168 | **114,524,160** | 36,393,440 | — | 3 |
| **KV-owner dK/dV mma（本原型，4 GEMM）** | **1.41 ms** | 168 | **0** | 50,989,925 | 6,036,301 | 3 |

⇒ **`red` 从 114.5M 精确归零**（F7 核心判据在真实张量核数据通路上成立）；
`sm__throughput 47.3%`、achieved occupancy 17.7%（168 regs、3 CTA/SM）；
`l1tex` 共享 bank conflict 14.8M。原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_ncu_s4096.out.txt`。

### 83.4 性能（event，iters=30，同 binary / 同 session）

| shape | KV-owner dK/dV main | 同 session Q-owner mma 主 kernel（5 GEMM+dQ） |
|---|---|---|
| S=512  H16 | **0.057 ms** | — |
| S=4096 H16 | **1.41 ms**（72.9 TF @dK/dV 口径） | **1.91 ms** |

**关键读数（重要，非结论性正收益）**：KV-owner dK/dV **只做 4 个 GEMM**（无 GEMM4 dQ），
却仍要 1.41ms；Q-owner 做 **5 个 GEMM + dK/dV 的 114.5M `red`** 只要 1.91ms。按「每 GEMM 等效
时间」折算：KV `1.41/4 = 0.353 ms`、Q `1.91/5 = 0.382 ms`——KV-owner **单 GEMM 反而略快**，
但 `op_read` 从 36.4M **涨到 51.0M（1.40×）**：KV-owner 每个 KV CTA 都要把（causal 范围内的）
全部 Q/dO 重读一遍（staging 只保证「逐 tile 合并读」，不消除**跨 CTA 的 Q/dO 复用**）。
再加上本原型**无 K/V 预取、无 TMA、无 persistent**，所以把 dQ 也补上后**并不构成净收益**。

### 83.5 结论 / 下一步

- **F7 的「单一 owner 消 `red`」判据在真实 fp8 mma 上成立**（`red` 114.5M → **0**，数值逐位同
  口径）。这是 F7 主体的必要不充分条件。
- **但 `red` 归零本身买不到墙钟**：主 kernel 是 occupancy/延迟 bound（§73、§80），不是 L2 带宽
  bound；KV-owner 把 `red` 换成 **Q/dO 的跨 CTA 读放大（1.40×）** 与**寄存器墙**（dVacc+dKacc
  占 64 regs）。要让 F7 真正转正，必须同时：① **persistent + 4D-TMA staging** 消除跨 CTA 读
  放大并对齐 TE 的 grid=132；② Q/dO/K/V 的 TMA 与 mma 重叠；③ 把 dQ 也纳入同一 KV-owner 循环
  （否则 dQ 另起一趟，收益被摊薄）；④ 降 `dVacc/dKacc` 的寄存器占用以提 occupancy。
- **本原型定位**：F7 主体的「真实数据通路 de-risk」，与 §81/§82 的标量机制 de-risk 互补。
  下一步 = 把上述 ①②③④ 落进主 kernel（对标 TE `..._flash_bprop_wgmma_f8_..._64x64x128`）。

```bash
# 运行 + 对拍（读 dump case）
scripts/run.sh src/fp8/fa_bwd_fp8_kvowner_mma.cu \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8
# ncu（red=0）
scripts/ncu.sh src/fp8/fa_bwd_fp8_kvowner_mma.cu -c 1 \
  --metrics lts__t_sectors_op_red.sum,lts__t_sectors_op_read.sum \
  --kernel-name regex:kvowner -- --dir=.../b1_s4096_h16_d128_causal_fp8
```

原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_{s512,s4096,ncu_s4096}.out.txt`。

## 84. F7 第四步（第一百五十一轮）：KV-owner mma 的 Q/dO staging 加 cp.async 双缓冲重叠——形状相关

> 承接 §83「结论 / 下一步」④②（重叠 staging）：第 150 轮 KV-owner mma 原型的 Q/dO staging 是
> **同步全局读 + `__syncthreads`**，每个 query 块的全局延迟串在 GEMM1 之前。本轮给它加
> **`cp.async.cg` 16B 双缓冲流水**（prologue 发 stage0，循环里先 issue 下一块、再
> `cp.async.wait_group 1`，把全局读延迟藏到上一块 compute 后面），做「重叠收益 vs occupancy
> 损失」的同 binary A/B。**默认路径一行未改**（纯新增 `fp8_kvowner_dkv_pipe_kernel`）。

### 84.1 改动（`src/fp8/fa_bwd_fp8_kvowner_mma.cu` 新增 `fp8_kvowner_dkv_pipe_kernel`）

- Q/dO 的 staging 缓冲 `Qs/dOs/Qp/dOp` **翻倍**（2 stage）：`issue_qdo(m0,s)` 用 16B
  `cp_async16` 搬 Q/dO（行越界 `cp_async16_z(...,0)` 零填充），`cp_async_commit`；
  循环里 `issue_qdo(next, cur^1)` → `wait_group 1`（末块 `wait_group 0`）→ `__syncthreads`
  → `build_paired(cur)`（从 `Qs/dOs` 在 smem 上重建 `Qp/dOp`，同 §83 的 `__byte_perm` 交织写）
  → 5 个 GEMM。
- **踩坑**：`Qs[STAGES]` 这类**运行期下标的指针数组**会被 ptxas 推到 **local memory**，
  首版把 regs 顶到 **243**（226→243）。改成「`STG_base + s*STG` 标量偏移」后 regs **243→239**、
  A/B 从 0.875× 回升到 **0.958×**（S4096）。smem：70,144 → **105,984 B**。

### 84.2 数值：与 base **逐位相同**（仅搬运时序）

| shape（causal, MHA, fp8） | base dk/dv (vs ref) | pipe dk/dv (vs ref) | **pipe vs base** |
|---|---|---|---|
| S=512  H16 | 2.975e-1 / 3.735e-1 | 2.975e-1 / 3.735e-1 | **0 / 0** |
| S=4096 H16 | 2.643e-1 / 3.216e-1 | 2.643e-1 / 3.216e-1 | **0 / 0** |

只换搬运时序、数学口径与 §83 完全一致；与 ref/TE 同量级、无系统误差。

### 84.3 性能（event，同 binary / 同 session）：**形状相关**

| shape | base main | **pipe main** | A/B（base/pipe） |
|---|---|---|---|
| S=512  H16（grid=16×16=256） | 0.0566 ms | **0.0493 ms** | **1.148×** |
| S=4096 H16（grid=128×16=2048） | 1.4218 ms | 1.4825 ms | 0.959× |

### 84.4 ncu：机制清楚——重叠**确实**消了全局延迟，但 base 不是全局延迟 bound（S4096 H16）

| kernel | Duration | regs | CTA/SM（限） | sm% | l1tex% | warps_active% | long | short | wait |
|---|---|---|---|---|---|---|---|---|---|
| base | **1.42 ms** | 168 | 3 | 46.98 | 50.46 | 17.69 | **0.58** | 0.88 | **1.44** |
| pipe | 1.47 ms | 239 | 2（regs+smem） | 40.97 | 47.74 | 12.16 | **0.15** | 0.88 | 1.28 |

- **重叠生效的铁证**：`long_scoreboard` **0.58 → 0.15**（全局读延迟被 cp.async 藏住）；
  `lts__t_sectors_op_read` 反而 51.0M → 67.4M（cp.async 16B 粒度 + 双缓冲的额外 L2 读）。
- **但 base 的头号 stall 不是全局延迟**：`wait 1.44`（固定延迟 / mma 依赖）+ `short 0.88`
  （smem→`ldmatrix`）才是墙，`long` 只有 0.58。**重叠一个非瓶颈** ⇒ 收益为 0。
- **代价**：staging 翻倍把 smem 顶到 106KB、regs 顶到 239 ⇒ **3→2 CTA/SM**（warps 17.7%→12.2%），
  加上 op_read 变多 ⇒ 大 S 时**净 −4.1%**；小 S（grid 撑不满 3 CTA/SM 容量）时 occupancy
  不是约束，重叠直接拿下 **+14.8%**。

### 84.5 结论 / 下一步

- **F7「重叠 Q/dO staging」是形状相关的结果**：小/中 grid（S≤512）正结果 **1.148×**，
  大 S 负结果 **0.959×**。根因经 ncu 钉死：**KV-owner mma 原型的墙是 `wait`+`short_scoreboard`，
  不是全局延迟**，所以「隐藏全局延迟」只在本来就延迟受限的小 shape 有效。
- 这把 F7 的杠杆进一步收窄：要在本卡让 KV-owner 真转正，必须直接打 **`wait`（mma/smem 依赖，
  如 GEMM 间指令级交错）** 与 **occupancy**（smem ≤58KB / regs ≤128 才能 4 CTA/SM）——而
  §83 已指出这需要 persistent + 4D-TMA + 降寄存器的大改（“F7 主体”），不是单个 micro-lever。
- **本原型定位**：F7 主体的「重叠」子项判决完成（形状相关、大 S 负）。原始输出
  `src/fp8/fa_bwd_fp8_kvowner_mma_p151_{s512,s4096,sol}.out.txt`。

## 85. F7 主体第一步（第一百五十二轮）：KV-owner mma 的 **persistent 调度**——负结果（对标 TE grid=132 不成立）

> 承接 §84「结论 / 下一步」与 ROADMAP「下一步候选 ①」：F7 主体要对标 TE 的
> `..._flash_bprop_wgmma_f8_..._64x64x128`（**grid=132 persistent、1 CTA/SM**），
> 所以先单独把「persistent 调度」这一子项在 KV-owner mma 原型上判决掉。**默认路径一行未改**
> （纯新增 `fp8_kvowner_dkv_persist_kernel`；base/pipe 变体原样保留）。

### 85.1 改动（`src/fp8/fa_bwd_fp8_kvowner_mma.cu` 新增 `fp8_kvowner_dkv_persist_kernel`）

- 栅格从 `dim3(nblk, H)`（每 (KV 块, head) 一个 CTA，S4096 时 **2048 CTA**）改成
  **1D persistent**：`grid = min(nblk*H, SM数×3)`（可 `--pgrid=` 覆盖），
  `for (tile = blockIdx.x; tile < total; tile += gridDim.x)`，`tile → (h = tile/nblk,
  j0 = (tile%nblk)*BN)`（按 head 连续）。每个 CTA 用同一块 smem/寄存器依次处理多个 tile。
- 数值口径与 §83/§84 **完全一致**（每 tile 独立、每输出元素仅被其 owner 写一次）⇒
  **`persistent vs base` 逐位 = 0**（只换栅格映射）。smem 与 base 相同（70,144 B），
  168 regs → 3 CTA/SM。

### 85.2 数值：与 base **逐位相同**

| shape（causal, MHA, fp8） | base dk/dv (vs ref) | persist dk/dv (vs ref) | **persist vs base** |
|---|---|---|---|
| S=512  H16 | 2.975e-1 / 3.735e-1 | 2.975e-1 / 3.735e-1 | **0 / 0** |
| S=4096 H16 | 2.643e-1 / 3.216e-1 | 2.643e-1 / 3.216e-1 | **0 / 0** |

### 85.3 性能：**persistent 全面更慢**（S4096 H16，event，iters=50，同 binary）

| pgrid | grid 波数 | persist main | persist/base |
|---|---|---|---|
| 132（1 CTA/SM，对标 TE） | 1 | 4.543 ms | **0.313×** |
| 264（2 CTA/SM） | 1 | 2.537 ms | 0.558× |
| 396（3 CTA/SM = 现行上限） | 1 | 2.013 ms | **0.707×** |
| 792 | 2 | 1.798 ms | 0.789× |
| 1188 | 3 | 1.530 ms | 0.931× |
| 2048（= 每 CTA 仍只 1 tile，纯换 1D kernel） | 5.17 | 1.520 ms | 0.931× |
| base（2D `(nblk,H)`，2048 CTA） | 5.17 | **1.414 ms** | 1.0 |

- **单调趋势**：pgrid 越小越慢，**永不反超 base**；即便 pgrid=2048（无循环，退化成 1D 版
  base）仍 **0.93×**（1D 解码 + 循环脚手架的开销，去掉 loop 顶的冗余 `__syncthreads` 后
  0.944→0.931×，基本不变）。S512（grid=256<396，无循环）persist/base = **0.921×**。
- 结论：**KV-owner dK/dV 原型的「persistent」不是杠杆**。

### 85.4 ncu：根因 = **causal 的块间负载极不均衡**，静态 persistent 无法动态回填（S4096 H16）

| kernel | Duration | Waves/SM | Compute% | Memory% | Achieved occ | Eligible warps/sched |
|---|---|---|---|---|---|---|
| base | **1.42 ms** | 5.17 | 46.9 | 50.3 | 17.69% | 0.82 |
| persist(pgrid=396) | 2.01 ms | **1.0** | **32.5** | **36.8** | 16.86% | 0.67 |

- persistent=1 波时，`Compute/Memory` 利用率从 46.9/50.3 掉到 32.5/36.8、`Eligible` 0.82→0.67：
  **SM 有大量空闲但无 block 可补**。根因是 causal 下每 tile 的工作量 = 它拥有的 KV 块被多少
  query 消费（j0 小 ⇒ ~S/BM 个 m-step，j0 接近 S ⇒ 1 个），**块间权重差 ~64×**；静态 strided
  划分给每个 CTA 的**总权重不均**，且没有 GPU 原生调度器的「块完成即回填」来消尾。
  base 的 2048-CTA 大栅格 + 硬件调度器天然把重/轻块混填（5.17 波）⇒ 利用率反而高。
- 注意：**TE 的 grid=132 之所以行，是因为它的 tile（64×64×128）工作划分是均匀的**（无 causal
  偏斜的静态映射 + 其 persistent 调度器做均衡分配）。**直接照搬「grid=132」到我们带 causal
  偏斜的 KV-owner 划分上是有害的**。

### 85.5 结论 / 下一步

- **F7「persistent 调度」子项判决为负结果**：KV-owner dK/dV 原型上，把栅格改成 persistent
  （含对标 TE 的 grid=132）**只会更慢**（1 CTA/SM 时 **0.31×**，3 CTA/SM 时 **0.71×**），
  根因是 causal 块间负载差 ~64× + 静态划分失去硬件动态回填。⇒ F7 主体若要 persistent，
  必须配 **动态负载均衡**（work-stealing / 按权重重排 tile）**或均匀化工作划分**，
  单纯「减 CTA 数 + strided 循环」是本卡的负优化。
- 与 §84 合并看，F7 要真转正的杠杆仍收窄为：**打 `wait`（mma/smem 依赖）+ occupancy**
  （smem≤58KB/regs≤128 才能 4 CTA/SM），或**先把 causal 工作划分均匀化**（例如把
  dK/dV-over-KV 的 KV 块与「消费它的 query 数」配对均衡），而非照搬 TE 的栅格形状。
- 本原型定位：F7 主体的「persistent」子项判决完成（负结果）。原始输出
  `src/fp8/fa_bwd_fp8_kvowner_mma_p152_s512.out.txt`、
  `..._p152_s4096.out.txt`、`..._p152_pgridsweep_s4096.out.txt`、
  `..._p152_ncu_base_s4096.out.txt`、`..._p152_ncu_persist_s4096.out.txt`。

---

## 86. F7 第六步（第一百五十三轮）：KV-owner mma 的 **dynamic work-queue 调度**——正结果（恢复 base 性能，`red` 仍 0）

> 承接 §85「结论 / 下一步」与 ROADMAP「下一步候选 ①」：§85 把 KV-owner mma 原型的 persistent
> 判负，根因是 **causal 块间负载差 ~64×、静态 strided 划分失去硬件「块完成即回填」**。本轮直接
> 落实那条「必须配**动态负载均衡**」——把持久栅格的 tile 分配从 **static strided** 换成
> **dynamic work-queue**（global `atomicAdd` 领任务）。**默认路径一行未改**（纯新增 `DYN` 模板分支；
> base/pipe/static-persist 变体原样保留）。

### 86.1 改动（`src/fp8/fa_bwd_fp8_kvowner_mma.cu`）

- `fp8_kvowner_dkv_persist_kernel` 加模板参 **`bool DYN = false`** 与 `int* wq`：
  - `DYN=false`：原 static 1D strided（`tile = blockIdx.x; tile += gridDim.x`），逐位不变；
  - `DYN=true`：循环顶 `if (tid==0) s_tile = atomicAdd(wq, 1); __syncthreads(); tile = s_tile;`
    直到 `tile >= nblk*H` 退出。tile 编号 `h*nblk + jblk` ⇒ 小 `j0`（重块，query 块数最多）
    先行领取 = **LPT（longest-processing-time-first）**，天然把重块前置、轻块填尾。
- host：新增 `d_wq`（int）缓冲；`launch_dyn` 每次 launch 前 `cudaMemsetAsync(d_wq,0,4)`，
  用 `<HD,BM,BN,true>` 实例；`cudaFuncSetAttribute` 对两个实例分别设 smem 上限。
- smem/regs/几何与 base/static-persist **完全相同**（70,144 B、168 regs、3 CTA/SM）。

### 86.2 数值：与 base **逐位相同**（仅换栅格映射）

| shape（causal, MHA, fp8） | base dk/dv (vs ref) | dyn dk/dv (vs ref) | **dyn vs base** |
|---|---|---|---|
| S=512  H16 | 2.975e-1 / 3.735e-1 | 2.975e-1 / 3.735e-1 | **0 / 0** |
| S=4096 H16 | 2.643e-1 / 3.216e-1 | 2.643e-1 / 3.216e-1 | **0 / 0** |

（`dyn vs static-persist` 亦逐位 = 0；与既有 Q-owner ours 的差仍是跨 CTA 加法次序。）

### 86.3 性能：**dynamic wq 追平硬件调度的 base**（event，iters=50，同 binary / 同 session）

| 变体（S4096 H16） | grid | main(dK/dV) | 相对 base | 相对 static-persist |
|---|---|---|---|---|
| base（2D `(nblk,H)`，2048 CTA，硬件调度） | 2048 | **1.4151 ms** | 1.000× | — |
| static persistent | 396 | 1.9799 ms | **0.715×** | 1.000× |
| **dynamic work-queue** | 396 | **1.4248 ms** | **0.993×** | **1.390×** |

- S512（grid=256=total，无循环）：base 0.0577 / static 0.0622 / dyn 0.0591 ms（dyn/base 0.977×）。

### 86.4 ncu：dynamic 的机制 = **消负载不均 + 恢复 L2 时间局部性**（S4096 H16，同 binary）

| kernel | Duration | Waves | `lts__t_sectors_op_red` | `lts__t_sectors_op_read` | SM% | L1/TEX% | stall wait / short / long | occ |
|---|---|---|---|---|---|---|---|---|
| base | **1.40 ms** | 5.17 | **0** | 51.0 M | 47.1 | 50.5 | 1.44 / 0.88 / 0.58 | 17.7% |
| **dynamic wq** | **1.40 ms** | 1.0 | **0** | 53.1 M | **47.5** | 50.4 | 1.47 / 0.86 / 0.61 | 17.6% |
| static persist | 2.00 ms | 1.0 | **0** | **72.2 M** | 32.3 | 36.5 | 1.70 / 0.77 / **1.11** | 16.9% |

- `red` 三者在**真实 mma 路径下均为 0**（KV-owner 单一 owner 的核心收益不变）。
- dynamic 与 base 的利用率/停顿/时长**逐项吻合**（1.40 vs 1.40 ms、SM 47.5 vs 47.1%）。
- **新证据（修正 §85 的单一「负载不均」归因）**：static 的 `lts__t_sectors_op_read` 涨到
  **72.2 M（base/dyn 的 ~1.42×）**——不只是时长/占用率问题。static strided 的 tile 步长
  `gridDim=396` 使同一时刻 396 个 CTA 的 `tile` 撒在 **~6 个不同 head** 上（`396/64≈6.2`），
  Q/dO 工作集被打散、L2 时间局部性差、重复读取增多；dynamic 连续领号让并发的 CTA 集中在
  相邻 `h`/`jblk`，工作集收敛（53 M，≈base 的 51 M）。⇒ **§85 的负结果 = 负载不均 + L2 局部性
  双重损失，二者都被 dynamic 领号一并消掉。**

### 86.5 结论 / 下一步

- **KV-owner mma 原型的「dynamic work-queue 调度」正结果**：在**同 3 CTA/SM、同 396 CTA**
  下把 §85 的 **0.715× 恢复到 0.993×**（vs 硬件调度的 base），且 `red` 仍为 **0**。
  ⇒ 「persistent 只能更慢」的结论被修正为「**静态 persistent 更慢**」；用 dynamic 领号即可在
  persistent 栅格上复现硬件调度器的「完成即回填」+ L2 局部性。这为 **F7 主体**（持久 CTA +
  TMA/smem staging + dQ 同循环）提供了**可用的调度骨架**——持久化本身不再是障碍。
- 注意：本轮 main 仍**只做 dK/dV**（无 dQ），dynamic wq 相对 base 是**中性**（0.993×），
  并未单独带来净收益；F7 主体要真正转正，仍需叠加 **4D-TMA 暂存 / 降 `dVacc/dKacc` 寄存器
  冲 4 CTA/SM / 打 `wait`**，否则杠杆回到 §5.65。**dynamic 调度的价值 = 让后续持久 CTA
  （要复用 smem/TMA 描述符）不被静态划分拖死。**
- 原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p153_s4096.out.txt`、
  `..._p153_s512.out.txt`、`..._p153_ncu_dyn_metrics_s4096.out.txt`、
  `..._p153_ncu_static_metrics_s4096.out.txt`、`..._p153_ncu_base_metrics_s4096.out.txt`、
  `..._p153_ncu_dyn_s4096.out.txt`（`--set full`）。

### 87 F7 第七步：KV-owner mma 原型的 GEMM1/2 换 Hopper `wgmma`（负→正结果，第一百五十四轮）

**动机**：§86（第 153 轮）把 KV-owner mma 原型的 dynamic work-queue 调度调到与硬件调度 base
持平（0.993×），但 main 仍只做 dK/dV 且**未转正**。ncu 钉死 base 的第一墙是
`wait 1.44 + short_scoreboard 0.88`（**mma / `ldmatrix` 依赖**），而 `long_scoreboard` 仅 0.58
⇒ **不是全局访存延迟**。F6 第二步（Q-owner）已证明「GEMM1/2 换 wgmma」能把两条独立 mma 的
等待重叠掉；本步把同款改造搬到 KV-owner 原型，检验它在**不牺牲 occupancy** 的前提下能否打 `wait`。

**改动**（`src/fp8/fa_bwd_fp8_kvowner_mma.cu` 新增 `fp8_kvowner_dkv_wgmma_kernel`，`-DFA_WGMMA` 构建）：

- Q/dO/K/V 的 smem 全部改存 **SW128 K-major**（fp8 一行 128B = 一个反交织 atom 的整行，
  `sw128_off_fp8`），不再存行主序 ASLD；`Qp/dOp` 的 K 配对布局由同一份 `q0/q1` 用
  `__byte_perm` 直接写出（不变）。
- GEMM1 `S=scale·QKᵀ`(e4m3×e4m3) 与 GEMM2 `dP=dO·Vᵀ`(e5m2×e4m3) 用 **`wgmma.m64n32k32`**
  （1 warpgroup=128 线程；BM=64、BN=32）**直读 smem 描述符**，两条异步 mma 一起发、统一
  `wgmma.wait_group 0`；epilogue 改用 wgmma 累加器映射（warp `w` 持行 `[16w,16w+16)`，
  `sacc[j*4+q]` ↔ `row=16w+g+(q>=2?8:0)`、`col=j*8+2*(lane%4)+(q&1)`），LSE/D 预装 2 个行槽。
- GEMM3(dV)/GEMM5(dK) **仍 `mma.m16n8k32 + ldmatrix`**（fp8 wgmma 无转置操作数、MN-major
  描述符无效，O9c-2 / 「阻塞」已三证判死），B 仍用 O4b 的 K 配对布局。
- **收益账**：SW128 tile 比 ASLD 行主序更紧凑（Ks/Vs 4096 vs 4608、Qs/dOs 8192 vs 9216），
  总 smem **67072B < base 70144B** ⇒ 仍 **3 CTA/SM**——这是与 §84 的 pipe 变体（103.5KB → 2
  CTA/SM）的关键区别：**在不掉 occupancy 的前提下拿到异步 mma**。

**数值**（ours-vs-fp32 ref，单 case）：S512 dk/dv `2.976e-1 / 3.732e-1`、S1024H32
`4.176e-1 / 3.535e-1`、S4096 `2.644e-1 / 3.216e-1`——与 base 同量级（base 为 2.975e-1/3.735e-1、
2.643e-1/3.216e-1），**无系统误差**。`wgmma vs base` 差 3.1e-2（S512）/ 4.1e-2（S1024H32）/
1.50e-1（S4096），与既有 `ours(Q-owner atomic) vs base` 同值 ⇒ 仅是 **fp8 GEMM1/2 累加次序**
（wgmma 与 mma）经 `exp` 放大后的 fp8 噪声，与「换 KV-owner 划分」同源，**非 bug**。

**性能**（CUDA event，同 binary A/B，iters=50，main 仅 dK/dV）：

| case | base | pipe | dynamic wq | **wgmma** | wg/base | wg/pipe | wg/dyn |
|---|---|---|---|---|---|---|---|
| S512 H16 | 0.0570 ms | 0.0492 | 0.0589 | **0.0571 ms** | 0.998× | 0.861× | 1.031× |
| S1024 H32 | 0.2407 ms | 0.2514 | 0.2411 | **0.2398 ms** | **1.004×** | 1.049× | 1.005× |
| S4096 H16 | 1.4128 ms | 1.4428 | 1.4074 | **1.3472 ms** | **1.049×** | 1.071× | 1.045× |

- 大 S 正收益（**S4096 1.049×**，且优于 §84 pipe 的 0.979×、§86 dynamic 的 1.004×），中等 S
  中性（1.004×），小 S 略负（0.998×，grid=256 单波、本就 grid-bound）。
- **ncu（S4096 H16）**：wgmma vs base —— Duration **1.42→1.38ms**、`Executed Instructions`
  **638.7M → 541.0M（−15.3%）**（mma+ldmatrix 换成 wgmma 的直接证据）、Compute **47.25→41.01%**、
  L1/TEX 56.77→56.51%、L2 12.78→**17.23%**、regs 168 / smem 65.5KB / **3 CTA/SM** 不变、
  achieved occ 17.7%。
  **stall（per-issue-active）**：`wait 1.44→1.52`、`short 0.88→0.96`、`long 0.58→1.40`、
  `not_selected 0.55→0.36`、`barrier 0.22→0.28`。
  ⇒ **`wait` 并未下降**（wgmma 的 `wait0` 本身仍等，且指令数变少后全局延迟相对更暴露，
  `long` 反升）；**净收益来自指令数 −15.3%**（去掉 `ldmatrix` 与 SM80 兼容 path 的多余发射），
  不是消除 `wait`。这与 F6 第二步（Q-owner）的「换 wgmma 但墙在别处 ⇒ 中性」不同：
  KV-owner 原型本就 issue/指令偏重，减指令能直接缩短关键路径。

**结论 / 下一步**：
- **正结果（大 S）**：KV-owner dK/dV 原型的 GEMM1/2 上 wgmma（配 SW128、**保持 3 CTA/SM**）
  把 S4096 main 1.049×（1.413→1.347ms）、指令 −15.3%，是 §84/§86 之后**第一个在大 S 转正的
  KV-owner 结构改动**；也优于 pipe（0.979×）与 dynamic（1.004×）。
- 但它仍是 **dK/dV-only** 原型（FLOPs 只算 2/3），且净收益仅 ~5%——**F7 主体要真正对标 TE
  （grid=132、1 CTA/SM、wgmma+TMA）还需**：① 把 Q/K/V/dO 的 `cp.async`/标量 staging 换成
  **4D-TMA**（对标 TE）；② 加 **dQ 同循环**（dQ 的跨 CTA 归约须 `cp.reduce.async.bulk` 或
  partial+reduce）；③ 降 `dVacc/dKacc` + `dqacc` 寄存器冲 **4 CTA/SM**。本轮把①③之外的
  「GEMM 指令层」先对齐 TE（wgmma 已进 KV-owner）。
- 原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p154_s512_h16_d128_causal_fp8.out.txt`、
  `..._p154_s1024_h32_d128_causal_fp8.out.txt`、`..._p154_s4096_h16_d128_causal_fp8.out.txt`、
  `..._p154_ncu_wgmma_s4096.out.txt`、`..._p154_ncu_base_s4096.out.txt`、
  `..._p154_stall_{base,wgmma}_s4096.out.txt`。

### 88 F7 第八步：KV-owner wgmma 原型的 Q/dO 4D-TMA staging（正结果）＋修正一处 p154 遗留的 async-proxy race（第一百五十五轮）

**动机**：§87 把 GEMM1/2 上 wgmma 后，KV-owner 原型的 Q/dO staging 仍是「逐 ROW-pair 的标量
global gather + `__byte_perm` 写 SW128」——每个 query 块都要标量读 Q/dO（ncu 相对暴露的
`long_scoreboard`）。F7 主体「对标 TE 的 TMA 数据通路」的第一子项就是把它换成 **4D-TMA**：fp8
一行 128B = 一个 SW128 atom 的整行，一条 `cp.async.bulk.tensor.4d` 即可搬完整块（无需 fp16 的
2×K=64 chunk 拆分，与主 kernel O37/O32 同构）。

**改动**（`src/fp8/fa_bwd_fp8_kvowner_mma.cu`，plant 路径一行未改）：

- 把原 `fp8_kvowner_dkv_wgmma_kernel` 抽成 device body
  **`fp8_kvowner_dkv_wgmma_body<HD,BM,BN,bool TMA>`**，两个 `__global__` 薄壳共用：原
  `..._wgmma_kernel`（`TMA=false`，逐字退化 = §87）与新 `..._wgmma_tma_kernel`（`TMA=true`，
  `const __grid_constant__ CUtensorMap qmap,dmap`）。避免复制 ~280 行。
- TMA 版 per m 块由 tid0 发两条 4D-TMA 把 Q/dO 搬进 SW128 tile（`make_kvowner_qd_map`：
  UINT8 `dims={D,S,H,B}`、`box={128,BM=64,1,1}`、SWIZZLE_128B），`mbarrier` 相位 `qph` 每迭代
  翻转；再从 SW128 用 `__byte_perm` **重建 `Qp/dOp` 的 K 配对布局**（与主 kernel O37 逐字同构）。
  其余（K/V 常驻、GEMM1/2 wgmma、GEMM3/5 mma、fold、本地累加、single-owner plain store）逐字不变。
  smem `67072 + 64B`（qbar/dbar）⇒ 仍 **3 CTA/SM**。
- **顺带修一处 p154 遗留 race**：wgmma 经 **async proxy** 读 smem，而 K/V（及非 TMA 路径的
  Q/dO）是 **generic 写**，p154 缺 `fence.proxy.async.shared::cta` ⇒ 偶发 nondeterminism
  （本轮实测同一 binary 8 次：`wgmma vs base` 在 `4.08e-2 / 6.4e-2 / 2.2e-1` 间跳；已在
  `git show HEAD` 的 p154 原文件上复现，确认非本步引入）。在 K/V staging 与（每迭代）Q/dO
  staging 后补 `bulk_reduce_fence()`（= `fence.proxy.async.shared::cta`）后 **8/8 次稳定**。

**数值**（ours-vs-fp32 ref，单 case）：S512 `2.976 / 3.732e-1`、S1024H32 `4.176 / 3.535e-1`、
S4096 `2.644 / 3.216e-1`——与 wgmma 版同量级、无系统误差。**`wgmma+TMA vs wgmma = 0.0000e+00
（逐位）**（S512/S1024H32/S4096 各多次；仅搬运通路）。修复 fence 后 `wgmma vs base` 也稳定：
S512 `3.058e-2`、S1024H32 `4.079e-2`、S4096 `1.500e-1`（= fp8 GEMM1/2 累加次序噪声，与
`ours(Q-owner) vs base` 同源）。

**性能**（CUDA event，同 binary A/B，iters=50，main 仅 dK/dV）：

| case | base | wgmma(§87) | **+Q/dO-TMA** | tma/wg | tma/base |
|---|---|---|---|---|---|
| S512 H16 | 0.0563 ms | 0.0577 | **0.0462 ms** | **1.247×** | **1.224×** |
| S1024 H32 | 0.2406 | 0.2404 | **0.1940** | **1.239×** | **1.241×** |
| S4096 H16 | 1.3863 | 1.3747 | **1.1183** | **1.229×** | **1.267×** |

- **三个 shape 一致 1.23–1.27×**，且是 §84/§86/§87 之后**最大的 KV-owner 结构收益**（§87 wgmma
  只在大 S 1.049×，§84/§86 中性/负）。原因是把「标量 global gather + SW128 逐 pair 写」换成
  单条 TMA，去掉的既是指令也是全局延迟；小 S（单波、grid-bound）同样受益。

**ncu（S4096 H16，main only）**：

| 指标 | wgmma(§87) | **+TMA** |
|---|---|---|
| Duration | 1.38 ms | **1.12 ms** |
| Executed Instructions | 541.5 M | **470.1 M（−13.2%）** |
| L1/TEX | 56.45% | 66.25% |
| Compute (SM) | 41.35% | 44.41% |
| L2 | 17.20% | 18.63% |
| DRAM | 1.85% | 2.27% |
| regs / smem | 168 / 67.07 KB | 168 / 67.14 KB（**3 CTA/SM**） |
| occ / Waves | 17.6% / 5.17 | 17.7% / 5.17 |
| `lts__t_sectors_op_read` | 58.86 M | **52.18 M（−11.4%）** |
| `lts__t_sectors_op_write` | 15.89 M | 12.46 M |
| `lts__t_sectors_op_red` | **0** | **0** |
| stall `long_scoreboard` | 1.40 | **0.43** |
| stall `wait` | 1.52 | **1.53（未降）** |
| stall `short_scoreboard` | 0.95 | 1.31 |
| stall `barrier` / `not_selected` | 0.28 / 0.36 | 0.32 / 0.42 |

⇒ **TMA 的收益 = 去全局 gather**（`long_scoreboard 1.40→0.43`、指令 −13.2%、L2 读扇区 −11%），
墙随即回到 **`wait`（wgmma/立即数延迟，未降）+ L1/TEX + `short_scoreboard`**——与 §87「收益来自
减指令、不是消 `wait`」完全一致，也正是 F7 主体下一步（降寄存器冲 4 CTA/SM / 打 `wait`）要啃的。

**对标**：本原型仍只做 dK/dV（FLOPs 2/3），非生产路径；同 session TE FP8 纯反向（全反向）S4096
= **0.3003 ms / 915.25 TF**（`docs/03` §71 口径）。本步把 F7 主体的「TMA staging」子项
**de-risk 并转正**。

**结论 / 下一步**：F7 主体剩余 = ① **dQ 同循环**（dQ 跨 CTA 归约须 `cp.reduce.async.bulk` 或
partial+reduce）；② **降 `dVacc/dKacc` + `dqacc` 寄存器冲 4 CTA/SM**；③ K/V 已 resident，
无需再 TMA。本步完成后 KV-owner 原型的 Q/dO 已是「wgmma + TMA」的 TE 同款数据通路。

- 原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p155_tma_{s512_h16_d128_causal_fp8,
  s1024_h32_d128_causal_fp8,s4096_h16_d128_causal_fp8}.out.txt`、
  `..._p155_nontma_{...}.out.txt`、`..._p155_ncu_{tma,wgmma}_s4096.out.txt`、
  `..._p155_stall_{tma,wgmma}_s4096.out.txt`。

### 89 F7 第九步（第一百五十六轮）：F7 主体的「降寄存器冲 4 CTA/SM + 微调」子项判决 —— 负结果（p155 定为局部最优）

**动机**：§88 后 F7 主体只剩两件大事：① **dQ 同循环**（跨 CTA 归约须 `cp.reduce.async.bulk` 或
partial+reduce）；② **降 `dVacc/dKacc` 寄存器冲 4 CTA/SM（smem≤58KB / regs≤128）**。①是大改，
②被列为「小步」。本轮先**把②以及几个便宜的微调旋钮一次性判决**，避免在 F7 主体大改前留盲点。

**改动**（`src/fp8/fa_bwd_fp8_kvowner_mma.cu`，仅加两个编译期旋钮，默认值 = p155 逐字行为）：

- **`FA_KV_CTA`**（默认 3）：wgmma 两壳的 `__launch_bounds__(THREADS, FA_KV_CTA)`，用于 CTA/SM 扫参。
- **`FA_KV_OVL`**（默认 0）：把 TMA 路径的 **Qp/dOp 重建**（SW128 → K 配对布局）抽成
  `build_paired_tma` lambda；=1 时挪到 wgmma GEMM1/2 **之后、`wait0` 之前**，试图让这段纯 smem
  搬运/交织与**异步 wgmma 的延迟重叠**（原顺序是「重建 → sync → wgmma → wait0」，重建串在 wgmma 前）。

**实验一：消 store bank conflict（机制验证，源码级 no-op）**。p155 ncu：`l1tex__data_bank_conflicts
_pipe_lsu_mem_shared_op_st = 13.7M`（store 波前的 37.8%，ncu Est. 25%）。先怀疑是 Qp/dOp 重建的
两次 4B 存（8B 步长 ⇒ 2-way），改成一次 8B `uint2` 存后：**`sm__inst_executed` 逐位不变
470,056,960**、store conflict 也不变 13.73M ⇒ **nvcc 早把两次相邻 `ST.32` 合成 `ST.64`**，
源级改动是 no-op（故已回退）。真正的 store 冲突源是 **Ps/Ss epilogue 的 `Ps[r*PSS+c]`**：8 行
（`g`）× 4 列（`sub`，步长 2）在 `PSS=37`（≡5 mod 32）下 8 个基址 `{0,5,10,15,20,25,30,3}`
各展开 `{0,2,4,6}` ⇒ 6 个 bank 二用；但**每行只用 4 个偶 bank、8 行需 32 个偶 bank 而全 warp 只有
16 个偶 bank**，数学上**至少 2-way**（ncu 实测 2.5-way 已近最优）。且 **store 不产生 stall**
（同为 dependency-bound 的 `wait`/`short`），故这条无杠杆。

**实验二：CTA/SM 扫参（`FA_KV_CTA` = 1/2/3/4，同 session，main 仅 dK/dV，iters=100）**：

| case | cta1 (237r/0spill) | cta2 (254r) | **cta3 (168r，默认)** | cta4 (128r，仍被 67KB smem 卡 3) |
|---|---|---|---|---|
| S512 H16 | **0.0449 ms** | 0.0450 | 0.0465 | 0.0498 |
| S1024 H32 | 0.2198 | 0.2188 | **0.1918** | 0.2054 |
| S4096 H16 | 1.2689 | 1.2683 | **1.1113** | 1.1803 |

⇒ **3 CTA/SM（168 regs）是最优点**：2 CTA/SM（更多寄存器、0 spill）在大 S 慢 ~14%（warps 12→8）、
1 CTA/SM 慢 ~14%、4 CTA/SM（强制 128 regs）慢 ~6%。**「降寄存器冲 4 CTA/SM」在本卡不成立**：
168→128 要砍 40 regs，而两个跨 m-loop 常驻的 `dVacc/dKacc`（各 `[1][8][4]`=32）、`sacc/dpacc`
（各 16）已占 96；除非把 dK/dV 拆成两个 kernel（读翻倍）或把累加器降 fp16（改数值口径）——
均不划算。且即便 regs 降到 128，smem 67KB 仍把 Block Limit Shared Mem 卡在 3（要到 4 须 ≤58KB，
需再去 Qp/dOp 的 17.4KB = F6 已判不可行的路）。**⇒ 4 CTA/SM 双硬墙（regs + smem）均未松动。**

**实验三：Qp/dOp 重建与 wgmma 重叠（`FA_KV_OVL` 0/1）**：bitwise `wgmma+TMA vs wgmma = 0/0`
（仅改重排），但性能 **S512/S1024 慢 ~1%、S4096 中性**。ptxas 报 **`C7517`：在 OVL=1 下注入了
额外 `wgmma.wait_group`**——编译器保守地把重建排在 wgmma 之后（防 GMMA 寄存器依赖），**重叠被
串行化**，故无收益（与 O22/O29/O46「`wait` 不是靠指令重排能解的」一致）。

**ncu（S4096，最终默认档）**：Duration 1.12ms、`Executed Instructions` 470.06M、`red` **0**、
`lts read` 52.2M / `write` 12.4M、L1/TEX 66.3%、Compute 44.4%、occ 17.7%（**3 CTA/SM**）、
stall **`wait 1.53` + `short 1.31` + `long 0.43`**、load conflict 5.51M（占 load 波前 5.7%，
`PSLD=136` 的 `ldmatrix.x2.trans` 16 行 stride 68 word ≡ 4 ⇒ 2-way，但量小）、store conflict
13.7M（不 stall）。

**结论**：**p155 的 KV-owner wgmma+TMA 原型是「微调空间已尽」的局部最优**——消 store 冲突无杠杆
（no-op / 数学下界 / 不 stall）、CTA/SM 3 为最优、指令重排被 ptxas 串行化、4 CTA/SM 被 regs+smem
双墙锁死。剩余唯一真杠杆 = **F7 主体本身（dQ 同循环 + GEMM3/5 的等待方式）**：GEMM3/5 上不了
fp8 wgmma（F6/「阻塞」三证），唯一路是把 **dQ 的跨 CTA 归约改成 `cp.reduce.async.bulk` / partial+
reduce** 并把三梯度收进同一循环以摊薄 `wait`——工程量大，转入「阻塞/下一步」。

- 原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p156_tma_{s512_h16_d128_causal_fp8,
  s1024_h32_d128_causal_fp8,s4096_h16_d128_causal_fp8}.out.txt`、
  `..._p156_ncu_tma_s4096.out.txt`、`..._p156_ctasweep_s4096.out.txt`、`..._p156_ovlsweep.out.txt`。

### 90 F7 第十步（第一百五十七轮）：KV-owner 原型补 **dQ 同循环**（三梯度全通 + atomic 归约判决）

**动机**：§89 判决 p155 的 KV-owner wgmma+TMA 原型（**仅 dK/dV**）微调空间已尽，F7 主体唯一
真杠杆 = **dQ 同循环**——`lts__t_sectors_op_red` 能不能真正降下来，取决于 dQ 的跨 CTA 归约方式。
本轮先把 **dQ 落进同一循环**，让 KV-owner 成为**首个三梯度全通的 fp8 反向原型**，并给出
「dQ 走 atomic」的定量判决（这正是 ROADMAP「候选①流量中性」预言的实证）。

**改动**（`src/fp8/fa_bwd_fp8_kvowner_mma.cu`，`fp8_kvowner_dkv_wgmma_body`，加编译期开关
`FA_KV_DQ` 默认 1；=0 退回 p155 逐字行为）：

- **A=`dS2`**：fold 阶段新增 `dS2[m][j]=dS[m][j]·ks[j]`（e5m2，per-m scale `sds2[m]`），与
  Q-owner 主 kernel 的 `dS2` fold **逐字同款**（4 warp × 16 行 × 2 lane，`foldpack4` + 16B 向量写）。
- **B=`Kp`**：K 的配对布局 `Kp[rp][d]`（uint16 = `{K[2rp][d], K[2rp+1][d]}`）从 SW128 `Ks` 重建，
  **整个 m-loop 只建一次**（K 常驻），供 `ldmatrix.x2.trans` 读出 `[N=d][K=j]` 片段。
- **GEMM4** `dQ += scale·(dS2·K)`：`mma_block_bt<GM5=32, GN5=64, BN, E5E4>(dS2, DSS2, Kp, PSLD, …)`
  输出 `[BM][HD]`，4 warp 各 32×64；epilogue 乘 `sds2[m]·scale` 后用 **`red_add2`（跨 CTA 原子）**
  归约进 global `dq`。
- smem：`+Kp(4352B) + dS2(3072B) + sds2(256B)` ⇒ wgmma smem 67072→**74752B**（仍 **3 CTA/SM**，
  上限 77482B）。

**数值（三梯度全通）**：`dq` vs fp32 ref，S512 **2.4262e-1** / S1024H32 **2.3993e-1** / S4096
**2.6355e-1**——与同 kernel 的 `dk/dv`（2.64/3.22e-1）**同量级 fp8 噪声**，无系统误差；
`wgmma+TMA vs wgmma` 的 dk/dv **逐位 0**（仅搬运通路）。⇒ **KV-owner 三梯度数学正确。**

**性能（CUDA event，main，iters=50~100，同 binary A/B）**：

| case | dK/dV-only（`FA_KV_DQ=0`，p155） | **三梯度**（`FA_KV_DQ=1`） | dQ 增量 | 比值 |
|---|---|---|---|---|
| S512 H16 | 0.0462 ms | **0.0750 ms** | +0.0288 | 1.62× |
| S1024 H32 | 0.1940 ms | **0.3044 ms** | +0.1104 | 1.57× |
| S4096 H16 | 1.1167 ms | **1.8318 ms** | +0.7151 | 1.64× |

三梯度口径 TFLOPS（`flops=2·S²·H·HD·causal·3`）：S512 21.48 TF / S1024H32 42.32 / S4096 56.27
（峰值 1978.8 的 1.09%/2.14%/2.84%）。**dQ 一步让 main 涨 ~1.6×**。

**ncu（S4096，`..._wgmma_tma`）**：Duration **1.84 ms**、`lts__t_sectors_op_red` **102,236,160**、
`l1tex…_op_red` 68.16M、`read` 52.6M / `write` 6.2M、L2 63.86%、DRAM 2.45%、occ 17.8%
（3 CTA/SM，168 regs）、Waves 5.17、stall `wait 1.64 + short 1.52 + long 0.89`。

**判决（对照 ROADMAP「候选①流量中性」）**：默认 fp8 `kvtma` main（三梯度）ncu 为 `red` **114.5M**
（O67）/ L2 76.96% / ~1.92ms（O41）。本原型把 dK/dV 的 red 打成 **0**，却因 dQ 走 atomic 新增
**102.2M** red ⇒ **总 red 只从 114.5M → 102.2M（−10.7%）**，L2 76.96%→63.86%、duration
1.92→1.84ms（−4%）。**数量级吻合「dQ red ≈ dK/dV red」的解析式**：dK/dV 的贡献数
`2×S²/(2·BM)=S²/BM`，dQ 的贡献数 `S²/(2·BN)`，`BM=64=2·BN` ⇒ **两者相等**。
⇒ **把原子从 dK/dV 搬到 dQ 是「流量中性」的，实证坐实 ROADMAP「候选①关闭」**；**单趟 KV-owner
（无论 atomic 还是等价的 bulk-reduce，字节数不变）不能解锁 F7 的 prize（dK/dV-only 的 1.12ms）**。

**结论 / 下一步**：F7 主体真杠杆只剩 **让 dQ 也「owned」（red=0）**，只有两条路：① **两 kernel**——
KV-owner 出 dK/dV（red=0）＋ **Q-owner 的 dQ-only pass**（dQ 在寄存器 owned、red=0，代价是 S/dS
重算一次，但算力仅 ~4% 峰值，L2 省 ~74% 值得）；② **两级 partial**——需 BN≥BM 或二次归约
（本原型 BN=32<BM=64 时 dQ red 恒 ≈ dK/dV）。**转入「下一步候选 ①」**。

- 原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p157_dq_{s512_h16_d128_causal_fp8,
  s1024_h32_d128_causal_fp8,s4096_h16_d128_causal_fp8}.out.txt`、`..._p157_dq0_s4096.out.txt`（A/B）、
  `..._p157_dq_ncu_s4096.out.txt`。

### 91 F7 第十一步（第一百五十八轮）：option(a)「两 kernel」——Q-owner **dQ-only pass** 的判决 —— **负结果**

**动机（ROADMAP「下一步候选 ①(a)」）**：§90 证明「把原子从 dK/dV 搬到 dQ」流量中性，F7 主体唯一
真杠杆 = **让 dQ 也 owned（red=0）**，二选一：**(a) 两 kernel**（KV-owner 出 dK/dV[red=0] ＋
Q-owner 出 dQ[red=0]），**(b) 两级 partial**（需 BN≥BM）。本轮把 (a) 落地并判决。

**改动（`src/fp8/fa_bwd_fp8_kernels.cuh`，一行默认路径未改）**：给 `fp8_mma_body` 尾部加编译期
`bool DQONLY = false`，并透传到两个壳（`fa_bwd_fp8_mma_kernel` / `fa_bwd_fp8_mma_kvtma_kernel`）：
- `DQONLY=true` 时 **跳过 Ap/dS3 fold + GEMM3/4（dV/dK）+ 其 epilogue/bulk-reduce**，保留
  GEMM1/2（`S=QKᵀ`、`dP=dO·Vᵀ`）、dS2 fold、GEMM5（`dQ=dS·K`）；
- `kRegDq` 下 dQ 寄存器 owned，**`DQONLY` 时用 plain store 写出**（而非 `red_add2`）⇒ L2 `red=0`。
`DQONLY=false` 逐字退化（CI fp8 全绿、数值与历史逐位一致）。

原型 `src/fp8/fa_bwd_fp8_kvowner_mma.cu` 新增：同一 binary 内
① `launch_dqonly`（`kvtma`, `DQONLY=1`, Q-owner 栅格 `(S/BM,H,B)`, `ksplit=1`）；
② `launch_main3`（生产默认 `kvtma`, `DQONLY=0`, 三梯度, 含 dK/dV 的跨 CTA `red`）；
③ KV-owner dK/dV-only 走 p155 的 `launch_wgtma`（编译 `-DFA_KV_DQ=0`）。

**数值（三 shape）**：`dQ-only` vs fp32 ref **S512 2.4262e-1 / S1024H32 2.3993e-1 / S4096 2.6355e-1**
——与历史 dQ 逐位一致（`dQ-only vs 三梯度基线 dq = 0.0000e+00`）；`main3` 的 dq/dk/dv 也与历史
逐位一致（S512 2.4262/2.9757/3.7318e-1、S1024H32 2.3993/4.1764/3.5346e-1、S4096
2.6355/2.6437/3.2157e-1）。⇒ **DQONLY 未动默认数值，dQ-only 数学正确。**

**性能（CUDA event，main，iters=100，同 binary A/B，ms）**：

| case | KV-owner dK/dV (TMA) | Q-owner **dQ-only** | 两 kernel 合计 | 单 kernel 三梯度 | 两/单 |
|---|---|---|---|---|---|
| S512 H16 | 0.0467 | 0.0539 | **0.1006** | 0.0948 | **0.944×** |
| S1024 H32 | 0.1903 | 0.1582 | **0.3485** | 0.3303 | **0.955×** |
| S4096 H16 | 1.1206 | 0.9230 | **2.0435** | 1.9425 | **0.955×** |

⇒ **两 kernel 反而慢 4.5–6%（负结果）**。

**ncu（S4096 H16，同 binary，`--metrics`）**：

| 口径 | Duration | `lts op_red` | `lts read` | `lts write` | L2% | DRAM% | SM% |
|---|---|---|---|---|---|---|---|
| dqonly（Q-owner, DQONLY=1） | **933.7µs** | **0** | 18.18M | 2.44M | 6.79 | 1.73 | 36.98 |
| main3（kvtma 三梯度，默认） | **1.99ms** | **103.8M** | 24.22M | 6.66M | 54.95 | 3.30 | 36.64 |
| wgtma（KV-owner dK/dV, TMA） | **1.12ms** | **0** | 52.30M | 12.43M | 18.77 | 2.27 | 43.94 |

**判决**：两 kernel 把跨 CTA `red` 从 **103.8M 扇区彻底打成 0**，但**总时间不降反升**（S4096
2.04ms > 1.94ms）。机制：单 kernel 里 GEMM1/2（`S`/`dP`）与 fold 出的 `P/dS` 被 dK/dV 与 dQ
**共享**；拆两 kernel 后 **dQ-only pass 必须重算一遍 S/dP 并重新流式读 Q/K/V/dO**（ncu：dQ-only
`read 18.2M`、KV-owner `read 52.3M`，合计 70.5M vs 单 kernel 24.2M，**L2 读放大 2.9×**），
省下的 `red` 打不过「重算 + 重复 operand 流量」。⇒ **ROADMAP「候选 ①(a)」判负**；
F7 主体只剩 **(b) 两级 partial / 单趟内非原子归约（BN≥BM）**，或放弃「拆 kernel」。

- 原始输出 `src/fp8/fa_bwd_fp8_p158_b1_{s512_h16,s1024_h32,s4096_h16}_d128_causal_fp8.out.txt`、
  `src/fp8/fa_bwd_fp8_p158_ncu_dqonly_s4096.out.txt`（full）、
  `src/fp8/fa_bwd_fp8_p158_ncu_metrics_s4096.out.txt`（三口径 metrics）。

## 92. F7 第十二步（第一百五十九轮）：用 **TMA 4D tensor store-reduce**（`cp.reduce.async.bulk.tensor.4d`，对标 TE 的 `UTMAREDG.4D.ADD`）替掉 dQ 的逐 lane `red.global.add` —— **机制正结果（指令数归零）+ 性能负结果（red 扇区一字不变、1.34× 慢）**

> 动机：第一百四十七轮（§80）用 ncu 重测 TE fp8 反向时，发现 TE 的全局归约只有 **3168** 条
> `smsp__inst_executed_op_global_red`（ours 9.54M 条 L1 red 请求）、且 **0 条** plain global
> store；本轮把 TE 的 SASS 拉出来，直方图里是 **`UTMAREDG.4D.ADD`**（TMA 4D 张量归约）+
> `UTMALDG.4D`（TMA 载入）+ `UTMASTG.4D`（TMA 存）+ `USETMAXREG`（setmaxnreg，warp-spec）。
> 于是做出**此前从未试过的最小判决**：把 KV-owner 原型的 dQ 跨 CTA 归约从逐 lane
> `red_add2`（8B `red.global.add.v2.f32`）换成 **一条 4D TMA tensor store-reduce**（硬件把
> 整块 `[BM][HD]` fp32 原子加回 global），验证「O42 的 bulk-reduce 思路」在**换成 4D tensor
> 变体 + 已换工作划分（KV-owner）**后是否真能吃到 O42 短路 red 的 prize（main 1.70×）。

### 92.1 改动（`src/fp8/fa_bwd_fp8_kvowner_mma.cu`，默认路径一行未改）

- `fa_bwd_fp8_kernels.cuh` 新增 `tma_reduce_add_4d_f32(map, smem_src, c0..c3)`——PTX 用
  **`cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group`**（类型由 tensormap
  的 element type 推断、指令**不带** `.f32`；对照 CUDA `cuda/__ptx` 生成头）。
- 原型加编译期开关 **`FA_KV_DQ_TMAR`**（默认 0）：=1 时 GEMM4 的 dQ 结果先写 smem 行主序
  staging（`dQst`），再由 `tid==0` 发 tensor reduce；=0 退回 p157 的 `red_add2`（逐位基线）。
  新增 `fp8_kvowner_dkv_wgmma_tma_r_kernel` 薄壳（多传一个 **fp32 的 dQ tensormap**：
  `make_kvowner_dq_map_f32`，dims={D,S,H,B}、box={64,BM}、SWIZZLE_NONE）+ 同 binary 的
  `launch_wgtmar` / `--only=wgtmar` / 计时与对拍。

### 92.2 两个关键坑（都实测踩过）

1. **`FA_FP8_HAS_TMA` 在本文件里晚于 helper 才 `#define`**（宏按出现顺序展开）⇒ 我一开始用它
   守卫新 helper 的 asm，导致 `#if FA_FP8_HAS_TMA` 恒为 0、**asm 被整段编掉**：SASS 里没有
   `UTMAREDG`、dQ 完全错（max_abs 2.96 vs 0.24）却**不报任何错**。改用
   `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900` 直接守卫后 SASS 才出现归约指令。
   判据：**新增 device asm helper 先 `cuobjdump -sass | grep 指令名` 确认被发射**。
2. **tensor-reduce 的 box 内维 ≤ 256B**（fp32 即 64 列）⇒ 128 列的 dQ tile 必须沿 D 拆
   **2 个 64 列 chunk**（每 chunk 在 smem 紧凑存成 `[BM][64]`、行距 64），各发一条 reduce。
   另 PTX 记法必须是 `.add.tile.bulk_group`（红操作在前、`.tile` 后接完成机制），写成
   `.bulk_group.add.f32` 会 `ptxas: Unexpected instruction types specified`。

### 92.3 数值（三 shape，单 binary A/B，全对拍通过）

`dQ 4D-TMA-reduce vs atomic(red_add2)`：S512/S1024H32/S4096 **dq max_abs = 1.1921e-07**
（仅 fp32 归约加序）；`4D-TMA-reduce vs wgmma+TMA` 的 **dk/dv max_abs = 0**（只改归约通路）；
`dq vs fp32 ref` S4096 = **2.6355e-1**（与 atomic 版逐位同量级）。三梯度全通、无系统误差。

### 92.4 性能（同 binary A/B，main 三梯度，event；ms）

| shape | wgmma+TMA（atomic `red_add2`） | **dQ 4D-TMA-reduce** | tmar/wgtma |
|---|---|---|---|
| S512 H16 | 0.0753 | 0.0820–0.0852 | **0.89–0.91×** |
| S1024 H32 | ~0.307 | 0.3926 | **0.78×** |
| S4096 H16 | ~1.856 | 2.3796 | **0.78×** |

### 92.5 ncu（S4096 H16，同 session A/B）——**判决性证据**

| 指标 | wgmma+TMA（atomic） | **dQ 4D-TMA-reduce** | 说明 |
|---|---|---|---|
| Duration | 1.84 ms | 2.47 ms | 慢 1.34× |
| `l1tex …op_red` 请求 | 8,519,680 | **0** | 逐 lane 原子**彻底消失** |
| `smsp__inst_executed_op_global_red` | 8,519,680 | **0** | 同上（指令层归零） |
| **`lts__t_sectors_op_red`** | **102,236,160** | **102,236,160** | **一字不变！** |
| `lts read / write` | 52.7M / 6.20M | 52.1M / 11.07M | write 反涨（reduce 写扇区） |
| `smsp__inst_executed` | 703.65M | 660.41M | −43M（省掉 8.5M 条 red） |
| dynamic smem | 74.82 KB（3 CTA/SM） | 107.58 KB（**2 CTA/SM**） | 32KB fp32 staging 的代价 |
| achieved occupancy | 17.79% | 12.19% | 掉一档 |

**结论（决定性）**：把归约从「逐 lane `red.global.add`」换成「TMA 4D tensor store-reduce」
后，**L1 层的 red 请求/指令归零，但 L2 的 `red` 扇区数一模一样（102,236,160）**、时间还因
3→2 CTA/SM 变慢 1.34×。⇒ **L2 `red` 流量由「每个输出元素被多少个 CTA 贡献」= 工作划分决定，
与归约指令机制无关**（O67「加宽归约无效」的同源结论再进一层）。TE 的 `UTMAREDG` 只是它的
**指令选择**，其 `red` 仅 25.96M（ours 1/3.9×）**来自它的 tile 调度/工作划分**（grid=132
persistent、384 线程、232KB smem、1 CTA/SM），**不是**用了 TMA 归约本身。

### 92.6 下一步

- **F7 的「TMA store-reduce」假设就此关闭**（机制已验证正确、但非杠杆）；F7 主体只剩
  **工作划分本身**（减少每个元素的贡献数）：① **BN≥BM**（KV-owner 的 dQ 贡献数 `S/BN`，
  BN 32→64 砍半；代价是寄存器/occupancy，见「阻塞」）；② **放大 tile / 持久化 tile 调度**
  （对标 TE 132 CTA，需先解决本卡 smem/寄存器硬墙）。③ 放弃 F7、转「换卡」。
- 原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p159_tmar_s4096.out.txt`、
  `..._p159_tmar_s512_h16_d128_causal_fp8.out.txt`、`..._p159_tmar_s1024_h32_d128_causal_fp8.out.txt`、
  `..._p159_default_regression_s512.out.txt`（默认档回归逐位）、
  `src/fp8/fa_bwd_fp8_p159_ncu_wgtmar_s4096.out.txt`、`..._p159_ncu_wgtma_s4096.out.txt`。

## 93. F7 第十三步（第一百六十轮）：BN≥BM（BN=64）——dK/dV-over-KV 的 dQ 贡献数减半，但撞寄存器墙（中性/偏负）

### 93.1 动机（落实第一百五十九轮「下一步候选 ①(a)」）

F7 的「KV-owner 单一 owner」划分把 dK/dV 的跨 CTA `red` 打成 **0**，但代价是 **dQ 变跨 CTA 原子**
（q157）：一个 dQ 元素 `(m,d)` 收到所有 `j<=m` 的 KV 块的贡献，贡献数 ≈ `S/BN`。q157 实测
KV-owner 三梯度的总 `red` **114.5M→102.2M（−10.7%）**——因为解析上 dK/dV 贡献 `S²/BM`、dQ 贡献
`S²/(2BN)`，`BM=64=2·BN` 时两者相等（「把原子从 dK/dV 搬到 dQ 流量中性」）。

第一百五十九轮据此给出唯一剩余的工作划分杠杆：**BN≥BM**。若把 KV-owner 的 tile 从 `BN=32`
放大到 `BN=64`（=BM），则：

- dQ 的贡献数 `S/BN` 从 `S/32` **减半到 `S/64`** ⇒ dQ `red` 减半；
- Q/dO 的跨 CTA **读放大**（每个 Q tile 被 `S/BN` 个 KV-owner CTA 重读）也减半；
- 代价：`Ks/Vs/Ap/dS3/Kp/dS2/Ps/Ss` 全随 BN 线性增长 ⇒ **smem 74.75→111.42KB**（3→2 CTA/SM）；
  `dVacc/dKacc`（`MTM34=BN/32` 从 1 变 2）与 GEMM1/2 的 wgmma 累加器（n32→n64）**翻倍**。

本轮把该选项**完整落进 p155/p157 的 `fp8_kvowner_dkv_wgmma_body`**（而非纸面估算），做同 binary
A/B + ncu，给出判决。

### 93.2 实现（`src/fp8/fa_bwd_fp8_kvowner_mma.cu`，device + host，默认路径一行未改）

- **body 参数化 BN∈{32,64}**：删 `static_assert(BN==32)` → 允许 64；GEMM1/2 新增
  `wgmma_mn_issue<BN,KIND>`（BN=32 走 `m64n32k32`（16 累加器）、BN=64 走 `m64n64k32`（32 累加器）；
  两者累加器映射同构，epilogue 的 j 上界由 `4` 改 `BN/8`）。其余（fold/GEMM3/4/5/epilogue/Kp
  重建/plain store）**逐字沿用**——`MTM34/NTM34/NTFOLD` 全部由 `BN` 派生。
- **壳**：`fp8_kvowner_dkv_wgmma_tma_kernel<HD,BM,BN>` 的 `__launch_bounds__` 改为
  `(BN==64?FA_KV_CTA64:FA_KV_CTA)`（`FA_KV_CTA64` 默认 2）。BN=32 实例逐字不变。
- **host**：新增 `Cfg64`/`BM64`/`BN64`、`wg_tma_smem64`（精确 111,424B）、BN=64 的
  `cudaFuncSetAttribute`/`launch_wgtma64`（grid.x 减半 `S/64`）/`--only=wgtma64`，以及对拍与计时。

### 93.3 数值（三 shape 全通，与 ref/TE 同量级）

| shape | 实现的 dk / dv / dq vs fp32 ref | BN=64 vs BN=32 |
|---|---|---|
| S512 H16 | 2.9757e-1 / 3.7318e-1 / 2.4262e-1 | dk/dv **0.0**、dq 5.9272e-2 |
| S1024 H32 | 4.1764e-1 / 3.5346e-1 / 2.3993e-1 | dk/dv **0.0**、dq 7.1432e-2 |
| S4096 H16 | 2.6355e-1 / 3.2161e-1 / 2.6355e-1 | dk/dv **0.0**、dq 5.9015e-2 |

⇒ **换工作划分不改变数学口径**：dK/dV（本地 owned、无跨 CTA 加序）**逐位相同**；dQ（跨 CTA
原子）只差贡献数变少导致的 fp32 加法次序，量级 ~5.9e-2 与 fp8 噪声一致。

### 93.4 性能（同 binary A/B，CUDA event，iters=30）

| shape | BN=32（wgmma+Q/dO-TMA 三梯度） | **BN=64** | 比值 | smem |
|---|---|---|---|---|
| S512 H16（grid 16→8 <132 SM，**grid-bound**） | 0.0748 ms | 0.0831 ms | **0.900×** | 111.4KB |
| S1024 H32 | 0.3349 ms | 0.3214 ms | **0.960×** | 111.4KB |
| S4096 H16 | 1.8483 ms | 1.7731 ms | **1.042×** | 111.4KB |

- S512：BN=64 的 grid 只有 `8×16=128 < 132 SM` ⇒ **连一个波都铺不满**（BN=32 是 256 CTA），
  净慢 10%。
- S1024/S4096：grid 足够，BN=64 把 `red` 与读放大减半，但只换来 **+4%（S4096，另一 session
  +1.5%）/ −4%（S1024）**——**基本中性**。

### 93.5 ncu（S4096 H16，同 session A/B）——**寄存器墙是判决**

| 指标 | BN=32 | **BN=64** | 说明 |
|---|---|---|---|
| **`lts__t_sectors_op_red`** | **102,236,160** | **51,118,080** | **精确减半**（工作划分生效） |
| `lts read` | 52.62M | 40.06M | −24%（Q/dO 重读减少） |
| `lts write` | 6.19M | **26.64M** | **+4.3×（溢出！）** |
| **`l1tex …mem_local_op_st`** | **65,536** | **13,754,368** | **+210× ⇒ `dVacc/dKacc` 翻倍把寄存器顶穿** |
| `l1tex …mem_local_op_ld` | 1.06M | 13.72M | 同上 |
| registers/thread | 168 | **255（上限）** | 2 CTA 上限 256，仍不够 |
| achieved occupancy | 17.86% | 11.85% | 3→2 CTA/SM |
| `smsp__inst_executed` | 703.65M | 539.34M | −23%（n64 + 少一半 CTA） |
| Duration | 1.83 ms | 1.79 ms | 中性 |

**结论（决定性）**：**BN≥BM 的算术假设（red 减半、读放大减半）完全成立**（`lts op_red`
102.2M→51.1M 一字不差地减半），但 **dQ/dK/dV 的「单一 owner」本质要求把整块 KV 的 `dK/dV`
累加器常驻寄存器**（BN=64 ⇒ `dVacc+dKacc = 2×64 = 128` 个 fp32），叠加 GEMM1/2 从 n32→n64 的
累加器翻倍，**顶穿 255 寄存器硬上限 → 每线程 112B 栈 + 13.75M 扇区的 local store 流量**，
其代价盖过了省下的 `red`/读。⇒ **F7 option(a)（BN≥BM）在本卡判为中性/偏负**：
`red` 的减半被**寄存器溢出**吃掉，与「阻塞」里 O17b/F6 的「寄存器文件锁死放大 tile」是**同一堵墙**。

### 93.6 下一步

- **F7 的「BN≥BM」子项就此判决（中性/偏负）**。到此 F7 主体已穷尽本卡可行的机制：
  single-owner（red=0）机制成立、但**任何「单趟内减少贡献数」的实现（两 kernel / TMA
  store-reduce / BN≥BM）都被 (i) L2 读放大、(ii) 寄存器文件 255 上限 之一挡住**。
  剩余唯一未试的 = **跨 warpgroup 偏和 + 二次归约**（需先把 dK/dV 累加器搬出寄存器，
  同样撞 smem/regs），或 **放弃 F7 / 换卡**。
- 原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p160_bn64_{s512,s1024h32,s4096}.out.txt`、
  `src/fp8/fa_bwd_fp8_p160_ncu_{bn32,bn64}_s4096.out.txt`。

---

## 94. O68（第一百六十二轮）：非 causal（full）D=128 的 LSE 接上均衡 `cp.async` 版 —— 正结果，默认

### 94.1 动机（发现一处被遗漏的路径）

F1（第 132 轮）把 fp8 **定长**默认切到 Hopper（wgmma+TMA），F5（第 136 轮）又把 causal 的
TMA LSE 做了「tile 内两趟 softmax」。但 **非 causal（full）D=128 的 LSE 一直走 O1 的
`lse_mma_kernel`**——那是全仓库最早的 LSE：每 CTA 64 行 × 64 列、**逐元素 `LDG.U8→STS` 载入
Q/K（无 `cp.async` 双缓冲）**、无 K 维 split。对照代码可见：

- 定长 causal：`launch_lse_bal_tma` / `launch_lse_bal_wgmma`（TMA/wgmma + `cp.async` 双缓冲）；
- 定长 full：`launch_lse<128>`（O1）——**没有跟上 O8b/O11/O54 的均衡/流水改造**；
- varlen full D=128：同样 `launch_lse<128>`（O1）。

而 O54（第 101 轮）其实早就给 `lse_mma_kernel_bal` 加了 **`FULL=true`** 模式（一个 CTA 一个
m 块 + `cp.async` 双缓冲），只是**只接到了 D=512（MLA）的 full 路径**，D=128 的定长/varlen
full 都漏接了。O68 = 把这两条 D=128 full 的 LSE 也切到 `lse_mma_kernel_bal<128,1,FULL=true>`。

### 94.2 实现（纯 host：`src/fp8/fa_bwd_fp8_main.cu` + `fa_bwd_fp8_mma_onefile.cu`，device 一行未改）

- 新增文件作用域开关 `g_lse_full_opt`（默认 **1**）+ CLI `--lsefull=0/1`（`--lsefull=0` 退回 O1
  做同 binary A/B）。定长与 varlen 两条 full 路径共用该开关。
- **定长 full D=128**（`run_preprocess` 的 `else` 分支）：`launch_lse<128>` 改为
  `launch_lse_bal<128, 1, true>(lg, …, nullptr, nullptr, 1)`（`lg=dim3(nblk,H,B)`，恰好是
  FULL 模式期望的「一个 CTA 一个 m 块」网格；ksplit=1，不需 partial/merge）。
- **varlen full D=128**（`run_varlen`）：同上，带 `d_cu`。
- 未改 device：kernel `lse_mma_kernel_bal<HD,PIPE,FULL=true>` 早已存在且经 D=512 full 验证。

### 94.3 数值（换 LSE 实现 → LSE 的 fp32 求和次序变化，噪声量级）

- **ours vs fp32 ref**（S1024 H16 full）：`dq/dk/dv = 5.520e-2 / 5.312e-2 / 4.024e-2`（fp8 噪声）。
- **lsefull=0 vs =1**（同输入落盘逐元素）：`dq 2.02e-3 / dk 1.15e-3 / dv 1.68e-3`——正是 O1 的
  「逐元素 online-softmax」与均衡版「tile 内两趟 softmax」的 fp32 求和次序差异，**远小于 fp8
  的 O(1e-1) 容差**（对照 `docs/03` §71 F5：同一类差异 ~1e-5–1e-3）。
- `--ci`（fp8）：单/两文件一致性 worst **7.153e-06** OK、`--check docs/04` OK。

### 94.4 性能（同 binary A/B，CUDA event）

| 路径 | lsefull=0（O1 LSE） | **lsefull=1（均衡 FULL）** | 比 |
|---|---|---|---|
| 定长 full S1024 H16 — **preprocess** | 0.1466 ms | **0.0397 ms** | **3.70×** |
| 定长 full S1024 H16 — total | 0.5388 ms / 15.94 TF | **0.3740 ms / 22.96 TF** | **1.44×** |
| varlen full b4_t4096 H16 — total | 1.5589 ms / 22.04 TF | **1.2293 ms / 27.95 TF** | **1.27×** |

单文件版逐指标一致（total 0.3739 vs 两文件 0.3745ms）。
**对标**（`harness/fa_bwd_bench.py bench --dtype fp8 --shape '1 1024 16 128 full'`）：
TE FP8 纯反向 **0.0562ms / 305.8 TF** ⇒ ours/ TE 时间比 **9.6× → 6.65×**；峰值占比
**0.81% → 1.16%**（fp8 峰值 1978.8 TF）。

### 94.5 ncu（定长 full S1024 H16，`regex:lse_mma_kernel`，`--launch-count 1`）——bound

| 指标 | lsefull=0（O1） | **lsefull=1（均衡 FULL）** |
|---|---|---|
| **Duration** | **208.4 µs** | **42.3 µs（4.93×）** |
| DRAM Throughput | 0.62% | 3.08% |
| L1/TEX Throughput | 9.43% | 22.87% |
| L2 Throughput | 2.59% | 15.33% |
| Compute (SM) | 23.36% | 45.14% |
| **Warp Cycles / Issued** | **8.11** | **3.86** |
| Registers / thread | 82 | 88 |
| Waves Per SM | 0.39 | 0.39 |
| Achieved Occupancy | 12.04% | 11.89% |

**结论**：O1 LSE 是**纯延迟 bound**（`Warp Cycles/Issued 8.11`、L1/TEX 仅 9.4%——每个 K tile 都
同步等 global→smem，再串行算），而均衡 FULL 版用 `cp.async` 16B 双缓冲把载入与 wgmma 重叠，
`Warp Cycles/Issued` 砍半、Duration **4.93×**。两条路径都 `Waves 0.39`（S1024 的 grid=256 只有
~0.4 个波），所以真实墙是**网格不足 + 串行载入延迟**；`cp.async` 直接打掉后者。
**这不是 main 的 L2 `red` 墙，而是 preprocess 内部的一条被漏改的 LSE 分支**。

### 94.6 复现 / 原始输出

```bash
# 两文件 A/B（Hopper 构建）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --full --lsefull=0 \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h16_d128_full_fp8
# ncu：regex:lse_mma_kernel --launch-count 1
```

原始输出：`src/fp8/fa_bwd_fp8_o68_ab_fixed_full_s1024.out.txt`、
`src/fp8/fa_bwd_fp8_o68_ab_onefile_full_s1024.out.txt`、
`src/fp8/fa_bwd_fp8_o68_ab_varlen_full_b4t4096.out.txt`、
`src/fp8/fa_bwd_fp8_o68_ncu_lse_full_s1024.out.txt`、
`src/fp8/fa_bwd_fp8_o68_te_full_baseline.out.txt`。

### 94.7 下一步

- **剩余 full 路径**：LSE 的 **TMA 化**（把 `lse_mma_kernel_bal<FULL>` 再换成 4D-TMA 版，对齐
  causal 的 O32）——当前 FULL 版是 `cp.async`，TMA 可再省 load 指令。
- 其余候选（F6/F7/放大 BM 等）均已判决/收口，见「阻塞」。

---

## 95. O70（第一百六十四轮）：非 causal（full）D=128 的 LSE 再上 **4D-TMA**（对齐 causal O32）—— 正结果，默认

### 95.1 动机

O68（§94）把定长 full D=128 的 LSE 从 O1 接到「均衡 + `cp.async`」版，preprocess 3.70×。但
**causal 的 LSE 早就是 4D-TMA 版**（O32，`lse_mma_kernel_bal_tma`：一条 `cp.async.bulk.tensor.4d`
搬整块 SW128，省掉逐 16B `cp.async` 的 load 指令/地址运算），full 用的仍是 `cp.async` 的
`lse_mma_kernel_bal`。本步把 full 也切到 **同一套 TMA 搬运**，把三 dtype full D=128 LSE 收敛的
第一步（fp8）做完（fp16/bf16 留下一轮，见 §95.7）。

### 95.2 实现（device + host，单/两文件同步）

- **device**（`src/fp8/fa_bwd_fp8_kernels.cuh` + `fa_bwd_fp8_mma_onefile.cu`，
  `lse_mma_kernel_bal_tma`）：模板加 `bool FULL = false`。
  - `for (t<2)` 里 `if constexpr (FULL) { if (t == 1) continue; }`，`mblk = FULL ? pair : …`
    ——full 各 m 块工作量相同，**用 grid.x=nblk 一个 CTA 一个 m 块**，不做镜像配对。
  - `ncols = FULL ? S : min(S, m0 + LBM)`。
  - 四个掩码条件 `jg <= qi` 改为 `(FULL || jg <= qi)`（full 不做因果掩码）。
  - 其余（TMA 双缓冲、rowwise scale、`tile 内两趟 softmax`、4-lane `shfl` 归约）逐字复用。
    `FULL=false` 编译出与 O32/O38 **逐位相同**的 causal 代码。
- **host**（`fa_bwd_fp8_main.cu` + onefile）：`launch_lse_bal_tma` 模板加 `bool FULL=false`；
  full 分支（`else if (g_lse_full_opt)`）当 `lse_tma` 为真时改调
  `launch_lse_bal_tma<128,1,true>(lg, …)`（`lg=dim3(nblk,H,B)` 正是 FULL 期望的网格，ksplit=1）；
  默认 `lse_tma = (D == 128) ? 1 : 0`（此前 `(D==128 && causal)`）。
  - A/B：`--lsetma=0` 退回 O68 的 `cp.async` 均衡版；`--lsefull=0` 仍退回 O1（无关 `lse_tma`）。
    `sm_90`（非 TMA）构建下 `lse_tma=0`，自动退回 O68，行为不变。

### 95.3 数值

- **ours vs fp32 ref**（定长 full S1024 H16）：`dq/dk/dv = 5.518e-2 / 5.310e-2 / 4.025e-2`；与 O68
  的 `cp.async` 版（5.520e-2 / 5.312e-2 / 4.024e-2）**仅差 LSE 的 fp32 求和次序**（~1e-5 量级），
  远小于 fp8 的 O(1e-1) 容差。vs TE：1.148e-1 / 9.946e-2 / 1.068e-1（同量级）。
- **单/两文件一致性**（`--hopper --consistency`，`fa_bwd_run.py`）：worst **1.192e-07**（fp8 容差
  1e-4）**OK**。`--check docs/04` **OK**（194 行，rtol 0.005 内，无需改表）。
- **causal 回归**：S512 H16 causal 仍 `lse backend=tma`，dq/dk/dv = 2.426/2.972/3.733e-1，与历史
  2.426/2.975/3.735e-1 一致（只差 atomic 次序/求和次序）；`sm_90` 默认构建 full 走 O68 逐位不变。

### 95.4 性能（同 binary A/B，Hopper 构建，CUDA event）

| 路径（定长 full S1024 H16） | O1（`--lsefull=0`） | cp.async（`--lsetma=0`，O68） | **TMA（默认，O70）** | TMA vs cp.async |
|---|---|---|---|---|
| **preprocess** | 0.1498 ms | 0.0401 ms | **0.0226 ms** | **1.77×** |
| total | 0.5399 ms / 15.91 TF | 0.3742 ms / 22.95 TF | **0.3569 ms / 24.07 TF** | **1.049×** |
| main | 0.3009 ms | 0.3005 ms | 0.3002 ms | 1.00×（不变） |

单文件版逐指标一致（total 0.3561 vs 两文件 0.3569ms；preprocess 0.0224 vs 0.0226ms）。
**对标**（O68 同 session TE FP8 纯反向 0.0562ms / 305.8 TF）：ours/TE 时间比 **6.65× → 6.35×**，
峰值占比 1.16% → **1.22%**（fp8 峰值 1978.8 TF）。收益全在 preprocess（main 不变）。

### 95.5 ncu（定长 full S1024 H16，`regex:lse_mma_kernel`，`--launch-count 1`）—— bound

| 指标 | O1（`--lsefull=0`，§94） | cp.async（`--lsetma=0`） | **TMA（默认，O70）** |
|---|---|---|---|
| **Duration** | **208.4 µs** | **42.46 µs** | **23.94 µs（vs cp.async 1.77×）** |
| DRAM Throughput | 0.62% | 3.07% | 5.43% |
| L1/TEX Throughput | 9.43% | 21.10% | 17.75% |
| Compute (SM) | 23.36% | 45.35% | 43.14% |
| `smsp__inst_executed` | — | 18,394,624 | **9,769,216（−46.9%）** |

**结论**：O1 是纯延迟 bound（串行 global→smem），均衡 `cp.async` 打掉它（4.93×），**4D-TMA 再用
一条 bulk 指令搬整块 SW128 把 load 指令数砍半（−46.9%）、Duration 再 1.77×**。两条流水版都仍是
`Compute ~43–45%` + 网格不足一个波（`Waves 0.39`），不是 DRAM/L2 bound。**这是 preprocess 内部
§94 同一分支的搬运方式升级，不是 main 的 L2 `red` 墙。**

### 95.6 复现 / 原始输出

```bash
# 两文件 A/B（Hopper 构建）：TMA(默认) / cp.async(--lsetma=0) / O1(--lsefull=0)
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --full \
  --dir=/home/xieminglin/proj/output/fa-bwd/b1_s1024_h16_d128_full_fp8
# 一致性 gate（单/两文件）
python3 harness/fa_bwd_run.py --dtype fp8 --glob b1_s1024_h16_d128_full_fp8 --hopper --consistency
# ncu：regex:lse_mma_kernel --launch-count 1
```

原始输出：`src/fp8/fa_bwd_fp8_o70_ab_fixed_full_s1024.out.txt`、
`src/fp8/fa_bwd_fp8_o70_onefile_full_tma_s1024.out.txt`、
`src/fp8/fa_bwd_fp8_o70_ncu_lse_full_s1024.out.txt`、
`src/fp8/fa_bwd_fp8_o70_consistency_s1024.out.txt`。

### 95.7 下一步

- **fp16/bf16 的同类改造**：把 `lse_mma_kernel_bal_tma`（fp16/bf16 版）也加 `bool FULL`，host
  两条 D=128 full 从 O69 的 `cp.async` 均衡版切到 4D-TMA（fp16 的 TMA LSE 是 **2× K=64 chunk**，
  bf16 逐字 dtype 化）——把三 dtype full D=128 LSE 统一到 TMA。
- 其余候选（F6/F7/放大 BM 等）均已判决/收口，见「阻塞」。

---

## 96. O72（第一百六十六轮）：varlen full D=128 的 LSE 也上 **4D-TMA**（补齐 F9→O70 链漏掉的一条分支）—— 正结果，默认

### 96.1 动机

F9（O68，§94）/O70（§95）把**定长** full D=128 的 LSE 从 O1 → 均衡 `cp.async` → 4D-TMA，
F11（O71）把它泛化到 fp16/bf16，ROADMAP 记为「三 dtype full D=128 LSE 统一到 TMA」。
但**这条统一漏了 varlen**：`run_varlen` 里 full D=128 仍硬编码走 O54 的
`launch_lse_bal<128,1,true>`（`cp.async` 双缓冲，ksplit=1），源码注释也自认
「仅定长（无 cu_seqlens）；varlen full 仍走 `lse_mma_kernel_bal`」
（`fa_bwd_fp8_kernels.cuh` O70 注释）。O40 早已把 varlen 的 LSE 接上 K 维 split，但那是
`cp.async` 分支；**4D-TMA 只服务定长**。本步把 varlen full D=128 也切到同一套 TMA 搬运
（fp8 先行，对齐 F9→F10→F11 的「fp8 先、再泛化」惯例）。

### 96.2 实现（device + host，单/两文件同步）

- **device**（`src/fp8/fa_bwd_fp8_kernels.cuh` + onefile，`lse_mma_kernel_bal_tma`）：模板加
  `const int* __restrict__ cu_seqlens = nullptr`。`cu` 非空时
  `qbase = cu_seqlens[b]`、`len = cu_seqlens[b+1]-qbase`、`nblk = ceil(len/LBM)`，否则逐式退化为
  `qbase=b*S / len=S`（**定长路径逐位不变**）。具体：
  - `if constexpr (FULL) { if (pair >= nblk) return; }`——短序列多余的对 CTA 直接退出
    （定长 `grid.x==nblk` ⇒ 恒不触发）。
  - Q/K 的 rowwise scale 索引由 `b*S + off` 改 `qbase + off`，越界判据 `off < S` → `off < len`。
  - TMA 行坐标 `m0/j0` → `qbase + m0 / qbase + j0`；packed 布局的 **batch 维恒 0**（描述符建在
    `dims={D,T,Hkv,1}` 上，基址由 `qbase` 给）。
  - `ncols`、四个 `jg < S` 掩码、输出写 `lse[(qbase+qi)*H+h]` 同步换成 `len/qbase`。
- **host**（`fa_bwd_fp8_main.cu` + onefile）：`launch_lse_bal_tma` 加 `const int* cu = nullptr`
  （定长调用不传 ⇒ 逐位不变）。`run_varlen` 在 LSE 段前（`#if FA_WGMMA && FA_TMA`）为
  D=128/full 建 packed 描述符 `qmap_v=make_lse_map_fp8(d_q8,H,T,D,1)`、
  `kmap_v=make_lse_map_fp8(d_k8,Hkv,T,D,1)`；full D=128 分支优先
  `launch_lse_bal_tma<128,1,true>(lg, qmap_v, kmap_v, …, d_cu)`。
  - A/B：`--lsetmavarlen=0` 退回 O68 的 `cp.async` 均衡版（同 binary）；`--lsefull=0` 仍退回 O1。
- **构建**：`harness/fa_bwd_run.py` 的 varlen 构建从 `-DFA_WGMMA` 扩到
  `-DFA_WGMMA -DFA_TMA -lcuda`（**仅 fp8**，`varlen_tma = is_varlen and dtype=="fp8" and not --mma`）。
  varlen 主 kernel 仍走 `launch_bwd_main`（无 TMA 模板参数）⇒ **只换 LSE 的搬运方式**，主 kernel
  不受影响。fp16/bf16 varlen 维持旧构建（其 varlen full LSE TMA 化尚未做）。

### 96.3 数值

- **vs fp32 ref（同 binary，逐位）**：`b4_t4096_full` TMA `dq/dk/dv = 8.881e-2 / 7.043e-2 /
  5.721e-2`，与 `--lsetmavarlen=0` 的 `cp.async` 版**打印逐位相同**；`b4_t3840_full`（长度不齐
  512/1024/2048/256）TMA `1.011e-1 / 9.656e-2 / 7.032e-2`，cp.async `1.010e-1 / 9.651e-2 /
  7.036e-2`（仅 LSE 的 fp32 求和次序，~1e-4，远小于 fp8 的 O(1e-1) 容差）。**不等长序列
  （短序列被 `pair>=nblk` 早退）对拍同样通过**，证明 packed 定界/掩码正确。
- **单/两文件一致性**（`fa_bwd_run.py --varlen-only --dtype fp8`，13 个 fp8 varlen case）：
  `ours vs ours_sf` worst **3.815e-06**（fp8 容差 1e-4）**OK**。
- **全量 CI**（`--no-run --ci`，73 case）：fp16 gate 3.906e-3 / bf16 7.812e-3 / fp8 7.153e-6
  全 **OK**，`--check docs/04` **OK**（194 行）。
- **固定长度回归**：`b1_s1024_h16_d128_full_fp8` 仍 `dq/dk/dv = 5.518e-2 / 5.310e-2 / 4.025e-2`
  （与 O70 §95.3 逐位一致）；causal/MLA 路径未动。

### 96.4 性能（同 binary A/B，Hopper 构建，CUDA event，iters=50）

| case（varlen full fp8 D=128 H16） | cp.async（`--lsetmavarlen=0`，O68） | **TMA（默认，O72）** | 加速 |
|---|---|---|---|
| `b4_t4096`（等长 1024×4）两文件 | 1.1810 ms / 29.09 TF | **1.1190 ms / 30.71 TF** | **1.055×** |
| `b4_t4096` 单文件 | 1.1841 ms / 29.02 TF | **1.1221 ms / 30.62 TF** | **1.055×** |
| `b4_t3840`（不齐 512/1024/2048/256）两文件 | 1.4724 ms / 30.99 TF | **1.4054 ms / 32.47 TF** | **1.048×** |
| `b4_t3840` 单文件 | 1.4806 ms / 30.82 TF | **1.4061 ms / 32.46 TF** | **1.053×** |

（`TFLOPS` 用 host 打印的 `sum_b 4HL²D` 口径；峰值占比 = `30.71/1978.8 ≈ 1.55%`——这是**整个
反向**的端到端口径，不只是 LSE。）
**对标**：同 session `harness/fa_bwd_bench.py bench --dtype fp8 --lengths 1024×4 --full` 的 TE FP8
纯反向 **0.1842 ms / 186.56 TF** ⇒ ours/TE 时间比 **6.39× → 6.10×**（收益全在 preprocess）。

### 96.5 ncu（b4_t4096 full，同 binary，`regex:lse_mma_kernel_bal`，`--launch-count 1`）—— bound

| 指标 | cp.async（`--lsetmavarlen=0`） | **TMA（默认，O72）** |
|---|---|---|
| **Duration** | **118.34 µs** | **63.01 µs（1.88×）** |
| `Smsp__inst_executed`（Executed Instructions） | 73,578,496 | **39,168,000（−46.8%）** |
| Registers / thread | 88 | **58** |
| Dynamic Shared / block | 28.42 KB | 26.43 KB |
| Achieved Occupancy | 25.84% | **35.93%** |
| Waves Per SM | 1.55 | **0.97** |
| DRAM / L2 / L1-TEX Throughput | 4.42% / 22.80% / 32.19% | 8.30% / 22.61% / 28.98% |
| Compute (SM) / Issue Slots Busy | 65.33% | 64.74% |
| Issued Ipc Active | 2.76 | 2.79 |

**结论**：varlen full D=128 的 LSE 与定长同源——`cp.async` 版是 **issue-bound**（逐 16B 的 load
指令 + 地址运算，`Compute≈Issue 65.3%`）。4D-TMA 用**一条 `cp.async.bulk.tensor.4d` 搬整块
SW128**，把 load/地址指令砍掉 **−46.8%**、寄存器 88→58、occupancy 25.8%→35.9%，**Duration
118.3→63.0 µs（1.88×）**。两条路径都**不是 DRAM/L2 bound**（DRAM 4–8%、L2 22%）——这是
preprocess 内一条被漏改的 LSE 分支的搬运升级，**与 main 的 L2 `red` 墙无关**。

### 96.6 复现 / 原始输出

```bash
# 两文件 A/B（Hopper+TMA 构建）：TMA(默认) / cp.async(--lsetmavarlen=0)
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --varlen --full \
  --dir=/home/xieminglin/proj/output/fa-bwd/varlen_b4_t4096_h16_d128_full_fp8
python3 harness/fa_bwd_run.py --varlen-only --dtype fp8 --iters 20   # 单/两文件一致性
# ncu：--kernel-name regex:lse_mma_kernel_bal --launch-count 1
```

原始输出：`src/fp8/fa_bwd_fp8_o72_varlen_lse_ab.out.txt`、
`src/fp8/fa_bwd_fp8_p166_ncu_lse_{tma,cpasync}_s4096.out.txt`、
`src/fp8/fa_bwd_fp8_p166_varlen_run.out.txt`、`src/fp8/fa_bwd_fp8_p166_ci.out.txt`、
`src/fp8/fa_bwd_fp8_p166_te_varlen_baseline.out.txt`。

### 96.7 下一步

- **fp16/bf16 的同类改造**：把本步的 `cu_seqlens` 参数化逐字 dtype 化到 fp16/bf16 的
  `lse_mma_kernel_bal_tma` 与 `run_varlen`（fp16 是 **2× K=64 chunk** 描述符、bf16 逐字参数化），
  并把 fp16/bf16 varlen 构建也加上 `-DFA_TMA`——把「三 dtype × {定长, varlen} full D=128 LSE
  统一到 TMA」真正做全。
- main 的 L2 `red` 墙仍受本卡寄存器/smem 硬墙锁定（F6/F7 全判死，见「阻塞」）。

---

## 97. F7 第十四步（第一百六十八轮）：KV-owner **column-owner**（每 CTA 拥有 RCOL 个连续 KV 块）—— 机制正结果 / 性能负结果

### 97.1 动机（F7 单趟「减少贡献数」的最后一条未试路）

§93.6 收口时记：「F7 单趟减少 dK/dV/dQ 贡献数的三条路（两 kernel / TMA store-reduce / BN≥BM）
全部判决；唯一未试 = **跨 warpgroup 偏和 + 二次归约**」。本轮把这条里**最可行的一个形态**做成真实
数据：**column-owner** —— 让同一个 CTA 拥有 **RCOL 个连续 KV 块**，并在一次 m-迭代里把它们对 dQ
的偏和**先在本地累加、再只发一次原子**。

为什么它不同于已判死的 **p160 BN≥BM**：p160 把 `BN` 从 32 放宽到 64，虽然 `red` 精确减半，但
**GEMM1/2 的累加器也随 BN 翻倍**（`m64n32k32`→`m64n64k32`），叠加 `dK/dV` 累加器翻倍 ⇒ 顶穿
255 寄存器文件。**column-owner 保持 BN=32**，GEMM1/2 累加器不变，**只**翻倍 `dK/dV` 累加器
（RCOL 份）+ 一个跨列的 dQ 偏和累加器 —— 寄存器账更省，是 p160 教训的正面解。

### 97.2 实现（`src/fp8/fa_bwd_fp8_kvowner_mma.cu` 新增 `fp8_kvowner_col_body` + 壳，默认路径一行未改）

新增 device body `fp8_kvowner_col_body<HD,BM,BN,TMA,RCOL>`（`HD=128/BM=64/BN=32`，128 线程，
causal，MHA）与 `__global__` 壳 `fp8_kvowner_dkv_col_tma_kernel`；host 加 `--only=col2` 与同
binary 的 A/B、对拍、计时。grid.x = `ceil(S/(RCOL·BN))`。每 CTA：

- **K/V/Kp（RCOL 份）常驻 smem**，只从 global 读一次；`dK/dV` **寄存器累加（RCOL 份）**，循环末
  每块一次 plain store ⇒ dK/dV `red`=0；
- Q/dO 的 4D-TMA staging 与 Qp/dOp 配对布局**每 m 一次、RCOL 列共享**（⇒ Q/dO 的跨 CTA 读放大
  也随 grid 减半）；
- 内层 `for cc in 0..RCOL`：GEMM1/2（wgmma 直读 SW128）→ P/S → fold(Ap/dS3/dS2，用本列 `ks` ) →
  GEMM3/5（累加进 `dVacc[cc]/dKacc[cc]`）→ GEMM4（dQ 偏和累加进跨列的 `dqacc`，乘本列 `sds2·scale`）；
- 内层退出后 `dqacc` 用 **一次** `red_add2` 写出 ⇒ **dQ 贡献数 ÷ RCOL**。

**踩坑**：`Kp` 是 `uint16_t*` 而 `sz_Kp` 是字节数，`Kp + cc*sz_Kp` 的指针算术前进 2×（首次运行
`dq` 错 max_abs=1.89）；改 `reinterpret_cast<unsigned char*>(Kp) + cc*sz_Kp` 后逐位修正。

### 97.3 数值（三 shape 全通，逐位级一致）

| case（S,H,D128 causal fp8） | dk/dv vs wgmma+TMA(BN=32) | dq vs wgmma+TMA |
|---|---|---|
| S512 H16 | **0.000e+00 / 0.000e+00** | 2.384e-07 |
| S1024 H32 | 0.000e+00 / 0.000e+00 | 1.788e-07 |
| S4096 H16 | 0.000e+00 / 0.000e+00 | 2.384e-07 |

dk/dv **逐位相同**（同口径、只在本地累加），dq 仅差跨 CTA 加法次序（~2e-7，远小于 fp8 容差）；
col-owner 三梯度 vs fp32 ref 与 wgmma+TMA **同量级**（如 S512 `2.976/3.732/2.426e-1`）。

### 97.4 性能（同 binary A/B，Hopper+TMA，event）：**慢 0.72–0.77×**

| case | wgmma+TMA BN=32（三梯度） | **col-owner RCOL=2** | 比 | smem |
|---|---|---|---|---|
| S512 H16 | 0.0749 ms | 0.1025 ms | **0.731×** | 87,616 B |
| S1024 H32 | 0.3075 ms | 0.4281 ms | **0.718×** | 87,616 B |
| S4096 H16 | 1.8352 ms | 2.3966 ms | **0.766×** | 87,616 B |

### 97.5 ncu（S=4096 H16，`--only=` 单 kernel，`regex:wgmma_tma_kernel` vs `regex:col_tma`）—— 机制成立、被寄存器墙吃掉

| 指标 | **wgmma+TMA BN=32** | **col-owner RCOL=2** | 说明 |
|---|---|---|---|
| `lts__t_sectors_op_red` | 102,236,160 | **51,118,080** | **精确减半**（设计目标达成） |
| `l1tex…global_op_red`（red 请求） | 8,519,680 | **4,259,840** | dQ 原子数减半 |
| `lts__t_sectors_op_read` | 52.78M | **105.69M（2.0×）** | Q/dO 重读**应减半**，但被 spill 读抵消 |
| `lts__t_sectors_op_write` | 6.20M | **113.73M（18.3×）** | **寄存器溢出（local store）** |
| registers/thread | 168 | **255（上限）** | `dVacc/dKacc` 翻倍 + `dqacc` |
| occupancy limit（reg / smem） | 3 / 3 block | **2 / 2 block** | 2 CTA/SM |
| Duration | 1.85 ms | 2.46 ms | |
| `smsp__inst_executed` | 703.65M | 628.78M | −10.6% |
| active warps | 17.77% | 11.93% | |

### 97.6 判决

**机制完全成立**：column-owner 把 dQ 的 L2 `red` **精确减半**（102.2M→51.1M）、L1 red 请求减半，
并让 Q/dO 的跨 CTA 读放大随 grid 减半 —— 正是 F7 追的「减少每个输出元素的贡献 CTA 数」。**但
`dK/dV` 累加器翻倍（RCOL 份）+ 跨列 `dqacc` 把寄存器顶到 255 上限**，每线程 112B 栈、local `write`
扇区 6.2M→**113.7M（18.3×）**，溢出流量盖过省下的 `red`/读，净慢 0.72–0.77×。

⇒ **与 p160（BN≥BM）、F6、O17b「放大 tile / 多 owner 撞 255 寄存器文件」是同一堵墙**：本卡
（128KB reg/CTA 上限、`__launch_bounds__` 170 regs@3CTA）下，**任何「单 CTA 拥有更多 KV」以减少
跨 CTA 贡献数的实现，都把 dK/dV 累加器翻倍 ⇒ 溢出**。至此 F7 单趟路线（两 kernel / TMA store-reduce
/ BN≥BM / column-owner）**全部判决**；剩余只有**换卡**（寄存器/smem 更大）或**多 warpgroup 把累加器
摊到更多线程**（256/384 线程，对标 TE 的 384 线程 1 CTA/SM —— 需先破 fp8 `wgmma` 无转置操作数的
限制，见「阻塞」）。

### 97.7 复现 / 原始输出

```bash
# 构建（Hopper+TMA）：含 --only=col2 的 ncu 单 kernel 剖析
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_kvowner_mma.cu --dir=<case>
scripts/ncu.sh src/fp8/fa_bwd_fp8_kvowner_mma.cu --metrics <...> --launch-count 1 \
  --kernel-name regex:col_tma -- --dir=<case> --only=col2
```

原始输出：`src/fp8/fa_bwd_fp8_kvowner_col_p168_{s512,s1024h32,s4096}.out.txt`、
`src/fp8/fa_bwd_fp8_kvowner_col_p168_ncu_s4096.out.txt`。

## 98. O74（第一百六十九轮）：MLA（head_dim=512）causal LSE 上 **4D-TMA** —— 补齐 fp8 LSE 的最后一条分支

### 98.1 动机

F5→F9→O70→F11→F12 把 fp8 的 **4D-TMA LSE** 逐步铺到 `D=128` 的 {causal, full, varlen-full}；
但 **MLA（`D=512`）的 causal LSE 一直走 `lse_mma_kernel_bal<512,...>`（`mma.m16n8k32` + `cp.async`）**，
从没有 TMA 版。`lse_mma_kernel_bal_tma<HD>` 里硬编码了 `static_assert(HD == 128)`（原因：TMA 的 box
内维固定 128B=128 fp8，`HD=128` 恰好一个 box 搬完整行；`HD=512` 需要拆 4 个 box）。本轮把这层
dtype/维度泛化补上，使 **fp8 的 LSE 四条路径（定长/变长 × causal/full × D=128/D=512）全部走 4D-TMA**。

判据（O39/O59 已把 MLA causal LSE 优化到 `cfg6`）：`lse_mma_kernel_bal<512,1,false,128,16>`（4 CTA/SM、
K 维 split auto=16）实测 S1024H2 Duration **19.62µs / Waves 0.48 / Compute 30.77%**；TMA 在 D=128 上
是 1.88×（O72），预期 MLA 的 LSE 也能降 1.5–1.9×。

### 98.2 实现（device + host，单/两文件 device 由 `sync_onefile_device.py` 逐字同步）

- **device**（`fa_bwd_fp8_kernels.cuh` 的 `lse_mma_kernel_bal_tma`）：
  - `static_assert(HD % 128 == 0)`；新增 `NCH = HD/128`、`CHUNK = (LBM/8)*1024`（=8KB）、
    `TILE = NCH*CHUNK`。
  - `issue_q`/`issue_k` 各发 **`NCH` 次 4D-TMA**，第 `c` 个 box 写入 `Qs + c*CHUNK`、global 内维坐标
    `c*128`；`mbar_arrive_expect` 仍只声明总字节 `TILE`（N 次 load 的 `complete_tx` 累加）。
  - 新增 `wgmma_qkt64_fp8_chunked(nch, chunk)`：因为 4 个 box 各写一块 canonical `[LBM][128]` tile，
    物理布局是 `[kg][rg]`（而非单 box 的 canonical `[rg][kg]`），故**每个 chunk 各用 SBO=1024 的描述符**
    累加（逐字对齐 fp16 O30/O71 的 2-chunk 写法）。`NCH==1` 时 `if constexpr` 仍走原
    `wgmma_qkt64_fp8` ⇒ **D=128 路径编译出逐位相同的代码**。
- **host**（`fa_bwd_fp8_main.cu` + `fa_bwd_fp8_mma_onefile.cu`，两处逐字相同）：
  - `cudaFuncSetAttribute(lse_mma_kernel_bal_tma<512,1>, ..., Fp8Cfg<512,64,32>::lse_smem_bytes_tma1)`
    （smem = `3*lse_tile_wgmma + (LBM+2*LBN)*4 + 1024 + 64` = **100,160B ⇒ 2 CTA/SM**）。
  - `lse_tma` 默认：`(D==128) ? 1 : (D==512 && causal ? 1 : 0)`；`--lsetma=0` 供同 binary A/B。
  - Q/K 的 LSE 描述符构建条件 `D==128` 改为 `D==128 || D==512`（`run_varlen` 的 D=512 **不受影响**，
    仍走 `cp.async` 版——本轮只动定长 causal 路径）。
  - D=512 的 causal 分支：`lse_tma` 时优先 `launch_lse_bal_tma_split<512,1>` / `launch_lse_bal_tma<512,1>`，
    否则退回原来的 `launch_lse_bal<512,...>`（cfg6/legacy）。

### 98.3 数值

- **D=128 逐位不变**（`NCH==1` 走原 wgmma 路径）：S4096 causal `2.635/2.644/3.216e-1`、S512
  `2.426/2.972/3.733e-1`，与 O37/O41 历史完全一致。`--ci` fp8 单/两文件一致性 **worst 8.583e-6 OK**、
  `--check docs/04` OK。
- **D=512 MLA causal**（TMA vs `--lsetma=0` 的 mma/cfg6，只差 LSE 的 fp32 求和次序）：
  | shape | 路径 | dq | dk | dv |
  |---|---|---|---|---|
  | `(1,256,2,512)` | TMA | 2.355e-1 | 2.284e-1 | 3.485e-1 |
  | | mma | 2.356e-1 | 2.290e-1 | 3.441e-1 |
  | `(1,512,4,512)` | TMA | 2.415e-1 | 2.992e-1 | 4.481e-1 |
  | | mma | 2.415e-1 | 2.992e-1 | 4.481e-1 |
  | `(1,1024,2,512)` | TMA | 2.228e-1 | 3.311e-1 | 3.611e-1 |
  | | mma | 2.232e-1 | 3.337e-1 | 3.602e-1 |
  全部在 fp8 噪声/求和次序量级内（≤4.7e-3，对应 split 的 fp32 求和次序差）。单文件与两文件逐指标
  一致（单文件 S1024H2 `2.228/3.311/3.611e-1`、total 0.1508ms）。

### 98.4 性能（同 binary A/B，Hopper+TMA 构建，CUDA event，iters=200）

| shape | preprocess mma | preprocess TMA | total mma | total TMA | total 加速 |
|---|---|---|---|---|---|
| `(1,256,2,512)` | 0.0175 ms | **0.0097 ms（1.80×）** | 0.0481 ms | **0.0400 ms** | **1.203×** |
| `(1,512,4,512)` | 0.0186 ms | **0.0116 ms（1.60×）** | 0.1052 ms | **0.0977 ms** | **1.077×** |
| `(1,1024,2,512)` | 0.0240 ms | **0.0160 ms（1.50×）** | 0.1601 ms | **0.1510 ms** | **1.060×** |

`lse k-split auto` 分别选 4/8/16；main 不变（MLA main 仍是 mma/cp.async，占 74–82%）。

### 98.5 ncu（`(1,1024,2,512)` causal，同 binary，`regex:lse_mma_kernel_bal`，`-c 1`）

| LSE 路径 | Duration | Waves/SM | Achieved Occ | Compute | L1/TEX | L2 | 寄存器 | Block Limit Shared Mem |
|---|---|---|---|---|---|---|---|---|
| mma `cfg6`（`--lsetma=0`） | 19.62 µs | 0.48 | 11.83% | 30.77% | 24.41% | 24.58% | 61 | 4 CTA/SM |
| **TMA（默认）** | **10.34 µs（1.90×）** | **0.97** | 11.55% | 15.08% | 15.56% | 21.60% | 61 | 2 CTA/SM |

⇒ **LSE kernel 1.90×**：TMA 把「网格不足半个波 + `cp.async`/ldmatrix 发射」换成「填满一个波 + 纯 TMA
搬运」，`Waves 0.48→0.97`、Compute 30.77%→15.08%、L1/TEX 24.41%→15.56%；新档 `Compute 15% / L1TEX 16%`
说明已不在指令/访存饱和区，墙回到 LSE 固有的 tile 级 `mma wait` + 2 CTA/SM（与 D=128 的 TMA LSE 一致）。

### 98.6 复现 / 原始输出

```bash
# 默认（TMA）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=.../b1_s1024_h2_d512_causal_fp8 --iters=200
# A/B（mma/cfg6）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
  scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=... --iters=200 --lsetma=0
```

原始输出：`src/fp8/fa_bwd_fp8_o74_mla_lse_tma_{s256h2,s512h4,s1024h2}.out.txt`、
`..._mla_lse_mma_{s256h2,s512h4,s1024h2}.out.txt`、`..._mla_lse_tma_onefile_s1024h2.out.txt`、
`..._mla_lse_tma_ncu_s1024h2.out.txt`、`..._mla_lse_mma_ncu_s1024h2.out.txt`、
`..._o74_d128_regression_s4096.out.txt`。

### 98.7 下一步

① ~~**把本步逐字 dtype 化到 fp16/bf16**~~ → **已完成（O75，第一百七十轮）**：fp16/bf16 的
`lse_mma_kernel_bal_tma` 去 `static_assert(HD==128)`、改 `NCH=HD/64`（box 内维 128B = 64 个
fp16/bf16），`NCH==2` 走原 `wgmma_qkt64_tma` ⇒ **HD=128 逐位不变**，新增 `wgmma_qkt64_tma_chunked`；
host `lse_tma` 默认 `(D==512&&causal)?1:0`、causal 优先 `lse_mma_kernel_bal_tma<512,1>`（smem
197,696B ⇒ 1 CTA/SM）。**LSE-only preprocess 1.20–1.56×（fp16）/1.21–1.54×（bf16）、端到端 S256H2
1.17× / S512H4 1.04× / S1024H2 1.07×，数值逐位相同、`--ci` 全绿**。见 `docs/01` §25、`docs/01b`
§6bb。**至此三 dtype × MLA 的 causal LSE 也统一到 4D-TMA**；
② main 的 L2 `red` 墙（F7 全判死、F6 不可行，受本卡寄存器/smem 硬墙锁定，见「阻塞」）。

## 99. O76（第一百七十一轮）：新增 **head_dim=256** 支持（fp8）—— 补齐 128/512 之间的形状

### 99.1 动机

此前 fa-bwd 的 fp8 反向只支持 `head_dim ∈ {128（MHA/GQA）, 512（MLA）}`（host 显式
`if (D != 128 && D != 512) 报错`）。但 **FA/TE 的反向都支持到 `head_dim=256`**（`docs/00` §4.1、
`docs/06`），256 是介于二者之间的一个标准档位（部分模型用 192/256）。本轮把 **fp8 的定长反向
扩到 `D=256`**，是 ROADMAP「换形状/维度覆盖」这一唯一还能产出的正结果方向的第一步。

为什么 fp8 的 `D=256` 可以「几乎零 device 改动」拿到：
- 主 kernel `fp8_mma_body`（`fa_bwd_fp8_mma_kernel<HD,...>`）对 `HD` 本就有模板化约束
  `HD % NTW == 0`（`NTW = WN*64 = 128`）⇒ **256 合法**；`HD/NTW = 2 ≠ 1` 时 `kRegDq` 自动关
  （与 `HD=512` 同），GEMM3/4/5 的 N 维自动分 2 遍。
- LSE 的 `lse_mma_kernel_bal<HD,...>`（D=512 在用的镜像配对/均衡 FULL 版）对 `HD` 是模板参数，
  256 直接复用。
- `delta_warp_kernel<HD>`、`quantize_*_warp_kernel<VPT>` 只需 `HD%4==0`、`VPT=D/32`（256→**8**）。

所以本轮是 **纯 host dispatch + 一个 VPT=8 实例**；device 的通用代码一行未改（单/两文件共用）。

### 99.2 实现（host，单/两文件逐字同步）

`src/fp8/fa_bwd_fp8_main.cu` 与 `src/fp8/fa_bwd_fp8_mma_onefile.cu` 各改 7 处、逐字相同：

1. 形状守卫：`D != 128 && D != 512` → 允许 **256**（报错信息同步）。
2. smem 选择：`Fp8Cfg<256,64,32>::smem_bytes` / `::lse_smem_bytes`（新增一档）。
3. `quant_new` / `quant_zero` / `quant_zero_delta`：`VPT=D/32`，256 走 **`<8>`** 实例
   （`quantize_row_warp_kernel<8,false/true>`、`quantize_zero[_delta]_warp_kernel<8>`）。
4. `run_preprocess` 新增 `else if (D==256)`：causal 走 `launch_lse_bal<256,1>`（镜像配对 + K 维
   split）、full 走 `launch_lse_bal<256,1,true>`（均衡 FULL）；再跑 `delta_warp_kernel<256>`。
   **不做 4D-TMA**（只有 128/512 有 TMA 版）。
5. `run_main` 顶部新增 `if (D==256)`：`launch_bwd_main<256,64,32,false>`（mma，4-warp/128 线程；
   wgmma 只做 128、TMA 只做 128）。
6. O26 A/B 诊断把 `else <512>` 拆成 `else if (D==256) <256> else <512>`（否则拿 512 的核去跑
   256 的缓冲 → 越界读、打印 5e30）。
7. `lse_tma` 默认仍是 `(D==128)?1:(D==512&&causal?1:0)`，256 自动 **0**（无 TMA）；Q/K 与主
   kernel 的 TMA 描述符只在 `D==128/512` 构建，256 跳过。

> fp16/bf16 的 `D=256` **本轮未做**（其 `*_mma_main.cu` 有 ~80–90 处 `D==128/512` 的
> wg2/wgmma4/cluster 分派，改动面大得多），留 backlog。故 `dump --requested` 不纳入 256
> （见 `harness/fa_bwd_bench.py` 的 `HD256_FP8_SHAPES` 注释）。

### 99.3 数值（ours vs fp32 ref，max_abs dq/dk/dv；单/两文件经 `fa_bwd_run.py` gate）

| shape (B,S,H,D;Hkv) | 模式 | ours（两文件） | ours_sf（单文件） |
|---|---|---|---|
| (1,1024,8,256) MHA | causal | 2.630e-1 / 2.795e-1 / 3.589e-1 | 同（逐指标一致） |
| (1,2048,8,256) MHA | causal | 2.220e-1 / 2.835e-1 / 3.584e-1 | 同 |
| (1,1024,16,256) Hkv=4 | causal | 2.477e-1 / 4.455e-1 / 6.157e-1 | 同 |
| (1,1024,8,256) MHA | full | 4.972e-2 / 5.571e-2 / 4.092e-2 | 同 |

与 `D=128` 的 fp8 噪声同量级（2–4e-1 causal、~5e-2 full）；GQA 的 `dk/dv` 略大（每个 KV 头承载
`H/Hkv=4` 个 Q 头的梯度，符合 `docs/06` §4.1 规律）。**FP8 的两半 `d[0..127]` / `d[128..255]` 误差
量级相同**（无「只算了一半」的 N-tile bug）。**单/两文件一致性**（`our vs ours_sf`）4 个 shape 的
worst = **2.384e-6**（fp8 gate tol 1e-4，**OK**）；全量 `--ci` 77 case 三 dtype gate 全绿
（fp16 1.953e-3 / bf16 7.812e-3 / fp8 5.722e-6）、`--check docs/04` OK（198 行）。

### 99.4 性能（CUDA event，iters 见原始输出；fp8 bwd FLOPs = `4BS2HD`）

| shape | total | main | main-only TFLOPS | 占 FP8 峰值 1978.8 |
|---|---|---|---|---|
| (1,1024,8,256) causal | 0.364 ms / 23.6 TF | 0.289 ms | **29.7 TF** | 1.50% |
| (1,2048,8,256) causal | 1.129 ms / 30.4 TF | 1.001 ms | **34.3 TF** | 1.73% |
| (1,1024,16,256) kv=4 causal | 0.664 ms / 25.9 TF | 0.555 ms | 30.9 TF | 1.56% |
| (1,1024,8,256) full | 0.558 ms / 15.4 TF | 0.449 ms | 19.1 TF | 0.97% |

对标（`harness/fa_bwd_bench.py bench`，纯反向 CUPTI；**FA/TE 的 fp8 反向不支持 256**，故给
同 shape 的 fp16/bf16 三列做量级参照）：

| shape | FA2.7.4 (fp16) | TE2.14 (fp16) | FA3 | ours fp8 main-only |
|---|---|---|---|---|
| (1,1024,8,256) causal | 168.5 TF | 216.3 TF | NA（本机只编 HDIM128） | 29.7 TF |
| (1,2048,8,256) causal | 228.0 TF | 320.1 TF | NA | 34.3 TF |

⇒ ours fp8 `D=256` 的 main 约为 TE fp16 的 **14%（S1024）/11%（S1024 之外）**——因为 fp8 的
`D=256` 走的是 **非 wgmma 的 mma.m16n8k32 路径 + 无 TMA**（`D=128` 才默认 wgmma+TMA），与
`D=128` 的 mma 档同源；差距量级与 `D=128`「ours mma vs TE」一致。

### 99.5 ncu（`(1,1024,8,256)` causal，主 kernel `regex:fa_bwd_fp8_mma_kernel`，`-c 1`）

| Duration | Registers | Dyn smem | Block Limit Shared Mem | Achieved Occ | Waves/SM |
|---|---|---|---|---|---|
| 327.2 µs | **254** | 117.76 KB | **1** | 6.25% | 3.88 |

| Compute | Memory | L2 | DRAM | L1/TEX | No Eligible | shared-store bank conflict |
|---|---|---|---|---|---|---|
| 15.97% | 40.85% | 41.41% | 3.53% | 30.82% | 78.87% | 1.6-way（26% 多余 wavefront） |

**bound = 低 occupancy + 延迟/并行度受限**，与 `D=512`（MLA）同类：`HD` 翻倍使 GEMM3/4/5 的
N 维分 2 遍，**寄存器顶到 254**（`Fp8Cfg` 的 fp32 累加器/P·S 缓冲随 `HD` 增大），加上 117.76KB
动态 smem（`ASLD=HD+16`、`Qp/dOp` 随 `HD` 增大）把 **Block Limit Shared Mem 锁到 1 CTA/SM**、
achieved occ 6.25%；`No Eligible 78.87%` 说明发射口大量空等。L2 仅 41%、DRAM 3.5%、Compute 16%
⇒ **不是带宽/算力 bound**。下一步若要提`D=256` 的性能，需（与 MLA 相同）降 smem 冲 2 CTA/SM、
或把 `D=256` 也接到 wgmma/TMA 快路（需先破 fp8 wgmma 只做 `HD=128` / SW128 atom 的 `HD` 约束）。

### 99.6 复现 / 原始输出

```bash
# dump（仅 fp8；FA3 本机只编 HDIM128、TE fp8 不支持 256）
python harness/fa_bwd_bench.py dump --dtype fp8 \
  --shape '1 1024 8 256 causal' --shape '1 2048 8 256 causal' \
  --shape '1 1024 16 256 kv=4 causal' --shape '1 1024 8 256 full'
# 默认（Hopper 构建；D=256 在 host 内自动退回 mma 主 kernel）
python harness/fa_bwd_run.py --dtype fp8 --glob '*d256*' --iters 20
# 单/两文件原始运行输出
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu        --dir=.../b1_s1024_h8_d256_causal_fp8 --o=ref_o
scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu --dir=.../b1_s1024_h8_d256_causal_fp8 --o=ref_o
# ncu
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full \
  --kernel-name regex:fa_bwd_fp8_mma_kernel --launch-count 1 \
  -- --dir=.../b1_s1024_h8_d256_causal_fp8 --o=ref_o --iters=1
```

原始输出：`src/fp8/fa_bwd_fp8_o76_d256_{s1024_h8_causal,s2048_h8_causal,s1024_h16_kv4_causal,
s1024_h8_full}_{twofile,onefile}.out.txt`、`src/fp8/fa_bwd_fp8_o76_ncu_d256_main_s1024.out.txt`、
`src/fa_bwd_o76_d256_baseline_fp16bf16.out.txt`。

### 99.7 下一步

① **把 `D=256` dtype 化到 fp16/bf16**（其 `*_mma_main.cu` 的 `D==128/512` 分派面较大：需为 256
选 `BM/BN/几何`，或复用一个通用 mma 分支）；② `D=256` 的 fp8 main 接 wgmma/TMA（受 fp8
wgmma `HD=128` 与 SW128 atom 约束，见「阻塞」）；③ main 的 L2 `red` 墙（F7 全判死、F6 不可行，
受本卡寄存器/smem 硬墙锁定，见「阻塞」）。

## 100. O77（第一百七十二轮）：F6-③ 收口 + fp8 main 的寄存器/smem/occupancy 三证 —— fp8 main 在本卡**已到硬件平台期**

> 背景：ROADMAP「下一批（fp8 继续）F6」的三条子项中，①（降 ksplit 消 Q/dO 重读）与 ②
> （dK/dV 跨 CTA `red` 改分块/column-owner）此前已被 F6/F7 系列判决；仅剩 **③（Q/dO 的
> 4D-TMA cache hint / L2 persist 减少重读）** 未实测。本轮把 ③ 实测收口，并把「为什么
> fp8 默认 main 在本卡无更多软件空间」用三组硬证据钉死。

### 100.1 F6-③：4D-TMA 的 L2 promotion（Q/dO/K/V）——**负结果**

- **动机**：fp8 默认 main（`fa_bwd_fp8_mma_kvtma_kernel`，ksplit=8）会**重复读取**
  每个 m 块的 Q/dO（每个 ksplit part 各读一次，共 8 次），直觉上给 Q/dO 的 TMA 描述符加
  L2 promotion（`CU_TENSOR_MAP_L2_PROMOTION_L2_128B/256B`）可提高 L2 命中、减少重读。
- **实现**（纯 host，device 一行未改）：`make_lse_map_fp8` 的 promotion 由文件作用域
  `g_l2promo` 决定（0=NONE 历史默认 / 1=L2_128B / 2=L2_256B），CLI `--l2promo=N` 强制，
  在**创建描述符之前**生效；默认 0 ⇒ 描述符与历史逐字节相同。
- **实测（同 binary 交替，S4096 causal，iters=30）**：

  | l2promo | total (ms) | main (ms) | TFLOPS |
  |---|---|---|---|
  | 0 = NONE | **1.7913** | 1.5533 | 76.73 |
  | 1 = L2_128B | 1.7936 | 1.5524 | 76.63 |
  | 2 = L2_256B | 1.8052 | 1.5691 | 76.13 |

  ⇒ **噪声内（±0.3%）、L2_256B 略负**。原因：ncu 显示默认 main 的 **L2 命中率 97.08%、
  DRAM 仅 4.32%**——Q/dO 的「重读」本来就在 L2 命中（并未打到 DRAM），promotion 只影响
  回填粒度、改变不了 L2 扇区总量。**L2 persist / cache hint 这条路对 fp8 main 无效。**
  原始输出 `src/fp8/fa_bwd_fp8_o77_l2promo_ab_s4096.out.txt`。

### 100.2 三证：ksplit / 编译期旋钮 / 寄存器账

- **ksplit 复扫（同 binary，S4096 causal，iters=30）**：1→**2.154ms**、2→1.913、4→1.805、
  **8→1.790（auto，最优）**、16→1.873。⇒ F6-①「降 ksplit 消 Q/dO 重读」在本卡被**并行度**
  锁死：少切分虽然 red 更低，但 grid 铺不满、延迟暴露，净更慢。auto=8 已是最优点。
  原始输出 `src/fp8/fa_bwd_fp8_o77_ksplit_sweep_s4096.out.txt`。
- **默认关的编译期开关在当前 TMA 构建下复测**（均在 `-DFA_WGMMA -DFA_TMA` 下，同 shape）：
  `FA_WS1=1` 1.7925 / `FA_ILV=1` 1.7894 / `FA_ILV34=1` 1.7944 / `FA_R4=1` 1.8180 /
  `FA_WS1+ILV34` 1.7916，baseline **1.7913**。⇒ 全部噪声内或有损（`FA_R4` 的 16B red
  再次确认「归约加宽」不降 L2 扇区）。这些旋钮**保持默认关是正确的**。
  原始输出 `src/fp8/fa_bwd_fp8_o77_macro_ab_s4096.out.txt`。
- **寄存器/spill 账（`-Xptxas -v`）**：默认实例
  `fa_bwd_fp8_mma_kvtma_kernel<128,64,32,REGDQ=1,...>` = **168 regs / 40B spill
  （40B st + 44B ld）/ 74.82KB smem**。`__launch_bounds__(128,3)` 的寄存器上限 =
  `65536/384 = 170` ⇒ **恰好顶格、溢出 10 个长生命期值**。而
  - 4 CTA/SM 需 **≤ 128 regs**（`65536/512`）**且 ≤ 58.1KB smem**（`232448/4`）；
  - 去掉 spill 需 > 170 regs（实测 ptxas 需求 ~180）。
  ⇒ 两者都**达不到**。ncu 亦记 **local memory 占 L1TEX 扇区 ~7.7%（local op_ld/st
  5.84M/5.67M sectors）、占 L2 ~4.5%**，属 spill + 长生命期地址的固有开销。
  原始输出 `src/fp8/fa_bwd_fp8_o77_ptxas_spill.out.txt`、
  `src/fp8/fa_bwd_fp8_o77_ncu_local_src_s4096.out.txt`。
- **`wait`/`short_scoreboard` 是头号 stall**（ncu，S4096）：`wait 27.48% + short_scoreboard
  21.65% + barrier 6.77% + long 6.71%`，`Active Warps/Sched 2.94`（3 CTA/SM = 12 warps）。
  即 kernel 在 L2 80% 之下**仍受 mma 依赖延迟 + smem→ldmatrix 依赖约束**；要打它需要更多
  warp（→4 CTA/SM，被 100.2 的寄存器/smem 双墙锁死）或跨-tile 软流水（→+32KB smem 掉
  2 CTA/SM，ROADMAP「阻塞」已核算为中性偏负）。

### 100.3 结论

fp8 默认 main（S4096：main 1.553ms / total 1.791ms / 76.7 TFLOPS、TE FP8 的 ~6.4×）在本卡
（3 CTA/SM、170 regs 上限、74.8↔77.5KB smem 硬间隙）**已无更多软件杠杆**：
- **减 L2 搬运量**只有「改工作划分」一条路，而它需要翻倍的长生命期累加器（dK/dV 或 dQ），
  撞 128-reg / 4-CTA 与 116KB / 2-CTA 双墙（F6/F7/O17b/p160/p168 五条路全部判决，见「阻塞」）；
- **TMA cache hint / L2 persist**（F6-③）已实测无效（L2 命中 97%、DRAM 4.3%）；
- **藏延迟**需要的额外 warp / 软流水被同一双墙锁死。

⇒ 继续产出正结果只能**换卡**（寄存器/smem 更大的目标卡）或**多 warpgroup 摊累加器**
（对标 TE 384 线程，需先破 fp8 `wgmma` 无转置操作数）。本轮默认路径数值逐位不变，
`--l2promo` 为默认关的诊断开关。

### 100.4 原始输出

- `src/fp8/fa_bwd_fp8_o77_l2promo_ab_s4096.out.txt`（F6-③ A/B）
- `src/fp8/fa_bwd_fp8_o77_ksplit_sweep_s4096.out.txt`（ksplit 复扫）
- `src/fp8/fa_bwd_fp8_o77_macro_ab_s4096.out.txt`（编译期开关复测）
- `src/fp8/fa_bwd_fp8_o77_ptxas_spill.out.txt`（寄存器/spill 账）
- `src/fp8/fa_bwd_fp8_o77_ncu_local_src_s4096.out.txt`（ncu local-memory 源级）

## 101. O78（第一百七十三轮）：端到端 overlap（quant/LSE 与 main 跨 stream）—— 负结果

**动机**：§100 把 fp8 main 钉成「硬件平台期」后，ROADMAP「下一步候选 ②」只剩一条能产正结果的
方向——**端到端重叠**：fp8 端到端里 main 只占 ~87%，`quant`（~0.106ms）+ `preprocess`（LSE，
~0.121ms）合计 ~12.7%，若能与 main 重叠即最多省 ~7–13%。本轮把这条候选人正式实现并判决。

**实现（纯 host，默认关；`--ovlp=N`）**：
- 新 `make_map_fp8_chunk(ptr, Hfull, h0, hc, S, D, B, box)`：把 4D-TMA 描述符的 **head 计数
  `dims[2]=hc`** 与 **物理行距 `strides={Hfull*D, D, S*Hfull*D}`** 解耦、基址前移 `h0*D`。于是
  kernel 内 head 坐标仍取 `blockIdx.y∈[0,hc)`，而物理地址与全量路径**逐元素同址**——**device
  代码一行未改**。
- 把 `d_q8/qs/do8/dos/delta/lse/dq_acc` 按 `h0`、`d_k8/ks/v8/vs/dk_acc/dv_acc` 按 `hkv0=h0`
  （MHA）做指针偏移；`grid.y` 由 H 改为 `hc`。LSE 强制 `ksplit=1`（避开 `lse_part` 的跨 head 步长）。
- 两条非阻塞 stream：`sA` 顺序发 `LSE(k)`、`sB` 发 `main(k)`；`main(k)` 用 event 等 `LSE(k)`；
  quant（含 delta/清零）整体留在 default stream，两 stream 先等一个 quant 完成的 event。
- 条件：定长 / D=128 / causal / MHA(`Hkv==H`) / 默认 wgmma+4D-TMA / `H%ovlp==0`；否则逐字退回串行。

**诊断（`--ovltest=1`，不改结果）**：把**整块** LSE 与**整块** main 放到两条 non-blocking stream
并发（LSE 输出丢弃、只测墙钟）：
```
main=1.5549 | lse=0.1210 | serial(sum)=1.6759 | concurrent wall=1.5939 => 0.951x
```
即**内核级**重叠确实可行——并发墙钟比串行和快 ~5%（LSE 被隐藏 ~0.082ms，67%）。

**实测（`--ovlp=N`，S4096 causal MHA，同 binary，`--iters=10`）**：
```
ovlp=0   1.7956 ms / 76.54 TF   (基线，串行)
ovlp=2   1.9032 ms / 72.22 TF   (+6.0%)
ovlp=4   2.1194 ms / 64.85 TF   (+18.0%)
ovlp=8   2.3798 ms / 57.75 TF   (+32.5%)
ovlp=16  2.7310 ms / 50.33 TF   (+52.1%)
```
**控制实验**（`--ovlp=N --ovlnolse=1`，只测「分块 main 串行」，去掉 LSE 变量）：
```
mainonly chunks=2  1.7625 ms   (full main+quant=1.676 → 分块 main +5.2%)
mainonly chunks=4  1.9258 ms   (+15%)
mainonly chunks=8  2.1622 ms   (+29%)
```
**归因**：把 main 的**单一 8192-CTA 栅格**切成 N 段后，每段的**尾波量化（tail quantization）**
被放大 ~N 倍——full main 是 20.7 波、尾波占比 ~5%，切成 2/4/8 段后每段只有 10.3/5.2/2.6 波，
尾波占比升到 ~10/~19/~38%。**分块 main 的尾波损失（+5%~+29%）远大于被隐藏的 LSE（~0.08ms，
4.5%）**；且并发时 LSE 与 main 争 SM 还会进一步拖慢 main（ovlp=4 的 LSE 实际加了 ~0.14ms 而非
0.12ms）。故 **O78 = 负结果**：端到端重叠在本卡对本 shape 不成立。

**结论**：本 shape 的 fp8 默认 main 已按「3 CTA/SM + 8192 CTA 铺满 20.7 波」调优，**任何在 head/
sequence 维上的切分都会重新引入尾波**；LSE 只有 ~7%，无法在「不切分 main」的前提下被隐藏。
⇒ 想真正吃到这 ~5% 的内核级重叠收益，只能**把 preprocess 融进 main**（warp-specialized
prologue / 单 kernel 内流水），属大改（同 F3b 的困境）。**默认路径未改，数值逐位不变。**

**原始输出**：
- `src/fp8/fa_bwd_fp8_o78_ovltest_s4096.out.txt`（诊断）
- `src/fp8/fa_bwd_fp8_o78_sweep_s4096.out.txt`（ovlp 扫参 + 分块 main 控制）
- `src/fp8/fa_bwd_fp8_o78_ncu_main_s4096.out.txt`（默认 main ncu，L2 77.10% / L1TEX 71.96% /
  DRAM 4.21% / Compute 48.27% / 168 regs / 74.82KB / 3 CTA/SM / Waves 20.69）
- `src/fp8/fa_bwd_fp8_o78_baseline_te.out.txt`（TE FP8 纯反向 0.3035ms/905.6TF ⇒ ours/TE 5.92×）

## 102. F3b 前置（第 174 轮）：TE SASS 的 **QGMMA RS_TN（A 在寄存器）** 发现 + fp8 wgmma RS 冒烟 —— 重新打开「GEMM3/4/5 上 wgmma」路径

### 102.1 动机与结论一句话

ROADMAP『fp8 专项冲刺』把「把 fp8 main 从 mma.sync 切到 wgmma」列为主线（F1→F5），
但 F1 只把 **GEMM1/2** 换成了 wgmma（SS_TN），GEMM3/4/5 仍 `HMMA.16816 + LDSM`；「阻塞」
里记的论据是「fp8 wgmma 只有 `SS_TN`，没有转置操作数，故 B 必须物理转置」。本轮用
`ncu --page source --print-source sass` 逐指令对照 TE 的反向 kernel 后发现：**该论据只覆盖
`SS_TN`（A/B 均描述符）**；TE 的 GEMM3/4/5 用的是 **`RS_TN`（A 在寄存器、B 走描述符）**，
于是「需要转置的那个操作数」可以经 `ldmatrix` 放进寄存器，绕开「B 必须物理转置 SW128」。
本轮的增量就是**把这条路径的第一步钉死**：写 `fa_bwd_fp8_wgmma_rs_smoke.cu` 验证
`ldmatrix.x4` 取回的 4×u32 恰是 `wgmma.m64n32k32` RS_TN 的 A 片段（`ALayout_64x32`），
**e4m3×e4m3 与 e5m2×e4m3 两组 max_abs=0（PASS）**。默认路径一行未改。

### 102.2 逐指令对照（S=512 causal，`ncu --page source --print-source sass`）

| opcode | TE `..._flash_bprop_wgmma_f8_..._64x64x128_1x4x1` (384 线程, grid=64) | ours `fa_bwd_fp8_mma_kvtma_kernel<128,64,32,...>` (128 线程, grid=...) |
|---|---|---|
| QGMMA | **16**（8×`64x64x32` + 8×`64x128x32`） | 8（仅 GEMM1/2 `64x32`） |
| HMMA | **0** | **96**（GEMM3/4/5） |
| LDSM | 20（12×`MT88.4` trans + 8×`M88.4`） | 46（40×`MT88.2` + 6×`M88.2/4`） |
| **STSM** | **24**（20×`M88.4` + 4×`MT88.4` trans） | **0** |
| REDG | 6（`MIN/MAX` amax；无逐元素 ADD） | **32×`REDG.E.ADD.F32`**（dK/dV/dQ 原子） |
| UTMA | 4×`UTMALDG.4D` + 2×`UTMASTG.4D` + **4×`UTMAREDG.4D.ADD`** | 7×`UTMALDG.4D`（Q/K/V/dO） |

TE 的 QGMMA 里半数带**寄存器 A 操作数**，例如
`QGMMA.64x128x32.F32.E4M3.E5M2 R152, R216, gdesc[UR20], R152`
（`D=R152, A=R216(寄存器), B=gdesc[UR20], C=R152`）。这正是 CUTLASS 的
`MMA_64x{32,64,128}x32_F32E*M3E*M3_RS_TN`（见 `cute/arch/mma_sm90_gmma.hpp`），
A 片段类型为 `uint32_t[N/8]`、`ALayout_64x32`（`mma_traits_sm90_gmma.hpp`）。

### 102.3 冒烟：`ldmatrix.x4` 产出 wgmma RS 的 A 片段（PASS）

`src/fp8/fa_bwd_fp8_wgmma_rs_smoke.cu`：M=64,N=32,K=128、128 线程、A 行主序 [64][128]、
B SW128 K-major [32][128]，A 用 `ldmatrix.x4`（与 `mma_block` 同款 `arow/acol` 模式）装进
4×u32，跑 `wgmma.mma_async...m64n32k32...{%0..%15}, {%16..%19}, %20, p, ...`，与 CPU fp32 参考逐元素比：

```
wgmma RS_TN m64n32k32 e4m3×e4m3 A(ldmatrix,x4) vs CPU: max_abs=0.000e+00  PASS
wgmma RS_TN m64n32k32 e5m2×e4m3 A(ldmatrix,x4) vs CPU: max_abs=0.000e+00  PASS
```

⇒ **A 片段 = mma.m16n8k32 的 A 片段在 4 warp 上铺 64 行**，可直接由现有 `mma_block` 的
`ldmatrix_x4` 取数模式喂给 wgmma RS。原始输出
`src/fp8/fa_bwd_fp8_wgmma_rs_smoke.out.txt`。

### 102.4 修正后的「真阻塞」与下一步

RS 只解决 A。每个非平凡 GEMM 仍有**恰好一个操作数需要（逐字节）转置**：
dV=PᵀdO（需 dOᵀ）、dK=dSᵀQ（需 Qᵀ）、dQ=dS·K（需 Kᵀ）。fp8 的 `ldmatrix.trans` **只交换
8×8 的 b16（2 字节）配对方向、不做逐字节转置**（O4b 已证）——所以不能靠 `.trans` 直接得到
dOᵀ/Qᵀ/Kᵀ 的 A 片段。TE 的 SASS 给出它真正的做法：**`STSM`（store-matrix，含
`STSM.MT88.4` 转置写）+ `LDSM`（含 `MT88.4` 转置读）** 共 44 条——即用矩阵搬运指令在寄存器
片段与 smem 之间做**配对粒度的转置搬运**（fold 出来的 P/dS 片段、以及 Q/K/dO 的转置副本都在
寄存器里用 `stmatrix` 落成 wgmma 可消费的 smem tile），而不是逐字节 scatter（那正是 O4b/F6
判负的 `+20KB smem + scatter` 方案）。**路径已明确、工程量中等偏大**：
① 冒烟 `stmatrix`（含 trans）能把 mma 累加器片段写成 SW128/none canonical B/G 操作数；
② 把 GEMM3/4/5 逐个换成 RS wgmma（A=ldmatrix 寄存器片段，B=stmatrix 落盘的 fold 操作数）；
③ 顺带评估去掉 `Qp/dOp`（17.4KB）后能否冲 **4 CTA/SM**（当前 168 regs/74.8KB→3 CTA/SM）。
默认路径数值逐位不变。见 `docs/08` §5.88、ROADMAP「下一步」。

**原始输出**：`src/fp8/fa_bwd_fp8_p174_sass_te_vs_ours.out.txt`（逐 opcode 对照）、
`src/fp8/fa_bwd_fp8_wgmma_rs_smoke.out.txt`（冒烟 PASS）。

## 103. F3b 主体第一步（第 175 轮，O80）：fp8 **逐字节转置**（`ldmatrix.x4.trans` + `PRMT`）+ **wgmma RS GEMM3** 冒烟 —— 打通 GEMM3/4/5 的 wgmma 操作数构造

### 103.1 结论一句话

第 174 轮（§102）用 TE SASS 发现 GEMM3/4/5 的「需转置操作数」可以经 `ldmatrix` 进寄存器 A
（RS_TN），但每个 GEMM 仍各有**一个逐字节转置**需求（dOᵀ/Qᵀ/Kᵀ），而 fp8 的 `ldmatrix.trans`
只交换 b16 配对方向。本轮把 **TE 真正的做法**（`LDSM.MT88.4` 转置读 → `PRMT` 逐字节重排 →
`STSM.M88.4` 矩阵写）用最小复现钉死，并跑到 wgmma RS 端到端，**全部逐字节 / max_abs=0 PASS**。
于是 GEMM3/4/5 上 wgmma 的**操作数构造问题已解**，剩下的是把它接进主 kernel + 几何调整。

### 103.2 TE SASS 复核（`ncu --page source --print-source sass`）

用 `harness/te_fp8_ncu.py` 跑 TE FP8 反向并抽 SASS，GEMM3/4/5 区域的指令序列为：

```
LDSM.16.MT88.4 R16, [R29]          # ldmatrix.x4.trans（转置读一个 8×8 b16 = 8 行×16 fp8）
PRMT R28, R16, 0x6420, R17         # 逐字节重排（修正 b16 配对方向）
PRMT R30, R16, 0x7531, R17
STSM.16.M88.4 [R64], R28           # stmatrix.x4（非转置）落盘成操作数
...
QGMMA.64x128x32.F32.E4M3.E5M2 R152, R216, gdesc[UR20], R152   # A=R216 寄存器(RS_TN)
LDSM.16.M88.4 R216, [R216+0x20400]                            # 非转置读回 A 片段
```

即「转置读 → 字节重排 → 矩阵写」，A 再用非转置 `ldmatrix` 读回喂 RS QGMMA。TE 用 PRMT
（而不是逐字节 scatter）解决 b16 配对方向——正是 O4b/F6 判负的「+20KB smem + scatter」之外的路。

### 103.3 逐字节转置的映射与冒烟（`src/fp8/fa_bwd_fp8_stmatrix_smoke.cu`）

对 [R][C] fp8 行主序 tile，取一个 **8 行 × 64 fp8** 块（= 4 个并排 8×8 b16 矩阵），一个 warp
`ldmatrix.x4.trans`（lane L：matrix=L>>3、row=L&7、地址 `&src[r0+(L&7)][c0+(L>>3)*16]`）。
推导并实测的 lane 映射（`p=L&3, q=L>>2`）：

```
reg_i  = ( Xb16[2p][8i+q] , Xb16[2p+1][8i+q] )          # 低/高 16-bit
       = ( (X[2p][2k],X[2p][2k+1]) , (X[2p+1][2k],X[2p+1][2k+1]) ) ,  k=8i+q
```

用 `__byte_perm(reg, reg>>16, 0x5140)` 得到：

```
word = ( X[2p][2k], X[2p+1][2k] | X[2p][2k+1], X[2p+1][2k+1] )
     = ( Y[2k][2p], Y[2k][2p+1] | Y[2k+1][2p], Y[2k+1][2p+1] )      # Y[c][r]=X[r][c]
```

于是把低 16-bit 写成 `Y[2k][2p..2p+2]`、高 16-bit 写成 `Y[2k+1][2p..2p+2]`（行主序或 SW128）。
`__byte_perm` 的 selector 是 **`0x5140`**（byte0←a0、byte1←b0、byte2←a1、byte3←b1）；写反成
`0x5410` 会退化成恒等、只错一半字节。**四种 shape `[128][64]→[64][128]`、`[64][64]`、
`[64][128]→[128][64]`、`[32][128]→[128][32]` 全部 mismatches=0 PASS**。

### 103.4 wgmma RS GEMM3 端到端（同一冒烟第二阶段）

`gemm3_wgmma_kernel`：`C[j][d] = Σ_m P[m][j]·dO[m][d]`（dV 形状，BM=128 / BN=64 / HD=64）。
- A = Pᵀ[64][128]：把源 P[128][64] 用上面的逐字节转置写成**行主序 K-major**（m 连续），
  再用 `ldmatrix.x4`（**非转置**）取 A 片段（O79 已证 = `ALayout_64x32`）。
- B = dOᵀ[64][128]：逐字节转置**直接写进 SW128 tile**（K=BM=128，行 128B = 一个 SW128 atom 整行），
  描述符 `make_desc_sw128_fp8(sw128_k32_addr(base,s), sbo=1024)`。
- `wgmma.mma_async.m64n32k32.f32.e4m3.e5m2`，4 个 k=32 步 × 2 个 n=32 块（N=HD=64）。

输入取 {-3..3} 整数（e4m3/e5m2 精确），CPU 参考直接整数乘加：

```
wgmma RS GEMM3 dV[j][d]=Σ_m P[m][j]·dO[m][d] (BM128 BN64 HD64,
  A=Pᵀ ldmatrix, B=dOᵀ SW128字节转置): max_abs=0.000e+00 bad=0  PASS
```

**SASS**（`ncu --page source`）：`gemm3_wgmma_kernel` = **8×`QGMMA.64x32x32.F32.E4M3.E5M2`**
（R56/R60/R64/R68 为寄存器 A 操作数）+ **0×HMMA** + `LDSM.MT88.4`（转置读）+ `PRMT 0x5140`；
ptxas **74 regs / 0 spill / 0 stack**。

### 103.5 对 F3b 的意义与剩余工作

- **正结果**：fp8 的逐字节转置用 `ldmatrix.trans + PRMT` 可解且逐字节正确，转置操作数可直接落成
  wgmma 消费的 SW128；RS wgmma 端到端数值精确。**「GEMM3/4/5 上不了 wgmma」的两道死结
  （ISA 无转置 + B 需物理转置）已全部打开。**
- **剩余工程量**（下一步 F3b 主体）：
  1. 把 `transpose_store` 接进 `fp8_mma_body` 的 GEMM3/4/5（替换 `mma_block_bt` 的
     `dOp/Qp/Kp` 配对读 + `ldmatrix.x2.trans`）；
  2. **几何调整**：GEMM3/4 的 M=BN，wgmma 最小 m64 ⇒ BN 需提到 **64**（或把两个 KV 块配成 M=64）；
     GEMM5 的 M=BM=64 已满足。BN=32→64 会改 smem/寄存器账（需重新核算 CTA/SM）；
  3. fold 的 P/dS 量化输出顺带落成 wgmma 操作数布局（目前 Ap/dS3 已是 K-major A，dS2 亦然）。
- 默认路径一行未改、数值逐位不变。**原始输出**：`src/fp8/fa_bwd_fp8_stmatrix_smoke.out.txt`
  （逐字节 + GEMM3 PASS）、`..._sass.out.txt`（QGMMA 直方图）、`..._ptxas.out.txt`（74 regs）。

## 104. F3b 主体（第 176 轮，O81）：fp8 **GEMM5（dQ）切到 wgmma RS** —— 正结果，默认

**动机**：O79/O80 已把「GEMM3/4/5 上 wgmma」的两道死结打开（RS 允许 A 在寄存器；逐字节转置用
`ldmatrix.x4.trans + PRMT 0x5140`）。但默认 fp8 main 的五个 GEMM 里只有 GEMM1/2 是 QGMMA，
GEMM3/4（dV/dK，M=BN=32）与 GEMM5（dQ，M=BM=64）仍是 `HMMA.16816 + LDSM`。本轮先把
**唯一满足 wgmma 最小 m64 的 GEMM5** 落进真实 `fp8_mma_body`，并解决它相对 GEMM1/2 的两个新问题：
（i）B=Kᵀ，K 归约维 = BN=32 < 128B，**用不了 SW128 描述符**；（ii）真实 K 以 **SW128** 存在 smem，
需要从 SW128 源做转置。

### 104.1 三个新钉死的数据通路事实（`fa_bwd_fp8_wgmma345_smoke.cu`，全 max_abs=0）

1. **no-swizzle（`layout_type=0`）K-major 描述符**：CUTLASS canonical INTERLEAVE 布局是
   `((8,n),2):((1,SBO),LBO)`（单位 uint128=16B）——**8 行的 stride 恒为 1 个 uint128**，
   即 core matrix 内 8 行各 16B 相邻、K core 与行组的 stride 分别是 LBO/SBO。元素 (r,k) 字节偏移
   `16*((r&7)+SBO*(r>>3)+LBO*(k>>4))+(k&15)`；K=32（BN=32）时自然编码 **LBO=8、SBO=16**。
   冒烟扫 4 组：`(LBO_u=8,SBO_u=16)` 与 `(8,32)` **max_abs=0 PASS**，`(16,16)/(2,16)` fail
   ⇒ 描述符正确、且「K-major 恒 LBO=16B」只对 SW128 成立（Interleave 必须按上式）。
2. **从 SW128 源逐字节转置**：把 O80 的 `transpose_store` 的每 lane 源地址由 `(r)*C+c` 改成
   `sw128_off_fp8(r,c,C)`（PRMT 数学不变），`[32][128]/[64][128] → [128][32]/[128][64]`
   **mismatches=0 PASS** ⇒ SW128 只在 16B chunk 粒度置换，`ldmatrix.x4.trans` 照常拿到逻辑行。
3. **端到端 GEMM5（dQ）**：A=dS2[64][32] e5m2 行主序（stride 48）经 `ldmatrix.x4`；
   B=Kᵀ[128][32] no-swizzle（由 (2) 从 SW128 K 造）；`wgmma.m64n32k32.e5m2.e4m3` ×4：
   `max_abs=0.000e+00 bad=0 PASS`。

### 104.2 落进真实 `fp8_mma_body`（`-DFA_WGMMA5`，现默认 1）

- **门控**：`constexpr bool kWg5 = FA_WGMMA5 && WGMMA && KVTMA && HD==128 && BN==32 &&
  !DQONLY && !DET && kRegDq`（仅默认 Hopper fp8 D=128 kvtma 快路；sm_90/mma、MLA、DET、
  DQONLY 一律回退，逐字不变）。
- **Kt 重建**：KVTMA 路径的 **Kp（配对布局）构建**换成
  `transpose_sw128_to_inter<BN,HD>(Kp_as_u8, Ks_stage, wid, lane)`——从当前 SW128 K stage
  逐字节转置成 Kᵀ 的 INTERLEAVE K-major，**复用 Kp 的 4352B 缓冲**（≥ HD*BN=4096）⇒ **smem 零增长**。
  两处（循环外来首 tile、循环内建下一 tile）都改。**关键坑**：generic 写完 smem 必须
  `fence.proxy.async.shared::cta`（`bulk_reduce_fence()`）才能被 wgmma 的 async proxy 读到，
  否则 dQ 出现 **O(1) 的偶发错**（首版 O64/O41 A/B 的 dq 差 2.36/5.14，补 fence 后回到 1e-7）。
- **GEMM5**：A=dS2 经 `ldmatrix.x4` 装 4×u32，`wgmma.m64n32k32_rs_e5e4` ×(HD/32=4)，累加器
  mapping 为 CLayout_64x32（warp wid 持行 [16wid,16wid+16)，列 = nn*32+j*8+c2+(q&1)）；
  折算累进取 `dqacc5[4][4][4]`，循环末按该 mapping 每元素一次 `red_add2`（与 mma 版同口径）。
- 默认路径（`FA_WGMMA5=0`）逐字不变。

### 104.3 数值（护栏全过）

| case | ours vs fp32 ref relL2 (dq/dk/dv) | max_abs (dq/dk/dv) | 结论 |
|---|---|---|---|
| S512 H16 causal | 8.179% / 8.298% / 6.341% | 2.426e-1 / 2.972e-1 / 3.733e-1 | 与 `FA_WGMMA5=0` **逐位打印相同** |
| S4096 H16 causal | 8.149% / 8.263% / 6.489% | 2.635e-1 / 2.644e-1 / 3.216e-1 | 同上；`wg5_1 vs wg5_0` dq 2.18e-4（累加次序）、dk/dv ~1e-6（原子噪声） |
| S1024 H32 kv4 causal | 8.15% / 8.22% / 6.32% | — | GQA 亦过 |
| S1024 H16 full | 8.11% / 8.23% / 6.71% | — | 非 causal 亦过 |

`--ci --dtype fp8 --hopper`：单/两文件一致性 gate **worst 7.629e-6 OK**、`--check docs/04` OK（198 行）、
29 个 fp8 case vs-ref 与历史**打印相同**（如 S4096 `2.635e-1/2.644e-1/3.216e-1`）。

### 104.4 性能与 ncu（同 binary / 同 session A/B，S4096 H16 causal）

| 口径 | wg5=0（mma GEMM5） | wg5=1（wgmma RS GEMM5） | 比 |
|---|---|---|---|
| event main | 1.5564 ms | **1.4810 ms** | **1.051×** |
| event total | 1.7885 ms / 76.84 TF | **1.7235 ms / 79.74 TF** | **1.038×** |
| ncu Duration | 1.56 ms | **1.49 ms** | 1.047× |
| ncu `smsp inst` | 722.28 M | **641.86 M** | **−11.1%** |
| ncu HMMA inst | 27.69 M | **20.23 M** | **−26.9%** |
| ncu `lts read` 扇区 | 30.91 M | **28.03 M** | −9.3% |
| ncu `lts red` 扇区 | 114.52 M | **114.52 M** | **不变** |
| ncu regs | 168 | 168 | 不变 |
| ncu L2 tput | 78.27% | 79.11% | — |

**判决**：**正结果，默认开启**。收益纯粹来自**指令路径**（GEMM5 的 16×HMMA/线程组 → 4×QGMMA，
并省掉 Kp 配对的 `__byte_perm` 构建），`red` 一字不变 ⇒ 与「降 L2 搬运量」正交，是 F3b 指令层的
一步。S4096 total 相对 TE FP8（纯反向 `0.3025 ms / 908.7 TF`）由 **5.93× → 5.70×**。

### 104.5 剩余（F3b 主体未完成部分）

- GEMM3/4（dV/dK）的 M=BN=**32** 仍 < wgmma 最小 m64 ⇒ 需 **BN=64**（或把两个 KV 块凑成 M=64）；
  BN 32→64 会翻倍 `dVacc/dKacc` 累加器、顶到 255 寄存器/2 CTA/SM（p160 已判负）⇒ 需先解决寄存器账。
- WS 完整化（producer/consumer + 更深 mbarrier 流水）仍待做。
- **原始输出**：`src/fp8/fa_bwd_fp8_wgmma345_smoke.{cu,out.txt}`、
  `src/fp8/fa_bwd_fp8_o81_ab_wgmma5_{0,1}_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o81_ncu_wgmma5_{0,1}_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o81_ci_fp8.out.txt`、`src/fp8/fa_bwd_fp8_o81_baseline_fa3_te.out.txt`。

## 105. F3b 主体续（第 177 轮，O82）：fp8 **GEMM3/4（dV/dK）切到 wgmma RS** —— **负结果，默认关**

**动机**：O81 已把 GEMM5（dQ）落进 wgmma RS，默认 fp8 Hopper main 的五个 GEMM 里只剩 GEMM3(dV)
与 GEMM4(dK) 还是 `HMMA.16816 + LDSM`。本轮把这两条也换到 wgmma RS，把 F3b 的「全 GEMM wgmma」
再往前推一块。

**障碍与绕法**：GEMM3/4 的 M=BN=32 < wgmma 最小 m64。TE 的解法是 BN=64（自然 m64），但 p160/O21
已判 BN 32→64 在本卡为负（累加器翻倍撞 255 regs）。本轮改走**零填充**：A 的 M 维补到 64，warp 2/3
的 A 寄存器置 0、输出行 32–63 丢弃——只多花一半张量核周期（tensor 利用率 <20%，非瓶颈）。

**新数据通路（两个可复用 helper）**：
1. `noswz_k_off_c(d,m,C) = (m>>4)*(C*16) + d*16 + (m&15)` + `transpose_sw128_to_noswz<R,C>`：
   把 SW128 源 [R][C] 逐字节转置成**紧凑 no-swizzle K-major** [C][R]。与 O81 的
   `inter_k_off_fp8(...,16,8)` 不同——后者 LBO=8 只对 **K≤32** 无冲突（K=64 时 `8*(k>>4)`
   的第 4 位与 `SBO*(r>>3)` 撞），故本轮 K=BM=64 用「K-core 主序、块内 C 行连续 16B」的新布局，
   描述符 `make_desc_noswz_fp8(addr, /*lbo_u=*/C, /*sbo_u=*/8)`；k-step `s` 前进 `s*2*C*16`
   字节、n-tile `nn` 前进 `nn*4*8*16=nn*512` 字节。
2. `wgmma_m64n32k32_rs_e4e5`：新增 e4m3×e5m2 的 RS_TN 封装（GEMM3 A=Ap e4m3、B=dOᵀ e5m2）；
   GEMM4（A=dS3 e5m2、B=Qᵀ e4m3）复用 O81 的 `wgmma_m64n32k32_rs_e5e4`。

**落地（`-DFA_WGMMA34`，默认 0）**：门控
`FA_WGMMA34 && WGMMA && KVTMA && HD==128 && BN==32 && !DET && !DQONLY && !ILV34 && !BULKRED && !R4`。
Q/dO 的配对缓冲（Qp/dOp）在 kWg34 时改存 Qᵀ/dOᵀ 紧凑 no-swizzle（8192B ≤ qp_bytes 8704B，
**smem 零增长**），由 `transpose_sw128_to_noswz<BM,HD>` 从 SW128 Qs/dOs 重建 + `fence.proxy.async`；
A=Ap/dS3 走 `ldmatrix.x4`（K=64 分两个 k-step）；累加器按 **2 个 n-tile 一组**（`acc[2][16]`），
两组各一次 wait 以压低寄存器；epilogue 只在 wid<2（有效 32 行）做 `red_add2`。

**数值（正确）**：S4096 vs fp32 ref relL2 dq/dk/dv = **8.149 / 8.263 / 6.489%**、max_abs
**2.635e-1 / 2.641e-1 / 3.218e-1**，与 `FA_WGMMA34=0` **打印完全相同**（护栏内）；A/B
`wg34_1 vs wg34_0` dq max_abs 1.19e-7（逐位）、dk 9.13e-4 / dv 2.28e-3（red 原子次序）。
S512 = 2.426e-1/2.973e-1/3.726e-1。`--ci --dtype fp8 --hopper` 全绿（gate 7.629e-6、docs check OK 198 行）。

**性能与 ncu（同 binary A/B，S4096 H16 causal）**：

| 口径 | wg34=0（mma GEMM3/4） | wg34=1（wgmma RS M-填充） | 比 |
|---|---|---|---|
| event main | 1.4738 ms | 1.5289 ms | **0.964×** |
| event total | 1.7119 ms / 80.28 TF | 1.7694 ms / 77.67 TF | **0.967×** |
| ncu Duration | 1.49 ms | 1.56 ms | 1.047×（更慢） |
| ncu Executed Instructions | 641.86 M | **594.83 M** | **−7.3%** |
| ncu L1/TEX tput | 74.44% | **67.55%** | −6.9pp |
| ncu L2 tput | 78.81% | **75.45%** | −3.4pp |
| ncu Compute (SM) | 45.06% | 40.30% | — |
| ncu No Eligible | 53.22% | **58.19%** | 延迟 bound 恶化 |
| ncu regs | 168 | 164 | — |

**判决（负结果，默认关）**：指令路径确实更优（−7.3% inst、L1/L2 吞吐降），但 **Duration 反升**。
根因：本 kernel 是 **L2 `red` bound（~75%，wgmma 对 `red` 一字不减）+ 延迟 bound**（No Eligible
升高）——零填充令张量核做 2× 无用功，Q/dO 的逐字节转置与 wgmma `fence/commit/wait` 又加延迟，
而被省下的 HMMA/LDSM 本就不是瓶颈。⇒ **F3b 的「无 BN=64 时上 GEMM3/4 wgmma」子路线判负**；
要把 GEMM3/4 真正 wgmma 化仍须 **BN=64（自然 m64）**，而那要先解寄存器账（256/384 线程摊累加器），
与 F7/p160 同源。默认路径一行未改、数值逐位不变。

**原始输出**：`src/fp8/fa_bwd_fp8_o82_ab_wg34_{0,1}_s4096.out.txt`、
`src/fp8/fa_bwd_fp8_o82_ncu_wg34_{0,1}_s4096.out.txt`。

---

## 106. F3b① 收口（第 178 轮，O83）：fp8 GEMM3/4「真 m64」三条路逐条资源核算 + `red` 成本分解

承接 O82 的「下一步候选 ①：真 m64 化 = BN=64 + 多 warpgroup 摊累加器；先解寄存器账」。本轮
**不动默认路径**，做三件事：① 用最新 ncu 把默认 fp8 `kvtma` main 的 bound 再钉一次；② 把
「GEMM3/4 真 m64（不零填充）」的**三条可行形式逐条做 smem/寄存器/访存核算**，证明在本卡
3 CTA/SM 工作点下**全部不可行**；③ 加一个编译期诊断 `FA_RED_STORE` 把「跨 CTA 原子归约」的
代价从「epilogue 写流量」里分出来。结论：**candidate ① 收口为「本卡资源墙 + 张量核空转」双锁死**。

### 106.1 默认 main 的 bound（ncu，S=4096 H16 causal，`--only=kvtma`）

| 指标 | 值 | 判读 |
|---|---|---|
| L2 throughput | **79.26%** | 墙 |
| L2 `red` 扇区 | **114,524,160** | 占 L2 总扇区（114.5M+28.0M+0.39M=143M）的 **80%** |
| L2 `read` 扇区 | 28,045,949 | 20%（K/V 跨 m-block 重读 + Q/dO） |
| L2 `write` 扇区 | 386,265 | 忽略 |
| DRAM throughput | **4.41%** | 纯 L2（命中 97%），非 DRAM |
| `sm__pipe_tensor_cycles_active` | **11.15%** | 张量核空转 |
| `sm__issue_active` | 46.66% | 一半发射口空 |
| warps active | 18.33%（3 CTA/SM、12 warp/SM） | 低 |
| regs / CTA/SM | 168 / 3 | `__launch_bounds__(128,3)` 顶格 |
| stall | `short_scoreboard 1.81` + `wait 1.55` | 内存/依赖延迟 |

⇒ **本 kernel 是 L2 `red`-bound（80%）+ 低 occupancy 延迟 bound；张量核只有 ~11% 忙**。
**这条本身即给 F3b 全族判了死缓**：把 GEMM3/4 的 `mma.sync` 换成 `wgmma` 只改指令路径、
对 `red` 一字不减，而张量核并非瓶颈——O81（GEMM5 wgmma）能拿 5% 只因它顺手去掉了 `Kp` 的
smem 构建（`lts read −9.3%`），O82（GEMM3/4 wgmma）连这点都没有、反被零填充 2× 无用功拖累。

### 106.2 「真 m64」三条形式的资源核算（本卡 3 CTA/SM 上限 = 77,482B、170 regs）

GEMM3/4 的 M=C（dV/dK 的 KV 行数）。wgmma 最小 M=64，而默认 `BN=32` ⇒ M=32。三条能在**不
零填充**下凑出 m64 的形式：

| 形式 | M 来源 | smem 增量 | 其余代价 | 判决 |
|---|---|---|---|---|
| **(a) 配对两 KV tile** | A=[Pᵀ₀;Pᵀ₁]（64 行） | **+4,096B**（存被配对 tile 的 `Ap`/`dS3`，`[BN][BM]`=2×2048） | A 跨两 tile 非连续（可分 warp 取，可行）；V 需双缓冲 | **74.8+4=78.8KB > 77.5 ⇒ 掉 2 CTA/SM**（p160/O21 已证 2 CTA/SM 可复现地为负） |
| **(b) 转置 GEMM** dVᵀ=dOᵀ[128][BM]·P | M=HD=128（自然） | 0（复用 `Qp`/`dOp` 缓冲；B=`Ap` 改 no-swizzle 布局） | **输出取向翻转**：`acc`（行=d、列=j）写 `dv[j][d]` ⇒ 相邻 lane 写 stride=`Hkv·HD`=512B 的**不同行**，red 扇区暴涨（每 warp-inst 8→32 扇区） | 负（red 墙恶化） |
| **(c) BN=64** | M=BN=64（自然 m64） | **+~45KB**（`Ps`/`Ss`=`[64][BN+5]`→17.7KB 各、`Ks`/`Vs` 翻倍、`dS2` 翻倍） | GEMM1/2 累加器翻倍 | smem≈120KB ⇒ **1 CTA/SM**，比 (a) 更差 |

⇒ **没有任何一条真 m64 形式能停在 3 CTA/SM**；而 2 CTA/SM 在本卡（O21/p160/F6 三度复现）
稳定为负。**candidate ① 至此按资源账收口**，与「阻塞」里 F6（去 Qp/dOp 冲 2 CTA 不可行）、
F7（多 owner 撞 255 regs）同源。真要 GEMM3/4 wgmma 化，只能**换卡**（更大 smem/regfile）或
**256/384 线程多 warpgroup**（把 `dVacc/dKacc`/`dqacc` 摊到更多线程，需先破 fp8 `wgmma` 无转置
操作数——O80 已解，但工程量大、且仍受同一 L2 墙）。

### 106.3 `red` 成本分解（诊断 `FA_RED_STORE`，**数值错误，仅诊断**）

把 `red_add2` 的 `atomicAdd(float2*)` 换成 **plain store**（`*(float2*)p = ...`，last-writer-wins），
写地址与字节数完全不变，只去掉原子语义：

| 版本 | main | 判读 |
|---|---|---|
| 默认（原子） | 1.493 ms | — |
| `FA_RED_STORE=1`（plain store） | **1.368 ms（1.09×）** | 原子 RMW 的成本 ≈ **0.125ms(~8%)** |
| O42（短路整个 dK/dV epilogue，无写） | 0.94 ms | epilogue 写流量本身 ≈ **0.43ms(~29%)** |

⇒ `red` 的成本里**大头是写流量本身（~29%）、原子语义只占 ~8%`**——与 O67（加宽归约宽度
`red_add4` 无效）、O42（TMA bulk-reduce 无效）一致：**L2 `red` 由「每输出元素被多少个 CTA
贡献」唯一决定，与归约指令/宽度/机制无关**。⇒ 唯一剩余杠杆仍是 **工作划分**（减少贡献 CTA 数），
即 F7 的 KV-owner/多 warpgroup 方向。

**新观察（留 backlog，不本轮做）**：GQA/MQA 下同一 KV 头被 `H/Hkv` 个 Q 头共享，若一个 CTA
内**先跨 Q 头本地累加 dK/dV 再原子**，`red` 可 ÷`(H/Hkv)`（MQA 最多 ÷64）。**阻塞**：dQ 的
寄存器累加 `dqacc`（`kRegDq`）要求「一个 CTA = 一个 Q 头」跨 nt 保持；跨 Q 头合并会使 `dqacc`
需 ×`(H/Hkv)` 份，撞寄存器墙。⇒ 与 F7 同墙，仅对 GQA/MQA 有效、对 MHA 零收益。

### 106.4 默认路径一行未改

`FA_RED_STORE` 默认 0；两文件 device 与单文件逐字同步。数值对拍（默认构建，S4096）：
`ours vs fp32 ref` **2.635/2.644/3.216e-1**（与历史逐位一致）、`ours vs TE` 同量级。
`--ci --dtype fp8 --hopper` 全绿。

**原始输出**：`src/fp8/fa_bwd_fp8_o83_ncu_main_s4096.out.txt`（ncu）、
`src/fp8/fa_bwd_fp8_o83_baseline_s4096.out.txt`（timing+数值）、
`src/fp8/fa_bwd_fp8_o83_redstore_s4096.out.txt`（`FA_RED_STORE` A/B）。

## 107. O84（第一百七十九轮）：fp8 `head_dim=256` 主 kernel 默认切 **wgmma** —— 正结果，默认

### 107.1 动机与现状

O76（第 171 轮）把 fp8 反向的 `head_dim` 覆盖补到 **256**，但当时 host 明确让 `D=256`
走 **非 wgmma 的 mma 主 kernel**（`launch_bwd_main<256,64,32,false,false,...>`），注释写
「无 fp8 wgmma——它只做 HD=128」。原因是 device 里有一条保守的编译期锁：

```cpp
static_assert(!WGMMA || (HD == 128), "WGMMA 主 kernel 目前只做 HD=128");
```

而 O76 自己在「下一步」里就点名了这条路：**「`D=256` 的 fp8 main 接 wgmma/TMA（受 fp8
wgmma 只做 HD=128 / SW128 atom 的 HD 约束）」**。本轮落实它——先只做 **wgmma（非 TMA）**。

**关键事实：fp8 的 SW128 K-major 数据通路 helper 本就支持 `HD` 为 128 的整数倍。**
`sw128_off_fp8(row,k,K)` 布局是 `[row/8][k/128][8][128]`（atom 1024B），描述符
`SBO=(K/128)*1024`、k32 步进 `sw128_k32_addr = (s>>2)*1024 + (s&3)*32`。当 `K=HD=256` 时：
- canonical 布局 = `[row/8][2 k-blocks][8][128]`，row-group 跨步 = `2*1024 = 2048 = SBO`；
- 第 2 个 128-block 的基址偏移 = `+1024`，恰是 `sw128_k32_addr` 的 `(s>>2)*1024`（s=4..7）。

`wgmma_mn32_issue`（GEMM1/2）与 `wgmma_qkt64_fp8`（LSE）的 K 循环都写的是 `s < HD/32`，
天然对 `HD=256` 正确。**所以本轮 device 侧只放开了那条 `static_assert`（改为
`HD==128 || HD==256`），host 把 `D=256` 的分派接到 `launch_bwd_main<256,64,32,false,true>`
（`WGMMA=true`，仍走 cp.async 载入，非 TMA），加 CLI `--d256wgm=0/1` 供同 binary A/B。**

### 107.2 四个 D=256 case 的同 binary A/B（两文件版，Hopper 构建，event iters=50）

| case | main wgmma | main mma | main × | total wgmma | total mma | total × | vs ref max_abs（wgmma） |
|---|---|---|---|---|---|---|---|
| S1024 H8 causal | 0.2450 ms | 0.2806 ms | **1.145×** | 0.3064 ms | 0.3603 ms | **1.176×** | 2.633e-1 / 2.797e-1 / 3.572e-1 |
| S2048 H8 causal | 0.7667 ms | 0.9883 ms | **1.289×** | 0.8891 ms | 1.1289 ms | **1.270×** | 2.227e-1 / 2.839e-1 / 3.586e-1 |
| S1024 H8 full | 0.3489 ms | 0.4438 ms | **1.272×** | 0.4524 ms | 0.5518 ms | **1.220×** | 5.012e-2 / 5.569e-2 / 4.044e-2 |
| S1024 H16 **GQA kv4** | 0.4537 ms | 0.5496 ms | **1.211×** | 0.5443 ms | 0.6606 ms | **1.214×** | 2.477e-1 / 4.412e-1 / 6.152e-1 |

**数值**：`wgmma vs mma` 的 max_abs 差均在 fp8 原子次序噪声内（≲1e-3）；与 O76 记录的
mma 结果同量级（causal 2.630/2.795/3.589e-1、full 4.972/5.571/4.092e-2、GQA 2.477/4.455/6.157e-1）。
**单/两文件一致性**（`--ci --dtype fp8` 的 gate）：16 个定长 case worst `6.676e-6` OK，
`docs/04` 内嵌表 `--check` OK（198 行）。

### 107.3 ncu：为什么能快（S1024 H8 causal）

| | `--d256wgm=1`（wgmma） | `--d256wgm=0`（mma） |
|---|---|---|
| Duration | **261.9 µs** | 329.6 µs |
| `launch__occupancy_limit_shared_mem` | **2 CTA/SM** | 1 CTA/SM |
| `launch__registers_per_thread` | 238 | 254 |
| `sm__warps_active` | **11.78%** | 6.25% |
| `smsp__inst_executed` | **46.01 M** | 49.72 M |
| `lts__throughput` | 51.61% | 41.11% |
| `lts red / read`（扇区） | 13.37M / 1.69M | 13.37M / 1.80M |

**两条机制**：
1. **SW128 布局更紧凑跨过 2 CTA/SM 门槛**：`D=256` 的 mma 动态 smem = **117,760B**
   （`232448/117760 = 1.97 ⇒ 1 CTA/SM`）；wgmma 的 SW128 布局 = **115,712B**
   （`≤116,224B ⇒ 2 CTA/SM`）。warps_active 翻倍是本轮最大收益来源。
2. **指令路径**：`smsp__inst_executed` −7.4%；SASS 直方图 **16×QGMMA + 192×HMMA + 92×LDSM**
   （mma 版 `0×QGMMA + 256×HMMA + 124×LDSM`）⇒ GEMM1/2 从 `HMMA+LDSM` 换成 `QGMMA`。
   `lts red` 一字不变（工作划分没动，符合 O67/O83「red 由贡献 CTA 数决定」）。

`sm__pipe_tensor_cycles_active` 仍只 ~7%、`red` 仍是头号 L2 项 ⇒ 与 D=128 一样，**D=256 的
墙仍是 L2 `red` + 延迟**；wgmma 的收益主要来自「顺手把 occupancy 抬到 2 CTA/SM」。

### 107.4 默认路径与回归

- `-DFA_WGMMA`（Hopper）构建下 `D=256` 定长默认走 wgmma；`--d256wgm=0` 退回 mma，
  `-arch=sm_90`（无 `-DFA_WGMMA`）构建自动退化 mma（`#ifdef` 门控）。
- **D=128 / D=512 一行未改**：`static_assert` 放宽对已实例化的 `HD=128` 无影响；host 只改
  `D==256` 分支。`--ci --dtype fp8`（定长 16 case + 变长）全绿、`docs/04` `--check` OK。
- 单/两文件 device 逐字同步（`scripts/sync_onefile_device.py`）。

### 107.5 剩余（D=256）

`D=256` 仍未上 **4D-TMA**（Q/dO/K/V 仍 cp.async 载入）；且 `smem=115.7KB` 只够 2 CTA/SM。
后续可选：① 把 Q/dO/K/V 也接 4D-TMA（需 `NCH=HD/128=2` 个 box，参考 O74 的
`lse_mma_kernel_bal_tma` HD=512 分块）；② 继续降 smem 冲 3 CTA/SM。二者都属 F6/F3b 同源
的「减 L2 搬运/提 occupancy」方向。

**原始输出**：`src/fp8/fa_bwd_fp8_o84_d256_wgmma_ab.out.txt`（4 case × wgmma/mma A/B）、
`src/fp8/fa_bwd_fp8_o84_ncu_d256_s1024.out.txt`（ncu + SASS 直方图）、
`src/fp8/fa_bwd_fp8_o84_d256_onefile.out.txt`（单文件版）。

## 108. O85（第一百八十轮）：fp8 `head_dim=256` 的 Q/dO 切 **4D-TMA**（chunk-major）—— **中性（±2%），默认关（opt-in）**

> 落实 O84 的「下一步候选 ①」：把 `D=256` 的 Q/dO 也接 4D-TMA，继续减 L1/LDSM 指令与
> 搬运延迟。**结论：A/B 中性偏负 ⇒ 默认关**（`--d256tma=1` opt-in），O84 的 cp.async
> wgmma 档仍是 `D=256` 默认。

### 108.1 动机与难点：`D>128` 的 TMA 布局与主 kernel 的 SW128 不一致

- `D=128` 时 fp8 一行 = 128B = 一个 SW128 atom 的整行 ⇒ **一个 4D-TMA box 搬完整块**，
  物理布局就是主 kernel `sw128_off_fp8` 的 `[row/8][8][128]`（rg-major），O37 直接生效。
- `D=256` 时一行 256B、TMA 的 SW128 box 内维固定 128B ⇒ 需要 **2 个 box**，各写一块
  canonical `[BM][128]` 到 `Qs + c*CHUNK`（`CHUNK=(BM/8)*1024`）。于是物理布局变成
  **chunk-major `[k/128][row/8][8][128]`**，与主 kernel 的 rg-major 不同（LSE 的
  `wgmma_qkt64_fp8_chunked` 早就走 chunk-major，主 kernel 一直没做）。
- 本步新增两个纯整数 helper：`sw128c_off_fp8(row,k,nrows)`（chunk-major 偏移）与
  `sw128c_k32_addr(base,s,nrows)`（跨 chunk 的 k32 起始地址，chunk stride `(nrows/8)*1024`），
  以及 `wgmma_mn32_issue_cm<KIND,AROWS>`（**A 走 chunk-major、B 仍 rg-major**）——
  `D=128`（NCH=1）与旧 `wgmma_mn32_issue` **逐位相同**。

### 108.2 实现（仅 `HD>128` 生效；`D=128`/其它路径一行未改）

- **device**（单/两文件 device 逐字一致，`sync_onefile_device.py` `identical: True`）：
  - `fp8_mma_body` 的 TMA Q/dO 载入改成 `for (c<NCH_Q) tma_load_4d(Qs+c*CHQ, qmap, c*128, ...)`
    （同一 mbarrier、expect 总量 `QS_SZ`）；
  - Qp/dOp 重建读源用 `sw128c_off_fp8`（`kQChunk = TMA && HD>128`）；
  - GEMM1/2 在 `kQChunk` 时走 `wgmma_mn32_issue_cm`；
  - `fa_bwd_fp8_mma_qdtma_kernel` 的 `__launch_bounds__` 对 `HD=256` 放开到 2 CTA/SM。
- **关键坑（smem 64B）**：`smem_bytes_wgmma_tma` 原为 `smem_bytes_wgmma + 64`（放 qbar/dbar）。
  但这 64B 其实落在 `smem_bytes_wgmma` 的 1024B 对齐 slack 里；对 `HD=256` 多这 64B 会把
  launched smem 从 115,712→115,776B、**allocated 从 116.74→116.86 KB，直接把 2 CTA/SM 挤成
  1 CTA/SM**（ncu `occupancy_limit_shared_mem=1`、warps_active 11.7%→6.25%）⇒ 首版慢 **~9–17%**。
  改成 `HD>128 ? 0 : 64` 后恢复 2 CTA/SM。
- **host**（两文件 + 单文件同源）：`D==256` 也建 Q/dO 描述符（`make_lse_map_fp8(d_q8,H,S,D,B)`），
  新增 CLI `--d256tma=1` 与同 binary A/B；`D==256 && wgmma && d256_tma` 分派到
  `launch_bwd_main_qdtma<256,64,32,false>`。**默认关**。

### 108.3 数值（护栏全过；chunk-major 不改变累加口径）

`b1_s1024_h8_d256_causal_fp8`，relL2（`‖a−b‖/‖b‖`）vs fp32 ref：

| 梯度 | tma=1 | tma=0（O84 档） | tma1-vs-tma0 max_abs |
|---|---|---|---|
| dq | **8.332%** | 8.332% | 1.19e-7 |
| dk | **8.435%** | 8.435% | 4.77e-7 |
| dv | **6.464%** | 6.464% | 7.15e-7 |

max_abs 2.633e-1/2.797e-1/3.572e-1，与 O84 逐位一致；两档仅差跨 CTA atomic 次序（~1e-7）。

### 108.4 性能（同 binary A/B / ncu；结论：中性）

event（`[timing]`，两文件，同 binary 交替）：

| case | main tma=1 / tma=0 | total tma=1 / tma=0 |
|---|---|---|
| S1024H8 causal | 0.2425–0.2442 / 0.2446–0.2460 ms | 0.3003–0.3032 / 0.3054–0.3066 ms（**+1.3–1.7%**） |
| S1024H8 full | 0.2397–0.2428 / 0.2439–0.2466 ms | 0.3001–0.3045 / 0.3051–0.3077 ms（**+1.3%**） |
| S2048H8 causal | 0.7742–0.7758 / 0.7626–0.7662 ms | 0.8992–0.9029 / 0.8884–0.8944 ms（**−1.1%**） |
| GQA h16kv4 | 0.4662–0.4811 / 0.4552–0.4597 ms | 0.5650–0.5651 / 0.5430–0.5462 ms（**−3.7%**） |

ncu（S1024H8 causal，单 kernel 隔离）：Duration **245.5 vs 255.6µs（1.041×）**、
`smsp__inst_executed` **44.52M vs 46.01M（−3.2%）**、L1TEX 31.3 vs 30.5%、L2 55.2 vs 52.9%、
regs 238、smem 115.71KB、**2 CTA/SM（两档相同）**。⇒ TMA 确实省了指令、隔离运行略快，
但在真实 event 计时下 **小 shape 微正、大 shape/GQA 微负，净中性**（TMA 的 prologue
barrier 等待与 K/V cp.async 未重叠，抵消了指令收益；`red`/工作划分一字未动）。

### 108.5 判决与后续

- `D=256` 的 Q/dO TMA **中性 ⇒ 默认关**；O84 的 cp.async wgmma 档仍是默认。代码保留为
  opt-in（`-DFA_WGMMA -DFA_TMA` 构建 + `--d256tma=1`），供复用 chunk-major helper。
- **保留的净收益**：`sw128c_*` / `wgmma_mn32_issue_cm` 是一套**可复用的 chunk-major SW128 基建**
  （后续 K/V TMA for `D>128`、或 `D=128` 的 MLA 复用）。
- **后续**：若要让 `D=256` TMA 转正，需把 K/V 也 TMA 化并让 TMA prologue 与 K/V cp.async 重叠
  （当前 Q/dO TMA 等待在 K/V 之前，串行化）；但 K 双缓冲会顶穿 116KB/2-CTA 门槛，属 backlog。

**原始输出**：`src/fp8/fa_bwd_fp8_o85_d256_ab.out.txt`（4 case × tma1/0 event 计时）、
`src/fp8/fa_bwd_fp8_o85_ncu_d256_s1024.out.txt`（两档 ncu）、
`src/fp8/fa_bwd_fp8_o85_accuracy_d256.out.txt`（relL2 / A/B 逐元素）。

## 109. O86（第一百八十一轮）：GQA/MQA「跨 Q 头折叠 dK/dV」可行性收口 —— **结构性不可行（负结果）**，默认路径一行未改

> 落实 F4b 的「新 backlog：GQA/MQA 跨 Q 头本地累加 dK/dV 可把 `red` ÷`(H/Hkv)`」。本轮**不写
> 新 kernel**，而是先用 ncu 把这条路的**收益上界**钉死，再对三种可行的 loop order 做寄存器/smem
> 解析核算，最后与 O25（cluster 分布式归约）的实测负结果合起来给出判决。**结论：该杠杆在本卡
> 被「dQ 与 dK/dV 相反的 loop-order 偏好」锁死，MQA 的 `red÷64` 拿不到；需要换卡或硬件 scatter-reduce。**

### 109.1 收益上界：默认 fp8 main 的 `red` **正比于 Q 头数 H，与 Hkv 无关**

用 ncu 探针（`--kernel-name regex:kvtma --launch-count 1`，S=1024 causal，同 session）测了
5 个点。关键规律：**MQA（H64kv1）与 GQA（H64kv4）的 `red` 逐字节相同（32,833,536 扇区）**，
而 H32→16.42M、H40→20.52M 都精确落在 `red ≈ (H/32)×16.42M` 线上 ⇒ `red` 只由 **Q 头数**决定：

| case（S1024 causal，fp8 默认 kvtma） | H | Hkv | G=H/Hkv | `lts op_red` | `op_read` | Duration | L2% |
|---|---|---|---|---|---|---|---|
| MHA q32/kv32 | 32 | 32 | 1 | 16,416,768 | 4,414,171 | 242.7µs | 71.0 |
| GQA q40/kv8  | 40 | 8  | 5 | 20,523,168 | 5,302,221 | 291.1µs | 73.0 |
| GQA q64/kv4  | 64 | 4  | 16| 32,833,536 | 8,077,902 | 444.7µs | 76.3 |
| **MQA q64/kv1** | 64 | 1 | 64| **32,833,536** | 7,712,055 | 437.6µs | 77.5 |
| MHA q16 （S4096） | 16 | 16 | 1 | 114,524,160 | 28,013,759 | 1.50ms | 78.3 |

（`red/(red+read)` ≈ 0.80，与 `docs/03` §106 的 MHA 结论一致；`dK/dV` 部分按 O67/§106 约占
`red` 的 ~55–90%，其余是 dQ 的跨-part 原子。）⇒ **理论上把「G 个 Q 头的 dK/dV 偏和折叠成一次贡献」
可把 dK/dV 的 `red` 压掉 (G-1)/G**（MQA 最多 ×64）——这是 F4b 里唯一还没做、且上界最大的杠杆。

### 109.2 为什么没有可用的 loop order：dQ 与 dK/dV 的偏好相反

令 head 组大小 `G=H/Hkv`、`BM=64`、`BN=32`、`HD=128`。默认 kernel 每个 CTA = 一个
`(mblk, Q头 h, part)`，`nt` 内层循环、**dQ 用寄存器沿 `nt` 累加**（`kRegDq`，每元素单写者，
只 flush 一次）、`dK/dV` 每个 `nt` tile 原子写回。要折叠 head，只有两种嵌套：

**(A) `for nt { for head { 算 } }`（head 内层）** —— `dK/dV` 可在一个 tile 的寄存器累加器里
跨 head 累加、每 `nt` 只 flush 一次（**red 成功 ÷G**）。但 `nt` 在外层 ⇒
1. **dQ 丢掉寄存器累加**：每个 `(nt,head)` 都要原子写 dQ。dQ 的 `red` 从「每 (mblk,h,part) 一次」
   涨到「每 (mblk,h,nt) 一次」——对 MQA S1024 = `H×Σntiles = 64×272` 个 tile、每 tile `BM·HD/8=1024`
   扇区 ≈ **+17.8M 扇区**，直接把省下的 dK/dV `red`(~18M) 吃回去；
2. **Q/dO 必须按 `(nt,head)` 重载**（Q/dO 只在 head 维度复用、`nt` 在外层无法跨 `nt` 复用）：
   MQA 多出的 Q/dO 读 = `(G-1)·#(mblk,kv,nt)·(BM·HD·2/32)` ≈ `63×4352×512 ≈ 140M 扇区`
   ⇒ 比全部 `red` 还大 4×，**致命**。

**(B) `for head { for nt { 算 } }`（head 外层）** —— dQ 保住寄存器累加、Q/dO 每个 head 只读一次
（无放大）。但要把 dK/dV 跨 head 折叠，`dK/dV` 的累加器必须在整个 head 循环内**常驻**，
即覆盖本 CTA 的**全部 nt tile 的 KV 行**（`part` 内 `Σntiles × BN × HD` 个 fp32）：
ksplit=4 时 MQA 小 m 块也要 8×32×128×4B = **128KB/dK + 128KB/dV**，ksplit=1 更大（~S×HD×8B）
——**smem 放不下**；退一步只在寄存器里存一个 tile、每个 `nt` flush，那 head 就没有折叠（`red` 不变）。

**(C) 只折叠 dK/dV、dQ 单独一遍（两 kernel / DQONLY + DKV pass）** —— 两遍都要重算
GEMM1/2/softmax 并各自读一遍 Q/K/V/dO；第 2 遍的 loop order 仍是 (A) 或 (B)，回到同一墙。
（唯一正收益场景是「compute 免费」——本卡张量核确实只 10% 忙、重算不贵，但**读/`red` 才是墙**，
重算第二遍只增读。）

### 109.3 cluster 分布式归约为什么对 GQA 也不成立

唯一能绕开 loop order 的是 Hopper thread block cluster（O25，fp16/bf16 已实现并实测）：
CTA 照常各算各的 head，用 `red.shared::cluster.add.f32` 把**逐元素**偏和推进 leader 的 smem，
leader 每 tile 只发一次全局原子。O25 实测 `red` **精确减半**（51.9M→26.7M）但 **main 慢 7.4×**
（`long_scoreboard` 0.75→5.83），根因是「逐元素远程 smem 原子 + 每 tile leader 串行 flush」。
**关键**：cluster 把 N 次全局原子换成 N 次远程原子是 **1:1**，省下的只是「flush 次数」；
GQA 只是让 cluster 的 `G` 更大，而**每个 CTA 的逐元素远程原子数不变** ⇒ 远程原子总成本随
CTA 数（=`H`）线性增长，与 MQA 的 `H` 倍 `red` 同阶。**换来的全局 `red` 削减 ≤ 远程原子开销**，
O25 的 7.4× 负结果原样成立（非原子 `st.async` 方案需 `CL×2×[BN][HD]` smem，GQA 下更大，放不下）。

### 109.4 判决与解锁条件

- **判决**：GQA/MQA 跨 Q 头折叠 dK/dV 在本卡（74.8KB smem / 170 regs / 3 CTA/SM）**结构性不可行**：
  loop order (A) 的 dQ 原子化 + Q/dO 重载、(B) 的 dK/dV 常驻累加器、(C) 的两遍重读、
  cluster 的 1:1 远程原子，四条路都撞同一组寄存器/smem/互连墙。**默认路径一行未改，数值逐位不变。**
- **解锁条件**（写入 backlog）：① **换卡**（更多 smem/寄存器，能放下 (B) 的常驻 dK/dV 累加器，
  或 `CL×[BN][HD]` 的 cluster staging）；② 硬件 **scatter/bulk-reduce 到 L2 的 primitive**
  且不按 CTA 数线性计费；③ 把 dK/dV 折叠到 **warp-specialized producer/consumer + 384 线程**
  （同 F6/F3b 的多 warpgroup 路线，受同一寄存器墙）。
- 这条结论**补全了 F4b 的 backlog**：F4b 的三条「减 `red`」路（归约宽度/机制、工作划分、
  GQA 头折叠）至此全部收口为负/不可行。

**原始输出**：`src/fp8/fa_bwd_fp8_o86_gqa_red_probe.out.txt`（5 case × ncu red/read/duration）、
`src/fp8/fa_bwd_fp8_o86_red_law.out.txt`（`harness/fa_fp8_red_law.py` 解析并打印 `red/H` 常数
= 5.1304e5±5.5e1）、`src/fp8/fa_bwd_fp8_o86_accuracy.out.txt`（4 个 GQA/MQA 的 relL2 护栏复核）、
`src/fp8/fa_bwd_fp8_o86_ci.out.txt`（`--no-run --ci` 全绿）。

## 110. O87（第一百八十二轮）：F3b GEMM3/4 wgmma 的 **wait-schedule 变体**（合并 commit/wait、`wait_group<1>` 流水）—— **仍为负结果**，默认路径一行未改

> 承接 O82（`-DFA_WGMMA34=1` 把 dV/dK 切 wgmma RS，M=BN=32 零填充到 m64，实测 **0.964×**）。
> O82 的 ncu 诊断是「指令 −7.3%、L1/L2 降，但 **No Eligible 升、Duration 反升**——wgnma `wait`
> 与零填充/转置的开销抵消了指令路径收益」。本轮**只改 wgmma 的 fence/commit/wait 时序**（不改
> 操作数几何），试两个最自然的「藏 wait」写法，看能否把 O82 翻正。**结论：翻不正，F3b 的
> GEMM3/4 wgmma（无 BN=64 时）路线彻底关闭。**

### 110.1 两个 wait-schedule 变体（只改 fence/commit/wait，不改数学）

O82 原版（`-DFA_WGMMA34=1`）：对每个 GEMM 分两组 n-tile（`ng=0/1`，每组 2 条 `wgmma`），
每组各自 `wgmma_fence_fp8(); issue×4; commit; wait0;` 后 epilogue ⇒ **只有 2 条 wgmma 在飞、
且 epilogue 前必等齐**。本轮两个变体：

- **变体 A「合并 commit/wait」**：`acc[4][16]` 一次零初始化、一次 fence、一次连发 8 条 wgmma、
  一次 `commit`、一次 `wait0`，再对 4 个 n-tile 一起 epilogue ⇒ 8 条在飞。
- **变体 B「`wait_group<1>` 流水」**：`acc[4][16]`、一次 fence、`ng=0`（2 条）`commit` 后
  再 `ng=1`（2 条）`commit`；先 `wgmma_wait_group_fp8<1>()`（只等 `ng=0`）做 `ng=0` 的
  epilogue（与 `ng=1` 仍在飞的 wgmma 重叠），再 `wait0` 做 `ng=1` 的 epilogue。
- （对照）**变体 C「GEMM5 wait 流水」**：O81 的 GEMM5（正结果、默认开）是单组 4 条 wgmma
  + 一次 `wait0`。把它拆成两个 `commit` 组，`wait_group<1>` 后先折算 `nn=0/1`、`wait0` 再
  折算 `nn=2/3`（`acc5` 数量不变）。**只验证 O81 路径有没有同款残余 wait。**

### 110.2 实测（同 session，`fa_bwd_fp8_main.cu`，S=4096 H16 causal，event iters=30，main ms）

| 构建 / 变体 | main (ms) | vs 默认 | 说明 |
|---|---|---|---|
| `-DFA_WGMMA34=0`（默认，GEMM3/4 mma） | **1.4788** | 1.000× | 基线 |
| `-DFA_WGMMA34=1`（O82 原版，分组 wait） | 1.5435 | 0.958× | 复现 O82 0.964× |
| 变体 A 合并 commit/wait | 1.5901 | 0.930× | 8 条在飞但 epilogue 前等齐 |
| 变体 B `wait_group<1>` 流水 | 1.5778 | 0.937× | 两个变体都比 O82 原版更慢 |
| 变体 C GEMM5 wait 流水（默认 O81 路径） | 1.4957 | 0.989× | 中性偏负 |

**两个 GEMM3/4 变体都跌破 O82 原版**（0.930/0.937 vs 0.958）——细粒度「每 2 条就等」反而比
「攒 4/8 条一起等」更贴近 wgmma 的发射-完成节奏；把 epilogue 前移（变体 B）也没换来收益。
变体 C 说明**连 O81 这条正结果路径也已到 wait 的平台期**（`acc5` 折算太短、藏不住）。

### 110.3 ncu 证据（S=4096 causal，`regex:kvtma_kernel --launch-count 1`）

| 指标 | `WGMMA34=0`（默认） | `WGMMA34=1`（O82） |
|---|---|---|
| Duration | **1.49 ms** | 1.55 ms |
| `smsp inst` | 641,862,784 | **594,828,928（−7.3%）** |
| `lts op_red`（dK/dV 跨 CTA 原子） | **114,524,160** | **114,524,160（一字不变）** |
| `lts op_read` | 28,038,921 | 27,712,491（−1.2%） |
| `sm__pipe_tensor_cycles_active` | 11.13% | 8.37% |
| stall `short_scoreboard` | 1.82 | **1.41** |
| stall `wait` | 1.56 | 1.56 |
| warps active | 18.35% | 18.35% |

⇒ wgmma **确实打掉了指令与 smem→mma 依赖（`short_scoreboard` 1.82→1.41）**，但
**`red` 一字不减、`wait` 不降**，Duration 反升。默认 fp8 main 是 **L2 `red` bound**（`red` 占
L2 扇区 ~80%、§106），`wgmma` 对 `red` 一字不减（O83 已证），因此**任何只改指令/wait 时序、
不改「每元素贡献 CTA 数」的改动都不可能转正**——本轮三个变体再次独立验证。

### 110.4 判决与解锁条件

- **判决**：F3b 的「无 BN=64 时把 GEMM3/4 切 wgmma」路线在 wait-schedule 维度上**再无空间**
  （O82 负 → O87 三个代表变体更负）。**默认路径一行未改、数值逐位不变**（三变体 vs fp32 ref 的
  `max_abs` 均 2.635/2.644/3.216e-1，与默认逐位相同）。寄存器账也复现：默认实例仍
  **168 regs / 40B spill**（合并 `acc[4][16]` 未进一步 spill，说明纯时序问题而非溢出）。
- **唯一剩余的真 m64 路** = **BN=64 + 多 warpgroup 摊累加器（256/384 线程，对标 TE 384）**，
  被本卡「3 CTA/SM ⇒ 77,482B / 170 regs」硬墙锁死（O83 §106.2：三条真 m64 形式全停不到
  3 CTA/SM）——与 F6/F7/p160/O86 同源。解锁需**换卡**（更大 smem/寄存器）或 **warp-specialized
  producer/consumer**。**F3b 至此与 F4b/F6/F7 一样收口为「本卡无软件解」。**
- 见 `docs/08` §5.96；原始输出 `src/fp8/fa_bwd_fp8_o87_wg34_{0,1_orig,1_mergewait,1_pipewait}_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o87_g5pipe_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o87_ncu_wg34_{0,1}_s4096.out.txt`。

## 111. O88（第一百八十三轮）：fp8 非 main「quant+zero」栅格封顶 A/B + 编译宏复扫 + `D=256` K/V-TMA 资源收口 —— **负结果/收口，默认一行未改**

> 承接 O77/O83/O87（F3b/F4b/F6/F7 在本卡收口为「无软件解」）。本轮按 ROADMAP「fp8 专项冲刺」
> 的剩余候选，做了三件事：把「非 main 固定开销」的最后一条（quant 块调度）、O81 之后的编译
> 宏、以及 O85 遗留的 `D=256` K/V-TMA backlog 逐一判决。**结论：三件全部负/不可行；默认路径
> 性能与数值一字未改。**

### 111.1 quant+zero 融合 kernel 的栅格封顶 A/B（负结果）

- **动机**：默认 fp8 端到端 = quant 0.106ms + preprocess 0.122ms + main 1.481ms + cvt ~0（S=4096
  H16 causal，两文件，Hopper）。融合 kernel `quantize_zero_delta_warp_kernel<4>` 以
  `grid = 任务数/4`（每 warp 一行、每 CTA 4 行）启动，S=4096 时 **grid=114,688 个 CTA**
  （458,752 个行任务），每个 CTA 只搬 4 行（~2KB）。怀疑 **块调度开销**是非 main 开销的来源。
- **改动（纯 host，`--qcap=N` 默认 0=历史行为；单/两文件同源）**：给 `quant_zero` /
  `quant_zero_delta` 的 `grid` 加一个上限 `qcap`，其余逐字不变（`--qcap=0` 与历史逐位一致）。
- **实测（同 binary 交替，S=4096 H16 causal，event iters=30）**：

  | `--qcap` | grid | quant (ms) | total (ms) |
  |---|---|---|---|
  | 0（历史） | 114688 | **0.1060** | **1.7277** |
  | 4096 | 4096 | 0.1100 | 1.7253 |
  | 2048 | 2048 | 0.1101 | 1.7214 |
  | 1056 | 1056 | 0.1194 | 1.7403 |
  | 528 | 528 | 0.1556 | 1.7676 |
  | 264 | 264 | 0.2503 | 1.8628 |
  | 132 | 132 | 0.4477 | 2.0724 |

  ⇒ **栅格封顶单调更慢**（quant 0.106→0.448ms）。块调度**不是**瓶颈：128-thread CTA 的
  dispatch 被硬件充分流水，「每 warp 一行」的最大栅格已是最优并行度。量化 kernel 的流量
  ~268MB（4×fp16→fp32 读 134MB + fp8 写 33.5MB + 3×fp32 清零写 100MB）/ 0.106ms ≈ **2.5 TB/s
  ≈ 峰值 75%**，已近带宽墙。**默认保持 grid=任务数/4，`--qcap` 仅留作诊断。**
- **顺带观察（backlog，非本轮）**：quant 读入的是 **fp32**（host 把 fp16 npy 读成 `float`），
  生产输入（已 fp16/bf16）可省一半读字节（~67MB、~2% 端到端），但属 harness 数据路径、
  非算法优化。

### 111.2 O81 之后的编译期优化宏复扫（负结果）

- **动机**：O81 把 GEMM5 切到 wgmma 后，主 kernel 的指令/等待结构变了；重扫 O77 那一批
  编译宏，看是否有宏因结构变化而转正。
- **实测（同 binary 交替，S=4096 H16 causal，main ms）**：

  | 宏 | main (ms) | 相对 baseline |
  |---|---|---|
  | baseline | **1.4802** | 1.000× |
  | `-DFA_ILV=1` | 1.4842 | 0.997× |
  | `-DFA_ILV34=1` | 1.5672 | **0.944×** |
  | `-DFA_WS1=1` | 1.4913 | 0.993× |
  | `-DFA_R4=1` | 1.5135 | **0.978×** |
  | `-DFA_ILV=1 -DFA_ILV34=1` | 1.5591 | 0.949× |
  | `-DFA_WS1=1 -DFA_ILV34=1` | 1.5570 | 0.951× |

  ⇒ **全部中性或有损**（ILV34 −5%、R4 −2.5%），复证 O77/O87：默认 fp8 main 是 **L2 `red`
  bound**，只改指令/发射顺序（不改「每元素贡献 CTA 数」）不可能转正。默认宏配置正确。

### 111.3 `D=256` K/V 4D-TMA 的资源收口（结构性不可行，解析）

- **背景**：O85 把 `D=256` 的 Q/dO 上 4D-TMA（chunk-major）判为中性、默认关，并留 backlog：
  「让 `D=256` TMA 转正需把 K/V 也 TMA 化，但 K 双缓冲会顶穿 116KB/2-CTA 门槛」。
- **定量（按 `Fp8Cfg` 常量精确核算）**：`D=256` 的 `smem_bytes_wgmma` = **115,712B**（O84
  ncu 实测，2 CTA/SM，上限 116,224B）。`smem_bytes_wgmma_kvtma = smem_bytes_wgmma +
  ks_sw_bytes + 64`，其中 `ks_sw_bytes = (BN/8)*(HD/128)*1024 = (32/8)*2*1024 = 8,192B`
  ⇒ **123,968B > 116,224B** ⇒ K 双缓冲直接把 `D=256` 从 2 CTA/SM 挤到 **1 CTA/SM**（与 O85
  的 smem 64B 坑同源、量级更大）。若 K 单缓冲（无额外 stage），则退回 O85 的「TMA prologue
  与 K/V cp.async 不重叠」≈中性 ⇒ **`D=256` K/V-TMA 在 2 CTA/SM（本卡唯一可用档）下结构性
  不可行**，与本卡 F6/F7/p160/O83/F3b/F4b 的 **smem 墙**同源。
- **解锁**：换卡（更大 smem）或把 `D=256` 的 Qp/dOp/Kp 配对副本消掉（fp8 逐字节转置，
  O80/O81 基建已具备，但需重算 smem/描述符）。

### 111.4 默认路径与回归

- 默认路径（`--qcap=0`、默认宏、`D=128` kvtma）**性能与数值一字未改**：S=4096 两文件 total
  1.7162ms / 80.08 TF、main 1.4829ms；vs fp32 ref max_abs **2.635/2.644/3.216e-1**、vs TE
  同量级。单/两文件 device 代码仍逐字同源（本轮只动 host 的栅格计算与一个诊断入参）。
- 原始输出：`src/fp8/fa_bwd_fp8_o88_qcap_ab_s4096.out.txt`、`..._o88_macro_sweep_s4096.out.txt`。

---

## 112. O89（第一百八十四轮）：fp8 主 kernel 的 **LPT m 块调度序**（causal 贵块先跑）—— **正结果，默认**

默认 fp8 `kvtma` 主 kernel 在本卡是 **L2 `red` bound + 尾波偏斜**。前几轮（O77/O83/O86/O87）
已把「减 L2 搬运量」的三条路（归约宽度/机制、工作划分、GQA 头折叠）全部收口为硬件墙，本轮
换一个**不改任何数据通路、不改 L2 搬运量**的角度：**只改「哪个 CTA 算哪个 m 块」的派发顺序**。

### 112.1 动机：causal 的块代价偏斜 vs 硬件升序派发

- 默认稠密网格 `grid=(nblk*ksplit, H, B)`，body 里 `mt = blockIdx.x / ksplit`、`mblk = mt`
  （定长无查询表路径，`fp8_mma_body` 第 2976–2979 行）。GigaThread 按线性 blockIdx **升序**
  派发 block。
- causal 下第 `m` 个 m 块的 K 循环长 ∝ `(m+1)`（`ncols = min(len, m0+BM)`）⇒ **便宜块先跑、
  最贵块排在每个 head 段末尾**，正是 LPT（Longest-Processing-Time-first）的反面，尾波全是重块。
- 这解释了 O77 的 ksplit 复扫：ksplit=8 比 ksplit=1 快 ~20%，靠的是**更细的块粒度**掩盖偏斜，
  代价是 Q/dO 被重读 8 次、dQ 多一轮跨 part `atomicAdd`。若能把平衡拿回来，理论上可降 ksplit
  以消 Q/dO 重读与 dQ 原子。

### 112.2 实现（纯 host + 一个透传参数，device 数学零改动）

- `fp8_mma_body` 早已支持 `mt_m` 查询表（varlen 紧凑网格用，`mblk = mt_m[mt]`）。给默认
  `fa_bwd_fp8_mma_kvtma_kernel` 壳加尾参 `const int* mt_m = nullptr` 并在调 body 时填到
  `mt_m` 槽（此前恒传 `nullptr`）；`launch_bwd_main_kvtma` 同样加尾参并透传。
- host 在 `--mrev=1` 时建反转表 `mt_m[i] = nblk-1-i`（`nblk = ceil(S/64)`）：执行序变成
  「每个 head 从最贵的 m 块开始、最便宜的收尾」。dK/dV 是跨 CTA `atomicAdd`（可交换）⇒
  **数值语义不变**，仅加法次序略变（fp8 噪声内）。
- 门控：`mrev_opt && causal && D==128 && nblk>=16`（S>=1024；小 S 网格太浅、反转只剩噪声）。
  单/两文件（`fa_bwd_fp8_main.cu` / `fa_bwd_fp8_mma_onefile.cu`）同步；`--mrev=0` 供 A/B；
  默认 `--mrev=1`。

### 112.3 实测（同 binary A/B，S=4096 H16 causal，event iters=30）

| 配置 | main (ms) | total (ms) | TFLOPS |
|---|---|---|---|
| `--mrev=0`（历史） | 1.4796 | 1.7159 | 80.10 |
| `--mrev=1`（默认） | **1.4475（1.022×）** | **1.6747（1.025×）** | **82.07** |

多次重复：main `mrev=0` 1.473–1.488 → `mrev=1` 1.429–1.440（**~3.2%**，稳定）。其它 causal
D=128：S1024H32 `0.2481→0.2442`（1.6%）、GQA kv4 `0.2410→0.2362`（2.0%）、GQA kv8
`0.2982→0.2944`（1.3%）、MQA kv1 `0.4442→0.4420`（0.5%）；S512 在门控外 ⇒ 逐位不变。

ksplit 复扫（`mrev=1`，main ms）：k=1 **1.792** / k=2 1.547 / k=4 1.445 / **k=8 1.448**。
反转把 ksplit=1 从 1.836 拉到 1.792（仅 1.024×）——**「削尾波」补不上 ksplit 的细粒度并行**
（1024 CTA 只 2.6 波），故 **ksplit auto=8 仍最优、Q/dO 重读消不掉**。

### 112.4 ncu 证据（S=4096 causal，`regex:kvtma_kernel --launch-count 1`）

| 指标 | mrev=0 | mrev=1 |
|---|---|---|
| `gpu__time_duration` | 1.48 ms | **1.46 ms** |
| `lts op_red` | 114,524,160 | 114,524,160（**一字不变**） |
| `lts op_read` | 28.04M | 27.92M |
| `lts op_write` | 0.384M | 0.384M |
| DRAM% | 4.40 | 4.50 |
| `short_scoreboard` / `wait` | 1.82 / 1.56 | 1.82 / 1.56 |

⇒ **收益纯粹来自尾波/负载均衡（LPT），与 L2 搬运量、stall 结构无关**——再次印证默认 main
是 `red` bound。**减 `red` 仍需改工作划分（换卡 / 多 warpgroup）**，本条不改 `red` 墙。

### 112.5 数值与回归

- `--ci --dtype fp8 --hopper`：单/两文件一致性 gate worst **7.629e-6 OK**、`--check docs/04`
  OK（198 行）。
- vs fp32 ref max_abs 与 `mrev=0` **打印相同**（S4096 `2.635/2.644/3.216e-1`、S512
  `2.426/2.972/3.733e-1`）；单/两文件 device 代码仍逐字同源（本轮只动 host 的查询表构建与
  壳的尾参透传）。默认 `--mma`（sm_90）构建不含 kvtma 路径 ⇒ mrev 无效、逐位不变。
- 原始输出：`src/fp8/fa_bwd_fp8_o89_ab_mrev_s4096.out.txt`、
  `..._o89_ab_mrev_shapes.out.txt`、`..._o89_ncu_mrev_{0,1}_s4096.out.txt`、
  `..._o89_ci_fp8.out.txt`。

## 113. 第 185 轮（O90 / F6-step3）：wgmma2 的 K/V 4D-TMA 化（K 双缓冲）—— 负结果（opt-in `--wg2tma`）

### 113.1 动机

默认 fp8 main（`kvtma`，BM=64）在本卡是 **L2 `red` bound**（`red` 114.5M 扇区、占 L2 流量
80%、Duration×L2% ≈ TE 的 6.9×），且 F3b/F4b/F6/F7 已证所有「不改工作划分」的杠杆失效。
唯一能真正减 `red` 的是 **BM 64→128**（每个 KV 元素被一半的 CTA 贡献 ⇒ `red` 精确砍半，
O17b/F6 的既有结论）。F6 第二步的 `wgmma2`（BM=128、2 warpgroup、GEMM1/2 wgmma + SW128）
已实现，但只有默认档的 **~0.56×**——除 1 CTA/SM 外，每 tile 的 K/V 仍是「标量 global 读 +
`__syncthreads`」串行。本轮（F6-step3）把 `wgmma2` 的 K/V 换成 **4D-TMA（K 双缓冲、V 单缓冲，
复现 O41 的时序）**，验证「TMA 化 + `red` 砍半」能否把 BM=128 档拉回竞争区。

### 113.2 实现

新增 `fa_bwd_fp8_wgmma2_tma_kernel<HD, BM=128, BN=32>`（`kernels.cuh` 3d 节；单/两文件 device
逐字一致，392 行核对 `identical=True`）+ host `wgmma2tma_smem_bytes`/`launch_bwd_wgmma2tma` +
CLI `--wg2tma`（**opt-in，默认路径一行未改**；smem 约 140KB ⇒ 1 CTA/SM）。

- Q/dO 仍**手工载入**（每 CTA 一次，非热点；SW128 + `__byte_perm` 重建 Qp/dOp，与 wgmma2 逐字相同）。
- K/V 走 `tma_load_4d` + mbarrier：prologue 发 K[nt_begin]→stage0、K[nt_begin+1]→stage1、
  V[nt_begin]→Vs；循环尾发 K[nt+2]→stage `stg`、V[nt+1]，等 K[nt+1]（stage `stg^1`）→
  从 SW128 K stage 重建 Kp（`__byte_perm` 配对）→ 等 V[nt+1]；相位用 `kc0/kc1/vuse` 标量
  （不落 local，复现 F3-a 的教训）。
- 编译：208 regs / 0 spill / 1 barrier（wgmma2 为 212 regs）。

### 113.3 实测（同 binary A/B，S=4096 H16 causal，event）

| 配置 | main (ms) | total (ms) | TFLOPS |
|---|---|---|---|
| 默认 `kvtma`（BM=64，wgmma） | **1.4518** | **1.6765** | **81.98** |
| `wg2`（BM=128，mma GEMM1/2） | 2.9533 | — | 40.3 |
| `wg2wgmma`（F6：GEMM1/2 wgmma） | 2.8308 | — | 42.0 |
| `wg2wgmma`+K/V-TMA（**本轮 O90**） | 2.7113 | 2.9494 | 46.6 |

同 session `[O90 A/B]`：**wg2wgmma 2.8062 → +KVTMA 2.7113 ms（1.035×）**；S=512 为
0.1067→0.1065（1.001×）。即 **K/V TMA 相对 wgmma2 仅省 ~3.5%（大 S）**，`wgmma2tma` 仍只有默认
档的 **0.54×（S4096）/ 0.66×（S512）**。

### 113.4 ncu 证据（S=4096 causal，`--launch-count 1`，默认 vs O90）

| 指标 | 默认 `kvtma`（BM=64） | O90 `wgmma2_tma`（BM=128） |
|---|---|---|
| Duration | 1.45 ms | 2.72 ms |
| `lts op_red` | 114,524,160 | **58,195,968（精确砍半）** |
| `lts op_read` | 27,916,478 | **14,467,149（砍半）** |
| `lts op_write` | 0.386M | 0.003M |
| L2 利用率 | **81.06%** | **21.99%（带宽大量空闲）** |
| L1/TEX | 74.82% | 51.13% |
| Active warps | 18.62% | 12.50% |
| CTA/SM | 3 | 1 |
| `smsp inst` | 642.1M | 838.3M |
| stall（short/wait/long） | 1.81 / 1.56 / 0.37 | 1.22 / 1.32 / 0.18 |

⇒ **BM=128 把 L2 搬运量精确砍半**（`red`/`read`）——机制完全成立；但 L2 利用率从 81% 掉到
22%（**省下来的带宽用不上**），根因是 **occupancy 18.62%→12.50%（3→1 CTA/SM）、仅 8 warp
藏不住延迟**（K/V TMA 只把 `long_scoreboard` 0.37→0.18）。这正是 F6/O83 早已判定的
「BM=128 在本卡停不到 2 CTA/SM」资源墙的直接体现。

### 113.5 数值（护栏全过）

- `ours_o90 vs fp32 ref` relL2：S4096 dq/dk/dv **8.15% / 8.39% / 6.52%**、S512 **8.18% /
  8.41% / 6.36%**（护栏 dq≤8.2 / dk≤8.3±0.3 / dv≤6.5±0.3；与默认档 ~8.2/8.3/6.5 同量级，
  差异来自 BM=128 的 atomic 次序）。
- `max_abs` vs ref：S4096 `2.635/2.760/3.325e-1`、S512 `2.426/2.996/3.713e-1`；
  O90 A/B `wg2wgmma_tma vs wg2wgmma` max_abs `1.19e-7/2.38e-7/4.77e-7`（纯 atomic 次序）。
- `ours_o90 vs 默认 ours_hp`：dq max_abs `4.6e-4`、dk `8.8e-2`、dv `4.5e-2`（fp8/BM 次序噪声）。
- `ours vs TE` relL2 dq/dk ~13.4%、dv S512 12.0% / S4096 **28.2%**——注意 **TE-vs-ref 的 dv
  本身在本 shape 就有 relL2 27.45%**，故该 28% 由 TE 自身误差主导，非本实现回退。
- 单/两文件默认路径数值逐值不变（`ours_hp`==`ours_sf_hp`）；`--mma`（sm_90）构建亦编译通过。

### 113.6 判决

**负结果（opt-in，默认关）。** K/V TMA 化确实把 `wgmma2` 提速 3.5%（指令/搬运路径收益），但
**BM=128 双 warpgroup 在本卡 1 CTA/SM 下是延迟/occupancy bound**（L2 利用率仅 22%），`red`
砍半的收益被 8 warp 藏不住延迟完全吃掉。与 F6/O83/O86/O87 同一堵墙：**除非 256/384 线程 +
多 warpgroup 摊累加器 / 换卡把 occupancy 拉起来，BM=128 无法转正**。本轮的 `wgmma2_tma` 与
K/V-TMA helper 仍可复用（是后续 `D>128`/WS 版的基建）。见 `docs/08` §5.99、`ROADMAP`
「fp8 专项冲刺 F6」。原始输出 `src/fp8/fa_bwd_fp8_o90_{ab_s4096,ab_s512,default_s4096,
ncu_wg2tma_s4096,ncu_default_s4096}.out.txt`。

## 114. 第 186 轮（O91 / F6-step4）：多 warpgroup（BM=192、3 WG、384 线程）—— 负结果（opt-in `--wg3`）

### 114.1 动机

O90（§113）把 `wgmma2`（BM=128、2 warpgroup、256 线程）补上 K/V 4D-TMA，`red` 精确砍半，
但只有默认档的 0.54×，根因钉为 **1 CTA/SM、仅 8 warp/SM 藏不住延迟**。O90 的「下一步候选 ①」
是 **多 warpgroup（256/384 线程）**——本卡上唯一未试、能同时（a）把 BM 继续放大以压 `red`、
（b）把每 SM 的 **warp 数**拉回默认档（默认 3 CTA/SM × 4 warp = 12 warp/SM；wgmma2 仅 1 CTA ×
8 warp = 8 warp/SM）的结构性杠杆。本轮把 `wgmma2` 泛化到 **`NWG` 个 warpgroup（`BM=NWG*64`）**，
用 `NWG=3`（BM=192、384 线程、1 CTA/SM）实测：**12 warp/SM + `red` 从 114.5M 再降到 49.6M**。

### 114.2 实现

- **device 泛化**（`kernels.cuh` 3c 节，单/两文件 device 逐字一致）：`fa_bwd_fp8_wgmma2_kernel`
  加 `int NWG=2` 模板参（`BM=NWG*64`、`TH=NWG*128`、`__launch_bounds__(NWG*128,1)`）。相 A 每个 WG
  各跑 `wgmma.m64n32k32` 算自己 64 行（A 描述符 +`wg*8192B`）；GEMM3/4（dV/dK，输出仅 BN×HD）
  仍用 8 个 warp（`if (wid<8)`，NWG=2 时恒真 ⇒ **逐位不变**）；GEMM5（dQ）按 `wm=wid>>1,wn=wid&1`
  自然铺 6×2；fold 的 `Ap/dS3` 沿 m **泛化到全 BM**（`NPC=BM/128` 个整块 + `REM` 尾块，
  `BM=128` 时 `NPC=1,REM=0` ⇒ 与旧代码逐位相同）；`dS2` fold 本就用 `wid*16` 覆盖全 BM。
- **`fence.proxy.async.shared::cta`**：SW128 的 Q/dO/K/V 是 generic 写、下一句 wgmma 走 async
  proxy 读。NWG=3 的线程时序更易命中 p155/O81 记过的 race——**首次实测 dk/dv 偶发 `inf`**
  （非确定），在每迭代顶部补 `bulk_reduce_fence()` 后 3/3 稳定。这是本轮最重要的踩坑。
- **host**：`wgmma_nw_smem_bytes<HD,BM,BN,NWG>`/`launch_bwd_wgmma_nw<...>`、CLI `--wg3`（opt-in，
  默认路径一行未改）+ `--ksplit3=`（自动 ~8 波）。`BM=192` 的 smem = **197,120B ⇒ 1 CTA/SM**。
  384 线程 `__launch_bounds__(384,1)` 上限 170 regs，实测 **168 regs / 0 spill**（未触寄存器墙）。

### 114.3 实测（同 binary A/B + 默认同 session，S=4096 / S=512 H16 causal，event）

| 配置（S4096） | main (ms) | total (ms) | TFLOPS | 相对默认 |
|---|---|---|---|---|
| **默认 `kvtma`（BM=64，3 CTA/SM）** | **1.4436** | **1.6706** | **82.27** | 1.00× |
| `wg2wgmma`（BM=128，2 WG） | 2.8330 | — | 41.9 | 0.51× |
| **`wg3`（BM=192，3 WG，**本轮 O91**）** | 2.7718 | 3.00 | 42.8 | **0.52×** |

- 同 session `[O91 A/B]`（S4096）：`wgmma2 2.8330 → wg3 2.7718 ms（1.022×）`；`wg3` 的 auto
  `ksplit3=8`、grid=176×16。**S512**：`wgmma2 0.1076 → wg3 0.1179ms（0.913×）`（grid-bound）。
- 即：`wg3` 相对 `wgmma2` 只快 2.2%，相对**默认档仍只有 0.52×**。

### 114.4 ncu 证据（S=4096 causal，`--launch-count 1`，默认 vs O91 `wg3`）

| 指标 | 默认 `kvtma`（BM=64） | **O91 `wg3`（BM=192）** |
|---|---|---|
| Duration | 1.44 ms | 2.78 ms |
| `lts op_red` | 114,524,160 | **49,643,520（0.43×）** |
| `lts op_read` | 27,896,828 | 16,987,685 |
| `lts sectors 总量` | 142,941,608 | **80,622,712（0.56×）** |
| L2 利用率 | **81.42%** | **21.18%（带宽大量空闲）** |
| DRAM | 4.52% | 2.51% |
| Active warps | 18.60% | **18.72%（= 12 warp/SM，与默认同）** |
| Tensor pipe | 11.26% | 8.38% |
| regs / CTA/SM | 168 / **3** | 168 / **1** |

⇒ **`red` 再降 2.3×、L2 搬运量降到 0.56×、warp 数已追平默认档（18.7%=12 warp/SM）**——机制完全成立；
但 **Duration 反升 1.93×、L2 利用率只剩 21%**。⇒ 卡点**不是「每 SM 的 warp 数」本身**，而是
**1 CTA/SM = 只有一个 barrier 域**：默认档的 12 warp 分属 **3 个独立 CTA**（互不 `__syncthreads`），
而 `wg3` 的 12 warp 挤在 **1 个 CTA** 内、被每 tile 的 5 个 `__syncthreads` 串成一条依赖链，
跨-tile 的延迟无法用另一 CTA 的工作填。**「多 warp 摊延迟」被证伪：需要的是多 CTA，不是多 warp。**

### 114.5 数值（护栏全过）

- `ours_wg3 vs fp32 ref` relL2：S4096 dq/dk/dv **8.149% / 8.449% / 6.532%**、S512 **8.179% /
  8.455% / 6.371%**（护栏 dq≤8.2 / dk≤8.3±0.3 / dv≤6.5±0.3；与默认档同量级，差异来自
  BM=192 的 atomic 次序）。
- `max_abs`：S4096 `2.635/2.783/3.330e-1`、S512 `2.426/3.027/3.678e-1`。
  O91 A/B `wg3 vs wgmma2` max_abs `1.19e-7/8.59e-2/3.95e-2`（fp8/BM 次序噪声）。
- `ours vs TE` relL2 dq/dk ~13.4%、dv S512 12.0% / S4096 **28.2%**；**TE-vs-ref 的 dv 本 shape
  就是 27.45%**，故 28% 由 TE 自身误差主导，非本实现回退（同 §113.5）。
- **单/两文件 device 逐字一致**（`sync` 区段 `identical=True`），`ours_wg3` vs `ours_wg3_sf`
  max_abs ≤ `1.19e-7`（纯 atomic 次序）；`--ci --dtype fp8 --hopper` gate 全绿、`docs/04` check OK。

### 114.6 判决

**负结果（opt-in，默认关）。** 这是本卡 fp8 主 kernel **最后一条未试的结构性杠杆（多 warpgroup /
放大 BM）** 的直接判决：它确实把 L2 `red` 从 114.5M 压到 49.6M（0.43×）、总 L2 搬运 0.56×，
并把 warp/SM 追平默认档，**但 1 CTA/SM 的单 barrier 域让 12 warp 无法互相填延迟**，净 0.52×。
**「多 warpgroup 摊累加器」只有在能同时维持 ≥2 个独立 CTA/SM 时才成立**——而那需要 ≤116KB smem
与 ≤128 regs，本卡 fp8 反向（BM≥128）达不到（F6/O83/O90 同墙）。⇒ 与 `ROADMAP`「阻塞」完全一致：
**默认 fp8 main 的 L2 `red` 墙在本卡无软件解；正结果只剩「换卡（更大 smem/寄存器）」**。
`--wg3` 与 `NWG` 泛化保留为换卡后的 `D>128`/WS 基建。见 `docs/08` §5.100、`ROADMAP`
「fp8 专项冲刺 F6」；原始输出 `src/fp8/fa_bwd_fp8_o91_{wg3_ab_s4096,wg3_ab_s512,
ncu_wg3_s4096,ncu_default_s4096,accuracy}.out.txt`。

---

## §115 O92（第一百八十七轮）：fp8 main L2 墙的「TE 侧对侧」闭环复核

### 115.1 目的与范围

`fp8 专项冲刺`（F1→F6）与 `下一批`（F6/F3b/F4b）在 O77 之后反复收敛到同一结论：
默认 fp8 `kvtma` main 是 **L2 `red` bound**、且所有软件杠杆（工作划分 / wgmma / wait-schedule /
BULKRED / GQA 折叠 / 多 warpgroup）均已判决。本轮**不再引入新算法**，而是做一次**闭环复核 + TE
逐指标对侧**，把「差距到底在哪一级、还差多少、剩余杠杆是什么」用同 session 的原始数据钉死，
并顺手复测两条在 O81（GEMM5 wgmma）/O89（LPT 调度）之后**尚未在最新默认上复跑**的候选
（`FA_BULKRED`、编译宏复扫、ksplit 复扫）。

**默认路径一行未改**；本节所有数字来自最新默认构建（`-DFA_WGMMA -DFA_TMA -lcuda`，sm90a）。

### 115.2 最新默认基线（S=4096 H16 B1 causal，event iters=30）

- `[timing] total(quant+pre+main+cvt) 1.6711 ms / 82.24 TFLOPS`；
  `quant 0.1061 | preprocess 0.1218 | main 1.4437 ms`（`main wgmma smem = 70656B`）。
- 与 O89/O91 基线逐位同档（复现稳定，非本轮回退）。

### 115.3 三条「已判决项」在最新默认上的复测

| 候选 | 实测（S4096 main，ms） | 相对默认 | 判定 |
|---|---|---|---|
| 默认 `kvtma` | **1.4397–1.4437** | 1.000× | — |
| `ksplit=1 / 2 / 4 / 8 / 16` | 2.032 / 1.804 / 1.694 / **1.667** / 1.821 | — | **auto=8 仍最优**（同 O77） |
| `-DFA_BULKRED=1`（TMA tensor-reduce） | 1.6454 | **0.877×** | 负（同 O42，O81/O89 后不翻转） |
| `-DFA_ILV34=1` | 1.5175 | 0.948× | 负（同 O88 宏复扫） |

- **ksplit**：mrev 已默认开（O89），复扫确认 **auto=8 最优**——`red` 随 ksplit 减小而降，
  但 causal 偏斜下的尾波/并行度损失盖过 `red` 收益 ⇒ **「消 Q/dO 重读的 ksplit 路径」仍关闭**。
- **BULKRED**：`cp.reduce.async.bulk.add.f32`（TMA 张量归约）替逐元素 `red.global.add`，机制上
  把「每 tile 2048 条原子」换成 per-warp staging + 一次 bulk，但 **staging 的 smem 流量 + TMA
  归约延迟** 使其 **0.877×**——与 O42 完全一致（O67/p159 已证 `red` 扇区数与归约机制无关）。
- **编译宏复扫**：ILV34（−5.2%）等仍全部中性/有损，复证 O88「只改指令/发射顺序、不改
  『每元素贡献 CTA 数』即不可能转正」。

### 115.4 TE vs ours：SASS 指令组合（S=4096 causal，`ncu --page source --print-source sass`）

| | TE `..._flash_bprop_wgmma_f8_..._64x64x128` | ours 默认 `kvtma<128,64,32>` |
|---|---|---|
| QGMMA | **16** | 12（GEMM1/2 共 8 + O81 的 GEMM5 共 4） |
| HMMA | **0** | **64**（GEMM3/4 dV/dK，O82/O87 已判负） |
| LDSM | 20 | 39 |
| STSM | **24** | 0 |
| TMA 归约 | **4× `UTMAREDG.4D.ADD`** + 4× `UTMALDG` + 2× `UTMASTG` | 7× `UTMALDG`（无 tensor-reduce） |
| 逐元素归约 | — | **64× `REDG.E.ADD.F32`** |

- ours 的 SASS 说明：**wgmma 已覆盖 GEMM1/2/5**（O81 起）；剩余 64 条 HMMA 是 GEMM3/4（dV/dK），
  其 wgmma 化（O82 零填充 m64 / O87 wait-schedule 变体）均已判负，**与 L2 `red` bond 无关**。
- TE 用 **TMA 4D 张量归约**写 dK/dV；但 O67/p159 已证 `red` 扇区数**只由工作划分决定**、与
  归约机制/宽度无关 ⇒ TE 的低 `red` 来自其**工作划分（persistent/每元素贡献数）**，不是 `UTMAREDG`。

### 115.5 TE vs ours：L2 / occupancy 六指标（S=4096 causal，同 session）

| 指标 | ours `kvtma`（BM=64） | TE `flash_bprop_wgmma_f8`（BM=64） | 比值 |
|---|---|---|---|
| Duration | **1.45 ms** | **258.34 µs** | **5.62×** |
| `lts sectors 总量` | 142.97M | 36.84M | **3.88×** |
| `lts op_read` | 27.91M | 10.00M | 2.79× |
| `lts op_red` | **114.52M** | **25.96M** | **4.41×** |
| `lts op_write` | 0.39M | 0.80M | — |
| L2 利用率 | **81.30%** | 70.70% | — |
| DRAM | 4.52% | 17.60% | — |
| Tensor pipe | 11.25% | **36.86%** | 3.28× |
| Active warps | 18.60% | 15.61% | — |
| regs / CTA/SM | 168 / **3** | 168 / **1** | — |

- **两者都 L2 bound**（81% vs 71%），但 ours 的 L2 搬运量是 TE 的 **3.9×**、其中 `red` 是 **4.4×**。
- 关键：**TE 的 tile 与 ours 同为 BM=64**（`64x64x128`）⇒ 差距**不是放大 BM**，而是
  **「每个 dK/dV 元素被多少 CTA 贡献」= 工作划分**（TE persistent grid=132，ours ksplit=8→8192 CTA）。
  把 ours 的 114.52M `red` 按 32B/sector 折算 = 830MB 原子流量，dK/dV 真实体量仅 67MB ⇒
  **每元素 ~54 次贡献**；TE 25.96M 对应 **~12 次**。这就是 F7 一直在追的 4.4×。
- **TE 的 tensor pipe 是 ours 的 3.3×**（36.9% vs 11.3%）⇒ TE 的 L2 墙背后算力更贴；ours 则
  在 L2 原子 + 低并行度上大量空转。

### 115.6 精度护栏（本轮默认，S=4096 causal）

- `ours vs fp32 ref` relL2：dq **8.149%** / dk **8.263%** / dv **6.489%**
  （护栏 dq≤8.2 / dk≤8.3 / dv≤6.5 ⇒ **全部通过**，与 O91 逐位同档）。
- `max_abs`：`2.635 / 2.644 / 3.216e-1`（O(0.2–0.9) 内）。
- `ours vs TE FP8` relL2：13.42% / 13.48% / 28.15%；其中 **TE-vs-ref 的 dv 自身就是 27.45%**
  ⇒ dv 的 28% 由 TE 误差主导、非本实现回退。

### 115.7 判决与剩余杠杆

**本轮为「闭环复核」：默认路径无新正结果。** 结论与 `ROADMAP`「阻塞」完全一致：

1. **默认 fp8 main 的 L2 `red` 墙在本卡无软件解**——F3b（GEMM3/4 wgmma）、F4b（归约机制/宽度/
   GQA 折叠）、F6/O90/O91（放大 BM / 多 warpgroup）、F7（工作划分：两 kernel / BULKRED /
   BN≥BM / column-owner）**全部判决**；本轮再复测 BULKRED/ksplit/宏亦无翻转。
2. **唯一经 ncu 钉死的真差距 = 工作划分**（同 BM 下 TE `red` 4.4× 低）——解锁需 **≥2 个独立
   CTA/SM 的放大 tile**（本卡 ≤116KB smem / ≤128 regs 达不到），或**换卡**（更大 smem/寄存器）。
3. **O91 的新认知**：1 CTA/SM 的瓶颈是 **单 barrier 域的 `__syncthreads` 串行**，而非 warp 数。
   若将来要复活 `wg3`/BM≥128 档，唯一路径是 **warp specialization（producer/consumer + mbarrier
   流水，彻底替换 `__syncthreads`）**——但这是多轮工程，且仍受同一 L2 墙约束，非本轮范围。

见 `docs/08` §5.101、`ROADMAP`「当前进度 第一百八十七轮」/「下一步」；原始输出
`src/fp8/fa_bwd_fp8_o92_{default_s4096,bulkred_s4096,sass_te_s4096,sass_ours_s4096,
ncu_te_s4096,ncu_ours_s4096}.out.txt`。

---

## 116. O93（第 188 轮）：跨 head 全局 LPT（grid 轴对调）—— 默认 fp8 causal 主 kernel 的调度正结果

> 承接 O89（`--mrev`，**per-head** 的 m 块降序 = LPT 贵块先跑，S4096 main 1.022×）与
> `ROADMAP`「fp8 专项冲刺 F6-①」的下一步候选 ③：**把 LPT 从「每个 head 内」升级到「跨 head
> 全局」**，并把由此解锁的**低 ksplit**（消 Q/dO 重读）一起利用。**不改任何数据/数学通路**，
> 只改「哪个 CTA 算哪个 `(h, mblk, part)`」。**默认开启**（`--hswap=0` 回退）。

### 116.1 动机

O89 的 `mrev` 只让**每个 head 内部**按 m 降序派发（`grid=(nblk*ksplit, H, B)`、`blockIdx.y=h`
是慢轴），序列是「`[64…1][64…1]…[64…1]`」的**锯齿**——每个 head 边界处负载重置，全局并非单调下降。
O92 的 ksplit 复扫（1/2/4/8/16 = 2.032/1.804/1.694/**1.667**/1.821ms）证明低 ksplit 的**尾波**
补不回来，而高 ksplit 正是 Q/dO 被重复读（8×）与 dQ 跨 part 原子的根源。

**O93 的思路**：把 head 放到 **`blockIdx.x` 快轴**——`grid=(H, nblk*ksplit, B)`，硬件按
`blockIdx.x` 最快的线性顺序派发 ⇒ 前 `H` 个 CTA 是 **所有 head 最贵的 m 块**、随后依次降档，
天然 = **跨 head 的全局 LPT**。全局单调下降让「长任务」全部尽早开始，尾波只剩最便宜块，
于是**低 ksplit 的负载均衡第一次够用**，可以放心把 ksplit 收到 2。

### 116.2 实现（device 一行数学未改；单/两文件 device 逐字同源）

- **device（`fa_bwd_fp8_kernels.cuh`）**：给 `fp8_mma_body` 加末位模板参 `bool HSWAP=false`，
  仅改解码三行：
  ```cpp
  const int part = HSWAP ? (blockIdx.y % ksplit) : (blockIdx.x % ksplit);
  const int mt   = HSWAP ? (blockIdx.y / ksplit) : (blockIdx.x / ksplit);
  const int h    = HSWAP ? blockIdx.x : blockIdx.y;
  ```
  `fp8_mma_kvtma_kernel` 加同名额模板参并透传；其余 kernel 用默认 `false`（零影响）。
- **host（`fa_bwd_fp8_main.cu`）**：`launch_bwd_main_kvtma` 加 `bool HSWAP`；默认分支在
  `hswap_opt && d_mrev`（即 O89 的 mrev 表已建：定长 causal D=128 nblk≥16）时，用
  `dim3(H, mg.x, mg.z)` 启动 + `HSWAP=true`。默认 `hswap_opt=1`，`--hswap=0` 回退。
- **自动 ksplit**：`hswap_elig` 且未显式 `--ksplit` 时把 auto ksplit 置 **2**（仅 Hopper
  `-DFA_WGMMA -DFA_TMA` 构建）。历史路径/非 D=128/非 causal/varlen 一律不变。
- **单文件** `fa_bwd_fp8_mma_onefile.cu` 由 `scripts/sync_onefile_device.py` 同步（`identical=True`），
  host 段手工同源（声明/CLI/launch/print）。

### 116.3 性能（S=4096 H16 B1 causal，同 binary A/B，event iters=40）

| 配置 | main (ms) | total (ms) | TFLOPS |
|---|---|---|---|
| `mrev=0 hswap=0`（O88 前） | 1.4768 | 1.7151 | 80.13 |
| `mrev=1 hswap=0`（**O89 默认**，ksplit=8） | 1.4304 | 1.6686 | 82.37 |
| `hswap=1` ksplit=8 | 1.8699（0.76×） | 2.0936 | 65.65 |
| `hswap=1` ksplit=4 | 1.4028 | 1.6349 | 84.07 |
| **`hswap=1` ksplit=2（新默认）** | **1.3742** | **1.6109** | **85.32** |
| `hswap=1` ksplit=1 | 1.4558 | 1.6900 | 81.32 |

- **主结果**：main **1.4304→1.3742ms（1.041×）**、total **1.6686→1.6109ms（1.036×，82.37→85.32 TF）**。
- **关键判据**：`hswap=1` 在 ksplit=8 时**反而慢**（跨 head 交错损 L2 读局部性、细粒度下更明显），
  但 **hswap=1 + ksplit=2** 才最优 —— 说明收益来自「全局 LPT 解锁的低 ksplit」而非 hswap 本身。
- **其它形状（默认 vs `--hswap=0`）**：S1024H32 main 0.2448→**0.2224（1.10×）**；
  GQA kv4 main 0.2367→**0.2100（1.13×）**；S512（nblk=8，门控外）不变。

### 116.4 ncu（S=4096 causal，同 session，`fa_bwd_fp8_mma_kvtma_kernel`）

| 指标 | `--hswap=0`（k=8） | 默认 `hswap`（k=2） | 变化 |
|---|---|---|---|
| Duration | 1.45 ms | **1.39 ms** | −4% |
| L2 总扇区 | 142.97 M | **129.87 M** | **−9.1%** |
| L2 `op_read` | 27.88 M | **24.26 M** | **−13.0%** |
| L2 `op_red` | 114.52 M | **105.38 M** | **−8.0%** |
| L2 利用率 | 81.0% | 76.9% | — |
| DRAM 字节 | 219 MB | 557 MB | **+2.5×** |
| warps active | 18.6% | 18.7% | — |

- **真降 L2 搬运**：ksplit=8→2 使 Q/dO 重读从 8× 降到 2×（`read −13%`），dQ 跨 part 原子
  也随之减少（`red` 中 dQ 部分 −8%）。`red` 的 dK/dV 主体（~104M）不变——仍是工作划分决定。
- **代价**：跨 head 交错使并发 CTA 落在 16 个 head ⇒ L2 局部性下降，**DRAM 字节 2.5×**。
  但主 kernel 墙是 **L2 吞吐（81→77%）**，DRAM 仅 4.5→11.7%（绝对量仍低），故 **L2 降幅盖过
  DRAM 升幅、净快**。换卡时需复核 L2/DRAM 比（见 `docs/05`）。

### 116.5 精度护栏（S=4096 causal，`ours_hp vs fp32 ref` relL2）

- dq **8.148%** / dk **8.263%** / dv **6.489%** —— 全在护栏（≤8.2 / 8.3 / 6.5）内，
  与 O91/O92 同档（调度只改 atomic 加法次序，数值在 fp8 噪声内）。
- `max_abs` 2.635 / 2.644 / 3.216e-1（O(0.2–0.9)）。`ours vs TE` relL2 13.4/13.5/28.2%
  （dv 的 28% 由 TE 自身 vs-ref 27.45% 主导，非回退）。
- `--ci --dtype fp8 --hopper`：一致性 gate worst **7.629e-6 OK**、`--check docs/04` OK（198 行）。

### 116.6 对标与判决

- 同 session TE FP8 纯反向 S4096 = **0.3049ms**（`harness/fa_bwd_bench.py bench --dtype fp8`）：
  ours total **5.30×**（O92 时 5.50×）、main **4.52×**（O92 时 4.75×）。**向 TE 靠近了一步。**
- **判决：正结果，默认开启。** 这是 `fp8 专项冲刺 F6-①`（降 L2 搬运：Q/dO 重读 + 跨 CTA red）
  在「不改工作划分、不改指令」前提下**第一条真正双向降 L2 的调度杠杆**（O89 的 mrev 只削尾波、
  `red` 一字不变；O93 的 `read/red` 都降）。`red` 的 dK/dV 主体墙（O83/O91/O92 已收口）不变。
- 原始输出：`src/fp8/fa_bwd_fp8_o93_{ab_s4096,ab_shapes,accuracy,tebench_s4096,
  ncu_hswap0_s4096,ncu_hswap1_s4096}.out.txt`。见 `docs/08` §5.102、`ROADMAP`「当前进度 第一百八十八轮」。

## 117. O95（第 189 轮）：**变 ks 调度**（`--ksm`：表驱动 per-m ksplit）—— **中性/负结果，opt-in，默认一行未改**

> 承接 `docs/03` §116（O93 跨 head 全局 LPT + 低 ksplit）与 ROADMAP「下一步候选 ③」的
> **「ksplit 随 m 变化（贵块多切、便宜块少切），在 O93 平衡点上再省 Q/dO 重读」**。
> 结论先行：**该子项判决为中性/负结果**——O93 的均匀 `ksplit=2` 已是全局最优点；
> 变 ks 只能改动 **Q/dO 重读**（L2 总量的一小部分），动不了 dK/dV `red` 主体，收益 <1% 且
> 常被「CTA 数下降 → 并行度/tail 变差」盖过。默认路径逐位不变；`--ksm` 保留为 opt-in 复现口。

### 117.1 动机与机制

O93 把 ksplit 从 8 收到 2，使 Q/dO 重读 8×→2×。一个自然猜想：**便宜 m 块不需要切**（它们
的 K 区间本来就短），只给最贵的若干 m 块保留更高 ksplit，即可在不抬长尾的前提下进一步减少
「Q/dO 重读 + dQ 跨 part 原子」。

**先算账（关键负结论的根因）**：主 kernel 的 L2 `read` 不是 ∝ 槽位数。它主要是 **K/V 读**，
而 K/V 读 = `Σ_slots (本 CTA 的 K tile 数) · BN·HD` = **总 (m,kv) tile 数 · BN·HD**（与 ksplit
无关，是总工作量）。ksplit/槽位数只影响 **Q/dO 的按-CTA 重复读**（每槽 `BM·HD·2B` 固定）。
S4096 H16：槽位 128→72（−44%）只省 `56·64·128·2·16 ≈ 14.7MB ≈ 0.46M` 扇区，占 L2 总量
（129.8M）的 **~0.4%**；`red` 的 dK/dV 主体（105M 的绝大多数）**一字不变**。⇒ 物理上就没有
空间。

### 117.2 实现（device 一行数学未改；单/两文件 device 逐字同源）

- **device（`fa_bwd_fp8_kernels.cuh`）**：`fp8_mma_body` 末尾加运行期参
  `const int* slot_tab = nullptr`。非空时从表解出 `(ks,part,mt)`：
  `enc = slot_tab[HSWAP ? blockIdx.y : blockIdx.x]`，`ks=enc&15`、`part=(enc>>4)&15`、
  `mt=enc>>8`；`nt_begin/nt_end` 用解出的 `ks_eff` 而非全局 `ksplit`。`slot_tab==nullptr`
  时逐式退化为 O93/历史路径（**位不变**）。非 DET 默认路径 dQ/dK/dV 仍是可交换跨 CTA
  `atomicAdd` ⇒ 变 ks 只改加法次序、数值在 fp8 噪声内。
- **host（`fa_bwd_fp8_main.cu`）**：`--ksm=N`（`-1`=关）+ `--ksmhi=K`。hswap eligible 时按
  **mt 升序（= 全局 LPT）** 构建表：前 N 个（最贵的）m 块 `ks=K`、其余 `ks=1`，每槽编码
  `(mt<<8)|(part<<4)|ks`，`grid=(H, nslots, B)`、`slot_tab` 传入 `kvtma` 主 kernel。`--ksm=64`
  即「表驱动的均匀 ks=2」，用于验证表解码与 O93 等价。
- **单文件** `fa_bwd_fp8_mma_onefile.cu` 手工同源（device 段与 host 段一并改）。

### 117.3 性能（S=4096 H16 B1 causal，同 binary A/B，event iters=40）

| 配置 | nslots | main (ms) | total (ms) | TFLOPS |
|---|---|---|---|---|
| `ksm=-1`（**O93 默认**，均匀 k=2） | 128 | **1.3670** | **1.6068** | 85.53 |
| `ksm=64 ksmhi=2`（表驱动均匀 k=2） | 128 | 1.3652 | 1.5978 | 86.02 |
| `ksm=8 ksmhi=2` | 72 | 1.3874 | 1.6187 | 84.91 |
| `ksm=16 ksmhi=2` | 80 | 1.3807 | 1.6195 | 84.86 |
| `ksm=32 ksmhi=2` | 96 | 1.4120 | 1.6431 | 83.64 |
| `ksm=48 ksmhi=2` | 112 | 1.4543 | 1.6862 | 81.51 |
| `ksm=0 ksmhi=2`（全 k=1） | 64 | 1.4488 | 1.6720 | 82.20 |
| `ksm=32 ksmhi=3`（同槽数重分配） | 128 | 1.4506 | 1.6961 | 81.03 |
| `ksm=16 ksmhi=4` | 112 | 1.4353 | 1.6814 | 81.74 |

- **`ksm=64`（128 槽）与默认 `ksm=-1` 逐项吻合**（1.3652 vs 1.3670）⇒ 表解码正确、且说明
  「变 ks 表本身」不是慢的原因。
- **所有非均匀档都更慢**；同槽数重分配（`ksm=32 ksmhi=3`，128 槽、贵块切 3）也慢 6% ⇒
  均匀 k=2 的**最小并行度最大**、负载最均衡。`ksplit=1`（64 槽）慢 6%，与 O93 一致。

### 117.4 ncu（S4096 causal，同 session，`fa_bwd_fp8_mma_kvtma_kernel`）

| 指标 | `ksm=-1`（默认） | `ksm=8`（nslots 72） | 变化 |
|---|---|---|---|
| Duration | 1.37 ms | 1.38 ms | ~中性 |
| L2 总扇区 | 129.81 M | 127.86 M | −1.5% |
| L2 `op_read` | 24.27 M | 23.74 M | −2.1% |
| L2 `op_red` | 105.38 M | 104.01 M | −1.3% |
| L2 `op_write` | 99 K | 59 K | — |
| DRAM 字节 | 553 MB | 560 MB | — |
| warps active | 18.66% | 18.56% | — |

- **证实 117.1 的账**：槽位砍 44% 但 `read` 只降 2.1%（Q/dO 重读只占 read 的小头，K/V 读
  不随 ksplit 变）；`red` 只降 1.3%（dQ 的跨 part 部分，dK/dV 主体不动）。**总 L2 降 1.5%
  低于并行度损失，Duration 不降反略升。**

### 117.5 其它形状

- S1024H32：默认 main 0.2232；`ksm=8` 0.2360（慢）、`ksm=16` 0.2228、`ksm=16 k=3` 0.2226 ⇒ 中性。
- GQA q32/kv4：默认 0.2112；`ksm=8` 0.2265（慢）、`ksm=16` 0.2101 ⇒ 中性。
- MQA q64/kv1（H=64）：默认 0.4308；**`ksm=8` 0.4198（1.026×，3 次重复稳定 0.4192–0.4202）**
  ⇒ **唯一的小正结果**，但也仅 ~2.6%，且只在此 MQA shape 观察到（H 越大、Q/dO 重读被 ×H 放大，
  收益才勉强盖过并行度损失）。**不值得做默认**（会对 MHA/GQA 造成 1–7% 回退）。

### 117.6 精度护栏与回归

- 默认路径（ksm 关）S4096 `ours_hp vs fp32 ref` relL2 dq/dk/dv **8.1485/8.2633/6.4894%**，
  `max_abs` 2.635/2.644/3.216e-1 —— 与 O93 **逐位相同**。
- `--ci --dtype fp8 --hopper`：单/两文件一致性 gate worst **7.629e-6 OK**、`--check docs/04` OK。
- **判决：中性/负结果**。`--ksm` 保留 opt-in（默认 `-1` 关，默认路径一行未改），ROADMAP
  「下一步候选 ③」的「ksplit 随 m 变化」子项到此关闭；O93 均匀 `ksplit=2` 仍是本卡默认最优点。
  原始输出 `src/fp8/fa_bwd_fp8_o95_{ab_s4096,ab_shapes,accuracy,ncu_s4096}.out.txt`。

## 118. O96（第 190 轮）：**full（非 causal）D=128 主 kernel 的 ksplit/regdq 重标定** —— **正结果，默认**

> 前 189 轮把 fp8 causal 主 kernel 调到本卡 L2 `red` 平台期后，「fp8 性能」的剩余空间一度只剩
> 「换卡」。本轮换一个角度：**检查与 causal 强绑定、但被无条件套用到 full 的两个启发式**
> （O29 的 ksplit target、O7 的 `use_regdq` 阈值）。结论先行：**full D=128 的 main 默认被这两条
> causal 标定的启发式系统性拖慢 1.3–1.6×**；改成 full 专属标定后，**L2 `red` 与总扇区各降 ~45%**，
> main 最高快 **1.59×**，且**数值与 causal 路径一字未动**。这是「降 L2 搬运量」在 full 分支上
> 一条被长期漏掉的、纯 host 的杠杆。

### 118.1 动机与机制（为什么 causal 的经验在 full 上反过来）

- **O29 的 ksplit target**（`D==128: S>=2048→8192，否则 max(2048,4*base)`）是按 **causal 的三角
  偏斜**标定的：causal 下第 m 个 m 块要扫 `(m+1)` 个 K 块，工作量从头到尾线性增长，尾波里全是
  「最贵的几块」，必须用**极细切分**（k=8/16）把它们摊到更多 CTA 上才能填满机器。
- **但 full（非 causal）每个 m 块工作量完全相同**，尾波只由「`grid=base*k` 是否落在整数个并发波
  上」决定，与块大小无关；此时**过细切分不再摊平任何东西，只剩纯浪费**：每个 CTA 要把
  Q/dO 重读一遍（L2 `read` 放大 k 倍）、dQ 还要跨 part 原子（L2 `red` 放大）。
- **O7 的 `use_regdq` 阈值**（`(S/32)/2/ksplit >= 4`）同样按 causal 标定：那个 `/2` 是「causal
  平均只扫一半三角」。full 下平均每 CTA 的 nt tile 数 = `(S/32)/ksplit`（**无折半**），阈值判断
  里的 `/2` 会让 full 在 `ksplit` 稍大时**误关寄存器 dQ 累加**（`kRegDq`），于是 dQ 退化成
  「每个 nt tile 都对本 CTA 的 dQ tile 做一次跨 CTA `atomicAdd`」——同一 (r,c) 被 RMW
  `ntiles` 次，`red` 直接翻数倍。

### 118.2 实测账（S=1024 H16 full，ncu 主 kernel `fa_bwd_fp8_mma_kvtma_kernel`）

| 指标 | O29 auto（k=8, regdq=**0**） | O96（k=3, regdq=**1**） | 变化 |
|---|---|---|---|
| Duration | 299.33 µs | **205.22 µs** | **−31%（1.46×）** |
| L2 总扇区 | 29,239,266 | **16,850,496** | **−42%** |
| L2 `op_red` | 25,165,824 | **13,762,560** | **−45%** |
| L2 `op_read` | 3,987,551 | **2,965,727** | −26% |
| L2 `op_write` | 3,554 | 39,780 | — |
| L2 利用率 | 84.64% | **68.34%** | 带宽从满降到有余 |
| sm throughput | 31.12% | 39.88% | — |
| DRAM 字节 | 39.09 MB | 39.32 MB | 中性 |

⇒ full 的 main 原本是 **L2 `red` 饱和（84.6%）**，一半的 `red` 来自「被误关 regdq 后 dQ 的
逐 tile 原子」；O96 把 `red` 砍半、L2 总量降 42%，Duration 直接 1.46×。**这不是换算法，而是
纠正了一条把 causal 经验错套到 full 上的启发式。**

### 118.3 实现（纯 host；单/两文件 device 一行未改）

- **device（`fa_bwd_fp8_kernels.cuh`）**：**未改**（`fp8_mma_body` 早已支持任意 `ksplit`/`REGDQ`）。
- **host（`fa_bwd_fp8_main.cu` + `fa_bwd_fp8_mma_onefile.cu`，两文件版与单文件版同源）**：
  1. **full D=128 的 ksplit 重标定**（`ksplit_auto && !causal && D==128`）：在 `k∈[1,8]` 里取
     「尾波空泡 `ceil(base*k/SLOTS)*SLOTS − base*k`」最小者（并列取更小 k），`SLOTS=396`
     = 本卡 D=128 fp8 main 的**3 CTA/SM × 132 SM**。显式 `--ksplit=K` 时**不覆盖**（保留 A/B）。
     实测该规则在 8 个 full shape 上**一致选到 k=3**（S4096H16 选 k=5，也确为最优）。
  2. **`use_regdq` 的 causal 折半只对 causal 生效**：
     `(D==128) && ((S/32) / (causal ? 2 : 1) / ksplit >= 4)`。causal 分支**逐字不变**。
- **单/两文件**：device 逐字同源；harness 一致性 gate 对 full case 检出
  `ours_hp vs ours_sf_hp` **逐值相同**。

### 118.4 性能（8 个 full D=128 shape，同 binary A/B，event iters=60）

| shape（B,S,H,D）full | O29 auto（k,regdq,main） | O96（k,regdq,main） | main 加速 | O96 main TFLOPS |
|---|---|---|---|---|
| (1,512,16,128) | 16,0, 0.0928ms | 3,1, **0.0665ms** | **1.40×** | 23.31 |
| (1,512,32,128) | 8,0, 0.1581ms | 3,1, **0.1189ms** | **1.33×** | 26.03 |
| (1,1024,8,128) | 16,0, 0.1670ms | 3,1, **0.1091ms** | **1.53×** | 29.91 |
| (1,1024,16,128) | 8,0, 0.3010ms | 3,1, **0.2064ms** | **1.46×** | 32.88 |
| (1,1024,32,128) | 4,1, 0.4172ms | 3,1, **0.3943ms** | 1.06× | 34.99 |
| (1,2048,8,128) | 16,0, 0.5968ms | 3,1, **0.3813ms** | **1.57×** | 37.88 |
| (1,2048,16,128) | 16,0, 1.1361ms | 3,1, **0.7159ms** | **1.59×** | 40.56 |
| (1,4096,16,128) | 8,1, 2.6713ms | 5,1, **2.6601ms** | 1.004× | 45.46 |

- **大 S（≥2048）仍受同一 L2 `red` 墙**（O93 结论不变），故 S4096 只 1.004×；收益集中在
  **base_grid 小 / 被过切或误关 regdq 的中小 shape**（1.3–1.6×）。
- k 扫描原始输出（`--regdq=1` 隔离 k 效应）见 `src/fp8/fa_bwd_fp8_o96_ksweep.out.txt`：8 个
  shape 的最优 k 一致落在 3（S4096 落在 3–8 近平）。

### 118.5 精度护栏与回归

- **数值不变量**：O96 只改「哪些 CTA 算哪个 (m,part)」与「dQ 是否寄存器累加」——两者都只改
  跨 CTA `atomicAdd` 的**加法次序/次数**，不改任何 fp8 量化口径。实测 S1024H16 full 新旧配置
  `ours vs fp32 ref` relL2 **完全相同 8.111/8.235/6.709%**（`max_abs` 5.521e-2/5.310e-2/4.025e-2，
  与旧 5.518e-2 差 ~0.05% = fp32 求和次序）。
- **causal 路径无回归**：`hswap_elig` 仍要求 `causal`，O96 的 full 分支不进 causal；实测
  S1024H32 causal 仍 **ksplit=2 / main 0.2209ms**、S4096H16 causal 仍 **ksplit=2 / main 1.3602ms**、
  `max_abs` 2.635/2.644/3.216e-1（与 O93 逐档一致），GQA kv4 / MQA kv1 亦 k=2。
- `--check docs/04` **OK（198 行）**；O96 的受影响 full case 一致性与数值用定向 A/B 核验
  （见原始输出），未重跑全量 73 case `--ci`（全量扫留作下一轮验收）。

### 118.6 结论 / 下一步

- **判决：正结果、默认开启。** full D=128 的 main 恢复「不做无谓切分」的正确工作点，
  `red`/L2 总扇区各 −45%/−42%，main 最高 **1.59×**。**注意口径**：这改善的是 **full 分支**，
  causal 旗舰（S4096）的 L2 `red` 墙仍是换卡前的天花板（O83/O91/O92 结论不变）。
- **新增的通用教训**：凡「按 causal 三角标定」的启发式（ksplit target、regdq 阈值、以及
  未来的 partial/split heuristic）都要逐条问一句「full/变长下还成立吗」，本轮已抓到两条。
- **原始输出**：`src/fp8/fa_bwd_fp8_o96_ab_full.out.txt`（A/B 主表）、
  `..._o96_causal_reg.out.txt`（causal 回归）、`..._o96_ksweep.out.txt`（k 扫描）、
  `..._o96_ncu_full_s1024.out.txt`（ncu）。见 `docs/08` §5.104。

---

## 119. O97（第 191 轮）：full（非 causal）D=256 / D=512 的 ksplit 重标定

> 承接 O96（full D=128 的同类审计）与 `ROADMAP`「下一步候选 ①」：D=256 与 D=512（MLA）
> **共用** O29 的 `target_ctas = S/2`（按 **causal MLA** 标定），而 D=128 早已改成
> `S>=2048→8192、否则 max(2048,4*base)`——两条被「causal 经验」覆盖的 full 路径一直没复核。
> **纯 host、device 一行未改、单/两文件同源**；`--ksplit=K` 仍可覆盖（保留 A/B）。

### 119.1 机制：两条路径的错法正好相反

- **D=512（MLA）主 kernel = `mma` 8-warp、smem ~223KB ⇒ 1 CTA/SM（132 并发槽）**。
  O29 的 `target=S/2, k=target/base`（`base=(S/64)*H*B`）⇒ `k = 32/H`：H=2 时恒为 **16**。
  实测 5 个 full shape 的最优 k ≈ **128/base**（S512H2→8、S1024H2/S512H4→4、S2048H2→2），
  正是「让 `grid=base*k` 对齐 132 个并发槽」——**旧档在 base 小时过切到 k=16**，白付
  Q/dO 重读与 dQ 跨 part 原子。
- **D=256 主 kernel = `fa_bwd_fp8_mma_kernel<256,64,32>`（cp.async wgmma、非 TMA，2 CTA/SM
  ⇒ **264 槽**）**。O29 给 D≠128 的 `target=S/2` ⇒ `k = S/2 / ((S/64)*H*B) = 32/H` **与 S 无关**：
  S≥2048 时 base 已大，k 只有 1–4 ⇒ **严重欠切**（并发不足、延迟藏不住）。实测 7 个 shape
  的最优 k 稳定在 **8–12**（auto 只 1–4）。S<2048 则 base 小，按 264 槽波对齐即可。

### 119.2 新规则（host，`fa_bwd_fp8_main.cu` / `fa_bwd_fp8_mma_onefile.cu` 同源）

```
if (ksplit_auto && !causal && (D == 256 || D == 512)) {
  if (D == 512 || S < 2048) {                 // 波对齐
    SLOTS = (D==512) ? 132 : 264;             // 1 / 2 CTA/SM × 132
    k = argmin_{k∈[1,KMAX]} ceil(base*k/SLOTS)*SLOTS - base*k   // KMAX=16/8，并列取更小
    if (D == 512 && k < 2) k = 2;             // 消大 base 的 k=1 回退（S4096H2）
  } else {                                    // D==256 && S>=2048：给足并发
    k = clamp(8192 / base, 1, 12);            // grid≈8192（≈31 波）
  }
  ksplit = k;
}
```

- D=512 给 k 设下限 2：S4096H2 的 base=128 已 ≈ 一整个波，波对齐给 k=1，但长 K 循环仍偏好
  ≥2 份并发（k=1 2.491ms vs k=2 2.474ms vs 旧 auto k=16 2.473ms）⇒ 下限 2 消除该回退。

### 119.3 ncu（同 binary，两条路径各一组）

| 指标 | D=512 S1024H2 full | | D=256 S2048H8 full | |
|---|---|---|---|---|
| ksplit | 16（旧） | **4（新）** | 4（旧） | **12（新）** |
| Duration | 211.33 µs | **190.78 µs** | 1.38 ms | **1.26 ms** |
| L2 `op_read` | 2,291,411 | **1,749,736（−24%）** | 8,690,755 | **12,489,421（+44%）** |
| L2 `op_red` | 12,582,912 | 12,582,912（不变） | 100,663,296 | 100,663,296（不变） |
| L2 利用率 | 60.84% | 66.83% | 72.31% | 79.64% |

⇒ 两条方向相反的机制都被证实：**D=512 是「减 k → 减 Q/dO 重读」**（read −24%、Duration −9.7%）；
**D=256 是「加 k → 升并发/藏延迟」**（read +44% 但 L2 利用率 72→80%、Duration −9%）。
`red` 主体（dK/dV 跨 m-block）与 ksplit 无关，故两组都不变——再次印证 O83/O86 的结论。

### 119.4 性能（14 个 full shape，同 binary A/B，event iters=60）

| shape（B,S,H,D）full | 新 auto（k, main） | 旧 auto（k, main） | main 加速 |
|---|---|---|---|
| (1,512,8,256)   | 4, 0.0922ms | 4, 0.0926ms | 1.004× |
| (1,512,16,256)  | 2, 0.1720ms | 2, 0.1723ms | 1.002× |
| (1,1024,8,256)  | 2, 0.3305ms | 4, 0.3493ms | 1.057× |
| (1,1024,16,256) | 1, 0.6558ms | 2, 0.7067ms | 1.078× |
| (1,2048,8,256)  | 12, 1.2457ms | 4, 1.3347ms | 1.071× |
| (1,2048,16,256) | 12, 2.4079ms | 2, 2.5697ms | 1.067× |
| (1,2048,32,256) | 8, 4.8031ms | 1, 5.1815ms | 1.079× |
| (1,4096,8,256)  | 12, 4.7285ms | 4, 5.0280ms | 1.063× |
| (1,4096,16,256) | 8, 9.4031ms | 2, 9.9368ms | 1.057× |
| (1,512,2,512)   | 8, 0.0541ms | 16, 0.0621ms | **1.148×** |
| (1,1024,2,512)  | 4, 0.1771ms | 16, 0.2034ms | **1.149×** |
| (1,2048,2,512)  | 2, 0.6564ms | 16, 0.6749ms | 1.028× |
| (1,4096,2,512)  | 2, 2.4636ms | 16, 2.4728ms | 1.004× |
| (1,512,4,512)   | 4, 0.0949ms | 8, 0.0987ms | 1.040× |

- 14 个 shape **全部 ≥ 1.00×，无回退**（D=256 大 S 5.7–7.9%、D=512 小 shape 14.8%）。
  D=512 的端到端收益被 **full MLA 的 preprocess（LSE，0.19–0.37ms）盖住**（main 只占 ~25%），
  故这里只报 main；D=256 的 preprocess 相对小。

### 119.5 精度护栏与回归

- **数值不变量**：O97 只改「哪些 CTA 算哪段 K」⇒ 只改跨 CTA `atomicAdd` 的次序。实测 5 个代表
  shape 的 `ours vs fp32 ref` relL2 新旧**逐位相同**（如 S4096H16 d256 full 8.151/8.302/6.759%；
  S2048H2 d512 full 8.184/8.324/6.787%），均在护栏内（dq≤8.2+/−0.3、dk≤8.3+/−0.3、dv≤6.5+/−0.3）。
  `max_abs` 亦不变（如 3.935e-2 / 2.452e-2 等）。
- **单/两文件一致**：`ours_hp vs ours_sf` 对受影响 full case 逐值相同（ksplit、`max_abs`、main 同档）。
- **CI**：`fa_bwd_run.py --ci` 全量三 dtype gate **OK**（fp16 1.953e-3 / bf16 7.812e-3 /
  fp8 **6.676e-6**），`--check docs/04` **OK（198 行）**。
- **causal 路径无回归**：O97 分支要求 `!causal`，且 D=128 causal 仍走 O93 的 hswap k=2；
  变长 causal 的 `use_regdq` `/2` 也逐字不变（仅 full 分支去掉 `/2`，当前 full varlen shape
  本就 regdq=1 ⇒ 无行为变化）。

### 119.6 结论 / 下一步

- **判决：正结果、默认开启。** full D=256/D=512 的 ksplit 从「causal MLA 的 `S/2`」改为按
  并发槽/工作量标定，main 1.00–1.15×、无回退。这仍属于「纠正被 causal 经验错套的启发式」，
  不改数据通路。
- **仍未复核的同类项**：变长 full 的 ksplit（本轮只把 `--ksplit` 接到了 varlen 做 A/B，
  4 个 shape 实测 auto 已近最优，仅 b8_t2904 有 ~3.6% 空间）；`partial/split` 的 full 标定。
- **原始输出**：`src/fp8/fa_bwd_fp8_o97_ab.out.txt`（A/B 主表）、`..._o97_ncu.out.txt`（ncu）。
  见 `docs/08` §5.105。

---

## 120. O98（第 192 轮）：full（非 causal）**变长**的 ksplit 重标定 —— **正结果，默认**

### 120.1 动机

O96（定长 D=128）与 O97（定长 D=256/D=512）已把「按 causal 三角标定、却无条件套到 full」的
ksplit 启发式逐条复核。**变长（`run_varlen`）路径仍是 O29 的 causal 标定**，且 O97 只把
`--ksplit` 接到 varlen 做 A/B、未改自动档。本轮补上这一条。

变长比定长多一层错：`base_grid = ceil(maxlen/BM)*H*B` 按 **maxlen** 计，短序列的 m 块是
**早退死 CTA**，于是名义网格被高估，`target/base` 推导出的 k 进一步偏。O97 结尾记「4 个
shape 实测 auto 已近最优，仅 b8_t2904 有 ~3.6% 空间」——本轮全扫后修正：**D=512 变长有
8.6% 空间，D=128 变长 b5/b8 各有 ~4–6% 空间**。

### 120.2 全扫（k∈[1,16]，total ms，iters=80，Hopper wgmma+TMA 构建）

| shape (varlen full) | maxlen/H/B | 最优 k | 备注 |
|---|---|---|---|
| `b4_t3840_h16_d128` (512/1024/2048/256) | 2048/16/4 | **4** (1.416) | k3=1.471（k3 反慢 3.9%，故不做全局 k=3） |
| `b4_t4096_h16_d128` (1024×4) | 1024/16/4 | **3** (1.120) | k4=1.162 |
| `b5_t3968_h32_d128` (128…2048) | 2048/32/5 | **3** (2.690) | old auto k=1 → 2.841（慢 5.6%） |
| `b8_t2904_h16_d128` (2048…8) | 2048/16/8 | **3~4** (1.149) | old auto k=2 → 1.194（慢 4.3%） |
| `b1_t512_h2_d512` | 512/2/1 | **8** (0.0751) | old auto k=16 → 0.0814（过切） |
| `b3_t1792_h2_d512` (256/512/1024) | 1024/2/3 | **8~11** (0.309) | old auto k=4 → 0.316 |

关键观察：**D=128 full 变长的最优 k 稳定在 3~4**（与 O96 定长 D=128 的 k=3 同源），而
causal 自动档在 `base` 偏大时给 k=1/2（欠切）；D=512 的最优正是「对齐 132 槽一个波」（k≈128/base）。

### 120.3 规则（纯 host、device 一行未改、单/两文件同源）

`run_varlen` 的自动 ksplit 增加 `!causal` 分支（`--ksplit=K` 显式时跳过）：

- **D=128**：`k = max(kp, 3)`。只把 causal 自动档的 k **抬下限到 3**——不碰 b4_t3840/b4_t4096
  已近最优的 k=4 档，只补齐被 causal 标定误判为 1/2 的欠切档（b5/b8）。这是本轮最保守、
  且对全部 4 个 shape 非回退的选择。
- **D=512（MLA，1 CTA/SM→132 槽）**：按 132 槽**波对齐**取 `k`（=O97 定长 D=512 同规则），
  下限 2；给 b1_t512 k=8、b3_t1792 k=11。
- **D=256**（无 varlen dump）：沿用 O97 定长规则（`maxlen≥2048` 取 `clamp(8192/base,1,12)`，
  否则 264 槽波对齐）。

### 120.4 性能（同 binary A/B，iters=100；total = quant+preprocess+main+convert）

| shape | new auto | new ms | 旧 auto | 旧 ms | gain |
|---|---|---|---|---|---|
| `b4_t3840_h16_d128` | k=4 | 1.4158 | k=4 | 1.4131 | 0.998×（同配置，噪声） |
| `b4_t4096_h16_d128` | k=4 | 1.1659 | k=4 | 1.1625 | 0.997×（同配置，噪声） |
| `b5_t3968_h32_d128` | **k=3** | 2.6904 | k=1 | 2.8480 | **1.059×** |
| `b8_t2904_h16_d128` | **k=3** | 1.1495 | k=2 | 1.1960 | **1.041×** |
| `b1_t512_h2_d512` | **k=8** | 0.0746 | k=16 | 0.0818 | **1.097×** |
| `b3_t1792_h2_d512` | **k=11** | 0.3108 | k=4 | 0.3165 | **1.018×** |

前两行 new==old（k 相同、仅两次 invocation），0.997–0.998 即本机 run-to-run 噪声底，
证明「不改已最优档」。其余四档 +1.8%~9.7%，**全 6 shape 无回退**。

### 120.5 ncu（main kernel = `fa_bwd_fp8_mma_kernel`）

| shape | 配置 | Duration | `op_read` | `op_red` | L2 利用率 |
|---|---|---|---|---|---|
| `b5_t3968` | k1 | 2.59ms | 28.18M | 137.1M | 54.6% |
| | **k3** | **2.49ms** | 38.67M | 143.2M | **60.4%** |
| `b8_t2904` | k2 | 1.00ms | 11.92M | 56.12M | 58.4% |
| | **k3** | **0.980ms** | 14.19M | 57.19M | **61.2%** |
| `b1_t512_d512` | k16 | 73.2µs | 0.661M | 3.146M | 44.7% |
| | **k8** | **64.2µs** | 0.533M | 3.146M | **50.8%** |
| `b3_t1792_d512` | k4 | 266.4µs | 2.473M | 16.52M | 63.0% |
| | **k11** | **258.2µs** | 3.036M | 16.52M | **65.3%** |

两条机制不同但都成立：**D=128** 下 ksplit 增加 Q/dO 重读（`op_read` +19~37%），但墙是
**延迟/并行度**，更高 L2 利用率把 Duration 压下去；**D=512** 下最优 k 恰好**减少** Q/dO
重读（k16→k8 时 `op_read` −19%），Duration 直接 −12%。两组 `op_red` 只随 ksplit 引起的
dQ 跨 part 原子略动（dK/dV 主体与 ksplit 无关，再次印证 O83/O86/O95）。

### 120.6 精度护栏与回归

- **数值不变量**：只改「哪些 CTA 算哪段 K」⇒ 只改跨 CTA `atomicAdd` 的次数/次序。6 个 shape
  `ours vs fp32 ref` relL2：dq 8.06–8.22%、dk 8.21–8.36%、dv 6.48–6.76%，全在护栏内
  （dq≤8.2+0.3 / dk≤8.3+0.3 / dv≤6.5+0.3）；`max_abs` O(0.05–0.25)。
- **单/两文件一致**：`ours vs ours_sf` 对 6 个 shape max_abs ≤ 3.58e-07（原子次序噪声，
  fp8 gate 6.676e-6 内）。
- **CI**：`fa_bwd_run.py --ci --no-run` 三 dtype gate **OK**（fp16 1.953e-3 / bf16 7.812e-3 /
  fp8 **6.676e-6**），`--check docs/04` **OK（198 行）**。
- **causal 与定长 full 无回归**：新分支要求 `!causal`；定长路径（O96/O97）逐字未改。

### 120.7 结论 / 下一步

- **判决：正结果、默认开启**（变长 full 的 ksplit 自动档）。这把 O96/O97 的「full 重标定」
  从定长补齐到变长；**fp8 的 full ksplit 自动档至此定长/变长 × D=128/256/512 全覆盖**。
- **残余**：D=128 变长 b4_t4096（k=4 已近最优但非 k=3 最优，差 ~3.7%）保留了 causal auto 的
  k=4 未动——因为全局 k=3 会反伤 b4_t3840；本轮取「下限 3」的保守非回退解。若要吃到这 3.7%，
  需要能区分「均匀长序列 vs 混合长度」的判据（`total_mt / (nmb*B)` 利用率）——留 backlog。
- **原始输出**：`src/fp8/fa_bwd_fp8_o98_varlen_full_ab.out.txt`（A/B 主表）、
  `..._o98_ksweep.out.txt`（全扫）、`..._o98_ncu.out.txt`（ncu）、
  `..._o98_fa3_baseline.out.txt`（纯反向 FA2/FA3/TE 基线）。见 `docs/08` §5.106。

---

## 121. O99（第 193 轮）：**causal D=256 的 ksplit 重标定** —— **正结果，默认**

> 承接 O96/O97/O98 的「把按 causal 经验错套到 full 的 ksplit 启发式逐条复核」。此前只审了
> **full**，而 **causal D=256** 一直沿用 O29 的 `target_ctas = S/2`——那是按 **causal MLA
> （D=512，1 CTA/SM）** 标的。本轮把 causal D=256 补上。**纯 host、device 一行未改、单/两文件
> 同源**；`--ksplit=K` 显式给出时不覆盖（保留 A/B）。

### 121.1 机制：`S/2` 对 causal D=256 是「与 S 无关的欠切」

- **D=256 主 kernel = `fa_bwd_fp8_mma_kernel<256,64,32>`（cp.async wgmma、非 TMA，2 CTA/SM
  ⇒ 264 并发槽）**。O29 给 `D != 128` 的 `target = S/2` ⇒ `k = S/2 / ((S/64)*H*B) = 32/(H*B)`
  **与 S 无关**：H=8 时恒 k=4、H=16 时恒 k=2。causal 三角偏斜本要靠细切分摊平尾波，于是
  小/中 S **严重欠切**（并发不足、全局延迟藏不住）。
- 实测 6 个 causal D=256 shape 全扫 k∈[1,32]（`src/fp8/fa_bwd_fp8_o99_ksweep.out.txt`）：

| shape（B,S,H,D；Hkv） | base | nblk | 旧 auto k | 旧 total | 最优 k | 最优 total | 加速 |
|---|---|---|---|---|---|---|---|
| (1,512,8,256)  | 64  | 8  | 4 | 0.1086ms | 8  | 0.1017ms | **1.068×** |
| (1,1024,8,256) | 128 | 16 | 4 | 0.3058ms | 8  | 0.2749ms | **1.112×** |
| (1,2048,8,256) | 256 | 32 | 4 | 0.8833ms | 16 | 0.8001ms | **1.104×** |
| (1,4096,8,256) | 512 | 64 | 4 | 2.9015ms | 16 | 2.7498ms | **1.055×** |
| (1,1024,16,256; Hkv4) | 256 | 16 | 2 | 0.5424ms | 6  | 0.4745ms | **1.143×** |
| (1,2048,16,256) | 512 | 32 | 2 | 1.6999ms | 12 | 1.5145ms | **1.122×** |

- 最优 k 一致地落在「`grid = base*k ≈ 2*S`」（即 `k ≈ 128/(H*B)`）附近：s1024→2048、
  s2048→4096、s4096→8192（H8 取到 k=16）；小 S 由 `k ≤ nblk` 封顶（s512 取 k=8）。
  这与 O29 给 **D=128 causal 大 S** 的目标 `8192 = 2*S` 同源。

### 121.2 新规则（host，`fa_bwd_fp8_main.cu` / `fa_bwd_fp8_mma_onefile.cu` 同源）

```
if (ksplit_auto && causal && D == 256) {
  nblk = ceil(S / 64);                 // 变长用 maxlen
  k = clamp(2*S / base_grid, 1, 16);   // grid = base*k ≈ 2*S
  if (k > nblk) k = nblk;              // 不切得比 m 块数还细
  ksplit = k;
}
```

选中：s512H8 **8** / s1024H8 **16** / s2048H8 **16** / s4096H8 **16** / s1024H16kv4 **8** /
s2048H16 **8**——全部距 per-shape 最优 **≤1.7%**，全部 ≥1.05×。变长 `run_varlen` 的同名分支
用 `maxlen` 当串长（**当前无 D=256 变长 dump，按定长实测外推、未单独测量**）。

### 121.3 ncu（main kernel；同 binary，new vs 旧 auto 两档）

| shape | 配置 | Duration | `op_read` | `op_red` | L2 利用率 |
|---|---|---|---|---|---|
| s2048H8 | 旧 k=4 | 769.6µs | 6.05M | 51.90M | 67.1% |
|  | **新 k=16** | **681.5µs** | 8.41M | 51.90M | **76.4%** |
| s1024H16kv4 | 旧 k=2 | 464.1µs | 2.93M | 26.74M | 57.4% |
|  | **新 k=8** | **397.1µs** | 4.14M | 26.74M | **67.7%** |

机制与 O97 的 D=256 full 完全同型：**加 k → `op_read`（Q/dO 重读）升，但并发/延迟隐藏变好、
L2 利用率抬高，Duration 直接下降**；**两组 `op_red` 一字不变**（dK/dV 跨 m-block 主体与 ksplit
无关，复证 O83/O86/O95）。故这是「不改工作划分、只买并发」的调度杠杆。

### 121.4 精度护栏与回归

- **数值不变量**：只改「哪些 CTA 算哪段 K」⇒ 只改跨 CTA `atomicAdd` 的次数/次序。6 个 shape
  `ours vs fp32 ref` relL2：dq 8.15–8.33%、dk 8.33–8.48%、dv 6.39–6.50%，全在护栏内
  （dq≤8.2+0.3 / dk≤8.3+0.3 / dv≤6.5+0.3）；`max_abs` O(0.21–0.62)。
- **单/两文件一致**：`ours vs ours_sf` 对这 6 个 shape **逐位相同**（relL2 完全一致）。
- **全量 CI**（`fa_bwd_run.py --ci`，**93 case**）：三 dtype 一致性 gate **OK**（fp16 1.953e-3 /
  bf16 7.812e-3 / fp8 **6.676e-6**）；顺带把 O96/O97/O98 遗留的 `docs/04` 内嵌表同步（198→214 行）。
- **无回归**：`!causal` 走 O96/O97/O98；D=128 causal 走 O93 的 hswap k=2；D=512 causal 仍 `S/2`。
  实测 D=128 causal S4096H16 仍 k=2 / total 1.616ms、full D=256/512 逐档不变。

### 121.5 性能与峰值占比

- ours fp8 causal D=256 total：s2048H8 **0.805ms（42.7 TF）** / s4096H8 **2.757ms（49.9 TF）**，
  FP8 峰值 1978.8 TF ⇒ **2.2% / 2.5%**。
- **外部基线**：TE fp8 `fused_attn` 对 D=256 **causal** 报
  `Invalid combination of data type and sequence`（dump 时确认），FA3 不支持 fp8 ⇒ 无 fp8 外部列。
  同 shape fp16 纯反向（`fa_bwd_bench.py bench`）TE 0.216/0.592ms（317.7/464.5 TF）、FA2
  0.301/0.863ms（228.6/318.5 TF），D=256 反向 `fa3=NA`——仅供数量级参照（dtype 不同、不可直接比）。

### 121.6 结论 / 下一步

- **判决：正结果、默认开启。** causal D=256 的 ksplit 自动档从「causal MLA 的 `S/2`」改为
  「`grid≈2*S`（`k≈128/(H*B)`，按 `nblk` 封顶）」，6 shape 全部 ≥1.05×（最高 1.14×）、无回退。
  仍属「纠正被 causal 经验错套的启发式」，不改数据通路。
- 至此 **fp8 ksplit 自动档：causal/full × 定长/变长 × D=128/256/512 全部复核过**。causal 旗舰
  （D=128 S4096）的 L2 `red` 主体墙仍是唯一真杠杆，本卡无软件解（见「阻塞」）。
- **原始输出**：`src/fp8/fa_bwd_fp8_o99_ab.out.txt`（A/B 主表）、`..._o99_ksweep.out.txt`
  （全扫）、`..._o99_ncu.out.txt`（ncu）、`..._o99_accuracy.out.txt`（relL2）、
  `..._o99_fa3_baseline.out.txt`（外部基线参照）、`..._o99_ci.out.txt`（全量 CI）。见 `docs/08` §5.107。
