# bf16 反向实现分析（P2-1）

> 代码：**单文件** `src/bf16/fa_bwd_bf16_onefile.cu`（自包含：preprocess + main kernel + launcher + 自测 main）；
> **两文件** `src/bf16/fa_bwd_bf16_kernels.cuh` + `fa_bwd_bf16_main.cu`（P2-2，见第 6b 节）。
> 由 fp16 单文件 `src/fp16/fa_bwd_fp16_onefile.cu` **dtype 参数化**而来（`__half`→`__nv_bfloat16`，
> `__half2float`→`__bfloat162float`，`__float2half`→`__float2bfloat16`），算法/三段式/线程映射完全一致。
> 在此基础上加了一处 **bf16 专属优化：K/V smem 行距 padding**（见第 3 节）。
> 后续新增 **GQA/MQA 支持**（P5-3，见第 6c 节；`Hkv` 映射同 fp16）。
> **O5b**：又把 main kernel 换成**张量核 `mma.m16n8k16` + `ldmatrix`**（单/两文件
> `fa_bwd_bf16_mma_{onefile.cu, kernels.cuh+main.cu}`，见第 6e 节；main 9.4–9.9×）。
>
> 实测原始输出：`fa_bwd_bf16_onefile_s512.out.txt`、`..._s4096.out.txt`、
> `..._ncu_main.out.txt`（ncu `--set full`）、`..._refbench.out.txt`（FA/TE 基线）、
> `fa_bwd_bf16_{main,onefile}_p53_gqa.out.txt`（P5-3 数值）、`..._p53_ncu_*.out.txt`、
> `fa_bwd_bf16_{main,onefile}_p53_mla_*.out.txt` 与 `..._p53_ncu_mla_*.out.txt`（P5-3 续：MLA d=512）。
> 对照：`docs/01-fp16-bwd-impl.md`、`docs/00-fa-bwd-optimization-catalog.md`、`../ROADMAP.md`。

---

## 1. 实现结构

与 fp16 **逐字同构**（三段式 FA2 风格）：

| 段 | 函数 | 职责 |
|---|---|---|
| preprocess | `preprocess_kernel` | 逐行求 `LSE=logsumexp(scale·QKᵀ)` 与 `D=rowsum(dO∘O)` |
| main | `fa_bwd_bf16_kernel` | 每个 Q 块固定，遍历 K/V 块（1colblock），recompute S/P，累加 dQ/dK/dV |
| convert | `convert_kernel` | fp32 累加缓冲 `dq/dk/dv` → bf16 写回 |

分块：`BM=64`、`BN=32`、`THREADS=128`、head_dim=128；fp32 累加；dQ 在 smem 累加、dK/dV 全局
`atomicAdd`；causal 整块跳过 + 对角 mask；动态 smem（~98.5KB）需 `cudaFuncSetAttribute`。
详见 `docs/01-fp16-bwd-impl.md` 第 1、2 节，此处不重复。

**bf16 与 fp16 唯一的语义差异只是元素类型**：preprocess/main/convert 的所有数学都是 fp32 累加，
输入 smem 存 bf16，误差由 bf16 输入量化决定（容差 ~1e-2，见第 4 节）。

---

## 2. 发现：bf16 标量生成比 fp16 慢 2.9×（同款代码）

dtype 参数化后跑 S=4096，main kernel 从 fp16 的 **68.2 ms 掉到 197.1 ms**（2.9× 慢），
而 preprocess（纯 global 读 + 标量点积）两者一致（69.4 ms），排除跑机/降频因素。
`ptxas -v` 两者寄存器相近（54 vs 52，0 spill），但 SASS 指令构成不同：

| 指令 | fp16 main | bf16 main |
|---|---|---|
| `HADD2.F32`（half2→float，成对） | 1042 | 0 |
| `LDS.U16`（标量取半字） | 512 | **1024** |
| `SHF.L.U32`（bf16→fp32 左移 16） | 86 | **679** |
| `IMAD.U32` | — | 430 |

原因：ptxas 会把 `__half2float` 识别成 `HADD2.F32`（消费一个 `LDS.32/64` 的 half2），
而 `__bfloat162float` 走「`LDS.U16` 标量取半字 + `SHF.L` 左移」路径，**K/V 行读被标量化**。
由于 smem 里 K/V 行距 = 128 个 bf16 = **256B，恰是 128B bank 周期的整数倍**，
「lane↔K/V 行」的读（跨 lane 步长 256B）会全部落到同一 bank：

```
ncu（padding 前，main S=512）：
  L1/TEX Cache Throughput   %   78.47   <- 最高单元
  Compute (SM) Throughput   %    4.59
  DRAM Throughput           %    0.09
  9.5-way bank conflict，585,108,389 次，89% 多余 shared wavefronts
```

对比 fp16（75% 多余 wavefronts、4.6-way）——bf16 把它放大成了真正的瓶颈。

---

## 3. 优化：K/V smem 行距 padding（+2 个元素）

把 `Ks/Vs` 的行距从 `kHeadDim=128` 改为 `kHeadDim + 2 = 130`（元素数）：

```cpp
static constexpr int kKVStride = kHeadDim + 2;  // +2 bf16 = +4B
bf16* Vs  = Ks + BN * kKVStride;
bf16* dOs = Vs + BN * kKVStride;
```

跨 lane 步长从 256B 变成 `130*2 = 260B`。按 bank word（4B）算：
`260/4 = 65`，`65 mod 32 = 1` ⇒ 地址 `lane` 的 bank = `(65·lane + d/2) mod 32 = (lane + d/2) mod 32`，
32 个 lane 恰好铺满 32 个 bank，**冲突归零**（`d` 对所有 lane 相同，`Qs/dOs` 的读是广播或连续，
不 padding）。仅改 `Ks/Vs` 的基址与三处索引（`QKᵀ`/`dP`/`dQ` 里的 `krow/vrow`），
`S/P/dS` 的数学完全不动。

| 指标（main，S=512） | padding 前 | padding 后 |
|---|---|---|
| Duration | 5.83 ms | **1.98 ms** |
| L1/TEX Cache Throughput | 78.47% | **24.87%** |
| Compute (SM) Throughput | 4.59% | 13.51% |
| 多余 shared wavefronts | 89%（9.5-way） | 已低于阈值，不再报警 |
| bank conflicts | 585.1 M | ~0 |

---

## 4. 数值对拍（vs fp32 ref，O 取 `ref_o.npy`）

容差口径：bf16 ~1e-2。`max_rel = max |a−b|/(|b|+1e-3)`。

### S=512, B=1, H=16, D=128, causal（`b1_s512_h16_d128_causal_bf16`）

| | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|
| **ours** | **6.892e-3** | **8.110e-3** | **1.365e-2** |
| FA 2.7.4 | 1.040e-2 | 1.261e-2 | 1.365e-2 |
| TE 2.14 | 1.374e-2 | 1.068e-2 | 1.365e-2 |

### S=4096, B=1, H=16, D=128, causal

| | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|
| **ours** | **8.895e-3** | **8.078e-3** | **1.494e-2** |
| FA 2.7.4 | 1.441e-2 | 1.332e-2 | 1.631e-2 |
| TE 2.14 | 1.524e-2 | 1.763e-2 | 1.631e-2 |

**结论**：我们的 bf16 实现与 ref 的偏差和 FA/TE **同量级（bf16 量化噪声，~1e-2）**，
且 dq/dk 略优于 FA/TE（因为我们全程 fp32 累加、无 split/atomic 误差），无系统性误差，正确性达标。
padding 前后数值**逐位一致**（同一 `max_abs` 数字）。

---

## 5. ncu 剖析（main kernel，S=512 causal，`--set full`，padding 后）

```
DRAM Throughput          %    0.26      <- HBM 几乎空闲
L2  Cache Throughput     %    0.82
L1/TEX Cache Throughput  %   24.87      <- 已从 78% 降下来
Compute (SM) Throughput  %   13.51
Issue Slots Busy         %   11.87
No Eligible              %   78.34      <- 发射口大多数时间没有可发射 warp
Achieved Occupancy       %    6.25      <- 受 98.5KB 动态 smem 限制，1 CTA/SM
Waves Per SM                   0.48      <- grid 只有 128 个 block（<132 SM）
Registers Per Thread     52
Dynamic Shared Memory    98.56 KB/block
Warp Cycles Per Issued Instruction  4.61
top stall: fixed-latency execution dependency (37.3%)
```

### bound 结论（与 fp16 版对比）
1. **bank conflict 已消**：L1/TEX 从 78.5% → 24.9%，不再报警；bf16 的「慢 2.9×」问题解决。
2. **不再是 smem 访问 bound**（fp16 版最高是 L1/TEX 53.5%）；现在最高单元只有 24.9%，
   且 **DRAM 0.26%、Compute 13.5% 都很低**。
3. **当前真正的瓶颈是「延迟 / 并行度」**：`No Eligible 78%`、`Issued Ipc` 低、
   top stall 是 **fixed-latency execution dependency（37.3%）**；根因是
   **occupancy 6.25%（1 CTA/SM，98.5KB smem）+ grid 0.48 wave**——SM 大量空转，
   没有足够的 warp 来隐藏 smem/ALU 延迟。
4. 下一步优先级（与 backlog 一致）：**降 smem / 提 occupancy → 上张量核 + 流水**。
   注意：padding 只对 bf16 做了（fp16 因成对读冲突仅 4.6-way，未改）；若要，fp16 同样可受益。

---

## 6. 性能对标（CUPTI 纯 device 时间，FA/TE 由 `fa_bwd_bench.py bench` 测得）

FLOPs 口径与 harness 一致：`4·B·S²·H·D`（causal 实际约一半，尚未折算）。
H100 峰值：BF16 Tensor Core dense ≈ 989 TFLOPS。

| shape | ours (total) | FA 2.7.4 | TE 2.14 | ours 峰值占比 |
|---|---|---|---|---|
| B1 S512 H16 D128 causal | 3.108 ms / **0.69 TF** | 0.0687 ms / 31.25 TF | 0.0455 ms / 47.18 TF | 0.07% |
| B1 S4096 H16 D128 causal | 111.65 ms / **1.23 TF** | 1.0479 ms / 131.16 TF | 0.5894 ms / 233.16 TF | 0.12% |

分段时间（ours）：

| shape | preprocess | main | convert |
|---|---|---|---|
| S=512 | 1.195 ms | 1.884 ms | 0.029 ms |
| S=4096 | 69.41 ms | 42.17 ms | 0.066 ms |

**结论**：
- 本版本仍是**正确性优先的标量（CUDA-core）实现**，性能约为 FA 的 ~0.5–2%、TE 的 ~0.5%。
- **padding 优化效果显著**：S=512 main 5.76→1.88 ms（**3.1×**），S=4096 main 197.1→42.2 ms（**4.7×**）；
  现在 bf16 的 main（1.88/42.2 ms）已经**快过 fp16 同款**（2.38/68.2 ms）。
- S=4096 时 **preprocess（69.4 ms）超过 main（42.2 ms）**：preprocess 的 LSE 重算没有分块，
  对 causal 是 O(S²) 标量点积，成为端到端新瓶颈（FA 的 LSE 来自前向，反向不付这笔）。
  若要端到端对标，需把 LSE 并入前向或对 preprocess 分块/向量化（列入 backlog）。
- 性能优化（张量核、流水、提 occupancy）仍是后续任务。

---

## 6b. 两文件版（P2-2）

由单文件版 `fa_bwd_bf16_onefile.cu` 拆分为：

- `src/bf16/fa_bwd_bf16_kernels.cuh`：**device** 部分（编译期常量含 `kKVStride` padding、
  `preprocess_kernel` / `fa_bwd_bf16_kernel` / `convert_kernel`、`SMEM_BYTES`）；
- `src/bf16/fa_bwd_bf16_main.cu`：**host** 部分（`#include "fa_bwd_bf16_kernels.cuh"` + npy 读取 /
  launcher / 自测对拍）。

kernel 代码与单文件**逐字一致**（仅移入 `.cuh` 并加 include guard）。逐指标核对与单文件**无差异**：

| 指标 | 单文件 | 两文件 |
|---|---|---|
| S=512 dq/dk/dv max_abs | 6.892 / 8.110 / 13.65e-3 | **6.892 / 8.110 / 13.65e-3** |
| S=4096 dq/dk/dv max_abs | 8.895 / 8.078 / 14.94e-3 | **8.895 / 8.078 / 14.94e-3** |
| S=512 main | 1.884 ms | 1.898 ms |
| S=4096 main | 42.17 ms | 42.16 ms |
| ncu DRAM / L1TEX / Compute | 0.26 / 24.87 / 13.51 % | 0.25 / 24.89 / 13.52 % |
| ncu occupancy / waves / regs / smem | 6.25% / 0.48 / 52 / 98.56KB | 6.25% / 0.48 / 52 / 98.56KB |
| Executed Instructions | 247,182,666 | 247,182,666 |

原始输出：`fa_bwd_bf16_main_s512.out.txt`、`..._s4096.out.txt`、`..._ncu_main.out.txt`。

---

## 6c. GQA / MQA（P5-3）

**改动**（单/两文件同源）：与 fp16（`docs/01` §8）完全同一口径——把 KV 头数 `Hkv` 作为运行参数，
第 `h` 个 Q 头映射到 KV 头 `hkv = h/(H/Hkv)`（对齐 `ref_attn` 的 `repeat_interleave`）。

- `preprocess_kernel` / `fa_bwd_bf16_kernel`：新增 `int Hkv` 入参，K/V 行索引
  `(((b*S+j)*H + h)*D)` → `(((b*S+j)*Hkv + hkv)*D)`；`dk_acc/dv_acc` 的基址同样换用 `Hkv/hkv`；
  Q/dO/dQ 仍用 `H/h`（Q 头数）。
- `convert_kernel`：收 `(n_q, n_kv)` 两个长度，`dk/dv` 按 `B*S*Hkv*D` 转换。
- host：从 `k.npy` 的 shape[2] 读 `Hkv`，按 `n`/`nkv` 分配与对拍；`Hkv==H` 时逐式退化为 MHA，
  **MHA 回归逐位不变**（S512 6.892/8.110/1.365e-2；S4096 8.895/8.078/1.494e-2，与第 4 节一致）。

**数值对拍（ours-vs-ref，bf16 causal，B1 S1024 D128）**：

| case | dq | dk | dv | FA vs ref (dq/dk/dv) | TE vs ref (dq/dk/dv) |
|---|---|---|---|---|---|
| h40 kv8 | 1.011e-2 | 1.885e-2 | 3.150e-2 | 1.233e-2 / 2.491e-2 / 3.411e-2 | 1.303e-2 / 2.491e-2 / 3.411e-2 |
| h32 kv4 | 9.631e-3 | 2.019e-2 | 3.094e-2 | 1.163e-2 / 3.110e-2 / 3.627e-2 | 1.238e-2 / 3.140e-2 / 3.627e-2 |
| h64 kv4 | 1.319e-2 | 2.716e-2 | 3.152e-2 | 1.710e-2 / 3.543e-2 / 6.511e-2 | 1.710e-2 / 3.534e-2 / 6.511e-2 |
| h64 kv1 (MQA) | 1.066e-2 | 3.385e-2 | 5.891e-2 | 1.793e-2 / 4.933e-2 / 8.386e-2 | 1.300e-2 / 6.018e-2 / 8.386e-2 |

**结论**：全部为 bf16 噪声量级（~1e-2），且 **ours 的 dq/dk/dv 均 ≤ FA/TE 同量级**（多数直接更小），
无系统误差。单文件与两文件**逐位相同**。原始输出 `src/bf16/fa_bwd_bf16_{main,onefile}_p53_gqa.out.txt`、
`src/bf16/fa_bwd_bf16_p53_reg.out.txt`。

**性能对标**（CUPTI，FLOPs 口径同文档 `4·B·S·H·S·D`；FA/TE 的 TFLOPS 由 harness 按 `(D+Dv)` 计）：

| case | ours total | ours pre / main | FA 2.7.4 | TE 2.14 | ours 峰值占比 |
|---|---|---|---|---|---|
| h40 kv8 | 18.816 ms / 1.14 TF | 10.949 / 7.886 ms | 0.2618 ms / 164.1 TF | 0.1687 ms / 254.6 TF | 0.12% |
| h32 kv4 | 15.539 ms / 1.11 TF | 8.784 / 6.748 ms | 0.2211 ms / 155.4 TF | 0.1427 ms / 240.7 TF | 0.11% |
| h64 kv4 | 28.443 ms / 1.21 TF | 17.432 / 10.963 ms | 0.3654 ms / 188.1 TF | 0.2448 ms / 280.8 TF | 0.12% |
| h64 kv1 | 28.394 ms / 1.21 TF | 17.430 / 10.895 ms | 0.3627 ms / 189.5 TF | 0.2655 ms / 258.8 TF | 0.12% |

> FA/TE 基线原始输出 `src/fa_bwd_bench_requested_bf16_p53.out.txt`（MLA d=512 两者均 NA）。

**ncu（main，h32 kv4 S1024）**：

```
DRAM Throughput          %    0.12      <- HBM 几乎空闲
L1/TEX Cache Throughput  %   43.75
L2  Cache Throughput     %    1.55
Compute (SM) Throughput  %   30.19
Achieved / Theoretical Occupancy  %  11.14 / 12.50   <- 98.56KB smem，2 CTA/SM 上限
Waves Per SM                   1.94
Registers Per Thread     52
bank conflicts           375,879 / 534.6M wavefronts（~0.07%，padding 生效）
stall: wait(fixed-latency) 1.74 | short_scoreboard 1.01 | long_scoreboard 0.39 | barrier 0.02
```

bound 与 bf16 MHA 同：**低 occupancy（1–2 CTA/SM）+ smem 依赖/固定延迟**，非带宽/算力
（DRAM 0.12%、Compute 30%）。原始输出 `src/bf16/fa_bwd_bf16_main_p53_ncu_gqa_kv4.out.txt`、
`..._p53_ncu_stall_kv4.out.txt`。

---

## 6d. MLA（head_dim=512，P5-3 续）

**背景**：fp16 已在 P5-2 把 `head_dim` 模板化并支持 MLA 的 `D=Dv=512`（`docs/01` §9）；
bf16 与 fp16 同源，本轮把同一改造落到 bf16（单/两文件），补齐 P5-3 的 MLA 部分。

**改动**（与 fp16 逐字同构，单/两文件同源）：

- 引入 `template <int HD, int BM> struct BwdTraits`：`WM_ROWS/WN_ROWS/NCH=HD/32`、
  `KVStride=HD+2`（padding 随 head_dim 走）、`smem_bytes`；删除全局 `kHeadDim=128/BM=64/...`。
- `preprocess_kernel` 增加运行时 `int HD`（`kHeadDim`→`HD`）。
- `fa_bwd_bf16_kernel` 改为 `template <HD,BM>`；三处 head-dim 分段循环
  `kk<4`→`kk<NCH`、`acc[4]`→`acc[NCH]`；K/V 行距 `kKVStride`→`T::KVStride`。
- host：`launch_bwd_main<HD,BM>` 按 `D` 分派——**`HD=128→BM=64`**（回归）、
  **`HD=512→BM=16`**（容量所限）。

**为什么 `HD=512` 用 `BM=16`**：`dQs[BM*HD]` 是 fp32、占比最大，`HD=512,BM=64` 时
`Qs+dOs(half)131072 + Ks+Vs(half,pad)65792 + Ss+Ps(fp32)16384 + dQs(fp32)131072 ≈ 336KB`
超 smem 上限；`BM=16` 时降到 **135.42KB**（ncu decimal；= 132.25KiB），1 CTA/SM。
与 fp16 的 `HD=512→BM=16` 一致；bf16 因 padding 比 fp16 多 256B（135.17→135.42KB）。

**数值对拍（ours-vs-ref，bf16 causal，D=Dv=512）**：

| MLA case (B1) | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|
| (1,256,2,512) | 8.240e-3 | 7.905e-3 | 1.504e-2 |
| (1,512,4,512) | 1.043e-2 | 1.033e-2 | 1.385e-2 |
| (1,1024,2,512) | 5.152e-3 | 7.742e-3 | 1.557e-2 |

均为 bf16 噪声量级（~1e-2），与 fp16 MLA（1.3–2.9e-3）同量级放大比例一致；
FA/TE 反向不支持 head_dim=512（`fa=NA`/`te=NA`），只有 fp32 ref 可对。
**MHA D=128 回归逐位不变**（S512 6.892/8.110/1.365e-2，与第 4 节一致）。
单文件与两文件**逐位相同**（同一 `max_abs` 数字，main 时间差 <1% 噪声）。

**性能（CUDA event，ours）**：

| MLA case (B1) | preprocess | main | total | TFLOPS | 峰值占比(989) |
|---|---|---|---|---|---|
| (1,256,2,512) | 0.213 ms | 0.748 ms | 0.972 ms | 0.28 | 0.03% |
| (1,512,4,512) | 1.297 ms | 1.461 ms | 2.872 ms | 0.75 | 0.08% |
| (1,1024,2,512) | 2.471 ms | 2.905 ms | 5.522 ms | 0.78 | 0.08% |

对比 fp16 MLA 的 main（1.058 / 2.080 / 4.143 ms），bf16 的 main **快 1.4–1.6×**——
与 MHA 一致，得益于 K/V padding 把 bf16 的 `LDS.U16+SHF` 标量读 bank conflict 消掉。

**ncu（main，S=1024 H2 D=512，`--set full` + stall metrics）**：

```
Duration                       ms   3.78        <- ncu 重放口径（event 2.91ms）
DRAM Throughput                %    0.14        <- HBM 几乎空闲
L2  Cache Throughput           %    2.33
L1/TEX Cache Throughput        %   26.68
Compute (SM) Throughput        %   13.07
Achieved / Theoretical Occupancy % 6.25 / 6.25  <- 135.42KB smem，1 CTA/SM
Waves Per SM                        0.97
Registers Per Thread       register  48
Dynamic Shared Memory      KB/block  135.42
bank conflicts (ld/st)              5,315 / 8   <- padding 生效，基本无冲突
stall: long_scoreboard 1.71 | wait(fixed-latency) 1.35 | short_scoreboard 0.25 | barrier 0.02
```

**bound 结论**：与 MHA/GQA 的 bf16 版不同——fp16 MLA 的 bound 是 **smem bank conflict
（MIO scoreboard 36%、shared load 76.5% 多余 wavefront）**；bf16 MLA 因为 padding 已在，
**冲突基本归零**（5.3K / 1.23e8 wavefronts），换成 **long_scoreboard 主导（1.71 / 4.55 ≈ 37.6%）+
fixed-latency wait 1.35**。即 bound = **全局访存延迟 + 低并行度**（occupancy 6.25%、1 CTA/SM、
`No Eligible 77.97%`），非带宽/算力（DRAM 0.14%、Compute 13.1%）。与 bf16 MHA 的
「延迟/并行度受限」定性一致；下一步同样是把 MLA 的 smem 降下来冲 2 CTA/SM、上张量核/流水。

原始输出：`fa_bwd_bf16_main_p53_mla_*.out.txt`、`fa_bwd_bf16_onefile_p53_mla_*.out.txt`、
`fa_bwd_bf16_main_p53_ncu_mla_s1024h2.out.txt`、`..._p53_ncu_mla_stall_s1024h2.out.txt`。

---

## 6e. 张量核版（O5b，单/两文件）

> 代码：**两文件** `src/bf16/fa_bwd_bf16_mma_kernels.cuh` + `fa_bwd_bf16_mma_main.cu`；
> **单文件** `src/bf16/fa_bwd_bf16_mma_onefile.cu`（由两文件拼接，device 代码逐字一致）。
> 由 O5 的 fp16 张量核版 `src/fp16/fa_bwd_fp16_mma_*.cu` **dtype 参数化**而来。

**动机**：P2 的 `fa_bwd_bf16_*.cu` 是**正确性优先的标量 golden**（CUDA-core FFMA），
main 只有 ~1 TFLOPS；O5 已把 fp16 反向换成张量核（main 11.8–14.9×）。O5b 把同一后端
移植到 bf16（对齐 ROADMAP「当前冲刺」第 1 项：O5 收尾）。

**改动（相对 fp16 张量核版逐字同构，仅 dtype 替换）**：
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` → `...f32.bf16.bf16.f32`；
- `__half`→`__nv_bfloat16`、`__half2float/__float2half`→`__bfloat162float/__float2bfloat16`；
- smem 布局、ldmatrix（`.b16`，bf16 与 fp16 位宽/布局同构）、padding（`LD=HD+8`、`LDP=BM+8`、
  `LDS=BN+8`）、dQ 寄存器累加、dK/dV `red_add2`（float2 atomicAdd）**全部不变**。

5 个 GEMM（HD=128, BM=64, BN=32）：① `S=scale·QKᵀ`（B=K `[N][K]`，`ldmatrix.x2`）② `dP=dO·Vᵀ`
（同）：③ `dV=Pᵀ·dO` ④ `dK=scale·dSᵀ·Q` ⑤ `dQ=scale·dS·K`（③④⑤ B 用 `ldmatrix.x2.trans`）。
P/dS 就地转 bf16（`PsT/dSsT/dSs`），D/LSE 用 fp32；`__launch_bounds__(128,3)` → 168 regs、0 spill、
66.56KB smem、3 CTA/SM。

**数值对拍（ours-vs-ref，bf16 causal，max_abs/max_rel）**：

| shape | dq | dk | dv | 参考：FA(FLA) | TE |
|---|---|---|---|---|---|
| S=512 H16 D128 (`b1_s512_h16_d128_causal_bf16`) | 9.00e-3 / 1.49 | 1.26e-2 / 1.78 | 1.37e-2 / 1.88 | 1.04e-2 / 1.26e-2 / 1.37e-2 | 1.37e-2 / 1.07e-2 / 1.37e-2 |
| S=4096 H16 D128 | 1.51e-2 / 1.42 | 1.34e-2 / 1.22 | 1.63e-2 / 1.98 | 1.44e-2 / 1.33e-2 / 1.63e-2 | 1.52e-2 / 1.76e-2 / 1.63e-2 |
| S=1024 H32 kv4 | 1.20e-2 | 2.13e-2 | 3.16e-2 | 1.16e-2 / 3.11e-2 / 3.63e-2 | 1.24e-2 / 3.14e-2 / 3.63e-2 |
| S=1024 H40 kv8 | 1.23e-2 | 1.93e-2 | 3.15e-2 | 1.23e-2 / 2.49e-2 / 3.41e-2 | 1.30e-2 / 2.49e-2 / 3.41e-2 |
| S=1024 H64 kv4 | 1.35e-2 | 3.09e-2 | 4.42e-2 | 1.71e-2 / 3.54e-2 / 6.51e-2 | 1.71e-2 / 3.53e-2 / 6.51e-2 |
| S=1024 H64 kv1 (MQA) | 1.19e-2 | 4.56e-2 | 7.20e-2 | 1.79e-2 / 4.93e-2 / 8.39e-2 | 1.30e-2 / 6.02e-2 / 8.39e-2 |

全部是 bf16 噪声量级（~1e-2），**多数情形 ≤ FA/TE**，无系统误差。单文件与两文件逐位一致
（S=512 三者 max_abs 完全相同）。

**性能（CUDA event；main-only TFLOPS 用与 `fa_vs_te_bwd_only.py` 一致的口径 `4·B·S·H·S·(D+Dv)`）**：

| shape | preprocess | main | convert | total | main TFLOPS | FA3 | TE2.14 | FA2.7.4 |
|---|---|---|---|---|---|---|---|---|
| S=512 H16 D128 | 1.201 ms | **0.190 ms** | 0.043 ms | 1.434 ms | 22.6 | — | — | — |
| S=4096 H16 D128 | 68.79 ms | **4.511 ms** | 0.136 ms | 73.43 ms | 60.9 | 860 | 622 | 379 |
| S=1024 H40 kv8 | 11.07 ms | **0.871 ms** | ~0 ms | 11.94 ms | 49.3 | 356 | 326 | 229 |
| S=1024 H32 kv4 | 8.826 ms | **0.639 ms** | 0.025 ms | 9.490 ms | 53.8 | 418 | 307 | 217 |
| S=1024 H64 kv4 | 17.64 ms | **1.119 ms** | 0.022 ms | 18.78 ms | 61.4 | 431 | 356 | 257 |
| S=1024 H64 kv1 (MQA) | 17.53 ms | **1.009 ms** | 0.084 ms | 18.62 ms | 68.1 | 440 | 322 | 259 |

- **main 相对标量 bf16 golden**：S=512 1.88→**0.190 ms（9.9×）**、S=4096 42.2→**4.51 ms（9.4×）**。
  （幅度小于 fp16 的 11.8–14.9×，是因为 bf16 标量版已做过 padding 优化、基线更快。）
- main-only 到 FA3 的 **5.7–15.5%**（MHA S4096 60.9/860；GQA/MQA 49–68 / 356–440）；
  到 TE 的 8–22%；到 FA2 的 16–26%。
- **端到端被标量 preprocess 拖住**：S=4096 时 preprocess 占 94%（68.8ms vs main 4.5ms），
  GQA S1024 亦占 ~93%。这正是 ROADMAP「当前冲刺」第 2 项 **O8（preprocess 的 LSE/D 改 mma 分块，
  对齐 fp8 的 O1）**要解决的。

**ncu（main, `--set full`）**：

| 指标 | S=512 | S=4096 |
|---|---|---|
| Duration | 320.5 µs（ncu 重放；event 190µs） | 4.55 ms |
| DRAM Throughput | 1.58% | 1.41% |
| L1/TEX Throughput | 10.24% | 33.24% |
| L2 Throughput | 7.69% | 24.84% |
| Compute (SM) | 4.39% | 17.33% |
| Achieved / Theoretical Occupancy | 6.25 / 18.75% | 16.40 / 18.75% |
| Registers / smem / CTA | 168 / 66.56KB / 3 | 168 / 66.56KB / 3 |
| Waves Per SM | 0.32（grid 128<132 SM，grid-bound） | 2.59 |
| No Eligible | 92.40% | 77.39% |
| stall | — | **long_scoreboard 7.35 / ~11.6 cycles = 63%**；wait 1.68、short 0.59、barrier 0.23 |

**bound 结论**：与 fp16 张量核版**逐项一致**（Duration 4.55ms、DRAM 1.41%、L1TEX 33.24%、
Compute 17.33%、168 regs、occ 18.75%、Waves 2.59）。S=4096 bound = **全局访存延迟**
（`long_scoreboard` 63%，每 tile 同步 global→smem、无 cp.async/预取）；S=512 是 **grid-bound**
（grid 128 < 132 SM）。非带宽（DRAM 1.4%）、非算力（Compute 17.3%）。
下一步：O6（`cp.async` 双缓冲 + 降 smem 提 occupancy）打 main 的墙，O8 打端到端的墙。

> 原始输出：`fa_bwd_bf16_mma_main_o5b_{s512,s4096,gqa}.out.txt`、
> `fa_bwd_bf16_mma_onefile_o5b_s512.out.txt`、`fa_bwd_bf16_mma_main_o5b_ncu_main_{s512,s4096}.out.txt`、
> `fa_bwd_bf16_mma_main_o5b_stall_s4096.out.txt`、`fa_bwd_bf16_mma_o5b_fa3_te_baseline.out.txt`。

---

## 6f. preprocess 的 mma 分块 LSE（O8，端到端 4.5–13.2×）

> 代码：两文件 `fa_bwd_bf16_mma_{kernels.cuh,main.cu}` + 单文件 `fa_bwd_bf16_mma_onefile.cu`，
> 与 fp16 的 O8（`docs/01` §11）**逐字同构**（仅 dtype 替换）。

§6e 的 bf16 张量核 main 后，端到端被**标量 preprocess** 拖住（S=4096 占 94%）。O8 照搬 fp8 O1：

**改动（与 fp16 O8 同构）**：
- 旧 `preprocess_kernel`（每 (s,h) 行一个 block、标量扫 K）替换为
  **`lse_mma_kernel<HD>`**（`mma.m16n8k16` `QKᵀ` + 累加器内 online-softmax + 同 row 4 lane
  `shfl_xor` 归约；LBM=64 行 Q × LBN=64 列 K/CTA，4 warp 各 16 行；smem 34816 B）
  + 独立 **`delta_kernel<HD>`**（D=`rowsum(dO∘O)`，O(S·H·D) 归约）。
- 单/两文件 device O8 代码块**逐字一致**（脚本核对 `device O8 block identical: True`）。

**数值（与 O5b 逐位相同，bf16 causal，max_abs）**：

| shape | dq | dk | dv |
|---|---|---|---|
| S=512 H16 D128 | 9.001e-3 | 1.261e-2 | 1.365e-2 |
| S=4096 H16 D128 | 1.510e-2 | 1.340e-2 | 1.631e-2 |
| S=1024 H32 kv4 | 1.201e-2 | 2.125e-2 | 3.156e-2 |
| S=1024 H40 kv8 | 1.233e-2 | 1.930e-2 | 3.150e-2 |
| S=1024 H64 kv4 | 1.351e-2 | 3.091e-2 | 4.420e-2 |
| S=1024 H64 kv1 (MQA) | 1.190e-2 | 4.558e-2 | 7.196e-2 |

与 §6e 表**逐位相同**，且与 ref/FA/TE 同量级。原始输出
`src/bf16/fa_bwd_bf16_mma_main_o8_*.out.txt`、`..._onefile_o8_s512.out.txt`。

**性能（CUDA-event，同 session）**：

| shape | preprocess 旧→新 | 加速 | total 旧→新 | total 加速 | total TFLOPS (占 989) |
|---|---|---|---|---|---|
| S=512 H16 D128 | 1.201→**0.072 ms** | **16.7×** | 1.434→**0.322 ms** | **4.45×** | 6.67 (0.67%) |
| S=4096 H16 D128 | 68.79→**0.993 ms** | **69.3×** | 73.43→**5.573 ms** | **13.2×** | 24.66 (2.49%) |
| S=1024 H40 kv8 | 11.07→**0.213 ms** | **52.0×** | 11.94→**1.110 ms** | **10.8×** | 19.35 (1.96%) |
| S=1024 H32 kv4 | 8.826→**0.186 ms** | **47.4×** | 9.490→**0.867 ms** | **10.9×** | 19.81 (2.00%) |
| S=1024 H64 kv4 | 17.64→**0.300 ms** | **58.8×** | 18.78→**1.453 ms** | **12.9×** | 23.65 (2.39%) |
| S=1024 H64 kv1 | 17.53→**0.306 ms** | **57.3×** | 18.62→**1.415 ms** | **13.2×** | 24.28 (2.46%) |

同 session **纯反向** FA3/TE/FA2（`harness/fa_vs_te_bwd_only.py bf16`）：S4096 MHA FA3
**0.3217ms/854TF**、TE 0.4415/623、FA2 0.7359/374；kv4 S1024 FA3 0.0822/418、TE 0.1122/306。
**ours total 现为 FA3 的 2.9–4.7%（TFLOPS，O5b 时 ~0.4%），时间比 ~10.6–17.3×**。

**ncu（`lse_mma_kernel<128>`，S=4096，`--set full`）**：

```
Duration                 us   967.01      DRAM Throughput        %   1.13
L1/TEX Cache Throughput  %   23.37       L2 Cache Throughput    %   5.55
Compute (SM) Throughput  %   44.70       No Eligible            %  43.12
Theoretical Occupancy    %   37.50       Achieved Occupancy     %  28.41
Waves Per SM                  1.29       Registers Per Thread         80
```

与 fp16 版逐项一致 ⇒ bound = **全局访存延迟 + 低 occupancy/尾波**（80 regs 把理论 occ
卡在 37.5%、grid 仅 1.29 波），非带宽/算力。原始输出
`src/bf16/fa_bwd_bf16_mma_main_o8_ncu_lse_s4096.out.txt`。

---

## 6g. main 的 `cp.async` 双缓冲（O6，main 2.27–2.40×）

O5b/§6e 的 bf16 main 与 fp16 版**逐字同构**，ncu 同是 **`long_scoreboard` ~63%**
（全局访存延迟）的墙；O8/§6f 把 preprocess 打下来后 main 成端到端第一瓶颈。
O6 把 fp16 版的 `cp.async` 双缓冲（见 `01-fp16-bwd-impl.md` §12）**做 dtype 参数化**
搬到 bf16：`cp_async16` 按字节搬（bf16 位宽同构）、`kv_issue_async` 的 `8 个 bf16=16B`
unit 划分、`PIPE` 模板、`Ks/Vs` 双缓冲、每 tile 2 个 barrier。

**改动**：`fa_bwd_bf16_mma_kernels.cuh` + `fa_bwd_bf16_mma_onefile.cu`（单/两文件 device
逐字一致）；host 加 `PIPE` 模板与 `--pipe/--nopipe`、smem 随 `PIPE` 翻 K/V 段。
smem `66.56→83.97KB`、3→2 CTA/SM（182 regs）。

**实测（同 session A/B，CUDA event，main-only）**：

| shape | nopipe (ms / TF) | **O6 pipe (ms / TF)** | 加速 |
|---|---|---|---|
| MHA S=512 | 0.1896 / 11.33 | **0.0837 / 25.66** | **2.27×** |
| MHA S=4096 | 4.4847 / 30.65 | **1.8718 / 73.43** | **2.40×** |
| GQA q32/kv4 S=1024 | 0.6377 / 26.94 | **0.3769 / 45.59** | **1.69×** |

端到端：S=512 0.1827ms（11.75 TF）、S=4096 **3.036ms（45.27 TF）**、GQA kv4 0.6006ms。
单文件与两文件逐指标相同（S512 pipe 0.0842ms、max_abs 逐位一致）。

**数值（与 O5b/O8 逐位相同）**：S=512 9.00/12.6/13.65e-3；S=4096 15.1/13.4/16.3e-3；
GQA kv4 12.0/21.25/31.56e-3——只改搬运、不改数学，结果 bitwise 不变。

**ncu（main, S=4096, PIPE）**：Duration **1.88ms**、DRAM 3.43% / **L1TEX 64.96%** /
**L2 62.19%** / Compute 25.49% / 182 regs / occ 12.5%（2 CTA/SM）/ Waves 3.88。
`long_scoreboard 7.35→1.12`（cp.async 吃掉全局延迟），新墙 = **`wait`（fixed-latency 依赖）
+ `short_scoreboard`（smem→ldmatrix）+ L1/TEX**，与 fp16 版逐项一致。

**对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）：FA3 MHA S4096
**0.3210ms/856 TF**、TE 0.4417/622、FA2 0.7347/374；ours total 3.036ms/45.3 TF ⇒
**FA3 的 5.3%**（时间比 9.5×，O8 时 13.2×）。原始输出
`src/bf16/fa_bwd_bf16_mma_main_o6_{s512,s4096,gqa_kv4}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o6_s512.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o6_ncu_s4096.out.txt`、
`src/bf16/fa_bwd_bf16_o6_fa3_te_baseline.out.txt`。

## 6h. O6b：K/V 双缓冲压回 3 CTA/SM + `ldmatrix.x4.trans` 消转置副本

bf16 复用 fp16 的 O6b（见 `01-fp16-bwd-impl.md` §12b），**逐字 dtype 参数化**：
① GEMM3/GEMM4 的 A 改 `ldmatrix.x4.trans` 从 `Ps/dSs[BM][BN]` 直读（bit3/bit4 互换），
删掉 `PsT/dSsT` 两份转置副本（`mma_block_bf16` 加 `ATRANS`）；② 只双缓冲 K、V 单缓冲
且在 GEMM2 后预取。`--pipe2`；host 按网格 `(S/64)×H×B ≥ 396` 自动选 O6b/O6。
smem `83.97→71.17KB`、Block Limit Shared Mem 2→3、occ 12.5%→**18.75%**（168 regs）。

**实测（同 session A/B，CUDA event，main-only）**：

| shape | nopipe (ms) | **O6 (ms)** | **O6b (ms)** | O6/O6b |
|---|---|---|---|---|
| MHA S=512 (grid=128, 自动选 O6) | 0.1900 | **0.0839** | 0.0873 | 0.96× |
| MHA S=4096 (grid=1024) | 4.4966 | 1.8941 | **1.8643** | 1.02× |
| GQA q32/kv4 S=1024 (grid=512) | 0.6346 | 0.3810 | **0.3584** | 1.06× |

端到端：S=512 0.1856ms（走 O6）、S=4096 **2.9605ms（46.42 TF，走 O6b）**、
GQA kv4 **0.5764ms（29.80 TF，走 O6b）**。单/两文件逐指标相同、数值**逐位一致**。

**数值（与 O5b/O8/O6 逐位相同）**：S=512 9.001/12.61/13.65e-3；S=4096 15.10/13.40/16.31e-3；
GQA kv4 12.01/21.25/31.56e-3。

**ncu（main, S=4096, PIPE=2）**：Duration **1.94ms**、DRAM 3.32% / **L1TEX 71.71%** /
**L2 60.39%** / Compute 31.42% / 168 regs / occ 16.86%（3 CTA/SM）。与 fp16 版逐项一致：
墙 = **L1/TEX + L2 吞吐 + fixed-latency(`wait`)**。

**对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）：FA3 MHA S4096
**0.3205ms/858 TF**、TE 0.4408/624；ours total 2.961ms/46.4 TF ⇒ **FA3 的 5.4%**。
原始输出 `src/bf16/fa_bwd_bf16_mma_main_o6b_{s512,s4096,gqa_kv4}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o6b_{s512,s4096}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o6b_ncu_main_s4096.out.txt`。

---

## 6i. O8b-bf16：LSE 预处理负载均衡 + `cp.async` 双缓冲（端到端 1.27×）

与 fp16 的 O8b（`01-fp16-bwd-impl.md` §13）**逐字 dtype 参数化**：新增
`lse_mma_kernel_bal<HD,PIPE>`（`__half`→bf16、`__float2half`→`__float2bfloat16`），
① **镜像配对**——每 CTA 处理 `m=blockIdx.x` 与 `m'=nblk-1-m`，因果下工作量恒 `nblk+1`
（grid.x 64→32，尾波 50%→~0）；② `PIPE=1` 的 K 用 `cp.async.cg` 16B 双缓冲
（`PIPE=0` 退回同步标量读用于消融）。仅 causal 走新 kernel，非 causal 走 O8 原版。
单/两文件同源，device 代码**逐字一致**（脚本核对 `using bf16` 后主体相同、仅尾部 include guard 差异）。

**消融（同 session，lse-only，CUDA event）**：

| shape | O8 lse (ms) | **bal 单缓冲 (ms)** | **bal+cp.async (ms)** | 总加速 |
|---|---|---|---|---|
| MHA S=512 | 0.0630 | 0.0552 (1.14×) | **0.0456 (1.38×)** | 1.38× |
| MHA S=4096 | 0.9696 | 0.4524 (2.14×) | **0.3503 (2.77×)** | **2.77×** |
| GQA q32/kv4 S=1024 | 0.1621 | 0.1013 (1.60×) | **0.0787 (2.06×)** | 2.06× |

**收益主要来自镜像配对负载均衡，`cp.async` 再叠加 ~1.29×**（与 fp16 一致）。

**端到端（CUDA event）**：preprocess S=4096 0.993→**0.392ms**；total S=4096
2.9605→**2.3427ms（58.67 TF，O6b 时 46.4 TF）**、S=512 0.1856→**0.1603ms**、
GQA kv4 0.5764→**0.4817ms（35.66 TF）**。**数值与 O5b/O8/O6/O6b 逐位相同**
（S=512 9.001/12.61/13.65e-3；S=4096 15.10/13.40/16.31e-3；GQA kv4 12.01/21.25/31.56e-3），
单/两文件逐指标一致。

**ncu（lse_bal, S=4096）**：Duration **356.4µs**、DRAM 3.07% / L1TEX 32.40% / L2 26.77% /
**Compute 60.40%** / 64 regs / 52.22KB smem（**4 CTA/SM**，理论 occ 25%、achieved 22.91%）/
Waves **0.97**；stall `wait 1.57 + short_scoreboard 1.33 + not_selected 0.79 + long_scoreboard 0.34`
（与 fp16 O8b 逐项相同）。**新墙 = Compute（QKᵀ mma + softmax exp）60% + smem 依赖**，
不再是 `long_scoreboard`（2.19→0.34）与尾波（1.29→0.97）。

**对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）：FA3 MHA S4096
**0.3194ms/861 TF**、TE 0.4419/622；GQA kv4 FA3 0.0825ms/416。ours total 2.343ms/58.7 TF
⇒ **FA3 的 6.8%**（O6b 5.4%）、时间比 7.3×；GQA kv4 total 0.482ms ⇒ FA3 的 8.6%、时间比 5.8×。
原始输出 `src/bf16/fa_bwd_bf16_mma_main_o8b_{s512,s4096,gqa_kv4}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o8b_{s512,s4096}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o8b_ncu_lse_bal_s4096.out.txt`、
`..._o8b_stall_lse_bal_s4096.out.txt`、`src/bf16/fa_bwd_bf16_o8b_fa3_te_baseline.out.txt`。

---

## 6j. O6c-bf16：主 kernel tile 几何参数化 + 小网格并行度自适应（main 最多 1.18×）

与 fp16 的 O6c（`01-fp16-bwd-impl.md` §13b）**逐字 dtype 参数化**，只把 `__half`→bf16、
`mma_f16`→`mma_bf16`。改动同 fp16：

1. **tile 几何参数化**：`fa_bwd_bf16_mma_kernel<HD,BM,BN,PIPE>` 的 2×2 warp 几何全部由
   `(BM,BN)` 派生（`GM1/GN1=BM/2,BN/2`；`GMV/GNV=BN/2,HD/2`；`GMQ/GNQ=BM/2,HD/2`；
   `MT*=WARP_M/16,WARP_N/8`），`pval/dqacc/acc` 按 `MT*` 定义，`__launch_bounds__` 改为
   `(THREADS,(BN>32)?2:3)`。`BM=64,BN=32` 展开与 O6b **完全同构**（回归逐位不变）。
2. **host 自动档**：`grid<132 且 S≤1024` ⇒ `(BM=32,BN=32,PIPE=1)`（grid/并行度翻倍）；
   否则 `BM=64`，`S≥4096` 用 `BN=64`，`grid≥396` 用 `PIPE=2`；CLI `--bm/--bn/--pipe/--sched` 覆盖。
3. `sched`（静态 mblk 重排）沿用 fp16 的结论，**保留默认 0、不作为优化**（本轮复测同一结论）。
4. 单文件 `fa_bwd_bf16_mma_onefile.cu` 由两文件 device 段 + host 段**重新拼接**生成
   （device 段与 `fa_bwd_bf16_mma_kernels.cuh` **逐字一致**，脚本已核对）。

### 配置 A/B（同 session，CUDA event，main-only，单位 ms）

| shape (grid) | (64,32,2) | (64,64,2) | (32,32,1) | (32,32,2) | best/原 |
|---|---|---|---|---|---|
| MHA S=512 H16 (128) | 0.0886 | 0.0835 | **0.0800** | 0.0831 | **1.108×** |
| GQA q32/kv4 S=1024 (512) | **0.3584** | 0.3875 | 0.4546 | 0.4583 | 1.00× |
| MHA S=4096 H16 (1024) | 1.8672 | **1.8349** | 2.6322 | 2.6342 | 1.018× |

自动档选中的配置：S=512⇒(32,32,1)、S=1024 kv4⇒(64,32,2)、S=4096⇒(64,64,2)。
**端到端**（`preprocess+main+convert`）：S=512 0.1603→**0.1501ms（1.06×，14.31 TF）**、
S=4096 2.3427→**2.3692ms（58.01 TF，持平，session 噪声）**、GQA kv4 0.4817→**0.4838ms（35.51 TF）**。
（`(32,32,1)` 在 S=512 的收益 run-to-run 在 **1.11–1.18×** 之间波动，机制一致。）
`[O6c A/B]` 的 `sched=0/1/2` 分别是 0.0845/0.0849/0.0846（S=512）与 1.8805/1.8547/1.8403（S=4096）
⇒ **静态重排 0~2% 且方向不稳**，再次证伪。

### 数值（与 O5b/O8/O6/O6b/O8b **逐位相同**）

`BM=64,BN=32` 路径展开后与 O6b 同构；`BM=32`/`BN=64` 只改 tile 划分、不改数学口径。
S=512 dq/dk/dv max_abs = 9.001/12.61/13.65e-3；S=4096 = 15.10/13.40/16.31e-3；
GQA kv4 = 12.01/21.25/31.56e-3（与 §6e–§6i 记录一致）。单/两文件逐指标一致。

### ncu（main）

- **S=512 `(128,32,32,1)`**：Duration **83.9µs**、DRAM 5.74% / L1TEX **43.94%** / L2 44.89% /
  Compute 11.91% / 145 regs / 59.90KB smem / 理论 occ **18.75%**、achieved **10.99%** / Waves **0.65**；
  stall `long 3.23 + wait 1.89 + short 1.08 + mio 0.68 + lg_throttle 0.30 + barrier 0.23`。
  与 fp16 O6c 同 config 逐项一致（fp16：83.87µs / L1 43.54% / occ 10.99%）。机制=**并行度**：
  grid 128→256、每 SM 1→2 个 CTA，把 SM 从「单 CTA 等延迟」救出。
- **S=4096 `(128,64,64,2)`**：Duration **1.86ms**、DRAM 3.48% / **L1TEX 57.05%** / **L2 63.02%** /
  Compute 26.77% / 242 regs / 105.47KB smem / 理论 occ **12.50%**（2 CTA/SM）/ Waves **3.88** /
  No Eligible 67.14%；stall `wait 1.94 + long 1.45 + short 0.46 + not_selected 0.18`。
  对比 O6b `(128,64,32,2)`：Duration 1.94ms、L1TEX **71.71%**、L2 60.39%、Compute 31.42%、
  168 regs / 71.17KB / occ **16.86%**（3 CTA/SM）/ Waves 2.59 ⇒ **`BN=64` 把 L1/TEX 71.7→57.1%
  （Q/dO 复用翻倍），但 smem 105KB 使占用从 3→2 CTA/SM，净收益仅 ~1.5%**。要同时拿低 L1 与高
  occupancy 须先把 smem 压到 ≤77.7KB（BN=64 需砍 ~28KB），留待 O9。**新墙 = L1/L2 吞吐 + `wait`**。

### 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`，含 `fa_bwd_o6c_fa3_te_s512.out.txt`）

| shape | FA3 | TE2.14 | ours total | ours/FA3 (TF) | 时间比 |
|---|---|---|---|---|---|
| MHA S=4096 | **0.3195ms / 860 TF** | 0.4360 / 630 | 2.3691ms / 58.0 | **6.7%** | 7.41× |
| MHA S=512 | **0.0265ms / 162 TF** | 0.0324 / 132 | 0.1510ms / 14.2 | **8.8%** | 5.70× |
| GQA q32/kv4 S=1024 | **0.0822ms / 418 TF** | 0.1116 / 308 | 0.4839ms / 35.5 | **8.5%** | 5.89× |

（FA2 对比：S=4096 0.7278/378、GQA 0.1584/217。）

原始输出 `src/bf16/fa_bwd_bf16_mma_main_o6c_{s512_h16,s4096_h16,s1024_h32_kv4}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o6c_{s512_h16,s4096_h16}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o6c_ncu_{s512_main,s4096_main}.out.txt`、
`..._o6c_stall_{s512,s4096}.out.txt`、`src/bf16/fa_bwd_o6c_fa3_te_baseline.out.txt`、
`src/bf16/fa_bwd_o6c_fa3_te_s512.out.txt`。

---

## 7. 复现命令

```bash
cd code/flash-attention/fa-bwd

# 1) dump bf16 输入/参考（S=512、S=4096 causal）
docker exec kernel_lab bash -lc "cd $PWD/harness && python fa_bwd_bench.py dump --dtype bf16 \
    --shape 1 512 16 128 causal --shape 1 4096 16 128 causal"

# 2) 编译运行 + 数值对拍
scripts/run.sh src/bf16/fa_bwd_bf16_onefile.cu \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_bf16 --iters=50
scripts/run.sh src/bf16/fa_bwd_bf16_onefile.cu \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_bf16 --iters=50

# 3) ncu（main kernel）
scripts/ncu.sh src/bf16/fa_bwd_bf16_onefile.cu --set full \
    --kernel-name regex:fa_bwd_bf16_kernel -- \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_bf16 --iters=1

# 4) FA/TE bf16 基线
docker exec kernel_lab bash -lc "cd $PWD/harness && python fa_bwd_bench.py bench --dtype bf16 \
    --shape 1 512 16 128 causal --shape 1 4096 16 128 causal"
```

---

## 6k. O7c-bf16：LSE/D 预装寄存器 + dK/dV float4 试错（main 1.14–1.19×）

把 fp16 的 O7c（`01` §14）**逐字 dtype 参数化**到 bf16：在 `fa_bwd_bf16_mma_kernel` 加
模板开关 `R4`（dK/dV 归约宽度，默认 false=float2）与 `PREL`（LSE/D 预装寄存器，默认 true），
device 代码与 fp16 同构。单文件、两文件同步，脚本核对 device 段 **逐字一致（identical: True）**。

**负结果（R4，float4）**：SASS 生成 `REDG.E.ADD.F32x4`，但四个几何一致变慢
（(64,32,2) S=4096 −4.6%、(64,64,2) −3.3%、(32,32,1) −2.4%、GQA kv4 −3.0%）⇒ 与 fp16
结论相同：dK/dV 归约**不是事务数 bound**，`shfl` 打包的代价超过省下的事务。

**正结果（PREL）**：LSE/D 只依赖 CTA 自己的 Q 行、与 K tile 无关，原来每个 tile 都在
GEMM1/2 epilogue 按 `qi` 去 global 读（ncu：每 thread 仅 4.4/32B）。循环前一次性装进
`lse_r[MTM1][2]/del_r[MTM1][2]` 后，数值**逐位不变**、性能：

| 几何 | base（f2） | f2+PREL | 提升 |
|---|---|---|---|
| (64,64,2) S=4096 | 1.8297 ms / 75.1 TF | **1.5566 / 88.3** | **+17.5%** |
| (64,32,2) S=4096 | 1.8057 / 76.1 | **1.5905 / 86.4** | +13.5% |
| (64,32,2) S=512 | 0.0851 / 25.3 | **0.0746 / 28.8** | +14.1% |
| (64,64,2) S=512 | 0.0808 / 26.6 | **0.0697 / 30.8** | +15.9% |
| (32,32,1) S=512 | 0.0797 / 26.9 | 0.0738 / 29.1 | +8.0% |
| GQA kv4 S=1024 (64,32,2) | 0.3580 / 48.0 | **0.3046 / 56.4** | +17.5% |
| GQA kv4 S=1024 (64,64,2) | 0.3821 / 45.0 | **0.3214 / 53.5** | +18.9% |

端到端（自动档 S=4096）：`main 42.2→…` 见 O5b 后已到 mma 版；本轮相对 O6c `main 1.87→1.59ms`、
total `2.3692→2.0986 ms`（1.13×）；S=512 total 0.1495ms。数值与 O5b/O8/O6/O6b/O8b/O6c
**逐位相同**（S=512 9.001/12.61/13.65e-3；S=4096 15.10/13.40/16.31e-3；GQA kv4
12.01/21.25/31.56e-3）。

**ncu（main，S=4096，(64,64,2)）**：Duration 1.87→**1.64ms**、L1/TEX 55.7%、
**L2 70.5%（新墙）**、Compute 23.4%、regs 250 / smem 105.47KB（2 CTA/SM，occ 11.8%）、
stall `wait 2.00 / long 1.39 / short 0.88 / mio 0.50`（`long` 从 ~1.8 降）——与 fp16 逐项一致。
**墙已从 L1/TEX 移到 L2 + occupancy**，进一步减 red 不划算 ⇒ 下一项 **O9（wgmma+TMA）**。

**对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）：S=4096 MHA FA3
0.3193ms/861TF、TE 0.4426/621；ours total 2.0986ms（时间 6.57×）。GQA kv4 S=1024 FA3
0.0827/415；ours total 0.4345（5.25×）。

原始输出 `src/bf16/fa_bwd_bf16_mma_main_o7c_{s512,s4096,gqa_kv4}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o7c_{s512,s4096}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o7c_ncu_s4096.out.txt`、`..._o7c_stall_s4096.out.txt`、
`src/bf16/fa_bwd_bf16_o7c_fa3_te_baseline.out.txt`。

---

## 6l. MLA（head_dim=512）张量核反向（bf16，main 3.7–4.1×）

把 fp16 的 MLA 张量核（`01` §14b）**逐字 dtype 参数化**到 bf16（`__half`→bf16、
`mma_f16`→`mma_bf16`、`__float2half`→`__float2bfloat16`），device 代码与 fp16 同构：

- `fa_bwd_bf16_mma_kernel<HD,BM,BN,PIPE,...>` 支持 HD=128/512：GEMM1/2 的 k-loop 变长；
  GEMM3/4/5 加 **N-tile 循环**（每遍 `NTW=128` 列）；HD>128 的 dQ 直接**全局累加**
  （每 `(qi,列)` 由唯一线程拥有 ⇒ 非原子 RMW 无竞争）。
- host：`D∈{128,512}` 分派；MLA 自动档 `BM=32,BN=32,PIPE=1`。单/两文件 device 段逐字一致。

**数值对拍（ours vs fp32 ref，bf16 causal，D=Dv=512）**：

| case | dq (max_abs) | dk | dv |
|---|---|---|---|
| S=256 H=2 | 1.230e-2 | 9.875e-3 | 1.50e-2 |
| S=512 H=4 | 8.753e-3 | 1.082e-2 | 1.740e-2 |
| S=1024 H=2 | 5.838e-3 | 9.519e-3 | ~1.5e-2 |

均 bf16 噪声量级（~1e-2），与 P5-3 标量 MLA（`01b` §6d：8.240/7.905/15.04e-3 等）同量级。
**FA3/TE 反向不支持 head_dim=512**，只有 fp32 ref 可对。**MHA D=128 回归逐位不变**
（S=512 9.001/12.61/13.65e-3），单/两文件一致。

**性能（同 session CUDA event，preprocess/main/total，ms）**：

| case | 标量 main | **张量核 main** | main 加速 | 标量 total | **张量核 total** | total 加速 |
|---|---|---|---|---|---|---|
| S=256 H=2 | 0.7531 | **0.2037** | **3.70×** | 0.9780 | **0.3003** | 3.26× |
| S=512 H=4 | 1.4880 | **0.3847** | **3.87×** | 2.8835 | **0.5355** | 5.38× |
| S=1024 H=2 | 2.9590 | **0.7248** | **4.08×** | 5.5703 | **0.9393** | 5.93× |

配置 A/B（S=512 H4）：`(32,32,PIPE=1)` **0.3786ms** < `(32,32,PIPE=0)` 0.6263 < `(64,32,PIPE=0)`
0.9208 ⇒ 1.65×。

**ncu（main，S=1024 H2 D=512，`(32,32,PIPE=1)`）**：Duration **769.9µs**、DRAM 0.84% /
L1/TEX 40.06% / L2 13.86% / Compute 2.22%、168 regs、207.36KB smem、1 CTA/SM、
occ 6.25%、**Waves 0.48**、No Eligible 91.65%——与 fp16 MLA 逐项相同。**bound = 全局访存延迟
+ 低并行度**（grid=64 < 132 SM、1 CTA/SM），非带宽/算力。进一步提速需降 smem 冲 2 CTA/SM 或
split-KV，留 backlog。

原始输出 `src/bf16/fa_bwd_bf16_mma_main_p5mla_{s256h2,s512h4,s1024h2}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_p5mla_s512h4.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_p5mla_reg_d128_s512.out.txt`、
`src/bf16/fa_bwd_bf16_main_p5mla_{s256h2,s512h4,s1024h2}.out.txt`（标量基线）、
`src/bf16/fa_bwd_bf16_mma_main_p5mla_ncu_sol_s1024h2.out.txt`。

---

## 6m. O10：Q/dO 载入向量化 + `cp.async` 重叠 + 打包写回（bf16，main/total 1.03–1.29×）

把 fp16 的 O10（`docs/01` §14c）**逐字 dtype 参数化**到 bf16：新增 `qdo_issue_async<HD,BM>`
（`__half`→bf16）、主 kernel `PIPE>=1` prologue 的 Q/dO 改 16B `cp.async.cg`（与 K/V 的
`wait_group 0` 一并等待）、`lse_mma_kernel_bal` 的 Q 改 `issue_q`、dQ 写回（含 HD>128 直接累加
路径）打包成 `float2`。单/两文件 device 代码**逐字一致**（脚本核对 `identical: True`）。

**数值与 O5b/O8/O6/O6b/O8b/O6c/O7c 逐位相同**（S=512 9.001/12.61/13.65e-3、S=4096
15.10/13.40/16.31e-3 等），单/两文件逐指标一致。

**性能（同 session A/B，CUDA event，ms）**：

| case | total base → O10 | total 加速 | main base → O10 | main 加速 |
|---|---|---|---|---|
| S=512 H16 d128 | 0.1460 → **0.1268** | **1.15×** | 0.0761 → **0.0720** | 1.06× |
| S=4096 H16 d128 | 2.0824 → **1.9950** | 1.04× | 1.5860 → **1.5237** | 1.04× |
| S=1024 H32 kv4 | 0.4341 → **0.3761** | **1.15×** | 0.3075 → **0.2621** | 1.17× |
| S=256 H2 d512 | 0.2985 → **0.2285** | **1.31×** | 0.2022 → **0.1672** | 1.21× |
| S=512 H4 d512 | 0.5319 → **0.4343** | **1.22×** | 0.3766 → **0.3257** | 1.16× |
| S=1024 H2 d512 | 0.9320 → **0.8205** | **1.14×** | 0.7160 → **0.6584** | 1.09× |
| S=1024 H40 kv8 | 0.5201 → **0.4589** | 1.13× | 0.3712 → 0.3215 | 1.15× |
| S=1024 H64 kv1 | 0.7151 → **0.6359** | 1.13× | 0.5106 → 0.4606 | 1.11× |

收益同 fp16：LSE 的 Q 向量化（preprocess 0.0533→0.0406）+ 主 kernel Q/dO 的 `cp.async` 重叠
（GQA/MQA/MLA main 1.1–1.2×）；大 S 收益最小（prologue 占比小）。

**ncu（main，S=4096，`(64,64,2)`）**：Duration **1.55 ms**、Executed Instructions
**342,052,864**（与 fp16 O10 **完全相同**）、regs 250 / smem 105.47KB / 2 CTA/SM、Waves 3.88；
`long_scoreboard` 0.89、`mio_throttle` 0.70、`wait` 2.04。**墙仍 = `wait` + L2 + 低 occupancy**。

原始输出 `src/bf16/fa_bwd_bf16_mma_main_o10_{allshapes,ncu_s4096}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o10_*.out.txt`、`src/bf16/fa_bwd_bf16_o10_fa3_te_baseline.out.txt`。

---

## 6n. O11：快速 exp/log（bf16，与 fp16 同款）

把 softmax 热点的 libdevice 精确 `expf`/`logf` 换成硬件内建 `__expf`/`__logf`
（MUFU.EX2/LG2，相对误差 ~2^-21），单/两文件 device 代码与 fp16 逐字同源（`FAST_EXP` 宏 A/B）。
bf16 容差 ~1e-2，无影响：S=512 dq/dk/dv vs ref 仍 9.001/12.61/13.65e-3、S=4096
15.10/13.40/16.31e-3（与 O5b/O8/O6/O6b/O8b/O6c/O7c/O10 逐位相同）。S=4096 `total` ~1.97ms。
原始输出 `src/bf16/fa_bwd_bf16_mma_main_o11_{s512,s4096}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o11_s512.out.txt`。fp8 的对应改动见 `docs/03` §19。

---

## 6o. O9a：LSE 预处理上 wgmma（bf16，单/两文件）

与 fp16 的 `docs/01` §14e **逐字同构**：bf16 与 fp16 同为 2 字节、SW128 布局/描述符/`m64n64k16`
累加器映射逐字节相同，仅指令 dtype 从 `f16.f16` 换成 `bf16.bf16`。冒烟复用 fp16 的
`fa_bwd_fp16_wgmma_smoke.cu`（PASS，max_abs=0）。

* **集成**：`lse_mma_kernel_bal_wgmma<HD,PIPE>`（仅 HD=128、causal），Q/K 以 SW128 存 +
  `cp.async` 发，8 条 `wgmma.m64n64k16` 完成 64×64×128 的 QKᵀ；镜像配对/online-softmax/
  行归约与 O8b 相同。默认走 mma 版，`--lsewgm=1`（需 sm_90a 构建）切 wgmma。
* **性能**（同 session，LSE-only，CUDA event）：bf16 S=4096 O8 0.8749 → O8b bal+cpasync
  0.3006 → **wgmma 0.2856ms（vs O8b 1.054×，vs O8 3.073×）**；端到端 total 1.9595ms（70.1 TF）。
  对照 fp16 同 shape wgmma 0.2871ms，一致。
* **ncu**（S=4096，同 session 对照）：Duration 303.62→**286.18µs**、**L1/TEX 37.62→18.05%**
  （`ldmatrix` 被 SS 直读取代）、L2 31.62→20.25%、Executed Ipc 2.36→2.52；regs 62、
  smem 52.22→50.18KB、occ ~23%、Waves 0.97。**墙 = softmax epilogue + 发射**（Compute ~60%），
  wgmma 只把访存那一半打掉，故 Duration 只快 ~6%——与 fp16 结论一致。
* **数值与 O8b 逐位相同**：S=4096 1.510/1.340/1.631e-2、S=512 9.001/12.61/13.65e-3。
* 原始输出 `src/bf16/fa_bwd_bf16_mma_{main,onefile}_o9_s4096.out.txt`、
  `src/bf16/fa_bwd_bf16_o9_fa3_te_baseline.out.txt`。

---

## 6p. O13-bf16：主 kernel auto tile 重新标定 + memset/convert 冗余（与 fp16 同款）

O6c 的 auto tile 启发式在 O7c-PREL 之后过时：S=512 MHA 上 `(64,64,2)` 的主 kernel 比自动档
`(32,32,1)` **快 1.27×**（bf16 O5c A/B 0.0703→0.0576ms）。本轮与 fp16 **逐字同款**改动：
1. 取消 `BM=32` 分支（`--bm=32` 仍可覆盖）；`BN=64` 判据改为 `S≥4096 || grid≤256 || grid>600`
   （`256<grid≤600` 保留 BN=32 以避开 2 CTA/SM 的 2-波坏量化点）。
2. HD=128 时 dQ 由主 kernel 覆盖写 ⇒ 省掉 `d_dq_acc` 的 memset（仅 HD=512 保留）。
3. `convert_kernel` 改 `float4` 读 + `bf162` 写。

**数值与 O5b~O10 逐位相同**（S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3）；
单/两文件一致。**性能**：S512 main 0.0727→**0.0572（1.27×）**、total 0.1267→**0.1112（1.14×）**；
S1024 kv1 main 0.4611→**0.4362（1.06×）**、kv4(h64) 0.4921→**0.4525（1.09×）**；S4096 不变。
convert 桶 S512 0.0157→0.0132、S4096 0.1037→0.0939ms。原始输出
`src/bf16/fa_bwd_bf16_mma_{main,onefile}_o13_*.out.txt`。（ncu 与 fp16 逐项同构。）

---

## 6q. O9b-bf16：主 kernel GEMM1/2 上 wgmma（bf16，单/两文件）

与 fp16 的 `docs/01` §14g **逐字同构**：bf16 与 fp16 同为 2 字节、SW128 布局/描述符/
`wgmma.m64n64k16` 累加器映射逐字节相同，仅指令 dtype 从 `f16.f16` 换成 `bf16.bf16`。
这是 fp16 O9b（第四十一轮）在 bf16 侧的补齐——`docs/01b` 此前只做到了 **O9a**（LSE 的
单个 QKᵀ 上 wgmma，§6o），主 kernel 的 5 个 GEMM 仍全部是 `mma.m16n8k16`。

**数据通路**（新增 `fa_bwd_bf16_wgmma_kernel<HD>` + `wgmma_mn64_issue`/`mma_block_swb`/
`kv_issue_async_sw`/`qdo_issue_async_sw`，仅 HD=128 / BM=BN=64）：

* Q/dO/K/V 存成 **SW128 K-major** tile（`sw128_off` 写、`cp.async.cg` 16B 发、wgmma 描述符直读）；
* GEMM1 `S=Q·Kᵀ`、GEMM2 `dP=dO·Vᵀ` 用 `wgmma.m64n64k16`（整 CTA 一个 64×64 tile），两 group
  一起 issue、统一 `wait0` 让两条异步 mma 重叠；
* GEMM3/4/5 仍 mma，但其转置 B（dO/Q/K）从**同一块 SW128 tile**用 `ldmatrix.x2.trans` 读
  （SW128 只在 16B 粒度置换，`ldmatrix` 每 lane 只要一个 16B 地址，冒烟已验证逐位一致）；
* P/dS 仍按 `[BM][BN]` 行主序（+8 行距）、GEMM3/4 的 A 用 `ldmatrix.x4.trans`（O6b）。
* smem（HD=128,BM=BN=64）：Q/dO 各 16KB + K 双缓冲 32KB + V 单缓冲 16KB + Ps/dSs 18KB
  ≈ **101.4KB → 2 CTA/SM**（与 O13 的 mma `(64,64,2)` 同 occupancy）。
* 构建：`ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" scripts/run.sh …`。
  host 加 `--wgmma=0/1`；自动档仅当 `D==128 && sel==(64,64)` 才走 wgmma，GQA（BN=32）自动回退 mma。

**性能**（CUDA event，同 session `[O9b A/B]`，单文件与两文件一致）：

| shape | main mma(64,64,2) | main wgmma(GEMM1/2) | 加速 |
|---|---|---|---|
| MHA S=512 | 0.0571–0.0573 ms | 0.0524–0.0526 ms | **1.088–1.089×** |
| MHA S=4096 | 1.4861 ms | 1.4515–1.4552 ms | **1.021–1.024×** |
| GQA kv4 S=1024（自动 BN=32，回退） | — | — | 走原 mma 路径 |

端到端 bf16 S=4096 **total 1.880 ms（73.1 TF，`4BS²HD` 口径）**、S=512 **0.1085 ms（19.8 TF）**、
GQA kv4 S=1024 **0.376 ms（45.7 TF）**。

**数值与 O5b/O8/O6/O6b/O8b/O6c/O7c/O10/O13 逐位相同**：MHA S=512 9.001/12.61/13.65e-3、
S=4096 15.10/13.40/16.31e-3、GQA kv4 12.01/21.25/31.56e-3；单/两文件逐指标一致。

**ncu（main，S=4096，`--set full`）**：Duration **1.46ms**、DRAM 4.44% / **L2 72.99%** /
L1/TEX 54.26% / Compute 26.51%、242 regs / 101.38KB smem → **2 CTA/SM**（occ 11.86%）、
**Waves 3.88**；stall **`wait` 1.50 + `long_scoreboard` 1.24 + short 0.56** + mio 0.26，
与 fp16 O9b 逐项一致。**bound = L2（残余 dK/dV 跨 CTA 原子）+ `wait`（mma 依赖）+ 2 CTA/SM**；
wgmma 打掉了 GEMM1/2 的 `ldmatrix`/发射，但 GEMM3/4/5 仍是 mma、dK/dV 仍是跨 CTA 原子，
所以 L2 与 occupancy 没变 ⇒ 收益真实但有限，与 fp16 侧结论相同。

**对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）：MHA S=4096 FA3
**0.3191ms/861 TF**、TE 0.4418/622、FA2 0.7307/376 ⇒ ours total 时间 **5.89×**（FA3）、
约 FA3 的 17% 吞吐；GQA kv4 S=1024 FA3 0.0823ms/417 TF。**下一步 O9b-2**（fp16 侧优先）：
GEMM3/4/5 也上 wgmma（P/dS 进 smem/SW128、dKV 转置 B 按 FA3 `dKV_swapAB`）＋ TMA 化 K/V
＋ 双缓冲 P/dS 跨-tile 流水，才能同时降 L2 与提 occupancy。

原始输出：`src/bf16/fa_bwd_bf16_mma_main_o9b_{s512,s4096,gqa_kv4}.out.txt`、
`..._mma_onefile_o9b_{s512,s4096}.out.txt`、`..._o9b_ncu_s4096.out.txt`、
`..._o9b_stall_s4096.out.txt`、`src/bf16/fa_bwd_bf16_o9b_fa3_te_baseline.out.txt`。

---

## 6r. O9b-2-bf16：主 kernel GEMM3/4/5 也上 wgmma（MN-major 转置读，bf16；数值逐位正确、性能中性）

与 fp16 的 `docs/01` §14h **逐字同构**（第四十四轮）。fp16 侧在第四十三轮已把主 kernel 的
**GEMM3/4/5**（`dV=Pᵀ·dO`、`dK=dSᵀ·Q`、`dQ=dS·K`）也用 `wgmma.m64n64k16` 实现了，bf16 侧
此前只到 **O9b**（§6q，仅 GEMM1/2 用 wgmma，GEMM3/4/5 仍 mma）。本步把 bf16 补齐。

### 6r.1 关键手段：`Major::MN` 描述符 = 对 K-major SW128 tile 的「转置读」

同一份 **K-major SW128** 存储的 tile，用 `Major::MN` 描述符 + `tnsp=1` 读，等于读它的转置
（FA3 `dKV_swapAB` 同思路）。bf16 与 fp16 同为 2 字节，SW128 布局/描述符/u128 步长**逐字节
同构**，故 fp16 的 `fa_bwd_fp16_wgmma_bwd_smoke.cu` 逐位验证（4 种转置组合 `max_abs=0`）
直接适用；bf16 只差 wgmma 指令 dtype（`f16.f16`→`bf16.bf16`）：

* `LBO=64`（相邻 64 列组的 u128 步长）；`SBO=(W/64)*64`（相邻 8 行组的 u128 步长）；
* trans 操作数第 `s` 个 k16 slab（K=行，前进 16 行=2 行组）地址 = `base + s*2*SBO*16` 字节；
* wgmma 尾部立即数 `p, 1, 1, tnspA, tnspB`，K-major 时 `tnsp=0`、MN-major 时 `tnsp=1`。

### 6r.2 实现（单/两文件 device 代码逐字一致，`scripts/sync_onefile_device.py` 核对 `identical: True`）

`fa_bwd_bf16_wgmma_kernel<HD>`（`#ifdef FA_WGMMA`）改动：

* 新增 `wgmma_m64n64k16_bf16_t<TA,TB>`（带 `tnspA/tnspB` 立即数）、`trans_k16_addr`/
  `desc_k16_mn`/`desc_k16_k`（dtype 无关）、`pds_store_sw128`（bf16 版，`__bfloat16_as_ushort`
  拼 `uint32` 写 16B 分块）。
* **P/dS 改 SW128 K-major**：GEMM1/2 的 epilogue 用 `pds_store_sw128` 写 `Ps/dSs`（不再按
  `[BM][LDS]` 行主序）。
* **5 个 GEMM 全 wgmma**：GEMM1/2 沿用 `wgmma_mn64_issue`；GEMM3/4/5 新增按 N 半（`nh=0/1`）
  分两遍，每遍三条一起发、统一 `wait0`（A/B 均为 MN-major 描述符，唯一例外是 GEMM5 的
  A=dS 用 K-major）。
* **dQ 寄存器累加重映射** `dqacc[nh][j][qq]`，循环后一次写出（每 Q 块唯一 CTA）。
* smem：`1024(对齐) + 2*16KB(Q/dO) + 3*16KB(K 双缓冲 + V 单缓冲) + 2*8KB(Ps/dSs SW128)`
  = **99.33KB → 2 CTA/SM**（与 fp16 O9b-2 逐字节相同）。
* host `launch_bwd_wgmma` 的 smem 公式改为 `1024 + TILE*2 + KTILE*3 + 2*BM*BN*sizeof(bf16)`。

### 6r.3 数值（与 O5b~O13 的 mma / O9b **逐位相同**）

| shape | dq / dk / dv max_abs（vs fp32 ref） |
|---|---|
| MHA S=512 | 9.001 / 12.61 / 13.65e-3 |
| MHA S=4096 | 15.10 / 13.40 / 16.31e-3 |
| GQA kv4 S=1024 | 12.01 / 21.25 / 31.56e-3 |

单文件与两文件逐指标一致；只换了数据通路（MN-major 描述符 + 全 wgmma），数学口径未动。
（对拍时用 `--wgmma=1` 让 `run_all` 走 wgmma 路径，`[compare]` 反映的就是 wgmma 输出。）

### 6r.4 性能（同 session `[O9b A/B]`，CUDA event，main-only，ms）—— **中性偏负**

| shape | main mma(64,64,2) | main 全 wgmma(O9b-2) | 加速 |
|---|---|---|---|
| MHA S=512 | 0.0568 ms (37.81 TF) | 0.0592 ms (36.29 TF) | 0.960× |
| MHA S=4096 | 1.4896 ms (92.26 TF) | 1.5221 ms (90.29 TF) | 0.979× |
| GQA kv4 S=1024（自动 BN=32，回退） | 0.2697 ms | 0.2698 ms | 1.000× |

端到端 bf16 total：S=512 **0.1148 ms**、S=4096 **1.9239 ms（71.4 TF，`4BS²HD` 口径）**、
GQA kv4 **0.3734 ms（46.0 TF）**。**全 wgmma 没有跑赢 mma 最优档**（与 fp16 O9b-2 结论相同）。

### 6r.5 ncu（main，`--wgmma=1`，`--launch-count 1`）

| 指标 | S=4096 | S=512 |
|---|---|---|
| Duration | **1.48 ms** | **59.3 µs** |
| DRAM / L1TEX / L2 | 4.35% / 40.05% / **71.60%** | 8.53% / 18.27% / 36.40% |
| Tensor / Compute(SM) | 12.03% / 23.37% | 5.21% / 10.71% |
| regs / smem / CTA per SM | 230 / 99.33KB / **2** | 230 / 99.33KB / **2** |
| achieved occupancy | 11.86% | 6.25%（grid-bound） |
| bank conflict（ld/st） | 0 / 638 | 0 / — |
| stall（wait/long/short/barrier） | 1.43 / 1.99 / **0.29** / 0.84 | 1.41 / 0.87 / 0.28 / 0.80 |

与 fp16 O9b-2 **逐项一致**（fp16：Duration 1.49ms、L1TEX 40.25%、L2 71.38%、Tensor 12.09%、
short 0.29、230 regs/99.33KB/2 CTA）。**结论**：全 wgmma 把 GEMM3/4/5 的 `ldmatrix` 也打掉
（`short_scoreboard` 0.56→**0.29**），但 **L2 仍 ~70%、occupancy 仍 2 CTA/SM**——墙是
**dK/dV 跨 CTA 原子归约 + 99.33KB smem**，不在 GEMM 指令。因此本步收益中性，但其价值是
**让 fp16/bf16 两套 wgmma 通路都具备「MN-major 转置读」**，为后续 TMA/流水/去原子打地基。

### 6r.6 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）

MHA S=4096 FA3 **0.3206ms/858 TF**、TE 0.4368/629、FA2 0.7307/376 ⇒ ours total 时间
**6.00×**（FA3）、4.40×（TE）；GQA kv4 S=1024 FA3 **0.0823ms/417 TF**、TE 0.1116/308
⇒ ours total **4.54×**（FA3）。与 fp16 O9b-2 的「~6.0×」一致。

原始输出：`src/bf16/fa_bwd_bf16_mma_main_o9b2_{s512,s4096,gqa_kv4}.out.txt`、
`..._mma_onefile_o9b2_{s512,s4096}.out.txt`、`..._o9b2_ncu_{s512,s4096}.out.txt`、
`src/bf16/fa_bwd_bf16_o9b2_fa3_te_baseline.out.txt`。

---

## 6s. O17-bf16：跨 warpgroup 归约（BM=128、2 warpgroups），dK/dV 的 red 字节砍半

> 与 fp16 的 O17（`docs/01` §14j）**逐字 dtype 参数化**：同为 2 字节，SW128 布局 / 描述符 /
> `wgmma.m64n64k16` 累加器映射逐字节同构，只差 `f16.f16`→`bf16.bf16`。

### 6s.1 动机

O15a / fp16 O17 已用 ncu 把 S=4096 主 kernel 的墙钉死：L2 扇区里 **`red`（dK/dV 跨 CTA
`atomicAdd`）占 73.1%**、DRAM 仅 ~4% ⇒ main 是 **L2 原子字节数 bound**（O16 错开 `wait_group`、
O7c float4 归约均证明「动等待 / 动事务数」无效）。**唯一杠杆**是让每个 KV 元素被更少的 CTA
贡献：`1colblock` 下每个 CTA 覆盖 BM 行 Q，dK/dV 沿 M 维归约 ⇒ 一个 KV 元素被 `nblk=S/BM`
个 CTA 各贡献一次。把 **BM 从 64 翻到 128**，贡献它的 CTA 数减半 ⇒ red 字节砍半。

### 6s.2 关键手段：dK/dV 让**一个** warpgroup 对全 BM 归约

**只让 wg0 做 GEMM3/4，并把两个 m64 半（Q 行 0–63 与 64–127，即 s=0..7 共 128 行）连续喂给
同一个 `wgmma.m64n64k16` 累加器**——两次 m64 的输出形状都是 `[BN][HD]`，累加器天然相加，
得到的**就是整个 BM 的 dV/dK**，于是每个 KV 元素只 `red` 一次。wg1 在并行做自己的 GEMM5
（dQ 只依赖本 wg 的 Q 行，互不干扰）。这套「对 [128][*] K-major SW128 tile 用 MN-major
描述符按 `s=0..7` 读转置」的逐位正确性，已由 fp16 冒烟 `fa_bwd_fp16_wgmma2_smoke.cu`
（max_abs=0）验证；bf16 与 fp16 同位宽、同布局，直接复用。

### 6s.3 实现（单/两文件 device 代码逐字一致，`scripts/sync_onefile_device.py` 核对 `identical: True`）

新增 `fa_bwd_bf16_wgmma2_kernel<HD>`（`#ifdef FA_WGMMA`，仅 HD=128，`__launch_bounds__(256,1)`）：
- **2 wg**：`wg = tid>>7`、`wid=(tid>>5)&3`；本 wg 的 Q/dO/P/dS tile 基址按「64 行 = 8 rowgroup」
  偏移；GEMM1/2 各 wg 用 `wgmma_mn64_issue`（m64n64）算自己的 S/dP，`pds_store_sw128` 写进
  **共享**的 `Ps/dSs[128][64]`；
- 中段 `__syncthreads` 后 **wg0** 做 GEMM3/4（`for nh: for s in 0..7`，`desc_k16_mn` 全 BM 归约
  + `red_add2`）；**wg1** 只做 GEMM5（A=`desc_k16_k(dSw,s,BN)`、B=`desc_k16_mn(Kt+nh*1024,s,HD)`，
  寄存器累加后一次 `float2` 写回）；
- K/V/Q/dO 用 **256 线程** `cp.async`：`kv_issue_async_sw`/`qdo_issue_async_sw` 加模板参数
  `NT=256`（默认 `THREADS=128` 时行为不变）；K 双缓冲、V 单缓冲后段预取（同 O6b/O9b）。
- smem：Q 32KB + dO 32KB + K 2×16KB + V 16KB + P 16KB + dS 16KB ≈ **149.5KB → 1 CTA/SM**
  （256 线程 = 8 warps/SM，与 O9b 的 2 CTA/SM × 4 warps 相同）。寄存器 **200**、0 spill。

单/两文件：`launch_bwd_wgmma2<HD>` + CLI `--wg2`；`run_main` 优先走 wg2；`[O17 A/B]` 段同 session
对比 mma 最优档 / O9b wgmma / O17 wgmma2，并打印 `max|diff|`。**没有**改默认行为（需 `--wg2`）；
`D=512`（MLA）时 `--wg2` 被忽略、走原路径。

### 6s.4 数值（ours-vs-ref，bf16 causal，max_abs）

| shape | dq | dk | dv | `max\|diff\|` wg2-vs-mma (dq/dk/dv) |
|---|---|---|---|---|
| MHA S=512 | 9.001e-3 | 1.261e-2 | 1.365e-2 | 0 / 8.8e-5 / 1.2e-4 |
| MHA S=4096 | 1.510e-2 | 1.340e-2 | 1.631e-2 | 0 / 2.2e-4 / 4.6e-4 |
| GQA h32kv4 S=1024 | 1.201e-2 | 2.125e-2 | 3.156e-2 | 0 / 8.4e-4 / 1.3e-3 |
| GQA h40kv8 S=1024 | 1.233e-2 | 1.930e-2 | 3.150e-2 | — |
| MQA h64kv1 S=1024 | 1.190e-2 | 4.558e-2 | 7.196e-2 | 0 / 5.1e-3 / 5.9e-3 |

**vs ref 与 O5b~O13 的历史值逐位一致**；`dq` 在 MHA 下 **逐位相同**（dQ 无跨 CTA 原子），
dk/dv 只差 atomic 累加次序。单/两文件逐指标一致。

### 6s.5 性能（同 session A/B，CUDA event，main-only，ms）

| shape | mma 最优档 | O9b wgmma | **O17 wgmma2(BM128)** | vs mma |
|---|---|---|---|---|
| MHA S=512 (64,64,1) | 0.0577 | 0.0598 | **0.0521** | **1.109×** |
| MHA S=4096 (64,64,2) | 1.4892 | 1.5050 | **0.9821** | **1.516×** |
| GQA kv8 S=1024 (64,64,2) | 0.2995 | 0.2985 | **0.1992** | **1.504×** |
| GQA kv4 S=1024 (64,32,2) | 0.2657 | 0.2639 | **0.1764** | **1.506×** |
| MQA kv1 S=1024 (64,64,2) | 0.4258 | 0.4221 | **0.2781** | **1.531×** |

端到端（preprocess+main+convert，`--wg2`）：S=4096 **1.4309ms（96.05 TF，`4BS²HD` 口径）**、
S=512 0.1075ms（19.97 TF）、GQA kv8 0.3334ms（64.4 TF）、GQA kv4 0.2892ms（59.4 TF）、
MQA kv1 0.4519ms（76.0 TF）。**main-only S=4096 139.9 TF**（O9b ~91 TF）。
（真反向 FLOPs 口径 ≈ ×2：S=4096 端到端 ~192 TF。）

### 6s.6 ncu（main，S=4096，同 session，O9b vs O17）

| 指标 | O9b wgmma (BM=64) | **O17 wgmma2 (BM=128)** |
|---|---|---|
| `lts__t_sectors_op_red` | 102,236,160 | **51,904,512（0.508×）** |
| `lts__t_sectors_op_read` | 35,845,803 | 18,005,333（0.50×，Q/dO 少读一遍） |
| `lts__t_sectors_op_write` | 1,575,930 | 1,575,919 |
| Duration | 1.48 ms | **997 µs** |
| L2 Throughput | 71.61% | **54.63%** |
| L1/TEX | 40.25% | 36.68% |
| DRAM | 4.35% | 6.47% |
| Compute (SM) | 23.49% | 27.36% |
| regs / smem | 230 / 100.35 KB | 200 / 149.50 KB |
| achieved occ | 11.89%（2 CTA/SM） | 12.41%（1 CTA/SM） |
| bank conflict (ld/st) | 0 / 747 | 0 / 0 |

**red 字节精确减半、read 也减半**（BM=128 让 Q/dO 只读一遍），L2 71.6%→54.6%、
Duration 1.48→1.00ms——**O17 的机制假设被 ncu 完全证实，且与 fp16 O17 逐项一致**。
**新墙仍是 L2**（red 51.9M 占余下 L2 扇区 ~72.6%），故进一步路线是 **BM=256 / 4 warpgroups**
（再砍半）或 dK/dV 分块累加（O7b）。

### 6s.7 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）

| shape | FA2.7.4 | FA3 | TE2.14 | ours total | ours/FA3（时间） |
|---|---|---|---|---|---|
| MHA S=4096 kv16 | 0.7343/374 | **0.3217/855** | 0.4422/622 | 1.4309 | **4.45×**（~22.5% TFLOPS） |
| GQA kv8 S=1024 | 0.1887/228 | 0.1214/354 | 0.1319/326 | 0.3334 | **2.75×**（~36%） |
| GQA kv4 S=1024 | 0.1590/216 | 0.0825/417 | 0.1122/306 | 0.2892 | **3.51×**（~28%） |
| MQA kv1 S=1024 | 0.2660/258 | 0.1567/439 | 0.2149/320 | 0.4519 | **2.88×**（~35%） |

（FA3 列为基准；MLA head_dim=512 FA/TE 反向均不支持。）

### 6s.8 原始输出

`src/bf16/fa_bwd_bf16_mma_main_o17_{s512_h16_d128,s4096_h16_d128,s1024_h32_d128_kv4,
s1024_h40_d128_kv8,s1024_h64_d128_kv1}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o17_{s512,s4096}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o17_ncu_wg2_s4096.out.txt`、`..._o17_ncu_o9b_s4096.out.txt`、
`src/bf16/fa_bwd_bf16_o17_fa3_te_baseline.out.txt`。

---

## 6t. O17-2-bf16：把 O17 的 GEMM3/GEMM4 拆分到两个 warpgroup（负载再平衡）

与 fp16 §14l **逐字 dtype 同构**（`__half`→bf16、`f16.f16`→`bf16.bf16`）。O17 的 phase B
里只有 wg0 串行做 GEMM3(dV)+GEMM4(dK)（4 条串行 red 链），wg1 只做 GEMM5 ⇒ 张量工作量 3:1、
wg1 在 GEMM3/4 期间 barrier 空等。把 **GEMM3→wg0、GEMM4→wg1**（各自仍对全 BM=128 归约、
每个 KV 元素仍只 `red` 一次）后，两 wg 各一条 GEMM+red 链，工作量 1:1。

`fa_bwd_bf16_wgmma2_kernel<HD, bool SPLIT=true>`；host `--wg2split=0/1`（默认 1）可同 session
A/B。单/两文件 device 代码逐字一致（`sync_onefile_device.py` 核对 `identical: True`）。

### 6t.1 性能（同 session A/B，CUDA event，main-only，ms）

| shape | O17（wg0 串行 dV+dK） | **O17-2（wg0=dV,wg1=dK）** | 比 |
|---|---|---|---|
| MHA S=512 | 0.0518 | 0.0520 | 0.996×（噪声） |
| MHA S=4096 | 1.0014 | **0.9647** | **1.038×**（142.5 TF） |
| GQA kv4 S1024 | 0.1859 | **0.1769** | **1.051×**（97.1 TF） |
| GQA kv8 S1024 | 0.2048 | **0.2023** | 1.012× |
| MQA kv1 S1024 | 0.2844 | **0.2837** | 1.003× |

端到端 S=4096 **1.942 ms（70.8 TF）**、S=512 0.112、GQA kv4 0.409、kv8 0.439、MQA 0.598 ms。
数值与 O5b~O17 历史值**逐位一致**（S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3、
GQA kv4 12.01/21.25/31.56e-3、kv8 12.33/19.30/31.50e-3、MQA kv1 11.90/45.58/71.96e-3），
`max|diff|` dk/dv ~5e-5~2e-4（仅 atomic 次序）、dq = 0。

### 6t.2 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）

FA3 MHA S=4096 **0.3204ms/858TF**、TE 0.4437/620、FA2 0.7283/377；GQA kv4 FA3 0.0829/414。
ours 端到端 S=4096 1.942ms ⇒ 时间为 FA3 的 **6.06×**（与 fp16 O17-2 同量级）。

### 6t.3 原始输出

`src/bf16/fa_bwd_bf16_mma_main_o17_2_sweep.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o17_2_s4096.out.txt`、
`src/fa_bwd_o17_2_fa3_te_baseline_bf16.out.txt`。

---

## 6u. O18-bf16：BN=128 版 wgmma2（kv-tile 翻倍，tile 数减半）

与 fp16 §14m **逐字 dtype 同构**（`__half`→bf16、`f16.f16`→`bf16.bf16`）。O17/O17-2（BM=128、
BN=64）后 main 仍 **1 CTA/SM（12.5%）+ 延迟受限**；把 per-CTA 的 KV-tile 从 `BN=64` 翻到
**`BN=128`**（`docs/01` §14k.7 item 3），per-CTA 的 tile 数减半 ⇒ `__syncthreads`、
`cp.async.wait_group`、wgmma `commit_group`+`wait0` 序列都减半；GEMM1/2 用
`wgmma.mma_async.m64n128k16`（一条指令算两倍、发射/依赖链减半）。代价是 smem
`148.5→230.4KB`（仍 1 CTA/SM）。

结构（HD=128、BM=128、BN=128、256 线程 = 2 warpgroup）与 O17 完全同构，仅：
① GEMM1/2 用 `wgmma_mn128_issue`（A/B 均 K-major SW128）；
② P/dS tile 变 `[128][128]`，softmax epilogue 列组 `j=0..15`；
③ GEMM3/4 输出 `[BN=128][HD=128]` ⇒ 分 2 个 m64 半（`mh=0/1`、转置描述符基址 `+mh*1024`）；
④ GEMM5 输出 `[64][HD=128]`（每 wg 自己 64 行）⇒ 一条 `m64n128`、`dqacc[16][4]`。
新增 `wgmma_m64n128k16_bf16_t<TA,TB>` + `wgmma_mn128_issue`；`fa_bwd_bf16_wgmma2b_kernel<HD,SPLIT>`。
host 加 `--wg2bn`（默认 `SPLIT=1`，与 O17-2 同：GEMM3(dV)→wg0、GEMM4(dK)→wg1，都对全 BM=128
归约）。单/两文件 device 代码逐字一致（`sync_onefile_device.py` 核对 `identical: True`）。

### 6u.1 数值（ours-vs-ref，bf16 causal，max_abs）—— 与历史**逐位一致**

| shape | dq | dk | dv |
|---|---|---|---|
| MHA S=512 | 9.001e-3 | 1.261e-2 | 1.365e-2 |
| MHA S=4096 | 1.510e-2 | 1.340e-2 | 1.631e-2 |
| GQA kv4 S1024 | 1.201e-2 | 2.125e-2 | 3.156e-2 |
| GQA kv8 S1024 | 1.233e-2 | 1.930e-2 | 3.150e-2 |
| GQA kv4(h64) S1024 | 1.351e-2 | 3.091e-2 | 4.420e-2 |
| MQA kv1 S1024 | 1.190e-2 | 4.558e-2 | 7.196e-2 |

与 O5b~O17 历史值**逐位相同**；单文件与两文件逐指标一致。`max|diff|` wg2b-vs-wg2：
dq `1.8–3.6e-7`、dk/dv `6e-5–8.2e-4`（仅跨 CTA `atomicAdd` 次序）。

### 6u.2 性能（同 session A/B，CUDA event，main-only，ms）

| shape | O17 wg2(BN64) | **O18 wg2b(BN128,split)** | 比 | O18 串行(BN128) | 比 |
|---|---|---|---|---|---|
| MHA S=512 | 0.0507 | **0.0492** | **1.030×**（43.6 TF） | 0.0497 | 1.019× |
| MHA S=4096 | 0.9799–0.9836 | **0.9558–0.9560** | **1.025–1.029×**（143.8 TF） | 0.984–0.985 | 0.995–1.000× |
| GQA kv4 S1024 | 0.1800 | 0.1806 | 0.996×（中性） | 0.1851 | 0.972× |
| GQA kv8 S1024 | 0.2037 | 0.2028 | 1.005× | 0.2080 | 0.980× |
| GQA kv4(h64) | 0.2846 | 0.2839 | 1.002× | 0.2896 | 0.983× |
| MQA kv1 S1024 | 0.2836 | 0.2816 | 1.007× | 0.2857 | 0.993× |

**结论**：MHA（大 tile、tile 数多）稳定 +2.5–3.0%；GQA/MQA（grid 小、tile 数本就少）中性。
串行版（wg0 串行 dV+dK）在 BN=128 下普遍更慢 ⇒ 保留 O17-2 的 SPLIT。

端到端（`--wg2bn`）：S=4096 **1.381 ms（99.5 TF）**、S=512 0.104 ms（20.6 TF）、
GQA kv4 0.291（59.1 TF）/ kv8 0.332（64.7）/ kv4(h64) 0.447（76.8）/ MQA kv1 0.447（76.8）。
O17-bf16 时 S=4096 为 1.431ms ⇒ 端到端 **1.036×**。

### 6u.3 ncu（main，S=4096，同 binary `--wg2bn` vs `--wg2`）

| 指标 | O17 wg2(BN64) | **O18 wg2b(BN128)** |
|---|---|---|
| Duration | 982.2 µs | **952.5 µs（1.031×）** |
| `lts__t_sectors_op_red` | 51,904,512 | **51,904,512（逐字节不变）** |
| `read` / `write` | 18.073M / 1.576M | 18.373M / 1.574M |
| L1/TEX / L2 / DRAM | 44.2% / 54.6% / 4.4% | 44.2% / 56.9% / 6.7% |
| Compute (SM) | 27.4% | 23.5% |
| regs / smem / occ | 200 / 148.5KB / 12.5% | **255 / 230.4KB / 12.5%（1 CTA/SM）** |
| Waves | 3.88 | 3.88 |
| stall barrier / wait / long / short | 0.46 / 1.19 / 0.75 / 0.40 | 0.93 / 1.25 / 0.60 / 0.43 |
| bank conflict (`op_ld`) | 0 | 0 |

`red` 逐字节不变（BN 不动归约结构）；收益来自 tile 数减半后 barrier/commit-wait 序列减半
（`barrier` 每-tile 占比升但总数降、`long_scoreboard` 0.75→0.60）。墙仍是 **L2（red 占 L2
扇区 ~72%）+ 1 CTA/SM**；BN 翻倍不再降 red，只有跨 CTA 归约（O7b）或提 occupancy 能再推进。

### 6u.4 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`，FA2/FA3/TE 三列）

| shape | FA3 (ms/TF) | TE (ms/TF) | ours total (ms/TF) | ours/FA3 时间 |
|---|---|---|---|---|
| MHA S=4096 | 0.3209 / 857 | 0.4423 / 621 | 1.381 / 99.5 | **4.30×** |
| GQA kv4 S1024 | 0.0826 / 416 | 0.1128 / 305 | 0.291 / 59.1 | 3.52× |
| GQA kv8 S1024 | 0.1214 / 354 | 0.1323 / 325 | 0.332 / 64.7 | 2.73× |
| GQA kv4(h64) | 0.1602 / 429 | 0.1940 / 354 | 0.447 / 76.8 | 2.79× |
| MQA kv1 S1024 | 0.1571 / 437 | 0.2136 / 322 | 0.447 / 76.8 | 2.85× |

ours 端到端相对 O17-bf16（S4096 4.45×、kv8 2.75×、kv4 3.51×、MQA 2.88×）**全 shape 小幅改善**。

### 6u.5 原始输出

`src/bf16/fa_bwd_bf16_mma_main_o18_{s512,s4096}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o18_gqa_{kv4,kv8,kv4h64,kv1}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o18_{s512,s4096}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o18_ncu_s4096.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o18_ncu_red_{s4096,o17_s4096}.out.txt`、
`src/fa_bwd_o18_fa3_te_baseline_bf16.out.txt`。

---

## 6v. O23-bf16：Hopper 路径默认化（主 kernel wgmma2/wgmma2b + LSE wgmma）

### 6v.1 动机 / 改动

与 fp16（`docs/01` §14n）**逐字同构**：O17-bf16/O18-bf16 的主 kernel 与 O9a 的 LSE wgmma 一直
是 `--wg2`/`--wg2bn`/`--lsewgm` 显式开关，默认仍是 mma。本项在 `-DFA_WGMMA`（sm_90a）构建下
把它们**默认打开**（`wg_forced`/`lse_forced` 标记 + 自动段：D==128 且 S>=4096→BN=128，否则
BN=64；causal D==128→LSE wgmma），`--wg2=0 --wg2bn=0`/`--lsewgm=0` 保留 mma 对照；非
`FA_WGMMA` 构建不变。单/两文件 host 逐字一致，device 代码未动。

### 6v.2 数值（ours-vs-ref，bf16 causal，max_abs）—— 与历史逐位一致

S512 9.001/12.61/13.65e-3；S4096 15.10/13.40/16.31e-3；GQA kv4 12.01/21.25/31.56e-3；
MQA kv1 11.90/45.58/71.96e-3；MLA S512H4 8.753/10.82/17.40e-3。单/两文件逐指标一致
（S4096 total 1.3677 vs 1.3666ms）。

### 6v.3 性能（同 session 端到端 total，CUDA event，ms）

| shape | 旧默认（mma） | **新默认** | × | 主 kernel 后端 |
|---|---|---|---|---|
| MHA S512 | 0.1119 | **0.1058** | 1.06 | wgmma2(BN=64) |
| GQA kv4 S1024 | 0.3763 | **0.2875** | 1.31 | wgmma2 |
| MQA kv1 S1024 | 0.6022 | **0.4438** | 1.36 | wgmma2 |
| MHA S4096 | 1.9429 | **1.3666** | **1.42** | wgmma2b(BN=128) |
| MLA S512 D512 | 0.4321 | 0.4366 | 1.00 | mma（D=512） |

main-only A/B：S4096 mma 1.4643 vs O17 0.9912（1.48×）vs O18 0.9607（**143.1 TF**）；S512 mma
0.0562 vs O17 0.0502（1.12×）；GQA kv4 mma 0.2714 vs O17 0.1806（1.50×）；MQA kv1 mma 0.4311 vs
O17 0.2848（1.51×）。

### 6v.4 ncu / 对标

默认路径即 `fa_bwd_bf16_wgmma2b_kernel`；ncu 与 fp16 O18-bf16 逐项一致（`red=51,904,512` 逐字节
不变、255 regs / 230.4KB smem / occ 12.5% / L2 ~57%），墙仍是 **L2 的 dK/dV 跨 CTA `red` +
1 CTA/SM**。同 session 纯反向基线（`harness/fa_vs_te_bwd_only.py bf16`）：MHA S4096 FA3
0.3210ms/856TF、TE 0.4410/623 ⇒ ours total 1.3666ms = **FA3 的 4.26× / TE 的 3.10×**；
GQA kv4 S1024 FA3 0.0826/416、TE 0.1121/307 ⇒ 3.48× / 2.56×；MQA kv1 FA3 0.1567/439、
TE 0.2140/321 ⇒ 2.83× / 2.07×。

### 6v.5 原始输出

`src/bf16/fa_bwd_bf16_mma_main_o23_{s4096,shapes}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o23b_s4096.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o23b_s4096.out.txt`、`src/fa_bwd_o23_default_ab.out.txt`、
`src/fa_bwd_o23_shapes_final.out.txt`、`src/fa_bwd_o23_fa3_te_baseline_bf16.out.txt`。

---

## 6w. O24-bf16：preprocess `delta` 向量化 + dQ 直写 bf16（与 fp16 逐字同构）

把 fp16 的 O24（见 `01` §14p）逐字 dtype 参数化到 bf16：`delta_warp_kernel<HD>`
（`__half2`→`__nv_bfloat162`、`__half22float2`→`__bfloat1622float2`）；`wgmma2`/`wgmma2b`
的 dQ epilogue 加 `bf16* dq_h`（非空时 `__floats2bfloat162_rn` 直写 `dq`，convert 跳过 dQ）。
单/两文件 device 由 `sync_onefile_device.py` 同步（`device region identical: True`）。

### 6w.1 数值（ours-vs-ref，bf16 causal，max_abs）—— 与历史**逐位一致**

S512 9.001/12.61/13.65e-3；S4096 15.10/13.40/16.31e-3；GQA kv4 12.01/21.25/31.56e-3；
MQA kv1 11.90/45.58/71.96e-3。

### 6w.2 性能（同 session A/B，CUDA event，端到端 total）

| shape | base(`--deltawarp=0 --dqdirect=0`) | **O24** | 加速 |
|---|---|---|---|
| MHA S=4096 | 1.3650 ms | **1.3388 ms** | 1.020× |
| MHA S=512 | 0.1050 | **0.0997** | 1.053× |
| GQA kv4 S=1024 | 0.2877 | **0.2733** | 1.053× |
| MQA kv1 S=1024 | 0.4423 | **0.4139** | 1.069× |

`delta` 单项：S=4096 0.0424→**0.0126ms（3.36×）**、S=512 0.0075→0.0035（2.15×）。
ncu 与 fp16 逐项一致（新 delta Duration **14.5µs**、DRAM 71% bound、指令数 −82%）。

### 6w.3 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`）

MHA S=4096 FA3 **0.3202ms/859TF**、TE 0.4359/631、FA2 0.7271/378 ⇒ ours total 1.3388ms =
**FA3 的 4.18×**（时间；O23 4.20×）。

### 6w.4 原始输出

`src/bf16/fa_bwd_bf16_main_o24_sweep.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o24_s4096.out.txt`、
`src/fa_bwd_o24_fa3_te_baseline_bf16.out.txt`。

---

## 6x. O31-bf16：LSE 4D-TMA（把 fp16 O30 逐字 dtype 参数化）

> fp16 侧见 `01` §14s（O30）。bf16 与 fp16 同为 2 字节、SW128 布局 / 描述符 / TMA box 内维
> （128B = 64 元素）/ 两个 `SBO=1024` 的 `2×K=64` chunk 拆分**逐字节同构**，仅把
> `wgmma.m64n64k16.f16` → `wgmma.m64n64k16.bf16`、tensormap 的 `CU_TENSOR_MAP_DATA_TYPE_FLOAT16`
> → `..._BFLOAT16`。因此 O30 的全部结论原样成立：LSE 的 Q/K 由「逐 16B `cp.async` + 地址运算」
> 换成 **4D TMA**（坐标 `{k0,row,head,batch}`，一条 bulk 指令搬一个 8KB chunk），复用同一份
> SW128 tile 供 wgmma 直读。

### 6x.1 实现（单/两文件 device 代码逐字一致）

* device（`fa_bwd_bf16_mma_kernels.cuh`）：新增 `mbar_init/arrive_expect/wait`、`tma_load_4d`、
  `wgmma_qkt64_tma`（bf16 版）与 `lse_mma_kernel_bal_tma<HD,PIPE>`（`static_assert(HD==128)`，
  镜像配对 + online-softmax + 4-lane `shfl` 归约与 `lse_mma_kernel_bal_wgmma` 相同）。
* host（`..._main.cu`）：`make_lse_map`（dtype=BFLOAT16）、`--lsetma=0/1`、`kLseSmemTma1`、
  `cudaFuncSetAttribute` + tensormap 建立、`run_pre` 里 D==128/causal 默认走 TMA，以及同 session
  `[O31 A/B]` 数值对拍。整体用 `-DFA_WGMMA -DFA_TMA -lcuda` 包裹；纯 `sm_90` 或仅 `-DFA_WGMMA`
  构建**完全不编译/不引用驱动符号**（已验证仍分别落回 `lse=mma` / `lse=wgmma`）。
* 单文件：device 区由同步脚本核对 `device region identical: True`，host 段与两文件同步重建。
  > 注：bf16 单文件的 `#include <algorithm>` 位于 device 区**之前**，`sync_onefile_device.py`
  > 的边界假定不成立；本轮改用「以 `using bf16 = ...` 到 `#endif` 为 device 区」的等价同步，
  > 并单独给单文件补 `#include <cuda.h>`（`CUtensorMap`）。

### 6x.2 数值（ours-vs-ref bf16 causal max_abs；`[O31 A/B]`）

* `[O31 A/B] max_abs(tma-vs-wgmma) = 0.000e+00`（S512/S4096 **逐位相同**）。
* ours-vs-ref 与历史逐位一致：S512 9.001/12.61/13.65e-3；S4096 15.10/13.40/16.31e-3。

### 6x.3 性能（同 session A/B，CUDA event）

| shape | LSE wgmma+cp.async | **LSE 4D-TMA** | LSE 加速 | 端到端 total（wgmma→TMA） | 端到端加速 |
|---|---|---|---|---|---|
| MHA S=4096 | 0.2819 ms | **0.2087 ms** | **1.351×** | 1.3390→**1.2577 ms**（102.6→109.3 TF） | 1.065× |
| MHA S=512 | 0.0331 ms | **0.0251 ms** | **1.318×** | 0.1002→**0.0936 ms**（21.4→22.9 TF） | 1.071× |

（TF 为 `4BS²HD` 口径；换算到 `fa_vs_te` 的 `4BS²H(D+Dv)` 口径，S4096 total ≈ **218.5 TF**。）

### 6x.4 ncu（S=4096，`--set full --launch-count 1`，同 binary）

| 指标 | LSE wgmma+cp.async | **LSE 4D-TMA** |
|---|---|---|
| Duration | 285.60 µs | **214.24 µs（1.333×）** |
| Executed Instructions | 165.30 M | **117.87 M（0.713×，−28.7%）** |
| Compute (SM) Throughput | 60.64 % | 57.85 % |
| L1/TEX / L2 | 18.03 / 20.27 % | 18.28 / 26.82 % |
| DRAM Throughput | 3.78 % | 5.05 % |
| Registers / thread | 62 | 58 |
| Waves Per SM / Occupancy | 0.97 / 23.0 % | 0.97 / 23.1 % |

结论与 fp16 O30 逐项一致：TMA 把 load 指令/地址运算压掉（**指令数 −28.7%**），LSE 仍由
**softmax epilogue 的 Compute（~58%）** 限速（非访存），故收益是「少发指令」而非「提带宽」；
`Duration 1.33×` 与 event 的 `1.35×` 同向。

### 6x.5 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`，FA2/FA3/TE 三列）

MHA S=4096：FA2 **0.7264ms/378TF**、FA3 **0.3194/861**、TE **0.4424/621**。ours total
1.2577ms（218.5 TF 同口径）= **FA3 的 3.94×**（O24 时 4.18×、O23 时 4.20×）。峰值占比：
bf16 dense 峰值 ~989 TF ⇒ main-only 144.6 TF ≈ **14.6% 峰值**（total 109.3/989 ≈ 11.0%）。

### 6x.6 原始输出

`src/bf16/fa_bwd_bf16_o31_lsetma_ab.out.txt`（同 session 4 跑 A/B）、
`src/bf16/fa_bwd_bf16_o31_lse_tma_s4096.out.txt` / `_s512.out.txt`、
`src/bf16/fa_bwd_bf16_o31_lse_tma_onefile_s4096.out.txt`（单文件）、
`src/bf16/fa_bwd_bf16_lse_tma_ncu_s4096.out.txt` / `fa_bwd_bf16_lse_wgmma_ncu_s4096.out.txt`、
`src/bf16/fa_bwd_bf16_o31_fa3_te_baseline.out.txt`。

---

## 6y. O34-bf16：主 kernel 的 Q/K/V/dO 逐 atom 4D-TMA（把 fp16 O33 逐字 dtype 参数化）

### 6y.1 动机 / 改动

O31（§6x）只把 **LSE** 的 Q/K 换成了 4D-TMA。主 kernel 的 Q/dO（prologue）与 K/V（每个 KV
tile）仍用逐 16B `cp.async` + `sw128_off` 地址运算。HD=128 的 K-major tile 的交织布局无法用
**一个 2D TMA box** 复现（O15a），因此照搬 fp16 O33 的做法：**逐 atom 发 TMA** —— 一个
`[8 行][64 列]` 的 box 恰好等于一个 1024B SW128 atom，dst 放到 `sw128_off(rg*8, kg*64, HD)`，
**原样复现交织布局** ⇒ 所有 wgmma 描述符零改动，搬的字节与 `cp.async` 逐字节相同。

实现（单/两文件 device 区逐字一致，均 `-DFA_WGMMA -DFA_TMA -lcuda` / `sm_90a` 构建）：

- `tma_fill_sw128<R,HD>`：逐 `(rg,kg)` 发 `tma_load_4d`（`cp.async.bulk.tensor.4d`），
  Q/dO 各 32 atom、K/V 各 32 atom，由 `tid0` 串行发射（异步、不占寄存器/不记 scoreboard）；
  同步改 mbarrier：Q/dO 一次性，K 双缓冲两 barrier（phase 每 `nt` 翻），V 单缓冲后段预取。
- `fa_bwd_bf16_wgmma2b_tma_kernel<HD,SPLIT>`：与 `fa_bwd_bf16_wgmma2b_kernel` 几何/数据流/
  描述符逐字相同，只换载入；仅 `HD=128`、BN=128（wgmma2b 几何）。
- host：`make_main_map`（4D 描述符 box `{64,8}`、`BFLOAT16`）+ `launch_bwd_wgmma2b_tma` +
  CLI `--maintma=0/1`（opt-in，与 cp.async 版同 binary A/B）；`make_main_map` 与 fp16 O33
  逐字节同构（同样 2B，字节 stride 不变）。

### 6y.2 数值（bf16 causal，max_abs）—— 与历史一致

`[O34 A/B] max|diff| tma-vs-cpasync`：**dq `0.00e+00`（逐位）**、dk/dv ~5e-5–1.4e-4
（仅跨 CTA `atomicAdd` 次序）；`ours vs ref`：dq 1.510e-2 / dk 1.340e-2 / dv 1.631e-2
（与 O18/O23/O24/O31 **逐位一致**）。

### 6y.3 性能（同 session A/B，CUDA event）

| 形态 | main O18 wg2b(cp.async) | main O34 wg2b(TMA) | main 比 | 端到端 total cp.async → TMA | 比 |
|---|---|---|---|---|---|
| 两文件 S4096 | 0.9644 ms / 142.5 TF | **0.9223 ms / 149.0 TF** | **1.046×** | 1.2581 → **1.2106 ms** (109.2→113.5 TF) | **1.039×** |
| 单文件 S4096 | 0.9599 ms / 143.2 TF | **0.9170 ms / 149.9 TF** | **1.047×** | 1.2591 → **1.2187 ms** (109.2→112.8 TF) | **1.033×** |

### 6y.4 ncu（S=4096，同 binary，`--launch-count 1`）

| 指标 | cp.async (O18) | TMA (O34) |
|---|---|---|
| Duration | 958.98 µs | **922.02 µs（1.040×）** |
| Executed Instructions | 214,705,152 | **161,438,720（−24.8%）** |
| `lts__t_sectors_op_red` | 51,904,512 | **51,904,512（逐字节不变）** |
| regs / smem / occ | 254 / 230.53KB / 12.46% | 255 / 230.53KB / 12.46% |
| L1/TEX / L2 / Compute / DRAM | 44.01 / 57.62 / 23.29 / 6.21 % | 45.84 / 60.26 / 19.52 / 6.48 % |
| stall long / short / wait / barrier | 0.61 / 0.44 / 1.23 / 0.94 | 1.37 / 0.47 / 1.33 / 1.47 |

⇒ **TMA 只省搬运（指令数 −24.8%）**，动不了主墙（dK/dV 的 L2 `red` 占 L2 ~72%）；1 CTA/SM
与 regs/smem 全不变。

### 6y.5 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`，FA2/FA3/TE 三列）

FA3 MHA S4096 **0.3194 ms / 861 TF**、TE 0.4420 / 622 ⇒ ours total（maintma）
**3.79×**（两文件）/ 3.82×（单文件）（O31 为 3.94×）。

### 6y.6 原始输出

`src/bf16/fa_bwd_bf16_o34_maintma{0,1}_s4096.out.txt`（两文件）、
`fa_bwd_bf16_o34_onefile_maintma{0,1}_s4096.out.txt`（单文件）、
`fa_bwd_bf16_o34_ncu_cpasync_s4096.out.txt` / `fa_bwd_bf16_o34_ncu_maintma_s4096.out.txt`（`--set full`）、
`fa_bwd_bf16_o34_ncu_red_stall_s4096.out.txt`、`fa_bwd_bf16_o34_fa3_te_baseline.out.txt`。

---

## 6z. O36-bf16：BN=64 版 `wgmma2` 主 kernel 的 Q/K/V/dO 也改用逐 atom 4D-TMA（补全 O34 的几何）

### 6z.1 动机 / 改动

O34（§6y）只把 **BN=128** 的 `wgmma2b` 主 kernel 的 Q/K/V/dO 换成 4D-TMA；而 **O23 的默认档
在 S<4096 与 GQA/MQA 走的是 BN=64 的 `wgmma2`**——这条更常用的路仍用逐 16B `cp.async` +
`sw128_off` 地址运算。本项把 O34 的做法搬到 BN=64 几何（对齐 fp16 O35，`docs/01` §14t），
补全 backlog 里「BN=64 的 `wgmma2` 几何待做」这一条，并量化它在不同 S 下的收益。

实现（单/两文件 device 区逐字一致，`scripts/sync_onefile_device.py` 核对 `identical: True`，
均 `-DFA_WGMMA -DFA_TMA -lcuda` / `sm_90a` 构建）：

- 新增 `fa_bwd_bf16_wgmma2_tma_kernel<HD,SPLIT>`：与 `fa_bwd_bf16_wgmma2_kernel` 的
  **几何/数据流/描述符逐字相同**（BM=128、BN=64、2 warpgroup，GEMM1/2 `wgmma_mn64_issue`、
  GEMM3/4/5 `wgmma_m64n64k16_bf16_t` + MN-major 转置描述符、GEMM5 的 `dqacc[2][8][4]` 寄存器
  累加），只把载入与同步换成 TMA + mbarrier：prologue `mbar_init` 4 barrier（`qbar,kbar0,kbar1,
  vbar`），tid0 发 Q/dO（`tma_fill_sw128<128,HD>`，expect `2*QTILE`）与 K0/V0
  （`tma_fill_sw128<64,HD>`，各 `KTILE`，共 8×2=16 atom）；循环内 K 双缓冲两 barrier、
  V 单缓冲后段预取；smem 与 cp.async 版同（+4 个 mbarrier），仍 **1 CTA/SM**。
- host：`launch_bwd_wgmma2_tma<HD,SPLIT>`（复用 O34 已建的 `make_main_map`，box `{64,8}`、
  `BFLOAT16`，无需新描述符），并让 `--maintma` 在 **BN=64 的 `wg2` 分支**也生效（此前只在
  `wg2b`/BN=128 生效）；新增 `[O36 A/B]`（mode 10）与 cp.async 版（mode 2）做同 session
  head-to-head + `max|diff|`。

### 6z.2 数值（bf16 causal，max_abs）—— 与历史一致

| shape | dq | dk | dv | `max|diff|`(TMA-vs-cp.async) dq/dk/dv |
|---|---|---|---|
| S=512 H16 | 9.001e-03 | 1.261e-02 | 1.365e-02 | `0.00e+00 / 4.20e-05 / 5.34e-05` |
| S=1024 H32 kv4 (GQA) | —（同 O18） | — | — | `0.00e+00 / 2.44e-04 / 1.83e-04` |
| S=4096 H16 | 1.510e-02 | 1.340e-02 | 1.631e-02 | `0.00e+00 / 9.16e-05 / 7.63e-05` |

**dq 逐位相同**；dk/dv 差异只来自跨 CTA `atomicAdd` 次序（搬的是与 `cp.async` 逐字节相同的
smem），不构成精度问题。`ours vs ref` 与 O18/O23/O24/O31/O34 **逐位一致**。

### 6z.3 性能（同 session A/B，CUDA event，main-only）

| shape | wg2(BN64, cp.async) | wg2(BN64)+TMA | 比 |
|---|---|---|---|
| S=512 H16（两文件） | 0.0507 ms (42.4 TF) | 0.0513 ms (41.8 TF) | **0.987×** |
| S=512 H16（单文件） | 0.0507 ms (42.4 TF) | 0.0513 ms (41.9 TF) | **0.989×** |
| S=1024 H32 kv4 | 0.1797 ms (95.6 TF) | 0.1803 ms (95.3 TF) | **0.997×** |
| S=4096 H16（强制 BN=64） | 0.9889 ms (139.0 TF) | 0.9654 ms (142.4 TF) | **1.024×** |

与 fp16 O35 的 0.992× / 0.996× / 1.021× 同量级：小 S/中等 S 是**延迟/grid bound**（S=512
`grid=128<132 SM`、Waves 0.48），省发射换不到时间；S=4096（强制 BN=64）才体现在 Duration。

> 注：S≥4096 的默认档是 **BN=128 的 `wgmma2b`**（O34，§6y），O36 只在用户显式
> `--wg2=1 --maintma=1` 或用 BN=64 几何时生效；故它主要价值是**补全几何覆盖、确认机制**，
> 不改变默认端到端数字（默认 S4096 total 仍 1.2676 ms / 108.4 TF）。

### 6z.4 ncu（main，S=512，同 session、同 binary，`-c 1`）

| 指标 | cp.async（BN64） | **+TMA** |
|---|---|---|
| Executed Instructions | 5,259,008 | **3,976,256（−24.4%）** |
| Registers/thread | 200 | **184** |
| Duration | 53.54 µs | **53.02 µs（持平）** |
| Waves Per SM / occ | 0.48 / 12.27% | 0.48 / 12.39% |
| L2 / Compute / DRAM | 25.89 / 10.62 / 9.46 % | 26.58 / 8.01 / 9.55 % |

结论：**TMA 只省「搬运的指令/地址运算」（−24.4%）**，动不了 BN=64 小 S 的延迟/grid 墙；
与 O33/O34/O35 完全一致。

### 6z.5 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py bf16`，FA2/FA3/TE 三列）

S=4096 MHA：FA2 0.7277ms/378TF、**FA3 0.3191ms/861TF**、TE 0.4426ms/621TF；GQA kv4 S=1024：
FA3 0.0825ms/416TF、TE 0.1118ms/307TF。默认端到端 ours（O36 不改默认）S=4096 total
1.2676ms（108.4 TF）⇒ **FA3/ours = 3.97×**；GQA kv4 S1024 total 0.2582ms（66.6 TF）⇒ 3.13×。

### 6z.6 原始输出

`src/bf16/fa_bwd_bf16_mma_main_o36_{s512,gqa_kv4,s4096}.out.txt`（两文件 A/B + 对拍）、
`fa_bwd_bf16_mma_onefile_o36_s512.out.txt`（单文件 `[O36 A/B]`）、
`fa_bwd_bf16_o36_ncu_main_{tma,cpasync}_s512.out.txt`（`--set full`）、
`fa_bwd_bf16_o36_fa3_te_baseline.out.txt`。

---

## 6aa. VARLEN：bf16 反向支持变长 / `cu_seqlens`（单/两文件）

> 把 fp16 VARLEN（`docs/01` §16）**逐字 dtype 参数化**到 bf16（`__half`→`bf16`、
> `__float2half`→`__float2bfloat16`、`__half2float`→`__bfloat162float`）。算法数据流、
> smem 布局、cu_seqlens 语义完全同构；FA/TE 变长在本机不可用 ⇒ 只对 fp32 ref。

### 6aa.1 实现（单/两文件 device 逐字一致，`sync_onefile_device.py` 核对 `identical: True`）

与 fp16 §16.2 相同的 3 处：4 个搬运 helper 的默认参数 `qbase`、`lse_mma_kernel_bal_wgmma`
与 `fa_bwd_bf16_wgmma2_kernel` 加 `cu_seqlens`（`qbase/len`、镜像配对早退）。host 加
`--varlen` 分支 `run_varlen`（`launch_bwd_wgmma2<128,true>`，non-TMA）。

### 6aa.2 数值（ours vs fp32 ref，bf16 causal；max_abs）

| case | lengths | dq | dk | dv | total (ms) | TFLOPS（Σ_b 4HL²D） |
|---|---|---|---|---|---|---|
| b4_t3840 不齐 | `[512,1024,2048,256]` | 1.340e-2 | 1.276e-2 | 1.911e-2 | 0.7756 | 58.84 |
| b4_t4096 等长 | `[1024]×4` | 1.355e-2 | 1.207e-2 | 1.772e-2 | 0.4507 | 76.24 |
| b5_t3968 GQA kv8 | `[128,256,512,1024,2048]` | 1.398e-2 | 2.396e-2 | 3.131e-2 | 1.4279 | 64.10 |
| b8_t2904 强倾斜 | `[2048,512,…,8]` | 1.464e-2 | 1.566e-2 | 1.863e-2 | 0.5835 | 63.00 |

全部 bf16 噪声量级（~1–3e-2），无 system error；单/两文件**逐位相同**（b4_t3840：
1.340e-2/1.276e-2/1.911e-2）。定长回归 `nullptr` **逐位不变**：S512 `9.001/12.61/13.65e-3`。

### 6aa.3 ncu（主 kernel，b4_t3840，`--set full --launch-count 1`）

Duration 551.55 µs、DRAM 10.02% / **L1/TEX 48.77% / L2 62.90%** / Compute 31.22%、
202 regs、Block Limit Shared Mem 1、**occ 12.43%**、Waves 7.76、No Eligible 63.46% ——
与 fp16 varlen（`docs/01` §16.5）逐项一致，bound = **L2 red + L1/TEX + 1 CTA/SM 延迟受限**。

### 6aa.4 对标（同口径 `8BS²HD`，等长 `[1024]×4` = B4 S1024 H16 D128）

ours total **0.4507 ms / 152.5 TF**；FA2.7.4 0.2504ms/274.4TF、**FA3 0.1454ms/472.7TF**、
TE2.14 0.1755ms/391.6TF ⇒ ours 时间 = FA3 的 **3.10×**、TFLOPS 为 FA3 的 **32%**。

### 6aa.5 原始输出

`src/bf16/fa_bwd_bf16_varlen_b{4_t3840,4_t4096,5_t3968,8_t2904}*.out.txt`（两文件）、
`..._onefile_b4_t3840.out.txt`（单文件）、`..._ncu_main_b4_t3840.out.txt`（ncu）、
`src/fa_bwd_varlen_fa3_te_baseline_bf16.out.txt`（FA2/FA3/TE 基线）。

### 6aa.6 非 causal（full attention）——第 79 轮

把 fp16 §16.8 **逐字 dtype 参数化**到 bf16：`lse_mma_kernel<HD>` 加默认参数
`const int* cu_seqlens = nullptr`（`qbase/len`、`if (m0>=len) return;`），`nullptr` 逐式退化、
定长逐位不变；host `run_varlen` 去掉「只做 causal」限制（causal→镜像配对 wgmma LSE，
非 causal→`lse_mma_kernel<128>` + `d_cu`）。单/两文件 device 逐字一致
（`sync_onefile_device.py` 核对 `identical: True`）。

**数值（ours vs fp32 ref，bf16 full，max_abs dq/dk/dv）**：b4_t3840 不齐
`5.764e-3/3.851e-3/3.031e-3`；b4_t4096 等长 `3.237e-3/2.392e-3/2.013e-3`；b5_t3968 GQA kv8
`5.324e-3/5.454e-3/5.881e-3`；b8_t2904 强倾斜 `1.153e-2/9.529e-3/1.083e-2` —— 全 bf16 噪声、
无 padding 泄漏；单/两文件逐位一致。**定长回归逐位不变**（causal S512
`9.001/12.61/13.65e-3`；fixed full S1024 `1.94/1.68/1.45e-3`）。

**性能（total，event，`Σ_b 4HL²D` 口径）**：b4_t3840 1.2874ms/35.45TF、b4_t4096（等长）
0.9088ms/37.81TF、b5_t3968 GQA 2.4621ms/37.18TF、b8_t2904 1.0547ms/34.85TF。
按定长口径 `4BS²H(D+Dv)` 乘 2 ⇒ 等长 **75.6 TF**；同 session TE bf16 定长 full
`0.2787ms/246.6TF`、FA2.7.4 `0.4663ms/147.4TF` ⇒ ours 为 TE 的 3.26×。

**ncu**：与 fp16 §16.8 同构（L2 red + 1 CTA/SM 延迟受限）。原始输出
`src/bf16/fa_bwd_bf16_varlen_full_sweep.out.txt`、`src/fa_bwd_varlen_full_regression.out.txt`。

### 6ab VARLEN 的 MLA（head_dim=512）——第 81 轮

把 fp16 §16.9 **逐字 dtype 参数化**到 bf16：`lse_mma_kernel_bal<HD>` 与 `fa_bwd_bf16_mma_kernel`
各加默认参数 `const int* cu_seqlens = nullptr`（`qbase/len`、短序列配对 CTA 早退、EV/边界全改），
`nullptr` 逐式退化、**定长逐位不变**；host `run_varlen` 按 D 分派（D=512 走 mma 主 kernel
`launch_bwd_mma<512,32,32,1,false,true>` + `lse_mma_kernel_bal<512,1>`/`lse_mma_kernel<512>`）。
单/两文件 device 逐字一致（`sync_onefile_device.py` 核对 `identical: True`）。

**数值（ours vs fp32 ref，max_abs dq/dk/dv）**：b3_t1792 `[256,512,1024]` H2 D512 causal
`1.267/1.217/1.796e-2`、full `3.100/3.526/2.316e-3`；**单/两文件逐位一致**；定长 D=512 回归
逐位不变（S1024H2 causal `5.838/9.519/1.568e-2`，与历史文件相同）；HD=128 varlen full 回归
b4_t3840 `5.764/3.851/3.031e-3` 与第 79 轮逐位相同。

**性能（total，event，`Σ_b 4HL²D` 口径）**：b3_t1792 causal **0.9347ms/6.03TF**、
full 1.4690ms/3.84TF；b1_t512 causal **0.4258ms/2.52TF**、full 0.5471ms/1.96TF，
同 shape 定长 causal 0.4247ms/2.53TF ⇒ **varlen 开销 +0.3%**。MLA 反向 FA3/TE 均不支持。

**ncu**：与 fp16 §16.9 同构（main = 1 CTA/SM + smem→mma 依赖，No Eligible 89%；
lse = Waves<1 的并行度 bound）。原始输出 `src/bf16/fa_bwd_bf16_varlen_mla_sweep.out.txt`。

## 6ac. O38-bf16：LSE 的 K 维 split + 二次归约（第 85 轮）—— **正结果，默认 auto**

> 把 fp16 **O38**（`docs/01` §14u）逐字 dtype 参数化到 bf16 的 **TMA LSE**
> （`lse_mma_kernel_bal_tma`，D=128/causal 默认快路）：`b=blockIdx.z/ksplit`、
> `ksp=blockIdx.z%ksplit`、每 `(pair,ksp)` 只扫 K tile 切片 `[nt0,nt1)`、流水 stage 用切片内
> 相对下标 `rnt&1`、`ksplit==1` 逐位退回 O31；新增 `lse_split_merge_kernel` 做二次归约。
> host 加 `--lsesplit=N`（`0=auto`，目标 `grid*split≈528`，上限 8）；单/两文件 device 逐字一致
> （`sync_onefile_device.py` 核对 `identical: True`）。

* **附修 bug**：bf16 **单文件** `fa_bwd_bf16_mma_onefile.cu` 之前在 `-DFA_TMA` 构建下会因缺
  `make_lse_map`/`make_main_map`（历次 device 同步把夹在 device 区与 `struct NpyF32` 之间的
  host 段吞掉）而编译失败；本轮把这俩 host 函数补回并移到 `struct NpyF32` **之后**，使其不在
  同步区的替换范围内。
* **数值（ours-vs-ref，bf16 causal，max_abs）** 与历史逐位一致：S512 9.001/12.61/13.65e-3、
  S4096 15.10/13.40/16.31e-3、GQA kv4 12.01/21.25/31.56e-3、kv8 12.33/19.30/31.50e-3、
  MQA kv1 11.90/45.58/71.96e-3；`max_abs(split-vs-split1)` ≤ 2e-6（fp32 求和次序）。单/两文件一致。
* **性能（同 session `[O38 A/B]`，CUDA event）**：LSE-only S512 0.0257→**0.0149ms（1.73×，split8）**、
  kv4 0.0496→**0.0364ms（1.36×，split2）**、S4096 0.2145（split1，1.00×）、kv8/kv1 1.00×；
  端到端 total S512 0.0938–0.0948→**0.0842ms（1.13×）**、kv4 0.2581–0.2594→**0.2451ms（1.06×）**、
  S4096 1.2591→1.2671ms（1.00×）。merge 仅 ~4µs。
* **ncu** 与 fp16 §14u.4 逐项一致（split1 Waves 0.12 / occ 6.25% → split8 Waves 0.97 / occ 20.61%，
  墙 = 并行度/临界路径，非指令/访存）。
* **对标（同 session `harness/fa_vs_te_bwd_only.py bf16`）**：S4096 ours 1.2671 vs FA3 0.3197/860、
  TE 0.4428/621（3.96×）；kv4 0.2451 vs FA3 0.0827/415（2.96×）；kv8 0.2916 vs 0.1213/354
  （2.40×）；MQA 0.3966 vs 0.1570/438（2.53×）。
* 原始输出 `src/bf16/fa_bwd_bf16_mma_main_o38_{b1_s512_h16_d128_causal_bf16,
  b1_s4096_h16_d128_causal_bf16,b1_s1024_h32_d128_kv4_causal_bf16,b1_s1024_h40_d128_kv8_causal_bf16,
  b1_s1024_h64_d128_kv1_causal_bf16}.out.txt`、`src/bf16/fa_bwd_bf16_mma_onefile_o38_s512.out.txt`、
  `src/fa_bwd_o38_split1_baseline.out.txt`、`src/fa_bwd_o38_fa3_te_baseline_fp16_bf16.out.txt`。

## 6ad. O39-bf16：MLA（head_dim=512）LSE 的 K 维 split + 二次归约（第八十六轮）—— **正结果，默认 auto**

把 O39-fp16（`docs/01` §14v）逐字 dtype 参数化到 bf16：`lse_mma_kernel_bal<HD,PIPE>` 加
`float* lse_part,int ksplit`（`grid.z: B→B*ksplit`、`b=z/ksplit`、每 CTA 扫 K tile 切片
`[nt0,nt1)`、stage 用相对下标 `rnt&1`、`ksplit==1` 逐位退回 O8b）；host `--lsesplit=N`
（`0=auto`：D=512 目标 `grid*split≈132`（smem ~202KB ⇒ 1 CTA/SM）、上限 16，按
`nblk=ceil(S/64)` 封顶）；D=128 的 TMA/非 TMA mma 路径与 fp16 同构。单/两文件 device
逐字一致（`sync_onefile_device.py` 核对 `identical: True`）。

**数值**（vs fp32 ref，bf16 causal，max_abs dq/dk/dv）：S256H2 1.230e-2/9.875e-3/1.686e-2；
S512H4 8.753e-3/1.082e-2/1.740e-2；S1024H2 5.838e-3/9.519e-3/1.568e-2 —— 与 §6l 历史逐位
一致；`max_abs(split-auto vs split1) ≤ 9.5e-7`。MHA D=128 回归逐位不变（S512
9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3、GQA kv4 12.01/21.25/31.56e-3）。

**性能**（CUDA event，同 session `--lsesplit=1` vs auto）：

| case | LSE split1 | LSE auto | LSE 倍数 | total split1 | total auto | total 倍数 |
|---|---|---|---|---|---|---|
| MLA S256H2 | 0.0350 ms | **0.0202** (split4) | **1.73×** | 0.2179 | **0.2025** | **1.08×** |
| MLA S512H4 | 0.0585 ms | **0.0221** (split8) | **2.65×** | 0.4182 | **0.3816** | **1.10×** |
| MLA S1024H2 | 0.1006 ms | **0.0276** (split8) | **3.64×** | 0.7855 | **0.7116** | **1.10×** |

**ncu** 与 fp16 逐项一致（LSE smem ~202KB ⇒ 1 CTA/SM）；split1 Waves 0.12 →
split8 Waves 0.97、Duration 4.2×，墙 = 并行度 → 1 CTA/SM 的 `mma wait` + smem 依赖。

原始输出：`src/bf16/fa_bwd_bf16_main_o39_b1_s{256_h2,512_h4,1024_h2}_d512_causal.out.txt`、
`..._o39_split1_*`、`..._o39_reg_*`、`..._o39_mmafallback_s512.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o39_*`、`..._o39_varlen_*`。

## 6ae. O40-bf16：非 TMA wgmma LSE 的 K 维 split + varlen 接入（第八十七轮）—— **正结果，默认 auto**

把 O40-fp16（`docs/01` §14w）逐字 dtype 参数化到 bf16：`lse_mma_kernel_bal_wgmma<HD,PIPE>` 加
`float* lse_part,int ksplit`（`b=blockIdx.z/ksplit`、每 `(pair,ksp)` 只扫本 m 块 K tile 切片
`[nt0,nt1)`、stage 用相对下标 `rnt&1`、`ksplit==1` 逐位退回 O9a/O9b）；`run_varlen` 加
`--lsesplit=N`（0=auto：D=128 目标 `grid*split≈528`（= 一个波）、D=512 `≈132`，上限 8/16，
按最大序列 `nblk` 封顶），D=128 causal 走 wgmma 版、D=512 causal 走 O39 的 mma 版，merge 行数
传 `T*H`；各加 `d_lse_part` 缓冲。单/两文件 device 逐字一致（`sync_onefile_device.py` 核对
`identical: True`）。

**数值**（vs fp32 ref，bf16 causal，max_abs dq/dk/dv）：定长 S512 `--lsetma=0`
9.001/12.61/13.65e-3；varlen b4_t3840_h16 D128 1.340e-2/1.276e-2/1.911e-2；varlen b1_t512_h2 D512
8.042e-3/1.097e-2/1.391e-2；b3_t1792_h2 D512 1.267e-2/1.217e-2/1.796e-2 —— 与 §6l/§6ab 历史
逐位一致；`max_abs(split vs split1)=0`。D=128 MHA 默认 TMA 路径回归逐位不变。

**性能**（CUDA event，同 session `--lsesplit=1` vs auto）：

| case | split1 | auto | 倍数 |
|---|---|---|---|
| MHA S512（定长 `--lsetma=0`，preprocess） | 0.0368 ms | **0.0215 ms** | **1.71×** |
| MHA S512（total） | 0.1015 ms | **0.0861 ms** | **1.18×** |
| varlen b1_t512_h2 D512（total） | 0.4100 ms | **0.3737 ms** | **1.10×** |
| varlen b3_t1792_h2 D512（total） | 0.9100 ms | **0.8630 ms** | **1.05×** |

bf16 的 D=128 varlen case base 已 ≥ 528 ⇒ auto=1、中性；D=512 MLA varlen 受益 1.05–1.10×。
单文件与两文件同量级（S512 0.0845 vs 0.0861）。

**ncu**（LSE `lse_mma_kernel_bal_wgmma`，S512）：split1 Waves 0.12 / occ 6.25% / Compute 9.31% /
Duration 33.50µs → split8 **Waves 0.97 / occ 19.27% / Compute 36.24% / Duration 14.91µs**，
与 fp16 逐项一致；墙 = 网格不足 → 填满后回到 `mma wait` + smem 依赖。

原始输出：`src/bf16/fa_bwd_bf16_mma_main_o40_fixed_s512_split{1,0}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o40_varlen_{b1_t512_h2_d512,b3_t1792_h2_d512,b4_t3840_h16_d128,
b4_t4096_h16_d128}_split{1,0}.out.txt`、
`src/bf16/fa_bwd_bf16_mma_main_o40_ncu_lse_wgmma_split{1,8}_s512.out.txt`、
`src/bf16/fa_bwd_bf16_mma_onefile_o40_fixed_s512_auto.out.txt`。

## 6af. O43-bf16：wgmma2 主 kernel 的 N 方向 split-K（第九十轮）—— **正结果，默认 auto（仅小 grid）**

把 fp16 O43（`docs/01` §14x）**逐字 dtype 参数化**到 bf16：`fa_bwd_bf16_wgmma2_kernel` 加
`int ksplit = 1`（`ksp=bx%ksplit`/`mblk=bx/ksplit`、KV tile 切片 `[nt_begin,nt_end)`、切片为空
早退、预取/列偏移改全局 tile 号），dQ 在 `ksplit>1` 时改 `red_add2(dq_acc)` 跨 CTA 原子累加；
`launch_bwd_wgmma2`/host 同步加 `--wg2ksplit=N`（默认 -1=auto：仅 D==128/BN=64/非 TMA 且
`ceil(S/128)*H*B<132` 时切到「填满一个波」，cap 8）；varlen 不做 auto（同 fp16 负结果）。
单/两文件 device 逐字一致（`sync_onefile_device.py` 核对 `identical: True`）。

**数值**（ours-vs-ref，bf16 causal，max_abs）：S=512 MHA ksplit=1/2/4 全为
`9.001/1.261e-2/1.365e-2`（逐位一致）；S=4096 causal `1.51/1.34/1.63e-2` 与旧一致。

**性能**（同 binary、同 session）：S=512 MHA main 0.0538→**0.0320ms（1.68×）**、
total 0.0869→**0.0692ms（1.26×，31.1 TF，单文件 0.0698 一致）**；ksplit=4 略差（0.0735）。
同 session 纯反向对标：FA2 0.0438 / **FA3 0.0262ms（164 TF）** / TE 0.0323 ⇒ ours/FA3
**3.32×→2.64×**。S=1024 / S=4096 走 wgmma2b，auto=1、不变。ncu 与 fp16 逐项同构
（Waves 0.48→0.97，墙 = grid 不足一个波、非 per-SM occupancy）。

原始输出：`src/bf16/fa_bwd_bf16_o43_sweep.out.txt`。

## 6ag. O44-bf16：MLA（D=512）mma 主 kernel 的 N 方向 split-K（第九十一轮）—— **正结果，默认 auto**

把 fp16 O44（`docs/01` §14y）**逐字 dtype 参数化**到 bf16。MLA（head_dim=512）走的是
`fa_bwd_bf16_mma_kernel<512,32,32,1>`，其 `grid=ceil(S/32)·H·B` 在 S1024H2=64 / S512H4=64 /
S256H2=16，ncu `Waves 0.48`、occ 6.25%（207KB smem→1 CTA/SM）⇒ 并行度不足。改动同 fp16：
kernel 加 `int ksplit=1`（`ksp=bx%ksplit`/`mblk=bx/ksplit`、KV tile 切片、空切片早退、
**prologue stage 用 `(nt_begin&1)` 对齐循环首 `nt&1`**）；`ksplit>1` 时 HD>128 的 GEMM5 dQ
从非原子 RMW 改 `red_add2`；host `launch_bwd_mma` 加 `ksplit`、`--mlaksplit=N`（auto 目标
`grid*sp≈528`、上限 16、按 `nblk` 封顶，仅 D==512）。单/两文件 device 逐字一致
（`sync_onefile_device.py` 核对 `identical: True`）。

**数值**（ours-vs-ref，bf16 causal，max_abs dq/dk/dv）与 O5c/§7.8 **逐位一致**：
(1,256,2,512) `1.230e-2/9.875e-3/1.686e-2`；(1,512,4,512) `8.753e-3/1.082e-2/1.740e-2`；
(1,1024,2,512) `5.838e-3/9.519e-3/1.568e-2`。MHA D=128 回归逐位不变。

**性能**（CUDA event，Hopper 构建，同 binary/同 session；`[O44 A/B]` main-only ksplit=1/2/4）：

| MLA case | ksplit=1 | 2 | 4 | **auto（=8）** | total（auto） | total 比 |
|---|---|---|---|---|---|---|
| (1,256,2,512) | 0.1852 | 0.0513 | 0.0279 | **0.0220 ms** | 0.0548 ms（4.90 TF） | 5.4× |
| (1,512,4,512) | 0.3624 | 0.1094 | 0.0932 | **0.0842 ms** | 0.1283 ms（16.74 TF） | 4.2× |
| (1,1024,2,512) | 0.7136 | 0.2121 | 0.1738 | **0.1518 ms** | 0.2000 ms（21.47 TF） | 4.7× |

对比 O5c 的 total 0.300/0.536/0.939ms ⇒ **4.2–5.4×**；ncu 与 fp16 O44 逐项同构（Waves
0.48→0.97、per-SM occ 不变 6.23%、Issue Ipc 0.33→0.57、头号 stall 从 long_scoreboard 变
fixed-latency `wait`）。FA/TE 反向后端不支持 D=512，无第三方对标；bf16 MLA total 已比
fp8 MLA（`docs/04` §7.7）快 ~5×。

原始输出：`src/bf16/fa_bwd_bf16_o44_mla_sweep.out.txt`；单文件
`src/bf16/fa_bwd_bf16_mma_onefile.cu` 同源。

## 6ah. O46-bf16：MLA（D=512）mma 主 kernel 的「256 线程 / 8-warp 几何」（第九十三轮）—— **正结果，D=512 默认**

把 fp16 O46（`docs/01` §14z）**逐字 dtype 参数化**到 bf16。O45 指出 MLA 主 kernel 是
**1 CTA/SM（207KB smem）× 4 warp ⇒ 每 scheduler 1 warp**（ncu `No Eligible 85.7%`、
`long 2.33 + wait 1.54 + short 0.76`），墙 = 并行度/延迟；唯一杠杆是 8-warp。

改动同 fp16：`fa_bwd_bf16_mma_kernel<…>` 加模板参数 `NTH`（线程数）/`NWAR`（N 方向 warp 数），
几何 `NWM=NTH/32/NWAR`、`GM1=BM/NWM, GN1=BN/NWAR, GMV=BN/NWM, GNV=NTW/NWAR, GMQ=BM/NWM,
GNQ=NTW/NWAR`（`NTW≡128`），`wr=wid/NWAR`、`wc=wid%NWAR`；`kv_issue_async`/`qdo_issue_async`
加默认 `NTH` 模板参数；kernel 内 `THREADS`→`NTH`。默认 `128/2` 与 O5b/O5c 逐字等价；MLA 走
`256/4`。host 加 `--mla8w=0/1`（D=512/BM=32/PIPE=1 默认 1）与 `LAUNCH_CFG_W`；`run_varlen`
保持 4-warp。单/两文件 device 逐字一致。

**数值**（ours-vs-ref，bf16 causal，max_abs）与 O5c/§6ag **一致**：(256,2,512)
`1.230e-2/9.875e-3/1.686e-2`；(512,4,512) `8.753e-3/1.082e-2/1.740e-2`；(1024,2,512)
`5.838e-3/9.519e-3/1.568e-2`。MHA D=128 回归逐位不变（S512 9.001/12.61/13.65e-3、
S4096 15.10/13.40/16.31e-3、GQA kv4 12.01/21.25/31.56e-3）。

**性能**（CUDA event，同 binary/同 session `[O46 A/B]`，main-only）：

| MLA case | 4-warp | 8-warp | 加速 | total 4w→8w |
|---|---|---|---|---|
| (1,256,2,512) | 0.0224 | **0.0205 ms** | 1.093× | 0.0534→0.0534 |
| (1,512,4,512) | 0.0843 | **0.0761 ms** | 1.108× | 0.1287→0.1196 |
| (1,1024,2,512) | 0.1514 | **0.1415 ms** | 1.070× | 0.1912→0.1898 |

ncu 与 fp16 O46 同构（occ 6.1→12.4%、Ipc 0.57→0.65、regs 168→151、0 spill、墙仍 L2 red+延迟）。
FA/TE 反向后端不支持 D=512。

原始输出：`src/bf16/fa_bwd_bf16_o46_sweep.out.txt`；单文件
`src/bf16/fa_bwd_bf16_mma_onefile.cu` 同源。

## 8. 下一步

> **O36-bf16（§6z）已完成**：把 O34 的逐 atom 4D-TMA 从 BN=128 的 `wgmma2b` 补到 **BN=64 的
> `wgmma2`**（对齐 fp16 O35）：S4096（强制 BN=64）main **1.024×**（139.0→142.4 TF）、
> S=512/GQA 中性（0.987–0.997×），指令数 **−24.4%**、regs 200→184、`dq` 逐位不变，
> 数值与历史一致。默认端到端数字不变（S≥4096 走 BN=128）。bf16 的 TMA 几何至此全覆盖。
> fp16/bf16 主 kernel 的 TMA 化家族（O30–O36）收口；**下一步首选 = fp8 主 kernel 的
> Q/K/V/dO TMA**（fp8 一行 128B = 一个 SW128 atom 的整行，TMA 更简单；但需处理 Kp/Qp/dOp
> 配对副本的 smem 重建与 dS3/Ap 复用）。
>
> **O34-bf16（§6y）已完成**：主 kernel Q/K/V/dO 逐 atom 4D-TMA（对齐 fp16 O33），
> main **1.046–1.047×**（142.5→149.0 TF）、端到端 **1.033–1.039×**（MHA S4096 1.2581→1.2106ms）、
> 指令数 **−24.8%**、`red` 逐字节不变，数值与历史逐位一致。
>
> **O31-bf16（§6x）已完成**：LSE 改 4D-TMA（对齐 fp16 O30），LSE-only **1.32–1.35×**、指令数
> **−28.7%**、端到端 **1.065–1.071×**（MHA S4096 1.3390→1.2577ms），`max_abs(tma-vs-wgmma)=0`
> 逐位一致；纯 `sm_90`/仅 `-DFA_WGMMA` 构建行为不变。
>
> **O24-bf16（§6w）已完成**：preprocess `delta` 改 warp-per-row 向量化（3.36×）+ dQ 直写 bf16，
> 端到端 1.02–1.07×，数值与历史逐位一致。
>
> **O23-bf16（§6v）已完成**：Hopper 快路默认化（主 kernel wgmma2/wgmma2b + LSE wgmma），
> 端到端 **1.06–1.42×**（MHA S4096 1.943→1.367ms），数值与历史逐位一致。

见 `../ROADMAP.md`。**O5b（bf16 张量核，§6e）、O8（preprocess mma，§6f）、O6（main
`cp.async` 双缓冲，§6g）、O6b（K/V 降 smem 回 3 CTA/SM + A 转置读，§6h）、O8b（LSE 负载
均衡 + cp.async，§6i）、O6c（tile 几何参数化 + 小网格自适应，§6j）、O7c（LSE/D 预装 +
float4 试错，§6k）、MLA 张量核（§6l）、O10（Q/dO 向量化 + cp.async 重叠，§6m）、
O11（快速 exp/log，§6n）、O9a（LSE wgmma，§6o）、O13（auto tile 重标定，§6p）、
O9b（主 kernel GEMM1/2 上 wgmma，§6q）、O9b-2（主 kernel GEMM3/4/5 也用 MN-major 转置读上
wgmma，数值逐位正确、性能中性，§6r）、**O17-bf16（跨 warpgroup 归约：BM=128、2 warpgroups，
dK/dV 的 red 字节砍半，main S=4096 1.52×、GQA/MQA 1.50–1.53×，§6s）已完成**、
**O17-2-bf16（GEMM3/GEMM4 拆分到两个 wg，消 barrier 空等，main S=4096 1.038×、GQA kv4 1.051×，
§6t）已完成**、**O18-bf16（BN=128 版 wgmma2，tile 数减半，MHA main 1.025–1.030×、GQA/MQA
中性，red 逐字节不变，§6u）已完成**；
接下来是 **O7b（去 dK/dV 跨 CTA 原子 → 确定性反向）**、**fp8 侧同构跨 wg 归约**、
**O15 TMA 化 Q/K/V/dO + P/dS 双缓冲跨-tile 流水 + 压 smem 冲更高 occupancy**。
backlog：MLA 降 smem 冲 2 CTA/SM / split-KV。
