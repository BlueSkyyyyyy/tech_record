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

## 8. 下一步

见 `../ROADMAP.md`。**O5b（bf16 张量核）已完成（第 6e 节）**；接下来是「当前冲刺」的
**O8（preprocess 的 LSE/D 改 mma 分块，对齐 fp8 的 O1）**——它现在是端到端第一瓶颈
（S=4096 preprocess 68.8ms vs main 4.5ms），然后 O6（main `cp.async` 双缓冲/提 occupancy）、
O7（去 atomic）、O9（wgmma+TMA 对标 FA3）。
backlog：fp8 侧残余 red（O7b）、MLA 降 smem / 张量核。
