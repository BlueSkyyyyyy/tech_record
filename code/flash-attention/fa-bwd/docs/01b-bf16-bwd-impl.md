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

## 8. 下一步

见 `../ROADMAP.md`。**O5b（bf16 张量核，§6e）、O8（preprocess mma，§6f）、O6（main
`cp.async` 双缓冲，§6g）、O6b（K/V 降 smem 回 3 CTA/SM + A 转置读，§6h）、O8b（LSE 负载
均衡 + cp.async，§6i）、O6c（tile 几何参数化 + 小网格自适应，§6j）、O7c（LSE/D 预装 +
float4 试错，§6k）、MLA 张量核（§6l）、O10（Q/dO 向量化 + cp.async 重叠，§6m）、
O11（快速 exp/log，§6n）、O9a（LSE wgmma，§6o）、O13（auto tile 重标定，§6p）、
O9b（主 kernel GEMM1/2 上 wgmma，§6q）、**O9b-2（主 kernel GEMM3/4/5 也用 MN-major 转置读上
wgmma，数值逐位正确、性能中性，§6r）已完成**；
接下来是 **O7b（去 dK/dV 跨 CTA 原子，唯一直接打掉 L2 墙的杠杆）** 与 **O9b-2b（TMA 化
Q/K/V/dO + P/dS 双缓冲跨-tile 流水 + 压 smem 冲 3 CTA/SM）**。
backlog：fp8 侧残余 red（O7b）、MLA 降 smem 冲 2 CTA/SM / split-KV。
