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
