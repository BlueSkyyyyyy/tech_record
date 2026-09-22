# FA 反向：数值 + 性能汇总（P4-1）

> 汇总 fp16 / bf16 / fp8 三种 dtype、单文件 / 两文件两种形态的实现结果：
> 数值对拍（vs fp32 ref、vs FA2.7.4、vs TE2.14）与性能对标（CUPTI 纯 device 时间 vs FA/TE）。
> 数据来源全部为实测原始输出（`src/**/*.out.txt`），本轮并重跑了两文件版与 FA/TE 基线：
> `src/fa_bwd_ours_summary.out.txt`（ours 两文件版）、`src/fa_bwd_refbench_summary.out.txt`（FA/TE）。
>
> 详细实现/优化说明见 `01-fp16-bwd-impl.md`、`01b-bf16-bwd-impl.md`、`02-fp8-bwd-design.md`、
> `03-fp8-bwd-impl.md`；状态见 `../ROADMAP.md`。

---

## 0. 口径与常量

- **测试机**：H100 80GB HBM3（sm90），CUDA 13.2 / nvcc / ncu，容器 `kernel_lab`。
- **基线 shape**（causal，D=128，scale=1/√D）：
  - `b1_s512_h16`（B=1,S=512,H=16）
  - `b1_s1024_h32`（B=1,S=1024,H=32）
  - `b1_s4096_h16`（B=1,S=4096,H=16）
- **FLOPs 口径**：`4·B·S²·H·D`（反向 ≈ 前向 2×）。**causal 未折半**，故所有实现的 TFLOPS
  都是下界；ours / FA / TE 用同一口径，横向可比。
- **峰值**（H100 dense）：FP16/BF16 Tensor Core ~989 TFLOPS；FP8 ~1978.8 TFLOPS。
- **计时**：ours 用 CUDA event（`run.sh` 内），FA/TE 用 CUPTI 纯 device 时间
  （`harness/fa_bwd_bench.py bench`）。“端到端”= preprocess+main+convert（fp8 另含 quant）；
  “main”单指反向主 kernel。
- **ref**：PyTorch fp32 autograd；**FA**：flash_attn 2.7.4（fp16/bf16）；**TE**：TransformerEngine 2.14
  （fp16/bf16 与 FP8 E4M3 前向 / E5M2 反向 + rowwise quantizer）。

> 注意：ours 现阶段的性能目标是「正确性打通 + 张量核化 + 找 bound」，尚未做流水/高 occupancy，
> 因此下面 ours 的绝对值远低于 FA/TE 属预期（详见 §3 的 bound 与 backlog）。

---

## 1. 数值对拍（max_abs，causal）

### 1.1 fp16（正确性容差 ~1e-3）

| shape | 对比 | dq | dk | dv |
|---|---|---|---|---|
| S=512 H16 | **ours vs ref** | 1.671e-3 | 1.680e-3 | 1.899e-3 |
| | FA vs ref | 1.679e-3 | 1.684e-3 | 1.899e-3 |
| | TE vs ref | 1.716e-3 | 2.287e-3 | 1.899e-3 |
| S=4096 H16 | **ours vs ref** | 1.499e-3 | 1.572e-3 | 2.225e-3 |
| | FA vs ref | 1.883e-3 | 1.734e-3 | 1.966e-3 |
| | TE vs ref | 1.883e-3 | 1.858e-3 | 1.966e-3 |

**结论**：ours 与 FA/TE 同量级（~1.5–2.3e-3），无系统误差，个别指标（S=4096 dq/dk）优于 FA/TE。

### 1.2 bf16（正确性容差 ~1e-2）

| shape | 对比 | dq | dk | dv |
|---|---|---|---|---|
| S=512 H16 | **ours vs ref** | 6.892e-3 | 8.110e-3 | 1.365e-2 |
| | FA vs ref | 1.040e-2 | 1.261e-2 | 1.365e-2 |
| | TE vs ref | 1.374e-2 | 1.068e-2 | 1.365e-2 |
| S=4096 H16 | **ours vs ref** | 8.895e-3 | 8.078e-3 | 1.494e-2 |
| | FA vs ref | 1.441e-2 | 1.332e-2 | 1.631e-2 |
| | TE vs ref | 1.524e-2 | 1.763e-2 | 1.631e-2 |

**结论**：ours 同量级（~7e-3–1.5e-2），dq/dk 略优于 FA/TE，dv 三者几乎一致（受 bf16 尾数限制）。

### 1.3 fp8（E4M3/E5M2 rowwise；容差 O(1)，见 `03` §2）

| shape | 对比 | dq | dk | dv |
|---|---|---|---|---|
| S=512 H16 | **ours vs ref** | 2.426e-1 | 2.975e-1 | 3.735e-1 |
| | TE vs ref | 4.859e-1 | 4.038e-1 | 5.906e-1 |
| | ours vs TE | 5.381e-1 | 4.429e-1 | 8.546e-1 |
| S=1024 H32 | **ours vs ref** | 2.400e-1 | 4.195e-1 | 3.536e-1 |
| | TE vs ref | 4.360e-1 | 4.500e-1 | 8.560e-1 |
| | ours vs TE | 4.652e-1 | 5.701e-1 | 9.431e-1 |
| S=4096 H16 | **ours vs ref** | 2.635e-1 | 2.643e-1 | 3.216e-1 |
| | TE vs ref | 3.763e-1 | 3.686e-1 | 6.688e-1 |
| | ours vs TE | 4.525e-1 | 5.324e-1 | 6.807e-1 |

**结论**：ours 与 fp32 ref、与 TE 的偏差都在 **同一量级（~0.24–0.94）**，无系统误差；
误差来自 FP8 量化本身（尾数 2–3 位），与 fp32 累加无关。**S=4096 时 ours-vs-ref 全面优于
TE-vs-ref**（ours 0.24–0.32 vs TE 0.37–0.67）——本版 dS/输出保留 fp32、dP 不进张量核，
口径更保守。`max_rel` 会被近零元素放大，不作判据（ref 梯度 amax 仅 3–6）。

---

## 2. 性能对标（CUPTI / event，causal）

`ours total` = 端到端（fp8 含 quant）；`ours main` = 反向主 kernel。峰值占比括号内。

### 2.1 fp16（峰值 989 TFLOPS）

| shape | ours total | ours main | FA2.7.4 | TE2.14 |
|---|---|---|---|---|
| S=512 H16 | 3.61 ms / 0.60 TF (0.06%) | 2.38 ms | 0.0703 ms / 30.54 TF (3.09%) | 0.0457 ms / 46.97 TF (4.75%) |
| S=1024 H32 | — | — | 0.2238 ms / 76.78 TF (7.76%) | 0.1372 ms / 125.20 TF (12.66%) |
| S=4096 H16 | 138.33 ms / 0.99 TF (0.10%) | 68.57 ms | 1.0582 ms / 129.88 TF (13.13%) | 0.5968 ms / 230.31 TF (23.29%) |

### 2.2 bf16（峰值 989 TFLOPS）

| shape | ours total | ours main | FA2.7.4 | TE2.14 |
|---|---|---|---|---|
| S=512 H16 | 3.12 ms / 0.69 TF (0.07%) | 1.88 ms | 0.0691 ms / 31.10 TF (3.14%) | 0.0455 ms / 47.24 TF (4.78%) |
| S=1024 H32 | — | — | 0.2212 ms / 77.68 TF (7.85%) | 0.1364 ms / 125.94 TF (12.73%) |
| S=4096 H16 | 111.33 ms / 1.23 TF (0.12%) | 41.91 ms | 1.0478 ms / 131.17 TF (13.26%) | 0.5918 ms / 232.23 TF (23.48%) |

> bf16 的 K/V smem 行距 +2 padding 已消除 9.5-way bank conflict，S=4096 main 从 197→42 ms
> （4.7×），现已快过同款 fp16 main；**端到端被 preprocess 拖住**（S=4096 69.4 ms > main 41.9 ms）。

### 2.3 fp8（峰值 1978.8 TFLOPS；FA 无反向 FP8，仅对标 TE）

| shape | ours total | ours main | TE FP8 |
|---|---|---|---|
| S=512 H16 | 1.69 ms / 1.27 TF (0.06%) | 0.450 ms | 0.0724 ms / 29.67 TF (1.50%) |
| S=1024 H32 | 10.88 ms / 1.58 TF (0.08%) | 1.827 ms | 0.1377 ms / 124.80 TF (6.31%) |
| S=4096 H16 | 80.63 ms / 1.70 TF (0.09%) | 10.22 ms | 0.4544 ms / 302.48 TF (15.29%) |

- fp8 **golden（标量）→ mma 张量核**：main 提速 **13–20×**（S=512 5.90→0.45 ms；
  S=4096 198.5→10.2 ms），数值不降反升（见 `03` §7.4）。
- main 相对 FP8 峰值仍只 0.24–0.68%、相对 TE FP8 约 4–16%（低 occupancy、无流水、每 tile 原子累加）。
- **端到端瓶颈已转移到 preprocess**（LSE 的 O(S²) 点积未分块）：S=4096 71.0 ms vs main 10.2 ms。

> **下表为 O1/O2 优化后的最新值（第十二轮）**，覆盖上表旧值；详细见 `03` §9（O1）与 §10（O2）。
> fp16/bf16 两表自 P4-1 起未变。

| shape | ours total（O2） | ours main（O2） | TE FP8（同 session） |
|---|---|---|---|
| S=512 H16 | 0.621 ms / 3.46 TF (0.17%) | 0.461 ms | 0.0724 ms / 29.67 TF |
| S=1024 H32 | 1.921 ms / 8.94 TF (0.45%) | 1.526 ms | 0.1371 ms / 125.35 TF |
| S=4096 H16 | 10.067 ms / 13.65 TF (0.69%) | 8.562 ms | 0.4550 ms / 302.08 TF |

- O1：`preprocess` 改 mma 分块 LSE + 独立 delta_kernel（14–62×），S=4096 端到端瓶颈回落 main。
- O2：`dS3/Ap` 折叠进 `Ks/Vs` 死空间，smem 80.13→75.01 KB → **3 CTA/SM**；main 1.16–1.19×。

---

## 3. ncu bound 小结（逐 dtype）

| dtype | kernel | DRAM | L1/TEX | Compute | Occ | Waves | 主 stall | bound 结论 |
|---|---|---|---|---|---|---|---|---|
| fp16 | main (S=512) | 0.21% | **53.5%**（75% 多余 wavefront） | 8.45% | 6.25%（96KB） | 0.48 | — | **smem 访问 + 低 occupancy** |
| bf16 | main (padding 后) | 0.26% | 24.9% | 13.5% | 6.25% | 0.48 | fixed-latency 37.3%、No Eligible 78% | **延迟 / 并行度不足** |
| fp8 | golden main | 0.06% | **75.96%**（90% 多余） | 4.98% | 6.25%（68KB） | 0.32 | MIO scoreboard 69% | **smem 冲突 + FP8 解码 + 低 occ** |
| fp8 | **mma main（O2 后, S=4096）** | 1.25% | 51.99% | 15.95% | **16.85%（75KB, 3 CTA/SM）** | 2.59 | No Eligible 79.6%、long_scoreboard/barrier | **延迟 / 并行度（3 CTA 后仍未饱和）+ 尾波** |

**共同结论**：三种 dtype 都不是 HBM 或算力 bound（DRAM <1%、Compute <16%）；真正的墙是
**低 occupancy（fp8 已做到 3 CTA/SM）+ 并行度不足**。
fp8 上 mma 后 bank conflict 已消失（L1/TEX 76%→19%），bound 从「smem 冲突」退化为「延迟受限」；
O2 降 smem 后 fp8 从 2→3 CTA/SM（theoretical 12.5%→18.75%），main 1.16–1.19×。
下一步优先级：**① `cp.async` 双缓冲流水（O3）；② 小 S 的 grid 太小 + 尾波（O2b）；
③ 4 CTA/SM 需消转置副本（O2c）；④ dQ/dK/dV 的 atomicAdd 换 `dQ_accum` 缓冲**。

---

## 4. 交付形态收口

| dtype | 单文件 | 两文件 | 对拍 | ncu | 文档 |
|---|---|---|---|---|---|
| fp16 | `src/fp16/fa_bwd_fp16_onefile.cu` | `_kernels.cuh` + `_main.cu` | ✅ | ✅ | `01-fp16-bwd-impl.md` |
| bf16 | `src/bf16/fa_bwd_bf16_onefile.cu` | `_kernels.cuh` + `_main.cu` | ✅ | ✅ | `01b-bf16-bwd-impl.md` |
| fp8 | `src/fp8/fa_bwd_fp8_mma_onefile.cu`（golden: `fa_bwd_fp8_onefile.cu`） | `_kernels.cuh` + `_main.cu` | ✅ | ✅ | `02-fp8-bwd-design.md` / `03-fp8-bwd-impl.md` |

两文件版与单文件版**逐指标一致**（fp16/bf16 数值与 ncu 逐项相同；fp8 数值逐位相同、
`Executed Instructions / regs / smem` 相同）。

---

## 5. 复现命令

```bash
cd code/flash-attention/fa-bwd

# ours（两文件版；默认 S=512 case）
scripts/run.sh src/fp16/fa_bwd_fp16_main.cu --iters=20
scripts/run.sh src/bf16/fa_bwd_bf16_main.cu --iters=20
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu  --iters=20
# 指定 case（S=1024H32 / S=4096）
scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --iters=5 \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s4096_h16_d128_causal_fp8

# FA / TE 基线（CUPTI 纯 device 时间）
H=harness/fa_bwd_bench.py
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python $H bench --dtype fp16 \
    --shape 1 512 16 128 causal --shape 1 1024 32 128 causal --shape 1 4096 16 128 causal
docker exec -e CUDA_VISIBLE_DEVICES=0 kernel_lab python $H bench --dtype fp8 \
    --shape 1 4096 16 128 causal

# ncu（示例：fp8 mma main）
scripts/ncu.sh src/fp8/fa_bwd_fp8_main.cu --set full --launch-count 1 \
    --kernel-name regex:fa_bwd_fp8_mma -- \
    --dir=/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8 --iters=1
```

---

## 6. 附件（本轮实测原始输出）

- `src/fa_bwd_ours_summary.out.txt`：ours 两文件版 fp16/bf16/fp8 三个 shape 的计时与对拍。
- `src/fa_bwd_refbench_summary.out.txt`：FA2.7.4 / TE2.14 CUPTI 基线（三 dtype × 三 shape）。
- 各 dtype 目录下的 `*_s512 / *_s4096 / *_ncu_main.out.txt`：单文件/两文件的历史原始输出。

---

## 7. 补充：GQA / MQA / MLA 形状（用户指定）

新增 7 个生产形状（均在 `harness/fa_bwd_bench.py` 的 `REQUESTED_SHAPES`，已 dump 到
`/home/xieminglin/proj/output/fa-bwd/<slug>/`，含 fp16/bf16/fp8 三份）：

| slug | 语义 |
|---|---|
| `b1_s1024_h40_d128_kv8_causal_*` | Qwen3-8B GQA（q=40, kv=8） |
| `b1_s1024_h32_d128_kv4_causal_*` | Qwen3-30B-A3B GQA（q=32, kv=4） |
| `b1_s1024_h64_d128_kv4_causal_*` | Qwen3-235B-A22B GQA（q=64, kv=4） |
| `b1_s1024_h64_d128_kv1_causal_*` | MQA（q=64, kv=1，如 DSA indexer） |
| `b1_s256_h2_d512_causal_*` | MLA（head_dim=512） |
| `b1_s512_h4_d512_causal_*` | MLA（head_dim=512） |
| `b1_s1024_h2_d512_causal_*` | MLA（head_dim=512） |

原始输出：`src/fa_bwd_bench_requested.out.txt`。

### 7.1 数值对拍（max_abs vs fp32 ref）

**fp16**（容差 ~1e-2）：

| shape | fa vs ref (dq/dk/dv) | te vs ref (dq/dk/dv) |
|---|---|---|
| kv8 | 1.76e-3 / 3.00e-3 / 4.33e-3 | 1.58e-3 / 2.89e-3 / 4.33e-3 |
| kv4 (h32) | 1.73e-3 / 3.32e-3 / 5.11e-3 | 2.07e-3 / 3.18e-3 / 5.11e-3 |
| kv4 (h64) | 1.91e-3 / 4.78e-3 / 5.65e-3 | 2.17e-3 / 3.82e-3 / 5.65e-3 |
| kv1 (h64) | 1.97e-3 / 7.59e-3 / 1.06e-2 | 2.14e-3 / 6.45e-3 / 1.06e-2 |

**bf16**（容差 ~1e-1）：

| shape | fa vs ref (dq/dk/dv) | te vs ref (dq/dk/dv) |
|---|---|---|
| kv8 | 1.23e-2 / 2.49e-2 / 3.41e-2 | 1.30e-2 / 2.49e-2 / 3.41e-2 |
| kv4 (h32) | 1.16e-2 / 3.11e-2 / 3.63e-2 | 1.24e-2 / 3.14e-2 / 3.63e-2 |
| kv4 (h64) | 1.71e-2 / 3.54e-2 / 6.51e-2 | 1.71e-2 / 3.53e-2 / 6.51e-2 |
| kv1 (h64) | 1.79e-2 / 4.93e-2 / 8.39e-2 | 1.30e-2 / 6.02e-2 / 8.39e-2 |

**fp8**（TE，容差 O(1)，见 `03` §2）：

| shape | te vs ref (o/dq/dk/dv) |
|---|---|
| kv8 | 2.23e-1 / 8.45e-1 / 6.62e-1 / 1.11 |
| kv4 (h32) | 2.68e-1 / 5.25e-1 / 7.66e-1 / 1.34 |
| kv4 (h64) | 2.08e-1 / 3.98e-1 / 1.01 / 1.84 |
| kv1 (h64) | 2.36e-1 / 4.10e-1 / 2.22 / 2.60 |

> 观察：**GQA/MQA 的 kv 头越少，dk/dv 的误差越大**（kv=1 时 dv 误差 ~2.6，明显高于 kv=8 的 1.11）。
> 原因是每个 KV 头要承载 `H/Hkv` 个 query 头的梯度，FP8 量化误差在更多累加项上叠加。

### 7.2 性能对标（CUPTI 纯 device 时间，bwd FLOPs=4·B·S·H·S·(D+Dv)）

**fp16（峰值 989 TFLOPS）**

| shape | FA 2.7.4 | TE 2.14 |
|---|---|---|
| kv8 | 0.2634 ms / 163.07 TF | 0.1697 ms / 253.12 TF |
| kv4 (h32) | 0.2225 ms / 154.45 TF | 0.1430 ms / 240.30 TF |
| kv4 (h64) | 0.3679 ms / 186.78 TF | 0.2462 ms / 279.10 TF |
| kv1 (h64) | 0.3675 ms / 186.99 TF | 0.2682 ms / 256.26 TF |

**bf16（峰值 989 TFLOPS）**

| shape | FA 2.7.4 | TE 2.14 |
|---|---|---|
| kv8 | 0.2639 ms / 162.77 TF | 0.1702 ms / 252.35 TF |
| kv4 (h32) | 0.2227 ms / 154.30 TF | 0.1432 ms / 239.96 TF |
| kv4 (h64) | 0.3686 ms / 186.41 TF | 0.2463 ms / 278.99 TF |
| kv1 (h64) | 0.3680 ms / 186.74 TF | 0.2675 ms / 256.87 TF |

**fp8（TE，峰值 1978.8 TFLOPS；FA 无反向 FP8）**

| shape | TE FP8 |
|---|---|
| kv8 | 0.2425 ms / 177.09 TF |
| kv4 (h32) | 0.2020 ms / 170.10 TF |
| kv4 (h64) | 0.3639 ms / 188.86 TF |
| kv1 (h64) | 0.4032 ms / 170.44 TF |

> TE 在 GQA/MQA 上稳定领先 FA 约 **1.4–1.6×**（fp16/bf16）；fp8 因每个 KV 头被更多 Q 头共享，
> TFLOPS 反而低于同 shape 的 fp16/bf16（kv1: 170 vs 256），是**访存/调度**而非算力问题。

### 7.3 MLA（head_dim=512）：FA / TE 均不支持反向

三个 MLA 形状下：
- `flash_attn` 报 **`FlashAttention forward only supports head dimension at most 256`**；
- `TE fused_attn` 报 `Invalid combination of data type and sequence length`（训练口径反向不支持 head_dim=512，
  与 `te-perf/results_summary.md` 一致：TE bwd 训练最大 head_dim=256，MLA qk≤192/v≤128 附近）。

因此这三个形状只有 fp32 ref（输入与 `ref_dq/dk/dv` 已 dump），**没有 FA/TE 基线**。

**P5-1/P5-2 之后 ours(fp16) 已支持 GQA/MQA 与 head_dim=512**（`HD`/`BM` 模板化，`docs/01` §9）：

| MLA case (B1, D=Dv=512, causal, fp16) | ours-vs-ref dq/dk/dv max_abs | ours preprocess/main/total | TFLOPS | 峰值占比 |
|---|---|---|---|---|
| S=256 H=2 | 1.638 / 1.582 / 1.753e-3 | 0.211 / 1.058 / 1.278 ms | 0.21 | 0.02% |
| S=512 H=4 | 2.324 / 2.916 / 1.724e-3 | 1.291 / 2.080 / 3.491 ms | 0.62 | 0.06% |
| S=1024 H=2 | 1.250 / 1.454 / 2.058e-3 | 2.463 / 4.143 / 6.789 ms | 0.63 | 0.06% |

数值均在 fp16 噪声量级；ncu bound = smem bank conflict + 1 CTA/SM 低 occupancy（135KB smem）。
后续待办看 ROADMAP「下一步」：bf16/fp8 复用同改造、MLA 优化（冲 2 CTA/SM / 张量核 / 对标 FlashMLA）。

### 7.4 ours 的 FP8 GQA/MQA（P5-3/P5-4）

FP8 反向（`src/fp8/` 单文件 + 两文件）现支持 GQA/MQA（`Hkv` 由 `k.npy` shape[2] 读出，
映射 `hkv=h/(H/Hkv)`，`docs/03` §13）。四个形状（B=1 S=1024 D=128 causal）**ours vs fp32 ref**
（max_abs dq/dk/dv）与同 session **TE FP8 vs ref**、以及性能：

| case | ours vs ref | TE vs ref | ours total | ours TF | TE ms | TE TF | ours/TE |
|---|---|---|---|---|---|---|---|
| h40 kv8 | 2.87e-1 / 5.39e-1 / 7.11e-1 | 8.45e-1 / 6.63e-1 / 1.11 | 1.635 ms | 13.1 | 0.2428 | 176.9 | 7.4% |
| h32 kv4 | 2.52e-1 / 5.41e-1 / 7.07e-1 | 5.25e-1 / 7.66e-1 / 1.34 | 1.365 ms | 12.6 | 0.2010 | 170.9 | 7.4% |
| h64 kv4 | 2.76e-1 / 8.46e-1 / 1.23 | 3.98e-1 / 1.01 / 1.84 | 2.369 ms | 14.5 | 0.3614 | 190.1 | 7.6% |
| h64 kv1 (MQA) | 4.10e-1 / 1.52 / 2.13 | 4.10e-1 / 2.22 / 2.60 | 2.293 ms | 15.0 | 0.4009 | 171.4 | 8.7% |

数值与 TE 同量级（fp8 噪声 O(1)，峰值 1978.8 TF 的 ~0.7%）；MHA 回归逐位不变，单/两文件逐位一致。
ncu（h32 kv4）：L1/TEX 70.5% / DRAM 0.9% / Compute 13.5% / occ 15.8% / short_scoreboard 主导
⇒ bound 与 MHA fp8 一致（smem 依赖 + L1/TEX）。详见 `docs/03` §13。
