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
- **shape 简写**：下表/正文里的 `S=512 H16` 等一律指 `(B,S,H,D)=(1,512,16,128)`、`causal`、MHA（D=128）；`S=1024 H32`=`(1,1024,32,128)`；`S=4096 H16`=`(1,4096,16,128)`；MLA 为 `head_dim=512`（§7）。所有 shape 的完整 slug 见 `/home/xieminglin/proj/output/fa-bwd/`。
- **FLOPs 口径**：`4·B·S²·H·D`（反向 ≈ 前向 2×）。**causal 未折半**，故所有实现的 TFLOPS
  都是下界；ours / FA / TE 用同一口径，横向可比。
- **峰值**（H100 dense）：FP16/BF16 Tensor Core ~989 TFLOPS；FP8 ~1978.8 TFLOPS。
- **单位**：性能**主指标是时间**（ms / µs，越小越好）；`TF`/`TFLOPS` = 每秒 Tera（10¹²）次浮点运算，是**吞吐/算力**单位、**不是时间**，由 `FLOPs ÷ 时间` 换算，仅作参考。
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
| (1,512,16,128) | **ours vs ref** | 1.671e-3 | 1.680e-3 | 1.899e-3 |
| | FA vs ref | 1.679e-3 | 1.684e-3 | 1.899e-3 |
| | TE vs ref | 1.716e-3 | 2.287e-3 | 1.899e-3 |
| (1,4096,16,128) | **ours vs ref** | 1.499e-3 | 1.572e-3 | 2.225e-3 |
| | FA vs ref | 1.883e-3 | 1.734e-3 | 1.966e-3 |
| | TE vs ref | 1.883e-3 | 1.858e-3 | 1.966e-3 |

**结论**：ours 与 FA/TE 同量级（~1.5–2.3e-3），无系统误差，个别指标（S=4096 dq/dk）优于 FA/TE。

### 1.2 bf16（正确性容差 ~1e-2）

| shape | 对比 | dq | dk | dv |
|---|---|---|---|---|
| (1,512,16,128) | **ours vs ref** | 6.892e-3 | 8.110e-3 | 1.365e-2 |
| | FA vs ref | 1.040e-2 | 1.261e-2 | 1.365e-2 |
| | TE vs ref | 1.374e-2 | 1.068e-2 | 1.365e-2 |
| (1,4096,16,128) | **ours vs ref** | 8.895e-3 | 8.078e-3 | 1.494e-2 |
| | FA vs ref | 1.441e-2 | 1.332e-2 | 1.631e-2 |
| | TE vs ref | 1.524e-2 | 1.763e-2 | 1.631e-2 |

**结论**：ours 同量级（~7e-3–1.5e-2），dq/dk 略优于 FA/TE，dv 三者几乎一致（受 bf16 尾数限制）。

### 1.3 fp8（E4M3/E5M2 rowwise；容差 O(1)，见 `03` §2）

| shape | 对比 | dq | dk | dv |
|---|---|---|---|---|
| (1,512,16,128) | **ours vs ref** | 2.426e-1 | 2.975e-1 | 3.735e-1 |
| | TE vs ref | 4.859e-1 | 4.038e-1 | 5.906e-1 |
| | ours vs TE | 5.381e-1 | 4.429e-1 | 8.546e-1 |
| (1,1024,32,128) | **ours vs ref** | 2.400e-1 | 4.195e-1 | 3.536e-1 |
| | TE vs ref | 4.360e-1 | 4.500e-1 | 8.560e-1 |
| | ours vs TE | 4.652e-1 | 5.701e-1 | 9.431e-1 |
| (1,4096,16,128) | **ours vs ref** | 2.635e-1 | 2.643e-1 | 3.216e-1 |
| | TE vs ref | 3.763e-1 | 3.686e-1 | 6.688e-1 |
| | ours vs TE | 4.525e-1 | 5.324e-1 | 6.807e-1 |

**结论**：ours 与 fp32 ref、与 TE 的偏差都在 **同一量级（~0.24–0.94）**，无系统误差；
误差来自 FP8 量化本身（尾数 2–3 位），与 fp32 累加无关。**S=4096 时 ours-vs-ref 全面优于
TE-vs-ref**（ours 0.24–0.32 vs TE 0.37–0.67）——本版 dS/输出保留 fp32、dP 不进张量核，
口径更保守。`max_rel` 会被近零元素放大，不作判据（ref 梯度 amax 仅 3–6）。

---

## 2. 性能对标（CUPTI / event，causal）

`ours total` = 端到端（fp8 含 quant）；`ours main` = 反向主 kernel。峰值占比括号内。

> **口径提醒**：本节的 FA/TE 数字来自 `harness/fa_bwd_bench.py`，其 lambda **包含 forward**（fwd+bwd 合计）。
> 若要与 `te-perf` 的纯 `fused_attn_bwd` 对齐，请看 `docs/06` §1 的**纯反向**口径
> （`harness/fa_vs_te_bwd_only.py`）：纯反向下 FA2.7.4 为 217–377 TF、TE2.14 为 305–618 TF
> （TE 快 1.2–1.6×），FA 慢的根因（FA2=SM80 kernel vs TE=SM90 wgmma）见 `docs/06` §3。

### 2.1 fp16（峰值 989 TFLOPS）

| shape | ours total | ours main | FA2.7.4 | TE2.14 |
|---|---|---|---|---|
| (1,512,16,128) | 3.61 ms / 0.60 TF (0.06%) | 2.38 ms | 0.0703 ms / 30.54 TF (3.09%) | 0.0457 ms / 46.97 TF (4.75%) |
| (1,1024,32,128) | — | — | 0.2238 ms / 76.78 TF (7.76%) | 0.1372 ms / 125.20 TF (12.66%) |
| (1,4096,16,128) | 138.33 ms / 0.99 TF (0.10%) | 68.57 ms | 1.0582 ms / 129.88 TF (13.13%) | 0.5968 ms / 230.31 TF (23.29%) |

> 上表 fp16 是**标量 golden**（`fa_bwd_fp16_{main,onefile}.cu`）。**O5** 新增张量核版
> `fa_bwd_fp16_mma_{onefile.cu, kernels.cuh+main.cu}`（`mma.m16n8k16`+`ldmatrix`），
> 同 session A/B：**main S=512 2.279→0.192ms（11.8×）、S=4096 67.55→4.555ms（14.9×）**
> （main-only 22.4 / 60.4 TF，峰值 2.3%/6.1%），数值与 ref/FA/TE 同量级、单/两文件逐位一致。
> 端到端仍被**标量 preprocess** 拖住（S=4096 preprocess 68.7ms > main 4.56ms，占 94%）→ 下一项 **O8**。
> 详见 `01-fp16-bwd-impl.md` §10。

> **O8（preprocess mma 分块 LSE）已完成**：标量 `preprocess_kernel` → `lse_mma_kernel<128>`
> （`mma.m16n8k16` `QKᵀ` + 累加器 online-softmax + 4-lane `shfl` 归约）+ 独立 `delta_kernel`。
> **preprocess S512 1.209→0.071 ms（17.0×）、S4096 68.70→0.986 ms（69.7×）**；端到端
> **total S512 1.440→0.326 ms（4.4×）、S4096 73.34→5.581 ms（13.1×，24.63 TF，峰值 2.49%）**。
> 同 session 纯反向 FA3 S4096 **0.3246ms/847TF**、TE 0.4441/619 ⇒ ours total 为 FA3 的
> **~2.9%（TFLOPS；O5 时 ~0.4%）**。数值与 O5 **逐位相同**。ncu（lse,S4096）= 访存延迟 +
> 低 occ（Compute 39.5%、DRAM 1.1%、occ 28.5%、Waves 1.29）。详见 `01-fp16-bwd-impl.md` §11。

> **O6（main 的 `cp.async` 双缓冲）已完成**：K/V 用 `cp.async.cg`（16B unit）双缓冲，
> 下一 tile 的全局读延迟被本轮 5 个 GEMM 覆盖，并省掉 1 个 `__syncthreads`/tile。
> 同 session A/B **main S512 0.1912→0.0847ms（2.26×）、S4096 4.510→1.873ms（2.41×）、
> GQA kv4 0.635→0.375ms（1.69×）**；端到端 **total S4096 5.581→3.043ms（45.16 TF，峰值 4.57%）**、
> S512 0.326→0.187ms。同 session 纯反向 FA3 S4096 0.3253ms/845TF ⇒ ours total 为 FA3 的
> **5.4%**（时间比 9.3×；O8 时 2.9%/13–17×）。数值与 O5/O8 **逐位相同**（只改搬运）。
> ncu（main,S4096）：Duration 4.55→1.95ms、**L1/TEX 65.2% / L2 60.1% / Compute 27.1% /
> 182 regs / 2 CTA/SM（smem 84KB）/ Waves 3.88**，**stall `long_scoreboard` 7.35→1.12**
> ⇒ 新墙 = `wait`（fixed-latency）+ `short_scoreboard`（smem→ldmatrix）+ L1/TEX。
> 详见 `01-fp16-bwd-impl.md` §12。

> **O6b（K/V 降 smem 回 3 CTA/SM + `ldmatrix.x4.trans` 消转置副本）已完成**：
> ① GEMM3/GEMM4 的 A 改 `ldmatrix.x4.trans` 从 `Ps/dSs[BM][BN]` 直读，删掉 `PsT/dSsT`
> （smoke `fa_bwd_fp16_atrans_smoke.cu` 验证逐位）；② 只双缓冲 K、V 单缓冲且在 GEMM2 后预取。
> **smem 83.97→71.17KB、Block Limit Shared Mem 2→3、occ 11.8%→16.9%**。同 session A/B main
> **S4096 1.900→1.860ms、GQA kv4 0.380→0.357ms**；S512（grid=128 单波）O6 反快，故 host 按
> 网格自动选（`≥396` 用 O6b）。端到端 **total S4096 2.952ms（46.56 TF）、GQA kv4 0.572ms（30.0 TF）**；
> 同 session FA3 S4096 0.3248ms/846 ⇒ ours total 为 FA3 的 **5.5%**、GQA kv4 为 7.2%；
> 数值与 O5/O8/O6 **逐位相同**。ncu（main,S4096）L1/TEX 71.9% / L2 63.1% / Compute 29.8% /
> 168 regs / 3 CTA/SM ⇒ **墙 = L1/L2 吞吐 + `wait`**。详见 `01-fp16-bwd-impl.md` §12b。

> **O8b（LSE 预处理负载均衡 + `cp.async` 双缓冲）已完成（fp16）**：O6b 后 preprocess 占端到端
> **34%**。① 因果下第 `mblk` 个 CTA 做 `mblk+1` 个 K tile，重块排最后（ncu 尾波 50%）⇒
> **镜像配对**：每 CTA 处理 `m` 与 `nblk-1-m`，工作量恒为 `nblk+1`（完美均衡）。
> ② K 改 `cp.async.cg` 16B 双缓冲，消掉每 tile 的同步读延迟。**lse S4096 0.985→0.348ms（2.81×）**、
> S512 1.38×、GQA kv4 2.09×；消融：镜像配对贡献 2.20×，cp.async 再叠加 1.28×。
> 端到端 **total S4096 2.952→2.336ms（58.8 TF，峰值 5.9%）**、GQA kv4 0.572→0.518ms；
> 同 session FA3 S4096 0.3241ms/848 ⇒ ours total 为 FA3 的 **6.9%**（O6b 5.5%）；数值逐位相同。
> ncu（lse,S4096）：Duration 1.03ms→354µs、`long_scoreboard` 2.19→0.34、Waves 1.29→0.97（尾波消除），
> 新墙 = Compute 60% + smem 依赖。详见 `01-fp16-bwd-impl.md` §13。

### 2.2 bf16（峰值 989 TFLOPS）

| shape | ours total | ours main | FA2.7.4 | TE2.14 |
|---|---|---|---|---|
| (1,512,16,128) | 3.12 ms / 0.69 TF (0.07%) | 1.88 ms | 0.0691 ms / 31.10 TF (3.14%) | 0.0455 ms / 47.24 TF (4.78%) |
| (1,1024,32,128) | — | — | 0.2212 ms / 77.68 TF (7.85%) | 0.1364 ms / 125.94 TF (12.73%) |
| (1,4096,16,128) | 111.33 ms / 1.23 TF (0.12%) | 41.91 ms | 1.0478 ms / 131.17 TF (13.26%) | 0.5918 ms / 232.23 TF (23.48%) |

> bf16 的 K/V smem 行距 +2 padding 已消除 9.5-way bank conflict，S=4096 main 从 197→42 ms
> （4.7×），现已快过同款 fp16 main；**端到端被 preprocess 拖住**（S=4096 69.4 ms > main 41.9 ms）。

**O5b：bf16 main 换张量核 `mma.m16n8k16` + `ldmatrix`**（单/两文件 `fa_bwd_bf16_mma_*`，第 6e 节）：

| shape | ours total | ours main | main TFLOPS | FA3 (SM90) | TE2.14 | FA2.7.4 |
|---|---|---|---|---|---|---|
| (1,512,16,128) | 1.434 ms | **0.190 ms** | 22.6 (2.3%) | — | — | — |
| (1,4096,16,128) | 73.43 ms | **4.511 ms** | 60.9 (6.2%) | 860 | 622 | 379 |
| (1,1024,40,128) kv8 | 11.94 ms | **0.871 ms** | 49.3 (5.0%) | 356 | 326 | 229 |
| (1,1024,32,128) kv4 | 9.490 ms | **0.639 ms** | 53.8 (5.4%) | 418 | 307 | 217 |
| (1,1024,64,128) kv4 | 18.78 ms | **1.119 ms** | 61.4 (6.2%) | 431 | 356 | 257 |
| (1,1024,64,128) kv1 | 18.62 ms | **1.009 ms** | 68.1 (6.9%) | 440 | 322 | 259 |

> main 相对标量 bf16 golden **9.4–9.9×**（S=512 1.88→0.190、S=4096 42.2→4.51 ms），
> 数值与 FA/TE 同为 bf16 噪声量级。main-only 到 FA3 的 5.7–15.5%。
> **端到端仍被标量 preprocess 拖住**（S=4096 preprocess 68.8ms 占 94%）——即下一项 **O8**。
> ncu（main, S=4096）：DRAM 1.41% / L1TEX 33.24% / L2 24.84% / Compute 17.33% / occ 18.75% /
> 168 regs / 66.56KB / 3 CTA/SM；**`long_scoreboard` 63%** ⇒ bound = 全局访存延迟（无 cp.async/预取）。
> FA2/FA3/TE 列为 `harness/fa_vs_te_bwd_only.py bf16` 纯反向 CUPTI 口径（`04` §7.2 同源）。

> **O8（preprocess mma 分块 LSE）已完成（与 fp16 逐字同构）**：**preprocess S512 1.201→0.072 ms
> （16.7×）、S4096 68.79→0.993 ms（69.3×）**；端到端 **total S512 1.434→0.322 ms（4.45×）、
> S4096 73.43→5.573 ms（13.2×，24.66 TF，峰值 2.49%）**。同 session 纯反向 FA3 S4096
> **0.3217ms/854TF**、TE 0.4415/623 ⇒ ours total 为 FA3 的 ~2.9%。数值与 O5b 逐位相同；
> ncu（lse,S4096）与 fp16 逐项一致。详见 `01b-bf16-bwd-impl.md` §6f。

> **O6（main 的 `cp.async` 双缓冲，与 fp16 逐字同构）已完成**：**main S512 0.1896→0.0837ms
> （2.27×）、S4096 4.485→1.872ms（2.40×）、GQA kv4 0.638→0.377ms（1.69×）**；端到端
> **total S4096 5.573→3.036ms（45.27 TF，峰值 4.58%）**、S512 0.183ms。同 session 纯反向
> FA3 S4096 0.3210ms/856TF ⇒ ours total 为 FA3 的 **5.3%**。数值与 O5b/O8 **逐位相同**。
> ncu（main,S4096）：Duration 1.88ms、L1TEX 64.96% / L2 62.19% / Compute 25.49% /
> 182 regs / 2 CTA/SM，`long_scoreboard` 63%→~1 成、新墙同 fp16。详见 `01b-bf16-bwd-impl.md` §6g。

> **O6b（与 fp16 逐字同构）已完成**：K/V 降 smem 回 3 CTA/SM + A 转置读，smem 83.97→71.17KB、
> occ 12.5%→18.75%。同 session A/B main **S4096 1.894→1.864ms、GQA kv4 0.381→0.358ms**
> （S512 单波走 O6）；端到端 **total S4096 2.961ms（46.42 TF）、GQA kv4 0.576ms**，
> 同 session FA3 S4096 0.3205ms/858 ⇒ ours 为 FA3 的 **5.4%**；数值逐位相同。
> 详见 `01b-bf16-bwd-impl.md` §6h。

### 2.3 fp8（峰值 1978.8 TFLOPS；FA 无反向 FP8，仅对标 TE）

| shape | ours total | ours main | TE FP8 |
|---|---|---|---|
| (1,512,16,128) | 1.69 ms / 1.27 TF (0.06%) | 0.450 ms | 0.0724 ms / 29.67 TF (1.50%) |
| (1,1024,32,128) | 10.88 ms / 1.58 TF (0.08%) | 1.827 ms | 0.1377 ms / 124.80 TF (6.31%) |
| (1,4096,16,128) | 80.63 ms / 1.70 TF (0.09%) | 10.22 ms | 0.4544 ms / 302.48 TF (15.29%) |

- fp8 **golden（标量）→ mma 张量核**：main 提速 **13–20×**（S=512 5.90→0.45 ms；
  S=4096 198.5→10.2 ms），数值不降反升（见 `03` §7.4）。
- main 相对 FP8 峰值仍只 0.24–0.68%、相对 TE FP8 约 4–16%（低 occupancy、无流水、每 tile 原子累加）。
- **端到端瓶颈已转移到 preprocess**（LSE 的 O(S²) 点积未分块）：S=4096 71.0 ms vs main 10.2 ms。

> **下表为 O1/O2 阶段值（第十二轮）**，见 `03` §9（O1）与 §10（O2）；最新值见本节末尾。
> fp16/bf16 两表自 P4-1 起未变。

| shape | ours total（O2） | ours main（O2） | TE FP8（同 session） |
|---|---|---|---|
| (1,512,16,128) | 0.621 ms / 3.46 TF (0.17%) | 0.461 ms | 0.0724 ms / 29.67 TF |
| (1,1024,32,128) | 1.921 ms / 8.94 TF (0.45%) | 1.526 ms | 0.1371 ms / 125.35 TF |
| (1,4096,16,128) | 10.067 ms / 13.65 TF (0.69%) | 8.562 ms | 0.4550 ms / 302.08 TF |

- O1：`preprocess` 改 mma 分块 LSE + 独立 delta_kernel（14–62×），S=4096 端到端瓶颈回落 main。
- O2：`dS3/Ap` 折叠进 `Ks/Vs` 死空间，smem 80.13→75.01 KB → **3 CTA/SM**；main 1.16–1.19×。

> **下表为 O1–O4d 全部优化后的最新值（本轮 O2b+O4d）**；逐项见 `03` §9–§15。
> `ours` 为同 session 实测，`TE` 为 CUPTI 端到端（FA 无反向 FP8）。fp16/bf16 两表自 P4-1 起未变。

| shape | ours total | ours main | main TF（峰值占比） | TE FP8（同 session） |
|---|---|---|---|---|
| (1,512,16,128) | 0.2916 ms / 7.36 TF | 0.1439 ms | 14.9（0.75%） | 0.1009 ms / 42.6 TF |
| (1,1024,32,128) | 1.2000 ms / 14.32 TF | 0.8084 ms | 21.3（1.07%） | 0.2061 ms / 166.8 TF |
| (1,4096,16,128) | 6.5265 ms / 21.06 TF | 4.9084 ms | 28.0（1.42%） | 0.5903 ms / 465.6 TF |

- O3：K/V 向量化 4B 读 + 寄存器双缓冲（main 1.15–1.18×）；O4a：消 GEMM1/2 barrier +
  fold 全线程并行（main 1.21–1.54×）。
- **O2b：split-K 自动切块调参**——d128 目标 grid≈4096、MLA 目标≈132（1 个波），
  相对旧自动档 main 1.08–1.21×、MLA 小 S 1.56×。
- **O4d：`Ps/Ss` 行距 `BN→BN+1`** 消 bank conflict（`op_ld` −61%、总 wavefronts −26%），
  main 1.05–1.12×；smem 73.2→73.8 KB（仍 3 CTA/SM）。
- 数值与 P3-5/O1–O4a **逐位相同**；端到端仍受 main 主导（S=4096 main 4.91/6.53 ms）。

> **下表为 O4c（第二十三轮）最新值**：把 dQ/dK/dV 的 `atomicAdd` 向量化为 `float2` red，
> red 请求/L2 扇区各 **0.50×**，**L2 81.5%→57.9%**（墙被打掉，新墙 L1/TEX 81.3%）。
> `TE` 为同 session CUPTI 端到端。逐项见 `03` §16。

| shape | ours total | ours main | main TF（峰值占比） | TE FP8（同 session） | main ours/TE |
|---|---|---|---|---|---|
| (1,512,16,128) | 0.2556 ms / 8.40 TF (0.42%) | 0.1090 ms | 19.7（1.00%） | 0.1014 ms / 42.4 TF | **93%** |
| (1,1024,32,128) | 1.0590 ms / 16.22 TF (0.82%) | 0.6772 ms | 25.4（1.28%） | 0.2059 ms / 166.9 TF | 30% |
| (1,4096,16,128) | 5.1601 ms / 26.64 TF (1.35%) | 3.5330 ms | 38.9（1.97%） | 0.5894 ms / 466.4 TF | 17% |
| MLA (1,1024,2,512) | 0.8515 ms / 5.04 TF | 0.3933 ms | 10.9（0.55%） | NA（FA/TE 不支持） | — |

- 数值与 O2b+O4d **逐位相同**；S=512 的 **main 已几乎追平 TE 整条反向**（0.1090 vs 0.1014 ms）。

> **下表为 O4b（第二十四轮）最新值**：fp8 `Kt/Qt/dOt` 三个逐字节转置副本 → **K 配对布局 +
> `ldmatrix.x2.trans`**（先在 `trans_smoke` 验证与 `ldmatrix.x2` 逐位一致）。smem 75520→70656 B
> （d128）、229120→205824 B（MLA）；`op_st` bank conflict **206.4M→69.4M（−66%）**、
> L1/TEX **81.3%→69.7%**、short_scoreboard 3.50→2.73。**数值与 O4c 逐位相同**。逐项见 `03` §17。

| shape | ksplit | ours total | ours main | main 加速 | main TF（峰值占比） | TE FP8（同 session） | main ours/TE |
|---|---|---|---|---|---|---|---|
| d128 (1,512,16,128) | 16 | 0.2190 ms / 9.80 TF | **0.0733 ms** | **1.49×** | 29.3（1.48%） | 0.1003 ms / 42.8 TF | **73%** |
| d128 (1,1024,32,128) | 8 | 0.8574 ms / 20.04 TF | **0.4552 ms** | **1.49×** | 37.7（1.91%） | 0.2049 ms / 167.7 TF | 222% |
| d128 (1,4096,16,128) | 4 | 4.6162 ms / 29.77 TF | **2.9473 ms** | **1.17×** | 46.6（2.36%） | 0.5847 ms / 470.1 TF | 504% |
| MLA (1,1024,2,512) | 4 | 0.7906 ms / 5.43 TF | **0.3251 ms** | **1.20×** | 13.2（0.67%） | NA（FA/TE 不支持） | — |
| MLA (1,256,2,512) | 16 | 0.1909 ms / 1.41 TF | **0.0522 ms** | **1.76×** | 5.14（0.26%） | NA | — |
| GQA h32kv4 (1,1024,32,128) | 8 | 0.7528 ms / 22.82 TF | **0.4404 ms** | **1.34×** | 39.0（1.97%） | 0.2013 ms / 170.7 TF | 219% |

- **矫正 O4b 前提**：fp8 里 A/B 主序相反、且 `ldmatrix.trans` 的配对方向是 N，
  转置副本**无法整个消掉**（详见 `03` §17.1）。真实收益来自「逐字节 scatter 写 → 4B 交织写」
  与更小的配对数组（无 `+16` 行距放大）。

> **下表为 O7（第二十五轮）最新值**：dQ 沿 nt 循环在**寄存器**里累加（折算后），每 CTA 只 flush
> 一次跨 CTA `atomicAdd`（dQ 的归约指令数 `O(ntiles)→O(1)`）。残余 red 再砍半
> （L1 red 17.0M→9.0M、L2 red 204.5M→108.5M），**L2 墙 69.1%→43.8%**；用
> `__launch_bounds__(128,3)` 把 254→168 regs、保住 3 CTA/SM。**数值与 O4b 逐位相同**；
> 主收益在 S=4096（REGDQ=true），其余 shape 走原路径。逐项见 `03` §18。

| shape | ksplit | ours total | ours main | main 加速 | main TF（峰值占比） | TE FP8（同 session） | main ours/TE |
|---|---|---|---|---|---|---|---|
| d128 (1,512,16,128) | 16 | 0.2191 ms / 9.80 TF | **0.0740 ms** | ~1.00× | 29.0（1.47%） | 0.1006 ms / 42.7 TF | 74% |
| d128 (1,1024,32,128) | 8 | 0.8648 ms / 19.86 TF | **0.4604 ms** | ~0.99× | 37.3（1.89%） | 0.2054 ms / 167.3 TF | 224% |
| d128 (1,4096,16,128) | 4 | 4.3410 ms / 31.66 TF | **2.6736 ms** | **1.10×** | 51.4（2.60%） | 0.5863 ms / 468.8 TF | 456% |
| GQA h32kv4 (1,1024,32,128) | 8 | 0.7569 ms / 22.70 TF | **0.4463 ms** | ~0.99× | 38.5（1.95%） | 0.2014 ms / 170.6 TF | 222% |
| MLA (1,1024,2,512) | 4 | 0.7910 ms / 5.43 TF | **0.3277 ms** | ~0.99× | 13.1（0.66%） | NA（FA/TE 不支持） | — |

- **O7 只改 dQ 的归约方式**（dK/dV 的跨 mblk/hkv 竞争未动，仍是残余 red 的大头），
  端到端 ours/TE FP8 S=4096 = **7.41×**（O4b 7.90×）；S=512 的 main 仍是 TE 整条反向的 ~74%。

---

## 3. ncu bound 小结（逐 dtype）

| dtype | kernel | DRAM | L1/TEX | Compute | Occ | Waves | 主 stall | bound 结论 |
|---|---|---|---|---|---|---|---|---|
| fp16 | main (S=512) | 0.21% | **53.5%**（75% 多余 wavefront） | 8.45% | 6.25%（96KB） | 0.48 | — | **smem 访问 + 低 occupancy** |
| bf16 | main (padding 后) | 0.26% | 24.9% | 13.5% | 6.25% | 0.48 | fixed-latency 37.3%、No Eligible 78% | **延迟 / 并行度不足** |
| fp16 | **mma main（O5, S=4096）** | 1.41% | 33.32% | 17.44% | 16.58%（66.56KB, 3 CTA/SM） | 2.59 | long_scoreboard 63.4% | **全局访存延迟**（无 cp.async/预取） |
| bf16 | **mma main（O5b, S=4096）** | 1.41% | 33.24% | 17.33% | 16.40% | 2.59 | long_scoreboard 63% | 同上（与 fp16 逐项一致） |
| fp16 | **lse_mma（O8, S=4096）** | 1.06% | 23.42% | 39.53% | 28.49%（80 regs） | 1.29 | long_scoreboard 2.17、wait 1.34、No Eligible 42.9% | **全局访存延迟 + 低 occupancy/尾波** |
| bf16 | **lse_mma（O8, S=4096）** | 1.13% | 23.37% | 44.70% | 28.41% | 1.29 | long_scoreboard（同 fp16） | 同 fp16 |
| fp16 | **lse_mma_bal<128,1>（O8b, S=4096）** | 3.07% | 32.36% | **60.43%** | 22.89%（52.2KB, 4 CTA/SM, 64 regs） | **0.97** | long 0.34、wait 1.57、short 1.33 | **Compute 60% + smem 依赖**（镜像配对消尾波、cp.async 消 long_scoreboard） |
| fp16 | **delta（O8, S=4096）** | 24.80% | 73.50% | 71.91% | 71.90%（17 regs） | 31.03 | — | 访存/算力均衡的轻量归约（<1% 端到端） |
| fp16 | **mma main（O6, S=4096, cp.async 双缓冲）** | 3.31% | 65.19% | 27.07% | 11.83%（**83.97KB, 2 CTA/SM**, 182 regs） | 3.88 | **long_scoreboard 7.35→1.12**；wait 1.95、short_scoreboard 0.79、barrier 0.10 | **fixed-latency(`wait`) + short_scoreboard(smem→ldmatrix) + L1/TEX**（全局访存延迟已被 cp.async 消掉） |
| bf16 | **mma main（O6, S=4096, cp.async 双缓冲）** | 3.43% | 64.96% | 25.49% | 11.79%（83.97KB, 2 CTA/SM） | 3.88 | long_scoreboard 同上降到 ~1、wait 主导 | 同 fp16（与 fp16 逐项一致） |
| fp16 | **mma main（O6b, S=4096, K 双缓冲+A 转置读）** | 3.47% | 71.87% | 29.77% | 16.90%（**71.17KB, 3 CTA/SM**, 168 regs） | 2.59 | wait 1.88、long_scoreboard 1.79、short 0.81、not_selected 0.37 | **L1/TEX 吞吐 + L2 吞吐 + fixed-latency(`wait`)**（occ 升但吞吐受限） |
| bf16 | **mma main（O6b, S=4096, K 双缓冲+A 转置读）** | 3.32% | 71.71% | 31.42% | 16.86%（71.17KB, 3 CTA/SM） | 2.59 | 同 fp16（与 fp16 逐项一致） | 同 fp16 |
| fp8 | golden main | 0.06% | **75.96%**（90% 多余） | 4.98% | 6.25%（68KB） | 0.32 | MIO scoreboard 69% | **smem 冲突 + FP8 解码 + 低 occ** |
| fp8 | **mma main（O2b+O4d 后, S=4096, ksplit=4）** | 1.41% | 69.91% | 21.70% | **18.27%（73.8KB, 3 CTA/SM）** | 10.34 | No Eligible 76.6%、long_scoreboard 4.46 + short_scoreboard 3.96 | **L2 带宽（81.5%）+ 延迟**（split-K 复读 Q/dO + 全局 atomic） |
| fp8 | **mma main（O4c 后, S=4096, ksplit=4）** | 1.99% | **81.30%** | 29.42% | 18.20%（73.8KB, 3 CTA/SM） | 10.34 | short_scoreboard 3.50、long_scoreboard 1.44 | **L1/TEX 81.3% + short_scoreboard**（全局 red 流量已减半，L2 退到 57.9%） |
| fp8 | **mma main（O4b 后, S=4096, ksplit=4）** | 2.46% | **69.69%** | 34.40% | 18.21%（**70.66KB**, 3 CTA/SM） | 10.34 | short_scoreboard 2.73、long_scoreboard 1.16 | **L1/TEX 69.7% + L2 69.1%（残余 red）+ short_scoreboard**（`op_st` 冲突 −66%） |
| fp8 | **mma main（O7 后, S=4096, ksplit=4, REGDQ=true）** | 2.60% | **64.4%** | 37.7% | 18.11%（70.66KB, 3 CTA/SM） | 10.34 | short_scoreboard 1.89、long_scoreboard 1.09 | **L1/TEX 64.4% + short_scoreboard 1.89 + 残余 L2 43.8%（dK/dV 跨 CTA red）**（dQ red 已 O(1)，L2 墙 69.1%→43.8%） |
| fp8 | **mma main（O4b 后, MLA S=1024 H2 D512）** | 1.46% | 11.38% | 7.45% | **6.25%（205.8KB, 1 CTA/SM）** | 0.97 | long_scoreboard 1.73、wait 1.67 | **低 occupancy/并行度**（smem 仍 205.8KB > 116KB 门槛） |

**共同结论**：三种 dtype 的 **main kernel** 都不是 HBM 或算力 bound（DRAM <3%、Compute <38%）；
真正的墙是**低 occupancy（fp8 已做到 3 CTA/SM）+ 并行度不足 / 全局访存延迟**。
fp16/bf16 的 main 也已换张量核（**O5/O5b**：`mma.m16n8k16`+`ldmatrix`，main 9–15×），
但 main 的新墙是 **`long_scoreboard` 63%（全局读延迟）**；端到端的第一瓶颈则是**标量 preprocess**，
已由 **O8**（fp16/bf16 的 `lse_mma_kernel`+`delta_kernel`，对齐 fp8 O1）打掉：
preprocess 17–70×、端到端 4.4–13.2×，端到端瓶颈回落到 **main**（S4096 占 ~80%）。
fp8 上 mma 后 bank conflict 已消失（L1/TEX 76%→19%），bound 从「smem 冲突」退化为「延迟受限」；
O2 降 smem 后 fp8 从 2→3 CTA/SM（theoretical 12.5%→18.75%），main 1.16–1.19×。
**O4b 完成**：`op_st` bank conflict 206.4M→69.4M、L1/TEX 81.3%→69.7%、main 1.17–1.76×，
但 fp8 的转置副本只能「变小 + 写得更省」而**不能消掉**（`03` §17.1）。
**O7 完成**：dQ 的每-tile 归约折成每-CTA 一次寄存器累加 + 单次 flush（`03` §18），
残余 red **再砍半**（L2 red 204.5M→108.5M）、**L2 墙 69.1%→43.8%**、main S=4096 1.10×，
且用 `__launch_bounds__(128,3)` 保住 3 CTA/SM。**新墙 = L1/TEX 64.4% + short_scoreboard 1.89
+ 残余 L2 43.8%**；剩余 red 全是 dK/dV 的跨 mblk/hkv 竞争，要动并行结构（或分块 `*_accum` + convert）。
下一步优先级：**① fp16/bf16 main 的 `cp.async` 双缓冲流水（O6，消 63% long_scoreboard，
现已是端到端第一瓶颈）；② fp16/bf16 去 atomic（O7，移植 fp8）；③ dK/dV 的跨 CTA 归约
（分块 `*_accum` + convert）；④ MLA 的 KV 分片/降 smem + 张量核；⑤ wgmma/TMA（O9 对标 FA3）。**

> **O6（已做）与 O6b（已做）**把 fp16/bf16 main 从 4.5ms 打到 1.86ms（端到端 S4096 3.0ms）；
> **O8b（fp16 已做）**再把 preprocess 从 1.0ms 打到 0.39ms（镜像配对消尾波 + K 的 cp.async 双缓冲），
> **端到端 S4096 2.95→2.34ms（58.8 TF，FA3 的 6.9%）**，lse 的 `long_scoreboard` 2.19→0.34、
> Waves 1.29→0.97，新墙变为 **Compute 60% + smem 依赖**。O6b 后的墙（L1/TEX 72% + L2 63%）
> 仍是 main 的下一刀：**O7（去 dK/dV atomic）/ O9（wgmma+TMA）**。
> 注：fp16/bf16 的 dK/dV 原子流量与 dQ 对称（无论 Q-resident 还是 KV-resident，归约贡献总数
> 都是 `S²/2`），单靠换归约维收益有限，真正的杠杆是**更大 `BM` 或寄存器累加**——O7 需谨慎设计。

> O4c（`atomicAdd`→`float2` 向量化 red）+ O4b（转置副本 → `ldmatrix.trans` + K 配对）+
> O7（dQ 寄存器累加 + 单次 flush）均已完成。

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
| (1,256,2,512) | 1.638 / 1.582 / 1.753e-3 | 0.211 / 1.058 / 1.278 ms | 0.21 | 0.02% |
| (1,512,4,512) | 2.324 / 2.916 / 1.724e-3 | 1.291 / 2.080 / 3.491 ms | 0.62 | 0.06% |
| (1,1024,2,512) | 1.250 / 1.454 / 2.058e-3 | 2.463 / 4.143 / 6.789 ms | 0.63 | 0.06% |

数值均在 fp16 噪声量级；ncu bound = smem bank conflict + 1 CTA/SM 低 occupancy（135KB smem）。
bf16 的 MLA（同一 `HD/BM` 改造）见 §7.6。后续待办看 ROADMAP「下一步」：fp8 复用同改造、
MLA 优化（冲 2 CTA/SM / 张量核 / 对标 FlashMLA）。

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

### 7.5 ours 的 fp16 / bf16 GQA/MQA（P5-1 / P5-3）

fp16/bf16 反向也已支持 GQA/MQA（`Hkv` 由 `k.npy` shape[2] 读出，映射 `hkv=h/(H/Hkv)`；
MHA 时退化为原式，逐位回归不变）。四个形状均为 **B=1, S=1024, D=128, causal**，
`ours vs fp32 ref`（max_abs dq/dk/dv）与 **FA/TE vs ref** 对照：

| shape | ours(fp16) vs ref | FA vs ref | TE vs ref |
|---|---|---|---|
| (1,1024,40,128) kv=8 | 2.134 / 3.078 / 3.963e-3 | 1.727 / 3.321 / 5.107e-3 | 2.075 / 3.175 / 5.107e-3 |
| (1,1024,32,128) kv=4 | 1.580 / 2.380 / 3.999e-3 | 1.757 / 2.996 / 4.326e-3 | 1.580 / 2.893 / 4.326e-3 |
| (1,1024,64,128) kv=4 | 1.974 / 5.704 / 4.938e-3 | 1.911 / 4.778 / 5.651e-3 | 2.167 / 3.818 / 5.651e-3 |
| (1,1024,64,128) kv=1 | 1.780 / — / — | 1.793 / 4.933 / 8.386e-3 | 1.300 / 6.018 / 8.386e-3 |

| shape | ours(bf16) vs ref | FA vs ref | TE vs ref |
|---|---|---|---|
| (1,1024,40,128) kv=8 | 1.011e-2 / 1.885e-2 / 3.150e-2 | 1.233e-2 / 2.491e-2 / 3.411e-2 | 1.303e-2 / 2.491e-2 / 3.411e-2 |
| (1,1024,32,128) kv=4 | 9.631e-3 / 2.019e-2 / 3.094e-2 | 1.163e-2 / 3.110e-2 / 3.627e-2 | 1.238e-2 / 3.140e-2 / 3.627e-2 |
| (1,1024,64,128) kv=4 | 1.319e-2 / 2.716e-2 / 3.152e-2 | 1.710e-2 / 3.543e-2 / 6.511e-2 | 1.710e-2 / 3.534e-2 / 6.511e-2 |
| (1,1024,64,128) kv=1 | 1.066e-2 / 3.385e-2 / 5.891e-2 | 1.793e-2 / 4.933e-2 / 8.386e-2 | 1.300e-2 / 6.018e-2 / 8.386e-2 |

**结论**：fp16/bf16 的 GQA/MQA 数值与 ref、FA、TE 同量级（fp16 ~1e-3、bf16 ~1e-2），
ours 在 bf16 上 dq/dk/dv 全面优于 FA/TE。**性能**：ours 仍是标量实现（见 §2 与 ROADMAP
「性能差距归因」），如 bf16 kv4(h64) main 10.96 ms / total 28.44 ms（≈1.2 TF，峰值 0.12%），
远低于 FA/TE；张量核化（O5）是下一步最大杠杆。
原始输出：`src/fp16/fa_bwd_fp16_main_p51_gqa.out.txt`、`src/bf16/fa_bwd_bf16_main_p53_gqa.out.txt`。

### 7.6 ours 的 bf16 MLA（head_dim=512，P5-3 续）

bf16 反向复用 fp16 的 `HD/BM` 模板改造（`HD=128→BM=64` 回归逐位不变、`HD=512→BM=16`），
单/两文件同源。FA/TE 反向均不支持 head_dim=512，故只有 fp32 ref 与 ours 性能数字：

| MLA case (B1, D=Dv=512, causal, bf16) | ours-vs-ref dq/dk/dv max_abs | ours preprocess/main/total | TFLOPS | 峰值占比 |
|---|---|---|---|---|
| (1,256,2,512) | 8.240 / 7.905 / 15.04e-3 | 0.213 / 0.748 / 0.972 ms | 0.28 | 0.03% |
| (1,512,4,512) | 10.43 / 10.33 / 13.85e-3 | 1.297 / 1.461 / 2.872 ms | 0.75 | 0.08% |
| (1,1024,2,512) | 5.152 / 7.742 / 15.57e-3 | 2.471 / 2.905 / 5.522 ms | 0.78 | 0.08% |

数值为 bf16 噪声量级（~1e-2）；bf16 MLA 的 main 比 fp16 MLA 快 1.4–1.6×（padding 消冲突）。
ncu（main, S=1024 H2）：DRAM 0.14% / L1/TEX 26.68% / Compute 13.07% / occ 6.25%（135.42KB smem,
1 CTA/SM）/ Waves 0.97 / 48 regs / **bank conflicts ~0（padding 生效）** /
stall `long_scoreboard 1.71` + `wait 1.35` ⇒ bound = 全局访存延迟 + 低并行度（与 fp16 MLA 的
「smem 冲突」不同，因为 padding 已消冲突）。详见 `docs/01b` §6d。

### 7.7 ours 的 fp8 MLA（head_dim=512，P5-3 收口）

fp8 反向（单/两文件）也复用了同一 `HD` 模板化：`Fp8Cfg<HD,BM,BN>` 参数化全部常量，
`GEMM1/2` 的归约维随 `HD` 加长 k-loop，`GEMM3/4/5`（输出 N 维=HD）加一层 **N-tile 循环**
（每 128 维一遍，共 `HD/128=4` 遍），`kVU` 预取在 HD>128 时关闭改走直接向量化读。
FA/TE 反向均不支持 head_dim=512，只有 fp32 ref 与 ours：

| MLA case (B1, D=Dv=512, causal, fp8) | ours-vs-ref dq/dk/dv max_abs | ours preprocess/main/total | total TFLOPS | 峰值占比 |
|---|---|---|---|---|
| (1,256,2,512) | 2.356e-1 / 2.290e-1 / 3.441e-1 | 0.107 / 0.162 / 0.308 ms | 0.87 | 0.044% |
| (1,512,4,512) | 2.415e-1 / 2.992e-1 / 4.481e-1 | 0.196 / 0.318 / 0.591 ms | 3.64 | 0.18% |
| (1,1024,2,512) | 2.232e-1 / 3.337e-1 / 3.602e-1 | 0.371 / 0.564 / 1.022 ms | 4.20 | 0.21% |

误差与 MHA fp8 同量级（`~2.4–4.5e-1`），按 head_dim 每 128 维分段的 max_abs 均匀（N-tile 四段
都正确）；MHA d128 回归逐位不变（`2.426/2.975/3.735e-1`）。fp8 张量核 MLA 的 main 比 fp16/bf16
标量 MLA 快 **6.5–7.3×**。ncu（S=1024H2）：DRAM 1.98% / L1TEX 26.28% / Compute 5.13% /
255 regs + spill / **223.2KB smem → 1 CTA/SM、occ 6.25%** / Waves 0.97 / No Eligible 90.7%
⇒ bound = **低 occupancy/并行度**（要冲 2 CTA/SM 需降 smem）。
**O4b（第二十四轮）** 已把 `Kt/Qt/dOt` 换成 K 配对布局，smem **223.2→200.9KB**、main 1.20×
（S=1024H2 0.3933→0.3251 ms）；但即使把三个配对数组也去掉仍 >116KB，MLA 冲 2 CTA/SM 需继续
降 `Qs/Ks/Vs/dOs/dS2`（见 `03` §17.6）。详见 `docs/03` §14、§17。
