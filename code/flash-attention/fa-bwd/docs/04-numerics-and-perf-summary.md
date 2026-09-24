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
>
> **O6c（主 kernel tile 几何参数化 + 小网格并行度自适应）已完成（fp16）**：把 `fa_bwd_fp16_mma_kernel`
> 的 2×2 warp 几何从写死的 `(BM=64,BN=32)` 改成由 `(BM,BN)` 派生；host 自动档在
> **`grid<132 且 S≤1024`** 用 `(BM=32,BN=32,PIPE=1)`（grid 翻倍、并行度翻倍），`S≥4096` 用 `BN=64`。
> **main S=512 0.0858→0.0776ms（1.106×）/ 端到端 0.1598→0.1504ms（1.06×）**、S=4096 `BN=64` main
> 1.017×；数值逐位相同。**证伪**了静态 mblk 重排负载均衡（0–2%、正负不稳）。ncu：S=512 achieved
> occ 6.24%→10.99%、Duration 99.3→83.9µs；S=4096 `BN=64` L1/TEX 71.9→57.3% 但掉到 2 CTA/SM（净 +1.5%）。
> 对标：S=512 FA3 0.0265ms/81TF ⇒ ours total 17.6%（时间 5.7×，O8b 时 12.3×）；S=4096 仍 6.9%。
> 详见 `01-fp16-bwd-impl.md` §13b。

> **O7c（LSE/D 预装寄存器 + dK/dV float4 试错）已完成（fp16）**：① **负结果**——把 dK/dV 的
> `red.global.add` 从 float2 提到 float4（quad `shfl` 打包成 `F32x4`、事务数减半）后**四个几何
> 全变慢 1–7%**，证明该归约**不是事务数 bound**。② **正结果**——`lse`/`delta` 只依赖 CTA 自己的
> Q 行、与 K tile 无关，原每 tile 在 GEMM1/2 epilogue 重复 global 读（ncu：4.4/32B）；循环前
> 预装进 `lse_r/del_r` 后 **main (64,64,2) S=4096 1.8464→1.5576ms（+18.5%）、(64,32,2) +11.9%、
> S=512 +14–16%、GQA kv4 +19.1%**。端到端 **total S=4096 2.3762→2.0935ms（1.13×）**、S=512
> 0.1504→0.1495ms。数值与 O5/O8/O6/O6b/O8b/O6c **逐位相同**；单/两文件 device 逐字一致。
> ncu（main,S4096）：Duration 1.99→**1.61ms**、**L2 58.7→71.6%（新墙）**、L1/TEX 57.5→55.7%、
> `long_scoreboard` 1.79→**1.38**、regs 250 / smem 105.47KB（仍 2 CTA/SM）⇒ 墙从 L1/TEX 移到
> **L2 + occupancy**，减 red 不划算 ⇒ 下一项 **O9（wgmma+TMA）**。对标同 session 纯反向 FA3
> S=4096 0.3235ms/850TF ⇒ ours total 时间 6.47×（真反向 FLOPs 口径 131TF ≈ FA3 的 15.4%）。
> 详见 `01-fp16-bwd-impl.md` §14。

> **O10（Q/dO 载入向量化 + `cp.async` 重叠 + 打包写回）已完成（fp16，第三十七轮）**：ncu 显示
> prologue 的 Q/dO 仍是逐元素 `LDG.U16/STS.U16`（global 仅用满 26.4/32 B/sector）、dQ 写回为两次
> 4B store，且 Q/dO 的同步读延迟串在 K/V 的 `cp.async` 之前。改动：① 新增 `qdo_issue_async`
> 把 Q/dO 用 16B `cp.async.cg` 发进 smem（与 K/V 同一 `wait_group 0` 等待 ⇒ 延迟重叠）；
> ② `lse_mma_kernel_bal` 的 Q 同样向量化；③ dQ 写回打包 `float2`。**数值与 O5/O8/O6/O6b/O8b/O6c/O7c
> 逐位相同**，单/两文件 device 逐字一致。**同 session A/B（改动前二进制 vs O10）**：端到端
> **total S512 0.1485→0.1256ms（1.18×）、S4096 2.0965→2.0044ms（1.05×）、GQA kv4 0.4360→0.3767ms
> （1.16×）、MLA S256H2 0.2990→0.2272ms（1.32×）、S512H4 0.5309→0.4357ms（1.22×）**；
> main GQA kv4 0.3062→0.2620ms（1.17×）、MLA S256H2 1.22×。ncu（main,S4096）：Duration 1.61→**1.48ms**、
> Executed Instructions 350.9M→**342.1M（−2.5%）**、`long_scoreboard` 1.38→**0.89**、shared load
> 多余 wavefront 15.4→12.2%，墙仍 = `wait`（mma 依赖）+ L2 + 2 CTA/SM。对标同 session 纯反向 FA3
> S=4096 0.3242ms/848TF ⇒ ours total 时间 6.18×（O7c 6.47×）。详见 `01-fp16-bwd-impl.md` §14c、
> `01b` §6m。bf16 同构：total S512 1.15×、S4096 1.04×、GQA kv4 1.15×、MLA S256H2 1.31×。

> **O13（主 kernel auto tile 重新标定 + memset/convert 冗余）已完成（fp16/bf16，第四十轮）**：
> O6c 的 auto tile 启发式在 O7c-PREL 之后过时——S=512 MHA 的 `(64,64,2)` 比自动档
> `(32,32,1)` 主 kernel **快 1.23×**。改动：取消 `BM=32` 分支；`BN=64` 判据改为
> `S≥4096 || grid≤256 || grid>600`（`256<grid≤600` 保留 `BN=32` 以避开 2-波坏量化点，
> 如 S1024/GQA-kv4）；HD=128 时 dQ 覆盖写 ⇒ 省掉 `d_dq_acc` 的 memset；`convert_kernel` 改
> `float4` 读 + `half2` 写。**数值与 O5~O10 逐位相同**（S512 1.671/1.771/1.899e-3、
> S4096 1.883/1.734/1.966e-3）。**同 session A/B（old auto vs O13）**：fp16 total
> S512 0.1255→**0.1121（1.12×）**、main 0.0721→**0.0574（1.26×）**；S1024 kv8 total
> 0.4555→0.4414、kv1 0.6354→0.6109、kv4(h64) 0.6635→0.6261；S4096 不变。bf16 同构
> （S512 total 0.1267→**0.1112（1.14×）**、main 1.27×）。ncu（main,S512）：Duration 83.9→**59.7µs**、
> L1/TEX 43.5→**20.2%**、occ 10.99→6.23%（grid=128<132 SM，尾波/grid-bound）。
> 对标同 session 纯反向 FA3 S4096 **0.3255ms/845TF** ⇒ ours total 时间 6.03×（O10 6.18×）。
> 详见 `01-fp16-bwd-impl.md` §14f、`01b` §6p。

> **O9b（主 kernel GEMM1/2 上 Hopper `wgmma`，fp16，第四十一轮）**：把 `S=QKᵀ`、`dP=dO·Vᵀ`
> 换成 `wgmma.m64n64k16`（Q/dO/K/V 存 **SW128 K-major**；GEMM3/4/5 的转置 B 用 `ldmatrix.x2.trans`
> 从同一 SW128 tile 读，冒烟 `fa_bwd_fp16_wgmma_main_smoke.cu` 逐位 PASS）；两条 wgmma 一起发、
> 统一 `wait0` 重叠。由 `#ifdef FA_WGMMA` 包裹、默认 `sm_90` 构建不变。**数值与 O13 逐位相同**
> （S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3；GQA 回退 mma 不变）。
> **同 session A/B（main-only）**：mma `(64,64,2)` vs wgmma = S4096 1.5103→**1.4435ms（1.046×）**、
> S512 0.0570→**0.0522ms（1.092×）**；两文件端到端 S4096 **1.8822ms**（真反向 FLOPs ≈146 TF）。
> ncu（main,S4096）：Duration 1.54→**1.43ms**、smem 105.5→**101.4KB**、`wait` 1.94→**1.50**、
> `long_scoreboard` 1.45→1.25，**L2 74.4% 仍封顶、occ 仍 2 CTA/SM** ⇒ 收益真实但有限
> （GEMM3/4/5 仍是 mma、dK/dV 仍是跨 CTA 原子）。对标同 session 纯反向 FA3 S4096 0.3255ms/844TF
> ⇒ ours total 时间 **5.8×**（O13 6.03×）。详见 `01-fp16-bwd-impl.md` §14g。

> **O9b-2 第一步（主 kernel GEMM3/4/5 上 Hopper `wgmma`：MN-major 转置读，fp16，第四十三轮）**：
> 把「K-major SW128 存储的 tile」用 **MN-major 描述符 + `tnsp=1`** 读 = 读它的转置（FA3
> `dKV_swapAB` 同思路；LBO=64、SBO=(W/64)*64、转置 k16 slab 地址 `base+s*2*SBO*16`）。
> 前置冒烟 `fa_bwd_fp16_wgmma_bwd_smoke.cu` 对 `dV=PᵀdO`/`dK=dSᵀQ`/`dQ=dS·K` 逐位 PASS；
> P/dS 改 **SW128 K-major**（4B 直写），**5 个 GEMM 全 wgmma**，smem 101.4→**99.33KB**、230 regs、
> 仍 2 CTA/SM。**数值与 O5~O13 逐位相同**（S512 1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、
> GQA kv4 2.134/3.305/3.850e-3）。**但性能中性偏负**（同 session main-only：S4096 mma `(64,64,2)`
> 1.471–1.484 vs 全 wgmma **1.500–1.523ms**；S512 0.0584 vs 0.0604；GQA 0.2482 vs 0.2664）——
> ncu 证实 `short_scoreboard` 0.56→**0.29**（GEMM3/4/5 的 ldmatrix 也消掉），**但墙不在 GEMM 指令**：
> **L2 ~70% 的 dK/dV 跨 CTA 原子 + 99KB smem 锁死的 2 CTA/SM** 才是瓶颈（`long_scoreboard` 反升到 ~1.96）。
> 结论：本步建立了「MN-major 转置读」数据通路（后续 TMA/流水/去原子的地基），但性能杠杆是 **O7b 去原子**。
> 对标同 session 纯反向 FA3 S4096 0.3241ms/848TF ⇒ ours total ~6.0×。详见 `01-fp16-bwd-impl.md` §14h。

> **O17（跨 warpgroup 归约：BM=128、2 warpgroups，fp16，第五十二轮）**：§14i 已把墙钉在
> **L2 的 dK/dV 跨 CTA `atomicAdd`（占 L2 扇区 73.1%）**。O17 用 2 个 warpgroup 各算 64 行 Q，
> **dK/dV 只由 wg0 对全 BM=128 归约**（两个 m64 半进同一 `wgmma.m64n64k16` 累加器 = 两半之和），
> 于是每个 KV 元素被 `nblk/2` 个 CTA 贡献 ⇒ **red 字节砍半**。前置冒烟
> `fa_bwd_fp16_wgmma2_smoke.cu` 对 128 行转置读逐位 PASS。**ncu 精确验证**：
> `lts__t_sectors_op_red` **102.2M→51.9M（0.508×）**、`read` 也 0.50×、L2 71.8%→**54.7%**、
> Duration 1.48→**1.00ms**；regs 200 / smem 149.5KB / 1 CTA/SM（8 warps）。
> **main-only S=4096 1.57×（140.5 TF）、GQA kv4 1.47×、MQA kv1 1.54×、S=512 1.11×**；
> 端到端 S=4096 **1.427ms（96.3 TF）**、为 FA3 的 **4.4×**（O9b ~6.0×）。数值 vs ref 历史逐位一致
> （dq 无跨 CTA 原子，MHA 下逐位相同；dk/dv 仅 atomic 次序差 ~1e-4）。需 `--wg2` 开启
> （`sm_90a` + `-DFA_WGMMA`），默认行为不变。**新墙仍是 L2**（red 占 ~72.6%）。
> 详见 `01-fp16-bwd-impl.md` §14j。
>
> **O17b（BM=256、4 warpgroups，fp16，第五十四轮）—— 负结果**：red 确实精确再减半
> （`lts__t_sectors_op_red` 51.9M→**26.7M = 0.515×**），但 **512 线程把每线程寄存器上限压到
> `65536/512 = 128`**，dQ 累加器（64）+ GEMM1/2 两条 wgmma 累加器（64）吃满后必然 spill：
> `op_write` 扇区 1.57M→**14.5–34M（10–21×）**、local 占 L2 ~48%。**main 反而变慢**
> （S4096 0.995→1.20ms=0.83×、S512 0.052→0.096=0.55×）。**结论：寄存器文件是硬墙，
> 「放大 BM」走不通**，下一步转 **O7b**（把跨 CTA `red` 换成 CTA 局部累加 + 非原子写 + 二次归约）。
> 详见 `01-fp16-bwd-impl.md` §14k。
>
> **O17-2（GEMM3/GEMM4 拆分到两个 wg，fp16/bf16，第五十五轮）**：O17 的 phase B 只有 wg0
> 串行做 dV+dK（张量工作量 wg0:wg1=3:1，wg1 在 GEMM3/4 期间 barrier 空等）。把 **GEMM3→wg0、
> GEMM4→wg1**（各自仍对全 BM=128 归约、每 KV 元素仍只 `red` 一次）后两 wg 各一条 GEMM+red 链。
> ncu（S4096，同 binary `--wg2split` 0/1）：**`red` 51,904,512 完全不变**、stall `barrier`
> **1.61→0.46（−3.5×）**、Duration 997.7→**982.4µs**；main S4096 **1.013×（fp16）/1.038×（bf16）**、
> GQA kv4 1.021×/1.051×，数值与历史**逐位一致**。red 不变 ⇒ 与 O7b 正交。详见 `01` §14l、`01b` §6t。
>
> **O18（BN=128 版 wgmma2，fp16，第五十六轮）**：O17/O17-2 后 main 仍 1 CTA/SM（12.5%）+
> 延迟受限。把 KV-tile 从 BN=64 翻到 **128**（`m64n128k16`）⇒ 每 CTA 的 tile 数减半、
> barrier/`cp.async.wait`/wgmma commit-wait 序列减半。先冒烟逐位验证 `m64n128` 布局
> （`max_abs=0`）。**ncu（S4096）**：Duration 982.4→**951.1µs**、**`red` 51,904,512 逐字节不变**
> （BN 不动归约结构）、L1/TEX 49.5→**44.3%**、L2 55.4→57.2%、smem 148.5→**224KB**、230 regs
> （0 spill），仍 1 CTA/SM。**main MHA S=4096 0.9895→0.9618ms（142.9 TF，1.029×）、S=512 1.028×**；
> GQA/MQA 中性（0.994–0.995×，保持 O17）；数值与历史逐位一致。墙仍是 **L2（red 占 ~72%）+
> `wait` + 低 occupancy**。详见 `01` §14m。
>
> **O23（Hopper 路径默认化，fp16/bf16，第六十三轮）**：O9a/O17/O18 的 Hopper 快路一直是 opt-in
> （需 `--wg2`/`--wg2bn`/`--lsewgm`），默认仍是慢 1.4–1.5× 的 mma。本项在 `-DFA_WGMMA` 构建下
> **默认打开**（对齐 fp8 O22）：D==128 时主 kernel `S≥4096`→wgmma2b(BN=128)、否则 wgmma2(BN=64)；
> LSE 也走 wgmma（仅 causal）。`--wg2=0 --wg2bn=0`/`--lsewgm=0` 保留 mma 对照，纯 `sm_90` 构建不变。
> **端到端 total（同 session A/B，ms）**：fp16 MHA S512 **0.1132→0.1045（1.08×）**、GQA kv4 S1024
> **0.3751→0.2864（1.31×）**、MQA kv1 **0.6024→0.4425（1.36×）**、MHA S4096 **1.9450→1.3641（1.43×）**；
> bf16 MHA S512 0.1119→0.1058、GQA kv4 0.3763→0.2875、MQA kv1 0.6022→0.4438、MHA S4096
> **1.9429→1.3666（1.42×）**；MLA（D=512）不变（无 wgmma2）。数值与历史**逐位一致**。
> 默认路径 ncu 即 `wgmma2b`：`red=51,904,512`（逐字节同 O18）、255 regs/231.4KB/occ 12.48%、
> L2 56.58%、stall `wait 1.25` ⇒ 墙仍是 **L2 red + 1 CTA/SM**。详见 `01` §14n、`01b` §6v。
>
> **O24（preprocess `delta` 向量化 + dQ 直写 fp16，fp16/bf16，第六十五轮）**：main 已是硬墙
> （L2 red）后，回头清 `delta_kernel`（旧版每行一个 128 线程 CTA + smem 树归约，S=4096 占 42.5µs、
> ncu Compute 72%/L1TEX 74%）与 `convert` 的 dQ 一趟。`delta_warp_kernel` 改 **warp-per-row
> `__half2` + `__shfl_xor`**（无 smem/barrier）⇒ **S4096 42.5→14.5µs（3.35×）、DRAM 74% bound、
> 指令数 −82%**；D=128 wgmma2/2b 主 kernel **直接写 fp16 dQ**、convert 跳过 dQ。端到端 total
> （同 session A/B）：fp16 S4096 1.3802→**1.3309ms（1.037×）**、S512 1.051×、GQA kv4 1.066×、
> MQA kv1 1.077×；bf16 S4096 1.3650→**1.3388（1.020×）**、S512 1.053×、GQA kv4 1.053×、
> MQA kv1 1.069×。数值与历史**逐位一致**。详见 `01` §14p、`01b` §6w。

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

> **O8b（LSE 负载均衡 + `cp.async`，与 fp16 逐字同构）已完成**：镜像配对（`m` 与 `nblk-1-m`，
> 工作量恒 `nblk+1`）+ K 的 `cp.async.cg` 16B 双缓冲。**lse S4096 0.970→0.350ms（2.77×）**、
> S512 1.38×、GQA kv4 2.06×（收益主要来自负载均衡、cp.async 再叠 1.29×）；**preprocess
> S4096 0.993→0.392ms**；端到端 **total S4096 2.961→2.343ms（58.67 TF，峰值 5.9%）**、
> S512 0.186→0.160ms、GQA kv4 0.576→0.482ms（35.66 TF）。数值与 O5b/O8/O6/O6b **逐位相同**。
> ncu（lse,S4096）：Duration 1.03ms→**356µs**、`long_scoreboard` 2.19→0.34、Waves 1.29→**0.97**、
> Compute 39.5%→**60.4%**、occ 22.9%（52.2KB smem/4 CTA/SM）；新墙 = **Compute 60% + smem 依赖**。
> 同 session 纯反向 FA3 S4096 **0.3194ms/861TF** ⇒ ours total 为 FA3 的 **6.8%**、时间比 7.3×；
> GQA kv4 FA3 0.0825ms/416 ⇒ ours 8.6%、时间比 5.8×。详见 `01b-bf16-bwd-impl.md` §6i。

> **O6c-bf16（主 kernel tile 几何参数化 + 小网格并行度自适应，与 fp16 逐字同构）已完成**：把
> `fa_bwd_bf16_mma_kernel` 的 2×2 warp 几何从写死 `(BM=64,BN=32)` 改成由 `(BM,BN)` 派生；host
> 自动档在 **`grid<132 且 S≤1024`** 用 `(BM=32,BN=32,PIPE=1)`（grid 翻倍、并行度翻倍），`S≥4096`
> 用 `BN=64`。**main S=512 0.0886→0.0800ms（1.11×，另一 session 1.18×）/ 端到端 0.1603→0.1501ms
> （1.06×，14.31 TF）**、S=4096 持平 2.369ms（58.01 TF）、GQA kv4 0.484ms（35.51 TF）**；
> **数值与 O5b/O8/O6/O6b/O8b 逐位相同**。
> **再次证伪**静态 mblk 重排（`[O6c A/B]` 0–2%、方向不稳）。ncu：S=512 `(32,32,1)` achieved occ
> 6.24%→**10.99%**、Duration 99.3→**83.9µs**；S=4096 `BN=64` L1/TEX 71.7→**57.1%** 但 168→242 regs、
> smem 105KB 使 3→2 CTA/SM（occ 16.9→11.8%），净 +1.5%，墙 = L1/L2 吞吐 + `wait`。
> 对标（纯反向 `fa_vs_te_bwd_only.py bf16`）：S=512 FA3 0.0265ms/162TF ⇒ ours **8.8%**（时间 5.7×）；
> S=4096 FA3 0.3195ms/860 ⇒ ours **6.7%**（7.4×）；GQA kv4 FA3 0.0822/418 ⇒ ours **8.5%**（5.9×）。
> 详见 `01b-bf16-bwd-impl.md` §6j。

> **O7c-bf16（LSE/D 预装 + dK/dV float4 试错，与 fp16 逐字同构）已完成**：float4 一致变慢
> （−2.4~−4.6%，同 fp16 负结论）；**PREL 正结果——main (64,64,2) S=4096 1.8297→1.5566ms（+17.5%）、
> (64,32,2) +13.5%、S=512 +14–16%、GQA kv4 +17.5~18.9%**。端到端 **total S=4096 2.3692→2.0986ms
> （1.13×）**、S=512 0.1495ms。数值与 O5b/O8/O6/O6b/O8b/O6c **逐位相同**；单/两文件逐字一致。
> ncu（main,S4096）：Duration 1.87→**1.64ms**、**L2 70.5%（新墙）**、L1/TEX 55.7%、regs 250 /
> smem 105.47KB（2 CTA/SM）、`wait 2.00 / long 1.39 / short 0.88`。对标纯反向 FA3 S=4096
> 0.3193ms/861TF ⇒ ours total 时间 6.57×（真 FLOPs 口径 ≈15.2%）。详见 `01b-bf16-bwd-impl.md` §6k。

> **O10-bf16（Q/dO 向量化 + `cp.async` 重叠 + `float2` 写回，与 fp16 逐字同构）已完成
> （第三十七轮）**：数值与 O5b/O8/O6/O6b/O8b/O6c/O7c **逐位相同**（S512 9.001/12.61/13.65e-3、
> S4096 15.10/13.40/16.31e-3）。同 session A/B：**total S512 0.1460→0.1268ms（1.15×）、S4096
> 2.0824→1.9950ms（1.04×）、GQA kv4 0.4341→0.3761ms（1.15×）、MLA S256H2 0.2985→0.2285ms
> （1.31×）、S512H4 0.5319→0.4343ms（1.22×）**；ncu（main,S4096）Duration 1.55ms、
> Executed Instructions 342.1M（与 fp16 O10 相同）、墙 = `wait` + L2 + 2 CTA/SM。
> 详见 `01b-bf16-bwd-impl.md` §6m。

> **O9a：LSE 预处理上 Hopper `wgmma`（fp16/bf16，第三十九轮）**：新增冒烟
> `fa_bwd_fp16_wgmma_smoke.cu`（SW128 + `wgmma.m64n64k16` 累加器映射，max_abs=0 PASS）+
> `lse_mma_kernel_bal_wgmma<HD,PIPE>`（Q/K 存 SW128、`cp.async` 发、8 条 wgmma 完成 QKᵀ）。
> 同 session LSE-only：fp16 S=4096 O8b 0.3004→**0.2871ms（1.046×）**、S=512 0.0338→0.0330、
> GQA kv4 0.0666→0.0642；bf16 S=4096 0.3006→**0.2856ms（1.054×）**。ncu（S=4096）：
> Duration 303.6→**286.2µs**、**L1/TEX 37.6→18.1%**（ldmatrix 消失）、L2 31.6→20.3%、
> Executed Ipc 2.36→2.52。**数值与 O8b 逐位相同**；端到端 total S=4096 1.95ms（70 TF）。
> 结论：wgmma 把 LSE 的访存那一半打掉，但 LSE 是 softmax epilogue/发射 bound（Compute 60%、
> Waves 0.97），故总收益有限 ⇒ 下一步 **O9b 主 kernel wgmma**。详见 `01` §14e、`01b` §6o。

> **O13：主 kernel auto tile 重标定（bf16，第四十轮）**：与 fp16 逐字同款（取消 `BM=32`、
> `BN=64` 判据 `S≥4096||grid≤256||grid>600`、HD=128 去 dQ memset、convert 向量化）。
> **数值与 O5b~O10 逐位相同**（S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3）。
> **同 session A/B**：total S512 0.1267→**0.1112（1.14×）**、main 0.0727→**0.0572（1.27×）**；
> S1024 kv1 total 0.6354→0.6109、kv4(h64) 0.6635→0.6261；S4096 不变。
> 详见 `01b-bf16-bwd-impl.md` §6p。

> **O9b-bf16：主 kernel GEMM1/2 上 Hopper `wgmma`（第四十二轮，单/两文件）**：把 fp16 O9b
> （§2.1）逐字 dtype 参数化到 bf16（同为 2 字节，SW128/描述符/`m64n64k16` 累加器映射逐字节同构）。
> Q/dO/K/V 存 SW128、GEMM1/2 两组 wgmma 统一 `wait0` 重叠；GEMM3/4/5 仍 mma、转置 B 用
> `ldmatrix.x2.trans` 从同一 SW128 tile 读。**数值与 O5b~O13 逐位相同**（S512 9.001/12.61/13.65e-3、
> S4096 15.10/13.40/16.31e-3、GQA kv4 12.01/21.25/31.56e-3）。**同 session A/B（main-only）**：
> S512 0.0571–0.0573→**0.0524–0.0526ms（1.089×）**、S4096 1.4861→**1.4515–1.4552ms（1.021×）**；
> 端到端 total S4096 **1.880ms（73.1 TF）**、S512 **0.1085ms**、GQA kv4 0.376ms。ncu（main,S4096）：
> Duration 1.46ms、**L2 72.99%** / L1/TEX 54.26% / Compute 26.51% / 242 regs / 101.38KB（2 CTA/SM）、
> stall `wait 1.50 + long 1.24 + short 0.56`，与 fp16 O9b 逐项一致。对标纯反向 FA3 S=4096
> 0.3191ms/861TF ⇒ ours total 时间 **5.89×**。详见 `01b-bf16-bwd-impl.md` §6q。

> **O17-bf16：跨 warpgroup 归约（BM=128、2 warpgroups，第五十三轮，单/两文件）**：把 fp16
> O17（§2.1）逐字 dtype 参数化到 bf16。**只让 wg0 做 GEMM3/4**、把两个 m64 半（128 行）连续
> 喂同一 `wgmma.m64n64k16` 累加器 ⇒ 每个 KV 元素只 `red` 一次；wg1 并行做自己的 GEMM5。
> **数值与 O5b~O13 逐位一致**（S512 9.001/12.61/13.65e-3、S4096 15.10/13.40/16.31e-3、
> GQA kv8 12.33/19.30/31.50e-3、GQA kv4 12.01/21.25/31.56e-3、MQA kv1 11.90/45.58/71.96e-3）。
> **同 session A/B（main-only）**：S512 0.0577→**0.0521（1.109×）**、S4096 1.4892→**0.9821
> （1.516×，139.9 TF）**、GQA kv8 1.504×、GQA kv4 1.506×、MQA kv1 1.531×。端到端 S=4096
> **1.4309ms（96.05 TF）**、为 FA3 的 **4.45×**（O9b ~6.0×）。ncu（main,S4096，同 session
> O9b vs O17）：**`lts__t_sectors_op_red` 102,236,160→51,904,512（0.508×）**、`read` 0.50×、
> Duration 1.48→**0.997ms**、**L2 71.61→54.63%**、L1/TEX 40.25→36.68%、regs 230→200 /
> smem 100.35→149.50KB / occ 11.89→12.41%，bank conflict 0；**机制假设被 ncu 完全证实，
> 与 fp16 O17 逐项一致**。需 `--wg2`（`sm_90a`+`-DFA_WGMMA`），默认行为不变；D=512 时忽略。
> 详见 `01b-bf16-bwd-impl.md` §6s。

> **O18-bf16：BN=128 版 wgmma2（第五十七轮，单/两文件）**：把 fp16 O18（§2.1）逐字 dtype
> 参数化到 bf16（kv-tile `BN=64→128`、`m64n128k16`，per-CTA tile 数减半 ⇒ barrier/commit-wait
> 序列减半；新增 `wgmma_m64n128k16_bf16_t`/`wgmma_mn128_issue`/`fa_bwd_bf16_wgmma2b_kernel`，
> host `--wg2bn`）。**数值与 O5b~O17 历史逐位一致**（S512 9.001/12.61/13.65e-3、S4096
> 15.10/13.40/16.31e-3、GQA kv4 12.01/21.25/31.56e-3、kv8 12.33/19.30/31.50e-3、kv4(h64)
> 13.51/30.91/44.20e-3、MQA kv1 11.90/45.58/71.96e-3）。**同 session A/B（main-only）**：
> MHA S512 0.0507→**0.0492（1.030×）**、S4096 0.9799–0.9836→**0.9558–0.9560（1.025–1.029×，
> 143.8 TF）**；GQA/MQA 中性（0.996–1.007×）；串行版普遍更慢（0.97–1.00×）。端到端 S4096
> **1.381ms（99.5 TF，FA3 的 4.30×，O17 4.45×）**、S512 0.104、GQA kv4 0.291 / kv8 0.332 /
> kv4(h64) 0.447 / MQA kv1 0.447ms。ncu（main,S4096，同 binary `--wg2bn` vs `--wg2`）：
> Duration 982.2→**952.5µs（1.031×）**、**`red` 51,904,512 逐字节不变**、regs 200→255 /
> smem 148.5→230.4KB / occ 12.5%（1 CTA/SM）；墙仍是 **L2 red（~72%）+ 1 CTA/SM**。
> 详见 `01b-bf16-bwd-impl.md` §6u。

> **O24-bf16（preprocess `delta` 向量化 + dQ 直写 bf16，第六十五轮）**：与 fp16 O24 逐字同构
> （见 §2.1 的 O24 条）。端到端 total 同 session A/B：S4096 1.3650→**1.3388ms（1.020×）**、
> S512 0.1050→**0.0997（1.053×）**、GQA kv4 0.2877→**0.2733（1.053×）**、MQA kv1
> 0.4423→**0.4139（1.069×）**；`delta` 单项 0.0424→**0.0126ms（3.36×）**；数值与历史逐位一致。
> 对标 FA3 MHA S4096 **0.3202ms/859TF** ⇒ ours total 时间 **4.18×**。详见 `01b` §6w。

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

> **下表为 O11（第三十八轮）最新值**：把 fp16/bf16 的 **O8b**（LSE 镜像配对负载均衡 +
> `cp.async` 双缓冲）移植到 fp8 的 `lse_mma_kernel_bal`（fp8 一行 `HD` 字节 = `HD/16` 个
> 16B unit）。**lse S=4096 0.936→0.345 ms（2.71×）**、preprocess 1.20→0.388 ms；数值与
> O7 **逐位相同**（单/两文件一致）。另附快速 exp/log（`__expf`/`__logf`，preprocess 再 ~8%）。
> main 未改。逐项见 `03` §19。

| shape | ksplit | ours total（含 quant） | preprocess | ours main | TE FP8（同 session，纯反向） |
|---|---|---|---|---|---|
| d128 (1,512,16,128) | 16 | 0.1797 ms | 0.0465 ms | 0.0725 ms | 0.1005 ms / 42.7 TF |
| d128 (1,1024,32,128) | 8 | 0.7116 ms / 24.1 TF | 0.1024 ms | 0.4463 ms | 0.1477 ms / 116.3 TF |
| d128 (1,4096,16,128) | 4 | **3.3104 ms / 41.5 TF** | **0.3878 ms**（1.20→0.388，3.1×） | 2.5591 ms | 0.5908 ms / 465.3 TF |
| GQA h32kv4 (1,1024,32,128) | 8 | 0.6380 ms | 0.1006 ms | 0.4308 ms | — |
| MQA h64kv1 (1,1024,64,128) | 8 | 1.1080 ms | 0.1457 ms | 0.8123 ms | — |
| MLA (1,1024,2,512) | 4 | 0.5390 ms | 0.1388 ms | 0.3250 ms | NA（FA/TE 不支持） |

- 端到端 S=4096 **4.13→3.31 ms（1.25×）**；纯反向（去掉 quant）≈3.13ms，**ours/TE 6.7×→5.3×**。
- 新墙 = **Compute 61% + 网格不足一个波**（lse_bal ncu：Duration 357µs、Waves 0.65、
  achieved occ 23.2%），与 fp16/bf16 O8b 一致。
- 同轮顺带 A/B **证伪**两条 fp16 主 kernel 假设：`(BM=32,BN=64,PIPE=2)`（慢 1.75×）、
  `cp.async .L2::256B`（无变化）。详见 `03` §19。

> **下表为 O7e（第四十五轮）最新值**：先用 `MemoryWorkloadAnalysis_Tables` 重新定位——
> O7 后 fp8 main 的**第一墙是 L1/TEX 66–71%**（L2 已降到 43.7%），其中 **register spill 占
> L1TEX sector 12%**（REGDQ 的 `dqacc` + O3 寄存器预取抢 168-reg 预算）、**shared store
> bank conflict 3.5-way/69.8%**（fold 段逐字节写）。① fold 的 `Ap/dS3/dS2` 改 **4B 向量化写**
> （store 指令 ÷4，store 冲突 68.6M→36.3M）；② `REGDQ` 生效时**关 O3 预取**（让出 16 regs）。
> **数值与 O7/O4b/O11 逐位相同**；收益集中在 S=4096（`REGDQ` 生效）。逐项见 `03` §20。

| shape | ksplit | ours total（含 quant） | ours main | main vs O7 | main TF（峰值占比） | TE FP8（同 session，纯反向） |
|---|---|---|---|---|---|---|
| d128 (1,512,16,128) | 16 | 0.180 ms | 0.0720 ms | ~1.00× | 29.0（1.47%） | 0.1011 ms / 42.5 TF |
| d128 (1,1024,32,128) | 8 | 0.710 ms / 24.2 TF | 0.4414 ms | **~1.04×** | 37.4（1.89%） | 0.2059 ms / 166.9 TF |
| d128 (1,4096,16,128) | 4 | **3.266 ms / 42.1 TF** | **2.5215 ms** | **1.036×** | 54.5（2.76%） | 0.5899 ms / 466.0 TF |
| MLA (1,1024,2,512) | 4 | 0.539 ms | 0.3243 ms | ~1.00× | 13.2（0.67%） | NA（FA/TE 不支持） |

- ncu（main, S=4096）：Duration 2.66→**2.57ms**、**L1/TEX 71.07→66.06%**、
  spill 占 L1TEX sector 12.09→**5.74%**、store 冲突 **68.6M→36.3M**；数值逐位不变、
  168 regs / 70.66KB / 3 CTA/SM 不变。
- **重要结论：O7b（去 dK/dV 跨 CTA red）已不是头号杠杆**（L2 43.7% < L1/TEX 66%）；
  下一步应转向 **fp8 main 的 Hopper `wgmma`（O9c）**。
- 同 session 纯反向基线（`fa_vs_te_bwd_only.py`，FA2/FA3/TE 三列）：MHA S=4096 FA3
  fp16 **0.3245ms/847TF**、TE fp16 0.4441/619、FA2 0.7286/377；bf16 FA3 0.3201/859；
  TE FP8 0.5899ms/466TF。ours fp8 main 2.5215ms ⇒ TE FP8 整条反向的 **~4.3×**
  （S=512 main 0.072ms 已快过 TE FP8 0.101ms）。

> **O9c 第一步（第四十六轮）**：建立 **fp8 Hopper `wgmma.m64n64k32` + SW128** 数据通路，
> 先在 **LSE** 上落地（对齐 fp16 O9a）。冒烟 `fa_bwd_fp8_wgmma_smoke.cu` 逐位 PASS
> （e4m3×e4m3 / e5m2×e4m3）。LSE `lse_mma_kernel_bal_wgmma<HD,PIPE>`（镜像配对 + `cp.async`
> 双缓冲）vs O11 mma：**event S=4096 0.3437→0.2697ms（1.275×）/ S=512 1.065× / MQA 1.302×**
> （ncu Duration 355.87→**279.07µs**、L1/L2 降、regs 77→64、理论 occ 37.5→50%）；端到端
> S=4096 3.2943→**3.1925ms（1.03×，43.05 TF）**。数值：最终 dq/dk/dv 与 ref 同 O7e 水平
> （S=4096 2.634/2.643/3.217e-1），LSE vs mma max_abs 5.6–7.4e-4（fp32 求和次序差）。
> ncu 结论：LSE 仍是 **Compute ~60%（softmax epilogue）+ 网格不足一个波**。ours/TE FP8
> S=4096 **5.42×**（O7e 5.56×）。**主 kernel 仍是 `mma.m16n8k32`（第一墙 L1/TEX 未动）**
> ⇒ O9c-2 把主 kernel GEMM1/2 上 `wgmma`。构建需 `-DFA_WGMMA` + `sm_90a`。详见 `03` §21。

> **O9c-2 第一步（第四十七轮）**：fp8 主 kernel 的 **GEMM1/2（`S=QKᵀ`、`dP=dO·Vᵀ`）上
> `wgmma.m64n32k32`**（Q/dO/K/V 存 **SW128**；fold + GEMM3/4/5 + dQ 归约逐字复用 mma 版；
> `smem 70.66→68.61KB`）。同 session A/B（main-only，event）：S512 **1.038×** / S1024H32
> **1.052×** / GQA kv4 **1.057×** / S4096 **1.056×**；端到端 total S=4096 **3.1374ms（43.8 TF）**、
> ours/TE FP8 **5.32×**（O7e 5.56×）。数值 vs ref 与 mma 版同量级（S4096 2.635/2.644/3.216e-1）。
> ncu：L1/TEX 66.06→**65.19%**、Duration 2.57→**2.46ms**、仍 3 CTA/SM；**第一墙仍是 L1/TEX
> （GEMM3/4/5 的 `ldmatrix` + fold）** ⇒ O9c-2b。详见 `03` §22。

> **O12（第四十八轮）**：fp8 主 kernel 的 **LSE/D 预装寄存器**（fp16 O7c-PREL 的 fp8 版；
> 原实现每 tile 每元素 global 读 lse/delta）。新增 `PREL` 开关，main **S512 1.042× / S1024H32
> 1.078× / S4096 1.100× / GQA 1.079× / MQA 1.108× / MLA 1.01×**；`--wgmma` 路径 S4096
> **1.125×**、端到端 **2.872ms（47.9 TF）**。ncu：uncoalesced global 38.0M→**4.70M（−87.6%）**、
> Executed Instructions **−8.4%**、Duration 2.60→**2.34ms**；数值与历史逐位一致。
> **O9c-2b（fp8 GEMM3/4/5 wgmma）查证为硬件阻塞**：fp8 wgmma 无转置操作数（CUTLASS 全为
> `_SS_TN`、asm 尾部无 `tnsp`），MN-major 转置读在 fp8 ISA 上不存在 ⇒ 记入 ROADMAP「阻塞」。
> 详见 `03` §23。

> **O14（第四十九轮）**：fp8 **输入量化**从「每行一个 CTA（smem 归约 + 7×`__syncthreads`）」
> 改为 **warp-per-row `float4` + `__shfl_xor` 树 + `uchar4` 写**（`quantize_row_warp_kernel`）。
> quant（q/k/v/dO 一组）**2.15×（S512）/ 2.44×（S1024H32）/ 2.65×（S4096）/ 2.22×（GQA kv4）
> / 1.22×（MLA S256H2）**，8 个量化输出**逐字节 bitwise mismatch=0**；端到端 total S=4096
> **3.0617→2.9210ms（47.1 TF，ours/TE FP8 5.22×→4.98×）**、S=1024H32 1.090×、S512 1.082×。
> ncu：量化 kernel Duration 46.2→**15.4µs**、墙从 **L1/TEX 73.7%+Compute 71.8%（DRAM 仅 24%）**
> 移到 **DRAM 71.3%（已达带宽上限）**、指令数 **−77%**。另证伪 convert 的 `float4`（中性）。
> 详见 `03` §24。

> **O7e-2（第五十轮）**：修 fp8 main fold 的 **shared-load bank conflict**。ncu Memory Tables
> 拆出 L1/TEX 第一来源是 **shared load（93.1M 请求、2.1-way、62.8M 冲突）**，根因是 fold 读
> `Ps/Ss` 列时 4 个 lane 组的 m 间距为 16（`16*PSS≡16 mod32` ⇒ 恒撞）。把每 lane 的 m 从
> `sub4*16+t` 改成两半 `sub4*8+t+half*32`（bank 铺满 0..31），并用 8B/16B 向量写。
> **shared load 冲突 62.8M→29.2M（−53.5%）、L1/TEX 64.8%→60.0%、Duration 2.41→2.27ms**；
> 同 session A/B（main-only）S512 **1.017×** / S1024H32 **1.029×** / GQA kv4 **1.033×** /
> MQA kv1 **1.040×** / S4096 **1.042×**；端到端 S=4096 **2.8942ms（47.5 TF，ours/TE FP8
> 4.98×→4.92×）**。数值与 O7/O12/O14 **逐位一致**（A/B 差 1e-7 仅 atomic 次序）。
> 另**证伪**单独把 fold 写折到 16B（只 1.007×、且增大 spill）⇒ fold 的墙在**读冲突**不在写指令数。
> 详见 `03` §25。

> **O7e-3（第五十八轮）**：fp8 main **GEMM1/2 epilogue `Ps/Ss` 的 store/回读 bank conflict**。
> 用 `--page source --csv` 按源码行拆开：shared-store 多余 wavefronts 的 **~98%** 来自
> `Ps/Ss[r*PSS+c]` 标量写（25.6M+12.8M）及其同模式回读（12.8M）；根因 `PSS=33`（≡1）使
> `(g·PSS+2l) mod32` 重合 ⇒ 4-way。改 **`PSS=BN+5=37`**（仍 ≡1 mod4，fold 掩码读无冲突；
> bank=(5g+2l) ⇒ 2-way）。**数值逐位不变**；smem 70.7→72.7KB 仍 3 CTA/SM。同 session A/B main
> S512 1.016× / S1024H32 1.010× / S4096 1.011× / GQA kv4 1.007× / MLA S1024H2 1.024×，
> 端到端 1.005–1.015×（S4096 total **2.8375ms，48.4 TF**）。ncu：store 冲突 29.46M→**12.33M
> （−58%）**、load 冲突 29.18M→**20.59M（−29%）**、L1/TEX 59.97→**55.83%**，**但 Duration 持平**
> ⇒ **证伪「L1/TEX 是 fp8 main 的限速器」**：墙是 `wait`(1.56)+`short_scoreboard`(1.50) 的
> mma 依赖延迟 + 3 CTA/SM（issue 45.9%）。对标：TE FP8 S4096 0.5887ms；FA3 fp16 MHA S4096
> 0.3251ms/846TF（fp8 无 FA 基线）。详见 `03` §26。

> **O19（第五十九轮）——fp8 跨 warpgroup 归约（BM=128、2 wg、256 线程）＝负结果**：`fa_bwd_fp8_wg2_kernel`
> 把 dK/dV 的跨 CTA `red` 砍到 **0.593×**（108.48M→64.29M）、`read` 0.537×、L2 49.9%→**22.6%**，
> 但 **1 CTA/SM（8 warp/SM，mma 为 3 CTA/SM、12 warp）**，issue 43→34%、No Eligible 54→64%，
> 同 session main 反而 **0.67–0.78×**（S4096 2.27→2.90ms）。数值仍为 fp8 噪声（dq 逐位级一致）。
> ⇒ **判决：fp8 main 的墙不是 L2 `red`，而是 mma 依赖延迟 + occupancy**；减 red/放大 BM 对 fp8
> 无收益，真正杠杆是提 occupancy（168 regs→≤128、72.7KB→≤58KB 才 4 CTA/SM）或减 mma stall。
> 详见 `03` §27、原始输出 `src/fp8/o19_wg2_ab_sweep.out.txt` / `o19_ncu_{wg2,mma}_red_s4096.out.txt`。
>
> **O20（第六十轮）——fp8 mma 路径 GEMM1/GEMM2 epilogue 融合（消 `Ps` 回读）**：O7e-3 遗留的
> 12.78M `Ps` 回读 wavefronts 被消掉（`FA_FUSE_EPI`，P 留 `preg[2][2][4]`，**数值逐位不变**）。
> ncu：shared 总 wavefronts 169.07→160.55M、`op_ld` 冲突 20.55→16.54M（−19.5%）、
> `short_scoreboard` 1.50→1.41、Duration 2.25→2.21ms；**main S4096 1.026×（2.1997→2.1433ms）、
> S1024H32 1.015×、kv8 1.017×**，小 S/MLA 中性。端到端 S4096 **total 2.77ms（49.6 TF）**、
> 同 session TE FP8 0.5909ms/465TF ⇒ ~4.7× TE。墙仍 = `wait`+3 CTA/SM + 残余 L2 red。
> 详见 `03` §28、原始输出 `src/fp8/fa_bwd_fp8_main_o20_fuse_ab.out.txt`。
>
> **O21（第六十一轮）——fp8 BN=64 负结果 + 消冗余 convert（O21b）**：① 把 O18 的「翻倍 KV-tile
> 减半串行相位」假设搬到 fp8 mma 路径（BN 全参数化，单/两文件 device 逐字一致）：**S512 0.900×、
> S1024H32 0.920×、GQA kv4 0.920×、S4096 0.816×**。ncu 证实 regs 168→255、smem 72.7→105.2KB、
> **occupancy 3→2 CTA/SM（18.05→12.28%）**；相位确实减半（Warp Cyc/Inst 6.19→5.92）但延迟受限下
> 少 1/3 在飞 warp 更亏 ⇒ **与 O19 一起双重证伪 fp8 的「放大 tile/减 red」**（fp16 O18 成立是因为
> 它从 1 CTA/SM 出发）。② **O21b**：fp8 输出本是 fp32，`convert_kernel` 是纯 fp32→fp32 拷贝 ⇒ 把
> acc 别名到输出、main 直接 `atomicAdd`，**端到端 S4096 2.7469→2.6497ms（1.037×）/ 单文件 1.046× /
> GQA 1.037×，数值逐位不变**，已设为默认。S4096 端到端 **2.65ms / 52.2 TF**、为 TE FP8（465TF）的
> **4.5×**。详见 `03` §29、原始输出 `src/fp8/o21_main_bn_ab_{s4096,gqa_kv4}.out.txt`、
> `o21_onefile_s4096.out.txt`、`o21_ncu_bn{32,64}_s4096.out.txt`。
>
> **O22（第六十二轮）——fp8 Hopper 路径（wgmma LSE + 主 kernel GEMM1/2）默认化**：`-DFA_WGMMA`
> （`sm_90a`）构建下 `lsewgm`/`wgmma` 默认开（`sm_90` 构建不变）。同 binary、同 session A/B
> （CUDA event，端到端 total）：**S512 0.1421→0.1341（1.060×）/ S1024H32 0.5623→0.5308（1.059×）/
> S4096 2.6937→2.4837（1.085×）/ GQA kv4 0.5296→0.4947（1.071×）**；S4096 **2.48ms / 55.3 TF**
> （main-only 67.3 TF，峰值 3.4%），为 TE FP8（同 session 0.5899ms/465.9TF）的 **4.21×**（O21b 4.5×）。
> 收益 = LSE `wgmma`（preprocess 1.246×）+ 主 kernel `wgmma`（main 1.065×）；数值 vs ref 逐位同级。
> 同轮量测并**否决**：`--regdq=0`（main 2.136 vs 2.480ms，保留 REGDQ——local spill 比多发 dQ red
> 便宜）、`FA_ILV`（GEMM1/2 交错，中性偏负）、ksplit 重标定（维持 k=4）、`PSS` 扫描（37 近最优）。
> 详见 `03` §30、原始输出 `src/fp8/o22_*`。

> **O26（第六十七轮）——fp8 `delta_kernel` → warp-per-row 向量化（对齐 fp16/bf16 O24）**：fp16/bf16
> 在 O24 已把 delta 改成 warp-per-row（S=4096 3.35×），但 **fp8 的 delta 一直是旧版**（每行一个
> 128 线程 CTA + smem 树归约）。新增 `delta_warp_kernel<HD>`（每 warp 一行，lane 沿 HD 用
> `float4`(O)/`uchar4`(dO) 读 + `__shfl_xor` 树，无 smem/无 barrier，grid-stride），
> `--deltawarp=0` 供 A/B；数值 `max_abs(new-vs-old)=1.9e-6~8.6e-6`（仅 fp32 求和次序），
> vs ref 与历史同水平。**delta 2.39–3.22×**（S4096 0.0433→**0.0151ms**）、preprocess **1.078×**、
> 端到端 S4096 **2.4837→2.4637ms（1.008×）/ 55.8 TF**，为 TE FP8（0.5904ms/465.6TF）的 **4.17×**。
> ncu：Duration 42.94→**17.34µs**、墙从 smem 归约（Compute 74.9%）移到 **DRAM 带宽 77%**
> （elementwise 上限）。详见 `03` §31、原始输出 `src/fp8/fa_bwd_fp8_main_o26_sweep.out.txt`、
> `o26_ncu_delta_{old,new}_s4096.out.txt`、`o26_te_fp8_bench.out.txt`。
>
> **O27（第六十八轮）——fp8 fold 量化「逐元素精确除法」→「每行 rcp + 乘法」**：fold 里每个
> (m,j) 元素都算一次 `Ps*[dos] / scA`（`scX` 是每输出行一个的常量），ptxas 默认 `prec-div`
> ⇒ ~10+ 指令/元素的精确除法。改成每行 `__frcp_rn` 一次 + `__shfl` 广播 + 乘法（新模板参数
> `RCP`，默认 true；`--foldrcp=0` 供同 binary A/B）。**main 1.08–1.18×**（S4096 2.043→
> **1.758ms**、MQA 1.178×、kv8 1.152×）；**端到端 S4096 2.4637→2.1770ms（63.1 TF）**、S512
> 0.1284→0.1246、S1024H32 0.5218→0.4895、GQA kv4 0.4875→0.4566、kv8 0.5606→0.5010、MQA
> 0.7926→0.6956、MLA（S256/S512/S1024）0.1169/0.2972/0.5175→0.1094/0.2688/0.4653。
> **vs fp32 ref 的 dq/dk/dv 与历史逐位一致**（9 shape 全部）；ncu main **Duration 2.06→1.78ms、
> executed inst 852.6M→735.2M（−13.7%）**，墙仍是 mma 依赖延迟（wait+short）+ 3 CTA/SM。
> 为 TE FP8（同 session 0.5904ms/465.6TF）的 **3.69×**（O26 4.17×）。详见 `03` §32、原始输出
> `src/fp8/o27_main_sweep.out.txt`、`o27_ncu_rcp_s4096.out.txt`、`o27_te_fp8_bench.out.txt`。
>
> **O28（第六十九轮）——fp8 fold 的转换指令向量化（`cvt ... x2` + `PACK_AB_MERGE_C`）**：
> O27 证明 fold 的指令数在关键路径上；本轮把 fold 里每个元素的 3 次 fp8 转换从「逐元素标量
> `__nv_cvt_float_to_fp8` + shift/OR」换成 **`__nv_cvt_float2_to_fp8x2`**（一次转 2 个，硬件
> `PACK_AB_MERGE_C` 直拼 32 位），新增 `foldpack4` + 开关 `FA_CVT2`（默认 1）。**数值与两次
> 标量转换逐位相同 ⇒ vs fp32 ref 与历史逐位一致**（S4096 `2.635/2.644/3.216e-1`、MLA S1024H2
> `2.232/3.337/3.602e-1`）。性能：**d128（MHA/GQA/MQA）全部中性**（fold 每 tile 只 1 遍，被 mma
> 依赖延迟掩盖），**MLA（`NDT=4`）main 1.03×**（S512H4 0.1423→0.1386、S1024H2 0.2727→0.2653ms）。
> ncu：d128 S4096 inst **735.4M→703.5M（−4.3%）**但 Duration 仅 1.80→1.77ms；MLA S1024H2
> 326.75→320.54µs。**再次确认 d128 的墙是 mma 依赖延迟、不是 fold 指令数**。详见 `03` §33、
> 原始输出 `src/fp8/fa_bwd_fp8_o28_ab.out.txt`、`o28_ncu_main_{s4096,mla_s1024h2}_ab.out.txt`。
>
> **O29（第七十轮）——fp8 自动 split-K 重新标定（正结果）+ GEMM3/4 交错（负结果）**：
> O2b 的固定 `ksplit` 目标 `(D==128)?4096:132` 在后续数据通路改动后已漂移（d128 base 小时
> 过切、S4096 欠切、MLA 严重欠切）。新标定 `D==128 → S>=2048?8192:max(2048,4*base_grid)`、
> `D==512 → S/2`（仅 host 自动档，`--ksplit=N` 可覆盖；不改 device/数学，只改 fp32 加法次序）。
> **main 最多 1.20×（GQA kv4 0.329→0.274ms）、MLA S1024H2 1.32×（0.265→0.200ms）、
> S1024H32 1.14×、S4096 1.01×**；端到端 total S4096 2.177→**2.148ms**、S1024H32 0.4895→
> **0.4464**、GQA kv4 0.4566→**0.4056**、MLA S1024H2 0.4653→**0.3900ms**，全 shape 不回退。
> ncu（S1024H32 k=8→4）：`red` 扇区 **26.74M→16.42M（0.61×）**、**L2 83.1%→57.6%**、
> `short_scoreboard` 2.40→1.50、Duration 332→305µs（收益 = 少切 + 自动开 `use_regdq`）。
> **`FA_ILV34`（先发 GEMM3/4 两条 mma 再 epilogue）负结果**：两个累加器同时存活使 spill 增加，
> 0.95–0.96×。详见 `03` §34、原始输出 `src/fp8/fa_bwd_fp8_o29_ksplit_sweep.out.txt`、
> `fa_bwd_fp8_o29_ncu_s1024h32.out.txt`、`o29_te_fp8_bench.out.txt`。

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
| bf16 | **lse_mma_bal<128,1>（O8b, S=4096）** | 3.07% | 32.40% | **60.40%** | 22.91%（52.2KB, 4 CTA/SM, 64 regs） | **0.97** | long 0.34、wait 1.57、short 1.33 | 同 fp16（与 fp16 逐项一致） |
| fp16 | **lse_mma_bal_wgmma<128,1>（O9a, S=4096）** | 3.77% | **18.05%**（ldmatrix 消失） | **60.66%** | 23.02%（50.2KB, 4 CTA/SM, 62 regs） | 0.97 | Executed Ipc 2.52（O8b 2.36） | **Compute 60% + 发射**（wgmma 打掉访存一半，但 LSE 是 softmax epilogue bound） |
| fp16 | **delta（O8, S=4096）** | 24.80% | 73.50% | 71.91% | 71.90%（17 regs） | 31.03 | — | 访存/算力均衡的轻量归约（<1% 端到端） |
| fp16 | **delta_warp（O24 warp-per-row, S=4096）** | **72.87%** | 25.84% | 29.42% | 67.94%（24 regs） | 7.76 | —（**4.19M inst, −82%**） | **DRAM 带宽 73%（elementwise 上限）**；Duration 42.5→**14.5µs（3.35×）**（对齐 fp8 O14 的 quant） |
| fp16 | **mma main（O6, S=4096, cp.async 双缓冲）** | 3.31% | 65.19% | 27.07% | 11.83%（**83.97KB, 2 CTA/SM**, 182 regs） | 3.88 | **long_scoreboard 7.35→1.12**；wait 1.95、short_scoreboard 0.79、barrier 0.10 | **fixed-latency(`wait`) + short_scoreboard(smem→ldmatrix) + L1/TEX**（全局访存延迟已被 cp.async 消掉） |
| bf16 | **mma main（O6, S=4096, cp.async 双缓冲）** | 3.43% | 64.96% | 25.49% | 11.79%（83.97KB, 2 CTA/SM） | 3.88 | long_scoreboard 同上降到 ~1、wait 主导 | 同 fp16（与 fp16 逐项一致） |
| fp16 | **mma main（O6b, S=4096, K 双缓冲+A 转置读）** | 3.47% | 71.87% | 29.77% | 16.90%（**71.17KB, 3 CTA/SM**, 168 regs） | 2.59 | wait 1.88、long_scoreboard 1.79、short 0.81、not_selected 0.37 | **L1/TEX 吞吐 + L2 吞吐 + fixed-latency(`wait`)**（occ 升但吞吐受限） |
| bf16 | **mma main（O6b, S=4096, K 双缓冲+A 转置读）** | 3.32% | 71.71% | 31.42% | 16.86%（71.17KB, 3 CTA/SM） | 2.59 | 同 fp16（与 fp16 逐项一致） | 同 fp16 |
| fp16 | **mma main（O6c, S=4096, (64,64,2)）** | 3.24% | 57.47% | 27.04% | 11.78%（**105.47KB, 2 CTA/SM**, 242 regs） | 3.88 | wait 1.88、long_scoreboard 1.79、short 0.81 | **L1/L2 吞吐 + `wait`**（BN=64 降 L1 但掉 occ） |
| fp16 | **mma main（O7c=PREL, S=4096, (64,64,2)）** | 4.01% | 55.73% | 23.37% | 11.84%（105.47KB, 2 CTA/SM, 250 regs） | 3.88 | wait 2.00、**long 1.79→1.38**、short 0.88、mio 0.49 | **L2 71.6%（新墙，残余 red）+ `wait` + 低 occupancy**（LSE/D 全局读已消；float4 red 更慢 ⇒ 非事务数 bound） |
| bf16 | **mma main（O7c=PREL, S=4096, (64,64,2)）** | 3.95% | 55.69% | 23.38% | 11.81%（105.47KB, 2 CTA/SM, 250 regs） | 3.88 | wait 2.00、long 1.39、short 0.88、mio 0.50 | 同 fp16（与 fp16 逐项一致） |
| fp16 | **mma main（O13, S=512, (64,64,2)）** | 8.5% | **20.2%** | 10.7% | 6.23%（grid=128<132 SM，1 CTA/SM） | — | wait 2.00、long 1.01、short 0.46 | **尾波/grid-bound + fixed-latency(`wait`)**（O6c 旧 auto `(32,32,1)` 同点 83.9µs→**59.7µs**、L1/TEX 43.5→20.2%） |
| fp16 | **wgmma main（O9b, S=4096, GEMM1/2 wgmma）** | 4.55% | 55.08% | 26.48% | 11.88%（**101.38KB, 2 CTA/SM**, 242 regs） | 3.88 | **wait 1.94→1.50**、long 1.45→1.25、short 0.56 | **L2 74.4%（dK/dV 原子，仍为墙）+ `wait` + 低 occupancy**（GEMM1/2 的 wgmma 打掉 ldmatrix/依赖，但 GEMM3/4/5 仍 mma） |
 | fp16 | **wgmma main（O9b-2, S=4096, 5 GEMM 全 wgmma）** | 4.35% | 40.05% | 23.37% | 11.86%（**99.33KB, 2 CTA/SM**, 230 regs） | 3.88 | short 0.29（ldmatrix 消）、long 2.01、wait 1.43、barrier 0.84 | **L2 68.8%：`red`（dK/dV `atomicAdd`）占 102.2M/139.8M=73.1% 扇区、DRAM 4.2%** ⇒ **L2 原子字节数 bound**（见下） |
  | fp16 | **wgmma2 main（O17, BM=128, 2 wg, S=4096）** | 6.47% | 36.85% | 27.48% | 12.41%（**149.5KB, 1 CTA/SM**, 200 regs, 256 thr） | — | Duration 1.48→**0.996ms** | **L2 54.7%（red 51.9M=0.508×/O9b、read 也 0.50×）**：跨 wg 归约把 dK/dV 的 red 字节精确砍半，但仍是第一墙（red 占 L2 扇区 ~72.6%）；bank conflict 0 |
   | bf16 | **wgmma2 main（O17-bf16, BM=128, 2 wg, S=4096）** | 6.47% | 36.68% | 27.36% | 12.41%（**149.50KB, 1 CTA/SM**, 200 regs, 256 thr） | — | Duration 1.48→**0.997ms** | **L2 54.63%（`red` 102,236,160→51,904,512=0.508×、`read` 0.50×）**：与 fp16 O17 逐项一致；新墙仍是 L2（red 占 ~72.6%）；bank conflict 0 |
   | fp16 | **wgmma2 main（O17-2, GEMM3/4 拆分, S=4096）** | 6.57% | 49.46% | 27.75% | 12.48%（148.48KB, 1 CTA/SM, 200 regs, 256 thr） | 3.88 | **`barrier` 1.61→0.46（−3.5×）**、wait 1.19、long 0.75、short 0.40 | **L2 red 完全不变（51,904,512）** ⇒ 收益来自消 wg1 在 GEMM3/4 的 barrier 空等（张量工作量 3:1→1:1）；Duration 997.7→**982.4µs**，墙仍是 **L2 red + `wait`** |
   | fp16 | **wgmma2b main（O18, BN=128, S=4096）** | 6.77% | **44.34%** | 23.60% | 12.50%（**224.0KB, 1 CTA/SM**, 230 regs, 256 thr） | 3.88 | wait 1.25、**barrier 0.46→0.93**、long 0.60、short 0.43 | **Duration 982.4→951.1µs（1.033×）**：tile 数减半摊薄 barrier/wgmma 序列；**`red` 51,904,512 逐字节不变**（BN 不动归约结构）；墙仍是 **L2 57.2%（red 占 ~72%）+ `wait` + 低 occ** |
 | fp8 | golden main | 0.06% | **75.96%**（90% 多余） | 4.98% | 6.25%（68KB） | 0.32 | MIO scoreboard 69% | **smem 冲突 + FP8 解码 + 低 occ** |
| fp8 | **mma main（O2b+O4d 后, S=4096, ksplit=4）** | 1.41% | 69.91% | 21.70% | **18.27%（73.8KB, 3 CTA/SM）** | 10.34 | No Eligible 76.6%、long_scoreboard 4.46 + short_scoreboard 3.96 | **L2 带宽（81.5%）+ 延迟**（split-K 复读 Q/dO + 全局 atomic） |
| fp8 | **mma main（O4c 后, S=4096, ksplit=4）** | 1.99% | **81.30%** | 29.42% | 18.20%（73.8KB, 3 CTA/SM） | 10.34 | short_scoreboard 3.50、long_scoreboard 1.44 | **L1/TEX 81.3% + short_scoreboard**（全局 red 流量已减半，L2 退到 57.9%） |
| fp8 | **mma main（O4b 后, S=4096, ksplit=4）** | 2.46% | **69.69%** | 34.40% | 18.21%（**70.66KB**, 3 CTA/SM） | 10.34 | short_scoreboard 2.73、long_scoreboard 1.16 | **L1/TEX 69.7% + L2 69.1%（残余 red）+ short_scoreboard**（`op_st` 冲突 −66%） |
| fp8 | **mma main（O7 后, S=4096, ksplit=4, REGDQ=true）** | 2.60% | **64.4%** | 37.7% | 18.11%（70.66KB, 3 CTA/SM） | 10.34 | short_scoreboard 1.89、long_scoreboard 1.09 | **L1/TEX 64.4% + short_scoreboard 1.89 + 残余 L2 43.8%（dK/dV 跨 CTA red）**（dQ red 已 O(1)，L2 墙 69.1%→43.8%） |
| fp8 | **mma main（O12=PREL 后, S=4096, ksplit=4, REGDQ=true）** | 2.60% | 65.92% | 40.94% | 18.10%（70.66KB, 3 CTA/SM） | 10.34 | short_scoreboard、long_scoreboard | **L1/TEX 65.9% + 残余 L2（dK/dV red）**（LSE/D 全局散读已消：uncoalesced global 38.0M→**4.70M（−87.6%）**、Executed Instructions **−8.4%**、Duration 2.60→**2.34ms**；对齐 fp16 O7c-PREL） |
| fp8 | **mma main（O7e-2 后, S=4096, ksplit=4, REGDQ=true, F16B）** | 3.22% | **59.97%** | 41.81% | 18.08%（70.66KB, 3 CTA/SM） | 10.34 | short/long_scoreboard | **L1/TEX 60.0% + L2 49.9%（dK/dV 跨 CTA red）+ spill(~2.7M local)**；shared load 冲突 **62.8M→29.2M（−53.5%，fold 列读改无冲突映射）**、总多余 wavefronts 79.9M→41.5M、Duration 2.41→**2.27ms** |
| fp8 | **mma main（O21, S=4096, BN=64）** | 2.45% | 39.77% | 31.55% | **12.28%（105.22KB, 2 CTA/SM, 255 regs）** | — | —（Warp Cyc/Inst 5.92） | **负结果**：tile 相位减半但 occupancy 3→2 CTA/SM ⇒ Duration 2.19→**2.74ms**（0.816×）；与 O19 一起证伪 fp8「放大 tile/减 red」 |
| fp8 | **mma main（O27 fold rcp-mul, S=4096, ksplit=4）** | — | 61.64% | 43.65% | 18.1%（70.66KB, 3 CTA/SM, 168 regs） | 10.34 | wait 1.48、short_scoreboard 1.43、long 0.87、barrier 0.27 | **Duration 2.06→1.78ms、executed inst 852.6M→735.2M（−13.7%）**；墙仍是 **mma 依赖延迟（wait+short）+ 3 CTA/SM**（O7e-3/O19/O20/O22 结论未变），但 fold 的精确除法（~10+ 指令/元素）换成每行一次 `__frcp_rn`+乘法 |
| fp8 | **quant 旧（per-row, S=4096）** | 24.29% | **73.73%** | **71.80%** | 87.82%（Waves 31.03） | 31.03 | —（34.6M inst） | **smem 归约（L1/TEX 73.7%）+ 标量加载（Compute 71.8%）**，DRAM 仅 24% |
| fp8 | **quant 新（O14 warp-per-row, S=4096）** | **71.31%** | 26.50% | 51.49% | 79.44%（Waves 7.76） | 7.76 | —（**7.93M inst, −77%**） | **DRAM 带宽 71%（elementwise 上限）**；Duration 46.2→**15.4µs** |
| fp8 | **delta 旧（per-row, S=4096）** | 31.13% | — | **74.93%** | 83.09%（17 regs） | 31.03 | smem 树归约 7×barrier | **smem 归约 + Compute 75%**；Duration **42.94µs** |
| fp8 | **delta_warp（O26 warp-per-row, S=4096）** | **77.01%** | — | 27.14% | 83.77%（18 regs） | 7.76 | — | **DRAM 带宽 77%（elementwise 上限）**；Duration 42.94→**17.34µs（2.48×）**（对齐 fp16 O24） |
| fp8 | **lse_mma_bal<HD,1>（O11, S=4096）** | 1.48% | 28.16% | **61.15%** | 23.24%（~27.7KB, 6 CTA/SM, 77 regs） | **0.65** | — | **Compute 61% + 网格不足一个波**（镜像配对消尾波 + cp.async 消 long_scoreboard；对齐 fp16/bf16 O8b） |
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

> **O7c（fp16/bf16，已做）把「减 red 事务数」这条杠杆证伪**：dK/dV 从 float2 提到 float4
> （quad `shfl` → `REDG.E.ADD.F32x4`，事务数减半）后**四个几何全变慢 1–7%**，说明该归约
> 不是事务数 bound。真正有用的是 **PREL**：`lse`/`delta` 只依赖 CTA 自己的 Q 行、与 K tile
> 无关，预装寄存器后 main **+14–19%**、端到端 S=4096 **2.376→2.094ms（1.13×）**，`long_scoreboard`
> 1.79→1.38。**新墙 = L2 71.6%（残余 dK/dV `red`）+ `wait` + 低 occupancy（105KB smem / 250 regs
> 锁死 2 CTA/SM）**。要同时拿低 L1/L2 与高 occupancy，只有**更低 smem 的数据通路（O9：wgmma+TMA）**
> 这条路；继续抠 red 宽度或 tile 几何的边际收益已很小。

> **O13（第四十轮，fp16/bf16）**：修正了 O6c 的过时 auto tile——S=512 MHA 改用 `(64,64,2)`
> （旧 `(32,32,1)`）；`BN=64` 判据改为 `S≥4096||grid≤256||grid>600`（`256<grid≤600` 用 BN=32
> 避开 2-波坏量化点）；HD=128 去掉 dQ 的 memset、`convert` 向量化。端到端 S512 **1.12×**
> （fp16）/ **1.14×**（bf16），S1024 部分 shape 1.03–1.06×，数值逐位不变。
>
> **下一步优先级**：**① O9（wgmma+TMA+warp specialization，对标 FA3）——当前唯一能同时
> 降 smem 与提 occupancy 的杠杆；② fp8 侧残余 dK/dV 跨 CTA red（O7b，分块 `*_accum`+convert）；
> ③ MLA 的 KV 分片/降 smem + 张量核。**

> **O15a/O16（第五十一轮）——把 fp16 main 的墙定量钉在 L2 原子**：ncu 把 S=4096 主 kernel
> 的 L2 扇区拆开——**`red`（dK/dV 的跨 CTA `atomicAdd`）102.2M / 139.8M = 73.1%**、
> `read` 25.6%、`write` 1.1%，`L2 Hit 96.2%`、DRAM 仅 4.2% ⇒ 主 kernel 是 **L2 原子字节数
> bound**。同轮：`src/fp16/fa_bwd_fp16_tma_smoke.cu` 用 `cp.async.bulk.tensor`(SWIZZLE_128B)
> + mbarrier 搬 [64][128] tile，**逐字节 == `sw128_off` 布局、wgmma 对拍 max_abs=0（PASS）**，
> 建好 TMA 通路（发现：TMA box 内维 128B=fp16 的 64 元素 ⇒ HD=128 tile 必须拆 2×K=64 chunk）。
> 另做 **O16**（分段 `wgmma.wait_group` 重叠 epilogue）**实测中性 0.99–1.00×**（`dq` 逐位相同，
> `dk/dv` 仅 atomic 次序差 ~1e-4）⇒ 动搬运/等待打不动原子墙。**唯一真杠杆 = 跨 warpgroup 归约
> （BM=128，一个 KV 元素被 nblk/2 个 CTA 贡献，red 字节砍半）**；详见 `docs/01` §14i。

> **O17（第五十二轮）——把这个「唯一真杠杆」落地并再次用 ncu 证实**：`fa_bwd_fp16_wgmma2_kernel`
> （BM=128、2 warpgroups、256 线程、仅 HD=128）：两组各算自己 64 行 Q 的 P/dS，**dK/dV 只由 wg0
> 对全 BM=128 归约**（两个 m64 半进同一 wgmma 累加器 = 两半之和），每个 KV 元素只 `red` 一次。
> ncu：`red` 102.2M→**51.9M（0.508×）**、`read` 0.50×、L2 71.8%→**54.7%**、Duration 1.48→
> **1.00ms**；main-only S=4096 **1.57×（140.5 TF）**、GQA kv4 1.47×、MQA kv1 1.54×、S=512 1.11×；
> 端到端 S=4096 1.427ms、为 FA3 的 **4.4×**（O9b ~6.0×）。数值 vs ref 历史逐位一致。**新墙仍是
> L2（red 占 ~72.6%）** ⇒ 下一步 O17b（BM=256/4 wg）。需 `--wg2`（`sm_90a`+`-DFA_WGMMA`）。
> 详见 `docs/01` §14j。

> **O17-bf16（第五十三轮）——逐字 dtype 参数化**：`fa_bwd_bf16_wgmma2_kernel`（同为 2 字节，
> SW128/描述符/`m64n64k16` 累加器映射逐字节同构）。ncu 与 fp16 O17 **逐项一致**：`red`
> 102,236,160→**51,904,512（0.508×）**、`read` 0.50×、L2 71.61%→**54.63%**、Duration 1.48→
> **0.997ms**；main-only S=4096 **1.516×（139.9 TF）**、GQA kv8 1.504×、GQA kv4 1.506×、
> MQA kv1 1.531×、S=512 1.109×；端到端 S=4096 **1.4309ms（96.05 TF）**、为 FA3 的 **4.45×**。
> 数值 vs ref 历史逐位一致。详见 `docs/01b` §6s。

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
- **O7c（本轮）**：`src/fp16/fa_bwd_fp16_mma_{main,onefile}_o7c_*.out.txt`、
  `src/fp16/fa_bwd_fp16_mma_main_o7c_ncu_s4096.out.txt` / `..._o7c_stall_s4096.out.txt` /
  `src/fp16/fa_bwd_fp16_o7c_fa3_te_baseline.out.txt`；bf16 同构文件在 `src/bf16/`（前缀 `..._o7c_`），
  改动前基线 `fa_bwd_fp16_mma_main_o7c_base_{s512,s4096}.out.txt`。

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
（S=1024H2 0.3933→0.3251 ms）；但即使把三个配对阵列也去掉仍 >116KB，MLA 冲 2 CTA/SM 需继续
降 `Qs/Ks/Vs/dOs/dS2`（见 `03` §17.6）。详见 `docs/03` §14、§17。

### 7.8 ours 的 fp16 / bf16 MLA 张量核（O5c，本节新增）

P5-2/P5-3 的 fp16/bf16 MLA 反向原是**标量 golden**；本轮把 `HD` 模板从 128 扩到 **128/512**：
GEMM1/2 的 k-loop 随 HD 加长，GEMM3/4/5（输出 N 维=HD）加 **N-tile 循环**（每遍 128 列），
HD>128 的 dQ 改**直接全局累加**（每 `(qi,列)` 唯一线程拥有 ⇒ 非原子 RMW 无竞争），
与 fp8 MLA（§7.7）同构。FA/TE 反向不支持 head_dim=512，仍只有 fp32 ref 与 ours。

**数值对拍（ours vs fp32 ref，B1 D=Dv=512 causal，max_abs dq/dk/dv）**：

| MLA case | fp16 | bf16 |
|---|---|---|
| (1,256,2,512) | 1.638 / 1.582 / 1.753e-3 | 1.230e-2 / 9.875e-3 / 1.50e-2 |
| (1,512,4,512) | 2.516 / 2.916 / 1.724e-3 | 8.753e-3 / 1.082e-2 / 1.740e-2 |
| (1,1024,2,512) | 1.987 / 1.712 / 1.848e-3 | 5.838e-3 / 9.519e-3 / ~1.5e-2 |

对应 fp16/bf16 噪声量级；MHA D=128 回归**逐位不变**（fp16 S=512 1.671/1.771/1.899e-3、
bf16 S=512 9.001/12.61/13.65e-3）。

**性能（同 session CUDA event；main 加速 = 标量 main / 张量核 main）**：

| MLA case | fp16 标量 main | fp16 TC main | 加速 | bf16 标量 main | bf16 TC main | 加速 |
|---|---|---|---|---|---|---|
| (1,256,2,512) | 1.065 ms | **0.204 ms** | 5.2× | 0.753 ms | **0.204 ms** | 3.7× |
| (1,512,4,512) | 2.120 ms | **0.385 ms** | 5.5× | 1.488 ms | **0.385 ms** | 3.9× |
| (1,1024,2,512) | 4.214 ms | **0.725 ms** | 5.8× | 2.959 ms | **0.725 ms** | 4.1× |

最优配置 `(BM=32,BN=32,PIPE=1)`（K 双缓冲），比 `(32,32,0)` 快 1.67×。ncu（S=1024H2，fp16/bf16
逐项相同）：Duration **769µs**、DRAM 0.83% / L1/TEX 40.1% / L2 13.9% / Compute 2.2%、
168 regs / **207.36KB smem → 1 CTA/SM**、occ 6.25%、**Waves 0.48**、No Eligible 91.6%、
stall `long_scoreboard 7.68 + wait 2.23` ⇒ **bound = 全局访存延迟 + 低并行度**（grid=64 < 132 SM、
1 CTA/SM），非带宽/算力；进一步提速需降 smem 冲 2 CTA/SM 或 split-KV（backlog）。
详见 `docs/01` §14b、`docs/01b` §6l。
