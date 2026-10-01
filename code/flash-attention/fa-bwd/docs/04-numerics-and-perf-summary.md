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
>
> **O30（LSE 的 4D-TMA 载入 Q/K，fp16，第七十一轮）**：O15a 的 TMA 通路（冒烟逐位 PASS）落进
> 真 kernel 的第一步。新增 `lse_mma_kernel_bal_tma<128,1>`（4D 描述符 `dims={D,S,H,B}`、
> 坐标 `{k0,row,head,batch}`、2×K=64 chunk、`SBO=1024`），数学与 `lse_mma_kernel_bal_wgmma`
> **完全一致**。**LSE-only 1.30–1.36×**（S512 0.0328→0.0249、S4096 0.2878→0.2128、
> GQA kv4 0.0640→0.0488、MQA kv1 0.0777→0.0590ms），**TMA-vs-wgmma `max_abs=0`（逐位相同）**，
> 端到端 S4096 total **1.2553ms（O24 1.3309，1.06×）/ 109.5 TF**。ncu：Duration 286.3→**214.6µs**、
> 指令数 165.3M→**117.9M（−28.7%）**、regs 62→58；墙不变（**Compute ~58% + `wait`**，softmax
> epilogue）。需 `-DFA_WGMMA -DFA_TMA -lcuda` 构建（否则不编译/不引用驱动符号）。对标同 session
> 纯反向 FA3 MHA S4096 0.3244ms/847TF ⇒ ours total 时间 **3.87×**（O24 4.10×）、GQA kv4 3.12×。
> 详见 `01` §14r。bf16/fp8 的 TMA 与主 kernel 的 Q/K/V/dO TMA 化列 backlog。
>
> **O33（主 kernel 的 Q/K/V/dO 4D-TMA，逐 atom，fp16，第七十四轮）**：TMA 一个 `[8 行][64 列]`
> box = 一个 1024B SW128 atom，逐 atom 发 TMA 即可**原样复现 `sw128_off` 交织布局** ⇒ 所有
> wgmma 描述符零改动、搬的字节与 cp.async 逐字节相同。`--maintma`（仅 BN=128 的
> wgmma2b；**自 O115 第 209 轮起该几何默认开启**，`--maintma=0` 退回 cp.async）。同 session A/B（S4096）：main **0.9638→0.9249ms（142.6→148.6 TF，1.042×）**、
> 单文件 1.044×；端到端 total **1.2289ms / 111.8 TF**。ncu：Duration 969→**924µs**、
> **`red` 51,904,512 逐字节不变**、regs/smem/occupancy（12.5%, 1 CTA/SM）全不变 ⇒ **TMA 只省
> 搬运那一半，动不了主墙（dK/dV 的 L2 `red`）**。`max|diff|`：dq `0`（逐位）、dk/dv ~2e-5
> （仅 atomic 次序）。对标同 session FA3 MHA S4096 0.3263/842、TE 0.4443/619 ⇒ ours total
> 时间 **3.77×**（O30 3.87×）。详见 `01` §14s。
>
> **O35（BN=64 的 `wgmma2` 也改用逐 atom 4D-TMA，fp16，第七十五轮）**：补全 O33 未覆盖的
> **BN=64 几何**（O23 默认档在 S<4096 / GQA/MQA 走它）。同 session A/B（main-only）：
> S=512 **0.992×**、S=1024 GQA kv4 **0.996×**、S=4096 强制 BN=64 **1.021×**（0.9884→0.9685ms）。
> ncu：TMA 把指令数砍 **−24%**、regs 200→184，但 S=512/1024 是 **延迟/grid bound**
> （Waves 0.48 / grid=128<132 SM）⇒ Duration 持平；S=4096（grid=512、tile 数 4×）才体现
> 1.02×。`max|diff|`：dq `0`（逐位）、dk/dv ~1e-4（仅 atomic 次序）。**S≥4096 默认仍是
> BN=128 的 wgmma2b（O33），O35 仅补几何覆盖、不改默认端到端。** 详见 `01` §14t。

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
>
> **O32（第七十三轮）——fp8 LSE 的 4D-TMA 载入（对齐 fp16 O30/bf16 O31；正结果）**：
> fp8 一行 128B = SW128 atom 整行 ⇒ Q/K 各只需**一次** 4D-TMA（box `{128,64,1,1}`、UINT8
> tensormap；fp16 需 2×K=64 chunk）。LSE-only **1.06–1.10×**（S4096 0.2726→**0.2475ms**），
> `max_abs(tma-vs-wgmma)=0` 逐位不变、`dq/dk/dv vs ref` 与历史逐位一致；端到端 S4096
> **2.1303ms/64.52 TF**、S512 0.1219、S1024H32 0.4468，同一 session TE FP8
> 0.1009/0.2059/0.5876 ⇒ ours/TE **1.21×/2.17×/3.63×**。ncu：`long_scoreboard 2.20→0.37`、
> `short 2.25→1.05`、`mio 0.67→0.03`，新墙 = Compute ~57% + `wait`。详见 `03` §35、
> 原始输出 `src/fp8/fa_bwd_fp8_o32_sweep.out.txt`、`..._o32_ncu_stall_lse_{tma,wgmma}_s4096.out.txt`。
>
> **O37（第七十六轮）——fp8 主 kernel 的 Q/dO 改用 4D-TMA（对齐 fp16 O33 / bf16 O34；正结果）**：
> 把主 kernel 函数体抽成 `fp8_mma_body<...,TMA>`，两个薄壳复用；TMA 版 `cp.async.bulk.tensor.4d`
> 一次性把 Q/dO（box `{128,64}`）搬进 SW128 tile，再从 smem 重建 Qp/dOp 的 K 配对布局。
> K/V/fold/GEMM3/4/5 逐字未动 ⇒ 数值 vs ref 与历史逐位一致、同 session `max_abs(tma-vs-cp)~1e-6`
> （仅原子次序）。**同 binary A/B：main S512 1.075× / S4096 1.050× / GQA 1.045× / MQA 1.074×**；
> 端到端 S4096 **2.0508ms/67.02 TF**，为 **TE FP8（0.5903ms/465.6TF）的 3.49×**（O27 3.69×）；
> ncu：指令数 −2.2%、`long_scoreboard 1.10→0.81`、Duration 1.77→1.66ms，墙仍是
> `short_scoreboard`+`wait`+3 CTA/SM。K/V TMA 因**单缓冲无重叠 + 双缓冲 smem 在 3 CTA/SM 下
> 顶格**，列入 backlog。详见 `03` §36、原始输出 `src/fp8/fa_bwd_fp8_o37_*.out.txt`。

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
    | fp16 | **wgmma2b+tma main（O33, Q/K/V/dO 逐 atom TMA, S=4096）** | 6.46% | 34.78% | 19.73% | 12.46%（230.5KB, 1 CTA/SM, 255 regs, 256 thr） | — | long 0.61→1.37、wait 1.23→1.33、barrier 0.94→1.47 | **Duration 968.96→923.87µs（ncu 1.049×、event main 1.042×）**；**`red` 51,904,512 逐字节不变**、occ/regs/smem 不变 ⇒ **TMA 只省搬运（发射/地址运算），主墙仍是 L2 red** |
    | fp16 | **wgmma2+tma main（O35, BN=64, S=512）** | 9.56% | 44.74% | 7.94% | 12.39%（148.6KB, 1 CTA/SM, 184 regs, 256 thr） | 0.48 | — | **指令数 5,259,008→3,976,256（−24.4%）、regs 200→184，但 Duration 53.02→52.96µs（持平）**：S=512 `grid=128<132 SM` 是**延迟/grid bound**，省发射换不到时间；S=4096 强制 BN=64 时 989.95→959.20µs（1.032×）。墙 = 低并行度/延迟，TMA 动不了 |
    | bf16 | **wgmma2+tma main（O36, BN=64, S=512）** | 9.55% | 44.7%（同 fp16） | — | 12.39%（148.6KB, 1 CTA/SM, 184 regs, 256 thr） | 0.48 | — | 把 O34 的逐 atom TMA 补到 BN=64（对齐 fp16 O35）：**指令数 5,259,008→3,976,256（−24.4%）、regs 200→184、Duration 53.54→53.02µs（持平）**；S4096 强制 BN=64 event **1.024×**（139.0→142.4 TF）、S=512/GQA 0.987–0.997×（延迟/grid bound），`dq` 逐位不变 |
 | fp8 | golden main | 0.06% | **75.96%**（90% 多余） | 4.98% | 6.25%（68KB） | 0.32 | MIO scoreboard 69% | **smem 冲突 + FP8 解码 + 低 occ** |
| fp8 | **mma main（O2b+O4d 后, S=4096, ksplit=4）** | 1.41% | 69.91% | 21.70% | **18.27%（73.8KB, 3 CTA/SM）** | 10.34 | No Eligible 76.6%、long_scoreboard 4.46 + short_scoreboard 3.96 | **L2 带宽（81.5%）+ 延迟**（split-K 复读 Q/dO + 全局 atomic） |
| fp8 | **mma main（O4c 后, S=4096, ksplit=4）** | 1.99% | **81.30%** | 29.42% | 18.20%（73.8KB, 3 CTA/SM） | 10.34 | short_scoreboard 3.50、long_scoreboard 1.44 | **L1/TEX 81.3% + short_scoreboard**（全局 red 流量已减半，L2 退到 57.9%） |
| fp8 | **mma main（O4b 后, S=4096, ksplit=4）** | 2.46% | **69.69%** | 34.40% | 18.21%（**70.66KB**, 3 CTA/SM） | 10.34 | short_scoreboard 2.73、long_scoreboard 1.16 | **L1/TEX 69.7% + L2 69.1%（残余 red）+ short_scoreboard**（`op_st` 冲突 −66%） |
| fp8 | **mma main（O7 后, S=4096, ksplit=4, REGDQ=true）** | 2.60% | **64.4%** | 37.7% | 18.11%（70.66KB, 3 CTA/SM） | 10.34 | short_scoreboard 1.89、long_scoreboard 1.09 | **L1/TEX 64.4% + short_scoreboard 1.89 + 残余 L2 43.8%（dK/dV 跨 CTA red）**（dQ red 已 O(1)，L2 墙 69.1%→43.8%） |
| fp8 | **mma main（O12=PREL 后, S=4096, ksplit=4, REGDQ=true）** | 2.60% | 65.92% | 40.94% | 18.10%（70.66KB, 3 CTA/SM） | 10.34 | short_scoreboard、long_scoreboard | **L1/TEX 65.9% + 残余 L2（dK/dV red）**（LSE/D 全局散读已消：uncoalesced global 38.0M→**4.70M（−87.6%）**、Executed Instructions **−8.4%**、Duration 2.60→**2.34ms**；对齐 fp16 O7c-PREL） |
| fp8 | **mma main（O7e-2 后, S=4096, ksplit=4, REGDQ=true, F16B）** | 3.22% | **59.97%** | 41.81% | 18.08%（70.66KB, 3 CTA/SM） | 10.34 | short/long_scoreboard | **L1/TEX 60.0% + L2 49.9%（dK/dV 跨 CTA red）+ spill(~2.7M local)**；shared load 冲突 **62.8M→29.2M（−53.5%，fold 列读改无冲突映射）**、总多余 wavefronts 79.9M→41.5M、Duration 2.41→**2.27ms** |
| fp8 | **mma main（O21, S=4096, BN=64）** | 2.45% | 39.77% | 31.55% | **12.28%（105.22KB, 2 CTA/SM, 255 regs）** | — | —（Warp Cyc/Inst 5.92） | **负结果**：tile 相位减半但 occupancy 3→2 CTA/SM ⇒ Duration 2.19→**2.74ms**（0.816×）；与 O19 一起证伪 fp8「放大 tile/减 red」 |
| fp8 | **mma main（O27 fold rcp-mul, S=4096, ksplit=4）** | — | 61.64% | 43.65% | 18.1%（70.66KB, 3 CTA/SM, 168 regs） | 10.34 | wait 1.48、short_scoreboard 1.43、long 0.87、barrier 0.27 | **Duration 2.06→1.78ms、executed inst 852.6M→735.2M（−13.7%）**；墙仍是 **mma 依赖延迟（wait+short）+ 3 CTA/SM**（O7e-3/O19/O20/O22 结论未变），但 fold 的精确除法（~10+ 指令/元素）换成每行一次 `__frcp_rn`+乘法 |
| fp8 | **mma main（O41 K/V-TMA, S=4096, ksplit=4）** | 4.24% | 69.18% | 47.25% | 18.35%（**74.82KB, 3 CTA/SM**, 168 regs） | 20.69 | wait 1.53、short 1.30、long **0.55**、barrier 0.42 | **mma 依赖延迟（wait+short）+ L2 77.9%**；K/V TMA 消掉 K/V 全局读地址运算（long 0.80→0.55、short 1.61→1.30），`red` 逐字节不变，3 CTA/SM 保住（smem 74816B）；O37 的 Q/dO-TMA 对照 Duration 1.67→**1.61ms** |
| fp8 | **quant 旧（per-row, S=4096）** | 24.29% | **73.73%** | **71.80%** | 87.82%（Waves 31.03） | 31.03 | —（34.6M inst） | **smem 归约（L1/TEX 73.7%）+ 标量加载（Compute 71.8%）**，DRAM 仅 24% |
| fp8 | **quant 新（O14 warp-per-row, S=4096）** | **71.31%** | 26.50% | 51.49% | 79.44%（Waves 7.76） | 7.76 | —（**7.93M inst, −77%**） | **DRAM 带宽 71%（elementwise 上限）**；Duration 46.2→**15.4µs** |
| fp8 | **delta 旧（per-row, S=4096）** | 31.13% | — | **74.93%** | 83.09%（17 regs） | 31.03 | smem 树归约 7×barrier | **smem 归约 + Compute 75%**；Duration **42.94µs** |
| fp8 | **delta_warp（O26 warp-per-row, S=4096）** | **77.01%** | — | 27.14% | 83.77%（18 regs） | 7.76 | — | **DRAM 带宽 77%（elementwise 上限）**；Duration 42.94→**17.34µs（2.48×）**（对齐 fp16 O24） |
| fp8 | **lse_mma_bal<HD,1>（O11, S=4096）** | 1.48% | 28.16% | **61.15%** | 23.24%（~27.7KB, 6 CTA/SM, 77 regs） | **0.65** | — | **Compute 61% + 网格不足一个波**（镜像配对消尾波 + cp.async 消 long_scoreboard；对齐 fp16/bf16 O8b） |
| fp8 | **lse_mma_kernel_bal_tma<128,1>（O32, S=4096）** | 2.08% | 25.00% | **56.79%** | 23.55%（26.43KB, 61 regs） | 0.48 | long 2.20→**0.37**、short 2.25→**1.05**、wait 2.34、mio 0.03 | **Compute ~57% + `wait`**（4D-TMA 消 cp.async 地址运算，对齐 fp16 O30/bf16 O31；Duration 251.97µs） |
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

## 8. 变长（VARLEN / `cu_seqlens`）：fp8 → fp16 → bf16（本节新增）

packed `[T,H,D]` + `cu_seqlens` 逐序列只算 `len_b×len_b` 因果注意力，避免 padding 到
`max_b len_b`。fp8 已在第 77 轮完成（`docs/03` §37）；本轮补 **fp16（`docs/01` §16）与
bf16（`docs/01b` §6aa）**，均为默认 D=128 的 wgmma2 主 kernel + wgmma LSE，非 TMA。

**数值对拍（ours vs fp32 ref，causal，max_abs dq/dk/dv）**：

| varlen case | lengths | fp16 | bf16 |
|---|---|---|---|
| b4_t3840 不齐 | `[512,1024,2048,256]` | 3.163 / 2.158 / 1.966e-3 | 1.340 / 1.276 / 1.911e-2 |
| b4_t4096 等长 | `[1024]×4` | 2.112 / 2.252 / 1.915e-3 | 1.355 / 1.207 / 1.772e-2 |
| b5_t3968 GQA kv8 | `[128,256,512,1024,2048]` | 2.438 / 3.433 / 3.843e-3 | 1.398 / 2.396 / 3.131e-2 |
| b8_t2904 强倾斜 | `[2048,512,…,8]` | 2.624 / 2.158 / 2.139e-3 | 1.464 / 1.566 / 1.863e-2 |

全部落在对应 dtype 噪声量级、无 padding 泄漏；单/两文件**逐位相同**，定长回归逐位不变。

**性能（ours 端到端 total，event；`Σ_b 4HL²D` 口径）与对标（等长 `[1024]×4` ≡ 定长 B4 S1024
H16 D128，统一 `8BS²HD`）**：

| case | fp16 ms / TF | bf16 ms / TF |
|---|---|---|
| b4_t3840 | 0.7764 / 58.77 | 0.7756 / 58.84 |
| b4_t4096（等长） | **0.4497 / 76.40** | **0.4507 / 76.24** |
| b5_t3968 GQA | 1.4388 / 63.62 | 1.4279 / 64.10 |
| b8_t2904 倾斜 | 0.5839 / 62.96 | 0.5835 / 63.00 |

等长 case 同口径：ours **152.8 TF**（fp16）; FA2.7.4 274.7、**FA3 471.5**、TE 390.2 ⇒
ours 时间 = FA3 的 **3.09×**、TFLOPS 32%。ncu（main，b4_t3840）：Duration 549µs、
DRAM 10.1% / **L1/TEX 48.7% / L2 63.2%** / Compute 31.0%、202 regs、**1 CTA/SM（occ 12.4%）**、
Waves 7.76 ⇒ bound = **L2 red + L1/TEX + 低 occupancy**，与定长 wgmma2 一致。

## 9. 变长（VARLEN）的非 causal（full attention）——fp8 / fp16 / bf16

第 77/78 轮的 varlen 只做 causal。本轮补 **非 causal（full）**：每个 m 块的 K 列数恒为 `len`
（不再随 `mblk` 递增）⇒ 工作天然均衡、无需镜像配对，LSE 从「causal 专用镜像配对 wgmma 版」
切到通用 mma `lse_mma_kernel`（加 `cu_seqlens` 默认参数，`nullptr` 时定长逐位不变）。
主 kernel 本就带 `causal`，无需改。三 dtype 单/两文件均完成，device 逐字一致。

> **O68（第 162 轮）更正**：上句里 fp8 的「通用 mma `lse_mma_kernel`」实为 **O1 版（无 `cp.async`
> 流水）**，是唯一没跟上 O11/O54 均衡/流水改造的分支。O68 把 **fp8 的定长 full 与 varlen full
> D=128** 都改走 O54 的 `lse_mma_kernel_bal<128,1,FULL=true>`（一个 CTA 一个 m 块 + `cp.async`
> 双缓冲）：定长 S1024 full **total 0.539→0.374ms（1.44×）**、varlen `[1024]×4` full
> **1.559→1.229ms（1.27×）**（下表 fp8 行随更新），LSE 本身 **208→42µs（4.93×）**、数值只差
> LSE 的 fp32 求和次序（~1e-3）。详见 `docs/03` §94。

**数值对拍（ours vs fp32 ref，full，max_abs dq/dk/dv）**：

| varlen case | lengths | fp16 | bf16 | fp8 |
|---|---|---|---|---|
| b4_t3840 不齐 | `[512,1024,2048,256]` | 5.60/6.84/2.34e-4 | 5.76/3.85/3.03e-3 | 1.01/0.97/0.70e-1 |
| b4_t4096 等长 | `[1024]×4` | 4.09/4.95/1.23e-4 | 3.24/2.39/2.01e-3 | 0.89/0.70/0.57e-1 |
| b5_t3968 GQA kv8 | `[128,256,512,1024,2048]` | 7.34/7.91/4.88e-4 | 5.32/5.45/5.88e-3 | 1.43/1.60/1.08e-1 |
| b8_t2904 强倾斜 | `[2048,512,…,8]` | 1.49/1.56/1.58e-3 | 11.5/9.53/10.8e-3 | 2.41/2.06/2.51e-1 |

全部对应 dtype 噪声量级、无 padding 泄漏；单/两文件逐位一致；定长回归逐位不变
（fp16 S512 `1.671/1.771/1.899e-3`、bf16 `9.001/12.61/13.65e-3`、fp8 `2.426/2.972/3.733e-1`）。

**性能（ours total，event，`Σ_b 4HL²D` 口径）与等长对标**：非 causal 总量约为 causal 的 2×，
故时间是 causal 的 ~2×。等长 `[1024]×4` ≡ 定长 B4 S1024 full H16 D128，按 `4BS²H(D+Dv)`：

| dtype | ours total ms | ours TF（口径换算） | TE 定长 full | FA2.7.4 定长 full |
|---|---|---|---|---|
| fp16 | 0.9090（0.450 causal 的 2.02×） | 75.6 | 0.2805 / 245.0TF | 0.4740 / 145.0TF |
| bf16 | 0.9088 | 75.6 | 0.2787 / 246.6TF | 0.4663 / 147.4TF |
| fp8 | **1.2293**（O68，旧 1.5800） | **55.9** | 0.4316 / 159.2TF（含 forward） | — |

ncu（main，b4_t3840）：fp16 Duration 705µs / DRAM 7.9% / L1TEX 49% / **L2 70.3%** /
Compute 35.8% / 1 CTA/SM（occ 12.5%）；fp8 Duration 1.20ms / DRAM 5.1% / L1TEX 62.5% /
L2 64.0% / Compute 40.9% / 3 CTA/SM（occ 18.4%）⇒ bound 与各 dtype 定长 main 一致
（fp16/bf16 = L2 red + 1 CTA/SM；fp8 = L1/L2 吞吐 + 3 CTA/SM），非带宽。

详见 `docs/03` §38（fp8）、`docs/01` §16.8（fp16）、`docs/01b` §6aa.6（bf16）。

## 10. 变长（VARLEN）的 MLA（head_dim=512）——fp8（本节新增，第 80 轮）

第 77–79 轮的 varlen 覆盖 fp8/fp16/bf16 × causal/full，但都限 **HD=128（MHA/GQA）**。
本轮把 **fp8 的 packed varlen 扩到 MLA（HD=512）**：`lse_mma_kernel_bal<HD>` 加 `cu_seqlens`，
`run_varlen` 按 D 分派（含 `quantize_row_warp_kernel` 的 `VPT=D/32` 必须随 D 切：512→16）。
主 kernel `fp8_mma_body` 已是 HD 参数化且第 77 轮带 cu ⇒ 无需改。单/两文件逐位一致。

**数值对拍（ours vs fp32 ref，fp8，max_abs dq/dk/dv）**：

| MLA varlen case | lengths | causal | full |
|---|---|---|---|
| b1_t512（H2 D512） | `[512]` | 1.61/2.24/3.86e-1 | 5.26/5.22/4.22e-2 |
| b3_t1792（H2 D512） | `[256,512,1024]` | 3.40/3.44/3.51e-1 | 7.99/9.63/4.15e-2 |

fp8 噪声量级；定长 D=512 回归逐位不变（MLA S1024H2 `2.232/3.337/3.602e-1`）。
FA/TE 反向均不支持 head_dim=512 ⇒ **无外部基线**，只能对 fp32 ref。

**性能（ours total，event，`Σ_b 4HL²D` 口径；MLA 的 `D=Dv=512`，×2 才是 `4BS²H(D+Dv)`）**：

| 形态 | case | total ms | TF（×2 口径） |
|---|---|---|---|
| varlen | b1_t512 causal | 0.1866 | 11.5 |
| varlen | b3_t1792 causal | 0.5875 | 19.2 |
| **定长对照** | S512 H2 D512 causal | 0.1836 | 11.7 |
| **定长对照** | S1024 H2 D512 causal | 0.3919 | 21.9 |

**varlen b1 与同 shape 定长仅差 1.6%** ⇒ varlen kernel 无额外固定开销。
ncu（main，b3 causal）：255 regs / **1 CTA/SM、occ 6.25%** / No Eligible 84.7% /
short_scoreboard ~30% / DRAM 2.6% / Compute 8.0% / L1TEX 22.8% ⇒ bound =
**低 occupancy + smem→mma 依赖**（非带宽/算力）；LSE `Waves 0.18`（grid 48<132 SM）⇒ 并行度 bound。
详见 `docs/03` §39。

## 11. 变长（VARLEN）的 MLA（head_dim=512）——fp16 / bf16（本节新增，第 81 轮）

把第 80 轮 fp8 的 varlen MLA 补到 **fp16/bf16**。device 改动 2 处：`lse_mma_kernel_bal<HD>`
（O8b mma 镜像配对 LSE，D=512 的 causal 无 wgmma 快路）与 `fa_bwd_{fp16,bf16}_mma_kernel`
（HD=128/512 通用 mma 主 kernel）各加默认参数 `const int* cu_seqlens`；`nullptr` 逐式退化 ⇒
**定长逐位不变**。host `run_varlen` 按 D 分派（D=512 走 `launch_bwd_mma<512,32,32,1,false,true>`
+ `lse_mma_kernel_bal<512,1>`/`lse_mma_kernel<512>`）。单/两文件 device 逐字一致。

**数值对拍（ours vs fp32 ref，max_abs dq/dk/dv）**：

| MLA varlen case（H2 D512） | dtype | causal | full |
|---|---|---|---|
| b3_t1792 `[256,512,1024]` | fp16 | 2.42/1.83/1.86e-3 | 5.52/4.45/2.39e-4 |
| b3_t1792 `[256,512,1024]` | bf16 | 1.27/1.22/1.80e-2 | 3.10/3.53/2.32e-3 |

同 dtype 噪声量级；**单/两文件逐位一致**。定长 D=512 回归逐位不变（S1024H2 causal fp16
`1.987/1.712/1.848e-3`、bf16 `5.838/9.519/1.568e-2`）；HD=128 varlen full 回归 b4_t3840
fp16 `5.603/6.841/2.338e-4` 与第 79 轮逐位相同。FA3/TE 反向不支持 head_dim=512 ⇒ 无外部基线。

**性能（ours total，event，`Σ_b 4HL²D` 口径）**：

| 形态 | case | fp16 ms/TF | bf16 ms/TF |
|---|---|---|---|
| varlen causal | b3_t1792 `[256,512,1024]` | 0.9382 / 6.01 | 0.9347 / 6.03 |
| varlen full | b3_t1792 | 1.4652 / 3.85 | 1.4690 / 3.84 |
| varlen causal | b1_t512 `[512]` | 0.4264 / 2.52 | 0.4258 / 2.52 |
| varlen full | b1_t512 | 0.5511 / 1.95 | 0.5471 / 1.96 |
| **定长对照 causal** | S512 H2 D512 | 0.4232 / 2.54 | 0.4247 / 2.53 |

**varlen b1 与同 shape 定长仅差 +0.8%（fp16）/ +0.3%（bf16）** ⇒ 无额外固定开销。
ncu（fp16 b3 causal）：main Duration 784µs / **1 CTA/SM、occ 6.25%** / No Eligible 89.0% /
DRAM 1.6% / Compute 4.4% / L1TEX 38.7% ⇒ **低 occupancy + smem→mma 依赖**；lse `Waves 0.36`
（grid 48<132 SM）⇒ **并行度 bound**。详见 `docs/01` §16.9、`docs/01b` §6ab。

### 11.1 O52（第 99 轮）：varlen MLA 补上 8-warp（+ fp8 K/V 回填流水）

O46/O47/O51 的 MLA 优化此前只在**定长** `D=512` 路径生效，`run_varlen` 的 MLA 主 kernel
仍是 4-warp/2×2。O52 只改 host（`run_varlen` 的 `D==512` 分支选 8-warp/256 线程，fp8 再加
O51 的 K/V `cp.async` 回填），device 未改 ⇒ 单/两文件 device 逐字一致、数值与历史同量级
（`max_abs(8w-vs-4w)` ≤1e-6，仅 dK/dV atomic 次序）。

| varlen causal（H2 D512） | O52 前 total | O52 total | 加速 |
|---|---|---|---|
| b1_t512（fp8，TF） | 0.1866 / 5.75 | **0.1006 / 10.68** | 1.86× |
| b3_t1792（fp8） | 0.5875 / 9.60 | **0.2987 / 18.87** | 1.97× |
| b1_t512（fp16） | 0.4264 / 2.52 | **0.2740 / 3.92** | 1.56× |
| b3_t1792（fp16） | 0.9382 / 6.01 | **0.6523 / 8.64** | 1.44× |
| b1_t512（bf16） | 0.4258 / 2.52 | **0.2747 / 3.91** | 1.55× |
| b3_t1792（bf16） | 0.9347 / 6.03 | **0.6552 / 8.60** | 1.43× |

ncu（fp8 b1_t512 causal）：Duration 116.6→**59.5µs**、warps_active 6.20%→**12.39%**、
Ipc 0.11→**0.23**、`lts__t_sectors_op_red` **逐字节不变**（1,769,472）。详见 `docs/03` §52。
**附带发现（非本轮引入）**：三 dtype 的 **非 causal（full）MLA varlen 在 HEAD 已是偏差**
（退回 O51 提交复现一致）——记入 ROADMAP backlog。

## 12. 变长（VARLEN）主 kernel 的 TMA 化 —— fp16，第 83 轮（判决：中性/偏负）

第 82 轮后 varlen 的两条候选之一。做法：packed 布局描述符按 `dims={D,T,H,1}`（`S=T,B=1`）
建，kernel 用行坐标 `cu_seqlens[b]+row`、batch 坐标 0，复用 O33/O35 的逐 atom TMA；仅 fp16、
`--varlentma` opt-in（默认 0）。**数值与 cp.async 逐位一致**（5 个 case 的 max_abs 及 @index 全同）。

| case（fp16） | cp.async ms | TMA ms | 比值 |
|---|---|---|---|
| b4_t3840 causal | 0.7761 | 0.7641 | 1.016× |
| b4_t4096 等长 causal | 0.4484 | 0.4461 | 1.005× |
| b5_t3968 GQA kv8 causal | 1.4102 | 1.4085 | 1.001× |
| b4_t3840 full | 1.2822 | 1.2746 | 1.006× |
| **b8_t2904 强倾斜 causal** | 0.5785 | 0.6073 | **0.953×** |

ncu（main, b4_t3840 causal）：指令数 163.7M→**125.9M（−23.1%）** 但 Duration 546.2→**543.6µs**，
occ 12.43%（1 CTA/SM）、Warp Cycles/Issued 5.45→7.06 ⇒ **延迟/occupancy bound**，非发射指令 bound
（同 O35 的 BN=64 定长 TMA）。**结论：varlen TMA 不成立**，保留 opt-in；剩余杠杆 = LSE 的 K 维
split + occupancy。对标（等长 `[1024]×4` == 定长 B=4,S=1024,H=16,D=128）：ours 0.446ms（真反向
~153TF）= FA3 0.1457/471.7 的 **3.06×**、TE 0.1760/390.6 的 2.53×、FA2 0.2507/274.1 的 1.78×。
详见 `docs/01` §16.10。

## 13. fp8 LSE 的 K 维 split + 二次归约（O38，第 84 轮）—— **正结果，默认 auto**

承接 `docs/03` §40/§41。动机：O32 的 TMA LSE 每 CTA 串行扫 `nblk+1` 个 K tile，小 S/H 下
grid 只有 64（S512）~512（S4096）个 CTA，ncu 实测 S512 是纯 grid/延迟受限（Compute 7.6%、
achieved occ 6.25%）。O38 把 K 范围切成 `ksplit` 段并行、再用 `lse_split_merge_kernel`
对 online-softmax 的 `(m,l)` 做二次归约。`--lsesplit=0`（默认 auto：目标 `grid*split≈2048`、
上限 8）在定长 causal、fp8、D=128 生效；`--lsesplit=1` 退回历史逐位。

**数值**（vs fp32 ref，`auto` 与历史逐位一致到打印精度；`split-vs-split1` max_abs ≤2e-6）：

| case（fp8 causal） | dq / dk / dv vs ref（auto） |
|---|---|
| S512 H16 | 2.426e-01 / 2.972e-01 / 3.733e-01 |
| S1024 H32 | 2.399e-01 / 4.177e-01 / 3.535e-01 |
| S1024 H32 kv4 | 2.517e-01 / 5.339e-01 / 7.173e-01 |
| S4096 H16 | 2.635e-01 / 2.644e-01 / 3.216e-01 |
| MLA S1024 H2 D512 | 2.232e-01 / 3.337e-01 / 3.602e-01（不生效） |

**性能**（端到端 CUDA event，同 session `--lsesplit=1` vs auto）：

| shape | split=1 (ms) | auto (ms) | 比 | LSE-only 最优 |
|---|---|---|---|---|
| S512 H16 | 0.1150 | **0.1030** | **1.12×** | 2.03×（split8） |
| S1024 H32 | 0.4185 | **0.4010** | 1.04× | 1.62×（split4） |
| S1024 H32 kv4 | 0.3879 | **0.3690** | 1.05× | 1.62×（split4） |
| S1024 H40 kv8 | 0.4684 | **0.4540** | 1.03× | 1.40×（split2） |
| S1024 H64 kv1 (MQA) | 0.6431 | **0.6395** | 1.006× | 1.10×（split2） |
| S4096 H16 | 2.0474 | **1.9970** | 1.025× | 1.25×（split8） |
| MLA S1024 H2 D512 | 0.3922 | 0.3908 | 1.00× | —（不生效） |

**对标**（同 session，纯反向）：TE FP8 S512 **0.1011 ms/42.50 TF**、S4096 **0.5917 ms/464.59 TF**。
ours total S512 **0.1030 ms ⇒ 1.02×**（O37 时 1.21×，**几乎追平 TE FP8**）；S4096 **1.9970 ms
⇒ 3.38×**（O37 3.49×）。FA3 fp16 MHA S4096 0.3242/848、GQA kv4 S1024 0.0824/417（fp8 无 FA 基线）。

**ncu**（LSE 主 kernel）：S512 grid 64→512、Duration 33.41→**13.38 µs（2.50×）**、achieved occ
6.25%→**19.78%**、Compute 7.57%→**27.69%**；S4096 grid 512→4096、Duration 250.85→**198.46 µs
（1.26×）**、occ 23.58%→**45.30%**、Compute 57.01%→**78.25%**。merge kernel（S4096 split8）仅
5.86 µs（~2.4%）。**结论：LSE 此前的墙不是指令/访存，而是并行度/临界路径**——split-K 把它补上。
详见 `docs/03` §41。

## 14. fp16 / bf16 LSE 的 K 维 split + 二次归约（O38-fp16/bf16，第 85 轮）—— **正结果，默认 auto**

把 fp8 O38（§13）逐字 dtype 参数化到 fp16（`docs/01` §14u）与 bf16（`docs/01b` §6ac）的 **TMA
LSE**（`lse_mma_kernel_bal_tma`，D=128/causal 默认快路）。host `--lsesplit=N`（`0=auto`）。
与 fp8 的差异只在 auto 目标：fp16/bf16 用 **`grid*split≈528`（4 CTA/SM × 132 SM = 填满一个波）**，
因为大 S 时 512 CTA 已满一波，`split>1` 反而慢 3–7%（fp8 是 `≈2048`）。

**数值**（ours-vs-ref，causal）：全 shape 与 O5–O37 历史**逐位一致**（fp16 S512
1.671/1.771/1.899e-3、S4096 1.883/1.734/1.966e-3、GQA kv4 2.134/3.305/3.850e-3；bf16 同构），
`max_abs(split-vs-split1)` ≤ 2e-6（纯 fp32 求和次序）。单/两文件逐指标一致。

**LSE-only 与端到端**（同 session，CUDA event）：

| case | LSE split1 (ms) | LSE auto (ms) | LSE 倍数 | total split1 (ms) | total auto (ms) | total 倍数 |
|---|---|---|---|---|---|---|
| fp16 MHA S512 | 0.0259 | **0.0150** (split8) | **1.72×** | 0.0935–0.0953 | **0.0831** | **1.13–1.15×** |
| fp16 GQA kv4 S1024 | 0.0506 | **0.0370** (split2) | **1.37×** | 0.2572–0.2598 | **0.2429** | **1.06–1.07×** |
| fp16 MHA S4096 | 0.2064 | 0.2064 (split1) | 1.00× | 1.260–1.266 | 1.261 | 1.00× |
| bf16 MHA S512 | 0.0257 | **0.0149** (split8) | **1.73×** | 0.0938–0.0948 | **0.0842** | **1.13×** |
| bf16 GQA kv4 S1024 | 0.0496 | **0.0364** (split2) | **1.36×** | 0.2581–0.2594 | **0.2451** | **1.06×** |
| bf16 MHA S4096 | 0.2145 | 0.2145 (split1) | 1.00× | 1.2591 | 1.2671 | 1.00× |

kv8/kv1 的 grid 已满一波，auto=1（手动 `--lsesplit=4` 在 kv8 上最多 1.035×）。merge kernel
（S512 split8）仅 4.16 µs。

**ncu**（fp16 LSE，S512，同 binary）：split1 **Duration 27.74µs / Waves 0.12 / occ 6.25% /
Compute 8.22% / Ipc 0.75 / No Eligible 81.3%** → split8 **12.64µs（2.19×）/ Waves 0.97 /
occ 20.61% / Compute 26.89% / Ipc 1.50 / No Eligible 61.3%**。**墙 = 并行度/临界路径**（与 fp8
O38 同结论）。

**对标**（同 session，纯反向 `harness/fa_vs_te_bwd_only.py`；ours total 的真反向 TF = 打印值 ×2）：

| shape | ours total (ms) | FA3 (ms/TF) | TE (ms/TF) | ours/FA3 时间 |
|---|---|---|---|---|
| fp16 MHA S4096 | 1.261 | 0.3250 / 846 | 0.4440 / 619 | 3.88× |
| fp16 GQA kv4 S1024 | 0.2429 | 0.0823 / 417 | 0.1124 / 306 | **2.95×** |
| fp16 GQA kv8 S1024 | 0.2894 | 0.1210 / 355 | 0.1320 / 325 | **2.39×** |
| fp16 MQA kv1 S1024 | 0.3957 | 0.1567 / 439 | 0.2148 / 320 | 2.53× |
| bf16 MHA S4096 | 1.2671 | 0.3197 / 860 | 0.4428 / 621 | 3.96× |
| bf16 GQA kv4 S1024 | 0.2451 | 0.0827 / 415 | 0.1121 / 306 | 2.96× |

## 15. MLA（D=512）LSE 的 K 维 split + 二次归约（O39，第八十六轮）—— **正结果，默认 auto**

O38（§13/§14）把 LSE 的 K 维 split 做进了 D=128 的 TMA LSE；但 **MLA（head_dim=512）反向**
的 causal LSE 走 mma 版 `lse_mma_kernel_bal<512>`，grid 极小（S1024H2 = 16 CTA，Waves 0.06）
—— 纯并行度墙。O39 把同一套「K tile 切片 + `lse_split_merge_kernel` 二次归约」移植到三
dtype 的 mma 版 LSE（单/两文件；device 同步 `identical: True`）。auto：**D=512 目标
`grid*split ≈ 256`（fp8，2 CTA/SM）/ `≈ 132`（fp16/bf16，1 CTA/SM）**、上限 16，并按
`nblk=ceil(S/64)` 封顶；D=128 维持 O38 的 `≈2048/528`。

**LSE-only 与端到端**（同 session，CUDA event；`--lsesplit=1` 为基线）：

| case | dtype | LSE split1 (ms) | LSE auto (ms) | LSE 倍数 | total split1 (ms) | total auto (ms) | total 倍数 |
|---|---|---|---|---|---|---|---|
| MLA S256H2 | fp8 | 0.0458 | **0.0237** (split4) | **1.93×** | 0.1110 | **0.0921** | **1.21×** |
| MLA S512H4 | fp8 | 0.0780 | **0.0254** (split8) | **3.07×** | 0.2601 | **0.2009** | **1.29×** |
| MLA S1024H2 | fp8 | 0.1447 | **0.0310** (split16) | **4.67×** | 0.4036 | **0.2851** | **1.42×** |
| MLA S256H2 | fp16 | 0.0350 | **0.0200** (split4) | **1.75×** | 0.2182 | **0.2014** | **1.08×** |
| MLA S512H4 | fp16 | 0.0583 | **0.0216** (split8) | **2.70×** | 0.4146 | **0.3785** | **1.10×** |
| MLA S1024H2 | fp16 | 0.1018 | **0.0282** (split8) | **3.62×** | 0.7846 | **0.7154** | **1.10×** |
| MLA S256H2 | bf16 | 0.0350 | **0.0202** (split4) | **1.73×** | 0.2179 | **0.2025** | **1.08×** |
| MLA S512H4 | bf16 | 0.0585 | **0.0221** (split8) | **2.65×** | 0.4182 | **0.3816** | **1.10×** |
| MLA S1024H2 | bf16 | 0.1006 | **0.0276** (split8) | **3.64×** | 0.7855 | **0.7116** | **1.10×** |

**数值**（vs fp32 ref，max_abs dq/dk/dv）：fp8 S256/S512/S1024 = 2.356/2.290/3.441e-1、
2.415/2.992/4.481e-1、2.232/3.337/3.602e-1；fp16 = 1.638/1.582/1.753e-3、2.516/2.916/1.724e-3、
1.987/1.712/1.848e-3；bf16 = 1.230e-2/9.875e-3/1.686e-2、8.753e-3/1.082e-2/1.740e-2、
5.838e-3/9.519e-3/1.568e-2 —— 全部与历史逐位一致；`max_abs(split-auto vs split1) ≤ 9.5e-7`
（纯 fp32 求和次序）。D=128 MHA/GQA 回归逐位不变（fp8 S512 2.426/2.972/3.733e-1、S4096
2.635/2.644/3.216e-1、kv4 2.517/5.339/7.173e-1；fp16/bf16 同历史）。单/两文件逐指标一致。

**ncu**（LSE 主 kernel，S1024H2，同 binary split1 vs split）：fp8 split1 Duration
152.42µs / Waves 0.06 / occ 6.25% / Compute 2.65% → split16 **26.11µs（5.8×）/ Waves 0.97 /
occ 10.77% / Compute 22.02%**；fp16 split1 108.80µs / Waves 0.12 / occ 6.25% → split8
**25.66µs（4.2×）/ Waves 0.97 / Compute 16.64%**。**结论：墙 = 网格不足/并行度；split 填满
一个波后回到 LSE 固有的 `mma wait` + smem 依赖（1–2 CTA/SM）**（与 fp8 O38 同结论）。

**对标**：FA3/TE/FA2 反向均**不支持 head_dim=512**（`fa=NA`/`te=NA`），MLA 性能仅 ours 提供；
D=128 的 MHA/GQA 对标不受影响（O39 只改 D=512 的 LSE，D=128 TMA 路径逐位回归）。
原始输出见 `src/{fp8,fp16,bf16}/fa_bwd_*_o39_*` 与各 dtype 文档 §42/§14v/§6ad。

## 16. 非 TMA wgmma LSE 的 K 维 split + varlen 接入（O40，第八十七轮）—— **正结果，默认 auto**

O38/O39 覆盖了 **TMA LSE**（D=128）与 **mma LSE**（D=512），但 **非 TMA 的 wgmma LSE**
`lse_mma_kernel_bal_wgmma` 未做 split，而它是 ① 定长 D=128/causal 的 `--lsetma=0` 回退、
② **VARLEN D=128/causal 的默认 LSE**。O40 把 O39 的「K tile 切片 + `lse_split_merge_kernel`
二次归约」移植到它（三 dtype 单/两文件，device 同步 `identical: True`），并把 split 接进
`run_varlen`（D=128 wgmma 版 + D=512 mma 版；merge 行数用 packed 的 `T*H`）。auto：D=128
`grid*split≈2048`（定长）/`≈528`（varlen，一个波）、D=512 `≈256`（fp8）/`≈132`（fp16/bf16），
再按 `nblk` 封顶。

**定长 `--lsetma=0`（wgmma LSE）与 VARLEN**（同 session，CUDA event；`--lsesplit=1` 基线）：

| case | dtype | split1 | auto | 倍数 | 备注 |
|---|---|---|---|---|---|
| MHA S512 preprocess | fp8 | 0.0367 ms | **0.0207** | **1.77×** | total 0.1205→0.1054（1.14×） |
| MHA S512 preprocess | fp16 | 0.0364 ms | **0.0211** | **1.73×** | total 0.1011→0.0847（1.19×） |
| MHA S512 preprocess | bf16 | 0.0368 ms | **0.0215** | **1.71×** | total 0.1015→0.0861（1.18×） |
| MHA S4096 preprocess | fp8 | 0.2958 ms | **0.2673** | **1.11×** | total 2.0804→2.0417（1.02×） |
| varlen b1_t512_h16 D128 total | fp8 | 0.1255 ms | **0.1072** | **1.17×** | auto=8 |
| varlen b4_t3840_h16 D128 total | fp8 | 0.9646 ms | **0.9285** | **1.04×** | auto=2 |
| varlen b1_t512_h2 D512 total | fp8 | 0.1954 ms | **0.1401** | **1.39×** | auto=8 |
| varlen b3_t1792_h2 D512 total | fp8 | 0.6067 ms | **0.4979** | **1.22×** | auto=4 |
| varlen b1_t512_h2 D512 total | fp16 | 0.4098 ms | **0.3737** | **1.10×** | auto=8 |
| varlen b3_t1792_h2 D512 total | fp16 | 0.9112 ms | **0.8607** | **1.06×** | auto=4 |
| varlen b1_t512_h2 D512 total | bf16 | 0.4100 ms | **0.3737** | **1.10×** | auto=8 |
| varlen b3_t1792_h2 D512 total | bf16 | 0.9100 ms | **0.8630** | **1.05×** | auto=4 |

**数值**（vs fp32 ref，max_abs dq/dk/dv）：定长 fp8 2.426/2.972/3.733e-1、fp16 1.671/1.771/
1.899e-3、bf16 9.001/12.61/13.65e-3；varlen fp8 b1_t512_h16 2.280e-1/3.108e-1/3.422e-1、
b1_t512_h2 D512 1.613e-1/2.238e-1/3.864e-1；fp16 b1_t512_h2 D512 1.303e-3/1.537e-3/1.557e-3；
bf16 b1_t512_h2 D512 8.042e-3/1.097e-2/1.391e-2 —— 全与历史**逐位一致**；
`max_abs(split-auto vs split1)=0`。D=128 MHA/GQA 默认 TMA 路径回归不受影响。

**ncu**（LSE `lse_mma_kernel_bal_wgmma`，S512，同 binary split1 vs split8）：fp8 split1
Duration 37.22µs / Waves 0.07 / occ 6.25% / Compute 8.21% → split8 **14.56µs（2.56×）/ Waves 0.55 /
occ 20.26% / Compute 33.32%**；fp16 33.66→**14.85µs**、bf16 33.50→**14.91µs**（Waves 0.12→0.97、
occ 6.25%→19.27%）。**墙 = 网格不足一个波；split 填满后回到 LSE 固有 `mma wait` + smem 依赖**。

**对标**：varlen / D=512 无 FA/TE 基线（仅 ours）；定长 D=128 默认 TMA 路径逐位回归、
对标不受影响（MHA S4096 ours total 约 FA3 的 3.9×）。原始输出见 `src/{fp8,fp16,bf16}/fa_bwd_*_o40_*`
与各 dtype 文档 §43/§14w/§6ae。

## 17. fp8 主 kernel 的 K/V 4D-TMA（O41，第八十八轮）—— **正结果，默认 auto**

> roadmap「下一步候选 ①」。O37 只把 fp8 主 kernel 的 Q/dO 改 4D-TMA；O41 把 **K/V 也 TMA 化**
> （K 双缓冲、V 单缓冲），K 双缓冲靠 `dS3` 复用当前 K stage、`Ap` 复用 `Vs` 实现零成本折叠，
> `smem 70656→74816B`（≤ 76800 的 3-CTA/SM 上限）⇒ 仍 3 CTA/SM。

**数值**（vs fp32 ref，max_abs dq/dk/dv）与历史**逐位一致**：S512 2.426/2.972/3.733e-1、
S4096 2.635/2.644/3.216e-1、GQA kv4 2.517/5.339/7.173e-1、MQA kv1 4.101e-1/1.572/2.126、
full S1024 5.520/5.312/4.024e-2；`max_abs(kvtma-vs-qdtma)≤1.2e-5`（仅 atomic 次序）。

**性能**（同 session A/B，CUDA event；`--kvtma` 0→1，main-only）：

| shape | Q/dO-TMA | **Q/dO/K/V-TMA** | 比 | ours total | TE FP8（同 session） |
|---|---|---|---|---|---|
| MHA S512 | 0.0734 ms | **0.0643** | **1.141×** | 0.1033 ms / 20.78 TF | 0.1008 ms / 42.60 TF |
| MHA S1024 H32 | 0.3188 | **0.2825** | **1.128×** | 0.3894 ms / 44.12 TF | 0.2063 ms / 166.54 TF |
| MHA S4096 | 1.7131 | **1.6056** | **1.067×** | **1.9215 ms / 71.53 TF** | 0.5905 ms / 465.48 TF |
| GQA q32/kv4 | 0.2856 | **0.2654** | **1.076×** | 0.3547 ms / 48.44 TF | — |
| MQA q64/kv1 | 0.5204 | **0.4930** | **1.056×** | 0.6280 ms / 54.71 TF | — |
| full S1024 H16 | 0.3384 | **0.3193** | **1.060×** | 0.5675 ms / 15.14 TF | — |

端到端 S4096 **2.0508→1.9215 ms**，为 TE FP8 的 **3.25×**（O37 3.49×）；S512 的 main
0.0643 ms 已快过 TE FP8 整条反向 0.1008 ms。单/两文件 device 逐字一致（单文件 total 1.934 ms）。

**ncu**（S=4096，同 binary `--kvtma` 0/1）：Duration 1.67→**1.61 ms**；`long_scoreboard`
0.80→**0.55**、`short_scoreboard` 1.61→**1.30**、`wait` 1.58→1.53、`barrier` 0.28→0.42；
**`lts__t_sectors_op_red` 114,524,160 逐字节不变**；lts throughput 70.9→**77.9%**；
168 regs / 74.82KB smem / **3 CTA/SM（Block Limit Shared Mem=3）**。**墙仍是 mma 依赖延迟
（`wait`+`short_scoreboard`）+ 抬头的 L2**；K/V 搬运这条路已到上限。

**对标**：fp8 无 FA 基线，仅 TE；同 session 纯反向 FA3 fp16 MHA S4096 0.3243ms/848TF
（ours fp8 total 距之仍量级差距）。详见 `docs/03` §44。原始输出
`src/fp8/fa_bwd_fp8_o41_sweep.out.txt`、`..._o41_ncu_ab_s4096.out.txt`。

## 18. fp16/bf16 `wgmma2` 主 kernel 的 N 方向 split-K（O43，第九十轮）—— **正结果，默认 auto（仅小 grid）**

> fp8 的 mma 主 kernel 早有 `ksplit`；fp16/bf16 默认档的 `wgmma2`（BM=128/BN=64、1 CTA/SM）
> 没有，**S=512 MHA grid=64 CTA < 132 SM**（ncu Waves 0.48，half-SM 空转）。O43 给它加运行时
> `ksplit`：KV tile 切片 + dQ 跨 CTA `atomicAdd`（`ksplit==1` 逐位退化）。**auto 仅在小 grid**，
> 大 S / wgmma2b 恒 1。varlen 切 K 反慢（opt-in）。

**数值**（vs fp32 ref，causal，max_abs）与 ksplit=1 **逐位一致**：fp16 S512 `1.671/1.771/1.899e-3`、
bf16 S512 `9.001/1.261e-2/1.365e-2`；回归 S1024 full / S4096 causal 不变。

**性能**（同 binary、同 session，S=512 MHA）：

| dtype | ksplit=1 main/total | **ksplit=2 main/total** | main 比 | total 比 | ours/FA3（时间） |
|---|---|---|---|---|---|
| fp16 | 0.0529 / 0.0871 ms | **0.0317 / 0.0687 ms** | **1.67×** | **1.27×** | 3.34×→**2.63×** |
| bf16 | 0.0538 / 0.0869 ms | **0.0320 / 0.0692 ms** | **1.68×** | **1.26×** | 3.32×→**2.64×** |

同 session 纯反向基线（`fa_vs_te_bwd_only.py`，S=512 MHA）：fp16 FA3 0.0261ms/164TF、
TE 0.0320ms/134TF（FA2 0.0437ms/98TF）；bf16 FA3 0.0262ms/164TF、TE 0.0323ms/133TF。

**ncu**（fp16 `wgmma2`，S=512）：ksplit 1→2 Duration 54.08→**33.50µs**、**Waves 0.48→0.97**、
Executed Ipc Elapsed 0.42→**0.77**、DRAM 9.4→18.8%、L2 25.6→49.8%；per-SM occupancy 恒 12.5%
（1 CTA/SM）。⇒ **墙是 grid 不足一个波、不是 per-SM occupancy**。详见 `docs/01` §14x、`docs/01b` §6af。
原始输出 `src/fp16/fa_bwd_fp16_o43_sweep.out.txt`、`..._o43_ncu_wg2_s512_ks{1,2}.out.txt`、
`src/bf16/fa_bwd_bf16_o43_sweep.out.txt`。

## 19. fp16/bf16 MLA（D=512）主 kernel 的 N 方向 split-K（O44，第九十一轮）—— **正结果，默认 auto**

> O43 只覆盖了 fp16/bf16 D=128 默认档 `wgmma2`；**MLA 走 mma 主 kernel（BM=32/BN=32/PIPE=1）
> 没有 ksplit**：S1024H2 / S512H4 / S256H2 的 base grid 只有 64 / 64 / 16 CTA，ncu Waves 0.48、
> occ 6.25%（207KB smem、1 CTA/SM）⇒ 并行度不足。O44 把 O43 的机制扩到 mma 主 kernel：
> `fa_bwd_{fp16,bf16}_mma_kernel` 加 `int ksplit=1`（KV tile 切片 + 空切片早退 + prologue stage
> 用 `nt_begin&1` 对齐），dQ 在 `ksplit>1` 时改跨 CTA `red_add2`；host `--mlaksplit=N`，auto 仅
> D==512、目标 `grid*sp≈528`（1 CTA/SM 的 4 个波）、上限 16。D!=512 逐位退化。

**数值**（ours-vs-ref，causal，max_abs dq/dk/dv）与 O5c（§7.8）**逐位一致**：

| MLA case (B1, D=Dv=512) | fp16 | bf16 |
|---|---|---|
| (1,256,2,512) | 1.638 / 1.582 / 1.753e-3 | 1.230e-2 / 9.875e-3 / 1.686e-2 |
| (1,512,4,512) | 2.516 / 2.916 / 1.724e-3 | 8.753e-3 / 1.082e-2 / 1.740e-2 |
| (1,1024,2,512) | 1.987 / 1.712 / 1.848e-3 | 5.838e-3 / 9.519e-3 / 1.568e-2 |

**性能**（CUDA event，Hopper 构建，同 binary/同 session；main 与 total，ms）：

| MLA case | dtype | ksplit=1 main | **auto main** | main 比 | **auto total** | total 比（vs O5c） | TFLOPS |
|---|---|---|---|---|---|---|---|
| (1,256,2,512) | fp16 | 0.1854 | **0.0222** | 8.4× | **0.0566** | 5.3× | 4.74 |
| (1,512,4,512) | fp16 | 0.3685 | **0.0840** | 4.4× | **0.1284** | 4.2× | 16.73 |
| (1,1024,2,512) | fp16 | 0.7171 | **0.1524** | 4.7× | **0.2009** | 4.7× | 21.38 |
| (1,256,2,512) | bf16 | 0.1852 | **0.0220** | 8.4× | **0.0548** | 5.4× | 4.90 |
| (1,512,4,512) | bf16 | 0.3624 | **0.0842** | 4.3× | **0.1283** | 4.2× | 16.74 |
| (1,1024,2,512) | bf16 | 0.7136 | **0.1518** | 4.7× | **0.2000** | 4.7× | 21.47 |

**ncu**（fp16 `fa_bwd_fp16_mma_kernel`，S=1024H2，`--set full --launch-count 1`）：ksplit 1→2
Duration 764.5→**228.7µs**、**Waves 0.48→0.97**、Issued Ipc 0.33→0.57、achieved occ 恒 ~6.25%
（1 CTA/SM）、No Eligible 91.8→85.7%、DRAM 0.83→2.78%、L1TEX 33.3→52.7%、L2 14.0→45.9%；
头号 stall 从 long scoreboard（7.4 cyc）变 fixed-latency `wait`（2.2 cyc）。⇒ **打掉的是
「SM 空转」而非延迟隐藏**（同 O43）。FA/TE 反向后端均不支持 D=512，无第三方对标。
~~**fp16/bf16 MLA total 已比 fp8 MLA（§7.7 0.308/0.591/1.022ms）快 4.9–5.4×**~~
—— **已由 O45（第 92 轮，`docs/03` §46）更正**：§7.7 是 P5-3 时代旧数字；fp8 MLA 早在
**O29** 就有同样的 auto ksplit（`target=S/2`）。实测 fp8 MLA total = 0.0896 / 0.1996 / 0.2835 ms
（vs 上面 fp16 的 0.0566 / 0.1284 / 0.2009），**fp16 仅快 1.4–1.9×**。O45 同时判决：
把 bulkred 开放到 D=512 仍 **0.84×**（负），`FA_ILV/ILV34` 中性/负；fp8 MLA 的墙是
**1 warp/scheduler 的延迟**（1 CTA/SM × 4 warp、No Eligible 85.7%），2 CTA/SM 因 smem 207.9KB
不可达，唯一剩余杠杆是 256 线程/8-warp 几何（backlog）。
详见 `docs/01` §14y、`docs/01b` §6ag、`docs/03` §46。原始输出 `src/{fp16,bf16}/fa_bwd_*_o44_mla_sweep.out.txt`、
`src/fp16/fa_bwd_fp16_o44_ncu_main_ks{1,2}_s1024h2.out.txt`、`src/fp8/fa_bwd_fp8_main_o45_sweep.out.txt`。

## 20. fp8 MLA（D=512）8-warp 几何（O47，第九十四轮）—— 正结果，D=512 默认

承接 O46（fp16/bf16 MLA 8-warp）与 O45（fp8 MLA 墙 = 1 warp/scheduler），把 `fp8_mma_body` /
`fa_bwd_fp8_mma_kernel` 的 warp 网格改成由 `NTH`/`NWAR` 派生：默认 `128/2` 与历史**逐字等价**，
MLA 用 `256/4`（2×4）。**同一 binary 的 `--mla8w=0/1` A/B**：

| MLA case | dtype | main 4w | **main 8w** | main 比 | total 4w→8w | main-only TF（峰值 1978.8） |
|---|---|---|---|---|---|---|
| (1,256,2,512) | fp8 | 0.0435 | **0.0237** | 1.84× | 0.0896→**0.0694** | 11.3（0.57%） |
| (1,512,4,512) | fp8 | 0.1192 | **0.0737** | 1.62× | 0.1996→**0.1332** | 29.1（1.47%） |
| (1,1024,2,512) | fp8 | 0.1998 | **0.1243** | 1.61× | 0.2835→**0.1927** | 34.6（1.75%） |

**数值**：vs fp32 ref 与历史逐值一致（2.356/2.290/3.441e-1、2.415/2.992/4.481e-1、
2.232/3.337/3.602e-1），`max_abs(8w-vs-4w)≤5e-7`（仅 atomic 次序）；D=128 MHA/GQA/MQA
回归逐位不变（S4096 2.635/2.644/3.216e-1、GQA kv4 2.517/5.34/7.17e-1）。FA/TE 反向不支持
D=512，仅 ours 数字。

**ncu**（S1024H2 main）：occ **6.25%→12.49%**、Active Warps/SM 4→8、Issued Ipc 0.57→1.13、
`sm__issue_active` 14.4%→28.1%、No Eligible 85.66%→71.95%、Duration 238.8→**132.4µs**、
255→245 regs（无 spill）、`lts__t_sectors_op_red` **6,684,672 逐字节不变**；stall 仍由
`long_scoreboard`（L2/全局）主导 + `wait`（mma 依赖）+ `short_scoreboard`（smem→mma）。
⇒ 提升来自**延迟隐藏/并行度**（每 scheduler 1→2 warp），非带宽/算力。详见 `docs/03` §47、
`docs/08` §5.12。

## 21. D=128 mma fallback 的 8-warp 几何（O48，第九十五轮）—— grid ≤ SM 时正、大 grid 负

承接 O47 的「下一步候选 ①」。对象是 **D=128 的 mma fallback**（纯 `sm_90` 构建 / `--wg2=0`；
生产 `wgmma2`/fp8 wgmma 已是 256 线程，结构不同）。同一 binary `--d128w=0/1` A/B：

| dtype | shape（grid） | main 4w | **main 8w** | 比 | 说明 |
|---|---|---|---|---|---|
| fp16 | S=512 MHA（128 < 132 SM） | 0.0622 | **0.0589** | **1.056×** | + 生产 `(64,64,2)`：main 1.105× / total 1.077× |
| bf16 | S=512 MHA（128） | 0.0618 | **0.0586** | **1.054×** | 同 |
| fp16 | S=1024 GQA kv4（512） | 0.2810 | 0.2744 | 1.024× | 接近中性 |
| fp16 | S=4096 MHA（1024） | 1.5842 | 1.7972 | 0.881× | occupancy 掉半 |
| bf16 | S=4096 MHA（1024） | 1.5673 | 1.7851 | 0.878× | 同 |
| fp8 | S=512 MHA（ksplit=16→2048） | 0.0718 | 0.0907 | 0.791× | 已有 split-K、机器本就填满 |
| fp8 | S=4096 MHA | 1.9139 | 2.7273 | 0.702× | 同 |

**判据 / ncu**：8-warp 只在「4-warp 的每 scheduler warp 数 <2」（即 grid ≲ SM 数）时赢——
fp16 S512 ncu：4w `Active Warps/Sched 1.00`、occ 6.24%、Duration 58.3µs →
8w `1.99`、occ 12.3%、**52.5µs**。fp8 因 auto split-K 把 grid 抬到 2048，4w 已有
`2.87 warp/sched`，8w 反降到 1.99 ⇒ 负。**数值**：`max_abs(8w-vs-4w)` ≤ ~2e-6（仅 dK/dV
atomic 次序），vs ref 与历史逐位不变。

**结论**：默认保持 4-warp（opt-in `--d128w`，不动历史逐位值）；若默认化，推荐
`D==128 && mma 路径 && grid < 132`。顺带修掉 O47 参数化留下的两个 `NTH>128` 才触发的
correctness bug（`kv_prefetch/commit_pair` 越界、`kRegDq` flush 硬编码几何），默认 `128/2`
路径逐位不变。详见 `docs/01` §14aa、`docs/01b` §6ai、`docs/03` §48。

## 22. D=128 mma fallback 8-warp 几何的自动档默认化（O49，第九十六轮）—— 正结果

O48 判明 8-warp 只在 `grid ≤ SM 数`（每 scheduler 4-warp 仅 1 warp）时赢，并留成 opt-in。
O49 把它**默认化**：auto 条件 `D==128 && mma 路径 && grid ≤ sm_count(132)`，`--d128w=0/1`
仍可强制。对 fp8 额外要求 `!wgmma`（不覆盖 Hopper 生产路径）且用**总网格** `mg.x·mg.y·mg.z`
判据（fp8 `mg.x` 已含 ksplit，单看 x 维会误开）。

**fp16/bf16（plain sm_90 mma，同 session）**：

| dtype | shape（grid） | main 4w | **main auto(8w)** | 比 | total 4w → auto | 比 |
|---|---|---|---|---|---|---|
| fp16 | S=512 MHA（128，auto on） | 0.0567 | **0.0509** | **1.114×** | 0.0960 → **0.0880** | 1.091× |
| bf16 | S=512 MHA（128，auto on） | 0.0570 | **0.0515** | **1.107×** | 0.0942 → **0.0899** | 1.048× |
| fp16 | S=4096 MHA（1024，auto off） | 1.5059 | 同（逐位） | 1.00× | 1.9288 | 1.00× |
| fp16 | S=1024 GQA kv4（512，auto off） | 0.2612 | 同 | 1.00× | 0.3457 | 1.00× |
| fp8 | S=512 MHA（2048，auto off） | 0.0630 | 同（逐位） | 1.00× | 0.1165 | 1.00× |
| fp8 | S=512 **`--ksplit=1`**（128，auto on） | 0.1188 | **0.0885** | **1.34×** | 0.1705 → 0.1368 | 1.25× |

**对标**（同 session 纯反向 `harness/fa_vs_te_bwd_only.py`）：fp16/bf16 S=512 MHA
**FA3 0.0263ms/163TF**、TE fp16 0.0319/135、TE bf16 0.0324/133 ⇒ ours total 时间比
fp16 **3.65×→3.35×**、bf16 **3.58×→3.42×**。

**数值**：auto on 只在 S=512 改 dK/dV 的 atomic 次序（`max_abs(8w-vs-4w)` ≤ 5e-7，dq 逐位），
vs ref 与历史同量级；auto off 的 shape **逐位不变**。详见 `docs/01` §14ab、`docs/01b` §6aj、
`docs/03` §49。

## 23. wgmma2 等待拆分（O50，第九十七轮，中性偏正/默认 fp16·bf16 开、fp8 关）+ MLA split-KV auto 重标定（正结果）

### 23.1 `FA_WS1`（fp16/bf16 wgmma2 的 GEMM1/2 等待拆分）

两条 wgmma（GEMM1 `S=QKᵀ`、GEMM2 `dP=dO·Vᵀ`）各自 commit 后，先 `wait_group<1>`（只等
GEMM1）算 P、再 `wait0` 取 dP 算 dS——用 CUDA-core 的 exp/量化掩盖 GEMM2。**数值逐位不变**。

| dtype | shape | main ws0 | main ws1 | 比 |
|---|---|---|---|---|
| fp16 | S512 MHA | 0.0315 | 0.0315 | 1.00× |
| fp16 | S4096 MHA | 0.9558 | **0.9543** | 1.002× |
| fp16 | GQA kv4 S1024 | 0.1801 | **0.1782** | 1.011× |
| fp16 | MQA kv1 S1024 | 0.2868 | 0.2865 | 1.00× |
| bf16 | S512 MHA | 0.0319 | **0.0316** | 1.009× |
| bf16 | S4096 MHA | 0.9561 | **0.9538** | 1.002× |
| bf16 | GQA kv4 S1024 | 0.1824 | **0.1807** | 1.009× |

ncu（fp16 S512）：Duration 33.57→33.47µs、`barrier` 0.39→0.25、`red` 1376256 / 指令数 6150400
**逐字节不变** ⇒ 幅度在噪声内、方向非负。**fp8 实测中性（±2%）故默认关**（保持旗舰路径逐位
不变）。**教训：3 CTA/SM（12 warp/SM）下「提前算依赖较轻的那半」拿不到额外重叠。**

### 23.2 fp16/bf16 MLA split-KV auto 重标定（正结果，默认）

O46 把 MLA 换 8-warp 后没重标 split。旧「`grid*sp≈528` + `nblk` 封顶」对 S1024H2 只切到 8、
S256H2 被 `nblk=8` 封顶到 8。改成 `sp = max(528 目标, ≤ceil(nblk/2))`、**去 `nblk` 封顶**：

| dtype | MLA shape | main 旧 | main 新 | 比 | total 旧→新 |
|---|---|---|---|---|---|
| fp16 | S1024H2 | 0.1411 | **0.1366** | **1.033×** | 0.1911 → **0.1855** |
| fp16 | S256H2 | 0.0204 | **0.0192** | **1.063×** | 0.0544 → **0.0533** |
| fp16 | S512H4 / S512H2 | 0.0756 / 0.0465 | 0.0758 / 0.0454 | 不变 | — |
| bf16 | S1024H2 | 0.1408 | **0.1365** | **1.031×** | — |
| bf16 | S256H2 | 0.0204 | **0.0192** | **1.063×** | — |

数值 vs fp32 ref **逐位不变**（fp16 MLA 1.987/1.712/1.848e-3 等）。

### 23.3 对标（同 session 纯反向 `harness/fa_vs_te_bwd_only.py`）

fp16：FA3 S512 0.0265/162TF、S4096 0.3240/848、GQA kv4 0.0826/416 ⇒ ours total S512 0.0684
（FA3 2.58×）、S4096 1.2532（3.87×）、GQA 0.2446（2.96×）。bf16：FA3 S4096 0.3193/861 ⇒
ours 1.2559（3.93×）。**MLA（D=512）FA/TE 均不支持，仅 fp32 ref**。

### 23.4 修复：bf16 单文件版缺失 TMA 描述符 helper

`fa_bwd_bf16_mma_onefile.cu` 自 O35/O36 加主 kernel TMA 后缺 host 侧 `make_lse_map`/
`make_main_map`、**编译不过**；补回（dtype 参数化自 fp16）后 `bf16_one_ok`，数值逐位一致。

---

## 24. fp8 MLA（D=512）主 kernel 的 K/V `cp.async` 回填流水（O51，第九十八轮）—— **正结果，D=512 默认**

承接 O45/O47 的头号 stall `long_scoreboard`（fp8 MLA 走 mma 后端、K/V 同步载入、1 CTA/SM）。
把每 tile 的 K/V 同步载入换成 `cp.async` 回填流水（K 双缓冲、V 单缓冲 + 后段回填；Ap/dS3
拆成独立缓冲），**只改搬运**，smem 207.9→229.9KB 仍 1 CTA/SM。

| MLA case (D=Dv=512) | dq / dk / dv vs fp32 ref | main sync→kvpipe | total（`4BS²HD`）|
|---|---|---|---|
| S256 H2 | 2.356 / 2.290 / 3.441e-1 | 0.0239→**0.0234ms (1.020×)** | 0.0701ms / 3.83 TF |
| S512 H4 | 2.415 / 2.992 / 4.481e-1 | 0.0739→**0.0716ms (1.032×)** | 0.1317ms / 16.30 TF |
| S512 H2 | 2.286 / 2.252 / 3.343e-1 | 0.0508→**0.0495ms (1.026×)** | 0.0985ms / 10.90 TF |
| S1024 H2 | 2.232 / 3.337 / 3.602e-1 | 0.1228→**0.1183ms (1.038×)** | 0.1894ms / 22.68 TF |

ncu（S1024 H2，同 session A/B）：`long_scoreboard` **1.97→1.72**、`short` 1.57→1.40、
指令数 **−2.0%**、`lts__t_sectors_op_red` **逐字节不变**（6,684,672）、occ 恒 12.48%（1 CTA/SM）。
D=128（MHA/GQA）与 varlen 路径**逐位不变**；单/两文件 device 逐字一致。
**MLA 反向 FA2/FA3/TE 均不支持（`fa=NA`/`te=NA`），只有 ours 数字**；详见 `docs/03` §51。

## 25. fp16/bf16 MLA（D=512）**varlen** 主 kernel 的 N 方向 split-K（O53，第 100 轮）—— **正结果，varlen MLA 默认 auto**

承接 O52（把 O46 的 8-warp 几何搬进 varlen MLA）：varlen 主 kernel 仍是「单 CTA 扫整条 K」，
`D=512/BM=32` 的 base grid 在 `b1_t512_h2` 只有 16 CTA（`b3_t1792_h2` 192），smem 207.36KB
锁死 1 CTA/SM ⇒ ncu `Waves 0.48`、SM 空转。本轮把定长 O44/O50 的 **split-KV** 搬进 varlen
（host-only；device 早已支持，`ksplit==1` 逐式退化）：`run_varlen` 加 `mlaksplit`，`D==512`
按 auto（target `grid*sp≈528` + 每 m 块 K 切 ≈2 份、cap 16）选 `mla_ks_eff`，`mg.x *= mla_ks_eff`，
dQ 走跨 CTA `red_add2`。

| fp16 varlen case (D=Dv=512) | dq / dk / dv vs fp32 ref | total O52→O53 | 加速 |
|---|---|---|---|
| b1_t512 causal | 1.303 / 1.537 / 1.557e-3 | 0.2740→**0.0819ms / 13.11 TF** | **3.35×** |
| b3_t1792 causal | 2.415 / 1.834 / 1.856e-3 | 0.6523→**0.3518ms / 16.02 TF** | **1.85×** |
| b1_t512 full | 3.046 / 4.449 / 1.327e-4 | — / 0.2853ms / 3.76 TF | — |
| b3_t1792 full | 5.516 / 4.451 / 2.385e-4 | — / 1.0093ms / 5.59 TF | — |

bf16 逐值同构（causal total 0.0830 / 0.3499ms，12.93 / 16.11 TF）。主 kernel-only sweep
（8-warp，`[O53 A/B]`）：causal b1 `0.2350→0.0455ms`（**5.17×**）、b3 `0.5540→0.2543ms`（2.18×，
auto 即最优）；full auto 偏大（b1 最优 k=4、b3 k=8），但相对 k=1 仍 3.6×/1.6×。
**单/两文件逐指标一致**；`max_abs(8w-vs-4w)` dq ≤2e-7、dk/dv ≤1e-6（仅 atomic 次序）。
MLA 反向 FA2/FA3/TE 均不支持 ⇒ 无外部基线。

**ncu**（fp16/bf16 b3_t1792 H2 D512 causal，逐项一致）：Duration **259µs**、DRAM 4.7% /
L1TEX ~50% / **L2 ~81%** / Compute 19%、occ 12.2%（1 CTA/SM）、stall **long 2.73 + wait 2.28 +
short 1.58**、L2 `op_red`/`op_read` = 18.41M/12.91M（red 占 **58.8%**）。**bound = L2（跨 CTA
dQ 归约）+ 访存延迟**（O52 的「grid 不足一个波」已消除）。详见 `docs/01` §14ae、`docs/01b` §6am。

**「非 causal full MLA varlen HEAD 偏差」更正**：O52 所记偏差实为**对拍脚本漏传 `--full`**
（按 causal 比 full ref）；显式 `--full` 后 fp16/bf16/fp8 三 dtype full 全部对拍通过
（fp16 3.0e-4–5.5e-4、bf16 1.6e-3–3.5e-3、fp8 ~5e-2），**不是回归、无需修复**。

## 26. 非 causal（full）MLA varlen 的 LSE 走 K 维 split（O54，第 101 轮）—— **正结果，full varlen 默认 auto**

承接 O53：full varlen 的主 kernel 已 split-KV，但端到端仍被 **LSE** 主导（b3 full：total 1.013ms、
main-only 0.392ms ⇒ LSE ~0.62ms = **60%**）——非 causal MLA 的 LSE 一直走 O1 的
`lse_mma_kernel<512>`（一个 CTA 一个 m 块、无 split、标量 K 载入）。本轮给三 dtype 的
`lse_mma_kernel_bal` 加 **FULL 模式**（一个 CTA 一个 m 块 + `cp.async` 双缓冲 + O40 K 维 split），
`run_varlen` 的 `D==512 && !causal` 分支改走它（`FULL=false` 编译出与历史逐位相同的代码）。
auto：fp16/bf16 目标 384（b1→8、b3→4）、fp8 目标 768（b1/b3→8）。

| dtype / case (D=Dv=512) | LSE old → new (split auto) | LSE 加速 | total old → new | 数值 vs ref (dq/dk/dv) |
|---|---|---|---|---|
| fp16 b3_t1792 full | 0.3791→**0.0411ms** (4) | **9.2×** | 1.013→**0.468ms / 12.05 TF** | 5.52/4.45/2.39e-4 |
| fp16 b1_t512 full | 0.1933→**0.0135ms** (8) | **14.3×** | ~0.28→**0.102ms** | 3.05/4.45/1.33e-4 |
| bf16 b3_t1792 full | 0.3747→**0.0413ms** (4) | **9.1×** | 1.013→**0.468ms / 12.03 TF** | 3.10/3.53/2.32e-3 |
| fp8 b3_t1792 full | 0.3753→**0.0374ms** (8) | **10.0×** | ~0.72→**0.346ms / 16.30 TF** | 7.99e-2/9.63e-2/4.15e-2 |
| fp8 b1_t512 full | 0.1899→**0.0156ms** (8) | **12.2×** | — | — |

**ncu（fp16 `lse_mma_kernel_bal<512,1,true>`，b3 full，split4）**：Duration 40.99µs、
DRAM 5.38% / L1TEX 28.52% / L2 24.00% / Compute 20.55%、regs 63、smem **199.68KB → 1 CTA/SM**、
occ 6.25%、**Waves 2.91**、No Eligible 74.1%、Warps/Sched 1.00、fixed-latency stall 37.3%（主导）。
**bound = 低 occupancy（1 CTA/SM，smem 硬约束）+ fixed-latency 依赖**，非带宽/算力。

**回归**：causal b3（fp16 2.415/1.834/1.856e-3、bf16 total 0.3508ms、fp8 3.40/3.44/3.51e-1）
与 D=128 full varlen（fp16 b4_t4096 total 0.9075ms）**逐位/同量级不变**；单/两文件逐指标一致。
**教训**：把一个路径（causal）已有的优化（split + 双缓冲）补到另一路径（full）时，即使
「full 各块工作量相同、无需镜像配对」，**缺少 K 维 split/异步载入仍是并行度与延迟的墙**——
一个 `bool FULL` 模板参数就够，且默认档编译出逐位相同的代码。详见 `docs/01` §15、`docs/01b` §6an、
`docs/03` §53、`docs/08` §5.19。

## 27. varlen MLA full 的 split-KV auto 重新标定（O55，第 102 轮）—— **正结果，full varlen 默认**

O53 给 varlen MLA（D=512）主 kernel 的 split-KV auto 对 causal/full 用了同一目标（`base*sp≈528`，
4 个波 + 每 m 块 K 切 ≈2 份）。`[O53 A/B]` sweep 显示 **full 的最优 split 更小**：full 主 kernel 每
m 块工作量相同，base grid 靠少量切分即可铺满「一个波」（b1 base=32，k=4→128 CTA≈132 SM），
再切到 16 只是**重复读 Q/dO + 增加 dQ 跨 CTA atomic**（纯亏）。host-only 改动：full 分支
`target=132`（1 个波）、`sp_min=2`；**causal 分支逐字不变以保回归**。

| case (D=512, fp16=bf16) | O53 auto | O55 auto | main（同 binary sweep） | total old→new |
|---|---|---|---|---|
| b1_t512 full | 16 | **4** | 0.0745→**0.0681ms**；sweep k4 **0.0647** vs k16 0.0748 | 0.1030→**0.0958ms**（1.075×） |
| b3_t1792 full | 16 | **2** | 0.3888→**0.3873ms**；sweep k2 0.3873 vs k8 0.3868 | 0.4680→**0.4636ms**（1.009×） |
| b1_t512 causal | 16 | 16 | 不变 | 0.0815ms（回归） |
| b3_t1792 causal | 16 | 16 | 不变 | 0.3512ms（回归） |

**数值 vs fp32 ref**：full b1 3.05/4.45/1.33e-4、b3 5.52/4.45/2.39e-4（与 O53/O54 逐位相同）；
causal 回归逐位不变（split 只改 dQ 的 fp32 atomic 次序）。单/两文件逐指标一致。
**ncu（fp16 b1 full main，`-c 1`）**：O53 k=16 grid 256×2、**Waves 3.88**、Duration **77.54µs**、
L2 67.95% → O55 k=4 grid 64×2、**Waves 0.97**、Duration **68.00µs（1.14×）**、L2 73.24%、occ ~12.4%。
**bound 仍是 L2（dK/dV 跨 CTA red）+ 1 CTA/SM 低 occupancy**；本次是**去过度切分**。
详见 `docs/01` §15b、`docs/01b` §6ao、`docs/08` §5.20。

## 28. O56（第一百零三轮）：full MLA varlen 的 LSE 8-warp（LBM=128）几何——混合/负结果

把 O46/O47 的「1 CTA/SM 时 4→8 warp（每 scheduler 1→2）」杠杆搬到 full MLA varlen 的 LSE
（`lse_mma_kernel_bal<512,1,true>`，O54）：模板参数化 `<...,NTH,LBN_>`，8-warp 用
`<512,1,true,256,32>`（LBM=128/LBN=32/PIPE=1，smem 199,680 B）。

| case (D=512 full) | LSE 4-warp 最优 | LSE 8-warp 最优 | 端到端 lse8w 0→1 |
|---|---|---|---|
| b3_t1792 (maxlen 1024) | 0.0410 ms (split4) | **0.0374 ms (split8, 1.09×)** | 0.4639 → **0.4601 ms（1.008×）** |
| b1_t512 (maxlen 512) | **0.0137 ms (split8)** | 0.0171 ms (0.80×) | 0.0957 → 0.1049 ms（0.91×） |

ncu（fp16 b3 full）：4w→8w `gpu__time_duration` 41.1→**37.2 µs**、`sm__warps_active` 6.25%→**12.49%**，
但 `short_scoreboard` 0.90→**2.59**（LBN=32 使 tile/barrier 翻倍）⇒ **短序列净负、长序列小正**。
**默认 `--lse8w=0`（opt-in）**；数值 vs ref 与历史同量级、单/两文件逐指标一致。
详见 `docs/01` §15c、`docs/01b` §6ap、`docs/08` §5.21。

## 29. O57（第一百零四轮）：full MLA varlen 的 LSE 真正冲 2 CTA/SM——混合结果

落实 O54/O56「下一步候选 ①」。O54/O56 的 FULL LSE 恒 smem 199,680 B（1 CTA/SM）。本轮把 smem
压到 99,840 B（`=232448/2` 以内）让**两个 CTA 同驻一个 SM**：4-warp 固定 LBM=64，取
**cfg6 `<512,1,true,128,16>`**（PIPE=1/LBN=16，保留 cp.async）与 **cfg5 `<512,0,true,128,32>`**
（PIPE=0/LBN=32，丢双缓冲）。`--lseocc=5/6` opt-in、默认 0 逐位不变；`[O57 A/B]` 同 binary 扫参。

| LSE 几何（D=512 full） | b3_t1792 (1024) | b1_t512 (512) | smem | CTA/SM |
|---|---|---|---|---|
| 默认 `P1/LBN64` | **0.0410 (split4)** | **0.0135 (split8)** | 199.7KB | 1 |
| cfg4 `P0/LBN64` | 0.1043 | 0.0277 | 133.1KB | 1 |
| cfg7 `P1/LBN32` | 0.0563 | 0.0159 | 133.1KB | 1 |
| cfg5 `P0/LBN32` | 0.0675 | 0.0283 | 99.8KB | **2** |
| **cfg6 `P1/LBN16`** | **0.0394（1.04×）** | 0.0149（**0.91×**） | 99.8KB | **2** |
| O56 8-warp（参考） | 0.0378（1.08×） | 0.0171（0.79×） | 199.7KB | 1 |

ncu（fp16 b3 full，split4）：默认 1 CTA `sm__warps_active` **6.25%** / `short_scoreboard` 0.89 /
Duration 40.9µs → cfg6 2 CTA **10.51%** / 1.56 / 44.0µs ⇒ **occupancy 精确翻倍**，但 LBN=16 让
tile/barrier 变 4×，`short_scoreboard`+`wait` 上升抵消之；只有长 K（tile 多）小胜，且仍不及
8-warp。**丢 cp.async（cfg4/5）灾难性变慢**。⇒ **LSE 的墙是 compute/softmax + smem→mma 的
tile 依赖，不是可被 CTA occupancy 掩盖的访存延迟；默认 `--lseocc=0`。**
数值 vs fp32 ref 与 O54–O56 **逐位相同**（fp16 b3 5.516/4.451/2.385e-4；bf16 3.100/3.526/2.316e-3）。
详见 `docs/01` §15d、`docs/01b` §6aq。

## 30. O58（第一百零五轮）：causal MLA varlen 的 LSE 冲 2 CTA/SM（fp16/bf16/fp8）—— **正结果，causal 默认**

补齐 O56/O57 只做 **full** 的缺口：把 LSE 的几何候选（cfg6 的 2 CTA/SM、8-warp）搬到
**causal 镜像配对版** `lse_mma_kernel_bal<512,1>`（fp16/bf16 host-only；fp8 顺带把 device
模板参数化为 `<HD,PIPE,FULL,NTH,LBN_>`）。**causal 默认切 cfg6**
（`<512,1,false,128,16>`，PIPE1/LBN16），`--lseocc=4` 退回旧默认、`--lse8w=1` opt-in；
full 仍走 O54 旧路（O56/O57 判其混合）。

| dtype / case (causal) | 旧默认 P1/LBN64 | **新默认 cfg6** | LSE 比 | 端到端 total 比 |
|---|---|---|---|---|
| fp16 b1_t512 | LSE 0.0213 (sp8) | **0.0180 (sp16)** | 1.18× | 0.0836 → **0.0783 ms（1.07×）** |
| fp16 b3_t1792 | LSE 0.0388 (sp4) | **0.0318 (sp8)** | 1.22× | 0.3517 → **0.3206 ms（1.10×）** |
| bf16 b1_t512 | 0.0213 | **0.0180** | 1.18× | 0.0836 → **0.0783 ms（1.07×）** |
| bf16 b3_t1792 | 0.0388 | **0.0321** | 1.21× | 0.3517 → **0.3206 ms（1.10×）** |
| fp8 b1_t512 | 0.0251 (sp8) | **0.0198 (sp16)** | 1.27× | 0.1006 → **0.0965 ms（1.04×）** |
| fp8 b3_t1792 | 0.0366 (sp8) | **0.0319 (sp16)** | 1.15× | 0.2997 → **0.2789 ms（1.07×）** |

**ncu / mechanism**：fp16/bf16 旧默认 199.7KB = **1 CTA/SM**，cfg6 99.8KB = **2 CTA/SM**：
b3 causal 同 split8 下 Duration **43.07→30.24µs（1.42×）**、`sm__warps_active` 6.25%→10.38%。
**fp8 特殊**：一元素 1B ⇒ 旧默认仅 100.6KB **本就 2 CTA/SM**，cfg6 49.9KB 给到 **4 CTA/SM**
（11.14%→18.26%）、Duration 40.70→**27.07µs（1.50×）**。bound 从「低 occupancy」转向
「compute/softmax + `wait` 固定延迟」。
**关键对照**：同一杠杆在 **full** 上为混合（O57 cfg6 长 K 1.04×、短 K 0.91×）——**causal 的
镜像配对让每 CTA 的 K 链更长、2 CTA/SM 真正吃进延迟**，故 causal 净正、full 不默认。

数值（ours vs fp32 ref，max_abs dq/dk/dv）：fp16 b3 causal `2.415/1.834/1.856e-3`、b1
`1.303/1.537/1.557e-3`；bf16 b3 `1.267e-2/1.217e-2/1.796e-2`；fp8 b3 `3.404e-1/3.436e-1/3.508e-1`;
与 O53 历史一致（fp16/fp8 的 dq 有跨 CTA atomic 归约次序的既有非确定性，多次运行 2.4–5.2e-3）。
full 路径（`--full`）默认不变。单/两文件逐指标一致。
详见 `docs/01` §15e、`docs/01b` §6ar、`docs/03` §54、`docs/08` §5.23。

## 31. O59（第一百零六轮）：把 causal MLA LSE `cfg6` 从 varlen 推广到**定长**（fp16/bf16/fp8）—— **正结果，定长 MLA 默认；含一处 device 竞争修复**

O58 的 `cfg6`（PIPE1/LBN16，压 smem 换 CTA 级并行度）只落在 **varlen** 路径；4 个**定长** MLA
生产形状的 causal LSE 一直用旧默认 `<512,1>`（1 CTA/SM）。O59 把它接到定长路径并默认化
（`--lseocc=4` 退回旧默认），同时修掉一处 latent `cp.async` 竞争（`nuse==0` 的切片不 drain Q
拷贝，导致 cfg6 在大 ksplit 下 LSE 非确定抖动 ~1e-2；修法：每 m 块末尾补 `cp.async.wait_group 0`）。

**LSE-only（同 binary A/B，`[O59 A/B]`）与端到端 total（定长 causal MLA）**：

| dtype | S256H2 LSE / total | S512H4 LSE / total | S1024H2 LSE / total |
|---|---|---|---|
| fp8 | 1.195× / 0.0720→**0.0649（1.11×）** | 1.206× / 0.1340→**0.1280（1.05×）** | 1.196× / 0.1882→**0.1808（1.04×）** |
| fp16 | 1.184× / 0.0534→**0.0502（1.06×）** | 1.197× / 0.1180→**0.1146（1.03×）** | 1.040× / 0.1884→**0.1836（1.03×）** |
| bf16 | 1.183× / 0.0535→**0.0504（1.06×）** | 1.204× / 0.1188→**0.1162（1.02×）** | 1.037× / 0.1874→**0.1832（1.02×）** |

**ncu（S512H4）**：fp8 legacy 102.14KB/2 CTA/SM、Duration 23.62µs → cfg6 **51.07KB/4 CTA/SM**、
**18.98µs（1.24×）**；fp16 legacy 199.68KB/**1 CTA/SM**、19.97µs → cfg6 **99.84KB/2 CTA/SM**、
**16.19µs（1.23×）**。bound：低 occupancy / 半个波 → compute/softmax + wave 量化。

**数值**：cfg6-vs-legacy LSE max_abs 4.768e-7；ours-vs-ref 与历史逐位一致（fp8 S256H2
`2.356e-1/2.290e-1/3.441e-1`、S512H4 `2.415e-1/2.992e-1/4.481e-1`、S1024H2
`2.232e-1/3.337e-1/3.602e-1`；fp16 1.6–2.9e-3；bf16 ~1e-2）；**D=128 回归逐位不变**。
单/两文件逐指标一致。详见 `docs/01` §15f、`docs/01b` §6as、`docs/03` §55、`docs/08` §5.24。

---

## 32. P3-3：把「ours vs ref vs FA/TE」对拍收敛成统一 harness（本轮新增）

**动机**：此前数值对拍散落在各 dtype 的 kernel 自测输出（`src/**/*.out.txt`）与
`harness/fa_bwd_bench.py`（只算 FA/TE 两列）里，`docs/04` §1/§7 的表格靠人工誊抄，
更新一处要重跑多份 out。P3-3 的目标是：**一个可复现的 harness 直接产出数值表**，
作为本文档数值节的唯一来源。

### 32.1 新增物

- **`harness/fa_bwd_compare.py`**：纯 numpy（不需 torch/GPU）。扫描
  `/home/xieminglin/proj/output/fa-bwd/<case>/`，以 `ref_{dq,dk,dv}.npy` 为 baseline，
  对目录里存在的 `<impl>_{dq,dk,dv}.npy`（`impl ∈ {fa, te, ours, ours_sf, ...}`）逐元素算
  `max_abs` 与 `max_rel`（`max_rel = max(|a-b|/(|b|+1e-3))`，与 kernel 内 `diff_stat` 一致）。
  支持 `--dtype/--glob/--case/--impls/--markdown/--out`；varlen 的 packed `[T,H,D]` 天然适用。
- **ours 输出落盘**：6 个 host（fp16/bf16/fp8 × 单文件/两文件）新增 `--dump=<prefix>`，
  把 ours 的 dq/dk/dv 写成 `<prefix>_{dq,dk,dv}.npy`（`src/fa_bwd_dump.h`，默认关，
  不影响任何既有行为/数值）。此前 kernel 只把 `[compare]` 打到 stdout、无法被 harness 复用。

### 32.2 实测（`scripts/run.sh` 默认 sm_90 构建、mma 路径；原始输出
`src/fa_bwd_compare_p33_summary.out.txt`，markdown `src/fa_bwd_compare_p33_markdown.md`）

**causal MHA 核心表（ours / FA / TE vs fp32 ref，max_abs；FA 仅 fp16/bf16，fp8 无 FA）**：

| shape | dtype | ours dq/dk/dv | FA dq/dk/dv | TE dq/dk/dv |
|---|---|---|---|---|
| (1,512,16,128) | fp16 | 1.671/1.771/1.899e-3 | 1.679/1.684/1.899e-3 | 1.716/2.287/1.899e-3 |
| (1,4096,16,128) | fp16 | 1.883/1.734/1.966e-3 | 1.883/1.734/1.966e-3 | 1.883/1.858/1.966e-3 |
| (1,512,16,128) | bf16 | 9.00/12.61/13.65e-3 | 10.40/12.61/13.65e-3 | 13.74/10.68/13.65e-3 |
| (1,4096,16,128) | bf16 | 15.10/13.40/16.31e-3 | 14.41/13.32/16.31e-3 | 15.24/17.63/16.31e-3 |
| (1,512,16,128) | fp8 | 0.2426/0.2975/0.3735 | — | 0.4859/0.4038/0.5906 |
| (1,1024,32,128) | fp8 | 0.2400/0.4195/0.3536 | — | 0.4360/0.4498/0.8558 |
| (1,4096,16,128) | fp8 | 0.2635/0.2643/0.3216 | — | 0.3763/0.3686/0.6688 |

**GQA/MQA（fp8，ours vs ref）**：h32kv4 `0.2517/0.5408/0.7072`、h40kv8
`0.2869/0.5390/0.7107`、h64kv4 `0.2760/0.8456/1.226`、h64kv1(MQA) `0.4097/1.519/2.127`
（均 ≤ TE-vs-ref，与 §7.4 历史一致）。

**MLA（head_dim=512，fp8，ours vs ref；FA/TE 反向均不支持）**：S256H2
`0.2356/0.2290/0.3441`、S512H4 `0.2415/0.2992/0.4481`、S1024H2 `0.2232/0.3337/0.3602`
（与 §14 逐位一致）。

**单文件 vs 两文件（同一 case、各跑一次）**：`dq` 逐位相同；`dk/dv` 差异仅来自跨 CTA
`atomicAdd` 的 fp32 求和次序（fp16 ≤2.44e-4、bf16 ≤1.95e-3、fp8 ≤4.77e-7），属既有
非确定性、非实现差异。

### 32.3 判读与对旧表的说明

- **一致**：fp8 全部 shape（MHA/GQA/MQA/MLA）与历史表**逐位/同量级一致**；bf16 一致。
- **一处需要注意**：`§1.1` 里 fp16 S4096 的 ours 旧值 `1.499/1.572/2.225e-3` 来自更早的
  混合构建/路径，本轮默认 sm_90 **mma** 构建实测为 `1.883/1.734/1.966e-3`（≈FA）。
  两者都在 fp16 噪声内、且都 ≤ TE；差异来自不同 kernel 路径的 fp32 累加次序。
  今后以 **`fa_bwd_compare.py` 的实测输出**为准（构建可复现）。
- 复现命令：
  `scripts/run.sh src/<dtype>/fa_bwd_<dtype>_mma_{main.cu,onefile.cu} --dir=<case> --dump=ours`；
  再 `python3 harness/fa_bwd_compare.py --markdown`。

## 33. P3-3b：把 varlen（含 full / MLA）纳入同一 dump/汇总（本轮新增）

**动机**：§32 的 harness 只覆盖定长三 dtype 的 17 个 case（FA/TE 列也在），**varlen
（`cu_seqlens` packed）与其中的 full / MLA 从未进入统一对拍**——它们的 ours 输出只在
各轮 kernel 自测 stdout 里，无法被 `fa_bwd_compare.py` 复用。本轮落实 §32 的下一步候选 ①：
把 varlen 也接入 `--dump` + 统一汇总。

### 33.1 新增物

- **`run_varlen` 支持 `--dump=<prefix>`**：6 个 host（fp16/bf16/fp8 × 单/两文件）的 varlen
  自测入口在算完 packed `h_dq/h_dk/h_dv` 后，按前缀落 `npy`（复用 `src/fa_bwd_dump.h`；
  默认关 ⇒ 既有 varlen 行为逐位不变）。两文件用 `--dump=ours`、单文件 `--dump=ours_sf`。
- 构建需 `-DFA_WGMMA`（fp16/bf16 的 varlen 入口包在 `#ifdef FA_WGMMA` 内）：
  `ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" scripts/run.sh <host> ...`。

### 33.2 实测（全部 **38 个 varlen case** × 单/两文件；原始输出
`src/{fp16,bf16,fp8}/fa_bwd_<dtype>_p33b_varlen_dump.out.txt`，
汇总 `src/fa_bwd_compare_p33b_varlen_summary.out.txt`）

**判据用 `max_abs`（baseline = fp32 ref）；FA/TE 本机版本不支持 varlen，仅 ours-vs-ref。**
按 dtype × causal/full 的区间归纳（完整逐 case 表见汇总 out）：

| dtype | causal varlen max_abs 区间 | full varlen max_abs 区间 | 量级判读 |
|---|---|---|---|
| fp16 | 4.1e-4 – 3.4e-3 | 3.0e-4 – 1.6e-3 | fp16 噪声（~1e-3） |
| bf16 | 5.3e-3 – 3.1e-2 | 1.6e-3 – 1.2e-2 | bf16 噪声（~1e-2） |
| fp8 | 1.6e-1 – 6.2e-1 | 4.1e-2 – 2.5e-1 | fp8 噪声（O(0.1–0.6)） |

代表性逐项（dq/dk/dv max_abs）：
- fp8 MLA causal `b1_t512_h2_d512` `1.613e-1/2.238e-1/3.864e-1`、`b3_t1792_h2_d512`
  `3.404e-1/3.436e-1/3.508e-1`（与 §30/O58 历史一致）；
- fp8 MLA full `b1` `5.26/5.22/4.22e-2`、`b3` `7.99/9.63/4.15e-2`；
- fp8 D=128 causal `b5_h32` 最大（`5.57/6.20e-1`，amax ~10，属 fp8 相对噪声）；
- fp16 full `b4_t4096_h16_d128` `4.09e-4/4.95e-4/1.23e-4`、bf16 full `b3_t1792_h2_d512`
  `3.10e-3/3.53e-3/2.32e-3`。

**单文件 vs 两文件（同一 case）**：114 个向量对逐元素比对，最大差
`fp16 9.77e-4 / bf16 3.91e-3 / fp8 2.86e-6`——均为一两个 dtype ulp，来自跨 CTA `atomicAdd`
求和次序（varlen 的 amax 大于定长，故 ulp 也略大），**非实现差异**；dQ 在两文件/单文件间
逐位相同（寄存器累加 + 唯一 CTA 写）。

### 33.3 复现

```bash
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA" \
  scripts/run.sh src/<dtype>/fa_bwd_<dtype>_mma_main.cu --varlen --dir=<case> [--full] --dump=ours
python3 harness/fa_bwd_compare.py --glob 'varlen_*' --out <out.txt>
```

（可选）§32 候选 ② 仍未做：让 `fa_bwd_compare.py` 直接驱动 `run.sh` 一键「跑 ours + 汇总」，
以及把 FA/TE 的 varlen 列接进 harness（本机 FA2.7.4/TE2.14 反向不支持 varlen，故暂无列）。

## 34. P3-3c：`fa_bwd_run.py` 一键「跑 ours + 汇总」（本轮新增）

**动机**：§32/§33 把 ours 的 `--dump` 与 `fa_bwd_compare.py` 接进了 harness，但每次仍要手写
`scripts/run.sh <host> --dir=<case> --dump=ours [--full] [--varlen]`（还要区分单/两文件、定长/
varlen 两套构建），再把 case 名喂给 `fa_bwd_compare.py`。选取的 case 一多就极易漏项、无法
一键复现。本轮落实 §33 结尾的候选 ①：把这两步收敛成**一个命令**。

### 34.1 新增物 `harness/fa_bwd_run.py`

- **自动发现**：扫描 `/home/xieminglin/proj/output/fa-bwd/<case>/`，按 `meta.json` 的
  `dtype` / `varlen` / `causal` 过滤并选 host；要求存在 `ref_{dq,dk,dv}.npy`。
- **host 映射**（两文件=`ours`，单文件=`ours_sf`，与 §32/§33 口径一致）：

  | dtype | 两文件 host | 单文件 host |
  |---|---|---|
  | fp16 | `src/fp16/fa_bwd_fp16_mma_main.cu` | `src/fp16/fa_bwd_fp16_mma_onefile.cu` |
  | bf16 | `src/bf16/fa_bwd_bf16_mma_main.cu` | `src/bf16/fa_bwd_bf16_mma_onefile.cu` |
  | fp8  | `src/fp8/fa_bwd_fp8_main.cu` | `src/fp8/fa_bwd_fp8_mma_onefile.cu` |

- **构建/运行**：定长走 `scripts/run.sh` 默认 `sm_90`（mma，与 §32 表一致）；varlen 入口在
  `#ifdef FA_WGMMA` 内，改用 `ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA"`。
  **同一 (host, 构建配置) 只编译一次**，其余 case 直接复用已产出的可执行文件（`docker exec`）。
- **落盘**：每个 case 的原始输出写 `src/<dtype>/fa_bwd_<dtype>_p33c_run.out.txt`；运行清单
  `src/fa_bwd_run_p33c_summary.out.txt`；最后调用 `fa_bwd_compare.py` 写
  `src/fa_bwd_compare_p33c_summary.out.txt`。
- CLI：`--dtype/--glob/--case/--impls {twofile,singlefile,both}/--fixed-only/--varlen-only/
  --iters/--no-run/--dry-run`；可从任意 cwd 运行，内部用绝对路径与 `kernel_lab` 容器。

### 34.2 实测（**全部 73 个 case × 单/两文件 = 146 次运行**；原始输出
`src/{fp16,bf16,fp8}/fa_bwd_<dtype>_p33c_run.out.txt`，汇总 `src/fa_bwd_run_p33c_summary.out.txt`、
`src/fa_bwd_compare_p33c_summary.out.txt`）

- **146 次运行、0 失败**；只编译 **12 次**（6 host × 定长/varlen 两配置），其余 **134 次**复用二进制。
- **判据 `max_abs`（baseline = fp32 ref）**，按 dtype × 定长/varlen × causal/full 的区间：

  | dtype | 定长 causal | 定长 full | varlen causal | varlen full | 量级 |
  |---|---|---|---|---|---|
  | fp16 | 1.6e-3 – 7.9e-3 | 1.2e-4 – 3.3e-4 | 1.3e-3 – 3.8e-3 | 1.2e-4 – 1.6e-3 | fp16 噪声 |
  | bf16 | 5.8e-3 – 7.2e-2 | 1.5e-3 – 1.9e-3 | 8.0e-3 – 3.1e-2 | 1.6e-3 – 1.2e-2 | bf16 噪声 |
  | fp8  | 2.2e-1 – 2.13 | 4.0e-2 – 5.5e-2 | 1.6e-1 – 6.2e-1 | 4.2e-2 – 2.5e-1 | fp8 噪声 |

  各区间的上界都落在 **GQA/MQA（`kv1`/`kv4`）**（多 Q 头共享 KV 头、amax 更大）：fp16 MQA
  `b1_s1024_h64_d128_kv1_causal_fp16` `2.29/7.93/7.52e-3`、bf16 同 case `1.19e-2/4.56e-2/7.20e-2`、
  fp8 同 case `4.10e-1/1.52/2.13`——与 §7.4/§7.5/§7.6 历史同量级。
- **与历史逐位/同量级一致**：fp16 S512 `1.671/1.771/1.899e-3`、S4096 `1.883/1.734/1.966e-3`
  （= §32）；fp8 S512 `0.2426/0.2975/0.3735`（= §32）；varlen 三 dtype causal/full 区间
  与 §33 完全一致。
- **单文件 vs 两文件**：全 73 case 的 `{dq,dk,dv}` 逐元素比对，最大差 **fp16 3.91e-3 /
  bf16 7.81e-3 / fp8 7.63e-6**（均 1–2 个 dtype ulp，来自跨 CTA `atomicAdd` 求和次序；GQA/MQA
  的 amax 更大故 ulp 略大于 §32 的 MHA 值），**非实现差异**；`dq` 基本逐位相同。

### 34.3 复现

```bash
# 一键跑全部（或 --dtype fp8 / --varlen-only / --impls twofile 等收窄）
python3 harness/fa_bwd_run.py
# 只重新汇总已有 npy（不编译）
python3 harness/fa_bwd_run.py --no-run
# 预览将执行的命令
python3 harness/fa_bwd_run.py --case b1_s512_h16_d128_causal_fp8 --dry-run
```

工具本身是纯 host 编排（不碰 device 代码）；数值与 §32/§33 的 kernel 完全一致，只是把
「选 case → 选 host → 编译 → 跑 → 落盘 → 汇总」自动化。**下一步候选**：① 把 FA/TE 的
varlen 列接进 harness（本机 FA2.7.4/TE2.14 反向不支持 varlen，暂无列）；② 让
`fa_bwd_compare.py --markdown` 直接产出 `docs/04` 的表格片段（文档与实测同步）。

## 35. P3-3d：`fa_bwd_compare.py --doc-table` 直出 docs/04 分组表（本轮新增）

**动机**（落实 §34 的下一步候选 ②）：§1/§7 的数值表一直是**人工誊抄**，与实测输出容易脱节
（§32.3 已发现 §1.1 fp16 S4096 的旧值来自更早的混合构建）。本轮给 harness 加一个
**`--doc-table`** 模式：把已算好的 `ours / FA2.7.4 / TE2.14` 的 `max_abs` 直接渲染成
「按 dtype × 家族（MHA / GQA-MQA / MLA / varlen）分组」的 markdown，供 docs/04 直接内联。

### 35.1 新增物

- `harness/fa_bwd_compare.py --doc-table`：新增家族判定（`varlen` / `D≠128→MLA` /
  `Hkv≠H→GQA-MQA` / 其余 MHA）与 `shape_str`（含 `Hkv`/`Dv`/`causal|full`）。默认
  **只列两文件版（ours）**、不重复列单文件（`--impls ours_sf` 可显式加回）；组内按
  `causal→S→B→H` 排序。
- `harness/fa_bwd_run.py --doc-table`：一键「跑 ours + 汇总」时额外调用上面的模式，落盘
  `src/fa_bwd_compare_p33d_table.md`（默认路径）。
- 原始输出：`src/fa_bwd_compare_p33d_table.md`（分组表）、
  `src/fa_bwd_compare_p33d_summary.out.txt`（同样的 stdout 存档）。

### 35.2 实测（`--doc-table`，baseline = fp32 ref，判据 max_abs）

下面是 **定长**（MHA / GQA-MQA / MLA）的分组表，均取自本轮实测（可直接与 §1/§7 对照）。
varlen 的逐 case 表较长，见 `src/fa_bwd_compare_p33d_table.md`，区间见 §33/§34。

**fp16**

| shape | impl | dq | dk | dv |
|---|---|---|---|---|
| (1,512,16,128) causal | FA2.7.4 | 1.679e-03 | 1.684e-03 | 1.899e-03 |
|  | TE2.14 | 1.716e-03 | 2.287e-03 | 1.899e-03 |
|  | **ours** | 1.671e-03 | 1.771e-03 | 1.899e-03 |
| (1,4096,16,128) causal | FA2.7.4 | 1.883e-03 | 1.734e-03 | 1.966e-03 |
|  | TE2.14 | 1.883e-03 | 1.858e-03 | 1.966e-03 |
|  | **ours** | 1.883e-03 | 1.734e-03 | 1.966e-03 |
| (1,1024,16,128) full | FA2.7.4 | 3.268e-04 | 2.523e-04 | 1.217e-04 |
|  | TE2.14 | 1.558e-04 | 2.297e-04 | 1.217e-04 |
|  | **ours** | 3.268e-04 | 2.523e-04 | 1.225e-04 |

GQA/MQA（causal）：`h32kv4` ours `2.134/3.305/3.850e-3`、`h40kv8` `2.008/2.931/3.891e-3`、
`h64kv1`(MQA) `2.292/7.934/7.517e-3`、`h64kv4` `2.348/5.704/3.893e-3`（FA/TE 同量级）。
MLA（D=512，causal；FA/TE 反向不支持）：S256H2 `1.638/1.582/1.753e-3`、S512H2
`1.584/1.580/1.859e-3`、S512H4 `2.516/2.916/1.724e-3`、S1024H2 `1.987/1.712/1.848e-3`；
S512H2 full `2.861/3.331/1.241e-4`。

**bf16**

| shape | impl | dq | dk | dv |
|---|---|---|---|---|
| (1,512,16,128) causal | FA2.7.4 | 1.040e-02 | 1.261e-02 | 1.365e-02 |
|  | TE2.14 | 1.374e-02 | 1.068e-02 | 1.365e-02 |
|  | **ours** | 9.001e-03 | 1.261e-02 | 1.365e-02 |
| (1,4096,16,128) causal | FA2.7.4 | 1.441e-02 | 1.332e-02 | 1.631e-02 |
|  | TE2.14 | 1.524e-02 | 1.763e-02 | 1.631e-02 |
|  | **ours** | 1.510e-02 | 1.340e-02 | 1.631e-02 |
| (1,1024,16,128) full | FA2.7.4 | 1.827e-03 | 1.684e-03 | 1.449e-03 |
|  | TE2.14 | 2.421e-03 | 2.970e-03 | 1.449e-03 |
|  | **ours** | 1.938e-03 | 1.684e-03 | 1.449e-03 |

GQA/MQA（causal）：`h32kv4` ours `1.201/2.125/3.156e-2`、`h40kv8` `1.233/1.930/3.150e-2`、
`h64kv1`(MQA) `1.190/4.558/7.196e-2`、`h64kv4` `1.351/3.091/4.420e-2`（多数 ≤ FA/TE）。
MLA（D=512，causal）：S256H2 `1.230e-2/9.875e-3/1.686e-2`、S512H2 `8.795e-3/8.073e-3/1.339e-2`、
S512H4 `8.753e-3/1.082e-2/1.740e-2`、S1024H2 `5.838e-3/9.519e-3/1.568e-2`。

**fp8**（FA 无反向 FP8，仅 TE 两列）

| shape | impl | dq | dk | dv |
|---|---|---|---|---|
| (1,512,16,128) causal | TE2.14 | 4.859e-01 | 4.038e-01 | 5.906e-01 |
|  | **ours** | 2.426e-01 | 2.975e-01 | 3.735e-01 |
| (1,1024,32,128) causal | TE2.14 | 4.360e-01 | 4.498e-01 | 8.558e-01 |
|  | **ours** | 2.400e-01 | 4.195e-01 | 3.536e-01 |
| (1,4096,16,128) causal | TE2.14 | 3.763e-01 | 3.686e-01 | 6.688e-01 |
|  | **ours** | 2.635e-01 | 2.643e-01 | 3.216e-01 |
| (1,1024,16,128) full | TE2.14 | 9.040e-02 | 7.548e-02 | 9.699e-02 |
|  | **ours** | 5.518e-02 | 5.309e-02 | 4.007e-02 |

GQA/MQA（causal，TE vs ours）：`h32kv4` `5.246/7.664/13.43e-1` vs `2.517/5.408/7.072e-1`、
`h40kv8` `8.453e-1/6.625e-1/1.106` vs `2.869e-1/5.390e-1/7.108e-1`、`h64kv1`(MQA)
`4.100e-1/2.218/2.597` vs `4.097e-1/1.519/2.127`、`h64kv4` `3.983e-1/1.011/1.844` vs
`2.760e-1/8.456e-1/1.226`——**ours 全面 ≤ TE**。
MLA（D=512，causal）：S256H2 `2.356/2.290/3.441e-1`、S512H2 `2.286/2.252/3.343e-1`、
S512H4 `2.415/2.992/4.481e-1`、S1024H2 `2.232/3.337/3.602e-1`。

### 35.3 判读

- **与 §1/§7 对照**：fp8 全部 shape、bf16 MHA 与 §1.2 逐位一致；fp16 MHA S512 逐位一致，
  **S4096 的 `ours` 与 §1.1 的旧值不同**（旧 `1.499/1.572/2.225e-3` vs 现
  `1.883/1.734/1.966e-3`）——差异来自不同 kernel 路径的 fp32 累加次序，两者都在 fp16 噪声
  内、且都 ≤ TE。**自本轮起，docs/04 的定长或 varlen 数值判据一律以 `--doc-table` 的实测输出为准。**
- **结论未变**：三 dtype 的 dq/dk/dv 均落在对应精度噪声量级；fp8 的 ours-vs-ref 全面优于
  TE-vs-ref（口径更保守：dS/输出 fp32、dP 不进张量核）。
- 复现：`python3 harness/fa_bwd_compare.py --doc-table --out <md>`，或
  `python3 harness/fa_bwd_run.py --doc-table`（先跑 ours 再出表）。

> 注：本节是**纯 harness 增量**（不碰 device 代码），没有新的 kernel/性能数字与 ncu 度量；
> §2/§3 的性能与 bound 结论仍适用。**下一步候选**：①（若本机版本将来支持）把 FA/TE 的
> varlen 列接进 harness；② 让 `--doc-table` 直接改写 docs/04 的对应小节（含自动 diff 校验）。

## 36. FA3（SM90）变长反向基线接入 + FA 口径切到 FA3（第 111 轮）

**动机**：§33/§35 的候选 ① 一直记「FA/TE 的 varlen 列：本机 FA2.7.4/TE2.14 反向不支持 varlen，
暂无列」。但**本机 `flash_attn_3` 3.0.0（FA3/SM90）的反向支持 varlen**（`flash_attn_varlen_func`
可 autograd，已验证 fp16/bf16、MHA 与 GQA、causal 与 full 均可；head_dim ≤ 128，fp8/D=512 不支持）。
同时用户要求「与 FA 对比一律用 FA3、不要用 FA2.7.4」——故本轮：

1. `harness/fa_bwd_bench.py` 新增 `fa3_bwd`（定长）与 `fa3_bwd_varlen`（packed + `cu_seqlens`），
   接入 `dump`（落 `fa3_{o,dq,dk,dv}.npy`，D≤128 的非 fp8 case）与 `bench`（纯反向 CUPTI device time）。
   `fa`（FA2.7.4）保留为历史对照列，不再作为默认口径。
2. `harness/fa_bwd_compare.py` 的 `IMPL_ORDER` 加 `fa3`，`--doc-table` 默认口径改为
   **`fa3 / TE / ours`**（要 FA2 需显式 `--impls fa`）。
3. 重跑 17 个 D≤128 的（定长 + varlen）fp16/bf16 case 补齐 `fa3_*`（固定 shape 沿用原 `seed`
   以保证输入不变、既有 `ours_*` 依旧有效），并生成 `src/fa_bwd_compare_p111_doc_table.md`。

### 36.1 数值对拍（max_abs vs fp32 ref）

FA3 变长反向的 `dq/dk/dv` 与 fp32 ref 全部落在对应 dtype 噪声量级，**与我们同量级或更小**；
代表值（`ours` / `fa3`）：

| case | dtype | dq | dk | dv |
|---|---|---|---|---|
| varlen [1024]×4 H16 D128 causal | fp16 | 2.112e-03 / 2.149e-03 | 2.252e-03 / 1.970e-03 | 1.915e-03 / 1.915e-03 |
| varlen [512,1024,2048,256] H16 D128 causal | fp16 | 3.163e-03 / 1.938e-03 | 2.158e-03 / 2.158e-03 | 1.966e-03 / 1.940e-03 |
| varlen [128..2048] H32 kv8 D128 causal | fp16 | 2.438e-03 / 2.146e-03 | 3.433e-03 / 3.392e-03 | 3.843e-03 / 3.843e-03 |
| varlen [2048..8] H16 D128 causal | bf16 | 1.464e-02 / 1.214e-02 | 1.566e-02 / 1.519e-02 | 1.863e-02 / 1.863e-02 |
| varlen [1024]×4 H16 D128 full | fp16 | 4.094e-04 / 4.094e-04 | 4.953e-04 / 4.953e-04 | 1.234e-04 / 1.213e-04 |
| MHA S512 causal | fp16 | 1.671e-03 / 1.679e-03 | 1.771e-03 / 1.684e-03 | 1.899e-03 / 1.899e-03 |
| MHA S4096 causal | fp16 | 1.883e-03 / 1.883e-03 | 1.734e-03 / 1.734e-03 | 1.966e-03 / 1.966e-03 |

定长 MHA 与 GQA/MQA 的 FA3 列与 §35 的 FA2.7.4 列**多数一致或更优**（如 fp16 kv1 dv：
FA3 `7.517e-3` vs FA2 `1.057e-2`；bf16 kv4 dv：FA3 `4.420e-2` vs FA2 `6.511e-2`）；
唯一系统性差异是 FA3/SM90 与 FA2/SM80 的 fp32 累加次序不同（都 ≤ TE）。完整分组表见
`src/fa_bwd_compare_p111_doc_table.md`（由 `--doc-table` 直出，判据以它为准）。

### 36.2 性能对标（纯反向 device time；`4·H·D·Σ_b L_b²` 口径）

ours = 两文件 `fa_bwd_{fp16,bf16}_mma_main` 端到端（preprocess+main+convert，CUDA event，iters=200）；
FA3 = `fa3_bwd_varlen`（CUPTI device time）。原始输出 `src/fa_bwd_p111_varlen_fa3_perf.out.txt`。

| varlen case（H=16 D=128） | ours ms/TF | FA3 ms/TF | ours/FA3 |
|---|---|---|---|
| causal [1024]×4 fp16 | 0.4502 / 76.3 | 0.2409 / 142.6 | **1.87×** |
| causal [512,1024,2048,256] fp16 | 0.7756 / 58.8 | 0.2632 / 173.4 | 2.95× |
| causal [128..2048] H32 kv8 fp16 | 1.4283 / 64.1 | 0.5272 / 173.6 | 2.71× |
| causal [2048..8] tilt fp16 | 0.5904 / 62.3 | 0.2237 / 164.4 | 2.64× |
| full [1024]×4 fp16 | 0.9054 / 38.0 | 0.3127 / 109.9 | 2.90× |
| full [512,1024,2048,256] fp16 | 1.2818 / 35.6 | 0.3677 / 124.1 | 3.49× |
| full [128..2048] H32 kv8 fp16 | 2.4158 / 37.9 | 0.7402 / 123.7 | 3.26× |
| full [2048..8] tilt fp16 | 1.0671 / 34.5 | 0.3094 / 118.8 | 3.45× |

bf16 与 fp16 逐项几乎相同（见原始输出）。**判读**：

- **varlen 的 ours/FA3 比值（1.87–3.55×）明显好于定长 MHA S4096 的 ~7×**——原因是
  FA3 的变长反向在**短序列**（S≤2048）上效率本就下降（[1024]×4 只 142.6 TF，而定长 S4096
  约 845 TF），分母变小；这也说明「单序列长 S」才是 FA3 的主场。
- full（非 causal）的 ours 吞吐 ~35–38 TF，约为 causal 的一半（工作量翻倍），符合预期。
- **fp8 与 MLA（D=512）：FA3 反向不支持**（分别受 dtype / head_dim≤128 限制），
  这两类仍只有 ours + fp32 ref（fp8 另有 TE-vs-ref 数值列，见 §33）；**FA3 无法覆盖最重点的
  fp8**，故 fp8 的口径仍以 ours/TE/ref 三者为准。

> 本轮为**harness + 基线增量**，未改任何 device 代码；我们 kernel 的 bound 结论（§2/§3、
> §26–§31）不变，本条只补齐「FA3 varlen 数值 + 性能」这一此前缺失的对照列。
> **下一步候选**：① 把 FA3（含 varlen）接入 `harness/fa_vs_te_bwd_only.py` 的纯反向基线；
> ② 继续 O59 候选（MLA 主 kernel 降 smem / fp8 MLA `short_scoreboard`）；
> ③ 让 `--doc-table` 直接改写 docs/04 的对应小节（含自动 diff 校验）。

## 37. P3-4b：`fa_vs_te_bwd_only.py` 补齐 varlen 纯反向基线（第 112 轮，harness + 基线）

**动机**：落实 §36 的候选 ①。`harness/fa_vs_te_bwd_only.py` 是用户指定的**纯反向口径**基准
（forward 建图一次放在计时区之外，只对 `autograd.grad` 计时），此前只有定长 `SHAPES`；
而 `harness/fa_bwd_bench.py` 的 varlen `bench` 把 `fa3_bwd_varlen()`（**含 forward**）
整段放在 `device_time` 里，所以 §36.2 里 FA3 的 varlen 数字其实混入了前向。

### 37.1 变更（纯 harness，device 一行未改）

`harness/fa_vs_te_bwd_only.py` 新增：

- `VARLEN_SHAPES`（5 个：MHA 不齐 / MHA 等长 / GQA q32-kv8 / 强倾斜 / MHA 等长 full）；
- `bench_fa_varlen`（FA2.7.4 `flash_attn_varlen_func` 反向）与 `bench_fa3_varlen`
  （FA3.0.0 变长），**forward 在计时区外建图**，只对 `autograd.grad` 计时；
- `ref_varlen`（逐序列切片 fp32 autograd）与 `--verify`：对一只小 shape 打印两列 vs ref 的
  `max_abs`，证明新列可信；
- main 末尾固定输出「定长表 + varlen 表」。**TE2.14 变长反向在本容器报错/非法访存**
  （`Ragged QKV input requires padding or padding_causal mask` → cuDNN err 700），故 varlen
  表只有 FA2/FA3 两个有效列、TE 列标 `NA`。

**重要更正（推翻旧记）**：ROADMAP 长期记「本机 FA2.7.4/TE2.14 反向不支持 varlen」——
本轮实测 **FA2.7.4 的 `flash_attn_varlen_func` 反向是可用的**（causal/full、MHA/GQA 均可
autograd），且数值与 FA3 逐点一致（§37.2）。不支持的是 **TE2.14** 的 ragged 反向。

### 37.2 数值校验（`--verify`，causal，lengths=[128,256,64] H4 D128）

| dtype | FA2 dq/dk/dv (max_abs vs fp32 ref) | FA3 dq/dk/dv |
|---|---|---|
| fp16 | 1.56e-03 / 1.38e-03 / 1.78e-03 | 1.56e-03 / 1.38e-03 / 1.78e-03 |
| bf16 | 9.41e-03 / 1.13e-02 / 1.82e-02 | 9.41e-03 / 1.13e-02 / 1.82e-02 |

两列逐位相同、均在 dtype 噪声内 ⇒ 新增的 FA2/FA3 varlen 列可信。

### 37.3 性能（纯反向 device time；`4·H·D·Σ_b L_b²`；原始输出
`src/fa_bwd_p112_varlen_fa2_fa3_te.out.txt`）

fp16（bf16 逐项几乎相同）：

| varlen case | FA2.7.4 ms/TF | FA3 ms/TF | FA3/FA2 |
|---|---|---|---|
| causal [512,1024,2048,256] H16 kv16 | 0.3356 / 136 | **0.1625 / 281** | 2.07× |
| causal [1024]×4 H16 kv16 | 0.2516 / 137 | **0.1480 / 232** | 1.70× |
| causal [128..2048] H32 kv8 | 0.6321 / 145 | **0.3750 / 244** | 1.69× |
| causal [2048..8] H16 kv16 tilt | 0.3009 / 122 | **0.1412 / 260** | 2.13× |
| full [1024]×4 H16 kv16 | 0.3443 / 100 | **0.2002 / 172** | 1.72× |

**判读**：

- **FA3 变长反向稳定快 FA2 1.69–2.13×**（TMA+wgmma+更深流水），与定长趋势一致。
- **纯反向口径下 FA3 比 §36.2 快约 1.6–1.7×**（如 [1024]×4 causal：`0.1480` vs 之前 `0.2409`ms）
  ——差额正是被 §36.2 误计入的**前向**。因此用 §36.2 的 FA3 数字算 ours/FA3 会**偏乐观**；
  按本表纯反向后重算，ours（两文件端到端 event 时间，沿用 §36.2）对 FA3 为
  [1024]×4 causal `0.4502/0.1480=`**3.04×**、full **4.52×**、不齐 causal **4.77×**，
  即 §36.2 的「1.87–3.5×」应更正为 **3.0–4.8×**。**教训：`device_time(fn)` 里若 `fn` 同时
  包含 forward+backward，得到的就不是纯反向——同一口径的两条曲线不能省掉建图位置这个细节。**
- TE2.14 的变长反向不可用（本容器），故 varlen 仍无 TE 列；fp8 与 MLA（D=512）FA3 不支持
  （§36），最重点的 fp8 口径仍以 ours/TE/ref 三者为准。

> 本轮为 **harness + 基线增量**，未改 device 代码；我们 kernel 的 bound 结论不变。
> **下一步候选**：① ~~把 FA3（含 varlen）接入 `fa_vs_te_bwd_only.py`~~ **本轮已完成**；
> ② 用同一纯反向口径重测 `fa_bwd_bench.py` 里定长/变长的 FA/TE 列（现 `bench_case*` 也含 forward），
> 让全站基线统一；③ 继续 O59 候选（MLA 主 kernel 降 smem / fp8 MLA `short_scoreboard`）。

## 38. P3-4c：全站基线统一到「纯反向」口径（第 113 轮，harness + 基线）

**动机**（落实 §37 的候选 ②、`ROADMAP` backlog 最后一条 `[ ]`）：上一轮发现 `harness/fa_bwd_bench.py`
的 `bench_case` / `bench_case_varlen` 把 `fa_bwd()` / `fa3_bwd()` / `te_bwd()` / `te_bwd_fp8()` /
`fa3_bwd_varlen()`（**都含 forward**）整段塞进 `CudaTimer.device_time()`，而用户指定的纯反向基准
`harness/fa_vs_te_bwd_only.py`（forward 建图在计时区外、只测 `autograd.grad` / `fused_attn_bwd`）
口径不同——§36.2 的 varlen FA3 数字因此偏慢 1.6–1.7×。本轮把 `fa_bwd_bench.py` 的 `bench` 默认切到
**纯反向**，让全站（定长 + varlen × FA2/FA3/TE）基线口径统一。

### 38.1 变更（纯 harness，device 一行未改）

- 新增 `make_fa_bwd_only` / `make_fa3_bwd_only` / `make_te_bwd_only` / `make_te_fp8_bwd_only` /
  `make_fa3_varlen_bwd_only`：**forward 只建图/算一次（计时区外）**，返回一个只跑反向的闭包；
  `bench_case` / `bench_case_varlen` 默认用它们，`CudaTimer.device_time(闭包)` 只计反向。
- 新增 `--with-fwd`：退回旧口径（整段含 forward）以做 A/B；`dump` 路径仍用原来的 `*_bwd()` 取数值，
  完全不变。
- 打印头由 `=== bench CUPTI device time (dtype=..) ===` 改为带 `pure-bwd` / `fwd+bwd` 标注。

### 38.2 定长基线（纯反向 CUPTI；ms / TFLOPS @ `4·B·S²·H·(D+Dv)`）

MHA（H=16 D=128，causal；fp16，bf16 逐项几乎相同）：

| (B,S) | FA2.7.4 | **FA3** | TE2.14 |
|---|---|---|---|
| (1,512) | 0.0442 / 97.3 | **0.0263 / 163.2** | 0.0322 / 133.5 |
| (1,1024) | 0.0820 / 209.5 | **0.0486 / 353.2** | 0.0582 / 295.2 |
| (4,2048) | 0.7374 / 372.8 | **0.3934 / 698.8** | 0.4747 / 579.1 |
| (2,2048) full | 0.5882 / 233.7 | **0.3205 / 428.9** | 0.3555 / 386.6 |
| (1,4096) | 0.7365 / 373.2 | **0.3236 / 849.5** | 0.4518 / 608.4 |

fp8 定长（FA2/FA3 无反向 fp8，只有 TE）：

| (B,S) | TE2.14 |
|---|---|
| (1,512) | 0.0357 / 120.2 |
| (1,1024) | 0.0550 / 312.1 |
| (1,4096) | 0.3031 / 906.8 |

GQA/MQA（B1 S1024 D128 causal，纯反向 ms / TF；fp16）：

| 形状 | FA2.7.4 | **FA3** | TE2.14 |
|---|---|---|---|
| q40 / kv8 | 0.1875 / 229.0 | **0.1218 / 352.6** | 0.1315 / 326.7 |
| q32 / kv4 | 0.1569 / 219.0 | **0.0823 / 417.3** | 0.1118 / 307.4 |
| q64 / kv4 | 0.2625 / 261.8 | **0.1595 / 431.0** | 0.1942 / 354.0 |
| q64 / kv1 (MQA) | 0.2649 / 259.4 | **0.1566 / 438.9** | 0.2154 / 319.1 |

fp8 GQA/MQA（TE2.14，纯反向）：q40/kv8 `0.1232/348.5`、q32/kv4 `0.1038/330.9`、
q64/kv4 `0.2065/332.7`、q64/kv1 `0.2468/278.5`。

### 38.3 varlen 基线（纯反向 CUPTI；ms / TF @ `4·H·D·Σ_b L_b²`）

fp16（bf16 逐项几乎相同）：

| varlen case (H16 D128) | FA3 纯反向 | 旧口径（含 fwd） |
|---|---|---|
| causal [512,1024,2048,256] | 0.1634 / 279.3 | 0.2625 / 173.8 |
| causal [1024]×4 | 0.1480 / 232.1 | 0.2404 / 142.9 |
| causal [128..2048] GQA kv8 | 0.3752 / 244.0 | 0.5265 / 173.8 |
| causal [2048..8] 强倾斜 | 0.1405 / 261.7 | 0.2241 / 164.1 |
| full [1024]×4 | 0.1991 / 172.6 | — |

fp8（FA3 无反向 fp8）：等长 `[1024]×4` 走 TE FP8 定长等价口径 `0.1472/233.4`（纯反向）；
其余不等长 case TE2.14 变长反向仍 segfault、无列。

### 38.4 A/B：forward 占了多少、旧口径偏乐观多少

同 binary、同 shape，`--with-fwd`（旧）vs 默认（纯反向）：

| case | 纯反向 | 含 fwd | forward 占比 |
|---|---|---|---|
| fp16 MHA S512 FA3 | 0.0263 | 0.0451 | +71% |
| fp16 MHA S4096 FA3 | 0.3236 | 0.4642 | +43% |
| fp16 MHA S4096 TE | 0.4518 | 0.5964 | +32% |
| fp8 MHA S512 TE | 0.0357 | 0.1011 | **+183%** |
| fp8 MHA S4096 TE | 0.3031 | 0.5900 | **+95%** |

**forward 在 fp8 上占比最大（几乎翻倍）**——`fa_bwd_bench.py` 的旧 TE-fp8 口径把整个前向
（含量化/`fused_attn_fwd`）都计了进去；定长/变长的旧数字据此应全部作废，以本轮纯反向为准。
`--with-fwd` 复现出的 varlen `[1024]×4` FA3 `0.2404ms` 与 §36.2 的 `0.2409ms` 一致，
证明口径切换只动了建图位置、未动其它。

### 38.5 对 ours/FA3、ours/TE 比值的更正

ours = 两文件 `fa_bwd_{fp16,bf16,fp8}_mma_main` 端到端 event（preprocess+main+convert，
`--iters=100`，同 session）；对照列取本轮纯反向。原始输出 `src/fa_bwd_p113_ours_ref.out.txt`。

| case | ours | 参考（纯反向） | 比值 |
|---|---|---|---|
| fp16 MHA S512 | 0.0910 / 23.6 TF | FA3 0.0263 | **3.46×** |
| fp16 MHA S4096 | 1.9242 / 71.4 TF | FA3 0.3236 | **5.95×** |
| fp16 varlen [1024]×4 causal | 0.4507 / 76.2 TF | FA3 0.1480 | **3.05×** |
| fp16 varlen [1024]×4 full | 0.9036 / 38.0 TF | FA3 0.1991 | **4.54×** |
| fp8 MHA S4096 | 2.3777 / 57.8 TF | TE 0.3031 | **7.85×** |
| fp8 varlen [1024]×4 causal | 0.7729 / 44.5 TF | TE-fixed 0.1472 | **5.25×** |

- **更正值**：旧口径下 fp8 S4096 的 ours/TE 只有 ~4.0×（TE 含 forward 0.59ms），纯反向后是
  **7.85×**；varlen `[1024]×4` causal 的 ours/FA3 从 §36.2 的 1.87×、§37 的 3.04× 定为
  **3.05×**（与本轮 ours 0.4507 一致）。
- 定长 fp16 的 ours/FA3 由 ~6.0×（§6）微调为 **5.95×**；fp16 varlen full 为 **4.54×**。

### 38.6 数值回归与结论

- **数值零变化**：本轮只改 `bench` 的计时位置，device 一行未动。`harness/fa_bwd_compare.py`
  对 73 个既有 case 重扫，ours-vs-ref 与 §32–§37 历史**逐位一致**
  （fp16 S512 `1.671/1.771/1.899e-3`、S4096 `1.883/1.734/1.966e-3`；fp8 S512
  `2.426/2.975/3.735e-1`、S4096 `2.635/2.643/3.216e-1`；varlen 三 dtype 区间同量级）；
  原始输出 `src/fa_bwd_compare_p113_summary.out.txt`。
- 本轮为 **harness + 基线增量**，未改任何 kernel；各 dtype 的 bound 结论（§2/§3、§26–§31）不变。
- **下一步候选**：① 继续 O59 候选（MLA 主 kernel 降 smem / fp8 MLA `short_scoreboard`）；
  ② ~~`--doc-table` 直接改写 docs/04 对应小节（含自动 diff 校验）~~ **已完成（第 114 轮
  P3-3e，§39）**：`--apply`/`--check` 原地同步并校验内嵌自动块，含原子噪声容差；
  ③ 把纯反向口径复用到 `fa_bwd_bench.py --requested` 的 MLA 行（现 FA/TE 均 `NA`，只能列 ours）。

---

## 39. P3-3e：docs/04 数值表自动同步（`--doc-table --apply/--check`，第 114 轮）

**动机**（落实 §38 候选 ②）：§1/§7 的数值表一直靠**人工誊抄** `fa_bwd_compare.py` 的输出，
与实测易脱节（§32 已记录一次：fp16 S4096 旧值来自更早的混合构建）。§35 的 `--doc-table`
已能直出按 dtype×家族分组、可直接内联的 markdown；本轮再进一步，把它做成
**可原地改写 + 可 diff 校验**——docs/04 内嵌一个被标记围起来的自动块，`--apply` 用实测同步、
`--check` 只校验并以非零退出码报告陈旧（可进 CI）。

> **下面内嵌的 `auto-doc-table` 块是数值对拍表（max_abs vs fp32 ref，causal/full 全家族）
> 的唯一机器可读来源。** §1/§7 的人工表保留作历史叙述与逐项解读，但**判据一律以本块
> 与 `--doc-table` 实测为准**；`--check` 保证文档与 dump 不再脱节。

**用法**：
```bash
# 校验（不一致时打印 unified diff 并退出码 1）
python3 harness/fa_bwd_compare.py --check docs/04-numerics-and-perf-summary.md
# 原地同步
python3 harness/fa_bwd_compare.py --apply docs/04-numerics-and-perf-summary.md
# 跑 ours + 同步一把梭（调用 scripts/run.sh，在 kernel_lab 内编译）
python3 harness/fa_bwd_run.py --doc-table-apply
```

> 实现说明：`fa_bwd_compare.py` 新增 `--apply PATH` / `--check PATH`，只改两个标记
> （`<!-- BEGIN:auto-doc-table ... -->` … `<!-- END:auto-doc-table -->`）之间的内容，
> 块外正文一行不动；`fa_bwd_run.py` 新增 `--doc-table-apply`（跑完 ours 后同步）与
> `--doc-table-check`（只读校验）。本轮**纯 harness**，device 一行未改，数值零变化。
>
> **容差（`--rtol`，默认 `5e-3`）**：fp8 的 dK/dV 走跨 CTA `atomicAdd`、split-K 的 dQ 亦然，
> 每次运行的**末位**会抖动（本轮重跑即出现 `7.108e-01 ↔ 7.107e-01`）。因此 `--check`/`--apply`
> 按「逐行结构相同 + 数值在 `rtol` 内」判定等价，原子次序噪声不会把文档判成陈旧；要逐位
> 复核时用 `--rtol 0`（本轮 `--rtol 0` 确实报出该噪声，证明校验是逐值生效的）。

<!-- BEGIN:auto-doc-table (generated by harness/fa_bwd_compare.py --doc-table; do not edit by hand) -->

### fp16

**MHA**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| (1,512,16,128) causal | FA3.0.0 | 1.679e-03 | 1.684e-03 | 1.899e-03 |
|  | TE2.14 | 1.716e-03 | 2.287e-03 | 1.899e-03 |
|  | **ours（两文件）** | 1.671e-03 | 1.771e-03 | 1.899e-03 |
| (1,4096,16,128) causal | FA3.0.0 | 1.883e-03 | 1.734e-03 | 1.966e-03 |
|  | TE2.14 | 1.883e-03 | 1.858e-03 | 1.966e-03 |
|  | **ours（两文件）** | 1.883e-03 | 1.734e-03 | 1.966e-03 |
| (1,1024,16,128) full | FA3.0.0 | 3.268e-04 | 2.523e-04 | 1.217e-04 |
|  | TE2.14 | 1.558e-04 | 2.297e-04 | 1.217e-04 |
|  | **ours（两文件）** | 3.268e-04 | 2.523e-04 | 1.225e-04 |

**GQA/MQA**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| (1,1024,32,128) Hkv=4 causal | FA3.0.0 | 1.727e-03 | 3.321e-03 | 3.850e-03 |
|  | TE2.14 | 2.075e-03 | 3.175e-03 | 5.107e-03 |
|  | **ours（两文件）** | 2.134e-03 | 3.305e-03 | 3.850e-03 |
| (1,1024,40,128) Hkv=8 causal | FA3.0.0 | 1.757e-03 | 2.648e-03 | 3.891e-03 |
|  | TE2.14 | 1.580e-03 | 2.893e-03 | 4.326e-03 |
|  | **ours（两文件）** | 2.008e-03 | 2.931e-03 | 3.891e-03 |
| (1,1024,64,128) Hkv=1 causal | FA3.0.0 | 1.971e-03 | 7.586e-03 | 7.517e-03 |
|  | TE2.14 | 2.141e-03 | 6.452e-03 | 1.057e-02 |
|  | **ours（两文件）** | 2.292e-03 | 7.934e-03 | 7.517e-03 |
| (1,1024,64,128) Hkv=4 causal | FA3.0.0 | 1.911e-03 | 4.104e-03 | 3.893e-03 |
|  | TE2.14 | 2.167e-03 | 3.818e-03 | 5.651e-03 |
|  | **ours（两文件）** | 2.348e-03 | 5.704e-03 | 3.893e-03 |

**MLA**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| (1,256,2,512) causal | **ours（两文件）** | 1.638e-03 | 1.582e-03 | 1.753e-03 |
| (1,512,2,512) causal | **ours（两文件）** | 1.584e-03 | 1.580e-03 | 1.859e-03 |
| (1,512,4,512) causal | **ours（两文件）** | 2.516e-03 | 2.916e-03 | 1.724e-03 |
| (1,1024,2,512) causal | **ours（两文件）** | 1.987e-03 | 1.712e-03 | 1.848e-03 |
| (1,1024,8,256) causal | TE2.14 | 1.372e-03 | 1.398e-03 | 1.447e-03 |
|  | **ours（两文件）** | 1.657e-03 | 1.405e-03 | 1.447e-03 |
| (1,1024,16,256) Hkv=4 causal | TE2.14 | 1.983e-03 | 2.434e-03 | 3.971e-03 |
|  | **ours（两文件）** | 2.480e-03 | 2.816e-03 | 1.976e-03 |
| (1,2048,8,256) causal | TE2.14 | 2.023e-03 | 1.664e-03 | 1.614e-03 |
|  | **ours（两文件）** | 2.023e-03 | 1.481e-03 | 1.614e-03 |
| (1,512,2,512) full | **ours（两文件）** | 2.861e-04 | 3.331e-04 | 1.241e-04 |
| (1,1024,8,256) full | TE2.14 | 1.523e-04 | 1.405e-04 | 2.109e-04 |
|  | **ours（两文件）** | 1.850e-04 | 2.775e-04 | 2.109e-04 |

**varlen**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| varlen B=2 T=500 [300,200] H=16 D=128 causal | FA3.0.0 | 1.968e-03 | 2.684e-03 | 2.055e-03 |
|  | **ours（两文件）** | 1.832e-03 | 1.593e-03 | 2.055e-03 |
| varlen B=1 T=512 [512] H=2 D=512 causal | **ours（两文件）** | 1.303e-03 | 1.537e-03 | 1.557e-03 |
| varlen B=3 T=1792 [256,512,1024] H=2 D=512 causal | **ours（两文件）** | 2.415e-03 | 1.834e-03 | 1.856e-03 |
| varlen B=8 T=2904 [2048,512,128,96...] H=16 D=128 causal | FA3.0.0 | 2.624e-03 | 2.231e-03 | 2.139e-03 |
|  | **ours（两文件）** | 2.624e-03 | 2.158e-03 | 2.139e-03 |
| varlen B=4 T=3840 [512,1024,2048,256] H=16 D=128 causal | FA3.0.0 | 1.938e-03 | 2.158e-03 | 1.940e-03 |
|  | **ours（两文件）** | 3.163e-03 | 2.158e-03 | 1.966e-03 |
| varlen B=5 T=3968 [128,256,512,1024...] H=32 D=128 Hkv=8 causal | FA3.0.0 | 2.146e-03 | 3.392e-03 | 3.843e-03 |
|  | **ours（两文件）** | 2.438e-03 | 3.433e-03 | 3.843e-03 |
| varlen B=4 T=4096 [1024,1024,1024,1024] H=16 D=128 causal | FA3.0.0 | 2.149e-03 | 1.970e-03 | 1.915e-03 |
|  | **ours（两文件）** | 2.112e-03 | 2.252e-03 | 1.915e-03 |
| varlen B=1 T=512 [512] H=2 D=512 full | **ours（两文件）** | 3.046e-04 | 4.449e-04 | 1.327e-04 |
| varlen B=3 T=1792 [256,512,1024] H=2 D=512 full | **ours（两文件）** | 5.516e-04 | 4.451e-04 | 2.385e-04 |
| varlen B=8 T=2904 [2048,512,128,96...] H=16 D=128 full | FA3.0.0 | 1.486e-03 | 1.555e-03 | 1.582e-03 |
|  | **ours（两文件）** | 1.486e-03 | 1.555e-03 | 1.582e-03 |
| varlen B=4 T=3840 [512,1024,2048,256] H=16 D=128 full | FA3.0.0 | 5.603e-04 | 6.841e-04 | 2.463e-04 |
|  | **ours（两文件）** | 5.603e-04 | 6.841e-04 | 2.338e-04 |
| varlen B=5 T=3968 [128,256,512,1024...] H=32 D=128 Hkv=8 full | FA3.0.0 | 7.515e-04 | 7.856e-04 | 4.880e-04 |
|  | **ours（两文件）** | 7.341e-04 | 7.906e-04 | 4.880e-04 |
| varlen B=4 T=4096 [1024,1024,1024,1024] H=16 D=128 full | FA3.0.0 | 4.094e-04 | 4.953e-04 | 1.213e-04 |
|  | **ours（两文件）** | 4.094e-04 | 4.953e-04 | 1.234e-04 |

### bf16

**MHA**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| (1,512,16,128) causal | FA3.0.0 | 1.040e-02 | 1.261e-02 | 1.365e-02 |
|  | TE2.14 | 1.374e-02 | 1.068e-02 | 1.365e-02 |
|  | **ours（两文件）** | 9.001e-03 | 1.261e-02 | 1.365e-02 |
| (1,4096,16,128) causal | FA3.0.0 | 1.441e-02 | 1.332e-02 | 1.631e-02 |
|  | TE2.14 | 1.524e-02 | 1.763e-02 | 1.631e-02 |
|  | **ours（两文件）** | 1.510e-02 | 1.340e-02 | 1.631e-02 |
| (1,1024,16,128) full | FA3.0.0 | 1.938e-03 | 1.684e-03 | 1.449e-03 |
|  | TE2.14 | 2.421e-03 | 2.970e-03 | 1.449e-03 |
|  | **ours（两文件）** | 1.938e-03 | 1.684e-03 | 1.449e-03 |

**GQA/MQA**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| (1,1024,32,128) Hkv=4 causal | FA3.0.0 | 1.163e-02 | 2.188e-02 | 3.156e-02 |
|  | TE2.14 | 1.238e-02 | 3.140e-02 | 3.627e-02 |
|  | **ours（两文件）** | 1.201e-02 | 2.125e-02 | 3.156e-02 |
| (1,1024,40,128) Hkv=8 causal | FA3.0.0 | 1.233e-02 | 2.071e-02 | 3.150e-02 |
|  | TE2.14 | 1.303e-02 | 2.491e-02 | 3.411e-02 |
|  | **ours（两文件）** | 1.233e-02 | 1.930e-02 | 3.150e-02 |
| (1,1024,64,128) Hkv=1 causal | FA3.0.0 | 1.793e-02 | 5.151e-02 | 7.196e-02 |
|  | TE2.14 | 1.300e-02 | 6.018e-02 | 8.386e-02 |
|  | **ours（两文件）** | 1.190e-02 | 4.558e-02 | 7.196e-02 |
| (1,1024,64,128) Hkv=4 causal | FA3.0.0 | 1.710e-02 | 2.769e-02 | 4.420e-02 |
|  | TE2.14 | 1.710e-02 | 3.534e-02 | 6.511e-02 |
|  | **ours（两文件）** | 1.351e-02 | 3.091e-02 | 4.420e-02 |

**MLA**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| (1,256,2,512) causal | **ours（两文件）** | 1.230e-02 | 9.875e-03 | 1.686e-02 |
| (1,512,2,512) causal | **ours（两文件）** | 8.795e-03 | 8.073e-03 | 1.339e-02 |
| (1,512,4,512) causal | **ours（两文件）** | 8.753e-03 | 1.082e-02 | 1.740e-02 |
| (1,1024,2,512) causal | **ours（两文件）** | 5.838e-03 | 9.519e-03 | 1.568e-02 |
| (1,1024,8,256) causal | TE2.14 | 1.010e-02 | 1.213e-02 | 1.460e-02 |
|  | **ours（两文件）** | 1.039e-02 | 1.213e-02 | 1.460e-02 |
| (1,1024,16,256) Hkv=4 causal | TE2.14 | 1.190e-02 | 1.800e-02 | 3.275e-02 |
|  | **ours（两文件）** | 1.300e-02 | 1.843e-02 | 3.075e-02 |
| (1,2048,8,256) causal | TE2.14 | 7.978e-03 | 1.300e-02 | 1.681e-02 |
|  | **ours（两文件）** | 7.978e-03 | 1.114e-02 | 1.681e-02 |
| (1,1024,8,256) full | TE2.14 | 1.491e-03 | 1.501e-03 | 2.454e-03 |
|  | **ours（两文件）** | 1.491e-03 | 1.597e-03 | 2.454e-03 |

**varlen**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| varlen B=1 T=512 [512] H=2 D=512 causal | **ours（两文件）** | 8.042e-03 | 1.097e-02 | 1.391e-02 |
| varlen B=3 T=1792 [256,512,1024] H=2 D=512 causal | **ours（两文件）** | 1.267e-02 | 1.217e-02 | 1.796e-02 |
| varlen B=8 T=2904 [2048,512,128,96...] H=16 D=128 causal | FA3.0.0 | 1.214e-02 | 1.519e-02 | 1.863e-02 |
|  | **ours（两文件）** | 1.464e-02 | 1.566e-02 | 1.863e-02 |
| varlen B=4 T=3840 [512,1024,2048,256] H=16 D=128 causal | FA3.0.0 | 1.340e-02 | 1.429e-02 | 1.911e-02 |
|  | **ours（两文件）** | 1.340e-02 | 1.276e-02 | 1.911e-02 |
| varlen B=5 T=3968 [128,256,512,1024...] H=32 D=128 Hkv=8 causal | FA3.0.0 | 1.390e-02 | 2.306e-02 | 3.131e-02 |
|  | **ours（两文件）** | 1.398e-02 | 2.396e-02 | 3.131e-02 |
| varlen B=4 T=4096 [1024,1024,1024,1024] H=16 D=128 causal | FA3.0.0 | 1.145e-02 | 1.237e-02 | 1.772e-02 |
|  | **ours（两文件）** | 1.355e-02 | 1.207e-02 | 1.772e-02 |
| varlen B=1 T=512 [512] H=2 D=512 full | **ours（两文件）** | 1.730e-03 | 1.692e-03 | 1.556e-03 |
| varlen B=3 T=1792 [256,512,1024] H=2 D=512 full | **ours（两文件）** | 3.100e-03 | 3.526e-03 | 2.316e-03 |
| varlen B=8 T=2904 [2048,512,128,96...] H=16 D=128 full | FA3.0.0 | 1.153e-02 | 9.529e-03 | 1.083e-02 |
|  | **ours（两文件）** | 1.153e-02 | 9.529e-03 | 1.083e-02 |
| varlen B=4 T=3840 [512,1024,2048,256] H=16 D=128 full | FA3.0.0 | 5.764e-03 | 3.851e-03 | 3.031e-03 |
|  | **ours（两文件）** | 5.764e-03 | 3.851e-03 | 3.031e-03 |
| varlen B=5 T=3968 [128,256,512,1024...] H=32 D=128 Hkv=8 full | FA3.0.0 | 5.324e-03 | 5.454e-03 | 5.881e-03 |
|  | **ours（两文件）** | 5.324e-03 | 5.454e-03 | 5.881e-03 |
| varlen B=4 T=4096 [1024,1024,1024,1024] H=16 D=128 full | FA3.0.0 | 3.237e-03 | 2.137e-03 | 2.013e-03 |
|  | **ours（两文件）** | 3.237e-03 | 2.392e-03 | 2.013e-03 |

### fp8

**MHA**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| (1,512,16,128) causal | TE2.14 | 4.859e-01 | 4.038e-01 | 5.906e-01 |
|  | **ours（两文件）** | 2.426e-01 | 2.972e-01 | 3.733e-01 |
| (1,1024,32,128) causal | TE2.14 | 4.360e-01 | 4.498e-01 | 8.558e-01 |
|  | **ours（两文件）** | 2.399e-01 | 4.177e-01 | 3.535e-01 |
| (1,4096,16,128) causal | TE2.14 | 3.763e-01 | 3.686e-01 | 6.688e-01 |
|  | **ours（两文件）** | 2.635e-01 | 2.644e-01 | 3.216e-01 |
| (1,8192,16,128) causal | TE2.14 | 2.694e-01 | 4.341e-01 | 5.070e-01 |
|  | **ours（两文件）** | 2.215e-01 | 2.680e-01 | 2.976e-01 |
| (1,1024,16,128) full | TE2.14 | 9.040e-02 | 7.548e-02 | 9.699e-02 |
|  | **ours（两文件）** | 5.521e-02 | 5.310e-02 | 4.025e-02 |

**GQA/MQA**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| (1,1024,32,128) Hkv=4 causal | TE2.14 | 5.246e-01 | 7.664e-01 | 1.343e+00 |
|  | **ours（两文件）** | 2.517e-01 | 5.339e-01 | 7.173e-01 |
| (1,1024,40,128) Hkv=8 causal | TE2.14 | 8.453e-01 | 6.625e-01 | 1.106e+00 |
|  | **ours（两文件）** | 2.869e-01 | 5.367e-01 | 7.032e-01 |
| (1,1024,64,128) Hkv=1 causal | TE2.14 | 4.100e-01 | 2.218e+00 | 2.597e+00 |
|  | **ours（两文件）** | 4.101e-01 | 1.572e+00 | 2.126e+00 |
| (1,1024,64,128) Hkv=4 causal | TE2.14 | 3.983e-01 | 1.011e+00 | 1.844e+00 |
|  | **ours（两文件）** | 2.761e-01 | 8.427e-01 | 1.233e+00 |

**MLA**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| (1,256,2,512) causal | **ours（两文件）** | 2.355e-01 | 2.284e-01 | 3.485e-01 |
| (1,512,2,512) causal | **ours（两文件）** | 2.287e-01 | 2.252e-01 | 3.360e-01 |
| (1,512,4,512) causal | **ours（两文件）** | 2.415e-01 | 2.992e-01 | 4.481e-01 |
| (1,512,8,256) causal | **ours（两文件）** | 2.863e-01 | 3.445e-01 | 3.655e-01 |
| (1,1024,2,512) causal | **ours（两文件）** | 2.228e-01 | 3.311e-01 | 3.611e-01 |
| (1,1024,8,256) causal | **ours（两文件）** | 2.632e-01 | 2.799e-01 | 3.584e-01 |
| (1,1024,16,256) Hkv=4 causal | **ours（两文件）** | 2.475e-01 | 4.412e-01 | 6.153e-01 |
| (1,2048,2,512) causal | **ours（两文件）** | 1.750e-01 | 2.557e-01 | 3.289e-01 |
| (1,2048,8,256) causal | **ours（两文件）** | 2.224e-01 | 2.838e-01 | 3.585e-01 |
| (1,2048,16,256) causal | **ours（两文件）** | 2.203e-01 | 3.006e-01 | 3.532e-01 |
| (1,4096,2,512) causal | **ours（两文件）** | 1.857e-01 | 2.544e-01 | 3.790e-01 |
| (1,4096,8,256) causal | **ours（两文件）** | 2.076e-01 | 3.052e-01 | 3.776e-01 |
| (1,512,2,512) full | **ours（两文件）** | 8.018e-02 | 6.265e-02 | 3.798e-02 |
| (1,512,4,512) full | **ours（两文件）** | 5.007e-02 | 5.066e-02 | 6.929e-02 |
| (1,512,8,256) full | **ours（两文件）** | 4.684e-02 | 4.923e-02 | 4.839e-02 |
| (1,512,16,256) full | **ours（两文件）** | 6.320e-02 | 5.856e-02 | 5.529e-02 |
| (1,1024,2,512) full | **ours（两文件）** | 3.271e-02 | 3.937e-02 | 3.199e-02 |
| (1,1024,8,256) full | **ours（两文件）** | 5.012e-02 | 5.570e-02 | 4.041e-02 |
| (1,1024,16,256) full | **ours（两文件）** | 3.936e-02 | 3.876e-02 | 3.110e-02 |
| (1,2048,2,512) full | **ours（两文件）** | 2.452e-02 | 2.657e-02 | 2.037e-02 |
| (1,2048,8,256) full | **ours（两文件）** | 5.154e-02 | 4.771e-02 | 2.724e-02 |
| (1,2048,16,256) full | **ours（两文件）** | 4.470e-02 | 2.960e-02 | 2.909e-02 |
| (1,2048,32,256) full | **ours（两文件）** | 2.907e-02 | 3.093e-02 | 2.761e-02 |
| (1,4096,2,512) full | **ours（两文件）** | 2.472e-02 | 2.883e-02 | 1.685e-02 |
| (1,4096,8,256) full | **ours（两文件）** | 1.622e-02 | 2.063e-02 | 1.648e-02 |
| (1,4096,16,256) full | **ours（两文件）** | 2.908e-02 | 3.037e-02 | 1.791e-02 |

**varlen**

| shape | impl | dq max_abs | dk max_abs | dv max_abs |
|---|---|---|---|---|
| varlen B=1 T=512 [512] H=2 D=512 causal | **ours（两文件）** | 1.613e-01 | 2.238e-01 | 3.864e-01 |
| varlen B=1 T=512 [512] H=16 D=128 causal | **ours（两文件）** | 2.280e-01 | 3.108e-01 | 3.422e-01 |
| varlen B=3 T=1792 [256,512,1024] H=2 D=512 causal | **ours（两文件）** | 3.404e-01 | 3.436e-01 | 3.508e-01 |
| varlen B=8 T=2904 [2048,512,128,96...] H=8 D=256 causal | **ours（两文件）** | 3.416e-01 | 4.407e-01 | 4.290e-01 |
| varlen B=8 T=2904 [2048,512,128,96...] H=16 D=128 causal | **ours（两文件）** | 3.136e-01 | 3.584e-01 | 4.030e-01 |
| varlen B=4 T=3840 [512,1024,2048,256] H=8 D=256 causal | **ours（两文件）** | 3.619e-01 | 3.385e-01 | 4.002e-01 |
| varlen B=4 T=3840 [512,1024,2048,256] H=16 D=128 causal | **ours（两文件）** | 2.935e-01 | 2.938e-01 | 4.179e-01 |
| varlen B=5 T=3968 [128,256,512,1024...] H=32 D=128 Hkv=8 causal | **ours（两文件）** | 3.094e-01 | 5.567e-01 | 6.203e-01 |
| varlen B=4 T=4096 [1024,1024,1024,1024] H=8 D=256 causal | **ours（两文件）** | 3.382e-01 | 3.485e-01 | 3.868e-01 |
| varlen B=4 T=4096 [1024,1024,1024,1024] H=16 D=128 causal | **ours（两文件）** | 2.651e-01 | 3.026e-01 | 3.920e-01 |
| varlen B=1 T=512 [512] H=2 D=512 full | **ours（两文件）** | 5.260e-02 | 5.222e-02 | 4.218e-02 |
| varlen B=3 T=1792 [256,512,1024] H=2 D=512 full | **ours（两文件）** | 7.994e-02 | 9.629e-02 | 4.148e-02 |
| varlen B=8 T=2904 [2048,512,128,96...] H=8 D=256 full | **ours（两文件）** | 1.874e-01 | 2.067e-01 | 2.334e-01 |
| varlen B=8 T=2904 [2048,512,128,96...] H=16 D=128 full | **ours（两文件）** | 2.412e-01 | 2.057e-01 | 2.512e-01 |
| varlen B=4 T=3840 [512,1024,2048,256] H=8 D=256 full | **ours（两文件）** | 7.204e-02 | 6.616e-02 | 8.387e-02 |
| varlen B=4 T=3840 [512,1024,2048,256] H=16 D=128 full | **ours（两文件）** | 1.011e-01 | 9.656e-02 | 7.032e-02 |
| varlen B=4 T=3840 [512,1024,2048,256] H=16 D=256 full | **ours（两文件）** | 8.477e-02 | 7.903e-02 | 6.954e-02 |
| varlen B=5 T=3968 [128,256,512,1024...] H=8 D=256 full | **ours（两文件）** | 1.135e-01 | 1.285e-01 | 8.087e-02 |
| varlen B=5 T=3968 [128,256,512,1024...] H=32 D=128 Hkv=8 full | **ours（两文件）** | 1.428e-01 | 1.597e-01 | 1.082e-01 |
| varlen B=4 T=4096 [1024,1024,1024,1024] H=16 D=128 full | **ours（两文件）** | 8.881e-02 | 7.043e-02 | 5.721e-02 |
| varlen B=4 T=4096 [1024,1024,1024,1024] H=16 D=256 full | **ours（两文件）** | 7.234e-02 | 6.459e-02 | 4.302e-02 |
<!-- END:auto-doc-table -->

## 40. P3-3f：单/两文件实现一致性自动报告（第 115 轮）—— 工具链 + 验证，正结果

> 本轮为 **harness + 验证增量**，device 一行未改。目标是把此前每一轮都要手工做的
> 「单文件 vs 两文件逐元素一致性」核对（§33/§34/§39 均手工誊抄过）**固化进 harness**，
> 并顺带修掉 `fa_bwd_run.py` 的 `--no-run` 失效 bug。

### 40.1 改动（纯 harness）

* `harness/fa_bwd_compare.py` 新增 **`--consistency`**：对每个 case 读取两个 impl 的
  `{dq,dk,dv}`（默认 `--ca ours --cb ours_sf`，可换成任意前缀），逐元素算 `max|A-B|`，
  按 dtype 分组打印并汇总**全 case 最坏值**；`--ctol` 给定时超过即退出码 1（CI 回归门）。
  该模式不需要 `ref_*`，也不需要 GPU（直接读 dump）。
* `harness/fa_bwd_run.py` 新增 **`--consistency` / `--consistency-tol` / `--consistency-out`**，
  跑完/复用 npy 后一键调用上面的一致性报告（产物默认
  `src/fa_bwd_consistency_p33f.out.txt`）。
* **修复 `--no-run`**：文档一直写「跳过 kernel、只重新汇总」，但主循环从未检查它，
  仍会编译并运行全部 case（一次全量扫 = 146 次运行）。现在 `--no-run` 真正跳过执行、
  复用已有 npy，且**不改写**上一轮全量扫的运行日志/清单。

### 40.2 实测（全部 73 个 case；`fa_bwd_run.py --consistency --no-run`，无 GPU）

73/73 case 的 `ours_*`（两文件）与 `ours_sf_*`（单文件）**都存在**，0 case 缺一侧。
两种文件形态的 device 代码逐字同源，唯一的期望差异是**跨 CTA `atomicAdd` 的 fp32 求和次序**。

| dtype | max\|ours−ours_sf\|（dq） | 最坏 dk | 最坏 dv | 最坏出现 | 判定 |
|---|---|---|---|---|---|
| fp16 | 2.441e-4 | 1.953e-3 | 3.906e-3 | GQA kv4 的 dv | dtype ulp（amax ~2） |
| bf16 | 1.953e-3 | 7.812e-3 | 7.812e-3 | GQA/MQA 的 dk/dv | dtype ulp（amax ~2） |
| fp8  | 1.192e-7 | 6.676e-6 | 9.537e-6 | MQA kv1 的 dk/dv | fp32 累加 ulp |

* `dq` 在两文件/单文件间**多数逐位相同**（MHA 无 split 时恒 0），只在 MLA / split-K / varlen
  的跨 CTA dQ 归约上差 1–2 ulp——与 §33/§34 的历史结论完全一致。
* 最坏值都出现在 **GQA/MQA**（KV 头共享、`amax` 更大）的 `dk/dv`，量级 = 1–2 个 dtype ulp。
  **无任何「实现分歧」型差异**（若同源 device 代码被改坏，会看到 O(输出幅度) 的差）。
* 复现：`python harness/fa_bwd_run.py --consistency --no-run`（或直接
  `python harness/fa_bwd_compare.py --consistency`）。原始输出
  `src/fa_bwd_consistency_p33f.out.txt`。

### 40.3 纯反向基线刷新（同机，CUPTI；`harness/fa_vs_te_bwd_only.py fp16`）

| shape | FA2.7.4 | FA3 | TE2.14 | FA3/FA2 |
|---|---|---|---|---|
| (1,1024,32,128) kv4 | 0.1577/218 | 0.0825/417 | 0.1120/307 | 1.91× |
| (1,4096,16,128) MHA | 0.7258/379 | **0.3237/849** | 0.4388/626 | 2.24× |
| varlen [1024]×4 causal | 0.2511/137 | 0.1470/234 | NA | 1.71× |
| varlen [1024]×4 full | 0.3453/100 | 0.2003/172 | NA | 1.72× |

与 §37/§38 的纯反向口径一致（FA3 > TE > FA2）；fp8/MLA（D=512）FA3 反向不支持。原始输出
`src/fa_bwd_p33f_fa_baseline_fp16.out.txt`。

### 40.4 本轮 ours 性能 / ncu（Hopper 快路：`-DFA_WGMMA -DFA_TMA`，同机）

| case | ours 端到端（event） | 对 FA3（时间） | 对 TE（时间） | 数值 vs ref（max_abs dq/dk/dv） |
|---|---|---|---|---|
| fp16 (1,4096,16,128) MHA | **1.2582 ms / 109.2 TF** | 3.89× | 2.87× | 1.883/1.734/1.966e-3 |
| fp8 (1,4096,16,128) MHA | **1.9466 ms / 70.6 TF** | —（FA3 无 fp8 bwd） | ≈6.4×（TE fp8 纯反向 0.303 ms，§38） | 2.635/2.644/3.216e-1 |

ncu（fp8 主 kernel `fa_bwd_fp8_mma_kvtma_kernel`，S=4096，`-c 1`）：Duration **1.60 ms**、
**L2 Cache Throughput 78.25%**、L1/TEX 70.48%、DRAM 4.23%、Compute 46.90%、occ 18.34%
（168 regs / 3 CTA/SM，Waves 20.69）；`lts__t_sectors_op_red=114.5 M`（dK/dV 跨 CTA 归约主导）、
stall `wait 1.59 + short_scoreboard 1.28 + long 0.58` ⇒ **bound = L2 dK/dV 原子归约 + mma 依赖延迟**，
与 §45（O42）/§44（O41）一致：red 是头号成本、已由 O19/O42 判决「换实现只会更慢」。
原始输出 `src/fp8/fa_bwd_fp8_p33f_perf_s4096.out.txt`、`src/fp8/fa_bwd_fp8_p33f_ncu_s4096.out.txt`、
`src/fp8/fa_bwd_fp8_p33f_ncu_stall_s4096.out.txt`；fp16 见
`src/fp16/fa_bwd_fp16_p33f_perf_s4096.out.txt`。

**结论**：三 dtype × 单/两文件共 73 case 的两形态一致性被自动化并全量通过（差异 ≤ 2 dtype ulp、
均来自 `atomicAdd` 次序）；刷新后的纯反向基线与 ours 数字确认当前边界未变（fp8 main 仍是
L2 red + mma 依赖，属文档已判决的硬件资源墙）。

## 41. P3-3g：单/两文件一致性 gate 接进端到端回归（第 116 轮）—— 工具链 + 验证，正结果

> 本轮为 **harness + 验证增量**，device 一行未改。落实第 115 轮（§40）「下一步候选 ①」：
> §40 把「单/两文件一致性」做成了 `--consistency` 工具，但仍要**人工记得**去调它；本轮把它
> **默认接进 `fa_bwd_run.py` 的全量扫**，并用**按 dtype 的容差**自动 gate——任何一次单文件
> 同步漏改都会被端到端回归立刻抓住。

### 41.1 改动（纯 harness）

* `harness/fa_bwd_compare.py`：`--consistency` 的容差从「单个全局标量」升级为 **按 dtype**。
  `--ctol auto` 使用 `CTOL_AUTO = {fp16: 1.6e-2, bf16: 3.2e-2, fp8: 1e-4}`（约实测 worst 的
  2–4× 余量），报告新增逐 dtype 的 `gate[dtype] worst=… tol=… -> OK/FAIL`，任一 dtype 超门即
  退出码 1；`--ctol <float>` 仍作全局标量（向后兼容），缺省只报不判。
* `harness/fa_bwd_run.py`：**默认在全量扫结束时自动调用**上述检查（两边文件形态都跑时才开；
  `--impls twofile` 只跑一边、或 `--no-consistency` 时跳过；`--consistency` 可显式强制，配
  `--no-run` 可用已有 npy 复核）。`--consistency-tol` 默认由 `None` 改为 `"auto"`，
  `--consistency-out` 默认产物改为 `src/fa_bwd_consistency_p33g.out.txt`。

**为什么必须按 dtype**：两形态的 device 代码逐字同源，唯一差异是跨 CTA `atomicAdd` 的 fp32
求和次序，其数值 = **1–2 个 dtype ulp**。而 fp16/bf16/fp8 的 ulp 相差 **3 个数量级**
（fp8 worst 9.5e-6 vs bf16 worst 7.8e-3）。任何单个全局标量要么放过 bf16 的错误、要么误杀
fp8 的正常原子噪声——这正是 §40 只做「报告」而没有直接上门的顾虑。

### 41.2 实测：全量 73 case 自动 gate（`fa_bwd_run.py --no-run`，无 GPU）

| dtype | worst `max|ours-ours_sf|` | 出现位置 | tol（auto） | 判定 |
|---|---|---|---|---|
| fp16 | 3.906e-03 | d512 GQA/MLA 的 dk/dv | 1.6e-2 | **OK** |
| bf16 | 7.812e-03 | GQA kv4/kv8 的 dk/dv | 3.2e-2 | **OK** |
| fp8  | 9.537e-06 | MQA kv1 的 dv | 1e-4 | **OK** |

73/73 case 两侧都在、0 case 缺一侧；全量 gate 退出码 **0**。与 §40 的实测逐位一致。原始输出
`src/fa_bwd_consistency_p33g.out.txt`。

### 41.3 负向验证（gate 确实会拦）

为防止「绿灯只是没接上」，做了两个负向测试（`src/fa_bwd_p33g_gate_negative.out.txt`）：

* `fa_bwd_compare.py --consistency --ctol 1e-3`（把全局容差收到 1e-3）：fp16 `3.906e-03`、
  bf16 `7.812e-03` 均判 **FAIL**，fp8 `9.537e-06` **OK**，退出码 **1**；
* `fa_bwd_run.py --no-run --consistency-tol 1e-9`：端到端 gate 退出码 **1**。

### 41.4 现场小样本重跑（证明自动 gate 挂在真实编译/运行路径上）

在 `kernel_lab` 容器里以 `fa_bwd_run.py --case <fp16 与 fp8 各一> --impls both` 真编译真运行
（4 次运行 / 2 次编译），随后默认 auto gate：fp16 worst `4.883e-04`、fp8 worst `4.768e-07`，
退出码 **0**；对拍值与历史一致（fp16 `1.671/1.771/1.899e-3`、fp8 `2.426/2.975/3.735e-1`）。
原始输出 `src/fa_bwd_p33g_live_sweep.out.txt`、`src/fa_bwd_p33g_gate_small.out.txt`。

### 41.5 纯反向基线刷新（同机，CUPTI；`harness/fa_vs_te_bwd_only.py fp16`）

| shape | FA2.7.4 | FA3 | TE2.14 | FA3/FA2 |
|---|---|---|---|---|
| (1,1024,32,128) kv4 | 0.1582/217 | 0.0822/418 | 0.1121/306 | 1.92× |
| (1,4096,16,128) MHA | 0.7277/378 | **0.3243/848** | 0.4399/625 | 2.24× |
| varlen [1024]×4 causal | 0.2508/137 | 0.1475/233 | NA | 1.70× |
| varlen [1024]×4 full | 0.3394/101 | 0.1980/174 | NA | 1.71× |

与 §37/§38/§40 的纯反向口径一致（FA3 > TE > FA2）；fp8/MLA（D=512）FA3 反向不支持。本轮
device 未改，故 ours 性能沿用 §40（fp16 S4096 `1.2582ms/109.2TF`、fp8 `1.9466ms/70.6TF`，
fp8 main 仍 **L2 78% red + mma 依赖**）。原始输出 `src/fa_bwd_p33g_fa_baseline_fp16.out.txt`。

**结论**：`--consistency --ctol auto` 已内建为 `fa_bwd_run.py` 全量扫的默认出口检查，用按
dtype 的 ulp 容差在 73/73 case 上全绿、且负向测试确认会拦；单/两文件「device 逐字同源」从
此由端到端回归持续背书，无需再逐轮人工誊抄。当前性能边界未变（属文档已判决的硬件资源墙）。

## 42. P3-4d：CI 单一入口 + Hopper 快路入标准 harness（第 117 轮）—— 工具链 + 验证，正结果

> 本轮落实第 116 轮（§41）「下一步候选 ②」：把「跑 ours + 汇总 + 单/两文件一致性 gate」再
> **收口成一条 CI 命令**，并在其上叠加 **docs/04 内嵌表新鲜度校验**；同时把此前只能在命令行手拼
> `NVCC_FLAGS` 的 **Hopper 快路（`-DFA_WGMMA -DFA_TMA -lcuda`）** 接进标准 harness（`--hopper`）。
> device 一行未改；但为保证「构建配置差异 ≠ 实现分叉」，`--hopper` 用**独立前缀**
> `ours_hp/ours_sf_hp`，绝不覆盖默认 mma 口径的 `ours/ours_sf`。

### 42.1 改动（纯 harness + 一个入口脚本）

* `harness/fa_bwd_run.py`：
  * **`--ci`**：跑完后（或 `--no-run` 时）自动调用 `fa_bwd_compare.py --check docs/04`，
    内嵌表陈旧即退出码 1；并把一致性 gate 结果 + doc-check 结果汇总到 `src/fa_bwd_ci.out.txt`。
  * **`--hopper`**：定长/变长构建都切到 `-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA
    -DFA_TMA -lcuda`（此前只有 varlen 入口在 sm90a 下编译），且输出前缀切到 `ours_hp/ours_sf_hp`
    （wgmma/TMA 与 mma 的数值差可达 O(1e-1)，见 `docs/03` O9c-2 A/B；混用会误触一致性 gate）。
  * **`--perf-baseline <dtype>`**：额外在容器内跑用户指定的纯反向基线 `fa_vs_te_bwd_only.py`
    （FA2/FA3/TE 三列，forward 在计时区外），原始输出落盘。
* `scripts/ci.sh`：`exec python harness/fa_bwd_run.py --ci "$@"`，一行命令的固定入口。

### 42.2 实测：CI 全绿（`scripts/ci.sh`，无 GPU 路径）

`python3 harness/fa_bwd_run.py --no-run --ci`（复用 73 case 已有 npy）退出码 **0**：

| dtype | worst `max\|ours-ours_sf\|` | tol（auto） | gate | docs/04 表 |
|---|---|---|---|---|
| fp16 | 3.906e-03 | 1.6e-2 | **OK** | `--check` OK（194 行，rtol=5e-3） |
| bf16 | 7.812e-03 | 3.2e-2 | **OK** | 同上 |
| fp8  | 9.537e-06 | 1e-4 | **OK** | 同上 |

**负向测试**（`src/fa_bwd_p117_ci_negative.out.txt`）：① `--consistency-tol 1e-9` → 三 dtype
全 FAIL、**rc=1**；② 把内嵌表某格从 `1.883e-03` 改成 `9.999e-03` → `--check` 报
`STALE（第 11 行数值超出 rtol=0.005）`、**rc=1**。⇒ 「实现分叉」与「文档陈旧」两类回归都被同一
条命令拦住。

### 42.3 现场 device 验证：`--hopper` 真编译真跑（fp8 S4096）

`python3 harness/fa_bwd_run.py --case b1_s4096_h16_d128_causal_fp8 --impls both --hopper`
（4 次运行 / 2 次编译，容器内）退出码 0：

* 数值 vs fp32 ref：`dq/dk/dv = 2.635e-01 / 2.644e-01 / 3.216e-01`（与 §32–§41 历史**逐位/同量级一致**）；
* 单/两文件（都是 Hopper 构建）一致性 worst `7.153e-07`（fp8 ulp 级，fp8 tol 1e-4）→ **OK**；
* 计时：`quant 0.0693 | preprocess 0.2219 | main 1.5945 | total 1.9359 ms / 70.99 TF`
  （Hopper 快路，`ksplit=8`、`grid=512×16`、`O7 use_regdq=1`、`O41 kv-tma=on`）；原始输出
  `src/fp8/fa_bwd_fp8_p117_hopper_run.out.txt`（对比 `src/fa_bwd_p117_hopper_compare.out.txt`）。

### 42.4 性能对标（纯反向口径）

同机纯反向基线（`src/fa_bwd_perf_baseline_fp16.out.txt`、`src/fa_bwd_p117_fp8_te_baseline.out.txt`）：

| shape | FA2.7.4 | FA3 | TE2.14 | TE FP8 | ours（Hopper） | ours/TE |
|---|---|---|---|---|---|---|
| fp16 MHA (1,4096,16,128) | 0.7341ms/374TF | **0.3245/847** | 0.4403/624 | — | 1.2582/109.2（§40） | 2.87× |
| fp8 MHA (1,4096,16,128) | — | — | — | **0.3025/908.7** | **1.9359/70.99** | **6.40×** |

⇒ fp8 ours/TE 6.40×（§41 记 ~6.4×，一致）；fp16 纯反向 FA3/TE/FA2 三列与 §37/§38/§40 一致。

### 42.5 ncu（fp8 main，Hopper，S=4096；`src/fp8/fa_bwd_fp8_p117_ncu_main_s4096.out.txt`）

| 指标 | 值 |
|---|---|
| `lts__throughput` / `lts__t_sectors_op_red` | **78.63%** / **114,524,160** 扇区 |
| `l1tex__throughput` / shared `op_ld` bank conflict | 68.23% / 15,033,972 |
| DRAM / Compute / tensor-pipe | 4.26% / 47.19% / 12.30% |
| stall（per issue active） | **wait 1.59 + short_scoreboard 1.29** + long 0.60 + barrier 0.43 |
| occ / regs / waves | 18.25% / **168** / 20.69 |

**bound 结论**：与 §41 完全一致——头号是 **L2 的 dK/dV 跨 CTA `red`（78.6%，114.5M 扇区）**，
其次是 **smem→mma 依赖（short_scoreboard）+ mma 等待（wait）**；DRAM 仅 4.3%、tensor pipe 12.3%。
非带宽/算力 bound。

### 42.6 结论 / 下一步

`scripts/ci.sh`（=`fa_bwd_run.py --ci`）已是「跑 ours + 数值 + 一致性 gate + 文档新鲜度」的单一
入口，正/负向测试均确认会拦；`--hopper` 让标准 harness 也能产 Hopper 快路口径而不污染默认基线。
**设备侧当前无未证伪的可行杠杆**：跨-tile `P/dS` 双缓冲流水（唯一能直接打 `wait+short_scoreboard`
的方向）经精确 smem 预算核算在本卡 3 CTA/SM 下**不可行**（详见 ROADMAP「阻塞」），与 L2 `red`
（`red` 只能靠放大 BM / 提 occupancy，两者皆撞寄存器/smem 硬墙）共同锁死了当前工作点。

## 43. P3-4e：fp8 确定性 dK/dV 归约（`--det`，第 118 轮）—— 正结果，opt-in

补齐 `docs/00` catalog §4.2 第 5 条（FP8 非确定性）与 backlog「deterministic 模式代价量化」。
fp16/bf16 早有 `--det`（O7b），fp8 本轮补齐：dK/dV 的跨 CTA `atomicAdd` 换成「按 (Q 头, Q 块)
分片的 partial 覆盖写 + `dkv_reduce_kernel` 固定次序求和」。单/两文件 device 逐字同源。

**数值**（`runs[1-2] bitwise-diff dk/dv = 0.00e+00`，全部 case 逐位可复现；`DET-vs-atomic`
~e-7–e-6 为 fp32 归约次序的末位差；`ours vs ref` 与历史逐位一致）：

| case（fp8） | atomic (ksplit=1) | DET | 比值 | `runs[1-2]` |
|---|---|---|---|---|
| S512 MHA causal（两/单文件） | 0.1657 / 0.1676 ms | 0.1723 / 0.1744 ms | 0.961× | 0 / 0 |
| S4096 MHA causal | 2.3649 ms | 2.8588 ms | 0.827× | 0 / 0 |
| S1024 GQA kv4 causal | 0.3882 ms | 0.4611 ms | 0.842× | 0 / 0 |
| S1024 MHA full | 0.4182 ms | 0.4671 ms | 0.895× | 0 / 0 |

**ncu**：DET 主 kernel（S4096 ksplit=1）DRAM 4.3%→**33.3%**、L2 78.6%→**36.2%**、
`red` 114.5M→**1.57M** 扇区、`write` 102.4M 扇区（partial 2.20 GB）；`dkv_reduce_kernel`
**731.6µs / DRAM 91.5% / 3.07 TB/s**（纯带宽 bound）。**代价**：ksplit 固定 1 + 一次纯带宽
归约；相对调优默认档（S4096 ksplit=4）约 **1.5–1.8×**。详见 `docs/03` §56、`docs/00` §4.2。

---

## 44. P3-4f：`--det` 扩展到 split-K（dQ 也走 partial，第 119 轮）—— 正结果，opt-in `--detk>1`

落实 §43 的「下一步候选 ①」。dK/dV 的 partial **天然无需 part 维**（一个 `(mblk,jg)` 只属于
一个 part，各 part 写不相交的 `jg`），故只需给 **dQ** 加 part 分片 partial +
`dq_reduce_kernel<HD>` 固定次序归约，即可让 DET 重新吃 split-K 并行度。新增 `--detk=N`。

**数值**：所有 ksplit（含 dQ）`runs[1-2] bitwise dq/dk/dv = 0.00e+00`；`DET-vs-atomic`
~e-7–e-6；默认路径（无 `--det`）数值逐位不变。

**性能**（同 session ksplit sweep，两文件；atomic vs DET）：

| case | DET k=1 | DET k=2 | DET k=4 | DET k=8 | 最优 / k=1 |
|---|---|---|---|---|---|
| S512 MHA causal | 0.1461 ms | 0.1116 | **0.1115** | 0.1215 | **1.31×** |
| S4096 MHA causal | 2.8380 ms | 2.6256 | **2.5514** | 2.6475 | **1.11×** |

atomic 随 ksplit 单调变快；DET 在 **k=4 触底**（固定 partial 写/读成本与 ksplit 无关）。

**ncu**：DET 主 kernel（S4096 k=4）Duration 1.72ms / DRAM **42.3%** / L1TEX 69.8% / occ 18.1%；
`dkv_reduce_kernel` **730.7µs / DRAM 91.56% / 3.07 TB/s**、`dq_reduce_kernel` 55.1µs /
DRAM 87.6%。**bound = reduce 的纯 DRAM 带宽 + 主 kernel 多写的 partial**（与 §43 同构）。
详见 `docs/03` §57。

---

## 45. P3-4m：`--det` 接进 MLA 的 K/V `cp.async` 回填流水（第 126 轮）—— 正结果，opt-in `--det`

`--det` 的 partial + 二次归约已覆盖定长/split-K/TMA/MLA/varlen（§43–44、`docs/03` §56–63），
但 host 的 `launch_bwd_main_det` 一直写死 `KVPIPE=false` ⇒ MLA（HD=512）的 DET 主 kernel 走
非 kvpipe 旧路，而默认 MLA 主 kernel 早在 O51 就用 K/V `cp.async` 回填流水。本轮加
`bool KVPIPE` 模板参（`kSmem` 选 `smem_bytes_kvpipe`，MLA 229888B ≤ 232448），把 DET 接上
同一条流水。**device 一行未改**（`fp8_mma_body` 本就支持 `KVPIPE && DET`），仅 host 接线。

**数值**：kvpipe DET 与非 kvpipe DET **逐位相同**（`runs[1-2]=0`、`kvpipe-vs-非kvpipe=0`）；
默认路 `ours vs ref` 与历史逐位一致。

**性能**（同 session，两文件；单文件 b3 复核）：

| case | ksplit | DET 非 kvpipe | DET kvpipe | 加速 |
|---|---|---|---|---|
| MLA 定长 S256H2 | 4 | 0.0652 ms | 0.0640 | 1.018× |
| MLA 定长 S512H4 | 4 | 0.1578 ms | 0.1477 | 1.068× |
| MLA 定长 S1024H2 | 4 | 0.2594 ms | 0.2394 | 1.084× |
| MLA varlen b1_t512 | 8 | 0.0928 ms | 0.0887 | 1.046× |
| MLA varlen b3_t1792 | 8 | 0.3550 ms | 0.3258 | 1.090× |
| varlen b3_t1792（单文件） | 8 | 0.3556 ms | 0.3248 | 1.095× |

**ncu**（b3_t1792 k=8，主 kernel）：Duration 243.0→**220.2µs**、`long_scoreboard` 3.93→**3.25**、
DRAM 29.8→33.0%、smem 207.87→229.89KB、occ 12.49%（1 CTA/SM）。机制与 O51 一致（K/V 全局读
延迟藏进计算）。**确定性模式下不再额外付出「非 kvpipe 主 kernel」这一层代价**。详见
`docs/03` §64、`docs/00` §4.2。

## 46. O63（第 143 轮）：fp16/bf16 LSE 的「tile 内两趟 softmax」—— 正结果，默认

把第 136 轮 F5（fp8，`docs/03` §71、`§5.51`）的 LSE 两趟 softmax epilogue 逐字回移/参数化到
**fp16 与 bf16** 的 4 个 LSE kernel（`lse_mma_kernel` / `_bal`（含 `FULL`）/ `_bal_wgmma` /
`_bal_tma`），并把 `_bal_tma` 的 `int kuse[2]`（→ local memory）改两标量。数学等价、只换
fp32 求和次序。单/两文件同源（bf16 按既定做法用 `using bf16=...` marker 同步）。

**数值**：与改前**逐位相同**（fp16 S4096 1.883/1.734/1.966e-3；bf16 1.510/1.340/1.631e-2）；
全量 `--ci --dtype fp16 bf16` gate fp16 **3.906e-3** / bf16 **1.562e-2** 均 OK、一致性 OK、
`--check docs/04` OK（194 行）。仅 full 的个别 dv 第 5 位移动（~1e-4），已 `--apply` 同步本表。

**性能**（同 session 同 binary，event，S4096 causal）：

| dtype / 构建 | 阶段 | OLD | NEW | 加速 |
|---|---|---|---|---|
| fp16 `sm_90` | LSE / preprocess / total | 0.3112 / 0.3246 / 1.9173 ms | **0.2159 / 0.2293 / 1.8233 ms** | 1.44× / 1.42× / 1.052× |
| bf16 `sm_90` | LSE / preprocess / total | 0.3098 / 0.3214 / 1.9213 ms | **0.2158 / 0.2279 / 1.8333 ms** | 1.44× / 1.41× / 1.048× |
| fp16 `-DFA_WGMMA`（varlen B4T3840） | total | 0.7792 ms | **0.7424 ms** | 1.050× |
| fp16 `-DFA_WGMMA -DFA_TMA`（Hopper） | preprocess / total | 0.2268 / 1.2571 ms | **0.1339 / 1.1657 ms** | 1.69× / 1.078× |

**ncu**（fp16 `lse_mma_kernel_bal<128,1>`，S4096）：Duration **310.27→221.18µs（1.40×）**、
`smsp__inst_executed` **163.8M→119.1M（−27.3%）**、local 扇区 **98K/16K→0**；wgmma 版
277.79→**195.17µs（1.42×）**。SASS：`MUFU` 172→112、`BSSY/BSYNC` 109→42、`FSETP` 260→140。

**结论**：三 dtype 的 LSE 现已统一走两趟 softmax；LSE 不是墙，默认路径的墙仍是 main。
详见 `docs/01` §19、`docs/01b` §6av、`docs/08` §5.57；原始输出 `src/fp16/fa_bwd_fp16_p143_*`、
`src/bf16/fa_bwd_bf16_p143_*`、`src/fa_bwd_p143_ci_fp16_bf16.out.txt`。

## 43. F8（第一百六十一轮，正结果，默认）：fp16/bf16 定长默认切 Hopper（wgmma+TMA）

§42 的 `--hopper` 快路此前只是**可选**（独立前缀 `ours_hp`），默认 `ours` 口径对 fp8 走 Hopper
（F1）、对 fp16/bf16 仍锁 `-arch=sm_90` 的 mma.sync。本轮把 F1 的默认化**泛化到 fp16/bf16**：
`harness/fa_bwd_run.py` 的 `HOPPER_DEFAULT_DTYPES = {"fp8","fp16","bf16"}`。**device 一行未改**。

| dtype | S4096 causal total（mma→Hopper） | 加速 | main | max_abs vs fp32 ref |
|---|---|---|---|---|
| fp16 | 1.8245→**1.1726 ms** | **1.556×** | 1.503→0.964 ms | 1.883/1.734/1.966e-3（**逐值不变**） |
| bf16 | 1.8325→**1.1643 ms** | **1.574×** | 1.495→0.958 ms | 1.510/1.340/1.631e-2（**逐值不变**） |

纯反向对标（`harness/fa_vs_te_bwd_only.py`，S4096 MHA fp16）：FA3 `0.3238ms/849TF`、
TE `0.4441ms/619TF`；ours total 5.6×→**3.6×** FA3。MLA（D=512）host 自动退回 mma（不变）、
full 走 mma LSE 亦正确、varlen 早已 Hopper。CI 73 case 全绿（fp16 worst 1.953e-3 / bf16
7.812e-3 / fp8 1.049e-5），`--check docs/04` OK（内嵌数值表不受默认切换影响，逐值/rtol 内）。
`--mma` 可一次退回三 dtype 的旧 mma 口径。详见 `docs/08` §5.75；原始输出
`src/fa_bwd_p161_hopper_default_ab.out.txt`、`src/fa_bwd_p161_baseline_{fp16,bf16}.out.txt`。

## 44. O69（第一百六十三轮，正结果，默认）：fp16/bf16 非 causal D=128 的 LSE 接上均衡 `cp.async` 版

O68/F9（§9 / 第 162 轮）修了 fp8「causal LSE 已均衡化、但 full D=128 仍走 O1 `lse_mma_kernel`」
这条漏改分支；本轮**核查发现 fp16/bf16 同一条分歧也漏改**（O54 的 `lse_mma_kernel_bal<FULL=true>`
此前只服务 D=512 MLA 的 full），逐个 dtype 补齐。**纯 host 改动（device 一行未改）**：定长与
varlen 的 D=128 full LSE 从 O8 `lse_mma_kernel<128>` 切到 `lse_mma_kernel_bal<128,1,FULL=true>`
（`ksplit=1`，对齐 fp8 F9）；新增 `--lsefull=0/1` 同 binary A/B。

| case（dtype） | 阶段 | lsefull=0（O8） | lsefull=1（bal FULL） | 加速 | max_abs vs fp32 ref |
|---|---|---|---|---|---|
| fp16 定长 full S1024 H16 D128 | preprocess | 0.1588 ms | **0.0463 ms** | **3.43×** | dq/dk/dv 同（3.27/2.52/1.23e-4） |
| fp16 定长 full S1024 H16 D128 | total | 0.3048 ms（28.2 TF） | **0.1687 ms（50.9 TF）** | **1.81×** | — |
| fp16 varlen full b4_t4096 H16 D128 | total | 0.8632 ms（39.8 TF） | **0.5942 ms（57.8 TF）** | **1.46×** | 4.09/4.95/1.23e-4（同） |
| bf16 定长 full S1024 H16 D128 | total | 0.3060 ms（28.1 TF） | **0.1710 ms（50.2 TF）** | **1.79×** | 1.94/1.68/1.45e-3（同） |
| bf16 varlen full b4_t4096 H16 D128 | total | 0.8706 ms（39.5 TF） | **0.5993 ms（57.3 TF）** | **1.45×** | 3.24/2.39/2.01e-3（同） |

收益全在 preprocess（main 不变）。纯反向对标（同 session CUPTI，`harness/fa_bwd_bench.py bench`）：
定长 full S1024 H16 D128 **fp16** FA3 `0.0512ms/335.6TF`、FA2 `0.0828ms/207.6TF`、
TE `0.0577ms/297.7TF` ⇒ ours total 为 FA3 时间 **3.29×**、TE **2.92×**（峰值 989 的 **5.1%**）；
**bf16** FA3 `0.0506ms/339.6TF`、FA2 `0.0828/207.5`、TE `0.0576/298.1` ⇒ 3.38×/2.94×；
varlen full `[1024]×4` fp16 FA3 `0.1996ms/172.1TF` ⇒ ours **3.0×**。

ncu（LSE，S1024 H16 full）：fp16 O8 **175.8µs / L1TEX 11.8% / Compute 25.6% /
No Eligible 73.7% / `long_scoreboard` 4.4cy / regs 80 / smem 34.82KB** → bal FULL **39.2µs
（4.49×）/ L1TEX 37.6% / Compute 39.6% / `short_scoreboard` 1.5cy / regs 64 / smem 52.22KB**；
bf16 逐项一致（175.2→38.8µs，4.51×）；varlen b4_t4096 fp16 LSE **375.5→108.5µs（3.46×）**。
⇒ O8 的墙是**串行全局载入延迟**，均衡 FULL 用 `cp.async` 双缓冲打掉它，新墙 = issue + smem
依赖 + 网格不足一个波（`Waves 0.48`）。**这是 preprocess 内一条被漏改的 LSE 分支，不是 main 的
L2 `red` 墙**（后者仍受本卡寄存器/smem 硬墙锁定，见「阻塞」）。CI：fp16 一致性 gate worst
**3.906e-3**、bf16 **7.812e-3**（均 OK）、`--check docs/04` OK（内嵌数值表 194 行，随本轮同步
更新 fp16 一处 full dv `1.234e-4→1.213e-4`）。详见 `docs/08` §5.77；原始输出
`src/fp16/fa_bwd_fp16_o69_*`、`src/bf16/fa_bwd_bf16_o69_*`、`src/fa_bwd_p163_full_baseline.out.txt`。

## 45. O71（第一百六十五轮，正结果，默认）：fp16/bf16 非 causal D=128 的 LSE 再上 **4D-TMA**

O69（§44）把 fp16/bf16 定长 full D=128 的 LSE 接到「均衡 + `cp.async`」，但 causal LSE 早就是
4D-TMA 版（O30/O31）。fp8 的对应改造（O70）已于第 164 轮完成；本轮把 fp16/bf16 也切到同一 TMA
搬运（`lse_mma_kernel_bal_tma` 加 `bool FULL`、host full 分支优先 TMA、`lse_tma=(D==128)?1:0`），
**三 dtype full D=128 LSE 至此统一到 TMA**。`--lsetma=0` 退回 O69 cp.async、`--lsefull=0` 退回 O8。
device 与 fp16 逐字同源（bf16 为 `__half`→bf16），`FULL=false` 与 O30/O31/O38 逐位相同。

| case（dtype） | 阶段 | O8（`--lsefull=0`） | cp.async（`--lsetma=0`） | **TMA（默认，O71）** | TMA vs cp.async | max_abs vs fp32 ref |
|---|---|---|---|---|---|---|
| fp16 定长 full S1024 H16 D128 | preprocess | 0.1602 ms | 0.0465 ms | **0.0362 ms** | **1.28×** | 同（3.27/2.52/1.23e-4） |
| fp16 定长 full S1024 H16 D128 | total | 0.3037 ms（28.3 TF） | 0.1687（50.9） | **0.1583 ms（54.3 TF）** | **1.066×** | — |
| bf16 定长 full S1024 H16 D128 | preprocess | 0.1602 ms | 0.0463 ms | **0.0364 ms** | **1.27×** | 同（1.94/1.68/1.45e-3） |
| bf16 定长 full S1024 H16 D128 | total | 0.3047 ms（28.2 TF） | 0.1700（50.5） | **0.1605 ms（53.5 TF）** | **1.059×** | — |

收益全在 preprocess（main 不变）。纯反向对标（同 session CUPTI，`harness/fa_bwd_bench.py bench`，
定长 full S1024 H16 D128）：**fp16** FA3 `0.0512ms/335.9TF`、TE `0.0577/297.9` ⇒ ours/FA3
**3.29×→3.09×**、ours/TE **2.92×→2.74×**；**bf16** FA3 `0.0504ms/340.7TF`、TE `0.0576/298.3` ⇒
**3.18×/2.79×**。

ncu（LSE，S1024 H16 full）：fp16 O8 **175.84µs / L1TEX 11.8%** → cp.async **38.62µs / L1TEX 37.6% /
Compute 39.6%** → TMA **28.16µs（vs O8 6.24×、vs cp.async 1.37×）/ L1TEX 18.8% / Compute 33.8% /
regs 64 / Waves 0.48**；bf16 TMA **28.32µs** 逐项一致。⇒ O8 是串行全局载入延迟 bound，`cp.async`
打掉它（4.55×），**4D-TMA 把 load 指令/地址运算交给 TMA 引擎（L1/TEX 37.6%→18.8%），再 1.37×**；
仍网格不足一个波、非 DRAM/L2/算力 bound。**这是 preprocess 内一条漏改分支的搬运升级，不是 main 的
L2 `red` 墙。** CI：fp16 一致性 gate worst 2.441e-4、bf16 4.883e-4（均 OK）、`--check docs/04` OK
（内嵌数值表不变，本步数值逐值一致）。详见 `docs/01` §23、`docs/01b` §6az、`docs/08` §5.79；原始
输出 `src/fp16/fa_bwd_fp16_o71_*`、`src/bf16/fa_bwd_bf16_o71_*`、`src/fa_bwd_o71_*`。

## 46. O73（第一百六十七轮，正结果，默认）：**varlen** full D=128 的 LSE 也上 **4D-TMA**（fp16/bf16）

第 166 轮 fp8 O72 把 varlen full D=128 的 LSE 切到 4D-TMA；O69→O70→O71 只覆盖**定长**，`run_varlen`
里 full D=128 仍走 O69 的 `cp.async` 均衡版。本轮把 `lse_mma_kernel_bal_tma` 加
`const int* cu_seqlens`（`nullptr` 逐式退化为定长 ⇒ 定长/因果路径逐位不变）、`run_varlen` 为
D=128/full 建 packed 描述符并优先走 TMA、`harness/fa_bwd_run.py` 的 varlen 构建对 fp16/bf16 也加
`-DFA_TMA -lcuda`。`--lsetmavarlen=0` 退回 cp.async 做同 binary A/B。**三 dtype × {定长, varlen}
full D=128 的 LSE 至此全部统一到 4D-TMA。**

| case（varlen D=128 full） | dtype | cp.async（`--lsetmavarlen=0`） | **TMA（默认，O73）** | 加速 | max_abs vs fp32 ref |
|---|---|---|---|---|---|
| b4_t3840 | fp16 | 0.9397 ms（48.56 TF） | **0.8810 ms（51.80 TF）** | 1.067× | 5.60/6.84/2.34e-4 |
| b4_t4096 | fp16 | 0.5908 ms（58.16 TF） | **0.5517 ms（62.28 TF）** | 1.071× | 4.09/4.95/1.23e-4 |
| b5_t3968（h32kv8） | fp16 | 1.8331 ms（49.93 TF） | **1.7218 ms（53.16 TF）** | 1.065× | 7.34/7.91/4.88e-4 |
| b8_t2904 | fp16 | 0.7562 ms（48.61 TF） | **0.7113 ms（51.69 TF）** | 1.063× | 1.49/1.56/1.58e-3 |
| b4_t3840 | bf16 | 0.9522 ms（47.92 TF） | **0.8878 ms（51.40 TF）** | 1.072× | 5.76/3.85/3.03e-3 |
| b4_t4096 | bf16 | 0.5986 ms（57.40 TF） | **0.5589 ms（61.48 TF）** | 1.071× | 3.24/2.39/2.01e-3 |
| b5_t3968（h32kv8） | bf16 | 1.8528 ms（49.41 TF） | **1.7496 ms（52.32 TF）** | 1.059× | 5.32/5.45/5.88e-3 |
| b8_t2904 | bf16 | 0.7622 ms（48.23 TF） | **0.7190 ms（51.13 TF）** | 1.060× | 1.15e-2/9.53e-3/1.08e-2 |

纯反向对标（同 session CUPTI，`harness/fa_bwd_bench.py bench --lengths 1024 1024 1024 1024 --H 16
--D 128 --full`）：`[1024]×4` full **fp16** FA3 `0.1993ms/172.4TF` ⇒ ours/FA3 **2.96×→2.77×**；
**bf16** FA3 `0.1978ms/173.7TF` ⇒ ours/FA3 **3.03×→2.83×**。

ncu（LSE，`b4_t4096` full，`--launch-count 1`，fp16/bf16 逐项一致）：cp.async **109.0µs /
inst 58.86M / L1TEX 50.9% / Compute 57.7%** → TMA **66.8µs（1.63×）/ inst 36.08M（−38.7%）/
L1TEX 27.1% / Compute 57.2%**；两者 DRAM 10–16%、L2 34–44% ⇒ O8 式的**串行载入延迟/issue bound**，
4D-TMA 把 Q/K 搬运交给 TMA 引擎。**这是 preprocess 内一条漏改分支的搬运升级，不是 main 的 L2 `red`
墙。** 一致性 gate：fp16 worst 2.441e-4、bf16 1.953e-3（均 OK）；`--check docs/04` OK。详见
`docs/01` §24、`docs/01b` §6ba、`docs/08` §5.81；原始输出
`src/fp16/fa_bwd_fp16_p167_lse_tma_ab.out.txt`、`..._p167_ncu_lse_tma.out.txt`、
`src/bf16/fa_bwd_bf16_p167_lse_tma_ab.out.txt`、`..._p167_ncu_lse_tma.out.txt`、
`src/fa_bwd_p167_fa3_varlen_full_baseline.out.txt`。

## 47. O75（第一百七十轮，正结果，默认）：MLA（D=512）causal LSE 上 **4D-TMA**（fp16/bf16）

第 169 轮 fp8 O74 把 `lse_mma_kernel_bal_tma` 从 `static_assert(HD==128)` 泛化为 `NCH=HD/128`
个 TMA box，让 MLA（D=512）causal LSE 也上 4D-TMA。本轮把同一步逐字 dtype 化到 fp16/bf16
（box 内维 128B = 64 个 fp16/bf16 ⇒ `NCH=HD/64`、HD=512 用 8 个 K=64 chunk），`NCH==2`（HD=128）
走原 `wgmma_qkt64_tma` ⇒ **HD=128 逐位不变**。host `lse_tma` 默认 `(D==512&&causal)?1:0`、causal
分支优先 `lse_mma_kernel_bal_tma<512,1>`（smem 197,696B ⇒ 1 CTA/SM）；`--lsetma=0` 退回 cfg6
（`<512,1,false,128,16>`）做同 binary A/B。**至此三 dtype × MLA 的 causal LSE 也统一到 4D-TMA。**

| case（MLA causal，fp16） | preprocess mma | **preprocess TMA** | LSE 加速 | total mma | **total TMA** | 端到端 | max_abs vs ref |
|---|---|---|---|---|---|---|---|
| S256 H2 D512 | 0.0193 ms | **0.0131 ms** | 1.47× | 0.0420 ms | **0.0359 ms** | 1.17× | 1.638/1.582/1.753e-3 |
| S512 H4 D512 | 0.0237 ms | **0.0197 ms** | 1.20× | 0.1092 ms | **0.1051 ms** | 1.04× | 2.516/2.916/1.724e-3 |
| S1024 H2 D512 | 0.0322 ms | **0.0206 ms** | 1.56× | 0.1785 ms | **0.1666 ms** | 1.07× | 1.987/1.712/1.848e-3 |

bf16 同构（preprocess 1.48×/1.21×/1.54×、端到端 1.18×/1.05×/1.07×，max_abs ~1e-2）。main 不变
（MLA main 仍 mma/cp.async，占 74–82%）；**MLA（D=512）反向 FA2/FA3/TE 均不支持**，故只有 ours
数字（S1024H2 total 0.1666ms / 25.8 TF，fp16 峰值 989 的 2.6%）。ncu（S1024H2 LSE，`--launch-count 1`）：
mma cfg6 **20.26µs / Compute 30.39% / L1TEX 42.10% / 2 CTA/SM** → TMA **12.99µs（1.56×）/
Compute 10.49% / L1TEX 16.63% / 1 CTA/SM**（bf16 12.77µs 逐项一致）⇒ **搬迁方式升级，非
DRAM/L2/算力 bound**。数值与 mma 版**打印逐位相同**；`--ci` 三 dtype gate 全 OK（fp16 1.953e-3 /
bf16 7.812e-3 / fp8 5.722e-6）、单/两文件一致性 worst 7.812e-3 OK、`--check docs/04` OK（内嵌
auto-table 在 rtol=0.005 内不变）。详见 `docs/01` §25、`docs/01b` §6bb、`docs/08` §5.84；原始输出
`src/fp16/fa_bwd_fp16_o74_mla_lse_ab.out.txt`、`src/fp16/fa_bwd_fp16_mma_onefile_o74_mla.out.txt`、
`src/bf16/fa_bwd_bf16_o74_mla_lse_ab.out.txt`、`src/bf16/fa_bwd_bf16_mma_onefile_o74_mla.out.txt`、
`..._o74_ncu_lse_tma_s1024h2.out.txt`。

## 48. O81（第一百七十六轮，正结果，默认）：fp8 GEMM5（dQ）切到 **wgmma RS**

- **改动**：fp8 默认 Hopper（kvtma）main 的 GEMM5（dQ，M=BM=64）从 `mma.m16n8k32` 换到
  `wgmma.m64n32k32` RS（A=dS2 寄存器、B=Kᵀ no-swizzle 描述符）；Kᵀ 由**从 SW128 K stage 逐字节
  转置**得到、**复用 Kp 缓冲 ⇒ smem 零增长**。`-DFA_WGMMA5=0` 回退 A/B（默认 1）。
- **数值**（不变）：S4096 vs fp32 ref dq/dk/dv max_abs = **2.635e-1 / 2.644e-1 / 3.216e-1**、
  relL2 **8.15% / 8.26% / 6.49%**（护栏内）；S512 = 2.426e-1/2.972e-1/3.733e-1（与历史打印相同）；
  GQA/full 亦过。`--ci --dtype fp8 --hopper` 全绿（gate 7.629e-6、docs check OK 198 行）。
- **性能**（同 binary A/B，S4096 H16 causal）：main **1.5564→1.4810ms（1.051×）**、total
  **1.7885→1.7235ms（1.038×，76.84→79.74 TF）**；ncu `smsp inst` −11.1%、HMMA −26.9%、
  `lts read` −9.3%、`lts red` 114.52M 不变、regs 168 不变。相对 TE FP8 纯反向
  （0.3025ms/908.7TF）**5.93×→5.70×**。
- 详见 `docs/03` §104、`docs/08` §5.90；原始输出 `src/fp8/fa_bwd_fp8_o81_*`、
  `src/fp8/fa_bwd_fp8_wgmma345_smoke.out.txt`。

## 49. O82（第一百七十七轮，**负结果**，默认关）：fp8 GEMM3/4（dV/dK）切到 **wgmma RS（M 零填充 m64）**

- **动机**：承接 O81（GEMM5 已 wgmma），把 fp8 默认 Hopper main 剩下的两条 HMMA GEMM——GEMM3(dV)、
  GEMM4(dK)——也换到 wgmma RS。二者 M=BN=32 < wgmma 最小 m64，故把 A（Ap/dS3）的 M 维**零填充到 64**
  （warp 2/3 的 A 寄存器置 0、其输出行 32–63 丢弃）。B=dOᵀ/Qᵀ 由 SW128 Q/dO 逐字节转置成**紧凑
  no-swizzle K-major [HD][BM]**（复用 Qp/dOp 缓冲，smem 零增长）；A 经 `ldmatrix.x4` 装入。
- **数值（正确，与 mma 版同量级）**：S4096 vs fp32 ref relL2 dq/dk/dv = **8.149% / 8.263% / 6.489%**
  （与 wg34=0 **打印完全相同**，护栏内）；A/B wg34_1 vs wg34_0：dq max_abs 1.19e-7（逐位）、
  dk 9.13e-4 / dv 2.28e-3（原子次序噪声）。S512 = 2.426e-1/2.973e-1/3.726e-1（基线 2.426/2.972/3.733e-1）。
- **性能（负结果，同 binary A/B，S4096 H16 causal）**：main **1.4738→1.5289ms（0.964×）**、
  total **1.7119→1.7694ms（0.967×，80.28→77.67 TF）**。ncu：Duration 1.49→1.56ms、
  Executed Instructions **641.86M→594.83M（−7.3%）**、L1/TEX **74.44→67.55%**、L2 **78.81→75.45%**、
  Compute 45.06→40.30%、regs 168→164，但 **No Eligible 53.22→58.19%**（延迟 bound 恶化）。⇒
  **本 kernel 是 L2 `red` bound（~75%，wgmma 一字不减）+ 延迟 bound**；零填充让张量核做 2× 无用功，
  Q/dO 转置与 wgmma `fence/commit/wait` 又加延迟，指令路径的收益被抵消。**默认关**
  （`-DFA_WGMMA34=1` 复现）。F3b 的「无 BN=64 时上 GEMM3/4 wgmma」子路线就此判负。
- 详见 `docs/03` §105、`docs/08` §5.91；原始输出 `src/fp8/fa_bwd_fp8_o82_ab_wg34_{0,1}_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o82_ncu_wg34_{0,1}_s4096.out.txt`。

## 50. O85（第一百八十轮，**中性，默认关**）：fp8 `head_dim=256` 的 Q/dO 切 4D-TMA（chunk-major）

- **改动**：`D=256` 的 Q/dO 从 cp.async 改 4D-TMA（chunk-major `[k/128][row/8][8][128]`，
  NCH=2）；新增 `sw128c_*` + `wgmma_mn32_issue_cm`（`D=128` 逐位不变）。`--d256tma=1` opt-in。
- **数值**：relL2 vs fp32 ref 与 O84 档**逐位相同**（S1024H8 causal dq/dk/dv 8.332%/8.435%/6.464%，
  max_abs 2.633/2.797/3.572e-1）；tma1-vs-tma0 max_abs ~1e-7。
- **性能（同 binary A/B，event）**：S1024H8 causal total 0.3003–0.3032 vs 0.3054–0.3066ms
  （**+1.3–1.7%**）、S1024H8 full **+1.3%**、S2048H8 causal **−1.1%**、GQA h16kv4 **−3.7%**；
  ncu 隔离 Duration **245.5 vs 255.6µs（1.041×）**、指令 **−3.2%**、2 CTA/SM。**净中性 ⇒ 默认关**。
- 详见 `docs/03` §108、`docs/08` §5.94；原始输出 `src/fp8/fa_bwd_fp8_o85_*`。

## 51. O89（第一百八十四轮，**正结果，默认**）：fp8 主 kernel 的 **LPT m 块调度序**（causal 贵块先跑）

- **改动（纯 host + 壳透传，device 数学零改动）**：默认 fp8 `kvtma` 主 kernel 的 m 块号就是
  `blockIdx.x/ksplit`，而硬件按 blockIdx 升序派发；causal 下便宜块在低 blockIdx ⇒ 尾波全重块。
  给 `fa_bwd_fp8_mma_kvtma_kernel` 壳加 `mt_m` 透传，host 在 `--mrev=1` 时建反转表
  `mt_m[i]=nblk-1-i`，改成「贵块先跑」（LPT）。门控 `causal && D==128 && nblk>=16`，
  单/两文件同步，`--mrev=0` 供 A/B，**默认开**。
- **数值（护栏全过）**：`--ci --dtype fp8 --hopper` 单/两文件一致性 gate worst **7.629e-6 OK**、
  `--check docs/04` OK；vs fp32 ref max_abs 与 `mrev=0` **打印相同**（S4096 2.635/2.644/3.216e-1、
  S512 2.426/2.972/3.733e-1）。`--mma`（sm_90）构建不含 kvtma ⇒ 无效、逐位不变。
- **性能（同 binary A/B，S4096 H16 causal，event）**：main **1.4796→1.4475ms（1.022×）**、
  total **1.7159→1.6747ms（1.025×，80.10→82.07 TF）**（重复实测 ~3.2%）。其它 causal D=128：
  S1024H32 main 1.6%、GQA kv4 2.0%、GQA kv8 1.3%、MQA kv1 0.5%；S512 门控外不变。
- **ncu（S4096）**：Duration **1.48→1.46ms**、**`lts op_red` 114,524,160 一字不变**、
  stall `short_scoreboard 1.82`/`wait 1.56` 均不变 ⇒ 收益 = **尾波/负载均衡**，与 L2 搬运量无关
  （再次印证 `red` bound；减 `red` 仍需改工作划分）。
- 详见 `docs/03` §112、`docs/08` §5.98；原始输出 `src/fp8/fa_bwd_fp8_o89_*`。

---

## 52. O93（第一百八十八轮，**正结果，默认**）：fp8 主 kernel 的**跨 head 全局 LPT**（grid 轴对调）+ 低 ksplit

承接 §51（O89，per-head LPT m 块降序，`red` 一字不变）：O93 把 LPT 升级为**跨 head 全局**
并在其解锁下把 **ksplit 8→2**（消 Q/dO 重读）。**device 数学一行未改**（`fp8_mma_body` 加
`bool HSWAP`，只改 `h/mt/part` 从哪个 blockIdx 取；host `grid=(H, nblk*ksplit, B)` head 走快轴），
`hswap_elig`（定长 causal D=128 nblk≥16）时 auto ksplit 收到 2。默认开（`--hswap=0` 回退）。

| S4096 H16 B1 causal（iters=40，同 binary A/B） | main (ms) | total (ms) | TFLOPS |
|---|---|---|---|
| `hswap=0`（§51 默认，ksplit=8） | 1.4304 | 1.6686 | 82.37 |
| `hswap=1` ksplit=8 | 1.8699 | 2.0936 | 65.65 |
| **`hswap=1` ksplit=2（新默认）** | **1.3742** | **1.6109** | **85.32** |

- **main 1.041× / total 1.036×**；其它 causal D=128：S1024H32 1.10×、GQA kv4 1.13×。
- **ncu（默认 vs `--hswap=0`）**：Duration 1.45→1.39ms、L2 总扇区 −9.1%、`read` −13.0%
  （Q/dO 重读 8×→2×）、`red` −8.0%（dQ 跨 part 原子减少）；代价 DRAM 219→557MB（跨 head 交错
  损 L2 局部性），但主 kernel 墙是 L2 吞吐 ⇒ 净快。
- **数值（vs fp32 ref，relL2）**：S4096 dq/dk/dv **8.148 / 8.263 / 6.489%**（护栏 8.2/8.3/6.5 内，
  与 §51 同档）；`max_abs` 2.635/2.644/3.216e-1。`--ci --dtype fp8 --hopper` gate worst
  **7.629e-6 OK**、`--check docs/04` OK。
- **对标**：同 session TE FP8 纯反向 S4096 = 0.3049ms ⇒ ours total **5.30×**（§51 时 5.50×）、
  main **4.52×**（§51 时 4.75×）。

见 `docs/03` §116、`docs/08` §5.102；原始输出 `src/fp8/fa_bwd_fp8_o93_*.out.txt`。

---

## 53. O99（第一百九十三轮，**正结果，默认**）：fp8 **causal D=256** 的 ksplit 重标定

承接 §52（O93）与 `docs/03` §118–120（O96/O97/O98 复核 **full** 的 ksplit）：O96/O97/O98 只审了
full，**causal D=256** 一直沿用 O29 的 `target_ctas = S/2`（按 causal MLA/D=512/1 CTA/SM 标），
套到 D=256（2 CTA/SM→264 槽）上 `k = 32/(H*B)` 与 S 无关 ⇒ 小/中 S **欠切**。**纯 host、device
一行未改、单/两文件同源**；`--ksplit=K` 显式覆盖。规则：`k = clamp(2*S/base,1,16)` 再按 `nblk`
封顶（`grid = base*k ≈ 2*S`；变长同名分支用 `maxlen`）。

| causal D=256 shape（B,S,H,D；Hkv） | 旧 auto k | 旧 total (ms) | 新 auto k | 新 total (ms) | 加速 |
|---|---|---|---|---|---|
| (1,512,8,256)  | 4 | 0.1086 | 8  | 0.1005 | **1.068×** |
| (1,1024,8,256) | 4 | 0.3058 | 16 | 0.2793 | **1.112×** |
| (1,2048,8,256) | 4 | 0.8833 | 16 | 0.8051 | **1.104×** |
| (1,4096,8,256) | 4 | 2.9015 | 16 | 2.7569 | **1.055×** |
| (1,1024,16,256; Hkv4) | 2 | 0.5424 | 8  | 0.4754 | **1.143×** |
| (1,2048,16,256) | 2 | 1.6999 | 8  | 1.5188 | **1.122×** |

- **6 shape 全 ≥1.05×、无回退**；ours fp8 total 42.7/49.9 TF（s2048H8/s4096H8，FP8 峰值
  1978.8 TF ⇒ 2.2%/2.5%）。TE fp8 对 D=256 causal 报 `Invalid combination of data type and
  sequence`、FA3 不支持 fp8 ⇒ 无 fp8 外部列（同 shape fp16 纯反向 TE 0.216/0.592ms 仅供参照）。
- **ncu（main）**：s2048H8 旧 k=4 769.6µs→新 k=16 **681.5µs**（`op_read` 6.05M→8.41M、L2
  67.1→76.4%）；s1024H16kv4 旧 k=2 464.1µs→新 k=8 **397.1µs**（L2 57.4→67.7%）；两组
  **`op_red` 一字不变**（dK/dV 主体与 ksplit 无关，续证 O83/O86/O95）⇒ 纯「加 k 买并发/藏延迟」。
- **精度护栏**：relL2 vs fp32 ref dq 8.15–8.33% / dk 8.33–8.48% / dv 6.39–6.50%（护栏内）、
  `max_abs` O(0.21–0.62)；单/两文件**逐位相同**。
- **全量 CI**（`fa_bwd_run.py --ci`，**93 case**）：三 dtype 一致性 gate **OK**（fp16 1.953e-3 /
  bf16 7.812e-3 / fp8 6.676e-6）；**顺带把 O96/O97/O98 遗留的 `docs/04` 内嵌数值表同步**
  （198→214 行，`--check` OK——新 D=256/D=512 full 行与 O96 的 full D=128 微调一并入表）。
- **无回归**：D=128 causal（hswap k=2）/ D=512 causal（`S/2`）/ 定长 full（O96/O97）逐档不变。

见 `docs/03` §121、`docs/08` §5.107；原始输出 `src/fp8/fa_bwd_fp8_o99_*`。

## 54. O103（第一百九十七轮，**正结果，默认**）：fp8 `head_dim=256` 的 LSE 上 **4D-TMA**

- **动机**：O102 收口后剩下覆盖型 backlog 里不撞 smem 墙的一条——D=256 的 LSE 仍走 mma +
  `cp.async`，而 LSE 的 TMA kernel 早在 O74 就泛化为 `NCH=HD/128`（D=512 已用）。
- **改动（纯 host、device 一行未改、单/两文件同源）**：`launch_lse_bal_tma_split` 加 `bool FULL`；
  D==256 建 LSE 描述符 + `cudaFuncSetAttribute<256,1>`；`lse_tma` 自动档纳入 D=256；D==256
  `run_preprocess` 加 TMA 优先路径（causal 镜像配对 + split / full FULL + split），`--lsetma=0`
  同 binary A/B。
- **性能（同 binary A/B，iters=40）**：preprocess **1.74–2.71×**、total **~5%**：S1024H8 causal
  0.2792→**0.2649ms（1.054×）**、S4096H8 causal 2.7648→**2.6373（1.048×）**、S1024H8 full
  0.2785→**0.2659（1.047×）**、S4096H16 full 5.4279→**5.1792（1.048×）**；main 段不变。
- **ncu（D=256 causal S4096 LSE）**：TMA **79.6µs / L1TEX 全局载入 1.196M / L2 10.49M / mem 45.1%**
  vs cp.async **211.6µs / 11.96M / 15.99M / 30.0%** ⇒ 载入扇区 **−10×**、Duration **2.66×**。
- **精度护栏**：D=256 fp8 无 FA/TE 外部列；15 个定长 shape `ours vs fp32 ref` relL2
  **8.14–8.33 / 8.29–8.48 / 6.39–6.76%**（护栏内，与 O84/O99 统计一致）；`max_abs` 与 mma 版同档；
  `--ci --dtype fp8` 单/两文件 gate **worst 7.629e-06 OK**、`--check docs/04` OK。

见 `docs/03` §125、`docs/08` §5.111；原始输出 `src/fp8/fa_bwd_fp8_o103_*`。

## 55. O104（第一百九十八轮，**正结果，默认**）：fp8 `head_dim=256` causal 的跨 head 全局 LPT

- **动机**：`fp8 专项冲刺`剩余唯一「不改工作划分/指令」的调度杠杆是 O93 的跨 head 全局 LPT
  （`HSWAP`）；O93 只接在 `kvtma`（D=128、需 TMA），而 **D=256 默认走通用
  `fa_bwd_fp8_mma_kernel`**（wgmma + cp.async、非 TMA），一直没有 LPT 排序。
- **改动（device 一行数学未改 + host、单/两文件同源）**：通用 kernel/launcher 加 `bool HSWAP`
  并透传 `fp8_mma_body`；host 对 D=256 定长 causal、`base_grid=nblk*H*B ≤ 256`、nblk≥16 建 O89
  `d_mrev` + grid 轴对调 + 自动 ksplit=4；`--hswap=0` A/B。
- **性能（同 binary A/B，iters=150，total）**：S1024H8 **1.141×**、S1024H16kv4 **1.093×**、
  S2048H8 **1.017×**；大 `base_grid`（S2048H16/S4096H8）为负 ⇒ 门控在 `base_grid ≤ 256`，
  门控外逐值不变。main-only（S1024H8）0.2051→0.1838ms。
- **ncu（D=256 causal S1024H8 main）**：Duration **231.5→194.5µs**、**`op_read` 2.665M→1.939M
  （−27%）**、**`op_red` 一字不变**（D=256 无 regdq）、L2 利用率 58.8%→69.9%。
- **精度护栏**：relL2 vs fp32 ref 与 hswap0 **逐位相同 8.332/8.434/6.464%**；`--ci --dtype fp8`
  gate **worst 7.629e-06 OK**、`--check docs/04` OK；D=256 fp8 无 FA/TE 外部列。

见 `docs/03` §126、`docs/08` §5.112；原始输出 `src/fp8/fa_bwd_fp8_o104_*`。

## 56. O129（第二百二十三轮，**负结果/覆盖**）：fp8 causal MHA `S=8192` —— 平台期延伸到 S4096 之外

把「fp8 主 kernel 默认档已全局最优」的结论从 S≤4096 延伸到 **S=8192**（`(1,8192,16,128) causal`，
nblk=128），新增 dump 目录 `b1_s8192_h16_d128_causal_fp8`（含 `ref_*`/`te_*`/`ours_*`/`ours_sf_*`）。

- **性能 / 对标（纯反向口径）**：ours total **5.84ms / main 5.17ms（≈94 TFLOPS，`4BS²HD`）；
  TE FP8 纯反向 **1.044ms / 1052.8 TFLOPS** ⇒ ours main **5.0×** / total **5.6×** TE**（与 S4096
  的 5.29× 同档）。**FA3 无 fp8 反向列**。
- **旋钮审计（同 binary）**：`--ksplit=1/4/8` = 0.971×/0.817×/0.579×（**auto=2 仍最优**）、
  `--hswap=0` 0.969×、`--mrev=0` 0.945×、`--ksm` 0.98×、`--det` 0.986×、`--wg2/--wg3` 0.49/0.59×、
  `--bn64` 0.66×、`--qdtma=0/--kvtma=0` 0.76/0.78×、`-DFA_WGMMA34=1` 0.916× ⇒ **无新正结果**。
- **ncu**：ours `red` **412.1M 扇区（L2 81.5%）**、L1→L2 **精确 1.50× 展宽**、3 CTA/SM/168 regs；
  vs TE `red` 102.2M（**4.03×**）、`read` 2.42×、时间 5.5× —— 与 S4096 的红字账逐项同构。
- **数值（护栏）**：ours vs fp32 ref `max_abs` dq/dk/dv = 2.215e-1/2.680e-1/2.976e-1（**≤ TE-vs-ref**）；
  单/两文件一致性 worst 2.38e-7（`--ci` OK）；`--check docs/04` 已同步（242 行）。

⇒ **负结果、默认一行未改**；S8192 作为覆盖 shape 纳入 harness（`S8192_FP8_SHAPES`）。
**正结果仍只剩换卡**（见 `docs/03` §149、「阻塞」）。
