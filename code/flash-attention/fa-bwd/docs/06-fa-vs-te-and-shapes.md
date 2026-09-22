# 目标形状重点分析 & FA 为何比 TE 慢（测试方法 + ncu 归因）

> 本文回答两件事：① 对用户指定的 GQA/MQA/MLA 形状做**重点分析**（数值 + 性能 + bound）；
> ② 仔细审查「FA 比 TE 慢」是不是**测试方法**的问题，并用 **ncu** 找到根因。
> 原始输出：`src/fa_vs_te_bwd_only.out.txt`（纯反向基准）、`src/fa_vs_te_kernel_profile.out.txt`（kernel 列表）、
> 以及下文 ncu/SASS 采集。硬件 H100 80GB HBM3（sm90），flash_attn **2.7.4**、TE **2.14**。

---

## 1. 先修测试口径：区分「正向+反向」与「纯反向」

之前的 `harness/fa_bwd_bench.py` 里，`fa_bwd()` 和 `te_bwd()` **都在 lambda 内先跑了一遍 forward 再 backward**，
所以之前表格里的 FA/TE 实际上是 **fwd+bwd 合计**，而不是纯反向（`te-perf` 那篇的 `fused_attn_bwd` 是纯反向）。
这不是「错」，但两个口径混用会误导，必须分开。

新增 `harness/fa_vs_te_bwd_only.py`：把 forward 移出计时区
（FA 用 `autograd.grad(o,[q,k,v],do,retain_graph=True)` 只跑反向；TE 先算好 `aux_ctx` 只跑 `fused_attn_bwd`），
CUPTI 纯 device 时间。结果（fp16；bf16 完全一致）：

| 形状 (B,S,H,D) | FA 2.7.4 纯反向 | TE 2.14 纯反向 | TE/FA |
|---|---|---|---|
| (1,1024,40,128) kv=8 GQA | 0.1874 ms / 229.1 TF | 0.1329 ms / 323.3 TF | **0.71×** |
| (1,1024,32,128) kv=4 GQA | 0.1584 ms / 216.9 TF | 0.1125 ms / 305.4 TF | **0.71×** |
| (1,1024,64,128) kv=4 GQA | 0.2667 ms / 257.7 TF | 0.1943 ms / 353.7 TF | **0.73×** |
| (1,1024,64,128) kv=1 MQA | 0.2666 ms / 257.8 TF | 0.2154 ms / 319.1 TF | **0.81×** |
| (1,256,2,512) MLA | FA 不支持（head_dim≤256） | TE 不支持（训练 bwd head_dim≤256） | — |
| (1,512,4,512) MLA | 同上 | 同上 | — |
| (1,1024,2,512) MLA | 同上 | 同上 | — |
| (1,4096,16,128) MHA | 0.7293 ms / 376.9 TF | 0.4447 ms / 618.1 TF | **0.61×** |

**结论**：纯反向口径下 TE 比 FA 快 **1.2–1.6×**（GQA/MQA 1.2–1.4×，大 MHA 1.64×）。
之前用 fwd+bwd 时差距被放得更大（因为 FA2 的 forward 也慢），所以「FA 比 TE 慢很多」一部分是
**口径问题（把较慢的 FA forward 也算进去了）**，但**纯反向也确实慢**——根因见 §3。

> 测试方法审查清单（都核对过，不是方法错）：
> - `q,k,v,do` 都 contiguous；`scale=1/√D`（两者一致）；`causal`；`softmax=vanilla`；`no_bias`。
> - CUPTI 纯 device 时间，autograd 的 CPU 开销不计入。
> - MLA 两侧都不支持，`nan` 是能力问题，不是测量问题。
> - 唯一要修的就是「fwd 是否计入」——上面的纯反向表已经修正。

---

## 2. 目标形状的重点分析

### 2.1 数值（max_abs vs fp32 ref）

fp16（容差 ~1e-2）与 bf16（~1e-1）下，FA/TE 与 ref 同量级；ours（fp16/bf16）也已支持这几个形状，
数值同样达标。逐形状数值表见 `docs/04` §7.1/§7.5（FA/TE）与 §7.3/§7.5（ours）。关键点：

- **GQA/MQA 的 kv 头越少，dk/dv 误差越大**（kv=1 时 dv 误差最大）：每个 KV 头承载 `H/Hkv` 个 Q 头的梯度，
  低精度误差在更多累加项上叠加。
- **MLA（head_dim=512）**：只有 fp32 ref 与 ours(fp16) 可比；FA/TE 无法给出反向。

### 2.2 性能与 bound（ours 视角）

ours 的 fp16/bf16/fp8 反向都已支持 4 个 GQA/MQA 形状（fp8 在 mma 路径上），fp16 已支持 MLA head_dim=512；
但 ours 的 fp16/bf16 仍是**标量 CUDA-core** 实现（见 `docs/04` 与 ROADMAP「性能差距归因」），
性能远低于 FA/TE：例如 bf16 kv4(h64) main 10.96 ms / total 28.44 ms（≈1.2 TF，峰值 0.12%）。
MLA 三个形状（fp16）：main 1.06 / 2.09 / 4.17 ms，bound = smem bank conflict + 1 CTA/SM。

**当前性能排序**：TE(cuDNN wgmma) > FA2(mma.sync) ≫ ours(标量)。补齐路径 = ROADMAP 的 O5（fp16/bf16 张量核化）。

---

## 3. FA 为何比 TE 慢：kernel 级与指令级归因（ncu）

### 3.1 两个实现跑的**不是同一代 kernel**

用 `torch.profiler` 列出一次反向 launch 的 kernel（以 (1,1024,40,128) kv=8 causal 为例）：

| FA 2.7.4（6 kernels） | 时间 | TE 2.14（8 kernels） | 时间 |
|---|---|---|---|
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<...>>` | 142.4 µs | `cudnn_generated_..._sm90_flash_bprop_`**`wgmma`**`_f16_...` | 90.1 µs |
| `flash_fwd_kernel<Flash_fwd_kernel_traits<...>>`（forward） | 57.6 µs | `cudnn_generated_..._sm90_flash_fprop_`**`wgmma`**`_f16_...` | 35.2 µs |
| `at::native::reduce_kernel`（**GQA 梯度归约**） | 20.7 µs | `cudnn::fusion::compute_dot_do_o_specialized` | 14.0 µs |
| `flash_bwd_dot_do_o_kernel` | 14.3 µs | `cudnn::fusion::fmha_reduce_head`（**GQA 归约进 kernel**） | 13.7 µs |
| `flash_bwd_convert_dq_kernel` | 14.0 µs | `cudnn::fusion::convert_dq_to_16bits` | 13.3 µs |
| `Memcpy DtoD` | 13.3 µs | —（含在 kernel 内） | — |

三条关键差异：
1. **FA 的 bwd 主 kernel 是 `Flash_bwd_kernel_traits`（FA2/SM80 实现）**；TE 的是
   **`..._sm90_..._wgmma_f16`（Hopper warpgroup MMA）**。
2. **FA 的 GQA 多一个 `at::native::reduce_kernel`**：FA 的 autograd 对 KV 广播做反向需要沿 Q 头求和，
   是一个**独立 kernel**；TE 用 `cudnn::fusion::fmha_reduce_head` 在 bwd kernel 内部完成。
   → 这解释了 **MQA/GQA 下 FA 的相对劣势更大**。
3. FA 还有一次 `Memcpy DtoD`（dQ 累加缓冲转换）；TE 的 `convert_dq_to_16bits` 已内联。

### 3.2 指令集（SASS）证据：Ampere mma.sync vs Hopper wgmma + TMA

用 `ncu --page source --print-source sass` 统计主反向 kernel 的 SASS 指令：

| 指令 | FA2 `flash_bwd_dq_dk_dv` | TE `cudnn ... flash_bprop` | 含义 |
|---|---|---|---|
| `HMMA.16816.F32` | **320** | 0 | Ampere `mma.sync.m16n8k16`（逐 warp 张量核） |
| `LDSM.16.M88.4` / `MT88.4` | 80 + 64 | 0 | `ldmatrix` |
| `LDGSTS` | 32 | 0 | `cp.async`（Ampere 异步拷贝） |
| `UTMALDG` / `UTMASTG` | 0 | 8 / 4 | **TMA**（Hopper `cp.async.bulk.tensor`） |
| `WARPGROUP.ARRIVE` / `DEPBAR` | 0 | 5 / 5 | **warpgroup/wgmma** 同步 |
| `STSM.16.M88.4` | 0 | 20 | `stmatrix` |

**FA2 用的是 Ampere 一代的 `mma.sync` + `ldmatrix` + `cp.async`；TE（cuDNN）用的是 Hopper 的
`wgmma` + TMA + `stmatrix` + warpgroup 调度。** 这就是性能差 ~1.5× 的指令级根因。

### 3.3 ncu SpeedOfLight / occupancy 对照（两个主反向 kernel，MHA S=4096）

| 指标 | FA2 bwd main | TE(cuDNN) bwd main |
|---|---|---|
| Duration | 682 µs | 399 µs |
| DRAM Throughput | 6.6% | 11.3% |
| L1/TEX Throughput | **72.2%** | **63.2%** |
| L2 Throughput | 29.2% | 58.1% |
| Compute (SM) | 39.8% | 46.7% |
| Executed IPC Active | 1.04 | 1.28 |
| Achieved Occupancy | 12.5%（8 warps/SM） | 15.6%（10 warps/SM） |

两者都**不是 HBM/算力 bound**（DRAM <12%），都是 **L1/TEX + 延迟受限**；TE 的 IPC、occupancy、
Compute 都更高，L1/TEX 更低（更省片上流量）——与「wgmma 一次算更大 tile、TMA 省地址计算/LD 指令」
的预期一致。

### 3.4 一句话结论

> **不是测试方法错**（口径已按 §1 修正），而是 **`flash_attn` 2.7.4 pip 包 = FA2 = SM80(Ampere)
> kernel（mma.sync + cp.async）**，而 **TE 在 H100 上走 cuDNN 的 SM90 Hopper kernel（wgmma + TMA）**。
> FA 的 Hopper 实现在 **FA3**（仓库 `hopper/`，包名 `flash_attn_3`），装的是 2.7.4，没有 FA3 内核。
> 加上 FA 对 GQA/MQA 多一个 `reduce` kernel，所以小 shape 差距更明显。

> **⚠️ 更正（见 `docs/07-fa3-sm90-comparison.md`）**：把 **FA3（SM90）编译出来重测**后，结论反转为
> **FA3 比 TE 更快**（GQA/MQA 355–438 TF vs TE 307–356 TF；MHA S=4096 **850 TF vs 618 TF**）。
> 本文「FA 比 TE 慢」仅对**FA2（SM80）**成立。**H100 上的正解对标是 FA3**，请以 `docs/07` 为准。

---

## 4. ncu 能 / 不能回答什么（方法论）

- **能**：确认跑的哪代 kernel（kernel 名/SASS 指令）、找瓶颈单元（SOL 的 DRAM/L1/L2/Compute）、
  看 occupancy、IPC、warp stall。上面 §3.2/§3.3 全部来自 ncu。
- **不能**：直接告诉你「哪个实现更先进」——那是**指令集/代际**问题，要靠 SASS 与 kernel 名判断；
  ncu 只给「这个 kernel 此刻被什么限制住」。
- 对 Python 里的库 kernel，用 `ncu --target-processes all -k regex:<名字>` + `--launch-skip` 取稳态，
  并用 `-k regex:... --page source --print-source sass` 看指令。

---

## 5. 对 ours 的启示

1. **ours 的补课目标**：fp16/bf16 从「标量 CUDA-core」升级到 **`mma.m16n8k16`+`ldmatrix`**（O5，
   预期 main 10–20×），再叠加 **`cp.async` 流水**与 **4 CTA/SM**（O6）——这正是 FA2 的水平；
2. 要超过 FA2、逼近 TE，需要再上 **Hopper `wgmma` + TMA + warp specialization**（O9）——即 FA3 的路线；
3. **GQA/MQA 的 KV 归约要放进 kernel**（别像 FA 那样单开 reduce），否则小 shape 吃亏；
4. 基准统一用**纯反向**口径（`harness/fa_vs_te_bwd_only.py`），并正确区分「对标 FA2」还是「对标 TE/FA3」。
