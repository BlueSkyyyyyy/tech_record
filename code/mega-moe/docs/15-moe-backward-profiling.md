# MoE 反向 grouped GEMM 性能归因：wgrad ~100 为什么只有 dgrad ~500 的 1/5（H100 / ncu）

> 承接第 12 篇（`docs/14-moe-backward-kernels.md`）：那篇把 SM90 的四个 grouped GEMM 接进真实
> MoE 反向图并量到吞吐——wgrad ~100 TFLOPS、dgrad ~450–600 TFLOPS，差 4~5 倍，但没解释**为什么**。
> 本篇用 `ncu` 采集硬指标（DRAM 字节、tensor pipe 利用率、GMMA 指令数、占用率），把差距归因到
> **访存受限 vs 计算受限** 的 roofline，并指出根因是 **权重梯度的形状 + SM90 累积语义**。

## 1. 问题：同一份反向图里，两类 GEMM 为何差 5 倍

DeepSeek 量级单 rank 反向（T=2048, H=7168, I=2048, E=8, topk=2）的四个大 GEMM：

| # | 算子 | API | 语义 | (M,N,K)（收缩维=分组轴） | 输出 D |
|---|---|---|---|---|---|
| w2g | linear2 wgrad | `k_grouped_fp8_gemm_nt_contiguous` | `dW2[e]=A_eᵀ dY_e` | M=I=2048, N=H=7168, **K=token** | `[E,I,H]` fp32 |
| w1g | linear1 wgrad | `k_grouped_fp8_gemm_nt_contiguous` | `dW1[e]=x_eᵀ dG_e` | M=H=7168, N=2I=4096, **K=token** | `[E,H,2I]` fp32 |
| d2g | linear2 dgrad | `m_grouped_fp8_gemm_nt_contiguous` | `dA_e=dY_e W2_eᵀ` | **M=token**, N=I=2048, K=H=7168 | `[Σt,I]` bf16 |
| d1g | linear1 dgrad | `m_grouped_fp8_gemm_nt_contiguous` | `dX_e=dG_e W1_eᵀ` | **M=token**, N=H=7168, K=2I=4096 | `[Σt,H]` bf16 |

第 12 篇实测（持续负载、event 计时）：w2g/w1g ≈ 103/106 TFLOPS，d2g/d1g ≈ 449/543 TFLOPS。
`ncu` 单次 launch（热 cache）则量到 w2g/w1g ≈ 240 TFLOPS、d2g/d1g ≈ 1030/1200 TFLOPS——
绝对值受**持续负载降频**影响翻了约 2 倍，但**两者比值稳定在 4~5 倍**。所以差距是结构性的，不是噪声。

## 2. 实验方法

单个 kernel 一个进程，避免四个 kernel 互相干扰，也避免 `ncu` 抓到 PyTorch 的量化前置 kernel：

- `harness/exp_backward_ncu.py`：构造与第 12 篇 `perf()` 完全相同的输入（`src`/`harness`），
  只跑一种 kernel；`kind ∈ {w2g,w1g,d2g,d1g}`。用 CUDA event 量 **单次**（热启 3 次后测 1 次）与
  **持续**（热启 10 次后测 100 次）两种时钟状态。
- `harness/exp_backward_ncu.sh`：`ncu --launch-count 1 --kernel-name 'regex:sm90_fp8_gemm'`，
  wgrad 命中 `sm90_fp8_gemm_1d1d_impl`、dgrad 命中 `sm90_fp8_gemm_1d2d_impl`；两段采集：
  ① 五个 section（SpeedOfLight/Occupancy/LaunchStats/Memory/Compute）；
  ② 定向 metric（单次时长、`dram__bytes_read/write`、`sm__inst_executed_pipe_tensor_op_gmma`、`sm__cycles_active`）。

原始输出见 `harness/exp_backward_ncu.out.txt`（H100 80GB HBM3, sm90, ncu 2026.1.1）。

## 3. 硬指标

`ncu` 单次 launch（热 cache，`Duration` 与 event 单次一致）：

| kernel | FLOPs | Duration | 计算(SM) % | tensor pipe % | DRAM 读 | DRAM 写 | 总字节 | 实际带宽 | GMMA 指令 |
|---|---|---|---|---|---|---|---|---|---|
| w2g | 120.3 GF | 499.9 us | **12.4** | 低 | 530.7 MB | 460.1 MB | 990.8 MB | 1.98 TB/s | 786 432 |
| w1g | 240.5 GF | 998.8 us | **12.2** | 低 | 1.09 GB | 932.5 MB | 2.02 GB | 2.02 TB/s | 1 548 288 |
| d2g | 120.3 GF | 111.0 us | **65.5** | **65.5** | 180.9 MB | 25.0 MB | 205.9 MB | 1.85 TB/s | 630 784 |
| d1g | 240.5 GF | 194.3 us | **75.7** | **75.7** | 293.2 MB | 64.2 MB | 357.4 MB | 1.84 TB/s | 1 474 560 |

四个 kernel 的占用率、寄存器、smem 几乎一样（168 regs/thread、~226 KB smem、理论占用 18.75%、
实际 ~14%、grid=132=SM 数、block=384，均为 1 CTA/SM 的 persistent 调度）。**差异不在占用率**，
而在：

- **计算 (SM) 吞吐**：wgrad 只有 **12%**，dgrad 是 **65–76%**（都是 tensor pipe 主导）。
- **算术强度**（FLOP / DRAM 字节）：
  - w2g = 120.3e9 / 990.8e6 ≈ **121** FLOP/B；w1g ≈ **119** FLOP/B
  - d2g = 120.3e9 / 205.9e6 ≈ **584** FLOP/B；d1g ≈ **673** FLOP/B
- H100 FP8 roofline 拐点 = 峰值算力 / HBM 带宽 ≈ 1979 TFLOPS / 3.35 TB/s ≈ **591** FLOP/B。

**wgrad 的算术强度不到拐点 1/5 → 访存受限；dgrad 在拐点附近/以上 → 计算受限。** 这就是全部。

## 4. 为什么 wgrad 是访存受限：三个叠加的原因

### 4.1 输出是「权重的梯度」——大且是 fp32

wgrad 的 D 就是整张专家权重矩阵的梯度，形状 `[E, M, N]`：

- w2g：`8 × 2048 × 7168 × 4 B = 470 MB`
- w1g：`8 × 7168 × 4096 × 4 B = 940 MB`

而 dgrad 的 D 是**激活**的梯度，形状 `[Σt, N]`（`Σt = T·k = 4096`）：

- d2g：`4096 × 2048 × 2 B(bf16) = 16 MB`
- d1g：`4096 × 7168 × 2 B = 59 MB`

即使按 element 数，wgrad 的输出也比 dgrad 大 7–57 倍，且是 fp32（4 B）而 dgrad 是 bf16（2 B）。
ncu 实测 DRAM 写正好落在 D 的大小上（w2g 460 MB、w1g 932 MB）。

### 4.2 SM90 的 `k_grouped_nt` 强制累积：再读一遍 D

`csrc/apis/gemm.hpp:403` 对 `k_grouped_fp8_gemm_nt_contiguous` 断言
`c.has_value() and c.value().is_contiguous()`——**C 是必填参数**（不像通用 `fp8_gemm_nt` 里 C 可选），
即该算子固定计算 `D = C + AᵀB`，**必须把 C 读进来**。第 12 篇的调用把 `d` 同时当 C 和 D 传入
（`exp_moe_backward_kernels.py`：`..., d, host_ks, layout, d, recipe=...`），所以每个 launch
都要**回读 D**。ncu 的 DRAM 读≈写+操作数正好坐实这点：

- w2g：读 530.7 MB ≈ 写 460 MB(D 回读) + A(8 MB) + B(29 MB) + SF
- w1g：读 1.09 GB ≈ 写 932 MB(D 回读) + A(29 MB) + B(17 MB) + SF

也就是说，**fp32 权重梯度每算一次要过两遍 DRAM**（读旧 D + 写新 D），把本就巨大的输出流量再放大一倍。

### 4.3 收缩维太浅：K = 每专家 token 数

wgrad 的 K 是**不等长的 token 轴**，DeepSeek 量级每专家只有 `T·k/E = 512` 个 token（对齐后 512，
即 4 个 BLOCK_K=128）。GMMA 的 K 循环只有 4 步，tensor pipe 还没热身 tile 就结束了。
实测 GMMA 指令数 w2g 786 432 ≈ d2g 630 784（因为两者 FLOPs 相同），**但 w2g 花了 4.5 倍时间**——
说明指令并不是发不出去，而是**大部分周期在等 DRAM**。

### 4.4 roofline 收口

| kernel | 访存下界 | 计算下界 | roofline 下界 | 实测 | 达下界 |
|---|---|---|---|---|---|
| w2g | 990.8MB/3.35TB/s = **295.8 us** | 60.8 us | **295.8 us** | 499.9 us | 59% |
| w1g | 2.02GB/3.35 = **603.6 us** | 121.5 us | **603.6 us** | 998.8 us | 60% |
| d2g | 61.5 us | 60.8 us | **61.5 us** | 111.0 us | 55% |
| d1g | 106.7 us | 121.5 us | **121.5 us** | 194.3 us | 63% |

四个 kernel 都跑到各自下界的 55–63%（本质是都只用到 ~60% DRAM 峰值）。**关键在下界本身**：
wgrad 的下界由访存决定（~300–600 us），dgrad 由计算决定（~60–120 us），两者相差约 **5 倍**——
与实测的 4.5 倍吻合。差距是 roofline 决定的，不是 kernel 写得差。

## 5. dgrad 为什么是计算受限

对称地看 dgrad：输出是 token 的激活梯度（小），K 是完整 hidden 维（`H=7168` 或 `2I=4096`，
折算 56/32 个 BLOCK_K=128），K 循环足够深，WGMMA 流水能吃满；输出又是 bf16、无累积读回，
DRAM 流量只有 206/357 MB。于是 D 的算子强度高、tensor pipe 65–76%，达到 FP8 峰值的 55–63%
（单次 ~1.0–1.2 PFLOPS）。

## 6. 单次 vs 持续：降频让绝对值减半

同一个 kernel，`ncu` 单次（GPU 空闲、时钟高）与 event 持续 100 次（持续负载）：

| kernel | 单次 | 持续(100×) | 比值 |
|---|---|---|---|
| w2g | 505 us / 238 TF | 1116 us / 108 TF | 2.2× |
| w1g | 1012 us / 238 TF | 2232–2892 us / 83–108 TF | 2.2–2.9× |
| d2g | 117 us / 1031 TF | 230 us / 522 TF | 2.0× |
| d1g | 200 us / 1202 TF | 425 us / 567 TF | 2.1× |

持续负载下四个 kernel **同比**变慢 ~2 倍（H100 持续消耗下 SM 降频），所以第 12 篇报的
~100/~450 TFLOPS 与本篇 `ncu` 的 ~240/~1100 TFLOPS 并不矛盾——**比值不变，结论不变**。
报告绝对性能时必须说明计时方式（单次 vs 稳态），否则容易差一倍。

## 7. 对 MoE 训练的工程含义

1. **wgrad 是天然的访存瓶颈**：权重梯度大（`E×M×N`）、fp32、还要累积回读。想让 wgrad 提速，
   应从**减少 D 的字节**入手，而不是优化 MMA：
   - 输出用 bf16/fp16 累积（本实验沿用 fp32 是为了对齐第 12 篇的精度实验）；
   - 避免「把旧 D 当 C 读回」——若上层能保证每组只被一个 kernel 写一次、无需 accumulate，
     可省掉一半 D 流量；
   - 反向本就「无 mega-kernel」、wgrad 留守专家本地（见 `docs/04`），因此**按专家拆分 + 并行写回**
     更契合。
2. **dgrad 才是算力主场**：它占了反向 FLOPs 的大头且能跑满 tensor pipe，是反向 kernel 优化的主战场。
3. **看数要对齐口径**：`k_grouped` 的「强制 accumulate」是 SM90 的实现约束（`gemm.hpp:403`），
   在 H100 实测里占了一半的 DRAM 读；评估 SM100 版 `k_grouped_*_tn`（MN-major）时也要核对是否读 C。

## 8. 复现

```bash
# H100 容器内（kernel_lab，已装 deep_gemm 2.8.1）
bash code/mega-moe/harness/exp_backward_ncu.sh   # 约 2~3 分钟
# 原始输出：code/mega-moe/harness/exp_backward_ncu.out.txt
```

单 kernel 手工跑：`python exp_backward_ncu.py w2g --warmup 3 --iters 1`（单次）或
`--iters 100`（持续）。

## 9. 小结

- wgrad（k_grouped，1d1d）与 dgrad（m_grouped，1d2d）的 4~5 倍吞吐差，**是访存受限 vs 计算受限
  的 roofline 必然结果**：wgrad 算术强度 ~120 FLOP/B（拐点 591 的 1/5），dgrad ~600 FLOP/B。
- 根因三条：**权重梯度输出巨大且 fp32**；**SM90 `k_grouped_nt` 强制累积读回 D（`gemm.hpp:403`）**，
  使 D 流量翻倍；**收缩维 K=token 太浅**（每专家 512），MMA 无法摊销访存。
- 证据：wgrad 计算 (SM) 吞吐仅 12%、DRAM 写≈D 大小、读≈写+操作数（累积读回）；dgrad tensor pipe
  65–76%、输出小且 bf16。两者都只达各自 roofline 下界的 55–63%，下界之比 ≈5 与实测吻合。
- 报告口径：单次 ≈ 持续 ×2（降频），比值稳定。
