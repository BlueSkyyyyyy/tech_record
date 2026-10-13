# Mega MoE 前向的 H100 数值复刻与误差分解

> 第 3/9/10/11 篇逐段精读了 SM100 的 `sm100_fp8_fp4_mega_moe.cuh` 主 kernel；第 41 篇拆了它的
> **端到端验收测试**（`tests/test_mega_moe.py`，逐位对比 + 性能账目）；第 4/12/13/14 篇把 **反向**
> 的四个 grouped GEMM 真的接进图里跑了一遍（`harness/exp_moe_backward_kernels.py`）。
> 但**前向**一直缺一块对应物：把 dispatch → linear1 → SwiGLU → linear2 → combine 这条流水线
> 用 H100 上**真实可跑**的算子逐段搭起来，量出「端到端误差是多少、每一段各贡献多少」。
>
> 本篇补上。**Mega MoE 主 kernel 是 SM100 专属、本机跑不了**；但它的**数学与量化语义架构无关**，
> 我们用 SM90 的 `m_grouped_fp8_gemm_nt_contiguous`（1D2D）与 `m_grouped_bf16_gemm_nt_contiguous`
> 逐段复刻，配 fp64/bf16/fp32 参考，做**误差来源分解**。所有数字来自
> `harness/exp_mega_moe_forward_emul.{py,sh,out.txt}`（H100，DeepGEMM 2.8.1）。

## 1. 前向数据流与「权重在哪乘」

Mega MoE 把五段融进一个 kernel。用真实行号把**数值语义**钉住（`impls/sm100_fp8_fp4_mega_moe.cuh`）：

1. **linear1**：`[m_e, H] @ W1_e^T [H, 2I]`，一次算出门（gate）与 up。UMMA 常量在 `:161-164`
   （`UMMA_M=LAYOUT_AD_M*2`、`UMMA_N=BLOCK_M`、`UMMA_BLOCK_K=128`、`UMMA_K=32`；A/B 互换）。
2. **SwiGLU（L1 epilogue，`:1066-1135`）**：gate/up 先各自 `__float22bfloat162_rn` **round 成 bf16**
   （`:1117-1118`），clamp（`:1121-1125`），`silu(gate)*up`（`:1127-1135`），**最后乘 top-k 权重**
   `activation_values = gate*up*weights`（`:1135`）。这里 `weights` 来自 `l1_topk_weights_buffer`，
   每 32 token 从显存读一次再做 warp 内 `ptx::exchange` 广播（`:1092-1101`）。
3. **L1 输出量化（`:1160-1192`）**：warp-pair 归约求 amax → `get_ue8m0_sf_exp` 得 UE8M0 指数、
   `get_ue8m0_sf_inv` 求逆 → `__fmul2_rn` 缩放 → `__nv_fp8x4_e4m3` 打包成 FP8，STSM 回 smem，
   SF 以 MN-major 写 `l2_sf_buffer`（`:1205-1213` 的 `transform_sf_token_idx`）。
   **这一步产出的就是 L2 的激活输入**。
4. **linear2**：`[m_e, I] @ W2_e^T [I, H]`。
5. **combine（`:1412-1519`）**：L2 epilogue 把 BF16 结果经 NVLink 直写远端 `combine_token_buffer`
   （`:1273/1311-1314/1346`）；combine 段对每个 token 的 top-k 槽（+共享槽）做 **`ptx::accumulate`
   的 fp32 累加**（`:1477-1490`）再 TMA 存回 `y`（`:1512-1515`）。**注意 combine 不再乘权重**——
   权重已在第 2 步进了激活。

**关键设计点**：top-k 权重被**乘在 L1 输出、L2 输入量化之前**。由于量化是 per-token 相对缩放，
先乘常量不改变相对误差，所以这在数值上是「免费」的；好处是把 combine 退化成纯求和（无权重账本）。

## 2. H100 复刻方案

主 kernel 跑不了，但每一段的算子在本机都有等价物：

| 段 | Mega MoE（SM100） | H100 复刻（SM90，真实 kernel） |
|---|---|---|
| dispatch | 自研 NVLink 前缀和 + round-robin | **跳过通信**，直接按 `(t,k)→expert` 建 m-grouped contiguous 布局 |
| linear1 | `SM100_MMA_MXF8F6F4_2x1SM_SS`，SF gran 32 | `m_grouped_fp8_gemm_nt_contiguous`（1D2D，激活 1D + 权重 128×128 2D） |
| SwiGLU | bf16 round + `silu*up` + ×w（`:1108-1135`） | torch：`silu(gate)*up*w`，gate/up 取自 kernel 的 **bf16 输出** |
| L2 激活量化 | UE8M0 per-token（`:1160-1192`） | `per_token_cast_to_fp8(gran_k=128, use_ue8m0=False)` |
| linear2 | 同 L1 | `m_grouped_fp8_gemm_nt_contiguous` |
| combine | NVLink 直写 + fp32 归约 | torch：把每行按 `(t,k)` 加回源 token |

两点必须讲清的**差异**（会体现在误差量级上）：① 真机 SM100 用 **gran 32**（`mega_moe.hpp:177-178`
断言 `rm==1 and rn==1 and rk==32` 且 `activation=="swiglu"`），而 SM90 1D2D 的 block-scale 粒度是
**128**（`sm90_fp8_gemm_1d2d.cuh:56` 的 `DG_STATIC_ASSERT(BLOCK_K == 128, "Only support
per-128-channel FP8 scaling")`）——本篇是数量级估算，不是逐位复现；② dispatch/combine 的**通信**被
替换成单卡 gather/scatter，故总量只反映「计算 + 量化」，不含 NVLink 误差。

对拍与分解三层（`harness/exp_mega_moe_forward_emul.py` 的 `run_case`）：

- `bf16-torch` / `fp32-torch`：同一条流水线的纯 torch 参考；
- `fp8-kernel`：真实 SM90 grouped FP8 kernel 搭出的前向；
- `deq` 参考：把喂进 kernel 的 FP8 操作数**反量化后在 fp32 重算**——`fp8-kernel` vs `deq`
  隔离 **kernel + 布局**误差（§4 的「分层」）；
- 消融：把 L1 激活 / L1 权重 / L2 激活 / L2 权重**选择性**替换成未量化原值，看谁主导（§6）。

## 3. 实验 A：小配置的误差分解

配置 `(T,H,I,E,topk)=(64,512,256,8,2) / (256,1024,512,8,2) / (512,1024,512,16,4)`：

```
T=  64 H=  512 I= 256 E= 8 topk=2 | m_total=  1024 (align 128) counts=[13,18,18,14,17,20,14,14]
   端到端 vs fp64:  fp8-kernel rel=6.2934e-02 cos=0.998031 | bf16-torch rel=5.1812e-03 | fp32-torch rel=5.1207e-04
   分层:  kernel+布局 L1 rel=1.4705e-03  L2 rel=1.4688e-03  端到端 vs 反量化参考 rel=3.2897e-04
   消融(反量化参考 vs fp64): none=1.5422e-03 仅激活=5.7187e-02 仅权重=4.5868e-02 两者=6.2940e-02
T= 256 H= 1024 I= 512 E= 8 topk=2 | m_total=  1024 (align 128) counts=[66,72,66,53,58,71,67,59]
   端到端 vs fp64:  fp8-kernel rel=6.3376e-02 cos=0.997991 | bf16-torch rel=5.1704e-03 | fp32-torch rel=5.1597e-04
   分层:  kernel+布局 L1 rel=1.4703e-03  L2 rel=1.4725e-03  端到端 vs 反量化参考 rel=3.2982e-04
   消融(反量化参考 vs fp64): none=1.5509e-03 仅激活=5.7590e-02 仅权重=4.6208e-02 两者=6.3378e-02
T= 512 H= 1024 I= 512 E=16 topk=4 | m_total=  3200 (align 128) counts=[109,114,136,...,115]
   端到端 vs fp64:  fp8-kernel rel=6.3381e-02 cos=0.997994 | bf16-torch rel=5.1817e-03 | fp32-torch rel=5.0911e-04
   分层:  kernel+布局 L1 rel=1.4691e-03  L2 rel=1.4723e-03  端到端 vs 反量化参考 rel=3.3212e-04
   消融(反量化参考 vs fp64): none=1.5476e-03 仅激活=5.7560e-02 仅权重=4.5998e-02 两者=6.3382e-02
```

读法：

- **kernel 本身是对的**：`fp8-kernel` 与「反量化后 fp32 重算」的端到端误差只有 **3.2e-4**，
  单层 L1/L2 各自 **1.47e-3**（这就是 bf16 输出的舍入地板，与第 27 篇的参考量化器对照一致）。
  也就是说 6.3e-2 里的 99% 来自**量化**，不是 kernel 或布局。
- **bf16 地板 vs fp8 地板**：纯 bf16-torch 端到端 5.2e-3、纯 fp32 5.1e-4，而 FP8 是 6.3e-2，
  相差约 12×。这与「E4M3 有效位 ~3 bit、per-token 相对缩放」的量级吻合。
- **误差对配置不敏感**：三个形状（含 E/topk 变化）都落在 **6.29–6.34e-2**，说明误差由**量化格式**
  决定、几乎与专家数/负载无关。

## 4. 实验 B：DeepSeek 量级 + 吞吐

配置 `T=2048, H=7168, I=2048, E=8, topk=2`（`m_total=4480`，每专家 ~470–670 token）：

```
   端到端 vs fp64: fp8-kernel rel=6.2832e-02 cos=0.998023 | bf16-torch rel=5.0388e-03
   分层: kernel+布局 L1 rel=1.4714e-03 L2 rel=1.4687e-03 | 端到端 vs 反量化参考 rel=3.2086e-04
   消融: none=1.5454e-03 仅激活=5.7037e-02 仅权重=4.5806e-02 两者=6.2833e-02
   吞吐: L1 m=4480 N=4096 K=7168 t=843us 312 TFLOPS | L2 N=7168 K=2048 t=434us 303 TFLOPS
```

- 端到端 **6.28e-2**、cos **0.998**，与实验 A 同量级 → 误差是格式属性而非规模属性。
- 两个前向 grouped GEMM 在 H100 上跑 **312 / 303 TFLOPS**（1D2D，m-grouped，G=8）。
  注意这与第 14 篇反向 dgrad 的 449/543 TFLOPS 不同——反向那两条的收缩维/输出维更大
  （N=7168 的 dX），而这里 L1 的 N=4096、L2 的 K 仅 2048，算术强度更低。**同一 kernel，
  形状决定一切**（选核见第 28 篇）。

## 5. 一个易踩的「吞吐陷阱」

首版本实验把 `run_grouped_fp8` 写成「每次调用 `torch.empty(d)` + 调 kernel + `torch.cuda.synchronize()`」，
于是测出 L1 只有 **54 TFLOPS**。原因不是 kernel：`_bench` 的循环里**每次都在分配输出并同步**，
把分配器/同步开销也算进了「吞吐」。改成预分配 `d` + 直接调 kernel 后是 **312 TFLOPS**（约 6×）。
这与第 40 篇 `bench_kineto` 强调「隔离发射/分配开销」是同一件事，记录在此以免误导。

## 6. 误差从哪来：消融的结论

消融（反量化参考里的哪几个操作数保持 FP8，其余用原值）给出很干净的结论：

| 量化对象 | 端到端 rel（vs fp64） |
|---|---|
| 不量化（纯 bf16 输出地板） | 1.55e-3 |
| **仅激活**量化 | 5.70e-2 |
| **仅权重**量化 | 4.58e-2 |
| 激活 + 权重都量化 | 6.28e-2 |

- **激活（per-token FP8）与权重（per-128 块 FP8）贡献同量级**（5.7e-2 vs 4.6e-2），
  两者叠加到 6.3e-2——不是某一侧单独主导。
- 这解释了为什么 Mega MoE 要做 **gran 32** 的细粒度激活缩放（比 SM90 的 gran 128 更细），
  以及为什么权重侧值得上 **packed UE8M0 + MN-major** 的块缩放（第 10/25 篇）：
  粗粒度块缩放是前向误差的主要来源，两者都要压。
- 「不量化」的 1.55e-3 就是 L2 输出 bf16 舍入的地板（与 §3 的单层 1.47e-3 一致）。

## 7. H100 实测证据台账

| 项 | 数字 |
|---|---|
| 端到端 fp8-kernel vs fp64（4 组形状） | rel 6.28–6.34e-2，cos ≥ 0.9980 |
| bf16-torch vs fp64 | rel 5.04–5.18e-3 |
| fp32-torch vs fp64 | rel 5.09–5.16e-4 |
| 分层 kernel+布局（L1 / L2） | rel 1.469–1.471e-3（bf16 输出地板） |
| 端到端 vs 反量化参考 | rel 3.21–3.33e-4 |
| 消融：仅激活 / 仅权重 / 两者 | 5.70e-2 / 4.58e-2 / 6.28e-2 |
| 前向吞吐 L1 / L2（DeepSeek 量级） | 312 / 303 TFLOPS；843 / 434 us |

原始输出：`harness/exp_mega_moe_forward_emul.out.txt`。

## 8. 边界与「仅分析」

- **SM100 专属、仅分析**：`sm100_fp8_fp4_mega_moe.cuh` 主 kernel 本体、`UMMA`/`UTCCP`/`TMEM`、
  packed FP4 路径、gran 32 的真实 SF 粒度、dispatch/combine 的 NVLink 通信。本篇的 6.3e-2 是
  **H100 gran 128** 的估算；真机 gran 32 + 访存融合的实际误差可能更小，需目标卡复测。
- **H100 真跑**：SM90 `m_grouped_fp8_gemm_nt_contiguous`（1D2D）、`per_token_cast_to_fp8` /
  `per_block_cast_to_fp8` 参考量化器（第 27 篇）、m-grouped contiguous 布局（第 13 篇）、
  以及本实验搭出的整条前向与三层对拍。
- 复刻把**通信折算成单卡 gather/scatter**，故不含 NVLink 带来的数值影响（通信是精确的搬运，
  不引入舍入）；重叠/带宽见第 2/11 篇与 `harness/exp_nvlink_access`。
