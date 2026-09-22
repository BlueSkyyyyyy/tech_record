---
title: "CUDA 算子调优（五十六）：DeepSeek-V4 的路由专家权重其实是 FP4 —— 真实 checkpoint 的格式、无损折进 FP8，以及 decode GEMV 的实测"
date: 2026-09-22T08:00:00+08:00
draft: false
weight: 56
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "FP4", "e2m1", "E8M0", "FP8", "e4m3", "DeepSeek-V4", "量化", "GEMV", "dp4a", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 34 篇]({{< relref "cuda-kernel-opt-34-fused-moe-fp8" >}}) 把 fused MoE expert FFN 做成了
per-block FP8（`e4m3 + ue8m0 + weight_block 128×128`），端到端相对 bf16 快 1.9~2.2×、专家权重
从 50.7 GB 压到 25.4 GB。那一篇（以及 31 / 36 / 37 篇）都**默认专家的权重是 FP8**，
因为 `config.json` 的 `quantization_config` 写着 `fmt: e4m3 / scale_fmt: ue8m0 / weight_block_size: [128,128]`。

直到我把真实 checkpoint 的 tensor 头列了出来：

```text
layers.0.ffn.experts.0.w1.weight   [3072, 3584]   I8       ←?? hidden 明明是 7168
layers.0.ffn.experts.0.w1.scale    [3072,  224]   F8_E8M0
```

`7168/2 = 3584`、`7168/32 = 224`——这不是 FP8，这是 **FP4**：每字节塞 **两个** `e2m1`
（2 位指数 + 1 位尾数，取值 $\{0,\pm0.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6\}$），每 **32 个 K**
一个 `E8M0`（纯 2 的幂）block scale。翻 `config.json` 才看到那一行之前被我忽略了：

```json
"expert_dtype": "fp4"
```

一句话结论先放这：

> **DeepSeek-V4-Pro 的路由专家（routed experts）是 FP4 e2m1 + E8M0 block-32；共享专家、注意力
> 仍是 FP8 e4m3 block-128；lm_head 是 bf16。** 因为 `e2m1 ⊂ e4m3` 且 scale 是 2 的幂，
> **FP4 可以逐位无损地折进 e4m3**（实测相对误差 $0$）。它意味着 34 篇的 fp8 grouped kernel
> 只要在 host 侧做一次无损预折叠就能原样消费 FP4 专家，而**专家权重显存/传输再砍一半**。

这一篇做三件事：**① 把真实 checkpoint 的格式钉死；② 证明 FP4→FP8 折叠是无损的；
③ 写一个读真实 FP4 权重的 decode GEMV，实测「现场解码」在 Hopper 上到底要付多少代价**
（结论：算术解码是 compute-bound，换成 smem LUT 后 2× 提速、并反超 fp8 权重路径）。

---

## 一、真实 checkpoint 的 dtype 全景

用 `safetensors` 的懒加载把 layer 0 的各类权重头列出来（`extract.py` 里可复现）：

| 张量 | 形状 | dtype | scale | 说明 |
|---|---|---|---|---|
| `ffn.experts.{0..383}.w1.weight` | `[3072, 3584]` | `I8` | `[3072, 224]` `F8_E8M0` | **FP4 e2m1**，block 32，两 nibble/byte |
| `ffn.experts.*.w3.weight` | `[3072, 3584]` | `I8` | `[3072, 224]` `F8_E8M0` | 同上（gate） |
| `ffn.experts.*.w2.weight` | `[7168, 1536]` | `I8` | `[7168, 96]` `F8_E8M0` | down，FP4 |
| `ffn.shared_experts.w1.weight` | `[3072, 7168]` | `F8_E4M3` | `[24, 56]` `F8_E8M0` | 共享专家：FP8 block 128 |
| `attn.wq_b.weight` | `[65536, 1536]` | `F8_E4M3` | `[512, 12]` `F8_E8M0` | 注意力：FP8 block 128 |
| `head.weight`（lm_head） | `[129280, 7168]` | `BF16` | — | 见 55 篇 |

**同一个 MoE 层里，路由专家用 4 bit、共享专家用 8 bit**——这是有意的：路由专家有 384 个、
占了整网权重的绝大部分（约 **822 GB / 883 GB**），必须最激进地压；共享专家每个 token 都过，
对精度更敏感，留在 FP8；lm_head 直接进 CE，留在 bf16。

### 1.1 解码验证

`w1.weight` 的每个 byte 是两个 `e2m1`：低 nibble 是偶数列、高 nibble 是奇数列。
按 `e2m1` 表解码、逐 32 列乘上 `E8M0` scale，得到真实权重。两个实测事实：

- **scale 全是 2 的幂**：`log2(scale)` 与最近整数的最大偏差 = **0.0**（scale 取值 $2^{-8}\sim 2^{-5}$）；
- 单行权重的 rms $\approx 2.49\times10^{-2}$，量级正常。

`E8M0` 是「无符号 8 位纯指数、bias 127」：`value = 2^(byte-127)`。

---

## 二、为什么 FP4 能**无损**折进 FP8

这是整篇最有价值的一条。`e2m1` 的 8 个幅值
$$\{0,\tfrac12,1,\tfrac32,2,3,4,6\}$$
**全部**能被 `e4m3` 精确表示（`e4m3` 有 3 位尾数，`1.5 = 1.100b`、`3 = 1.5\cdot2`、`6 = 1.5\cdot4`）。
再把 $2^{e}$（$e\in[-8,-5]$）乘上去，只是**指数平移**，尾数一位不丢；最小值 $\tfrac12\cdot2^{-8}=2^{-9}$
恰好是 `e4m3` 的最小次正规数。于是

$$W_{\text{fp8}} = \mathrm{e4m3}\big(W_{\text{fp4}}\cdot 2^{e}\big) \quad\text{逐位无损。}$$

用真实权重验证（`extract.py`，把 FP4 解码后转回 FP8 再反量化）：

```text
w1: packed(3072, 3584) scale(3072, 224)  fp8_fold_relRMS=0.00e+00  maxabs=0.00e+00
w3: packed(3072, 3584) scale(3072, 224)  fp8_fold_relRMS=0.00e+00  maxabs=0.00e+00
w2: packed(7168, 1536) scale(7168,  96)  fp8_fold_relRMS=0.00e+00  maxabs=0.00e+00
```

**相对误差、最大绝对误差都是 0**。理论上必然如此，实测再次确认。

> 这条和 [第 37 篇]({{< relref "cuda-kernel-opt-37-fp8-pb-prescale" >}}) 的「`ue8m0` 是 2 的幂
> $\Rightarrow$ 可把 scale 精确折进 e4m3 操作数」是同一件事。37 篇用它把 **per-block FP8 退化成
> per-tensor GEMM**；这里用它把 **FP4 折算成 FP8**——两次都靠「2 的幂只会平移指数」。

### 2.1 存储账

一个 expert（w1+w3+w2）：

| 格式 | 每 expert | 384 experts × 61 层 |
|---|---|---|
| bf16（未量化） | 132.1 MB | 3094.8 GB |
| FP8 e4m3 | 66.1 MB | 1547.4 GB |
| **FP4 e2m1** | **33.0 MB**（+scale 2.1 MB） | **773.7 GB + 48.4 GB** |

FP4 是 bf16 的 **25%**、是 FP8 的 **50%**（含 scale 约 53%）。对 MoE 的 **all-to-all / 专家换入换出**
来说，这是实打实的 2×——而且因为第 2 节的无损性，**推理时零精度损失、零额外运行时改写**
（host 侧一次性折叠，或者直接让 GEMM 消费折好的 FP8）。

问题是：如果想省掉那一次性折叠、直接**在现场读 FP4 并解码**，代价多大？下面写 kernel 量一量。

---

## 三、decode GEMV：现场解码 FP4 值不值

形状取真实 expert 的 up 投影：`w1: [N=3072, K=7168]`，decode（`M=1`）时这是**权重带宽受限**的
GEMV。实现（`fp4_gemv.cu`）：一 warp 若干行，warp 内沿 K 并行；激活量化成 int8（per-128 动态），
权重 nibble 解码成 int8（注意把 `e2m1` 值 **×2 整数化**，$\{0,\pm1,\pm2,\pm3,\pm4,\pm6,\pm8,\pm12\}$，
正好落进 int8），用 `__dp4a` 一条指令做 4 个 MAC；每 32-K 的 `E8M0` scale 折成 float 在点积后乘回：

$$y_n = \sum_{c}\underbrace{2^{b_c-127}}_{\text{block-32 scale}}\cdot\tfrac12\underbrace{\sum_{k\in c} (2w_{k})\,q^x_k}_{\texttt{dp4a}}.$$

（×2 整数化后要补一个 $\tfrac12$；激活 scale 是 per-128。）

### 3.1 v1：算术解码 → compute-bound

第一版把 nibble 逐位算术解码：`e=(n>>1)&3, m=n&1`，
$v'=((e\,?\,2\!:\!0)+m)\ll(e\,?\,e{-}1\!:\!0)$。32 个权重要 ~7 条 ALU，**比 int4 的
`__vsubss4` 偏移技巧（~1.25 条/权重）贵 5 倍**。ncu 直接判了死刑：

```text
[gemv fp4 v1]  RWW=1 BN=8   0.0199 ms   587.6 GB/s (17.5% HBM)
ncu: Compute(SM) 65.5% | DRAM 19.6% | IPC 1.95 | 40 regs | occ 27%
```

**Compute(SM) 65.5%、DRAM 只有 19.6%**——访存很闲，全卡在解码的 ALU 发射口。

### 3.2 v2：smem LUT 解码 → 2× 提速

把「一个输入 byte（两个 nibble）→ 两个 int8」预存成一张 **256 项 `uint16` LUT**（512 B，
`__shared__`），解码从 ~7 条/nibble 降到 **一次 smem 读/byte**：

```text
[gemv fp4 v2]  RWW=1 BN=8   0.0100 ms  1165.6 GB/s (34.8% HBM)
ncu: Memory 52.7% | DRAM 37.6% | Compute 20.5% | IPC 1.11 | 38 regs | occ 32.8%
```

**0.0199 → 0.0100 ms，1.99×**；瓶颈从 `Compute 65.5%` 翻到访存/延迟（`Compute` 掉到 20.5%）。

### 3.3 和「34 篇假设的 FP8 权重」对拍

同样形状、同样 GEMV，把权重换成 1 byte/elem 的 FP8（第 2 节无损折叠的结果，无 scale）：

| 权重格式 | 字节/元素 | 最佳时间 | 有效权重带宽 | 相对 |
|---|---|---|---|---|
| FP4 e2m1 + LUT 解码 | 0.5 | **0.0100 ms** | 1165.6 GB/s | **1.00×** |
| FP8 e4m3（34 篇假设） | 1.0 | 0.0216 ms | 1017.2 GB/s | 2.16× |
| bf16 | 2.0 | — | — | ~4× |

**FP4 路径比 FP8 路径快 2.16×**——尽管它多背了一个解码步骤，字节减半还是把它拉回来了。
（注：单 expert 权重 11 MB < 50 MB L2，纯读 roof 也就 86~90%「HBM」，其实是 L2 带宽；
这里的对比是**同量级 footprint、同一读路径**下的，结论是相对关系。）

---

## 四、ncu：下一堵墙是 LUT 的 bank conflict

v2 的 stall 指纹（`ncu_lut.out.txt`）：

```text
long_scoreboard   5.96   ← 等权重 global load
short_scoreboard  4.54   ← 等 smem LUT（随机的 2B 读撞 bank）
mio_throttle      2.84
```

`L1 Wavefronts Shared Excessive` 报「**809012 条多余 wavefront，占 59%**」——LUT 的随机
`uint16` 访问每两个相邻 nibble 落进同一个 32-bit bank。下一步该做的（本轮的「下一堵墙」）：
把 LUT 换成**免冲突布局**（按 nibble 的 4-bit 值重排、或每 lane 私有副本），
或用 [第 48/50/52 篇]({{< relref "cuda-kernel-opt-52-w4a8-imma-tma" >}}) 那套「K-blocked 权重布局 + TMA」
把解码摊到更多线程上。目标是把 35% 推向 70%+。

> 这和 44/45 篇的结论一脉相承：**低比特 decode 的真账不在像素带宽，在「解码的 ALU / smem 延迟」**。
> FP4 比 int4 更难解（`e2m1` 不是简单的偏置整数），所以 LUT 这一步比 int4 更值钱（那里 `__vsubss4`
> 就够了，这里必须上表）。

---

## 五、对模型的意义

1. **存量算法无需重写**：34 篇的 FP8 grouped kernel 消费的是「FP8 权重 + `ue8m0` block scale」。
   只要 host 侧把 FP4 **无损预折叠**成 FP8，那个 kernel 原样可用，且专家权重显存/传输减半。
   代价是推理前一次性折算（相对整个 decode 可以忽略）。
2. **现场解码只在「不想预折叠」时才有账**：Hopper 没有原生 FP4 张量核，只能软件解码；
   实测 LUT 解码已经把字节优势兑现成 2.16×（相对 fp8）——但那是 GEMV（权重带宽受限）的账。
   prefill 的 grouped GEMM 里每个权重 tile 会被 $B_M$ 个 token 复用，**解码 ALU 被摊薄**，
   FP4 的收益会更接近纯字节比（~2×）；反过来 decode 时收益取决于解码多便宜。
3. **Blackwell 才是 FP4 的主场**：`tcgen05` 原生吃 FP4，解码税归零，字节优势直接兑现。
   这也是 V4 把路由专家压到 FP4 的底气——目标硬件本就该有 FP4 张量核。

### 五、1 与 SOTA 的差距

| 阶段 | 实测 | 同口径上界 | 差距 |
|---|---|---|---|
| FP4 GEMV（LUT 解码） | 1165.6 GB/s | 纯读 roof ~2886 GB/s | 2.5× |
| FP4→FP8 无损折叠 | 0 误差 | — | 无损 |
| 存储 vs FP8 | 50%+scale | — | 2× 省 |

---

## 小结

- 真实 DeepSeek-V4-Pro 的 **routed expert 是 FP4 e2m1 + E8M0 block-32**（不是 34 篇假设的 FP8）；
  共享专家/注意力是 FP8，lm_head 是 bf16。全网路由专家从 bf16 的 3095 GB 压到 **773.7 GB**。
- **FP4→FP8 折叠逐位无损**（实测 relRMS $=0$）：`e2m1 ⊂ e4m3` 且 scale 是 2 的幂。
  这让存量 FP8 算子零成本消费 FP4 专家，同时省一半显存/传输。
- 现场解码：算术解码是 **compute-bound（65.5%）**，换 smem LUT 后 **2× 提速**，
  并比 fp8 权重路径 **快 2.16×**；下一堵墙是 LUT 的 shared bank conflict（59% 多余 wavefront）。
- 和 37 篇同源：**「2 的幂 scale 可以精确平移指数」** 是低比特量化里反复出现的免费午餐。

## 复现

```bash
cd code/kernel-opt
python3 56-v4-fp4-moe/extract.py 0 0          # 从真实 checkpoint 取 expert 0 的 w1/w2/w3
scripts/run.sh 56-v4-fp4-moe/fp4_gemv.cu 1 all
scripts/ncu.sh 56-v4-fp4-moe/fp4_gemv.cu --set full \
  --kernel-name regex:gemv_fp4_kernel --launch-skip 1 --launch-count 1 -- 1 ncu
```

代码：`code/kernel-opt/56-v4-fp4-moe/`（`extract.py` 真实权重解码 + 无损折叠验证；
`fp4_gemv.cu` 的 `gemv_fp4_kernel`（v1 算术 / v2 smem LUT）与 `gemv_fp8_kernel` 对照；
`real_weight_stats.out.txt`、`fp4_gemv_M1.out.txt`、`ncu_lut.out.txt`、`ncu_arith.out.txt`）。

## 下一篇

- **主题 21h**：把 FP4 无损折叠后的权重接回 [34 篇]({{< relref "cuda-kernel-opt-34-fused-moe-fp8" >}})
  的 grouped kernel，量 prefill 端到端（预期 ~1.5~1.9×）；
- 或 **主题 26m**：把 LUT 换成免冲突布局 / IMMA，把 FP4 decode 从 35% 推向 70%+。
