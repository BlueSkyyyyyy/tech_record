---
title: "CUDA 算子调优（三十二）：RMSNorm / QK-Norm 与残差、FP8 量化的融合 —— 一个纯访存算子怎么贴到 87% HBM"
date: 2026-09-21
draft: false
weight: 32
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "RMSNorm", "QK-Norm", "LayerNorm", "FP8", "量化", "残差", "DeepSeek-V4", "Qwen3", "GLM", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 24 篇]({{< relref "cuda-kernel-opt-24-fp8-gemm-pb" >}})把 DeepSeek-V4 的 per-block FP8 缩放做进了 GEMM，
[第 27 篇]({{< relref "cuda-kernel-opt-27-moe-router" >}})则说明 MoE 前门里「搬运」才是大头。这一篇换一个
**纯访存、没有任何 Tensor Core** 的模型算子：**RMSNorm 家族**——Transformer 每一层都会跑、
DeepSeek-V4 / Qwen3 / GLM 都用、而且在 FP8 部署里还常常和「残差相加」「动态量化」绑在一起。

它是最好的 roofline 练手题：算术强度趋近于 0，**唯一的目标就是把 HBM 打满**。系列北极星要求
内存受限算子 ≥ 90~95% HBM，这篇就从 Python 级的朴素写法一路推到 **87% HBM**，并把踩过的
FP8 工具链坑记下来。

shape 全部取自本地模型 config：

| 模型 | hidden | heads | head_dim | eps |
|---|---|---|---|---|
| DeepSeek-V4-Pro | 7168 | 128（kv=1，MLA） | 512（qk_rope=64） | 1e-6 |
| Qwen3-8B | 5120 | 40 q / 8 kv | 128 | 1e-6 |
| GLM-5.2-FP8 | 6144 | 64 | 192（v=256） | 1e-5 |

实测用 `M=8192` 行、`H=7168`（DeepSeek-V4-Pro），QK-Norm 用 Qwen3 的 `40 q / 8 kv × 128`。

## RMSNorm 与「融合链」

对一行 `x ∈ R^H`、权重 `w ∈ R^H`：

$$
\mathrm{RMS}(x)=\sqrt{\tfrac1H\sum_{i=1}^{H}x_i^2},\qquad
y_i=\frac{x_i}{\sqrt{\mathrm{RMS}(x)^2+\epsilon}}\,w_i .
$$

Transformer block 里它从来不单独出现，而是和邻居粘在一起：

```text
不融合（每层 3 个 kernel）：
  s = x + residual            # add：读 x,res 写 s
  y = rmsnorm(s) * w          # norm：读 s 写 y
  yq, scale = quant_fp8(y)    # quant（FP8 部署）：读 y 写 yq(1B)+scale

融合（1 个 kernel）：
  s = x + residual;  y = rmsnorm(s)*w;  yq = fp8(y)（per-128 动态 scale）
```

融合省的不是 FLOPs（RMSNorm 只有 `3H` 次浮点运算），而是**中间张量的读写**。先算笔账（每元素字节数）：

| 版本 | 读 | 写 | 合计 |
|---|---|---|---|
| 不融合 add+norm | x, res(2+2) | s, y(2+2) | 8 B |
| **融合 add+norm（v2）** | x, res(4) | s, y(4) | 8 B？ |
| 融合 add+norm+fp8（v3） | x, res(4) | s(2), yq(1) | **7 B** |
| 单独 rmsnorm（v1） | x(2) | y(2) | 4 B |

等一下——融合 add+norm 的字节数和不融合一样？是的。不融合是 `add`（读 2 写 1 = 3 个 pass）+ `norm`（读 1 写 1 = 2 个 pass）= 5 个 pass；融合后是 2 读 + 2 写 = **4 个 pass**，省掉的是 `s` 被第二次读取的那 1 个 pass。所以「融合收益」= 少读一次中间张量。

## 实测：四步把 44% 推到 87%

先给结论（H100 SXM，`M=8192, H=7168`，`2·M·H` 字节为有效读+写，带宽换算 `bytes/time`）：

| 版本 | 做法 | 时间 (ms) | 有效带宽 (GB/s) | % HBM | ncu DRAM% |
|---|---|---|---|---|---|
| v0 | 朴素：2 趟 global、无 smem | 0.1586 | 1481 | 44.2% | — |
| v1 | 行缓存 smem、全局只读一遍 | 0.1006 | 2334 | 69.6% | — |
| v1r | **寄存器缓存**、不落 smem | **0.0819** | **2869** | **85.6%** | 82.9% |
| v2 | 融合 add+rmsnorm（smem） | 0.1644 | 2858 | 85.2% | — |
| **v2r** | **融合 add+rmsnorm（寄存器）** | **0.1607** | **2923** | **87.2%** | **86.2%** |
| v3 | +FP8 per-128 量化（smem） | 0.1894 | 2170 | 64.7% | — |
| v3r | +FP8 per-128 量化（寄存器） | 0.1864 | 2205 | 65.8% | 64.8% |
| v4 | QK-Norm（Qwen3 40/8×128） | 0.0462 | 2179 | 65.0% | — |

`2·M·H` 口径下，v2r 的 87.2% 已接近「一个读两个写」的物理极限。

### v0 → v1：无 smem 的朴素版为什么只有 44%

`rmsnorm_v0_kernel` 每行一个 block，**读两遍 x**：第一遍求 `Σx²`，第二遍归一化。它没有 smem 缓存，
第二遍要么命中 L2（7168×2B=14 KB/行，8192 行 = 117 MB，装不进 50 MB L2），要么重新回 HBM。
于是有效流量实际是 `3·M·H·2` 而不是 `2·M·H·2`——**有效带宽肯定虚高**，但真实 HBM 利用率只有 44%。

### v1：行缓存进 smem

把一行 14 KB 缓存进共享内存，全局只读一遍、写一遍。速度到 69.6%。但 ncu 显示
`l1tex__throughput = 76%`、`sm__throughput = 31.5%`——**L1/TEX（含 smem 的读写）成了新瓶颈**：
每元素要在 smem 里「写一次 + 读一次」，L1 吞吐翻倍。

### v1r：把整行留在寄存器里

`H/8 = 896` 个 16B 向量，`T=256` 个线程每线程 `ceil(896/256)=4` 个 `uint4`。既然一次读进来，
**为什么不直接放寄存器、归约完就地归一化**？于是 smem 往返整段消失：

```text
for g in 0..NG-1:  v[g] = load_uint4(x[row, c8(g)]);  ss += Σ v[g]²   // 留寄存器
block-reduce(ss) -> inv = rsqrt(ss/H + eps)
for g:             y[row, c8] = bf16(v[g] * inv * w)                  // 就地写回
```

ncu：`l1tex` 从 76% 掉到 53.6%，DRAM 从 44% 抬到 **82.9%**，带宽 2334 → 2869 GB/s。
寄存器只用了 48 个，occupancy 不受限。**这是本篇最大的单步收益（+16 个百分点 / +23% 带宽）。**

### v2 / v2r：融合残差相加

`y = rmsnorm(x + res)`。若分两个 kernel 写，`s = x+res` 会被 norm kernel 再读一遍（多 1 个 pass）。
融合后每线程一次读 `x`、`res` 两个 `uint4`，算出 `s` 后既写回全局（给下一层当残差）、又留在寄存器里
做 rmsnorm。`v2r` 达 **87.2% HBM / ncu DRAM 86.2%**——此时 L2 利用率 79.7% 反超 DRAM，是下一道墙。

### v3 / v3r：再融合 FP8 动态量化

DeepSeek 的 FP8 激活量化是 **1×128 的动态 block scale**：每 128 个元素算 `amax`，
`scale = amax/448`，`yq = e4m3(y/scale)`。把这一步也融进来，输出从 2B/元素降到 1B。
但实测只有 65.8%，**不如不融合**。ncu 说出了原因：`sm__throughput = 50.2%`、`l1tex = 36.7%`、
DRAM 只有 64.8%——**瓶颈从 HBM 转移到了「算 amax 的 shared atomicMax + 逐字节拼 fp8」**。

每行 56 个 128-block，512 个线程对 56 个 `amax` 槽位做 `atomicMax`，竞争严重；而且 `e4m3` 是 1 字节，
凑 8 个要逐字节拼 `uint2`。它换来的只是 8B→7B（−12.5%）的流量，性价比不高。
**结论：FP8 量化融合在「输出字节占比大」时才划算；RMSNorm 的输出本来就不大，省不出钱。**

## ncu 定位：瓶颈在 L1 还是在 HBM

| kernel | DRAM% | L1TEX% | L2% | SM% | 寄存器 | 瓶颈 |
|---|---|---|---|---|---|---|
| `rmsnorm_v1r` | 82.9 | 53.6 | 79.5 | 31.4 | 48 | L2 |
| `add_rmsnorm_v2r` | **86.2** | 31.6 | 79.7 | 19.5 | 48 | HBM/L2 |
| `add_rmsnorm_fp8_v3r` | 64.8 | 36.7 | 63.5 | 50.2 | 61 | 计算（amax/打包） |

判据很清楚：**`DRAM Throughput` 才是 HBM 利用率**（见第 03/10 篇的坑）。
v1 时代 `l1tex` 76% 掩盖了 DRAM 只有 44%；把数据留在寄存器把 L1 让出来，DRAM 才真正上去。

## 踩坑：这个工具链下 `float(fp8)` 是坏的

写 v3 时出现一个极隐蔽的坑：量化后的误差稳定在 **155%**，但 kernel 逻辑看着没错。定位过程：

```cpp
// fp8_test.cu：把 float 转 e4m3 再转回来
1.000 -> 56.000      // 期望 1.0
1.130 -> 56.000
448.000 -> 128.000
```

`float(fp8_value)` 返回的竟是**原始 1 字节位模式**（`1.0` 的 e4m3 位模式是 `0x38 = 56`），
不是数值。正确路径是显式走存储类型 + `__nv_cvt_fp8_to_halfraw`：

```cpp
__nv_fp8_storage_t s = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
float back = __half2float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));   // 1.0 -> 1.0
```

同理，**不要写 `fp8 out = fp8(storage_byte)` 再存**（重载会选错把 1 字节当 float 用），
直接把 `__nv_fp8_storage_t`（`uint8`）写进输出字节流。改完误差立刻从 155% 降到 **3.31%**
（e4m3 的正常量化误差）。复现见 `32-fused-norm/fp8_test.cu` 与 `fp8_test2.cu`。

另一个小坑：`__shared__` 里对 `amax` 做 `atomicMax` 时，float 要用
`atomicMax((int*)&a, __float_as_int(fabsf(v)))`——**非负数**的 IEEE 位模式按整数比大小仍然保序。

## QK-Norm

Qwen3 / GLM 对 q、k 的每个 head 在 `head_dim` 上单独做 RMSNorm（`head_dim=128`，权重长度 128）。
实现是「一个 warp 管一个 (token, head)」：`128` 个元素 / 32 lane = 每 lane 4 个，一次 `uint2` 载入，
`__shfl_down_sync` 归约出 `Σx²`，再归一化写回。实测 `M=4096` 时 `(q40 + k8)` 共 50 MB 流量、
**2179 GB/s（65% HBM）**。

它比 v1r 低，是因为每个 warp 只搬 256 B、ILP 太低（一次载入就进归约，没有第二个独立访存流，
延迟藏不住）。若要再往上推，应让一个 warp 串行处理多个 head（把 4~8 个 head 的载入先发出去）。

## 小结

- **纯访存算子的优化顺序是「先减 pass、再消 L1、最后才谈指令」**：44% → 70%（smem 单读）→ 86%
  （寄存器缓存）→ 87%（融合残差）。
- **行能放进寄存器就别放 smem**：RMSNorm 每行 14 KB 在这台机器上 smem 够用，但 smem 的往返会
  把 L1/TEX 顶到 76%，成为比 HBM 更早到的墙。用 `NG = ceil((H/8)/T)` 个 `uint4` 留寄存器即可，
  代价只有 ~16 个寄存器。
- **融合的收益只看「省掉几个 pass」**：add+norm 省 1 个（5→4），FP8 量化只省 0.5 个（8→7），
  却引入了 amax 归约与逐字节打包——**不划算**。这也解释了为什么 27 篇里 permute/unpermute 值得融，
  而这里的 quant 不值得。
- **代码不能只看逻辑对**：`float(fp8)` 这种「看着是标准库」的转换在这个 CUDA 工具链下直接返回位模式，
  害得误差稳定停在 155%。**任何精度类算子，先用一个 1+1 的最小样例核对转换本身**。

## 下一篇

第五/六部分还剩几个高价值目标：MLA 的 MN-major 免转置已被实测证伪（Hopper 的 `wgmma` **B 操作数
只认 K-major**，LBO 字段被硬件忽略，见 TECHNIQUES 的 J4），后续会把 MLA 极限冲刺的结论单独成文；
MoE 侧继续把 per-block FP8 叠进 fused FFN；此外 DSA 的 compressor、paged KV/flash-decoding 也在队列里。

---

配套代码：[`code/kernel-opt/32-fused-norm/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/kernel-opt/32-fused-norm)
（`norm_fused.cu` 八个变体 + `fp8_test*.cu` 最小复现）。
