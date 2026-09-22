---
title: "CUDA 算子调优（五十七）：FP4 专家权重接回 grouped MoE —— 现场折 vs 预折叠，与一份被「变换税」吃掉的字节红利"
date: 2026-09-22T08:00:00+08:00
draft: false
weight: 57
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "DeepSeek-V4", "FP4", "e2m1", "E8M0", "FP8", "e4m3", "量化", "grouped GEMM", "wgmma", "SW128", "TMA", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 56 篇]({{< relref "cuda-kernel-opt-56-v4-fp4-moe" >}}) 从真实 checkpoint 里钉死了一件事：
**DeepSeek-V4-Pro 的路由专家（routed experts）权重是 FP4**（`e2m1` + `E8M0` block-32），
并且在 host 侧无损折进 e4m3 后 `relRMS = 0`（逐位相等）。而 [第 34 篇]({{< relref "cuda-kernel-opt-34-fused-moe-fp8" >}})
的 fused MoE expert FFN 端到端 kernel **默认权重是 FP8**（`e4m3 + ue8m0 + weight_block 128×128`），
它比 bf16 快 1.9~2.2×、权重从 50.7 GB 压到 25.4 GB。

两篇合起来自然有一个问题：

> **既然真实权重只有 FP8 的一半字节（fp4 8.46 GB vs fp8 16.91 GB，全部 384 个专家），
> 那么直接把 FP4 喂给 grouped kernel、在 kernel 里现场折成 FP8，能不能把 prefill 的
> 权重带宽真正砍半、换来 ~1.5~1.9×？**

这一篇给出实测答案，而且是一份**「半否决」**：折叠本身完全无损（现场折的结果与预折叠**逐位相同**），
但在这套实现下 **现场折反而慢 1.5×**——因为 SW128 置换 + FP4 折叠的**变换税**（~5.3 ms 的
SM 发射时间）远大于省下的内存字节（~2.4 ms）。ncu 一句话钉死：**Compute (SM) 95%，DRAM 只有 47%**。

先给全系列目录：见 [01 篇]({{< relref "cuda-kernel-opt-01-overview" >}}) 开头的索引。

---

## 1. 背景：问题的一句话版

设一个 expert 的 up/gate 权重（V4-Pro：`hidden=7168`，`moe_intermediate=3072`，所以
up/gate 输出维 `N=2I=6144`）：

- 存 FP8：`N × K = 6144 × 7168 = 44.0 MB` / expert；
- 存 FP4：`N × K/2 = 22.0 MB` + `E8M0` scale `N × K/32 = 1.38 MB` = **23.4 MB** / expert。

384 个专家、up/gate 一层：**fp8 16.91 GB vs fp4 8.98 GB**（含 scale）。按 H100 HBM 3.35 TB/s，
纯读时间分别是 `5.04 ms` 与 `2.68 ms`——**理论上 FP4 快 1.88×**。

所以问题被简化成：**能不能在不付出额外代价的前提下，把 FP4 变成 wgmma 能吃的 FP8？**

---

## 2. 两个路径的设计

我把整个 grouped up/gate GEMM（`D[p,n] = Σ_k sa[p,k/128]·A[p,k]·B[g,n,k]`，
A 是 fp8 激活、SA 是 per-1×128 动态 scale）做成一个 kernel，
只差 B（权重）的**格式与搬运**：

```text
MODE 0  预折叠 (pre-fold)            MODE 1  现场折 (on-the-fly)
────────────────────────────         ────────────────────────────
global:  W8 [E*N, K] fp8             global:  Wp [E*N, K/2] u8 packed fp4
         (host 一次性 fold)                   WsT[K/32, E*N] u8 E8M0 (转置)
   │ cp.async 16B/thread                │ cp.async 16B/thread
   ▼                                    ▼
smem raw [BN, K] fp8                 smem raw [BN, K/2] fp4 + scale
   │ SW128 置换 (16B chunk)             │ 4096 项 LUT: (exp,nibble)->e4m3
   ▼                                    ▼
Bs [BN, K] SW128 fp8   ────► wgmma ◄──── Bs [BN, K] SW128 fp8
```

两条路径的 **consumer（wgmma + per-128 的 `sa` 折算）逐字节相同**，A 也走同一套
cp.async + 手写 SW128。于是耗时差 = 「读 FP4 半字节 + 现场折」的净效应。

### 2.1 SW128 置换在做什么

wgmma SS 要求操作数在 smem 里是 **K-major SW128** 布局。对 fp8，一个 1024B atom = 8 行 × 128B，
逻辑 `(row, k)` 的字节偏移是

$$
\text{off}(r,k) = \Big\lfloor \tfrac{r}{8}\Big\rfloor\cdot 1024
+ \big(r\!\bmod 8\big)\cdot 128
+ \Big(\big(\lfloor \tfrac{k}{16}\rfloor \bmod 8\big) \oplus (r\!\bmod 8)\Big)\cdot 16
+ (k\!\bmod 16)
$$

也就是：**128B 行内的 8 个 16B chunk 按行号异或**。这不是数据重排，只是把 16B chunk 换个位置——
所以可以 **16B 粒度**向量化搬：读 `uint4`、写 `uint4`，一条指令搬 16 个元素。
（[第 20 篇]({{< relref "cuda-kernel-opt-20-mla-wgmma-sw128" >}}) 早已给出这个公式；当时是给 MLA 用。）

### 2.2 现场折：4096 项 LUT

把 `e2m1` 折进 `e4m3` 是逐位无损的，但要对每个元素做「查表（e2m1 值）× 2 的幂（scale）→ e4m3」。
直接 `exp2f + __nv_cvt_float_to_fp8` 是 compute-bound（56 篇实测算术解码 Compute 65%）。
这里预先把 `(exp, nibble) -> e4m3 字节` 做成 **4096 项 smem LUT**（256 个 E8M0 指数 × 16 个 nibble），
现场折就是两次 smem 字节查表 + 一次拼接：

```c
// 一个 packed byte = 2 个 fp4；一个 16B chunk = 32 个 fp4 = 32 个 k
// 32 个 k 落在 1 个 32-k scale block 里 -> 整块共用一个 exp
for (int b = 0; b < 8; ++b) {
  o[b]   = foldlut[(exp << 4) | ( ww[q]        & 0xF)];  // 低 nibble -> k=2p
  o[8+b] = foldlut[(exp << 4) | ((ww[q] >> 4)  & 0xF)];  // 高 nibble -> k=2p+1
}
```

关键观察：**E8M0 是 block-32，而一个 16B packed chunk 恰是 32 个 fp4 = 32 个 k**，
所以**一个 16B 输入 chunk 共用一个 scale 字节**，LUT 的 `exp` 整块相同。

---

## 3. 正确性：现场折 ≡ 预折叠（逐位）

最重要的验证不是「和 CPU 参考差不多」，而是 **MODE 0 与 MODE 1 的输出逐位相等**：
既然折叠无损，现场折出来的 Bs 应与预折叠的 Bs 逐字节相同，则两模式的 fp32 累加器也应逐位相同。

```text
[fold check] 抽样 1048576 元素 fp4->fp8：最大位差 = 0  OK (逐位无损)
[bitwise] MODE0(pre-fold fp8) vs MODE1(fp4 on-the-fly): max_abs_diff=0.000e+00 bad=0 OK
```

再加上一个（更弱的）CPU 参考对拍：`199/200` 个抽样点相对误差 < 3%，唯一一个 `6.5e-2`
的离群点出现在 `ref ~ 1.7e4` 处——那是 fp32 累加在大量相消下的条件数问题，不是逻辑错。

> 顺带一个**非常大的坑**（见 §7）：我最初的 `fold_fp4_to_fp8_kernel` 和 `repack_scale_kernel`
> 忘了写 grid-stride，固定 grid 只折了 0.4% 的权重，其余全是未初始化显存。而 CPU 参考读的是
> **同一份坏掉的 dW8**，所以它对得上、把错误藏了整整一轮调试。

---

## 4. 实测：现场折反而更慢

`M=8192` balanced routing（每 expert 恰好 128 行 → `Pp=49152`，零 padding），
up/gate grouped GEMM，BN 为 n-tile 宽度：

| 配置 | 路径 | 耗时 | TFLOPS | 有效权重带宽 |
|---|---|---:|---:|---:|
| BN=128 | MODE0 预折叠 fp8 | 11.41 ms | 379.4 (19.2%) | 1482 GB/s (fp8) |
| BN=128 | MODE1 现场折 fp4 | 16.42 ms | 263.7 (13.3%) | 547 GB/s (fp4) |
| **BN=256** | **MODE0 预折叠 fp8** | **7.76 ms** | **557.8 (28.2%)** | **2179 GB/s (fp8, 65% HBM)** |
| **BN=256** | **MODE1 现场折 fp4** | **11.69 ms** | **370.4 (18.7%)** | **769 GB/s (fp4, 23% HBM)** |

**结论：现场折慢 1.44×（BN=128）~ 1.51×（BN=256）**，与「权重带宽减半应该更快」的预期完全相反。
锯齿状看，MODE1 的 fp4 权重带宽只有 769 GB/s——**离 HBM（3352）差 4.4×，说明它根本不是带宽受限**。

---

## 5. 根因：ncu 判「变换受限」

对两个 kernel 各取一组 SpeedOfLight / Occupancy（BN=256，M=8192）：

| 指标 | MODE0 预折叠 fp8 | MODE1 现场折 fp4 |
|---|---:|---:|
| Compute (SM) Throughput | **95.26%** | **95.25%** |
| DRAM Throughput | 47.48% | 47.12% |
| L2 Cache Throughput | 59.13% | 58.68% |
| Achieved Occupancy | 94.65% | 94.66% |
| 主导 stall | 等发射口 39.8% + 未选中 39.7% | 同 |

两个版本都**把 SM 发射口打到 95%**，DRAM 只有 47%。也就是说：**时间花在「搬/变换数据」的
指令上，而不是在等 HBM**。这和 [34 篇]({{< relref "cuda-kernel-opt-34-fused-moe-fp8" >}})
（TMA 版 K1 DRAM 80%、Compute 较低）形成鲜明对照。

### 5.1 诊断实验：把变换短路掉看地板

保留全部骨架，只在 `swz_stage` 入口 `diag&1` 时直接 `return`（跳过 SW128 置换与折叠，结果无意义）：

| 配置（BN=256, M=8192） | 正常 | **DIAG=1（无变换）** | 变换税 |
|---|---:|---:|---:|
| MODE0 预折叠 fp8 | 7.76 ms | **6.60 ms**（2561 GB/s, 76% HBM） | 1.16 ms |
| MODE1 现场折 fp4 | 11.69 ms | **5.25 ms**（1711 GB/s fp4, 51% HBM） | **6.44 ms** |

这两行把问题讲得清清楚楚：

- **地板（无变换）上 FP4 确实更快：6.60 → 5.25 ms（1.26×）**。省下的字节红利是真的。
- 但 MODE1 的**变换税高达 6.44 ms**（减去与 MODE0 同等的 SW128 置换 ~1.16 ms，
  纯折叠约 **5.3 ms**）；MODE0 的置换税只有 1.16 ms。
- **6.44 ms > 省下的 ~2.4 ms 带宽**，于是净效果为负。

为什么会这样？逐字节算：MODE1 每 16B packed 输入要产出 32 个 e4m3 字节，即
**32 次 smem 字节查表 + 2 次 `uint4` 写**；而 MODE0 每 16B fp8 输入只需 **1 读 + 1 写**。
LUT 查表虽然是 1 条指令，但 32 条/16B 把发射口填满了。

> 这也解释了为什么 34 篇用 **TMA**：`cp.async.bulk.tensor` 的 SW128 是**硬件免费**做的，
> 消费者侧零置换代码。一旦被迫走 cp.async（因为要在 kernel 里折 FP4），
> SW128 置换就得自己用指令搬——**FP4 把你从 TMA 快车道赶了下来**。

---

## 6. 判据与「怎么才能赢」

设省下的字节为 $\Delta B$、有效带宽 $W$，变换的单位成本为 $C_\text{tr}$：

$$
\text{现场折划算} \iff C_\text{tr} < \frac{\Delta B}{W}
$$

本例：$\Delta B = 16.91-8.98 = 7.93\ \text{GB}$，$W=3.35\ \text{TB/s}$，右端 $= 2.37\ \text{ms}$；
而 $C_\text{tr}\approx 6.4\ \text{ms}$。**差 2.7×**。

要让现场折赢，需要把 $C_\text{tr}$ 压到 2.4 ms 以下。可行的方向：

1. **host 预排布 FP4 布局**：把 packed fp4 按「一个 16B chunk 恰好对 2 个 SW128 输出 chunk」排好
   （一次性 host 重排），kernel 里就是「读 16B→折→写 32B」的 chunk-local 操作，**没有随机访问**。
   本质上是把 SW128 置换搬到 host，让 kernel 只剩折叠。
2. **把折叠的 32 次查表降下来**：`e2m1` 的 8 个幅值是固定位模式，可用位运算（`(e+6)<<3 | m<<2`）
   或按输出字节做 256 项 LUT，减少一半查表。
3. **prefill 用预折叠（推荐）**：反正 host 折一次是无损的，存 FP8 的代价只是**存储翻倍**
   （专家权重 0.77 TB → 1.55 TB），换来 1.5× 的 kernel。对离线部署，这笔账通常是赚的。

一句话台账：

> **FP4 的价值在存储/传输，不在 prefill 的算力路径。** 要在 prefill 里兑现它，
> 必须让「FP4→FP8」的变换免费（硬件 swizzle 或 host 预排布）；否则每元素变换的发射成本，
> 会超过一半字节省下的内存时间。**带宽敏感（decode）场景才更可能翻盘**——那里 SM 有空、
> 缺的是 DRAM 时间。

---

## 7. 踩过的坑（都命中了）

1. **固定 grid 的 fold kernel 没有 grid-stride**：`fold_fp4_to_fp8_kernel`、`repack_scale_kernel`
   都写了 `if (i >= tot) return` + 固定 `<<<8192,256>>>`，只覆盖 0.4% 的权重；而 CPU 参考读同一份
   坏数据，**对拍全过**，把 bug 藏了一整轮。写任何「大规模逐元素」kernel，一律 grid-stride。
2. **host 侧 `float(fp8)` / `(unsigned char)fp8` 都不是「读原始字节」**：`std::vector<fp8>` 里
   `ha[k]` 会走 `__nv_fp8_e4m3` 的转换运算符，`(float)` 给位模式、`(unsigned char)` 先转 float 再截断。
   要么用 `reinterpret_cast<unsigned char*>` 取字节，要么手写 e4m3 解码。这一条和
   [32 篇]({{< relref "cuda-kernel-opt-32-fused-norm" >}}) 的坑是同源的。
3. **SW128 的 chunk 下标是 `k/16` 不是 `k/8`**：fp8 一行 128B = 128 个元素（不是 bf16 的 64），
   16B chunk 数 = 8。照抄 bf16 的 `(k/8)^r` 会写错（我用 `swz_test.cu` 与 TMA 逐字节对比后才发现）。
4. **wgmma 是 warpgroup 级**：单 warpgroup 只能算 m64；BM=128 需要 **2 个 warpgroup**（256 线程）
   各算一半行，A 描述符按 `wg*64*BK` 偏移。最小复现里只用 256 线程却按单 warpgroup 写，会漏一半行。
5. **mbarrier 方案与「非 grid-stride 的 host 参考」叠加**，让我误以为死锁/竞态是主因，
   换了两版流水都不对——最后发现是 host 数据坏了。**先验证数据，再怀疑流水。**

---

## 8. 代码与复现

```text
code/kernel-opt/57-v4-fp4-moe-fold/
  fold_moe.cu                  主实验：MODE0 预折叠 / MODE1 现场折，bitwise 对拍 + 计时
  swz_test.cu                  SW128 布局最小复现（手写 vs TMA 逐字节对比）
  gemm_smoke.cu                单/双 warpgroup 单 tile wgmma GEMM 冒烟
  fold_M8192_bal_BN128.out.txt 实测原始输出
  fold_M8192_bal_BN256.out.txt
  fold_M8192_bal_BN256_diag.out.txt   DIAG=1 无变换地板
  ncu_pf_BN256.out.txt         MODE0 ncu
  ncu_fp4_BN256.out.txt        MODE1 ncu
```

```bash
# 容器内（H100 / sm_90a）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/run.sh 57-v4-fp4-moe-fold/fold_moe.cu 8192 bal all 256
# 诊断（跳过变换，结果无效）
DIAG=1 ... scripts/run.sh 57-v4-fp4-moe-fold/fold_moe.cu 8192 bal all 256
```

---

## 9. 小结

- **真实语义**：V4-Pro routed experts 是 FP4 e2m1 + E8M0 block-32（56 篇），
  无损折进 e4m3 后可以接进 [34 篇]({{< relref "cuda-kernel-opt-34-fused-moe-fp8" >}}) 的 grouped kernel。
- **现场折 ≡ 预折叠（逐位相等）**，`max_abs_diff = 0`。折叠的正确性没问题。
- **但现场折慢 1.44~1.51×**（BN=256：11.69 vs 7.76 ms）。ncu：两者 **Compute 95% / DRAM 47%**。
- **诊断地板**：去掉变换后 FP4 5.25 ms vs FP8 6.60 ms（**1.26×**）——字节红利真实存在，
  但被 **6.44 ms 的变换税**吃掉。交换判据：$C_\text{tr} < \Delta B / W = 2.37\ \text{ms}$，实测 6.4 ms。
- **出路**：host 预排布 FP4（把 SW128 置换搬到 host）或彻底走**预折叠存 FP8**；prefill 用预折叠更划算，
  decode（带宽敏感）才轮到现场折。

**下一篇候选**：把「host 预排布 FP4 + chunk-local 折叠」做出来，验证能否把
$C_\text{tr}$ 压进 2.4 ms 并把 FP4 prefill 拉到 1.2× 以上；或转去
[主题 25d]({{< relref "cuda-kernel-opt-42-mla-decode-wgmma" >}}) 的 MLA decode producer/consumer 专门化。
