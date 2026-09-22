---
title: "CUDA 算子调优（五十九）：FP4 decode 的 grouped GEMV —— 把 top-6 的 6B 个 (token, expert) 对按专家折叠，权重只读一遍"
date: 2026-09-22T09:36:00+08:00
draft: false
weight: 59
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "FP4", "e2m1", "DeepSeek-V4", "GEMV", "grouped", "MoE-decode", "dp4a", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 56 篇]({{< relref "cuda-kernel-opt-56-v4-fp4-moe" >}}) 钉死了 DeepSeek-V4-Pro 路由专家的真实格式
（FP4 e2m1 + E8M0 block-32），[第 58 篇]({{< relref "cuda-kernel-opt-58-v4-fp4-decode" >}}) 又把**单专家**
的 M=1 decode GEMV 推到 0.0090 ms / 1296.6 GB/s / 38.7% HBM，并给出判词：

> **38.7% 就是这一层的天花板**：`w1` 只有 `N=3072` 行，「一行一 warp」只有 3072 个 warp
> （`Waves Per SM = 0.36`、`Issued Ipc = 1.08`），GPU 并行度根本喂不满；split-K、位运算解码、
> 无 smem 都救不了它。

但 58 的实验看的是**一个 (token, expert) 对**。真实 decode 一步要算 **top-6 个专家**——也就是说
一个 token 要跑 6 次 GEMV。把视野放大到「一步 decode」：

- 一个 decode step（batch = `B`）要算 `B × 6` 个 `(token, expert)` 对，落在 384 个路由专家上；
- 每个专家的权重是 `N=3072 × K=7168` 的 FP4（`11.0 MB`）+ E8M0 scale（`2.75 MB`）；
- 一步 decode 的权重字节 = `pairs × 13.76 MB`，而**最少**只需要 `active_experts × 13.76 MB`。

这就是 58 标题里那句「形状决定天花板」的正确打开方式：**decode 根本不是一个 GEMV，而是一个
以 expert 为分组、每组的 M 只有几个的 grouped GEMV**。本篇把它写出来，结论：

> 把 `B×6` 个 `(token, expert)` 对按 expert 折叠进一个 kernel，**权重每个专家只从 HBM 读一次**：
> `B=256`（每专家 4 个 token）时 **2.63 ms / 有效权重带宽 2008 GB/s / 59.9% HBM**，
> 相对「逐对 launch」的 naive 基线 **5.9×**、相对「一次性 batched 但 M=1」**3.2×**。
> 关键的 60% 不是靠读得更少自动拿到的：**先修掉激活在共享内存里的 8-way bank conflict**
> （两平面布局），batched 基线从 67.6% → 74.5%，grouped 从 40.6% → 59.9%。

---

## 一、把一个 decode step 的账算清楚

先看 shape（全部取自 `/ssd/models/DeepSeek-V4-Pro/config.json`）：

| 参数 | 值 | 含义 |
|---|---|---|
| `hidden_size` | 7168 | K |
| `moe_intermediate_size` | 3072 | N（w1/w3 的输出维） |
| `n_routed_experts` | 384 | 每层专家数 |
| `num_experts_per_tok` | 6 | top-6 |
| `expert_dtype` | `fp4` | e2m1 + E8M0 block-32（56 篇实测） |

单专家权重（FP4 + E8M0 scale）：`3072×7168/2 = 11.0 MB`（fp4）+ `3072×7168/32×4 = 2.75 MB`（scale）
= **13.76 MB**。384 个专家一层 `w1` 就是 **5.28 GB**（`w1/w2/w3` 三张共约 15.8 GB，这就是 FP4 让
MoE 能塞进显存的意义）。

一步 decode（batch `B`）的**工作量固定**：`B×6` 个 `(token, expert)` 对，每个是
`2·N·K = 4.4e10` FLOP？不，是 `2×3072×7168 = 4.4×10^7` FLOP，总 FLOP 与 B 成正比。

而**权重字节**取决于实现：

```text
naive（逐对）      : W_bytes = pairs × 13.76 MB = 6B × 13.76 MB
grouped（按专家）  : W_bytes = active_experts × 13.76 MB ≈ E × 13.76 MB（每个专家只要有一个 token 就要读）

复用倍数 reuse = pairs / active_experts ≈ 6B / E
  B=64  (m_e=1) : reuse 1×   两者一样
  B=256 (m_e=4) : reuse 4×   grouped 只读 1/4
  B=512 (m_e=8) : reuse 8×   grouped 只读 1/8
```

**「按专家折叠」是本篇唯一重要的结构决策**：同一个专家的 m 个 token 共享一次权重读取，
算术强度从 58 的 `AI≈3.76`（M=1）抬到 `AI≈3.76·m`。`m→∞` 时 grouped 逼近「读一遍全部专家权重」
的 roofline：

```text
roofline(grouped) = active_experts × 13.76 MB / 3.35 TB/s
                  = 384 × 13.76 MB / 3.35 TB/s ≈ 1.58 ms   （B 足够大、全专家激活）
```

这就是本篇要在 `B=256/512` 上打的靶子。

---

## 二、三个实现，一个 kernel

为了干净地分离「并行度」和「权重复用」两个变量，写三个口径（同一份真实权重）：

```text
naive_loop : 58 的 pipe kernel，每个 (token,expert) 对单独 launch（6B 次），M=1
             → 有 launch 税、有并行度上限，权重读 6B 遍

batched_m1 : 一个 kernel，grid.y = pairs，每 CTA 一个 (token,expert)，M=1
             → 只赢「一次 launch + 网格铺满」，权重仍读 6B 遍

grouped    : 一个 kernel，grid.y = 384 experts，每 CTA 吃下该 expert 的全部 m 个 token
             → 一次 launch + 网格铺满 + 权重每专家只 decode/读一次、dp4a 复用 m 次
```

三个口径共用**同一个模板 kernel**，靠「组」的元数据区分：

```cpp
// grid = (N/BN, G)
//   gid = blockIdx.y 是「组」编号
//   e   = wexp[gid]      权重用的专家    （grouped: wexp[e]=e；batched: wexp[pair]=expert）
//   m   = counts[gid]    该组 token 数   （grouped: B/64；batched: 1）
//   off = toff[gid]      token 列表起点
//   toks[off..off+m)     该组的 token
template <int RWW, int MTMAX, int DEPTH, bool USELUT = true>
__global__ void gemv_fp4_grouped(...) {
  const int gid = blockIdx.y;
  const int m = counts[gid];
  const int e = wexp[gid];
  // 1) 把 m 个 token 的 int8 激活 + per-128 scale 搬进 smem
  // 2) 每个 warp 负责 RWW 行 N，沿 K 的 16B chunk 流式读权重：
  //      对每个 chunk：decode 一次 w -> w8[8]（8 个 int8 word），
  //      然后对 m 个 token 各做 8 次 __dp4a
  //    => 权重 decode 被 m 个 token 摊薄
  // 3) 写 Y[(off+mm)*N + row]（mm 是组内 token 下标）
}
```

权重 decode 复用是全篇的收益来源：**每个 16B 权重 chunk 只 decode 一次，喂给 m 个 token 的 dp4a**。
`m=1` 时它退化成 58 的 kernel；`m=4` 时 decode/读权重的成本被摊薄 4 倍。

### 路由与工作点

用**平衡路由** `(t*6+j) % 384`（周期 64），让每个专家恰好拿到 `B/64` 个 token，于是
`m_e ∈ {1,2,4,8}` 对应 `B ∈ {64,128,256,512}`，无 padding 浪费。

### 权重怎么来

只有一个真实专家的权重文件，但要在显存里铺 384 个专家（否则权重会全部命中 L2，带宽是假的）。
做法是 host 侧逐字节 `wpack ^ (e*37)`（任何字节都是合法 e2m1），scale 乘 `1+0.05*(e%5)`——**既
撑出了 5.28 GB 的真实 HBM 工作集，又让每个专家数值不同，能抓「权重/scale 索引错」**。正确性用
CPU 参考抽查 8 个 `(组, 行)` 点。

---

## 三、第一版：权重复用兑现了，但带宽只有 40%

直接跑 `B=256`（m=4），第一版结果：

```text
  naive_loop    15.42 ms  (weight read 21.14 GB, 1370 GB/s, 40.9% HBM)
  batched_m1     9.33 ms  (weight read 21.14 GB, 2265 GB/s, 67.6% HBM)
  grouped RWW=1  5.42 ms  (weight read  5.28 GB,  974 GB/s, 29.1% HBM)
  grouped RWW=2  3.89 ms  (weight read  5.28 GB, 1360 GB/s, 40.6% HBM)
```

好消息：grouped 的**绝对时间**已经是 batched 的 2.4×（权重读少了 4 倍）。
坏消息：grouped 的**有效带宽只有 40.6%**，离 roofline 的 80%+ 差很远，而且远低于 batched 的 67.6%。

用 ncu 看 grouped RWW=2 第一版：

```text
  DRAM Throughput        37.6%
  Registers Per Thread   126          -> Block Limit Registers 2
  Dynamic Shared Mem     59.14 KB/CTA -> Block Limit Shared Mem 2
  Achieved Occupancy     24.2%
  No Eligible            58.7%（每 scheduler 只有 0.48 warp eligible）

  OPT: uncoalesced shared accesses -> 376,720,896 excessive wavefronts (54% of 691,613,184)
```

两个病：

1. **占用率被寄存器和 smem 同时锁在 25%**。smem 按模板上限 `MTMAX=8` 分配（59 KB），
   即使 `m=4` 也占满；
2. **共享内存 54% 是多余 wavefront（bank conflict）**。激活按 `[token][K]` 行主序存，
   而 warp 内 `lane l` 处理 chunk `c=lane+32i`，读激活地址是 `base + c*32` 字节——**相邻 lane
   差 32 B**。16 B 的 `uint4` 在 32 B 步长下每 8 个 lane 就回到同一批 bank → **8-way conflict**。
   每个 chunk、每个 token 读 2 个 `uint4`，冲突被 `m` 放大。

---

## 四、两个修复：亲和 m 的 smem + 两平面激活

**修复 1：让 smem/寄存器随真实 `m` 缩放。** 把 `MTMAX` 做成按 `MMAX = B/64` 分派的模板
（1/2/4/8），smem 从固定 59 KB 变成 `MTMAX×(K + K/128×4)`（m=4 时 29.6 KB），寄存器也随
`x8[MTMAX][8]` 下降。occupancy 的 smem 限制从 2 块放到 4 块。

**修复 2：激活改成「两平面」布局，消掉 8-way conflict。** 关键观察：一个 16 B 权重 chunk 对应
32 个 k 值，而激活是 int8，32 个 k 值 = 32 B = **两个 `uint4`**（记作 `xa`、`xb`）。把这两个
`uint4` 分开放进两块平面：

```text
plane0[token][c] = 第 c 个 32-k chunk 的前 16B
plane1[token][c] = 第 c 个 32-k chunk 的后 16B
```

这样 lane `l` 读 `plane0 + (token*CH + c)*16` 时，地址是 `... + lane*16`——**相邻 lane 差 16 B**，
一个 warp 恰好铺满 512 B 连续、零冲突。

```cpp
// 装载（global -> smem 两平面）
for (int i = tid; i < m * CH; i += 256) {
  int mm = i / CH, c = i % CH;
  const uint4* src = (const uint4*)(Aq + toks[off+mm]*KK);
  *(uint4*)(xs0 + (mm*CH + c)*16) = src[2*c];      // 前 16B
  *(uint4*)(xs1 + (mm*CH + c)*16) = src[2*c+1];    // 后 16B
}
// 计算（lane 地址相邻 16B，合并）
uint4 xa = *(const uint4*)(xs0 + (mm*CH + c)*16);
uint4 xb = *(const uint4*)(xs1 + (mm*CH + c)*16);
```

两个修复叠加后：

```text
  naive_loop    15.78 ms  (21.14 GB, 1340 GB/s, 40.0% HBM)
  batched_m1     8.47 ms  (21.14 GB, 2496 GB/s, 74.5% HBM)
  grouped RWW=1  3.51 ms  ( 5.28 GB, 1503 GB/s, 44.9% HBM)
  grouped RWW=2  3.16 ms  ( 5.28 GB, 1673 GB/s, 49.9% HBM)
  grouped RWW=4  2.63 ms  ( 5.28 GB, 2008 GB/s, 59.9% HBM)   <- 新最佳
  grouped RWW=8  3.31 ms  ( 5.28 GB, 1596 GB/s, 47.6% HBM)
```

两平面布局对 **batched 基线也生效**（67.6% → 74.5%），说明它修的是激活访存这个公共瓶颈，不是
grouped 特有的。`RWW=4`（一个 warp 4 行 N，BN=32）成为甜点：行数越多、一次激活读取服务的
dp4a 越多；到 `RWW=8`（BN=64，n-tile 只有 48）寄存器涨到 occupancy 掉一档，反而更慢。

---

## 五、跨 batch / 复用倍数扫一遍

固定 `RWW` 最优随 `m` 走（小 m 用大 RWW、大 m 用小 RWW：激活复用多时行数反而没那么值钱）。
每个 `B` 取最优 `RWW`：

| B | m | 权重读 | naive_loop | batched_m1 | **grouped（最优）** | grouped vs naive | vs batched | 有效带宽 |
|---|---|---|---|---|---|---|---|---|
| 64 | 1 | 5.28 GB | 4.07 ms | 2.13 ms | **1.91 ms**（RWW=2） | 2.14× | 1.12× | **82.7%** |
| 128 | 2 | 5.28 GB | 7.87 ms | 4.25 ms | **2.12 ms**（RWW=8） | 3.71× | 2.00× | 74.3% |
| 256 | 4 | 5.28 GB | 15.44 ms | 8.47 ms | **2.63 ms**（RWW=4） | 5.87× | 3.22× | 59.9% |
| 512 | 8 | 5.28 GB | 30.54 ms | 16.92 ms | **4.42 ms**（RWW=2） | 6.91× | 3.83× | 35.7% |

三个读法：

1. **绝对时间全线赢**。grouped 比 naive_loop 快 2.1–6.9×，比 batched_m1 快 1.1–3.8×；
   复用越大赢得越多——这正是「按专家折叠」的意义。
2. **单 token 成本随 batch 下降**：`1.91/64 = 29.8 µs` → `10.3 µs`（B=256）→ `8.6 µs`（B=512）。
   B 越大，那 5.28 GB 固定权重被越多 token 摊薄。
3. **有效带宽的「百分比」在大 batch 反而下降**，别误读：grouped 的**分母是 5.28 GB 固定权重**，
   而 `m=8` 时激活/LUT 的共享内存工作随 m 线性涨。B=512 时 42 GB 的 naive 流量被压到 5.28 GB，
   但 kernel 已经从「等 HBM」滑向「等共享内存」。B=64（m=1、无复用）反而是最纯粹的带宽
   benchmark，跑出 **82.7%**。

---

## 六、ncu：墙从 DRAM 挪到了 L1/TEX

对 `B=256, RWW=4`（当前最佳）采 ncu：

```text
  Memory Throughput      83.49%     <- L1/L2/DRAM 里最高那一级
  DRAM Throughput        60.12%     <- 与外部口径 59.9% 吻合
  L1/TEX Cache Throughput 83.87%    <- 真正的墙
  L2 Cache Throughput    63.69%
  Compute (SM)           41.12%
  Registers/Thread       117         -> Block Limit Registers 2（occupancy 上限）
  Achieved Occupancy     24.44%
  short_scoreboard stall 3.00（每 issue 的 warp-cycles）
```

**DRAM 60%、L1/TEX 83.9%**：grouped 已经把 HBM 用到六成，但每 cycle 的瓶颈是**共享内存管道**。
剩下的多余 wavefront（244.6 M / 508.6 M ≈ 48%）几乎全来自 256 项 `uint16` 的 **LUT 随机读**
（激活的 8-way 已经被两平面修掉了）。于是自然要问：

> 把 LUT 换成纯寄存器的 `__byte_perm`（PRMT）解码，能消掉这 48% 吗？

## 七、负结果：PRMT 解码在 grouped 里也更慢

给同一 kernel 加 `USELUT=false` 的 PRMT 路径（[58 篇]({{< relref "cuda-kernel-opt-58-v4-fp4-decode" >}})
的位运算解码表），`B=256, RWW=4` 同进程对照：

```text
  grouped LUT  (RWW=4)  2.63 ms  (2008 GB/s, 59.9% HBM)
  grouped PRMT (RWW=4)  3.50 ms  (1509 GB/s, 45.0% HBM)   <- 慢 33%
  正确性：max_rel=1.15e-7 OK（两条路逐位一致）
```

**和 58 在 M=1 上得到的结论一致**：LUT 的 bank conflict 是「访存税」，PRMT 把它换成
「ALU 税」（~3 条指令/nibble），而这个 kernel 还有 41% 的 SM 空闲不代表 ALU 有富余——
`__dp4a` 本身就在 ALU/整数管道上，额外塞进的展开指令直接顶住了发射口。**「把访存税换成 ALU 税」
在已经偏指令密集的核里方向是反的。**

---

## 八、和 SOTA / 上一版的差距

- **相对 58 的单专家 M=1**（0.0090 ms / 1296.6 GB/s / 38.7%）：一步 decode 要 6 个专家，
  58 口径下要 `6×0.0090 = 0.054 ms`（单 token）。本篇 grouped 在 `B=1` 时会退化成同样的形状
  （6 个专家各 1 个 token），但**批量 decode** 下把固定权重摊薄到 `10.3 µs/token`（B=256），
  即单 token 成本降到 58 的 **~1/5**。
- **相对 batched_m1**（同网格、M=1）：grouped 的权重字节是它的 `m/... ` （B=256 时 1/4），
  时间快 3.2×。
- **相对 roofline**：全专家激活下 `384×13.76 MB / 3.35 TB/s ≈ 1.58 ms`；B=256 实测 2.63 ms，
  **达 roofline 的 60%**；B=64（m=1）实测 1.91 ms / 82.7%。
- 没有可用的 Marlin / DeepGEMM FP4 grouped decode 入口做精确对照（DeepGEMM 在本容器仍无法编译，
  见系列「阻塞」），这里只给「读一遍全部专家权重的物理下限」作为上界参考。

---

## 九、小结与下一步

- **decode 不是 6 个独立 GEMV，而是一个 grouped GEMV**：`B×6` 个 `(token, expert)` 对按 expert
  折叠后，权重每个专家只读一次，复用倍数 = `pairs/active ≈ B/64`。
- 一个模板 kernel 三种口径（逐对 launch / batched M=1 / grouped），把「并行度」和「权重复用」
  两个变量干净分离。
- **两平面激活布局**把激活 smem 的 8-way conflict 修掉，batched 67.6%→74.5%、grouped 40.6%→59.9%；
  这是本篇最可移植的一条工程技巧（任何「lane 处理 32-k chunk、要读两个 `uint4`」的 int8 GEMV 都适用）。
- `B=256` grouped **2.63 ms / 2008 GB/s / 59.9% HBM，vs naive 5.9×、vs batched 3.2×**；
  `B=64` 达 **82.7%**。
- ncu 判决：墙从 DRAM 挪到 **L1/TEX (83.9%)**，残余是 LUT 随机读；**PRMT 换 LUT 是负结果**（慢 33%）。
- 正确性：`max_rel ~1.1e-7`；权重/scale 索引错会被 per-expert 扰动抓出来。

**下一步**（写进 ROADMAP）：

1. **冲突无关的 LUT**：既然 LUT 随机读是唯一残余（48% 多余 wavefront），试 `lut[i]` 加 padding
   或把 256 项表拆成两个 128 项半表落到不同 bank 段，看能否在不加 ALU 的前提下压掉冲突；
2. **真正的稀疏路由**：平衡路由让 384 个专家都激活；真实 router 是幂律分布，很多专家分配 0 个
   token——`counts[gid]==0` 的 CTA 现在直接 return，可以把「活跃专家列表」压紧、进一步降权重流量；
3. **把这套 grouped GEMV 接到 decode 的 MoE FFN 端到端**（w1/w3 → SwiGLU → w2），量端到端收益；
4. **Blackwell `tcgen05`**：grouped 的 M 小但仍想吃张量核，等有 sm_100 机器再试。

代码在 `code/kernel-opt/59-v4-fp4-grouped/`（`fp4_grouped.cu`，`grouped_B{64,128,256,512}.out.txt`，
`ncu_grouped_g4.out.txt`，`ncu_batched_m1.out.txt`）。
