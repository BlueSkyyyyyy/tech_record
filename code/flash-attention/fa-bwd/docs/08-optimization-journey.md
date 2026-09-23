# 反向算子的性能调优历程与方法论（从 138 ms 到 1.95 ms）

> 本文把 ours（借鉴 FA2/FA3/TE 写出来的反向）从「能跑对」到「尽量快」的**调优过程**整理成一条线：
> 每一轮做了什么、ncu 说瓶颈在哪、打掉后**墙迁移到哪**、收益多少、有没有负结果。
> 主指标一律是**时间**（ms/µs，越小越好）；`TF/TFLOPS` 只是 `FLOPs ÷ 时间` 的换算。
> 数据来源均为实测原始输出（`src/**/*.out.txt`）。对标基准 = **FA3（SM90）** 纯反向。

---

## 0. 方法论：一个闭环

```
建基准(纯反向, CUPTI) → ncu 找 bound(SOL/stall/扇区) → 用对应手段打这个 bound
      ↑                                                        │
      └── 对标 FA3/TE + 数值校验(逐位/容差) ← 记录 before/after ┘
```

四条自己给自己立的规矩：

1. **先量化，再动手**：不猜瓶颈。例：O4c 前先用 `lts__t_sectors_op_*` 把「L2 81.5%」拆开，
   才发现 92% 的 L2 扇区来自 dQ/dK/dV 的全局 `red`（原子），而不是访存。
2. **数值护栏**：每一轮优化都要求**数值逐位不变**（除非有意改数学口径），否则不予采纳。
   这让我们能大胆改搬运/布局而不担心悄悄改错。
3. **bound 会迁移**：打掉当前墙，下一个墙立刻暴露——这份日志的重点就是这张「墙迁移」表。
4. **负结果也记**：变慢的改动同样写下来（避免重复踩），并据此判断 bound 类型。

---

## 1. 总览：起点 → 当前（H100，causal）

| dtype | shape | 起点（标量 golden，端到端） | **当前（端到端）** | 提速 |
|---|---|---|---|---|
| fp16 | (1,512,16,128) | 3.61 ms | **0.126 ms**（O11） | **28.7×** |
| fp16 | (1,4096,16,128) | 138.33 ms | **1.95 ms**（O9/O10） | **70.9×** |
| fp16 | (1,1024,64,128) kv4 GQA | ~35 ms | **0.377 ms**（O9） | ~93× |
| bf16 | (1,4096,16,128) | 111.33 ms | **1.95 ms**（O9） | **57×** |
| fp8 | (1,4096,16,128) | 80.63 ms | **3.40 ms**（O11） | **23.7×** |

对标（同 shape、纯反向、CUPTI）：

| shape | ours（端到端） | FA3 (SM90) | TE (cuDNN) | ours/FA3（时间） | ours/FA3（TFLOPS） |
|---|---|---|---|---|---|
| (1,512,16,128) fp16 | 0.126 ms | ~0.027 ms | ~0.046 ms | ~4.7× | ~18% |
| (1,4096,16,128) fp16 | 1.95 ms | 0.324 ms | 0.444 ms | **6.0×** | **~8%** |
| (1,1024,64,128) kv4 fp16 | 0.377 ms | 0.082 ms | 0.112 ms | ~4.6× | ~22% |
| (1,4096,16,128) fp8 | 3.40 ms | —（FA3 无 FP8 bwd） | 0.454 ms | ~7.5× | ~12% |

> 结论：**从 FA3 的 ~1% 一路推到 ~8–22%**（时间从 400× 缩到 6×）。离 FA3 还有距离，
> 差距主要在 **wgmma+TMA 的完整数据通路 + occupancy**（见 §5）。

---

## 2. 逐轮记录：fp8（先把张量核跑通，范式从这里来）

fp8 版最早实现「四段式」(quantize → preprocess → main → convert)，main 用 `mma.m16n8k32`。

| 轮次 | 手段 | before → after（时间） | 打掉的墙 / 新墙 |
|---|---|---|---|
| 起点 | 标量 golden | main S=4096 **198.5 ms** | smem 冲突 + 低 occ |
| mma | 5 个 GEMM 上 `mma.m16n8k32`+`ldmatrix` | 198.5 → **10.19 ms（19.5×）** | 墙转「低 occupancy / 延迟」 |
| **O1** | preprocess 的 LSE 改 mma 分块 + 独立 delta | S=4096 pre 71.0 → **1.14 ms（62×）**；端到端 80.9 → **11.5 ms（7×）** | 端到端瓶颈回落 main |
| **O2** | `dS3/Ap` 折叠进 `Ks/Vs` 死空间（smem 80→75KB） | main S=4096 9.92 → **8.56 ms（1.16×）**；**2→3 CTA/SM** | 仍 L1/TEX |
| **O3** | K/V 向量化 4B 读 + 寄存器预取流水 | main S=4096 8.67 → **7.51 ms（1.15×）** | `long_scoreboard` 2.94→2.21；新墙 = barrier |
| **O4a** | 删 GEMM1/2 间 barrier + prologue 合并 + fold 全线程并行 | main **1.21–1.54×**；barrier 5.13→**0.46** | 新墙 = `short_scoreboard` + L1/TEX |
| **O2b+O4d** | split-K 自动切块 + `Ps/Ss` 行距 +1 消 bank | main d128 1.08–1.21×；`op_ld` 冲突 −61% | L2 81.5% |
| **O4c** | dQ/dK/dV 归约打包成 `atomicAdd(float2*)` | **L2 81.5%→57.9%**；main 1.19–1.45× | 墙 = L1/TEX 81.3% |
| **O4b** | 转置副本 → `ldmatrix.x2.trans` + K 配对布局 | `op_st` 冲突 −66%、**L1/TEX 81.3%→69.7%**；main 1.17–1.76× | 墙 = L2（残余 red）+ short |
| **O7** | dQ 沿 nt 在寄存器累加，每 CTA 只 flush 一次 | L2 red 204→**108 M**、**L2 69.1%→43.8%**；main 1.10× | 墙 = L1/TEX + 残余 red |
| **O11** | LSE 镜像配对负载均衡 + `cp.async.cg` 双缓冲 | pre 3.1×；端到端 **4.13 → 3.40 ms（1.25×）** | lse 墙 = Compute + smem 依赖 |

**fp8 的经验**：张量核只是入场券（第一步 19.5×），后面 100 多个百分点全靠
**逐项打掉 ncu 指出的具体墙**（barrier → bank → 原子/L2 → L1/TEX → 负载均衡）。

---

## 3. 逐轮记录：fp16 / bf16（把 fp8 的范式搬过来，再补 MLA/GQA）

| 轮次 | 手段 | before → after | 备注 |
|---|---|---|---|
| 起点 | 标量 CUDA-core（`HMMA` 都没有） | total S4096 **138.3 ms** | main 68.6 ms |
| **O5** | main 上 `mma.m16n8k16`+`ldmatrix`（照搬 fp8 范式） | main 68.6 → **4.56 ms（14.9×）** | 端到端反被**标量 preprocess** 拖住（68.7ms） |
| **O8** | preprocess LSE 改 mma 分块 | pre 68.7 → **0.99 ms（69.7×）**；total 73.3 → **5.58 ms（13.1×）** | 墙回落 main |
| **O6** | K/V `cp.async.cg` 双缓冲 | main 4.51 → **1.87 ms（2.41×）**；total **3.04 ms** | `long_scoreboard` 7.35→1.12 |
| **O6b** | `ldmatrix.x4.trans` 消转置副本 + 只双缓冲 K | smem 84→71KB、**2→3 CTA/SM**；main 1.02× | 墙 = L1/L2 + wait |
| **O8b** | LSE 镜像配对（因果负载均衡）+ cp.async | lse 2.81×；**尾波消除**；total 2.95 → **2.34 ms** | 消融：配对 2.20×、cp.async 1.28× |
| **O6c** | tile 几何参数化 + 小网格并行度自适应 | S512 main 1.11×；occ 6.2→11.0% | 证伪静态重排（0–2%） |
| **O7c** | LSE/D 预装寄存器（避免每 tile 重复 global 读） | main +12–19%；total 2.38 → **2.09 ms** | 新墙 = L2 71.6% + occupancy |
| **O10** | Q/dO 载入 16B `cp.async` + dQ 打包写回 | 端到端 1.04–1.32× | 墙仍是 wait+L2 |
| **O9a** | LSE 预处理上 Hopper `wgmma` | L1/TEX 减半，数值逐位同 | 通往 O9 的第一步 |
| **O5c** | fp16/bf16 的 **MLA head_dim=512** 反向也上张量核 | main **5.2–5.8× / 3.7–4.1×** | MLA 从标量升级 |

**fp16/bf16 的经验**：搬迁 fp8 范式时，**每换一种 dtype 都会冒出专属问题**
（bf16 曾有 9.5-way smem bank conflict；MLA D=512 的 smem 容量/寄存器不同），
`HD/BM/BN` 必须模板化，并按容量自动选。

---

## 4. 「墙迁移」总表（这份日志最值钱的部分）

把每一轮的 ncu 头号瓶颈连起来，就是优化的路线图：

```
smem 冲突 + 低 occ
   → 低 occupancy / 延迟
   → preprocess（LSE/D 未分块）
   → CTA barrier（每 tile 多个 __syncthreads）
   → smem bank conflict（Ps/Ss 行距、转置副本）
   → L2 原子流量（dQ/dK/dV 的 red 占 92% 扇区）      ← O4c
   → L1/TEX 吞吐（ldmatrix/ldmatrix.trans）          ← O4b
   → L2 残余 red + short_scoreboard                  ← O7
   → long_scoreboard(访存延迟) → cp.async 双缓冲     ← O6/O8b
   → 尾波不均衡（causal 负载）                        ← O8b 镜像配对
   → L2 + occupancy（但 smem 已 >100KB，只能 2 CTA/SM）← O7c 后的新墙
   → 需要更低 smem 的数据通路 = wgmma + TMA           ← O9（进行中）
```

**读懂这张表，就知道「为什么不能一步到位」**：每个手段只对**当前那堵墙**有效，
墙一变，手段就得换。这也是为什么调优是**迭代**而不是一次性重写。

---

## 5. 当前仍未解决的（下一步）

1. **O9（wgmma + TMA + warp specialization）**：当前 main 的墙是「L2 吞吐 + 2 CTA/SM」，
   而 smem 已 >100KB，只用 `mma.sync` 的数据通路降不下来。只有换 Hopper 的
   `wgmma`（寄存器操作数、更省 smem）+ TMA（省地址/搬运指令），才能同时降 smem、提 occupancy。
   O9a（LSE 上 wgmma）已完成，main 的 wgmma 化进行中。
2. **O7b**：dK/dV 的跨 CTA 残余 red（108 M）→ 分块 `*_accum`+convert，顺带拿到**确定性反向**。
3. **MLA（head_dim=512）**：已是张量核，但 1 CTA/SM（smem/寄存器大），需继续降 smem 或 persistent。

---

## 6. 可复用的经验（写给别人 / 未来的自己）

1. **对标要选同代**：FA2（SM80）≠ FA3（SM90）。拿错代际会得出相反结论（见 `docs/06`）。
2. **先确认 bound 再优化**：`ncu` 的 SOL 看单元，`--page source --print-source sass` 看指令，
   `lts__t_sectors_op_*` 拆 L2 到底被什么占满。
3. **数值逐位不变是最好的回归测试**：它让「只改搬运」的优化可以放心上。
4. **负结果要记**：dK/dV 归约提到 float4 反而慢 → 该归约不是事务数 bound；
   静态负载重排 0–2% → causal 的动态负载要靠镜像配对（2.2×）。
5. **利用对称性做负载均衡**：因果 mask 让第 `m` 个 CTA 的活是 `m+1` 个 tile，
   把 `m` 与 `nblk-1-m` 配对后每个 CTA 恒为 `nblk+1`（O8b）。
6. **单文件 / 两文件保持逐字一致**：便于教学与工程两用。
