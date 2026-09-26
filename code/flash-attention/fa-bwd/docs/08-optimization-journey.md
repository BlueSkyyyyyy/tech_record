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

> 第 51 轮（O15a/O16）把 fp16 main 的墙**定量钉死**：ncu 拆 L2 扇区，**`red`（dK/dV 的
> 跨 CTA `atomicAdd`）占 73.1%**、DRAM 仅 4.2% ⇒ main 是 **L2 原子字节数 bound**。
> 同一轮：TMA+SW128 冒烟逐位 PASS（建好 Hopper bulk-tensor 通路，并发现 HD=128 的 K-major
> tile 必须拆成 2×K=64 chunk 才能喂 TMA），而「分段 `wait_group` 重叠 epilogue」实测**中性**
> （0.99–1.00×）——证明动搬运/等待打不动原子墙。详见 `docs/01` §14i。

0. **清非-main 开销（O24，第 65 轮，已完成）**：main 已是硬墙（L2 red）后，回头清
   preprocess/convert 的工程浪费——`delta_kernel` 从「每行一个 CTA + smem 树归约」改成
   **warp-per-row `vector2` + `shfl`**（S4096 42.5→**14.5µs，3.35×**、DRAM 74% bound、指令 −82%），
   D=128 的 wgmma2/2b 主 kernel **直接写 fp16 dQ**、convert 跳过 dQ。端到端 fp16/bf16
   S4096 **1.037×/1.020×**、S512 1.05×、GQA 1.05–1.08×，数值逐位不变。详见 `01` §14p、`01b` §6w。
0b. **Hopper 快路默认化（O23，第 63 轮，已完成）**：O9a/O17/O18 把 fp16/bf16 的 main/LSE 做到   Hopper `wgmma`，但一直是 opt-in（默认仍走 mma）。O23 在 `-DFA_WGMMA` 构建下把它们**默认打开**
   （D==128：S≥4096→wgmma2b(BN=128)，否则 wgmma2(BN=64)；LSE wgmma），端到端 **1.08–1.43×**
   （MHA S4096 1.945→1.364ms），数值逐位不变；`--wg2=0 --wg2bn=0`/`--lsewgm=0` 保留对照。
   默认档的墙不变（**L2 red + 1 CTA/SM**）。详见 `docs/01` §14n、`docs/01b` §6v。
0c. **fp8 delta 向量化（O26，第 67 轮，已完成）**：fp8 的 `delta_kernel` 是 preprocess 里唯一
   没跟上向量化的子 kernel（O11/O14 只覆盖 LSE/quant）。改成 **warp-per-row**（`float4`(O) +
   `uchar4`(dO) + `shfl` 树，无 smem/无 barrier），delta **2.39–3.22×**（S4096 42.9→17.3µs）、
   端到端 S4096 1.008×（55.8 TF，TE FP8 的 4.17×）；ncu 墙移到 **DRAM 带宽 77%**。
   **fp8 preprocess 三子 kernel 至此全部向量化**。详见 `docs/03` §31。

1. **跨 warpgroup 归约（BM=128，2 warpgroups）→ 已完成（O17，第 52/53 轮）**：唯一能直接砍
   half dK/dV 原子字节的杠杆（一个 KV 元素由 `nblk/2` 个 CTA 贡献，两组的 dV/dK 偏和在 smem
   合并一次再写）。实测 `red` 102.2M→51.9M（0.508×）、main S4096 1.56×。
   **再翻倍到 BM=256/4wg（O17b，第 54 轮）是负结果**：red 确实再减半（→26.7M），但 512 线程
   把每线程寄存器上限压到 128，dQ 累加器 + 两条 wgmma 累加器必然 spill，local 占 L2 ~48%，
   净 Duration +21%。**结论：寄存器文件是硬墙，「放大 BM」走不通**（详见 `docs/01` §14k）。
2. **O7b（当前第一优先级）**：dK/dV 的跨 CTA red → 分块 `*_accum`+convert（顺带拿到**确定性反向**）；
   把原子 RMW 换成「CTA 局部累加 + 非原子写 + 二次归约」，直接消 L2 原子。会多一趟读回、字节不减。
3. **fp8 侧同构跨 wg 归约**：fp8 是 1 字节 operand、smem 更省，4wg 的寄存器压力比 fp16 小一档。
4. **O17 的 `BN=128` 微优化**：tile 数/barrier 减半（smem 恰好 224KB、2 wg 寄存器够用）。
5. **TMA 化 operand**：**LSE 已落地（O30，第 71 轮）**——4D-TMA 载入 Q/K（两个 K=64 chunk），
   LSE-only **1.30–1.36×**、指令数 −28.7%、数值逐位不变；但墙不变（Compute ~58% + `wait`，
   即 softmax epilogue），且动不了 main 的 L2 red。**主 kernel 的 Q/K/V/dO TMA 化**（需把
   HD=128 tile 改 2×K=64 chunk、并改 GEMM3/4/5 的转置描述符）与 **bf16/fp8 版 TMA** 仍列 backlog。
6. **MLA（head_dim=512）**：已是张量核，但 1 CTA/SM（smem/寄存器大），需继续降 smem 或 persistent。
7. **fp8 主 kernel 的 `red` 天花板（O42，第 89 轮）**：ncu 重测确认 dK/dV 的跨 CTA `red` 仍是
   头号成本——占 L2 扇区 ~70%（114.5 M）、`wait` 1.53 + `short_scoreboard` 1.30；**短路掉
   dK/dV red 后 main 1.60→0.94 ms（天花板 1.70×）**。但把 red 换成「smem staging +
   `cp.reduce.async.bulk`」实测 **慢 11%（0.89×，负结果）**——staging 的 smem 往返把小粒度 TMA
   顶在已经 71.8% 的 L1/TEX 上。⇒ red 只能靠**减少每元素贡献数（放大 BM，撞寄存器墙）**或
   **提 occupancy（smem 74.8 KB + regs 168 双卡 3 CTA/SM）**解决，两者都是硬件资源硬约束。
   详见 `docs/03` §45。
8. **小 grid 的 N 方向 split-K（O43，第 90 轮，正结果）**：fp16/bf16 默认档下 **S=512 MHA 的
   `wgmma2` grid 只有 64 CTA < 132 SM**（ncu `Waves 0.48`，half-SM 空转）。给 `wgmma2` 加
   运行时的 `ksplit`（KV tile 切片 + dQ 跨 CTA 原子累加，`ksplit==1` 逐位退化）：S=512
   main **1.67–1.68×**、端到端 **1.26–1.27×**（0.087→0.069ms），Waves **0.48→0.97**、
   elapsed IPC 0.42→0.77，per-SM occupancy 不变（12.5%）。⇒ **打掉的是「SM 空转」而非延迟隐藏**；
   varlen 短序列切 K 反慢（opt-in）。详见 `docs/01` §14x、`docs/01b` §6af。
9. **MLA（D=512）主 kernel 的 split-KV（O44，第 91 轮，正结果）**：O43 只覆盖 D=128 的
   `wgmma2`；**MLA 走 mma 主 kernel（`fa_bwd_{fp16,bf16}_mma_kernel`, BM=32/BN=32/PIPE=1）
   没有 ksplit**，S1024H2 / S512H4 / S256H2 的 grid 只有 64/64/16（Waves 0.48、occ 6.25%、
   207KB smem）。把 O43 的机制扩到 mma 主 kernel（KV tile 切片 + 空切片早退 + prologue stage
   对齐 `nt_begin&1`，dQ 在 split>1 时改 `red_add2`）：main **4.4–8.4×**、端到端 **4.2–5.4×**
   （S1024H2 fp16 total 0.939→0.201ms），Waves 0.48→0.97、Ipc 0.33→0.57、per-SM occ 不变。
   ⇒ **fp16/bf16 MLA total 现已比 fp8 MLA 快 ~5×**。详见 `docs/01` §14y、`docs/01b` §6ag、
   `docs/04` §19。**教训：split-K 的 auto 目标别只填「一个波」——MLA（1 CTA/SM）实测 4 个波
   （`grid*sp≈528`）更优**（同 O29 对 fp8 MLA 用 `S/2` 的有效目标）。
10. **fp8 MLA（D=512）主 kernel 的墙复核（O45，第 92 轮，负结果 + 更正）**：先更正一条**错记**——
   上面第 9 条「fp16 比 fp8 MLA 快 ~5×」是拿 O44 去比 **P5-3 时代（第 21 轮）** 的旧数字；
    fp8 MLA 早在 **O29** 就有 auto ksplit（`target=S/2`），实测 total 仅比 fp16 慢 **1.4–1.9×**。
    ncu 复核：fp8 MLA main 是 **1 CTA/SM × 4 warp = 1 warp/scheduler**（No Eligible 85.7%、
    Active Warps/Sched 1.00、stall `long 2.33 + wait 1.54 + short 0.76`），墙是**并行度/延迟**，
    不是带宽/算力（DRAM 2.2% / L1TEX 19.8% / Compute 11.9%）。判决两条新机制：把 O42 的
    **bulkred 开放到 D=512 仍 0.84×**（即使 L1/TEX 有余量，staging+小粒度 TMA 本身净亏）、
    **`FA_ILV/ILV34` 中性/负**；K/V 全局载入的天花板只有 **1.16×**（探针）。⇒ 2 CTA/SM 因
     smem 207.9KB 不可达（消掉全部配对副本仍 >124KB），唯一剩余杠杆是 **256 线程/8-warp 几何**
     （fp16 O6c 式参数化，多轮，backlog）。详见 `docs/03` §46。
11. **MLA（D=512）主 kernel 的 8-warp 几何（O46，第 93 轮，正结果，D=512 默认）**：把
    `fa_bwd_{fp16,bf16}_mma_kernel` 的 warp 网格从写死的 2×2 改成由 `NTH`（线程数）/`NWAR`
    （N 方向 warp 数）派生（`NWM=NTH/32/NWAR`；`NTW≡128` 与 warp 数解耦）；默认 `128/2` 与
    O5c 逐位等价，MLA 用 `256/4`（2×4 网格）。**1 CTA/SM 下每 scheduler 的 warp 数 1→2**：
    ncu **occ 6.10%→12.36%、Ipc 0.57→0.65、Duration 155.1→145.4µs**，regs 168（+spill）→
    **151（0 spill）**；main **1.070–1.108×**、total 1.00–1.07×（S256H2 4w/8w: 0.0221/0.0204、
    S512H4 0.0839/0.0757、S1024H2 0.1512/0.1409ms）。数值 vs ref 与 4-warp 档逐值一致、
    D=128 MHA/GQA 回归逐位不变。墙仍是 **L2 red + 延迟**。`--mla8w=0` 供 A/B。详见 `docs/01`
    §14z、`docs/01b` §6ah。**下一步候选**：① 把同一 8-warp 几何搬到 **fp8 MLA**（fp8 是最重点
    且 128 线程/4 warp 同病，但需重排 fp8 的 `fp8_mma_body` 2×2 几何与 fold）；② fp16/bf16/fp8
    **D=128** 主 kernel 是否也能吃 8-warp（wgmma 路径已 256 线程，mma fallback 与 GQA 待测）；
    ③ fp8 MLA 的其它杠杆（O42/O45 一致：硬件资源锁死）。
12. **fp8 MLA（D=512）主 kernel 的 8-warp 几何（O47，第 94 轮，正结果，D=512 默认）**：把
    同一参数化搬到 fp8——`fp8_mma_body`/`fa_bwd_fp8_mma_kernel` 的 warp 网格由 `NTH`/`NWAR` 派生
    （`NWM=NTH/32/NWAR`、`GM1/GN1/GM34/GN34/GM5/GN5` 与 m/n-tile 同步、`kv_*` helper 加 `NT`、
    fold 只由前 4 warp 覆盖 `BN≤64`）。默认 `128/2` 与历史**逐字等价**，MLA 用 `256/4`。
    ncu（S1024H2 main）：**occ 6.25%→12.49%、Warps/SM 4→8、Ipc 0.57→1.13、issue_active
    14.4%→28.1%、No Eligible 85.66%→71.95%、Duration 238.8→132.4µs、255→245 regs**，
    `lts__t_sectors_op_red` **6,684,672 逐字节不变**；墙仍是 `long_scoreboard + wait + short`。
    **main 1.84×/1.62×/1.61×**、total **0.0694/0.1332/0.1927ms**（0.0896/0.1996/0.2835），
    fp8 MLA main 现比 fp16/bf16 MLA（O46）更快。数值 vs ref 与历史逐值一致、`max_abs(8w-vs-4w)≤5e-7`，
    D=128/GQA/MQA 回归逐位不变。`--mla8w=0` 供 A/B。详见 `docs/03` §47。
     **教训：把「每 scheduler warp 数」从 1 提到 2 在 1 CTA/SM 的 MLA 上是通用杠杆**——
    fp16/bf16（O46 1.07–1.11×）与 fp8（O47 1.6–1.84×，因 fp8 的 4-warp 档寄存器 255 更死）。
13. **D=128 mma fallback 的 8-warp（O48，第 95 轮，正/负取决于 grid）**：把 O46/O47 的
    `NTH`/`NWAR` 几何用到 D=128 的 **mma fallback**（生产 `wgmma2` 已 256 线程）。判据是
    **「4-warp 的每 scheduler warp 数是否 <2」= grid 是否 ≲ SM 数**：fp16/bf16 MHA S=512
    （grid=128 < 132 SM，ncu `Active Warps/Sched 1.00`）**main 1.05×、端到端 1.08×**；
    S=4096（grid=1024）**0.88×**；fp8 因 auto split-K 把 S=512 的 grid 抬到 2048
    （4w 已 2.87 warp/sched）**0.79×**。默认保持 4-warp（opt-in `--d128w`）。顺带修掉
    O47 参数化留下、只在 `NTH>128` 触发的两个 correctness bug（`kv_prefetch/commit_pair`
    越界、`kRegDq` flush 硬编码几何）。详见 `docs/01` §14aa、`docs/01b` §6ai、`docs/03` §48。
14. **D=128 mma 8-warp 自动档默认化（O49，第 96 轮，正结果）**：把 O48 的 opt-in 按它自己
    推荐的 auto 条件落地——`D==128 && mma 路径 && grid ≤ SM 数`（每 scheduler 4-warp 仅 1 warp）
    默认开 8-warp；`--d128w=0/1` 仍可强制。fp16/bf16 S=512 MHA main **1.11×**、端到端
    **1.05–1.09×**（total 0.0960→0.0880ms），大 grid（S=4096/GQA）**逐位不变**；fp8 因 auto
    split-K 把 grid 抬到 2048，auto **不触发**（逐位不变），但 `--ksplit=1` 的 128 网格探针下
    8-warp main **1.34×**。教训：**「4-warp 每 scheduler warp 数 <2」是比「grid < 132」更本质的
    判据**——fp8 的 `mg.x` 已含 split-K、必须用**总网格**判据（单看 x 维会误开）。原始输出
    `src/{fp16,bf16}/fa_bwd_*_o49_*`、`src/fp8/fa_bwd_fp8_o49_*`；详见 `docs/01` §14ab、
     `docs/01b` §6aj、`docs/03` §49、`docs/04` §22。

15. **O50（第九十七轮，中性 + 正结果 + bug 修复）**：换两个不碰 occupancy/red 结构的角度。
    ① **wgmma2 的 GEMM1/2 等待拆分**（`FA_WS1`）：`wait0` → 先 `wait_group<1>` 算 P、再
    `wait0` 取 dP，用 CUDA-core 的 exp/量化掩盖 GEMM2。**fp16/bf16 小幅非负（≤1%、方向一致）、
    fp8 中性**（ncu S512 fp16：Duration 33.57→33.47µs、`red`/指令数逐字节不变）⇒ fp16/bf16 默认
    开、fp8 默认关。**教训：3 CTA/SM/8 warp 下「提前算依赖较轻的那半」拿不到额外重叠——
    `wait` 是 warp 数不足的症状，不是发射顺序问题。** ② **fp16/bf16 MLA 的 split-KV auto
    重标定**：O46 把 MLA 换成 8-warp 后，旧的「`grid*sp≈528` + `nblk` 封顶」对 S1024H2 只切到 8、
    S256H2 被 `nblk=8` 封顶到 8；改成 `max(528 目标, ≤ceil(nblk/2))`、去封顶 ⇒ S1024H2 main
    **1.033×**、S256H2 **1.063×**、S512H4/S512H2 不变，端到端 1.02–1.03×，数值逐位不变。
    ③ **修复 bf16 单文件版**：`fa_bwd_bf16_mma_onefile.cu` 自 O35/O36 加主 kernel TMA 后
    一直缺 host 侧 `make_lse_map`/`make_main_map`、**编译不过**；本轮补回（逐字节 dtype 参数化
    自 fp16）⇒ 重新可编译、数值逐位一致。详见 `docs/01` §14ac、`docs/01b` §6ak、`docs/03` §50、
    `docs/04` §23。

16. **O51（第九十八轮，正结果，D=512 默认）**：fp8 MLA（D=512）主 kernel 的 **K/V `cp.async`
    回填流水**。fp8 MLA 走 mma 后端、K/V 每 tile 同步载入（`NPU=16` 禁用了 O3 寄存器预取），
    ncu 头号 stall 是 `long_scoreboard`。改动只碰搬运：**K 双缓冲**（GEMM1 前发下一 tile 的 K）、
    **V 单缓冲 + 后段回填**（GEMM1/2 后发，把 `Ap` 从 `Vs` 拆开）、`kp_build_rows` 从行主序 Ks
    重建 Kp；smem 207.9→229.9KB，仍 1 CTA/SM。**main 1.02–1.04×**（S256H2/S512H2/S512H4/
    S1024H2），`long_scoreboard` 1.97→1.72、指令 −2.0%、`red` 扇区逐字节不变、数值 vs ref
    与历史同量级（差异仅 atomic 次序）；D=128 与 varlen 回归逐位不变。**教训：fp8 MLA 是
    mma+1 CTA/SM，K/V 载入的「零成本重叠」只值 ~2–4%；上面是 `short_scoreboard`/`wait` 的
     mma 依赖延迟 + 1 CTA/SM，仍受 smem 硬约束（2 CTA/SM 不可达）。** 详见 `docs/03` §51、
     `docs/04` §24。

17. **O52（第九十九轮，正结果，varlen MLA 默认）**：**把 O46/O47/O51 的 MLA 优化搬进 varlen**。
     O46（8-warp）与 O47（fp8 8-warp）/O51（fp8 K/V 回填）此前只落在**定长** `D=512` 路径，
     `run_varlen` 的 MLA 主 kernel 仍是 4-warp/2×2。O52 只改 host（`run_varlen` 加 `mla8w`/
     `mla_kvp`，`D==512` 分支三档选择 4w / 8w / 8w+kvpipe，主函数透传 `--mla8w=`/`--mlakvp=`，
     末尾加 `[O52 A/B]`），**device 一行未改**（早由 O46/O47/O51 参数化）⇒ 单/两文件 device
     仍逐字一致。**main 1.5–1.9×（fp8 b1_t512 1.78× / b3_t1792 1.86×；fp16/bf16 1.5–1.6×）；
     端到端 fp8 1.86×/1.97×、fp16 1.56×/1.44×、bf16 1.55×/1.43×**；ncu（fp8 b1_t512）
     Duration 116.6→59.5µs、warps_active 6.20%→12.39%、Ipc 0.11→0.23、`red` 扇区**逐字节不变**；
     数值 vs ref 同量级，`max_abs(8w-vs-4w)` dq~1e-7/dk,dv~1e-6（仅 atomic 次序），D=128 varlen
     回归逐位不变。**教训：一个只在定长路径落地的优化（8-warp / cp.async 回填）要显式检查
     varlen 分支是否也吃到——O46/O47/O51 连续三轮都漏了 `run_varlen`。** 附带「发现」非 causal
     （full）MLA varlen 偏差——**O53 已更正为对拍脚本漏传 `--full` 的假警报**（见下条）。详见
     `docs/03` §52、`docs/01` §14ad、`docs/01b` §6al。

18. **O53（第 100 轮，正结果，varlen MLA 默认 auto）**：**把定长 MLA 的 N 方向 split-K
     搬进 varlen**（fp16/bf16，host-only）。O52 把 8-warp 搬进 varlen 后，主 kernel 仍是「单
     CTA 扫整条 K」；`D=512/BM=32` 的 base grid 在 `b1_t512_h2` 只有 16 CTA、`b3_t1792_h2`
     192，而 smem 207.36KB 锁死 1 CTA/SM ⇒ `Waves 0.48`、SM 空转。`run_varlen` 加
     `mlaksplit`，`D==512` 按定长 O44/O50 同款 auto（target `grid*sp≈528` + 每 m 块 K 切 ≈2 份、
     cap 16）选 `mla_ks_eff`，`mg.x *= mla_ks_eff`，两处 `launch_bwd_mma<512,32,32,1,...>` 传
     `mla_ks_eff`（dQ 走跨 CTA `red_add2`）；`main` 透传 `--mlaksplit=`，末尾加 `[O53 A/B]` sweep。
     **device 一行未改**（O44 早已支持，`ksplit==1` 逐式退化）。**main：causal b1
     0.2350→0.0455ms（5.17×）、b3 0.5540→0.2543ms（2.18×）；端到端 fp16 3.35×/1.85×、
     bf16 3.31×/1.87×**（total 0.2740→0.0819 / 0.6523→0.3518ms，13.11/16.02 TF）；full auto
     偏大（最优 k=4/8），但相对 k=1 仍 3.6×/1.6×。ncu（fp16/bf16 b3 causal，逐项一致）：
     Duration **259µs**、**L2 81.0%** / DRAM 4.7% / Compute 19%、occ 12.2%（1 CTA/SM）、
     stall `long 2.73 + wait 2.28 + short 1.58`、L2 `red` 占 58.8% ⇒ **bound 从 O52 的「grid
     不足一个波」变为「L2 跨 CTA dQ 归约 + 访存延迟」**（与定长 O44 同结论）。数值与历史同量级、
     单/两文件逐指标一致。**教训：把一个机制从一个路径搬到另一个路径时，「数据流改造」（O52）
     与「并行度改造」（O53）要分两步、分别验证——O52 只搬了 warp 几何，grid 仍不足一个波。**
     另：**O52 记的「非 causal full MLA varlen 偏差」经查是对拍脚本漏传 `--full` 的假警报**
     （显式 `--full` 后三 dtype full 全部通过：fp16 3e-4–5.5e-4、bf16 1.6e-3–3.5e-3、
     fp8 ~5e-2）——**教训：跑 full 用例时先核对输出头的 `causal=` 字段**。详见 `docs/01`
     §14ae、`docs/01b` §6am、`docs/04` §25。

19. **O54（第 101 轮，正结果，full varlen 默认 auto）**：**非 causal（full）MLA varlen 的 LSE
     走 K 维 split**。O53 把 split-KV 搬进 varlen **主 kernel** 后，`b3_t1792` full 端到端仍是
     1.013 ms，而 main-only 只有 0.392 ms —— **LSE 占 60%**。原因：非 causal 的 MLA LSE 一直走
     O1 的 `lse_mma_kernel<512>`（一个 CTA 一个 m 块、**无 K 维 split、标量 K 载入**），而 causal
     早已用带 **镜像配对 + `cp.async` 双缓冲 + O40 split** 的 `lse_mma_kernel_bal`。full 下各 m
     块工作量相同（无需配对），但缺 split/双缓冲 ⇒ `b3` base grid 仅 `16·2·3=96 < 132 SM`、
     单 CTA 顺序扫 16 tile。改动：给三 dtype 的 `lse_mma_kernel_bal` 加模板 **`bool FULL=false`**
     （`FULL=true`：`grid.x=nblk`、一个 CTA 一个 m 块、`ncols=len`、无因果掩码；`FULL=false`
     经 `if constexpr` 化简出与历史**逐位相同**的代码）；host `run_varlen` 的 `D==512&&!causal`
     分支在 `lse_split_eff>1` 时走它 + `lse_split_merge_kernel`。auto：fp16/bf16 目标 384
     （b1→8、b3→4），fp8 目标 768（b1/b3→8）。**LSE 9–14×、端到端 fp16/bf16 2.17×、fp8 ~2.1×**：
     fp16 total 1.013→**0.468ms/12.05TF**、b1 ~0.28→**0.102ms**；bf16 同；fp8 **0.346ms/16.30TF**。
     数值 vs ref 与 old 同量级（split 只改 fp32 求和次序）；causal b3 与 D=128 full varlen 回归
     **逐位/同量级不变**，单/两文件逐指标一致。ncu（fp16 FULL 版 LSE，b3 split4）：Duration
     40.99µs、**L2 24% / Compute 20.6% / DRAM 5.4%**、smem 199.68KB → 1 CTA/SM、**Waves 2.91**、
     No Eligible 74.1%、fixed-latency stall 37.3% ⇒ **bound = 低 occupancy + fixed-latency**。
     **教训：把一个路径已有的优化（split+双缓冲）补到另一路径（full）时，别被「full 工作均衡、
     无需镜像配对」迷惑——`bool FULL` 一个模板参数就够，且默认档必须编译出逐位相同的代码以保回归。**
     顺带修复单文件 `fa_bwd_bf16_mma_onefile.cu` 缺 `make_lse_map`/`make_main_map`（`-DFA_TMA`
     构建一直编译不过）的既有 bug。详见 `docs/01` §15、`docs/01b` §6an、`docs/03` §53、`docs/04` §26。

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
