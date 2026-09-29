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

20. **O55（第 102 轮，正结果，full varlen 默认）**：**varlen MLA full 的 split-KV auto 重新标定**。
      O53 的 split-KV auto 对 causal/full 用同一目标（`base*sp≈528` + 每 m 块 K 切 `nt_cap/2`），
      但 `[O53 A/B]` sweep 显示 **full 的最优 split 明显更小**——full 主 kernel 每 m 块工作量相同，
      base grid 靠少量切分即可铺满一个波（b1 `base=32`，k=4→128 CTA≈132 SM）；再切到 16 只是
      **重复读 Q/dO + 增加 dQ 跨 CTA atomic**（纯亏）。host-only：full 分支 `target=132`（1 个波）、
      `sp_min=2`；**causal 分支逐字不变以保回归**。**b1 full main 0.0745→0.0681ms（1.09×）、
      同 binary sweep k4 0.0647 vs k16 0.0748（1.16×）、total 0.1030→0.0958ms（1.075×）**；
      b3 full total 1.009×；causal b1/b3 与 O53 逐位/持平。ncu（b1 full main）：O53 k16 grid 256×2、
      **Waves 3.88**、77.54µs → O55 k4 grid 64×2、**Waves 0.97**、**68.00µs（1.14×）**。
      **bound 仍是 L2（dK/dV 跨 CTA red）+ 1 CTA/SM 低 occupancy**；本次是**去过度切分**。
      **教训：`split-K` 的 auto 目标必须按「每 m 块工作量是否均衡」分路径标定——causal 的镜像配对
      每 CTA 工作量随 m 变化、需要多切几个波来均衡，full 均匀时只需一个波；否则过切反而更慢。**
      单/两文件 host 同步、device 一行未改。详见 `docs/01` §15b、`docs/01b` §6ao、`docs/04` §27。

---


21. **O56（第 103 轮，混合/负结果，opt-in）**：把 O46/O47 的「1 CTA/SM 时 4→8 warp」杠杆
     搬到 **full MLA varlen 的 LSE**（O54 的 `lse_mma_kernel_bal<512,1,true>`）——模板参数化
     `<HD,PIPE,FULL,NTH,LBN_>`，8-warp 用 `<512,1,true,256,32>`（LBM=128/LBN=32/PIPE=1，
     smem 199,680B）。**同 session A/B：长 K 的 b3_t1792 LSE 1.09×（0.0410→0.0374ms）、端到端
     +0.8%；短 K 的 b1_t512 LSE 0.80×、端到端 −9.7%**。ncu：8-warp 把 `sm__warps_active`
     6.25%→12.49%（精确 2 warp/scheduler）、Duration 41.1→37.2µs，但 `LBN=32` 使 tile/barrier
     翻倍、`short_scoreboard` 0.90→2.59。**教训：主 kernel 的「8-warp」靠的是 wgmma/ldmatrix
     的访存并行度，LSE 已偏 compute/softmax epilogue，减半 LBN 又引入额外 barrier ⇒ 该杠杆
     **不能无条件移植**；只有 K 足够长（tile 多）才回本。** 默认 opt-in（`--lse8w=0`），
     代码与 A/B 留档。详见 `docs/01` §15c、`docs/01b` §6ap、`docs/04` §28。

22. **O57（第 104 轮，混合结果，opt-in）**：**full MLA varlen 的 LSE 真正冲 2 CTA/SM**（O56
     的姊妹实验）。O54/O56 的 FULL LSE smem 恒 199,680B ⇒ 1 CTA/SM。本轮把 smem 压到
     99,840B（`≤232448/2`）让**两个 CTA 同驻一个 SM**：4-warp 固定 LBM=64，唯一可行的是
     **cfg6 `P1/LBN16`**（保留 cp.async）与 **cfg5 `P0/LBN32`**（丢双缓冲）。host-only
     （`--lseocc=5/6`，O56 已把 `<HD,PIPE,FULL,NTH,LBN_>` 参数化），`[O57 A/B]` 同 binary 扫参。
     **丢 cp.async 灾难性**（cfg4/5：b3 0.041→0.104/0.068）；**保留双缓冲的 cfg6 occupancy
     精确翻倍**（ncu `sm__warps_active` 6.25%→10.51%），但 LBN=16 把 tile/barrier 变 4×，
     `short_scoreboard` 0.89→1.56 ⇒ **长 K 小胜（b3 1.04×）、短 K 反负（b1 0.91×），且长 K 仍
     不及 O56 的 8-warp（1.08×）**。**教训：LSE 的墙是 compute/softmax + smem→mma 的 tile 级
     依赖，不是可被 CTA occupancy 掩盖的访存延迟——「压 smem 换 2 CTA/SM」在 LSE 上不成立。**
     默认 `--lseocc=0`，数值 vs ref 与 O54–O56 逐位相同。详见 `docs/01` §15d、`docs/01b` §6aq、
     `docs/04` §29。

23. **O58（第 105 轮，正结果，causal varlen 默认）**：**causal MLA varlen 的 LSE 冲 2 CTA/SM**
     ——补齐 O56/O57 只做 **full** 的缺口。O56/O57 在 full 上把「8-warp / 压 smem 换 2 CTA/SM」
     判为混合/负；本轮把它搬到 **causal 的镜像配对版** `lse_mma_kernel_bal<512,1>`（fp16/bf16
     host-only；fp8 顺带把 device 参数化为 `<HD,PIPE,FULL,NTH,LBN_>`）。**causal 默认切 cfg6**
     （`<512,1,false,128,16>`，PIPE1/LBN16；fp16/bf16 smem 99.8KB ⇒ 2 CTA/SM，**fp8 仅 49.9KB
     ⇒ 4 CTA/SM**），`--lseocc=4` 退旧默认、`--lse8w=1` opt-in；full 仍 O54 旧路。
     **结果：fp16/bf16 LSE 1.18–1.22×、端到端 b1 1.07×/b3 1.10×；fp8 LSE 1.15–1.27×、
     端到端 1.04×/1.07×**。ncu：fp16 b3 causal 同 split8 下 Duration 43.07→**30.24µs（1.42×）**、
     `sm__warps_active` 6.25%→10.38%；fp8 11.14%→18.26%、Duration 40.70→**27.07µs（1.50×）**。
     **教训：同一个「压 smem 提 occupancy」杠杆在 full 上负、在 causal 上正——镜像配对让每 CTA
     的 K 链更长，CTA 级并行度才吃得进；「full 判负」不能外推到 causal。** 数值与 O53 历史一致
     （fp16/fp8 的 dq 有跨 CTA atomic 的既有非确定性）。bound 由「低 occupancy」转向
     「compute/softmax + `wait` 固定延迟」。详见 `docs/01` §15e、`docs/01b` §6ar、`docs/03` §54、
     `docs/04` §30。

24. **O59（第 106 轮，正结果，定长 MLA 默认；含一处竞争修复）**：**把 O58 的 causal MLA LSE
     `cfg6` 从 varlen 推广到定长**。O58 只改了 `run_varlen`；定长 `run_pre` 的 `D==512 && causal`
     仍用旧默认 `<512,1>`（1 CTA/SM）。本轮把它接上并默认化（fp8/fp16/bf16；`--lseocc=4` 退旧），
     拆分 auto 目标 fp8 256→1024、fp16/bf16 132→528（cfg6 并发槽翻倍）。
     **附带修复一处 latent `cp.async` 竞争**：镜像配对循环里当切片 `nuse==0` 时循环内的
     `wait_group 0` 不执行 ⇒ 本 m 块的 Q 拷贝不被 drain，下一 m 块的 `issue_q` 又写同一 `Qs`，
     两异步拷贝竞争；LBN=16 + 大 ksplit 让空切片变多，暴露成 LSE ~1e-2 的非确定抖动
     （`cfg6 vs cfg6` 两次跑各不相同、`cfg6(sp=1) vs legacy(sp=1)` 逐位 0）。修法：每个 m 块
     末尾补 `if constexpr (PIPE) asm volatile("cp.async.wait_group 0;\n");`（`lse_mma_kernel_bal`
     与 `..._wgmma` 同修，三 dtype）。修后 `cfg6 vs cfg6` = 0、`cfg6 vs legacy` = 4.768e-7。
     **结果：LSE 1.04–1.20×、端到端 1.02–1.11×**（小 shape 收益最大）；ncu：fp8 102.14KB/2 CTA/SM
     →51.07KB/**4 CTA/SM**、Duration 1.24×；fp16 199.68KB/1 CTA/SM→99.84KB/**2 CTA/SM**、1.23×。
     **教训：压 smem 提 occupancy 的杠杆要沿「同一算法结构」推广（varlen→定长）；而一个只在高
     并行度/细 tile 下才暴露的异步竞争，会在推广时跳出来——先证明「同几何逐位、跨几何 1e-6」
     再谈收益。** 数值 D=128 回归逐位不变。详见 `docs/01` §15f、`docs/01b` §6as、`docs/03` §55、
     `docs/04` §31。

25. **P3-3c（第 109 轮，工具链，正结果）**：**`harness/fa_bwd_run.py` 一键「跑 ours + 汇总」**
     ——落实 §33 候选 ①。此前每次对拍要手写 `scripts/run.sh <host> --dir=<case> --dump=ours
     [--full] [--varlen]`（还要区分单/两文件、定长/varlen 两套构建）再手动跑 `fa_bwd_compare.py`。
     新脚本自动扫 dump 目录、按 `meta.json`（dtype/varlen/causal）选 host 与参数、调 `run.sh`
     在 `kernel_lab` 编译运行、最后汇总。**关键工程点：同一 (host, 构建配置) 只编译一次**，
     其余 case 复用已产出的可执行文件。**实测：全部 73 个 case × 单/两文件 = 146 次运行、0 失败，
     仅编译 12 次（134 次复用）**；数值与 §32/§33 的 kernel **逐位/同量级一致**（fp16 S512
     `1.671/1.771/1.899e-3`、S4096 `1.883/1.734/1.966e-3`；fp8 S512 `0.2426/0.2975/0.3735`；
     varlen 区间与 §33 一致）。**教训：对拍脚手架的「可复现」和 kernel 的「快」一样值钱——
     选 case / 选构建 / 编译复用 / 落盘 / 汇总 里任何一处靠手抄，都会随 case 数增长变成错误源。**
     详见 `docs/04` §34。

26. **P3-3d（第 110 轮，工具链，正结果）**：**`fa_bwd_compare.py --doc-table` 直出 docs/04 分组表**
     ——落实 §34 候选 ②。§1/§7 的数值表此前靠人工誊抄，已出现与实测脱节（§1.1 的 fp16 S4096
     旧值来自更早混合构建）。新 `--doc-table` 把 `ours/FA/TE` 的 `max_abs` 按
     **dtype × 家族（MHA / GQA-MQA / MLA / varlen）** 分组渲染成可直接内联的 markdown；
     家族判定由 `meta.json`（`varlen`/`D≠128`/`Hkv≠H`）自动完成，`shape_str` 带 `Hkv`/`Dv`/
     `causal|full`；默认只列两文件版、避免单文件重复列。`fa_bwd_run.py --doc-table` 顺手接上，
     一键「跑 ours + 出表」。**实测全部 73 个 case**：fp8 全部 shape、bf16 MHA、fp16 MHA S512
     与 §1/§7 **逐位一致**；唯一差异是 fp16 S4096 的 ours（旧 `1.499/1.572/2.225e-3` vs 现
     `1.883/1.734/1.966e-3`，都在 fp16 噪声内、都 ≤ TE），**今后判据以 `--doc-table` 为准**。
     **教训：文档与实测之间不该有手抄环节——把「产出表」做进 harness，文档的陈旧值会自动暴露
     并收敛。** 详见 `docs/04` §35。本节为纯 harness 增量，无新 kernel/性能/ncu 数字。

27. **P3-4-lite（第 111 轮，harness + 基线，正结果）**：**接入 FA3（SM90）变长反向基线 +
    FA 口径切到 FA3**——落实 §33/§35 候选 ①（「FA/TE 反向不支持 varlen，暂无列」）。核查发现
    本机 `flash_attn_3` 3.0.0 的**反向支持 varlen**（`flash_attn_varlen_func` 可 autograd，
    fp16/bf16、MHA/GQA、causal/full；head_dim≤128，fp8 与 D=512 不支持），而 FA2.7.4/TE2.14
    确实不支持。故 `fa_bwd_bench.py` 加 `fa3_bwd` / `fa3_bwd_varlen`（dump 落 `fa3_*`、bench 出
    纯反向 device time），`fa_bwd_compare.py` 的 `--doc-table` 默认口径改为 `fa3/TE/ours`。
    **数值**：17 个 D≤128 的 varlen case 补齐 `fa3_*`，dq/dk/dv 全在 dtype 噪声内、与 ours
    同量级或更小（定长 MHA/GQA 的 FA3 列多数也 ≤ FA2.7.4）。**性能**：varlen causal 的
    ours/FA3 = **1.87–2.95×**、full = **2.90–3.52×**，明显好于定长 MHA S4096 的 ~7×
    （FA3 变长在短序列效率低，分母小）。**教训：别把「某实现不支持」当永久结论——同代的不同
    版本（FA2 vs FA3）能力边界不同；补一条基线列，往往就能让此前「只有 ours」的对照变得完整。**
    详见 `docs/04` §36；原始输出 `src/fa_bwd_p111_varlen_fa3_perf.out.txt`、
    `src/fa_bwd_compare_p111_doc_table.md`、`src/fa_bwd_fa3_varlen_bench_p111.out.txt`。
    本轮为 harness 增量，无新 kernel/ncu 数字。

28. **P3-4b（第 112 轮，harness + 基线，含一处口径更正）**：**`fa_vs_te_bwd_only.py`
    补齐 varlen 纯反向基线**——落实 §5.27 候选 ①。该脚本是用户指定的**纯反向口径**基准
    （forward 建图放在计时区外，只对 `autograd.grad` 计时），此前只有定长 `SHAPES`。
    本轮加 `VARLEN_SHAPES`（MHA 不齐/等长、GQA q32-kv8、强倾斜、full）、`bench_fa_varlen`
    / `bench_fa3_varlen`（forward 移出计时区）、`ref_varlen`+`--verify`（小 shape 对拍 ref
    证明新列可信），main 固定输出「定长表 + varlen 表」。**TE2.14 变长反向在本容器报错
    （ragged QKV 需 padding mask → cuDNN err 700），varlen 表 TE 列标 NA。**
    **两项发现**：① **FA2.7.4 的变长反向其实可用**（`flash_attn_varlen_func` 可 autograd，
    causal/full、MHA/GQA 均可），数值与 FA3 逐点一致（fp16 `1.56/1.38/1.78e-3`、bf16
    `9.41e-3/1.13e-2/1.82e-2`）——推翻旧记「FA2 反向不支持 varlen」；② **纯反向下 FA3 比
    §5.27 里的 varlen 数字快 1.6–1.7×**（[1024]×4 causal `0.1480` vs `0.2409`ms），因为
    §5.27 的 `fa_bwd_bench.py` `bench_case_varlen` 把**含 forward** 的 `fa3_bwd_varlen()`
    整段计了时。**故 ours/FA3 应从 §5.27 的 1.87–3.5× 更正为 3.0–4.8×**（纯反向口径）。
    **教训：`device_time(fn)` 里 `fn` 是否包含 forward，是「纯反向」与「前向+反向」的分水岭；
    对标脚本必须先钉死建图位置再谈吞吐。** FA3 varlen 稳定快 FA2 1.69–2.13×；fp8/MLA FA3
    不支持，最重点的 fp8 口径仍以 ours/TE/ref 为准。本轮为 harness 增量，无新 device 代码。
    详见 `docs/04` §37；原始输出 `src/fa_bwd_p112_varlen_fa2_fa3_te.out.txt`。

29. **P3-4c（第 113 轮，工具链 + 基线，正结果）**：**把全站基线统一到「纯反向」口径**
     ——落实 §5.28 候选 ① / backlog 最后一条 `[ ]`。`fa_bwd_bench.py` 的 `bench_case` /
     `bench_case_varlen` 一直把 `*_bwd()`（**含 forward**）整段放进 `device_time`，
     与用户指定的 `fa_vs_te_bwd_only.py` 纯反向口径不一致（§36.2 的 varlen FA3 因此偏慢
     1.6–1.7×）。本轮加 `make_*_bwd_only()`（forward 在计时区外）并把 `bench` 默认切过去，
     `--with-fwd` 保留旧口径做 A/B。**实测（同 binary A/B）**：forward 占 FA3 S512 `71%`、
     S4096 `43%`，而 **fp8 TE 占 `95–183%`**（旧 fp8 口径几乎把前向也算成反向）；
     纯反向下 FA3 MHA S4096 `0.3236ms/849.5TF`、varlen `[1024]×4` causal `0.1480ms/232.1TF`，
     与 `fa_vs_te_bwd_only.py` 一致。**更正 ours/参考比值**：fp8 S4096 ours/TE 由 ~4.0× 改为
     **7.85×**、varlen `[1024]×4` ours/FA3 = **3.05×**、fp16 S4096 ours/FA3 = **5.95×**。
     **教训：`device_time(fn)` 里 `fn` 的建图位置必须钉死在脚本入口——两个都号称「反向基准」
     的脚本，口径能差出一整个 forward；且不同后端的 forward 占比天差地别（fp8 最重）。**
      数值零变化（`fa_bwd_compare.py` 73 case 重扫逐位一致）；详见 `docs/04` §38。
      原始输出 `src/fa_bwd_p113_*.out.txt`、`src/fa_bwd_compare_p113_summary.out.txt`。

30. **P3-3e（第 114 轮，工具链，正结果）**：**docs/04 数值表自动同步（`--doc-table
     --apply/--check`）**——落实 §38 候选 ②。§35 的 `--doc-table` 已能直出分组表，但仍是
     「人工把输出粘进文档」，且已发生过陈旧（§32 的 fp16 S4096 旧值）。本轮在 docs/04 里放一个
     被标记（`<!-- BEGIN/END:auto-doc-table -->`）围起来的自动块，并给 `fa_bwd_compare.py`
     加 **`--apply PATH`（原地改写块）** 与 **`--check PATH`（diff 校验、陈旧则退出码 1）**；
     `fa_bwd_run.py` 加 `--doc-table-apply` / `--doc-table-check` 一键化。**容差**：fp8 的
     dK/dV（及 split-K 的 dQ）走跨 CTA `atomicAdd`，末位会抖动，故 `--check`/`--apply` 按
     「逐行结构相同 + 数值在 `--rtol`（默认 `5e-3`）内」判等价；`--rtol 0` 可做逐值复核。
     **实测**：本轮重跑全部 fp8（28 case）后，`--check` 在 rtol=5e-3 下 OK、`--rtol 0` 精确报出
     原子噪声 `7.108e-01↔7.107e-01`；把某格改成 `9.999e-03` 后 `--check` 退出码 1 且打印
     unified diff；`--apply` 幂等（rtol 内不改写文件）。另修 `IMPL_LABEL` 缺 `fa3`（此前
     `--doc-table` 把 FA3 列显示为裸 `fa3`）。**教训：把「产出表」再往前一步到「自动改写 + 可
     校验」，文档与实测之间就不再有手抄环节，且原子非确定性必须用容差吸收，否则 CI 会抖。**
     本轮纯 harness，device 一行未改。详见 `docs/04` §39；原始输出
     `src/fa_bwd_p33e_doc_table.md`、`src/fa_bwd_p33e_doc_table_check.out.txt`。

31. **P3-3f（第 115 轮，工具链 + 验证，正结果）**：**单/两文件实现一致性自动报告**
     ——把每轮都要手工做的「单文件 vs 两文件逐元素一致性」核对应固化为 harness。
     `fa_bwd_compare.py` 新增 **`--consistency`**（默认 `ours vs ours_sf`，逐 case 算 `max|A-B|`、
     按 dtype 汇总、`--ctol` 可作 CI 回归门；不需要 ref/GPU）；`fa_bwd_run.py` 加
     `--consistency`/`--consistency-tol` 一键化。**顺带修一个真 bug**：`--no-run` 文档说
     「跳过 kernel、只重新汇总」，但主循环从未检查它——仍会编译+运行全部 case（全量扫 146 次）；
     现在真正跳过且不改写上一轮日志。**实测全部 73 case × 两形态**：worst `max|ours-ours_sf|`
     fp16 3.906e-3 / bf16 7.812e-3 / fp8 9.537e-6，全部 = 1–2 个 dtype ulp、来自跨 CTA
     `atomicAdd` 次序（`dq` 多数逐位相同），**无实现分歧**。并刷新纯反向基线
     （FA3 MHA S4096 `0.3237ms/849TF`、TE `0.4388/626`）与 ours（fp16 S4096 `1.2582ms/109.2TF`、
     fp8 `1.9466ms/70.6TF`）；ncu 复核 fp8 main 仍 **L2 78%（red 114.5M 扇区）+ mma 依赖**。
     **教训：两形态「device 逐字同源」这件事应当由脚本持续背书，而不是每轮手抄——把一致性
     检查自动化后，任何一次单文件同步漏改都会被 `--ctol` 立刻抓住。** 本轮为 harness/验证增量，
     device 一行未改。详见 `docs/04` §40；原始输出 `src/fa_bwd_consistency_p33f.out.txt`、
     `src/fa_bwd_p33f_fa_baseline_fp16.out.txt`、`src/{fp16,fp8}/fa_bwd_*_p33f_*.out.txt`。

32. **P3-3g（第 116 轮，工具链 + 验证，正结果）**：**把单/两文件一致性 gate 接进端到端回归**
     ——落实 §5.31 候选 ①。§5.31 的 `--consistency` 仍是「人工记得去调」的工具；本轮让
     `fa_bwd_run.py` 的**全量扫默认在结束时自动 gate**（`--no-consistency` 可关、`--impls
     twofile` 只跑一边时跳过）。关键改动是容差**从单个全局标量升级为按 dtype**：
     fp16/bf16/fp8 的 ulp 差 3 个数量级（worst fp8 9.5e-6 vs bf16 7.8e-3），全局标量必然
     放过 bf16 的错误或误杀 fp8 的正常原子噪声；故 `--ctol auto` 用
     `{fp16:1.6e-2, bf16:3.2e-2, fp8:1e-4}`（≈实测 worst 的 2–4×），报告逐 dtype 打
     `gate[...] -> OK/FAIL`。**实测全量 73 case**：fp16 `3.906e-3` / bf16 `7.812e-3` /
     fp8 `9.537e-6`，全部 OK、退出码 0；**负向验证**（`--ctol 1e-3` 与 `--consistency-tol
     1e-9`）均退出码 1，确认 gate 真的接上了；并在容器里真编译真跑 2 个 case 证明自动 gate
     挂在真实运行路径上。刷新纯反向基线（FA3 MHA S4096 `0.3243ms/848TF`、TE `0.4399/625`、
     varlen `[1024]×4` causal `0.1475ms/233TF`）。**教训：校验工具「做出来」和「接进回归」
     是两件事——只有挂进默认出口并配好按量级的容差，它才会在漏改时真正变红；负向测试是
     验证「gate 会不会拦」的唯一办法。** 本轮 harness 增量、device 一行未改。详见
      `docs/04` §41；原始输出 `src/fa_bwd_consistency_p33g.out.txt`、
      `src/fa_bwd_p33g_gate_negative.out.txt`、`src/fa_bwd_p33g_fa_baseline_fp16.out.txt`。

33. **P3-4d（第 117 轮，工具链 + 验证，正结果）**：**CI 单一入口 + Hopper 快路入标准 harness**
     ——落实 §5.32 候选 ②。`fa_bwd_run.py` 加 **`--ci`**（跑完后自动 `fa_bwd_compare.py --check
     docs/04`，表陈旧即 rc=1；一致性 gate + doc-check 汇总到 `src/fa_bwd_ci.out.txt`）、
     **`--hopper`**（定长/变长都走 `-DFA_WGMMA -DFA_TMA -lcuda`；用**独立前缀** `ours_hp/ours_sf_hp`，
     绝不覆盖默认 mma 的 `ours/ours_sf`——因为 wgmma/TMA 与 mma 的数值差可达 O(1e-1)，混用会
     误触一致性 gate）、**`--perf-baseline <dtype>`**（容器内跑 `fa_vs_te_bwd_only.py` 落盘）。
     `scripts/ci.sh` 是一行固定入口。**实测**：`--no-run --ci` 在 73 case 上全绿（fp16 3.906e-3 /
     bf16 7.812e-3 / fp8 9.537e-6，`--check` OK 194 行），**负向测试**（tol 1e-9、改错一格）均 rc=1；
     现场 `--hopper` 真编译真跑 fp8 S4096：数值 2.635/2.644/3.216e-1、`ours_hp vs ours_sf_hp`
     worst 7.153e-7、total 1.9359ms/70.99TF（TE FP8 0.3025/908.7 ⇒ **6.40×**）；ncu 复核 fp8 main
     仍 **L2 red 78.6%（114.5M 扇区）+ wait/short**。**教训：把「多条校验」收口成一条 CI 命令只是
     第一步，更要防「构建配置差异」被误当作「实现分叉」——`--hopper` 必须走独立前缀。** 本轮
     device 一行未改；详见 `docs/04` §42、`src/fa_bwd_ci_p34d.out.txt`、`src/fa_bwd_p117_ci_negative.out.txt`。

34. **P3-4e（第 118 轮，device 增量，正结果/opt-in）**：**fp8 反向的确定性 dK/dV 归约**
     ——补齐 `docs/00` §4.2 第 5 条（fp16/bf16 早在 O7b 就有 `--det=1`）。把 dK/dV 的跨 CTA
     `atomicAdd` 换成「按 (Q 头, Q 块) 分片的 partial 覆盖写 + `dkv_reduce_kernel` 固定次序
     求和」（causal 下从 `jg/BM` 起求和；partial 必须按 Q 头而非 KV 头分片，否则 GQA 广播组
     互相覆盖）。**两次跑 `bitwise-diff = 0`（逐位可复现）**；`DET-vs-atomic` ~e-6（fp32 次序
     末位）。**代价 ncu 定量**：DET 主 kernel DRAM 4.3%→33.3%、L2 78.6%→36.2%、`red` 114.5M→
     1.57M 扇区，`dkv_reduce_kernel` **731.6µs / DRAM 91.5% / 3.07 TB/s**（纯带宽 bound）；
     因 DET 要求 `ksplit=1`，相对调优默认档 S=4096 约 **1.5–1.8×**。**教训：确定性的售价就是
     「把 L2 原子换成一次性 DRAM partial 写读」——在 atomic 归约已是墙的工作点上，确定性与
     性能直接对立；opt-in 而非默认是对的。** 单/两文件 device 逐字同源（`sync_onefile_device.py`
     核对）；默认路径一行未改（回归数值与历史逐位一致）。详见 `docs/03` §56、`docs/04` §43。

35. **P3-4f（第 119 轮，device 增量，正结果/opt-in）**：**把 `--det` 扩展到 split-K（dQ 也走
     partial）**——P3-4e 为「每 `(h,mblk)` 一个 CTA 写 partial」把 ksplit 锁死为 1，白扔了
     split-K。分解后发现 **dK/dV 的 partial 天然无需 part 维**（ksplit 切的是同一 m 块的 K
     tile，一个 `(mblk,jg)` 只属于一个 part），真正需要分片的只有 **dQ**。于是新增
     `dq_reduce_kernel<HD>` + dQ partial（`[row][h][part]`），`--detk=N` 扫 ksplit。
     **所有 ksplit（含 dQ）两次跑 `bitwise = 0`**；DET 在 **k=4 触底**，相对 k=1 提速
     S512 **1.31×** / S4096 **1.11×**。ncu：DET 主 kernel（S4096 k=4）1.72ms/DRAM 42.3%、
     `dkv_reduce` 730.7µs/DRAM 91.6%/3.07 TB/s、`dq_reduce` 55µs ⇒ **bound 仍是 reduce 的
     纯 DRAM 带宽**（固定成本与 ksplit 无关，故 k 再大反而回升）。**教训：先想清楚「哪些
     中间量的竞争域到底多大」再下「必须 ksplit=1」的结论——一个看似必要的限制常常只对一半
     的累加器成立（这里 dQ 需要分片，dK/dV 不需要）。** 单/两文件 device 逐字同源；默认路径
      数值逐位不变。详见 `docs/03` §57、`docs/04` §44。

36. **P3-4g（第 120 轮，device 增量，正结果/opt-in）**：**把 `--det` 扩到 Hopper TMA 快路**——
     P3-4e/f 的 `--det` 只挂默认 mma 路径；`--hopper`（`-DFA_WGMMA -DFA_TMA`）的主路径是
     Q/dO/K/V 全 4D-TMA 的 `kvtma` kernel，dK/dV 仍是跨 CTA `atomicAdd` ⇒ 快路无法确定性复现。
     因为 DET 的 epilogue 早在 `fp8_mma_body` 里，**device 一行数学都没改**，只把 `DET` 与
     `dk_part/dv_part/nblk/dq_part` 从两个 TMA 壳透传进 body（默认 `DET=false` ⇒ 既有 TMA 逐位
     不变）。**三 shape（S512/S1024-GQA/S4096）× 单/两文件，两次跑 `bitwise dk/dv = 0`**，与
     atomic 差 e-7–e-6；`ours vs ref` 与历史逐位一致；`--no-run --ci` 73 case 全绿。代价同 P3-4e/f：
     DET 主 kernel 把 L2 `red`（78.6%）换成 DRAM partial 写（38.3%），`dkv_reduce` **729.8µs /
     DRAM 91.7% / 3.07 TB/s**（纯带宽 bound），S4096 0.78×。**教训：只要把确定性做在共享的
     「计算体」而不是某个后端壳里，换后端（cp.async→TMA）时确定性几乎免费继承——新增一个后端
      只需要把模板参数透传过去，风险被 `DET=false` 默认值完全隔离。** 单/两文件 device 逐字同源；
      默认路径一行未改。详见 `docs/03` §58。

37. **P3-4h（第 121 轮，host 增量，正结果/opt-in）**：**把 `--det` 扩到 MLA（HD=512）**——
     落实第 119/120 轮「下一步候选 ①」的 MLA 部分。MLA 的 dK/dV 同样是跨 CTA `atomicAdd`，
     而且它的 dQ **无法用寄存器累加**（`kRegDq = REGDQ && HD/NTW==1`，HD=512 时 `HD/NTW=4`
     ⇒ 恒 false），所以「确定 dQ」只能靠 **ksplit=1 的单写者 `red_add2`**。好消息是
     `dkv_reduce_kernel<HD,BM>` 与 body 的 DET 分支本来就 HD 无关，**device 数学一行未改**，
     只给 `launch_bwd_main_det` 加了 `NTH/NWAR`（默认 `THREADS/WN` ⇒ D=128 逐字不变）以复用
     MLA 的 8-warp/256 几何，host 补一条 A/B。**三 shape（S256H2/S512H4/S1024H2）× 单/两文件，
     两次跑 `runs[1-2] bitwise dq/dk/dv = 0`**；`DET-vs-atomic` **dq 恒 0**（单写者，与 atomic
     逐位同值）、dk/dv e-7–e-6；`ours vs ref` 与历史逐位一致（S256H2 `2.356/2.290/3.441e-1`、
     S512H4 `2.415/2.992/4.481e-1`、S1024H2 `2.232/3.337/3.602e-1`）。代价（同 256/4 几何、
     仅 DET 一个变量）：S256H2 **1.035×**、S512H4 **0.923×**、S1024H2 **0.940×**。ncu
     （S1024H2）：DET 主 kernel 541.4µs / DRAM 3.53 / L2 10.24 / L1TEX 57.96 / Compute 3.78%、
     255 regs / occ 12.5% / Waves 0.24；atomic 主 kernel 611.4µs / DRAM 0.83 / L2 11.23%；
     `dkv_reduce_kernel` **29.8µs / DRAM 76.4% / 2.56 TB/s / occ 65.6%** ⇒ **bound 仍是 reduce
     的纯 DRAM 带宽**（与 P3-4e/f/g 一致）。**教训：先在每个 dtype/几何上核对该路径的「dQ 累加
     方式」再谈确定性——同是 fp8 反向，D=128 能用寄存器累加（ksplit>1 也确定），MLA 不能，
     于是只能锁 ksplit=1，代价与收益都随形状变号（S256 净赚、大 S 净亏）。** 单/两文件 device
     逐字同源、默认路径数值逐位不变。详见 `docs/03` §59。

38. **P3-4i（第 122 轮，device+host 增量，正结果/opt-in）**：**把 `--det` 扩到 varlen**——
     落实第 121 轮「下一步候选 ①」的最后一块。关键观察：body 的 DET 分支只依赖
     `(b,h,mblk,jg,S,nblk)`，**与定长/变长无关**——varlen 的 main kernel 早已把 `S=maxlen`
     （packed 索引由 `qbase` 定界）传进 body，故**只需 host 传 `nblk=nblk_max`**，device 数学
     一行未改；真正的新代码是 `dkv_reduce_varlen_kernel<HD,BM>`（按 `cu_seqlens` 的逐序列
     `len_b/nblk_b` 定界、输出按 packed token 定位）。`launch_bwd_main_det` 尾部加
     `WGMMA=false` 与 `cu_seqlens/mt_b/mt_m` 默认实参 ⇒ 定长调用逐字不变。**4 个 case
     （b1 单长 / b4 不齐 / b5 GQA q32kv8）× 单/两文件、ksplit=1/4，两次跑
     `bitwise dq/dk/dv = 0`**；`DET-vs-atomic` e-7–e-6；`ksplit=1` 时 dQ 逐位等于 atomic。
     代价（同 binary A/B，含 reduce）**0.71–0.91×**（ksplit=1 最轻）；varlen reduce
     **274.0µs / DRAM 87.4% / L2 85.3%**（b4_t3840）⇒ **bound 仍是 reduce 的纯 DRAM 带宽**
     （与 P3-4e/f/g/h 一致）。**教训：把 DET 的 partial 布局选成「定长式 + `S/nblk` 参数化」
     后，变长只剩「归约端按 `cu_seqlens` 定界」这一处新逻辑——好的布局抽象让第 5 个后端几乎
     零成本接入。** 单/两文件 device 逐字同源（`sync_onefile_device.py` 核对 `identical: True`），
     默认路径数值逐位不变（`--no-run --ci` 73 case 全绿）。详见 `docs/03` §60。

39. **P3-4j（第 123 轮，host 增量，正结果/opt-in）**：**把 `--det` 扩到 MLA（HD=512）的
     varlen**——落实第 122 轮「下一步候选 ②」。关键观察与 P3-4h 同：MLA 的 dQ 无法用寄存器
     累加（`kRegDq` 恒 false）⇒ 确定 dQ 只能靠 **ksplit=1 的单写者 `red_add2`**；而 dK/dV 的
     partial 布局在 P3-4i 已被抽象成「定长式 + `S/nblk` 参数化」的 HD 无关形式 ⇒
     **`dkv_reduce_varlen_kernel<512,64>` 直接可用，device 一行未改**，只补 host A/B（+76 行
     /文件名）。**3 个 case（b1 causal/b3 causal/b1 full）× 单/两文件两次跑
     `bitwise dq/dk/dv = 0`**；`DET-vs-atomic` **dq 恒 0**、dk/dv e-7–e-6；`ours vs ref` 与历史
     逐位一致（b1 causal `1.613e-1/2.238e-1/3.864e-1`、b3 causal `3.404e-1/3.436e-1/3.508e-1`、
     b1 full `5.260e-2/5.222e-2/4.218e-2`）。代价（只差 DET）**0.95–1.02×**（main 本体几乎免费），
     但 **ksplit 锁 1 相对 auto split 的主 kernel ~3.3–5.6×**（b1 auto=16、b3 auto=4）；
     reduce **48.03µs / DRAM 65.8% / L2 67.6%**（纯带宽 bound，与 P3-4e/f/g/h/i 一致）。
     **教训：确定性在 MLA 上不是「免费换归约方式」，而是「用单写者换掉 split-K 并行度」；
     要既确定又并行，得让 MLA 的 dQ 也进 partial（对唯一 CTA 的 `red_add2` 写 per-part buffer
     仍确定）——留作下一步候选。** 单/两文件 device 逐字同源、默认路径数值逐位不变。
     详见 `docs/03` §61。

40. **P3-4k（第 124 轮，device+host 增量，正结果/opt-in `--detk>1`）**：**让 MLA（HD=512）
     的 dQ 也进 partial，从而支持 DET 的 split-K**——落实第 123 轮「下一步候选 ①」。关键观察：
     非 `kRegDq` 的 dQ epilogue 是对**唯一 CTA**（固定 `(mblk,part)`）的逐 tile 写，把目标从
     `dq_acc` 换成按 part 分片的 `dq_part`（同一 partial 元素只被一个 CTA 写、CTA 内同一
     `(row,c)` 由同一线程按 nt 程序序写）⇒ **确定性天然成立**，无需寄存器累加；再接早已就位的
     `dq_reduce_kernel<512>` 按 part 固定次序求和。**device 只动 dQ epilogue 一处**（`DET=false`
     与 `ksplit=1` 逐位不变），host 把 P3-4h 的 A/B 扩成 P3-4f 同构（`dq_part` 清零 +
     `dkv_reduce_kernel<512>` + `dq_reduce_kernel<512>`）。**3 个定长 MLA shape（S256H2/S512H4/
     S1024H2）× 单/两文件、ksplit=1/4/8/16 全部 `runs[1-2] bitwise dq/dk/dv = 0`**；
     `DET-vs-atomic` k=1 时 dq 恒 0、k>1 时 e-7。**DET 在 k=4 触底**（S1024H2：k=4 0.2589 <
     k=8 0.2650 < k=16 0.2738），**相对旧「锁 k=1」主链 2.4–2.8×**（0.7256→0.2589ms），把
     P3-4h/j 白扔的 split-K 并行度收回一大截；但仍比 atomic 慢 0.61–0.73×（partial 写+两次
     reduce）。ncu（S1024H2 k=4）：DET 主 kernel 133.34µs/L2 51.5%/occ 12.5%/Waves 3.88、
     `dkv_reduce_kernel<512,64>` **29.60µs / DRAM 77.1% / L2 75.8% / occ 65.3%**、
     `dq_reduce_kernel<512>` 8.74µs ⇒ **bound 仍是 reduce 的纯 DRAM 带宽**。默认路径与
     D=128 回归逐位不变。详见 `docs/03` §62。

41. **P3-4l（第 125 轮，host 增量，正结果/opt-in `--detk>1`）**：**把 MLA（HD=512）varlen 的
      DET 也接上 split-K（DET 候选 ① 收口）**——落实第 124 轮「下一步候选 ①」。观察与 P3-4k
      同构：varlen body 的 DET 分支只依赖 `(b,h,mblk,jg,S,nblk)`，dQ 的 per-part partial 用
      `qbase+qi`（就是 packed 全局 q token，与定长 `dq_reduce_kernel` 的 `row` 语义逐字相同）
      ⇒ **device 一行未改**，host 只补 `dq_part` 分配/清零 + `dq_reduce_kernel<512><<<(T,H),512>>>`。
      **b1_t512（k=1/4/8/16）、b3_t1792（k=1/4/8/16）、b1 full（k=4）× 单/两文件全部
      `runs[1-2] bitwise dq/dk/dv = 0`**；`ksplit=1` 时 `DET-vs-atomic` dq 恒 0、k>1 时 e-7；
      `ours vs ref` 与历史逐位一致（b1 causal `1.613e-1/2.238e-1/3.864e-1`）。**DET 在 k=8 触底**
      （varlen 短序列 k=4 并行度未吃饱），**相对 P3-4j「锁 k=1」b1 3.69×、b3 2.13×**；ncu
      `dkv_reduce_varlen_kernel<512,64>` 48.13µs/DRAM 65.7%、`dq_reduce_kernel<512>` 23.33µs/
      DRAM 81.7% ⇒ **bound 仍是 reduce 的纯 DRAM 带宽**。`--no-run --ci` 73 case 全绿、默认路径
      逐位不变。MLA d512 的 FA2/FA3/TE 反向均不支持故无同 shape 对标。详见 `docs/03` §63。

42. **P3-4m（第 126 轮，host 增量，正结果/opt-in `--det`）**：**把 `--det` 接进 MLA 的
      K/V `cp.async` 回填流水（kvpipe）**。`fp8_mma_body` 的模板里 `KVPIPE && DET` 本就并存，
      但 host 的 `launch_bwd_main_det` 一直写死 `KVPIPE=false` ⇒ MLA（HD=512）的 DET 主 kernel
      走的是「每 tile 同步载入 K/V」的旧路，白扔 O51 的 1.78–1.86×。本轮加模板参
      `bool KVPIPE=false`（`kSmem` 选 `smem_bytes_kvpipe`，MLA 229888B ≤ 232448）、P3-4k/P3-4l
      的 A/B 各加一个 kvpipe DET 变体，**device 一行未改**。5 shape × 单/两文件
      **kvpipe DET 与非 kvpipe DET 逐位相同**（`kvpipe-vs-非kvpipe=0`、`runs[1-2]=0`）；
      非 kvpipe→kvpipe 1.018–1.095×（b3_t1792 0.3550→0.3258，单文件 1.095×）；ncu 主 kernel
      243.0→220.2µs、`long_scoreboard` 3.93→3.25（机制同 O51）。顺带修
      `sync_onefile_device.py` 的既有坑（单文件 device 区 marker 在 `#include "../fa_bwd_dump.h"`
       之前，每次同步都会误删该 include），把它移到 marker 之前。`--ci` 73 case 全绿、默认路径
       逐位不变。详见 `docs/03` §64。

43. **P3-4o（第 128 轮，device+host 增量，负结果（性能）/正结果（显存），opt-in `--partcompact`）**：
     **varlen `--det` 的 dK/dV partial 试换 compact per-sequence 布局**——落实第 127 轮候选 ① 的
     「compact per-sequence offset」。`fp8_mma_body` 的两个 DET 写点与两个 varlen 归约 kernel
     加行前缀和 `part_base`；非空时按 `part_base[b]+(h*nblk_b+mblk)*len_b+jg` 编址，空时逐字
     退化旧 `maxlen`-strided 布局。**求和集合/次序不变 ⇒ compact-vs-legacy 逐位相同**（`runs[1-2]`
     与 `DET-vs-atomic` 不变；默认路径数值逐位不变）。**分配大幅缩小**（b4_t3840 D128 2147→713MB
     =33%、b8_t2904 4295→575MB=13%、D512 b3 201→88MB=44%），但**耗时中性偏负 0.944–0.975×**；
     ncu（b4_t3840 融合 reduce）两布局都 **DRAM/L2 带宽 bound**（旧 373.76µs/DRAM 86.9%、compact
     384.86µs/84.4%）。**教训：reduce 本就只读被写过的条目，maxlen-stride 的空洞不产生额外
     DRAM 流量；compact 只缩地址跨度，拿不到带宽收益，反而破坏跨序列的通道/页并行。真正的
     「减 partial 字节」只能靠 BM=128 跨 warpgroup 偏和（fp8 撞 smem 硬墙，见「阻塞」）。**
     故默认保持旧布局，`--partcompact` opt-in 供显存受限场景。`--ci` 73 case 全绿（顺带按项目
     工作流 `--doc-table-apply` 同步 docs/04 两处末位原子次序噪声）。详见 `docs/03` §66、
     `docs/00` §4.2。


44. **O60（第 129 轮，device+host 增量，混合结果：reduce 正 / 端到端中性偏负，opt-in A/B）**：
     **把 fp16 `--det` 的 dK/dV partial 从 fp32 降精度存 fp16**——落实第 128 轮「降 partial
     字节」候选。`dkv_det_store` 加 `__half*` 重载 + `dkv_det_store_p<P16>(base, off, …)`（**坑：
     `off` 必须在 `__half*` 上做加法，否则字节地址翻倍、越界**）；`fa_bwd_fp16_wgmma2b_kernel`
     加 `DET_HALF` 模板参、4 处写点透传；`dkv_reduce_kernel<HD, P16>` 逐元素 `__half2float` 进
     fp32 累加器，**索引/求和集合/次序逐字不变**。host `--det` A/B 扩成 atomic/DET(fp32)/DET(fp16)
     × main-only/reduce-only。**两种 partial 都 `runs[1-2]` 逐位可复现**，fp16-vs-atomic 仅差
     fp16 舍入 0.95–2.4e-3；默认路径数值逐位不变；单/两文件逐指标一致。
     **实测（同 session，event）**：**reduce 单向 1.27×（S512）/1.50×（GQA kv4）/1.66×（S4096）**
     （S4096 0.3877→0.2341ms），**但 DET 主 kernel 慢到 0.84–0.87×**（S4096 0.8004→0.9496ms），
     端到端只 1.006×（S4096）/0.93–0.97×（S512/GQA）。ncu（S4096）：reduce DRAM read
     1.11→0.55GB、L2 read 扇区 51.9M→26.0M ⇒ 纯带宽 bound、1.66×；**main 的 DRAM 写字节减半
     （1.12→0.57GB）但 store 扇区数一字不变（35,651,584）**——DET partial 写是**扇区粒度 bound
     而非字节 bound**（每 `(j,row)` 仅 16B 落不满 32B 扇区），半宽写更碎 + `__floats2half2_rn`
     转换 ⇒ 主 kernel 慢 19%。**教训：降精度只对「读被写过的字节」的 reduce 成立；要让写侧也降
     字节，必须让一次 store 落满扇区——下一步候选：把相邻 `j` 的 half 拼成 32B 连续写，或 staging
     到 smem 再整行 128B 写。** 详见 `docs/01` §17；原始输出 `src/fp16/fa_bwd_fp16_*_o60_*`。

45. **O61（第 130 轮，device+host 增量，功能补齐 + 混合结果，opt-in A/B）**：**把 fp16 的确定性
     dK/dV 归约（`--det`/O7b）+ partial 降精度（O60）逐字 dtype 参数化到 bf16**——补齐三 dtype
     里 bf16 唯一缺的 `--det`。device 新增 `dkv_det_store`/`dkv_det_store_p<P16>` 与
     `dkv_reduce_kernel<HD,P16>`；`fa_bwd_bf16_wgmma2b_kernel<HD,SPLIT,DET,DET_HALF>` 加参、
     4 处写点透传；host 加 `--det` A/B。**两档 partial `runs[1-2]=0`**（确定性），bf16-vs-atomic
     仅 bf16 舍入（7.8e-3–2.2e-2），默认路径数值逐位不变（`--no-run --ci` 73 case 全绿、
     `--check docs/04` OK 194 行）。**性能与 fp16 O60 逐项同构**：reduce 单向 1.27×（S512）/
     1.50×（GQA kv4）/1.66×（S4096）、纯 DRAM 带宽 bound（ncu 90.4%→78.9%）；**DET 主 kernel
     写侧 0.84–0.87×**——bf16 把 DRAM 写字节减半（ncu 44.5%→20.3%）但 `st.global` **store 扇区数
     几乎不变**（35,653,558→35,537,820），partial 写是扇区粒度 bound，端到端 `--det` 仍中性偏负。
      对标默认路径不变：MHA S4096 FA2 379 / FA3 862 / TE 630 TF（ours total 时间 4.14× FA3）。
      详见 `docs/01b` §6at；原始输出 `src/bf16/fa_bwd_bf16_p61_*`。

46. **O62（第 131 轮，device+host 增量，正结果：确定性 `--det` 端到端转正，opt-in A/B）**：
      **DET partial 写扇区化**——落实第 129/130 轮明确点名的「唯一能同时降 reduce 与 main 写侧
      字节的候选」。O60/O61 把 partial 降到 fp16/bf16 后，每 lane 只写 4B（half2），quad 4 lane
      合起来 16B **落不满 32B 扇区**（ncu：store 扇区数 35.65M 一字不变），主 kernel 写侧反慢
      15–19%。O62 把**相邻两个列组 `j`/`j+1` 的 4B 拼成一次 8B 写**（`uint2`），并使 partial 的
      HD 维做**16 列块内置换**（`dkv_p16_perm`）让 quad 的 4×8B 恰为连续 32B；reduce 端按同一
      置换读回 ⇒ **求和集合/次序不变、数值逐位不变**。fp16/bf16 同步（bf16 单文件因
      `#include <algorithm>` 在 device 区之前、`sync_onefile_device.py` 边界启发式误判，改手工
      同步）。**store 扇区数精确减半**（S4096 35.65M→18.35M）；DET 主 kernel 写侧从
      **0.84–0.87× 翻正为 1.10–1.17×**，reduce 仍 1.24–1.66×，**端到端 `--det` 首次全面快过
      非确定 atomic**（fp16 DET/atomic=1.19×/1.09×/1.03×；bf16 1.19×/1.10×/1.02×）。默认路径
      一行未动；全量 `--ci` gate 全绿（fp16 7.812e-3 / bf16 3.125e-2 / fp8 7.629e-6）、
      `--check docs/04` OK 194 行。详见 `docs/01` §18、`docs/01b` §6au、`docs/00` §4.2。
      原始输出 `src/fp16/fa_bwd_fp16_*_o62_*`、`src/bf16/fa_bwd_bf16_*p62*`。

47. **F1（第 132 轮，harness-only，正结果，默认化）**：**fp8 主路径默认切到 Hopper**
      （wgmma GEMM1/2 + Q/K/V/dO 4D-TMA）——落实 ROADMAP『fp8 专项冲刺』第一步。此前 ours 的
      `ours` 口径一直是 `-arch=sm_90` 的 mma.sync（SASS：160×HMMA + 78×LDSM、无 QGMMA/TMA），
      而 O9c-2/O32/O37/O41 早已把 wgmma+TMA 实现好、只是 opt-in。F1 只改标准入口
      `harness/fa_bwd_run.py`（`FP8_HOPPER_DEFAULT`：fp8 定长用
      `sm90a -DFA_WGMMA -DFA_TMA -lcuda`，新增 `--mma` 回退），**device 一行未改**；host 自动打印
      `qd-tma=on / kv-tma=on` 并选到 `fa_bwd_fp8_mma_kvtma_kernel`。**SASS：QGMMA 0→8、HMMA
      160→96、LDSM 78→46、新增 7×UTMA**（GEMM3/4/5 仍是 mma.sync，因 fp8 wgmma 无转置操作数）。
      数值与历史在 fp8 容差内一致（S4096 `2.635/2.644/3.216e-1`），单/两文件 gate `worst=1.05e-5`。
      **端到端 1.14–1.23×**（S4096 total 2.399→**1.946ms**、main 1.885→1.608ms），相对同 session
      TE FP8 仍 2.83×/5.23×/6.43×（S512/S1024/S4096）。**ncu：墙未变**——`wait 1.59 +
      short_scoreboard 1.29`、L2 `red` 114.5M 扇区、L2 77.9%、3 CTA/SM，与 ROADMAP「阻塞」的
      核算逐项吻合（下一步 F3/F4）。详见 `docs/03` §67；原始输出 `src/fp8/fa_bwd_fp8_p132_*`、
      `src/fp8/fa_bwd_fp8_p132_{sass_counts,ncu_main_s4096,stall_s4096,te_baseline}.out.txt`。

### 5.48 F4-第一步（第一百三十三轮）：fp8 DET 的 dK/dV partial「降精度 + 写扇区化」

- **背景**：F1 后默认路径的墙是 `wait + short_scoreboard`（GEMM3/4/5 的 mma 依赖，见「阻塞」）
  与 L2 `red` 114.5M；F4 想做的是打掉 dK/dV 归约。第 131 轮把 fp16/bf16 的 DET partial 做完
  了「降精度（O60）+ 扇区化（O62）」，fp8 一直没做——本轮把这块补齐（落实 §67.7 候选 ②）。
- **fp8 的特殊性**：fp32 的 DET partial 在 `m16n8k32` 布局下**已经落满 32B 扇区**（quad 的
  4 个 `float2` 连续），所以只把存储降 fp16 会变成 16B/quad、扇区不减 ⇒ 必须叠加 O62 的
  「相邻两列组拼 8B + 16 列块内置换」。
- **实现**：`dkv_det_store_h4` + `dkv_p16_perm`（公式与 fp16/bf16 逐字相同）；`fp8_mma_body`
  加 `DET_HALF`，`epi_dv/epi_dk` 在 `DET_HALF` 时一次写 8B；两个 `dkv*_reduce_kernel` 加 `P16`
  按置换列读回 fp16。单/两文件 device 逐字一致、默认路径一行未改（`--ci` 全绿）。
- **正结果（DET 路径）**：S4096 DET-fp32 2.597ms → **DET-fp16 2.214ms（1.17×）**，相对非确定
  atomic 由 0.78× 抬到 **0.92×**（S512 0.88→0.93、GQA 0.80→0.90）；`runs[1-2]` 逐位=0（确定性
  保留），**ncu**：主 kernel store 扇区 68.2M→**34.1M（−50%）**、reduce 734.7→**420.9µs
  （1.75×）**、读扇区 102.2M→51.1M。数值只差 partial 的 fp16 舍入（~1e-3），`ours-vs-ref` 不变。
- **尚未转正**：fp8 的 partial 即使 fp16 也有 1.1GB 写 + 1.1GB 读（atomic 在 L2 RMW 无额外
  DRAM），故 ksplit=1 下 DET 仍 0.90–0.93×——与 fp16/bf16 O62「反超 atomic」不同（fp8 原
  partial 已满扇区，降精度只减字节）。默认路径的 `red` 仍被 O42 的双硬约束锁定。详见 `docs/03`
  §68；原始输出 `src/fp8/fa_bwd_fp8_p133_*`。

### 5.49 F4-b（第一百三十四轮）：把 fp16 partial + 写扇区化扩到 fp8 的 MLA 与 varlen DET

- **背景**：第 133 轮把 fp8 的 DET partial 降精度+扇区化只接到了定长 D=128 的 Hopper `kvtma`
  快路；`docs/03` §68.6 候选 ③ 点名要把这条扩到 varlen/MLA。本轮补齐 fp8 DET 的全部三条路径。
- **实现**：两个 varlen 归约内核加 `bool P16`，按 `row + dkv_p16_perm(c)` 读回 fp16（求和集合/
  次序逐字不变 ⇒ 仍确定性）；boot host 给三处 DET A/B 各加 `run_dth`（varlen D=128、varlen MLA、
  定长 MLA）。**HD=512 成立的关键**：`dkv_p16_perm` 只在 16 列块内置换，而 GEMM3/4 的 warp
  n-tile 起点 `c0=wc*GN34`（HD=512/4-N-tile 的 `GN34=32`）是 16 的倍数 ⇒ 全局列索引直接套用。
  顺带补上单文件 `launch_bwd_main_det` 缺失的 `DET_HALF` 模板参数（§68 的历史遗漏）。
- **正结果（DET 路径）**：DET-fp32→fp16 端到端 **1.06–1.11×**（大 varlen ksplit=4 达 1.20×）；
  `runs[1-2]` 逐位=0、`fp16-vs-fp32` ~1e-3（纯 partial 舍入）、`ours-vs-ref` 逐位不变；
  **ncu**：store 扇区在定长 MLA（2.228M→1.114M）与 varlen D=128（1.180M→0.590M）上**精确减半**，
  证明 O62 扇区化对 HD=512/varlen 普适。单/两文件逐指标一致；`--ci` 全绿。
- **仍未转正**：与 §68 同——fp8 partial 即使 fp16 仍有 ~1.1GB 写 + 读，ksplit=1 下 DET 仍略慢于
  非确定 atomic；默认路径 `red` 仍由 O42 双硬约束锁定。详见 `docs/03` §69；原始输出
  `src/fp8/fa_bwd_fp8_p134_*`。

### 5.50 F3-a（第一百三十五轮）：fp8 主 kernel 去 local 化 + F3（WS）可行性评估

- **先评估后动手**：F3 = warp specialization / 更深 mbarrier 流水。用 ncu 对标 TE（S4096 causal）：
  TE 384 线程/132 CTA/1 CTA/SM/**232KB smem**/258µs；ours 128 线程/8192 CTA/3 CTA/SM/74.8KB/1.61ms。
  **两者都 L2 bound，但 ours 的 L2 搬运量 ≈ TE 的 6.9×**（`Duration×L2%`）——因为 ours 受 3 CTA/SM
  的 smem 限制，靠 ksplit=8 凑并行度，Q/dO 重读 + dK/dV 跨 CTA `red`（114.5M 扇区）。
- **判决**：WS 只能*重叠*、不能*减少*这笔 L2 流量（O48 判据不满足）；更深流水被 3 CTA/SM 的
  74.8KB↔77.5KB 硬间隙锁死。**F3 不是最优点，真杠杆是 F4（减 L2 `red`）/ 改工作划分（dK/dV-over-KV）。**
- **正结果（微优化）**：ncu 报默认 kernel local memory 占 L1TEX 扇区 18.29%。定位到
  `int kuse[2]` 按运行期 `stg^1` 动态下标 ⇒ 落 local（8B stack + 每 tile LDL/STL，同 kernel-opt
  「mbarrier 相位别用动态下标数组」坑）。改两标量后：local `op_ld/st` 扇区 **7.33/8.48M→
  5.84/5.67M**；main S4096 6 次交替 **1.588→1.557ms（~2%↑）**、端到端 1.9285→**1.8969ms**、
  S512/GQA 亦 1.8–2.4%；数值**逐位不变**、单/两文件 gate worst 6.7e-6 OK。剩余 local 是默认
  `REGDQ` 实例 168-reg 硬墙下的 60B spill（寄存器粒度 8 ⇒ 封顶 168）。详见 `docs/03` §70；
  原始输出 `src/fp8/fa_bwd_fp8_p135_*`、`..._f3_te_ncu_s4096.out.txt`。

### 5.51 F5（第一百三十六轮）：fp8 LSE 的「tile 内两趟 softmax」+ mbarrier 相位去 local

- **动机**：main 的杠箱被 F3-a 判为「3 CTA/SM 的 L2 流量锁死」后，转 preprocess。ncu 定位
  D=128 定长 causal 的 LSE（`lse_mma_kernel_bal_tma`）是**纯 issue-bound**：S=4096 Duration
  201µs / 指令 141.0M / Issue Slots Busy 73.6% / Ipc 3.09。
- **SASS 归因**（`cuobjdump -sass` opcode 直方图，每 tile）：`MUFU.EX2` **64**（每元素 2 个
  `fexp`，其中一次恒为 `exp(0)=1` 纯浪费）、`BSSY/BSYNC` **107/107**（逐元素 `if (sv!=-INF)`）、
  `FSETP.*` **180**、`LDS` 316。
- **改法**：epilogue 从「逐元素 online-softmax」改成 **tile 内两趟**——第一趟算本 lane 各 `s`
  的列 max（顺手把掩码后的 `sv` 写回累加器），第二趟统一 `Σexp(sv-mn)` + 一次 rescale。数学
  等价、只换 fp32 求和次序。**`fexp` 64→34、去逐元素分支**。覆盖 4 个 LSE kernel（TMA / wgmma
  / mma-bal，含 `FULL`/`MTN`）。顺带把 TMA LSE 的 `int kuse[2]`（运行期动态下标 → local
  memory，同 F3-a 在 main 修的坑）改两标量。
- **结果**：LSE **201→120µs（1.67×）**、指令 **141.0M→86.4M（−38.7%）**、local 扇区
  532k/598k→**0**；**preprocess 1.21–1.56×（S4096 0.216→0.138ms）**、端到端 **1.03–1.04×**
  （S4096 1.888→**1.817ms/75.6TF**）；varlen D=128 1.02–1.04×。vs-ref/TE 打印位一致、单/两文件
  gate 1.335e-5 OK、`--check docs/04` OK。原始输出 `src/fp8/fa_bwd_fp8_f5_*`。详见 `docs/03` §71。
- **教训**：**softmax epilogue 的「逐元素 online」常隐含一次恒为 `exp(0)` 的废物 `exp` 与
  逐元素分支**；改成「先求 tile/lane 列 max 再统一 exp」既省 exp 又消分支，是 issue-bound
  softmax 的常规手段。

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
7. **同步脚本的「边界」要用验证兜住**：`sync_onefile_device.py` 只校验替换后的 device 区等于
   源 `.cuh`，**不校验替换区间内是否夹带了 host 行**。fp8 单文件的 device 区 marker
   （`#include <cuda_runtime.h>`）一度在 `#include "../fa_bwd_dump.h"` 之前，于是每次同步都
   把它误删、单文件编译报 `fa_bwd_save_npy_f32 undefined`。修法：把 host-only 的 include 移到
   marker **之前**。（对照：单项修改后一定重编译单文件，别只跑两文件。）
