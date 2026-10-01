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

### 5.52 F4-c：fp16 DET partial 归约的读取向量化（第一百三十七轮，正结果，opt-in `--det`）

- **动机**：F4/F4-b 把 fp8 `--det` 的 dK/dV partial 降到 fp16 + 写扇区化，但 reduce 端仍
  **逐列 2B 标量读**（按 `dkv_p16_perm` 反查）⇒ ncu 显示 fp16 reduce 的 **DRAM 只有 ~78%**
  （fp32 版反而 91.2%）：**降精度减半字节的一半收益被 2B 读的低效率吃掉**（reduce 是纯带宽 bound）。
- **改法**：`dkv_det_store_h4` 写的 8B 单元物理上就是 4 个连续列，reduce 端**每线程按物理 4 列
  一组用 `uint2` 读回**（每 block 4 行、`HD/4` 个列组），把每个输出列的求和集合/次序**逐字保持**
  ⇒ 逐位不变。四个归约 kernel 的 `P16` 分支 + 7 处 host grid/`dkv_blocks` 相应改；默认路径一行未改。
- **结果**：reduce **421→370µs（1.14×）**、**DRAM 78%→92.5%**（追平 fp32 版）；DET 端到端
  S4096 2.180→**2.147ms**（同 session 交替 A/B）、DET/atomic **0.899→0.908×**；MLA/varlen
  fp32→fp16 比也小幅上行。三个 `runs[1-2] bitwise=0`、`fp16-vs-fp32` 与 F4/F4-b **逐位相同**、
  `ours-vs-ref` 逐位不变；单/两文件一致；`--ci` 全绿。详见 `docs/03` §72。
- **教训**：**「把精度降一半」只在读侧也做到「宽读」时才兑现带宽收益**——O60 在写侧踩过
  「扇区粒度」的坑，本轮在读侧踩到同一个镜像问题（**标量窄读达不到 HBM 峰值效率**）。

### 5.53 F3/F4-default/GEMM3·5-wgmma 收口 + LSE split auto 微调（第一百三十八轮）

- **不是新优化，是把「墙」钉死**：F1→F5 后 `下一步候选` 一直收敛到「默认路径 L2 `red` 是唯一
  真杠杆、受寄存器/smem 双硬墙锁定」。本轮用 ncu 定量复核：默认 fp8 main（S4096 causal）
  **Duration 1.59ms、L2 76.96%、L1/TEX 71.75%、Compute 47.89%、DRAM ~4.3%（L2 hit 97.08%）**；
  L2 扇区 **154.0M，其中 `red` 114.5M = 74.3%**；stall **`wait` 1.61 + `short_scoreboard` 1.27**
  （症状，非发射序）、Active Warps/Scheduler 2.94。**L2 吞吐就是墙。**
- **ISA 层验证**：读 CUTLASS `mma_sm90_gmma.hpp`——fp8 `SS_TN` 的 asm 尾操作数是
  `p, scaleA, scaleB`，fp16 `SS` 是 `p, scaleA, scaleB, tnspA, tnspB`。**fp8 wgmma 无运行时转置**，
  GEMM3/4/5 的 B 是 N 连续 ⇒ 必须物理转置（+20KB smem + scatter，O4b 已判净负）。阻塞判据成立。
- **负结果两条**（都同 session/同 binary 交替）：① 累加缓冲 `cudaMemset` 旁路 stream 重叠 →
  **1.8072 vs 1.8096ms（噪声内）**；② 量化 grid cap → **0.0686 vs 0.0687ms（无效）**
  （quant 单 kernel 15.5µs、DRAM 70.8%，是真带宽 bound）。非 main 固定开销无余量。
- **正结果（小）**：LSE K 维 split auto 在 S≥2048 多切一档（S4096：auto split=4，
  实测最优 split=2：0.1185 vs 0.1189/0.1219ms）。`target` 在 `S≥2048` 降到 1024，S<2048 不变；
  S512/S1024H32 无回归，端到端噪声内。数值逐位不变，`--ci` fp8 全绿（gate 7.629e-6）。
- **教训**：**当所有 quick lever 都在噪声内、且结构杠杆已被 ISA/硬件判死时，正确的增量是
  「把阻塞证据补全到不可再辩」**，而不是制造一个噪声级 commit。真杠杆只剩：跨 CTA 分块偏和
  （FA2 式 dK/dV-over-KV）、放大 BM（寄存器墙）、或换卡。详见 `docs/03` §73。

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

### 5.54 fp8 主 kernel「唯一真杠杆」定位（第一百三十九轮）

- **不是新优化，是把「red 到底是谁」钉死**：第 138 轮已证默认 fp8 main 是 L2-bound（77%）、
  `red` 114.5M 占 L2 扇区 74.3%。本轮用 ncu 把 `red` 拆开：
  - **ksplit=1**：`red` **103.8M**（dQ 无原子）⇒ **dK/dV 的跨 m-block `red` ≈ 104M，占 ~90%**；
    ksplit=8 只多 ~10.7M 的 **dQ 跨 part 原子**。⇒ 「dQ 改 partial」收益 <10%，不值。
  - **候选① `dK/dV-over-KV`：流量中性（负杠杆）**——把原子负担从 dK/dV 搬到 dQ，总数
    `Σ(nblk-j)·BN·HD` ≡ `Σ(m+1)·BM·HD`（BM=BN），MHA 对称、**GQA 更差 `H/Hkv` 倍**。关闭。
  - **候选② BM=128：单独不够**——现有 `wg2<128>`（无 TMA/wgmma）实测 `red` 砍半
    （114.5M→58.2M）但 **L2 仅 20.4%**（1 CTA/SM、12.5% warps 的延迟 bound），慢 0.53×。
    必须 **TMA+wgmma+双 warpgroup 的 BM=128**；`fp8_mma_body` 现锁死单 warpgroup ⇒ 需新 body。
  - **候选③ ksplit auto=8 已最优**（1/2/4/8/12/16/24/32 → 1.92/1.69/1.58/**1.53**/1.57/1.62/2.30/2.31ms）。
- **LSE（F5 后）issue_active 77.6%、sm 73.3%** ⇒ 已近指令侧下限，无新低垂果实。
- **教训**：**当「候选清单」里的路没被定量验证前，先花一轮把每条路算/测清楚再动手**——
  本轮避免了一次基于「dK/dV-over-KV」的错误大重写（该路流量中性），并把唯一真路
  （双 warpgroup wgmma+TMA BM=128，F6）钉成可执行目标。详见 `docs/03` §74。

### 5.55 F6 第二步：双 warpgroup 主 kernel 的 GEMM1/2 wgmma 化（第一百四十一轮，正确但中性）

- **做了什么**：把第 140 轮冒烟落进主 kernel——新增 `fa_bwd_fp8_wgmma2_kernel`（= `wg2`
  的 GEMM1/2 换 `wgmma.m64n32k32`，Q/dO/K/V 存 SW128，Qp/dOp/Kp 由 SW128 重建，fold 与
  GEMM3/4/5 逐字沿用），host `--wg2wgmma`；单/两文件同步。
- **正确**：vs fp32 ref S512 `2.426/2.996/3.713e-1`、S4096 `2.635/2.760/3.325e-1`（fp8 噪声）；
  单文件逐位一致；`[F6 A/B]` wg2wg-vs-wg2 差 ~5e-2（同既有 wgmma-vs-mma 量级）。
- **性能：中性**。相对 wg2（mma）只快 **1.021×（S512）/1.044×（S4096）**，仍只有默认
  BM=64 档的 **0.66–0.68×**。
- **原因（ncu）**：wg2wgmma **212 regs / 136.4KB smem → 1 CTA/SM**（wg2 是 217/131.3KB），
  occupancy 12.5%、`wait 1.67 + long_scoreboard 1.49` ⇒ 延迟/occupancy bound 未变。冒烟的
  90 regs 只在「只有 GEMM1/2」成立；叠上 `dqacc[2][8][4]`+fold+3/4/5 的 mma 后寄存器照旧。
  且本轮为 Ap/dS3 独立缓冲把 smem 推高，方向相反。
- **教训**：
  1. **别把「子模块的孤岛测量」当整 kernel 的预算**——90 regs 是 GEMM1/2 单独的数，不等于
     整 kernel 能到 128 regs / 2 CTA/SM。
  2. **BM=128 的最大 smem 项是 Qp/dOp（34.8KB）**；F6 要 2 CTA/SM 必须先干掉它（SW128 tile
     直读 `ldmatrix`），而不是先换 GEMM1/2 的指令。下一小步据此排序。
  详见 `docs/03` §76。

### 5.56 F6 判定：「去 Qp/dOp 冲 2 CTA/SM」在 fp8 上不可行（第一百四十二轮，负结果 + 收口）

- **做了什么**：落实第 141 轮「下一步候选 ①」，用**三条证据**判决 F6 的最小步：
  ① ISA 探针（`fa_bwd_fp8_mma_variant_probe.cu`）：fp8 `mma.m16n8k32` **只有 `.row.col`**，
  `.col.row/.row.row/.col.col` 全被 ptxas 拒 ⇒ B 必须 col-major（必须转置）；
  ② pairing 分析：fp8 的 1 个 b16 = 2 个 fp8，`.trans` 不改变 b16 配对方向 ⇒ K-major tile
  的配对轴仍是 N，不是 mma 要的 K（O4b 已证，本轮显式化）；
  ③ 穷举复现（`fa_bwd_fp8_f6_directb_smoke.cu`）：从 K-major tile `ldmatrix.x4.trans` 取回
  的 4 个寄存器里**任意 2 个都拼不出** col-major `x2` 的 B 片段（**0/32 lane 匹配**）。
- **资源账**：wgmma2（F6）`cuobjdump` **212 regs / 135,424B smem → 1 CTA/SM**；2 CTA/SM 上限
  `116,224B`，缺口 19,200B。去 Qp/dOp（34,816B）本可解，但被上面判不可行；改 `Ps/Ss` 半精度
  只能刚够且改数值口径，regs 212→≤128 必大 spill。
- **性能复核**（S4096 causal，同 session）：默认 BM=64 `kvtma` main **1.569ms** vs F6
  wgmma2 **2.808ms**（默认的 0.56×）；TE FP8 纯反向 0.3003ms ⇒ F6 是 TE 的 ~9.4×。
- **结论**：F6 在本卡「冲 2 CTA/SM」不成立；继续只能走**物理转置 SW128 B**（O4b 已判净负）
  或**换卡**，转 backlog。默认路径一行未改。
- **教训**：**「别的 dtype 成立的技巧」跨到 fp8 前先做 ISA 句法级验证**——kernel-opt 42 篇的
  「SW128 16B chunk 可 `ldmatrix` 转置读」是 **bf16（1 个 b16 = 1 元素）** 的结论；fp8 的
  2 元素/b16 让「转置」只发生在 8×8 矩阵层、不改元素配对，直接搬运会导致整块 B 错位。
  详见 `docs/03` §77。

### 5.57 O63：fp16/bf16 LSE 的「tile 内两趟 softmax」（第一百四十三轮，正结果，默认）

- **背景**：第一百三十六轮 F5 给 **fp8** 的 4 个 LSE kernel 上了「tile 内两趟 softmax」
  （`docs/03` §71、`docs/08` §5.51），fp8 LSE **1.67×**。但 F5 只覆盖 fp8；fp16/bf16 的 LSE
  仍是旧的「逐元素 online-softmax」（每元素 `fmax` + 两个 `fexp` + 逐元素 `if (sv!=-INF)`）。
- **做了什么**：把 F5 的两趟 epilogue **逐字回移/参数化**到 fp16 与 bf16 的 4 个 LSE kernel
  （`lse_mma_kernel` / `_bal` 含 `FULL` / `_bal_wgmma` / `_bal_tma`），并顺手把 `_bal_tma` 的
  `int kuse[2]`（运行期动态下标 → local memory，F3-a 的坑）改两标量。数学等价，只换 fp32 求和次序。
  单/两文件同源：fp16 用 `#include <cuda_runtime.h>` marker；bf16 按既定做法用
  `using bf16 = ...` marker（其 `#include <algorithm>` 在 device 区之前）。
- **数值**：与改前**逐位相同**（fp16 S4096 1.883/1.734/1.966e-3；bf16 1.510/1.340/1.631e-2；
  fp16 varlen B4 3.163/2.158/1.966e-3）；CI gate fp16 3.906e-3 / bf16 1.562e-2 均 OK；
  仅 full 的个别 dv 第 5 位移动（~1e-4 级），`--apply` 同步 docs/04。
- **性能**（同 session 同 binary，event，S4096 causal）：fp16 LSE **0.3112→0.2159ms（1.44×）**、
  preprocess 0.3246→**0.2293（1.42×）**、total 1.9173→**1.8233（1.052×）**；bf16 LSE
  **0.3098→0.2158（1.44×）**、total 1.9213→**1.8333（1.048×）**；fp16 varlen 0.7792→**0.7424
  （1.050×）**；Hopper（wgmma LSE）preprocess 0.2268→**0.1339（1.69×）**、total 1.2571→**1.1657
  （1.078×）**。main 一行未动。
- **ncu/SASS**：fp16 `lse_mma_kernel_bal<128,1>` Duration **310.27→221.18µs（1.40×）**、
  指令 **163.8M→119.1M（−27.3%）**、local 扇区 **98K/16K→0**、stall `wait` 2.16→1.05；
  wgmma 版 277.79→**195.17µs（1.42×）**、指令 −26.0%。SASS：`MUFU` **172→112**、
  `BSSY/BSYNC` **109→42**、`FSETP` **260→140**、静态指令 **4544→3776**。
- **结论**：F5 的两趟 softmax 对 fp16/bf16 同样成立；**三 dtype 的 LSE 现已统一走两趟**。
  LSE 不再是墙（S4096 fp16 preprocess 仅 ~13% 端到端），默认路径的墙仍是 main。
  详见 `docs/01` §19、`docs/01b` §6av；原始输出 `src/fp16/fa_bwd_fp16_p143_*`、
  `src/bf16/fa_bwd_bf16_p143_*`、`src/fa_bwd_p143_ci_fp16_bf16.out.txt`。

### 5.58 O64：fp8 量化 + 清零融合成单 launch（第一百四十四轮，正结果，默认）

- **背景**：F1→F6 收口后默认 fp8 main 受本卡寄存器/smem 硬墙锁定（见 ROADMAP「阻塞」），转查
  **非 main 固定开销**。nsys 拆 S1024H32 causal 默认路径：per-call 有 4 个 quant kernel + 3 个
  `cudaMemset` 共 7 次串行小 launch；ncu 每个 quant 只到 **DRAM 58% / Compute 43%**（8.7µs/个，
  尾波 + 间隙截断），quant+残留 ≈ 端到端 19%。
- **做了什么**：新增 `quantize_zero_warp_kernel<VPT>`——每 warp 一行、单一 grid 覆盖「量化 Q/dO/K/V +
  清零 dQ/dK/dV」7 类任务；量化抽成 `quant_row_warp`（与 O14 逐字相同），清零 `zero_row_warp`（float4）。
  定长 + varlen 的 `run_all` 默认走它，`--qfuse=0` 退回旧路径做 A/B。**数值逐位不变**。仅 fp8（fp16/bf16
  无量化）。
- **结果**（同 binary A/B）：定长 S512 **1.168×** / S1024H32 **1.046×** / GQA kv4 **1.048×** /
  S4096 **1.016×**；varlen MLA b1_t512 **1.177×** / b4_t3840 **1.020×**。
  `max_abs(fused-vs-unfused)` dq/dk/dv ≤ 7.2e-7（仅 atomic 次序）。ncu 融合 kernel DRAM **58→73%**；
  nsys per-call kernel 数 **10→5**、间隙残留 ~31→~12µs。`--ci --dtype fp8` 全绿、`--check docs/04` OK。
- **教训**：**「小 launch 的固定开销」是继 main 之后的第二梯队**——多个独立、各自不满带宽的小 kernel
  （量化/清零）串行时，融合成一个大 kernel 能让 DRAM 流水连续、吃掉 per-launch 尾波与间隙；对
  launch 占比高的小 shape 收益可达 15–18%，对 main 主导的大 S 则边际（~1%）。详见 `docs/03` §78。

### 5.59 O65：fp16/bf16 prologue 融合（清零 dq/dk/dv + delta 单 launch，第一百四十五轮，小/中 shape 正结果，默认）

- **背景**：O64（fp8）证明「小 launch 融合」有效，其原因是 fp8 的 4 个 quant kernel 各自只到
  DRAM 58%（尾波 + 间隙截断）。ROADMAP 的下一步候选 ② 是把同款思路扫到 **fp16/bf16 的固定开销**
  （无 quant，只剩 `cudaMemset` 与 `delta`）。先量化：nsys `cuda_gpu_mem_time_sum` 显示默认路径
  per-call 有 **2–3 次 `cudaMemset`**（S4096 每次 10.6µs / 33MB ⇒ **~3.1 TB/s，已近 HBM 峰值**）+
  1 次 `delta_warp_kernel`（~14µs，DRAM 71%）。
- **做了什么**：新增 `zero_delta_warp_kernel<HD>`（fp16/bf16 单/两文件同源）——一个 grid-stride 内核里
  先 float4 清零 dq_acc（可选，`n_q`）/dk_acc/dv_acc，再按 **与 O24 `delta_warp_kernel` 逐字相同的
  几何**（warp-per-row、lane 沿 HD 以 half2 读、`__shfl_xor_sync` 同树）算 delta；host 在 `run_pre`
  里替换 `delta` 那个 launch，并去掉 `run_all` 的 memset（`zfuse_sel` 开关，`--zfuse=0` 同 binary A/B）。
- **数值**：delta 段与 `delta_warp` 逐字同几何 ⇒ **结果逐位相同**；dump 对拍 `dq` 逐位 0，
  dk/dv 仅 132/155 个元素差 **1 ulp**（跨 CTA `atomicAdd` 的调度次序，本就非确定）。CI：fp16
  gate worst 3.906e-3 / bf16 3.125e-2 均 OK、`--check docs/04` OK。
- **性能**（同 binary A/B，event，total，3 rep 中位）：fp16 **S512 0.0855→0.0812ms（1.053×）**、
  GQA kv4 S1024 0.3347→0.3308（1.012×）、MLA S1024H2 0.1826→0.1765（**1.035×**）、S4096 1.8271→1.8301
  （中性）；bf16 同构（S512 1.054×、GQA 1.012×、MLA 1.033×、S4096 中性）。
- **ncu**：S4096 融合 kernel `zero_delta_warp_kernel<128>` Duration **33.95µs / DRAM 70.8% / L2 76%**；
  拆开看 `delta_warp` 14.3µs + 2×memset ~21µs ≈ 35µs ⇒ **大 S 处本质是「同字节数、已带宽 bound」，
  融合不省字节、也省不出时间**。S512 融合 kernel **6.1µs / DRAM 20.5%**（远未饱和、launch 占比高），
  而旧路径 2×memset(~2µs) + delta(~3.6µs) + 3 次 launch 开销 ⇒ 融合实测省 ~4µs（5.3%）。
- **教训**：**融合只对「不满带宽的小 kernel」有效**。O64 的 fp8 quant 只到 DRAM 58%，融合能把流水拉满；
  而 fp16/bf16 的 memset 已到 3.1 TB/s（近峰值）、delta 也到 71%，融合最多省 launch 开销，故只在
  **小/中 shape（launch 占比高）**有 1–5% 收益，大 S 中性。结论与 ROADMAP 对候选 ② 的预判一致
  （fp16/bf16「无 quant，仅剩 memset/convert」）。详见 `docs/01` §20、`docs/01b` §6aw；
  原始输出 `src/fp16/fa_bwd_fp16_o65_ab.out.txt`、`..._o65_ncu_s{512,4096}.out.txt`、
  `src/bf16/fa_bwd_bf16_o65_ab.out.txt`。

### 5.60 O66：fp8 把 delta 融进「量化 + 清零」单 launch（第一百四十六轮，正结果，默认）

- **背景**：O64（§5.58）把 fp8 的「4 次量化 + 3 次清零」并成 1 个 `quantize_zero_warp_kernel`，
  但 `delta`（`D=rowsum(dO∘O)`）仍是紧随的一次独立 `delta_warp_kernel` launch。delta 只依赖
  dO/O 且是逐 query 行归约——与 dO 的 rowwise 量化同一域、同一 warp-per-row 几何。
- **做了什么**：新增 `quant_delta_row_warp<VPT>`（量化 dO 时，用寄存器里的 dO 值与同步读入的 O 行
  顺便算 delta）与 `quantize_zero_delta_warp_kernel<VPT>`（O64 任务序逐字相同，dO 档换成融合版）；
  host `--dfuse=`（默认 1）在 `run_all` 调融合 kernel、让 `run_preprocess(false)` 跳过独立 delta。
  **关键正确性点**：delta 必须**耦合进 dO 的量化任务**（同一 warp 内），不能作独立任务——同一
  launch 内无跨 warp 同步，独立任务读 `do8` 会与量化写竞争。数值逐位不变（`max_abs=0.000e+00`）。
- **结果**（同 binary `[O66 A/B]`，event）：S512 **1.047×**、S1024H32 **1.025×**、MLA S1024H2
  **1.021×**、varlen D128 b1_t512 **1.036×**、S4096 **1.003×**；数值 vs ref 与历史逐位一致
  （S4096 `2.635/2.644/3.216e-1`）、`--ci --dtype fp8`（定长+varlen）全绿（gate worst 9.537e-06）、
  `--check docs/04` OK 194 行。ncu 融合 kernel S1024H32 `Duration 50.85µs / DRAM 76.3% / L2 84.7%
  / 31 regs` ⇒ 仍 DRAM/L2 带宽 bound。
- **教训**：继 O64/O65 之后再次印证——**「小 launch 融合」只值「省一次尾部小 kernel 的 launch 与
  尾波」**，故小/中 shape 有 2–5%、大 S 边际；但把依赖同一数据域的逐行归约（delta）并进量化任务
  几乎零成本（只多 O 的读，被 DRAM 流水吸收）。**fp8 非 main 固定开销的融合至此收口。**
  详见 `docs/03` §79；原始输出 `src/fp8/fa_bwd_fp8_o66_*`。

### 5.61 O67：fp8 dK/dV 归约 float4（负结果）+ 用 ncu 重测 TE 定位真差距（第一百四十七轮）

- **做了什么**：默认 fp8 `kvtma` main 的 L2 `red`（114.5M 扇区=74% L2）是唯一真杠杆（见 §5.53/§5.56）。
  O4c 已把归约从标量提到 `red.v2.f32`（8B）。本轮再提到 **`red.v4.f32`（16B，`FA_R4`，默认 0）**：
  fp8 `m16n8k32` 的一个 quad 的 `c2=(lane&3)*2` 是同 row 连续 8 列，用 `shfl_down(1)` 拼两个
  float4（列 0-3 lane0 写、列 4-7 lane2 写），请求数应再减半。`epi_dv`/`epi_dk` 的
  非 DET/BULKRED 分支同步改，单/两文件 device 同源。
- **结果：负**。SASS 确认生成了 `REDG.E.ADD.F32x4`（base 320×`F32x2` → 256×`F32x2`+64×`F32x4`），
  但 **ncu 的 red 请求/扇区一字不变**（`l1tex 9,543,680` / `lts 114,524,160`）——red 的 L2 计数
  不随「同 warp 内合并列」而变；同 shape 交替 A/B main **-0.4~-0.8%**（S512/S1024/S4096）。
  与 O7c 在 fp16 上的结论一致（§5.12），本轮补齐 fp8。默认关，旗舰路径逐位不变。
- **真正价值（重新定位）**：同轮用 `harness/te_fp8_ncu.py` 重测 TE fp8 反向
  （`..._flash_bprop_wgmma_f8_..._64x64x128_1x4x1_cga1x1x1`，grid=132、384 线程、**BM=64 与
  ours 相同**）：TE `red=25.96M`、read 10.2M、write 0.79M、**L2 总量 ~36.9M（ours 153.9M 的
  1/4.2）**、Duration 260.7µs（ours 1/6）。TE 的 SASS 是 `REDG.4D.ADD`（同为 128-bit red）。
  ⇒ **同 BM 下 red 仍可压 4.4×** ⇒ 差距是**工作划分 / tile 调度**（TE 132 CTA persistent，
  ours ksplit=8 → 8192 CTA），**不是归约宽度、也不只是放大 BM**。
- **教训**：**「同 BM、同归约位宽」也能有 4× 的 red 差**——瓶颈是「每个 KV 元素被多少 CTA
  贡献」，即 FA2/TE 的 **dK/dV-over-KV 单一 owner** 工作划分（+ TMA store-reduce / persistent）。
  O42 的 `cp.reduce.async.bulk` 失败是因为**没换工作划分**（归约次数没降、只加 staging）。
  新立 **F7**。详见 `docs/03` §80；原始输出 `src/fp8/fa_bwd_fp8_p147_*`。

### 5.62 F7 第一步：dK/dV「KV-owner 单一 owner」划分 smoke（第一百四十八轮）

- **背景**：O67（§5.61）把默认 fp8 main 的 L2 `red`（S4096 114.5M 扇区、占 L2 74%）真差距钉到
  **工作划分**：TE 同 **BM=64** 下 `red` 仅 ours 的 **1/4.4×**、L2 1/4.2×、时间 1/6×
  （TE grid=132 persistent vs ours ksplit=8→8192）。ours 是 **Q-owner**（每 CTA 一块 query、遍历
  KV、dK/dV 跨 CTA `atomicAdd`）。F7 = 换 **KV-owner**（每 CTA 一块 KV、遍历 query、dK/dV
  本地累加一次写）。
- **做了什么**：新增自包含最小复现 `src/fp8/fa_bwd_fp8_kvowner_smoke.cu`——同一批随机
  Q/K/V/dO（S512 H8 HD128 causal），`mowner_kernel`（现有划分，atomic）与 `kvowner_kernel`
  （F7 划分，smem 累加 + 一次 plain store）对拍 double host 参考。fp32 标量（隔离变量）。
- **结果（机制正）**：数值 KV-owner **正确**（vs double ref dk/dv max_abs 3.64e-5/2.28e-6，
  与 Q-owner 差 5.7e-6 = fp32 次序）；ncu `lts__t_sectors_op_red`：Q-owner **1,671,168** →
  KV-owner **0**（L1 red 请求 278,528 → **0**）⇒ 跨 CTA 原子归约**彻底消除**，与 TE 低 red 一致。
- **关键教训（下一步判据）**：这个**最小原型** KV-owner 的 `op_read` 反涨到 **44M**（Q-owner 1.5M）、
  标量耗时 0.64×——因为**没做 operand staging**：Q-owner 的 K/V tile 被并发 query CTA 大量复用、
  L2 命中极高；KV-owner 每个 KV CTA 反复从 global 重读 Q/dO（`dot` 里跨行未合并）把 red 的收益
  吃回。⇒ **F7 主体不能只翻转 grid**，必须让 persistent CTA **拥有 KV 块并把 Q/dO/K/V 经
  smem/TMA staging 复用**（对标 TE grid=132 persistent + 本卡 TMA 通路）。prize 由 O42 钉死：
  短路 dK/dV red ⇒ main 1.70×、total 1.53×。
- 详见 `docs/03` §81；原始输出 `src/fp8/fa_bwd_fp8_kvowner_smoke.out.txt`、
  `..._kvowner_ncu_red.out.txt`。

### 5.63 F7 第二步：persistent KV-owner + smem/cp.async 暂存（第一百四十九届）

- **动机（落实 §5.62 的判据）**：§5.62 的 KV-owner `red=0`，但 `op_read` 从 1.5M 暴涨到 44M、
  耗时 0.64×——因**没做 operand staging**。本轮验证「persistent CTA 拥有 KV 块 + Q/dO 经
  smem/cp.async staging」能把读放大消除、`red` 仍为 0。
- **做了什么**：在同一 smoke 里新增 `kvowner_stage_kernel`：① 拥有的 `Ks/Vs[BN][HD]` 只从 global
  读一次常驻 smem；② Q/dO 用 `cp.async.cg` 16B **双缓冲流水**搬入 `Qs/dOs[2][BM][HD]`；③ phase1/2
  全从 smem 读，dK/dV 本地累加一次 plain store。动态 smem 139,264B（`extern __shared__` +
  `cudaFuncSetAttribute`）。
- **结果（机制正）**：数值 vs double ref **3.64e-5/2.28e-6**、与无 staging 的 KV-owner **逐位相同**
  （只换 operand 来源）。ncu（同 shape）：`red = 0`（保持），`op_read` **44.07M → 1.38M（31.9×）**、
  甚至略低于 Q-owner 的 1.50M；global 载入请求 26.8M → **147K（182×）**。
- **性能（event，iters=20）**：Q-owner 3.46ms / KV-owner 5.41ms（0.64×）/ **KV-owner+stage 0.876ms
  （vs Q-owner 3.95×、vs 无 stage 6.17×）**。ncu stage kernel：880µs、32 regs、139.26KB smem
  →1 CTA/SM、stall `short 4.49 + wait 1.67` ⇒ 墙回到 **smem 依赖 + 低 occupancy**（标量 dot 特有），
  **不再是 global 读放大、也不是原子归约**。
- **教训 / 下一步**：F7 机制两根支柱（单一 owner 消 red + staging 消读放大）均已 de-risk；
  **F7 主体** = 把这两点落进真实 fp8 主 kernel（persistent 拥有 KV 块、Q/dO/K/V 走 4D-TMA 暂存、
  dK/dV 本地累加后 TMA store / `cp.reduce.async.bulk` 一次写出，对标 TE grid=132）。prize 由 O42
  钉死：main **1.70×**、total **1.53×**。
- 详见 `docs/03` §82；原始输出 `src/fp8/fa_bwd_fp8_kvowner_stage_run.out.txt`、
  `src/fp8/fa_bwd_fp8_kvowner_stage_ncu.out.txt`。

### 5.64 F7 第三步：KV-owner 落进真实 fp8 张量核（dK/dV 的 mma 原型，第一百五十轮）

- **动机（落实 §5.63 的下一步）**：§5.62/§5.63 的 KV-owner + staging 是在 **fp32 标量** smoke 上做的，
  而真实 fp8 主 kernel 走 **E4M3/E5M2 + rowwise scale + `mma.m16n8k32`**。必须把「单一 owner 消 red」
  落进真实数据通路，才能判断它是否成立。
- **做了什么**：新增 `src/fp8/fa_bwd_fp8_kvowner_mma.cu`（独立文件，默认路径一行未改）——
  `fp8_kvowner_dkv_kernel<128,64,32>`：grid=`(S/BN,H)`，每 CTA 拥有 KV 行块、K/V 常驻 smem 只读
  一次、Q/dO staging 到 smem（Qs/dOs + Qp/dOp 配对）、遍历 query 块、dK/dV 在**寄存器**本地累加后
  **一次 plain store**。4 个 GEMM（S、dP、dV、dK）全 mma，量化/折算记账与 Q-owner 主 kernel 完全一致。
- **数值（正）**：S512 dk/dv vs ref **2.975e-1/3.735e-1**、S4096 **2.643e-1/3.216e-1**——与既有
  Q-owner ours **完全相同**；`KV vs Q` 仅跨 CTA 加法次序差（≤1.5e-1）。
- **ncu（同 session，S4096 H16 causal）**：Q-owner mma 主 kernel `red=114,524,160`、1.91ms；
  **KV-owner dK/dV `red=0`**、`op_read 36.4M→51.0M（1.40×）`、**1.41ms**（168 regs、3 CTA/SM）。
- **结论（非正收益，方向明确）**：`red` 114.5M→**0** 在真实 mma 上成立；但按每 GEMM 等效折算
  KV `1.41/4=0.353ms` 仅略优于 Q `1.91/5=0.382ms`。KV-owner 把 `red` 换成了 **Q/dO 的跨 CTA 读
  放大（1.40×）+ 寄存器墙**（dVacc+dKacc 占 64 regs），且无持久化/预取/TMA，补齐 dQ 后不构成
  净收益。**F7 主体须同时做**：persistent（grid≈132）+ Q/dO/K/V 4D-TMA 暂存重叠 + dQ 同循环 +
  降寄存器占用。
- 详见 `docs/03` §83；原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_{s512,s4096,ncu_s4096}.out.txt`。

### 5.65 F7 第四步：KV-owner mma 的 Q/dO staging 加 cp.async 双缓冲重叠（形状相关，第一百五十一轮）

- **动机（落实 §5.64 的 F7 主体子项「重叠」）**：§5.64 的 KV-owner mma 原型 staging 是同步全局读，
  全局延迟串在 GEMM1 前。给 Q/dO staging 加 `cp.async.cg` 16B **双缓冲流水**，做同 binary A/B。
- **做了什么**：`src/fp8/fa_bwd_fp8_kvowner_mma.cu` 新增 `fp8_kvowner_dkv_pipe_kernel`（默认路径
  一行未改）：staging 缓冲 ×2（`Qs/dOs/Qp/dOp` 双 stage），循环里 `issue_qdo(next)`→`wait_group 1`
  →`build_paired(cur)`→5 GEMM；末块 `wait_group 0`。
  **踩坑**：`Qs[STAGES]` 运行期下标的**指针数组**被 ptxas 推到 local memory，首版 regs 243；
  改成「基址 + 标量偏移」后 239、A/B 0.875×→**0.958×**（S4096）。smem 70,144→**105,984B**。
- **数值（逐位）**：S512 base/pipe dk/dv 均 2.975e-1/3.735e-1、S4096 均 2.643e-1/3.216e-1，
  **pipe vs base = 0/0**（只换搬运时序）。
- **性能（event，同 binary/session）**：**形状相关**——S512（grid 256）**1.148×**（0.0566→0.0493ms）、
  S4096（grid 2048）**0.959×**（1.4218→1.4825ms）。
- **ncu 机制（S4096）**：base long 0.58 / short 0.88 / **wait 1.44**（墙）；pipe long **0.15**
  （重叠生效，全局延迟被藏住）但 regs 168→**239**、smem 70→106KB ⇒ **3→2 CTA/SM**、
  warps 17.7%→12.2%、op_read 51.0M→67.4M ⇒ 净 −4.1%。
- **结论**：KV-owner mma 原型的墙是 **`wait`+`short_scoreboard`（mma/smem 依赖）而非全局延迟**，
  所以「隐藏全局延迟」只在小 grid（未撑满 3 CTA/SM 容量）时有效。F7 要转正必须直接打 `wait` 与
  occupancy，属 §83 的 persistent+4D-TMA+降寄存器大改。**本子项判决完成。**
- 详见 `docs/03` §84；原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p151_{s512,s4096,sol}.out.txt`。

### 5.66 F7 主体第一步：KV-owner mma 的 persistent 调度（负结果，第一百五十二轮）

- **动机**：F7 主体要对标 TE `..._flash_bprop_wgmma_f8_..._64x64x128`（grid=132 persistent、
  1 CTA/SM），先把「persistent 调度」子项单独判决。`src/fp8/fa_bwd_fp8_kvowner_mma.cu` 新增
  `fp8_kvowner_dkv_persist_kernel`（默认路径一行未改）：栅格从 `(nblk,H)`（S4096=2048 CTA）改
  1D persistent `grid = min(nblk*H, SM×3)`（`--pgrid` 可覆盖）+ `tile += gridDim.x` 循环，
  `tile→(h=tile/nblk, j0=(tile%nblk)*BN)`。
- **数值（逐位）**：S512/S4096 **persistent vs base = 0/0**（只换栅格映射），
  vs ref 与 base 同（S512 2.975e-1/3.735e-1、S4096 2.643e-1/3.216e-1）。
- **性能（event，同 binary，S4096 H16 iters=50）**：pgrid=132→**0.313×**、264→0.558×、
  396→**0.707×**、792→0.789×、1188→0.931×、**2048（无循环，纯 1D 版 base）→0.931×**；
  base(2D 2048 CTA)=1.0。**单调、永不反超**。S512（grid 256<396 无循环）=0.921×。
- **ncu 根因（S4096）**：base waves 5.17、Compute/Memory 46.9/50.3%、Eligible 0.82；
  persist(pgrid=396) waves **1.0**、Compute/Memory **32.5/36.8%**、Eligible 0.67
  ⇒ **SM 空闲但无 block 可补**。causal 下每 tile 权重（被多少 query 消费）差 ~64×，
  静态 strided 划分总权重不均 + 失去硬件「块完成即回填」⇒ 单波利用率骤降。
- **结论**：**KV-owner dK/dV 原型的 persistent 不是杠杆**（负结果）。TE grid=132 可行是因为其
  tile 工作划分**均匀**；把「grid=132」直接搬到带 causal 偏斜的 KV-owner 划分上有害。F7 主体若
  要 persistent 必须配**动态负载均衡 / 均匀化工作划分**，否则杠杆仍是 §5.65 的 `wait`+occupancy。
- 详见 `docs/03` §85；原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p152_*`。

### 5.67 F7 第六步：KV-owner mma 的 dynamic work-queue 调度（正结果，第一百五十三轮）

- **动机**：承接 §5.66 的「必须配动态负载均衡」——把持久栅格的静态 strided tile 分配换成
  **dynamic work-queue**（global `atomicAdd` 领任务），验证是否能在 persistent 栅格上复现硬件
  调度器的「完成即回填」。`fp8_kvowner_dkv_persist_kernel` 加模板参 `bool DYN`：`DYN=true` 时
  循环顶 `tid==0 atomicAdd(wq,1)` + `__syncthreads` 广播 tile 直至越界。tile=`h*nblk+jblk`，
  小 `j0` 先领 = **LPT 重块前置**。host 加 `d_wq` + `cudaMemsetAsync` 清零。
- **数值（逐位）**：S512/S4096 **dyn vs base = 0/0**、dyn vs static=0/0（只换栅格映射），
  vs ref 与 base 同（S512 2.975e-1/3.735e-1、S4096 2.643e-1/3.216e-1）。
- **性能（event，同 binary，iters=50）**：S4096 H16 base **1.4151ms**、static(396) 1.9799ms
  （**0.715×**）、**dynamic(396) 1.4248ms（0.993× / dyn-vs-static 1.390×）**；S512
  base 0.0577 / static 0.0622 / dyn 0.0591ms（0.977×）。
- **ncu（S4096）机制**：base / dyn / static 的 `red` **均=0**；Duration 1.40 / 1.40 / 2.00ms、
  SM% 47.1 / 47.5 / 32.3、`lts__t_sectors_op_read` 51.0M / 53.1M / **72.2M**。⇒ 修正 §5.66
  的单因归因：static strided 的 step=`gridDim=396` 把并发 CTA 撒在 ~6 个 head 上、Q/dO 工作集
  打散 ⇒ **L2 局部性损失（读扇区 1.42×）+ 负载不均**，二者都被 dynamic 连续领号一并消除，
  dyn 与 base 逐项吻合。
- **结论**：**dynamic work-queue 正结果**——同 3 CTA/SM、同 396 CTA 下把 §5.66 的 0.715×
  恢复到 **0.993×**（追平硬件调度的 base），`red` 仍 0。「persistent 只能更慢」修正为
  「**静态 persistent 更慢**」；持久化本身不再是障碍，为 F7 主体（持久 CTA + TMA staging +
  dQ 同循环）提供了可用的调度骨架。但 dyn 相对 base 仅中性——F7 主体要真正转正仍需叠加
  4D-TMA / 降寄存器冲 4 CTA/SM / 打 `wait`，否则杠杆回到 §5.65。
- 详见 `docs/03` §86；原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p153_*`。

### 5.68 F7 第七步：KV-owner mma 的 GEMM1/2 上 Hopper wgmma（正结果，第一百五十四轮）

- **动机**：承接 §5.67 的「F7 主体须打 `wait` + occupancy」。base 第一墙是
  `wait 1.44 + short 0.88`（mma/`ldmatrix` 依赖），非全局延迟（`long` 0.58）。F6 第二步（Q-owner）
  已用 wgmma 重叠 GEMM1/2；本步把它搬到 KV-owner，且**关键约束 = 不掉 occupancy**。
- **做法**：新增 `fp8_kvowner_dkv_wgmma_kernel`（`-DFA_WGMMA`）：Q/dO/K/V 全存 **SW128**，
  GEMM1/2 用 `wgmma.m64n32k32` 直读描述符（异步、一起发、统一 wait0），epilogue 改 wgmma
  累加器映射；GEMM3/5 仍 mma+ldmatrix（fp8 wgmma 无转置操作数，O9c-2 判死）。SW128 tile 比
  ASLD 更紧凑 ⇒ smem **67072B < base 70144B，仍 3 CTA/SM**（区别于 §5.65 pipe 的 2 CTA/SM）。
- **数值**：wgmma vs ref 与 base vs ref 同量级（S512 2.976e-1/3.732e-1、S4096 2.644e-1/3.216e-1）；
  wgmma-vs-base 差 = 既有 Q-owner-vs-base 的累加次序噪声，非 bug。
- **性能（同 binary A/B，main 仅 dK/dV）**：S4096 H16 base 1.4128 / pipe 1.4428 / dyn 1.4074 /
  **wgmma 1.3472ms（wg/base 1.049×、wg/pipe 1.071×、wg/dyn 1.045×）**；S1024H32 1.004×；
  S512 0.998×（grid-bound）。
- **ncu（S4096）**：Duration 1.42→**1.38ms**、`Executed Instructions` **638.7M→541.0M（−15.3%）**、
  Compute 47.25→41.01%、`wait 1.44→1.52`（**未降**）、`long 0.58→1.40`、regs/smem/3 CTA/SM 不变。
  ⇒ **收益来自指令数 −15.3%（去掉 ldmatrix + SM80 兼容发射），不是消除 `wait`**。
- **结论**：§5.65/§5.67 之后**第一个在大 S 转正的 KV-owner 结构改动**（1.049×，优于 pipe 0.979×
  与 dyn 1.004×），且保持 3 CTA/SM。但仍是 dK/dV-only；F7 主体仍欠 4D-TMA staging、dQ 同循环、
  降寄存器冲 4 CTA/SM。详见 `docs/03` §87；原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p154_*`。

### 5.69 F7 第八步：KV-owner wgmma 的 Q/dO 4D-TMA staging（正结果）＋修一处 p154 遗留 race（第一百五十五轮）

- **动机**：承接 §5.68 的「F7 主体仍欠 4D-TMA staging」。§87 后 Q/dO 仍是逐 ROW-pair 的标量
  global gather（ncu 相对暴露的 `long_scoreboard`）。本步把它换成 **4D-TMA**（对标 TE 数据通路）。
- **做法**：`fp8_kvowner_dkv_wgmma_kernel` 抽成 device body `<...,bool TMA>` + 两个薄壳（`TMA=false`
  逐字退化 = §87；`TMA=true` 新加）。TMA 版 per m 块发两条 `cp.async.bulk.tensor.4d` 搬 Q/dO 进
  SW128 tile（fp8 一行 128B = SW128 atom 整行，一条 TMA 整块），mbarrier 相位每迭代翻转，再从
  SW128 重建 `Qp/dOp`（同 O37）；smem 67072+64B ⇒ 仍 3 CTA/SM。
- **附带修 race**：wgmma 经 async proxy 读 smem，而 K/V（及非 TMA 的 Q/dO）是 generic 写，p154
  **缺 `fence.proxy.async.shared::cta`** ⇒ 偶发 nondeterminism（同一 binary 8 次
  `wgmma vs base` 在 4.08e-2/6.4e-2/2.2e-1 间跳；已在 HEAD 原文件复现）。补
  `bulk_reduce_fence()` 后 8/8 稳定，`wgmma+TMA vs wgmma` 逐位 = 0。
- **数值**：TMA-vs-wgmma **0.0000e+00（逐位）**；vs ref 与 wgmma 同量级（S512 2.976/3.732e-1、
  S1024H32 4.176/3.535e-1、S4096 2.644/3.216e-1）。
- **性能（同 binary A/B，main 仅 dK/dV）**：S512 0.0577→**0.0462ms（1.247×）**、
  S1024H32 0.2404→**0.1940（1.239×）**、S4096 1.3747→**1.1183（1.229×）**，tma/base 1.22–1.27×
  ⇒ **§84/§86/§87 之后最大的 KV-owner 结构收益**（小 S 也转正）。
- **ncu（S4096）**：Duration 1.38→**1.12ms**、`Executed Instructions` **541.5M→470.1M（−13.2%）**、
  `lts op_read` 58.86M→**52.18M（−11.4%）**、**`red` 仍 = 0**、`long_scoreboard` **1.40→0.43**
  （全局 gather 延迟被 TMA 去掉）、`wait 1.52→1.53（未降）`、short 0.95→1.31、regs 168 / 67.1KB /
  3 CTA/SM 不变。⇒ 收益 = **去全局 gather**；墙回到 `wait` + L1/TEX + short（F7 主体下一步）。
- **结论**：F7 主体的「TMA staging」子项 **de-risk 并转正**（1.23–1.27×，全部 shape）。剩余 =
  ① dQ 同循环（跨 CTA 归约/partial+reduce）；② 降寄存器冲 4 CTA/SM。详见 `docs/03` §88；
  原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p155_*`。

### 5.70 F7 第九步：KV-owner wgmma+TMA 的「降寄存器冲 4 CTA/SM + 微调」判决（负结果，第一百五十六轮）

- **动机**：§5.69 后 F7 主体只剩 dQ 同循环 + 降寄存器冲 4 CTA/SM（smem≤58KB/regs≤128）。先一次
  性判决「便宜旋钮」，避免大改前留盲点。
- **做了什么**（`src/fp8/fa_bwd_fp8_kvowner_mma.cu`，仅加两编译期旋钮，默认 = p155 逐字行为）：
  ① `FA_KV_CTA`（wgmma 两壳的 `__launch_bounds__`，默认 3）；② `FA_KV_OVL` + `build_paired_tma`
  lambda（=1 时把 TMA 路径 Qp/dOp 重建挪到 wgmma 之后、`wait0` 之前做重叠）。
- **实验一（store bank conflict）**：源码把 Qp/dOp 两次 4B 存合成 `uint2` → **`sm__inst_executed`
  逐位不变 470,056,960、store conflict 也不变** ⇒ nvcc 早合成了 `ST.64`，源级 no-op（已回退）。
  真冲突源是 **Ps/Ss epilogue**（`PSS=37`，8 行×4 列，每行 4 个偶 bank ⇒ 数学下界 ≥2-way）；
  且 store 不 stall，无杠杆。
- **实验二（CTA/SM 扫参 1/2/3/4，main dK/dV，iters=100）**：S4096 cta1 1.269 / cta2 1.268 /
  **cta3 1.111** / cta4 1.180 ms；S1024H32 0.220/0.219/**0.192**/0.205；S512 0.0449/0.0450/
  0.0465/0.0498 ⇒ **3 CTA/SM（168regs）最优**。**「冲 4 CTA/SM」双硬墙未松动**：168→128 要砍
  40 regs（`dVacc/dKacc` 64 + `sacc/dpacc` 32 已占 96），且 smem 67KB 仍卡 Block Limit Shared
  Mem=3（要到 4 须 ≤58KB，需去 Qp/dOp = F6 已判不可行）。
- **实验三（Qp/dOp 重建与 wgmma 重叠，`FA_KV_OVL` 0/1）**：bitwise `wgmma+TMA vs wgmma = 0/0`，
  但 **S512/S1024 慢 ~1%、S4096 中性**；ptxas **`C7517`** 在 OVL=1 注入额外 `wgmma.wait_group`
  ⇒ 重叠被串行化（同 O22/O29/O46「`wait` 非指令重排可解」）。
- **ncu（最终默认档 S4096）**：Duration 1.12ms、指令 470.06M、`red` **0**、L1/TEX 66.3% / Compute
  44.4% / occ 17.7%（3 CTA/SM）、stall `wait 1.53 + short 1.31 + long 0.43`。
- **结论**：**p155 原型微调空间已尽（局部最优）**；F7 主体真杠杆只剩 **dQ 同循环（`cp.reduce.async
  .bulk` / partial+reduce）+ 三梯度同循环摊薄 `wait`**（GEMM3/5 上不了 fp8 wgmma，F6 三证）。
  工程量大，转「阻塞/下一步」。详见 `docs/03` §89；原始输出 `src/fp8/fa_bwd_fp8_kvowner_mma_p156_*`。

### 5.71 F7 第十步：KV-owner 原型补 dQ 同循环（三梯度全通；atomic 归约判决，第一百五十七轮）

- **动机**：§5.70 后 p155 的 KV-owner wgmma+TMA 原型（仅 dK/dV）微调已尽，F7 主体只剩「dQ 同
  循环」。本轮把 dQ 落进同一循环，让 KV-owner 成为**首个三梯度全通的 fp8 反向原型**，并判决
  「dQ 走 atomic」。
- **做了什么**（`src/fp8/fa_bwd_fp8_kvowner_mma.cu`，`FA_KV_DQ` 默认 1，=0 退回 p155）：
  ① fold 新增 `dS2[m][j]=dS[m][j]·ks[j]`（e5m2，per-m `sds2`，与 Q-owner 逐字同款）；
  ② K 的配对布局 `Kp` 从 SW128 `Ks` **一次重建**；③ **GEMM4** `dQ += scale·(dS2·K)` =
  `mma_block_bt<32,64,BN,E5E4>`，epilogue 乘 `sds2·scale` 后 `red_add2` 原子归约进 `dq`。
  smem 67072→**74752B**（仍 3 CTA/SM）。
- **数值**：`dq` vs fp32 ref S512 **2.4262e-1** / S1024H32 **2.3993e-1** / S4096 **2.6355e-1**
  → 与 dk/dv **同量级 fp8 噪声**；`wgmma+TMA vs wgmma` dk/dv **逐位 0**。⇒ 三梯度数学正确。
- **性能（同 binary A/B，main）**：dK/dV-only → 三梯度：S512 0.0462→**0.0750ms**、
  S1024H32 0.1940→**0.3044**、S4096 1.1167→**1.8318ms**（dQ 增量 +1.6×）。
- **ncu（S4096）**：Duration **1.84ms**、`red` **102.2M**（dK/dV-only 原型为 **0**）、`read` 52.6M /
  `write` 6.2M、L2 63.86%、DRAM 2.45%、occ 17.8%（3 CTA/SM）、stall `wait 1.64 + short 1.52 +
  long 0.89`。
- **判决**：默认 fp8 `kvtma` main（三梯度）`red` **114.5M** / L2 76.96% / ~1.92ms。本原型 dK/dV red=0
  但 dQ atomic 新添 **102.2M** ⇒ 总 red 114.5M→**102.2M（−10.7%）**、L2 −13pt、时间 −4%。
  **解析式吻合**：dK/dV 贡献 `2·S²/(2BM)=S²/BM`，dQ 贡献 `S²/(2BN)`，`BM=64=2BN` ⇒ 相等。
  ⇒ **原子从 dK/dV 搬到 dQ 是流量中性**，坐实 ROADMAP「候选①关闭」；**单趟 KV-owner 解锁不了
  F7 的 prize（dK/dV-only 1.12ms）**。
- **下一步**：让 dQ 也 **owned（red=0）**——① 两 kernel（KV-owner dK/dV ＋ Q-owner dQ-only pass，
  S/dS 重算但省 ~74% L2）；② 两级 partial（需 BN≥BM）。详见 `docs/03` §90；原始输出
  `src/fp8/fa_bwd_fp8_kvowner_mma_p157_dq_*`。

### 5.72 F7 第十一步：option(a)「两 kernel」——Q-owner dQ-only pass 的判决（负结果，第一百五十八轮）

- **动机**：§5.71 把 dQ 落进 KV-owner 同循环后，dQ 走 atomic ⇒ 流量中性。F7 主体真杠杆只剩
  「dQ 也 owned（red=0）」。ROADMAP「候选 ①(a)」= **两 kernel**（KV-owner dK/dV ＋ Q-owner
  dQ-only pass）。本轮落地并判决。
- **做了什么**（`src/fp8/fa_bwd_fp8_kernels.cuh`，默认路径一行未改）：给 `fp8_mma_body` 加尾部
  编译期 `bool DQONLY=false` 并透传到 `fa_bwd_fp8_mma_kernel` / `fa_bwd_fp8_mma_kvtma_kernel`。
  `DQONLY=true` 跳过 Ap/dS3 fold + GEMM3/4（dV/dK）及其 epilogue，保留 GEMM1/2 + dS2 fold +
  GEMM5（dQ）；`kRegDq` 下 dQ 用 **plain store** 写出（`red=0`）。原型
  `fa_bwd_fp8_kvowner_mma.cu` 加 `--only={dqonly,main3,wgtma}` 与同 binary 的 `launch_dqonly`
  （`DQONLY=1`, Q-owner 栅格）/ `launch_main3`（默认 `kvtma` 三梯度）/ p155 的 KV-owner
  dK/dV-only。
- **数值**：`dQ-only` vs ref 与历史 dQ **逐位一致**（`vs 三梯度基线 dq = 0.0`）；`main3` 的
  dq/dk/dv 也与历史逐位一致。CI `--dtype fp8`（mma 与 `--hopper`）均 rc=0、数值逐位不变。
- **性能（同 binary，iters=100）**：两 kernel 合计 / 单 kernel 三梯度 = S512 **0.944×**、
  S1024H32 **0.955×**、S4096 **0.955×**（S4096 2.04 vs 1.94ms）⇒ **慢 4.5–6%，负结果**。
- **ncu（S4096）**：dqonly Duration **933.7µs / red 0** / read 18.2M / L2 6.8%；main3
  **1.99ms / red 103.8M** / read 24.2M / L2 55.0%；wgtma **1.12ms / red 0** / read 52.3M / L2 18.8%。
  两 kernel 红归零但 **L2 读放大 2.9×**（70.5M vs 24.2M）——单 kernel 的 GEMM1/2（S/dP）与 P/dS
  被 dK/dV 与 dQ 共享，拆开后 dQ-only 必须重算并重读 Q/K/V/dO。省下的 `red` 打不过重算。
- **判决 / 下一步**：**「候选 ①(a) 两 kernel」判负**。F7 主体只剩 **①(b) 两级 partial / 单趟内
  非原子归约（需 BN≥BM）**，或放弃「拆 kernel」；「让 dQ owned」的收益必须在**单趟**内取。详见
  `docs/03` §91；原始输出 `src/fp8/fa_bwd_fp8_p158_*`。

### 5.73 F7 第十二步：用 TMA 4D tensor store-reduce（TE 的 `UTMAREDG`）替掉 dQ 逐 lane 原子（机制正/性能负，第一百五十九轮）

- **背景**：§5.61（O67）里 TE 的全局 red 只有 **3168** 条指令、ours 9.54M；本轮把 TE SASS 拉出来
  直方图确认是 **`UTMAREDG.4D.ADD`（TMA 4D 张量归约）** + `UTMALDG/UTMASTG` + `USETMAXREG`。
  于是试做 F7 里点名的「TMA store-reduce」：KV-owner 原型的 dQ 归约从 `red_add2` 换成
  **`cp.reduce.async.bulk.tensor.4d`**（一条指令归约整块 `[BM][HD]` fp32 回 global，box 内维 ≤256B
  故拆 2×64 列 chunk）。
- **两个坑**：① `FA_FP8_HAS_TMA` 晚于 helper 定义 ⇒ 用它守卫 asm 会**恒 0 编掉**（SASS 无
  `UTMAREDG`、dQ 错 max_abs 2.96 却**不报错**）——**新 device asm helper 必须 `cuobjdump -sass`
  验指令被发射**；② PTX 记法必须是 `.add.tile.bulk_group`。
- **数值**：`TMA-reduce vs atomic` dq max_abs **1.19e-7**、dk/dv **0**，dq vs ref 与 atomic 同量级。
- **性能（同 binary，main 三梯度）**：S512 0.89–0.91×、S1024H32 0.78×、S4096 0.78× ⇒ **更慢**。
- **ncu（S4096，判决性）**：`l1tex …op_red` 8.52M → **0**、`smsp …global_red` 8.52M → **0**、
  `smsp inst` 703.65M→660.41M，**但 `lts__t_sectors_op_red` 102,236,160 → 102,236,160 一字不变**；
  dynamic smem 74.82→107.58KB ⇒ 3→2 CTA/SM、occ 17.79→12.19%、Duration 1.84→2.47ms。
- **判决 / 下一步**：**L2 `red` 流量由工作划分（每元素贡献 CTA 数）决定，与归约指令机制无关**；
  TE 的 `UTMAREDG` 只是指令选择，其 4.4× 低的 red 来自 tile 调度。**F7「TMA store-reduce」假设关闭**，
  F7 主体只剩工作划分本身（BN≥BM / 放大 tile / persistent 调度）。详见 `docs/03` §92。

### 5.74 F7 第十三步：BN≥BM（BN=64）——dQ 贡献数减半，但撞寄存器墙（中性/偏负，第一百六十轮）

- **动机**（落实第一百五十九轮「下一步候选 ①(a)」）：KV-owner 下 dQ 的跨 CTA 贡献数 ≈ `S/BN`。
  把 KV-owner 的 tile 从 `BN=32` 放大到 `BN=64=BM`，dQ `red` 与 Q/dO 读放大**各减半**；
  代价是 `dK/dV` 累加器与 GEMM1/2 wgmma 累加器翻倍、smem 74.75→111.42KB（3→2 CTA/SM）。
- **做了什么**（`src/fp8/fa_bwd_fp8_kvowner_mma.cu`，默认路径一行未改）：把 `fp8_kvowner_dkv_wgmma_body`
  参数化到 `BN∈{32,64}`——GEMM1/2 新增 `wgmma_mn_issue<BN,KIND>`（BN=32 `m64n32k32` / BN=64
  `m64n64k32`）、epilogue 的 `j<4`→`j<BN/8`；壳的 `__launch_bounds__` 按 BN 选 `FA_KV_CTA64=2`；
  host 加 `Cfg64`/`wg_tma_smem64`/`launch_wgtma64`/`--only=wgtma64` 与同 binary 对拍+计时。
- **数值**：三 shape BN=64 的 `dk/dv` 与 BN=32 **逐位相同**（0.0）、`dq` 只差 ~5.9e-2（fp8 噪声，
  贡献数变少的加法次序）；vs fp32 ref 与 BN=32 同量级。
- **性能（同 binary，event，iters=30）**：S512 **0.900×**（grid 8×16=128<132 SM，grid-bound）、
  S1024H32 **0.960×**、S4096 **1.042×（另一 session 1.015×）** ⇒ **中性**。
- **ncu（S4096，判决）**：`lts__t_sectors_op_red` **102,236,160→51,118,080（精确减半）**、
  `lts read` 52.6M→40.1M（−24%），**但 `lts write` 6.2M→26.6M、local `op_st` 65,536→13,754,368
  （+210×）**，registers 168→**255（上限）**、occ 17.9%→11.9%（3→2 CTA/SM）。⇒ **`dVacc+dKacc`
  翻倍顶穿 255 寄存器文件 → 溢出流量盖过省下的 `red`/读**。
- **判决 / 下一步**：**F7 option(a)「BN≥BM」判为中性/偏负**。至此 F7 的单趟「减少贡献数」路线
  （两 kernel / TMA store-reduce / BN≥BM）**全部判决**；剩余只有「跨 warpgroup 偏和 + 二次归约」
   （同样撞 smem/regs）或「放弃 F7 / 换卡」。与阻塞里 O17b/F6 的「寄存器文件锁死放大 tile」同源。
   详见 `docs/03` §93。

### 5.75 F8：fp16/bf16 定长默认切 Hopper（wgmma+TMA）（第一百六十一轮，正结果，默认）

- **动机（F1 的 dtype 泛化）**：F1（第一百三十二轮）只把 **fp8** 定长默认切到
  `sm90a -DFA_WGMMA -DFA_TMA`；fp16/bf16 的 `ours` 口径仍锁在 `-arch=sm_90` 的 `mma.sync`——
  尽管 host 自 O5/O5b 起已有 wgmma+TMA 快路（一直只由 `--hopper` 独立前缀在跑，未做默认）。
  本轮把同一默认化推到 fp16/bf16（用户优先级「逐步把 main 切到 wgmma+TMA 路径」的 dtype 收尾）。
- **改动（纯 harness，device 一行未改）**：`harness/fa_bwd_run.py` 把 `FP8_HOPPER_DEFAULT`
  泛化为 `HOPPER_DEFAULT_DTYPES = {"fp8", "fp16", "bf16"}`：定长（`not is_varlen`）且非
  `--mma`/`--hopper` 时用 `HOPPER_FLAGS`（`-gencode=compute_90a -DFA_WGMMA -DFA_TMA -lcuda`）
  构建；`--mma` 一次退回三 dtype 的旧 mma.sync 口径做 A/B；`--hopper` 仍走独立前缀
  `ours_hp/ours_sf_hp`。
- **数值**：S=4096 causal `max_abs` vs fp32 ref **逐值不变**（fp16 `1.883/1.734/1.966e-3`、
  bf16 `1.510/1.340/1.631e-2`）——wgmma 与 mma 只差 fp32 累加次序（A/B `wg2b-vs-wg2` ~1e-5），
  被 fp16/bf16 舍入盖住；MLA（HD=512）host 自动退回 mma（不变）、full 走 mma LSE 亦正确。
  CI 73 case 全绿（fp16 worst `1.953e-3`/bf16 `7.812e-3`/fp8 `1.049e-5`，`--check docs/04` OK）。
- **性能（同 session，S4096 causal MHA，total/event）**：**fp16 1.8245→1.1726ms（1.556×）**、
  **bf16 1.8325→1.1643ms（1.574×）**；main 分别 1.503→0.964ms / 1.495→0.958ms。相对纯反向
  FA3（`0.3238ms/849TF`，fp16）时间比 **5.6×→3.6×**。
- **原始输出**：`src/fa_bwd_p161_hopper_default_ab.out.txt`、`src/fa_bwd_p161_baseline_{fp16,bf16}.out.txt`。

### 5.76 O68：非 causal（full）D=128 的 LSE 接上均衡 `cp.async` 版（第一百六十二轮，正结果，默认）

- **动机（一处被遗漏的分支）**：F1/F5 把 fp8 causal 的 LSE 做到 TMA/wgmma + 两趟 softmax，
  O54 也给 `lse_mma_kernel_bal` 加了 `FULL=true`（一个 CTA 一个 m 块 + `cp.async` 双缓冲）——
  但该 FULL 模式**只接到了 D=512（MLA）**；D=128 的**定长 full** 与 **varlen full** 仍走 O1 的
  `lse_mma_kernel`（逐元素 `LDG.U8→STS`、无 `cp.async`、无 split）。
- **做了什么**（纯 host，`src/fp8/fa_bwd_fp8_main.cu` + `fa_bwd_fp8_mma_onefile.cu`，device 一行
  未改）：新增文件作用域开关 `g_lse_full_opt`（默认 1）+ CLI `--lsefull=0/1`；把两条 D=128 full
  的 LSE 调用从 `launch_lse<128>`（O1）改为 `launch_lse_bal<128,1,true>`（O54 FULL）。
- **数值**：ours vs fp32 ref（S1024 full）`5.52e-2 / 5.31e-2 / 4.02e-2`（fp8 噪声）；lsefull=0 vs 1
  只差 LSE 的 fp32 求和次序 `~1e-3`；`--ci` fp8 单/两文件一致性 worst **7.153e-06** OK、
  `--check docs/04` OK。
- **性能（同 binary A/B）**：定长 full S1024 **preprocess 0.1466→0.0397ms（3.70×）/ total
  0.5388→0.3740ms（1.44×，22.96 TF）**；varlen full b4_t4096 **total 1.5589→1.2293ms（1.27×）**。
  对标 TE FP8 纯反向 `0.0562ms/305.8TF` ⇒ ours/TE **9.6×→6.65×**。
- **ncu（LSE）**：O1 **208.4µs / Warp-Cycles-Issued 8.11 / L1TEX 9.4%**（纯延迟 bound）→
  均衡 FULL **42.3µs（4.93×）/ 3.86 / L1TEX 22.9%**。两路径 `Waves 0.39`（网格不足）。
  ⇒ **这是 preprocess 内一条被漏改的 LSE 分支，不是 main 的 L2 `red` 墙**。
- **下一步**：LSE 再上 **4D-TMA**（对齐 causal O32）；其余候选见「阻塞」。
- 详见 `docs/03` §94；原始输出 `src/fp8/fa_bwd_fp8_o68_*`。

### 5.77 O69：fp16/bf16 非 causal（full）D=128 的 LSE 接上均衡 `cp.async` 版（第一百六十三轮，正结果，默认）

- **动机（F9 的 dtype 泛化 / 补漏改分支）**：O68/F9 发现并修了 fp8「causal LSE 已均衡化、但
  full D=128 仍走 O1 `lse_mma_kernel`」这条漏改分支；**核查 fp16/bf16 发现同一条分歧也漏改**
  （O8b/O54 的 `lse_mma_kernel_bal<FULL=true>` 此前只服务 D=512 MLA 的 full）。定长
  （`fa_bwd_{fp16,bf16}_mma_main.cu` 的 `!causal` D=128 分支）与 varlen（`run_varlen` 的
  `!causal` D=128 分支）都直接 `lse_mma_kernel<128><<<...>>>`。
- **做了什么**（纯 host，四个文件 `fa_bwd_{fp16,bf16}_mma_{main,onefile}.cu`，device 一行未改）：
  新增文件作用域开关 `g_lse_full_opt`（默认 1）+ CLI `--lsefull=0/1`；把两条 D=128 full 的 LSE
  从 O8 `lse_mma_kernel<128>` 切到 `lse_mma_kernel_bal<128, PIPE=1, FULL=true>`（`ksplit=1`，
  与 fp8 F9 逐字一致）；加对应 `cudaFuncSetAttribute` 实例。`--lsefull=0` 逐字退回 O8。
- **数值**：LSE 只差 fp32 求和次序 ⇒ 同一 case 的 dq/dk/dv **逐值一致**——fp16 定长 full S1024
  `3.268/2.523/1.225e-4`、varlen b4_t4096 `4.094/4.953/1.234e-4`；bf16 定长 full S1024
  `1.938/1.684/1.449e-3`、varlen `3.237/2.392/2.013e-3`。单/两文件逐指标一致
  （fp16 定长 full 两文件/单文件 total 0.1687ms；bf16 0.1710/0.1713ms）；fp16 一致性 gate
  worst **3.906e-3**、bf16 **7.812e-3**（均 OK）、`--check docs/04` OK（194 行）。
- **性能（同 binary A/B，event）**：
  - fp16 定长 full S1024：preprocess **0.1588→0.0463ms（3.43×）**、total **0.3048→0.1687ms
    （1.81×，28.2→50.9 TF）**；varlen full b4_t4096 total **0.8632→0.5942ms（1.46×）**。
  - bf16 定长 full S1024：preprocess **0.1578→0.0467ms（3.38×）**、total **0.3060→0.1710ms
    （1.79×）**；varlen full b4_t4096 total **0.8706→0.5993ms（1.45×）**。
  - 收益全在 preprocess（main 不变）。纯反向对标（同 session CUPTI）：定长 full S1024 fp16
    FA3 `0.0512ms/335.6TF`、FA2 `0.0828/207.6`、TE `0.0577/297.7` ⇒ ours total 为 FA3 时间
    **3.29×**、TE **2.92×**（峰值 989 的 **5.1%**）；bf16 同量级；varlen full `[1024]×4` fp16
    FA3 `0.1996ms/172.1TF` ⇒ ours **3.0×**。
- **ncu（LSE，`regex:lse_mma_kernel`，S1024 H16 full，同 session 同 binary）**：fp16 O8
  **175.8µs / L1TEX 11.8% / Compute 25.6% / Ipc 1.05 / No Eligible 73.7% / long_scoreboard 4.4cy /
  regs 80 / smem 34.82KB** → bal FULL **39.2µs（4.49×）/ L1TEX 37.6% / Compute 39.6% / Ipc 1.72 /
  No Eligible 57.1% / short_scoreboard 1.5cy / regs 64 / smem 52.22KB**；bf16 逐项一致
  （175.2→38.8µs，4.51×）；varlen full b4_t4096 fp16 LSE **375.5→108.5µs（3.46×）**。
  ⇒ O8 的墙是**串行全局载入延迟**；均衡 FULL 用 `cp.async` 双缓冲打掉它，新墙 = issue +
  smem 依赖 + 网格不足一个波（`Waves 0.48/1.94`）。与 fp8 F9（208→42µs）同源。
- **下一步**：① 同路径再上 **4D-TMA**（对齐 causal O32/O30）；② main 的 L2 `red` 墙仍受本卡
  寄存器/smem 硬墙锁定（见「阻塞」）。
- 详见 `docs/01` §22、`docs/01b` §6ay、`docs/04` §44；原始输出
  `src/fp16/fa_bwd_fp16_o69_*`、`src/bf16/fa_bwd_bf16_o69_*`、`src/fa_bwd_p163_full_baseline.out.txt`。

### 5.78 O70：fp8 非 causal（full）D=128 的 LSE 再上 **4D-TMA**（第一百六十四轮，正结果，默认）

- **动机**：O68/F9 把 fp8 定长 full D=128 的 LSE 从 O1 接到「均衡 + `cp.async`」；但 **causal 的
  LSE 早就是 4D-TMA 版**（O32 `lse_mma_kernel_bal_tma`，一条 `cp.async.bulk.tensor.4d` 搬整块
  SW128，省掉逐 16B `cp.async` 的 load 指令/地址运算）。本步把 full 也切到同一套 TMA 搬运（fp8
  先行，fp16/bf16 留下一轮）。
- **做了什么**（device+host，单/两文件同步）：
  - device（`fa_bwd_fp8_kernels.cuh` + onefile 的 `lse_mma_kernel_bal_tma`）：模板加
    `bool FULL=false`。`FULL=true` 时只做 `t=0`（无镜像配对，`grid.x=nblk` 一个 CTA 一个 m 块）、
    `ncols=S`、四个掩码 `jg<=qi` 改 `(FULL || jg<=qi)`；其余（TMA 双缓冲、rowwise scale、tile 内
    两趟 softmax、4-lane shfl）逐字复用。`FULL=false` 与 O32/O38 **逐位相同**。
  - host：`launch_lse_bal_tma` 加 `bool FULL`；full 分支当 `lse_tma` 真时调
    `launch_lse_bal_tma<128,1,true>`；默认 `lse_tma = (D==128) ? 1 : 0`。`--lsetma=0` 退回 O68
    的 cp.async 版、`--lsefull=0` 仍退回 O1；`sm_90` 构建 `lse_tma=0` 自动退回 O68。
- **数值**：ours vs fp32 ref（定长 full S1024）`dq/dk/dv=5.518e-2 / 5.310e-2 / 4.025e-2`，与 O68
  cp.async 版仅差 LSE 的 fp32 求和次序（~1e-5）；单/两文件一致性 worst **1.192e-07 OK**
  （fp8 容差 1e-4）、`--check docs/04` OK；causal S512 回归 2.426/2.972/3.733e-1 与历史一致。
- **性能（同 binary A/B，Hopper，event）**：定长 full S1024 — preprocess
  **O1 0.1498 / cp.async 0.0401 / TMA 0.0226ms（TMA vs cp.async 1.77×）**；total
  **0.5399 / 0.3742 / 0.3569ms**（TMA 24.07 TF），单文件一致。收益全在 preprocess（main 不变）；
  ours/TE 时间比 6.65×→**6.35×**（TE 0.0562ms/305.8TF），峰值占比 1.16%→**1.22%**。
- **ncu（`regex:lse_mma_kernel`，`--launch-count 1`，S1024 H16 full）**：O1 **208.4µs** →
  cp.async **42.46µs（Compute 45.4% / inst 18.39M）** → TMA **23.94µs（Compute 43.1% /
  inst 9.77M，−46.9%）**。⇒ O1 是串行载入延迟；TMA 用一条 bulk 指令搬整块 SW128 把 load 指令
  砍半，再 1.77×。仍 `Waves 0.39`（网格不足一个波）、非 DRAM/L2 bound——**是 preprocess 内部的
  搬运方式升级，不是 main 的 L2 `red` 墙**。
- **下一步**：① fp16/bf16 的同类改造（把 `lse_mma_kernel_bal_tma` 加 `FULL`，fp16 是 2×K=64
  chunk、bf16 逐字 dtype 化）——三 dtype full D=128 LSE 统一到 TMA；② main 的 L2 `red` 墙仍受
  寄存器/smem 硬墙锁定（见「阻塞」）。
- 详见 `docs/03` §95；原始输出 `src/fp8/fa_bwd_fp8_o70_*`。

### 5.79 O71：fp16/bf16 非 causal（full）D=128 的 LSE 再上 **4D-TMA**（第一百六十五轮，正结果，默认）

- **动机**：O69/F10 把 fp16/bf16 定长 full D=128 的 LSE 从 O8 接到「均衡 + `cp.async`」；但
  **causal 的 LSE 早就是 4D-TMA 版**（O30 fp16 / O31 bf16）。O70 已把 fp8 的对应改造做完，本轮
  把 fp16/bf16 也切到同一套 TMA 搬运——**至此三 dtype 的 full D=128 LSE 统一到 TMA**。
- **做了什么**（device+host，单/两文件同步）：
  - device（`fa_bwd_{fp16,bf16}_mma_kernels.cuh` + onefile 的 `lse_mma_kernel_bal_tma`）：模板加
    `bool FULL=false`。`FULL=true` 时只做 `t=0`（无镜像配对，`grid.x=nblk` 一个 CTA 一个 m 块）、
    `ncols=S`、掩码 `jg<=qi` 改 `(FULL || jg<=qi)`；其余（fp16 的 2×K=64 chunk TMA 双缓冲、tile
    内两趟 softmax、4-lane shfl）逐字复用。`FULL=false` 与 O30/O31/O38 **逐位相同**。
  - host（四个文件 `fa_bwd_{fp16,bf16}_mma_{main,onefile}.cu`）：`cudaFuncSetAttribute` 增
    `lse_mma_kernel_bal_tma<128,1,true>` 实例；full 分支当 `lse_tma` 真时改调
    `lse_mma_kernel_bal_tma<128,1,true><<<lg,…>>>(qmap_lse,kmap_lse,d_lse,nullptr,…,1)`；默认
    `lse_tma = (D==128) ? 1 : 0`（此前 `&&causal`）。`--lsetma=0` 退回 O69 cp.async、`--lsefull=0`
    退回 O8；`sm_90` 构建 `lse_tma=0` 自动退回 O69。
- **数值**：ours vs fp32 ref（定长 full S1024 H16）fp16 `3.268e-4 / 2.523e-4 / 1.225e-4`、bf16
  `1.938e-3 / 1.684e-3 / 1.449e-3`，与 O69 cp.async 版**逐值一致**（只差 LSE fp32 求和次序）；
  单/两文件一致性 gate fp16 worst 2.441e-4 / bf16 4.883e-4（均 OK）、`--check docs/04` OK；
  causal S512 回归 fp16 1.671/1.771/1.899e-3 与历史逐位。
- **性能（同 session、同 binary A/B，Hopper，event）**：定长 full S1024 —
  **fp16** preprocess **O8 0.1602 / cp.async 0.0465 / TMA 0.0362ms（TMA vs cp.async 1.28×）**、
  total **0.3037 / 0.1687 / 0.1583ms（54.27 TF）**；**bf16** preprocess **0.1602 / 0.0463 /
  0.0364ms（1.27×）**、total **0.3047 / 0.1700 / 0.1605ms（53.52 TF）**。收益全在 preprocess。
  纯反向对标（同 session CUPTI）：fp16 FA3 0.0512ms/335.9TF、TE 0.0577/297.9 ⇒ ours/FA3
  3.29×→**3.09×**、ours/TE 2.92×→**2.74×**；bf16 FA3 0.0504/340.7、TE 0.0576/298.3 ⇒ **3.18×/2.79×**。
- **ncu（`regex:lse_mma_kernel_bal_tma`，`--launch-count 1`，S1024 H16 full）**：fp16
  O8 **175.84µs / L1TEX 11.8%** → cp.async **38.62µs / L1TEX 37.6%** → TMA **28.16µs /
  L1TEX 18.8% / Compute 33.8% / regs 64 / Waves 0.48**（bf16 28.32µs 逐项一致）。⇒ O8 是串行
  载入延迟 bound，`cp.async` 打掉它（4.55×），**4D-TMA 再把 load 指令/地址运算交给 TMA 引擎
  （L1/TEX 37.6%→18.8%），Duration 再 1.37×**。仍网格不足一个波、非 DRAM/L2 bound——**是
  preprocess 内部的搬运方式升级，不是 main 的 L2 `red` 墙**。
- **下一步**：main 的 L2 `red` 墙受本卡寄存器/smem 硬墙锁定（见「阻塞」）；full LSE 的三 dtype
  搬运至此统一到 TMA。
- 详见 `docs/01` §23、`docs/01b` §6az；原始输出 `src/fp16/fa_bwd_fp16_o71_*`、
  `src/bf16/fa_bwd_bf16_o71_*`、`src/fa_bwd_o71_*`。

### 5.80 O72：varlen full D=128 的 LSE 也上 **4D-TMA**（第一百六十六轮，正结果，默认）

- **动机**：O68→O70→O71（F9/F11）把「三 dtype full D=128 LSE 统一到 TMA」，但**只覆盖定长**；
  `run_varlen` 里 full D=128 仍硬编码走 O54 的 `launch_lse_bal<128,1,true>`（`cp.async`，源码
  自认「仅定长（无 cu_seqlens）；varlen full 仍走 lse_mma_kernel_bal」）。本轮把这条**漏掉的分支**
  补上（fp8 先行，沿用 F9→F11 的「fp8 先、再泛化」惯例）。
- **做了什么**（device+host，单/两文件同步）：
  - device（`fa_bwd_fp8_kernels.cuh` + onefile 的 `lse_mma_kernel_bal_tma`）：模板加
    `const int* cu_seqlens = nullptr`。`cu` 非空时 `qbase=cu_seqlens[b]`、`len=cu_seqlens[b+1]
    -qbase`、`nblk=ceil(len/LBM)`，否则退化为 `qbase=b*S/len=S`（**定长逐位不变**）。`FULL` 下
    `pair>=nblk` 早退；scale 索引、TMA 行坐标（`qbase+m0/j0`，batch 维恒 0）、`ncols`、四个
    `jg<len` 掩码、输出写全部换 `qbase/len`。
  - host（`fa_bwd_fp8_main.cu` + onefile）：`launch_lse_bal_tma` 加 `const int* cu=nullptr`；
    `run_varlen` 为 D=128/full 建 packed 描述符（`dims={D,T,Hkv,1}`）并优先
    `launch_lse_bal_tma<128,1,true>(lg, qmap_v, kmap_v, …, d_cu)`。`--lsetmavarlen=0` 退回
    `cp.async` 版做同 binary A/B、`--lsefull=0` 退回 O1。
  - 构建（`harness/fa_bwd_run.py`）：varlen 构建对 **fp8** 也加 `-DFA_TMA -lcuda`（主 kernel 仍
    `launch_bwd_main`，无 TMA 模板参数 ⇒ 只换 LSE 搬运）。fp16/bf16 varlen 维持旧构建。
- **数值**：ours vs fp32 ref 与 `cp.async` 版**打印逐位相同**（`b4_t4096` 8.881e-2/7.043e-2/5.721e-2；
  不齐序列 `b4_t3840` 1.011e-1/9.656e-2/7.032e-2，仅 LSE fp32 求和次序 ~1e-4）；13 个 fp8 varlen
  case 单/两文件一致性 worst **3.815e-6 OK**；全量 CI `--no-run --ci` 73 case 三 dtype gate 全 OK、
  `--check docs/04` OK；定长 full S1024 回归 `5.518e-2/5.310e-2/4.025e-2` 逐位不变。
- **性能（同 binary A/B，Hopper，event，iters=50）**：`b4_t4096` full fp8 两文件
  **cp.async 1.1810 → TMA 1.1190ms（1.055×）**、单文件 1.1841→1.1221ms（1.055×）；不齐 `b4_t3840`
  两文件 **1.4724 → 1.4054ms（1.048×）**、单文件 1.4806→1.4061ms（1.053×）。同 session TE FP8
  纯反向 0.1842ms/186.56TF ⇒ ours/TE **6.39× → 6.10×**。
- **ncu（`regex:lse_mma_kernel_bal`，`--launch-count 1`，b4_t4096 full）**：cp.async
  **118.34µs / inst 73.58M / regs 88 / occ 25.8% / waves 1.55** → TMA **63.01µs（1.88×）/
  inst 39.17M（−46.8%）/ regs 58 / occ 35.9% / waves 0.97**；两者 `Compute≈Issue ~65%`
  （issue-bound）、DRAM 4–8%、L2 22% ⇒ **搬迁升级，不是 DRAM/L2 bound**。
- **下一步**：把 `cu_seqlens` 参数化逐字 dtype 化到 fp16/bf16 varlen（fp16 是 2×K=64 chunk 描述符）
  并给其 varlen 构建加 `-DFA_TMA`，做全「三 dtype × {定长, varlen} full D=128 LSE 统一到 TMA」；
  main 的 L2 `red` 墙仍受过本卡寄存器/smem 硬墙锁定（见「阻塞」）。
- 详见 `docs/03` §96；原始输出 `src/fp8/fa_bwd_fp8_o72_varlen_lse_ab.out.txt`、
  `src/fp8/fa_bwd_fp8_p166_ncu_lse_{tma,cpasync}_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_p166_{varlen_run,ci,te_varlen_baseline}.out.txt`。

### 5.81 O73：varlen full D=128 的 LSE 也上 **4D-TMA**（fp16/bf16 泛化，第一百六十七轮，正结果，默认）

- **动机**：第 166 轮 fp8 O72（§5.80）把 varlen full D=128 的 LSE 切到 4D-TMA，并留「逐字 dtype
  化到 fp16/bf16」为下一步。O69→O70→O71（F9/F11）只覆盖**定长**；`run_varlen` 里 full D=128 仍硬
  编码走 O69 的 `lse_mma_kernel_bal<128,1,true>`（`cp.async`）。本轮补齐，**三 dtype × {定长,
  varlen} full D=128 的 LSE 全部统一到 4D-TMA**。
- **做了什么**（device+host，单/两文件同步）：
  - device（`fa_bwd_{fp16,bf16}_mma_kernels.cuh` + onefile 的 `lse_mma_kernel_bal_tma`，与 fp8 O72
    逐字同构）：模板加 `const int* cu_seqlens = nullptr`。`cu` 非空时 `qbase=cu_seqlens[b]`、
    `len=cu_seqlens[b+1]-qbase`、`nblk=ceil(len/LBM)`，否则退化（**定长逐位不变**）。`FULL` 下
    `pair>=nblk` 早退；TMA 行坐标 `qbase+m0/j0`（batch 维恒 0）、`ncols`、掩码 `jg<len`、输出写
    全部换 `qbase/len`。Q/K 仍 2×K=64 chunk TMA（fp16 是 2 chunk 描述符，fp8 是 1 chunk）。
  - host（四个文件）：新增文件作用域 `g_lse_tma_varlen`（默认 1）+ CLI `--lsetmavarlen=0/1`；
    `run_varlen` 为 D=128/full 建 packed 描述符（`make_lse_map(d_q,H,T,D,1)`）并优先调
    `lse_mma_kernel_bal_tma<128,1,true>(…, d_cu)`。`--lsetmavarlen=0` 退回 cp.async 版做同 binary
    A/B、`--lsefull=0` 退回 O8。
  - 构建（`harness/fa_bwd_run.py`）：`varlen_tma` 判据由 `dt=="fp8"` 扩为
    `dt in HOPPER_DEFAULT_DTYPES` ⇒ fp16/bf16 varlen 也加 `-DFA_TMA -lcuda`（主 kernel 不变）。
- **数值**：ours vs fp32 ref 与 `cp.async` 版**打印逐位相同**（fp16 b4_t4096
  4.094e-4/4.953e-4/1.234e-4；bf16 b4_t4096 3.237e-3/2.392e-3/2.013e-3；个别元素仅 LSE fp32 求和
  次序 ~1e-4）；12 个 varlen full case 单/两文件一致性 gate fp16 worst 2.441e-4、bf16 1.953e-3 均
  OK；全量 CI `--no-run --ci` 73 case 三 dtype gate 全 OK、`--check docs/04` OK；定长 causal
  S512/S4096 回归逐位不变。
- **性能（同 binary A/B，Hopper，event，iters=50）**：varlen full D=128 total **1.06–1.07×**：
  fp16 b4_t3840 `0.9397→0.8810ms`、b4_t4096 `0.5908→0.5517`、b5_t3968(h32kv8)
  `1.8331→1.7218`、b8_t2904 `0.7562→0.7113`；bf16 同量级（`0.9522→0.8878` 等）。对标纯反向
  `[1024]×4` full：fp16 FA3 `0.1999ms` ⇒ ours/FA3 **2.96×→2.76×**；bf16 FA3 `0.1980ms` ⇒
  **3.02×→2.82×**。
- **ncu（`regex:lse_mma_kernel_bal`，`--launch-count 1`，b4_t4096 full，fp16/bf16 逐项一致）**：
  cp.async **109.0µs / inst 58.86M / L1TEX 50.9%** → TMA **66.8µs（1.63×）/ inst 36.08M
  （−38.7%）/ L1TEX 27.1%**；两者 `Compute ~57%`（issue/延迟 bound）、DRAM 10–16%、L2 34–44%
  ⇒ **搬迁升级，不是 DRAM/L2/算力 bound**。
- **下一步**：main 的 L2 `red` 墙仍受本卡寄存器/smem 硬墙锁定（见「阻塞」）。
- 详见 `docs/01` §24、`docs/01b` §6ba；原始输出 `src/fp16/fa_bwd_fp16_p167_lse_tma_ab.out.txt`、
  `..._p167_ncu_lse_tma.out.txt`、`src/bf16/fa_bwd_bf16_p167_*`、`src/fp16/fa_bwd_fp16_p33c_run.out.txt`。

### 5.82 F7 第十四步：KV-owner **column-owner**（第一百六十八轮，机制正结果 / 性能负结果）

- **一句话**：让每个 CTA 拥有 **RCOL=2 个连续 KV 块**，dK/dV 仍本地累加（red=0），dQ 跨列先在
  本地累加再每 m 一次原子 ⇒ dQ 的 L2 `red` **精确减半**（102.2M→51.1M）；但 `dVacc/dKacc` 翻倍
  + 跨列 `dqacc` 把寄存器顶到 **255 上限**、local spill（write 6.2M→113.7M 扇区）盖过收益，净
  **慢 0.72–0.77×**。
- **为什么值得一试**：这是 §93.6「唯一未试 = 跨 warpgroup 偏和 + 二次归约」的最可行形态。与已判死的
  **p160 BN≥BM** 的关键差别：p160 把 BN 32→64 时**连 GEMM1/2 累加器一起翻倍**；column-owner
  **BN 仍 32**，只翻倍 dK/dV 累加器，寄存器账更省 —— 结果证明**仍然不够**。
- **数值**：三 shape（S512H16 / S1024H32 / S4096H16）dk/dv vs wgmma+TMA **逐位 0.000e+00**、dq
  ~2e-7（仅跨 CTA 加法次序）；vs fp32 ref 与 wgmma+TMA 同量级。
- **ncu（S4096）**：`lts op_red` 102,236,160→**51,118,080**、`l1tex op_red` 8.52M→4.26M、
  registers 168→**255**、write 6.20M→**113.73M**、occupancy 3→2 CTA/SM、Duration 1.85→2.46ms、
  inst 703.65M→628.78M。
- **结论**：**与 p160 / F6 / O17b「放大 tile / 多 owner 撞 255 寄存器文件」同源**。F7 单趟路线
  （两 kernel / TMA store-reduce / BN≥BM / column-owner）**全部判决**；剩余只有换卡或「多 warpgroup
  摊累加器」（需先破 fp8 wgmma 无转置）。默认路径一行未改。
- 详见 `docs/03` §97；原始输出 `src/fp8/fa_bwd_fp8_kvowner_col_p168_*`。

### 5.83 O74：MLA（D=512）causal LSE 上 **4D-TMA**（第一百六十九轮，正结果，默认）

- **一句话**：补齐 fp8 4D-TMA LSE 的最后一条分支——**MLA（`head_dim=512`）的 causal LSE**。此前
  `lse_mma_kernel_bal_tma` 锁死 `HD==128`（TMA box 内维 128B 恰好搬一行）；本轮泛化为
  `NCH=HD/128` 个 box，`HD=512` 用 **4 次 4D-TMA** 搬完整行（每 box 一块 `[64][128]` canonical SW128
  tile，`[kg][rg]` 布局 ⇒ 每 chunk 各用 SBO=1024 的描述符累加，逐字对齐 fp16 的 2-chunk 写法）。
  `NCH==1` 走原路径 ⇒ **D=128 逐位不变**。
- **为什么值得**：`lse_mma_kernel_bal<512,...>`（`mma`+`cp.async`，O39/O59 的 cfg6）在 S1024H2 上
  ncu **Duration 19.62µs / Waves 0.48 / Compute 30.77%**——半个波都铺不满、纯发射受限；TMA 版直接
  填满一个波。
- **性能（同 binary A/B，event，iters=200）**：LSE-only preprocess **S256H2 1.80× / S512H4 1.60× /
  S1024H2 1.50×**；端到端 total **0.0481→0.0400（1.20×）/ 0.1052→0.0977（1.08×）/
  0.1601→0.1510（1.06×）**（MLA main 占 74–82%，故端到端收益被摊薄）。
- **ncu（S1024H2 LSE）**：mma **19.62µs / Waves 0.48 / Compute 30.77% / L1TEX 24.41%** → TMA
  **10.34µs（1.90×）/ Waves 0.97 / Compute 15.08% / L1TEX 15.56%**、2 CTA/SM ⇒ 新墙回到 LSE 固有的
  tile 级 `mma wait`（与 D=128 的 TMA LSE 同）。
- **数值**：D=128 与历史**逐位不变**（S4096 `2.635/2.644/3.216e-1`）；D=512 只差 LSE 的 fp32 求和
  次序（≤4.7e-3，fp8 容差内）；`--ci` fp8 单/两文件 gate **8.583e-6 OK**、`--check docs/04` OK
  （已 `--doc-table-apply` 同步 3 行 MLA causal）。单/两文件 device 由 `sync_onefile_device.py`
  逐字同步。
- **下一步**：把本步 dtype 化到 fp16/bf16（`HD=512` 需 8 个 K=64 chunk、`TILE=64KB`、192KB smem
  ⇒ 1 CTA/SM）；main 的 L2 `red` 墙仍受本卡硬墙锁定。
- 详见 `docs/03` §98；原始输出 `src/fp8/fa_bwd_fp8_o74_*`。

### 5.84 O75：MLA（D=512）causal LSE 上 **4D-TMA**（fp16/bf16 泛化，第一百七十轮，正结果，默认）

- **一句话**：落实 §5.83 的「dtype 化到 fp16/bf16」，把 fp8 O74 的 `NCH=HD/128` 泛化逐字搬到
  fp16/bf16（box 内维 128B = 64 个 fp16/bf16，故 `NCH=HD/64`；`HD=512` 用 **8 个 K=64 chunk**），
  `NCH==2`（HD=128）走原 `wgmma_qkt64_tma` ⇒ **HD=128 逐位不变**；host `lse_tma` 默认
  `(D==512&&causal)?1:0`、causal 优先 `lse_mma_kernel_bal_tma<512,1>`（smem 197,696B ⇒ 1 CTA/SM）。
  **至此三 dtype × MLA 的 causal LSE 也统一到 4D-TMA。**
- **为什么值得**：`lse_mma_kernel_bal<512,1,false,128,16>`（mma+`cp.async`，cfg6）在 S1024H2 上 ncu
  **Duration 20.26µs / Compute 30.39% / L1TEX 42.10%**（发射受限）；TMA 版把 load 指令/地址运算交给
  TMA 引擎。
- **性能（同 binary A/B，event，iters=50）**：LSE-only preprocess **fp16 S256H2 1.47× / S512H4 1.20× /
  S1024H2 1.56×**（bf16 1.48×/1.21×/1.54×）；端到端 total fp16 **0.0420→0.0359（1.17×）/
  0.1092→0.1051（1.04×）/ 0.1785→0.1666（1.07×）**（MLA main 占 74–82%）。
- **ncu（S1024H2 LSE）**：mma **20.26µs / Compute 30.39% / L1TEX 42.10% / 2 CTA/SM** → TMA
  **12.99µs（1.56×）/ Compute 10.49% / L1TEX 16.63% / 1 CTA/SM**；bf16 TMA 12.77µs 逐项一致 ⇒ 搬迁
  方式升级，非 DRAM/L2/算力 bound。
- **数值**：fp16 MLA causal `S256H2 1.638/1.582/1.753e-3`、`S512H4 2.516/2.916/1.724e-3`、
  `S1024H2 1.987/1.712/1.848e-3`（bf16 同量级 ~1e-2），与 mma 版**打印逐位相同**；D=128 定长
  causal/full 与 varlen 回归逐位不变；`--ci` 三 dtype gate 全 OK（fp16 1.953e-3 / bf16 7.812e-3 /
  fp8 5.722e-6）、单/两文件一致性 worst 7.812e-3 OK、`--check docs/04` OK。单/两文件 device 由
  `sync_onefile_device.py` 逐字同步（fp16 `#include <cuda_runtime.h>` / bf16 `using bf16 = …` 为界）。
- **下一步**：main 的 L2 `red` 墙仍受本卡寄存器/smem 硬墙锁定（见「阻塞」）。
- 详见 `docs/01` §25、`docs/01b` §6bb；原始输出 `src/fp16/fa_bwd_fp16_o74_*`、
  `src/bf16/fa_bwd_bf16_o74_*`。

### 5.85 O76：新增 **head_dim=256** 支持（fp8，第一百七十一轮，能力覆盖）

- **一句话**：补齐 fa-bwd fp8 反向在 `D=128` 与 `D=512` 之间缺失的 **`head_dim=256`**（FA/TE
  的反向都支持到 256）。**纯 host dispatch + 一个 `VPT=8` 量化实例**，device 通用代码一行未改
  （单/两文件逐字同步）：放形状守卫、加 `Fp8Cfg<256,64,32>` smem 档、`quantize_*_warp_kernel<8>`
  （`VPT=D/32`）、`lse_mma_kernel_bal<256,1>`（causal）/`<256,1,true>`（full）、
  `launch_bwd_main<256,64,32,false>`（mma，4-warp；wgmma↑128、TMA↑128/512）。
- **为什么值得**：主 kernel 对 `HD` 本就只要求 `HD%128==0`（`HD/NTW=2` 时 `kRegDq` 自动关、与
  MLA 同），LSE/delta/quant 也全是 `HD`/`VPT` 模板，所以 256 是**几乎零 device 成本**的形状扩展。
- **数值（ours vs fp32 ref）**：causal MHA `(1,1024,8,256)` 2.630/2.795/3.589e-1、
  `(1,2048,8,256)` 2.220/2.835/3.584e-1、GQA kv4 `(1,1024,16,256)` 2.477/4.455/6.157e-1、
  full `(1,1024,8,256)` 4.972/5.571/4.092e-2——与 `D=128` 的 fp8 噪声同量级；两半 `d[0..127]`/
  `d[128..255]` 误差同量级。单/两文件一致性 worst **2.384e-6**（gate 1e-4 OK）；全量 `--ci`
  77 case 三 dtype 全绿、`--check docs/04` OK（198 行）。
- **性能**：main-only `(1,1024,8,256)` **29.7 TF（峰值 1.50%）**、`(1,2048,8,256)` **34.3 TF
  （1.73%）**、GQA 30.9 TF、full 19.1 TF。FA/TE 的 **fp8 反向不支持 256**，同 shape 只给
  fp16/bf16 参照：TE fp16 `(1,1024,8,256)` 216.3 TF、`(1,2048,8,256)` 320.1 TF（FA2 168.5/228.0；
  FA3 本机只编 HDIM128 → NA）。
- **ncu（S1024H8 causal 主 kernel）**：Duration 327µs、**Registers 254**、Dyn smem 117.76KB、
  **Block Limit Shared Mem 1 → 1 CTA/SM、occ 6.25%**、Waves 3.88、Compute 15.97%、L2 41.41%、
  DRAM 3.53%、L1/TEX 30.82%、**No Eligible 78.87%** ⇒ **低 occupancy + 延迟/并行度受限**（与
  MLA `D=512` 同类：`HD` 翻倍让 GEMM3/4/5 累加器/P·S 缓冲把寄存器顶到 254、smem 115KB 卡 1 CTA）。
- **下一步**：① 把 `D=256` dtype 化到 fp16/bf16（分派面大，留 backlog）；② `D=256` 的 fp8 main
  接 wgmma/TMA（受 fp8 wgmma `HD=128` 锁死）；③ main 的 L2 `red` 墙（F7/F6 已判死，见「阻塞」）。
- 详见 `docs/03` §99；原始输出 `src/fp8/fa_bwd_fp8_o76_d256_*.out.txt`、
  `src/fp8/fa_bwd_fp8_o76_ncu_d256_main_s1024.out.txt`、`src/fa_bwd_o76_d256_baseline_fp16bf16.out.txt`。

### 5.86 O77：F6-③（TMA L2 promotion）收口 + fp8 main 的「寄存器/smem/occupancy」三证（第一百七十二轮，负结果 / 平台期确认）

- **一句话**：把「fp8 继续」里唯一未测的 **F6-③（Q/dO 的 4D-TMA cache hint / L2 persist 减重读）**
  实测判决为**负结果**，并用 ksplit、编译期开关、ptxas 寄存器账三组证据把「fp8 默认 main 在本卡
  已到硬件平台期」钉死。默认路径数值逐位不变。
- **F6-③**（纯 host，`--l2promo=0/1/2` = NONE/L2_128B/L2_256B）：S4096 causal 同 binary 交替
  **1.7913 / 1.7936 / 1.8052 ms**——噪声内、256B 略负。根因：ncu 默认 main **L2 命中 97.08%、
  DRAM 4.32%**，Q/dO 的 ksplit 重读本就在 L2 命中，promotion 改不了 L2 扇区总量。**L2 persist /
  cache hint 对 fp8 main 无效**。
- **ksplit 复扫**：1/2/4/**8**/16 = 2.154/1.913/1.805/**1.790**/1.873 ms ⇒ auto=8 最优；少切分
  red 更低但 grid 铺不满、延迟暴露，净更慢。F6-① 被并行度锁死（同既往结论）。
- **编译期开关复测**（`FA_WS1/FA_ILV/FA_ILV34/FA_R4` 及组合，当前 TMA 构建）：全部 1.789–1.818ms，
  baseline 1.791ms ⇒ 噪声内或有损；`FA_R4`（16B red）再次确认不降 L2 扇区。**默认关是正确的。**
- **寄存器账**：默认实例 **168 regs / 40B spill / 74.82KB smem**，`__launch_bounds__(128,3)` 上限
  170 ⇒ 顶格溢出 10 个长生命期值。4 CTA/SM 需 ≤128 regs **且** ≤58.1KB smem，两者都达不到；
  去 spill 需 >170 regs（ptxas 需求 ~180）。ncu：local memory 占 L1TEX ~7.7%、L2 ~4.5%；
  头号 stall `wait 27.5% + short_scoreboard 21.7%`（3 warps/scheduler），要藏它需更多 warp /
  跨-tile 软流水，均被同一双墙锁死。
- **结论**：fp8 默认 main（S4096 main 1.553ms / total 1.791ms / **76.7 TFLOPS**、TE FP8 ~6.4×）
  在本卡无更多软件杠杆；减 L2 搬运量只剩「换工作划分」（撞 128-reg/116KB 双墙，F6/F7/O17b/p160/
  p168 五路全判死）、藏延迟只剩「换卡 / 多 warpgroup 摊累加器」。
- 详见 `docs/03` §100；原始输出 `src/fp8/fa_bwd_fp8_o77_{l2promo_ab,ksplit_sweep,macro_ab,
  ptxas_spill,ncu_local_src}_s4096.out.txt`。

### 5.87 O78：端到端 overlap（quant/LSE 与 main 跨 stream）—— 负结果（第一百七十三轮）

- **一句话**：把 §100「下一步候选 ②（端到端重叠）」正式实现——按 head 分块把 LSE 与 main 放到
  两条非阻塞 stream 流水——实测**单调更慢**（+6%~+52%），根因是**切分 main 栅格引入的尾波量化
  损失**远大于被隐藏的 LSE。默认路径未改。
- **机制/实现**（纯 host，`--ovlp=N`）：新 `make_map_fp8_chunk` 让 4D-TMA 描述符的 head 计数与
  物理行距解耦（`dims[2]=hc`、`strides` 用全 H），配合指针偏移 ⇒ **device 一行未改**；`sA` 发
  LSE(k)、`sB` 发 main(k)、event 依赖、quant 整体在 default 前置。
- **诊断**（`--ovltest=1`，整块 main || 整块 LSE、只计时）：`main 1.555 + lse 0.121 = serial 1.676`
  → `concurrent 1.594（0.951×）` ⇒ **内核级重叠可行**（LSE 藏住 ~0.082ms、67%）。
- **实测**（S4096 causal MHA，同 binary）：ovlp=0/2/4/8/16 = **1.796 / 1.903 / 2.119 / 2.380 /
  2.731 ms**（单调 +6%→+52%）。**控制**（`--ovlnolse=1` 只测分块 main 串行）：chunks=2/4/8 →
  full main **+5%/+15%/+29%**。
- **归因**：full main 是 8192 CTA / 20.7 波、尾波 ~5%；切 N 段后每段 10.3/5.2/2.6 波、尾波占比
  ~10/19/38%。**尾波损失 > LSE 可藏量（4.5%）**；并发时 LSE 与 main 争 SM 又额外拖慢 main。
- **结论**：本 shape 的 fp8 main 已按「3 CTA/SM、铺满 20.7 波」调优，head/seq 切分必回尾波；
  吃这 ~5% 只能「把 preprocess 融进 main」（WS prologue，同 F3b 大改）。默认逐位不变。
- 详见 `docs/03` §101；原始输出 `src/fp8/fa_bwd_fp8_o78_{ovltest,sweep,ncu_main,baseline_te}*`。

### 5.88 第 174 轮（O79）：TE SASS 的 QGMMA **RS_TN** 发现 + fp8 wgmma RS 冒烟——重新打开 GEMM3/4/5 wgmma 路径

- **动机**：ROADMAP『fp8 专项冲刺』主线是「fp8 main 从 mma.sync 切到 wgmma」（F1→F5），但 F1
  只把 GEMM1/2 换成 wgmma；GEMM3/4/5 仍是 `HMMA`，「阻塞」记为「fp8 wgmma 无转置操作数」。
  本轮按任务要求用 `ncu --page source --print-source sass` 逐指令对照 TE 的反向 kernel。
- **发现**：TE 的 `..._flash_bprop_wgmma_f8_..._64x64x128`（384 线程、grid=64）**16×QGMMA**
  （8×`64x64x32` + 8×`64x128x32`）、**0×HMMA**；其中一半带**寄存器 A 操作数**
  （`QGMMA.64x128x32... R152, R216, gdesc[UR20], R152`）= CUTLASS `RS_TN`。即「需转置的那个
  操作数」可经 `ldmatrix` 进寄存器 A，绕开「B 必须物理转置」。此外 TE 有 **24×STSM + 20×LDSM**
  （含 `MT88.4` 转置变体）——它靠**矩阵搬运指令做配对粒度的转置**，不是逐字节 scatter。
- **冒烟（正结果）**：`src/fp8/fa_bwd_fp8_wgmma_rs_smoke.cu` 验证 `ldmatrix.x4`（行主序 [64,K]
  K-major，mma 同款 `arow/acol`）取回的 4×u32 **恰是** `wgmma.m64n32k32` RS_TN 的 A 片段
  （`ALayout_64x32`）：e4m3×e4m3 与 e5m2×e4m3 **max_abs=0 PASS**。
- **修正后的真阻塞**：RS 只解决 A；dV/dK/dQ 仍各有**一个逐字节转置**需求（dOᵀ/Qᵀ/Kᵀ），而 fp8
  `ldmatrix.trans` 只交换 b16 配对方向。TE 的解法是 `stmatrix/ldmatrix` 配对粒度搬运（44 条）。
  ⇒ GEMM3/4/5 wgmma 路径**明确但工程量中等偏大**，非本轮 quick lever。默认路径逐位不变。
- 详见 `docs/03` §102；原始输出 `src/fp8/fa_bwd_fp8_p174_sass_te_vs_ours.out.txt`、
  `src/fp8/fa_bwd_fp8_wgmma_rs_smoke.out.txt`。

### 5.89 第 175 轮（O80）：fp8 `stmatrix`/`ldmatrix.trans` + PRMT **逐字节转置**冒烟 PASS + wgmma RS GEMM3 端到端 PASS——F3b 主体去风险

- **动机**：第 174 轮（§5.88）把 F3b 的下一步定为「① `stmatrix` 冒烟 → GEMM3/4/5 逐个换 RS wgmma」。
  本轮把这条路径的**最不确定一步**（fp8 逐字节转置）用最小复现钉死，并跑到 wgmma RS 端到端。
- **TE SASS 复核**（`ncu --page source --print-source sass`，`harness/te_fp8_ncu.py`）：TE 的
  `..._flash_bprop_wgmma_f8_..._64x64x128` 里，GEMM3/4/5 的操作数是
  `LDSM.16.MT88.4`（`ldmatrix.x4.trans`，转置读）→ **`PRMT R, R, 0x5140/0x6420/0x7531, R`**
  （逐字节重排 b16 配对方向）→ `STSM.16.M88.4`（`stmatrix.x4` 落盘）造出来的；
  A 用 `LDSM.16.M88.4`（非转置）读回喂 `QGMMA ... R216(寄存器A) gdesc`（RS_TN）。
- **冒烟（正结果）**：新增 `src/fp8/fa_bwd_fp8_stmatrix_smoke.cu`，两阶段：
  1. **逐字节转置**：对 [R][C] fp8 行主序 tile，warp 用 `ldmatrix.x4.trans`（4 个 8×8 b16 矩阵并排）
     + `__byte_perm(reg, reg>>16, 0x5140)` + 两次 16-bit 存，得到 [C][R]（R 连续）。
     推导的 lane 映射 = `lo16=(X[2p][2k],X[2p+1][2k])`、`hi16=(X[2p][2k+1],X[2p+1][2k+1])`
     （`p=lane&3, k=8·reg+lane>>2`）。四种 shape `[128][64]/[64][64]/[64][128]/[32][128]`
     **全部 mismatches=0（逐字节）PASS**。**关键坑**：PRMT selector 是 **`0x5140`**（byte0←a0,
     byte1←b0, byte2←a1, byte3←b1），写反成 `0x5410` 会退化成恒等、只错一半字节——与 fp8 kernel
     里已有的 `__byte_perm(q0,q1,0x5140)` 交织写同源。
  2. **wgmma RS GEMM3**：`C[j][d]=Σ_m P[m][j]·dO[m][d]`（dV 形状，BM=128/BN=64/HD=64）。
     A=Pᵀ[64][128]（逐字节转置成行主序 K-major，`ldmatrix.x4` 非转置取片段），
     B=dOᵀ[64][128]（逐字节转置**直接写进 SW128 tile**），跑 `wgmma.m64n32k32.e4m3.e5m2`
     （4 个 k=32 步 × 2 个 n=32 块），与 CPU `max_abs=0.000e+00 bad=0 PASS`。
- **SASS 证据**：`gemm3_wgmma_kernel` = **8×`QGMMA.64x32x32.F32.E4M3.E5M2`**（其中 R56/R60/R64/R68
  为寄存器 A）+ **0×HMMA** + `LDSM.16.MT88.4`（转置读）+ `PRMT 0x5140`；ptxas **74 regs / 0 spill**。
- **结论 / 对 F3b 的意义**：① fp8 的「逐字节转置」不再是死结——`ldmatrix.trans + PRMT` 可用且
  逐字节正确；② 转置操作数可直接落成 wgmma 消费的 SW128，RS 端到端正确。**剩余工程量**：
  把这套 `transpose_store` 接进 `fp8_mma_body` 的 GEMM3/4/5（替换 `mma_block_bt` 的
  `dOp/Qp/Kp` 配对读），并把 GEMM3/4 的 M=BN 从 32 提到 **64**（wgmma 最小 m64；GEMM5 的
  M=BM=64 已满足）。默认路径一行未改、数值逐位不变。
- 详见 `docs/03` §103；原始输出 `src/fp8/fa_bwd_fp8_stmatrix_smoke.out.txt`、
  `..._sass.out.txt`、`..._ptxas.out.txt`。

### 5.90 第 176 轮（O81）：fp8 **GEMM5（dQ）切到 wgmma RS**（正结果，默认）——F3b 主体第一块落地

- **承接**：O79/O80 把「GEMM3/4/5 上 wgmma」的操作数死结打开（RS 允许 A 在寄存器；
  `ldmatrix.x4.trans + PRMT 0x5140` 逐字节转置）。默认 fp8 main 里唯一满足 wgmma 最小 **m64**
  的 GEMM 是 **GEMM5（dQ，M=BM=64）**，先把它落进真实 `fp8_mma_body`。
- **三个新事实**（冒烟 `fa_bwd_fp8_wgmma345_smoke.cu`，全 max_abs=0）：
  ① **no-swizzle K-major 描述符**可用，但 CUTLASS canonical INTERLEAVE 布局是
  `((8,n),2):((1,SBO),LBO)`（8 行 stride 恒 1 个 uint128），K=32 时 **LBO_u=8/SBO_u=16**；
  ② 从 **SW128 源**做逐字节转置只需把 `transpose_store` 的源地址改成 `sw128_off_fp8`（PRMT 不变）；
  ③ 端到端 `dQ=dS2·K`（A=dS2 e5m2 经 ldmatrix、B=Kᵀ no-swizzle、`m64n32k32.e5m2.e4m3` ×4）精确。
- **落地**（`-DFA_WGMMA5`，**默认 1**）：KVTMA 路径的 Kp 配对构建换成从 SW128 K stage 转置成
  Kᵀ INTERLEAVE（**复用 Kp 缓冲 ⇒ smem 零增长**），GEMM5 换 `wgmma RS` + CLayout_64x32 epilogue
  （`dqacc5[4][4][4]`）。门控 `WGMMA && KVTMA && HD==128 && BN==32 && !DET && !DQONLY && kRegDq`，
  其余路径逐字不变。**关键坑**：generic 写 smem 后必须 `fence.proxy.async.shared::cta`
  （`bulk_reduce_fence()`），否则 wgmma 的 async proxy 读到半成品，dQ 出现 **O(1) 偶发错**
  （首版 O64/O41 A/B 的 dq 差 2.36/5.14；补 fence 后回到 1e-7）。
- **数值**：S512/S4096 causal、GQA、full 四例 vs fp32 ref relL2（dq/dk/dv）8.15–8.18% /
  8.22–8.30% / 6.32–6.71%，与 `FA_WGMMA5=0` **逐位打印相同**；`--ci --dtype fp8 --hopper`
  单/两文件 gate 7.629e-6 OK、docs/04 OK。
- **性能（同 binary A/B，S4096 H16）**：main **1.5564→1.4810ms（1.051×）**、total
  **1.7885→1.7235ms（1.038×，76.84→79.74 TF）**。ncu：Duration 1.56→1.49ms、
  `smsp inst` **−11.1%**、HMMA **−26.9%**、`lts read` −9.3%、**`lts red` 114.52M 不变**、
  regs 168 不变。⇒ 收益是**指令路径**（GEMM5 由 16×HMMA/线程组 → 4×QGMMA + 省 Kp 构建），
  与「降 L2 搬运量」正交。S4096 相对 TE FP8（0.3025ms/908.7TF）**5.93× → 5.70×**。
- **剩余（F3b 主体）**：GEMM3/4 的 M=BN=32 < m64 ⇒ 需 BN=64（翻倍 dVacc/dKacc 撞 255 regs，
  p160 已判负，需先解寄存器账）；WS 完整化。详见 `docs/03` §104；原始输出
  `src/fp8/fa_bwd_fp8_wgmma345_smoke.out.txt`、`src/fp8/fa_bwd_fp8_o81_ab_wgmma5_{0,1}_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o81_ncu_wgmma5_{0,1}_s4096.out.txt`、`src/fp8/fa_bwd_fp8_o81_ci_fp8.out.txt`。

### 5.91 第 177 轮（O82）：fp8 **GEMM3/4（dV/dK）切到 wgmma RS（M 零填充 m64）**（负结果，默认关）

- **承接**：O81 把 GEMM5（dQ，M=BM=64）落进 wgmma RS。本轮把默认 fp8 Hopper main 剩下的
  GEMM3(dV)/GEMM4(dK)（M=BN=32）也换 wgmma RS，把 F3b「全 GEMM wgmma」再推一块。
- **障碍/绕法**：M=32 < wgmma 最小 m64。TE 用 BN=64；p160/O21 已判 BN 32→64 在本卡为负
  （累加器翻倍撞 255 regs）。故用**零填充**：A 的 M 补到 64，warp 2/3 的 A 置 0、输出行 32–63 丢弃。
- **新 helper**：① 紧凑 no-swizzle K-major（`noswz_k_off_c` + `transpose_sw128_to_noswz<R,C>`，
  K=BM=64；O81 的 `inter_k_off_fp8` 的 LBO=8 只对 K≤32 无冲突，K=64 第 4 位会撞）；②
  `wgmma_m64n32k32_rs_e4e5`（GEMM3），GEMM4 复用 O81 的 `rs_e5e4`。Q/dO 转置复用 Qp/dOp
  缓冲（smem 零增长）；累加器按 2 个 n-tile 一组（`acc[2][16]`）压低寄存器。
- **数值（正确）**：S4096 vs fp32 ref relL2 8.149/8.263/6.489%、max_abs 2.635/2.641/3.218e-1，
  与 `FA_WGMMA34=0` **逐位打印相同**；A/B dq 1.19e-7、dk 9.13e-4、dv 2.28e-3。
  `--ci --dtype fp8 --hopper` 全绿（gate 7.629e-6、docs check OK）。
- **性能（负结果，同 binary A/B，S4096）**：main **1.4738→1.5289ms（0.964×）**、total
  **1.7119→1.7694ms（0.967×）**。ncu：inst **−7.3%**、L1/TEX 74.4→67.6%、L2 78.8→75.5%、Compute
  45.1→40.3%，但 **No Eligible 53.2→58.2%**、Duration 反升。⇒ **L2 `red` bound（75%，wgmma 不减
  `red`）+ 延迟 bound**；零填充的 2× 无用张量功 + Q/dO 转置 + wgmma wait 抵消了指令路径收益。
- **判决**：F3b「无 BN=64 时上 GEMM3/4 wgmma」子路线判负。真做 GEMM3/4 wgmma 仍须 **BN=64**
  （自然 m64），需先解寄存器账（256/384 线程摊累加器，与 F7/p160 同源）。默认一行未改。
  详见 `docs/03` §105、`docs/04` §49；原始输出 `src/fp8/fa_bwd_fp8_o82_ab_wg34_{0,1}_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o82_ncu_wg34_{0,1}_s4096.out.txt`。

### 5.92 第 178 轮（O83）：fp8 **GEMM3/4「真 m64」三条路资源核算收口** + **`red` 成本分解**（负结果/收口）

- **承接**：O82 的「下一步候选 ①：真 m64 化（BN=64 + 多 warpgroup 摊累加器）」。
- **默认 main 的 bound 再钉（ncu，S4096 H16 causal）**：L2 **79.26%**、其中 `red` **114.52M
  扇区 = L2 总扇区的 80%**、`read` 28.05M、`write` 0.39M；DRAM **4.41%**；
  `sm__pipe_tensor_cycles_active` **11.15%**、`sm__issue_active` 46.66%、warps active 18.33%
  （3 CTA/SM/168 regs）；stall `short_scoreboard 1.81 + wait 1.55`。⇒ **red-bound + 张量核空转**，
  这本身即说明「F3b 把 GEMM3/4 换 wgmma」不可能有收益（wgmma 对 `red` 一字不减、张量核非瓶颈）。
- **「真 m64」三形式资源核算（3 CTA/SM 上限 77,482B / 170 regs）**：
  (a) **配对两 KV tile**（A=[Pᵀ₀;Pᵀ₁]）需 +4KB 存被配对 tile 的 `Ap`/`dS3` ⇒ 78.8KB > 77.5 ⇒
  **掉 2 CTA/SM**（p160/O21 已证负）；(b) **转置 GEMM** dVᵀ=dOᵀ·P（M=HD=128，smem 中性）但
  **输出取向翻转** ⇒ `acc`（行=d、列=j）写 `dv[j][d]`，相邻 lane 写 stride=`Hkv·HD`=512B 的
  不同行 ⇒ red 扇区每 warp-inst 8→32、**red 墙恶化**；(c) **BN=64** 令 `Ps`/`Ss`=`[64][BN+5]`
  （各 17.7KB）+`K/V`/`dS2` 翻倍 ⇒ smem≈120KB ⇒ **1 CTA/SM**。⇒ **三条全部停不到 3 CTA/SM，
  candidate ① 按资源账收口**（与 F6/F7/p160 同源）。
- **`red` 成本分解（新诊断 `FA_RED_STORE`，plain store 替原子，数值错误仅诊断）**：默认原子
  main **1.493ms** vs plain store **1.368ms（1.09×）** ⇒ 原子 RMW ≈ 0.125ms(~8%)；对照 O42
  短路整个 epilogue **0.94ms** ⇒ **写流量本身 ≈ 0.43ms(~29%)**。⇒ `red` 大头是写流量、
  与归约指令/宽度/机制无关（复证 O42/O67），**唯一杠杆仍是减少贡献 CTA 数（工作划分）**。
- **新观察（backlog）**：GQA/MQA 下同 KV 头被 `H/Hkv` 个 Q 头共享，跨 Q 头本地累加 dK/dV 再原子
  可把 `red` ÷`(H/Hkv)`（MQA 最多 ÷64）；**阻塞**：dQ 的 `kRegDq` 寄存器累加需「一 CTA 一 Q 头」，
  跨头合并会使 `dqacc` ×`(H/Hkv)` 撞寄存器墙（仅 GQA/MQA 有效、MHA 零收益）。
- 默认路径一行未改（`FA_RED_STORE` 默认 0、单/两文件同步），数值 vs ref **逐位不变**
  （2.635/2.644/3.216e-1）。详见 `docs/03` §106；原始输出 `src/fp8/fa_bwd_fp8_o83_*`。

### 5.93 第 179 轮（O84）：fp8 **`head_dim=256` 主 kernel 默认切 wgmma**（正结果/默认）

- **承接**：O76（第 171 轮）把 fp8 反向的 `head_dim` 覆盖补到 **256**，但当时 host 明确让
  `D=256` 走 **非 wgmma 的 mma 主 kernel**，并在「下一步」里点名「D=256 的 fp8 main 接
  wgmma/TMA（受 fp8 wgmma 只做 HD=128 的约束）」。本轮落实 **wgmma（非 TMA）那半**。
- **发现**：fp8 的 SW128 K-major helper（`sw128_off_fp8`/`sw128_k32_addr`/
  `make_desc_sw128_fp8`、`wgmma_mn32_issue`/`wgmma_qkt64_fp8`）**本就按 `SBO=(HD/128)*1024`、
  `s < HD/32` 编写**，`HD=256` 的 canonical 布局 `[row/8][2 k-blocks][8][128]` 与描述符
  逐字相容。真正的锁只是一条保守的 `static_assert(!WGMMA || HD==128)`。
- **改了什么**：把 `static_assert` 放宽到 `HD==128||256`（device）；host 把 `D=256` 分派接到
  `launch_bwd_main<256,64,32,false,true>`（`WGMMA=true`，仍 cp.async 载入），加
  `--d256wgm=0/1` 同 binary A/B。**只动 `D==256`，D=128/512 一行未改**（单/两文件逐字同步）。
- **性能（4 个 D=256 case 同 binary A/B，event iters=50）**：main **1.145×（S1024 causal）/
  1.289×（S2048 causal）/1.272×（S1024 full）/1.211×（GQA kv4）**，total 同量级
  （1.176/1.270/1.220/1.214×）。数值与 mma 版同量级（差 ≲1e-3）、与 O76 记录一致。
- **ncu（S1024 causal）**：`D=256` mma 动态 smem **117,760B（1 CTA/SM）** → wgmma SW128
  **115,712B（≤116,224B ⇒ 2 CTA/SM）**，warps_active 6.25%→**11.78%**；`smsp__inst_executed`
  49.72M→**46.01M（−7.4%）**；SASS **16×QGMMA + 192×HMMA + 92×LDSM**（mma 版无 QGMMA）；
  `lts red` 一字不变（工作划分没动）。⇒ 收益 = **跨过 2 CTA/SM 门槛 + 指令路径**。
- 单/两文件 gate worst **6.676e-6** OK、`docs/04` auto 表已同步（D=256 行 +fp8 次序噪声）。
  `D=256` 仍未上 4D-TMA（留 backlog）。详见 `docs/03` §107；原始输出
  `src/fp8/fa_bwd_fp8_o84_d256_wgmma_ab.out.txt`、`..._o84_ncu_d256_s1024.out.txt`、
  `..._o84_d256_onefile.out.txt`。

### 5.94 第 180 轮（O85）：fp8 `head_dim=256` 的 Q/dO 切 **4D-TMA（chunk-major）**（中性，默认关）

- **一句话**：落实 O84 的「D=256 Q/dO 上 4D-TMA」。因 TMA 一个 SW128 box 只能搬 128 列，
  `D=256` 要 2 个 box ⇒ smem 物理布局变 **chunk-major `[k/128][row/8][8][128]`**（与 LSE
  一致），主 kernel 新增 `sw128c_off_fp8`/`sw128c_k32_addr`/`wgmma_mn32_issue_cm`（A 走
  chunk-major、B 仍 rg-major；`D=128` NCH=1 时逐位相同）。**结论：同 binary A/B 净中性**
  （S1024 +1.3–1.7%、S1024 full +1.3%、S2048 −1.1%、GQA −3.7%；ncu 隔离 Duration 245.5
  vs 255.6µs=1.041×、指令 −3.2%）⇒ **默认关**（`--d256tma=1` opt-in）。
- **关键坑**：`smem_bytes_wgmma_tma = smem_bytes_wgmma + 64` 的 64B 对 `D=128` 无害，但对
  `D=256` 把 launched smem 115,712→115,776B、allocated 116.74→116.86KB，**2 CTA/SM 直接掉到
  1**（ncu `occupancy_limit_shared_mem=1`），首版慢 9–17%；改成 `HD>128 ? 0 : 64` 后恢复。
- **数值**：chunk-major 只改 smem 布局/搬运，不变量化口径 ⇒ relL2 vs ref 与 O84 **逐位相同**
  （dq/dk/dv 8.332%/8.435%/6.464%），tma1-vs-tma0 max_abs ~1e-7（atomic 次序）。
- **保留价值**：chunk-major SW128 基建可复用于后续 `D>128` 的 K/V TMA 或 MLA；要让 D=256 TMA
  转正需把 K/V 也 TMA 化并与 cp.async 重叠（K 双缓冲顶穿 116KB，backlog）。
- 见 `docs/03` §108；原始输出 `src/fp8/fa_bwd_fp8_o85_d256_ab.out.txt`、
  `..._o85_ncu_d256_s1024.out.txt`、`..._o85_accuracy_d256.out.txt`。

### 5.95 第 181 轮（O86）：GQA/MQA 跨 Q 头折叠 dK/dV —— **结构性不可行（负结果/收口）**

落实 F4b 的新 backlog（GQA/MQA 把 `red` ÷`(H/Hkv)`，MQA 最多 ÷64）。本轮先用 ncu 钉死收益
上界，再对三种 loop order + cluster 逐条解析核算：

- **收益上界（ncu 实测）**：默认 fp8 `kvtma` main 的 L2 `red` **只由 Q 头数 H 决定**——
  MQA（H64kv1）与 GQA（H64kv4）的 `red` **逐字节相同 32,833,536**；H32→16.42M、H40→20.52M
  精确落 `red≈(H/32)×16.42M`。⇒「跨 Q 头折叠」可把 dK/dV 的 `red` 压 (G-1)/G（MQA ×64），
  是 F4b 里唯一没做、上界最大的杠杆。
- **为何拿不到**：dQ 与 dK/dV 的 loop-order 偏好相反——(A) head 内层可折叠 dK/dV 但 dQ 丢掉
  寄存器累加（+17.8M 扇区）且 Q/dO 按 `(nt,head)` 重载（MQA +140M 扇区，致命）；(B) head 外层
  保住 dQ/Q 读但 dK/dV 累加器须常驻整个 head 循环（ksplit=4 已 128KB/dK，smem 放不下）；
  (C) 两遍重算回到同一墙。
- **cluster 也不成立**：O25（fp16/bf16，§5.x）实测 `red` 精确减半但慢 **7.4×**（远程逐元素
  smem 原子）；它把 N 次全局原子换成 N 次远程原子是 **1:1**，GQA 只放大 `G` 不减少每 CTA 的
  远程原子数 ⇒ 同阶成本，负结果原样成立。
- **判决**：本卡（74.8KB smem/170 regs/3 CTA/SM）**结构性不可行**，默认路径一行未改、数值逐位
  不变。解锁需换卡 / 硬件 scatter-reduce / 多 warpgroup WS（同 F6/F3b 寄存器墙）。

详见 `docs/03` §109；原始输出 `src/fp8/fa_bwd_fp8_o86_gqa_red_probe.out.txt`。

### 5.96 第 182 轮（O87）：F3b **GEMM3/4 wgmma 的 wait-schedule 变体** —— **仍负（收口）**

承接 O82（`-DFA_WGMMA34=1`：dV/dK 切 wgmma RS、M=BN=32 零填充到 m64，实测 0.964×，诊断
「wgnma `wait` + 零填充/转置抵消指令收益」）。本轮只改 **fence/commit/wait 时序**，试三个
「藏 wait」写法，看能否翻正：

- **变体 A 合并 commit/wait**：4 个 n-tile 的 8 条 wgmma 一次 fence/连发/commit/wait。
  S4096 main **1.5901ms（0.930×）**——比 O82 原版（1.5435/0.958×）更慢。
- **变体 B `wait_group<1>` 流水**：两组各 commit，先 `wait_group<1>` 做 `ng=0` epilogue、
  再 `wait0` 做 `ng=1`。**1.5778ms（0.937×）**。
- **变体 C GEMM5（O81 正结果路径）wait 流水**：**1.4957ms（0.989×）**中性偏负。

**ncu（S4096）**：`WGMMA34=0→1` 的 `smsp inst` 641.86M→594.83M（−7.3%）、`short_scoreboard`
1.82→1.41，但 **`lts op_red` 114,524,160 一字不变、`wait` 1.56 不降、Duration 1.49→1.55ms**。
⇒ 默认 fp8 main 是 **L2 `red` bound**，wgmma 对 `red` 一字不减（O83），**任何只改指令/wait
时序、不改「每元素贡献 CTA 数」的改动都不可能转正**——三个变体独立复证。寄存器账复现
（默认实例 168 regs/40B spill；合并 `acc[4][16]` 未进一步溢出 ⇒ 纯时序问题）。

**判决**：F3b 的「无 BN=64 时 GEMM3/4 切 wgmma」路线在 wait 维度**再无空间**，默认路径一行未改、
数值逐位不变（三变体 vs fp32 ref 的 max_abs 均 2.635/2.644/3.216e-1）。唯一剩余真 m64 路 =
**BN=64 + 多 warpgroup（256/384 线程）**，被 3 CTA/SM 的 77,482B/170-reg 硬墙锁死（O83），
与 F6/F7/p160/O86 同源。**F3b 至此与 F4b/F6/F7 一样收口为「本卡无软件解」。**

详见 `docs/03` §110；原始输出 `src/fp8/fa_bwd_fp8_o87_wg34_{0,1_orig,1_mergewait,1_pipewait}_s4096.out.txt`、
`..._o87_g5pipe_s4096.out.txt`、`..._o87_ncu_wg34_{0,1}_s4096.out.txt`。

### 5.97 第 183 轮（O88）：fp8 非 main 栅格封顶 + 宏复扫 + `D=256` K/V-TMA 资源收口 —— **负结果/收口**

承接 O77/O83/O87 的「默认 fp8 main 在本卡已收口」。本轮按『fp8 专项冲刺』剩余候选，把
**非 main 固定开销的最后一条（quant 块调度）**、**O81 之后的编译宏**、**O85 遗留的
`D=256` K/V-TMA backlog** 逐一判决。**三件全部负/不可行，默认路径一行未改。**

- **quant+zero 栅格封顶（`--qcap`，纯 host，默认 0=历史）**：融合 kernel 以 `grid=任务数/4`
  （每 warp 一行、每 CTA 4 行）启动，S=4096 时 **grid=114,688**。怀疑块调度开销。封顶到
  4096/2048/1056/528/264/132 后 quant **0.110/0.110/0.119/0.156/0.250/0.448ms**（历史
  **0.106**）⇒ **单调更慢**。quant 流量 ~268MB/0.106ms ≈ **2.5TB/s ≈ 峰值 75%**，已近带宽墙；
  128-thread CTA 的 dispatch 被硬件充分流水，**「每 warp 一行」的最大栅格已最优**。`--qcap`
  仅留作诊断。
- **O81 之后的宏复扫**：baseline main 1.480ms；`ILV` 1.484、`ILV34` **1.567（−5%）**、`WS1`
  1.491、`R4` **1.514（−2.5%）**、`ILV+ILV34` 1.559、`WS1+ILV34` 1.557 ⇒ **全部中性/有损**，
  复证「默认是 L2 `red` bound、只改指令/发射顺序不改贡献 CTA 数即不可能转正」。
- **`D=256` K/V 4D-TMA 资源收口（解析）**：`D=256` 的 `smem_bytes_wgmma=115,712B`（2 CTA/SM，
  上限 116,224B）；`smem_bytes_wgmma_kvtma = 115,712 + ks_sw_bytes(8,192) + 64 = 123,968B
  > 116,224` ⇒ **K 双缓冲把 2 CTA/SM 挤成 1**，退回 O85 的中性档（K 单缓冲则无预取重叠）
  ⇒ **结构性不可行**，与本卡 F6/F7/p160/O83/F3b/F4b 的 **smem 墙同源**。

**默认路径回归**：S=4096 两文件 total 1.7162ms/80.08TF、main 1.4829ms，vs fp32 ref max_abs
2.635/2.644/3.216e-1，单/两文件 device 逐字同源（只动 host 栅格）。见 `docs/03` §111；
原始输出 `src/fp8/fa_bwd_fp8_o88_qcap_ab_s4096.out.txt`、`..._o88_macro_sweep_s4096.out.txt`。

### 5.98 第 184 轮（O89）：fp8 主 kernel 的 **LPT m 块调度序（causal 贵块先跑）** —— **正结果，默认**

承接 O77/O83/O87 的「默认 fp8 main 在本卡是 L2 `red` bound、张量核空转、无软件杠杆」。本轮换一个
此前**从未试过**的角度——**不改数据通路、不改 L2 搬运量，只改「哪个 CTA 算哪个 m 块」的调度序**。

- **动机**：默认稠密网格是 `grid=(nblk*ksplit, H, B)`，`mt = blockIdx.x/ksplit` 直接就是 m 块号，
  而 GigaThread 按线性 blockIdx **升序**派发。causal 下第 `m` 个 m 块的 K 循环长度 ∝ `(m+1)` ⇒
  **便宜的 m 块先跑、最贵的 m 块排在每个 head 段的末尾** ⇒ 尾波全是重块（LPT 的反面）。
  ksplit=8 之所以比 ksplit=1 快 ~20%，正是用**更细的块粒度**掩盖这个偏斜——代价是 Q/dO 被重读
  8 次、dQ 多一轮跨 part 原子。若能把平衡拿回来，理论上可再降 ksplit 消 Q/dO 重读。
- **改动（纯 host + 1 个透传参数，device 数学一行未改）**：主 kernel body 早已支持 `mt_m` 查询表
  （varlen 紧凑网格用）。给默认 `fa_bwd_fp8_mma_kvtma_kernel` 壳加一个 `const int* mt_m` 尾参
  并透传给 body；host 在 `--mrev=1` 时建一个**反转表** `mt_m[i] = nblk-1-i`（`nblk=ceil(S/64)`）。
  于是执行序变为「每个 head 从最贵的 m 块开始、最便宜的收尾」。dK/dV 是跨 CTA `atomicAdd`
  （可交换）⇒ **数值语义不变**，仅加法次序略变。门控 `causal && D==128 && nblk>=16`（S>=1024；
  小 S 网格太浅、反转只剩噪声），单/两文件同步。`--mrev=0` 供 A/B。
- **同 binary A/B（S=4096 H16 causal，event iters=30）**：main **1.4796→1.4475ms（1.022×）**、
  total **1.7159→1.6747ms（1.025×，80.10→82.07 TF）**；多次重复 main 1.473–1.488 → 1.429–1.440
  （**~3.2%**，稳定）。其它 causal D=128：S1024H32 main 0.2481→0.2442（1.6%）、GQA kv4
  0.2410→0.2362（2.0%）、GQA kv8 0.2982→0.2944（1.3%）、MQA kv1 0.4442→0.4420（0.5%）；
  S512（nblk=8）在门控外 ⇒ 逐位不变。
- **ncu（S=4096，`regex:kvtma_kernel --launch-count 1`）**：Duration **1.48→1.46ms**；
  **`lts op_red` 114,524,160 一字不变**、`op_read` 28.04M→27.92M、`op_write` 0.384M、
  DRAM 4.40→4.50%、`short_scoreboard 1.82`/`wait 1.56` 均不变 ⇒ **收益纯粹来自尾波/负载均衡
  （LPT），与 L2 搬运量无关**——再次印证默认 main 是 `red` bound + 尾波偏斜。
- **ksplit 复扫（mrev=1，main ms）**：k=1 **1.792** / k=2 1.547 / k=4 1.445 / **k=8 1.448**；
  反转把 ksplit=1 从 1.836 拉到 1.792（仅 1.024×）——**即「削尾波」并不能替代 ksplit 的
  细粒度并行**（1024 个 CTA 只有 2.6 波，静态排序再好也补不上粒度），故 **ksplit auto=8 仍最优**、
  Q/dO 重读消不掉。结论：调度序是**免费的小幅正收益**，不是「降 L2」的钥匙。
- **数值（护栏全过）**：`--ci --dtype fp8 --hopper` 单/两文件一致性 gate worst **7.629e-6 OK**、
  `--check docs/04` OK（198 行）；vs fp32 ref max_abs 与 mrev=0 **打印相同**
  （S4096 2.635/2.644/3.216e-1，S512 2.426/2.972/3.733e-1），单/两文件逐字同源。

**判决**：LPT m 块调度为**正结果、默认开启**（无需改 device、可随时 `--mrev=0` 回退）。它是
「不改 L2 搬运量」类微优化里目前唯一转正的一条；也再次证明**减 `red` 仍需改工作划分（换卡/
多 warpgroup）**；`red` 墙不受影响。见 `docs/03` §112；原始输出
`src/fp8/fa_bwd_fp8_o89_ab_mrev_s4096.out.txt`、`..._o89_ab_mrev_shapes.out.txt`、
`..._o89_ncu_mrev_{0,1}_s4096.out.txt`、`..._o89_ci_fp8.out.txt`。

### 5.99 第 185 轮（O90 / F6-step3）：wgmma2 的 **K/V 4D-TMA** 化 —— **负结果（opt-in）**

落实 F6 的「下一步」：给 F6 第二步的 `wgmma2`（BM=128、2 warpgroup、GEMM1/2 wgmma + SW128）
补上 **K/V 4D-TMA（K 双缓冲、V 单缓冲）**，看「TMA 化 + `red` 砍半」能否把 BM=128 档拉回竞争区。

- **实现**：新增 `fa_bwd_fp8_wgmma2_tma_kernel<128,128,32>`（`kernels.cuh` 3d 节，单/两文件
  device 逐字一致、392 行 `identical=True`）+ `launch_bwd_wgmma2tma` + `--wg2tma`（opt-in，
  默认一行未改）。Q/dO 仍手工载入（每 CTA 一次）；K/V 走 `tma_load_4d` + mbarrier（prologue
  发 K[0]/K[1]/V[0]；循环尾发 K[nt+2]/V[nt+1]、等 K[nt+1] 重建 Kp、等 V[nt+1]；相位标量）。
  **208 regs / 0 spill / 1 barrier**。
- **性能（同 binary A/B，S4096 H16 causal）**：默认 `kvtma` main **1.4518ms/81.98TF**；
  `wg2wgmma` **2.8062ms** → `+KVTMA` **2.7113ms（1.035×）**。即 K/V TMA 相对 wgmma2 只省
  ~3.5%，`wgmma2tma` 仍只有默认档的 **0.54×（S4096）/ 0.66×（S512）**。
- **ncu（S4096，默认 vs O90）**：`lts op_red` **114.52M → 58.20M（精确砍半）**、`read`
  27.92M → **14.47M（砍半）**——**BM=128 减 L2 搬运量的机制完全成立**；但 **L2 利用率
  81.06% → 21.99%**（省下的带宽用不上）、**warps 18.62% → 12.50%（3→1 CTA/SM）**、
  `smsp inst` 642M → 838M、Duration 1.45 → 2.72ms。K/V TMA 只把 `long_scoreboard` 0.37→0.18。
- **判决**：**负结果、opt-in 默认关**。BM=128 双 warpgroup 在本卡 1 CTA/SM 下是 **延迟/occupancy
  bound**，`red` 砍半被「8 warp 藏不住延迟」吃掉——与 F6/O83/O86/O87 同一堵墙；转正需
  **256/384 线程多 warpgroup 摊累加器**或**换卡**。`wgmma2_tma` + K/V-TMA helper 留作后续
  `D>128`/WS 版基建。数值：`ours_o90 vs fp32 ref` relL2 S4096 8.15/8.39/6.52%、S512
  8.18/8.41/6.36%（护栏内）；单/两文件默认路径逐值不变。见 `docs/03` §113；原始输出
  `src/fp8/fa_bwd_fp8_o90_{ab_s4096,ab_s512,default_s4096,ncu_wg2tma_s4096,ncu_default_s4096}.out.txt`。

### 5.100 第 186 轮（O91 / F6-step4）：**多 warpgroup（BM=192、3 WG、384 线程）** —— **负结果（opt-in）**

落实 O90 的「下一步候选 ①」——本卡 fp8 主 kernel **最后一条未试的结构性杠杆**：多 warpgroup。
O90 证 `wgmma2`（BM=128、2 WG）`red` 砍半却只有 8 warp/SM（1 CTA×2 WG）⇒ 延迟 bound。本轮把
`wgmma2` 泛化成 **`NWG` 个 warpgroup（`BM=NWG*64`）**，`NWG=3`（BM=192、384 线程、1 CTA/SM）：
既把 `red` 再压（BM 64→192），又把 warp/SM 拉回默认档（12 warp/SM）。

- **实现**：`fa_bwd_fp8_wgmma2_kernel<HD,BM,BN,NWG=2>` 泛化（相 A 每 WG 算 64 行、GEMM3/4 仍
  8 warp `if(wid<8)`、GEMM5 铺 6×2、`Ap/dS3` fold 沿 m 泛化 `NPC/REM`）；`NWG=2` 逐位不变。
  CLI `--wg3`（opt-in）+ `--ksplit3=`；host `launch_bwd_wgmma_nw`。BM=192 smem=197,120B → 1 CTA/SM。
  **踩坑**：SW128 是 generic 写、wgmma 走 async proxy 读，NWG=3 时序下**首次 dk/dv 偶发 `inf`**；
  每迭代补 `bulk_reduce_fence()`（`fence.proxy.async.shared::cta`）后 3/3 稳定（p155/O81 同坑）。
- **性能（同 binary A/B + 默认同 session）**：S4096 默认 main **1.4436ms/82.27TF**；`wgmma2`
  2.8330 → `wg3` **2.7718ms（1.022×）** ⇒ **wg3 仍只有默认档的 0.52×**。S512 `wgmma2 0.1076 →
  wg3 0.1179ms（0.913×）`（grid-bound）。
- **ncu（S4096，默认 vs O91）**：`lts op_red` **114.52M → 49.64M（0.43×）**、`read` 27.90M →
  16.99M、L2 总扇区 142.9M → **80.6M（0.56×）**、**L2 利用率 81.4% → 21.2%**；**warps 18.60% →
  18.72%（12 warp/SM，已追平默认）**；但 **Duration 1.44 → 2.78ms、CTA/SM 3→1**。⇒ 卡点**不是
  warp 数**，是 **1 CTA/SM 的单一 barrier 域**：默认 12 warp 分属 3 个独立 CTA，wg3 的 12 warp
  挤在一个 CTA、被每 tile 5 个 `__syncthreads` 串成依赖链。**「多 warp 摊延迟」被证伪：要多 CTA，
  不是多 warp。**
- **数值（护栏内）**：`ours_wg3 vs fp32 ref` relL2 S4096 8.149/8.449/6.532%、S512 8.179/8.455/
  6.371%（护栏 dq≤8.2/dk≤8.3/dv≤6.5±0.3）；单/两文件 device 逐字一致（`identical=True`）、
  `ours_wg3` vs `ours_wg3_sf` ≤1.19e-7；`--ci --dtype fp8 --hopper` 全绿。
- **判决**：**负结果、opt-in 默认关**。多 warpgroup / 放大 BM 这条结构性杠杆在本卡（1 CTA/SM，
  ≤116KB smem / ≤128 regs 才能 2 CTA/SM）**判死**；正结果只剩 **换卡**。见 `docs/03` §114；
  原始输出 `src/fp8/fa_bwd_fp8_o91_*.out.txt`。

### 5.101 第 187 轮（O92）：fp8 main L2 墙的「TE 侧对侧」闭环复核 —— **无新正结果（默认一行未改）**

`fp8 专项冲刺`（F1→F6）与 `下一批`（F6/F3b/F4b）在 O77 后反复收敛到「默认 fp8 `kvtma` main 是
L2 `red` bound、软件杠杆已尽」。本轮不做新算法，而是**闭环复核 + TE 逐指标对侧**，并用同 session
原始数据把「差距在哪一级、剩余杠杆是什么」钉死；顺带在 O81（GEMM5 wgmma）/O89（LPT 调度）之后
**复测三条尚未在最新默认上复跑的候选**。

- **基线（S4096 causal，event iters=30）**：`total 1.6711ms / 82.24TF`、`main 1.4437ms`
  （70656B smem）——与 O89/O91 同档，复现稳定。
- **复测（默认一行未改）**：① **ksplit 复扫** 1/2/4/**8**/16 = 2.032/1.804/1.694/**1.667**/
  1.821ms ⇒ **auto=8 仍最优**（`red` 随 ksplit 降但尾波补偿更贵）；② **`FA_BULKRED` 复测**
  1.6454ms（**0.877×**）⇒ O42 的负结果在 O81/O89 后**不翻转**（staging smem 流量 + TMA 归约延迟）；
  ③ **编译宏复扫** ILV34 0.948× 等仍全负 ⇒ 复证 O88「不改贡献数即不可能转正」。
- **TE vs ours SASS（S4096，`ncu --page source --print-source sass`）**：TE = **16 QGMMA + 0 HMMA
  + 4×`UTMAREDG.4D.ADD` + 24 STSM**；ours = **12 QGMMA**（GEMM1/2+O81 的 GEMM5）+ **64 HMMA**
  （GEMM3/4，O82/O87 已判负）+ **64×`REDG.E.ADD.F32`** + 39 LDSM。⇒ ours 的 wgmma 覆盖已到顶
  （剩余 HMMA 与 `red` bond 无关）；TE 的低 `red` 来自**工作划分**，不是 `UTMAREDG`（O67/p159 已证
  `red` 扇区与归约机制无关）。
- **TE vs ours L2/occupancy 六指标（同 session，S4096）**：Duration **1.45ms vs 258.34µs（5.62×）**、
  L2 总扇区 **142.97M vs 36.84M（3.88×）**、`read` 27.91M vs 10.00M、**`red` 114.52M vs 25.96M
  （4.41×）**、L2 利用率 **81.30% vs 70.70%**、DRAM 4.52% vs 17.60%、**tensor pipe 11.25% vs
  36.86%（3.28×）**、warps 18.60% vs 15.61%、regs 168/168、CTA/SM **3 vs 1**。**TE tile 同为
  BM=64（64x64x128）** ⇒ 差距**不是放大 BM**，而是「每个 dK/dV 元素的贡献 CTA 数」：ours 的
  114.52M `red`=830MB 原子流量 / dK/dV 真实 67MB ⇒ **~54 次/元素**；TE 25.96M ⇒ **~12 次**。
- **精度护栏（S4096）**：`ours vs fp32 ref` relL2 dq/dk/dv **8.149/8.263/6.489%**（护栏内，与 O91
  逐位同档）；`max_abs` 2.635/2.644/3.216e-1；`ours vs TE` 13.42/13.48/28.15%（TE-vs-ref 的 dv
  自身 27.45%，非回退）。
- **判决**：**无新正结果，默认一行未改**。默认 fp8 main 的 L2 `red` 墙在本卡**无软件解**
  （F3b/F4b/F6/O90/O91/F7 全收口，本轮复测亦无翻转）；真差距 = 工作划分（同 BM 下 TE `red` 4.4× 低），
  解锁需 ≥2 独立 CTA/SM 的放大 tile（本卡达不到）或**换卡**。**O91 的新认知**：1 CTA/SM 的瓶颈是
  **单 barrier 域 `__syncthreads` 串行**，复活 BM≥128 档唯一路径是 **warp specialization**
  （多轮工程，且仍受同一 L2 墙）。见 `docs/03` §115；原始输出 `src/fp8/fa_bwd_fp8_o92_*.out.txt`。

### 5.102 第 188 轮（O93）：跨 head 全局 LPT（grid 轴对调）+ 低 ksplit —— **正结果（默认）**

O92 把 `fp8 专项冲刺 F6-①`（降 L2 搬运）的 ksplit 路径判为关闭（「auto=8 仍最优」），并复证
「不改工作划分/指令即不可能转正」。本轮回到 **O89 的下一步候选 ③**：O89 的 `--mrev` 只是
**per-head** 的 m 块降序，全局仍是锯齿。**O93 = 把 LPT 升级为跨 head 全局，并利用它解锁低 ksplit。**

- **实现（device 数学一行未改）**：`fp8_mma_body` 加末位模板参 `HSWAP`，只改解码——`h=blockIdx.x`、
  `mt=blockIdx.y/ksplit`、`part=blockIdx.y%ksplit`；host 用 `grid=(H, nblk*ksplit, B)` 启动（head
  走**快轴**）⇒ 硬件按 `blockIdx.x` 最快的顺序派发 = 所有 head 的最贵 m 块一起先跑 = 全局 LPT。
  `hswap_elig`（定长 causal D=128 nblk≥16）时自动把 ksplit 收到 **2**。**默认开**（`--hswap=0` 回退）；
  单/两文件 device 逐字同源。
- **性能（S4096 causal，同 binary A/B，iters=40）**：`mrev=1 hswap=0`（O89 默认）main 1.4304 /
  total 1.6686ms（82.37TF）→ **hswap k=2：main 1.3742 / total 1.6109ms（85.32TF）= main 1.041× /
  total 1.036×**。**关键**：hswap 在 k=8 时**反而慢**（1.87ms，跨 head 交错损 L2 局部性），只有
  **hswap + k=2** 才最优 ⇒ 收益来自「全局 LPT 解锁的低 ksplit」，不是 hswap 本身。
  S1024H32 main 1.10×、GQA kv4 1.13×；S512（门控外）不变。
- **ncu（S4096，同 session）**：Duration 1.45→**1.39ms**、L2 总扇区 142.97M→**129.87M（−9.1%）**、
  **`read` 27.88M→24.26M（−13.0%，Q/dO 重读 8×→2×）**、**`red` 114.52M→105.38M（−8.0%，dQ 跨 part
  原子减少；dK/dV 主体不变）**、L2 利用率 81.0%→76.9%；**代价**：DRAM 219→557MB（2.5×，跨 head
  交错损 L2 局部性），但主 kernel 墙是 L2 吞吐、DRAM 绝对量仍低 ⇒ 净快。
- **精度护栏**：`ours vs ref` relL2 dq/dk/dv **8.148/8.263/6.489%**（护栏内，与 O91/O92 同档）；
  `max_abs` 2.635/2.644/3.216e-1；`--ci --dtype fp8 --hopper` gate worst **7.629e-6 OK**、
  `--check docs/04` OK。
- **对标**：同 session TE FP8 纯反向 S4096 = 0.3049ms ⇒ ours total **5.30×**（O92 5.50×）、
  main **4.52×**（O92 4.75×）。**判决：正结果、默认**；这是「不改工作划分/指令」类里第一条
  `read`/`red` 双向下降的调度杠杆（对照 O89 只削尾波、`red` 一字不变）。见 `docs/03` §116；
  原始输出 `src/fp8/fa_bwd_fp8_o93_*.out.txt`。

### 5.103 第 189 轮（O95）：变 ks 调度（`--ksm`，表驱动 per-m ksplit）—— **中性/负结果（opt-in）**

落实 O93 的下一步候选 ③「ksplit 随 m 变化」。给 `fp8_mma_body` 加运行期 `slot_tab`（表驱动
`(ks,part,mt)` 解码，`--ksm=N --ksmhi=K` = 最贵 N 个 m 块 `ks=K`、其余 `ks=1`，host 按全局 LPT
序建表）——**device 一行数学未改**，默认 `slot_tab=nullptr` 逐位退化。

- **性能（S4096 causal，同 binary A/B，iters=40）**：默认（均匀 k=2）main **1.3670ms**；
  `ksm=64`（表驱动均匀 k=2）1.3652（**验证表解码等价**）；**所有非均匀档都更慢**——
  `ksm=8`（72 槽）1.3874、`ksm=16` 1.3807、`ksm=32` 1.4120、`ksm=0`（全 k=1）1.4488、
  同槽数重分配 `ksm=32 k=3`（128 槽）1.4506。
- **ncu（S4096）**：槽位 128→72（−44%）但 L2 总扇区只 **129.81M→127.86M（−1.5%）**、
  `read` −2.1%、`red` −1.3%，Duration 不降反略升。
- **根因（账）**：主 kernel 的 L2 `read` 主要来自 **K/V 读 = 总 (m,kv) tile 数（与 ksplit 无关）**，
  ksplit 只改 **Q/dO 重读**（占 read 小头）；`red` 的 dK/dV 主体也不随 ksplit 变。⇒ 物理上无空间，
  且槽位下降损并行度。**唯一小正结果**：MQA q64/kv1 `ksm=8` main 0.4308→**0.4198（1.026×，稳定）**，
  但只此一 shape、不值得做默认（会回退 MHA/GQA 1–7%）。
- **精度/回归**：默认 relL2 **8.1485/8.2633/6.4894%**（与 O93 逐位相同）、`max_abs` 2.635/2.644/3.216e-1；
  `--ci --dtype fp8 --hopper` gate worst **7.629e-6 OK**、`--check docs/04` OK。**默认一行未改**，
  `--ksm` opt-in。⇒ O93 的均匀 `ksplit=2` 是本卡默认最优点。见 `docs/03` §117；
  原始输出 `src/fp8/fa_bwd_fp8_o95_*.out.txt`。

### 5.104 第 190 轮（O96）：full（非 causal）D=128 的 ksplit/regdq 重标定 —— **正结果（默认）**

**思路转向**：causal 旗舰调到 L2 `red` 平台期（O83/O91/O92）后，本轮不再攻 causal，而是审计
**两条按 causal 标定、却被无条件套用到 full 的启发式**——O29 的 ksplit target 与 O7 的
`use_regdq` 阈值。**full 每块工作量相同**，O29 的「细切分摊平三角尾波」在 full 下不成立，
只剩 Q/dO 重读 + dQ 跨 part 原子；O7 阈值里的 `/2`（causal 平均只扫半个三角）在 full 下会让
`ksplit` 稍大时**误关寄存器 dQ 累加**，使 dQ 退化成逐 tile 跨 CTA `atomicAdd`。

- **改动（纯 host，device 一行未改，单/两文件同源）**：① full D=128 的 ksplit 改为在 k∈[1,8] 里
  取「尾波空泡 `ceil(base*k/396)*396 − base*k`」最小者（`396` = 3 CTA/SM × 132 SM），8 个 full
  shape 一致选到 **k=3**（S4096 选 k=5）；② `use_regdq` 的 `/2` 只对 causal 生效。
- **性能（8 个 full D=128 shape，同 binary A/B，iters=60）**：main 最高 **1.59×**
  （S2048H16 1.136→0.716ms；S1024H16 0.301→0.206ms；S1024H8 0.167→0.109ms；S512H16
  0.093→0.067ms），大 S S4096 1.004×（仍受 L2 `red` 墙）。
- **ncu（S1024H16 full）**：Duration **299→205µs**、L2 总扇区 **29.24M→16.85M（−42%）**、
  `op_red` **25.17M→13.76M（−45%）**、`op_read` **−26%**、L2 利用率 **84.6%→68.3%** ⇒
  **L2 `red` 从饱和（84.6%）降到有余**，是教科书式的「降 L2 搬运量」正结果。
- **数值/回归**：只改 atomic 加法次序 ⇒ S1024H16 full `ours vs fp32 ref` relL2 新旧**完全相同**
  （8.111/8.235/6.709%）；**causal 路径逐档不变**（S1024H32/S4096H16 仍 k=2、main 0.221/1.360ms、
  max_abs 2.635/2.644/3.216e-1）。`--check docs/04` OK。
- **教训**：凡「按 causal 三角标定」的启发式（ksplit、regdq、未来的 partial/split）都要逐条
  复核 full/变长。见 `docs/03` §118；原始输出 `src/fp8/fa_bwd_fp8_o96_*.out.txt`。

### 5.105 第 191 轮（O97）：full（非 causal）D=256 / D=512 的 ksplit 重标定 —— **正结果（默认）**

**思路**：O96 的「凡按 causal 三角标定的启发式都要逐条复核 full」的同类审计，本轮落到
**D=256 与 D=512（MLA）**——两者共用 O29 的 `target_ctas = S/2`（按 causal MLA 标定），
而 D=128 早已单独重标定。**纯 host、device 一行未改、单/两文件同源**。

- **两条路径错法相反**：① **D=512**（1 CTA/SM→132 槽）：`S/2/base = 32/H` 恒把 k 顶到 16，
  base 小时**过切**——5 个 full shape 最优 k≈`128/base`（即「对齐一个波」），main 1.15×；
  ② **D=256**（2 CTA/SM→264 槽）：`32/H` 与 S 无关 ⇒ S≥2048 时**欠切**——7 个 shape 最优
  k=8–12（auto 只 1–4），main 5.7–7.9%。
- **新规则**：D=512 / D=256(S<2048) 按并发槽做**波对齐**（SLOTS=132/264；D=512 设 k≥2 下限
  消大 base 回退）；D=256(S≥2048) 取 `k=clamp(8192/base,1,12)`（grid≈8192）。`--ksplit` 仍可覆盖。
- **性能（14 个 full shape，同 binary A/B，iters=60）**：**全部 ≥1.00×、无回退**；D=512 小 shape
  **1.15×**（S512H2 0.0621→0.0541ms、S1024H2 0.2034→0.1771ms）、D=256 大 S 1.06–1.08×
  （S2048H8 1.3347→1.2457ms）。D=512 端到端被 full MLA 的 LSE preprocess 盖住，只报 main。
- **ncu（两条方向）**：D=512 S1024H2 k16→k4：`op_read` **−24%**、Duration **211→191µs**；
  D=256 S2048H8 k4→k12：`op_read` **+44%** 但 L2 利用率 **72→80%**、Duration **1.38→1.26ms**；
  两组 `op_red` 均**一字不变**（dK/dV 主体与 ksplit 无关，再次印证 O83/O86）。
- **数值/回归**：只改 atomic 加法次序 ⇒ 5 个代表 shape 的 relL2 vs fp32 ref **逐位相同**，
  全量 `--ci` 三 dtype gate **OK**（fp8 6.676e-6）、`--check docs/04` **OK**；causal 逐档不变。
- **教训延续**：D=256 的「同 base 但不同 S」表现不同（S1024H16 要 k=1、S2048H8 要 k=12）——
  **full 的 ksplit 不是纯波对齐能全包的**，大 S 还需要「足够的并行度/working-set 切分」。

见 `docs/03` §119；原始输出 `src/fp8/fa_bwd_fp8_o97_{ab,ncu}.out.txt`。

### 5.106 第 192 轮（O98）：full（非 causal）**变长**的 ksplit 重标定 —— **正结果（默认）**

**思路**：O96/O97 已复核定长 full 的 ksplit，但 **`run_varlen` 仍是 O29 的 causal 标定**
（`target_ctas` 按 maxlen 计），且变长多一层错——`base_grid=ceil(maxlen/BM)*H*B` 含短序列的
**早退死 CTA**，名义网格被高估。O97 只把 `--ksplit` 接到 varlen 做 A/B、未改自动档。本轮补上。

- **全扫（6 个 dumped varlen full shape，k∈[1,16]，iters=80）**：D=128 的最优稳定在 **k=3~4**
  （`b4_t3840`→4、`b4_t4096`/`b5_t3968`/`b8_t2904`→3），与 O96 定长 D=128 的 k=3 同源；
  D=512 的最优是 **k≈128/base**（`b1_t512`→8、`b3_t1792`→11），即「对齐 132 槽一个波」。
  causal 自动档在 base 偏大时给 **k=1/2（欠切）**：b5 k1 慢 5.6%、b8 k2 慢 4.3%；D=512 则
  k=16 **过切**（b1 慢 8.2%）。
- **规则（纯 host、device 一行未改、单/两文件同源）**：`!causal` 时——**D=128** `k=max(kp,3)`
  （只抬下限，不碰已最优的 k=4 档）；**D=512** 按 132 槽波对齐（下限 2，同 O97）；**D=256**
  沿用 O97（`maxlen≥2048` 取 `clamp(8192/base,1,12)`，否则 264 槽波对齐）。`--ksplit` 仍可覆盖。
- **性能（同 binary A/B，iters=100）**：`b4_t3840`/`b4_t4096` 配置不变（0.998×/0.997× = 噪声），
  `b5_t3968` **1.059×**、`b8_t2904` **1.041×**、`b1_t512_d512` **1.097×**、
  `b3_t1792_d512` **1.018×** ⇒ **6 shape 无回退**。
- **ncu（两条机制）**：**D=128** k 增大 ⇒ Q/dO 重读 `op_read` +19~37%，但墙是延迟/并行度，
  L2 利用率 54.6→60.4%（b5）、58.4→61.2%（b8），Duration −3.9%/−2.0%；
  **D=512** 最优 k 反而**减少**重读（k16→8 `op_read` −19%、L2 44.7→50.8%），Duration −12%。
  两组 `op_red` 仅随 dQ 跨 part 原子微动（dK/dV 主体与 ksplit 无关，续证 O83/O86/O95）。
- **数值/回归**：只改 atomic 次序 ⇒ relL2 vs fp32 ref dq 8.06–8.22% / dk 8.21–8.36% /
  dv 6.48–6.76%（护栏内）、`max_abs` O(0.05–0.25)；单/两文件 `max_abs ≤3.58e-7`；
  全量 `--ci` 三 dtype gate **OK**（fp8 6.676e-6）、`--check docs/04` **OK**；causal/定长 full 逐字不变。
- **教训**：**变长的 ksplit 不能用名义网格做波对齐**——早退死 CTA 使 `base_grid` 高估，
  O96 的 396 槽规则套到变长会给 1/2/5/8（偏差最大 4.3%）。改用「按 D 的最优 k 下限 + 按槽波对齐」
  才稳。残余：D=128 `b4_t4096` 的 k=4 比 k=3 差 ~3.7%，但全局 k=3 会反伤 `b4_t3840`，
  需「均匀 vs 混合长度」判据才能吃下（backlog）。

见 `docs/03` §120；原始输出 `src/fp8/fa_bwd_fp8_o98_{varlen_full_ab,ksweep,ncu,fa3_baseline}.out.txt`。

### 5.107 第 193 轮（O99）：**causal D=256** 的 ksplit 重标定 —— **正结果（默认）**

**思路**：O96/O97/O98 只复核了 **full**，而 **causal D=256** 一直沿用 O29 的 `target_ctas=S/2`
——那是按 **causal MLA（D=512，1 CTA/SM）** 标的。套到 D=256（2 CTA/SM⇒264 槽）上
`k=S/2/base=32/(H*B)` 与 S 无关，小/中 S 严重欠切。

- **全扫（6 个 causal D=256 shape，k∈[1,32]，iters=100）**：最优 k 一致落在 **`grid=base*k≈2*S`**
  （`k≈128/(H*B)`）附近——s512H8→8、s1024H8→8/16（差 1.7%）、s2048H8→16、s4096H8→16、
  s1024H16kv4→6（k=8 差 0.5%）、s2048H16→12（k=8 差 0.6%）。旧 auto（k=4/4/4/4/2/2）慢 5–14%。
- **规则（纯 host、device 一行未改、单/两文件同源）**：`k = clamp(2*S/base,1,16)` 再按 `nblk` 封顶；
  变长 `run_varlen` 同名分支用 `maxlen`（**无 D=256 变长 dump，按定长外推、未单独测量**）。
- **性能（同 binary A/B，iters=100）**：s512H8 **1.068×**、s1024H8 **1.112×**、s2048H8 **1.104×**、
  s4096H8 **1.055×**、s1024H16kv4 **1.143×**、s2048H16 **1.122×** ⇒ **6 shape 全 ≥1.05×、无回退**。
- **ncu**：s2048H8 旧 k=4 769.6µs → 新 k=16 **681.5µs**（`op_read` 6.05M→8.41M、L2 67.1→76.4%）；
  s1024H16kv4 旧 k=2 464.1µs → 新 k=8 **397.1µs**（L2 57.4→67.7%）。**两组 `op_red` 一字不变**
  （dK/dV 主体与 ksplit 无关，续证 O83/O86/O95）⇒ 纯「加 k 买并发/藏延迟」。
- **数值/回归**：只改 atomic 次序 ⇒ relL2 vs fp32 ref dq 8.15–8.33% / dk 8.33–8.48% /
  dv 6.39–6.50%（护栏内）、`max_abs` O(0.21–0.62)；单/两文件**逐位相同**；**全量 `--ci`（93 case）**
  三 dtype gate **OK**（fp8 6.676e-6），并顺带把 O96/O97/O98 遗留的 `docs/04` 表同步（198→214 行）；
  D=128 causal（hswap k=2）/ D=512 causal / 定长 full 逐档不变。
- **教训**：**causal 的 ksplit 也不能拿「别的 dtype（D=512 MLA）的 target」通用**。至此 fp8 的
  ksplit 自动档在 causal/full × 定长/变长 × D=128/256/512 全部经实测复核。

见 `docs/03` §121；原始输出 `src/fp8/fa_bwd_fp8_o99_{ab,ksweep,ncu,accuracy,fa3_baseline,ci}.out.txt`。

### 5.108 第 194 轮（O100）：fp8 full 变长 D=128 的 ksplit 收口（k=3）—— **正结果（默认）**

**背景**：O98（§5.106）给 D=128 full 变长的自动 ksplit 设成 `max(kp,3)`，并留下一个未决：
*「`b4_t4096` 最优 k=3、但 `b4_t3840` 最优 k=4，需『均匀 vs 混合长度』判据」*。本轮复核该前提。

- **干净机器重测（k∈[1,5]，iters=150，3 次，GPU 无残留）**：四个 D=128 full 变长 shape 的最优
  **一致为 k=3**（b4_t3840 1.386 / b4_t4096 1.114 / b5_t3968 2.684 / b8_t2904 1.126 ms），
  O98 记录的「b4_t3840 k3 慢 3.9%」在本轮消失（k3 1.386 vs k4 1.395）⇒ 判为**当轮噪声/残留**。
  「均匀 vs 混合长度」不是真判别维度。
- **规则（纯 host、device 一行未改、单/两文件同源）**：`!causal && D==128` 时 `ksplit=min(3, nblk32)`
  （把 O98 的 `max(kp,3)` 改为精确 3）。D=256/D=512 full 与 causal 路径逐字不变。
- **性能（同 binary A/B，iters=150，3 次）**：**`b4_t4096` 1.009–1.011×**（1.115 vs 1.127ms）；
  `b4_t3840` 1.002–1.005×；`b5`/`b8`/D=512 均 ≈1.00×（同配置，噪声内）。只有等长 shape 有
  稳定正收益，量级小但**非回退**。
- **数值/回归**：只改 atomic 次序 ⇒ 4 个 shape 的 relL2 vs fp32 ref **新旧逐位相同**
  （dq 8.06–8.12% / dk 8.21–8.26% / dv 6.48–6.68%，护栏内）、`max_abs` O(0.07–0.25)；
  单/两文件一致性 gate **OK**（worst 5.722e-6）、`--check docs/04` **OK（214 行）**；
  causal 与定长 full 逐字不变。
- **教训**：**别让「单轮 sweep 里某个 shape 的反向结论」直接进入启发式**——先查 GPU 残留、
  多次重复再下判断（这是本项目第 54 篇踩过的同类坑在启发式标定上的再现）。
- **残余**：fp8 唯一未复核的同类启发式 = `partial/split` 的 full 标定（backlog）。causal 旗舰
  （D=128 S4096）的 L2 `red` 主体墙仍是唯一真杠杆，本卡无软件解（见「阻塞」）。

见 `docs/03` §122；原始输出 `src/fp8/fa_bwd_fp8_o100_{varlen_full_k_sweep,ab,accuracy,fa3_baseline}.out.txt`。

### 5.109 第 195 轮（O101）：定长 full 的 LSE 补齐均衡/流水/split —— **正结果（默认）**

**背景**：O54（第 101 轮，本文第 19 条 / `docs/03` §53）只把 **varlen** full MLA 的 LSE 从 O1 切到均衡 FULL+K split；
O68/O70 只补了定长 full 的 **D=128**（TMA）与 **D=256**（均衡 FULL 但无 split）。O100 结尾留的
`partial/split 的 full 标定` 中「split」这一半，**定长 full D=512（MLA）**从未做——实测其 LSE
（O1 `lse_mma_kernel`）占 full MLA 端到端 **~65%**（S1024H2：preprocess 0.376ms > main 0.183ms）。

- **改动（纯 host，device 一行未改、单/两文件同源）**：① D=512 full 的 LSE 从 O1 切到
  `lse_mma_kernel_bal<512,1,true>` + K 维 split（`--lse512old=1` 退回 O1 做同 binary A/B）；
  ② D=256 full 的均衡 LSE 接上 K 维 split（此前恒 1）；③ split auto 的 base 从「恒 `lg_bal.x`」
  改为 `causal ? lg_bal.x : lg.x`（full 一个 CTA 一个 m 块 = `nblk`），target 取 full D=512 **256**、
  full D=256 **512**（5+8 个 full shape 全扫确认 `base*split≈256/512` 最优）。causal 路径逐字不变。
- **性能（同 binary A/B，iters=30，两文件）**：**D=512 full 端到端 S512H2 3.46× / S512H4 2.53× /
  S1024H2 2.64× / S2048H2 2.19× / S4096H2 1.68×**（LSE 单向 6.5–15.8×）；D=256 full 小/中 S
  1.02–1.14×、大 S 中性（LSE 只占 ~14%）。
- **ncu（D=512 S1024H2）**：old `lse_mma_kernel<512>` 544µs / `sm 3.38%` / warps 6.25%（1 CTA/SM）/
  DRAM 0.12%（纯「单线程顺序扫 K」的延迟 bound）→ new `lse_mma_kernel_bal<512,1,1,128,64>`
  grid(16,2,**8**) **21.1µs（25.8×）/ sm 35.4% / warps 12.1% / L1TEX 18.4%**（`cp.async` 藏载入 +
  split 铺满 2 CTA/SM 的并发槽）。新墙是 issue/延迟，不再主导端到端。
- **数值/回归**：D=512/D=256 full 的 relL2 vs fp32 ref 全在护栏内（dq 8.14–8.24 / dk 8.29–8.39 /
  dv 6.69–6.79 %），`max_abs` O(0.02–0.09)；单/两文件 worst 8.94e-8；全量 `--ci --dtype fp8`
  一致性 gate **worst 6.199e-6 OK**、`--check docs/04` **OK（214 行）**；causal 与 D=128 full 逐位不变。
- **外部基线**：MLA D=512 full 的 FA2.7.4/FA3/（fp16/ fp8）TE **均 NA**，仅 ours。
- **教训**：**「把某路径已有的优化补到另一条路径」时，必须核对所有分支**——O54 补 varlen、
  O68/O70 补 D=128/D=256，「定长 full MLA」被两次改造同时漏掉，成了 full MLA 端到端的头号成本。

见 `docs/03` §123；原始输出 `src/fp8/fa_bwd_fp8_o101_{ab,ncu}.out.txt`。

### 5.110 第 196 轮（O102）：`--det` 确定性路径的 Hopper split-K + full/causal 标定 —— **正结果（非默认路径）**

**背景**：O101（本文 §5.109 / `docs/03` §123）把「`partial/split` 的 full 标定」里的 **split** 收口，
点明**唯一未复核的同类启发式** = **确定性 `--det` 的 partial/`split` 标定**。核查发现：mma 路径的
`--det` 早在 P3-4f 支持 `--detk>1`（causal 下 k=4 触底），但 **Hopper `kvtma` 快路**的 `--det`
（P3-4g）**把 ksplit 写死为 1**——而 F1 之后 fp8 的生产默认构建就是 Hopper，于是默认构建上 `--det`
一直白扔 split-K 并行度。

- **改动（纯 host，device 一行未改、单/两文件同源）**：`launch_bwd_main_kvtma_det` 加
  `int ksplit = 1, float* dq_part = nullptr` 并透传（`fp8_mma_body` 本就支持 `DET && ksplit>1` 的
  per-part dQ partial）；P3-4g 的 A/B 段参数化到 `--detk`：grid `(nblk*ks,H,B)`、`ks>1` 时分配/清零
  `dq_part` + `dkv_reduce_kernel` + `dq_reduce_kernel<128>` 固定次序求和（与 P3-4f 同款）。`det_ksplit`
  默认由 1 改 **0=auto**：定长 D=128（mma 与 Hopper 两条）取 `causal ? min(4,nblk) : 1`；`--detk=1`
  可复现旧状；varlen/MLA 三条 DET 路径仍 `⇒1`（逐位不变）。
- **标定（causal S4096 H16 Hopper，iters=20）**：DET-fp32 k=1 **2.5344** → k=2 2.3472 → **k=4 2.2860
  (1.109×)** → k=8 2.3583；DET-fp16(扇区化) k=1 **2.1241** → **k=4 1.8743 (1.133×)**；auto 选中 k=4
  （2.2845 / 1.8729）。**full（S1024 H16）** k=1 0.3353 < k=4 0.3465 ⇒ auto 取 k=1（无三角偏斜）。
- **确定性/精度**：所有 ks、单/两文件、causal/full **`runs[1-2] bitwise dq/dk/dv = 0`**；
  `DET-vs-atomic` ~e-7–e-4；`fp16-vs-fp32 dk/dv` ~1e-3（partial 过 fp16）。**默认路径一行未改**：
  causal S4096 total 1.6079ms/85.48TF、max_abs 2.635/2.644/3.216e-1（与历史同档），
  `--doc-table-check` OK（214 行）。
- **ncu**：`dkv_reduce_kernel<128,64>`（k=4）Duration **732.7µs** / **DRAM 91.31%** / L2 88.76% /
  Compute 12.97% / 3.06 TB/s ⇒ 墙仍是**二次归约的纯 DRAM 带宽**（与 §56/§57/§68 一致），与 split-K 无关。
- **判决**：正结果（非默认路径，`--det`）。**`partial/split` 的 full 标定至此全部收口**（LSE split =
  O101；DET partial/split = 本轮）。剩余只有换卡 / 减 DET partial 字节（§68 候选 ②，非默认）/ 覆盖型 backlog。

见 `docs/03` §124；原始输出 `src/fp8/fa_bwd_fp8_p196_det_hopper_ksweep_s4096.out.txt`、
`..._p196_det_hopper_full_1file.out.txt`、`..._p196_default_regression.out.txt`、
`..._p196_ncu_reduce_s4096.out.txt`。

### 5.111 第 197 轮（O103）：head_dim=256 的 LSE 上 4D-TMA —— **正结果（默认）**

- **动机**：O96–O102 收口「错套启发式」后，剩下的是覆盖型 backlog。取其中不撞 smem 墙的一条：
  **D=256 的 LSE 仍走 mma + `cp.async`**（O76/O101 注释「不做 4D-TMA（只服务 128/512）」）。
  但 LSE 的 TMA kernel 早在 O74 就泛化为 `NCH=HD/128`（D=512 已用），`make_lse_map_fp8` 也早已
  服务 D=256 的主 kernel Q/dO；**缺口纯在 host 未接 D=256**。
- **改动（纯 host，device 一行未改，单/两文件同源）**：`launch_lse_bal_tma_split` 加 `bool FULL`
  模板参；D==256 建 LSE 描述符 + `cudaFuncSetAttribute`；`lse_tma` 自动档纳入 D=256；D==256
  的 `run_preprocess` 加 TMA 优先路径（causal 镜像配对 + split / full FULL + split），
  `--lsetma=0` 退回 mma+cp.async 做同 binary A/B。`lse_smem_bytes_tma1`(256)=51008B ⇒ 4 CTA/SM。
- **性能（同 binary A/B，event）**：preprocess **1.74–2.71×**、端到端 total **~5%**——
  S1024H8 causal 0.2792→**0.2649ms（1.054×）**、S4096H8 causal 2.7648→**2.6373（1.048×）**、
  S1024H8 full 0.2785→**0.2659（1.047×）**、S4096H16 full 5.4279→**5.1792（1.048×）**；main 不变。
- **ncu（D=256 causal S4096 LSE）**：TMA `lse_mma_kernel_bal_tma<256,1,0>` **79.6µs / L1TEX 1.196M
  扇区 / L2 10.49M / mem 45.1%** vs cp.async `lse_mma_kernel_bal<256,…>` **211.6µs / L1TEX 11.96M /
  L2 15.99M / mem 30.0%** ⇒ 载入扇区 **−10×**、Duration **2.66×**，从「载入指令 bound」转均衡。
- **精度/回归**：15 个 D=256 定长 shape `ours vs fp32 ref` relL2 **8.14–8.33 / 8.29–8.48 / 6.39–6.76%**
  （护栏内，与 O84/O99 统计一致）；`--ci --dtype fp8` 单/两文件 gate worst **7.629e-06 OK**、
  `--check docs/04` OK。D=128/D=512/varlen 逐字不变。
- **判决**：正结果、默认。fp8 4D-TMA LSE 覆盖 D=128/256/512 × causal/full。主 kernel 仍是
  L2 `red` bound（本卡无软件解，见「阻塞」）。

见 `docs/03` §125；原始输出 `src/fp8/fa_bwd_fp8_o103_d256_ab.out.txt`、
`..._o103_ncu_lse_d256_s4096.out.txt`。

### 5.112 第 198 轮（O104）：fp8 `head_dim=256` causal 的跨 head 全局 LPT（`hswap256`）—— **正结果（默认）**

- **动机**：`fp8 专项冲刺`剩余的唯一「不改工作划分/指令」杠杆是 **O93 的跨 head 全局 LPT**
  （`HSWAP`）。O93 只接在 `kvtma` 快路（D=128，需 TMA），而 **D=256 默认走通用
  `fa_bwd_fp8_mma_kernel`**（wgmma 档、cp.async、非 TMA），一直没有 LPT 排序。`fp8_mma_body`
  本就支持 `HSWAP`，缺的只是通用 kernel 的模板透传 + host。
- **改动（device 一行数学未改 + host）**：`fa_bwd_fp8_mma_kernel` / `launch_bwd_main` 加
  `bool HSWAP=false` 并透传；host 对 D=256 定长 causal、`base_grid=nblk*H*B ≤ 256`、nblk≥16
  建 O89 的 `d_mrev`、grid 改 `(H,nblk*ksplit,B)`、自动 ksplit 收 4；`--hswap=0` 回退。
  单/两文件 device 由 `sync_onefile_device.py` 同步（`identical: True`）。
- **关键判据（形状相关）**：hswap256 只在小 `base_grid` 为正——S1024H8 **1.133×**、
  S1024H16kv4 **1.090×**、S2048H8 **1.014×**；而 S2048H16（base=512）/S4096H8（base=512）
  hswap 在任意 k 都**回退**（0.84–0.95×，大工作集下 hswap 损 L2 局部性盖过全局 LPT 收益）
  ⇒ 故按 `base_grid ≤ 256` 门控（门控外逐值不变）。
- **ncu（D=256 causal S1024H8 main）**：Duration **231.5→194.5µs（1.19×）**、**`op_read`
  2.665M→1.939M（−27%）**、**`op_red` 13.369M 一字不变**（D=256 无 regdq，降低 ksplit 不减
  dK/dV 的 red）、L2 利用率 58.8%→69.9%。机制 = 全局 LPT 削尾波 + Q/dO 重读 k×→4×。
- **精度/回归**：relL2 vs fp32 ref 与 hswap0 **逐位相同 8.332/8.434/6.464%**；hswap-vs-hswap0
  max_abs ~1e-7（atomic 次序）；`--ci --dtype fp8` gate worst **7.629e-06 OK**、`--check docs/04`
  OK。D=128/D=512/full/varlen/D=256 非 causal 逐字不变。TE/FA3 无 fp8 D=256 列。
- **判决**：正结果、默认。D=256 第一条 `op_read` 下降的调度杠杆。`op_red` 主体墙仍无软件解。

见 `docs/03` §126；原始输出 `src/fp8/fa_bwd_fp8_o104_ab.out.txt`、
`..._o104_ncu_s1024h8.out.txt`、`..._o104_fa3_te_baseline.out.txt`、`..._o104_ci.out.txt`。

### 5.113 第 199 轮（O105）：fp8 D=512（MLA）causal 的 per-head LPT（`mrev`）—— **正结果（默认）**

- **动机**：落实 O104「下一步候选 ③」里不撞 smem 墙的一条。O104 把 O93 的跨 head 全局 LPT
  （`HSWAP`，grid 轴对调）推广到 D=256，但大 `base_grid` 下因 L2 局部性受损变负；而 **D=512
  （MLA）causal 一直没有任何 LPT**（`d_mrev=nullptr`）。MLA 1 CTA/SM、`target=S/2` 细切、
  causal 偏斜 ⇒ 最贵 m 块落尾波。
- **改动（host-only、device 一行未改、单/两文件同源）**：`mrev_elig` 加 **D=512**（`nblk>=16`
  沿用 O89）；定长 D=512 的三个主 kernel launcher 透传 `d_mrev` 当 `mt_m`
  （`fp8_mma_body` 早已消费 `mt_m`）。只做 **per-head 反转**（grid/head 排布不动，不改 L2 局部性）；
  `--mrev=0` 回退锯齿序。D=256 大 `base_grid` 的 mrev-only 对照仅 1.003×（噪声内）⇒ 不纳入。
- **性能（同 binary A/B，iters=300，3×400 复测）**：**S1024H2 D=512 causal main
  0.1184→0.1106ms（1.071×）、total 0.1505→0.1464ms（1.028×）**，稳定 1.069–1.071×；
  S512H4/H2（nblk=8）门控外中性；D=256 大 base/ D=128 O93 默认逐值不变。
- **ncu（D=512 causal S1024H2 main，1 CTA/SM/249 regs）**：**Duration 130.1–132.7→126.0µs**、
  **`op_read`/`op_red`/`op_write` 逐位不变（8.17M 扇区）**、L2 利用率 51.8→**54.5%**、warps 12.48%
  ⇒ 收益**纯来自尾波削平**（与 O93/O104 降 `op_read` 的机制不同）。
- **精度/回归**：relL2 vs fp32 ref **8.163/8.564/6.507%**（护栏内，与默认逐位同）；mrev-vs-mrev0
  max_abs ~1e-7（atomic 次序）；单/两文件一致性 gate worst **5.722e-06 OK**、`--check docs/04` OK。
  fp8 MLA 无 FA3/TE 列。
- **判决**：正结果、默认。MLA causal 首条主 kernel 调度杠杆（不改 L2 搬运量）。`op_red` 主体墙
  仍无软件解（见「阻塞」）。

见 `docs/03` §127；原始输出 `src/fp8/fa_bwd_fp8_o105_ab.out.txt`、
`..._o105_ncu_s1024h2_mrev{0,1}.out.txt`。

### 5.114 第 200 轮（O106）：fp8 变长（varlen）causal 主 kernel 的 per-head LPT（`mrev_varlen`）—— **正结果（默认）**

- **动机**：O89→O93→O104→O105 把「只改 CTA→m 块派发顺序、不改 L2 搬运量」的 LPT 调度杠杆逐个
  补到 fp8 定长因果路径（D=128/256/512），**变长 causal 一直没接**。变长因果偏斜更重（短序列
  产生早退死 CTA + 每序列 K 循环长 ∝`ceil(len_b/BM)`），默认非紧凑网格是 LPT 的反面（便宜块先跑、
  贵块压尾）。LSE 早已按降序工作量排镜像对表，主 kernel 却没接。
- **改动（纯 host、device 一行未改、单/两文件同源）**：`run_varlen` 加 `mrev_flag`（默认 1），建
  `d_mrev_v`（`nblk_max` 项反转表），非紧凑网格把它当 `mt_m` 传入（`mt_b=null` ⇒ `b=blockIdx.z`；
  `fp8_mma_body` 早已消费 `mt_m`）。`--compact` 不叠加。**形状门控**：
  `... && nblk_max>=16 && total_mt < nblk_max*B`（名义 maxlen 网格被短序列 padding）。
- **性能（同 binary A/B，iters=200，3× 复测）**：D=128 varlen causal **b4_t3840 1.015×**、
  **b8_t2904 1.010×**、b5_t3968 1.008×（均不齐、maxlen=2048）；**D=512 b3_t1792 1.080×**；
  等长 b4_t4096（无 padding、门控外）逐值不变（测得 mrev 为 −1.1% ⇒ 正是门控要挡的）；nblk=8 的
  b1_t512 门控外中性。
- **ncu**：b8 D128 main Duration **593.5→582.6µs**；b3 D512 main **203.8→180.96µs（1.126×）**；
  两者 **`op_red` 一字不变**（30.29M / 8.95M）、`op_read` 噪声内不变、regs/warps 不变，L2 利用率
  53.95→54.96% / 44.91→50.60% ⇒ 收益**纯来自尾波削平**（非降 L2 搬运量，同 O89/O105）。
- **精度/回归**：`ours vs fp32 ref` max_abs 与 mrev=0 **逐位相同**（如 b3 D512 3.404/3.436/3.508e-1，
  护栏内）；单/两文件 varlen fp8 一致性 gate **worst 1.907e-06 OK**；全量
  `--ci --dtype fp8 --hopper`（45 case）gate **9.537e-06 OK**、`--check docs/04` OK（214 行）。
- **判决**：正结果、默认。补齐 LPT 调度在 fp8（D=128/256/512 × 定长/变长 × causal）的最后一块。
  `op_red` 主体墙仍无软件解（见「阻塞」）。

见 `docs/03` §128；原始输出 `src/fp8/fa_bwd_fp8_o106_ab.out.txt`、
`..._o106_ncu_b8_t2904_h16_d128.out.txt`、`..._o106_ncu_b3_t1792_h2_d512.out.txt`、
`..._o106_consistency.out.txt`、`..._o106_ci.out.txt`。

### 5.115 第 201 轮（O107）：fp8 变长 causal 主 kernel 的 ksplit 重标定（D=128/D=512）—— **正结果（默认）**

- **动机**：O96/O97/O98/O99/O100 复核了定长/变长 full 与 causal D=256 的 ksplit，但 **causal 变长
  D=128/D=512 一直用 O29 原始档、从未单独测量**。O29 的 D=128 target 固定 8192，当
  `base_grid > target` 时 `kk<1` ⇒ **欠切到 k=1**（O106 A/B 里 b5_t3968 默认就是 k=1）；
  D=512 的 `target=maxlen/2` 同理。
- **改动（纯 host、device 一行未改、单/两文件同源）**：7 个 dumped casual 变长 shape 全扫
  k∈[1,16]（150–250 iters，3×）。
  · **D=128**：`auto_k = max(auto_k, 3)`（只抬下限，对齐 O100 full 的 k=3；**不按 nblk 封顶**）。
  · **D=512**：`target` 从 `maxlen/2` 提到 `maxlen`、`k = pow2floor(min(8, maxlen/base_grid))`、
    下限 2 ⇒ b3/b1 均取 8。
- **性能（同 binary A/B，iters=250，3× 复测）**：**D=128 b5_t3968 1.102× / b8_t2904 1.062×**；
  **D=512 b3_t1792 1.113× / b1_t512 1.060×**；b1_t512（k=16）、b4_t3840（k=4）auto 未变 ⇒ 中性。
- **ncu（main, launch-skip 1/count 1）**：b5 D=128 **1.42→1.28ms（1.109×）**、`op_read`
  19.29M→**23.50M**、`op_red` 73.1→79.0M、L2 53.6→**65.7%**；b8 **589.5→546.1µs（1.079×）**、
  L2 54.4→61.1%；b3 D=512 **182.9→163.0µs（1.122×）**、**`op_red` 8,945,664 一字不变**、L2
  50.2→56.5% ⇒ **机制 = 加 ksplit 换并行度/占用率**（`op_read` 上升、per-CTA K 循环变短），
  **不是** O89/O105/O106 的纯尾波。
- **精度/回归**：`ours vs fp32 ref` max_abs 与新/旧 ksplit **逐位相同**；relL2 D=128
  8.11–8.25/8.21–8.37/6.20–6.38%、D=512 8.42/8.48/6.47%（护栏内）。单/两文件一致性 gate
  **worst 5.722e-06 OK**；**全量 `--ci --dtype fp8 --hopper`（45 case）gate OK**、
  `--check docs/04` OK（214 行）；定长与 full 路径逐字不变。
- **判决**：正结果、默认。`op_red` 主体墙仍无软件解（见「阻塞」）；下一步候选：① 换卡；
  ② 把同类复核推广到 fp16/bf16 的 causal 变长；③ 覆盖型 backlog（D=256 变长 / MLA 降 smem）。

见 `docs/03` §129；原始输出 `src/fp8/fa_bwd_fp8_o107_ab.out.txt`、`..._o107_ncu.out.txt`、
`..._o107_ci.out.txt`。

### 5.116 第 202 轮（O108）：fp8 `head_dim=256` **变长**支持 + ksplit 标定 —— **正结果（默认）**

- **动机**：落实 O107「下一步候选 ③」覆盖型 backlog 的 **`D=256` 变长**。此前 `run_varlen` 的
  guard 只放行 `HD=128/512`（`D!=128&&D!=512` 直接 `return 1`），而 O99 已按定长写好 causal
  D=256 的 ksplit 公式并注明「无 D=256 变长 dump、按定长外推」——公式挂在跑不了的路径上。
  D=256 的 LSE/delta/主 kernel 实例在**定长**路径早已存在，varlen 只缺 host 接线。
- **改动（host 为主、设备数学一行未改、单/两文件同源）**：① guard 放行 256；② 量化 VPT=8
  （fused/non-fused 各加一支）；③ `delta_warp_kernel<256>`；④ LSE causal `launch_lse_bal<256,1>`、
  full `launch_lse_bal<256,1,true>`（同定长非 TMA 档，`d_cu` 定界）；⑤ 主 kernel
  `launch_bwd_main<256,64,32,false,true>`（wgmma SW128 + cp.async，同定长默认档）；⑥ O106 mrev
  门控扩到 `D∈{128,256,512}`；⑦ **ksplit 下限 `nblk/4`**（每 CTA 约 4 个 K tile）；⑧ 对拍打印加
  `relL2`。
- **性能（同 binary A/B，iters=250，3×）**：O99 旧档 → 新档：b4_t3840 causal（k4→8）1.109→1.069ms
  **1.038×**；b4_t4096 等长（k4→4）中性；**b8_t2904 causal（k2→8）1.060→0.893ms 1.186×**；
  full（O98，k=8）不变 1.898ms。ours 端到端 24–43 TF（`sum_b 4HL²D`）；D=256 变长**无外部基线**
  （TE fp8 D=256 causal 报 invalid、FA3 不支持 fp8、FA3 fp16 varlen D=256 本机 `fa3=NA`）。
- **ncu（b4_t3840 causal main）**：**L2 79.4%**、`op_red` 69.80M（**占 L2 87%**）、`op_read`
  10.59M、DRAM 6.83%、SM 29.7%、warps 12.4%（238 regs/115.71KB smem→2 CTA/SM）；stall
  `long 1.77 / short 1.47 / wait 1.35`。bound = **dK/dV 跨 CTA `red`**（与 D=128/256 定长旗舰同源，
  本卡无软件解，见「阻塞」）；ksplit/LPT 只买并行度、不动 `red` 总量。
- **数值/回归（护栏）**：relL2 dq/dk/dv = 8.31/8.42/6.41（b4_t3840 causal）、8.23/8.38/6.44、
  8.25/8.40/6.29、full 8.15/8.29/6.70% —— **全部在护栏内**；`max_abs` O(0.066–0.44)、与 D=128/512
  同量级；**单/两文件逐位一致**（≤1.4e-6）；**全量 `--ci --dtype fp8 --hopper`（49 case）**一致性
  gate **worst 6.676e-06 OK**、`docs/04` 表同步 **218 行**、`--check` OK、rc=0；D=128/512 varlen 与
  定长/full 路径逐字不变。
- **判决**：正结果、默认。fp8 变长覆盖补齐 `D=128/256/512 × causal/full`。`op_red` 主体墙仍无
  软件解；下一步候选：① 换卡；② cause 变长 ksplit 复核推广到 fp16/bf16；③ MLA 降 smem。

见 `docs/03` §130；原始输出 `src/fp8/fa_bwd_fp8_o108_ab.out.txt`、
`..._o108_ncu_d256_varlen.out.txt`、`..._o108_baseline.out.txt`、`..._o108_ci.out.txt`。

### 5.117 第 203 轮（O109）：量化分相 + LSE 跨 stream 重叠 —— **正结果（默认）**

- **背景**：fp8 main 的 L2 `red` 墙（105.4M 扇区 / L2 77.8%）已由 F3b/F4b/F6/F7/O83/O90/O91/
  O92/O95 收口为本卡无软件解。本轮 ncu 复核默认（main 1.38ms、L2 77.8%、张量核 11.65%）+
  编译宏复扫（`FA_WS1` 中性、`ILV34` 0.911×、`FA_R4` 0.951×）确认后，**转向非 main**。
- **观察**：S=4096 非 main 223µs 中，量化（99µs，DRAM 84%/SM 62%）与 LSE（121µs，SM 73%/DRAM 4%）
  资源互补却串行——O64/O66 的单颗融合量化 kernel 同时产出 LSE 与 main 的全部输入。
- **做法（device 一行数学未改 + host 一条 stream）**：新增
  `quantize_zero_delta_phase_kernel<VPT>`（phase0 = Q/K；phase1 = dO+delta/V/清零，逐行例程与
  合并版逐字同款 ⇒ 输出逐位相同）；host 令 `phase0(default) → {phase1(aux) || LSE(default)} →
  main(default.wait e1)`。仅定长 D=128 默认融合路径；`--ovlql=0` 关。单/两文件同步。
- **性能（同 binary A/B，iters=400，3×）**：**S512 1.030×**、S1024H32 1.017×、S1024 **kv4(GQA)
  1.016×**、S4096 1.005×；D=256（门控外）1.000×。
- **nsys**：一行迭代里 `phase0(34µs)` 后 `lse(stream7)` 与 `phase1(stream20)` **区间重叠**。
- **数值/回归**：quant 输出逐位相同 ⇒ `max_abs` 与历史逐位一致；`--ci --dtype fp8 --hopper`
  （49 case）gate **worst 7.153e-06 OK**、`docs/04` `--check` OK、rc=0。
- **判决**：正结果、默认。压的是**非 main 串行**，不动 main 的 `red` 墙（后者仍需换卡）。

见 `docs/03` §131；原始输出 `src/fp8/fa_bwd_fp8_o109_ab.out.txt`、`..._o109_nsys_overlap.out.txt`、
`..._o109_nsys_kernsum.out.txt`、`..._o109_macrosweep.out.txt`。

### 5.118 第 204 轮（O110）：量化分相/LSE 重叠扩到 D=256/D=512 + phase1 栅格封顶 —— **正结果（默认）**

- **动机**：落实 O109（§5.117）明确留下的「D=256/512 与 varlen 未接入」。D>128 的非 main 占比
  更大，且三者复用同一 `quantize_zero_delta_phase_kernel<VPT=D/32>`——LSE 与 phase1 同样资源互补。
- **nsys 揭示真瓶颈是「重叠率」而非「有没有重叠」**：phase1 默认栅格巨大（D=256 10240 CTA /
  D=512 2560 CTA）几乎占满 SM，LSE 只能等其尾部，**不封顶时实测重叠 ≈0**（D=256 phase1
  14.5µs/LSE 12.4µs 严格背靠背）。⇒ **给 phase1 栅格封顶**（auto `clamp(ntask/64,132,4096)`）
  让 LSE 拿到 SM 槽，封顶后重叠 ~3.5–3.8µs（phase1 自身略慢但净时间下降）。
- **改动（device 数学一行未改、单/两文件同源）**：① gating 放开到 `D∈{128,256,512}`，
  但 D>128 加 `rows_q=B*S*H>=2048`（否则小 shape 的分相 launch/event 开销盖过收益，
  如 D=512 S512H2 full 实测 0.993×）；② `quant_phase` 对 `phase1` 做 `grid=min(grid,cap)`，
  `--ovlcap=0(auto)/N/<0(off)`；封顶只改 `gridDim.x`，kernel 本就 grid-stride ⇒ 输出逐位相同。
- **性能（同 binary A/B，iters=300，3×）**：D=128 causal **S512 1.058× / S1024H32 1.028× /
  MQA(kv1) 1.031× / GQA(kv8) 1.044× / S4096 1.004×**；D=256 causal S1024 1.018× / S2048 1.015×、
  full S1024H16 1.013×；D=512 causal S1024H2 1.012× / S512H4 1.026×、full S2048H2 1.011×。
- **数值/回归**：`ours vs ref` 的 `max_abs` 在 `--ovlql=0/1` 间**逐位相同**（14 case×3 全
  `identical=True`）；单/两文件 gate 与 `--check docs/04` 见 `..._o110_ci.out.txt`。
- **对标**：TE fp8 仅支持 D=128（S1024H32 0.0743ms / S4096 0.3022ms）= ours total 的
  **4.07× / 5.22×**（O109 时 4.11× / 5.20×）；D=256/512 无 FA3/TE fp8 反向基线。
- **判决**：正结果、默认。main 的 L2 `red` 主体墙仍无软件解（见「阻塞」）；
  下一步候选：① 换卡；② causal 变长 ksplit 复核推广到 fp16/bf16；③ MLA 降 smem；
  ④ 把分相+封顶重叠推广到 varlen。

见 `docs/03` §132；原始输出 `src/fp8/fa_bwd_fp8_o110_ab.out.txt`、`..._o110_ci.out.txt`。

### 5.119 第 205 轮（O111）：量化分相 + LSE 跨 stream 重叠推广到变长（varlen）—— **正结果（默认）**

- **动机**：落实 O110（§5.118）明确留下的「`run_varlen` 仍是合并量化串行」。变长非 main 占比
  同样可观，且 LSE（SM bound、DRAM 4–7%）与量化 phase1（DRAM 78%、SM 30%）资源互补。
- **做法（device 一行未改、单/两文件 host 同源）**：`run_varlen` 透传 `--ovlql/--ovlcap`，新增
  `quant_phase_v(ph,stream)`（复用 O109 的 `quantize_zero_delta_phase_kernel<VPT=D/32>`）+
  aux stream/两个 event；`run_all` 的量化段改为 `phase0(default) → {phase1(aux) || LSE(default)}
  → default.wait(aux)`。门控同 O110（D=128 无条件、D=256/512 需 `rows_q=T*H>=2048`、
  `qfuseflag && dfuseflag`）；phase1 栅格封顶沿用 auto `clamp(ntask/64,132,4096)`。
- **性能（同 binary A/B，iters=200，3×）**：causal **b1_t512 d128 1.051× / b4_t3840 d128 1.028× /
  b5_t3968 d128 1.029× / b4_t3840 d256 1.018× / b4_t4096 d256 1.012× / b3_t1792 d512 1.030×**；
  full 0–0.5%（**1.005× / 1.001× / 1.000× / 1.000×**）。10 shape 无回退。
- **nsys**：稳态一行 `phase1(stream13, grid 4096, 69.7µs)` 与 `LSE(stream7, ~107µs)` **区间重叠**。
- **ncu**：phase1 DRAM **78.3%** / SM 30.5%，phase0 DRAM 77.7% / SM 57.8%，LSE DRAM **7.4%** /
  SM **57.7%** ⇒ 资源互补正是重叠收益来源。
- **数值/回归**：`max_abs`/`relL2` 在 `--ovlql=0/1` 间**逐位相同**（10 shape ×3），relL2 全在
  护栏内；`--ci --dtype fp8 --hopper`（49 case）gate **5.722e-06 OK**、`docs/04 --check` OK。
- **判决**：正结果、默认。main 的 L2 `red` 主体墙仍无软件解（见「阻塞」）；下一步候选：
  ① 换卡；② causal 变长 ksplit 复核推广到 fp16/bf16；③ MLA 降 smem；④ `--det` partial 并行化。

见 `docs/03` §133；原始输出 `src/fp8/fa_bwd_fp8_o111_ab.out.txt`、`..._o111_nsys.out.txt`、
`..._o111_ncu.out.txt`、`..._o111_ci.out.txt`。

### 5.120 第 206 轮（O112）：fp8 定长 causal `head_dim=512`（MLA）的 ksplit 重标定 —— **正结果（默认）**

- **动机**：O96–O108 的 ksplit 复核把 full、causal D=128/256、变长 causal/full 都审计了，唯独
  **定长 causal `D=512`（MLA）** 一直沿用 **O29** 的 `target=S/2`——那是 **pre-O51**（MLA K/V
  `cp.async` 回填流水）/ **pre-O105**（mrev）时代的标定。O107 已证明变长 causal `D=512` 需把
  target 提到 `maxlen`（cap 8）⇒ 定长 causal `D=512` 是其未复核姊妹。
- **做法（纯 host、device 一行未改、单/两文件同源）**：定长自动档 O99 之后新增
  `if (ksplit_auto && causal && D==512)`，规则与 O107 逐字一致
  `k=pow2floor(min(8, S/base_grid))`、下限 2（`base_grid=ceil(S/64)*H*B`）。full→O97、
  causal D=256→O99、causal D=128→O93/O96 均在本段前判定；`--ksplit=K` 不覆盖。只改跨 CTA
  `atomicAdd` 次序。
- **性能（同 binary ksplit sweep，iters=200，3× 复测）**：6 个 causal D=512 定长 shape 全扫
  `k∈[2,32]`（S2048H2/S4096H2 为本轮新 dump 的 causal），最优稳定 **k≈8**。auto 旧档→新档：
  **S512H2 1.065–1.075×、S1024H2 1.020–1.024×，S2048H2 +3.6%、S4096H2 +2.7%**，S256H2/S512H4
  中性（本就在/接近 k=8）。端到端 auto 新档 S512H2 0.0636 / S1024H2 0.1427 / S4096H2 1.3474 ms。
- **ncu（S1024H2 main，同 binary k=16 vs k=8）**：**`lts op_red` 逐位不变 6,684,672**、
  **`lts op_read` 1,402,132→1,172,906（−16.3%）**、Duration 124.8→**123.1µs**；`sm__warps_active`
  恒 12.5%、SM 23% ⇒ 仍 1 CTA/SM 延迟 bound，收益来自「减 Q/dO 重读 + 保并发」。
- **数值/回归**：k=8 vs k=16 `max_abs`/relL2 **逐位相同**（纯 atomic 次序）；`--ci --dtype fp8
  --hopper` gate **5.722e-06 OK**、`docs/04 --check` OK（218 行）。D=512 无 FA3/TE fp8 外部列。
- **判决**：正结果、默认。**这是「降 L2 搬运量」里可动的 Q/dO 重读那一半**（`red` 另一半受本卡
  寄存器/smem 墙锁定，见「阻塞」）。下一步候选：① 换卡；② causal 变长 ksplit 推广 fp16/bf16；
  ③ MLA 降 smem（实测为 **regs 249 → 1 CTA/SM 的寄存器墙，非 smem**）；④ `--det` 并行化。

见 `docs/03` §134；原始输出 `src/fp8/fa_bwd_fp8_o112_ab.out.txt`、`..._o112_default_sweep.out.txt`、
`..._o112_onefile.out.txt`、`..._o112_ncu_s1024h2.out.txt`、`..._o112_ci.out.txt`。

### 5.121 第 207 轮（O113）：fp8 full 变长 `head_dim=256` 的 ksplit 重标定 —— **正结果（默认）**

- **缺口**：O96–O112 的 ksplit 复核几乎覆盖 fp8 全维度，唯独 **full（非 causal）变长
  `D=256`** 沿用 O98 的「无 dump、套 O97 定长公式 `k=8192/base`」。本轮先补 dump 5 个
  D=256 full 变长 shape，再单独全扫 `k∈[1,16]`（iters=200，min of 3）。
- **做法（纯 host、device 一行未改、单/两文件同源）**：`!causal && D==256 && maxlen>=2048`
  分支改为 `k=max(6, 2048/base_grid)`、cap 12（旧 `k=8192/base_grid`）。只改跨 CTA
  `atomicAdd` 次序，数值在 fp8 噪声内；`--ksplit=K` 不覆盖。
- **性能（同 binary，event）**：**b4_t3840_h8 1.899→1.859ms（1.022×，旧 k=8→6）、
  b8_t2904_h8 1.554→1.504ms（1.033×，旧 k=4→6）、b4_t3840_h16 3.617→3.552ms（1.018×，
  旧 k=4→6）**；b5_t3968_h8 中性（旧已 k=6）；等长 b4_t4096_h16（maxlen=1024）走既有波支路
  k=9，不受影响。
- **ncu（b8_t2904_h8 main，同 binary k=4/6/8）**：**`lts op_red` 逐位不变 107.77M**、
  `op_read` 10.07/11.77/11.94M、Duration **1.38/1.33/1.39 ms**、warps_active 恒 ~12.4%
  ⇒ 2 CTA/SM 的并行度/尾波 bound，ksplit 是「换并发」而非「减 Q/dO 重读」的旋钮（同 O112）。
- **数值/回归**：5 个新 shape relL2 全在护栏内；`--ci --dtype fp8` 55 case 单/两文件
  **worst 7.153e-06 OK**、`docs/04 --check` **OK（224 行，已同步新 case）**。
- **判决**：正结果、默认。**fp8 的 ksplit 自动档至此 full/causal × 定长/变长 ×
  D=128/256/512 全覆盖复核**。下一步：① 换卡（L2 `red` 主体墙无软件解）；② causal 变长
  ksplit 推广 fp16/bf16；③ MLA 降 smem（实为寄存器墙）；④ `--det`/量化并行化。

见 `docs/03` §135；原始输出 `src/fp8/fa_bwd_fp8_o113_varlen_full_d256_ksweep.out.txt`、
`..._o113_ncu_kvtma_d256_full.out.txt`、`..._mma_onefile_o113_d256_full.out.txt`。

### 5.122 第 208 轮（O114）：fp8 `dK/dV` `red` 元素宽度收窄（fp32→fp16）—— **负结果（opt-in）**

- **动机**：fp8 causal 旗舰的 L2 墙 = dK/dV 跨 CTA `red`（S4096 `lts op_red` 105.4M ≈ L2 的
  80%）。「阻塞」把 `red` 判为工作划分决定、与归约机制无关——但前提是**元素宽度恒 fp32**。
  本轮试正交维度：**元素 fp32→fp16**（写入字节减半 ⇒ 扇区应减半）。
- **冒烟（决定性）**：`atomicAdd(__half2*)` 退化成 `ATOM`（读改写）；手写 PTX
  `red.global.add.noftz.f16x2` 才是纯 RED。**连续地址模式减半扇区**（3.24M→1.62M、1.34×）；
  **模仿 `mma.m16n8` 片段的行散列模式扇区一字不变**（3,244,032→3,244,032）。
- **接线（`-DFA_REDHALF=1` opt-in）**：`fp8_mma_body` 非 DET/BULKRED/FA_R4 的 dK/dV red 走
  `red_addh2`，主机换 fp16 累加缓冲 + `redhalf_finalize`。SASS 生效（新增 2144 条
  `REDG.E.ADD.F16x2`）。
- **真机 ncu（S4096 默认 kvtma main）**：`lts op_red` **105,381,888 → 105,381,888 逐位不变**、
  Duration 1.38→1.38 ms。**根因**：mma 片段使一次 warp red 请求覆盖 8 个不同行 ⇒ 固定 8 扇区；
  fp32 float2 (4 lane×8B=32B/行) 已填满扇区，fp16 f16x2 (4 lane×4B=16B/行) 仍占满但只写一半
  ⇒ 扇区不减。要减半需「8 lane/行」或「64-bit fp16 RED」，PTX/`mma` 均不提供。
- **精度/性能 A/B（6 shape）**：`max_abs` 变化 ≤0.01（护栏内）；total 一律更慢 1.3–20%
  （多出的 memset+finalize），main 本体中性。ours S4096 87.2 TF（峰值 4.4%）、≈5.2× TE FP8
  （与 O112/O113 持平）。
- **判决**：负结果，默认 0（两文件定长 D=128/256 opt-in 探针）。「收窄元素宽度」四证关闭；
  `red` 墙仍只剩换工作划分（已判死）或换卡。副产品：可复现的 `red` 扇区最小冒烟。

见 `docs/03` §136；原始输出 `src/fp8/fa_bwd_fp8_o114_redhalf_smoke.out.txt`、
`..._o114_redhalf_smoke_ncu.out.txt`、`..._o114_sass.out.txt`、`..._o114_ab.out.txt`、
`..._o114_ncu_ab_s4096.out.txt`。

### 5.123 第 209 轮（O115）：fp16/bf16 定长 causal D=128（S≥4096 的 `wgmma2b`）主 kernel 默认走逐 atom 4D-TMA —— **正结果（默认）**

- **背景**：fp8 的「main 切 wgmma+TMA」在 F1/F2/O37/O41 已**默认全开**；fp16/bf16 的
  对应物 O33/O34（BN=128 的 `wgmma2b` 逐 atom 4D-TMA）与 O35/O36（BN=64 的 `wgmma2`）
  却一直 **`--maintma` opt-in**。F8（第 161 轮）已把 fp16/bf16 定长**默认构建**切成 Hopper
  （`-DFA_TMA -lcuda`），TMA 主 kernel 可默认用，却仍在跑 `cp.async` ⇒ 一条被漏掉的默认化。
- **改动（纯 host、device 一行未改、单/两文件 + fp16/bf16 同源）**：`maintma_sel` 默认 `0`→`-1`，
  在 `wg2bn/wg2` 与 cluster 覆盖定型后自动：**仅 `D==128 && wg2bn_sel`（S≥4096 的 `wgmma2b`）
  时=1**，其余 0（BN=64 的 `wgmma2` 小 S 是 grid/latency bound，TMA 反慢）；无 `FA_TMA` 的老
  sm_90 构建逐字退化为 0。`--maintma=0/1` 仍可强制 A/B；顺带补上 bf16 的 `+maintma` 打印。
- **性能（同 binary A/B，S4096 causal，iters=200）**：fp16 两文件 total **1.1665→1.1222ms
  （1.039×）** / 单文件 1.1709→1.1199（1.046×）；bf16 两文件
  **1.1581→1.1207（1.033×）** / 单文件 1.1581→1.1154（1.038×）。
- **门控证据**：BN=64 的 `wgmma2` 若强制 TMA，S512 total **0.0588→0.0757（0.78×）**、
  S1024 kv4 1.01× ⇒ 保持 `cp.async`。
- **ncu（fp16 `wgmma2b` S4096）**：`lts op_red` **51,904,512 逐字节不变**、Duration
  955.7→**918.6µs（1.040×）**、`sm__inst_executed` 214.77M→**161.44M（−24.8%）**、L2% 57.8→59.9
  ⇒ 收益 100% = **TMA 省下的载入指令/地址运算**，主墙（dK/dV 的 L2 `red`）未动（同 O33 结论）。
- **数值/护栏**：`ours vs fp32 ref` `max_abs` 在 `--maintma=0/1` 间**逐位相同**（fp16
  `1.883/1.734/1.966e-3`、bf16 `1.510/1.340/1.631e-2`），仅差跨 CTA atomic 次序（~2e-5）；
  `--ci --dtype fp16 bf16` 单/两文件 gate **fp16 1.953e-3 OK / bf16 3.906e-3 OK**、
  `docs/04 --check` OK（224 行）。
- **与本轮 fp8 的关系**：fp8 默认 main 仍是本卡唯一真杠杆——L2 `red` 主体墙（占 L2 ~80%）已被
  F3b/F4b/F6/F7/O90–O95/O114 全部收口为「无软件解」，只剩换卡/多 warpgroup WS；本轮因此把
  fp8 的 main-TMA 默认化**补到 fp16/bf16**（同一「降搬运/逼近 TE」的方法论）。

见 `docs/01` §26、`docs/01b` §6bc；原始输出 `src/fp16/fa_bwd_fp16_o115_maintma_ab.out.txt`、
`src/bf16/fa_bwd_bf16_o115_maintma_ab.out.txt`、`src/fp16/fa_bwd_fp16_o115_ncu_maintma_s4096.out.txt`、
`src/fa_bwd_o115_ci.out.txt`。

### 5.124 第 210 轮（O116）：fp8 主 kernel 的 host/运行期旋钮**系统复核**（ksplit / hswap / mrev / 端到端量化重叠 / 栅格封顶 × 4 形状）＋ TE-vs-ours L2 红字账 —— **负结果（默认一行未改）**

- **动机**：O115 之后 fp8 默认路径被逐条收口（main 的 L2 `red` 主体墙 = F3b/F4b/F6/F7/O90–O95/O114
  全部「无软件解」；非 main 已被 O109–O111 的量化分相 + LSE 跨 stream 重叠压到带宽墙）。**唯一
  还没系统复核的维度是「纯 host/运行期旋钮的联合最优性」**——O93/O96–O113 每次只审一条分支，
  没有把 ksplit / hswap / mrev / ovlql / ovlcap 放在同一 binary、同一 session、同一批形状上一起扫。
  本轮补这一刀，并顺带把「主 kernel 的 L2 `red` 到底还能不能动」用 TE 同 session ncu 再钉一次。
- **方法（新工具 `harness/fa_fp8_main_sweep.py`，纯 harness）**：把 Hopper 构建
  （`-DFA_WGMMA -DFA_TMA`）出的 `src/fp8/fa_bwd_fp8_main.out` 在 kernel_lab 容器里按
  「4 case × 9 配置」矩阵跑，解析 `[timing] total` / `quant|preprocess|main`，给出「默认 vs 每档」
  的 total×/main×。case 覆盖 MHA S4096、大 H GQA（q32/kv4、q40/kv8）、MQA（q64/kv1）。
- **结果（`src/fp8/fa_bwd_fp8_o116_knob_sweep.out.txt`，iters=50/100）**：
  * **ksplit**：默认 auto（`hswap` 生效时=2）在 4 个 shape 上**全最优**；k=1 慢 1.5–15.9%、
    k=4 慢 2.7–6.4%（S4096 例外：k=4 仅慢 1.1%）。
  * **hswap=0 / mrev=0**：4 个 shape **全负**（total 慢 3.2–11.7%），且默认档自动把 ksplit
    从 8 收到 2（grid 512→128）——证实 O93 的「跨 head 全局 LPT + 低 ksplit」在 MQA/GQA 上同向。
  * **ovlql=0**（关量化分相/LSE 重叠）：4 个 shape total 慢 0.2–4.8% ⇒ O109/O110/O111 默认开正确。
  * **ovlcap**(-1/1024/4096)：与 auto 档（`ntask/64` clamp [132,4096]）在 ~1% 噪声内，无稳定赢家。
  * ⇒ **默认档已是这五类 host 旋钮的全局最优，无一条可转正**；默认一行未改。
- **TE-vs-ours L2 红字账（S4096 causal，同 session ncu；`_o116_ncu_{ours,te}_main_s4096.out.txt`）**：

  | | kernel | grid/线程 | smem | regs | Duration | `op_red` | `op_read` | L2% | SM% | warps% |
  |---|---|---|---|---|---|---|---|---|---|---|
  | ours | `fa_bwd_fp8_mma_kvtma_kernel<128,64,32,...>` | 128×16 / 128 | 74.82KB(3 CTA/SM) | 168 | **1.37 ms** | **105,381,888** | 24,266,474 | 78.20 | 46.92 | 18.66 |
  | TE | `..._flash_bprop_wgmma_f8_..._64x64x128` | 132 / **384** | **232.45KB(1 CTA/SM)** | 168 | **257.9 µs** | **25,957,064** | 10,232,698 | 70.86 | 45.07 | 15.61 |

  **同为 BM=64；差距全在「搬运量」**：`red` **4.06×**、`read` **2.37×**、时间 **5.31×**。
  ours 的 `red` 占其 L2 总量 `24.27+105.38+0.10=129.75M` 的 **81.2%**。
- **red 扇区账（本轮把 1.5× 说清）**：causal S4096 下 `ntiles(mblk)=2(mblk+1)`、
  `Σ_{mblk=0}^{63} ntiles = 4160`/head；每 tile 的 dV/dK 各写 `BN×HD=32×128` 个元素、
  warp 内 8 行 × 8 列 = 256B = **8 扇区（刚好填满、无浪费，O114 已证）** ⇒ 理论下界
  `2(dK/dV)·4160·(4096元素/8)·16头 = 68.2M` 扇区。实测 **105.4M = 1.545×** —— 多出的
  ~37M 扇区来自**跨 CTA 贡献（每个 KV 元素被多少 m-block 归约）**，不是扇区浪费、也不是归约机制；
  要压只能减「每元素贡献 CTA 数」= 放大 BM / KV-owner（寄存器/smem 墙，见「阻塞」）。
- **L2 读侧同理**：`op_read` 24.27M = 777MB，其中 K/V 因「每 (m,nt) tile 重读一次」占 ~545MB、
  Q/dO（ksplit=2）~34MB；要压也需放大 BM（同一堵墙）。
- **判决**：**负结果、默认一行未改**。fp8 主 kernel 的 host 旋钮前沿至此全部扫清；真差距 =
  TE 的「大 tile（384 线程 / 232KB smem / 1 CTA/SM）+ 更好的工作划分」把每元素贡献 CTA 数压到
  ours 的 ~1/4，本卡寄存器/smem 装不下（F6/F7/p160/O83/O86/O90/O91/O114 同源）。
- **数值/护栏**：默认档本轮未改任何 device/host 默认；对 dump 的 ref/te npy（S4096 causal）
  ours relL2 **8.149/8.263/6.489%**（全在 fp8 硬护栏 8.2/8.3/6.5%±0.3 内、且优于 TE 的
  10.574/10.546/27.452%）、`max_abs` O(0.26–0.32)；`--ci --no-run --dtype fp8` **55 case
  单/两文件 gate worst 5.722e-06 OK**、`docs/04 --check` OK（224 行）。
- **原始输出**：`src/fp8/fa_bwd_fp8_o116_knob_sweep.out.txt`、
  `src/fp8/fa_bwd_fp8_o116_ncu_ours_main_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o116_ncu_te_main_s4096.out.txt`；工具 `harness/fa_fp8_main_sweep.py`。

### 5.125 第 211 轮（O117）：fp8 主 kernel 平台期**最终收口**（小 shape host 旋钮 + 更细 ksplit 探针）—— **负结果（默认一行未改）**

- **动机**：O116（§5.124）把 5 类 host 旋钮在 4 个 S≥1024 shape 上判定默认最优，但**没覆盖
  小 shape（S512、D=256）与 ksplit=8/32/64 更细档**。本轮补这一刀并用 fresh 同 session ncu 复证。
- **结果**：默认 auto ksplit 在 MHA/GQA/MQA × D=128/256 × S512–4096 **全部最优**；
  `b1_s512_h16_d128` k=64 与 default 齐平（0.0785 vs 0.0767）、k=8/32 的 2× 波动系共享 GPU
  邻容器负载（k=64 两次一致）；`b1_s512_h8_d256` 各档全在 ~1.6% 噪声内；S1024 GQA/MQA 与
  S4096 flagship 的 k=8/32/64 单调更慢（同 O116）。⇒ **host 旋钮前沿在所有 shape 尺度扫清。**
- **fresh ncu（ours vs TE，S4096 causal）**：ours `kvtma<128,64,32>`（3 CTA/SM/128 线程）
  **1.38ms / `op_red` 105,381,888 / `op_read` 24,238,059 / `sm inst` 617,655,296 / L2 78.09%**
  vs TE（1 CTA/SM/384 线程）**257.63µs / 25,957,024 / 10,197,700 / 108,334,714 / 70.97%** ——
  同 BM=64，差距全在搬运量（red **4.06×**、read **2.38×**、指令 **5.70×**、时间 **5.36×**）。
- **判决**：**负结果、默认一行未改**。`red` 主体墙 = 本卡工作划分的硬件下界。**唯一未试的
  搬运量杠杆 = TMA multicast + cluster 共享 K/V 读**（直打 read 2.38× 差距，工程量大）；否则
  换卡 / 覆盖型 backlog（fp16/bf16 `head_dim=256`）。
- **原始输出**：`src/fp8/fa_bwd_fp8_o117_smallshape_sweep.out.txt`、
  `src/fp8/fa_bwd_fp8_o117_ncu_ours_main_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o117_ncu_te_main_s4096.out.txt`、`src/fp8/fa_bwd_fp8_o117_summary.out.txt`。

### 5.126 第 212 轮（O118）：候选②「TMA multicast + cluster 共享 K/V 读」的 de-risk —— **机制正结果 / prize 有界、不集成（默认一行未改）**

- **动机**：O117（§5.125）后，`red` 主体墙已收口；ROADMAP「下一步候选 ②」= TMA cluster
  multicast（同 cluster 多 CTA 读同一 K/V 时 leader 发一条 `...multicast::cluster` 广播，摊薄
  L2 读扇区）。在改 `fp8_mma_body` 前先按 O42/O79/O80 惯例做独立 de-risk。
- **冒烟（`src/fp8/fa_bwd_fp8_mcast_smoke.cu`）**：在**与主 kernel 相同**的 K/V 4D-TMA 几何
  （UINT8 / dims={D,S,Hkv,B} / SWIZZLE_128B / box={128,32} / `KS_SZ=4096` / `sw128_off_fp8`）上，
  leader 发 multicast+mask、每 CTA 各自 `arrive.expect_tx`+等本地 mbar（kernel-opt 26/36 篇协议），
  `cudaLaunchKernelEx` 指定 `clusterDim.x=CN`。
- **正确性**：CN=1/2/4（R=1）multicast 的 SW128 tile 与「逐 CTA 各自发 TMA」及 host swizzle 参考
  **逐字节相同（mismatch=0/0/0）** ⇒ 机制在 fp8 K/V 几何上完全正确。
- **搬运量（ncu，CN=2，R=128，grid=4096，copy=0，同 binary A/B）**：`lts op_read`
  **52.6–52.9M → 33.82M（0.64×，−36%）**、`srcunit_tex op_read` 52.4M→33.55M、**L2 利用率
  49.6%→31.5%**；`dram bytes_read` 恒 8.40MB（K 常驻 L2）；**Duration 391→393µs（1.00×，该
  microbench 不在 L2 墙上）**。CN=4 的 ncu replay 在本工具链下不稳定（残留死锁进程，需手动清理），
  只留 CN=4 的正确性。
- **不集成判据**：① 主 kernel L2 ≈ `op_red` 105.38M（~80%）+ `op_read` 24.24M（**~19%**）+ 写
  0.14M，read 占 <1/5，且 multicast **不动 `red`** ⇒ prize 上限 ~9% L2、实测只到 ~6% L2；
  ② causal 的 cluster 配对不天然（`ntiles` 随 m 块差 BM/BN=2，需锁步 padding 或镜像配对重设计）；
  ③ cluster≥4 代表负载下 ncu replay 脆弱 + 残留进程风险。
- **判决**：**de-risk 机制正结果、prize 有界、ROI 低 ⇒ 暂不集成，记为有数据支撑的 backlog；
  默认路径一行未改（独立冒烟，不触数值）。** `red` 主体墙仍只剩换卡 / 覆盖型 backlog
  （fp16/bf16 `head_dim=256`）。
- **原始输出**：`src/fp8/fa_bwd_fp8_o118_mcast_smoke.out.txt`、
  `src/fp8/fa_bwd_fp8_o118_mcast_ncu.out.txt`。

### 5.127 第 213 轮（O119）：fp8 K/V TMA cluster multicast **落地集成**（Q 头轴 / GQA-MQA） —— **负结果（opt-in，默认关）**

- **动机**：O118（§5.126）把候选② multicast de-risk 为「机制正、prize 有界、不集成」，其顾虑之一是
  causal 的 **m 轴** cluster 配对不天然（`ntiles` 随 m 块差 BM/BN=2）。本轮换**天然锁步的 Q 头轴**：
  O93 的 HSWAP 快路（head=blockIdx.x）上，cluster 沿 x 分组的 Q 头同属一个 `(mt,part)` ⇒ 行程完全
  一致；对 **GQA/MQA**（`G=H/Hkv>1` 共享 KV 头）即读同一批 K/V tile ⇒ 直接可 multicast。
- **改动（device 单/两文件逐字一致）**：`fp8_mma_body`/`kvtma` kernel 加模板参 `int MCAST=1`；
  新增 `fp8_cluster_rank/sync`、`fp8_fence_mbar_init`、`tma_load_4d_mc`；K/V 的 4D-TMA 由
  leader(rank0) 发 multicast（mask=`(1<<W)-1`），每 CTA 各自 `arrive.expect_tx` 等本地 mbar；
  **每 tile 回填前一次 `barrier.cluster` 锁步**（去锁步会串 tile、实测死锁）。host 用
  `cudaLaunchKernelEx`+`cudaLaunchAttributeClusterDimension` 启动（`cudaFuncAttributeRequiredClusterWidth`
  在 `<<<>>>` 下报 `cluster misconfiguration`）。`--mcast=C`（2/4/8）或 `--mcast`=auto
  （最大 2 的幂 ≤ min(G,8)），门控 `HSWAP && kv_tma && D==128 && causal && H%W==0 && G%W==0`；
  默认 `--mcast=1`（关）。
- **正确性**：MQA/GQA 的 `ours vs fp32 ref` 三梯度 `max_abs` 在 `--mcast=1/2/4/8` 间**逐位相同**
  （`4.101e-01/1.572/2.126`）；单/两文件一致；MHA `G=1` auto ineligible、路径逐字退回
  （S4096 `2.635/2.644/3.216e-1`、total 1.5796ms 与历史一致）。
- **L2 / 性能（ncu + 计时，MQA q64kv1 S1024，同 binary A/B）**：`lts op_read`
  **6,264,250 → 2,464,328（0.39×，−60.7%）**、`op_red` **逐位不变** 29,884,416、Duration
  **450.3→585.7µs（1.30×）**；计时 main **0.4156 → 0.4524(W2) / 0.5117(W4) / 0.5640(W8)**、
  GQA q64kv4 main `0.3946 → 0.5517(W8)`。⇒ **读确实大幅摊薄，但 multicast 正确性必需的每-tile
  `barrier.cluster` 锁步代价 > 读节省（结构性倒亏 8–40%）。**
- **判决**：**负结果、`--mcast` opt-in、默认关**。与 O118 合并：候选② multicast 在**本卡当前
  非-WS、非-persistent 的 Q-owner 主体**上判死（锁步成本结构性）；解锁需**完整 warp specialization
  + persistent**（用 mbarrier 而非 `barrier.cluster` 做跨 CTA 生产/消费同步）。`red` 主体墙仍只剩
  换卡 / 覆盖型 backlog（fp16/bf16 `head_dim=256`）。数值逐位不变、默认逐字退回。
- **原始输出**：`src/fp8/fa_bwd_fp8_o119_ab.out.txt`、
  `src/fp8/fa_bwd_fp8_o119_ncu_mqa_kv1_mcast{1,8}.out.txt`。

### 5.128 第 214 轮（O120）：F3b② 的 **warp specialization de-risk** —— 「单一 barrier 域」病因的直接检验 —— **机制正结果（独立冒烟，默认一行未改）**

- **动机**：fp8 main 的 `red` 墙唯一软件杠杆是放大 BM（red 砍半），但 BM≥128 ⇒ 1 CTA/SM。O90
  （`wgmma2` BM=128 0.66×）/O91（`wg3` BM=192 0.52×）实测：red 真降到 0.43×、时间反翻倍，O91 把
  病因钉为「**1 CTA/SM 单一 barrier 域**：每 tile 5 个 `__syncthreads` 把 12 warp 串成依赖链、
  K/V 搬运与 wgmma 不重叠」。但此前**只从『改多 warpgroup 后变慢』间接推断，从未把同步机制换掉
  直接检验**。F3b② 留的「WS 完整化（producer/consumer + mbarrier 替换 `__syncthreads`）」是本卡
  fp8 主 kernel 上唯一未判决的软件杠杆。
- **方法**：独立冒烟 `src/fp8/fa_bwd_fp8_ws_smoke.cu`，在与主 kernel 相同的 fp8 K/V 几何
  （UINT8 / SW128 / 4D-TMA box={128,32} / KS_SZ=4096B）上比较三种同步机制的流水吞吐（1 CTA/SM、
  grid=132、K 常驻 L2）：**(A) SYNC** 每 tile TMA + `__syncthreads`（O91 形态）；**(A2)
  SYNC-PREFETCH** ring buffer + thread0 预取但仍全 CTA `__syncthreads`；**(B) WS** 独立 producer
  warpgroup 预取 + consumer 只等 mbarrier(full)→wgmma→arrive(empty)，无 `__syncthreads`。
  正确性：三者 sink 在 fp32 原子次序噪声内一致（~1e-6）、`compute-sanitizer` **0 errors**。
- **性能（GPU1，iters=20，read=2214.6MB/launch）**：A SYNC **1.648ms / 1343GB/s**；
  A2 **1.087–1.089ms / 2035GB/s（1.50×）**；**B WS 0.798–0.860ms / 2576–2775GB/s
  （1.89× NS=2 → 2.04× NS=8）**。
- **ncu（同 binary A/B，1 CTA/SM，12.5% warps）**：`lts op_read` **19,330,267 vs 19,403,553
  （≈1.00×，同字节）**、`dram bytes_read` **8.40 vs 8.41MB（K 常驻 L2）**、warps 12.50 vs 12.49%、
  **Duration 1.67ms → 0.824ms（2.03×）**。⇒ **同一份 L2 字节、同样 warp 占用，纯延迟暴露差异**。
- **判决**：**机制正结果**——O91「单一 barrier 域」病因被直接检验成立，**warp specialization 是
  1 CTA/SM 下恢复吞吐的解锁路径（2.0×，L2 字节不变）**。副结论：单靠预取深度（A2）只得 1.50×，
  多出的 1.33× 来自「把 producer 移出 `__syncthreads` 域」。**下一步 = F3b 主体化**：把该
  producer/consumer 模式落进 `wgmma2`/`wg3`（BM=128/192、1 CTA/SM），把 O90/O91 的延迟 bound
  拉回 L2 bound（结合 `wg3` 的 red 0.43×，理论上界 ~默认档 1.7–1.8×，逼近 TE），工程量中等偏大
  （重排 GEMM3/4/5 的 warp 归属 + 全 mbarrier 化）。默认路径一行未改。
- **原始输出**：`src/fp8/fa_bwd_fp8_o120_ws_smoke.out.txt`、
  `src/fp8/fa_bwd_fp8_o120_ncu_{sync,ws}.out.txt`。

### 5.129 第 215 轮（O121）：F3b 主体化第一步 —— 把 O120 的 WS 落进 `wgmma2_tma`（BM=128）—— **中性/负结果（opt-in `--wg2ws`）**

- **动机**：O120（§5.128）在独立冒烟上证明「把 K/V TMA 的 producer 移出 `__syncthreads` 域」值
  **2.0×**（同 L2 字节、同 warp），据此立项 F3b 主体化——把该 producer/consumer 模式落进
  `wgmma2`（BM=128、`red` 已砍半到 58.2M），预期把 O90 的 0.54× 默认档拉回 ~1.7–1.8×。
- **改动**（`src/fp8/fa_bwd_fp8_kernels.cuh` §3e 新增 `fa_bwd_fp8_wgmma2_ws_kernel`；`main.cu`
  加 launcher + `[O121 A/B]`）：**384 线程 = 2 compute WG + 1 producer WG**；compute 侧 5 个
  `__syncthreads` → 3 个命名 barrier `bar.sync 1,256`；K/V 双 ring 用 `full/empty` mbarrier
  （empty count=256）；GEMM 数据通路逐字沿用 O90。`--wg2ws` opt-in、默认一行未改。
  首测死锁坑：K/V 若写成两个独立前瞻循环会把 producer 卡在 `emptyK`、V 后继永不发出 ⇒ consumer
  等 `fullV` 死锁（S512 `NT≤2` 不触发、S4096 必现）；合并成单循环交替推进即解。
- **数值**：ws vs tma `max_abs` dq/dk/dv = 1.19e-7/2.38e-7/4.77e-7（仅原子/归约次序）；
  **ws vs fp32 ref relL2** S512 **8.18/8.41/6.36%**、S4096 **8.15/8.39/6.52%** ⇒ 护栏内。
- **性能（同 session A/B，iters=100）**：S512 **0.1052→0.1050ms（1.002×）**、
  S4096 **2.6999→2.6877ms（1.005×）** ⇒ **中性**，远不及 O120 冒烟的 2.0×。
- **ncu（S4096，同 binary）**：wg2tma vs wg2ws —— Duration **2.72 vs 2.71ms**、`lts op_red`
  **58,195,968 逐位相同**、SM **32.5 vs 32.9%**、L2 **22.0 vs 23.6%**、warps **12.5 vs 13.9%**、
  **stall `barrier` 仅 0.29 vs 0.26**、主导 stall = `short_scoreboard 1.22/1.22` + `wait 1.32/1.46`。
- **判决**：**F3b 主体化中性/负结果**。真实反向在 BM=128、1 CTA/SM 下的墙是 **compute 流水延迟**
  （只有 8 个 compute warp 跑 5 个 GEMM+fold，无可填延迟的第二个独立 CTA），**不是 O91/O120 归因的
  「单一 barrier 域」**（ncu `barrier` stall 实测 ~0.28，本就极低）。O120 冒烟的 consumer 每 tile
  只做 1 个 wgmma（纯访存延迟 bound）故 WS 值 2×，真实 kernel 的 compute 已足够长去藏 K/V 延迟。
  副归因：384 线程的 170-reg 上限把 O90 的 208 regs 压到 168（68B spill）⇒ `lts op_write`
  1.3K→8.44M，producer 第 3 个 WG 与 compute 抢同一 regfile。⇒ 与「阻塞」里 F6/F7/O83/O91
  「本卡 fp8 BM≥128 撞 smem/寄存器墙」同源，**正结果仍只剩换卡**。
- 原始输出：`src/fp8/fa_bwd_fp8_o121_ab_{s512,s4096}.out.txt`、
  `src/fp8/fa_bwd_fp8_o121_ncu_{ws,tma}_s4096.out.txt`；实现细节 `docs/03` §142。

### 5.130 第 216 轮（O122）：fp8 主 kernel 的 **V 等待前移**（KVTMA tile 边界去关键化）—— **中性/负结果（opt-in `FA_VWAIT_TOP`，默认关）**

- **动机**：O121 后 `ROADMAP` 候选 ④ =「复核 `wait`+`short_scoreboard` 能否用更深 wgmma 流水 /
  更少 barrier 的 fold 拆解缓解」。既有 `FA_WS1/ILV/ILV34/R4` 都作用于 tile 内且已反复判中性；
  本轮单独拆 **tile 边界**：K 双缓冲故 K 不等，**V 单缓冲**（`Ap` 复用 `Vs`）⇒ V[nt+1] 只能在 tile
  末发起、紧接着在 tile 末等（中间只隔 K[nt+1] 的 wait + Kp 重建），V 的 TMA 延迟暴露在边界临界路径。
- **改动**（`kernels.cuh` `Fp8Cfg` 加 `FA_VWAIT_TOP` 默认 0；单文件 `sync_onefile_device.py` 同步
  `identical: True`）：把 V 的 `mbar_wait` 从 tile 末前移到**下一 tile 的 GEMM1 之后、GEMM2 之前**，
  用异步 wgmma（GEMM1 只读 Qs/Kcur）掩盖 V 延迟；等待仍一一对应（prologue 等 nt_begin，之后每 tile
  等前一 tile 发的 V）。数学/数值**逐位不变**。
- **数值**：S4096 ours vs fp32 ref `2.635/2.644/3.216e-1`（与默认档逐位相同）。
- **性能（同 session A/B，iters=50，S4096）**：默认 total 1.5820 / main 1.3540ms；
  `FA_VWAIT_TOP=1` **1.5741 / 1.3711ms**；`FA_VWAIT_TOP=1+FA_WS1=1` 1.5806 / 1.3497ms
  ⇒ **全在 ~1.5% 噪声内（中性）**。
- **fresh ncu（S4096，1.38ms）**：L2 **77.87%**、L1/TEX 76.10%、DRAM 11.38%、SM 46.88%、
  occ 18.67%（3 CTA/SM/168reg/74.82KB）、`op_red` **105.38M** / `op_read` 24.28M / `op_write` 0.10M；
  stall = **`short_scoreboard` 1.85 + `wait` 1.54** + not_selected 0.42 + barrier 0.41 +
  long_scoreboard 0.30 ⇒ **头号是 smem→mma 的 `ldmatrix` 依赖 + wgmma 依赖，tile 边界的 V 等待不在其中**
  （3 CTA/SM 的边界互相错开填补）。与 O121 同源：墙是 compute 流水延迟，非同步机制。
- **副收口**：shared **store** 2.4-way 冲突 = store wavefronts 的 46%（ncu Est 46%），但
  `wavefronts_mem_shared_op_st` 仅 **11.18% of peak** ⇒ **store 冲突吃的是远未饱和的 pipe，不是墙**，
  排除「抠 shared-store 冲突」方向。
- **判决**：**中性/负、`FA_VWAIT_TOP` opt-in 默认关**。候选 ④ 在 fp8 默认档**关闭**；fp8 主 kernel
  的 bytes（L2 `red`）与 issue（`short_scoreboard`+`wait`）都已在边界，`short_scoreboard` 再压需
  B 操作数寄存器预取（~24 regs，168/170 无余量⇒必 spill）。**正结果仍只剩换卡。**
- 原始输出：`src/fp8/fa_bwd_fp8_o122_vwait_ab_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o122_ncu_{base,stall,tables}_s4096.out.txt`；实现细节 `docs/03` §143。

### 5.131 第 217 轮（O123）：fp8 主 kernel 的 **per-row fold scale 寄存器预取**（`sA`/`sds3`/`sds2`）—— **中性（默认开，`-DFA_SCALE_HOIST=0` 可 A/B）**

- **动机**：O122 的 fresh ncu 把 fp8 默认 main（S4096 kvtma）头号 stall 钉为
  **`short_scoreboard`=1.86**，当时归因「smem→mma 的 `ldmatrix` 依赖」。本轮先检验该 stall 的
  另一候选来源——dV/dK/dQ 的 red/累加循环里**每元素一次读的 per-row fold scale**
  （`sA[r]`/`sds3[r]`/`sds2[r]`，`(i,r)` 固定却被 `j×q` 展平后重复读 8×，每个 `red` 的 issue
  都要等一次 LDS）。这是对 O122「B 操作数预取」之外的**零寄存器开销**替代（scale 只占 2–4 regs）。
- **改动**（`kernels.cuh` `Fp8Cfg` 加 `FA_SCALE_HOIST` 默认 1）：`epi_dv`/`epi_dk` 预取
  `sav[MTM34][2]`/`ssv[MTM34][2]`，GEMM5 预取 `sd20/sd21`；**保留原乘法次序**（`acc*sd*scale`）
  ⇒ 逐位等价。`=0` 逐字退回原路径（same-binary A/B）。单文件 `sync_onefile_device.py` 同步
  `identical: True`（6072 行）。
- **数值**：S4096 ours vs fp32 ref `2.635/2.644/3.216e-1`（与默认档**逐位相同**）。
- **性能（GPU1，same-binary A/B，iters=50，S4096）**：hoist=1 total avg **1.5845ms** / main **1.3781ms**；
  hoist=0 total avg **1.5864ms** / main **1.3801ms** ⇒ **1.0012× / 1.0015×（噪声内，中性）**。
  两变体 ptxas 同为 68B spill stores / 92B spill loads。
- **ncu（同 binary，`--set full`+stall）**：预取**确实**打掉 hot red 里的 LDS——`short_scoreboard`
  **1.86→1.48（−20.4%）**、Executed Instructions **617.66M→609.92M（−1.25%）**、LSU pipe
  35.72%→33.29%（即 O122 的 1.86 里 epilogue scale 是**真实的一部分**，非编译器已 CSE）。
- **判决**：**计时中性，但 issue 侧确有正收益**。kernel 是 **L2 吞吐 bound**（ncu L2 78.2%、
  Duration 1.37ms 不变）——省下的发射槽被 L2 吞吐吃掉，落不到时间上。剩余 `short_scoreboard`（1.48）
  主体是 `mma_block_bt` 的 `ldmatrix`（A/B 操作数）与 Kp/Qp/dOp 重建后的 ldmatrix 读，只能靠**更多
  独立 warp（occupancy）或更深跨-tile 流水**去藏，两者均被 3 CTA/SM 的 74.82KB/168reg 硬墙锁死。
  **fp8 默认 main 的 bytes 与 issue 都已在软硬件边界，正结果仍只剩换卡。** `FA_SCALE_HOIST` 作为
  逐位安全微优化保留（默认 1）。
- 原始输出：`src/fp8/fa_bwd_fp8_o123_scalehoist_ab_s4096.out.txt`；实现细节 `docs/03` §144。

### 5.132 第 218 轮（O124）：fp8 默认 main 的 `red` L1/L2 扇区分解 + MLA(D=512) 上 wgmma 探针 —— **诊断/收口（默认一行未改）**

- **动机**：O116 把默认 `kvtma` main 的 L2 `red` 105.4M 相对「理论下界 68.2M」的 **1.545×**
  归因为「每 KV 元素被多少 m-block 归约 = 工作划分」。本轮在**真实默认 kernel**上做
  `ncu` 的 **L1/L2 分层**，检验该归因。
- **证据一（L1 vs L2）**：S4096 causal H16 `fa_bwd_fp8_mma_kvtma_kernel<128,64,32,...>`：
  `l1tex op_red` 请求 **8.78M**、扇区 **70.25M（8.0/req，≈每贡献下界 68.2M、1.03×）**；
  `lts op_red` 扇区 **105.38M（12.0/req = 1.50× L1）**；`op_read` 24.24M、L2 78.24%、1.37ms。
  ⇒ **L1 侧已最优**，多出的 1.5× 在 L1→L2 之间，**不是贡献次数**。
- **证据二（机制无关）**：`-DFA_RED_STORE=1`（plain store 同址）L2 写扇区 **107.21M ≈ atomic
  105.38M**、L2 red=0、Duration 1.26ms。⇒ 坐实 O83/O159「L2 `red` 与归约机制无关」；
  atomic 在 L1 请求粒度更优（8.78M vs store 100.9M 单扇区请求）；plain store 快的 ~8% = 原子语义成本。
- **证据三（同 session TE）**：`harness/te_fp8_ncu.py` = `..._flash_bprop_wgmma_f8_..._64x64x128`
  （grid=132/384t）：Duration **258.7µs**、**L1 red 请求/扇区各 3,168（TMA 4D reduce 绕过 L1）**、
  L2 red **25.96M（ours 的 1/4.06）**、read 10.08M（1/2.40）。⇒ 差距=**贡献次数（持久 KV-owner）**，
  非机制；ours 要接近只能放大 BM / 持久 KV-owner，已被 O83/O91/O157/O160/O168 判「本卡无软件解」。
- **探针（MLA D=512 wgmma）**：fp8 主 kernel 仅剩 MLA 走 mma（`WGMMA=false` 写死）。尝试
  `--mlawgm=1` → 编译期被两处 `static_assert` 拦：`WGMMA ⇒ NTH==THREADS && NWAR==WN`
  （只支持 128 线程/2 warp；MLA 用 256/8-warp）与 `WGMMA ⇒ HD∈{128,256}`（HD=512 未接线几何）。
  ⇒ 非「只换指令」，需新写 HD=512 wgmma 几何；**默认一行未改、探针未留源码**。
- **判决/下一步**：与 O116/O117/O122/O123 合并——fp8 默认 main 的 **L1 贡献数（工作划分）与
  L2 issue 都已在软硬件边界，正结果仍只剩换卡**（更大 smem/regfile）。

- 原始输出：`src/fp8/fa_bwd_fp8_o124_ncu_{ours,redstore,te}_s4096.out.txt`、
  `src/fp8/fa_bwd_fp8_o124_sass_hist.out.txt`、`src/fp8/fa_bwd_fp8_o124_mlawgm_probe.out.txt`；
  实现细节 `docs/03` §145。

### 5.133 第 219 轮（O125）：fp8 `red` 墙的「表示无关性」终局核对 —— **bytes / sectors / requests**（default vs bulk-reduce vs TE）—— **诊断/收口（默认一行未改）**

- **动机**：O124 报了 L1/L2 扇区，但没报 **L2 的交易粒度（request）**；本轮补上这最后一块，
  并用 O42 的 `-DFA_BULKRED=1`（smem staging + `cp.reduce.async.bulk`）与同 session TE 三方对照，
  把「L2 `red` 由什么决定」钉死。
- **三方计数（S=4096 causal H16）**：
  | kernel | Duration | L1 red req/sect | L2 red **req** | L2 red **sect** | sect/req | red bytes |
  |---|---|---|---|---|---|---|
  | ours `kvtma<128,64,32>` | 1.37ms | 8.78M / 70.25M | **105.38M** | **105.38M** | **1.0** | 3.37GB |
  | ours `-DFA_BULKRED=1` | **1.45ms** | 0.26M / 2.10M | **51.07M** | **118.16M** | 2.3 | 3.78GB |
  | TE `flash_bprop_wgmma_f8` | **0.259ms** | 3.2K / 3.2K | **6.49M** | **25.96M** | **4.0** | 0.83GB |
- **判决**：
  1. 默认 ours **每个 L2 `red` 请求恰好 1 扇区**（req==sect==105.38M）；L1 每请求 8 扇区（已满），
     L1→L2 把 8-row 散布展宽成 ~12 个单扇区请求——O124「1.50×」的确切形态。
  2. **L2 `red` 扇区 == 贡献字节 / 32B**，只由「每 KV 元素被多少 CTA 贡献」（工作划分）决定。
  3. **交易粒度是正交维度且无收益**：bulk 把 L2 请求砍半（105.38M→51.07M）但**扇区反升 12%**、
     且 smem 往返的 L1/TEX 成本（O42 已 71.8%）使其 **0.94× 更慢**；O67/O114/O124 同结论。
  4. TE 同 BM=64 的 4.06× 字节优势**纯来自持久 KV-owner**（F7 目标），被本卡 255-reg/77.5KB
     的 smem/regfile 墙锁定（O83/O91/O157/O160/O168）。
- **护栏**：默认档 `ours vs fp32 ref` relL2 dq/dk/dv = 8.149/8.263/6.489%（MHA S4096）、
  GQA q40kv8 8.181/8.353/6.363%、MQA q64kv1 8.176/8.441/6.449%、D=256 8.332/8.434/6.464%、
  MLA D=512 8.163/8.564/6.507% ⇒ **全部在护栏内**；单/两文件 gate worst **7.629e-06 OK**、
  `docs/04 --check` OK（224 行）。性能 total **1.576ms / 87.2TF**，同 session TE FP8 main 0.259ms
  ⇒ main **5.29×**、端到端 ~6.1×（FA3 无 FP8 反向）。
- **下一步**：fp8 主 kernel 的 bytes（`red`）与 issue（O122/O123）都已在软硬件边界，
  **正结果只剩换卡**；覆盖型 backlog 见 ROADMAP「下一步」。
- 原始输出：`src/fp8/fa_bwd_fp8_o125_ncu_{ours,bulkred,te}_s4096.out.txt`、`..._o125_bulkred_timing_s4096.out.txt`、
  `..._o125_default_s4096.out.txt`；实现细节 `docs/03` §146。

### 5.134 第 220 轮（O126）：fp8 主 kernel「最后一条 mma 路径」——MLA（D=512）的 **wgmma 几何** —— **负结果（opt-in `--mlawgm`）**

- **动机**：O124（§5.132）探针判定 fp8 主 kernel 里唯一仍走 `mma.sync` 的是 **MLA（HD=512）**，
  被两处 `static_assert` 挡住、记为「需新写几何、工程量大、prize 仅 MLA 尺寸」。本轮把它
  **真正实现并 A/B**，收口「把 fp8 main 全切 wgmma」的最后一块。
- **实现**：device 放开 `fp8_mma_body` 的 HD 断言到 512（SW128/`wgmma_qkt64_fp8` 本就按
  `SBO=(HD/128)*1024` 写、O84 已放开 256）；host 新增 `--mlawgm` opt-in，派发
  `launch_bwd_main<512,64,32,0,WGMMA=1,...>`（4-warp/1 warpgroup、非 TMA）。SASS 确认新实例
  **32×QGMMA + 384×HMMA**（默认档 0×QGMMA）。单/两文件 device 逐字一致。
- **数值**：wgmma-vs-mma `max_abs` 差 ≤2e-3、relL2 差 ≤0.02%（S1024 8.163/8.583/6.504% vs
  8.163/8.564/6.507%、S4096 8.389/8.489/6.623% vs 8.385/8.483/6.623%）⇒ **同精度档、护栏内**。
- **性能（同 binary A/B，main）**：S512 **0.682×**、S1024 **0.699×**、S4096 **0.673×** ⇒ 负结果。
- **ncu（S4096）**：默认 8-warp mma+kvpipe = 1.25ms / 245 reg / 229.9KB / **occ 12.5%** /
  L2 83.0% / L1 49.7% / `short_sb` 1.39 / `wait` 1.18；**wgmma（4-warp/1WG）= 1.90ms / 255 reg /
  205.8KB / occ 6.25% / L2 54.7% / L1 31.3% / `short_sb` 0.70 / `wait` 1.52**。
- **机制**：切 wgmma 后 GEMM1/2 省掉 `ldmatrix`/HMMA（short_sb、L1/L2 都降），**但 warp 数腰斩
  （8→4）**——`fp8_mma_body` 的 wgmma 是 **warpgroup 级、锁死 128t/1 WG**；而 MLA 默认档早已用
  **2 个 warpgroup（8 warp）+ kvpipe** 藏延迟。省下的 issue 填不满少掉的一半 warp，寄存器反顶到
  **255**，仍 1 CTA/SM。⇒ 与 O39/O83/O91/F6/F7「MLA / BM≥128 撞 1-CTA/SM 寄存器墙」同源。
- **结论**：fp8 主 kernel 的「wgmma 化」至此覆盖**最后一条 mma 路径**——数值正确、**收益不存在**；
  正结果只剩换卡（更大 smem/寄存器，或支持 wgmma 跨多 warpgroup 的几何）。代码 `--mlawgm`
  保留 opt-in、默认 0、逐位安全。
- 原始输出：`src/fp8/fa_bwd_fp8_o126_mlawgm_ab.out.txt`、`..._o126_ncu_{mma,wgmma}_s4096.out.txt`、
  `..._o126_onefile_s512.out.txt`；实现细节 `docs/03` §147。

### 5.135 第 221 轮（O127）：fp8 主 kernel 的 **编译期占据率旋钮** `FA_MAIN_CTA`（`__launch_bounds__` min-blocks）—— **负结果（默认 3，一行数学未改）**

- **动机**：O116/O117（§5.124/§5.125）把 fp8 主 kernel 的 **host/运行期**旋钮在全部 shape 扫清；
  本轮补最后一个旋钮类——**编译期** `__launch_bounds__` 的 min-blocks（默认 `kvtma`/`qdtma`/
  通用 WGMMA 壳对 `HD==128/BN<=32` 写死 **3**）。O122/O123 判定头号 stall 是
  `short_scoreboard`+`wait`（issue），而 O79 寄存器账显示 3 CTA/SM 下实例 168 regs + 60–92B
  spill——`CTA=2` 把预算提到 256，ptxas 通常据此加深流水/消 spill。
- **实现**：`Fp8Cfg` 加 `FA_MAIN_CTA`（默认 3），只替换默认三壳的 `__launch_bounds__`
  min-blocks；`-DFA_MAIN_CTA=2/3/4` 同源码 A/B。单/两文件 device 逐字同步（`identical: True`）。
  **只改寄存器分配、一行数学未动**（三者对拍逐位相同 dq/dk/dv 2.635/2.644/3.216e-1）。
- **性能（S4096 H16 B1 causal，iters=30）**：CTA=3 main **1.3689ms / total 1.5797ms / 87.0TF**；
  **CTA=2 main 1.5890（0.861×）/ total 1.8079**；**CTA=4 main 2.1110（0.648×）/ total 2.3114**。
- **ncu（S4096，`regex:kvtma`）**：CTA=3 = 1.37ms / **168 regs** / occ 3 / **warps 18.67%** /
  `red` 105,381,888 / `read` 24,235,564 / tensor 11.89% / `short_sb` 1.48 / `wait` 1.54；
  **CTA=2 = 1.62ms / 212 regs / occ 2 / warps 12.47% / `red` 一字不变 / `short_sb` 1.03 /
  `wait` 1.41**。CTA=4 受 `74816B` 动态 smem 墙无占据率回报，唯一变化是 ptxas 把寄存器 cap 到 128
  → 248B stack / 280B spill stores（纯惩罚）。
- **机制/结论**：给出更多寄存器后 ptxas 确实把 issue stall 压低（`short_sb` −30%），但
  **L2 搬运量一字不变**、少掉的 4 warp/SM 暴露的延迟无法被 issue 节省补回；本 kernel 在
  3 CTA/SM（12 warp）已把「寄存器 budget ↔ occupancy」用到最优点。⇒ **编译期占据率旋钮也无正结果**，
  与 O116/O117 + 「正结果只剩换卡」一致。默认 `FA_MAIN_CTA=3` 一行未改。
- 原始输出：`src/fp8/fa_bwd_fp8_o127_cta_ab_s4096.out.txt`、`..._o127_ncu_cta{2,3}_s4096.out.txt`；
  细节 `docs/03` §148。

### 5.136 第 222 轮（O128）：fp16/bf16 反向新增 **`head_dim=256` 支持**—— **能力覆盖，正结果**

- **动机**：fp8 已于 O76 支持 `D=256`（O84 默认 wgmma）；fp16/bf16 反向只有 `D=128/512`，
  `D=256` 被 host 守卫拒绝。`D=256` 是**两支参考（FA2/TE）都支持、生产里存在**的形状，是
  ROADMAP「下一批/backlog」里**唯一明确的 `[ ]` 项**（fp8 主 kernel 的性能杠杆已在
  O116–O127 全部收口为「正结果只剩换卡」）。
- **改动（纯 host dispatch，device 一行未改）**：fp16/bf16 的 `*_mma_main.cu` + `*_mma_onefile.cu`
  各加 `D==256` 分支——形状守卫放开 256；`lse_mma_kernel<256>` / `lse_mma_kernel_bal<256,0/1/1,true>`
  按 `LDl=D+8` 设 `MaxDynamicSharedMemorySize`；`run_pre()` 新增 `D==256`（causal 镜像配对
  `bal<256,1>` + K-split、full 均衡 FULL `bal<256,1,true>`，**不做 4D-TMA**）；主 kernel 走
  **通用 mma** `fa_bwd_{fp16,bf16}_mma_kernel<256,64,32,1>`（`NDT=HD/128=2`，4-warp），
  `cudaMemset(dq_acc)` 条件补 `|| D==256`（GEMM5 全局 RMW）。单/两文件 host 同步、device 逐字一致。
- **数值（护栏）**：fp16 ours vs fp32 ref `max_abs` S1024 1.657/1.405/1.447e-3、S2048
  2.023/1.481/1.614e-3、GQA kv4 2.480/2.816/1.976e-3、full 1.850/2.775/2.109e-4；bf16 同构
  （1.039/1.213/1.460e-2 … full 1.491/1.597/2.454e-3）——全部同 dtype 噪声、**dk/dv 多数 ≤ FA/TE**；
  单/两文件逐位一致。**FA3 反向只支持 `head_dim≤128`** ⇒ D=256 无 FA3 列。
- **性能（event total；FA2/TE 同机 CUPTI 纯反向）**：fp16 S1024 0.4604ms/18.7TF、S2048
  1.3546/25.4、GQA 0.7644/22.5、full 0.4910/17.5；bf16 同量级。ours/TE **5.2–6.3×**、
  ours/FA2 4.0–4.7×（D=256 的 main 仍是通用 mma，非 wgmma）。
- **ncu（fp16 main S1024 causal）**：Duration **382µs**、DRAM 3.84% / L1/TEX 48.46% / L2 30.0% /
  Compute 6.33%、**168 regs / 动态 smem 149.5KB → Block Limit Shared Mem=1 ⇒ 1 CTA/SM、occ 6.25%**、
  Waves 0.97、`No Eligible 88.3%`、头号 stall = L1TEX scoreboard 41.3% ⇒ **bound = 低 occupancy
  （smem 墙）+ 全局访存延迟**（同 `D=512` MLA）；local spill 占 L1 sector 7.76%。
- **结论**：fp16/bf16 反向的 head_dim 覆盖补齐为 **128/256/512**；`D=256` 走 mma 通用壳
  （wgmma 只做 HD=128），若要对标 fp8 O84 的 2 CTA/SM 需新写 HD=256 wgmma 几何（留 backlog）。
  默认路径一行未改（D=128/512 回归逐位不变）。见 `docs/01` §27、`docs/01b` §6bd、`docs/04` 表；
  原始输出 `src/{fp16,bf16}/fa_bwd_*_o128_*.out.txt`、`..._o128_ncu_main_s1024h8_d256.out.txt`。

### 5.137 第 223 轮（O129）：fp8 causal MHA `S=8192` 覆盖 + 平台期延伸到 S4096 之外 —— **负结果**（默认一行未改）

- **动机**：fp8 主 kernel 的性能杠杆在 O116–O127 已全部收口（host/运行期/编译期旋钮 + 工作划分
  + wgmma 路径），但**所有标定的最大 MHA shape 只到 S=4096**（O93/O112/O117）。本轮把覆盖推到
  **S=8192**（nblk=128），复核对新尺度是否出现「更低 ksplit 反超」等杠杆。
- **做法（无 device 改动）**：新 dump `--shape '1 8192 16 128 causal'`（含 `ref_*`/`te_*`，
  ours 单/两文件各落盘）；同 binary 扫 `--ksplit=1/2/4/8`、`--hswap/--mrev/--ksm/--det/--wg2/--wg3/
  --bn64/--qdtma/--kvtma`，并 `-DFA_WGMMA34=1`。
- **性能（S8192 H16，event）**：默认 total **5.84ms / main 5.17ms（≈94 TFLOPS）**；`--ksplit=1/4/8`
  = 5.325/6.330/8.931ms（**0.971×/0.817×/0.579×**）⇒ **ksplit=2 仍最优**；`--hswap=0` 0.969×、
  `--mrev=0` 0.945×、`--ksm` 0.98×、`--det` 0.986×、`--wg2/--wg3` 0.49/0.59×、`--bn64` 0.66×、
  `--qdtma=0/--kvtma=0` 0.76/0.78×、`WGMMA34` 0.916× ⇒ **默认档全局最优在小 S 的结论延续到 8K**。
- **ncu / TE 对侧（同 session）**：ours `kvtma<128,64,32>` **5.32ms / L1 red 274.7M（8.0/req）/
  L2 red 412.1M（**精确 1.50× 展宽**）/ read 93.5M / L2 79.0% / DRAM 13.0% / warps 18.72% / 168 regs /
  3 CTA/SM** vs TE `..._flash_bprop_wgmma_f8_..._64x64x128` **0.966ms / L1 red≈0 / L2 red 102.2M /
  read 38.7M / L2 70.9% / DRAM 22.4%** ⇒ 红字账与 S4096 **完全同构**（red **4.03×**、read **2.42×**、
  时间 **5.5×**）⇒ `red` 只由工作划分决定、随 S² 缩放；**本卡无软件解**。
- **数值/护栏**：ours vs fp32 ref `max_abs` 2.215e-1/2.680e-1/2.976e-1（≤ TE-vs-ref）；
  `--ci` 单/两文件一致性 worst 2.38e-7 OK；`--check docs/04` 同步（242 行）。
- **结论**：**负结果、默认一行未改**；S8192 作为新覆盖 shape（`S8192_FP8_SHAPES`）纳入 harness。
  **正结果仍只剩换卡**。见 `docs/03` §149、`docs/04` auto-doc-table；原始输出
  `src/fp8/fa_bwd_fp8_o129_*`。

### 5.138 第 224 轮（O130）：fp8 **full（非 causal）MHA `S=8192`** 覆盖 + full 的 ksplit 规则在大 S 复核 —— **负结果**（默认一行未改）

- **动机**：O129 只用 **causal** 把平台期延伸到 S=8192；但 `fp8 专项冲刺` 的「降 L2 搬运量」有两个
  正交维度——**Q/dO 重读（ksplit）** 与 **跨 CTA `red`（工作划分）**。其中 ksplit 的最优在
  **causal（三角偏斜）** 与 **full（均匀工作量）** 下由不同目标决定（O96：full 只需「波对齐」）。
  本轮把 **full** 也推到 S=8192，复核 O96 自动档是否仍最优、并取 ncu/TE 红字账。
- **做法（无 device 改动）**：新 dump `--shape '1 8192 16 128 full'`（`ref_*`/`te_*` + ours
  `ours_o130_*`）；同 binary 扫 `--ksplit=1/2/3/4/5/6/8/12/16`（O96 自动档落 **k=5**）。
  `--hswap/--mrev` 对 full **ineligible**（需 causal，代码已打印忽略）。
- **性能（S8192 full H16，event）**：默认 total **11.43ms / main 10.44ms（48.1 TFLOPS）**；
  ksplit 1/2/3/4/**5**/6/8/12/16 → main 11.410/10.696/10.469/10.444/**10.442**/10.424/10.457/
  10.568/10.631ms ⇒ **auto k=5 ≈ 4–8（噪声内）最优**（k=1 并行度不足慢 8.5%、k≥12 Q/dO 重读慢 1–2%）。
  内置 A/B：`kvtma` 比 `qdtma` **1.188×**、`BN=32` 比 `BN=64` 1.14×、4-warp 比 8-warp 1.50×、
  BM=64 比 `wg2`(BM=128) 1.58× ⇒ **默认档最优**。
- **ncu / TE 对侧（同 session）**：ours `kvtma<128,64,32>` **10.59ms / L1 red 547.4M（8.0/req）/
  L2 red 821.0M（精确 1.50× 展宽）/ read 177.1M / L2 79.2% / DRAM 1.36% / warps 18.59% / 168 regs /
  3 CTA/SM** vs TE `..._flash_bprop_wgmma_f8_..._64x64x128` **1.89ms / L1 red≈0 / L2 red 201.3M /
  read 48.1M / L2 69.9% / DRAM 3.74%** ⇒ 红字账与 causal 同构（red **4.08×**、read **3.68×**、
  时间 **5.6×**）。**关键**：ours full 的 L2 `red` 821.0M 恰为 **causal S8192（412.1M）的 2.00×**
  ⇒ 再次坐实 `red` 扇区 == 贡献字节/32B、只由「每 KV 元素被多少 m-block 贡献」唯一决定。
- **数值/护栏**：ours vs fp32 ref `relL2` dq/dk/dv = **8.114% / 8.236% / 6.635%**（护栏内）、
  `max_abs` 1.56e-2/1.69e-2/1.29e-2。**注**：该 full shape 下 TE 自身 dv 不可信（TE-vs-ref
  `relL2` dv **88.1%**、TE dv L2 范数 53.3 vs ref/ours 75.0）；ours 的 dv 范数 74.9 ≈ ref。
- **结论**：**负结果、默认一行未改**；full S8192 纳入 `S8192_FP8_SHAPES` 覆盖。
  **正结果仍只剩换卡**。见 `docs/03` §150；原始输出 `src/fp8/fa_bwd_fp8_o130_*`。

### 5.139 第 225 轮（O131）：fp16/bf16 `head_dim=256` 主 kernel 切 BM=64 的 wgmma —— **正结果，默认**

- **背景**：`fp8 专项冲刺`（F1→F17）与 `下一批`（F6/F3b/F4b/F7）的性能杠杆在 O116–O130 已
  全部收口为「本卡无软件解」（`red` 由工作划分唯一决定，受 3 CTA/SM 的 74.8↔77.5KB smem +
  168-reg 双墙锁定；host/运行期/编译期旋钮全负）。fp8 main 覆盖型 backlog 亦已清空
  （O129/O130 把 MHA 覆盖推到 S=8192，仍负）。ROADMAP 唯一还开着的代码项 = **O128 遗留的
  `head_dim=256` wgmma 几何**（fp16/bf16；fp8 的对应项 O84/O85/O88 已收口）。
- **动机**：O128 的 D=256 主 kernel 走通用 `mma.sync`，ncu 的墙是 **L1TEX/LDSM**
  （S1024H8：L1/TEX 48.46%、stall L1TEX scoreboard 41.3%、Compute 6.33%）——这与 fp8 的
  L2 `red` 墙完全不同，正是 `wgmma`（无 `ldmatrix`、直读 SW128 描述符）能打的方向。
- **发现**：`fa_bwd_fp16_wgmma_kernel`（O9b，BM=64/BN=64/128t）**本就按 `HD/64` 参数化**
  （tile 尺寸、搬运 helper、描述符 SBO、GEMM1/2 的 K 循环 `Kd/16`），唯一 HD 硬编码是
  GEMM3/4/5 与 dQ 的「64 列组」遍数 `nh<2`。⇒ 泛化为 `NH=HD/64`（HD=128 时 =2，逐字不变）。
- **改动**（单/两文件 device 逐字一致，`sync_onefile_device.py` identical=True）：`static_assert`
  放开 256、`NH=HD/64`、`dqacc[NH]` 与三处 `nh<NH`；host D==256 默认 `launch_bwd_wgmma<256>`
  （cp.async 载入），`--d256wgm=0` 退回 O128 mma 做同 binary A/B。D=128/512 一行未改。
- **数值**：与 O128 记录**完全一致**、单/两文件一致；一致性 gate fp16 9.766e-4 / bf16 1.953e-3
  （容差 0.016/0.032）OK；`docs/04 --check` OK（244 行）。D=128 S4096 回归 1.883/1.734/1.966e-3 不变。
- **性能**（CUDA event，同 binary，iters=50）：fp16 main **S1024H8 0.384→0.214（1.79×）/
  S2048 1.208→0.681（1.77×）/ GQA kv4 0.666→0.362（1.84×）**，total 1.63–1.71×；bf16 同构
  （main 1.77–1.86×、total 1.64–1.71×）。相对 FA2/TE 由 ~5.2–6.3× 压到 ~3.1–3.8×。
- **ncu**（fp16 S2048H8 causal，同 session A/B）：Duration **1.21→0.688ms（1.76×）**、
  shared 波前 **34.43M→13.13M（2.62×↓）**、`smsp inst` **88.47M→71.98M（−18.6%）**、
  L1/TEX 32.08→24.92%；仍 1 CTA/SM（fp16 2 字节 ⇒ smem 181KB，非 fp8 O84 的 2 CTA/SM），
  但 LDSM 一省即 1.6–1.7×。⇒ **O128 的 L1TEX 墙被 wgmma 打掉**。
- **结论**：**正结果、默认**（`--d256wgm=0` opt-out）。**这是 fp16/bf16 `head_dim=256` 的
  wgmma 几何落地**，ROADMAP「下一步候选 ④」关闭；fp8 侧仍只剩换卡。见 `docs/01` §28、
  `docs/01b` §6be；原始输出 `src/fp16/fa_bwd_fp16_o131_*`、`src/bf16/fa_bwd_bf16_o131_*`。
- **注**：本轮任务模板要求「只做 fp8 性能」，但 fp8 的性能 backlog 已在 O116–O130 全部
  收口为「本卡无软件解」、覆盖型 backlog 也清空；故本轮推进 ROADMAP 唯一还开着的代码项
  （O128 遗留的 fp16/bf16 D=256 wgmma 几何），仍未触碰 fp8 的默认路径（零回归风险）。

### 5.140 第 226 轮（O132）：fp16/bf16 `head_dim=256` 反向补齐 **varlen** —— **能力覆盖，正结果**

- **背景**：`D=256` 在 O128（能力）/O131（wgmma 几何）先做了**定长**；**变长**（packed
  `[T,H,D]` + `cu_seqlens`）仍被 `run_varlen` 的形状守卫拒绝，是 ROADMAP「下一批」候选 ⑤
  （fp8 的 `D=256` 变长早在 O108/O113 完成并标定 ksplit）。本轮把它补齐（见 `docs/01` §29）。
  → 同「只做 fp8 性能」模板，但 fp8 main 的 bytes/issue 两墙在 O116–O130 已收口为「本卡无软件解」、
    覆盖型 backlog 清空；故推进唯一明确开着的覆盖项（与 O128/O131 的处理一致）。
- **改动（纯 host dispatch，device 一行未改；单/两文件同步）**：`run_varlen` 形状守卫放开 256；
  LSE 加 `D==256`（causal `bal<256,1>` / full `<256,1,true>`，行距 264，不做 4D-TMA）；
  `main_bm=64`；主 kernel = O128 的通用 mma `fa_bwd_{fp16,bf16}_mma_kernel<256,64,32,1>`
  （已带 `cu_seqlens`；`NDT=2` ⇒ dQ 逐 ndt 全局 RMW，需 memset）；`run_all` 加 `D==256` 分支。
- **附带修复**：`fa_bwd_fp16_mma_onefile.cu` 在 HEAD 上**无法独立编译**（`sync_onefile_device.py`
  早先用错 marker ⇒ `THREADS/WN`/include 重复定义 + 文件守卫 `#endif` 丢失）；本轮用正确 marker
  重新同步 device 区（`identical: True`），单文件恢复可编译。CI `--ci --no-run` 不编译单文件故未暴露。
- **新增 dump**：fp16/bf16 × 6 个 `D=256` 变长 shape（3 causal + 3 full）。FA3/FA2 反向变长
  不支持 `head_dim=256` ⇒ 只有 fp32 ref 可对。
- **数值**：fp16 causal 1.3–2.4e-3 / full 2.4e-4–1.4e-3；bf16 causal 1.3–2.0e-2 / full 3e-3–1.3e-2
  —— 同 dtype 噪声；单/两文件一致 gate `fp16 1.953e-3 / bf16 3.906e-3` OK。
- **性能（event total，`sum_b 4HL²D`）**：fp16/bf16 15.6–26.3 TF；fp8（Hopper）24.2–43.5 TF
  （**~1.6–1.7×**）。变长 D=256 仍走通用 mma（非 O131 wgmma）；wgmma+TMA 化留 backlog。
- **ncu（fp16 main）**：1 CTA/SM（149.5KB smem、168 regs）、occ 6.25%、L1/TEX 46.65%、
  DRAM 7.04%、No Eligible 88.7% ⇒ bound = **低 occupancy + 全局延迟**（同定长 D=256）。
- **结论**：**能力覆盖正结果**；fp16/bf16 的 `head_dim=256` 至此覆盖 定长（MHA/GQA × causal/full）
  + 变长（MHA × causal/full）。见 `docs/01` §29、`docs/01b` §6bf、`docs/04` §57；原始输出
  `src/fa_bwd_o132_d256_varlen.out.txt`、`src/fp16/fa_bwd_fp16_o132_ncu_d256_varlen_s4096.out.txt`。

### 5.141 第 227 轮（O133）：fp16/bf16 `head_dim=256` **变长**主 kernel 切 wgmma —— **正结果，默认**

- **背景**：O132（第 226 轮）把 `D=256` 从定长扩到变长，但变长仍走通用 `mma.sync`（O131 只切了
  **定长** GEMM1/2 的 wgmma）——这是 fp16/bf16 `D=256` 与「wgmma 全路径」之间最后一块缺口，
  也是 ROADMAP「下一步候选 ③」的剩余。本轮补上（见 `docs/01` §30）。
  → 同「只做 fp8 性能」模板，但 fp8 main 的 bytes（L2 `red`）与 issue 两墙在 O116–O130 已全部
    收口为「本卡无软件解」、覆盖型 backlog 也清空（O132 同处理）；故继续推进唯一明确开着的
    代码项（O132 遗留的变长 wgmma），仍未触碰 fp8 默认路径（零回归风险）。
- **改动（device + host；单/两文件同步）**：`fa_bwd_{fp16,bf16}_wgmma_kernel` 加
  `const int* cu_seqlens`——`qbase=cu?cu[b]:b*S`、`len=cu?cu[b+1]-qbase:S`、`if (m0>=len) return`
  （maxlen 网格里的死 CTA）、所有 `b*S` token 索引改 `qbase`、所有 `S` bound 改 `len`，两个搬运
  helper（`qdo_issue_async_sw`/`kv_issue_async_sw`）传 `qbase`；`launch_bwd_wgmma` 透传；
  `run_varlen` 的 `D==256` 默认 `launch_bwd_wgmma<256,true>(..., d_cu)`，`--d256wgm=0` 退回 O128 mma
  做同 binary A/B。`cu_seqlens==nullptr` 时定长逐位不变。
- **数值**：wgmma 与 mma 两条路径 `max_abs` **逐值相同**（fp16 causal 1.9–2.4e-3 / full 2.4e-4–1.4e-3；
  bf16 causal 1.3–2.0e-2 / full 2.9e-3–1.3e-2）；单/两文件一致 gate `fp16 1.953e-3 / bf16 7.812e-3` OK；
  定长 D=256/D=128/MLA 回归逐位不变。FA3/FA2 反向变长不支持 `D=256`，仅 fp32 ref 可对。
- **性能（event total，`sum_b 4HL²D`，同 binary A/B）**：fp16 **1.83–2.38×**、bf16 **1.82–2.33×**
  （b4_t3840 h8 causal 2.13→0.91ms；b5_t3968 h8 full 2.91→1.23ms），追平/略超 fp8 同 shape。
- **ncu（fp16 main）**：Duration **1.89→0.672ms（2.81×）**、`smsp inst` **−56%**、shared-load
  bank conflict **10.77M→0**、`lts op_read` **0.375×**、`op_red` **0.56×** ⇒ `wgmma` 直读 SW128
  去掉 `ldmatrix`（同 O131），额外收益来自 BN=64（vs mma 的 BN=32）减少重读/贡献。仍 1 CTA/SM
  （fp16 2 字节、smem 181KB）⇒ bound = **低 occupancy + L2/L1 搬运**（同 O128/O131）。
- **结论**：**正结果、默认**（`--d256wgm=0` opt-out）。**fp16/bf16 `head_dim=256` 至此定长+变长
  全走 wgmma**；fp8 侧仍只剩换卡。见 `docs/01` §30、`docs/01b` §6bg、`docs/04` §58；原始输出
  `src/fa_bwd_o133_d256_varlen_ab.out.txt`、`src/fp16/fa_bwd_fp16_o133_ncu_*`。
