---
title: "CUDA 算子调优（六十三）：K2 的权重在显存里其实是连续的 —— 用 TMA 1D bulk 搬整 warp row-tile，把 FP4 decode FFN 推到 90% 权重 roofline"
date: 2026-09-22T13:40:00+08:00
draft: false
weight: 63
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "FP4", "e2m1", "TMA", "cp.async.bulk", "mbarrier", "DeepSeek-V4", "GEMV", "grouped", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 62 篇]({{< relref "cuda-kernel-opt-62-v4-fp4-ffn2" >}}) 把 K1 的解码从「每 byte 一次 LDS」
换成了两条 `prmt` 的整 chunk 位拼装，端到端做到 **4.968 ms / 81.1% 权重 roofline**。它留下的
判词里有一条当时没解：

```
K2 (w2, N=7168, K=3072):  DRAM 78.2%  L2 84.7%  occ 34.3%  long_scoreboard 40.5%
                          NITER = K/1024 = 3   ← 太短
                          Block Limit Shared Mem = 3（动态 smem 68.7KB，其中 cp.async staging 64KB）
```

K2 的 `RWW/DEPTH/PIPE` 几何扫参已经全负（见 62 篇 `sweep_B64.out.txt`），说明**得改结构**。
本篇做的事很小：**发现一个 warp 负责的那几行权重，在显存里本来是连续的一段**，于是用一条
`cp.async.bulk`（1D TMA）把整个 row-tile 一次搬进 smem，配一个 warp 私有的 `mbarrier`。

结论先放：K2 单独 **1.718 → 1.506 ms（1.14×）**，端到端 **4.979 → 4.487 ms（1.10×）**，
权重 roofline 从 **81.0% → 90.0%**；K1 也在同一轮里被一个「先发 TMA 再搬激活」的排序
顺手推到了 **92.2% HBM**。

---

## 一、K2 的权重，一个 warp 读的是连续内存

先把 K2 的访存模式摊开。K2 是 `w2 [N=7168, K=3072]` 的 grouped GEMV：一个 warp 负责连续的
`RWW` 行，沿 K 方向以 `16 B/lane` 扫。对 `RWW=4`：

```
专家 e 的权重 W_e 在显存里是 [N, K/2] 行主序（K/2 = 1536 B/行）：

  row_base+0 : |<---------------- 1536 B ---------------->|
  row_base+1 : |<---------------- 1536 B ---------------->|
  row_base+2 : |<---------------- 1536 B ---------------->|
  row_base+3 : |<---------------- 1536 B ---------------->|
               ^
               We + row_base*(K/2)  ← RWW*(K/2) = 6144 B 完全连续！
```

关键点：**行主序下，连续的几行 = 连续的一段字节**。62 的 cp.async 版本每个 warp 要发
`DEPTH(2) × RWW(8) = 16` 条 `cp.async`（每条 16 B、且要自己算 SW 地址），而 TMA 只需要：

```
lane0: mbarrier.arrive.expect_tx [mbar], WBYTES + SBYTES
       cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes \
            [wsm + warp*WBYTES], [We + row_base*(K/2)], WBYTES,      [mbar]
       cp.async.bulk... [ssm + warp*SBYTES], [Wse + row_base*NS32], SBYTES, [mbar]
其余 lane: mbarrier.try_wait.parity
```

scale 段同理：`Ws_e [N, K/32]`，连续 `RWW` 行也是连续的 `RWW*(K/32)` 字节。

> 一句话：**62 的 cp.async 是在「用 instruction 手工拼一个连续 DMA」，而 TMA 本来就是干这个的。**

### 1.1 一个容易踩的坑：mbarrier 的初始化顺序

每个 warp 私有自己的 `mbarrier`，`count=1`（只有 lane0 经由 `arrive.expect_tx` 到达）。
`mbarrier.init` 是 shared 写，必须对后续 `try_wait` 的线程可见。我最初的写法多加了一条
`__syncthreads()`：

```cuda
if (lane == 0) mbar_init(&mbar[warp], 1);
__syncthreads();            // ← 多余，K2 短 kernel 里实测值 ~3%
...
load activations
__syncthreads();
mbar_wait(...)
```

其实**不需要**：lane0 是「先 init 再 issue」，同线程程序序保证 init 早于 transaction；
其余 lane 在第二个 `__syncthreads()` 之后才 `wait`，而那条 barrier 本身就保证 lane0 的 init
已可见。删掉之后 K2 从 1.583 ms 回到 1.55 ms。

---

## 二、TMA 的延迟该藏在哪：一个排序问题

TMA 搬 6–12 KB 的延迟只能靠「等待期间有没有别的事做」来藏。最初我把顺序写成
「搬激活 → `__syncthreads` → 发 TMA → 等待」，于是**发完 TMA 立刻干等**。

改成：

```
lane0:  init mbar
lane0:  expect_tx + 发 2 条 bulk          ← 先把 DMA 发出去
all:    搬激活（xs0/xs1/sxs，全局读）      ← DMA 飞行期间干活
all:    __syncthreads()
all:    mbar_wait
```

效果**因 tile 大小而异**，这一点很有意思：

| | K1（tile 57 KB，NITER=7） | K2（tile 12 KB，NITER=3） |
|---|---|---|
| 激活在前，TMA 后发 | 3.126 ms（85.7%） | **1.530 ms（87.6%）** |
| **TMA 先发，激活在后** | **2.914 ms（92.0%）** | 1.551 ms（86.4%） |

K1 的 tile 大、激活也大（`GATHER` 按 token 随机读 7168 B×token），TMA 飞行期间有活干，白赚
**6.8%**；K2 的 tile 小、激活只 3 KB，遮不住多少，反而早发 TMA 会和激活抢一点带宽，稍退。
这是典型的「**优化是否有效取决于能不能把等待填满**」，和 47/48 篇 warp specialization 的教训同源。

---

## 三、结果

### 3.1 同一份二进制里 K2 的 head-to-head

`63-v4-fp4-ffn3/k2_tma.cu` 把两条 K2 路径放进同一个进程、同样 warmup/iters：

```
== K2 head-to-head (B=64, balanced 384 experts, m=1) ==
  pipe (cp.async RWW8 D2)  1.7182 ms  2614.4 GB/s (78.0%)
  tma  (row-tile RWW4 NW8) 1.5066 ms  2981.6 GB/s (88.9%)     ← 1.14×
[check pipe-vs-tma] max_rel=0.000e+00 OK
```

### 3.2 几何 sweep（K2）

```
== K2 sweep (baseline cp.async) ==
  PIPE RWW8 DEPTH2  1.7191 ms (77.9%)    PIPE RWW4 DEPTH2  1.8019 (74.4%)
  PIPE RWW8 DEPTH3  2.0020 (66.9%)       PIPE RWW4 DEPTH3  2.1515 (62.3%)
== K2 sweep (TMA row-tile) ==
  TMA  RWW8 NW8 MT1  1.5594 ms (85.9%)  smem=105KB
  TMA  RWW4 NW8 MT1  1.5119 ms (88.6%)  smem=54KB   ← late-TMA 最优
  TMA  RWW2 NW8 MT1  1.7743 (75.5%)     TMA RWW16 NW4  1.9255 (69.6%)
```

cp.async 版对几何极敏感（77.9 → 62.3%），TMA 版整个矩形都在 85% 以上——**TMA 把「手算
SW 地址 + 短流水」这个易碎环节从优化空间里拿掉了**。（注意：这张表是「激活在前」的 late 版；
开启「TMA 先发」后 K2 的最优几何变成 `RWW8 NW8`，见下。）

### 3.3 端到端（DeepSeek-V4-Pro 一步 decode，balanced 路由）

shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：`hidden=7168, moe_intermediate=3072,
n_routed_experts=384, topk=6, expert_dtype=fp4`；权重是 FP4 e2m1 + E8M0 block-32，384 专家
共 **13.48 GB**（packed + uint8 scale），roofline = 13.48/3352.3 = **4.020 ms**。

```
== per-stage (B=64, 同一二进制) ==
                          62 基线(cp.async)        63 TMA
  K1 gemv   3.2185 ms (83.3%)              2.9230 ms (91.7%)
  K2 gemv   1.7291 ms (77.5%)              1.5241 ms (87.9%)
  ---- sum  4.9650 ms (81.0% roof)         4.4651 ms (90.0% roof)   ← 1.110×
  end-to-end 4.9792 ms                     4.4874 ms                ← 1.110×
[check K1 GU / swiglu / K2 O / unperm / e2e]  全部 OK（err 1e-7 量级）
```

跨 batch（都是 balanced、零 padding）：

| B | m=路由复用 | 62 e2e (DEC=8) | 63 e2e (TMA) | 加速 |
|---|---|---|---|---|
| 64 | 1 | 4.9686 ms | **4.4874 ms** | **1.107×** |
| 128 | 2 | 5.6438 ms | **4.8309 ms** | **1.168×** |
| 256 | 4 | 6.8929 ms | **6.4640 ms** | **1.066×** |

B=128 提升最大（K2 79.4%），因为 `MTMAX=2` 让一次权重读被 2 个 token 复用，而 TMA 又消掉了
每个 stage 的地址计算。

---

## 四、ncu：墙从「staging / 延迟」挪到 DRAM

同一进程用 `ncu --set full` 抓 K1/K2 两个 TMA kernel（完整输出
`63-v4-fp4-ffn3/ncu_tma_k1k2.out.txt`）：

| 指标 | 62 基线 K2 | **63 TMA K2** | 62 基线 K1 | **63 TMA K1** |
|---|---|---|---|---|
| DRAM Throughput | 78.23% | **90.35%** | — | **92.24%** |
| L2 Cache Throughput | 84.68% | **90.90%** | — | **92.41%** |
| Compute (SM) | 68.16% | 79.58% | — | 83.77% |
| Achieved Occupancy | 34.3% | **24.4%** | 34.3% | 36.3% |
| Registers/thread | 62 | 64 | — | 72 |
| Dynamic smem | 68.7 KB | 107.7 KB | 68.4 KB | 68.4 KB |
| Block Limit (smem) | 3 | **2** | 3 | 3 |

最反直觉的一行是 **K2 的 occupancy 从 34% 掉到 24%**（smem 107.7 KB → 2 CTA/SM），
**但 DRAM 反而从 78% 涨到 90%**。为什么？因为 62 版的 `long_scoreboard` 占 stall 的 40.5%，
warps 都在等 `cp.async` 的依赖；TMA 把数据搬运从 lane 的 dependency 里摘出来，
`mbarrier` 的等待由「同一个 dependency scoreboard」变成「整个 warp 一起等一个事件」，
**延迟不再需要靠堆 warp 数来填**。这正好回答了 62 篇留下的「occ 34% 却只有 78%」：
瓶颈不是并发不够，而是搬运方式不对。

K1 现在是 **DRAM 92.2% / L2 92.4%**，离同进程纯读上界（8.46 GB @ **95.7% HBM**）只差 3.5 pp。

---

## 五、负结果与小坑

1. **TMA 的 `L2::cache_hint`（`createpolicy.fractional.L2::evict_first`）中性**。
   权重是流式读、本应 `evict_first`，但实测 K1 91.7%/K2 87.6%、sum 4.4721 ms，与不开
   hint 的 4.4651 ms 在噪声内（`ffn_B64_hint_negative.out.txt`）。L2 命中率本来就只有 ~19%，
   没有可保护的复用。

2. **几何不能乱扫到 `DEPTH > NITER`**。62 的 `issue(s)` 对 `s ≥ NITER` 会算出
   `(lane + 32*s)*16 > K/2` 的行外偏移；扫参 `DEPTH=4 > NITER=3` 直接 illegal memory access。
   本篇在 `issue` 里加了 `ss = min(s, NITER-1)` 兜底。

3. **`RWW` 不是越大越好**。`RWW=16` 的 tile 到 105 KB、`RWW=32` 到 207 KB，前者 69.6%、
   后者崩到 18%。tile 必须在「一次搬得够多」和「smem 别把 CTA 数压死」之间取平衡。

4. **短 kernel 里多一条 `__syncthreads` 值 ~3%**（见 §1.1），别把「保险起见」的同步留在热路径上。

---

## 六、小结

- **「连续行即连续内存」是 TMA 1D bulk 的入场券**：grouped GEMV 一个 warp 负责的 `RWW` 行
  行主序天然连续，`RWW*(K/2)` 字节一条 `cp.async.bulk` 搬完，配 warp 私有 `mbarrier`。
  每 warp 的搬运指令从 `16` 条 `cp.async` 降到 `2` 条 bulk。
- **端到端 4.979 → 4.487 ms（1.10×）**，权重 roofline **81.0% → 90.0%**；K2 单独 1.14×；
  B=128 达 **1.17×**；K1 经「TMA 先发、激活后搬」的排序推到 **92.2% HBM**。
- **TMA 的收益是「把 latency 从 dependency 里拿出来」**，所以它能让 occupancy **下降**而 DRAM
  利用率**上升**——`long_scoreboard` 高的 kernel，先想搬运方式，别先想堆 warp。
- **排序决定成败**：TMA 飞行期间有没有活干，决定了「先发 TMA」是 +6.8%（K1）还是 −1%（K2）。
- **判据**：搬运的是一段「连续、够大、行数固定」的字节 → 用 1D `cp.async.bulk`；
  否则才回去手工 `cp.async`。

**下一篇候选**：① K1 已到 92.2%，L2 92.4% 是下一道墙——试 `cluster multicast` / 更大的
row-span（一个 CTA 覆盖更多行）；② 把这条「连续行 → 1D bulk」推广到 34/30 篇的 prefill
grouped GEMM（A tile 也连续）；③ B=256 的 K1 只有 63.7%：`MTMAX=4` 的激活 smem 把 occupancy
压到 2 CTA/SM，试「激活分片/流式复用」。

---

**配套代码**：`code/kernel-opt/63-v4-fp4-ffn3/`

- `fp4_ffn3.cu` —— 主实验：`KTMA=0/1/2` 切 cp.async/TMA，`EK` 位控「TMA 先发」，
  `HINT` 控 L2 hint；内建 CPU 参考对拍 + 分段计时 + `sweep`/`tsweep`
- `k2_tma.cu` —— K2 单算子的同进程 head-to-head 与几何 sweep
- `ffn_B64_128_256.out.txt`、`ffn_B64_baseline_pipe.out.txt`、`tsweep.out.txt`
- `k2_tma_B64_headtohead.out.txt`、`k2_tma_sweep.out.txt`
- `ncu_tma_k1k2.out.txt`、`ffn_B64_hint_negative.out.txt`

运行：

```bash
cd code/kernel-opt
scripts/run.sh 63-v4-fp4-ffn3/fp4_ffn3.cu 64 1 8 none 1 0 3   # B=64, DEC=8, KTMA=1, HINT=0, EK=3
scripts/run.sh 63-v4-fp4-ffn3/fp4_ffn3.cu 64 1 8 none 0 0 3   # 同二进制 cp.async 基线
scripts/run.sh 63-v4-fp4-ffn3/fp4_ffn3.cu 64 1 8 tsweep       # TMA 几何 sweep
```
