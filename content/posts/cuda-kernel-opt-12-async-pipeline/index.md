---
title: "CUDA 算子调优（十二）：异步拷贝与流水线 — cp.async 多级流水线把访存藏起来"
date: 2026-09-21
draft: false
weight: 12
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "cp.async", "流水线", "pipeline", "异步拷贝", "GEMM", "TMA", "系列"]
categories: ["算子开发"]
---

上一篇（{{< relref "cuda-kernel-opt-11-launch-occupancy" >}}）的结论是：当瓶颈是「等访存」（`long_scoreboard` 高）时，可以用更多 warp（occupancy）来盖住延迟；但第 09 篇的 GEMM 被寄存器卡在 25% occupancy，提不上去，于是 09 用了 **`cp.async` 双缓冲**把全局内存载入和计算重叠，拿到 51.5%。

这篇把双缓冲推进到 **N 级软流水线（software pipeline）**：让更多分块同时在飞，看看是不是级数越深越快。配套代码在 [`code/kernel-opt/12-async-pipeline/async_pipeline.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/12-async-pipeline/async_pipeline.cu)，原始输出与 ncu 数据同目录。硬件仍是 H100 80GB HBM3（132 SM，每 SM 65536 寄存器、228 KB smem）。

核心结论先放这里：

| 版本 | 级数 | 耗时 | 算力 | %fp32 峰值 | `long_scoreboard` |
|---|---|---|---|---|---|
| `sync` | 1（同步） | 0.5692 ms | 30.18 TFLOPS | 45.1% | 0.91 |
| `pipe2` | 2（双缓冲） | 0.5094 ms | 33.72 TFLOPS | 50.4% | 0.03 |
| `pipe3` | 3 | 0.4816 ms | 35.67 TFLOPS | 53.3% | 0.03 |
| `pipe4` | 4 | **0.4794 ms** | **35.84 TFLOPS** | **53.6%** | 0.03 |
| `pipe5` | 5 | 0.4805 ms | 35.75 TFLOPS | 53.4% | 0.03 |

**级数从 2 加到 3 有收益，到 4/5 就平了。** 为什么？往下看。

---

## 1. 为什么「同步载入」会把发射口堵住

第 09 篇的 `gemm_reg_vec` 主循环长这样（`gemm_sync` 与它等价）：

```
for k0 in 0..K step BK:
    load_tile_vec(...)        // 每个线程读一个 float4 到 smem
    __syncthreads()           // 等所有线程搬完
    compute_tile(...)         // 从 smem 读，做 64 次 FMA
    __syncthreads()           // 等算完，才能覆盖 smem
```

问题在于 `load_tile_vec` 是普通的全局读写：线程发出 `LDG` 后，**必须等数据真的落到寄存器**才能写进 smem。这段时间里这条 warp 停在 `long_scoreboard` 上。ncu 也证实了——`sync` 的 `long_scoreboard` 是 **0.91**（每条已发射指令平均等 0.91 个 cycle），是除 `not_selected` 外最大的 stall。

更关键的是：载入和计算在同一个 warp 里**串行**。搬数据时没人算，算的时候没人搬。要重叠，就得让「搬」不阻塞「算」。

---

## 2. `cp.async`：把「搬」变成不占寄存器的后台任务

`cp.async`（CUDA 11 / sm_80 起）是一条 **直接从全局内存拷到共享内存、不经过寄存器** 的指令：

```cpp
__pipeline_memcpy_async(&As[row][q*4], &A[gr*K + gc], 16);  // 16 字节异步拷贝
__pipeline_commit();                                        // 把这一批拷贝打包成一个 group
...
__pipeline_wait_prior(N);  // 等到只剩 N 个 group 在飞，即更早的都已落 smem
```

对比一下两条路径：

```
普通 LDG + STS：
  LDG ──(等 ~400cy)──► reg ──► STS          这段 warp 被 stall 住

cp.async：
  cp.async ─────────────► smem （后台 DMA，不占寄存器、不占发射口）
  只有 __pipeline_wait_prior 处才需要等
```

于是可以把循环改造成：**算第 i 块的同时，预取第 i+1 块**。这就是第 09 篇的双缓冲。

---

## 3. 从双缓冲到 N 级流水线

双缓冲只允许「一块在算、一块在飞」。如果访存延迟很长，仅靠预取一块可能还不够盖住。多级流水线就是开一个 **环形缓冲队列**，让 `STAGES-1` 个分块同时在飞：

```
STAGES = 4 的环形缓冲（A 分块示意，B 同理）

  smem 槽位:   [0]      [1]      [2]      [3]
              ┌────┐  ┌────┐  ┌────┐  ┌────┐
  时间 t:     │计算│  │在飞│  │在飞│  │在飞│   ← 1 个在算，3 个 cp.async 在飞
              └────┘  └────┘  └────┘  └────┘
  时间 t+1:   │新飞│  │计算│  │在飞│  │在飞│   ← 算完的槽位立刻补发更后面的块
              └────┘  └────┘  └────┘  └────┘

  主循环每轮：
    ① 给「STAGES-1 之后的那一块」发 cp.async 并 commit
    ② __pipeline_wait_prior(STAGES-1)  ← 只等当前要算的这块到齐
    ③ __syncthreads()
    ④ compute_tile(当前槽位)
    ⑤ __syncthreads()
    ⑥ 槽位前移
```

代码里就是模板参数 `STAGES` 控制缓冲深度（`async_pipeline.cu:141` 起的 `gemm_pipe<STAGES>`）：

```cpp
__shared__ float As[STAGES][BM][BK];
__shared__ float Bs[STAGES][BK][BN];

// 序言：先把前 STAGES-1 块排进队列
for (int s = 0; s < STAGES - 1; ++s)
    prefetch_tile(..., As[s], Bs[s], ..., s * BK);

int stage = 0;
for (int k0 = 0; k0 < K; k0 += BK) {
    const int next = k0 + (STAGES - 1) * BK;        // 提前 STAGES-1 块预取
    const int nslot = (stage + STAGES - 1) % STAGES;
    if (next < K) prefetch_tile(..., As[nslot], Bs[nslot], ..., next);
    else          __pipeline_commit();              // 尾块：空组，保持 group 计数整齐
    __pipeline_wait_prior(STAGES - 1);              // 当前块到齐，更晚的继续飞
    __syncthreads();
    compute_tile(As[stage], Bs[stage], acc);
    __syncthreads();
    stage = (stage + 1) % STAGES;
}
```

几个容易踩的细节：

- **`wait_prior(STAGES-1)` 的语义**是「最多允许 STAGES-1 个最近提交的 group 未完成」。因为当前要算的块是第 `STAGES-1` 新的，所以留 STAGES-1 个在飞正好。
- **尾块必须 commit 一个空 group**（`else` 分支）。若不 commit，group 总数变少，`wait_prior` 可能提前返回而当前块还没到齐——这是数据竞争，不是性能问题，务必注意。
- 每个 stage 的 A、B 分块各自 commit 成一个 group，`__syncthreads` 在 `wait_prior` 之后、compute 之前，保证全 block 看到一致的 smem。

---

## 4. 实测：级数越多越快吗？

`scripts/run.sh 12-async-pipeline/async_pipeline.cu`，M=N=K=2048（17.18 GFLOP），原始输出见 `async_pipeline.out.txt`：

| 版本 | 寄存器 | smem/block | occupancy | 耗时 | 算力 | %fp32 峰值 |
|---|---|---|---|---|---|---|
| `sync` | 118 | 8 KB | 25% | 0.5692 ms | 30.18 TFLOPS | 45.1% |
| `pipe2` | 125 | 16 KB | 25% | 0.5094 ms | 33.72 TFLOPS | 50.4% |
| `pipe3` | 127 | 24 KB | 25% | 0.4816 ms | 35.67 TFLOPS | 53.3% |
| `pipe4` | 127 | 32 KB | 25% | **0.4794 ms** | **35.84 TFLOPS** | **53.6%** |
| `pipe5` | 127 | 40 KB | 25% | 0.4805 ms | 35.75 TFLOPS | 53.4% |

（`nvcc -Xptxas -v`，fp32 峰值 66.9 TFLOPS；同口径 cuBLAS SGEMM 参考为 0.3386 ms / 50.73 TFLOPS / 75.8%，见 09 目录。）

读法：

- **同步 → 双缓冲**：45.1% → 50.4%，+5.3 个点。这一步是把 `long_scoreboard` 消掉。
- **双缓冲 → 三级**：50.4% → 53.3%，再 +2.9 个点。三级让「补发」更早，同步屏障的等待也变短。
- **三级以上**：`pipe4` 53.6%、`pipe5` 53.4%。**基本平了，再加深只多花 smem 不涨性能。**

到 `pipe4` 时，我们达到了同口径 cuBLAS 的 **70.6%**（35.84 / 50.73），比 09 的 `reg_db`（51.5%）又往前推了 2 个点。

---

## 5. ncu 证据：`long_scoreboard` 被干掉了，剩下的是别的

只看 TFLOPS 容易以为是「流水线越多越好」。ncu 的 stall 分解说明白了真正发生了什么（`ncu_stalls.out.txt`，单位：每条已发射指令平均 stall cycle）：

| stall reason | `sync` | `pipe2` | `pipe3` | `pipe4` |
|---|---|---|---|---|
| `long_scoreboard`（等全局/local 访存） | **0.91** | 0.03 | 0.03 | 0.03 |
| `barrier`（等 `__syncthreads`） | 0.50 | 0.53 | 0.41 | 0.33 |
| `not_selected`（warp 够多、抢发射口） | 2.29 | 2.01 | 1.97 | 2.06 |
| `wait`（等固定延迟） | 0.21 | 0.26 | 0.28 | 0.28 |
| `mio_throttle`（MIO 队列拥塞） | 0.20 | 0.08 | 0.12 | 0.12 |
| `short_scoreboard`（等 smem/常量） | 0.16 | 0.18 | 0.19 | 0.17 |
| **`issue_active`（发射口利用率）** | 56.8% | 67.8% | 70.3% | 70.9% |
| `sm__throughput` | 54.7% | 65.5% | 68.0% | 68.3% |
| 耗时 | 570.9 µs | 510.4 µs | 482.4 µs | 483.8 µs |

三点判断：

1. **`long_scoreboard` 从 0.91 掉到 0.03**：cp.async 一上，等全局内存的 stall 直接归零——这正是它要做的事。`issue_active` 也从 56.8% 提到 70.9%，发射口被填得更满。
2. **`pipe2` 之后收益递减的原因**：`long_scoreboard` 在 `pipe2` 时就已经接近 0，**再深的流水线没有任何访存延迟可隐藏了**。`pipe3` 额外的 +2.9 点来自 `barrier` 从 0.53 降到 0.41（更早补发让同步等待变短），之后 `barrier` 也触底，自然就平了。
3. **现在的头号 stall 是 `not_selected`（~2.0），这是个「好」信号**：warp 已经多到抢发射口，说明延迟被盖住了、瓶颈转移到了计算/发射本身。结合 SOL（`ncu_sol.out.txt`）：

| 指标 | `sync` | `pipe3` |
|---|---|---|
| Compute (SM) Throughput | 55.1% | **67.9%** |
| L1/TEX Cache Throughput | 46.0% | 45.4% |
| DRAM Throughput | 1.9% | 2.3% |
| L2 Hit Rate | 90.2% | 94.5% |
| Executed IPC | 2.28 | **2.81** |
| Achieved Occupancy | 23.9% | 22.4% |

注意 **DRAM 只有 2%**：2048³ 的 fp32 GEMM 数据量（50 MB）能大部分驻留 L2，瓶颈从来不是 HBM，而是 SM 自己的计算/L1 管道。所以「继续堆级数」是在解一个已经不存在的问题。想把 53.6% 再往上推，得动内层计算（更大 thread tile、更宽的数据类型）或真正换硬件单元（Tensor Core），而不是继续加深流水线。

---

## 6. 更进一步：TMA 与生产者-消费者

`cp.async` 是 **sm_80 时代**的做法：每个线程各发一条 16 字节的拷贝，地址计算、group 管理都由线程自己做。到了 Hopper（sm_90），有了更彻底的方案 **TMA（Tensor Memory Accelerator）**：

```
cp.async（本例）                           TMA（cp.async.bulk.tensor）
─────────────────────────────            ─────────────────────────────
每线程发 16B 拷贝                          单个线程发一条「张量描述符」拷贝
地址计算消耗线程的指令                      地址/边界由 TMA 硬件生成
用 __pipeline group 管理完成             用 mbarrier 管理完成
拷贝 = 线程的额外工作                      拷贝 = 硬件 DMA，线程几乎零成本
适合 sm_80 兼容 / 小分块                    适合 Hopper+ / 大分块 / warp specialization
```

在这条演进线上，真正的 **producer-consumer（warp specialization）** 是这样的：把 block 内的 warp 分成两组——**producer warp** 只负责用 TMA 发拷贝、等 mbarrier；**consumer warp** 只负责从 smem 计算。两组通过 mbarrier / named barrier 解耦，谁都不干对方的活。可以把它理解成：

```
  生产者 warp ──TMA──► smem 环形缓冲 ──► 消费者 warp
       │                                     │
       └──────── mbarrier 同步 ──────────────┘
```

本篇的 `gemm_pipe<STAGES>` 还属于**「自我消费者」模型**：同一批 warp 又搬又算，只是用异步 group 队列把两件事在时间上错开。这已经拿到了大部分收益（`long_scoreboard`≈0）；warp specialization 的额外好处主要在更细的调度自由度和更少寄存占用，是把 53.6% 继续往上推的下一块拼图，也是后续「Tensor Core GEMM」篇要用的地基。

---

## 小结

- **`cp.async` 把全局内存载入从「占寄存器的阻塞读」变成「后台 DMA」**：`long_scoreboard` 从 0.91 降到 0.03，发射口利用率 56.8% → 70.9%。
- **多级流水线有效但迅速饱和**：`sync` 45.1% → `pipe2` 50.4% → `pipe3` 53.3%，到 `pipe4`（53.6%）/`pipe5`（53.4%）基本平——因为访存延迟在 `pipe2` 就已被完全隐藏，加深只剩 `barrier` 的小幅改善。
- **别被「occupancy」误导**：所有版本都是 25% occupancy（寄存器 118~127 限制），性能提升完全来自延时隐藏，不是并行度。
- **判断该不该继续加深**：看 `long_scoreboard` 是否已经接近 0；若已接近 0 而 `not_selected` 高，说明瓶颈已转移到发射/计算，继续加级数是白费 smem。
- 本实验最佳 **35.84 TFLOPS（53.6% 峰值）**，是同口径 cuBLAS（50.73 TFLOPS / 75.8%）的 **70.6%**。
- **下一步的方向**：不再堆流水线深度，而是换执行单元（Tensor Core）和调度模型（TMA + warp specialization）。

> 下一篇：CUDA 算子调优（十三）· Tensor Core 入门——从 `mma` PTX / WMMA 开始，用 `m16n8k16` 指令写一个能跑的 TC GEMM，看看 BF16 算力能从 fp32 的 ~54% 跳到多少。
