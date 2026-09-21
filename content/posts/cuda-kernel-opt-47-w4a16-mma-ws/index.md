---
title: "CUDA 算子调优（四十七）：给 W4A16 小 M 的 mma_b16 叠 warp specialization——一个负结果，和真正的墙（反量化指令 + KSPLIT 并行度）"
date: 2026-09-22T04:20:00+08:00
draft: false
weight: 47
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "decode", "量化", "W4A16", "int4", "AWQ", "GPTQ", "warp-specialization", "mbarrier", "mma.sync", "tensor-core", "KSPLIT", "Qwen3", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」量化推理方向的第七篇。承接
[第 40 篇 W4A16 dequant-GEMM]({{< relref "cuda-kernel-opt-40-w4a16-gemm" >}})、
[第 41 篇 warp specialization]({{< relref "cuda-kernel-opt-41-w4a16-warp-specialization" >}})、
[第 43 篇跨 item 持久化流水]({{< relref "cuda-kernel-opt-43-w4a16-persist" >}})、
[第 45 篇 W4A8+dp4a]({{< relref "cuda-kernel-opt-45-w4a8-dp4a-gemv" >}}) 和
[第 46 篇小 M 分派]({{< relref "cuda-kernel-opt-46-w4a16-smallm-dispatch" >}})。

46 篇把 `M∈[3,16]` 这个「小 batch decode」区间的最优交给了 `mma.sync.m16n8k16` 的
BM=16 小 tile（`mma_b16_k8`）。但它 M=16 时只有 **22.6% HBM**，ncu 报
`Compute 47.7% / L1 40.3% / DRAM 22.6%`——三个数都没打满，典型的「延迟/发射受限」。
顺着 41 篇的成功经验，本篇的第一个假设很自然：

> **把反量化从主循环里拆出去，交给一个专门的 producer warpgroup，让 consumer
> 只管 `ldmatrix` + `mma`。反量化的 ALU 与张量核不就重叠了吗？**

结论先说：**这个假设错了**。同一份反量化搬到另一个 warpgroup，指令一条没少、
延迟反而暴露得更厉害，所有 WS 变体都比不加 WS 慢。真正把这条路径推快的是另一件
「朴素」的事：**把 K 切得更细（`KSPLIT=8→16`），用更多短命 CTA 去互相填延迟**。
在 `M∈[3,16]` 这一档，新配置相对 46 篇提升 **5–11%**，运行时自适应总分从
always-wgmma 的 1.19× 提到 **1.23×**。

---

## 一、先看墙：这个 kernel 根本不是张量核受限

运行环境同前作：kernel_lab 容器、H100 80GB HBM3（132 SM，HBM 3.35 TB/s，
bf16 TC 峰值 989 TFLOPS）。shape 仍取自 `/ssd/models/qwen3-8B/config.json`：
`hidden=5120, intermediate=17408, group_size=128`，对称 int4
（权重打包 `[N, K/2]` u8 + `[N, K/128]` fp32 scale）。默认 `N=17408（up/gate）,
K=5120`，权重 int4 **44.6 MB** + scale **2.8 MB**。

先把 46 篇最好的 `mma_b16_k8` 放到 ncu 下数一数指令（M=16，`--metrics` 取管线：

```text
kernel gemm_mma_w4a16_kernel<16,128,64,2,1,4,8>  grid (136,1,8)  block 128
  gpu__time_duration.sum                   71.36 us
  sm__pipe_tensor_op_hmma_cycles_active     7.29 %     <-- 张量核只忙 7%
  dram__throughput                         21.51 %
  lts__throughput (L2)                     46.02 %
  l1tex__throughput                        34.13 %
  sm__warps_active                         22.79 %     (4 CTA/SM × 128 线程)
  sm__inst_executed.sum                    30.60 M
    pipe_alu                               14.53 M  (47.5%)
    pipe_fma                                9.44 M  (30.8%)
    pipe_lsu                                2.30 M  ( 7.5%)
    pipe_tensor                             0.70 M  ( 2.3%)   <-- 真·张量核指令
  stalls: long_scoreboard 1.59 / wait 1.31 / short_scoreboard 0.85
          not_selected 0.79 / math_pipe_throttle 0.47
```

一句话：**这个 kernel 78% 的指令是反量化产生的 ALU + FMA，张量核指令只占 2.3%、
张量流水线活跃度只有 7.3%**。瓶颈是指令发射 + 全局访存延迟（`long_scoreboard`
最高），不是张量算力。

这直接解释了为什么 41 篇的 WS 只快 7%：**把一个不是瓶颈的东西搬走，不会变快**。
但这还不足以预测「搬到 producer 后会更慢」，我们真的写出来测了。

---

## 二、设计：producer 专职反量化，consumer 只做 mma

改法很小：把 46 篇 `gemm_mma_w4a16_kernel` 的每一轮主循环
（`load cp.async → 反量化写 Bs → ldmatrix + mma`）拆成两个 warpgroup，
用 `mbarrier` 的 full/empty 握手（完全复用 23/41 篇的协议）。

```text
              46 篇（单 WG 串行）                 47 篇（producer/consumer）
        ┌───────────────────────────┐      ┌───────────────────────────────┐
        │ 128 threads               │      │ producer WG (128)             │
 每轮:  │  cp.async W → Wraw        │      │   cp.async W → Wraw           │
        │  int4→bf16 → Bs   (ALU)   │      │   int4→bf16 → Bs   (ALU)      │
        │  ldmatrix A,Bs + mma      │      │   arrive(full)                │
        │  __syncthreads ×2         │      ├───────────────────────────────┤
        └───────────────────────────┘      │ consumer WG×N (128 each)      │
                                           │   wait(full)                  │
       ALU 与 mma 在同一条链上串行          │   ldmatrix A,Bs + mma         │
                                           │   arrive(empty)               │
                                           └───────────────────────────────┘
                                             ALU 与 mma 理论上可重叠
```

smem 布局、`ldmatrix` 地址、B 的 `[N][K]` 行主序（非转置 `ldmatrix.x4`）全部照抄
46 篇，保证和 baseline **逐位等价**（对拍相对误差 2.6e-3，与 46 一致）。
`producer` 先 `__pipeline_wait_prior`、`named_bar_sync(1, NPROD)` 保证本 WG 的
cp.async 与反量化都写完后才 `mbar_arrive(full)`；`empty` 的 count = 所有 consumer
线程数（23/41 篇的坑）。代码在 `code/kernel-opt/47-w4a16-mma-ws/w4a16_mma_ws.cu`。

---

## 三、实测：WS 全线更慢

同一进程、同 warmup/iters（`bench_ms(run,5,50)`），M=16：

| kernel（M=16） | 结构 | ms | TFLOPS | 相对 `mma_b16_k8` |
|---|---|---|---|---|
| `mma_b16_k8`（46 baseline） | 单 WG，KSPLIT=8 | **0.0752** | 39.2 | 1.00× |
| `ws_b16_s2c1k8` | 1 producer + **1** consumer，s2 | 0.1006 | 29.3 | **0.75×** |
| `ws_b16_s3c1k8` | 1 producer + 1 consumer，s3 | 0.0904 | 32.6 | 0.83× |
| `ws_b16_s3c2k8` | 1 producer + 2 consumer，s3 | 0.0901 | 32.7 | 0.83× |
| `ws_b16_s2c2k8` | 1 producer + 2 consumer，s2 | 0.0771 | 38.3 | **0.98×**（最好的 WS，仍慢） |

最好的 WS（1 producer + 2 consumer、2 级流水）也只是追平 baseline，其余全输。
这不是调参问题——把 `STAGES`、consumer 数、KSPLIT 都扫过，WS 没有一档赢。

**为什么？** 再上 ncu，把 WS 版（`ws_b16_s2c2k8`）和 baseline 并排：

| 指标（M=16） | `mma_b16_k8`（单 WG） | `ws_b16_s2c2k8`（WS） | 说明 |
|---|---|---|---|
| tensor pipe 活跃度 | 7.29% | **6.34%** | 张量核本来就闲，WS 没帮上 |
| `pipe_alu` | 14.53 M | 15.01 M | 反量化指令**一条没少** |
| `pipe_fma` | 9.44 M | 11.26 M | 反而更多（多了 barrier/地址指令） |
| `pipe_tensor` | 0.696 M | 0.696 M | 完全一样 |
| `inst_executed.sum` | 30.60 M | 33.33 M | **总指令多了 9%**（握手开销） |
| stall `long_scoreboard` | **1.77** | **10.19** | 生产者 warp 赤裸裸地等全局访存 |
| `sm__warps_active` | 22.8% | 50.6% | warp 是变多了，但都在等 |

关键的一行是 `long_scoreboard`：**1.77 → 10.19（5.8×）**。在单 WG 版本里，
反量化 warp 在等 `__ldg` scale / 权重时，同一条 warp 上还有别的指令（mma 的
`ldmatrix`、地址计算）可以发射；拆成 producer 后，producer 的 128 个线程除了
「等 cp.async + 等 scale」之外**无事可做**，延迟无处藏身。消费者的 `mbar_wait`
也在空转。于是：

> **WS 把「反量化」这件本来就不该在这条路径上的工作，原封不动地搬到了一个
> 延迟暴露更严重的 warpgroup 里，还额外付了 9% 的握手指令税。**

这与 41 篇（wgmma 版 WS 只快 7%）是同一类教训、但这次是彻底的负结果。
判据很清楚：**先看 ncu 里目标工作占多少指令/多少 pipe**。如果它只占 2.3%
（tensor）而 78% 是你要「搬走」的 ALU/FMA，那 WS 不会凭空让 ALU 变少。

---

## 四、真正的杠杆：KSPLIT 并行度

既然墙是「指令数 + 访存延迟」，那就只能 (a) 减指令，(b) 用更多并行来藏延迟。
(a) 需要换数制（W4A8 + 整数张量核，见「下一篇」），本篇先把 (b) 做透。

46 篇已经发现 `KSPLIT` 有用（`mma_b16` 从 k1 的 0.1607 到 k8 的 0.0749，2.1×，
waves 0.26→2.06）。当时只扫到 k8。这次把 KSPLIT 一路加到 32：

| kernel | M=3 | M=4 | M=6 | M=8 | M=12 | M=16 |
|---|---|---|---|---|---|---|
| `mma_b16_k8` | 0.0683 | 0.0684 | 0.0702 | 0.0717 | 0.0729 | 0.0752 |
| `mma_b16_k12` | 0.0693 | 0.0696 | 0.0704 | 0.0709 | 0.0728 | 0.0751 |
| **`mma_b16_k16`** | 0.0638 | **0.0644** | **0.0657** | **0.0665** | **0.0692** | **0.0728** |
| `mma_b16_k20` | 0.0636 | 0.0644 | 0.0660 | 0.0669 | 0.0702 | 0.0744 |
| `mma_b16_k24` | 0.0645 | 0.0654 | 0.0674 | 0.0691 | 0.0733 | 0.0783 |
| `mma_b16_k32` | **0.0634** | 0.0644 | 0.0666 | 0.0689 | 0.0749 | 0.0820 |

（ms；同进程同 warmup。加粗为每列最优。）

规律很干净：

- **`M∈[3,16]` 的甜点是 `KSPLIT=16`**：`nblk=K/BKi=80`，k16 让每个 CTA 只算
  5 个 k-tile，网格 `136×16=2176` 个 CTA，waves 从 k8 的 2.06 升到 **4.12**。
- KSPLIT 再往上，单 CTA 的 k-tile 数（5→4→3…）少到被 prologue/drain 与
  `atomicAdd` 归约税吃掉，**M 越大越明显**（M=16：k16 0.0728 < k24 0.0783 <
  k32 0.0820）。
- 小 M（M=3–4）可以再贪一点到 k20/k32，但收益已是 1% 量级。

为什么加 waves 有用？因为这个 kernel 每个 CTA 的串行链是
`load → dequant → mma` 的 5~10 个 k-tile，链长而并发 CTA 数不变（4 CTA/SM 受
smem 限制）。**把每个 CTA 切短，就能让更多 CTA 同时在飞、互相填彼此的
`long_scoreboard` 空隙**——这正是 41 篇 wave quantization 的另一面：
不是消尾波，而是增加「同时可发射的独立访存流」。

### 还顺手改了 M=24 的档

`mma_b16` 一遍只覆盖 16 行；M=24 要跑两遍。但**更宽的小 tile**（`mma_b32`，
一遍覆盖 32 行）配上 k16 后，M=24 反超了 wgmma：

| M=24 | ms | TFLOPS |
|---|---|---|
| `wgmma_m64_k5`（46 篇该档最优） | 0.0858 | 51.9 |
| **`mma_b32_k16`** | **0.0824** | **52.0** |
| `mma_b32_k8` | 0.0844 | 50.8 |

于是分派表从 46 篇的「`≤16` 用 `mma_b16_k8`，`≥17` 用 `wgmma`」细化为三段：

```text
M ≤ 2        → gemv（d=1，每遍最便宜）            0.0309 / 0.0377 ms
3 ≤ M ≤ 16   → mma_b16_k16（d=16，一遍覆盖 + 细 KSPLIT）   0.0638–0.0728
17 ≤ M ≤ ~25 → mma_b32_k16（d=32）                 0.0824 @ M=24
M ≥ ~26       → wgmma_m64_k5（d=64）               0.0862–0.0953
```

### 分派总分

| 策略 | 12 个 M 点求和 | 相对 always-wgmma |
|---|---|---|
| always GEMV | 3.956 ms | 3.90× |
| always mma（全用 `mma_b16_k8`） | 0.948 ms | 0.93× |
| always wgmma | 1.015 ms | 1.000 |
| 46 篇自适应 | 0.853 ms | 0.840 |
| **本篇自适应（k16 / b32k16 / wgmma）** | **0.825 ms** | **0.813** |

即相对 always-wgmma **1.23×**（46 篇是 1.19×）、always-GEMV **4.79×**。
收益主要来自 `M∈[3,16]` 那 6 个点各快 5–11%。

---

## 五、完整对比表（M=16）

口径同前作：`AI` 与带宽按权重最小搬运量（int4 44.6MB + scale 2.8MB = 47.3 MB），
HBM 峰值 3.35 TB/s。同进程纯读权重上界 `roof_g1056 = 0.0182 ms / 2606 GB/s / 77.7%`。

| 版本 | 手段 | ms @M=16 | 有效权重带宽 | % HBM | ncu 结论 |
|---|---|---|---|---|---|
| 46 `mma_b16_k8` | 单 WG，`cp.async` + 主循环 dequant，KSPLIT=8 | 0.0752 | 629 GB/s | 18.8% | tensor 7.3%、ALU+FMA 78%、`long_scoreboard` 1.59、occ 22.8% |
| **47 `mma_b16_k16`（新最佳）** | 同上但 KSPLIT=16（waves 4.12） | **0.0728** | **650 GB/s** | **19.4%** | tensor 6.9%、`long_scoreboard` **1.77**、waves 4.12 |
| 47 `ws_b16_s2c2k8` | WS：producer 反量化 + 2 consumer mma | 0.0771 | 613 GB/s | 18.3% | tensor 6.3%、`long_scoreboard` **10.19**、总指令 +9% |
| 47 `ws_b16_s2c1k8` | WS：1 producer + 1 consumer | 0.1006 | 470 GB/s | 14.0% | 消费者饿死，最差 |
| 纯读上界 | 只读 W+scale | 0.0182 | 2606 GB/s | 77.7% | 与最佳差 **4.0×** |

---

## 六、小结

- **WS 不是「低比特 dequant」的答案**。这个 kernel 78% 的指令是反量化
  的 ALU+FMA，张量核只占 2.3%；把反量化搬到 producer WG 既没减少指令
  （ALU 15.1M→15.0M），也没藏住它的访存延迟（`long_scoreboard` 1.77→10.19），
  还多了 9% 的握手指令。**先量目标工作占多少指令/pipe，再决定要不要 WS。**
- **`long_scoreboard` 是判断「该不该拆 warpgroup」的指纹**：拆完如果它不降反升，
  说明你把「有伴可躲延迟」的 warp 拆成了「独自干等」的 warp。23/41 篇的 WS 之所以
  有效（或至少不亏），是因为被拆出去的是 TMA 那种「发射即返回」的异步操作，
  而不是需要等结果的反量化。
- **小 M 延迟受限时，KSPLIT 粒度是免费杠杆**：`k8→k16` 让 waves 2.06→4.12，
  `M∈[3,16]` 一致快 5–11%；再往上被 prologue/drain 与归约税吃回去。
- **更宽的 `mma_b32` tile 在 M=24 反超 wgmma**（0.0824 vs 0.0858），把
  46 篇的两段分派细化为三段。
- 距纯读上界仍有 **4×**——剩下的 gap 必须靠**减少反量化指令**，即换数制
  （W4A8 + 整数张量核 `mma.m16n8k32.s8`），这是路线图上真正的下一步。

配套代码：`code/kernel-opt/47-w4a16-mma-ws/w4a16_mma_ws.cu`
（含 46 篇 GEMV/wgmma 基线作同进程对拍、`mma_ws_kernel<BM,BN,BKi,STAGES,NCONS,KSPLIT>`
WS 变体、`M` 可覆盖便于 ncu 定点剖析）。原始输出见同目录
`sweep.out.txt`、`ncu_mma_b16k8_M16.out.txt`、`ncu_ws_b16_s2c2k8_M16.out.txt`、
`ncu_mma_b16_k16_M16.out.txt`。

## 下一篇预告

反量化指令占 78% 说明：**继续在访存/并行度上抠已经到头**。下一篇走 ROADMAP 上
挂了很久的一条路——**把 W4A16 换成 W4A8，用 Hopper 的整数张量核
`mma.m16n8k32.s8.s8.s32` 做整数点积**，让「反量化 + 乘加」直接进张量核，
把那份 78% 的 ALU/FMA 税一次性砍掉。45 篇已经用标量 `__dp4a` 在 M=1 验证过
方向（1.53×），这次是把它搬进小 M 的 GEMM。
