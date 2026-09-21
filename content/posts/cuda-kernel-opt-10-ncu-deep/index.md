---
title: "CUDA 算子调优（十）：ncu 深潜 — occupancy、warp stall、roofline 与 source/SASS"
date: 2026-09-21
draft: false
weight: 10
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "ncu", "occupancy", "roofline", "warp stall", "SASS", "系列"]
categories: ["算子开发"]
---

上一篇结束时留了一个悬念：`gemm_reg` 被 128 个寄存器卡在 **25% occupancy**，`ncu` 又建议「降低驻留 warp 数可能更好」——到底该不该提高 occupancy？这一篇不写新算法，专门把 `ncu` 报告里最常看的四块读明白：

1. **Occupancy 到底怎么算**（手算一遍，和 `ncu` / 运行时 API 对账）；
2. **Warp stall reasons** 怎么读（五类 stall 指纹）；
3. **Roofline** 怎么用（Compute vs Memory 两个轴，别把 L1 管道当 HBM）；
4. **Source/SASS 对照**（把指标落到具体指令上）。

配套代码：[`code/kernel-opt/10-ncu-deep/ncu_deep.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/10-ncu-deep/ncu_deep.cu)。前一篇见 {{< relref "cuda-kernel-opt-09-gemm-advanced" >}}。

---

## 0. 实验设计：一组「只在一个维度上有特征」的样本

为了把每个概念拆干净，我写了几个刻意做对比的 kernel（H100，block=256，1024+ 个 block）：

| kernel | 干什么 | 预期特征 |
|---|---|---|
| `k_occ<1/2/4/8>` | 同一段 64 个独立累加器的计算体，用 `__launch_bounds__(256, N)` 请求不同驻留 block 数 | 寄存器 / occupancy / spill 此消彼长 |
| `k_latency` | 每个线程沿随机指针链走 256 步，地址依赖上一步 load | 长延迟 stall（long scoreboard） |
| `k_stream` | grid-stride `float4` 拷贝 128 MB | 压满 HBM，DRAM 顶线 |
| `k_smem_bar` | 每轮写 smem → `__syncthreads` → 读 → `__syncthreads` | barrier stall |

先看主机侧输出（`10-ncu-deep/ncu_deep.out.txt`）：

```
Device : NVIDIA H100 80GB HBM3
  SMs : 132   clock=1.980 GHz   HBM=3352.3 GB/s
  max threads/SM=2048   regs/SM=65536   smem/SM=233472 B
```

---

## 1. Occupancy：手算一遍，再和工具对账

`ncu` 的 Occupancy 小节只给结论（「limited by the number of required registers」）。真正看懂要会自己算。H100 每个 SM 的资源上限是：

```
寄存器   65536 / SM
线程     2048  / SM   (= 64 warp)
共享内存 228 KB / SM
block    32    / SM
```

**关键细节：寄存器按 warp 粒度分配，每个 warp 的寄存器数向上取整到 256 的倍数。** 于是：

$$
\text{warps}_{regs} = \left\lfloor \frac{65536}{\lceil \frac{r \times 32}{256} \rceil \times 256} \right\rfloor,
\qquad
\text{blocks} = \left\lfloor \frac{\min(\text{warps}_{regs}, 64)}{8} \right\rfloor
$$

（8 = 256 线程 / 32。）我在程序里用 `cudaFuncGetAttributes` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 把这条公式实现了一遍，跑出来的手算表：

```
kernel          regs  local  | blocks/SM: regs smem blk thr -> blocks(occ%, limit)
k_occ<1>         96    0     |   2   32   32   8  ->  2 (25.0%, registers)
k_occ<2>         96    0     |   2   32   32   8  ->  2 (25.0%, registers)
k_occ<4>         64    56    |   4   32   32   8  ->  4 (50.0%, registers)
k_occ<8>         32   344    |   8   32   32   8  ->  8 (100.0%, registers)
k_stream         14    0     |  16   32   32   8  ->  8 (100.0%, threads)
```

和 `ncu` 对账（`k_occ<2>`，`ncu_occ2.out.txt`）：

```
Block Limit Registers  block   2
Block Limit Shared Mem block  32
Theoretical Active Warps per SM  warp  16
Theoretical Occupancy  %      25
Achieved Occupancy     %      23.72
```

手算 2 个 block / 25%，`ncu` 也是 2 / 25%，achieved 23.7%（差的那点来自 kernel 收尾时的 wave 尾巴）。**结论：occupancy 不是玄学，就是这四个约束取最小值。**

注意 `k_occ<1>` 和 `<2>` 结果完全一样：`__launch_bounds__(256,1)` 只是「至少 1 个 block」，编译器自然用了 96 个寄存器，刚好能塞下 2 个。`<4>`/`<8>` 是**强制**编译器把寄存器压到 64/32 —— 代价马上来了。

### 1.1 强制提高 occupancy 反而慢 34 倍

同样是那段 64 个独立 FMA 链的计算体，只改 `__launch_bounds__` 的第二个参数：

| 版本 | 寄存器/线程 | 局部内存/线程 | 理论 occ | 实测耗时 | 相对 |
|---|---|---|---|---|---|
| `occ<1>` | 96 | 0 B | 25% | **10.75 ms** | 1.00× |
| `occ<2>` | 96 | 0 B | 25% | **10.75 ms** | 1.00× |
| `occ<4>` | 64 | 56 B | 50% | 54.46 ms | 5.06× |
| `occ<8>` | 32 | 344 B | 100% | **367.75 ms** | **34.2×** |

occupancy 从 25% 提到 100%，速度掉了 34 倍。多出来的寄存器放不下 64 个累加器，只能 **spill 到 local memory**（本质是显存），于是每次 FMA 前后夹一次 `STL`/`LDL`。用 `ncu` 量一下 local 流量（`ncu_spill.out.txt`）：

| 版本 | local load 指令 | local store 指令 | local 读字节 | local 写字节 |
|---|---|---|---|---|
| `occ<1>`/`<2>` | 0 | 0 | 0 | 0 |
| `occ<4>` | 1.52 G | 1.52 G | 194.7 GB | 194.7 GB |
| `occ<8>` | 10.22 G | 10.22 G | **1.31 TB** | **1.31 TB** |

`occ<8>` 光 local 读写就搬了 **2.6 TB**，指令数从 11.2 G 涨到 31.6 G（2.8×），`ncu` 的 stall 也换成了 LSU 排队：

```
k_occ<8>  lg_throttle    78.65   ← LSU 指令队列满（在发 spill load/store）
          long_scoreboard 40.68
          Memory Throughput 82.06%   DRAM 53.38%
```

**这就是第 09 篇那个问题的答案。** `gemm_reg` 的 25% occupancy 不是坏事：它 64 个累加器全在寄存器里，`occ<2>` 版本在 25% occupancy 下仍然把 FMA 管道压到 95.5%、IPC 3.94（上限 4.0）：

```
k_occ<2>  Compute (SM) Throughput 98.47%   Executed IPC 3.94   FMA pipe 95.5%
          stall 里唯一非零的是 not_selected 2.72 —— 有足够多 warp 在抢发射口
```

> 一句话：**occupancy 是手段不是目的，真正的目标是「发射口别空」。** 只要有足够的 ILP（独立指令），低 occupancy 也能喂饱计算单元；反过来，为了数字好看硬压寄存器，会把自己拖进 local memory 泥潭。

---

## 2. Warp stall reasons：一张「指纹表」

`ncu` 的 Warp State Statistics 会说「每个 warp 平均 X 个周期才发射一条指令」，但真正有用的是**它卡在哪**。我采集了 `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`（越大代表越多 warp 在等这件事）：

| kernel | 主 stall（ratio） | 次 stall | Compute % | Memory % | DRAM % | 结论 |
|---|---|---|---|---|---|---|
| `occ<2>` | not_selected 2.72 | — | **98.5** | 0.0 | 0.0 | 计算饱和，无需更多 warp |
| `occ<8>` | **lg_throttle 78.7** | long_sb 40.7 | 20.6 | 82.1 | 53.4 | LSU 被 spill 压垮 |
| `latency` | **long_scoreboard 798** | wait 2.0 | 1.8 | 97.4 | 0.1 | 依赖访存延迟 |
| `stream` | **long_scoreboard 359** | wait 2.7 | 4.6 | 84.0 | 84.0 | 带宽/延迟混合 |
| `smem_bar` | **barrier 17.4** | short_sb 7.9 / mio 5.5 | 92.0 | 79.0 | 0.0 | 被 `__syncthreads` 串住 |

怎么读这张表：

- **long_scoreboard**：等全局内存（L1TEX）返回数据。数值巨大（几百）说明访存延迟没被掩盖。`latency` 的 798 是因为地址依赖上一步结果，编译器**无法**把多个 load 重叠；`stream` 的 359 是典型带宽受限下的延迟排队。
- **short_scoreboard**：等 MIO 管道（shared memory / 特殊函数）。`smem_bar` 的 7.9 就是在等 smem 读。
- **mio_throttle / lg_throttle**：队列**满了**（不是等数据，是发不出去）。`lg_throttle` 78.7 是 spill 的直接证据。
- **barrier**：等同一 block 的兄弟 warp 到齐。`smem_bar` 17.4，被两次 `__syncthreads()` 串行化。
- **not_selected**：warp 已经就绪，但发射口被别的 warp 抢了。它非零通常是**好事**——说明 warp 够多、延迟被掩盖住了。
- **math_pipe_throttle**：计算管道排队。`occ<2>` 的 FMA 95.5% 但该项只有 0.03，因为它是靠 ILP 打满、不是靠堆积 warp。

> 经验：先看 `long_scoreboard`/`lg_throttle` 判断是不是访存问题；再看 `barrier`/`short_scoreboard` 判断同步与 smem；最后看 `not_selected`——**只有当它很小、发射口真的空着时，才考虑加 occupancy。**

---

## 3. Roofline：先搞清「Memory」是哪一级缓存

Roofline 用两个数把 kernel 分类：**算力**（FMA/CUDA core/Tensor core 利用率）和**带宽**（DRAM 利用率），中间那条折线的拐点在

$$
AI^\* = \frac{66.9\ \text{TFLOPS}}{3.352\ \text{TB/s}} \approx 20\ \text{FLOP/byte}
$$

算术强度 $AI$ 大于 20 就是 compute-bound，小于就是 memory-bound。把五个 kernel 放上去：

```
TFLOPS
 66.9 ┤█████████████████████████████████  ← FP32 计算屋顶
      │                              ● occ<2>  (64.4 TFLOPS, AI≈6e5, 96%)
      │                            ╱
      │                          ╱
      │                        ╱
      │                      ╱
      │                    ╱
      │                  ╱
      │                ╱
      │              ╱
      │            ╱                 ▲ smem_bar (memory 79% / compute 92%)
      │          ╱
      │        ╱
      │      ╱  ● stream (2.74 TB/s @ AI≈0)
      │    ╱  ● latency (0.1% DRAM！卡在 L1 延迟，不是带宽)
      │  ╱
    0 ┤╱────────────────────────────────────────►
      0      20              1e3          AI (FLOP/byte)
              ▲ 拐点
```

**这里有个大坑：`ncu` 的 "Memory Throughput" 是 L1/L2/DRAM 里最高的那一级的利用率，不一定是 HBM。** 看 `k_latency`：

```
latency : Memory Throughput 97.35%   DRAM Throughput 0.13%
```

看起来「内存打满了」，其实 97% 是 **L1TEX 管道**——256 步指针链的工作集只有 1 MB（n=2^20），全在 L2/L1 里，HBM 几乎没动。它真实的瓶颈是**访存延迟**（long_scoreboard 798），不是带宽。反过来 `k_stream` 的 84% Memory 和 84% DRAM 一致，才是真的 HBM 受限。

所以读 SOL 小节必须同时看 `L1/TEX Cache Throughput`、`L2 Cache Throughput`、`DRAM Throughput` 三个，别只信那个 "Memory Throughput"。

`ncu` 的 Roofline Chart 在 CLI 下不画图，只给一句判定：

```
INF  The workload achieved 95% of this device's FP32 peak ...
```

配合手算的 64.4 TFLOPS（96%）就能定位：`occ<2>` 已经贴着计算屋顶，唯一的提升方向是把重复计算砍掉或换 Tensor Core。

---

## 4. Source/SASS 对照：把 stall 落到具体指令

`ncu --page source --print-source cuda,sass` 能把指标关联到每一行源码/汇编。看 `k_latency` 的核心循环（`ncu_source.out.txt`）：

```sass
/* 50  for (int i = 0; i < steps; ++i) { */
0x...6b0   IMAD.WIDE R6, R17, 0x4, R12
0x...6c0   LDG.E.CONSTANT R2, desc[UR4][R6.64]     ← 发起 load
0x...6d0   IMAD.WIDE R16, R2, 0x4, R12             ← 用 R2 算下一个地址
0x...6e0   LDG.E.CONSTANT R26, desc[UR4][R16.64]   ← 下一次 load 依赖它
0x...6f0   IMAD.WIDE R18, R26, 0x4, R12
0x...700   LDG.E.CONSTANT R29, desc[UR4][R18.64]
...
```

读法一目了然：每条 `LDG` 的目标寄存器（`R2`）立刻被下一条 `IMAD.WIDE` 用来算地址，形成一条**串行依赖链**。硬件只能一个接一个等，`long_scoreboard` 高到 798 就是这么来的。这也解释了为什么 `k_latency` 的 occupancy 已经 96.8%，却依然慢——**加再多 warp 也无法把单条依赖链变短**，只能靠更多 warp 来互相填空（Fill）。

对照看 `k_occ<8>` 的 SASS，会出现大量 `STL`/`LDL`：

```sass
STL  [R1], R8          ← 寄存器放不下，写回 local memory
...
LDL  R8, [R1]          ← 下次用再从 local 读回来
```

`lg_throttle` / local 1.31 TB 的源头就在这两条指令。

> source/SASS 对照的正确用法：**先用 section 找到「哪种指令」是瓶颈（FMA？LDG？STL？barrier？），再用 source view 找到它在源码的哪一行、为什么编译器这么排。** 不要一上来就逐行读 SASS。

---

## 5. 回到第 09 篇：三条结论

1. **`gemm_reg` 的 25% occupancy 不是主要矛盾**。它的 stall 主要是 `not_selected`，说明 warp 够用；真正的天花板是 FMA 管道与 `cp.async` 流水深度，所以第 11、12 篇的方向是对的（`__launch_bounds__` 调优 + 更深流水），但不能盲目压寄存器。
2. **看 SOL 要分清缓存层级**：`Memory Throughput` 高 ≠ HBM 瓶颈，先看 `DRAM Throughput`。
3. **occupancy 是四个约束的最小值**：寄存器、smem、block、线程上限。想提它，先确认瓶颈真的是「发射口空着」（`not_selected` 很小 + `no_eligible` 高），否则可能换来 2.6 TB 的 spill 流量。

---

## 小结

- **occupancy 手算**：`min(寄存器, smem, block, 线程)`，寄存器按 warp 向上取整到 256 倍数；手算与 `ncu` 完全一致（`k_occ<2>` = 2 block / 25%）。
- **强制提 occupancy 的代价**：`occ<8>` 把寄存器从 96 压到 32，spill 出 **1.31 TB** local 流量，比 `occ<2>` 慢 **34 倍**；stall 从 `not_selected` 变成 `lg_throttle 78.7`。
- **stall 指纹**：long_scoreboard=访存延迟、short_scoreboard=smem、lg/mio_throttle=队列满、barrier=同步、not_selected=warp 够多（好事）。
- **roofline 拐点** AI\*≈20 FLOP/byte；`Memory Throughput` 是缓存层级的最大值，`k_latency` 的 97% 其实卡在 L1 延迟，DRAM 只有 0.13%。
- **source/SASS**：`LDG → IMAD.WIDE → LDG` 的依赖链一眼可见；`STL/LDL` 就是 spill。
- `occ<2>` 在 **25% occupancy** 下打到 **98.5% Compute / IPC 3.94 / FMA 95.5%**——低 occupancy 不等于慢。

> 下一篇：CUDA 算子调优（十一）· launch 配置与 occupancy：`__launch_bounds__` 怎么用、寄存器/共享内存如何取舍、循环展开与 ILP。
