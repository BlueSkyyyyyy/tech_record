---
title: "CUDA 算子调优（三）：正确测量 — 计时陷阱、有效带宽与 ncu 入门"
date: 2026-09-21
draft: false
weight: 3
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "ncu", "性能分析", "系列"]
categories: ["算子开发"]
---

优化算子之前，先要学会**正确地测量**。这听起来像废话，但新手最容易栽在这一步：同一段 kernel，计时方法不对，结论能差好几倍，于是「优化」变成了在噪声里瞎调。这一篇用三个小实验把最常见的坑摆出来，再介绍怎么用 `ncu` 看到 kernel 内部的真实指标。

配套代码：[`code/kernel-opt/03-measurement/timer_pitfalls.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/03-measurement/timer_pitfalls.cu)。前两篇见 {{< relref "cuda-kernel-opt-01-overview" >}}、{{< relref "cuda-kernel-opt-02-first-kernel" >}}。

---

## 0. 先看工具：三种「耗时」不是一回事

同一个 kernel，你会看到至少三个不同的数字：

| 量 | 含义 | 怎么测 |
|---|---|---|
| **墙钟 / wall** | 从 host 发起调用到 kernel 结束的总时间，**含 launch 与主机派发开销** | CUDA event 包住调用 |
| **纯 device 时间** | kernel 真正在 GPU 上执行的时间，**不含 launch** | `ncu` / CUPTI（如 `torch.profiler`） |
| **端到端** | 整个程序/整层的时间线，含拷贝、同步、气泡 | `nsys` |

用 CUDA event 测出来的是「墙钟」，对**单个** kernel 会严重高估（把 launch 算进去了）。这一篇的第一件事就是把 launch 开销量化出来。

## 1. 坑一：launch 开销不是零

写一个空的 kernel（什么也不做），连发 2000 次，用 CPU 计时：

```
=== A. launch overhead (empty kernel) ===
  2000 empty launches: host_issue=3.749 ms (1.874 us/launch), +sync=3.775 ms
  CUDA-event avg over 1000 launches: 1.886 us/launch (includes event overhead)
```

**每次 launch 约 1.9 微秒**。这个数字单独看不大，但它意味着：

- 如果一个小 kernel 只跑 2 微秒，你测到的「墙钟 4 微秒」里有一半是 launch；
- 如果有 1000 个小 kernel 串起来（比如逐元素的算子链），光 launch 就是 2 毫秒——这正是「算子融合」能省下的钱。

所以：**小 kernel 的墙钟时间几乎全是 launch**，要么用 CUDA Graph 合并、要么把多个算子融合成一个大 kernel（本系列后面会反复回到这个点）。

> 顺便说，`ncu --set full` 报告里的 `Duration` 是纯 device 时间。第一篇里提到 TE 的基准测试改用 CUPTI 测纯 device 时间，也是同一个道理。

## 2. 坑二：冷启动不是稳态

第一次调用一个 kernel，往往比稳态慢。原因很多：CUDA module 要加载、常量/纹理缓存要填充、context/lazy 初始化、显存页第一次触达等。

```
=== B. cold start vs steady state ===
  first (cold) launch : 0.025 ms
  steady state        : 0.009 ms  (3734.2 GB/s)
```

**第一次 25 微秒，稳态 9 微秒，差了 2.8 倍。** 如果只测一次就下结论，会得出完全错误的性能数字。所以任何基准测试都要**先 warmup 若干次，再对多次取平均/中位数**。本系列的公共工具 `bench_ms` 就是先 warmup 10 次、再测 50 次取平均。

## 3. 坑三：小数据会「住进」L2，带宽虚高

H100 有 **52 MB 的 L2 缓存**。如果反复搬同一份小数据，它会一直待在 L2 里，你测到的「带宽」是 L2 带宽，不是 HBM 带宽。用一个 copy kernel（读 in、写 out，工作集 = 2×数组）扫不同大小：

```
=== C. L2 residency (L2 = 52.4 MB) ===
        size    working set           ms         GB/s
        1 MB         2.0 MB       0.0025        810.2
        4 MB         8.0 MB       0.0036       2205.0
       16 MB        32.0 MB       0.0081       3954.1     ← 超过 HBM 峰值！
       32 MB        64.0 MB       0.0247       2595.2
       64 MB       128.0 MB       0.0458       2792.8
      256 MB       512.0 MB       0.1709       2995.0
     1024 MB      2048.0 MB       0.6748       3035.1
```

几个值得注意的点：

- 工作集 32 MB 时算出 **3954 GB/s，超过 HBM 峰值 3352 GB/s**——显然是假的，因为一半数据命中了 L2。这类「超峰值」数字不要当成优化成果。
- 工作集 64 MB 时反而掉到 2595 GB/s（L2 装不下了），之后随规模增大才慢慢爬回 ~3035 GB/s（真正的 HBM 效率，约 90%）。
- 1 MB 时只有 810 GB/s，不是带宽不行，而是**时间太短（2.5 微秒），被 launch 开销淹没**了。

**方法论**：判断 HBM 效率一定要用**远大于 L2（≥ 128 MB）**的工作集；小 shape 的漂亮数字要打问号。

## 4. 有效带宽 / FLOPs 怎么算

测到时间后，怎么换算成「离峰值还有多远」？记住数据的搬入搬出量。

- **有效带宽**：`GB/s = 搬运字节数 / 耗时`。copy kernel 读 N 字节、写 N 字节，共 `2N`；vector add 读 2N、写 N，共 `3N`。
- **有效算力**：`TFLOPS = FLOPs / 耗时`。GEMM 是 `2·M·N·K`；attention 前向约 `4·b·s²·h·d`。
- **算术强度 AI** = FLOPs / Bytes，用来判断 roofline 上落在哪一段。

换算示例：copy，工作集 2048 MB，0.6748 ms：

```
2 × 2.0 GB / 0.6748 ms = 4.0 GB / 0.0006748 s ≈ 3035 GB/s ≈ 90.5% of 3352 GB/s
```

## 5. ncu 入门：从「快慢」到「为什么」

CUDA event 只能告诉你「多快」，`ncu` 能告诉你「为什么」——它把 kernel 的硬件计数器采集下来，分成一节节的报告。用 `--section` 只打印关心的几节：

```bash
scripts/ncu.sh 03-measurement/timer_pitfalls.cu \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
    --kernel-name regex:copy_v4 --launch-count 1
```

针对 copy kernel（工作集 32 MB，正好在 L2 临界）的实测输出：

```
Section: GPU Speed Of Light Throughput
--------------------------------------- ----------- ------------
Metric Name                             Metric Unit Metric Value
--------------------------------------- ----------- ------------
DRAM Frequency                              Ghz         2.61
SM Frequency                                Ghz         1.98
Elapsed Cycles                             cycle        22878
Memory Throughput                             %        48.41
DRAM Throughput                               %        48.41
Duration                                      us        11.52
L1/TEX Cache Throughput                       %        26.14
L2 Cache Throughput                           %        72.03
SM Active Cycles                          cycle     17100.70
Compute (SM) Throughput                       %        15.54

Section: Memory Workload Analysis
--------------------------------------- ----------- ------------
Memory Throughput                        Tbyte/s         1.62
Mem Busy                                      %        38.59
Max Bandwidth                                 %        48.41
L1/TEX Hit Rate                               %            0
L2 Hit Rate                                    %        52.19

Section: Occupancy
------------------------------- ----------- ------------
Theoretical Occupancy                         %          100
Achieved Occupancy                            %        77.17
Achieved Active Warps Per SM                warp        49.39
```

怎么读这三节：

1. **Speed Of Light（SOL）**：一眼看瓶颈。这里 `Compute (SM) 15.5%` 很低、`DRAM 48.4%` 也不高，说明既没吃满算力也没吃满带宽——**有别的因素在拖后腿**（这里是数据小、L2 命中一半、延迟没盖住）。
2. **Memory Workload**：`L2 Hit Rate 52%` 解释了为什么 `DRAM Throughput` 只有 48%——一半数据从 L2 拿，DRAM 没被喂饱；`L1 Hit Rate 0%` 对纯流式 copy 是正常的。
3. **Occupancy**：`Theoretical 100%` vs `Achieved 77%`。理论占用率由寄存器/共享内存决定，达到 100% 说明启动配置没浪费资源；实际 77% 的差距来自 warp 调度开销或负载不均。

> 注意：ncu 会把 kernel 多次 replay 来采集不同计数器，因此**在 ncu 下程序自己打印的耗时会被拉长**，不能用 ncu 运行时的屏幕输出去做性能结论，只能看 ncu 报告的那些比率指标。测时间老老实实用 event/CUPTI，看内部指标才用 ncu。

## 6. 一份「正确测量」清单

写基准测试时照着检查：

- [ ] **先 warmup**（≥10 次），再测多次取平均/中位数，别用单次。
- [ ] **区分墙钟和 device 时间**：小 kernel 的墙钟被 launch 支配。
- [ ] **工作集要远大于 L2**（≥128 MB）才是真实 HBM 带宽；小数据「超峰值」是 L2。
- [ ] **明确定义了搬运字节数和 FLOPs**，再算 GB/s / TFLOPS / AI。
- [ ] **对拍正确性**：优化不能改变结果（浮点容差内）。
- [ ] **ncu 只用来解释原因**，不用来做绝对计时。

## 小结

- launch 开销实测 **~1.9 µs/次**，小 kernel 的墙钟一大半是它——这是算子融合 / CUDA Graph 的价值所在。
- **冷启动**（25 µs）与**稳态**（9 µs）差 2.8 倍，必须 warmup。
- **L2（52 MB）会让小数据的带宽虚高甚至超峰值**（本例 3954 GB/s）；测 HBM 效率要用大工作集。
- 有效带宽 = 搬运字节 / 时间；AI = FLOPs / 字节，用来定位 roofline 位置。
- `ncu` 的 SOL / Memory / Occupancy 三节是入门必看：先看哪条「throughput」高，再看 L2 命中、occupancy 找原因。

下一篇正式进入优化：**访存合并与向量化**。我们会故意写一个「访存错位」的 kernel，看它如何把带宽打到零头，再一步步修回峰值。

> 下一篇：CUDA 算子调优（四）· 访存合并与向量化
