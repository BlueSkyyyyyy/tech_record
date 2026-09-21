---
title: "CUDA 算子调优（一）：开篇 — GPU 怎么跑一个算子，怎么判断它快不快"
date: 2026-09-21
draft: false
weight: 1
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "roofline", "性能分析", "系列"]
categories: ["算子开发"]
---

这个博客之前写了不少「读 kernel 源码」的文章（FlashAttention 精读、Liger-Kernel 对照、TE 算子 roofline）。读得多了会有一个缺口：**知道别人怎么写的，但自己动手从零优化一个算子是什么体验？** 这个系列就来补这一课——从 CUDA 编程入门开始，一个算子一个算子地写、测、用 `ncu` 剖析，把「为什么慢 / 怎么想到优化 / 改完快了多少」完整走一遍。

这是第一篇，先不写代码，把三件事讲清楚：**GPU 是怎么执行一个 kernel 的**、**用什么模型判断一个算子快不快**、**本系列的实验环境与工具链**。

配套代码在 [`code/kernel-opt/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/kernel-opt)，路线图与进度见 [ROADMAP.md](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/ROADMAP.md)。

---

## 1. 一页纸看懂 GPU 执行模型

写 CUDA 之前，脑子里要先有一张图。CPU 和 GPU 的分工是这样：

- **Host（CPU）** 负责准备数据、发起「内核（kernel）」调用；
- **Device（GPU）** 上跑的是成千上万个线程，每个线程执行同一段代码、但处理不同的数据（SIMT：Single Instruction, Multiple Threads）。

一次 kernel 调用的线程组织是三层：

```
grid  (一次 kernel 启动的全部线程)
 ├── block 0            ← 调度到某个 SM 上执行；block 内可共享 shared memory、可同步
 │    ├── warp 0        ← 32 个线程，GPU 调度的最小单位，锁步执行同一条指令
 │    ├── warp 1
 │    └── ...
 ├── block 1
 └── ...
```

关键点：

- **warp 是 32 个线程**，硬件以 warp 为单位取指令、发射执行。所以「一个 warp 里的 32 个线程访问内存是否连续」极大影响性能（后续第 04 篇专门讲）。
- **block 是资源分配单位**：一个 block 被塞进一个 SM（Streaming Multiprocessor），block 内的线程可以用 `__shared__` 共享内存、用 `__syncthreads()` 同步。**不同 block 之间不能直接通信**。
- 一个 SM 上可以同时驻留多个 block（取决于寄存器/共享内存/线程数），这就是后面「occupancy（占用率）」讨论的东西。

存储层次（延迟和带宽差好几个数量级，是优化的核心矛盾）：

| 层级 | 归属 | 典型容量 | 典型延迟 | 谁能访问 |
|---|---|---|---|---|
| 寄存器 Register | 每个线程私有 | 255 个/线程 | ~1 cycle | 本线程 |
| 共享内存 Shared Memory | 每个 block | 几十~200 KB | ~20-30 cycle | block 内所有线程 |
| L1 / L2 Cache | SM / 全 GPU | L2 52 MB (H100) | ~200 cycle | 硬件缓存 |
| 全局显存 HBM | 全 GPU | 80 GB | ~500+ cycle | 所有线程 |

一句话总结这个系列的多数主题：**算子的瓶颈往往不在「算」，而在「搬数据」**，而搬数据的效率取决于你有没有用好上面这张表（合并访存、把数据放进 shared memory 复用、用寄存器挡住延迟）。

## 2. 判断快不快的标尺：Roofline

光有「跑得快/慢」的直觉不够，我们需要一把尺子。最常用的叫 **Roofline 模型**，它只问两个问题：

1. 这个算子每搬运 1 字节，做了多少次浮点运算？记为**算术强度 AI = FLOPs / Bytes**。
2. 机器有两个峰值：**峰值带宽**（搬数据的上限）和**峰值算力**（算的上限）。

于是性能上限是一条折线：

```
性能
 ^                                  计算受限区（平顶 = 峰值算力）
 |                                 ┌──────────────────────────
 |                                /
 |                               / ← 内存受限区（斜率 = 峰值带宽）
 |                              /
 |_____________________________/______________________________> 算术强度 AI
                          ridge 拐点
```

- **拐点左边（低 AI）**：内存受限。想更快，只能减少搬运字节、或提高带宽利用率——**再怎么优化计算都没用**。
- **拐点右边（高 AI）**：计算受限。想更快，要减少 FLOP、或用上更快的计算单元（比如 Tensor Core）。

拐点 = 峰值算力 ÷ 峰值带宽。本系列全程用 H100 SXM，常量来自实测 `device_info`（见下节）：

| 常量 | 值 | 说明 |
|---|---|---|
| HBM 带宽 | ≈ **3.35 TB/s** | 内存天花板 |
| FP32 CUDA core 峰值 | ≈ 66.9 TFLOPS | 普通标量计算 |
| BF16/FP16 Tensor Core 峰值（dense） | ≈ 989 TFLOPS | 矩阵乘用 |
| FP8 Tensor Core 峰值 | ≈ 1978 TFLOPS | |
| ridge（BF16/FP16 TC） | ≈ 295 FLOP/byte | 989 T / 3.35 T |
| ridge（FP32 CUDA core） | ≈ 20 FLOP/byte | 66.9 T / 3.35 T |

> 注意：同一个「AI」下，走 Tensor Core 和走 CUDA core 的天花板完全不同。判断内存受限时，看的是**对应计算单元**的那条线；但绝大多数算子 AI 都很低，连 CUDA core 的 20 都够不到，所以「内存受限」是常态。

举个本系列会反复出现的例子：元素级算子（elementwise，如 `c = a + b`）每个元素读 2 次写 1 次（12 字节），只做 1 次加法（1 FLOP），**AI ≈ 0.08**，低到尘埃里——它 100% 是内存受限的，理论上限就是 3.35 TB/s。第 02 篇的 vector add 就是这样一个算子。

## 3. 工具链：写、跑、剖析

本系列用到三样工具，分工很明确：

| 工具 | 干什么 | 什么时候用 |
|---|---|---|
| `nvcc` | 编译 `.cu`（host + device 代码） | 写代码时 |
| **`ncu`**（Nsight Compute） | 剖析**单个 kernel 内部**：带宽、算力、occupancy、warp 停顿原因 | 「这个 kernel 为什么慢」 |
| `nsys`（Nsight Systems） | 看**端到端时间线**：kernel 顺序、拷贝、气泡、launch 开销 | 「整个流程哪里空转了」 |

初学者最容易混的是 `ncu` 和 `nsys`：**单算子内部指标用 ncu，端到端时序用 nsys。** 本系列主要用 `ncu`。

### 实验环境：一个专门的容器 `kernel_lab`

本机是 8 × H100 80GB SXM，宿主 CUDA 驱动能跑 CUDA 13.2。但这里有个坑：宿主的 `RmProfilingAdminOnly=1`，意味着**只有具备特权能力的进程才能读 GPU 性能计数器**；普通容器里跑 `ncu` 会直接报：

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

解决办法是起一个带 `--cap-add SYS_ADMIN --cap-add SYS_PTRACE` 的容器。本系列把它固化成了脚本 [`code/kernel-opt/scripts/lab.sh`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/scripts/lab.sh)：

```bash
cd ~/proj/tech_record/code/kernel-opt
scripts/lab.sh up        # 创建/启动 kernel_lab（幂等）
scripts/lab.sh enter     # 进去交互
```

代码在宿主机写，容器挂了同一个目录，路径一致。编译运行和剖析也各有一个脚本：

```bash
scripts/run.sh 02-first-kernel/vector_add.cu
scripts/ncu.sh 03-measurement/timer_pitfalls.cu --section SpeedOfLight --kernel-name regex:copy_v4
```

先跑一下环境自检程序 `01-overview/device_info.cu`，它会打印本机 GPU 的真实参数（后面所有文章换算都以此为准）：

```
Device : NVIDIA H100 80GB HBM3
  SMs                : 132
  clock              : 1.980 GHz
  HBM bandwidth(peak): 3352.3 GB/s
  memory             : 85.0 GB
  max threads/SM     : 2048  regs/SM: 65536  smem/SM: 233472 B
  warp size          : 32
  L2 cache           : 52.43 MB
  memory bus         : 5120 bit
  compute capability : 9.0
```

132 个 SM、1.98 GHz、3.35 TB/s、52 MB L2——这几个数字请记住，本系列会一直用它们做对照。

## 4. 系列地图

由浅入深，每篇一个主题、一份能跑的代码：

1. **开篇**（本篇）：执行模型、roofline、工具链
2. 第一个 CUDA kernel：线程层次、vector add 的三种写法
3. 正确测量：计时陷阱、有效带宽怎么算、ncu 入门
4. 访存合并与向量化：elementwise 逼近带宽峰值
5. 共享内存与 bank conflict：矩阵转置
6. 归约与 warp shuffle：softmax / LayerNorm 的底座
7. Softmax / LayerNorm 优化：online softmax、融合
8. GEMM 入门：从 naive 到 shared-memory tiled
9. GEMM 进阶：寄存器分块、double buffering
10. ncu 深潜：occupancy、warp stall、SASS
11. launch 配置与 occupancy
12. 异步拷贝与流水线（`cp.async` / TMA）
13. Tensor Core 入门：WMMA / `mma` PTX
14. 融合与 epilogue：GEMM+bias+激活、split-K

## 小结

- GPU 以 **warp（32 线程）** 为调度单位、以 **block** 为资源分配单位；存储层次从寄存器到 HBM 延迟差几百倍，优化的主线是**搬数据**。
- **Roofline** 用「算术强度 AI」判断内存受限还是计算受限；H100 的内存天花板是 **3.35 TB/s**，绝大多数 elementwise / reduction 算子都撞在这条线上。
- 工具链：`nvcc` 编译、**`ncu` 看单 kernel 内部**、`nsys` 看端到端；本系列在特权容器 `kernel_lab` 里跑，否则 `ncu` 没有权限。

下一篇进入正题：写出第一个 CUDA kernel，并第一次看到「一个算子可以多快」。

> 系列代码与路线图：[`code/kernel-opt/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/kernel-opt)
