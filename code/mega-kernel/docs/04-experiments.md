# MegaKernel 实验：环境探测与实测

> 本文是「MegaKernel 资料与实现分析」的实验篇。先记录 **P0-3 环境探测**的结果——
> 目标卡（H100 / sm90）上 persistent kernel、CUDA Graph、PDL 三项能力是否可编译可运行，
> 作为后续 P4/P5 实测的基线；后续 P5 的 launch 开销 / 两算子融合实验继续写在本文件。

## 0. 环境探测（P0-3）

### 0.1 目的

MegaKernel 的实测需要三块基础能力，先确认它们在本机容器 `kernel_lab` 上「能编译、能跑」：

1. **persistent kernel + cooperative grid sync**：单 kernel 常驻、跨 block 同步——P5「单持久化 kernel」的前提。
2. **CUDA Graph capture/replay**：消除 launch 开销——P4-1 / P5-1 的对照组。
3. **PDL（Programmatic Dependent Launch）**：让相邻 kernel 部分重叠——P4-2 / P5-3。

探测程序：`src/probe_env.cu`（自包含，单文件，含三项的编译与运行自检）。

### 0.2 硬件 / 工具链

| 项 | 值 |
|---|---|
| GPU | NVIDIA H100 80GB HBM3（8 卡，本探测用 1 卡） |
| compute capability | 9.0（**sm90**） |
| SM 数 | 132 |
| shared mem / SM | 233,472 B（228 KB） |
| shared mem / block | 49,152 B（48 KB，默认） |
| 寄存器 / SM | 65,536 |
| 驱动 | 570.211.01 |
| nvcc | release 13.2, V13.2.78 |
| CUDA runtime | 13020 |

### 0.3 编译与运行

在容器内经 `scripts/lab.sh` 执行（绝对路径，workdir 为容器内 `/tmp`）：

```
scripts/lab.sh exec "nvcc -O3 -arch=sm_90 -rdc=true \
  /ssd/.../code/mega-kernel/src/probe_env.cu -o probe_env && ./probe_env 200"
```

原始输出见 `harness/probe_env.out.txt`。

### 0.4 结果

**(1) persistent kernel + cooperative grid sync —— 支持**

| 指标 | 值 |
|---|---|
| `cudaDevAttrCooperativeLaunch` | 1（YES） |
| `persistent_kernel` occupancy | 16 blocks/SM @128 threads |
| persistent kernel（grid=132，200 轮，每轮 grid sync） | **0.330 ms** |
| 对照：200 个 trivial kernel 串行 launch（无 sync） | **0.575 ms** |

一个常驻 kernel 内做 200 次 grid-wide 屏障（跨 132 个 block）总耗时 0.330 ms，
平均每次屏障约 1.65 µs；而 200 次独立 kernel launch 反而要 0.575 ms。
这说明 **cooperative grid sync 在本机可跑**，且在这些规模下跨 block 屏障比反复 launch 更省。
（注：这是能力探测而非公平的 launch 开销基准，量级仅供定位；正式的 launch 开销量在 P5-1。）

**(2) CUDA Graph capture/replay —— 支持**

| 指标 | 值 |
|---|---|
| 图中节点数 | 32 个小 kernel |
| plain：200 × 32 次串行 launch | 18.007 ms |
| graph：200 次 replay | 11.567 ms |
| 加速比 | **1.56×** |

用独立非默认 stream 以 `cudaStreamCaptureModeThreadLocal` 捕获（legacy default stream 在
global 模式下不可捕获，见下方「坑」）。32 个节点的图捕获后 replay，相对逐次 launch 约 **1.56×**
加速，印证 CUDA Graph 消除的正是 host 侧逐 kernel launch 开销。**但 graph 仍保留 kernel barrier**——
它不改变串行性（对照 `docs/03-related-tech.md` P4-1）。

**(3) PDL（Programmatic Dependent Launch）—— 支持**

| 项 | 结果 |
|---|---|
| precheck（sm90 + CUDA 13.2） | likely YES |
| `cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization` 启 producer | ok |
| 同属性启 consumer | ok |
| device sync | ok |
| 运行后 flag | producer=1, consumer=1 |

producer 用 `cudaTriggerProgrammaticLaunchCompletion()` 提前放行，consumer 用
`cudaGridDependencySynchronize()` 等待上游数据就绪，两个 kernel 都正常完成。
**PDL 在本机可编译可运行**，P5-3 可做真实的重叠度量。

### 0.5 结论与对后续实验的意义

| 能力 | 状态 | 支撑的子项 |
|---|---|---|
| persistent kernel + cooperative grid sync | 可编译可运行 | P5-2（两算子融进单持久化 kernel） |
| CUDA Graph capture/replay | 可编译可运行 | P4-1、P5-1（launch 开销对照） |
| PDL | 可编译可运行 | P4-2、P5-3（跨 kernel 重叠） |

三项均**实测通过**，无「仅资料分析」降级项。后续 P4/P5 的实测均可落地。

### 0.6 踩到的坑

- **legacy default stream 不能用 `cudaStreamCaptureModeGlobal` 捕获**：会报
  `operation not permitted when stream is capturing`。改用独立非默认 stream +
  `cudaStreamCaptureModeThreadLocal` 即正常。
- **CUDA 13 移除了 `cudaDeviceProp::cooperativeMultiDeviceLaunch`**：引用它会编译报错，
  多设备 cooperative 查询改用设备属性 API。
