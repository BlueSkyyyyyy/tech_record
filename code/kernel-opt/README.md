# kernel-opt — CUDA 算子调优系列配套代码

配套博客系列：`content/posts/cuda-kernel-opt-*`。路线图与进度见 [ROADMAP.md](ROADMAP.md)。

## 目录

| 目录 | 对应文章 | 内容 |
|---|---|---|
| `common/` | — | 公共头 `cuda_utils.cuh`（设备信息 / 计时 / 带宽换算） |
| `scripts/` | — | `lab.sh` 容器管理、`run.sh` 编译运行、`ncu.sh` 剖析 |
| `01-overview/` | 01 开篇 | 环境自检小程序 |
| `02-first-kernel/` | 02 第一个 kernel | vector add 的三种写法 |
| `03-measurement/` | 03 正确测量 | 计时陷阱 / launch 开销 / 带宽实验 |
| `04-coalescing/` | 04 访存合并与向量化 | 合并 vs 跨行访问 vs float4 的带宽对比 |
| `05-transpose/` | 05 共享内存与 bank conflict | 矩阵转置：naive / smem 无 padding / padding |

## 怎么跑

宿主机上直接调用脚本即可（自动进 `kernel_lab` 容器编译运行）：

```bash
cd ~/proj/tech_record/code/kernel-opt
scripts/lab.sh up
scripts/run.sh 02-first-kernel/vector_add.cu
scripts/ncu.sh 02-first-kernel/vector_add.cu --set full --kernel-name regex:add
```

环境变量：`ARCH`（默认 `sm_90`）、`GPU`（默认 0）、`NVCC_FLAGS`。

## 约定

- 每个 `.cu` 都能独立编译运行，自带参考实现对拍与计时输出。
- 实测原始输出存到同目录 `*.out.txt`，供写文章引用。
- 公共代码只放 `common/`，不放实验目录。
