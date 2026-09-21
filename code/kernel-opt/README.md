# kernel-opt — CUDA 算子调优系列配套代码

配套博客系列：`content/posts/cuda-kernel-opt-*`。路线图与进度见 [ROADMAP.md](ROADMAP.md)，
优化技巧与性能榜见 [TECHNIQUES.md](TECHNIQUES.md)。

> 系列分两阶段：**01–14 通用 CUDA 优化**；**15 起为模型场景算子**（DeepSeek-V4.x / Kimi-K2.6 / Qwen3 的
> MLA、DSA 稀疏注意力、MoE、FP8 GEMM、MuonClip 等），并把性能持续推向极致。

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
| `06-reduction/` | 06 归约与 warp shuffle | 求和：原子反面教材 / smem 树 / warp shuffle / float4 |
| `07-softmax/` | 07 Softmax 优化 | 多趟 / 融合 / shared memory 缓存整行 |
| `08-gemm/` | 08 GEMM 入门 | naive / smem tiling（TFLOPS + ncu 瓶颈）|
| `09-gemm-advanced/` | 09 GEMM 进阶 | 寄存器分块 / float4 向量化 / cp.async 双缓冲 + cuBLAS 对照 |
| `10-ncu-deep/` | 10 ncu 深潜 | occupancy 手算 / warp stall 指纹 / roofline / source-SASS 对照 |
| `11-launch-occupancy/` | 11 launch 配置与 occupancy | ILP × occupancy / `__launch_bounds__` 扫描 / thread tile 扫描（落地 09 GEMM）|
| `12-async-pipeline/` | 12 异步拷贝与流水线 | cp.async 2/3/4/5 级软流水线 + ncu stall 对照（落地 09 GEMM）|
| `13-tensor-core/` | 13 Tensor Core 入门 | WMMA / 裸 mma.m16n8k16 + ldmatrix / bank conflict padding + cuBLAS BF16 对照 |
| `14-fusion-epilogue/` | 14 融合与 epilogue | GEMM+bias+GELU/ReLU 寄存器融合 vs 独立 epilogue kernel（11%/26%/33% 提速，随 K 缩小放大）|
| `15-mla-attn/` | 15 MLA 注意力（一） | MLA 吸收成 MQA：naive/head_reuse/smem 标量版（15~19 TFLOPS，算力受限）+ TC 三 kernel（115.6 TFLOPS），扫 S=1k/2k/4k |

## 怎么跑

宿主机上直接调用脚本即可（自动进 `kernel_lab` 容器编译运行）：

```bash
cd ~/proj/tech_record/code/kernel-opt
scripts/lab.sh up
scripts/run.sh 02-first-kernel/vector_add.cu
scripts/ncu.sh 02-first-kernel/vector_add.cu --set full --kernel-name regex:add
```

环境变量：`ARCH`（默认 `sm_90`）、`GPU`（默认 0）、`NVCC_FLAGS`。

## 无人值守自驱（autopilot）

`scripts/autopilot.sh` 会**反复启动 opencode 无头会话**（`opencode run --auto`），每轮推进
`ROADMAP.md` 里的一篇文章增量（写代码 → 实测 → ncu → 写文 → 发布 → 更新路线图），
适合离开电脑后持续工作：

```bash
scripts/autopilot.sh start     # 后台启动（setsid 脱离终端，关终端也不停）
scripts/autopilot.sh status    # 查看状态 + 最近日志
scripts/autopilot.sh stop      # 停止
scripts/autopilot.sh run       # 前台运行（调试）
```

- 日志：`code/kernel-opt/autopilot.log`；停止文件：`AUTOPILOT_STOP`（agent 做完整个系列也会自动创建）。
- 可调：`MAX_ROUNDS`（默认 40）、`SLEEP_BETWEEN`（默认 30s）、`TIMEOUT_PER_ROUND`（默认 5400s）。
- 连续失败 3 次会自行停止；`autopilot.lock` 防止重复启动。

## 约定

- 每个 `.cu` 都能独立编译运行，自带参考实现对拍与计时输出。
- 实测原始输出存到同目录 `*.out.txt`，供写文章引用。
- 公共代码只放 `common/`，不放实验目录。
