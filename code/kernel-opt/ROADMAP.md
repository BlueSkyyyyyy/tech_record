# CUDA 算子调优系列 · 路线图（活文档）

> 这是「CUDA 算子调优」系列持续自驱工作的**控制面板**。每完成一篇，
> 更新这里的状态与「下一步」。AI agent 每次接管时：先读本文 → 读
> `agent_guide.md` → 读本系列的 [agent skill](../../agent_skills/kernel-opt.md) → 从「下一步」做起。

## 目标

一条**由浅入深、每篇都能自己跑出来**的算子调优学习线：
从 CUDA 编程入门、正确测量，到访存/共享内存/归约/GEMM/Tensor Core/异步流水线，
每篇都包含：可运行代码 + 本地实测数据 + ncu 剖析 + 通俗的优化思路说明。

原则：

1. **代码必须能跑**：每个实验都有 `__main__` 式自测 / 参考实现对拍，实测数据写进文章。
2. **数据必须真实**：文章里的数字来自本机 H100 实测（记录环境、命令、原始输出）。
3. **先说人话**：讲清楚「为什么慢 → 怎么想到的 → 改了什么 → 快了多少」。
4. **循序渐进**：每篇只引入 1~2 个新概念，且依赖前面的结论。
5. **持续演进**：路线图不是圣旨，跑出来有新发现就调整。

## 硬件 / 环境

| 项 | 值 |
|---|---|
| GPU | 8 × NVIDIA H100 80GB HBM3 (SXM)，CC 9.0，132 SM，峰值 ~3.35 TB/s，BF16 TC dense ~989 TFLOPS |
| 容器 | `kernel_lab`（镜像 `dsv4-inf:latest`，CUDA 13.2，含 nvcc / ncu / nsys / PyTorch） |
| 关键点 | 宿主 `RmProfilingAdminOnly=1`，普通容器跑 ncu 报 ERR_NVGPUCTRPERM；`kernel_lab` 加了 `SYS_ADMIN`/`SYS_PTRACE` 才能 profiling |

```bash
cd ~/proj/tech_record/code/kernel-opt   # 注意本机有软链：/home/xieminglin -> /ssd/home/xieminglin

scripts/lab.sh up                  # 确保容器在跑（幂等）
scripts/lab.sh enter               # 进容器
scripts/run.sh 02-first-kernel/vector_add.cu   # 编译 + 运行
scripts/ncu.sh 03-measurement/foo.cu --set full --kernel-name regex:foo   # ncu 剖析
```

> 容器可重建：`scripts/lab.sh rm && scripts/lab.sh up`。

## 系列大纲

状态：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 已完成并发布 · `[-]` 暂缓。

### 第一部分：入门与测量

- [x] **01 开篇**：为什么算子调优重要 · GPU 执行模型一页纸 · roofline 性能模型（带宽 vs 算力）· 工具链（nvcc/ncu/nsys）· 环境搭建
- [x] **02 第一个 CUDA kernel**：线程层次（grid/block/thread/warp）· vector add 的三种写法（单元素 / grid-stride / float4）· 编译与错查
- [x] **03 正确测量**：CUDA event 计时的陷阱（warmup、launch 开销、L2 常驻）· 有效带宽/FLOPs 怎么算 · ncu SpeedOfLight & Memory Workload 入门

### 第二部分：内存是瓶颈

- [x] **04 访存合并与向量化**：coalescing 原理（一个 warp 一次 128B 事务）· copy 行/列优先 7.4× 差距 · ncu sectors-per-request · float4 89% 峰值
- [x] **05 共享内存与 bank conflict**：矩阵转置（naive → 分块 smem → padding 消冲突）· bank 是怎么分的 · ncu 看冲突（6531 万→40 万）
- [x] **06 归约与 warp shuffle**：树形归约 · `__shfl_down_sync` · 多 block + atomics · 每元素原子加 0.1% → 两级归约 91%

### 第三部分：计算与融合

- [x] **07 Softmax 优化**：多趟(29.8%) → 融合(37.4%) → 整行 smem 缓存(85.3%) · online softmax 引出 FlashAttention · LayerNorm 同套路
- [x] **08 GEMM 入门**：naive(8.2%) → smem tiled(13.5%) · ncu 指出瓶颈是 L1/TEX 载入管道 · 引出寄存器分块
- [x] **09 GEMM 进阶**：寄存器分块（thread tile）· 向量化 + double buffering · 逼近 cuBLAS 的百分比

### 第四部分：进阶专题

- [x] **10 ncu 深潜**：occupancy 计算 · warp stall reasons · roofline section · source/sass 对照
- [x] **11 launch 配置与 occupancy**：寄存器/共享内存限制 · `__launch_bounds__` · 循环展开/ILP · thread tile 落地 09 GEMM
- [x] **12 异步拷贝与流水线**：`cp.async` 多级软流水线（2/3/4/5 级）· `long_scoreboard` 0.91→0.03 · TMA 与 warp specialization 概念 · 与第 09 篇结合
- [x] **13 Tensor Core 入门**：WMMA API / `mma` PTX · `m16n8k16` + `ldmatrix` · BF16 TC GEMM（220.98 TFLOPS / 22.3%）
- [ ] **14 融合与 epilogue**：GEMM+bias+激活 · split-K · 生产算子（attention/GEMM）串讲

## 每篇的 Definition of Done

- [ ] 文章 `content/posts/cuda-kernel-opt-NN-<slug>/index.md`，`draft: false`，含 `weight`
- [ ] 配套代码 `code/kernel-opt/NN-<slug>/`，实测通过，原始输出留存（`*.out.txt` / `.log`）
- [ ] 文章含：背景 → 现象/数据 → 根因 → 优化 → 实测对比表 → 小结 → 下一篇预告
- [ ] 大数/结论有 ncu 或实测支撑；引用的行号/API 核对过
- [ ] README.md 与 code/README.md 索引更新
- [ ] `hugo --gc --minify` 无 ERROR；publish 技能验证线 200

## 工作循环（agent 自驱）

1. 读 ROADMAP 的「下一步」，选一篇。
2. 写 / 改代码到 `code/kernel-opt/NN-*/`，用 `scripts/run.sh` 跑通，`scripts/ncu.sh` 采集关键指标。
3. 写文章，把实测数字填进去（不要编）。
4. 走 `agent_skills/publish.md`：构建 → commit → push → 验证线上 200。
5. 更新本文件状态、`content`/`code` 索引、`agent_skills/kernel-opt.md` 里踩到的坑。
6. commit & push 阶段性成果（建议每 1~2 篇一次）。

## 无人值守（autopilot）

`scripts/autopilot.sh` 会循环启动 opencode 无头会话，每轮独立完成一篇文章增量，
自动提交推送。启动/查看/停止：`scripts/autopilot.sh {start|status|stop}`。
日志在 `autopilot.log`。**agent 每轮开始前先看本文的「下一步」；整个系列完成后 `touch AUTOPILOT_STOP` 让循环停下。**

## 阻塞

（当前无）

## 当前进度

- 2026-09-21：搭建容器 `kernel_lab` 与 `scripts/`、`common/cuda_utils.cuh`，起草路线图。
- 2026-09-21：完成并发布 **01–08**（… / Softmax / GEMM 入门）。已 push 且线上 200。
- 2026-09-21：完成并发布 **09 GEMM 进阶**：寄存器分块 8×8 + 交错映射消 bank conflict → 39.2%；float4 向量化 → 45.0%；cp.async 双缓冲 → **51.5%**（34.46 TFLOPS）；同口径 cuBLAS 75.8%（50.73 TFLOPS），达其 ~68%。
- 2026-09-21：完成并发布 **10 ncu 深潜**：手算 occupancy（`k_occ<2>` = 2 block/25%，与 ncu 一致）；`__launch_bounds__` 强制提 occupancy → spill 1.31 TB local 流量、慢 34×；五类 stall 指纹表；roofline 拐点 AI\*≈20 并澄清 `Memory Throughput` 是缓存层级最大值（`k_latency` 97% 卡 L1、DRAM 仅 0.13%）；source/SASS 依赖链与 STL/LDL 对照。结论：`occ<2>` 25% occupancy 仍达 98.5% Compute / IPC 3.94。
- 2026-09-21：完成并发布 **12 异步拷贝与流水线**：把 09 的 cp.async 双缓冲推广成模板化 N 级软流水线（`gemm_pipe<STAGES>`，环形缓冲 + `__pipeline_wait_prior(STAGES-1)`）：`sync` 30.18%（0.5692 ms）→ `pipe2` 50.4% → `pipe3` 53.3% → `pipe4` **53.6%**（35.84 TFLOPS）→ `pipe5` 53.4%，三级后饱和。ncu：`long_scoreboard` 0.91→0.03（双缓冲即归零），`issue_active` 56.8%→70.9%，头号 stall 变成 `not_selected`（~2.0，好信号）；DRAM 仅 2%（2048³ 驻 L2），瓶颈在 SM/发射。同口径 cuBLAS 50.73 TFLOPS（75.8%），达其 **70.6%**。附 TMA / warp specialization 演进说明。原始输出与 ncu 在 `12-async-pipeline/`。
- 2026-09-21：完成并发布 **13 Tensor Core 入门**：WMMA API（62 TFLOPS / 6.3%）→ WMMA+cp.async（72.9）→ 裸 `mma.sync.aligned.m16n8k16` + 标量 LDS（~65）→ 裸 mma + `ldmatrix`（无 padding 76.2，有 padding 115）→ +cp.async 双缓冲 **220.98 TFLOPS / 22.3%**。最大杠杆是 **smem 行距 +8 padding 消 ldmatrix bank conflict**（3565 万 → 0，mma_pipe 76→221，2.9×）；瓶颈随优化 WMMA L1/TEX 77% → padding 后 Compute 37% / L2 70% / DRAM 7%。同口径 cuBLAS BF16 672.70 TFLOPS（68%），达其 32.8%，为 12 的 fp32（35.84）的 6.2×。原始输出/ncu 在 `13-tensor-core/`。
- 2026-09-21：完成并发布 **11 launch 配置与 occupancy**：①ILP×occupancy 二维实验——12.5% occ 下 ILP=1→2 从 29.9 翻到 53.6 TFLOPS（`wait` stall 2.96→0.06），ILP=1 时 100% occ 也能到 63.7 TFLOPS，证明两者可互相替代；②在 09 GEMM（128×128/8×8/256 线程）上 `__launch_bounds__(256,MINB)` 扫描：`lb<2>`=128 寄存器/25%/30.6 TFLOPS 持平基线，`lb<3>`/`lb<4>` spill 296/520 B、occupancy 37.5%/50% 却是 11.3%/5.3%（最惨慢 8.7×，`long_scoreboard` 0.94→16.06）；③thread tile 扫描：`2×2`(100% occ) 18.8% vs `8×8`(25% occ) 46.8%，且 `8×8` 需 256 线程/block 才行（64 线程仅 23.7%）。模板化 GEMM 与 09 的 `gemm_reg_vec` 对齐（30.6 vs 30.1 TFLOPS）。

## 下一步（明确到可执行）

- [ ] 完成 **14 融合与 epilogue**：在 13 的 `mma_pipe` 骨架上做 GEMM+bias+激活（GELU/ReLU）融合，写一个带 epilogue 的生产形态 GEMM；可选顺带讲解 split-K 与 attention/GEMM 串讲
- [ ] 每完成一篇：更新本文件、README 索引，提交推送
- [ ] 可选：给 01 的 roofline 画一张 mermaid 图

## 灵感 / backlog（想到就记，别丢）

- 用 `torch.profiler`/CUPTI 与 CUDA event 对照，量化 launch 开销（呼应 te-perf 的结论）。
- 拿 `How_to_optimize_in_GPU`（本地 `~/github`）里的例子做交叉验证。
- cutlass 的 threadblock 层次与本文系列对照。
- 单独一篇：H100 vs A100 的 smem/TC 差异对调优的影响。
- 把每篇的 kernel 存成版本序列（v0/v1/v2），做一个 "diff 视图" 展示优化步骤。
