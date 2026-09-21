# kernel-opt — CUDA 算子调优系列的自驱工作流程

## 触发场景

用户说「继续整理/推进算子优化系列」「CUDA 算子调优」（本仓库的长期自驱任务）。
本技能把「写代码 → 本地实测 → ncu 剖析 → 写文章 → 发布」串成可重复的循环。

> **无人值守模式**：本技能常被 `code/kernel-opt/scripts/autopilot.sh` 以无头会话反复调用，
> 每轮只做一篇文章增量，做完就停，更新路线图后本轮结束。若你是被 autopilot 唤醒的：
> 全程不要请求人工确认（别用 question 类工具），**永远不要自己创建 `AUTOPILOT_STOP`**
> （只有用户运行 `autopilot.sh stop` 才停）。**当 ROADMAP 的「下一步」做完/为空时，
> 自己从第五/六/七部分或 backlog 里补充具体、可测的新任务再继续**，持续把算子性能往极致推。

## 前置条件

- 已读根目录 `agent_guide.md`（目录约定、已知坑）。
- 已读系列控制面板 `code/kernel-opt/ROADMAP.md`（**每次先读它，从「下一步」做起，做完更新它**）。
- 已读技巧台账 `code/kernel-opt/TECHNIQUES.md`（**每篇必须往里加技巧条目 + 更新性能榜**）。
- 环境：宿主机有 `docker` 权限；镜像 `dsv4-inf:latest` 已存在（含 CUDA 13.2 / nvcc / ncu / nsys / PyTorch）。

## 环境：kernel_lab 容器（关键）

宿主机 `nvidia-smi` 显示 8×H100，但**宿主 `RmProfilingAdminOnly=1`**，普通容器里 ncu 会报
`ERR_NVGPUCTRPERM`。必须用带 `--cap-add SYS_ADMIN` 的容器——已封装为 `code/kernel-opt/scripts/lab.sh`：

```bash
cd ~/proj/tech_record/code/kernel-opt
scripts/lab.sh up          # 创建/启动 kernel_lab（幂等，可重复调用）
scripts/lab.sh enter       # 交互 shell
scripts/lab.sh status
```

代码在宿主机与容器内路径一致（容器挂了 `/home/xieminglin` 与 `/ssd`），所以直接在宿主侧写代码即可。

## 步骤（每篇的循环）

1. **选一篇**：读 ROADMAP「下一步」。
2. **写代码**：`code/kernel-opt/NN-<slug>/*.cu`，`#include "../common/cuda_utils.cuh"`。
   - 每个 .cu 自带参考实现对拍 + 计时；编译运行：`scripts/run.sh NN-<slug>/foo.cu`。
   - 剖析：`scripts/ncu.sh NN-<slug>/foo.cu --set full --kernel-name regex:foo`
     （ncu 参数写在前，程序参数用 `--` 分隔）。
   - 实测原始输出另存 `NN-<slug>/foo.out.txt`，写文章时引用真实数字。
3. **写文章**：按 `write-post.md`，slug `cuda-kernel-opt-NN-<slug>`，加 `weight`（=NN）与系列 tag。
   结构：背景 → 现象/数据 → 根因 → 优化 → 实测对比表 → 小结 → 下一篇预告。讲人话，配图优先。
4. **发布**：按 `publish.md`（构建 → commit → push → 验证线上 200）。
5. **回写控制面板**：更新 `ROADMAP.md` 状态与「下一步」、`README.md`/`code/README.md` 索引，
   把新踩的坑追加到本文件或 `agent_guide.md`。
6. **回写台账**：在 `TECHNIQUES.md` 新增技巧条目（原理/场景/代码/实测收益/坑）并刷新性能榜；
   模型类文章更新「模型场景台账」。
7. **模型场景优先**：第五部分（MLA/DSA/MoE/FP8/MuonClip…）是当前重点，要用 `/ssd/models/*/config.json`
   的真实参数构造 shape，并对标 `~/github/` 里的 FlashMLA / DeepGEMM / cutlass 等实现给出差距。
8. **提交**：每篇 commit & push 一次，信息格式 `post:` / `bench:` / `skill:`。**不要**创建 `AUTOPILOT_STOP`。

## 验收标准

- [ ] 配套代码在 `kernel_lab` 里实跑通过，原始输出已留存
- [ ] 文章数字全部来自实测（不是猜的），关键结论有 ncu 支撑
- [ ] ROADMAP 状态、「下一步」、索引已更新
- [ ] `hugo --gc --minify` 无 ERROR，线上页面 200

## 已知坑

- **ncu 权限**：必须 `kernel_lab`（含 SYS_ADMIN）。用 `kimi26_train` 会 `ERR_NVGPUCTRPERM`。
- **容器 root 建目录**：首次以容器 workdir 创建目录会属 root，宿主写不进去；
  用 `docker exec kernel_lab chown -R 1001:1001 <dir>` 修。
- **`/home/xieminglin` 是软链**到 `/ssd/home/xieminglin`；`readlink -f` 后路径一致，脚本已处理。
- **H100 峰值常量**（写文章换算用）：HBM ~3.35 TB/s；FP32 CUDA core ~66.9 TFLOPS；
  BF16/FP16 Tensor Core dense ~989 TFLOPS；FP8 ~1978 TFLOPS。
- CUDA event 计时**含 launch 开销**，测单 kernel 纯 device 时间用 ncu / CUPTI（见第 03 篇）。
- 小 shape 反复跑会命中 L2（H100 ~50MB），带宽会虚高，判断 HBM 效率要用大 shape。
- **`ncu` 的 "Memory Throughput" 是 L1/L2/DRAM 里最高那一级的利用率**，不等于 HBM；判断带宽瓶颈必须看 `DRAM Throughput`。
- **强制提高 occupancy 可能是负优化**：`__launch_bounds__(threads, N)` 压寄存器会 spill 到 local memory（实测 `occ<8>` spill 1.31 TB，慢 34×）。先看 `not_selected`/`no_eligible` 是否真的空发射口。
- `scripts/ncu.sh` 支持用 `--` 分隔 ncu 参数与程序参数（如 `... --set full -- 20000 occ2`）；收集 stall 用 `--metrics smsp__average_warps_issue_stalled_*_per_issue_active.ratio`。
- 源码/SASS 对照：`ncu --page source --print-source cuda,sass --metrics smsp__inst_executed.sum ...`。
- **`ldmatrix` 的 bank conflict 极其致命**：smem 行主序数组若不加 padding，`ldmatrix` 一次读 8 行会撞 bank（B 行距 128 bf16 时 8 行全在 bank 0，实测 3565 万次冲突）。给行距 **+8 个元素**（`LDA=BK+8`、`LDB=BN+8`）即可归零，同一 kernel 从 76→221 TFLOPS。写任何用 ldmatrix 的 kernel 先查 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`。
- **WMMA `load_matrix_sync` 在 sm_90 上是负优化**：抽象导致共享内存流量失控（L1/TEX 77%、Compute 17%），同尺寸只有 62 TFLOPS；手写 `mma.m16n8k16` + `ldmatrix` 能到 221。入门可讲 WMMA，但性能路径必须自己控布局。
- **`ldmatrix.x4.trans` 取 B 片段**：B 在 smem 里存成行主序 `[K][N]`，mma 需要 `.col` 布局，用转置 ldmatrix 让硬件顺带转置；地址 lane 公式：`row=(lane&7)+((lane>>3)&1)*8, col=(lane>>4)*8`。
- **静态 smem 上限 48KB**：`__shared__` 数组超过 0xc000 会 ptxas 报错；BK=64 的 tc kernel 会超。要更大需 `cudaFuncSetAttribute` + 动态 smem。
- **模板化 kernel 的「通用循环 loader」会悄悄多花寄存器**：把第 09 篇的单发 float4 载入改成
  `for (q = t; q < N4; q += T)` 的通用循环后，同一 128×128×8 GEMM 从 118/128 寄存器涨到 168，
  occupancy 25%→12.5%，算力掉 ~35%。当某配置下每线程恰好搬 1 个 float4 时用 `if constexpr`
  走单发写法可恢复。**写通用模板后务必用 `NVCC_FLAGS="-Xptxas -v"` 核对寄存器数**，别只看功能正确。
- **epilogue 融合的寄存器开销常常为 0**：第 14 篇把 `store_acc` 模板化加 bias+GELU/ReLU 后，
  三个实例（MODE=0/2/3）都是 128 寄存器、0 spill，occupancy 与瓶颈指标逐项不变。别怕融合
  「拖慢」GEMM；但每次都要用 `-Xptxas -v` 复核，激活换成更复杂的函数未必还免费。
- **对照「两 kernel vs 融合」时，独立 epilogue kernel 必须自己先向量化**（float4 + grid-stride，
  第 14 篇跑 ~4 TB/s），否则标量版只有几百 GB/s，会得到虚高的融合收益（稻草人对比）。
- **device 辅助函数要在 host 参考实现里复用**（如 GELU）时，记得标 `__host__ __device__`，
  否则 nvcc 报 `calling a __device__ function from a __host__ function is not allowed`。
- **CUDA event 口径与 ncu 口径会有差**：第 14 篇独立 epilogue kernel event 测 8.4 µs、ncu 报
  11.26 µs（含 replay）。写文章时同一结论尽量用同一口径，或两者都标注。
- **别被「读放大」骗了**：第 15 篇 MLA，`naive` 版读放大 131072×，但 `c_kv` 只有 1~5MB、常驻
  50MB L2，ncu 实测 `DRAM Throughput` 仅 0.71%——真正瓶颈是 `Compute (SM)` 83.5%。判断优化方向
  **先看 ncu 的 DRAM%**，DRAM 低就别做访存复用。
- **attention/MLA 的 prefill 是算力受限**：`AI_ideal = 2H(DC+DR+DV)/(DC+DR)`，H=128/576/512 时≈242，
  接近 bf16 TC ridge（295），但标量 FFMA ridge 只有 ~20。标量实现天花板 <7%，必须上 Tensor Core。
- **按 head 复用 KV 会把寄存器吃爆**：第 15 篇 `head_reuse<8>` 达 255 寄存器 + 328B 栈溢出，比
  `HG=2` 还慢。`-Xptxas -v` 核对；换来的复用若不减少 DRAM（见上条），纯属负优化。
- **三 kernel 物化 attention 中间量的隐性税**：`S`(fp32)+`P`(bf16) 各写读一遍，流量 `∝H·S²`。第 15
  篇 S=4096 时 25.8GB≈7.7ms，占 TC 总时长 21%。能融合就别物化。
