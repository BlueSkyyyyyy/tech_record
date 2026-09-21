# kernel-opt — CUDA 算子调优系列的自驱工作流程

## 触发场景

用户说「继续整理/推进算子优化系列」「CUDA 算子调优」（本仓库的长期自驱任务）。
本技能把「写代码 → 本地实测 → ncu 剖析 → 写文章 → 发布」串成可重复的循环。

> **无人值守模式**：本技能常被 `code/kernel-opt/scripts/autopilot.sh` 以无头会话反复调用，
> 每轮只做一篇文章增量。若你是被 autopilot 唤醒的：全程不要请求人工确认（别用 question 类工具），
> 做完一篇就停，更新 ROADMAP 后本轮即可结束；整个系列完成时 `touch code/kernel-opt/AUTOPILOT_STOP`。

## 前置条件

- 已读根目录 `agent_guide.md`（目录约定、已知坑）。
- 已读系列控制面板 `code/kernel-opt/ROADMAP.md`（**每次先读它，从「下一步」做起，做完更新它**）。
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
5. **回写控制面板**：更新 `ROADMAP.md` 状态、`README.md`/`code/README.md` 索引，
   把新踩的坑追加到本文件或 `agent_guide.md`。
6. **提交**：每 1~2 篇 commit & push 一次，信息格式 `post:` / `bench:` / `skill:`。

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
