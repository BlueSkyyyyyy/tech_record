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
