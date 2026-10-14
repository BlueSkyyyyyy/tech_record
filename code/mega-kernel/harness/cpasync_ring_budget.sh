#!/usr/bin/env bash
# P53-1 复现：**worker「取 task」深环的第三种载体（16 B cp.async）与 §41 的 TMA bulk 对跑；
#   以及 cp.async「不占寄存器」对下游 task 执行的实际收益。**
# 补齐 P52-1（docs/04 §41.8）明留的两条尾巴：
#   「载体固定为 TMA bulk + mbarrier，未同时跑 16 B `cp.async` 环（其预取机制 `wait_group` 是软件计数，
#     深环写法见 §40.8 的 drain 坑）；未测 `cp.async` 释放的寄存器对下游 task 执行的实际收益。」
# 真实锚点（mirage 快照 f9eb70c）：worker 取 task = 全线程协作 16 B cp.async 进 SMEM 单缓冲
#   （persistent_kernel.cuh:1063-1078，cp_async_wait<0>）；load_smem = cp.async.ca.shared.global.L2::128B
#   （tasks/common/copy_sm80.cuh:44-57）；TaskDesc 352 B（论文 §6.1 行 1365）；论文 §5.3 pre-fetching
#   （行 1364-1368）/ paged shared memory（行 1313-1325）。
# 机制：G 个常驻消费者 block，全局一大块自描述描述符（DW=88 → 352 B，≫L2）。每 block 处理自己的 batch
#   序列，batch 内 K 条连续描述符。两条取数载体：bulk（TMA bulk + 硬件 mbarrier，运行时 D）与
#   ca（每 lane 16 B cp.async.cg + 软件 wait_group(D-1)，模板 D）；环形缓冲 = D 个槽（SMEM = D·K·352 B）。
# 四个 Section：A 正确性 / B 载体对照 K×SMEM 平面 / C 环深 D 扫描 / D 载体对下游 task 的寄存器足迹。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/cpasync_ring_budget.cu
BIN=/tmp/cpasync_ring_budget
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench
