#!/usr/bin/env bash
# P52-1 复现：**worker「取 task」的 SMEM 预算：批量 K × 环形深度 D 的联合约束 + 与 task 执行争 SMEM 的占用率代价。**
# 补齐 P51-1（docs/04 §40.8）明留的第一条尾巴：
#   「未测：`K` 与 SMEM 容量的联合约束、`cp.async` 释放的寄存器对下游 task 执行的实际收益、
#     与 DeepGEMM SM100 路径对跑（SM100 专属，本机不具备）。」
# 真实锚点（mirage 快照 f9eb70c）：worker 取 task 用全线程协作 16B cp.async 进 SMEM 单缓冲
#   （persistent_kernel.cuh:1063-1078，cp_async_wait<0>）；硬件 mbarrier + TMA bulk 形态在 task 单体内部
#   （tasks/hopper/tma_2d.cuh:119；barrier.cuh:42/59/75）；TaskDesc 352 B（论文 §6.1 行 1365）；
#   论文 §5.3 pre-fetching（行 1364-1368）/ paged shared memory（行 1313-1325）。
# 机制：G 个常驻消费者 block，全局一大块自描述描述符（DW=88 → 352 B，≫L2）。每 block 处理自己的 batch
#   序列，batch 内 K 条连续描述符。取数载体 = TMA bulk + 硬件 mbarrier（§40 的最优载体）；
#   环形缓冲 = D 个槽（SMEM = D·K·352 B），槽 slot=t%D，用 try_wait.parity 奇偶相位等待。
# 五个 Section：A 正确性 / B 环深 D 扫描 / C K×SMEM 联合约束 / D SMEM 上限 / E 占用率与带宽。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/fetch_smem_budget.cu
BIN=/tmp/fetch_smem_budget
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench
