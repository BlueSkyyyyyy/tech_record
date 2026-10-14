#!/usr/bin/env bash
# P50-1 复现：**worker 侧「批量取 task」值多少钱——`cp.async` 批量加载 TaskDesc 进 SMEM
#   相对寄存器/向量读的收益是多少？**
# 补齐 P49-1（docs/04 §38.7）明留的尾巴：「`cp.async` 只是等效语义（§38 用普通 ld.relaxed 读，
#   未实测 cp.async 批量加载本身的收益——那是另一条线）」「warp 下 cp.async/mbarrier 的批量加载形态」。
# 真实锚点（mirage 快照 f9eb70c）：worker 用所有线程协作 16B cp.async 把 TaskDesc 批量搬进 SMEM
#   （persistent_kernel.cuh:1063-1078）；load_smem = cp.async.ca.shared.global.L2::128B（16B，预取 128B，
#   绕过寄存器；tasks/common/copy_sm80.cuh:44-57）；TaskDesc（runtime_header.h:296-336，alignas(16)，~128B）。
# 机制：单常驻 kernel、单 block；全局预写一大块自描述描述符（≫L2），消费者按批取进来做 XOR 校验。
#   唯一变量 = 取数路径：scalar(逐 4B)/vec4(128-bit)/cpasync(SMEM 单缓冲)/cpasync_pf(双缓冲预取)。
# 目标卡 H100 / sm90，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/cpasync_desc_fetch.cu
BIN=/tmp/cpasync_desc_fetch
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 1200 "$BIN" bench
