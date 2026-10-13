#!/usr/bin/env bash
# P62-1 复现：**动态生产者 TMA store 的「正确握手 + 流水」**——补齐 §50.7（P61-1）明留的「最值得做的
# 下一条尾巴」：serialized 的 472 GB/s 来自「每批 wait_group 0」这一过保守握手；一个**流水但正确**的形态
# （FIFO commit + 发布第 j 批 head 前只 wait_group(win-1)、head 落后 win 批）能否拉回 rolling 的 ~722-828。
# 唯一新变量 = 发布握手：serialized（win=1）vs pipelined（win>=2）；载体固定 TMA bulk store。
# 四个 Section：A 正确性（static/ser/pipe win2..4 逐位一致）/ B gen 握手 × P（有 worker）/
#   C 纯写吞吐（ser/pipe/rolling/v4/coop × P）/ D 接数据面后 ser vs pipe。
# 真实锚点（mirage 快照 f9eb70c）：worker 取 task 用 TMA bulk 1D；scheduler 只做「事件→task id」映射入队
#   （persistent_kernel.cuh:1254-1523），payload 来自编译期预计算的 FullTaskDesc 数组（runtime.cc），
#   **不在设备侧现场生成**——本节给 MPK「relaxed 写数据 + 单条 release 推游标 + 批量摊薄」以同构理由。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/dyn_producer_tma_pipe.cu
BIN=/tmp/dyn_producer_tma_pipe
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 5400 "$BIN" bench
