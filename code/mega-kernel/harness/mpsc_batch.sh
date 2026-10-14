#!/usr/bin/env bash
# P49-1 复现：**MPSC in-kernel 队列的「批量出队 + 多线程消费者 + 真实 task 描述符」**
# 补齐 P48-1（docs/04 §37.7）明留的两条尾巴：「消费者是单线程，每项 ~2.3µs 同步往返主导成本」
#   与「payload 是纯 uint32、未做真实 task 描述符」。
# 真实锚点（mirage 快照 f9eb70c）：worker 拉取 task 不是逐条——thread0 先读 task 区间，
#   再用 128-bit cp.async 把 TaskDesc（runtime_header.h:296-336，alignas(16)，~128B）
#   批量搬进 shared memory（persistent_kernel.cuh:1063-1078）；队列容量 per_worker_queue_len=8192
#   （:1790-1791）；scheduler 每处理完一个 batch 走 prepare_next_batch（:1378-1423）。
# 机制：单常驻 kernel，grid=P+1（前 P block=生产者，末 block=消费者）。生产者协议 = §37 `mpk`
#   （占位 release atomicAdd → relaxed 写描述符 → 连续 CAS 放行）。消费者三档：
#   serial（1 线程逐条）/ batchK（1 线程一次等 K 条、一次放行 K 槽）/ warpK（1 warp，每线程并行校验 1 条）。
# 描述符布局（真实）：w0=task id、w1=op_type(id%4)、w2=layer(id/4)、w3=iter、w[i>=4]=mix(id,i)。
# 目标卡 H100 / sm90，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/mpsc_batch.cu
BIN=/tmp/mpsc_batch
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 1200 "$BIN" bench
echo
echo "===== 追加：正确性连跑 3 次（DW=32=128B，P=8，batch/warp K=32） ====="
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 180 "$BIN" run 100000 1024 32 8 1 32 1 1 | sed 's/^/  /'; done
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 180 "$BIN" run 100000 1024 32 8 2 32 1 1 | sed 's/^/  /'; done
