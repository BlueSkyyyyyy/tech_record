#!/usr/bin/env bash
# P51-1 复现：**多 worker（多 block）取 task 的描述符带宽标度 + `cp.async` 的 TMA/mbarrier 形态值多少？**
# 补齐 P50-1（docs/04 §39.7）明留的两条尾巴：
#   (1) 「单 block ⇒ 测的是『取一批的延迟』而非『访存带宽』…要多 worker 的带宽标度需另做多 block 实验」；
#   (2) 「`mbarrier` 形态未测：cp.async 用 commit_group/wait_group 软件计数，未测硬件 mbarrier（TMA 风格）」。
# 真实锚点（mirage 快照 f9eb70c）：worker 用所有线程协作 16B cp.async 把 TaskDesc 批量搬进 SMEM
#   （persistent_kernel.cuh:1063-1078，单缓冲 cp_async_wait<0>）；硬件 mbarrier + TMA bulk 形态在 task
#   单体内部（tasks/hopper/tma_2d.cuh:119；barrier.cuh:42/59/75）；TaskDesc 352 B（论文 §6.1 行 1365）；
#   论文 §5.3「Pre-fetching task descriptions」（行 1364-1368）。
# 机制：G 个常驻消费者 block，全局一大块自描述描述符（≫L2），batch（K 条/批）round-robin 分给 G 个 block。
#   唯一变量 = 取数路径：scalar / vec4 / cpasync(单缓冲) / cpasync_pf(双缓冲) / bulk(TMA+mbarrier 单) / bulk_pf(双)。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/multi_worker_desc_fetch.cu
BIN=/tmp/multi_worker_desc_fetch
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench
