#!/usr/bin/env bash
# P59-1 复现：**动态生产者的多块并行**——把 §47 的单块 producer 扩成 P 个 SM，量 gen/sched 代价
# 是否按 1/P 下降、在哪饱和、以及「生成必须落在调度」的结论是否被多块翻案。
# 补齐 §47.8 明留的尾巴①（gen 单块，分散到多块则代价按块数下降）与尾巴③（sched 1.36ms 是单块上界，
# MPK 实际用 16 scheduler SM）。
# 五个 Section：A 正确性（P 段子环+背压）/ B 标度（gen/sched vs P）/ C 额外计算 PW 随 P /
#   D 多 producer 有界环容量 / E 接数据面后结论是否变。
# 真实锚点（mirage 快照 f9eb70c）：H100 128 worker/16 scheduler（utils.py:25-29）；
#   execute_scheduler 现场入队 persistent_kernel.cuh:1254-1523；队列有界 per_sched_queue_len=4096:1790-1791；
#   MPSC push 连续 CAS:1206-1228；论文 §5.1 行 1168 / §5.2 行 1232-1308。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/dyn_producer_scale.cu
BIN=/tmp/dyn_producer_scale
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" bench
