#!/usr/bin/env bash
# P58-1 复现：**取 task × 动态生产者**——描述符由设备侧 producer（等价 scheduler）现场生成进
# **有界环 + 背压**，worker 用 TMA bulk 预取；对照「host 预写大数组」（= §39–§46 口径）。
# 补齐 §39.7/§40.8/§42.7/§43.7/§44.7/§45.7/§46.7 每一节都明留的同一条尾巴「未叠动态生产者」。
# 五个 Section：A 正确性 / B 控制面来源（HBM 数组 vs L2 环）/ C 生产者成本 PW /
#   D 有界环容量 CAP（背压/死锁）/ E 接数据面后结论是否变。
# 真实锚点（mirage 快照 f9eb70c）：scheduler 现场入队 execute_scheduler:1254-1523；worker 取 task
#   fr ld_acquire + 批量 cp.async:1063-1078；队列有界 per_worker_queue_len=8192:1790-1791；
#   MPSC push 连续 CAS 放行:1206-1228；论文 §5.1 行 1175-1231 / §5.2 行 1232-1308。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/dyn_producer_fetch.cu
BIN=/tmp/dyn_producer_fetch
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" bench
