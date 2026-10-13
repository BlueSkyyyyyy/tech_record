#!/usr/bin/env bash
# P48-1 复现：**MPSC（多生产者单消费者）in-kernel 队列的「发布协议」**
# 补齐 P47-1（docs/04 §36.6）明留的尾巴：「环仍是合成模型……未测多生产者（MPSC）」。
# 真实锚点：MPK 的 sched_queue 是 MPSC——128 个 worker 用 get_rand_sched_id()（persistent_kernel.cuh:857）
#   把事件推到同一个 scheduler 队列，协议 = 占位(atom.add.release) → relaxed 写 payload
#   → 连续 CAS 放行 atom.cas.release(last_ready,pos,pos+1)（persistent_kernel.cuh:1206-1228）。
# 机制：单常驻 kernel，grid=P+1（前 P block=生产者、末 block=消费者，各 1 线程），全局环形缓冲 CAP 槽，
#   payload 自描述（首字 = 占位号 r、其余 = mix(r,i)），消费者按占位号 0,1,2,... 顺序逐字校验。
# 模式：mpk / pertile（逐槽 release）/ reserve_count（非连续完成计数，结构错误）/
#   pub_then_write（占位即就绪，结构错误）/ pubslp（拉长窗口正对照）/ mpk_relaxed / spsc（P=1 无 release 基线）。
# 目标卡 H100 / sm90，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/mpsc_ring.cu
BIN=/tmp/mpsc_ring
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 900 "$BIN" bench
echo
echo "===== 追加：正确协议连跑 3 次（稳定性，P=8） ====="
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" ring 50000 256 8 8 0 1 | sed 's/^/  /'; done
echo "===== 追加：结构不严谨替代自然时序连跑 3 次（多为侥幸 0 错） ====="
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" ring 50000 256 8 8 2 1 | sed 's/^/  /'; done
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" ring 50000 256 8 8 3 1 | sed 's/^/  /'; done
echo "===== 追加：决定性正对照 stall0（低号槽延迟）——错误协议连跑 3 次应稳定读错，正确协议恒 0 ====="
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" ring 20000 256 8 8 2 1 1 | sed 's/^/  /'; done
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" ring 20000 256 8 8 3 1 1 | sed 's/^/  /'; done
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" ring 20000 256 8 8 0 1 1 | sed 's/^/  /'; done
for i in 1 2 3; do CUDA_VISIBLE_DEVICES="$DEV" timeout 120 "$BIN" ring 20000 256 8 8 1 1 1 | sed 's/^/  /'; done
