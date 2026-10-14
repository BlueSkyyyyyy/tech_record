#!/usr/bin/env bash
# P43-1 复现：**有界 ready 池 × 多级依赖 DAG**——死锁边界的「深度×层宽」归因，以及
# 拓扑层序（level）/ 依赖门控（gated）两个更优准入策略。
# 接续 P42-1（docs/04 §31，src/ring_deadlock.cu）：把单层 fan-in 推广到分层 DAG（L 层 × W 宽 × FIN 扇入）。
# 对应 DeepGEMM Mega MoE `get_num_l1_warmup_waves()`（scheduler/mega_moe.cuh:15-45）与
# `get_num_max_live_pool_blocks()`（:47-80）、MPK §5.1 事件驱动入队（docs/06 §3）。
# 目标卡 H100 / sm90，用 DEV 选择空闲 device（默认 3）。
#
#   bench : §A 死锁边界 / §A2 边界随深度 L / §B warmup 深度下界 / §C 吞吐 vs 容量 / §D worker 扫描 / §E 扇入 FIN。
#   run   : 单次（policy CAP G L W FIN D iters）。
#
# 安全：设备侧无进度守门狗（clock64）检测死锁并置 flag，kernel 正常返回（host 不挂）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/layered_pool.cu
BIN=/tmp/layered_pool
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

echo "===== P43-1 layered_pool bench ====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 600 "$BIN" bench
echo

echo "===== §A 边界精扫（L=4,W=16）：wide 应在 CAP=(L-1)*W=48 两侧翻转（48 死、49 活）====="
for CAP in 47 48 49 50; do
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run wide "$CAP" 64 4 16 2 0 20000 | sed 's/^/  /'
done
echo

echo "===== §A3 联合边界精扫（L=8,W=16 → B=(L-1)*W=112；wide 须 CAP>B 且 G>B）====="
for cfg in "112 256" "113 256" "256 112" "256 113"; do
  set -- $cfg
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run wide "$1" "$2" 8 16 2 0 20000 | sed 's/^/  /'
done
echo
