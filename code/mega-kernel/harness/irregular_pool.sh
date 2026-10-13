#!/usr/bin/env bash
# P44-1 复现：**有界 ready 池 × 不规则依赖 DAG**——把 P43-1（docs/04 §32）的规则分层 DAG
# 推广到「每层宽度不等 / 跨层 skip / 多级扇入 / 层号乱序」，并给出普适的死锁边界量 B(π)
# （贪心模拟的峰值池占用），对照 §32 的 (L-1)*W 闭式。
# 接续 DeepGEMM Mega MoE `get_num_l1_warmup_waves()`（scheduler/mega_moe.cuh:15-45）与
# `get_num_max_live_pool_blocks()`（:47-80）、MPK §5.1 事件驱动入队（docs/06 §3）、
# MPK 依赖分析器 Kahn linearization（docs/08 §5）。
# 目标卡 H100 / sm90，用 DEV 选择空闲 device（默认 3）。
#
#   bench : §A 不规则宽度 / §B 跨层 skip / §C 层号乱序（level 失效）/ §D 多级扇入 gated vs topo / §E warmup。
#   run   : 单次（policy shape CAP G L W0 FIN span scramble D iters）。
#
# 安全：设备侧无进度守门狗（clock64）检测死锁并置 flag，kernel 正常返回（host 不挂）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/irregular_pool.cu
BIN=/tmp/irregular_pool
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

echo "===== P44-1 irregular_pool bench ====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 600 "$BIN" bench 30 64
echo

echo "===== §A 精扫：shrink（W=[16,15,14,13], N=58, B=N-W0=42）边界应在 CAP=42/43 翻转 ====="
for CAP in 41 42 43; do
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run wide shrink "$CAP" 256 4 16 2 1 0 0 20000 | sed 's/^/  /'
done
echo

echo "===== §C 精扫：层号乱序 scramble=3（wide 此时 B=32）边界应在 CAP=32/33 翻转 ====="
for CAP in 31 32 33; do
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run wide reg "$CAP" 256 4 16 2 1 3 0 20000 | sed 's/^/  /'
done
echo

echo "===== §C 关键：scramble=1 时 level(层号升序) B=16——CAP=1 死锁、CAP=17 才活；topo 恒 B=0、CAP=1 即活 ====="
for cfg in "level 1" "level 17" "topo 1"; do
  set -- $cfg
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run "$1" reg "$2" 64 4 16 2 1 1 0 20000 | sed 's/^/  /'
done
echo

echo "===== §B 精扫：跨层 skip span=4，wide 边界仍 = B+1 = 43 ====="
for CAP in 42 43; do
  CUDA_VISIBLE_DEVICES="$DEV" timeout 60 "$BIN" run wide shrink "$CAP" 64 4 16 2 4 0 0 20000 | sed 's/^/  /'
done
echo
