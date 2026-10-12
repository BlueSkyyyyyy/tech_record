#!/usr/bin/env bash
# P29-1 复现：Thread Block Clusters + DSM（目标卡 H100 / sm90）。
# 每个配置单独一次进程调用——一次 cluster launch failure 会污染 CUDA context（sticky），
# 必须隔离。用 DEV 选择空闲 device。
set -u
DEV=${DEV:-1}
SRC=~/proj/tech_record/code/mega-kernel/src/cluster_dsm.cu
BIN=/tmp/cluster_dsm
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1
run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$@"; }

echo "== A. DSM 远端读 vs 本地 SMEM 读（C=1 为本地基线；grid=128, SB=16KB, R=20000）=="
for C in 1 2 4 8 16; do run A $C 128 4096 20000; done
echo
echo "== HBM 参照：plain 全局流式读（grid=132, 256 MB > 50 MB L2）=="
run Lg 132 256
echo
echo "== B. 跨 SM ring 段位移（ping-pong, 1 屏障/轮，grid=128, R=200）：段放 HBM(scratch) vs DSM =="
for K in 1 2 4 8; do
  for C in 2 4 8; do run Sh $C 128 $K 200; run Sd $C 128 $K 200; done
done
echo
echo "== C1. cluster.sync() 成本（grid=128, R=200）=="
for C in 2 4 8 16; do run C1 $C 128 200; done
echo
echo "== C2. cluster 容量：单 cluster 尝试 launch（portable<=8, non-portable<=16）=="
for C in 1 2 4 8 16 32; do run C2 $C $C; done
echo
echo "== C2b. 硬边界：C 硬上限 16；grid 非 C 的整数倍非法 =="
run C2 8 132          # 132 % 8 != 0 -> invalid cluster size
run C2 16 64          # 64  % 16 == 0 -> ok
run C2 4 132          # 132 % 4 == 0 -> ok（非 GPC 平铺也可）
echo
echo "== C3. grid.sync() 成本（cooperative, grid=132, R=200）=="
run C3 132 200
