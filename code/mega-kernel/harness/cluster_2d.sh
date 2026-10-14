#!/usr/bin/env bash
# P31-1 复现：2D/3D Thread Block Cluster 的 rank 布局 / DSM / TMA multicast 子集掩码（目标卡 H100 / sm90a）。
# 每个配置单独一次进程调用——一次 cluster launch failure 会污染 CUDA context（sticky）。
# 用 DEV 选择空闲 device。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/cluster_2d.cu
BIN=/tmp/cluster_2d
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" -lcuda 2>/dev/null || exit 1
run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$@"; }

echo "== A. rank 布局：验证 rank == x + y*cx + z*cx*cy（打印每 block 的 rank）=="
run A 2 2 1 4 4 1
run A 4 2 1 8 4 1
run A 4 4 1 8 8 1
run A 2 2 2 4 4 2
echo
echo "== S. 2D cluster 的 DSM 环形段位移（K=4, R=200, grid=128；对照 P29-1 §18 1D）=="
for cd in "1 2" "2 2" "2 4" "4 4"; do run S $cd 4 200; done
echo "  -- 扫 K（cluster=(2,2)）--"
for K in 1 2 4 8; do run S 2 2 $K 200; done
echo
echo "== M. 2D cluster 的 TMA multicast：整群 vs 逐行子集（chunk=8KB, R=500）=="
echo "  -- cluster=(2,2), grid=(8,8) --"
for m in 0 1 2; do run M 2 2 8 8 8 500 $m; done
echo "  -- cluster=(4,4), grid=(8,8) --"
for m in 0 1 2; do run M 4 4 8 8 8 500 $m; done
echo "  -- chunk 扫描（cluster=(4,4), grid=(8,8), mode=2）--"
for KB in 4 8 16 32; do run M 4 4 8 8 $KB 500 2; done
echo
echo "== P. cluster.sync() 成本（2D 形状；对照 P29-1 §18 1D）=="
run P 1 2 8 8 200
run P 2 2 8 8 200
run P 2 4 8 8 200
run P 4 4 8 8 200
echo
echo "== C2. cluster 容量/合法性（2D/3D；硬上限 16，各维 grid 须为 cluster 维整数倍）=="
run C2 4 4 1 8 8 1     # C=16 ok
run C2 2 2 2 4 4 2     # C=8 3D ok
run C2 4 4 2 8 8 2     # C=32 超上限
run C2 8 8 1 8 8 1     # C=64 超上限
run C2 4 4 1 8 6 1     # grid y=6 非 4 倍数 -> 非法
echo
echo "== D. ncu：源侧读取流量 mode0/1/2（cluster=(4,4), grid=(8,8), chunk=8KB, R=200）=="
for m in 0 1 2; do
  echo "\$ ncu ... M 4 4 8 8 8 200 $m"
  CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all -k regex:bcast2d_kernel -c 1 \
    --metrics lts__t_bytes.sum,lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum \
    "$BIN" M 4 4 8 8 8 200 $m 2>&1 | grep -E 'Metric Name|lts__|dram__'
done
