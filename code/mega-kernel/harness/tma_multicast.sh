#!/usr/bin/env bash
# P30-1 复现：cluster 内 TMA multicast 广播（目标卡 H100 / sm90）。
# 每个配置单独一次进程调用——一次 cluster launch failure 会污染 CUDA context（sticky）。
# 用 DEV 选择空闲 device。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/tma_multicast.cu
BIN=/tmp/tma_multicast
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" -lcuda || exit 1
run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$@"; }

echo "== A. 单 cluster（grid=C）扫 cluster 大小：mode0=每 CTA 各读 / mode1=rank0 multicast（chunk=8KB, R=500）=="
for C in 2 4 8 16; do
  run B $C $C 8 500 0
  run B $C $C 8 500 1
done
echo
echo "== A2. 多 cluster（grid=128）扫 C（chunk=8KB, R=500）=="
for C in 4 8 16; do
  run B $C 128 8 500 0
  run B $C 128 8 500 1
done
echo
echo "== B. chunk 扫描（C=8, grid=128, R=500）=="
for KB in 4 8 16 32; do
  run B 8 128 $KB 500 0
  run B 8 128 $KB 500 1
done
echo
echo "== C. cluster.sync() 成本 sanity（对比 P29-1）=="
for C in 2 8 16; do run P $C 0 200; done
echo
echo "== D. ncu：源侧读取流量（L2 read sectors / L2 bytes / DRAM read）mode0 vs mode1（C=8, grid=128, chunk=8KB, R=200）=="
for m in 0 1; do
  echo "\$ ncu ... B 8 128 8 200 $m"
  CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all -k regex:bcast_kernel -c 1 \
    --metrics lts__t_bytes.sum,lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum \
    "$BIN" B 8 128 8 200 $m 2>&1 | grep -E 'Metric Name|lts__|dram__'
done
