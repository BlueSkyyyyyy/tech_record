#!/usr/bin/env bash
# P33-1 复现：2SM 协作的 sm90 投影 —— 一对 CTA 的「独立 / 复制(multicast) / 分区(split)」载入对照
#           （目标卡 H100 / sm90a）。每个配置单独一次进程调用（cluster launch failure 会 sticky）。
set -u
DEV=${DEV:-2}
SRC=~/proj/tech_record/code/mega-kernel/src/two_sm_load.cu
BIN=/tmp/two_sm_load
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" -lcuda 2>/dev/null || exit 1
run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$@"; }

echo "== A. 单 pair（grid=2）扫 chunk 大小：mode0 indep / mode1 mcast / mode2 split（R=500）=="
for KB in 4 8 16 32; do
  for m in 0 1 2; do run L 2 2 $KB 500 $m; done
done
echo
echo "== B. 多 pair（grid=128，64 pair）扫 chunk 大小（R=500）=="
for KB in 4 8 16 32; do
  for m in 0 1 2; do run L 2 128 $KB 500 $m; done
done
echo
echo "== C. 多 pair（grid=128）扫轮数 R（chunk=8KB）=="
for R in 100 500 2000; do
  for m in 0 1 2; do run L 2 128 8 $R $m; done
done
echo
echo "== D. ncu：源侧读取流量（L2 read sectors / L2 bytes / DRAM read），grid=128 chunk=8KB R=200 =="
for m in 0 1 2; do
  echo "\$ ncu ... L 2 128 8 200 $m"
  CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all -k regex:load2sm_kernel -c 1 \
    --metrics lts__t_bytes.sum,lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum \
    "$BIN" L 2 128 8 200 $m 2>&1 | grep -E 'Metric Name|lts__|dram__'
done
