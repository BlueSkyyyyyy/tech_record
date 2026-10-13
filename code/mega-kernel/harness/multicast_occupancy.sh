#!/usr/bin/env bash
# P168-1（§93）复现：TMA multicast × 占用率 / 寄存器预算——补 §19.6 / §86 明留的组合。
#   每配置一次独立进程调用：一次 cluster launch failure 会 sticky 污染 CUDA context。
#   目标卡 H100 / sm90a，nvcc 13.2。DEV 选空闲 device（可 DEV=n 覆盖）。
set -u
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/multicast_occupancy.cu
BIN=/tmp/multicast_occupancy

echo "########## §93 TMA multicast × 占用率 / 寄存器预算（P168-1） ##########"
echo "# DEV=$DEV  nvcc -O3 -arch=sm_90a"
echo
echo "===== ptxas -v：mcast_nat 与 __maxnreg__ 变体的寄存器 / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" -lcuda 2>/tmp/mcast_occ_ptxas.txt || exit 1
grep -E "Function properties for _Z|bytes stack frame|registers, used" /tmp/mcast_occ_ptxas.txt
run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$@"; }

echo
echo "===== §O 占用率表：不同 SMEM 足迹下各变体的 occ（blocks/SM）====="
for L in 2 4 8 16 24; do run O $L 0; done
for pad in 32 64 96 128 160; do run O 2 $pad; done

echo
echo "===== §B（Q1）占用率用 grid 钉死（小 SMEM=L2=16KB 不 binding，chunk=8KB, R=300, C=4）：grid=numSM·O, O=1/2/4/8 ====="
NSM=$(CUDA_VISIBLE_DEVICES="$DEV" "$BIN" O 2 0 | sed -n 's/.*numSM=\([0-9]*\).*/\1/p')
echo "# numSM=$NSM"
for O in 1 2 4 8; do
  g=$((NSM*O))
  for m in 0 1; do run B 4 $g 8 300 $m 2 0 0; done
done

echo
echo "===== §C（Q2）占用率用 SMEM padding 钉死（固定 grid=numSM·8, L=2, chunk=8KB, R=300, C=4）====="
g=$((NSM*8))
for pad in 0 16 48 96 144 200; do
  for m in 0 1; do run B 4 $g 8 300 $m 2 $pad 0; done
done

echo
echo "===== §D（Q3）寄存器预算 __maxnreg__(R)（固定 grid=numSM·8, L=2, pad=0, chunk=8KB, R=300, C=4）====="
g=$((NSM*8))
for r in 0 24 32 40 64 96; do
  for m in 0 1; do run B 4 $g 8 300 $m 2 0 $r; done
done

echo
echo "===== §E（Q4）ncu：multicast 省 L2 请求在不同占用率下是否成立（C=4, chunk=8KB, L=2, R=200）====="
for O in 1 2 4 8; do
  g=$((NSM*O))
  for m in 0 1; do
    echo "\$ ncu ... B 4 $g 8 200 $m 2 0 0  (O=$O mode=$m)"
    CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all -k regex:mcast_nat -c 1 \
      --metrics lts__t_bytes.sum,lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum \
      "$BIN" B 4 $g 8 200 $m 2 0 0 2>&1 | grep -E 'Metric Name|lts__|dram__'
  done
done
echo
echo "-- §E2 复现 §19 的 C=8/L=24/grid=128 配置（大 request 流）：multicast 的 L2 收益随 C/L 放大 --"
for m in 0 1; do
  echo "\$ ncu ... B 8 128 8 200 $m 24 0 0  (C=8 L=24 mode=$m)"
  CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all -k regex:mcast_nat -c 1 \
    --metrics lts__t_bytes.sum,lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum \
    "$BIN" B 8 128 8 200 $m 24 0 0 2>&1 | grep -E 'Metric Name|lts__|dram__'
done
