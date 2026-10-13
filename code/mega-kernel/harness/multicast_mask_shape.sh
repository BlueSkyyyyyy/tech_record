#!/usr/bin/env bash
# P170-1（§95）复现：TMA multicast 掩码的「非连续形状」与「分裂发送」——
#   补 §94.7 明留的「未测非连续 / 重叠子集掩码」。
#   每配置一次独立进程调用：一次 cluster launch failure 会 sticky 污染 CUDA context。
#   目标卡 H100 / sm90a，nvcc 13.2。DEV 选空闲 device（可 DEV=n 覆盖）。
set -u
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/multicast_mask_shape.cu
BIN=/tmp/multicast_mask_shape

echo "########## §95 TMA multicast 掩码形状 + 分裂发送（P170-1） ##########"
echo "# DEV=$DEV  nvcc -O3 -arch=sm_90a"
echo
echo "===== ptxas -v：shape_kernel 的寄存器 / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" -lcuda 2>/tmp/mcast_shape_ptxas.txt || exit 1
grep -E "Function properties for|bytes stack frame|registers, used" /tmp/mcast_shape_ptxas.txt
run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$@"; }

NSM=$(CUDA_VISIBLE_DEVICES="$DEV" python3 -c 'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null)
if [ -z "${NSM:-}" ]; then NSM=132; fi
echo "# numSM=$NSM"

echo
echo "===== §A（Q1）形状不变性：contig(0) vs stride(1)，固定 g，C=8, chunk=8KB, L=2, grid=numSM*4, R=200, it=20 ====="
for g in 2 4 8; do for p in 0 1; do run B 8 $((NSM*4)) 8 200 2 $p $g 20; done; done

echo
echo "===== §B（Q2）分裂发送：单发起者(0/1) vs 双发起者轮流(2/3)，C=8, chunk=8KB, L=2, grid=numSM*4, R=200, it=20 ====="
for g in 2 4 8; do for p in 0 2 1 3; do run B 8 $((NSM*4)) 8 200 2 $p $g 20; done; done

echo
echo "===== §C（Q3）C×pattern 交叉：C=4/8/16，g=min(C/2,8) 下 contig/stride/contig-split（chunk=8KB, L=2, grid=numSM*4, R=200, it=20）====="
for Cg in "4 2" "8 4" "16 8"; do
  set -- $Cg; C=$1; g=$2
  for p in 0 1 2; do run B $C $((NSM*4)) 8 200 2 $p $g 20; done
done

echo
echo "===== §D（Q4）ncu：离散掩码源侧 L2 读 sector 是否仍 = delivered/(32*g)（C=8, chunk=8KB, L=2, R=100, grid=numSM*4）====="
for pg in "0 4" "1 4" "0 8" "1 8"; do
  set -- $pg; p=$1; g=$2
  echo "\$ ncu ... B 8 grid=$((NSM*4)) 8 100 2 $p $g 1  (pat=$p g=$g)"
  CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all -k regex:shape_kernel -c 1 \
    --metrics lts__t_bytes.sum,lts__t_sectors_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum \
    "$BIN" B 8 $((NSM*4)) 8 100 2 $p $g 1 2>&1 | grep -E 'Metric Name|lts__|dram__'
done

echo
echo "===== §E 复跑核对：§A 的 g=8 三点连跑一次（稳定性）====="
for p in 0 1; do run B 8 $((NSM*4)) 8 200 2 $p 8 20; done
