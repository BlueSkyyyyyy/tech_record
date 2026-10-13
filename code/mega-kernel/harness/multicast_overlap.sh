#!/usr/bin/env bash
# P171-1（§96）复现：TMA multicast 的「真正重叠掩码」——同一 CTA 被多笔 multicast 打到
#   同一 SMEM 槽时的语义 / 正确性 / 成本。补 §95.7 明留的尾巴。
#   每配置一次独立进程调用：一次 cluster launch failure 会 sticky 污染 CUDA context。
#   目标卡 H100 / sm90a，nvcc 13.2。DEV 选空闲 device（可 DEV=n 覆盖）。
set -u
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/multicast_overlap.cu
BIN=/tmp/multicast_overlap

echo "########## §96 TMA multicast 真正重叠掩码（P171-1） ##########"
echo "# DEV=$DEV  nvcc -O3 -arch=sm_90a"
echo
echo "===== ptxas -v：overlap_kernel 的寄存器 / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" -lcuda 2>/tmp/mcast_overlap_ptxas.txt || exit 1
grep -E "Function properties for|bytes stack frame|registers, used" /tmp/mcast_overlap_ptxas.txt
run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$@"; }

NSM=$(CUDA_VISIBLE_DEVICES="$DEV" python3 -c 'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null)
if [ -z "${NSM:-}" ]; then NSM=132; fi
echo "# numSM=$NSM"

echo
echo "===== §A 语义 / 正确性探针（R=1，reps=16，逐次读回）：one/idem/nslot 确定性 vs race 竞争 ====="
echo "#   mode=0(one) N=1 / mode=1(idem) N=2,4 / mode=3(nslot) N=2,4 / mode=4(nslot1) N=4 / mode=2(race) N=2,4"
run P 8 $((NSM*4)) 4 4 0 1 16
for N in 2 4; do run P 8 $((NSM*4)) 4 4 1 $N 16; done
for N in 2 4; do run P 8 $((NSM*4)) 4 4 3 $N 16; done
run P 8 $((NSM*4)) 4 4 4 4 16
for N in 2 4; do run P 8 $((NSM*4)) 4 4 2 $N 16; done

echo
echo "===== §A2 race 非确定性复核：mode2 N=4 独立跑 3 次（winner_hist 是否逐次不同）====="
for i in 1 2 3; do run P 8 $((NSM*4)) 4 4 2 4 16; done

echo
echo "===== §B 成本：固定 issue 数 N，目标槽 1 个(idem/race) vs N 个(nslot1) —— chunk=4KB L=4 grid=numSM*4 R=200 it=20 ====="
run B 8 $((NSM*4)) 4 200 4 0 1 20
for m in 1 2 4 3; do for N in 2 4; do run B 8 $((NSM*4)) 4 200 4 $m $N 20; done; done

echo
echo "===== §C 带宽区间：chunk=8KB L=4（occupancy 降到 6）—— issue 数 vs 目标槽 ====="
run B 8 $((NSM*4)) 8 200 4 0 1 20
for m in 1 2 4; do run B 8 $((NSM*4)) 8 200 4 $m 4 20; done
run B 8 $((NSM*4)) 8 200 4 3 4 20

echo
echo "===== §D ncu：源侧 L2 读 sector（one N=1 vs idem N=4 同源行 vs nslot1 N=4 异源行）====="
for cfg in "0 1" "1 4" "2 4" "4 4"; do
  set -- $cfg; m=$1; N=$2
  echo "\$ ncu ... B 8 grid=$((NSM*4)) 4 20 4 $m $N 1  (mode=$m N=$N)"
  CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all -k regex:overlap_kernel -c 1 \
    --metrics lts__t_sectors_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_bytes.sum,dram__bytes_read.sum \
    "$BIN" B 8 $((NSM*4)) 4 20 4 $m $N 1 2>&1 | grep -E 'Metric Name|lts__|dram__'
done

echo
echo "===== §E C×overlap 交叉：C=4/16，N=C/2，one/idem/race/nslot（chunk=4KB L=4 grid=numSM*4 R=200 it=20）====="
for CN in "4 2" "16 8"; do
  set -- $CN; C=$1; N=$2
  run B $C $((NSM*4)) 4 200 4 0 1 20
  for m in 1 2; do run B $C $((NSM*4)) 4 200 4 $m $N 20; done
  if [ "$N" -le 4 ]; then run B $C $((NSM*4)) 4 200 4 3 $N 20; fi
done

echo
echo "===== §F 稳定性复跑：§B 的 mode1/mode4 N=4 各两点连跑 ====="
run B 8 $((NSM*4)) 4 200 4 1 4 20
run B 8 $((NSM*4)) 4 200 4 4 4 20
