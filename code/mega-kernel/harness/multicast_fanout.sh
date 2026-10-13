#!/usr/bin/env bash
# P169-1（§94）复现：TMA multicast 的「子集扇出 g」与「chunk 维度」——
#   补 §93.8 明留的「未扫 chunk、未测 multicast 子集掩码」。
#   每配置一次独立进程调用：一次 cluster launch failure 会 sticky 污染 CUDA context。
#   目标卡 H100 / sm90a，nvcc 13.2。DEV 选空闲 device（可 DEV=n 覆盖）。
set -u
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/multicast_fanout.cu
BIN=/tmp/multicast_fanout

echo "########## §94 TMA multicast 子集扇出 g + chunk 维度（P169-1） ##########"
echo "# DEV=$DEV  nvcc -O3 -arch=sm_90a"
echo
echo "===== ptxas -v：fanout_kernel 的寄存器 / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" -lcuda 2>/tmp/mcast_fanout_ptxas.txt || exit 1
grep -E "Function properties for|bytes stack frame|registers, used" /tmp/mcast_fanout_ptxas.txt
run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$@"; }

NSM=$(CUDA_VISIBLE_DEVICES="$DEV" python3 -c 'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null)
if [ -z "${NSM:-}" ]; then NSM=132; fi
echo "# numSM=$NSM"

echo
echo "===== §A（Q1）fan-out 扫描 g=0(plain)/1/2/4/8（C=8, chunk=8KB, L=2, grid=numSM*4, R=200, it=20）====="
for g in 0 1 2 4 8; do run B 8 $((NSM*4)) 8 200 2 $g 20; done

echo
echo "===== §A2 chunk×g 交互：g=0/4/8 在 chunk=2KB 与 16KB（C=8, L=2, grid=numSM*4, 固定总字节 2400KB, it=20）====="
for KB_R in "2 1200" "16 150"; do
  set -- $KB_R; KB=$1; R=$2
  for g in 0 4 8; do run B 8 $((NSM*4)) $KB $R 2 $g 20; done
done

echo
echo "===== §B（Q2）chunk 扫描（固定总字节 chunk*R=2400KB；C=8, L=2, grid=numSM*4, it=20）plain vs g=8 ====="
for KB_R in "2 1200" "4 600" "8 300" "16 150"; do
  set -- $KB_R; KB=$1; R=$2
  for g in 0 8; do run B 8 $((NSM*4)) $KB $R 2 $g 20; done
done

echo
echo "===== §C（Q3）C×g 交叉（chunk=8KB, L=2, grid=numSM*4, R=200, it=20）：缩减由 g 定还是 C 定？ ====="
for Cg in "4 0" "4 2" "4 4" "8 0" "8 2" "8 4" "8 8" "16 0" "16 4" "16 8" "16 16"; do
  set -- $Cg; C=$1; g=$2
  run B $C $((NSM*4)) 8 200 2 $g 20
done

echo
echo "===== §D（Q4）ncu：源侧 L2 请求是否按 1/g 缩放（C=8, chunk=8KB, L=2, R=100, grid=numSM*4）====="
for g in 0 1 2 4 8; do
  echo "\$ ncu ... B 8 grid=$((NSM*4)) 8 100 2 $g 1  (g=$g)"
  CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all -k regex:fanout_kernel -c 1 \
    --metrics lts__t_bytes.sum,lts__t_sectors.sum,lts__t_sectors_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum \
    "$BIN" B 8 $((NSM*4)) 8 100 2 $g 1 2>&1 | grep -E 'Metric Name|lts__|dram__'
done
echo
echo "-- §D2 chunk 对源侧 L2 请求的影响（plain vs g=8，固定总字节 2400KB）--"
for KB_R in "2 1200" "8 300" "16 150"; do
  set -- $KB_R; KB=$1; R=$2
  for g in 0 8; do
    echo "\$ ncu ... B 8 grid=$((NSM*4)) $KB $R 2 $g 1  (chunk=${KB}KB g=$g)"
    CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all -k regex:fanout_kernel -c 1 \
      --metrics lts__t_bytes.sum,lts__t_sectors.sum,lts__t_sectors_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum \
      "$BIN" B 8 $((NSM*4)) $KB $R 2 $g 1 2>&1 | grep -E 'Metric Name|lts__|dram__'
  done
done
