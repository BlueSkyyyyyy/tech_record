#!/usr/bin/env bash
# P174-1（§99）复现：占用率上界的 ncu cross-check + cluster 每槽握手的指令级归因。
#   补 §93.8 / §97.9 / §98.9 明留的同一条尾巴——「占用率用
#   cudaOccupancyMaxActiveBlocksPerMultiprocessor 的**静态**上界，未用 ncu 的 achieved
#   occupancy 交叉验证」「未做 ncu 归因」。
#   复用既有源码：src/multicast_occupancy.cu（§93 load-path kernel）、
#   src/cluster_concurrency.cu（§87–§98 的跨 SM 交换载体），**不改既有语义**。
#   目标卡 H100 / sm90a，nvcc 13.2。DEV 选空闲 device（可 DEV=n 覆盖）。
#   每配置一次独立进程调用：一次 cluster launch failure 会 sticky 污染 CUDA context。
set -u
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}
DIR=${HOME}/proj/tech_record/code/mega-kernel
MO=$DIR/src/multicast_occupancy.cu
CC=$DIR/src/cluster_concurrency.cu
BIN_MO=/tmp/mcast_occ_ncu
BIN_CC=/tmp/cluster_cc_ncu

OCCMET="sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_blocks,launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,\
launch__waves_per_multiprocessor"
INSTMET="smsp__inst_executed.sum,smsp__inst_executed_op_shared_ld.sum,\
smsp__inst_executed_op_shared_st.sum,smsp__inst_executed_op_global_st.sum"

echo "########## §99 占用率上界的 ncu cross-check（P174-1） ##########"
echo "# DEV=$DEV  nvcc -O3 -arch=sm_90a  (CUDA 13.2)"
echo
echo "===== 编译 ====="
nvcc -O3 -arch=sm_90a "$MO" -o "$BIN_MO" -lcuda 2>/dev/null || { echo "compile MO failed"; exit 1; }
nvcc -O3 -arch=sm_90a "$CC" -o "$BIN_CC" -lcuda 2>/dev/null || { echo "compile CC failed"; exit 1; }
echo "ok: $BIN_MO $BIN_CC"

run_mo(){ echo "\$ MULTICAST_OCC $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN_MO" "$@"; }
run_cc(){ echo "\$ CLUSTER_CONCURRENCY $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN_CC" "$@"; }
prof(){    # prof <kernel-regex> <metrics> <bin> <args...>
  local k="$1"; local met="$2"; shift 2; local bin="$1"; shift
  echo "\$ ncu -k $k --metrics {occ/inst} $bin $*"
  CUDA_VISIBLE_DEVICES="$DEV" timeout 600 ncu --target-processes all -k "regex:$k" -c 1 \
      --metrics "$met" "$bin" "$@" 2>&1 \
    | grep -E "Metric Name|sm__warps_active|launch__occupancy|launch__waves|smsp__inst|Metric Value|^  [a-z0-9_]+\(" \
    | sed 's/^/    /'
}

NSM=$(CUDA_VISIBLE_DEVICES="$DEV" python3 -c 'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null)
if [ -z "${NSM:-}" ]; then NSM=132; fi
echo "# numSM=$NSM"

echo
echo "===== §A §93 grid 轴：静态 occ=13（L=2,16KB），grid=numSM·O，mode0；timing + ncu achieved occ ====="
for O in 1 2 4 8; do
  g=$((NSM*O))
  run_mo B 4 $g 8 300 0 2 0 0
  prof mcast_nat "$OCCMET" "$BIN_MO" B 4 $g 8 300 0 2 0 0
done

echo
echo "===== §B §93 SMEM-pad 轴：grid=numSM·8，L=2，pad 改静态 occ；timing + ncu achieved occ ====="
for pad in 0 16 48 96 144 200; do
  run_mo B 4 $((NSM*8)) 8 300 0 2 $pad 0
  prof mcast_nat "$OCCMET" "$BIN_MO" B 4 $((NSM*8)) 8 300 0 2 $pad 0
done

echo
echo "===== §C 短工作尾波：grid=numSM·8，pad=0，R=1（极短）vs R=300（长）；achieved occ 之差即尾波 ====="
run_mo B 4 $((NSM*8)) 8 1 0 2 0 0
prof mcast_nat "$OCCMET" "$BIN_MO" B 4 $((NSM*8)) 8 1 0 2 0 0

echo
echo "===== §D §97/§98 跨 SM 载体：C=2 grid=264（=2 块/SM）U=8；静态 occ=3，ncu achieved occ（应≈eff=2） ====="
for k in mbar_8 mtex_8 mtex2_8 ring1_8; do
  case $k in
    mbar_8)  cmd=B;; mtex_8) cmd=D;; mtex2_8) cmd=I;; ring1_8) cmd=M;;
  esac
  run_cc $cmd 2 264 32 64 8
  prof $k "$OCCMET" "$BIN_CC" $cmd 2 264 32 64 8
done

echo
echo "===== §E 指令级归因：同配置，mbar/mtex/mtex2/ring1 的 inst_executed 与 shared ld/st（验证 §97/§98「省 arrive」） ====="
for k in mbar_8 mtex_8 mtex2_8 ring1_8; do
  case $k in
    mbar_8)  cmd=B;; mtex_8) cmd=D;; mtex2_8) cmd=I;; ring1_8) cmd=M;;
  esac
  prof $k "$INSTMET" "$BIN_CC" $cmd 2 264 32 64 8
done

echo
echo "########## 完 ##########"
