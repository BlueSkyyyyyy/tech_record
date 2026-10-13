#!/usr/bin/env bash
# P175-1（§100）复现：cluster 放置墙的细化——cluster 感知占用率 API
#   （cudaOccupancyMaxActiveClusters）对账 + 扫 cluster 尺寸 C + 扫 grid 找「放置悬崖」。
#   补 §99.8 明留的尾巴「只测 H100/sm90a、单一 cluster 尺寸 C=4（未扫 C 对放置墙的影响）」。
#   复用既有源码：src/multicast_occupancy.cu（§93 load-path kernel `mcast_nat`，
#   本节只新增**只读**子命令 `P`，不改任何既有语义）。
#   目标卡 H100 / sm90a，nvcc 13.2。DEV 选空闲 device（可 DEV=n 覆盖）。
#   每配置一次独立进程调用：一次 cluster launch failure 会 sticky 污染 CUDA context。
set -u
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}
DIR=~/proj/tech_record/code/mega-kernel
MO=$DIR/src/multicast_occupancy.cu
BIN=/tmp/mcast_occ_pw

OCCMET="sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_blocks,launch__occupancy_limit_shared_mem,\
launch__waves_per_multiprocessor"

echo "########## §100 cluster 放置墙的细化（P175-1） ##########"
echo "# DEV=$DEV  nvcc -O3 -arch=sm_90a  (CUDA 13.2)"
echo
echo "===== 编译（复用 src/multicast_occupancy.cu，新增只读子命令 P） ====="
nvcc -O3 -arch=sm_90a "$MO" -o "$BIN" -lcuda 2>/dev/null || { echo "compile failed"; exit 1; }
echo "ok: $BIN"

run(){ echo "\$ MULTICAST_OCC $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$@"; }
prof(){  # prof <kernel-regex> <bin> <args...>
  local k="$1"; shift; local b="$1"; shift
  echo "\$ ncu -k $k --metrics {occ} $b $*"
  CUDA_VISIBLE_DEVICES="$DEV" timeout 600 ncu --target-processes all -k "regex:$k" -c 1 \
      --metrics "$OCCMET" "$b" "$@" 2>&1 \
    | grep -E "sm__warps_active|launch__occupancy|launch__waves" | sed 's/^/    /'
}

NSM=$(CUDA_VISIBLE_DEVICES="$DEV" python3 -c 'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null)
[ -z "${NSM:-}" ] && NSM=132
echo "# numSM=$NSM"

echo
echo "===== §A 三种占用率读数并置（grid=numSM·8=1056，L=2/16KB，KB=8，mode0） ====="
echo "#   P 行 = cluster 感知 API（cudaOccupancyMaxActiveClusters）；B 行 = blocks/SM API 的 occ/eff"
for C in 1 2 3 4 6 8 12 16; do
  run P $C $((NSM*8)) 8 2 0 0
done

echo
echo "===== §B ncu achieved occupancy 对账（grid=1056，R=300，mode0；应随 §A 的 cluster-API 排序） ====="
for C in 2 4 8 16; do
  run B $C $((NSM*8)) 8 300 0 2 0 0
  prof mcast_nat "$BIN" B $C $((NSM*8)) 8 300 0 2 0 0
done

echo
echo "===== §C 扫 C 的 deliver 吞吐（grid=1056，R=300；C=2 无放置墙 vs C≥4 有） ====="
for C in 2 3 4 6 8 12 16; do
  run B $C $((NSM*8)) 8 300 0 2 0 0
done

echo
echo "===== §D 放置悬崖：固定 C=4 扫 grid（cluster-API 上限 = 248 cluster = 992 块） ====="
for g in 264 528 792 896 928 992 1024 1056; do
  run B 4 $g 8 300 0 2 0 0
done
echo "# C=2 对照（cap = 528 cluster = 1056 块，grid≤1056 内无墙）："
for g in 992 1056; do
  run B 2 $g 8 300 0 2 0 0
done

echo
echo "===== §E SMEM-pad 轴：cluster API vs blocks API（grid=1056） ====="
echo "# C=4（有墙）：blocks API=occ，cluster API 应低于 occ"
for pad in 0 16 48 96 144; do
  run P 4 $((NSM*8)) 8 2 $pad 0
done
echo "# C=2（无墙）：cluster API 应 == occ 逐档"
for pad in 0 48 96 144; do
  run P 2 $((NSM*8)) 8 2 $pad 0
done

echo
echo "===== §F 隔离 cluster 同步成本：grid=numSM·1≈1 块/SM（grid=128，无放置压力）扫 C ====="
for C in 2 4 8 16; do
  run P $C 128 8 2 0 0
  run B $C 128 8 300 0 2 0 0
done

echo
echo "########## 完 ##########"
