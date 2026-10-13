#!/usr/bin/env bash
# P176-1（§101）复现：cluster 放置悬崖的**逐块时间线**——坐实/修正 §100 的「尾波」解释。
#   补 §100.8(5) 明留的尾巴「尾波机制是吞吐差的推断解释（未做逐波时间线/逐 SM 占用）」。
#   复用既有源码：src/multicast_occupancy.cu（§93 load-path kernel，§100 新增只读子命令 P，
#   本节再新增**时间线探针**：语义与 mcast_body 逐字相同的 mcast_tl + 只读子命令 T，不改既有语义）。
#   目标卡 H100 / sm90a，nvcc 13.2。DEV 选空闲 device（可 DEV=n 覆盖）。
#   每配置一次独立进程调用：一次 cluster launch failure 会 sticky 污染 CUDA context。
set -u
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}
DIR=~/proj/tech_record/code/mega-kernel
MO=$DIR/src/multicast_occupancy.cu
BIN=/tmp/mcast_tl

echo "########## §101 cluster 放置悬崖的逐块时间线（P176-1） ##########"
echo "# DEV=$DEV  nvcc -O3 -arch=sm_90a  (CUDA 13.2)"
echo "# T = 逐块记录 (smid, t0, t1)（%globaltimer ns），聚合 start/finish 分位 + 驻留块 50 桶时间线"
echo
echo "===== 编译（复用 src/multicast_occupancy.cu，新增时间线探针 mcast_tl + 只读子命令 T） ====="
nvcc -O3 -arch=sm_90a "$MO" -o "$BIN" -lcuda 2>/dev/null || { echo "compile failed"; exit 1; }
echo "ok: $BIN"

run(){ echo "\$ T $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" T "$@"; }

NSM=$(CUDA_VISIBLE_DEVICES="$DEV" python3 -c 'import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null)
[ -z "${NSM:-}" ] && NSM=132
echo "# numSM=$NSM"

echo
echo "===== §A（headline）C=4 放置悬崖：grid=992(峰值=cap) / 996(+1 cluster) / 1024(+8) / 1056(+16) ====="
echo "#   cap = maxActiveClusters(248) × C(4) = 992 块 = 一次可驻留上限；超出者为「尾波」"
for g in 992 996 1024 1056; do
  run 4 $g 8 300 0 2 0 0 5
done

echo
echo "===== §B 对照 C=2（cap = 528×2 = 1056，grid≤1056 内无墙）：grid=992 / 1056 ====="
for g in 992 1056; do
  run 2 $g 8 300 0 2 0 0 5
done

echo
echo "===== §C 固定 grid=1024 扫 C：尾波大小 = grid − cap_blocks(C) ====="
echo "#   C=2: cap=1056(无尾波) | C=4: cap=992(尾32) | C=8: cap=992(尾32) | C=16: cap=928(尾96)"
for C in 2 4 8 16; do
  run $C 1024 8 300 0 2 0 0 5
done

echo
echo "===== §D R 敏感性：C=4，grid=992(无尾波) vs 1024(尾波)，R=50/150/300/600 ====="
for R in 50 150 300 600; do
  echo "# R=$R"
  run 4 992 8 $R 0 2 0 0 5
  run 4 1024 8 $R 0 2 0 0 5
done

echo
echo "########## 完 ##########"
