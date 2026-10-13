#!/usr/bin/env bash
# P95-1（§83）复现：补 P94-1（§82）§82.7 明留的尾巴——
#   「§82.6 的『L2 延迟受限混合（scalar+l2res）少而胖更好』只给现象 + 机制假设，
#     未做 ncu 归因到具体的 L2 停顿 / 调度槽指标」。
#   用 ncu 的 warp stall-reason 直方图 + L2/L1 sector 命中 + issue/warp-active，
#   在固定 Wsm 意图=2048 下扫 T，把「块粒度 → 停顿分布」这一环补上。
#   复用 §82 的内核（同一 src/hetero_sm_tax_T.cu，不改一行），只是换 ncu 指标与
#   只跑 scalar+l2res（seq=0,0,1,1）与其两个单模式对照 + coal+scalar 控制组。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_tax_T.cu
BIN=/tmp/hetero_sm_tax_T_attr
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §83 「少而胖更好」的 ncu 归因：L2 延迟受限混合的块粒度代价落在哪个停顿指标（P95-1，补 §82.7 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a；ncu 指标 = warp stall-reason 直方图 + L2/L1 命中 + issue/warp-active"
echo "# DEV=$DEV"
which ncu || { echo "ncu 不可用，退出"; exit 1; }

echo
echo "===== 构建（--ptxas-options=-v，与 §82 同内核同配置）====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_tax_T_attr_ptxas.txt || exit 1
grep -E "Function properties for _Z5pat_T|Used [0-9]+ registers|bytes spill" /tmp/hetero_sm_tax_T_attr_ptxas.txt | head -60

METRICS="gpu__time_duration.sum,\
dram__bytes_read.sum,\
lts__t_sector_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_not_selected_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct,\
smsp__warp_issue_stalled_drain_per_warp_active.pct"

run_ncu () {
  local tag="$1" T="$2" seq="$3"
  echo "----- [$tag] U=8 R=32 T=$T map=1(bind) seq=$seq -----"
  CUDA_VISIBLE_DEVICES="$DEV" timeout 400 ncu --target-processes all \
    --replay-mode application --cache-control none \
    -k "regex:pat_T" --launch-count 1 --launch-skip 0 \
    --metrics "$METRICS" \
    "$BIN" run 8 32 "$T" 1 "$seq" 2>&1 \
    | grep -E "pat_T|duration|dram__bytes_read|lts__t_sector_hit_rate|l1tex__t_sector_hit_rate|sm__warps_active|smsp__issue_active|stalled_|OK ms"
  echo
}

echo "===== §A 混合 scalar+l2res（seq=0,0,1,1）扫 T（固定 Wsm 意图=2048，R=32,U=8）====="
for T in 128 256 512; do run_ncu "scalar+l2res mixed" "$T" "0,0,1,1"; done

echo "===== §B 单模式对照：l2res solo（seq=0,0,0,1）扫 T ====="
for T in 128 512; do run_ncu "l2res solo" "$T" "0,0,0,1"; done

echo "===== §C 单模式对照：scalar4B solo（seq=0,0,1,0）扫 T ====="
for T in 128 512; do run_ncu "scalar solo" "$T" "0,0,1,0"; done

echo "===== §D 控制组：coal+scalar（seq=1,0,1,0）扫 T（应『税只认 Wsm』，T 无关）====="
for T in 128 512; do run_ncu "coal+scalar mixed" "$T" "1,0,1,0"; done

echo "########## §83 ncu 归因完成 ##########"
