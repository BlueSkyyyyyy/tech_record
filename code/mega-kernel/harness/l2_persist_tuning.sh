#!/usr/bin/env bash
# P41-1 复现：L2 persisting 的精细化调参（hitRatio<1 概率化分区）+ 多窗口/多 stream 共存。
# 目标卡 H100 / sm90，用 DEV 选择空闲 device（默认 3）。
#
#   bench : 扫 hitRatio（§A）、carveout × hitRatio（§B）、双窗口共存（§C），K2-only 计时表。
#   ncu   : 每配置用 `--replay-mode application --cache-control none` 采样 K2 的
#           dram__bytes_read.sum + lts__t_sector_hit_rate.pct（**不能**用默认 --cache-control all，
#           否则 ncu 会在 replay 前 flush L2，看不到 persisting 差异——P40-1 §29.2 已踩过）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/l2_persist_tuning.cu
BIN=/tmp/l2_persist_tuning
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

NCU="ncu --target-processes all --replay-mode application --cache-control none"
M="dram__bytes_read.sum,lts__t_sector_hit_rate.pct"

echo "== A/B/C. K2-only 计时表（iters=30）=="
CUDA_VISIBLE_DEVICES="$DEV" "$BIN" bench 30
echo

echo "== ncu：单窗口 hitRatio 扫描（S=16MB, carveout=16MB）=="
for f in 0.00 0.25 0.50 0.75 1.00; do
  echo "S=16 carve=16 f=$f"
  CUDA_VISIBLE_DEVICES="$DEV" $NCU -k "regex:l2_k2$" -c 1 --metrics "$M" \
    "$BIN" one win 16 "$f" 16 2>&1 | grep -E 'dram__bytes_read.sum|lts__t_sector_hit_rate'
done
echo

echo "== ncu：单窗口 carveout 边界（S=8MB, f=1.0）=="
for c in 8 16 31; do
  echo "S=8 carve=$c f=1.0"
  CUDA_VISIBLE_DEVICES="$DEV" $NCU -k "regex:l2_k2$" -c 1 --metrics "$M" \
    "$BIN" one win 8 1.00 "$c" 2>&1 | grep -E 'dram__bytes_read.sum|lts__t_sector_hit_rate'
done
echo "S=8 no-window (pollute)"
CUDA_VISIBLE_DEVICES="$DEV" $NCU -k "regex:l2_k2$" -c 1 --metrics "$M" \
  "$BIN" one pollute 8 0 0 2>&1 | grep -E 'dram__bytes_read.sum|lts__t_sector_hit_rate'
echo

echo "== ncu：对照（pfull carveout=31MB f=1.0 vs 匹配 carveout=16MB f=1.0）=="
for cfg in "win 16 1.00 31" "win 16 1.00 16" "base 16 0.00 0" "pollute 16 0.00 0"; do
  echo "one $cfg"
  CUDA_VISIBLE_DEVICES="$DEV" $NCU -k "regex:l2_k2$" -c 1 --metrics "$M" \
    "$BIN" one $cfg 2>&1 | grep -E 'dram__bytes_read.sum|lts__t_sector_hit_rate'
done
echo

echo "== ncu：双窗口/双 stream 共存（S=8MB each）=="
echo "-- carveout=16MB（=总需求）：仅 stA 窗口 / 仅 stB 窗口 / 两个都挂 --"
for cfg in "8 1.0 16 -1" "8 -1 16 1.0" "8 1.0 16 1.0"; do
  echo "one2 $cfg"
  CUDA_VISIBLE_DEVICES="$DEV" $NCU -k "regex:l2_k2" -c 4 --metrics "$M" \
    "$BIN" one2 $cfg 2>&1 | grep -E 'l2_k2[ab]|dram__bytes_read.sum'
done
echo "-- carveout=8MB（< 总需求）：两个都挂（应崩）--"
CUDA_VISIBLE_DEVICES="$DEV" $NCU -k "regex:l2_k2" -c 4 --metrics "$M" \
  "$BIN" one2 8 1.0 8 1.0 2>&1 | grep -E 'l2_k2[ab]|dram__bytes_read.sum'
echo "-- 诊断：stB 单独使用（单 stream，无竞争）--"
CUDA_VISIBLE_DEVICES="$DEV" $NCU -k "regex:l2_k2$" -c 1 --metrics "$M" \
  "$BIN" oneB 8 1.0 16 2>&1 | grep -E 'dram__bytes_read.sum|lts__t_sector_hit_rate'
