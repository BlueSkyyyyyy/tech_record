#!/usr/bin/env bash
# P40-1 复现：L2 常驻（persisting L2 / access policy window）作为「损失三」的第三档中间量落点。
# 目标卡 H100 / sm90，用 DEV 选择空闲 device（默认 3）。
#
#   bench : 扫中间量尺寸 S × {base,pollute,persist}，K2-only 计时表（连续跑，L2 自然演化）。
#   ncu   : 每配置用 `--replay-mode application --cache-control none` 采样 K2 的
#           dram__bytes_read.sum + lts__t_sector_hit_rate.pct（**不能**用默认 --cache-control all，
#           否则 ncu 会在 replay 前 flush L2，看不到常驻差异——首版就栽在这）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/l2_residency.cu
BIN=/tmp/l2_residency
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1

echo "== A. K2-only 计时表（S × {base,pollute,persist,pfull}，iters=30）=="
CUDA_VISIBLE_DEVICES="$DEV" "$BIN" bench 30
echo

echo "== B. ncu：K2 读中间量 y 的 DRAM 读字节 + L2 命中率（--replay-mode application --cache-control none）=="
for S in 4 8 16 32 48 64; do
  for m in base pollute persist pfull; do
    r=$(CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all \
          --replay-mode application --cache-control none \
          -k regex:l2_k2 -c 1 \
          --metrics dram__bytes_read.sum,lts__t_sector_hit_rate.pct \
          "$BIN" one "$m" "$S" 2>&1 | \
        grep -E 'dram__bytes_read.sum|lts__t_sector_hit_rate')
    echo "S=$S $m"
    echo "$r"
  done
done
