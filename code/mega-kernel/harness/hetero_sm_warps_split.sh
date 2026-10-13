#!/usr/bin/env bash
# P91-1（§79）复现：补 P90-1（§78）§78.8 明留的头号尾巴
#   「Wsm=2048 只经 O=8/T=256，未补 O=2/T=1024 → 『warp 数 ≠ block 数』在 2048 处未完全解耦」。
#   在**每个固定 Wsm∈{512,1024,2048} 内把 (T,O) 拆分铺全**（O=Wsm/T），唯一新变量 = 同一 Wsm 下的
#   (T,O) 拆分；仍复用 §72–§78 的四访存模式 / 逻辑字节 / XOR 参考 / 误差扩散 assign[] / SM 分配 /
#   per-SM 原子子段游标。检验 §78 的 headline「税随 Wsm 收回」在 2048 处是否对**所有** T 拆分成立，
#   以及同一 Wsm 内 meas/E 与 solo 是否与拆分无关（合量 Wsm 是否唯一支配量）。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_warps_split.cu
BIN=/tmp/hetero_sm_warps_split
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §79 SM 角色绑定 × 每 SM warp 数 Wsm 固定下的 (T,O) 拆分（P91-1，补 §78.8 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_hetero 的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_warps_split_ptxas.txt || exit 1
grep -E "Function properties for _Z11pat_hetero|bytes stack frame|registers, used" /tmp/hetero_sm_warps_split_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 O=1/2/4/8/16 / §B solo×10 列 / §C 三映射×拆分 / §C2 平铺 / §C3 同 Wsm 内 spread / §D SM 分配）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 3000 "$BIN" bench

echo
echo "===== ncu 归因：coal+scalar 在 Wsm=2048 的三个拆分（T1024/O2、T512/O4、T256/O8）下 bind 的 DRAM 字节 / 活跃 warp / 发射 ====="
if command -v ncu >/dev/null 2>&1; then
  for TO in "1024 2" "512 4" "256 8"; do
    set -- $TO
    echo "--- map=1 (bind), seq 1,0,1,0 (coal+scalar), T=$1 O=$2 (Wsm=$(( $1 * $2 ))) ---"
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_hetero" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active \
      "$BIN" run "$1" "$2" 1 1 1,0,1,0 2>&1 | grep -E "pat_hetero|dram__bytes_read|l1tex__t_sectors|sm__warps_active|smsp__inst_executed" || true
  done
else
  echo "ncu 不可用，跳过"
fi
