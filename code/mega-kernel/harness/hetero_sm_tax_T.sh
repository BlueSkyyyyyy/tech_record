#!/usr/bin/env bash
# P94-1（§82）复现：补 P93-1（§81）§81.7 明留的尾巴——
#   「U ≤ 32、R ≤ 128、T=256 固定；R<32 未测」+「§C 固定 R=128 时 occ 随 U 变（ILP 与占用率仍部分纠缠）」。
#   把每块线程数 T 变成第三旋钮，用「固定 Wsm=2048」分离纯 ILP 与占用率；并补 R<32 角点。
#   仍复用 §72–§81 的四访存模式 / 逻辑字节 / XOR 参考 / 误差扩散 assign[] / SM 分配 / per-SM 游标。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_tax_T.cu
BIN=/tmp/hetero_sm_tax_T
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §82 每块线程数 T 作为第三旋钮：占用率「由谁封顶」、R<32 角点、固定 Wsm 的纯 ILP 曲线（P94-1，补 §81.7 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_T<U,R> 的 registers / spill（U=8 扫 R，含 R=24/28）====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_tax_T_ptxas.txt || exit 1
grep -E "Function properties for _Z5pat_T|Used [0-9]+ registers|bytes spill" /tmp/hetero_sm_tax_T_ptxas.txt | head -120

echo
echo "===== 运行 bench（§A 正确性 / §B 资源表 T×R / §C 固定 Wsm 纯 ILP / §D 固定 Wsm 扫 T / §E R<32 / §F scalar+l2res）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 3000 "$BIN" bench

echo
echo "===== ncu 归因：coal+scalar（bind）在 (U=8,R=32) 的 T=64/256/512 ====="
if command -v ncu >/dev/null 2>&1; then
  for T in 64 256 512; do
    echo "--- map=1 (bind), seq 1,0,1,0 (coal+scalar), U=8 R=32 T=$T ---"
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_T" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active,launch__occupancy_limit_registers,launch__occupancy_limit_threads,launch__occupancy_limit_blocks,launch__occupancy_per_register_count \
      "$BIN" run 8 32 "$T" 1 1,0,1,0 2>&1 | grep -E "pat_T|dram__bytes_read|sm__warps_active|smsp__inst_executed|occupancy" || true
  done
else
  echo "ncu 不可用，跳过"
fi
