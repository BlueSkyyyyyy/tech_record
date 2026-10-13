#!/usr/bin/env bash
# P93-1（§81）复现：补 P92-1（§80）§80.7 明留的尾巴——
#   「`U ≤ 8` 且 `__maxnreg__(32)` 固定（放宽寄存器预算后的 `U∈{16,…}` 未测）」；
#   「`U×Wsm` 替代性只用 `coal+scalar` 一个混合」。
#   把寄存器预算 R（`__maxnreg__(R)`）与每线程 ILP U 都当独立旋钮，做 U×R 二维；
#   每个 (U,R) 用 occupancy API 实测 occ，令 O=occ、grid=numSM·occ（全图常驻、不丢 work）。
#   仍复用 §72–§80 的四访存模式 / 逻辑字节 / XOR 参考 / 误差扩散 assign[] / SM 分配 / per-SM 游标。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_tax_reg_ilp.cu
BIN=/tmp/hetero_sm_tax_reg_ilp
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §81 寄存器预算 R × 每线程 ILP U：ILP 天花板与「占用率 vs 在飞 load」解耦（P93-1，补 §80.7 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_reg<U,R> 的 registers / spill（U×R，节选）====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_tax_reg_ilp_ptxas.txt || exit 1
grep -E "Function properties for _Z7pat_reg|Used [0-9]+ registers|bytes spill" /tmp/hetero_sm_tax_reg_ilp_ptxas.txt | head -200

echo
echo "===== 运行 bench（§A 正确性 / §B 资源表 / §C ILP 天花板 / §D R 扫 / §E scalar+l2res）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 3000 "$BIN" bench

echo
echo "===== ncu 归因：coal+scalar（bind）在 (U=8,R=32)/(U=16,R=64)/(U=32,R=128) 的 DRAM 字节 / 活跃 warp / 发射 ====="
if command -v ncu >/dev/null 2>&1; then
  for UR in "8 32" "16 64" "32 128"; do
    set -- $UR; U=$1; R=$2
    echo "--- map=1 (bind), seq 1,0,1,0 (coal+scalar), U=$U R=$R ---"
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_reg" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active,launch__occupancy_limit_registers \
      "$BIN" run "$U" "$R" 1 1,0,1,0 2>&1 | grep -E "pat_reg|dram__bytes_read|l1tex__t_sectors|sm__warps_active|smsp__inst_executed|occupancy_limit" || true
  done
else
  echo "ncu 不可用，跳过"
fi
