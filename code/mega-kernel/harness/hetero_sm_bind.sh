#!/usr/bin/env bash
# P85-1（§73）复现：把 §72 的「按 block 交错」共驻，对照「**按 SM 角色绑定**」（每 SM 只跑一种模式）。
#   补 §72.8 明文尾巴「共驻映射按 block 交错，非真实按 task 类型分区/SM 角色绑定」——
#   判定共驻税/毒化是 SM 内（发射槽共享）还是 SM 间（HBM/全局）现象。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_bind.cu
BIN=/tmp/hetero_sm_bind
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §73 按 SM 角色绑定 vs 按 block 交错（P85-1，补 §72.8 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_hetero 的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_bind_ptxas.txt || exit 1
grep -E "Function properties for _Z11pat_hetero|bytes stack frame|registers, used" /tmp/hetero_sm_bind_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 / §B solo / §C 三映射对照 / §D SM 分配 / §E scalar 份额 / §F）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 1800 "$BIN" bench

echo
echo "===== ncu 归因：coal+scalar 的 interleave vs bind（DRAM 字节 / 活跃 warp / 发射）====="
if command -v ncu >/dev/null 2>&1; then
  for MAP in 0 1; do
    echo "--- map=$MAP (0=interleave,1=bind), seq 1,0,1,0 (coal+scalar), T=1024,O=1 ---"
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_hetero" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active \
      "$BIN" run 1024 1 $MAP 1 1,0,1,0 2>&1 | grep -E "pat_hetero|dram__bytes_read|l1tex__t_sectors|sm__warps_active|smsp__inst_executed" || true
  done
else
  echo "ncu 不可用，跳过"
fi
