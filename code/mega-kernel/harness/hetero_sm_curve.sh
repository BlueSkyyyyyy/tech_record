#!/usr/bin/env bash
# P86-1（§74）复现：把 §73 的**常数速率 SM 分配**（bind/bind_bal）对照**实测「SM 数→速率」曲线配平**
#   （bind_opt，贪心 makespan 最小化）。补 §73.8 明文尾巴「bind_bal 用常数每SM速率模型，未按曲线最优配平」。
#   问：①按曲线配平能否收回 bind_bal 的分区自伤？ ②能否收回 coal+scalar 的全局发射税？
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_curve.cu
BIN=/tmp/hetero_sm_curve
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §74 SM 分配器：常数速率 vs 实测曲线配平（P86-1，补 §73.8 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_hetero 的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_curve_ptxas.txt || exit 1
grep -E "Function properties for _Z11pat_hetero|bytes stack frame|registers, used" /tmp/hetero_sm_curve_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 / §B solo+速率曲线 / §C 四映射 / §D SM 分配 / §E scalar 份额 / §F）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench

echo
echo "===== ncu 归因：coal+scalar 的 bind vs bind_opt（DRAM 字节 / 活跃 warp / 发射）====="
if command -v ncu >/dev/null 2>&1; then
  for MAP in 1 3; do
    echo "--- map=$MAP (1=bind,3=bind_opt), seq 1,0,1,0 (coal+scalar), T=1024,O=1 ---"
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_hetero" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active \
      "$BIN" run 1024 1 $MAP 1 1,0,1,0 2>&1 | grep -E "pat_hetero|dram__bytes_read|l1tex__t_sectors|sm__warps_active|smsp__inst_executed" || true
  done
else
  echo "ncu 不可用，跳过"
fi
