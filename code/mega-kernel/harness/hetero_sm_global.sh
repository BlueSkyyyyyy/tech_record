#!/usr/bin/env bash
# P88-1（§76）复现：把 §75 的**单起点** in-situ 爬山（bind_insitu）升级为**多起点**（bind_global）
#   并做 oracle 穷举，补 §75.9 头号尾巴「bind_insitu 是局部爬山、结果不可复现（equal4 两次运行
#   收敛到不同分配 [25,32,30,45]/[8,84,20,20]，因 l2res16M 的 L2 驻留对运行序敏感）」。
#   四问：Q1 可复现性 / Q2 多起点 / Q3 oracle 穷举 / Q4 评价噪声地板。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_global.cu
BIN=/tmp/hetero_sm_global
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §76 SM 分配器：单起点局部爬山 vs 多起点 vs oracle 穷举（P88-1，补 §75.9 头号尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_hetero 的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_global_ptxas.txt || exit 1
grep -E "Function properties for _Z10pat_hetero|bytes stack frame|registers, used" /tmp/hetero_sm_global_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 / §B solo+单跑曲线 / §C 可复现性 / §D 六映射 / §E oracle / §F 噪声+轨迹 / §G）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench

echo
echo "===== ncu 归因：coal+sparse 的 bind_insitu vs bind_global（DRAM 字节 / 活跃 warp / 发射）====="
if command -v ncu >/dev/null 2>&1; then
  for MAP in 4 5; do
    echo "--- map=$MAP (4=bind_insitu,5=bind_global), seq 1,1,0,0 (coal+sparse), T=1024,O=1 ---"
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_hetero" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active \
      "$BIN" run 1024 1 $MAP 1,1,0,0 2>&1 | grep -E "pat_hetero|dram__bytes_read|l1tex__t_sectors|sm__warps_active|smsp__inst_executed" || true
  done
else
  echo "ncu 不可用，跳过"
fi
