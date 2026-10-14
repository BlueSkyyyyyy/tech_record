#!/usr/bin/env bash
# P87-1（§75）复现：把 §74 的**单跑**速率曲线分配器（bind_opt）对照 **in-situ 迭代不动点**分配器
#   （bind_insitu，仅用带真实并发争用的实测 makespan 做局部爬山）。补 §74.9 明文尾巴
#   「bind_opt 曲线是单跑测得，未做 in-situ/迭代不动点」。
#   问：in-situ 能否收回 §74 里 coal+sparse 从 0.936 崩到 0.664 的分配崩盘？
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_insitu.cu
BIN=/tmp/hetero_sm_insitu
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §75 SM 分配器：单跑曲线(bind_opt) vs in-situ 迭代不动点(bind_insitu)（P87-1，补 §74.9 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_hetero 的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_insitu_ptxas.txt || exit 1
grep -E "Function properties for _Z10pat_hetero|bytes stack frame|registers, used" /tmp/hetero_sm_insitu_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 / §B solo+单跑曲线 / §C 五映射 / §D 条件 in-situ 曲线 / §E 分配明细 / §F 迭代轨迹 / §G）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench

echo
echo "===== ncu 归因：coal+sparse 的 bind vs bind_insitu（DRAM 字节 / 活跃 warp / 发射）====="
if command -v ncu >/dev/null 2>&1; then
  for MAP in 3 4; do
    echo "--- map=$MAP (3=bind_opt,4=bind_insitu), seq 1,1,0,0 (coal+sparse), T=1024,O=1 ---"
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_hetero" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active \
      "$BIN" run 1024 1 $MAP 1,1,0,0 2>&1 | grep -E "pat_hetero|dram__bytes_read|l1tex__t_sectors|sm__warps_active|smsp__inst_executed" || true
  done
else
  echo "ncu 不可用，跳过"
fi
