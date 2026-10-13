#!/usr/bin/env bash
# P92-1（§80）复现：补 P91-1（§79）§79.6 明留的两条尾巴——
#   (tail 1)「`meas/E` 的 spread 混入 `E` 的拆分依赖，故比值需与绝对量并读」；
#   (tail 2)「`U=4`/`__maxnreg__(32)` 单一（每线程 ILP 未当独立变量）」。
#   把「共驻税」从**比值 meas/E**换成**绝对量**（mixed 绝对 GB/s + 固定参考归一 taxRef），
#   并把**每线程 ILP `U`** 变成独立旋钮（固定 Wsm=1024/T256/O4 扫 U∈{1,2,4,8}），再测 U×Wsm 替代性。
#   仍复用 §72–§79 的四访存模式 / 逻辑字节 / XOR 参考 / 误差扩散 assign[] / SM 分配 / per-SM 原子子段游标。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_tax_ilp.cu
BIN=/tmp/hetero_sm_tax_ilp
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §80 共驻税的「口径」与「每线程 ILP」：绝对量 vs meas/E、U 独立旋钮（P92-1，补 §79.6 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_hetero_u<U> 的 registers / spill（U=1/2/4/8）====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_tax_ilp_ptxas.txt || exit 1
grep -E "Function properties for _Z13pat_hetero_u|bytes stack frame|registers, used" /tmp/hetero_sm_tax_ilp_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 U=1/2/4/8 / §B solo×U / §C 口径 mixed×(Wsm=2048 四拆分) / §C2 口径 spread / §D ILP×5混合 / §E U×Wsm 替代性）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 3000 "$BIN" bench

echo
echo "===== ncu 归因：coal+scalar（bind, Wsm=1024/T256/O4）在 U=1/4/8 下的 DRAM 字节 / 活跃 warp / 发射 ====="
if command -v ncu >/dev/null 2>&1; then
  for U in 1 4 8; do
    echo "--- map=1 (bind), seq 1,0,1,0 (coal+scalar), T=256 O=4 U=$U (Wsm=1024) ---"
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_hetero_u" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active \
      "$BIN" run 256 4 "$U" 1 1,0,1,0 2>&1 | grep -E "pat_hetero_u|dram__bytes_read|l1tex__t_sectors|sm__warps_active|smsp__inst_executed" || true
  done
else
  echo "ncu 不可用，跳过"
fi
