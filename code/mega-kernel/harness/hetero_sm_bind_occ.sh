#!/usr/bin/env bash
# P89-1（§77）复现：补 §73.8/§74.9/§75.9/§76.10 反复出现的尾巴
#   「bind 用 O=1（每 SM 一 block），未测每 SM 多块的角色绑定」。
#   把 §73 的「按 SM 角色绑定」正确推广到 **每 SM 多块（O>1）**（per-SM 原子子段游标），
#   在**同一 O·T=1024** 下对跑 O=1/T=1024、O=2/T=512、O=4/T=256，检验 §73「共驻税是全局」的
#   O 稳健性，并兑现 §72.8 缓解 (a) 的「加 warp 摊」。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_sm_bind_occ.cu
BIN=/tmp/hetero_sm_bind_occ
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §77 SM 角色绑定 × 每 SM 多块（O>1）（P89-1，补 §73.8/§74.9/§75.9/§76.10 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_hetero 的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_sm_bind_occ_ptxas.txt || exit 1
grep -E "Function properties for _Z11pat_hetero|bytes stack frame|registers, used" /tmp/hetero_sm_bind_occ_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 O=1/2/4 / §B solo×3档 / §C 三映射×O / §D SM 分配 / §E coal+scalar 份额×O）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench

echo
echo "===== ncu 归因：coal+scalar 在 O=1 vs O=4 下 bind 的 DRAM 字节 / 活跃 warp / 发射 ====="
if command -v ncu >/dev/null 2>&1; then
  for O in 1 4; do
    echo "--- map=1 (bind), seq 1,0,1,0 (coal+scalar), O=$O ---"
    T=1024; [ "$O" = "4" ] && T=256
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_hetero" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active \
      "$BIN" run "$T" "$O" 1 1 1,0,1,0 2>&1 | grep -E "pat_hetero|dram__bytes_read|l1tex__t_sectors|sm__warps_active|smsp__inst_executed" || true
  done
else
  echo "ncu 不可用，跳过"
fi
