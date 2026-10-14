#!/usr/bin/env bash
# P80-1（§68）复现：把「寄存器预算」与「占用率」解耦——补 §67.4 明留的尾巴
#   （「MINB 同时改寄存器预算与占用率两个物理量，本轮未把二者解耦」）。
#   占用率用 grid=numSM·O 钉死（每 SM 恰好 O 块，只要 O ≤ reg-limit）；寄存器预算用
#   __launch_bounds__(256,MINB) 钉死（MINB∈{8,6,4,2} → regs ≤ 32/42/64/128）。
#   再加 SMEM-padding 作独立占用率口径交叉验证，以及 cp.async 载体的免疫性对照。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/reg_occ_decouple.cu
BIN=/tmp/reg_occ_decouple
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §68 寄存器预算 vs 占用率 解耦（P80-1，补 §67.4 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：ro_kernel 各实例的 registers / spill（载体 × MINB）====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/reg_occ_decouple_ptxas.txt || exit 1
grep -E "Function properties for _Z9ro_kernel|bytes stack frame|registers, used" \
    /tmp/reg_occ_decouple_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 / §B 固定占用率扫寄存器预算 / §C 固定预算扫占用率 / §D SMEM-pad 对照 / §E 载体交叉 / §F 属性）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 1800 "$BIN" bench
