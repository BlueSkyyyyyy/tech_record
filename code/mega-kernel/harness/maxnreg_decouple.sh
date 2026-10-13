#!/usr/bin/env bash
# P81-1（§69）复现：用 __maxnreg__ 做精确寄存器上限，检验 §68 的「32 vs ≥40 regs」分界，
#   并把「每线程寄存器预算」与「每 SM 线程数 / 占用率」再拆一层。
#   （补 §68.7 明留的尾巴「MINB 只是 hint、未用 __maxnreg__ 精确钉；占用率是 block 粒度、未扫 T」。）
#   占用率用 grid=numSM·O 钉死；每块线程数 T 独立；寄存器上限用 __maxnreg__(R) 精确钉死。
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/maxnreg_decouple.cu
BIN=/tmp/maxnreg_decouple
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §69 精确寄存器上限（__maxnreg__）× 每 SM 线程数（P81-1，补 §68.7 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：mr_*（__maxnreg__）与 lb_*（__launch_bounds__）各实例的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/maxnreg_decouple_ptxas.txt || exit 1
grep -E "Function properties for _Z2mr_|Function properties for _Z2lb_|bytes stack frame|registers, used" \
    /tmp/maxnreg_decouple_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 / §B 固定 O=1 扫精确 R / §C maxnreg vs launch_bounds / §D 扫 T / §E 扫 O / §F 属性）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 1800 "$BIN" bench
