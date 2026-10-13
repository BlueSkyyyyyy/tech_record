#!/usr/bin/env bash
# P82-1（§70）复现：固定每线程寄存器预算（__maxnreg__）下，同时扫每块线程数 T 与每 SM 块数 O，
#   检验数据面是否只认「每 SM 总线程 O·T」（×每线程 MLP U）这一个合量。
#   （补 §69.7 明文尾巴「未测 T 与 O 同时大于 1 的二维面（O·T 联合钉『每 SM 总线程』）」。）
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/t_occ_surface.cu
BIN=/tmp/t_occ_surface
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §70 T×O 二维面：固定 __maxnreg__，数据面是否只认『每 SM 总线程 O·T』（P82-1，补 §69.7 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：mr_32 / mr_40 / mr_64 各实例的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/t_occ_surface_ptxas.txt || exit 1
grep -E "Function properties for _Z2mr_|bytes stack frame|registers, used" /tmp/t_occ_surface_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 / §B R=32 2D / §C R=64 2D / §D R=40 2D / §E 等积聚合 / §F 属性）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 1200 "$BIN" bench
