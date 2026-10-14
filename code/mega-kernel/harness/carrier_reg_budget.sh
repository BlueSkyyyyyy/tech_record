#!/usr/bin/env bash
# P79-1（§67）复现：把 §66.9 明留的「跨二进制 codegen 偏移」钉死在寄存器预算上——
#   §66 的同一内核里 plain/TMA 载体在 `__launch_bounds__(256,8)`（=8 blk/SM，32 regs/线程）下
#   报 8 B 局部栈溢出，而 cp.async 载体（单缓冲/深环）0 溢出。本节把 MINB（每 SM 最小 block 数）
#   当寄存器预算旋钮扫 {8,6,4}，用 (a) ptxas -v 的 spill 行 + (b) `cudaFuncGetAttributes` 的
#   numRegs/localB + (c) 端到端运行时，检验：**8 B 溢出是不是 32-reg 上限打的，而不是「运行期
#   分支选择载体」**；并给出「抬寄存器上限（降占用率）能否收回 §66 的 plain/TMA 慢」。
# 目标卡 H100 / sm90a，DEV 自动选空闲 device（可 DEV=n 覆盖）。
set -u
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/worker_carrier_cpasync_ring.cu
BIN=/tmp/carrier_reg_budget
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §67 codegen 偏移归因：载体 × MINB 的寄存器/局部栈预算（P79-1） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：各 kernel 实例的 stack frame / spill / registers ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/carrier_reg_budget_ptxas.txt || exit 1
# 只保留 wcr_kernel 的 per-function 段（函数名 + stack/regs 行）
grep -E "Function properties for _Z10wcr_kernel|bytes stack frame|registers, used 1 barriers" \
    /tmp/carrier_reg_budget_ptxas.txt

echo
echo "===== 运行 §A（函数属性）+ §B（MINB 扫描运行时）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" regs
