#!/usr/bin/env bash
# P84-1（§72）复现：同一常驻 grid 内不同 block 跑不同访存模式（异构共驻），
#   复检 §71.7 headline 四「一个常驻 kernel 内 task 类型越杂、越难一套调度参数通吃」。
#   （补 §71.7 明文尾巴「四模式各自单独跑、未测同一 grid 内多模式共驻」。）
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/hetero_access_mix.cu
BIN=/tmp/hetero_access_mix
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §72 异构访存模式共驻（P84-1，补 §71.7 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_mix 的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/hetero_access_mix_ptxas.txt || exit 1
grep -E "Function properties for _Z7pat_mix|bytes stack frame|registers, used" /tmp/hetero_access_mix_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 / §B 单模式基线 / §C 混合矩阵 / §D (T,O) 通吃 / §E 稀疏份额 / §F）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 1800 "$BIN" bench

echo
echo "===== §C ncu 归因：equal-4way 混合（T=256,O=4）的 DRAM 读 / L1 sectors ====="
if command -v ncu >/dev/null 2>&1; then
  echo "--- mix equal 1,1,1,1 ---"
  CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
    --replay-mode application --cache-control none \
    -k "regex:pat_mix" --launch-count 1 --launch-skip 0 \
    --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    "$BIN" run 256 4 1 1,1,1,1 2>&1 | grep -E "pat_mix|dram__bytes_read|l1tex__t_sectors" || true
  echo "--- mix 50/50 coal+sparse 1,1,0,0 ---"
  CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
    --replay-mode application --cache-control none \
    -k "regex:pat_mix" --launch-count 1 --launch-skip 0 \
    --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    "$BIN" run 256 4 1 1,1,0,0 2>&1 | grep -E "pat_mix|dram__bytes_read|l1tex__t_sectors" || true
else
  echo "ncu 不可用，跳过"
fi
