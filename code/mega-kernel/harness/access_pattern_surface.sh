#!/usr/bin/env bash
# P83-1（§71）复现：固定每线程寄存器预算（__maxnreg__(32)）与载体（寄存器读 + U=4 ILP）下，
#   把「数据面只认『每 SM 总线程 O·T』合量」的结论拿到四种访存模式上复检。
#   （补 §70.7 明文尾巴「只测连续/规则的 __ldcs uint4 流，未扫 strided / 非对齐 / L2 命中等访存模式」。）
# 目标卡 H100 / sm90a，DEV 选空闲 device（自动选最闲；可 DEV=n 覆盖）。
set -u
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/access_pattern_surface.cu
BIN=/tmp/access_pattern_surface
DEV=${DEV:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)}

echo "########## §71 访存模式 × O·T 合量（P83-1，补 §70.7 尾巴） ##########"
echo "# nvcc -O3 -arch=sm_90a --ptxas-options=-v；DEV=$DEV"

echo
echo "===== ptxas -v：pat_m0..m3 各实例的 registers / spill ====="
nvcc -O3 -arch=sm_90a --ptxas-options=-v "$SRC" -o "$BIN" 2>/tmp/access_pattern_surface_ptxas.txt || exit 1
grep -E "Function properties for _Z5pat_m|bytes stack frame|registers, used" /tmp/access_pattern_surface_ptxas.txt

echo
echo "===== 运行 bench（§A 正确性 / §B 每模式扫 O·T / §C 每模式扫 T / §D 汇总）====="
CUDA_VISIBLE_DEVICES="$DEV" timeout 1800 "$BIN" bench

echo
echo "===== §D ncu 归因：每模式 T=256,O=4（O·T=1024）的 DRAM 读 / L1 sectors ====="
if command -v ncu >/dev/null 2>&1; then
  for PAT in 0 1 2 3; do
    case $PAT in 0) NM=coalesced;; 1) NM=sparse8x;; 2) NM=scalar4B;; 3) NM=l2res16M;; esac
    echo "--- pat=$PAT ($NM) ---"
    CUDA_VISIBLE_DEVICES="$DEV" timeout 300 ncu --target-processes all \
      --replay-mode application --cache-control none \
      -k "regex:pat_" --launch-count 1 --launch-skip 0 \
      --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
      "$BIN" run "$PAT" 256 4 1 2>&1 | grep -E "pat_m|dram__bytes_read|l1tex__t_sectors" || true
  done
else
  echo "ncu 不可用，跳过"
fi
