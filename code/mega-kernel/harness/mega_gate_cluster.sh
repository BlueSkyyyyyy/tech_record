#!/usr/bin/env bash
# P34-1 复现：DeepGEMM Mega Gate 的 `cluster_dim` 路径的 sm90 等价复刻
#           （目标卡 H100 / sm90a）。固定空闲 device。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/mega_gate_cluster.cu
BIN=/tmp/mega_gate_cluster
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" -lcuda 2>/dev/null || exit 1
run(){ echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $*"; CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$@"; }

echo "== A. 正确性：split-K × CTA-pair（cluster_dim）× gate/fused 对参考逐 token 核对 =="
echo "   （字段：idx_mismatch=top-k 专家集合与参考不同的 token 数；wmax=权重最大绝对误差）"
for cfg in \
  "128 128 128 512 1 1 4 0" \
  "128 128 128 512 2 1 4 0" \
  "128 128 128 512 4 1 4 0" \
  "128 128 128 512 1 2 4 0" \
  "128 128 128 512 2 2 4 0" \
  "128 128 128 512 4 2 4 0" \
  "128 128 128 512 1 1 4 1" ; do
  run L $cfg 100
done
echo

echo "== B. 占满机器：固定少量 token block（NT=16），靠 split-K / CTA-pair 增加 logical CTA 数 =="
echo "   （grid = NT * SPLIT * MMA；K=2048）"
for sm in "1 1" "2 1" "4 1" "1 2" "2 2" "4 2"; do
  run L 16 128 128 2048 $sm 4 0 100
done
echo

echo "== C. 已饱和（NT=128，grid 已 =128≈#SM）：split-K 只加协调开销，fused 免 scratch =="
echo "   （K=512）"
for cfg in "128 128 128 512 1 1 4 0" "128 128 128 512 2 1 4 0" "128 128 128 512 4 2 4 0" \
           "128 128 128 512 1 1 4 1"; do
  run L $cfg 200
done
echo

echo "== D. ncu：score scratch 的 L2 写/读扇区（NT=128, T=128, E=128, K=512），取第 2 次 launch =="
for cfg in "128 128 128 512 1 1 4 0" "128 128 128 512 4 2 4 0" "128 128 128 512 1 1 4 1"; do
  echo "\$ ncu ... L $cfg 3"
  CUDA_VISIBLE_DEVICES="$DEV" ncu --target-processes all -k regex:gate_kernel --launch-skip 1 -c 1 \
    --metrics lts__t_sectors_op_write.sum,lts__t_sectors_op_read.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    "$BIN" L $cfg 4 2>&1 | grep -E 'Metric Name|lts__t_sectors|dram__bytes'
done
