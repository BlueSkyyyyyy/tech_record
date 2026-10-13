#!/usr/bin/env bash
# P147-1 实验 driver：MQA logits（lightning indexer）训练反向。
#   bash harness/exp_mqa_logits_indexer_backward.sh > harness/exp_mqa_logits_indexer_backward.out.txt
#   段一：nvcc 编 src/mqa_indexer_backward.cu（真实内核：正确性 [A]、吞吐 [B]）。
#   段二：harness/exp_mqa_logits_indexer_backward.py —— KL 蒸馏梯度、打分算子反向、
#         top-k 直通、FP8 STE（torch fp64 对拍 autograd + 中心差分）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SRC="$ROOT/src/mqa_indexer_backward.cu"

CMD=$(cat <<EOS
set -e
echo "== 内核源码 =="
sha256sum '$SRC'
echo
echo "== 编译（sm_90）=="
nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/mqa_indexer_backward
echo "  built /tmp/mqa_indexer_backward"
echo
echo "== 设备端自带报告（src/mqa_indexer_backward.cu，单进程）=="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
/tmp/mqa_indexer_backward
echo
echo "== Python：KL 蒸馏梯度 + 打分算子反向 + top-k 直通 + FP8 STE ==="
cd '$H'
python3 -u exp_mqa_logits_indexer_backward.py
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
