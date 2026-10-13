#!/usr/bin/env bash
# P149-1 实验 driver：稀疏（块选择）MQA indexer 前向的反向。
#   bash harness/exp_mqa_logits_sparse_backward.sh > harness/exp_mqa_logits_sparse_backward.out.txt
#   段一：nvcc 编 src/mqa_indexer_backward_sparse.cu（真实内核：正确性 [A]、吞吐 [B]）。
#   段二：harness/exp_mqa_logits_sparse_backward.py —— 压缩伴随(scatter)、稀疏 KL 梯度、
#         打分算子反向、别名累加、真实 SM90 dense kernel 锚。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SRC="$ROOT/src/mqa_indexer_backward_sparse.cu"

CMD=$(cat <<EOS
set -e
echo "== 内核源码 =="
sha256sum '$SRC'
echo
echo "== 编译（sm_90）=="
nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/mqa_indexer_backward_sparse
echo "  built /tmp/mqa_indexer_backward_sparse"
echo
echo "== 设备端自带报告（src/mqa_indexer_backward_sparse.cu，单进程）=="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
/tmp/mqa_indexer_backward_sparse
echo
echo "== Python：压缩伴随 + 稀疏 KL 梯度 + 打分算子反向 + 别名 + 真实 kernel 锚 =="
cd '$H'
python3 -u exp_mqa_logits_sparse_backward.py
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
