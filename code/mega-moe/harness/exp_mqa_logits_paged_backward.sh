#!/usr/bin/env bash
# P148-1 实验 driver：分页 KV 的 MQA logits（lightning indexer）反向。
#   bash harness/exp_mqa_logits_paged_backward.sh > harness/exp_mqa_logits_paged_backward.out.txt
#   段一：nvcc 编 src/mqa_indexer_backward_paged.cu（真实内核：正确性 [A]、吞吐 [B]）。
#   段二：harness/exp_mqa_logits_paged_backward.py —— 分页前向官方锚、解析反向 vs autograd、
#         页别名累加、KL 蒸馏梯度、真实 SM90 分页 kernel 前向锚。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SRC="$ROOT/src/mqa_indexer_backward_paged.cu"

CMD=$(cat <<EOS
set -e
echo "== 内核源码 =="
sha256sum '$SRC'
echo
echo "== 编译（sm_90）=="
nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/mqa_indexer_backward_paged
echo "  built /tmp/mqa_indexer_backward_paged"
echo
echo "== 设备端自带报告（src/mqa_indexer_backward_paged.cu，单进程）=="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
/tmp/mqa_indexer_backward_paged
echo
echo "== Python：分页前向锚 + 解析反向 + 页别名 + KL 梯度 + 真实 kernel =="
cd '$H'
python3 -u exp_mqa_logits_paged_backward.py
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
