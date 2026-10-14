#!/usr/bin/env bash
# P166-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 indexer↔主注意力【真实低秩共享 q/k】头结构——
#   逐头对齐 rho_h 谱、头间相关矩阵/有效头数 H_eff、真实头结构上重跑逐头权重 lambda 方案，
#   检验第 175 篇「按值范数 nu 而不是熵」在头间相关下是否稳健。
#   bash harness/exp_indexer_lowrank_structure.sh > harness/exp_indexer_lowrank_structure.out.txt
# 对应源码 exp_indexer_lowrank_structure.py；参考 ~/github/DeepGEMM
#   csrc/apis/attention.hpp、tests/test_attention.py、README:88-110。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_indexer_lowrank_structure.py"
