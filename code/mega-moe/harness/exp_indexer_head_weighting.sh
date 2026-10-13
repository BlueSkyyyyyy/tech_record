#!/usr/bin/env bash
# P165-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 indexer 蒸馏 loss 的【逐头权重】研究——
#   逐头权重 λ 决定 indexer 学到的目标分布 p_λ=Σ_h λ_h P_h（KL 不动点）、支持集 Top-k(p_λ)，
#   再拿真实下游输出误差（per-head softmax + GQA 共享）评判 λ 该按熵 / 值范数 / 均匀。
#   bash harness/exp_indexer_head_weighting.sh > harness/exp_indexer_head_weighting.out.txt
# 对应源码 exp_indexer_head_weighting.py；参考 ~/github/DeepGEMM
#   csrc/apis/attention.hpp、tests/test_attention.py、README:88-110。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_indexer_head_weighting.py"
