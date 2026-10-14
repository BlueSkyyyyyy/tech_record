#!/usr/bin/env bash
# P106-1 实验 driver：DeepEP 的 MoE 路由分数/拓扑生成器 `deep_ep/utils/gate.py`
#   （generate_topk_idx / generate_rank_count / get_precise_unbalanced_scores /
#    map_unbalanced_ratio_to_factor / get_unbalanced_scores）在 H100 真机核验，
#   复刻 tests/utils/test_gate.py 的验收判据。
#   bash harness/exp_deepep_gate.sh > harness/exp_deepep_gate.out.txt
# 对应源码：~/github/DeepEP@93eb6eb deep_ep/utils/gate.py:4-180
#           ~/github/DeepEP@93eb6eb tests/utils/test_gate.py:7-53
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEP=${HOME}/github/DeepEP
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && TORCHDYNAMO_DISABLE=1 DEEPEP_ROOT='$DEP' python exp_deepep_gate.py"
