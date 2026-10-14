#!/usr/bin/env bash
# P104-1 实验 driver：DeepEP 的 MoE 语义工具层 `deep_ep/utils/math.py`
#   （calc_diff / per_token_cast_to_fp8 / per_token_cast_back / inplace_unique /
#    create_grouped_scores / hash_tensor(s) / count_bytes / safe_div）在 H100 上真机核验，
#   并与 DeepGEMM 同名但不同定义的 `testing/numeric.py`、`utils/math.py` 对照。
#   bash harness/exp_deepep_utils_math.sh > harness/exp_deepep_utils_math.out.txt
# 对应源码：~/github/DeepEP@93eb6eb deep_ep/utils/math.py:5-103
#           ~/github/DeepGEMM@057ca59 deep_gemm/testing/numeric.py:5-27、
#                                  deep_gemm/utils/math.py:19-49
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEP=${HOME}/github/DeepEP
DGEMM=${HOME}/github/DeepGEMM
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && TORCHDYNAMO_DISABLE=1 DEEPEP_ROOT='$DEP' DEEPGEMM_ROOT='$DGEMM' python exp_deepep_utils_math.py"
