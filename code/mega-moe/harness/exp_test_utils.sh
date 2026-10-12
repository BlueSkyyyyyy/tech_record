#!/usr/bin/env bash
# P33-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM 测试断言辅助 tests/utils.py。
#   先 nvcc 编 src/test_utils_sr.cu（device 侧随机舍入 oracle），再跑 exp_test_utils.py 逐条对拍。
#   bash harness/exp_test_utils.sh > harness/exp_test_utils.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="/tmp/opencode/p33_test_utils_sr"
"$ROOT/scripts/lab.sh" exec "mkdir -p /tmp/opencode && nvcc -arch=sm_90 -O2 -o '$BIN' '$ROOT/src/test_utils_sr.cu' && \
  cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && \
  TEST_UTILS_SR_BIN='$BIN' python exp_test_utils.py"
