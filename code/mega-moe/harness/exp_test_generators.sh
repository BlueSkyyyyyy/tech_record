#!/usr/bin/env bash
# P31-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM 测试操作数/布局生成器
#   tests/generators.py 的枚举器 / QuantConfig / operand major / 三种 grouped 布局 /
#   PSUM 转换 / K-grouped SF 语义与真实 SM90 kernel 端到端对拍。
#   bash harness/exp_test_generators.sh > harness/exp_test_generators.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_test_generators.py"
