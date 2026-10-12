#!/usr/bin/env bash
# P40-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM 运行时对象生命周期实验。
#   bash harness/exp_runtime_lifecycle.sh > harness/exp_runtime_lifecycle.out.txt
# 对应源码 exp_runtime_lifecycle.py；参考 ~/github/DeepGEMM 的 csrc/runtime/runtime.hpp、
# csrc/apis/config.hpp、deep_gemm/__init__.py、deep_jit/utils/{lazy,env}.hpp、tests/test_lazy_init.py。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_runtime_lifecycle.py"
