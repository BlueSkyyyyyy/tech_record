#!/usr/bin/env bash
# P38-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM 全局运行时/编译开关面。
#   bash harness/exp_runtime_switches.sh > harness/exp_runtime_switches.out.txt
# 对应源码 exp_runtime_switches.py；参考 ~/github/DeepGEMM 的 csrc/apis/config.hpp、heuristics/runtime.hpp、
# runtime/runtime.hpp、heuristics/{sm90,sm100}.hpp。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_runtime_switches.py"
