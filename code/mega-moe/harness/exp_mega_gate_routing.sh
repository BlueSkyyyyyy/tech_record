#!/usr/bin/env bash
# P5-1 实验 driver：在 kernel_lab(H100) 编译并运行 Mega Gate 路由算法移植实验。
#   ./harness/exp_mega_gate_routing.sh
# 对应源码 src/mega_gate_routing.cu，输出见 harness/exp_mega_gate_routing.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mega_gate_routing.cu"
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/mega_gate_routing && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && /tmp/mega_gate_routing"
