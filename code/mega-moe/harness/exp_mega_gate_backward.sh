#!/usr/bin/env bash
# P117-1 实验 driver：在 kernel_lab(H100) 上
#   1) 编译并运行 src/mega_gate_backward.cu（sqrt-softplus 前向/导数、排名单调性、
#      归一化反向 Jacobian、整 token gate 反向的设备代码对拍）；
#   2) 运行 harness/exp_mega_gate_backward.py（手写反向 vs torch autograd）。
#   ./harness/exp_mega_gate_backward.sh
# 输出见 harness/exp_mega_gate_backward.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mega_gate_backward.cu"
PY="$ROOT/harness/exp_mega_gate_backward.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- CUDA (sm90) ---' && nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/mega_gate_backward && /tmp/mega_gate_backward && echo '--- PYTHON ---' && python3 '$PY'"
