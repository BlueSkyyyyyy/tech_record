#!/usr/bin/env bash
# P10-1 实验 driver：在 kernel_lab(H100) 上
#   1) 编译并运行 src/bf16_mega_moe_epilogue.cu（BF16 L1/L2 epilogue 数值语义）；
#   2) 运行 harness/exp_bf16_mega_moe.py（block config / symm buffer 尺寸 / 权重重排）。
#   ./harness/exp_bf16_mega_moe.sh
# 输出见 harness/exp_bf16_mega_moe.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/bf16_mega_moe_epilogue.cu"
PY="$ROOT/harness/exp_bf16_mega_moe.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- CUDA ---' && nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/bf16_mega_moe_epilogue && /tmp/bf16_mega_moe_epilogue && echo '--- PYTHON ---' && python3 '$PY'"
