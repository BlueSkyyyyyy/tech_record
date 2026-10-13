#!/usr/bin/env bash
# P120-1 实验 driver：在 kernel_lab(H100) 上
#   1) 编译并运行 src/mega_mhc_backward.cu（4x4 Sinkhorn 反向 vs 有限差分、
#      整 token 变换反向 device(fp32) vs double、double 参考的有限差分抽样复核）；
#   2) 运行 harness/exp_mega_mhc_backward.py（手写反向 vs torch autograd，float64）。
#   ./harness/exp_mega_mhc_backward.sh
# 输出见 harness/exp_mega_mhc_backward.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mega_mhc_backward.cu"
PY="$ROOT/harness/exp_mega_mhc_backward.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- CUDA (sm90) ---' && nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/mega_mhc_backward && /tmp/mega_mhc_backward && echo '--- PYTHON ---' && python3 '$PY'"
