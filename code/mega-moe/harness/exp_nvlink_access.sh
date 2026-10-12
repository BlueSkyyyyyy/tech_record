#!/usr/bin/env bash
# P3-4 实验 driver：在 kernel_lab(H100) 编译并运行「本地 vs NVLink 远端访存」实验。
#   ./harness/exp_nvlink_access.sh
# 对应源码 src/nvlink_access.cu，输出见 harness/exp_nvlink_access.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/nvlink_access.cu"
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/nvlink_access && nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader | head -1 && /tmp/nvlink_access"
