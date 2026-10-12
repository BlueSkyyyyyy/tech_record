#!/usr/bin/env bash
# P5-2 实验 driver：在 kernel_lab(H100) 编译并运行 Mega mHC 的架构无关数值语义移植实验。
#   ./harness/exp_mega_mhc_semantics.sh
# 对应源码 src/mega_mhc_semantics.cu，输出见 harness/exp_mega_mhc_semantics.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mega_mhc_semantics.cu"
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/mega_mhc_semantics && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && /tmp/mega_mhc_semantics"
