#!/usr/bin/env bash
# P3-2 实验 driver：在 kernel_lab(H100) 编译并运行 mbarrier 多级流水实验。
#   ./harness/exp_mbarrier_pipeline.sh
# 对应源码 src/mbarrier_pipeline.cu，输出见 harness/exp_mbarrier_pipeline.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mbarrier_pipeline.cu"
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/mbarrier_pipeline && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && /tmp/mbarrier_pipeline"
