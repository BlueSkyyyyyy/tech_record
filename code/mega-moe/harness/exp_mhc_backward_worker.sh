#!/usr/bin/env bash
# P122-1 实验 driver：在 kernel_lab(H100) 上
#   1) 编译并运行 src/mhc_backward_worker.cu —— per-token 融合 worker（K1+G1a+K2 一个 block）
#      device(fp32) vs host 单片 double 参考对拍，并与六段分段路径计时对比（H=64）。
#   2) 运行 harness/exp_mhc_backward_worker.py —— 真实形状（H=7168）下「融合省掉的全局往返」
#      的字节账 + H100 实测带宽 → 投影的省时。
#   ./harness/exp_mhc_backward_worker.sh
# 输出见 harness/exp_mhc_backward_worker.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mhc_backward_worker.cu"
PY="$ROOT/harness/exp_mhc_backward_worker.py"
OUT=~/tmp/mhc_bwd_worker
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- CUDA (sm90) ---' && nvcc -arch=sm_90 -O3 -std=c++17 -Xptxas -v '$SRC' -o $OUT 2>&1 | grep -E 'error|Used|spill' && echo '--- RUN CU ---' && $OUT && echo '--- PYTHON ---' && python3 '$PY'"
