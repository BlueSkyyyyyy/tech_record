#!/usr/bin/env bash
# P121-1 实验 driver：在 kernel_lab(H100) 上
#   1) 编译并运行 src/mhc_backward_fused.cu —— 把 mHC 反向拆成 6 段（K0/G0/K1/G1a/G1b/K2），
#      device(fp32) 六段 vs host 单片 double 参考对拍；并计时元素级三段（H=64）。
#   2) 运行 harness/exp_mhc_backward_fused.py —— 真实形状（H=7168）的手写反向 vs autograd、
#      逐段 FLOPs/字节/AI 账、三段 GEMM 与元素级 op 的真实计时。
#   ./harness/exp_mhc_backward_fused.sh
# 输出见 harness/exp_mhc_backward_fused.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mhc_backward_fused.cu"
PY="$ROOT/harness/exp_mhc_backward_fused.py"
OUT=${HOME}/tmp/mhc_bwd_fused
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- CUDA (sm90) ---' && nvcc -arch=sm_90 -O3 -std=c++17 -Xptxas -v '$SRC' -o $OUT 2>&1 | grep -E 'Used|stack frame|spill' && echo '--- RUN CU ---' && $OUT && echo '--- PYTHON ---' && python3 '$PY'"
