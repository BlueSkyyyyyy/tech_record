#!/usr/bin/env bash
# P130-1 实验 driver：DeepEP `comm::Context`（csrc/kernels/comm/context.cpp）的 host 几何算术
# 与 NCCL 运行时版本闸门，在 H100(sm90) 上真机对拍。
#
# 为什么 H100 可跑：
#   * 常量 / advance_ptr / 域分解来自**真实头文件**（src/deepep_context_arith.cu 直接 include
#     deep_ep/common/compiled.cuh + math.cuh），不需要 GPU。
#   * NCCL 版本用 ctypes 直接 ncclGetVersion 读两个真实 libnccl（system 2.29.7 / pip 2.32.3）。
#   * 不碰 `import deep_ep`（会触发 __init__.py:37 check_nccl_so() 二进制比对 → abort）。
#
# 用法：bash harness/exp_deepep_context.sh > harness/exp_deepep_context.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEEPEP="${DEEPEP_ROOT:-${HOME}/github/DeepEP}"
INC="$DEEPEP/deep_ep/include"

echo "DeepEP comm::Context host 算术 + NCCL 运行时版本闸门 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version(), 'sms', torch.cuda.get_device_properties(0).multi_processor_count)\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

"$ROOT/scripts/lab.sh" exec "nvcc -std=c++17 -I '$INC' -o /tmp/deepep_context_arith '$ROOT/src/deepep_context_arith.cu' && /tmp/deepep_context_arith > /tmp/deepep_context_arith.txt && echo 'CUDA host-arithmetic program OK'"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT='$DEEPEP' CUDA_ARITH_OUT=/tmp/deepep_context_arith.txt python '$ROOT/harness/exp_deepep_context.py'"
