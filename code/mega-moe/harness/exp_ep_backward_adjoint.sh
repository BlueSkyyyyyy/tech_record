#!/usr/bin/env bash
# P85-1 实验 driver：MoE dispatch/combine 的反向（伴随）语义与路由权重梯度。
#
# 为什么 H100(sm90) 能跑：
#   * dispatch/combine 反向恒等式（「dispatch 的反向是 combine、combine 的反向是 dispatch」）与其伴随性
#     是纯线性代数事实，与 SM100 / NCCL 设备 API 无关。
#   * 端到端 `deep_ep.EPBuffer.dispatch/combine` 依赖 C++/NCCL 设备 API（ncclDevComm/ncclGin），
#     本机 NCCL 2.29.7 不匹配、运行需 ≥2.31 → **仅分析**；但真实 `refs.dispatch` 只用 torch 自带
#     `all_to_all_single`，可在 8×H100 真机 NCCL 上跑真实跨 rank 通信。
#   * refs/math 按文件路径 exec（sys.modules 注入合成包绕过 check_nccl_so）。
#
# 用法：bash harness/exp_ep_backward_adjoint.sh > harness/exp_ep_backward_adjoint.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_ep_backward_adjoint.py"
DEEPEP=${HOME}/github/DeepEP

echo "MoE dispatch/combine 的反向（伴随）语义与路由权重梯度 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP EP_BWD_WORLD=8 python '$PY'"
