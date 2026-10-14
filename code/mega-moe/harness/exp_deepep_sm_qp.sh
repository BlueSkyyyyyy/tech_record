#!/usr/bin/env bash
# P63-1 实验 driver：DeepEP dispatch/combine 的【自动 SM 数 / QP 数选择模型】与带宽探测层。
#
# 为什么 H100(sm90) 可跑：
#   * 这一层是纯 host / Python 解析式 roofline 模型，不碰 CUDA / NCCL 设备 API。
#   * `import deep_ep` 会触发 `deep_ep/__init__.py:55 check_nccl_so()`（本机 NCCL 2.29.7 与编译期
#     不匹配 → abort）。故脚本用 sys.modules 注入合成 `deep_ep._C` / `deep_ep.comm` stub 后按文件
#     路径 exec 真实 `envs.py`，再用 ast 从真实 `ep.py` 抽取三个方法源码运行真实方法体。
#   * 带宽探测走真实 `nvidia-smi nvlink -s`（本机无 ibstat → rdma=0）。
#
# 用法：bash harness/exp_deepep_sm_qp.sh > harness/exp_deepep_sm_qp.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_sm_qp.py"
DEEPEP=${HOME}/github/DeepEP

echo "DeepEP dispatch/combine 自动 SM/QP 选择模型 + 带宽探测层 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version(), 'sms', torch.cuda.get_device_properties(0).multi_processor_count)\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP python '$PY'"
