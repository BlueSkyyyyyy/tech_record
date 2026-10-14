#!/usr/bin/env bash
# P76-1 实验 driver：DeepEP 的【NCCL 通信域绑定与 Python 引导层】。
#
# 为什么 H100(sm90) 可跑：
#   * 本层纯 host / Python：包发现、NCCL 版本核对、comm 句柄生命周期、域大小算术，不碰
#     CUDA / NCCL 设备 API 与任何 kernel。
#   * `import deep_ep` 会触发 `deep_ep/__init__.py:55 check_nccl_so()`（本机 PyTorch 链系统
#     NCCL 2.29.7 ≠ DeepEP 链接的 pip NCCL 2.32.3 → AssertionError）。故脚本用 sys.modules
#     注入合成 `deep_ep._C` stub 后按文件路径 exec 真实 `comm/handle.py`、`comm/barrier.py`，
#     并 ast 抽取真实 `check_nccl_so` 函数体运行。
#
# 用法：bash harness/exp_deepep_nccl_binding.sh > harness/exp_deepep_nccl_binding.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_nccl_binding.py"
DEEPEP=${HOME}/github/DeepEP

echo "DeepEP NCCL 通信域绑定与 Python 引导层 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'torch-nccl', torch.cuda.nccl.version(), 'sms', torch.cuda.get_device_properties(0).multi_processor_count)\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
"$ROOT/scripts/lab.sh" exec "python -c \"import importlib.util; print('installed deep_ep:', importlib.util.find_spec('deep_ep').origin)\""
echo ""

set +e
"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP python '$PY'"
rc=$?
set -e
echo ""
echo "exit code: $rc"
exit $rc
