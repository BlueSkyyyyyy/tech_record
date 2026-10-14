#!/usr/bin/env bash
# P73-1 实验 driver：DeepEP 的【EPHandle 句柄契约 + dispatch/combine host 编排层】。
#
# 为什么 H100(sm90) 可跑：
#   * `EPHandle` 是纯 Python 数据容器；`EPBuffer.dispatch`/`combine` 的 host 编排（auto SM/QP、
#     SF/bias 解包、句柄复用断言、deferred epilogue、确定性排序钩子接线）全在 Python，与 NCCL
#     设备 API 无关。脚本把**未修改的真实方法体**绑到假 buffer 上，假 runtime 记录参数并回放
#     合成 result，逐字段核对编排。
#   * `deterministic_sort` 用真实 CUDA 张量在 H100 上跑。
#   * `import deep_ep` 会触发 NCCL so 版本校验 abort，故用 sys.modules 注入合成 `deep_ep._C`
#     与 `deep_ep.comm` 桩后按文件路径 exec 真实文件。
#
# 用法：bash harness/exp_deepep_epbuffer_handle.sh > harness/exp_deepep_epbuffer_handle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_epbuffer_handle.py"
DEP=${HOME}/github/DeepEP

echo "DeepEP EPHandle 句柄契约 + dispatch/combine host 编排层 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo "DeepEP 源码：$(git -C "$DEP" rev-parse --short HEAD 2>/dev/null || echo '?')  ($(git -C "$DEP" log -1 --format='%ci' 2>/dev/null || echo '?'))"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEP python '$PY'"
