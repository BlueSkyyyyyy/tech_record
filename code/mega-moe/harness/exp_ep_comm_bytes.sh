#!/usr/bin/env bash
# P152-1 实验 driver：多卡 EP MoE 真实通信字节口径（FP8 dispatch / BF16 combine）+ dtype 伸缩。
#
# 为什么 H100(sm90) 可跑：
#   * dispatch/combine 的 a2a 用 `torch.distributed`(NCCL) 的 `all_to_all_single`（架构无关）；
#   * 字段口径来自 DeepEP 参考实现 `deep_ep/utils/refs.py`（纯 host 语义）；
#   * 真实 header 字节账见 harness/exp_deepep_token_layout_bytes.sh（直接 include token.cuh）。
#   Mega MoE 融合 kernel 本体 SM100 专属 → 仅分析。
#
# 用法：bash harness/exp_ep_comm_bytes.sh > harness/exp_ep_comm_bytes.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_ep_comm_bytes.py"

echo "多卡 EP MoE 真实通信字节口径（FP8/1B dispatch + BF16/2B combine）—— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo ""

"$ROOT/scripts/lab.sh" exec "EPCB_WORLD=8 python '$PY'"
