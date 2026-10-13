#!/usr/bin/env bash
# P131-1 实验 driver：8×H100 上的【多卡 EP 端到端 MoE 前向/反向】与单卡稠密参考的梯度核验。
#
# 为什么 H100(sm90) 能跑：
#   * 全篇是**纯 PyTorch 张量 + NCCL all_to_all_single**（架构无关，不碰 UMMA/TMA/tcgen05）。
#   * 端到端 `deep_ep.EPBuffer.dispatch/combine` 依赖 NCCL 设备 API（ncclDevComm/ncclGin），
#     本机 NCCL 2.29.7 不匹配、运行需 ≥2.31 → **仅分析**；这里用 `all_to_all_single` 实现
#     expand 版 dispatch/combine（行空间与 DeepEP `do_expand=True` 一致），在真机 NCCL 上跑真实通信。
#   * Mega MoE 融合 kernel 本体是 SM100 专属（`sm100_fp8_fp4_mega_moe.cuh`）→ 仅分析。
#
# 用法：bash harness/exp_ep_e2e_moe.sh > harness/exp_ep_e2e_moe.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_ep_e2e_moe.py"

echo "多卡 EP 端到端 MoE（dispatch→L1→SwiGLU→L2→combine）前向/反向 vs 单卡稠密 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo ""

"$ROOT/scripts/lab.sh" exec "EP_E2E_WORLD=8 python '$PY'"
