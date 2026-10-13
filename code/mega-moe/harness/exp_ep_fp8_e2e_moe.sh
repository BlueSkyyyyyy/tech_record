#!/usr/bin/env bash
# P151-1 实验 driver：8×H100 上的【多卡 EP MoE + 真实 SM90 FP8 grouped kernel】端到端误差。
#
# 为什么 H100(sm90) 能跑：
#   * EP 通信用 `torch.distributed`(NCCL) 的 `all_to_all_single`（架构无关），
#     dispatch/combine 的行空间与 DeepEP `do_expand=True` 一致（docs/141）；
#   * 计算腿是**真实 DeepGEMM SM90 FP8 grouped kernel**（`m_grouped_*`/`k_grouped_*`，sm90 有实现）；
#   * Mega MoE 融合 kernel 本体是 SM100 专属（`sm100_fp8_fp4_mega_moe.cuh`）→ **仅分析**。
#
# 用法：bash harness/exp_ep_fp8_e2e_moe.sh > harness/exp_ep_fp8_e2e_moe.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_ep_fp8_e2e_moe.py"

echo "EP + 真实 SM90 FP8 grouped kernel 端到端 MoE（前向/反向 vs 单卡稠密 fp64）—— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo ""

"$ROOT/scripts/lab.sh" exec "EPFP8_WORLD=8 python '$PY'"
