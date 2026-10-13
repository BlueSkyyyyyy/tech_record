#!/usr/bin/env bash
# P155-1 实验 driver：跨 micro-batch 流水线能不能藏 EP 通信（dispatch_{i+1} ∥ compute_i）。
#
# 两个部分：
#   A) 负结果（真机通信）：真实 SM90 grouped GEMM × NCCL a2a 的 micro-batch 流水线（8×H100，多进程）。
#   B) 正对照（SM 预算）：单进程 8 卡 NVLink 的 D_i→K_i→C_i 依赖链，扫通信块数 B（src/pipeline_overlap_nvlink.cu）。
#
# 为什么 H100(sm90) 能跑：
#   * 通信腿 = `torch.distributed`(NCCL) / NVLink peer scatter（架构无关）；
#   * 计算腿 = **真实 DeepGEMM SM90 FP8 grouped kernel** `m_grouped_fp8_gemm_nt_contiguous`；
#   * Mega MoE 融合 kernel 本体是 SM100 专属 → **仅分析**。
#
# 用法：bash harness/exp_pipeline_overlap.sh > harness/exp_pipeline_overlap.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_pipeline_overlap.py"
CU="$ROOT/src/pipeline_overlap_nvlink.cu"

echo "跨 micro-batch 流水线能否藏 EP 通信（NCCL a2a × 真实 SM90 grouped GEMM）—— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo ""

echo "===== A) 负结果：NCCL a2a 的 micro-batch 流水线（8×H100，多进程） ====="
"$ROOT/scripts/lab.sh" exec "POV_WORLD=8 POV_ITERS=10 python '$PY'"
echo ""

echo "===== B) 正对照：SM 预算下的 NVLink 流水线（单进程 8 卡，扫 B / M） ====="
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90 -O3 -std=c++17 '$CU' -o /tmp/pipeline_overlap_nvlink && /tmp/pipeline_overlap_nvlink"
