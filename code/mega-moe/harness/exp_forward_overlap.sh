#!/usr/bin/env bash
# P134-1 实验 driver：Mega MoE **前向 step 的通信-计算重叠**量化（第 142/143 篇「必须重叠」结论的真机验证）。
#
# 两部分：
#   A) 负结果（NCCL a2a × 真实 grouped GEMM）：把 NCCL a2a 发在侧流、DeepGEMM grouped GEMM 发在主
#      **流，扫 NCCL_MAX_NCHANNELS，量 overlap_factor。→ harness/exp_forward_overlap.py（8 进程 NCCL）
#   B) 正对照（SM 预算分区）：单进程 8 卡 NVLink 的**限块数** peer scatter ∥ 计算受限 kernel，
#      扫通信块数 B，量 overlap_factor。→ src/forward_overlap_nvlink.cu
#
#   bash harness/exp_forward_overlap.sh > harness/exp_forward_overlap.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/forward_overlap_nvlink.cu"
PY="$ROOT/harness/exp_forward_overlap.py"

echo "P134-1 Mega MoE 前向 step 的通信-计算重叠 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version(), 'sms', torch.cuda.get_device_properties(0).multi_processor_count)\""
echo ""

echo "======================================================================"
echo "A) 负结果：侧流 NCCL a2a（dispatch fp8 + combine bf16，~88MB/rank）"
echo "   × 真实 grouped GEMM（DeepGEMM SM90 L1+L2，DeepSeek 量级），扫 NCCL_MAX_NCHANNELS"
echo "======================================================================"
for mc in 0 8 16 32; do
  for r in 1 3; do
    "$ROOT/scripts/lab.sh" exec \
      "cd '$ROOT/harness' && CUDA_VISIBLE_DEVICES=\${FOV_GPUS:-1,2,3,4,5,6,7,0} python exp_forward_overlap.py --maxch $mc --reps $r --iters 20" \
      2>&1 | grep -E '^RESULT'
  done
done
echo ""

echo "======================================================================"
echo "B) 正对照：单进程 8 卡 NVLink 限块数 peer scatter ∥ 计算受限 kernel"
echo "   （SM 预算分区 B 给通信、其余给计算）"
echo "======================================================================"
"$ROOT/scripts/lab.sh" exec \
  "nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/forward_overlap_nvlink 2>/dev/null && \
   CUDA_VISIBLE_DEVICES=\${FOV_GPUS:-1,2,3,4,5,6,7,0} /tmp/forward_overlap_nvlink"
