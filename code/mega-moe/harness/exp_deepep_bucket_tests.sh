#!/usr/bin/env bash
# P124-1 实验 driver：H100(sm90) 8 卡上复现 DeepEP bucket/buffer 家族最后三份验收测试的测试侧语义。
#   Part C/D 单进程纯 host（缓冲区算术 / BufferAllocator 物化 / 多 group / SM 模型）
#   Part A/B 8×H100 NCCL（all_gather / reduce_scatter / all_reduce 参考与容差梯度）
#   ./harness/exp_deepep_bucket_tests.sh
# 输出见 harness/exp_deepep_bucket_tests.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_bucket_tests.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python3 '$PY'"
