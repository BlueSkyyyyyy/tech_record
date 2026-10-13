#!/usr/bin/env bash
# P84-1 实验 driver：DeepEP BucketBuffer 混合集合算子的验收测试即正确性规范
#（tests/bucket/test_mixed.py + tests/bucket/test_perf.py）。
#
# 为什么 H100(sm90) 只能跑「测试侧架构无关部分」：
#   * 端到端要构造 `deep_ep.BucketBuffer`，依赖 DeepEP 的 C++/NCCL 设备 API（ncclDevComm/ncclGin）；
#     本机 NCCL 2.29.7 与编译期不匹配 → `deep_ep/__init__.py:55 check_nccl_so()` abort。故 buffer
#     端到端**仅分析**。
#   * 测试侧规范逻辑（supported / make_sizes / make_schedule / make_groups / make_inputs /
#     nccl_reference / verify / measure / get_skip_reason）是纯 Python / 纯张量，可原样抽取运行。
#   * 8 卡 H100 真机 mp.spawn + NCCL 跑真实 `nccl_reference` 与 `rs_ag == all_reduce` 不变量。
#
# 用法：bash harness/exp_deepep_bucket_test_oracle.sh > harness/exp_deepep_bucket_test_oracle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_bucket_test_oracle.py"
DEEPEP=~/github/DeepEP

echo "DeepEP BucketBuffer 混合集合算子验收测试即正确性规范 —— H100 真机（测试侧架构无关部分）"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP python '$PY'"
