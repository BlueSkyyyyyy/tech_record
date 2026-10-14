#!/usr/bin/env bash
# P90-1 实验 driver：DeepEP EPLB「权重预取 / 梯度归约」的验收测试即正确性规范
# （tests/ep/test_prefetch_weights.py + tests/ep/test_reduce_grads.py）。
#
# 为什么 H100(sm90) 只能跑「测试侧架构无关部分」：
#   * 端到端测试要构造 `deep_ep.EPBuffer` 并调用 `lb_prefetch_weights`/`lb_reduce_grads`，其设备
#     kernel 用 NCCL 设备 API（ncclDevComm/ncclGin）；本机系统 NCCL 2.29.7 与 pip-linked 2.32.3
#     二进制不一致 → deep_ep/__init__.py 的 check_nccl_so 先拦。故 kernel 路径**端到端仅分析**。
#   * 但测试把「什么叫对」全编码在纯 Python/纯张量 oracle 里：SENTINEL/make_blob（唯一 key ramp）、
#     make_redundancy_mapping（own-expert 排除 / empty_ratio / ring vs spread）、check（哨兵 vs blob）、
#     make_pattern（精确 fp32 + 原子加顺序无关）、expected_expert_grads。按文件路径 ast 抽取原样运行。
#
# 用法：bash harness/exp_deepep_eplb_test_oracle.sh > harness/exp_deepep_eplb_test_oracle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_eplb_test_oracle.py"
DEEPEP=${HOME}/github/DeepEP

echo "DeepEP EPLB 权重预取/梯度归约的验收测试即正确性规范 —— H100 真机（测试侧架构无关部分）"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP python '$PY'"
