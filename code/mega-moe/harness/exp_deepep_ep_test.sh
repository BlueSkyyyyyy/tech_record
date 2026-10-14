#!/usr/bin/env bash
# P75-1 实验 driver：DeepEP 的 EP dispatch/combine 端到端验收测试即正确性规范（tests/ep/test_ep.py）。
#
# 为什么 H100(sm90) 只能跑「测试侧架构无关部分」：
#   * 端到端测试要构造 `deep_ep.EPBuffer`，依赖 DeepEP 的 C++/NCCL 设备 API（ncclDevComm/ncclGin）；
#     本机 NCCL 2.29.7 与编译期不匹配 → `deep_ep/__init__.py:55 check_nccl_so()` abort；跑真正 dispatch
#     还需 ≥2.31 的 NCCL。故 buffer 端到端**仅分析**。
#   * 测试侧大量纯 Python/张量逻辑可原样抽取运行：`enumerate_ep_modes`（模式矩阵）、`fold_expanded`
#     （expand 折叠不变量）、handle 两套 psum 前缀和语义、`get_unique_and_valid_dst_count`（带宽账目）。
#   * refs/gate/math 按文件路径 exec（sys.modules 注入合成包绕过 check_nccl_so）。
#   * 8 卡 H100 真机 mp.spawn + NCCL 跑 refs.dispatch，坐实测试「按 src 全局索引升序」断言的根。
#
# 用法：bash harness/exp_deepep_ep_test.sh > harness/exp_deepep_ep_test.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_ep_test.py"
DEEPEP=${HOME}/github/DeepEP

echo "DeepEP EP dispatch/combine 端到端验收测试即正确性规范 —— H100 真机（测试侧架构无关部分）"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP python '$PY'"
