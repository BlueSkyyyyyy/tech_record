#!/usr/bin/env bash
# P91-1 实验 driver：DeepEP 屏障的验收基准 `tests/comm/test_barrier.py` 与「通信 kernel 精确测时」。
#
# 为什么 H100(sm90) 可跑：
#   * `import deep_ep` 会触发 `deep_ep/__init__.py` 的 check_nccl_so()（本机 NCCL 2.29.7 与编译期
#     不匹配 → abort）。脚本用 sys.modules 注入合成包 + stub `deep_ep._C` + stub `envs`，再按文件
#     路径 exec 真实源码（打印 sha256）。
#   * 真实 barrier kernel（`impls/comm/barrier.cuh`）走 NCCL GIN 设备 API，本机没有 → 仅分析；
#     但 host 启动公式、域语义、Python barrier 转发、`bench_kineto(barrier_comm_profiling=...)`
#     全在纯 host/Python + torch profiler 里 → H100 真机可跑。
#   * Part F 用真实 `deep_ep.utils.testing.bench_kineto` + 真实 DeepGEMM FP8 kernel，在 8×H100 NCCL
#     上比较三种测时模式的稳定性。
#
# 用法：bash harness/exp_deepep_barrier_profiling.sh > harness/exp_deepep_barrier_profiling.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_barrier_profiling.py"
DEEPEP=~/github/DeepEP

echo "DeepEP 屏障的验收基准与「通信 kernel 精确测时」方法论 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP python '$PY'"
