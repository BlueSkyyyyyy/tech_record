#!/usr/bin/env bash
# P55-1 实验 driver：DeepEP 的 dispatch/combine【参考实现（语义规范）】与流量/基准生成器。
#
# 为什么 H100(sm90) 可跑：
#   * refs.py / gate.py / testing.py 全是纯 PyTorch（refs 只用 dist 的 get_rank/get_world_size +
#     all_to_all_single，即 torch 自带 NCCL），不碰 DeepEP 的 C++/NCCL 设备 API。
#   * `import deep_ep` 会触发 `deep_ep/__init__.py:55 check_nccl_so()`（本机 NCCL 2.29.7 与编译期
#     不匹配 → abort）。故脚本用 sys.modules 注入合成包 + stub `deep_ep.utils.envs`，再按文件路径
#     exec 真实源码（打印 sha256）。
#   * 8 卡 H100 真机 mp.spawn + NCCL 跑真实 all_to_all，对 refs.dispatch 做独立 oracle 对拍；
#     combine/gate/testing 是单进程张量运算，rank0 直接跑。
#
# 用法：bash harness/exp_deepep_refs.sh > harness/exp_deepep_refs.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_deepep_refs.py"
DEEPEP=~/github/DeepEP

echo "DeepEP dispatch/combine 参考实现（语义规范）+ 流量/基准生成器 —— H100 真机"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1"
"$ROOT/scripts/lab.sh" exec "python -c \"import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())\""
echo "DeepEP 源码：$(git -C "$DEEPEP" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo ""

"$ROOT/scripts/lab.sh" exec "DEEPEP_ROOT=$DEEPEP python '$PY'"
