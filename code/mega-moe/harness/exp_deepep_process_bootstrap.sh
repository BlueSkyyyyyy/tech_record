#!/usr/bin/env bash
# P108-1 实验 driver：在 kernel_lab(H100/sm90) 上验证 DeepEP 的测试/进程引导层。
#   harness/exp_deepep_process_bootstrap.py（纯 Python，直接加载真实 deep_ep/utils/envs.py）
#   bash harness/exp_deepep_process_bootstrap.sh > harness/exp_deepep_process_bootstrap.out.txt
# 对应源码：~/github/DeepEP@93eb6eb
#   deep_ep/utils/envs.py:22-37/40-51/54-61/64-71/74-86/89-133/184-190
#   deep_ep/utils/refs.py:151、deep_ep/buffers/engram.py:81-85、deep_ep/buffers/ep.py:18-21/639/813
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DE=${HOME}/github/DeepEP
PY=${HOME}/proj/tech_record/code/mega-moe/harness/exp_deepep_process_bootstrap.py

echo "== 环境坐标 =="
( cd $DE && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
python -c "import torch; print('torch', torch.__version__)"
sha256sum $DE/deep_ep/utils/envs.py $DE/deep_ep/utils/semantic.py | awk '{n=split($2,a,"/"); print $1, a[n]}'

echo
TORCHDYNAMO_DISABLE=1 DEEPEP_ROOT="$DE" python "$PY"
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
