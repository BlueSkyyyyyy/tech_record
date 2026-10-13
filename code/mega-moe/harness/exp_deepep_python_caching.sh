#!/usr/bin/env bash
# P113-1 实验 driver：在 kernel_lab(H100/sm90) 上验证 DeepEP 的 Python 缓存基座。
#   harness/exp_deepep_python_caching.py（纯 Python，按路径加载真实 semantic/find_pkgs/envs.py）
#   bash harness/exp_deepep_python_caching.sh > harness/exp_deepep_python_caching.out.txt
# 对应源码：~/github/DeepEP@93eb6eb（v1.2.1-41, __version__ 2.5.0）
#   deep_ep/utils/semantic.py:5-26、deep_ep/buffers/ep.py:430-431/845-846、
#   deep_ep/utils/{find_pkgs.py:58,envs.py:193/200/207/237/260}、deep_ep/buffers/engram.py:130
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DE=~/github/DeepEP
PY=~/proj/tech_record/code/mega-moe/harness/exp_deepep_python_caching.py

echo "== 环境坐标 =="
( cd $DE && git -c safe.directory='*' log --oneline -1 )
( cd $DE && git -c safe.directory='*' describe --tags )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
python -c "import torch; print('torch', torch.__version__)"

echo
TORCHDYNAMO_DISABLE=1 DEEPEP_ROOT="$DE" python "$PY"
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
