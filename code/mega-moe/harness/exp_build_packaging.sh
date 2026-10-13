#!/usr/bin/env bash
# P114-1 实验 driver：在 kernel_lab(H100/sm90) 上验证两库的「构建 / 打包 / 安装期地基」。
#   harness/exp_build_packaging.py（纯 host/Python，按路径加载真实 find_pkgs.py + ast 抽取两个 setup.py）
#   bash harness/exp_build_packaging.sh > harness/exp_build_packaging.out.txt
# 对应源码：~/github/DeepEP@93eb6eb（v1.2.1-41, __version__ 2.5.0）
#   deep_ep/utils/find_pkgs.py:9-55/58-69/72-78、setup.py:14-15/17-20/23-40/43-74/77-140
#           ~/github/DeepGEMM@057ca59（v2.1.1.post3-24）
#   setup.py:22-24/80-96/99-108/111-174/176-200、deep_gemm/__init__.py:5-13
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DE=~/github/DeepEP
DG=~/github/DeepGEMM
PY=~/proj/tech_record/code/mega-moe/harness/exp_build_packaging.py

echo "== 环境坐标 =="
( cd $DE && git -c safe.directory='*' log --oneline -1 )
( cd $DE && git -c safe.directory='*' describe --tags )
( cd $DG && git -c safe.directory='*' log --oneline -1 )
( cd $DG && git -c safe.directory='*' describe --tags )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import nvidia.nccl, os; print('nvidia-nccl pkg:', os.path.dirname(nvidia.nccl.__file__))" 2>&1 | tail -1 || true

echo
DEEPEP_ROOT="$DE" DEEPGEMM_ROOT="$DG" python "$PY"
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
