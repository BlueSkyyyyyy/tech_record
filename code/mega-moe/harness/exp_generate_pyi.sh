#!/usr/bin/env bash
# P115-1 实验 driver：在 kernel_lab(H100/sm90) 上验证 DeepGEMM 的 Python 类型桩生成器。
#   harness/exp_generate_pyi.py（纯 host/Python，按路径加载真实 scripts/generate_pyi.py）
#   bash harness/exp_generate_pyi.sh > harness/exp_generate_pyi.out.txt
# 对应源码：~/github/DeepGEMM@057ca59（deep_gemm 2.8.1, v2.1.1.post3-24-g057ca59）
#   scripts/generate_pyi.py:5-94/97-148/151-222/225-342/345-386/389-414/417-462/
#                           494-578/608-696/699-720/723-778/781-825/828-869/872-890
#   setup.py:19/120/136-146（generate_pyi_file(name='_C', root='./csrc', output_dir='./stubs')）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=~/github/DeepGEMM
PY=~/proj/tech_record/code/mega-moe/harness/exp_generate_pyi.py

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' describe --tags )
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
python -c "import sys; print('python', sys.version.split()[0])"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo
DEEPGEMM_ROOT="$DG" python "$PY"
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
