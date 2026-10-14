#!/usr/bin/env bash
# P118-1 实验 driver：在 kernel_lab(H100/sm90) 上验证 DeepGEMM 的 NCU PM-sampling 离线绘图工具。
#   harness/exp_quick_plot_pm.py（纯 host/Python，importlib 按路径加载真实 scripts/quick_plot_pm.py）
#   bash harness/exp_quick_plot_pm.sh > harness/exp_quick_plot_pm.out.txt
# 对应源码：~/github/DeepGEMM@057ca59（deep_gemm 2.8.1, v2.1.1.post3-24-g057ca59）
#   scripts/quick_plot_pm.py:1-9/21-43/47-55/57-74/77-125/126-132/134-153/156-161/
#                            164-181/184-198/201-224/227-248/251-271/274-417/420-448
# 真机部分用 ncu 2026.1.1.0 的 `--query-metrics --chip gh100`（H100，非目标卡）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=${HOME}/github/DeepGEMM
PY=${HOME}/proj/tech_record/code/mega-moe/harness/exp_quick_plot_pm.py

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' describe --tags )
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
ncu --version | sed -n '1,4p'
python -c "import sys; print('python', sys.version.split()[0])"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo
DEEPGEMM_ROOT="$DG" python "$PY"
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
