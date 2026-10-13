#!/usr/bin/env bash
# P109-1 实验 driver：在 kernel_lab(H100/sm90) 上验证 DeepEP 的 NVLink 连接自检与 P2P 拓扑。
#   harness/exp_deepep_nvlink_p2p.py（纯 Python + pynvml，直接加载真实 deep_ep/utils/envs.py）
#   src/nvlink_p2p_matrix.cu      （真机 P2P 拷贝带宽矩阵）
#   bash harness/exp_deepep_nvlink_p2p.sh > harness/exp_deepep_nvlink_p2p.out.txt
# 对应源码：~/github/DeepEP@93eb6eb
#   deep_ep/utils/envs.py:146-181（check_nvlink_connections）、消费点 engram.py:67 / ep.py:325
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DE=~/github/DeepEP
PY=~/proj/tech_record/code/mega-moe/harness/exp_deepep_nvlink_p2p.py
CU=~/proj/tech_record/code/mega-moe/src/nvlink_p2p_matrix.cu

echo "== 环境坐标 =="
( cd $DE && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
python -c "import torch, pynvml; print('torch', torch.__version__, 'pynvml ok')"
sha256sum $DE/deep_ep/utils/envs.py | awk '{n=split($2,a,"/"); print $1, a[n]}'

echo
echo "== 编译 P2P 带宽矩阵 =="
nvcc -arch=sm_90 -O3 -std=c++17 "$CU" -o /tmp/nvlink_p2p_matrix
echo "compiled: /tmp/nvlink_p2p_matrix"

echo
TORCHDYNAMO_DISABLE=1 DEEPEP_ROOT="$DE" P2P_BIN=/tmp/nvlink_p2p_matrix python "$PY"
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
