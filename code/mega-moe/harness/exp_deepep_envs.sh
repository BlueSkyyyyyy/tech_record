#!/usr/bin/env bash
# P107-1 实验 driver：在 kernel_lab(H100/sm90) 上验证 DeepEP 的环境变量 / 开关面。
#   Part A: harness/deepep_env_probe.cpp（g++ host，真实 include csrc/utils/system.hpp::get_env）
#   Part B: harness/exp_deepep_envs.py（纯 Python，复用真实生成的 deep_ep/envs.py + check_nccl_so）
#   bash harness/exp_deepep_envs.sh > harness/exp_deepep_envs.out.txt
# 对应源码：~/github/DeepEP@93eb6eb
#   csrc/utils/system.hpp:12-29、csrc/runtime/jit.hpp:43、csrc/kernels/comm/context.cpp:103-154
#   csrc/buffers/ep.hpp:163/454/653、csrc/buffers/bucket.hpp:51
#   deep_ep/__init__.py:8-16/19-41、deep_ep/comm/handle.py:64、deep_ep/buffers/engram.py:61-66
#   deep_ep/utils/testing.py:145/154、setup.py:14-15/63-74
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DE=${HOME}/github/DeepEP
DG=${HOME}/github/DeepGEMM
PROBE=${HOME}/proj/tech_record/code/mega-moe/harness/deepep_env_probe.cpp
PY=${HOME}/proj/tech_record/code/mega-moe/harness/exp_deepep_envs.py

echo "== 环境坐标 =="
( cd $DE && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
python -c "import torch; print('torch', torch.__version__)"
sha256sum $DE/csrc/utils/system.hpp $DE/setup.py $DE/deep_ep/__init__.py | awk '{n=split($2,a,"/"); print $1, a[n]}'
g++ --version | head -1

echo
echo "############ Part A: get_env<T> 语义（g++ host，真实 system.hpp）############"
g++ -std=c++20 -O0 -w -I$DE -I$DE/deep_ep/include "$PROBE" -o /tmp/deepep_env_probe
echo "> build OK"
/tmp/deepep_env_probe

echo
echo "############ Part B: persistent envs / 闸门 / 清单普查（Python）############"
TORCHDYNAMO_DISABLE=1 DEEPEP_ROOT="$DE" DEEPGEMM_ROOT="$DG" python "$PY"
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
