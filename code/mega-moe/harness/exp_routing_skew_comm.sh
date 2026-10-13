#!/usr/bin/env bash
# P156-1 实验驱动：路由偏斜对 EP 通信/显存的真实代价（H100 sm90，DeepEP 真实源码 + 8 卡真实 a2a）。
# 运行：bash code/mega-moe/harness/exp_routing_skew_comm.sh > code/mega-moe/harness/exp_routing_skew_comm.out.txt
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
LAB="$REPO/code/kernel-opt/scripts/lab.sh"
PY="$HERE/exp_routing_skew_comm.py"

export EPSC_WORLD="${EPSC_WORLD:-8}"
export EPSC_T="${EPSC_T:-512}"

"$LAB" exec "cd $REPO/code/kernel-opt && EPSC_WORLD=$EPSC_WORLD EPSC_T=$EPSC_T python $PY"
