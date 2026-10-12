#!/usr/bin/env bash
# P18-1: SM90 heuristic get_best_config Python 复刻 vs 真实 C++ 选核（H100 真机）。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(cd "$HERE/../../kernel-opt/scripts" && pwd)/lab.sh" exec \
  "cd ~/proj/tech_record/code/mega-moe && DG_PRINT_CONFIGS=1 PYTHONUNBUFFERED=1 python harness/exp_best_config.py"
