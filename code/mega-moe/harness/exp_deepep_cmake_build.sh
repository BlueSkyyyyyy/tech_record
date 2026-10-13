#!/usr/bin/env bash
# P119-1 实验 driver：在 kernel_lab(H100/sm90) 上真跑 DeepEP 的 CMake 调试构建链。
#   harness/exp_deepep_cmake_build.py（纯构建 / 文件解析，不加载 GPU）
#   bash harness/exp_deepep_cmake_build.sh > harness/exp_deepep_cmake_build.out.txt
# 对应源码：~/github/DeepEP@93eb6eb（v2.5.0）
#   CMakeLists.txt、csrc/{CMakeLists.txt,kernels/CMakeLists.txt,kernels/{comm,driver}/CMakeLists.txt}
#   setup.py:54-61、.gitignore、csrc/indexing/main.cu
#   ~/github/DeepGEMM@057ca59：CMakeLists.txt（同一套调试 CMake 对偶）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
export DEEPEP_ROOT=~/github/DeepEP
export DEEPGEMM_ROOT=~/github/DeepGEMM
python '~/proj/tech_record/code/mega-moe/harness/exp_deepep_cmake_build.py'
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
