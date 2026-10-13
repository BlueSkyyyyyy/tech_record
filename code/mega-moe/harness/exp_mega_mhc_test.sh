#!/usr/bin/env bash
# P57-1 实验 driver：Mega mHC 的端到端验收测试与正确性 oracle（tests/test_mega_mhc.py）。
#
# 为什么 H100(sm90) 可跑：
#   * `mhc_reference` / `extra_sf_rows` / `check_correctness` 全是纯 torch / host 逻辑，架构无关；
#   * SF 页布局行索引 `get_shared_sf_row_idx`（epilogue/sm100_mega_mhc.cuh:116-122）是纯整数，
#     直接 include 真实头文件在 H100 上跑真实设备代码；
#   * host 契约 csrc/apis/mega_mhc.hpp:116-221 在 H100 上真实执行（非法组合逐条拒绝，
#     合法调用才撞到 :223 的 arch 闸门）。
# 主体 kernel `sm100_mega_mhc` 是 SM100 专属（mega_mhc.hpp:223 arch==10），不涉及。
#   bash harness/exp_mega_mhc_test.sh > harness/exp_mega_mhc_test.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INC="~/github/DeepGEMM/deep_gemm/include"

CMD=$(cat <<EOS
set -e
echo "== 真实头文件 sha256 =="
sha256sum $INC/deep_gemm/epilogue/sm100_mega_mhc.cuh $INC/deep_gemm/layout/mega_mhc.cuh
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
echo
echo '== Part 1: 设备代码 —— kernel get_shared_sf_row_idx vs 测试 extra_sf_rows =='
nvcc -arch=sm_90a -O2 -std=c++17 --expt-relaxed-constexpr \
  -I"$INC" -I"$INC/cutlass" -I"$INC/cute" \
  '$ROOT/src/mhc_sf_row_idx.cu' -o /tmp/mhc_sf_row_idx
/tmp/mhc_sf_row_idx
echo
echo '== Part 2: Python —— 验收 oracle + host 契约 =='
cd '$ROOT/harness'
python3 '$ROOT/harness/exp_mega_mhc_test.py'
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
