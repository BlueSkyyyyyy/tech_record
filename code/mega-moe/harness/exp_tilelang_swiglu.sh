#!/usr/bin/env bash
# P153-1 实验 driver：精读 + 实测 TileLang 基线 SwiGLU 算子
#   third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py
# （`tests/test_mega_moe.py` 的 run_baseline 用它，是 SM100 mega kernel L1 epilogue 的逐位对照物）。
#
#   * host 公式 + 真实 tilelang kernel 真机跑 → harness/exp_tilelang_swiglu.py
#   * mega L1 epilogue 设备端算术复刻 → src/swiglu_mega_epilogue.cu（nvcc include 真实 DeepGEMM 头）
#   bash harness/exp_tilelang_swiglu.sh > harness/exp_tilelang_swiglu.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DG=${HOME}/github/DeepGEMM
INC="$DG/deep_gemm/include"

CMD=$(cat <<EOS
set -e
echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
python -c "import tilelang, torch; print('tilelang', tilelang.__version__, '| torch', torch.__version__)"
echo "== 真实源码 sha256 =="
sha256sum $DG/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py $DG/third-party/tilelang_ops/utils.py
sha256sum $INC/deep_gemm/common/math.cuh

echo
echo "== 编译 src/swiglu_mega_epilogue.cu（nvcc 直接 include 真实设备头）=="
nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr \
  -I"$INC" -I"$INC/cutlass" -I"$INC/cute" \
  '$ROOT/src/swiglu_mega_epilogue.cu' -o /tmp/swiglu_mega_epilogue
echo "build OK"

echo
python3 '$ROOT/harness/exp_tilelang_swiglu.py'
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
