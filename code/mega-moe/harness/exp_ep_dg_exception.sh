#!/usr/bin/env bash
# P116-1 实验 driver：在 kernel_lab(H100/sm90) 上验证两库共用的「错误 / 断言层」。
#   src/exception_layer.cu    （host 宏 + 设备 trap，直接 #include 两库真实头文件）
#   src/exception_bridge.cpp  （pybind11：EPException/DGException → Python RuntimeError）
#   harness/exp_ep_dg_exception.py（驱动 + 宏普查 + 负向编译）
#   bash harness/exp_ep_dg_exception.sh > harness/exp_ep_dg_exception.out.txt
# 对应源码：~/github/DeepEP@93eb6eb、~/github/DeepGEMM@057ca59
#   DeepEP   deep_ep/include/deep_ep/common/exception.cuh（EPException / EP_* / CUDA_*.*_CHECK / NCCL_CHECK）
#   DeepGEMM deep_gemm/include/deep_gemm/common/exception.cuh（device DG_* / DG_UNIFIED_ASSERT）
#            csrc/utils/exception.hpp（host DGException / DG_HOST_ASSERT / DG_CUBLASLT_CHECK）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DE=~/github/DeepEP
DG=~/github/DeepGEMM
SRC=~/proj/tech_record/code/mega-moe
TORCH=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")
NCCL_INC=/tmp/nccl229/include
NCCL_LIB=/tmp/nccl229/lib

INC="-I $DE/deep_ep/include -I $DE/csrc -I $DG/deep_gemm/include -I $DG/csrc \
     -I $TORCH/include -I $TORCH/include/torch/csrc/api/include -I $NCCL_INC"

echo "== 编译 src/exception_layer.cu（真实头文件）=="
nvcc -arch=sm_90 -std=c++17 -O2 $INC "$SRC/src/exception_layer.cu" -o /tmp/exception_layer \
     -L$NCCL_LIB -lnccl -lcublasLt -lcublas -lcuda -ldl
echo "compiled: /tmp/exception_layer"
echo

export DEEPEP_ROOT="$DE" DEEPGEMM_ROOT="$DG" EXC_BIN=/tmp/exception_layer
export LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH"
python "$SRC/harness/exp_ep_dg_exception.py"
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
