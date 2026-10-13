#!/usr/bin/env bash
# P81-1 实验 driver：DeepGEMM 官方 SF 契约（docs/scaling-factor-format.md）与 host 契约代码的一致性核验。
#   * csrc/utils/layout.hpp                    （check_sf_layout / get_tma_aligned_size）
#   * csrc/jit_kernels/impls/smxx_layout.hpp   （check_k_grouped_packed_ue8m0_tensor）
#   源码 src/sf_contract_conformance.cu（nvcc 直接 include 真实头文件 + torch）。
#   在 kernel_lab(H100/sm90) 上编译并运行真实契约函数（这两个函数不含 arch 门控）。
#   ./harness/exp_sf_contract_conformance.sh > harness/exp_sf_contract_conformance.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=~/github/DeepGEMM
TORCH=/usr/local/lib/python3.12/dist-packages/torch
PYSITE=/usr/local/lib/python3.12/dist-packages
SRC=~/proj/tech_record/code/mega-moe/src/sf_contract_conformance.cu

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
nvcc --version | tail -1
sha256sum $DG/csrc/utils/layout.hpp $DG/csrc/jit_kernels/impls/smxx_layout.hpp | sed 's#.*/##'

echo
echo "== 编译（nvcc 直接 include 真实头文件 + torch）=="
nvcc -std=c++20 -O1 -g -w -arch=sm_90a -x cu \
  -I$DG/csrc -I$DG -I$DG/deep_gemm/include -I$DG/third-party/deep_jit/include \
  -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
  -I$PYSITE/pybind11/include -I/usr/include/python3.12 -I/usr/local/cuda/include \
  "$SRC" -o /tmp/sf_contract_conformance \
  -L$TORCH/lib -L/usr/local/cuda/lib64 -Xlinker -rpath -Xlinker $TORCH/lib \
  -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcudart -lcuda -lcublas -lcublasLt -lpython3.12 -ldl -lpthread
echo "build OK"

echo
/tmp/sf_contract_conformance
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
