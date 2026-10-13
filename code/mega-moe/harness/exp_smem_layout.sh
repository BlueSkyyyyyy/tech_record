#!/usr/bin/env bash
# P82-1 实验 driver：DeepGEMM 的共享内存布局结构层。
#   * deep_gemm/include/deep_gemm/layout/gemm.cuh        （SM100BF16GemmSharedStorage / SM100FP8FP4GemmSharedStorage）
#   * deep_gemm/include/deep_gemm/layout/mqa_logits.cuh  （MQALogitsSharedStorage）
#   源码 src/smem_layout.cu（nvcc 直接 include 真实 layout 头文件，纯 host 程序）。
#   这些结构体不含 UMMA/TMA 设备指令，故可在 H100(sm90) 上编译、逐字段对拍。
#   ./harness/exp_smem_layout.sh > harness/exp_smem_layout.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=~/github/DeepGEMM
SRC=~/proj/tech_record/code/mega-moe/src/smem_layout.cu

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
nvcc --version | tail -1
sha256sum $DG/deep_gemm/include/deep_gemm/layout/gemm.cuh \
          $DG/deep_gemm/include/deep_gemm/layout/mqa_logits.cuh | sed 's#.*/##'
wc -l $DG/deep_gemm/include/deep_gemm/layout/gemm.cuh \
      $DG/deep_gemm/include/deep_gemm/layout/mqa_logits.cuh

echo
echo "== 编译（nvcc 直接 include 真实 layout 头文件，纯 host 程序）=="
nvcc -std=c++20 -O1 -w -arch=sm_90a -x cu \
  -I$DG/deep_gemm/include \
  "$SRC" -o /tmp/smem_layout
echo "build OK"

echo
/tmp/smem_layout
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
