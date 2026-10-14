#!/usr/bin/env bash
# P184-1 实验 driver：Mega MoE dispatch 段的「round-robin min-peeling」token 认领协议。
#   * 忠实设备代码：src/mega_moe_dispatch_claim.cu（逐行复刻 sm100_fp8_fp4_mega_moe.cuh:439-534，
#     只用 warp intrinsics，与 SM100 无关，H100(sm90a) 可真机编译运行）。
#   * 独立 host/Python oracle：harness/exp_mega_moe_dispatch_claim.py（不做 lane/ballot 模拟，
#     只按聚合语义解析落点，另跑 sweep + 交错性统计）。
#   bash harness/exp_mega_moe_dispatch_claim.sh > harness/exp_mega_moe_dispatch_claim.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=${HOME}/github/DeepGEMM
SRC=${HOME}/proj/tech_record/code/mega-moe/src/mega_moe_dispatch_claim.cu

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
nvcc --version | tail -1
sha256sum $DG/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh | sed 's#.*/##'

echo
echo "== 编译（nvcc -arch=sm_90a，真实设备代码）=="
nvcc -std=c++20 -O2 -arch=sm_90a -x cu "$SRC" -o /tmp/mega_moe_dispatch_claim
echo "build OK"
echo
/tmp/mega_moe_dispatch_claim

echo
echo "== 独立 host/Python oracle（不做 lane/ballot 模拟）=="
cd ${HOME}/proj/tech_record/code/mega-moe/harness
python3 -u exp_mega_moe_dispatch_claim.py
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
