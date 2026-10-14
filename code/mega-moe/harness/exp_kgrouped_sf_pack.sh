#!/usr/bin/env bash
# P16-1 实验 driver：在 kernel_lab(H100) 直接编译并运行 DeepGEMM 真实设备代码
#   deep_gemm::pack_fp32_into_ue8m0<kNumGroups>（K-grouped SF 打包）。
#   ./harness/exp_kgrouped_sf_pack.sh
# 对应源码 src/kgrouped_sf_pack.cu，输出见 harness/exp_kgrouped_sf_pack.out.txt。
#
# 该头文件两个 kernel 声明了同名不同类型的 extern __shared__ 数组、不能同 TU 编译，
# 故在容器内先按行号原样抽取真实 kernel（smxx_layout.cuh:193-293）到 /tmp 生成文件再编译
# （见 src 顶部注释）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/kgrouped_sf_pack.cu"
INC="${HOME}/github/DeepGEMM/deep_gemm/include"
DEEPG="$INC/deep_gemm/impls/smxx_layout.cuh"

CMD=$(cat <<'EOS'
set -e
INC=${HOME}/github/DeepGEMM/deep_gemm/include
DEEPG=$INC/deep_gemm/impls/smxx_layout.cuh
GEN=/tmp/kgrouped_pack_kernel.inc
{
  echo '#pragma once'
  echo '#include <deep_gemm/common/math.cuh>'
  echo '#include <deep_gemm/common/utils.cuh>'
  echo '#include <deep_gemm/ptx/ld_st.cuh>'
  echo '#include <deep_gemm/ptx/utils.cuh>'
  echo 'namespace deep_gemm {'
  sed -n '193,293p' "$DEEPG"
  echo '}'
} > "$GEN"
echo "extracted kernel sha256: $(sha256sum "$GEN" | cut -d' ' -f1) ($DEEPG:193-293)"
EOS
)

CMD="$CMD
nvcc -arch=sm_90 -O3 -std=c++17 -I'$INC' -I/tmp '$SRC' -o /tmp/kgrouped_sf_pack && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && /tmp/kgrouped_sf_pack
echo
echo '== F) host wrapper 在 sm90 的拒绝（真实 python API，smxx_layout.hpp:225 arch==10 断言）=='
python3 - <<'PY'
import torch, deep_gemm
sf = torch.zeros(8, 512, dtype=torch.float32, device='cuda')
grouped = torch.tensor([256, 384, 128], dtype=torch.int32, device='cuda')
ks = [256, 384, 128]
try:
    deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, grouped, ks, 32, 128, False)
    print('  !! 意外通过')
except Exception as e:
    msg = str(e).strip().splitlines()
    print('  sm90 上 get_k_grouped_... 被拒：', type(e).__name__)
    print('  断言信息：', msg[-1] if msg else str(e)[:160])
PY"

"$ROOT/scripts/lab.sh" exec "$CMD"
