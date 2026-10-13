#!/usr/bin/env bash
# P86-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证两个「mega 家族」JIT 启动包装层
#   csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp  （P3/P79 拆过 kernel，本篇补 FP8 wrapper）
#   csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp    （P5-1/P47 拆过 kernel，本篇补 gate wrapper）
#   ./harness/exp_mega_moe_gate_jit_wrapper.sh
# 对应源码 src/mega_moe_gate_jit_wrapper.cu（nvcc 直接 include 真实头文件 + torch）。
#   Part 1 FP8/FP4 vs BF16 Mega MoE 的 config 推导不变量
#   Part 2 FP8 wrapper 的 18 个 TMA 描述符组合 + 共享专家回退 + 独立编码互证 + JIT 模板参数个数
#   Part 3 Mega Gate 的 config / 常量 / RoutingArgs ABI / scratch 公式
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=~/github/DeepGEMM
TORCH=/usr/local/lib/python3.12/dist-packages/torch
PYSITE=/usr/local/lib/python3.12/dist-packages
SRC=~/proj/tech_record/code/mega-moe/src/mega_moe_gate_jit_wrapper.cu

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
sha256sum $DG/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp \
          $DG/csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp \
          $DG/csrc/jit_kernels/heuristics/mega_moe.hpp \
          $DG/csrc/jit_kernels/heuristics/mega_gate.hpp \
          $DG/deep_gemm/include/deep_gemm/layout/mega_gate.cuh | sed 's#.*/##'
nvcc --version | tail -1

echo
echo "== 编译（nvcc 直接 include 真实头文件 + torch）=="
nvcc -std=c++20 -O1 -w -arch=sm_90a -x cu \
  -I$DG/csrc -I$DG -I$DG/third-party/deep_jit/include -I$DG/third-party/cutlass/include \
  -I/usr/local/cuda/targets/x86_64-linux/include/cccl -I$DG/deep_gemm/include \
  -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
  -I$PYSITE/pybind11/include -I/usr/include/python3.12 -I/usr/local/cuda/include \
  "$SRC" -o /tmp/mega_moe_gate_jit_wrapper \
  -L$TORCH/lib -L/usr/local/cuda/lib64 -Xlinker -rpath -Xlinker $TORCH/lib \
  -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcudart -lcublas -lcublasLt -lpython3.12 -ldl -lpthread
echo "build OK"

echo
echo "== 主实验 =="
/tmp/mega_moe_gate_jit_wrapper
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
