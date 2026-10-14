#!/usr/bin/env bash
# P151-1 复现：BF16 SM100 linear device 身体 `linear_sm100_mpk.cuh`（docs/65）的
#   host 侧结构探测——把该内核用 CuTe 布局代数（make_tiled_mma /
#   partition_shape_{A,B,C} / UMMA::tile_to_mma_shape）算出的 SMEM 描述符布局
#   在 host 上重算，并与源码注释（linear_sm100_mpk.cuh:190-231 的注释掉的
#   debug print）逐项对照；同时验证「手工 smem_tma stage 步长 == CuTe 布局
#   stage 步长」这条 ABI 一致性。
# 所有 CuTe layout 都是 CUTE_HOST_DEVICE，故整套代数在 host 上可实例化/求值，
# 无需 SM100 硬件（目标卡 H100/sm90 没有 SM100）；用 MPK 快照钉死的
# deps/cutlass（f3fde583）。
set -u
DEV=${DEV:-3}
CUTLASS=${CUTLASS:-/ssd/mirage_mpk_build/deps/cutlass}
MPK=${MPK:-/ssd/mirage_mpk_build}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/linear_sm100_bf16_layouts.cu
BIN=/tmp/linear_sm100_bf16_layouts
nvcc -std=c++17 -O2 -arch=sm_90a -I"$CUTLASS/include" "$SRC" -o "$BIN" 2>/dev/null \
  || nvcc -std=c++17 -O2 -arch=sm_90a -I"$CUTLASS/include" "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"

echo
echo "=== consumer/wiring audit (grep, MPK $MPK) ==="
printf 'task_header includes linear_sm100_mpk.cuh        : %s\n' \
  "$(grep -rc 'linear_sm100_mpk.cuh' "$MPK/include/mirage/persistent_kernel/tasks/blackwell/task_header.cuh")"
printf 'register_linear_sm100_task call sites (task_register.cc): %s\n' \
  "$(grep -rc 'register_linear_sm100_task' "$MPK/src/kernel/task_register.cc")"
printf 'register_splitk_linear_sm100_task call sites            : %s\n' \
  "$(grep -rc 'register_splitk_linear_sm100_task' "$MPK/src/kernel/task_register.cc")"
printf 'device impl instantiations (linear_sm100_mpk_task_impl) : %s\n' \
  "$(grep -rc 'linear_sm100_mpk_task_impl' "$MPK/src/kernel/task_register.cc")"
printf 'unit-test wrapper instantiations                        : %s\n' \
  "$(grep -rc 'linear_sm100_mpk_task_impl' "$MPK/tests/runtime_python/blackwell/sm100_linear/runtime_kernel_wrapper_sm100.cu")"
printf 'graph.cc op names (linear_sm100/splitk/residual)        : %s\n' \
  "$(grep -Ec 'linear_sm100|splitk_linear_sm100|linear_with_residual_sm100' "$MPK/src/kernel/graph.cc")"
printf 'TaskType 251/252/253 (SPLITK/RESIDUAL/LINEAR SM100)      : %s\n' \
  "$(grep -Ec 'TASK_SPLITK_LINEAR_SM100|TASK_LINEAR_WITH_RESIDUAL_SM100|TASK_LINEAR_SM100' "$MPK/include/mirage/persistent_kernel/runtime_header.h")"
printf 'relinquish for megakernel (commented)                   : %s\n' \
  "$(grep -rc 'relinquish for megakernel' "$MPK/include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh")"
