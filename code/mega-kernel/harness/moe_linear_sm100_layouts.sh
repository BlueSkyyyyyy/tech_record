#!/usr/bin/env bash
# P152-1 复现：BF16 SM100 *MoE* linear device 身体 `moe_linear_sm100.cuh`
#   （docs/66）的 host 侧结构探测——把该内核用 CuTe 布局代数（make_tiled_mma /
#   partition_shape_{A,B,C} / UMMA::tile_to_mma_shape）算出的 SMEM 描述符布局
#   在 host 上重算，并与源码注释（moe_linear_sm100.cuh:186-198 的注释掉的
#   debug print）逐项对照；额外把 MoE-only 的 `sB_cp_layout`（cp.async gather
#   视图）与 `MoESharedStorage` 的 SMEM 预算算出来；并验证
#   「手工 smem_tma/cp.async stage 步长 == CuTe 布局 stage 步长」这条 ABI 一致性。
# 所有 CuTe layout 都是 CUTE_HOST_DEVICE，故整套代数在 host 上可实例化/求值，
# 无需 SM100 硬件（目标卡 H100/sm90 没有 SM100）；用 MPK 快照钉死的
# deps/cutlass（f3fde583）。
set -u
DEV=${DEV:-3}
CUTLASS=${CUTLASS:-/ssd/mirage_mpk_build/deps/cutlass}
MPK=${MPK:-/ssd/mirage_mpk_build}
SRC=~/proj/tech_record/code/mega-kernel/src/moe_linear_sm100_layouts.cu
BIN=/tmp/moe_linear_sm100_layouts
nvcc -std=c++17 -O2 -arch=sm_90a -I"$CUTLASS/include" "$SRC" -o "$BIN" 2>/dev/null \
  || nvcc -std=c++17 -O2 -arch=sm_90a -I"$CUTLASS/include" "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"

echo
echo "=== consumer/wiring audit (grep, MPK $MPK) ==="
printf 'task_header includes moe_linear_sm100.cuh             : %s\n' \
  "$(grep -rc 'moe_linear_sm100.cuh' "$MPK/include/mirage/persistent_kernel/tasks/blackwell/task_header.cuh")"
printf 'register_moe_linear_sm100_task call sites (graph.cc)  : %s\n' \
  "$(grep -rc 'register_moe_linear_sm100_task' "$MPK/src/kernel/graph.cc")"
printf 'device impl instantiations (task_register.cc)         : %s\n' \
  "$(grep -rc 'moe_linear_sm100_task_impl' "$MPK/src/kernel/task_register.cc")"
printf 'unit-test wrapper instantiations (sm100_moe)          : %s\n' \
  "$(grep -rc 'moe_linear_sm100_task_impl' "$MPK/tests/runtime_python/blackwell/sm100_moe/runtime_kernel_wrapper_sm100.cu")"
printf 'unit-test bf16 wrapper instantiations (sm100_fp8_moe) : %s\n' \
  "$(grep -rc 'moe_linear_sm100_task_impl' "$MPK/tests/runtime_python/blackwell/sm100_fp8_moe/runtime_kernel_wrapper_bf16.cu")"
printf 'front-end layer register_task (moe_w13/w2_linear_sm100): %s\n' \
  "$(grep -Ec 'moe_w13_linear_sm100|moe_w2_linear_sm100' "$MPK/python/mirage/mpk/persistent_kernel.py")"
printf 'builder files calling (moe_w13/w2_linear_layer)       : %s\n' \
  "$(grep -rlE 'moe_w13_linear_layer|moe_w2_linear_layer' --include='*.py' "$MPK/python/mirage/mpk/models/" 2>/dev/null | wc -l)"
printf 'TaskType 254/255 (MOE W13/W2 LINEAR SM100)             : %s\n' \
  "$(grep -Ec 'TASK_MOE_W13_LINEAR_SM100|TASK_MOE_W2_LINEAR_SM100' "$MPK/include/mirage/persistent_kernel/runtime_header.h")"
printf 'sm90 sibling registered (moe_w13/w2_linear_sm90)       : %s\n' \
  "$(grep -Ec 'moe_w13_linear_sm90|moe_w2_linear_sm90' "$MPK/src/kernel/graph.cc")"
printf 'relinquish for megakernel (commented)                 : %s\n' \
  "$(grep -rc 'relinquish for megakernel' "$MPK/include/mirage/persistent_kernel/tasks/blackwell/moe_linear_sm100.cuh")"
