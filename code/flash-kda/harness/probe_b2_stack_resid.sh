#!/usr/bin/env bash
# P7-108: isolate and characterize the **`(K,V)=(64,256)` bank-conflict secondary interaction** of the
# `bf16mma_tma` multi-knob stack -- the item `docs/07` §110.3/§110.6 (P7-107) left verbatim as
# "机制待验证" (~-0.6 M, ~2%, reproducible) and the ROADMAP "当前进度 / 下一步" restated.
#
# Base = "bmt" = `-DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_BF16MMA -DB2M_BVK=64 -DB2M_TMA=1`, K=64 V=256:
#   bmt            bf16mma_tma
#   bmt_da         + -DB2M_DAMMA      (DA -> mma.sync.tf32)
#   bmt_intra      + -DB2M_INTRA      (B3 diagonal folded)
#   bmt_da_intra   + both
#
# Reports (all captured to probe_b2_stack_resid.out.txt):
#   [A] static SASS shared-load opcode histogram per build (`cuobjdump -sass`) -- tests whether the
#       combined build's instruction multiset is the union of the two knobs' changes;
#   [B] `ncu` bank-conflict / warps / occupancy over a **grid sweep** {128,512,1024,2048,4096,8192}
#       at fixed `(K,V)` -- tests whether the residual is work/grid-dependent;
#   [C] `ncu` aggregate at grid=8192: dynamic shared-load instruction count (`smsp__inst_executed_op_shared_ld`),
#       total shared-load wavefronts and bank conflicts -- tests additivity of each layer;
#   [D] `ncu` conflict-**type** split at grid=8192 (`type_arbitration` vs `mem_shared_op_ld`).
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   CUDA_VISIBLE_DEVICES=1 bash .../probe_b2_stack_resid.sh > .../probe_b2_stack_resid.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"
DRV="$DIR/harness/ncu_b2_stack_resid_driver.py"
K=64
V=256

echo "== prebuild (outside ncu) =="
build() {  # tag extra...
  tag=$1; shift
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
       -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_BVK=64 -DB2M_TMA=1 \
       -DB2M_KD=$K -DB2M_VD=$V "$@" "$CU" -o /tmp/b2res_${tag}_${K}_${V}.so \
    && echo "  built ${tag}" || { echo "  FAIL ${tag}"; exit 1; }
}
build bmt
build bmt_da        -DB2M_DAMMA=1
build bmt_intra     -DB2M_INTRA=1
build bmt_da_intra  -DB2M_DAMMA=1 -DB2M_INTRA=1
TAGS="bmt bmt_da bmt_intra bmt_da_intra"

echo ""
echo "########## [A] SASS shared-load opcode histogram (K=${K} V=${V}) ##########"
for t in $TAGS; do
  echo "########## SASS tag=$t ##########"
  cuobjdump -sass /tmp/b2res_${t}_${K}_${V}.so 2>/dev/null | grep -oE '\bLDS[A-Z0-9.]*' | sort | uniq -c
  echo "    total_LDS $(cuobjdump -sass /tmp/b2res_${t}_${K}_${V}.so 2>/dev/null | grep -cE '\bLDS')"
done

GRIDS="128:1:512:4 512:2:512:8 1024:4:512:8 2048:8:512:8 4096:8:1024:8 8192:8:2048:8"
MBASE="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_shared_mem"

echo ""
echo "########## [B] grid sweep: bank conflicts / warps / occ_smem ##########"
for cfg in $GRIDS; do
  g=${cfg%%:*}; rest=${cfg#*:}; B=${rest%%:*}; rest=${rest#*:}; T=${rest%%:*}; H=${rest##*:}
  for t in $TAGS; do
    echo "########## GRID grid=$g B=$B T=$T H=$H tag=$t ##########"
    B2STK_SO=/tmp/b2res_${t}_${K}_${V}.so \
    ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$MBASE" \
      python "$DRV" --B "$B" --T "$T" --H "$H" --K "$K" --V "$V" --tag "$t" \
        --so /tmp/b2res_${t}_${K}_${V}.so --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d' \
      | grep -E "bank_conflicts|warps_active|occupancy_limit_shared"
  done
done

MAGG="smsp__inst_executed_op_shared_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"

echo ""
echo "########## [C] aggregate at grid=8192: dynamic shared-ld instr / wavefronts / conflicts ##########"
for t in $TAGS; do
  echo "########## AGG tag=$t ##########"
  B2STK_SO=/tmp/b2res_${t}_${K}_${V}.so \
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$MAGG" \
    python "$DRV" --B 8 --T 2048 --H 8 --K "$K" --V "$V" --tag "$t" \
      --so /tmp/b2res_${t}_${K}_${V}.so --reps 6 2>&1 \
    | grep -E "inst_executed_op_shared_ld|wavefronts_mem_shared|bank_conflicts"
done

MSUB="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_type_arbitration.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"

echo ""
echo "########## [D] conflict-type split at grid=8192 ##########"
for t in $TAGS; do
  echo "########## SUB tag=$t ##########"
  B2STK_SO=/tmp/b2res_${t}_${K}_${V}.so \
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$MSUB" \
    python "$DRV" --B 8 --T 2048 --H 8 --K "$K" --V "$V" --tag "$t" \
      --so /tmp/b2res_${t}_${K}_${V}.so --reps 6 2>&1 \
    | grep -E "bank_conflicts"
done

echo ""
echo "== DONE =="
