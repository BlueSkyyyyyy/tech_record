#!/usr/bin/env bash
# P7-108 (companion to probe_b2_stack_resid.sh): does the `(64,256)` bank-conflict residual depend on
# the grid *size* or on how the grid splits into `(B,T,H)`?  Sweeps three shapes with the SAME grid
# (`B*ceil(T/16)*H = 2048`) at fixed `(K,V)=(64,256)`.  Reuses the `.so` built by probe_b2_stack_resid.sh.
# Usage (inside kernel_lab):  bash .../probe_b2_stack_resid_knee.sh > ..._knee.out.txt 2>&1
set -uo pipefail
DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
K=64; V=256
DRV="$DIR/harness/ncu_b2_stack_resid_driver.py"
M="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
for cfg in "8 512 8" "4 1024 8" "2 2048 8"; do
  set -- $cfg; B=$1; T=$2; H=$3
  echo "########## KNEE grid=2048 B=$B T=$T H=$H ##########"
  for t in bmt bmt_da bmt_intra bmt_da_intra; do
    B2STK_SO=/tmp/b2res_${t}_${K}_${V}.so \
    ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
      python "$DRV" --B "$B" --T "$T" --H "$H" --K "$K" --V "$V" --tag "$t" \
        --so /tmp/b2res_${t}_${K}_${V}.so --reps 6 2>&1 \
      | grep bank_conflicts | sed "s/^/    tag=$t /"
  done
done
echo "== DONE =="
