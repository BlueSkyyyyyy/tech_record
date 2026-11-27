#!/bin/bash
# One-shot reproduction of every number in results_moe_ep.md.
#
#   bash run_all.sh          # ~45 min on 8x H100, runs everything sequentially
#
# Stages
#   1. uniEP    : unfused ops + MegaKernel1/2 + fused/sequential e2e
#   2. DeepEP   : dispatch/combine (plain, expanded, async) + comm-only e2e
#   3. TE       : grouped GEMM FC1/FC2 (+ fwd/bwd, + FC1+swiglu+FC2 layer fwd & fwd+bwd)
#   4. uniEP bwd: dgrad/wgrad/swiglu bwd composed from group_gemm + torch-reference check
#   5. checks   : uniEP vs pure-torch MoE reference; SM-budget sweep; topk=8 probe
#
# Host quirks baked in (see results_moe_ep.md §环境):
#   NCCL_NVLS_ENABLE=0                  no NVSwitch/Fabric Manager on this box
#   FLASH_COMM_BUFFER_BACKEND=cuda_ipc  VMM/fabric shareable handles not permitted
#   LD_PRELOAD=/opt/nccl232/...         DeepEP V2.5 needs NCCL >= 2.32.3
#   EP_DISABLE_GIN=1                    intranode-only, no RDMA Gin domain
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR"

NPROC=${NPROC:-8}
TOKENS=${TOKENS:-"1024 128"}     # tokens *per rank*
HIDDEN=${HIDDEN:-"1024 3072"}    # H = I
ITERS=${ITERS:-50}
WARMUP=${WARMUP:-10}
NCCL232=/opt/nccl232/nvidia/nccl/lib/libnccl.so.2
FLASH="NCCL_NVLS_ENABLE=0 FLASH_COMM_BUFFER_BACKEND=cuda_ipc NCCL_LSA_TEAM_SIZE=$NPROC"

merge() {
  python3 - "$1" "$2" <<'PY'
import glob, sys, csv
files, out = sorted(glob.glob(sys.argv[1])), sys.argv[2]
rows = []
for f in files:
    with open(f) as fh:
        rows.extend(list(csv.DictReader(fh)))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"merged {len(files)} files, {len(rows)} rows -> {out}")
PY
}

# each (tokens, hidden) pair runs in a FRESH process on purpose: reusing one process
# across shapes poisons the numbers (allocator state + JIT caches made DeepEP's third
# config 10x slower in an early run).
run_uni() {
  env $FLASH torchrun --nproc_per_node=$NPROC --master_port=29520 bench_uniep_ep.py \
    --tokens $1 --hidden $2 --iters $ITERS --warmup $WARMUP --csv "moe_ep_uniep_M$1_H$2.csv" \
    2>&1 | tee "moe_ep_uniep_M$1_H$2.log" | grep -E "^  |^==="
}
run_deepep() {
  env LD_PRELOAD=$NCCL232 EP_DISABLE_GIN=1 NCCL_NVLS_ENABLE=0 \
    torchrun --nproc_per_node=$NPROC --master_port=29521 bench_deepep_ep.py \
    --tokens $1 --hidden $2 --iters $ITERS --warmup $WARMUP --csv "moe_ep_deepep_M$1_H$2.csv" \
    2>&1 | tee "moe_ep_deepep_M$1_H$2.log" | grep -E "^  |^===|buffer:"
}
run_unibwd() {
  env $FLASH torchrun --nproc_per_node=$NPROC --master_port=29527 bench_uniep_bwd.py \
    --tokens $1 --hidden $2 --iters $ITERS --warmup $WARMUP --csv "moe_ep_uniep_bwd_M$1_H$2.csv" \
    2>&1 | tee "moe_ep_uniep_bwd_M$1_H$2.log" | grep -E "^  |^==="
}
run_te() {
  env NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=$NPROC --master_port=29522 bench_te_group_gemm.py \
    --tokens $1 --hidden $2 --iters $ITERS --warmup $WARMUP --csv "moe_ep_tegmm_M$1_H$2.csv" \
    2>&1 | tee "moe_ep_tegmm_M$1_H$2.log" | grep -E "^  |^==="
}
echo "############ stage 1/5: uniEP ############"
for T in $TOKENS; do for H in $HIDDEN; do
  echo "=== uniEP    T=$T/rank H=I=$H ==="; run_uni $T $H; sleep 3
done; done
merge "moe_ep_uniep_M*_H*.csv" moe_ep_uniep.csv

echo "############ stage 2/5: DeepEP ############"
for T in $TOKENS; do for H in $HIDDEN; do
  echo "=== DeepEP   T=$T/rank H=$H ==="; run_deepep $T $H; sleep 3
done; done
merge "moe_ep_deepep_M*_H*.csv" moe_ep_deepep.csv

echo "############ stage 3/5: TE grouped GEMM ############"
for T in $TOKENS; do for H in $HIDDEN; do
  echo "=== TE       T=$T/rank H=I=$H ==="; run_te $T $H; sleep 3
done; done
merge "moe_ep_tegmm_M*_H*.csv" moe_ep_tegmm.csv

echo "############ stage 4/5: uniEP backward ############"
for T in $TOKENS; do for H in $HIDDEN; do
  echo "=== uniEP bwd T=$T/rank H=I=$H ==="; run_unibwd $T $H; sleep 3
done; done
merge "moe_ep_uniep_bwd_M*_H*.csv" moe_ep_uniep_bwd.csv

echo "############ stage 5/5: correctness + probes ############"
for cfg in "1024 1024" "1024 3072"; do set -- $cfg
  env $FLASH torchrun --nproc_per_node=$NPROC --master_port=29524 check_uniep_moe.py \
    --tokens $1 --hidden $2 2>&1 | tee -a check_uniep_moe.log | grep -E "rank [01]\]"
done
for cfg in "1024 1024" "1024 3072"; do set -- $cfg
  env $FLASH torchrun --nproc_per_node=$NPROC --master_port=29525 sweep_sm_uniep.py \
    --tokens $1 --hidden $2 --iters 30 --warmup 10 \
    --csv "moe_ep_uniep_sm_M${1}_H${2}.csv" 2>&1 | tee "moe_ep_uniep_sm_M${1}_H${2}.log" | grep -E "^  "
done
merge "moe_ep_uniep_sm_M*_H*.csv" moe_ep_uniep_sm_sweep.csv

env $FLASH torchrun --nproc_per_node=$NPROC --master_port=29526 bench_uniep_ep.py \
  --tokens 1024 --hidden 1024 --topk 8 --iters 30 --warmup 10 \
  --csv "moe_ep_uniep_topk8_M1024_H1024.csv" 2>&1 | tee moe_ep_uniep_topk8.log | grep -E "^  "

echo "############ all done ############"
