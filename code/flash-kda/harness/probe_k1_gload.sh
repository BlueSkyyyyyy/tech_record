#!/usr/bin/env bash
# P7-4: attribute the K1 "global load only uses 4/32 bytes per sector" warning
# (docs/07 SS5.2, left as "待验证 (a)") to a concrete instruction.
#
# Method:
#   1. Static: disassemble the installed FlashKDA .so and count every *scalar*
#      global load (LDG.E) vs every TMA load (UTMALDG) in both K1 variants
#      (`_flash_kda_fwd_prepare`, IsVarlen=false and IsVarlen=true).
#   2. Dynamic: run ncu with `MemoryWorkloadAnalysis_Tables` (the section that
#      emits the 4/32 warning) and `SourceCounters` (per-SASS-line sector
#      counts) on the non-varlen K1 to show which instruction the warning is.
#   3. Budget: measure K1 DRAM read at small + design point and compute the
#      share contributed by that load (sectors * 32B).
#
# Usage (inside kernel_lab):
#   bash ${HOME}/proj/tech_record/code/flash-kda/harness/probe_k1_gload.sh \
#       > ${HOME}/proj/tech_record/code/flash-kda/harness/probe_k1_gload.out.txt 2>&1

set -uo pipefail

HARNESS=${HOME}/proj/tech_record/code/flash-kda/harness
DRIVER=$HARNESS/ncu_driver.py
FK=${HOME}/github/FlashKDA
SO=$FK/flash_kda_C.cpython-312-x86_64-linux-gnu.so
SASS=/tmp/k1_gload_sass.txt
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

# ---------------------------------------------------------------------------
echo ""
echo "################################################################################"
echo "# 1. Static SASS: global loads in K1 _flash_kda_fwd_prepare (both variants)"
echo "################################################################################"
( cd "$FK" && cuobjdump -sass "$SO" > "$SASS" 2>&1 )
awk '
/Function : / {
    if ($0 ~ /_flash_kda_fwd_prepare/) {
        name = ($0 ~ /Lb1EEvT_/) ? "K1 varlen  (IsVarlen=true)" : "K1 non-varlen (IsVarlen=false)"
        print "  " name
    } else { name = "" }
    next
}
name != "" {
    if ($0 ~ /[^A-Z]LDG\.E /) {
        le[name]++
        inst = $0; sub(/^ +/, "", inst)
        p = index(inst, "/*"); inst = substr(inst, p)
        s = index(inst, ";"); if (s) inst = substr(inst, 1, s)
        lex[name] = lex[name] "        " inst "\n"
    }
    if ($0 ~ /UTMALDG/) ut[name]++
}
END {
    for (k in le)
        printf "    scalar LDG.E = %d   TMA UTMALDG = %d\n%s", le[k], ut[k], lex[k]
}' "$SASS"

# ---------------------------------------------------------------------------
echo ""
echo "################################################################################"
echo "# 2. ncu MemoryWorkloadAnalysis_Tables: the 4/32 warning (K1 B=1 T=1024 H=4)"
echo "################################################################################"
( cd "$FK" && ncu -k "regex:flash_kda_fwd_prepare" -c 1 --launch-skip 3 \
    --section MemoryWorkloadAnalysis_Tables \
    python "$DRIVER" --B 1 --T 1024 --H 4 2>&1 ) \
  | grep -A3 -iE "global loads from L1TEX|4\.0 of the 32|stride between"

# ---------------------------------------------------------------------------
echo ""
echo "################################################################################"
echo "# 3. ncu SourceCounters: per-SASS-line global load (K1 B=1 T=1024 H=4)"
echo "#    key columns: L1 Tag Requests Global | L2 Theoretical Sectors Global | ... Ideal"
echo "################################################################################"
( cd "$FK" && ncu -k "regex:flash_kda_fwd_prepare" -c 1 --launch-skip 3 \
    --section SourceCounters --page source --print-source sass --csv \
    python "$DRIVER" --B 1 --T 1024 --H 4 2>&1 ) > /tmp/k1_sc.csv
python - <<'PY'
import csv
rows = list(csv.reader(open('/tmp/k1_sc.csv')))
hdr = None
for r in rows:
    if r and r[0] == 'Address':
        hdr = r; break
if hdr is None:
    print("  (no source counters captured)"); raise SystemExit
idx = {n: i for i, n in enumerate(hdr)}
want = ["Source", "Instructions Executed", "Address Space", "Access Operation",
        "Access Size", "L1 Tag Requests Global",
        "L2 Theoretical Sectors Global Excessive", "L2 Theoretical Sectors Global",
        "L2 Theoretical Sectors Global Ideal"]
print("  " + " | ".join(want))
for r in rows:
    if len(r) != len(hdr):
        continue
    if r[idx["Address Space"]] != "Global":
        continue
    print("  " + " | ".join(r[idx[w]] for w in want))
    print("  -> sectors {} == ideal {} (excessive {}) => no fixable uncoalescing; "
          "the 4/32 is a uniform (broadcast) scalar read".format(
              r[idx["L2 Theoretical Sectors Global"]],
              r[idx["L2 Theoretical Sectors Global Ideal"]],
              r[idx["L2 Theoretical Sectors Global Excessive"]]))
PY

# ---------------------------------------------------------------------------
echo ""
echo "################################################################################"
echo "# 4. Budget: that load's DRAM share vs K1 DRAM read"
echo "    SASS runs once per warp; warps/CTA=8. share = warps*32B / K1 dram_read"
echo "################################################################################"
run_k1() {  # B T H
  ( cd "$FK" && ncu -k "regex:flash_kda_fwd_prepare" -c 1 --launch-skip 3 \
      --metrics dram__bytes_read.sum,smsp__inst_executed_op_global_ld.sum \
      python "$DRIVER" --B "$1" --T "$2" --H "$3" 2>&1 ) \
    | grep -E "dram__bytes_read|smsp__inst_executed_op_global_ld" \
    | awk -v B="$1" -v T="$2" -v H="$3" '
        /dram__bytes_read/ {rd=$NF}
        /inst_executed_op_global_ld/ {ldg=$NF}
        END { tiles=int((T+15)/16); warps=tiles*H*8; share=warps*32/1048576;
              printf "  B=%s T=%s H=%s: inst_global_ld=%d, dram_read=%.2f MB, load bytes=%.2f MB, share=%.2f%%\n",
                     B, T, H, ldg, rd, share, (rd>0?100*share/rd:0) }'
}
echo "  (smsp__inst_executed_op_global_ld == warps confirms 1 LDG.E per warp)"
run_k1 1 1024 4
run_k1 1 8192 64

echo ""
echo "DONE"
