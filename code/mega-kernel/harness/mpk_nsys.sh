#!/usr/bin/env bash
# P25-1 continuation: attempt an nsys host-side CUDA API trace of the MPK
# launch path, comparing a HANG prewarm (none) with a PASS prewarm (cast_1elem).
#
# This profiling attempt did NOT yield a usable host timeline (see docs/09
# §10.7): the hang case exits via os._exit(124), which skips CUPTI's buffer
# flush, so only the first few API rows survive; adding cudaProfilerStop() /
# --capture-range made the nsys supervisor itself stall on the spinning kernel.
# The script is kept (with a hard per-case `timeout`) so the attempt is
# reproducible and its limits documented.
#
# Run inside kernel_lab:
#   bash harness/mpk_nsys.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-/ssd/mirage_mpk_build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
PROF=/tmp/mpkprof
mkdir -p "$PROF"

for pw in none cast_1elem; do
  echo "===== nsys profile prewarm=$pw ====="
  timeout 120 nsys profile -t cuda --sample=none --cpuctxsw=none \
    --force-overwrite=true -o "$PROF/$pw" \
    python3 -u "$HERE/mpk_race_probe.py" --device 0 --hang-timeout 10 \
    --prewarm "$pw" >"$PROF/$pw.log" 2>&1
  echo "  nsys wrapper exit=$?"
  tail -3 "$PROF/$pw.log" | sed 's/^/    /'
  if [ -f "$PROF/$pw.nsys-rep" ]; then
    nsys export --type sqlite --force-overwrite=true \
      -o "$PROF/$pw.sqlite" "$PROF/$pw.nsys-rep" >/dev/null 2>&1
    echo "  --- runtime API rows captured ---"
    timeout 60 python3 "$HERE/mpk_nsys_parse.py" "$PROF/$pw.sqlite" 2>&1 \
      | sed 's/^/    /' | head -12
  else
    echo "  (no .nsys-rep produced)"
  fi
done
