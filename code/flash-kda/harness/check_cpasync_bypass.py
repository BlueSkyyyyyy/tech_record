"""P7-110: parse the two `ncu` outputs for `probe_cpasync_bypass.cu` into the
summary tables and check the correctness half.

  Part A  bit-identical reads: at a fixed pattern every read instruction
          (`cp.async.cg/ca` at 16/8/4 B, `ld.global[.cg].v4`, scalar) must
          accumulate the *same* value -- the probe only changes the load path,
          not the addresses/values.
  Part B  the isolation table (pat x mode): l1tex sectors/request, L2 srcunit
          read sectors/request, srcunit total, DRAM read, duration.
  Part C  the reuse-factor sweep (srcunit vs `iters`) and the alignment sweep
          (l1tex/request vs rows-pattern pitch).

Raw evidence: `harness/probe_cpasync_bypass_ncu.out.txt` (Part B) and
`harness/probe_cpasync_bypass_sweep.out.txt` (Part C).

Run:  python .../harness/check_cpasync_bypass.py   (pure CPU, no GPU needed)
"""
import os
import re

HARNESS = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HARNESS, "probe_cpasync_bypass_ncu.out.txt")
SWEEP = os.path.join(HARNESS, "probe_cpasync_bypass_sweep.out.txt")

MODES = {0: "cg16", 1: "ca16", 2: "ca8", 3: "ca4", 4: "ld16", 5: "ldcg16", 6: "ld4"}
LDREQ = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
LDSEC = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
SRCREAD = "lts__t_sectors_srcunit_tex_op_read.sum"
L2HIT = "lts__t_sectors_op_read_lookup_hit.sum"
DRAMR = "dram__bytes_read.sum"
DUR = "gpu__time_duration.sum"


def parse(path):
    """Return a list of run dicts: {mode,pat,iters,pitch,outsum,metrics{}}."""
    runs = []
    cur = None
    for ln in open(path):
        m = re.match(r'^\s{0,4}mode=(\d+)\(.*?\) pat=(\d+) nunits=(\d+) iters=(\d+) '
                     r'pitch=(\d+)', ln)
        if m:
            if cur:
                runs.append(cur)
            cur = {"mode": int(m.group(1)), "pat": int(m.group(2)),
                   "nunits": int(m.group(3)), "iters": int(m.group(4)),
                   "pitch": int(m.group(5)), "outsum": None, "m": {}}
            continue
        if cur is None:
            continue
        mo = re.search(r'out\.sum=([-+0-9.eE]+)', ln)
        if mo:
            cur["outsum"] = float(mo.group(1))
            continue
        for key in (LDREQ, LDSEC, SRCREAD, L2HIT, DRAMR, DUR):
            if ln.strip().startswith(key):
                try:
                    cur["m"][key] = float(ln.split()[-1].replace(",", ""))
                except ValueError:
                    pass
    if cur:
        runs.append(cur)
    return runs


def main():
    runs = parse(MAIN)
    by = {(r["pat"], r["mode"]): r for r in runs}

    print("=" * 78)
    print("Part A - correctness: every read path accumulates the same value")
    print("=" * 78)
    ok = True
    for pat in (0, 1):
        sums = [by[(pat, m)]["outsum"] for m in MODES if (pat, m) in by]
        same = len(set(sums)) == 1
        ok &= same
        vals = " ".join(f"{MODES[m]}={by[(pat, m)]['outsum']:.6e}"
                        for m in MODES if (pat, m) in by)
        print(f"  pat={pat}: identical={same}  {vals}")
    print(f"  => {'PASS' if ok else 'FAIL'}")

    print()
    print("=" * 78)
    print("Part B - isolation table (pat x mode), 8 MB x iters=4")
    print("=" * 78)
    hdr = (f"{'pat':>3s} {'mode':7s} {'req':>8s} {'l1sec/req':>9s} "
           f"{'src/req':>8s} {'srcunit':>10s} {'l2hit':>9s} {'dramMB':>7s} {'us':>6s}")
    print(hdr)
    for pat in (0, 1):
        for mode in MODES:
            r = by.get((pat, mode))
            if not r:
                continue
            req = r["m"].get(LDREQ, 0) or 1
            print(f"{pat:3d} {MODES[mode]:7s} {req:8.0f} "
                  f"{r['m'].get(LDSEC, 0)/req:9.2f} "
                  f"{r['m'].get(SRCREAD, 0)/req:8.2f} "
                  f"{r['m'].get(SRCREAD, 0):10.0f} "
                  f"{r['m'].get(L2HIT, 0):9.0f} "
                  f"{r['m'].get(DRAMR, 0):7.2f} {r['m'].get(DUR, 0):6.2f}")
    print("  note: l1sec/req = 16 for every mode (512 B/warp / 32 B = access minimum;")
    print("        path-independent).  src/req splits by *L1 caching*: L1-allocating")
    print("        (ld16/ca16/ca8/ca4) read the 8 MB working set from L2 once")
    print("        (srcunit=262144); L1-bypass (cg16/ldcg16) read it once per reuse.")

    print()
    print("=" * 78)
    print("Part C - sweeps")
    print("=" * 78)
    sw = parse(SWEEP)
    print("  (C1) reuse factor: srcunit_read vs iters (pat=0, 8 MB working set =")
    print("       262144 sectors; DRAM read 8.39 MB for all)")
    print(f"  {'mode':7s} " + " ".join(f"iters={it:<8d}" for it in (1, 2, 4, 8)))
    for mode in (0, 1, 4):
        row = []
        for it in (1, 2, 4, 8):
            cand = [r for r in sw if r["pat"] == 0 and r["mode"] == mode
                    and r["iters"] == it]
            row.append(cand[0]["m"].get(SRCREAD, 0) if cand else 0)
        print(f"  {MODES[mode]:7s} " + " ".join(f"{v:13.0f}" for v in row) +
              "   /262144 = " + " ".join(f"{v/262144:.2f}x" for v in row))
    print("  (C2) alignment: l1tex sectors/request vs rows-pattern pitch (pat=1, iters=4)")
    print(f"  {'mode':7s} " + " ".join(f"pitch={p:<6d}" for p in (32, 40, 68)))
    for mode in (0, 4):
        row = []
        for p in (32, 40, 68):
            cand = [r for r in sw if r["pat"] == 1 and r["mode"] == mode
                    and r["pitch"] == p and r["iters"] == 4]
            if cand:
                r = cand[0]
                req = r["m"].get(LDREQ, 0) or 1
                row.append((r["m"].get(LDSEC, 0) / req, r["m"].get(SRCREAD, 0) / req))
            else:
                row.append((0, 0))
        print(f"  {MODES[mode]:7s} " +
              " ".join(f"l1={a:4.1f}/src={b:5.1f}" for a, b in row))


if __name__ == "__main__":
    main()
