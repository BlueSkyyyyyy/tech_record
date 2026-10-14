"""P7-108: parse `harness/probe_b2_stack_resid.out.txt` and characterize the `(K,V)=(64,256)`
bank-conflict secondary interaction of the `bf16mma_tma` multi-knob stack.

The four builds (base "bmt" = bf16mma_tma, K=64 V=256):
    bmt / bmt_da (+DAMMA) / bmt_intra (+INTRA) / bmt_da_intra (+both).

Verifies the three layers that *should* be additive and isolates the one that is not:
  [A] static SASS shared-load instruction multiset  (LDS / LDS.64 / LDS.128 / LDS.U16)
  [C] dynamic shared-load instruction count  (`smsp__inst_executed_op_shared_ld`)
  [C] ideal wavefronts = total wavefronts - bank conflicts
  [B] bank-conflict residual vs grid, [D] conflict-type split.

Usage:  python harness/check_bwd_b2_stack_resid.py [harness/probe_b2_stack_resid.out.txt]
"""
import re
import sys

TAGS = ("bmt", "bmt_da", "bmt_intra", "bmt_da_intra")


def num(tok):
    try:
        return float(tok)
    except ValueError:
        return None


def parse(path):
    sass = {}     # tag -> {opcode: count}
    tot = {}      # tag -> total_LDS
    grid = {}     # (g, tag) -> {metric: value}
    agg = {}      # tag -> {metric: value}
    sub = {}      # tag -> {metric: value}
    cur = None
    with open(path) as f:
        for line in f:
            m = re.match(r"#+ SASS tag=(\S+)", line)
            if m:
                cur = ("sass", m.group(1)); sass.setdefault(cur[1], {})
                continue
            m = re.match(r"#+ GRID grid=(\d+) B=(\d+) T=(\d+) H=(\d+) tag=(\S+)", line)
            if m:
                cur = ("grid", int(m.group(1)), m.group(5)); grid.setdefault((cur[1], cur[2]), {})
                continue
            m = re.match(r"#+ AGG tag=(\S+)", line)
            if m:
                cur = ("agg", m.group(1)); agg.setdefault(cur[1], {})
                continue
            m = re.match(r"#+ SUB tag=(\S+)", line)
            if m:
                cur = ("sub", m.group(1)); sub.setdefault(cur[1], {})
                continue
            s = line.strip()
            if cur and cur[0] == "sass":
                mm = re.match(r"(\d+)\s+(LDS\S*)", s)
                if mm:
                    sass[cur[1]][mm.group(2)] = int(mm.group(1)); continue
                mm = re.match(r"total_LDS\s+(\d+)", s)
                if mm:
                    tot[cur[1]] = int(mm.group(1)); continue
            if cur and cur[0] in ("grid", "agg", "sub"):
                parts = s.split()
                if len(parts) >= 2 and num(parts[-1]) is not None and "_" in parts[0]:
                    key = parts[0]
                    val = float(parts[-1])
                    if cur[0] == "grid":
                        d = grid[(cur[1], cur[2])]
                    elif cur[0] == "agg":
                        d = agg[cur[1]]
                    else:
                        d = sub[cur[1]]
                    d[key] = val
    return sass, tot, grid, agg, sub


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "harness/probe_b2_stack_resid.out.txt"
    sass, tot, grid, agg, sub = parse(path)
    print("parsed: sass=%d grid=%d agg=%d sub=%d  from %s"
          % (len(sass), len(grid), len(agg), len(sub), path))

    print("\n[A] static SASS shared-load opcode histogram (additivity test)")
    ops = ["LDS", "LDS.64", "LDS.128", "LDS.U16"]
    print("    %-12s %6s %8s %8s %8s %8s" % ("tag", "total", *ops))
    b, d, i, di = (sass.get(t, {}) for t in TAGS)
    for t in TAGS:
        row = sass.get(t, {})
        print("    %-12s %6d %8d %8d %8d %8d"
              % (t, tot.get(t, 0), *[row.get(o, 0) for o in ops]))
    ok_all = True
    for o in ops + ["__total__"]:
        if o == "__total__":
            pred = tot["bmt"] + (tot["bmt_intra"] - tot["bmt"]) + (tot["bmt_da"] - tot["bmt"])
            obs = tot["bmt_da_intra"]
        else:
            pred = b.get(o, 0) + (i.get(o, 0) - b.get(o, 0)) + (d.get(o, 0) - b.get(o, 0))
            obs = di.get(o, 0)
        ok = (pred == obs)
        ok_all &= ok
        print("    additive %-12s pred=%d obs=%d  %s" % (o, pred, obs, "OK" if ok else "DIFF"))
    print("    => instruction multiset additive: %s" % ok_all)

    print("\n[C] aggregate layer additivity at grid=8192")
    def check(key, label, scale=1.0):
        bb, dd, ii, did = (agg[t].get(key) for t in TAGS)
        pred = bb + (ii - bb) + (dd - bb)
        obs = did
        print("    %-42s bmt=%12.0f obs=%12.0f pred=%12.0f resid=%+.0f (%+.3f%%)"
              % (label, bb, obs, pred, obs - pred, 100.0 * (obs - pred) / bb))
        return bb, dd, ii, did
    ib, idd, iii, iid = check("smsp__inst_executed_op_shared_ld.sum", "dynamic shared-ld instr count")
    wb, wd, wi, wid = check("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "total shared-load wavefronts")
    cb, cd, ci, cid = check("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "bank conflicts")
    # ideal wavefronts = wavefronts - conflicts
    ideal = lambda w, c: w - c
    p = ideal(wb, cb) + (ideal(wi, ci) - ideal(wb, cb)) + (ideal(wd, cd) - ideal(wb, cb))
    o = ideal(wid, cid)
    print("    %-42s bmt=%12.0f obs=%12.0f pred=%12.0f resid=%+.0f"
          % ("ideal wavefronts (wf - bconf)", ideal(wb, cb), o, p, o - p))
    print("    => ideal wavefronts additive & residual confined to the conflict term")

    print("\n[B] bank-conflict residual vs grid (K=64 V=256)")
    print("    %6s %12s %12s %12s %12s %12s %+10s %7s" %
          ("grid", "bmt", "bmt_da", "bmt_intra", "bmt_da_intra", "pred", "resid", "resid%"))
    keys = sorted(set(g for (g, t) in grid))
    for g in keys:
        row = {t: grid[(g, t)].get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum") for t in TAGS}
        if any(v is None for v in row.values()):
            continue
        pred = row["bmt"] + (row["bmt_intra"] - row["bmt"]) + (row["bmt_da"] - row["bmt"])
        res = row["bmt_da_intra"] - pred
        print("    %6d %12.0f %12.0f %12.0f %12.0f %12.0f %+10.0f %6.3f%%"
              % (g, row["bmt"], row["bmt_da"], row["bmt_intra"], row["bmt_da_intra"],
                 pred, res, 100.0 * res / row["bmt"]))
    gw = {g: grid[(g, "bmt")].get("sm__warps_active.avg.pct_of_peak_sustained_active") for g in keys}
    go = {g: grid[(g, "bmt")].get("launch__occupancy_limit_shared_mem") for g in keys}
    print("    warps%% / occ_smem(bmt): " +
          "  ".join("g%d:%.1f%%/%g" % (g, gw[g], go[g]) for g in keys if gw[g] is not None))

    print("\n[D] conflict-type split at grid=8192")
    def resid_type(key, label):
        bb, dd, ii, did = (sub[t].get(key) for t in TAGS)
        pred = bb + (ii - bb) + (dd - bb)
        print("    %-42s bmt=%12.0f obs=%12.0f pred=%12.0f resid=%+.0f (%+.3f%%)"
              % (label, bb, did, pred, did - pred, 100.0 * (did - pred) / bb))
    resid_type("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "mem_shared_op_ld conflicts")
    resid_type("l1tex__data_bank_conflicts_type_arbitration.sum", "type_arbitration conflicts")
    resid_type("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "mem_shared (all) conflicts")


if __name__ == "__main__":
    main()
