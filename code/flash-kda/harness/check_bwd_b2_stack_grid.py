"""P7-107: parse the full-grid `(K,V) in {64,128,256}^2` `ncu` scan produced by
`harness/probe_b2_stack_ncu.sh` (P7-106 scanned only `(128,128)`/`(128,64)`; section 109.5 named the
verbatim open boundary "未扫全 K,V∈{64,128,256} 网格；(64,128) 与 K=256 未剖析").

The base is `bmt = -DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_BF16MMA -DB2M_BVK=64 -DB2M_TMA=1`
(bf16mma_tma).  Four tags per grid point (except `K=256`, where `-DB2M_INTRA`'s fp32 `q/k/g`
staging `3*BT*K*4` exceeds the 227 KiB smem cap -> only the `DAMMA` axis is scanned):

  bmt / bmt_da / bmt_intra / bmt_da_intra

The P7-106 "additive model" to re-test on the full grid:
  * `DAMMA` = pure shared-load **bank-conflict / wavefront** item, byte-neutral
    (global ld/st sectors, L2 bytes, DRAM read/write unchanged);
  * `INTRA` = global **ld sectors** increment (only K-dependent: reread `q/k/g` + `dAqk`) plus an
    smem occupancy stair, with **write** sectors / `dram__bytes_write` bit-unchanged;
  * their effects on bank conflict / wavefronts / ld sectors / duration are **additive**.

Usage:
    python harness/check_bwd_b2_stack_grid.py [harness/probe_b2_stack_ncu.out.txt]
"""
import re
import sys

KS = (64, 128, 256)
VS = (64, 128, 256)
TAGS = ("bmt", "bmt_da", "bmt_intra", "bmt_da_intra")
B, T, H, BT = 8, 2048, 8, 16
NT = (T + BT - 1) // BT

# (short, kind); kind in {"count-6" (value*1e-6), "dram-rd", "dram-wr", "l2", "time", "raw"}.
# `ncu` picks the byte/time unit per launch (Mbyte/Gbyte, ns/us/ms), so byte+time metrics must be
# converted by their printed unit (section 103.6 caveat).
METRICS = {
    "dram__bytes_read.sum": ("dram_rd_MB", "dram-rd"),
    "dram__bytes_write.sum": ("dram_wr_MB", "dram-wr"),
    "lts__t_bytes.sum": ("lts_GB", "l2"),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": ("ld_sec_M", "count-6"),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": ("st_sec_M", "count-6"),
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": ("bconf_M", "count-6"),
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": ("wf_M", "count-6"),
    "gpu__time_duration.sum": ("dur_ms", "time"),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("warps%", "raw"),
    "launch__registers_per_thread": ("regs", "raw"),
    "launch__occupancy_limit_registers": ("occ_reg", "raw"),
    "launch__occupancy_limit_shared_mem": ("occ_smem", "raw"),
}

BYTE_MB = {"Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3}
TIME_MS = {"s": 1e3, "ms": 1.0, "us": 1e-3, "ns": 1e-6}


def conv(kind, val, unit):
    if kind in ("count-6",):
        return val * 1e-6
    if kind == "dram-rd":
        return val * BYTE_MB.get(unit, 1.0)
    if kind == "dram-wr":
        return val * BYTE_MB.get(unit, 1.0)
    if kind == "l2":
        return val * BYTE_MB.get(unit, 1.0) * 1e-3  # -> GB
    if kind == "time":
        return val * TIME_MS.get(unit, 1.0)
    return val


def analytic_intra_ld_sectors(K):
    """`INTRA` fold rereads `q/k/g` as fp32 `[BT,KD]` staging + the `[BT,BT]` diagonal `dAqk`,
    once per chunk (`NT*B*H` chunks); sectors = bytes/32 (section 106.3)."""
    per_chunk = 3 * BT * K * 4 + BT * BT * 4
    return per_chunk / 32.0 * (NT * B * H)


def parse(path):
    hdr = re.compile(r"# b2prod_launch\s+shape=(\S+)\s+K=(\d+)\s+V=(\d+)\s+tag=(\S+)")
    rows, cur = {}, None
    with open(path) as f:
        for line in f:
            m = hdr.search(line)
            if m:
                cur = {"K": int(m.group(2)), "V": int(m.group(3)), "tag": m.group(4)}
                rows[(cur["K"], cur["V"], cur["tag"])] = cur
                continue
            if cur is None:
                continue
            s = line.strip()
            for key, (short, kind) in METRICS.items():
                if s.startswith(key):
                    parts = s.split()
                    try:
                        val = float(parts[-1])
                    except ValueError:
                        continue
                    unit = parts[-2] if len(parts) >= 3 else ""
                    cur[short] = conv(kind, val, unit)
    return rows


def show(rows, K, V, tags):
    print("\n(K=%d,V=%d):" % (K, V))
    hdrk = ("  %-13s %8s %9s %9s %8s %8s %9s %9s"
            % ("tag", "dur_ms", "bconf_M", "wf_M", "ld_M", "st_M", "dram_rd", "dram_wr"))
    print(hdrk)
    for t in tags:
        r = rows.get((K, V, t))
        if not r or "dur_ms" not in r:
            print("  %-13s  (not measured)" % t)
            continue
        print("  %-13s %8.3f %9.3f %9.3f %8.3f %8.3f %9.1f %9.1f"
              % (t, r["dur_ms"], r["bconf_M"], r["wf_M"], r["ld_sec_M"], r["st_sec_M"],
                 r["dram_rd_MB"], r["dram_wr_MB"]))
    b, d, i = rows.get((K, V, "bmt")), rows.get((K, V, "bmt_da")), rows.get((K, V, "bmt_intra"))
    di = rows.get((K, V, "bmt_da_intra"))
    if b and d and i and di:
        print("  -- additive check (bmt_da_intra vs damma+intra deltas) --")
        for key in ("bconf_M", "wf_M", "ld_sec_M", "dur_ms"):
            dd = d[key] - b[key]
            di_ = i[key] - b[key]
            obs = di[key] - b[key]
            print("     %-8s damma=%+.3f intra=%+.3f sum=%+.3f obs=%+.3f resid=%+.3f"
                  % (key, dd, di_, dd + di_, obs, obs - (dd + di_)))
        print("     DAMMA byte-neutral: dld=%.3f dst=%.3f dlts=%.3f dram_rd=%.1f dram_wr=%.1f"
              % (0.0, d["st_sec_M"] - b["st_sec_M"], d["lts_GB"] - b["lts_GB"],
                 d["dram_rd_MB"] - b["dram_rd_MB"], d["dram_wr_MB"] - b["dram_wr_MB"]))
        print("     dld sectors from DAMMA = %.3f  (expect ~0)" % (d["ld_sec_M"] - b["ld_sec_M"]))
        ana = analytic_intra_ld_sectors(K) * 1e-6
        print("     INTRA dld = %.3f M  analytic(3*BT*K*4+BT*BT*4) = %.3f M  ratio=%.3f"
              % (i["ld_sec_M"] - b["ld_sec_M"], ana, (i["ld_sec_M"] - b["ld_sec_M"]) / ana))
        print("     INTRA write bit-neutral: dst=%.3f  (expect 0)"
              % (i["st_sec_M"] - b["st_sec_M"]))
    if b and d and not i:
        print("  -- K=256: INTRA unlaunchable (smem); DAMMA-only axis --")
        for key in ("bconf_M", "wf_M", "ld_sec_M", "st_sec_M", "lts_GB", "dur_ms"):
            print("     DAMMA delta %-8s = %+.3f" % (key, d[key] - b[key]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "harness/probe_b2_stack_ncu.out.txt"
    rows = parse(path)
    print("parsed %d (K,V,tag) rows from %s" % (len(rows), path))
    print("stats: regs/occ by tag")
    for K in KS:
        for V in VS:
            for t in TAGS:
                r = rows.get((K, V, t))
                if r and "regs" in r:
                    print("  (K=%d,V=%d) %-13s regs=%d occ_reg=%d occ_smem=%d warps%%=%.2f"
                          % (K, V, t, int(r["regs"]), int(r["occ_reg"]), int(r["occ_smem"]),
                             r.get("warps%", float("nan"))))
    for K in KS:
        for V in VS:
            tags = TAGS if K != 256 else ("bmt", "bmt_da")
            show(rows, K, V, tags)

    # grid-wide summary
    print("\n=== grid-wide summary ===")
    print("(a) DAMMA: exactness of the 'pure conflict/wavefront, byte-neutral' claim")
    w_ld = w_st = 0.0
    w_l2 = w_rd = w_wr = 0.0
    w_intra_st = 0.0
    for K in KS:
        for V in VS:
            b, d = rows.get((K, V, "bmt")), rows.get((K, V, "bmt_da"))
            if b and d:
                w_ld = max(w_ld, abs(d["ld_sec_M"] - b["ld_sec_M"]))
                w_st = max(w_st, abs(d["st_sec_M"] - b["st_sec_M"]))
                w_l2 = max(w_l2, abs(d["lts_GB"] - b["lts_GB"]))
                w_rd = max(w_rd, abs(d["dram_rd_MB"] - b["dram_rd_MB"]))
                w_wr = max(w_wr, abs(d["dram_wr_MB"] - b["dram_wr_MB"]))
            i = rows.get((K, V, "bmt_intra"))
            if b and i:
                w_intra_st = max(w_intra_st, abs(i["st_sec_M"] - b["st_sec_M"]))
    print("    max |dld_sec|  = %.4f M   (expect exactly 0)" % w_ld)
    print("    max |dst_sec|  = %.4f M   (expect exactly 0)" % w_st)
    print("    max |dL2|      = %.3f GB  (L2-window noise)" % w_l2)
    print("    max |dRAM_rd|  = %.1f MB / |dRAM_wr| = %.1f MB  (L2-window noise)" % (w_rd, w_wr))
    print("    INTRA max |dst_sec| = %.4f M   (expect exactly 0)" % w_intra_st)

    print("(b) INTRA: ld-sector increment is K-only and analytic")
    for K in (64, 128):
        vals = []
        for V in VS:
            b, i = rows.get((K, V, "bmt")), rows.get((K, V, "bmt_intra"))
            if b and i:
                vals.append(i["ld_sec_M"] - b["ld_sec_M"])
        ana = analytic_intra_ld_sectors(K) * 1e-6
        print("    K=%d: dld = %s M (V=64/128/256); analytic = %.3f M" %
              (K, "/".join("%.3f" % x for x in vals), ana))

    print("(c) additivity residual (bmt_da_intra vs damma+intra) over K!=256")
    for K in (64, 128):
        for V in VS:
            b, d, i, di = (rows.get((K, V, t)) for t in TAGS)
            if not (b and d and i and di):
                continue
            res = []
            for key in ("bconf_M", "wf_M", "ld_sec_M", "dur_ms"):
                obs = di[key] - b[key]
                pred = (d[key] - b[key]) + (i[key] - b[key])
                res.append("%s=%+.3f" % (key, obs - pred))
            print("    (K=%d,V=%d): %s" % (K, V, "  ".join(res)))


if __name__ == "__main__":
    main()
