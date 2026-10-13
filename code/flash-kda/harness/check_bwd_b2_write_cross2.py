"""P7-99: parse the `ncu` write-path decomposition of the two crosses left open by docs/07
section 101.7 (P7-98):

  (1) INTRA/DAMMA cross at V=128 (section 101 only did V=256);
  (2) the three-way cross INTRA/DAMMA x TMA/BF16MMA (never measured);
  (3) a `-DB2M_VECST` sanity on top of INTRA/TMA (does the S_L1 halving survive?).

判据 (sections 99/100/101): compare write volume by **L1 global store sectors**
(`l1tex__t_sectors_pipe_lsu_mem_global_op_st` = S_L1), which tracks the analytic fp32 output
byte volume U(K,V); `dram__bytes_write` is a L2-writeback-window artifact and must not be
compared across configs.  The two new claims to test:

  * S_L1 is bit-identical across {prod,damma,intra,intra_damma,tma,damma_tma,intra_tma,
    bf16mma,bf16mma_damma,bf16mma_intra,bf16mma_intra_tma} at fixed (K,V);
  * S_L1*32/U = 2 default, = 1 with `-DB2M_VECST`, for INTRA/TMA too.

Usage:
    python harness/check_bwd_b2_write_cross2.py [probe_b2_write_cross2_ncu.out.txt]
"""
import re
import sys

# output volume U(K,V) in bytes for `-DB2M_PROD -DB2M_HDW` (section 99.5):
#   4x [B,T,H,K] (dqg,dkg,dg,dkgb) + [B,NT,H,K] (dgk)
# + 2x [B,T,H,V] (dv2,dvb) + 2x [B,T,H,BT] (dApre,dA) + [B,T,H] (db)
B, T, H, BT = 8, 2048, 8, 16


def out_volume(K, V):
    NT = (T + BT - 1) // BT
    e = 4  # fp32
    return (4 * (B * T * H * K) + (B * NT * H * K)
            + 2 * (B * T * H * V) + 2 * (B * T * H * BT) + (B * T * H)) * e


METRICS = {
    "dram__bytes_read.sum": "rd_GB",
    "dram__bytes_write.sum": "wr_MB",
    "lts__t_bytes.sum": "lts_GB",
    "lts__t_sectors_op_write.sum": "l2_wr",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "l1_ld",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "l1_st",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": "l1_st_req",
    "gpu__time_duration.sum": "dur_ms",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "warps%",
    "launch__registers_per_thread": "regs",
    "launch__occupancy_limit_registers": "occ_reg",
    "launch__occupancy_limit_shared_mem": "occ_smem",
}


def parse(path):
    hdr = re.compile(r"# b2prod_launch\s+shape=(\S+)\s+K=(\d+)\s+V=(\d+)\s+tag=(\S+)\s+launch_skip=(\d+)")
    blocks, cur = [], None
    with open(path) as f:
        for line in f:
            m = hdr.search(line)
            if m:
                if cur:
                    blocks.append(cur)
                cur = {"shape": m.group(1), "K": int(m.group(2)), "V": int(m.group(3)),
                       "tag": m.group(4), "skip": int(m.group(5))}
                continue
            if cur is None:
                continue
            for key, short in METRICS.items():
                if line.strip().startswith(key):
                    parts = line.split()
                    try:
                        cur[short] = float(parts[-1])
                    except ValueError:
                        pass
    if cur:
        blocks.append(cur)
    return blocks


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "harness/probe_b2_write_cross2_ncu.out.txt"
    blocks = parse(path)
    print(f"parsed {len(blocks)} ncu blocks from {path}\n")
    hdr = (f"{'tag':>18} {'K':>4} {'V':>4} | {'occ s/r':>8} {'regs':>5} {'warps%':>7} {'dur_ms':>7} | "
           f"{'S_L1':>10} {'S_L1_req':>9} {'l2_wr':>10} | {'L1*32/U':>8} {'DRAM_wr':>8} {'DRAM_rd':>8}")
    print(hdr)
    print("-" * len(hdr))
    for b in blocks:
        U = out_volume(b["K"], b["V"])
        l1_st = b.get("l1_st", 0)
        D = b.get("wr_MB", float("nan"))
        print(f"{b['tag']:>18} {b['K']:>4} {b['V']:>4} | "
              f"{int(b.get('occ_smem', 0))}/{int(b.get('occ_reg', 0)):<5} "
              f"{int(b.get('regs', 0)):>5} {b.get('warps%', 0):>7.2f} {b.get('dur_ms', 0):>7.2f} | "
              f"{int(l1_st):>10} {int(b.get('l1_st_req', 0)):>9} {int(b.get('l2_wr', 0)):>10} | "
              f"{l1_st * 32 / U:>8.3f} {D:>8.2f} {b.get('rd_GB', 0):>8.2f}")

    # Part A2 + D: S_L1 / req identical across all non-vecst knobs at a fixed (K,V)
    print("\n== S_L1 invariance across non-vecst knobs ==")
    for (K, V) in [(128, 128), (128, 256)]:
        grp = [b for b in blocks if (b["K"], b["V"]) == (K, V) and "vecst" not in b["tag"]]
        l1 = {int(b.get("l1_st", -1)) for b in grp}
        req = {int(b.get("l1_st_req", -1)) for b in grp}
        print(f"  (K,V)=({K},{V}) tags={[b['tag'] for b in grp]}")
        print(f"     distinct S_L1={l1}  distinct req={req}  "
              f"=> identical: {len(l1) == 1 and len(req) == 1}")

    # Part E: vecst halves S_L1, S_L1*32/U -> 1.0, for INTRA and INTRA+TMA
    print("\n== Part E: -DB2M_VECST on top of INTRA/TMA ==")
    pairs = [("intra", "intra_vecst"), ("intra_tma", "intra_tma_vecst")]
    for a, b in pairs:
        ra = [x for x in blocks if x["tag"] == a]
        rb = [x for x in blocks if x["tag"] == b]
        if ra and rb:
            x0, x1 = ra[-1], rb[-1]
            U = out_volume(x0["K"], x0["V"])
            print(f"  {a:>12} S_L1={int(x0['l1_st']):>10} (S_L1*32/U={x0['l1_st']*32/U:.3f}) -> "
                  f"{b:>16} S_L1={int(x1['l1_st']):>10} (S_L1*32/U={x1['l1_st']*32/U:.3f})  "
                  f"ratio={x1['l1_st']/x0['l1_st']:.3f}")


if __name__ == "__main__":
    main()
