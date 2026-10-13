"""P7-98: parse the `ncu` write-path decomposition of the remaining production knobs
(`-DB2M_INTRA` / `-DB2M_DAMMA` / `K!=V`, left open by docs/07 section 100.6) out of
`probe_b2_write_cross_ncu.out.txt` and print the three-metric table + the analytic output
volume U(K,V) and the residual Delta = D - U.

判据 (section 99/100): compare write volume by **L1 global store sectors**
(`l1tex__t_sectors_pipe_lsu_mem_global_op_st`), which is invariant in dtype / BVK / staging /
mma-ISA and tracks the fp32 output byte volume; `dram__bytes_write` is a L2-writeback-window
artifact and must not be compared across configs.

Usage:
    python harness/check_bwd_b2_write_cross.py [harness/probe_b2_write_cross_ncu.out.txt]
"""
import re
import sys

# output volume U(K,V) in bytes for `-DB2M_PROD -DB2M_HDW` (section 99.5):
#   4x [B,T,H,K]  (dqg,dkg,dg,dkgb) + [B,NT,H,K] (dgk)
# + 2x [B,T,H,V]  (dv2,dvb) + 2x [B,T,H,BT] (dApre,dA) + [B,T,H] (db)
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
}


def parse(path):
    hdr = re.compile(r"# b2prod_launch\s+shape=(\S+)\s+K=(\d+)\s+V=(\d+)\s+tag=(\S+)\s+launch_skip=(\d+)")
    blocks = []
    cur = None
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


def to_bytes(v, unit):
    return v * {"Gbyte": 1e9, "Mbyte": 1e6, "Kbyte": 1e3, "byte": 1}[unit]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "harness/probe_b2_write_cross_ncu.out.txt"
    blocks = parse(path)
    print(f"parsed {len(blocks)} ncu blocks from {path}\n")
    hdr = (f"{'tag':>12} {'K':>4} {'V':>4} | {'occ/reg':>8} {'warps%':>7} {'dur_ms':>7} | "
           f"{'L1_st':>10} {'L1_st_req':>10} {'l2_wr':>10} | {'DRAM_wr':>8} {'DRAM_rd':>8} | "
           f"{'U_MB':>8} {'D-U':>7}")
    print(hdr)
    print("-" * len(hdr))
    for b in blocks:
        U = out_volume(b["K"], b["V"])
        D = b.get("wr_MB", float("nan"))
        l1_st = b.get("l1_st", 0)
        print(f"{b['tag']:>12} {b['K']:>4} {b['V']:>4} | "
              f"{'-':>8} {b.get('warps%', float('nan')):>7.2f} {b.get('dur_ms', float('nan')):>7.2f} | "
              f"{int(l1_st):>10} {int(b.get('l1_st_req', 0)):>10} "
              f"{int(b.get('l2_wr', 0)):>10} | {D:>8.2f} {b.get('rd_GB', float('nan')):>8.2f} | "
              f"{U / 1e6:>8.1f} {D - U / 1e6:>7.1f}  {l1_st * 32 / U:>5.3f}")

    # group Part A by (K,V)=(128,256), tag != prodvec: S_L1 must be invariant across knobs
    pa = [b for b in blocks if (b["K"], b["V"]) == (128, 256) and b["tag"] != "prodvec"]
    l1 = {int(b.get("l1_st", -1)) for b in pa}
    req = {int(b.get("l1_st_req", -1)) for b in pa}
    l2 = {int(b.get("l2_wr", -1)) for b in pa}
    print("\nPart A (K=V=128, V=256): distinct L1_st =", l1, " distinct L1_st_req =", req,
          " distinct l2_wr =", l2)
    print("  -> L1 store sectors/requests identical across {prod,damma,intra,intra_damma,bf16,...}:",
          len(l1) == 1 and len(req) == 1)

    # Part C: store vectorization halves S_L1 while U is unchanged
    print("\nPart C (-DB2M_VECST): S_L1 halves, S_L1*32/U -> 1.0 (store flow = exact output bytes):")
    for V in (256, 128):
        base = [b for b in blocks if b["tag"] == "prod" and (b["K"], b["V"]) == (128, V)]
        vec = [b for b in blocks if b["tag"] == "prodvec" and (b["K"], b["V"]) == (128, V)]
        if base and vec:
            b0, b1 = base[-1], vec[-1]
            print(f"  V={V}: prod L1_st={int(b0['l1_st'])} -> prodvec L1_st={int(b1['l1_st'])} "
                  f"(ratio {b1['l1_st'] / b0['l1_st']:.3f}); "
                  f"U={out_volume(128, V) / 1e6:.1f} MB")

    # Part B: does S_L1 track U across K!=V?
    pb = [b for b in blocks if b["tag"] == "prod" and (b["K"], b["V"]) != (128, 256)]
    print("\nPart B (K!=V, fp32 prod): S_L1 / U ratio (sectors per output byte):")
    for b in pb:
        U = out_volume(b["K"], b["V"])
        l1_st = b.get("l1_st", 0)
        print(f"  K={b['K']:>3} V={b['V']:>3}  U={U/1e6:7.1f} MB  L1_st={int(l1_st):>10}  "
              f"L1_st*32/U={l1_st * 32 / U:.4f}")


if __name__ == "__main__":
    main()
