"""P7-100: parse the `ncu` write-path decomposition of the two crosses left open by docs/07
section 102.7 (P7-99):

  (1) `K!=V` crossed with `INTRA`/`DAMMA` (section 101 Part B only did the fp32 `prod` kernel at
      K!=V; sections 101/102 measured INTRA/DAMMA only at K==V);
  (2) `V=64` on the write path (never measured).

判据 (sections 99/100/101/102): compare write volume by **L1 global store sectors**
(`l1tex__t_sectors_pipe_lsu_mem_global_op_st` = S_L1), which tracks the analytic fp32 output
byte volume U(K,V); `dram__bytes_write` is an L2-writeback-window artifact and must not be
compared across configs.  The claims to test:

  * at fixed (K,V), S_L1 is bit-identical across {prod,damma,intra,intra_damma} for both K<V and K>V;
  * S_L1*32/U = 2 (default 32-bit strided store) for V=64 too.

Usage:
    python harness/check_bwd_b2_write_cross3.py [probe_b2_write_cross3_ncu.out.txt]
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
                        val = float(parts[-1])
                    except ValueError:
                        continue
                    if short == "dur_ms":
                        # ncu picks the unit per launch (ns/us/ms/s); convert to ms.
                        unit = parts[-2] if len(parts) >= 2 else "ms"
                        val *= {"s": 1e3, "ms": 1.0, "us": 1e-3, "ns": 1e-6}.get(unit, 1.0)
                    cur[short] = val
    if cur:
        blocks.append(cur)
    return blocks


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "harness/probe_b2_write_cross3_ncu.out.txt"
    blocks = parse(path)
    print(f"parsed {len(blocks)} ncu blocks from {path}\n")
    hdr = (f"{'tag':>13} {'K':>4} {'V':>4} | {'occ s/r':>8} {'regs':>5} {'warps%':>7} {'dur_ms':>7} | "
           f"{'S_L1':>10} {'S_L1_req':>9} {'l2_wr':>10} | {'L1*32/U':>8} {'DRAM_wr':>8} {'U_MB':>7}")
    print(hdr)
    print("-" * len(hdr))
    for b in blocks:
        U = out_volume(b["K"], b["V"])
        l1_st = b.get("l1_st", 0)
        print(f"{b['tag']:>13} {b['K']:>4} {b['V']:>4} | "
              f"{int(b.get('occ_smem', 0))}/{int(b.get('occ_reg', 0)):<5} "
              f"{int(b.get('regs', 0)):>5} {b.get('warps%', 0):>7.2f} {b.get('dur_ms', 0):>7.2f} | "
              f"{int(l1_st):>10} {int(b.get('l1_st_req', 0)):>9} {int(b.get('l2_wr', 0)):>10} | "
              f"{l1_st * 32 / U:>8.3f} {b.get('wr_MB', 0):>8.2f} {U/1e6:>7.1f}")

    # Part A: S_L1 / req identical across {prod,damma,intra,intra_damma} at a fixed K!=V
    print("\n== Part A: S_L1 invariance across INTRA/DAMMA at K!=V ==")
    for (K, V) in [(64, 128), (128, 64)]:
        grp = [b for b in blocks if (b["K"], b["V"]) == (K, V)]
        l1 = {int(b.get("l1_st", -1)) for b in grp}
        req = {int(b.get("l1_st_req", -1)) for b in grp}
        print(f"  (K,V)=({K},{V}) tags={[b['tag'] for b in grp]}")
        print(f"     distinct S_L1={l1}  distinct req={req}  "
              f"=> identical: {len(l1) == 1 and len(req) == 1}")

    # Part B: S_L1*32/U stays 2 (default) at V=64 / K=64,V=256
    print("\n== Part B: S_L1*32/U = 2 calibration at V=64 and (64,256) ==")
    for (K, V) in [(64, 64), (256, 64), (64, 256)]:
        for b in blocks:
            if (b["K"], b["V"]) == (K, V):
                U = out_volume(K, V)
                print(f"  (K,V)=({K},{V}) tag={b['tag']:>6}  S_L1={int(b.get('l1_st', 0)):>10}  "
                      f"S_L1*32/U={b.get('l1_st', 0)*32/U:.3f}")


if __name__ == "__main__":
    main()
