"""P7-111: parse `harness/probe_l2_write_evict_ncu.out.txt` (the ncu collection
from `harness/probe_l2_write_evict_ncu.sh`) and turn it into the write-side
`dram__bytes_write` accounting story that §99.5 / §102.4 left open.

Claims under test:
  [A] the store *stream* is identical across read/write orderings:
      `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` (S_L1) and
      `lts__t_sectors_srcunit_tex_op_write.sum` (L2-received writes) do not move.
  [B] `dram__bytes_write` *does* move, and is set by whether L2 traffic after
      the stores evicts the dirty output lines before the ncu window closes:
      read-then-store keeps them dirty in L2 (tiny flush), store-then-read /
      interleaved let the read stream cast them out (flush ~ output).
  [C] cycling the output working set past L2 forces the flush regardless of the
      read ordering (store-only already flushes).
  [D] the cross-run / cross-replay jitter lives entirely in this flush term; the
      store-identical counters are stable.

Usage:  python harness/check_l2_write_evict.py [harness/probe_l2_write_evict_ncu.out.txt]
"""
import re
import sys
from collections import OrderedDict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from probe_l2_write_evict import MODES  # noqa: E402

SL1 = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
L2W = "lts__t_sectors_srcunit_tex_op_write.sum"
DW = "dram__bytes_write.sum"
DR = "dram__bytes_read.sum"
DUR = "gpu__time_duration.sum"
BYTES = {SL1, L2W, DW, DR}
UNIT = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}

META = re.compile(r"^# TAG (\S+)\s+mode=(\d+) outMB=(\S+) readMB=(\S+)")


def parse(path):
    tags = OrderedDict()
    cur = None
    with open(path) as f:
        for line in f:
            m = META.match(line)
            if m:
                cur = tags.setdefault(m.group(1), dict(
                    mode=int(m.group(2)), outmb=float(m.group(3)),
                    readmb=float(m.group(4)), runs=[]))
                continue
            if cur is None:
                continue
            if "writeKernel<" in line:      # one ncu block per kernel instance
                cur["runs"].append({})
                continue
            toks = line.split()
            if not toks or toks[0] not in (SL1, L2W, DW, DR, DUR):
                continue
            name = toks[0]
            try:
                val = float(toks[-1])
            except ValueError:
                continue
            unit = toks[-2] if len(toks) >= 2 else "byte"
            if name in BYTES:
                val *= UNIT.get(unit, 1.0)
            cur["runs"][-1][name] = val
    return tags


def mb(x):
    return "-" if x is None else "%.2f" % (x / 1e6)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "harness/probe_l2_write_evict_ncu.out.txt"
    tags = parse(path)
    print("parsed %d tags from %s" % (len(tags), path))

    print("\n[i] per-config table (outMB is MiB; output bytes = outMB*1MiB)")
    print("  %-24s %-22s %6s %7s %10s %10s %9s %7s %9s %8s"
          % ("tag", "mode", "outMB", "readMB", "S_L1", "L2_wr", "dram_wr",
             "wr/out", "dram_rd", "dur_us"))
    for tag, d in tags.items():
        for r in d["runs"]:
            if SL1 not in r:
                continue
            ob = d["outmb"] * 1024 * 1024
            print("  %-24s %-22s %6.1f %7.1f %10.0f %10.0f %9s %7.3f %9s %8.1f"
                  % (tag, MODES[d["mode"]], d["outmb"], d["readmb"],
                     r.get(SL1, 0), r.get(L2W, 0), mb(r.get(DW)),
                     r.get(DW, 0) / ob, mb(r.get(DR)), r.get(DUR, 0)))

    def first(tag):
        if tag not in tags:
            return None
        for r in tags[tag]["runs"]:
            if SL1 in r:
                return r
        return None

    print("\n[A] store stream identical across read/write orderings (Part A, out=8 MiB)")
    base = None
    for tag in ("A1_sO8r0_mode0", "A2_sO8r128_mode1", "A3_sO8r128_mode2",
                "A4_sO8r128_mode3"):
        r = first(tag)
        if not r:
            continue
        if base is None:
            base = (r[SL1], r[L2W])
        print("  %-22s S_L1=%-10.0f L2_wr=%-10.0f  identical=%s  dram_wr=%s MB"
              % (tag, r[SL1], r[L2W],
                 (r[SL1] == base[0] and r[L2W] == base[1]), mb(r[DW])))

    print("\n[B] dram__bytes_write = dirty-line FLUSH share, set by ordering"
          " (out=8 MiB < L2~50 MiB)")
    for tag in ("A1_sO8r0_mode0", "A2_sO8r128_mode1", "A3_sO8r128_mode2",
                "A4_sO8r128_mode3", "A5_sO8r64_mode1", "A6_sO8r256_mode1"):
        r, d = first(tag), tags.get(tag)
        if not r:
            continue
        ob = d["outmb"] * 1024 * 1024
        print("  %-22s %-22s dram_wr=%-8s MB = %.2fx output (dram_rd=%s MB)"
              % (tag, MODES[d["mode"]], mb(r[DW]), r[DW] / ob, mb(r[DR])))

    print("\n[C] output > L2 forces the flush regardless of ordering (Part B)")
    for tag in ("B1_sO64r0_mode0", "B2_sO64r128_mode1", "B3_sO96r0_mode0"):
        r, d = first(tag), tags.get(tag)
        if not r:
            continue
        ob = d["outmb"] * 1024 * 1024
        print("  %-22s out=%-4.0f MiB %-22s dram_wr=%-8s MB = %.2fx output"
              % (tag, d["outmb"], MODES[d["mode"]], mb(r[DW]), r[DW] / ob))

    print("\n[D] repeatability: jitter lives in the flush term, not the store term")
    for tag in ("C1_sO8r0_mode0_x3", "C2_sO8r128_mode1_x3", "C3_sO8r0_mode0_x3b"):
        if tag not in tags:
            continue
        runs = [r for r in tags[tag]["runs"] if SL1 in r]
        print("  %-22s dram_wr = [%s] MB    S_L1 = [%s]"
              % (tag, ", ".join(mb(r.get(DW, 0)) for r in runs),
                 ", ".join("%.0f" % r.get(SL1, 0) for r in runs)))


if __name__ == "__main__":
    main()
