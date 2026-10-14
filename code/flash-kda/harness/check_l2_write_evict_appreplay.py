"""P8-17 / docs/07 §115: parse `harness/probe_l2_write_evict_appreplay_ncu.out.txt`
(from `harness/probe_l2_write_evict_appreplay_ncu.sh`) and check two things:

  [R] REPLAY ROBUSTNESS -- the §114 mechanism (store term bitwise invariant;
      `dram__bytes_write` set by the read/write *ordering*) is not a
      kernel-replay artifact: the kernel-replay (`KR_*`) and application-replay
      (`AR_*`) rows must agree on (a) `S_L1`/`L2_wr` exactly, (b) the ordering
      effect on `dram__bytes_write` (store-only ~0 / read-then-store ~0.3x /
      store-then-read ~1.3x).

  [C] READ-SIDE COMPLEMENT -- unlike the write side, `dram__bytes_read` is the
      read working set exactly, invariant to the read/write ordering and to the
      replay mode (read-only data has no eviction-share ambiguity).

Usage: python harness/check_l2_write_evict_appreplay.py [out.txt]
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
            if "writeKernel<" in line:
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


def first(tags, tag):
    if tag not in tags:
        return None
    for r in tags[tag]["runs"]:
        if SL1 in r:
            return r
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "harness/probe_l2_write_evict_appreplay_ncu.out.txt"
    tags = parse(path)
    print("parsed %d tags from %s" % (len(tags), path))

    print("\n[i] per-config table (outMB is MiB)")
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

    def pair(prefix, suffix):
        return first(tags, prefix + suffix), first(tags, "AR_" + suffix)

    print("\n[R] replay robustness: kernel (KR_) vs application (AR_) replay")
    print("  %-22s %-18s %-18s %-18s %-18s"
          % ("cfg", "S_L1 KR/AR", "L2_wr KR/AR", "dram_wr KR/AR (MB)",
             "dram_rd KR/AR (MB)"))
    for suffix in ("A1_sO8r0_mode0", "A2_sO8r128_mode1", "A3_sO8r128_mode2",
                   "A4_sO8r128_mode3", "B1_sO64r0_mode0", "B2_sO64r128_mode1"):
        kr = first(tags, "KR_" + suffix)
        ar = first(tags, "AR_" + suffix)
        if not kr or not ar:
            continue
        ok = (kr[SL1] == ar[SL1] and kr[L2W] == ar[L2W])
        print("  %-22s %-18s %-18s %-18s %-18s  store_identical=%s"
              % (suffix,
                 "%d/%d" % (kr[SL1], ar[SL1]),
                 "%d/%d" % (kr[L2W], ar[L2W]),
                 "%s/%s" % (mb(kr[DW]), mb(ar[DW])),
                 "%s/%s" % (mb(kr[DR]), mb(ar[DR])), ok))

    print("\n[R] ordering effect present in BOTH replays (dram_wr / output)")
    for suffix in ("A1_sO8r0_mode0", "A2_sO8r128_mode1", "A3_sO8r128_mode2",
                   "A4_sO8r128_mode3"):
        kr = first(tags, "KR_" + suffix)
        ar = first(tags, "AR_" + suffix)
        if not kr or not ar:
            continue
        ob = tags["KR_" + suffix]["outmb"] * 1024 * 1024
        print("  %-22s %-22s KR=%.3fx  AR=%.3fx"
              % (suffix, MODES[tags["KR_" + suffix]["mode"]],
                 kr[DW] / ob, ar[DW] / ob))

    print("\n[C] read side: dram__bytes_read == read working set, invariant")
    for suffix in ("A2_sO8r128_mode1", "A3_sO8r128_mode2", "A4_sO8r128_mode3",
                   "B2_sO64r128_mode1", "C1_sO8r32_mode1", "C2_sO8r64_mode1",
                   "C3_sO8r256_mode1"):
        kr = first(tags, "KR_" + suffix)
        if not kr:
            continue
        d = tags["KR_" + suffix]
        exp = d["readmb"]
        exp_b = exp * 1024 * 1024
        print("  %-22s read=%.0f MiB  dram_rd=%-9s MB (= %.4fx read footprint)"
              % (suffix, exp, mb(kr[DR]),
                 kr[DR] / exp_b if exp_b else 0.0))


if __name__ == "__main__":
    main()
