#!/usr/bin/env python3
# P97-1（§85）：把 `hetero_sm_tax_T xproc` 跨进程 / 跨 device 的原始样本汇总成
#   「T 趋势的进程间/设备间方差」证据——补 §84.4 明留的尾巴「CI 只含同一进程内 30 次采样」。
#
# 输入：由 harness/hetero_sm_tax_T_xproc.sh 生成的行，每行前缀 `P <dev> <rep> `，其后是二进制输出：
#   P 1 1 META numSM=132 S=30 seed=1001 cvd=1 name="..." clock_khz=...
#   P 1 1 RAW 0 128 0.0562 ...            # c 0..6，后接 S 个单采样 ms
#   P 1 1 MED 0 128 0.0568                # 该进程内该配置的 median
#   P 1 1 DONE
#
# 输出：(A) 每进程的 T512/T128、T256/T128 median 比；(B) 跨进程/跨 device 的比值分布
#   （median/mean/std/min/max + 对进程做 bootstrap 的 95% CI）；(C) 方差分解（进程内 vs 进程间）。
import sys, math, random

CFG_T = {0: 128, 1: 256, 2: 512, 3: 128, 4: 512, 5: 128, 6: 512}
CFG_NAME = {0: "mixed scalar+l2res", 1: "mixed scalar+l2res", 2: "mixed scalar+l2res",
            3: "l2res16M solo", 4: "l2res16M solo",
            5: "scalar4B solo", 6: "scalar4B solo"}
TOT_BYTES = 16777216 * 16  # 256 MiB


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n & 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean(v):
    return sum(v) / len(v)


def std(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def boot_ci(vals, B=4000, seed=7):
    rnd = random.Random(seed)
    n = len(vals)
    ms = sorted(median([vals[rnd.randrange(n)] for _ in range(n)]) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def boot_ratio_ci(a, b, B=4000, seed=11):
    # 进程内：median(b)/median(a) 的有放回 bootstrap（a,b 是同一进程的两组单采样）
    rnd = random.Random(seed)
    n = len(a)
    out = []
    for _ in range(B):
        ra = [a[rnd.randrange(n)] for _ in range(n)]
        rb = [b[rnd.randrange(n)] for _ in range(n)]
        ma, mb = median(ra), median(rb)
        out.append(mb / ma if ma > 0 else float("nan"))
    out = sorted(out)
    return out[int(0.025 * B)], out[int(0.975 * B)]


def main():
    if len(sys.argv) < 2:
        print("usage: hetero_sm_tax_T_xproc.py <raw.txt>")
        return 1
    path = sys.argv[1]
    procs = {}   # (dev,rep) -> {c: [samples]}
    meta = {}    # (dev,rep) -> meta string
    with open(path) as f:
        for line in f:
            if not line.startswith("P "):
                continue
            parts = line.rstrip("\n").split(" ", 3)
            if len(parts) < 4:
                continue
            dev, rep, body = parts[1], parts[2], parts[3]
            key = (dev, rep)
            tok = body.split()
            if not tok:
                continue
            if tok[0] == "META":
                meta[key] = body
            elif tok[0] == "RAW":
                c = int(tok[1])
                vals = [float(x) for x in tok[3:]]
                procs.setdefault(key, {})[c] = vals

    if not procs:
        print("no data parsed")
        return 1

    keys = sorted(procs.keys(), key=lambda k: (int(k[0]), int(k[1])))
    devices = sorted({k[0] for k in keys}, key=int)
    S = max(len(v) for p in procs.values() for v in p.values())

    print("########################################################################")
    print("# §85 块粒度 T 趋势的**跨进程 / 跨 device** 稳健性（P97-1，补 §84.4 尾巴）")
    print("#   每个进程跑 `xproc S=%d`（round 内 Fisher-Yates 打乱访问顺序）；" % S)
    print("#   device=%s；每 device %d 个独立进程，共 %d 进程。" %
          (",".join(devices), len(keys) // max(len(devices), 1), len(keys)))
    print("########################################################################")

    # (A) 每进程 median 与比值
    print("\n===== §A 每进程 median 与 T 比值（时间比；<1 = 大 T 更快）=====")
    print("  %-4s %-4s %10s %10s %10s %10s %10s  %8s %8s %8s %8s" %
          ("dev", "rep", "mixT128", "mixT256", "mixT512", "l2T128", "l2T512",
           "mix512/128", "mix256/128", "l2_512/128", "sc512/128"))
    per_proc = {}   # (dev,rep) -> dict of ratios
    for k in keys:
        p = procs[k]
        if not all(c in p for c in range(7)):
            continue
        m = {c: median(p[c]) for c in range(7)}
        r = {"mix512": m[2] / m[0], "mix256": m[1] / m[0],
             "l2": m[4] / m[3], "sc": m[6] / m[5]}
        per_proc[k] = r
        print("  %-4s %-4s %10.4f %10.4f %10.4f %10.4f %10.4f  %8.4f %8.4f %8.4f %8.4f" %
              (k[0], k[1], m[0], m[1], m[2], m[3], m[4],
               r["mix512"], r["mix256"], r["l2"], r["sc"]))

    def dist_report(title, key, invert=False):
        vals = [per_proc[k][key] for k in keys if k in per_proc]
        if not vals:
            return
        lo, hi = boot_ci(vals)
        md, mn, mx = median(vals), min(vals), max(vals)
        print("\n  --- %s：%d 进程 ---" % (title, len(vals)))
        print("      median=%.4f mean=%.4f std=%.4f min=%.4f max=%.4f  CI95(对进程)=[%.4f,%.4f]" %
              (md, mean(vals), std(vals), mn, mx, lo, hi))
        # 换算成吞吐
        print("      吞吐比(倒数) median=%.4f  (i.e. 大 T 吞吐 %+.1f%%)" %
              (1.0 / md, (1.0 / md - 1.0) * 100.0))
        sig = (lo > 1.0 or hi < 1.0)
        print("      => 跨进程 95%% CI %s 1.000：%s" %
              ("不含" if sig else "含",
               "T 趋势在进程/设备间稳健" if sig else "进程间不可区分于无效应"))

    print("\n===== §B 跨进程 / 跨 device 的比值分布（对「进程」做 bootstrap 95% CI）=====")
    dist_report("mixed scalar+l2res  T512/T128", "mix512")
    dist_report("mixed scalar+l2res  T256/T128", "mix256")
    dist_report("l2res16M solo       T512/T128", "l2")
    dist_report("scalar4B solo       T512/T128", "sc")

    # (C) 方差分解：进程内 vs 进程间
    print("\n===== §C 方差分解：同一配置「进程内样本 std」 vs 「跨进程 median std」=====")
    print("  (a) 每个进程内 S 个单采样的相对 std 之均值（测量噪声）")
    print("  (b) 跨进程 median 的相对 std（含进程/设备系统差）——若 (b)>>(a) 说明系统差不可忽略")
    print("  %-22s %12s %12s" % ("config", "withinStd%", "betweenStd%"))
    for c in (0, 1, 2, 3, 4, 5, 6):
        within = []
        meds = []
        for k in keys:
            p = procs[k]
            if c not in p:
                continue
            v = p[c]
            m = mean(v)
            within.append(std(v) / m if m > 0 else 0.0)
            meds.append(median(v))
        mm = mean(meds)
        bs = std(meds) / mm if mm > 0 else 0.0
        print("  %-22s %11.2f%% %11.2f%%" %
              ("%s T%d" % (CFG_NAME[c], CFG_T[c]), 100 * mean(within), 100 * bs))

    # (D) 每 device 汇总 + 进程内 CI 显著计数
    print("\n===== §D 每 device 的混合比值（设备间一致性）=====")
    for dev in devices:
        vals = [per_proc[k]["mix512"] for k in keys if k in per_proc and k[0] == dev]
        if vals:
            print("  dev=%-3s n=%d mix512/128 median=%.4f mean=%.4f std=%.4f" %
                  (dev, len(vals), median(vals), mean(vals), std(vals)))
    dev_meds = []
    for dev in devices:
        vals = [per_proc[k]["mix512"] for k in keys if k in per_proc and k[0] == dev]
        if vals:
            dev_meds.append(median(vals))
    if len(dev_meds) > 1:
        print("  设备间 median 的 spread = %.4f..%.4f（std=%.4f）" %
              (min(dev_meds), max(dev_meds), std(dev_meds)))

    print("\n===== §E 进程内 bootstrap CI：每个进程各自 T512/T128 是否显著 =====")
    sig_cnt = 0
    tot = 0
    for k in keys:
        p = procs[k]
        if not all(c in p for c in (0, 2)):
            continue
        lo, hi = boot_ratio_ci(p[0], p[2], B=2000, seed=hash(k) & 0xffff)
        tot += 1
        if lo > 1.0 or hi < 1.0:
            sig_cnt += 1
    print("  %d/%d 个进程的**进程内** CI95 不含 1.0（混合 T512/T128）" % (sig_cnt, tot))
    print("  => 若跨进程 CI 仍不含 1.0 且逐进程多数显著，则 §84 的 +4.3% 不是单进程偶发。")

    print("\n########## §85 xproc 汇总完成 ##########")
    return 0


if __name__ == "__main__":
    sys.exit(main())
