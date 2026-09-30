#!/usr/bin/env python3
"""O86（第一百八十一轮）：量化默认 fp8 `kvtma` main 的 L2 `red` 与「Q 头数 H」的线性律，
并给出「GQA/MQA 跨 Q 头折叠 dK/dV」的收益上界。

背景（见 docs/03 §109 / ROADMAP 阻塞）：默认 fp8 反向的 L2 `red`（跨 CTA 原子归约）只由
**Q 头数 H** 决定、与 KV 头数 Hkv 无关——MQA（H64kv1）与 GQA（H64kv4）的 `red` 逐字节相同。
因此「把共享同一 KV 头的 G=H/Hkv 个 Q 头的 dK/dV 偏和折叠成一次贡献」理论上可把
dK/dV 的 `red` 压 (G-1)/G（MQA 最多 ×64）。本脚本把这条律与上界测量/打印出来，作为
F4b 收口的可复现证据（判决见 docs/03 §109：loop-order 冲突 ⇒ 本卡不可行）。

用法（在 kernel_lab 容器里跑，或宿主机用 --file 解析已存输出）：
  python harness/fa_fp8_red_law.py                 # 现场 ncu 测 5 个点
  python harness/fa_fp8_red_law.py --file src/fp8/fa_bwd_fp8_o86_gqa_red_probe.out.txt
"""
import argparse
import os
import re
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "fp8")
BIN = "fa_bwd_fp8_main.out"
OUTBASE = "/home/xieminglin/proj/output/fa-bwd"
# (case, H, Hkv, causal)
CASES = [
    ("b1_s1024_h32_d128_causal_fp8", 32, 32, True),
    ("b1_s1024_h40_d128_kv8_causal_fp8", 40, 8, True),
    ("b1_s1024_h64_d128_kv4_causal_fp8", 64, 4, True),
    ("b1_s1024_h64_d128_kv1_causal_fp8", 64, 1, True),
    ("b1_s4096_h16_d128_causal_fp8", 16, 16, True),
]
METRICS = ",".join([
    "gpu__time_duration.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_red.sum",
    "lts__t_sectors_op_write.sum",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
])


def run_ncu(case):
    cmd = ["ncu", "--metrics", METRICS, "--kernel-name", "regex:kvtma",
           "--launch-count", "1", "./" + BIN, "--dir=" + os.path.join(OUTBASE, case),
           "--iters=10"]
    p = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    return p.stdout + p.stderr


def parse(text):
    """解析 ncu 文本，返回 metric -> 数值（sector 取整数，百分比/时间取 float）。"""
    out = {}
    for m in re.finditer(
            r"^\s+([a-z0-9_.]+(?:\.sum|\.pct_of_peak_sustained_elapsed))\s+"
            r"([a-z%]+)\s+([0-9.eE+-]+)", text, re.M):
        name, unit, val = m.group(1), m.group(2), float(m.group(3))
        if unit == "ms":
            val *= 1000.0
        out[name] = int(val) if name.endswith(".sum") else val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="解析已保存的 ncu 文本（离线，不跑 GPU）")
    args = ap.parse_args()

    rows = []
    for case, H, Hkv, _ in CASES:
        text = open(args.file).read() if args.file else run_ncu(case)
        # 离线解析：把 `===== <case> =====` 之后的正文段取出
        if args.file:
            parts = re.split(r"^=====\s*(.*?)\s*=====\s*$", text, flags=re.M)
            text = ""
            for i in range(1, len(parts) - 1, 2):
                if case in parts[i]:
                    text = parts[i + 1]
                    break
        mt = parse(text)
        if not mt.get("lts__t_sectors_op_red.sum"):
            print(f"[skip] {case}: 未解析到 red（ncu 失败或不支持）", file=sys.stderr)
            continue
        rows.append((case, H, Hkv, H / Hkv, mt))

    print(f"{'case':42s} {'H':>3s} {'Hkv':>4s} {'G':>3s} "
          f"{'red':>12s} {'read':>10s} {'red/H':>10s} {'dur':>9s} {'L2%':>6s}")
    red_per_h = []
    for case, H, Hkv, G, mt in rows:
        red = mt["lts__t_sectors_op_red.sum"]
        read = mt.get("lts__t_sectors_op_read.sum", 0)
        red_per_h.append(red / H)
        print(f"{case:42s} {H:3d} {Hkv:4d} {G:3.0f} {red:12d} {read:10d} "
              f"{red/H:10.3e} {mt.get('gpu__time_duration.sum', 0):7.1f}{'us':>2s} "
              f"{mt.get('lts__throughput.avg.pct_of_peak_sustained_elapsed', 0):6.1f}")

    if red_per_h:
        import statistics
        s1024 = [mt["lts__t_sectors_op_red.sum"] / H
                 for (case, H, Hkv, G, mt) in rows if "s1024" in case]
        print(f"\nred/H 常数（S1024 各点）：均值 {statistics.mean(s1024):.4e}"
              f"（极差 {max(s1024) - min(s1024):.2e}）")
        print("⇒ `red` 只由 Q 头数 H 决定（MQA H64kv1 == GQA H64kv4）；"
              "跨 Q 头折叠 dK/dV 的上界 = red 的 dK/dV 份额 ÷G（MQA ×64）。")
        print("⇒ 判决（docs/03 §109）：本卡 loop-order/寄存器/smem 墙锁定，不可行。")


if __name__ == "__main__":
    main()
