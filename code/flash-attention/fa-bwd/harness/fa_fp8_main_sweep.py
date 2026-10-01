#!/usr/bin/env python3
"""fp8 主 kernel 的 host/runtime 旋钮系统扫描（O116）。

背景：ROADMAP「fp8 专项冲刺 / 下一批 F6·F3b·F4b」把 fp8 主 kernel 的 L2 墙逐条收口后，
唯一还没系统复核的是**纯 host/运行期旋钮**（ksplit / hswap / mrev / 端到端量化重叠 / 栅格封顶）
在同一 binary 里的联合最优性。本脚本把 `src/fp8/fa_bwd_fp8_main.cu` 编译出的可执行文件在
kernel_lab 容器里按「case × 配置」矩阵跑一遍，解析 `[timing]` 行，给出
「默认 vs 每档」的 main / total 比，用于一次性判定默认档是否已是最优。

用法（宿主机，任意目录）：
    python3 harness/fa_fp8_main_sweep.py                       # 默认 4 case × 9 配置
    python3 harness/fa_fp8_main_sweep.py --cases a b --configs default --ksplit=1
    python3 harness/fa_fp8_main_sweep.py --json out.json --iters 100

前置：先编译出二进制（`fa_bwd_fp8_main.out`，Hopper 构建）：
    ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda" \
        scripts/run.sh src/fp8/fa_bwd_fp8_main.cu --dir=... --iters=5
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = Path("/home/xieminglin/proj/output/fa-bwd")
CONTAINER = "kernel_lab"
BIN = "fa_bwd_fp8_main.out"
BIN_DIR = "src/fp8"

# (label, 追加到命令行的参数)。`default` 必须是空参数（走构建里的默认档）。
DEFAULT_CONFIGS = [
    ("default", []),
    ("ksplit=1", ["--ksplit=1"]),
    ("ksplit=4", ["--ksplit=4"]),
    ("hswap=0", ["--hswap=0"]),
    ("mrev=0", ["--mrev=0"]),
    ("ovlql=0", ["--ovlql=0"]),
    ("ovlcap=-1", ["--ovlcap=-1"]),
    ("ovlcap=1024", ["--ovlcap=1024"]),
    ("ovlcap=4096", ["--ovlcap=4096"]),
]

# 代表性 fp8 定长 causal case（覆盖 MHA / 大 H GQA / MQA / 小 S）。
DEFAULT_CASES = [
    ("b1_s4096_h16_d128_causal_fp8", 50),
    ("b1_s1024_h32_d128_kv4_causal_fp8", 100),
    ("b1_s1024_h64_d128_kv1_causal_fp8", 100),
    ("b1_s1024_h40_d128_kv8_causal_fp8", 100),
]

RE_TOTAL = re.compile(r"\[timing\] total\(quant\+pre\+main\+cvt\)\s+([0-9.]+)\s+ms")
RE_BREAK = re.compile(r"\[timing\] quant\s+([0-9.\-]+)\s+ms \| preprocess\s+([0-9.\-]+)\s+"
                      r"ms \| main\s+([0-9.\-]+)\s+ms")


def run_case(case, args, iters, bin_dir=BIN_DIR):
    inner = (f"cd {ROOT}/{bin_dir} && CUDA_VISIBLE_DEVICES=0 ./{BIN} "
             f"--dir={OUT_ROOT}/{case} --iters={iters} " + " ".join(args))
    p = subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=0", CONTAINER,
                        "bash", "-lc", inner], capture_output=True, text=True)
    out = p.stdout + p.stderr
    m_t = RE_TOTAL.search(out)
    m_b = RE_BREAK.search(out)
    if not m_t:
        return None
    res = {"total": float(m_t.group(1))}
    if m_b:
        res["quant"] = float(m_b.group(1))
        res["pre"] = float(m_b.group(2))
        res["main"] = float(m_b.group(3))
    # 参考打印：抓 grid/ksplit 行，便于确认配置确实生效。
    for line in out.splitlines():
        if "grid main" in line:
            res["grid"] = line.strip()
            break
    return res


def main():
    ap = argparse.ArgumentParser(description="fp8 main host-knob sweep (O116)")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--configs", nargs="+", default=None,
                    help="只跑这些 label（默认全部）；label 见 DEFAULT_CONFIGS")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--json", default=None)
    ap.add_argument("--bin-dir", default=BIN_DIR)
    args = ap.parse_args()

    cases = DEFAULT_CASES
    if args.cases:
        cases = [(c, args.iters) for c in args.cases]
    configs = DEFAULT_CONFIGS
    if args.configs:
        want = set(args.configs)
        configs = [c for c in DEFAULT_CONFIGS if c[0] in want]
        if not configs:
            print("没有匹配的 config label:", args.configs, file=sys.stderr)
            return 2

    out_json = {"cases": {}, "configs": [c[0] for c in configs]}
    print("=== fa_fp8_main_sweep (O116) ===")
    print(f"cases={[c for c, _ in cases]}  configs={[c[0] for c in configs]}")
    for case, iters in cases:
        print(f"\n--- {case} (iters={iters}) ---")
        base = None
        rows = []
        for label, cfargs in configs:
            r = run_case(case, cfargs, iters, args.bin_dir)
            rows.append((label, r))
            if label == "default":
                base = r
        if base is None:
            base = rows[0][1]
        hdr = f"{'config':<14} {'total ms':>9} {'main ms':>9} {'total×':>7} {'main×':>7}  grid"
        print(hdr)
        for label, r in rows:
            if r is None:
                print(f"{label:<14} {'FAIL':>9}")
                continue
            rx = r["total"] / base["total"]
            mx = r.get("main", float("nan")) / base.get("main", float("nan"))
            print(f"{label:<14} {r['total']:9.4f} {r.get('main', float('nan')):9.4f} "
                  f"{rx:7.3f} {mx:7.3f}  {r.get('grid', '')}")
            out_json["cases"].setdefault(case, {})[label] = r
    if args.json:
        Path(args.json).write_text(json.dumps(out_json, indent=2))
        print(f"\n[written] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
