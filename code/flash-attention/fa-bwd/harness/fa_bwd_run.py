#!/usr/bin/env python3
"""fa_bwd_run.py —— 一键「跑 ours + 汇总」（ROADMAP 第一百零八轮 下一步候选 ①，P3-3c）。

背景：P3-3/P3-3b 把 ours 的对拍落盘接进了 harness，但每次仍要手写
`scripts/run.sh <host> --dir=<case> --dump=ours [--full] [--varlen]` 再手动跑
`fa_bwd_compare.py`。本脚本把这两步串成一个命令：

  * 扫描 `/home/xieminglin/proj/output/fa-bwd/<case>/`（要求有 `ref_{dq,dk,dv}.npy`）；
  * 按 `meta.json` 的 `dtype` / `varlen` / `causal` 选 host 与参数，调用 `scripts/run.sh`
    （**在 kernel_lab 容器内编译运行**）生成 `<prefix>_{dq,dk,dv}.npy`；
  * 同一 (host, 构建配置) 只编译一次，其余 case 直接复用已产出的可执行文件；
  * 最后调用 `fa_bwd_compare.py` 汇总 ours vs ref/FA/TE，并落盘原始输出。

用法（宿主机，从 `code/flash-attention/fa-bwd/` 或任意目录）：
  python harness/fa_bwd_run.py                         # 全部 case、两文件+单文件
  python harness/fa_bwd_run.py --dtype fp8             # 只看 fp8
  python harness/fa_bwd_run.py --fixed-only            # 只跑定长（默认 sm_90/mma）
  python harness/fa_bwd_run.py --varlen-only --dtype fp8
  python harness/fa_bwd_run.py --impls twofile         # 只跑两文件版
  python harness/fa_bwd_run.py --case b1_s512_h16_d128_causal_fp16 --impls both
  python harness/fa_bwd_run.py --no-run                # 只用已有 npy 重新汇总
  python harness/fa_bwd_run.py --dry-run               # 只打印将执行的命令

产物：
  src/<dtype>/fa_bwd_<dtype>_p33c_run.out.txt    每个 case 的原始运行输出（逐 dtype）
  src/fa_bwd_run_p33c_summary.out.txt           本脚本的运行清单/命令/结果摘要
  src/fa_bwd_compare_p33c_summary.out.txt       fa_bwd_compare.py 的数值汇总
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # code/flash-attention/fa-bwd
OUT_ROOT = Path("/home/xieminglin/proj/output/fa-bwd")
RUN_SH = ROOT / "scripts" / "run.sh"
CONTAINER = os.environ.get("LAB_NAME", "kernel_lab")

# host 源文件/可执行文件名（单文件 vs 两文件）。两文件 = 当前最优 mma 路径的 host，
# 与 P3-3/P3-3b 的 `ours` 口径一致；单文件用 `ours_sf`。
HOSTS = {
    "fp16": {
        "twofile":    ("src/fp16/fa_bwd_fp16_mma_main.cu",     "fa_bwd_fp16_mma_main"),
        "singlefile": ("src/fp16/fa_bwd_fp16_mma_onefile.cu",  "fa_bwd_fp16_mma_onefile"),
    },
    "bf16": {
        "twofile":    ("src/bf16/fa_bwd_bf16_mma_main.cu",     "fa_bwd_bf16_mma_main"),
        "singlefile": ("src/bf16/fa_bwd_bf16_mma_onefile.cu",  "fa_bwd_bf16_mma_onefile"),
    },
    "fp8": {
        "twofile":    ("src/fp8/fa_bwd_fp8_main.cu",           "fa_bwd_fp8_main"),
        "singlefile": ("src/fp8/fa_bwd_fp8_mma_onefile.cu",    "fa_bwd_fp8_mma_onefile"),
    },
}
PREFIX = {"twofile": "ours", "singlefile": "ours_sf"}

# 构建配置：定长走默认 sm_90（mma，与 P3-3 表一致）；varlen 入口在 `#ifdef FA_WGMMA` 内，
# 需 `sm_90a` + `-DFA_WGMMA`（fp16/bf16 必需；fp8 同样可行）。
BUILD = {
    "fixed":  {"ARCH": "sm_90", "NVCC_FLAGS": ""},
    "varlen": {"ARCH": "", "NVCC_FLAGS": "-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA"},
}


def discover(args):
    cases = []
    for c in sorted(p for p in OUT_ROOT.iterdir() if p.is_dir()):
        if args.case and c.name not in args.case:
            continue
        if args.glob:
            import fnmatch
            if not any(fnmatch.fnmatch(c.name, g) for g in args.glob):
                continue
        mp = c / "meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        dt = meta.get("dtype", "")
        if args.dtype and dt not in args.dtype:
            continue
        if not all((c / f"ref_{v}.npy").exists() for v in ("dq", "dk", "dv")):
            continue
        is_varlen = bool(meta.get("varlen", False)) or c.name.startswith("varlen_")
        if args.fixed_only and is_varlen:
            continue
        if args.varlen_only and not is_varlen:
            continue
        cases.append((c, meta, is_varlen))
    return cases


def build_args(case_dir: Path, meta, is_varlen: bool, prefix: str, iters: int):
    a = [f"--dir={case_dir}", f"--dump={prefix}", f"--iters={iters}"]
    if is_varlen:
        a.append("--varlen")
    if not meta.get("causal", True):
        a.append("--full")
    return a


def run_via_runsh(src: str, prog_args, build_env):
    env = dict(os.environ)
    env.update(build_env)
    cmd = [str(RUN_SH), src, *prog_args]
    return subprocess.run(cmd, cwd=str(ROOT), env=env,
                          capture_output=True, text=True)


def run_via_binary(src: str, bin_name: str, prog_args):
    src_dir = Path(os.path.realpath(ROOT / src)).parent
    inner = (f"cd '{src_dir}' && CUDA_VISIBLE_DEVICES=0 ./{bin_name}.out "
             + " ".join(prog_args))
    return subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=0",
                           CONTAINER, "bash", "-lc", inner],
                          capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser(description="one-click run ours + aggregate (P3-3c)")
    ap.add_argument("--dtype", nargs="+", default=None)
    ap.add_argument("--glob", nargs="+", default=None)
    ap.add_argument("--case", nargs="+", default=None)
    ap.add_argument("--impls", nargs="+", default=["twofile", "singlefile"],
                    choices=["twofile", "singlefile", "both"])
    ap.add_argument("--fixed-only", action="store_true")
    ap.add_argument("--varlen-only", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--no-run", action="store_true", help="跳过 kernel，只重新汇总")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    impls = ["twofile", "singlefile"] if "both" in args.impls else args.impls
    cases = discover(args)
    if not cases:
        print("没有匹配的 case（检查 --dtype/--glob/--case 或 dump 目录）")
        return 1

    summary = ["=== fa_bwd_run.py (P3-3c) 一键跑 ours + 汇总 ===",
               f"cases={len(cases)} impls={impls} iters={args.iters}"]
    logs = {}  # dtype -> [lines]
    built = {}  # (src, cfg) -> bin_name

    compare_cases = []
    for case_dir, meta, is_varlen in cases:
        dt = meta.get("dtype", "")
        if dt not in HOSTS:
            summary.append(f"[skip] {case_dir.name}: 未知 dtype={dt}")
            continue
        cfg = "varlen" if is_varlen else "fixed"
        compare_cases.append(case_dir.name)
        for impl in impls:
            src, bin_name = HOSTS[dt][impl]
            prefix = PREFIX[impl]
            prog_args = build_args(case_dir, meta, is_varlen, prefix, args.iters)
            hdr = (f"=== [P3-3c] {dt} {src} {case_dir.name} {prefix} "
                   f"({'varlen' if is_varlen else 'fixed'}) ===")
            logs.setdefault(dt, []).append(hdr)
            summary.append(f"[run] {dt} {impl} {case_dir.name}")
            if args.dry_run:
                summary.append(f"      run.sh {src} {' '.join(prog_args)}")
                logs[dt].append("  (dry-run)")
                continue

            key = (src, cfg)
            if key in built and Path(os.path.realpath(ROOT / src)).with_suffix(".out").exists():
                r = run_via_binary(src, built[key], prog_args)
                how = f"binary {built[key]}.out"
            else:
                r = run_via_runsh(src, prog_args, BUILD[cfg])
                built[key] = bin_name
                how = f"run.sh ({cfg})"
            out = (r.stdout or "") + (r.stderr or "")
            logs[dt].append(f"  [{how}] rc={r.returncode}")
            logs[dt].append(out.rstrip())
            # 打印关键对拍行
            for ln in out.splitlines():
                if "max_abs" in ln or "[dump]" in ln or "max_abs=" in ln:
                    logs[dt].append("  | " + ln.strip())
            if r.returncode != 0:
                summary.append(f"      !! rc={r.returncode} ({how})")

    # 落盘
    for dt, lines in logs.items():
        p = ROOT / "src" / dt / f"fa_bwd_{dt}_p33c_run.out.txt"
        p.write_text("\n".join(lines) + "\n")
        summary.append(f"[written] {p}")

    sp = ROOT / "src" / "fa_bwd_run_p33c_summary.out.txt"
    sp.write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\n[written] {sp}")

    if args.no_run or args.no_compare or not compare_cases:
        return 0

    cmp_out = ROOT / "src" / "fa_bwd_compare_p33c_summary.out.txt"
    cmd = [sys.executable, str(ROOT / "harness" / "fa_bwd_compare.py"),
           "--case", *compare_cases, "--out", str(cmp_out)]
    print("\n[compare] " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
