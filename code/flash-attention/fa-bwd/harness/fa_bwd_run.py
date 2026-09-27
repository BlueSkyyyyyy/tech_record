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
  python harness/fa_bwd_run.py --doc-table             # 额外产出 docs/04 分组表（P3-3d）
  python harness/fa_bwd_run.py --doc-table-apply       # 跑完 ours 后原地同步 docs/04 内嵌表（P3-3e）
  python harness/fa_bwd_run.py --doc-table-check       # 只校验 docs/04 内嵌表是否最新（不跑 kernel）
  python harness/fa_bwd_run.py --consistency --no-run  # 单文件 vs 两文件一致性报告（用已有 npy）
  python harness/fa_bwd_run.py --no-consistency        # 关掉默认的「单/两文件一致性 gate」（P3-3g）
  python harness/fa_bwd_run.py --dry-run               # 只打印将执行的命令
  python harness/fa_bwd_run.py --hopper                # 定长也走 Hopper 快路（-DFA_WGMMA -DFA_TMA，P3-4d）
  python harness/fa_bwd_run.py --ci --no-run           # 一条命令：汇总 + 一致性 gate + docs/04 表校验（P3-4d）
  python harness/fa_bwd_run.py --ci --perf-baseline fp16   # CI + 纯反向基线（FA2/FA3/TE）落盘

P3-3g：一次「全量扫」（两边形态都跑，默认）结束时会**自动**调用
`fa_bwd_compare.py --consistency --ctol auto`（按 dtype 的 ulp 容差，见该文件的 `CTOL_AUTO`），
任一 dtype 的 `max|ours-ours_sf|` 超门即以非零退出码失败——单/两文件 device 代码分叉会被
端到端回归自动抓住，不必再逐轮人工誊抄。`--impls twofile`（只跑一边）或 `--no-consistency`
时跳过。`--consistency` 可显式强制（如配 `--no-run` 用已有 npy 复核）。

产物：
  src/<dtype>/fa_bwd_<dtype>_p33c_run.out.txt    每个 case 的原始运行输出（逐 dtype）
  src/fa_bwd_run_p33c_summary.out.txt           本脚本的运行清单/命令/结果摘要
  src/fa_bwd_compare_p33c_summary.out.txt       fa_bwd_compare.py 的数值汇总
  src/fa_bwd_consistency_p33g.out.txt           一致性报告（P3-3g 默认产物）

P3-4d：把「跑 ours + 汇总 + 一致性 gate」再收口成**一条 CI 命令**（`--ci`），并在跑完后
再调用 `fa_bwd_compare.py --check` 校验 `docs/04` 内嵌表是否与实测一致（陈旧则退出码 1），
于是「单/两文件分叉」与「文档陈旧」两类回归都被同一条命令拦住。`--hopper` 让**定长** case
也走 `-DFA_WGMMA -DFA_TMA`（Hopper 快路，此前只有变长入口在 sm90a 下编译；与
`docs/04` 表的默认 sm_90 mma 口径互补）。`--perf-baseline <dtype>` 额外调用用户指定的纯反向
基线 `fa_vs_te_bwd_only.py`（FA2/FA3/TE 三列）并把原始输出落盘到
`src/fa_bwd_perf_baseline_<dtype>.out.txt`。
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
# P3-4d：Hopper 快路（--hopper）用**独立前缀**，避免把默认 mma 口径的 `ours`/`ours_sf` npy
# 覆盖掉——wgmma/TMA 与 mma 的数值差可达 O(1e-1)（`docs/03` O9c-2 A/B），若混用会让
# 「单/两文件一致性 gate」把「构建配置差异」误判成「实现分叉」。
HOPPER_PREFIX = {"twofile": "ours_hp", "singlefile": "ours_sf_hp"}

# 构建配置：定长走默认 sm_90（mma，与 P3-3 表一致）；varlen 入口在 `#ifdef FA_WGMMA` 内，
# 需 `sm_90a` + `-DFA_WGMMA`（fp16/bf16 必需；fp8 同样可行）。
BUILD = {
    "fixed":  {"ARCH": "sm_90", "NVCC_FLAGS": ""},
    "varlen": {"ARCH": "", "NVCC_FLAGS": "-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA"},
}
# P3-4d：Hopper 快路（`--hopper`）——定长也走 TMA+wgmma。`-lcuda` 是 TMA 描述符
# （`cuTensorMapEncodeTiled`）必需的链接项；`ARCH=""` 让 flags 里的 gencode 生效
# （CUDA 13 的 nvcc 不认 `-arch=sm_90a`）。
HOPPER_FLAGS = "-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA -DFA_TMA -lcuda"


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
    ap.add_argument("--doc-table", action="store_true",
                    help="额外产出 docs/04 可内联的分组表（P3-3d）")
    ap.add_argument("--doc-table-out", default=str(ROOT / "src" / "fa_bwd_compare_p33d_table.md"))
    ap.add_argument("--doc-table-apply", action="store_true",
                    help="跑完 ours 后用实测原地同步 docs/04 的 auto-doc-table 块（P3-3e）")
    ap.add_argument("--doc-table-check", action="store_true",
                    help="只校验 docs/04 的 auto-doc-table 块是否最新（P3-3e；不跑 kernel）")
    ap.add_argument("--consistency", action="store_true",
                    help="P3-3f：强制报告单文件 vs 两文件（ours vs ours_sf）的逐 case 一致性")
    ap.add_argument("--no-consistency", action="store_true",
                    help="P3-3g：关掉全量扫默认的单/两文件一致性 gate（两边形态都跑时才默认开）")
    ap.add_argument("--consistency-tol", default="auto", metavar="TOL",
                    help="P3-3g：'auto'=按 dtype 的 ulp 容差（推荐）；浮点=全局标量；"
                         "缺省 auto（超出则退出码 1）")
    ap.add_argument("--consistency-out",
                    default=str(ROOT / "src" / "fa_bwd_consistency_p33g.out.txt"))
    ap.add_argument("--docs-md", default=str(ROOT / "docs" / "04-numerics-and-perf-summary.md"))
    ap.add_argument("--hopper", action="store_true",
                    help="P3-4d：定长也走 Hopper 快路构建（-DFA_WGMMA -DFA_TMA -lcuda，sm90a）")
    ap.add_argument("--ci", action="store_true",
                    help="P3-4d：一条命令收口——跑完后自动校验 docs/04 内嵌表最新（陈旧则退出码 1）")
    ap.add_argument("--perf-baseline", default=None, metavar="DTYPE",
                    help="P3-4d：额外跑纯反向基线 fa_vs_te_bwd_only.py <DTYPE>（FA2/FA3/TE）并落盘")
    ap.add_argument("--perf-baseline-out", default=None,
                    help="--perf-baseline 的产物路径（默认 src/fa_bwd_perf_baseline_<dtype>.out.txt）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # P3-4d：--hopper 覆盖定长/变长的构建配置（同一次调用内一致），并切到独立前缀，
    # 以免污染默认 mma 口径的 ours/ours_sf（两者数值可差 O(1e-1)，见 HOPPER_PREFIX 注释）。
    prefix_map = dict(PREFIX)
    if args.hopper:
        BUILD["fixed"] = {"ARCH": "", "NVCC_FLAGS": HOPPER_FLAGS}
        BUILD["varlen"] = {"ARCH": "", "NVCC_FLAGS": HOPPER_FLAGS}
        prefix_map = dict(HOPPER_PREFIX)

    # --doc-table-check 是只读校验：不跑 kernel、不依赖 case 过滤。
    if args.doc_table_check:
        cmd = [sys.executable, str(ROOT / "harness" / "fa_bwd_compare.py"),
               "--check", args.docs_md]
        print("[doc-table-check] " + " ".join(cmd))
        return subprocess.run(cmd).returncode

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
            prefix = prefix_map[impl]
            prog_args = build_args(case_dir, meta, is_varlen, prefix, args.iters)
            hdr = (f"=== [P3-3c] {dt} {src} {case_dir.name} {prefix} "
                   f"({'varlen' if is_varlen else 'fixed'}) ===")
            logs.setdefault(dt, []).append(hdr)
            summary.append(f"[run] {dt} {impl} {case_dir.name}")
            if args.dry_run:
                summary.append(f"      run.sh {src} {' '.join(prog_args)}")
                logs[dt].append("  (dry-run)")
                continue
            if args.no_run:
                # P3-3f 修复：`--no-run` 原文档说「跳过 kernel、只重新汇总」，但循环未检查它，
                # 仍会编译+运行所有 case（一次全量扫会跑 146 次）。这里真正跳过执行，
                # 复用已有 npy（compare/consistency 直接读 dump）。
                summary.append("      (--no-run: 复用已有 npy)")
                logs[dt].append("  (--no-run: 复用已有 npy)")
                continue

            key = (src, cfg, args.hopper)
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

    # 落盘（--no-run 时不改写既有的运行日志/清单，避免用空内容覆盖上一轮全量扫的产物）
    if args.no_run:
        print("\n".join(summary))
    else:
        for dt, lines in logs.items():
            p = ROOT / "src" / dt / f"fa_bwd_{dt}_p33c_run.out.txt"
            p.write_text("\n".join(lines) + "\n")
            summary.append(f"[written] {p}")
        sp = ROOT / "src" / "fa_bwd_run_p33c_summary.out.txt"
        sp.write_text("\n".join(summary) + "\n")
        print("\n".join(summary))
        print(f"\n[written] {sp}")

    if args.doc_table_apply:
        # 同步 docs/04 内嵌块：扫全部 dump（不限定本轮 case），保证表覆盖完整。
        # 放在 compare_cases 早退之前，使 `--no-run --doc-table-apply` 也能用已有 dump 重出表。
        cmd3 = [sys.executable, str(ROOT / "harness" / "fa_bwd_compare.py"),
                "--apply", args.docs_md]
        print("\n[compare --apply] " + " ".join(cmd3))
        r3 = subprocess.run(cmd3, capture_output=True, text=True)
        print(r3.stdout)
        if r3.returncode != 0:
            print(r3.stderr, file=sys.stderr)
            return r3.returncode

    # P3-3g：两边文件形态都跑（默认）时自动 gate 一致性；--impls 只跑一边或显式
    # --no-consistency 时跳过；--consistency 可强制（含 --no-run 用已有 npy 复核）。
    both_impls = ("twofile" in impls) and ("singlefile" in impls)
    auto_consistency = both_impls and not args.no_consistency
    if args.consistency or auto_consistency:
        cmd_c = [sys.executable, str(ROOT / "harness" / "fa_bwd_compare.py"),
                 "--consistency", "--ca", prefix_map["twofile"], "--cb", prefix_map["singlefile"],
                 "--out", args.consistency_out]
        if compare_cases:
            cmd_c += ["--case", *compare_cases]
        if args.consistency_tol is not None:
            cmd_c += ["--ctol", str(args.consistency_tol)]
        tag = "auto" if (auto_consistency and not args.consistency) else "explicit"
        print(f"\n[consistency:{tag}] " + " ".join(cmd_c))
        rc = subprocess.run(cmd_c).returncode
        if rc != 0:
            print(f"[consistency] gate FAIL (rc={rc})", file=sys.stderr)
            return rc
    elif both_impls and args.no_consistency:
        summary.append("[consistency] skipped (--no-consistency)")
        print("\n[consistency] skipped (--no-consistency)")

    # P3-4d：--perf-baseline —— 调用用户指定的纯反向基线（FA2/FA3/TE 三列，forward 在计时区外），
    # 在容器内跑（该脚本 import torch），原始输出落盘。与 ours 无关，`--no-run` 也可用。
    ci_lines = list(summary)
    if args.perf_baseline:
        dt = args.perf_baseline
        pbo = (Path(args.perf_baseline_out) if args.perf_baseline_out
               else ROOT / "src" / f"fa_bwd_perf_baseline_{dt}.out.txt")
        inner = (f"cd '{ROOT}' && python harness/fa_vs_te_bwd_only.py {dt}")
        print(f"\n[perf-baseline] docker exec {CONTAINER}: {inner}")
        rp = subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", inner],
                            capture_output=True, text=True)
        pbo.write_text((rp.stdout or "") + (rp.stderr or ""))
        print(rp.stdout)
        ci_lines.append(f"[perf-baseline] {dt} rc={rp.returncode} -> {pbo}")
        if rp.returncode != 0:
            print(rp.stderr, file=sys.stderr)

    # P3-4d：--ci —— 在「一致性 gate」之上再校验 docs/04 内嵌表是否最新；陈旧即退出码 1。
    # 跑 `fa_bwd_compare.py --check`（扫全部 dump、按 rtol 吸收原子次序噪声）。
    if args.ci:
        cmd_chk = [sys.executable, str(ROOT / "harness" / "fa_bwd_compare.py"),
                   "--check", args.docs_md]
        print("\n[ci doc-table-check] " + " ".join(cmd_chk))
        rc = subprocess.run(cmd_chk).returncode
        ci_lines.append(f"[ci] doc-table-check rc={rc}")
        (ROOT / "src" / "fa_bwd_ci.out.txt").write_text("\n".join(ci_lines) + "\n")
        if rc != 0:
            print("[ci] docs/04 内嵌表陈旧（rc=%d）" % rc, file=sys.stderr)
            return rc

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

    if args.doc_table:
        cmd2 = [sys.executable, str(ROOT / "harness" / "fa_bwd_compare.py"),
                "--doc-table", "--case", *compare_cases, "--out", str(args.doc_table_out)]
        print("\n[compare --doc-table] " + " ".join(cmd2))
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode != 0:
            print(r2.stderr, file=sys.stderr)
            return r2.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
