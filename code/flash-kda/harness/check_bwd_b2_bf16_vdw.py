"""P7-80: close the open item left by §79 (P7-76) -- the bf16 precision of the **raw `v` / `dw`**
scratch operands was never *separately* ablated.

P7-73 gave the base B2 K2-mma GEMM core a bf16-native operand entry (`-DB2M_BF16`); P7-75 lifted it
to the six-segment production launch, but kept the two **scratch / raw** operands (`vv` = raw `v`,
`dwm` = `dw`) fp32 in the non-TMA `cp.async` path -- only the TMA path (§79) already staged them
bf16.  The §79 boundary note says verbatim: "bf16+TMA 下 `vv`/`dwm` 被降为 bf16（生产核组合时
raw `v`/`dw` 的 bf16 精度未单独消融）".

This round adds the missing knob `-DB2M_BF16_VDW` (a *layout / staging-dtype* change only: non-TMA
`vv`/`dwm` become `op_t`/`OPS` bf16 tiles, `prefetch` moves them at the bf16 `cp.async`
granularity, `rds()` widens bf16->fp32 exactly as `rd()`), and uses it to run the ablation.

  Part A  sanity: on bf16-valued `v`/`dw`, the bf16-staged build is **bit-identical** to the
          fp32-staged build (bf16->fp32 widening is exact) -- the flag is arithmetically inert.
  Part B  precision ablation: feed full-precision `v`/`dw` to the fp32-staged baseline vs the
          bf16-staged build; report per-output max_abs/max_rel and isolate `v` only / `dw` only.
  Part C  resources: regs / spill / smem (the smem drop from fp32->bf16 scratch tiles).
  Part D  timing: kernel-only `dev`, one production launch.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_bf16_vdw.py \
        > .../harness/check_bwd_b2_bf16_vdw.out.txt 2>&1
    B2VDW_MODE=a | b | c | d
"""
import ctypes
import os
import re
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs    # noqa: E402
from check_bwd_ws_reuse import cdiv                                   # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                         # noqa: E402
from check_bwd_b2_prod import run_core, run_prod, ptr, stream         # noqa: E402

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 1),
    ("B2 T1024 H8", 2, 1024, 8, 1),
    ("T512 H4", 1, 512, 4, 1),
    ("T250 H3 (non-div)", 1, 250, 3, 1),
]
BT = 16
CU = os.path.join(HARNESS, "b2_mma_probe.cu")

# the five streamed operands that P7-75 already stages bf16
BF16_KEYS = ("v_new", "h", "dh", "do", "dv")

BUILDS = {
    "core":          [],   # plain GEMM core (no -DB2M_PROD): emits `dw`
    "prod":          [],
    "prod_bf16":     ["-DB2M_BF16=1"],
    "prod_bf16_vdw": ["-DB2M_BF16=1", "-DB2M_BF16_VDW=1"],
    "prod_bf16_vdw_mma": ["-DB2M_BF16=1", "-DB2M_BF16_VDW=1", "-DB2M_BF16MMA=1"],
}

# per-output operand provenance for the ablation reading:
#   vv (raw v) -> dApre/dA (SAND), db (DB), and the gate chain whose raw dq/dk feed GATE
#   dwm (dw)   -> dgk, dkgb (DKGB) and the gate chain
ALL_OUT = ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dw", "dgk")


def build_v(tag, extra):
    so = f"/tmp/b2_vdw_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v"]
    if tag != "core":
        cmd += ["-DB2M_PROD=1"]
    cmd += extra + ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    return so, r.stderr


def build(tag, extra):
    return build_v(tag, extra)[0]


def load(so):
    return ctypes.CDLL(so)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def five_bf16(inp):
    """bf16-valued fp32 copies of the five streamed operands (v / dw untouched)."""
    out = dict(inp)
    for k in BF16_KEYS:
        out[k] = inp[k].bfloat16().float().contiguous()
    return out


def rel(a, b):
    d = (a.float() - b.float()).abs()
    den = b.float().abs().clamp_min(1e-12)
    return (d / den).max().item()


def report(name, a, b, refmag=None):
    d = (a.float() - b.float()).abs().max().item()
    mag = b.float().abs().max().item() if refmag is None else refmag
    # scale-relative error: max_abs / |ref|max (robust to near-zero cancellation entries)
    sr = d / max(mag, 1e-12)
    print(f"    {name:22s} max_abs={d:.3e}  max_abs/|ref|={sr:.3e}  (|ref|max={mag:.3e})")
    return d, sr


# ------------------------------------------------------------------------- parts
def part_a(cases):
    print("=" * 80)
    print("Part A: on bf16-valued v/dw, prod_bf16_vdw == prod_bf16 (bit-identity)")
    print("=" * 80)
    libs = {t: load(build(t, e)) for t, e in BUILDS.items()}
    worst = 0.0
    for (name, B, T, H, seed) in cases:
        print(f"\n# {name}")
        inp = make(B, T, H, seed)
        q = five_bf16(inp)
        dw_f = run_core(libs["core"], q)["dw"]
        # baseline: bf16-valued fp32 wire for v/dw (fp32 staging)
        qb = dict(q)
        qb["v"] = q["v"].bfloat16().float().contiguous()
        dwb_f = dw_f.bfloat16().float().contiguous()
        # vdw: bf16 wire for v/dw
        vb = q["v"].bfloat16().contiguous()
        dwb = dw_f.bfloat16().contiguous()
        pa = run_prod(libs["prod_bf16"], qb, dwb_f)
        qv = dict(q)
        qv["v"] = vb
        pb = run_prod(libs["prod_bf16_vdw"], qv, dwb)
        for k in ALL_OUT:
            d, _ = report(f"{k}", pb[k], pa[k])
            worst = max(worst, d)
    print(f"\n  worst over all outputs: max_abs={worst:.3e}  "
          f"[{'BIT-IDENTICAL' if worst == 0.0 else 'MISMATCH'}]")


def part_b(cases):
    print("=" * 80)
    print("Part B: precision cost of bf16 raw v/dw -- fp32-staged baseline vs bf16-staged build")
    print("=" * 80)
    libs = {t: load(build(t, e)) for t, e in BUILDS.items()}
    for (name, B, T, H, seed) in cases:
        print(f"\n# {name}  (baseline = fp32-staged, full-precision v/dw)")
        inp = make(B, T, H, seed)
        q = five_bf16(inp)
        dw_f = run_core(libs["core"], q)["dw"]
        # reference: full-precision v/dw, fp32 staging
        ref = run_prod(libs["prod_bf16"], q, dw_f)
        # (1) only v quantized to bf16 (fp32 staging)
        qv = dict(q); qv["v"] = q["v"].bfloat16().float().contiguous()
        rv = run_prod(libs["prod_bf16"], qv, dw_f)
        # (2) only dw quantized to bf16 (fp32 staging)
        dwb_f = dw_f.bfloat16().float().contiguous()
        rw = run_prod(libs["prod_bf16"], q, dwb_f)
        # (3) both quantized, bf16 staging (the new build)
        qvb = dict(q); qvb["v"] = q["v"].bfloat16().contiguous()
        dwb = dw_f.bfloat16().contiguous()
        rb = run_prod(libs["prod_bf16_vdw"], qvb, dwb)
        print("  -- v only (fp32 staging) vs ref --")
        for k in ALL_OUT:
            report(f"v.{k}", rv[k], ref[k])
        print("  -- dw only (fp32 staging) vs ref --")
        for k in ALL_OUT:
            report(f"dw.{k}", rw[k], ref[k])
        print("  -- both bf16-staged (new build) vs ref --")
        for k in ALL_OUT:
            report(f"vdw.{k}", rb[k], ref[k])
        # the bit-identity sanity between (v,dw) both-quantized fp32 staging and bf16 staging
        both_fp32 = dict(q)
        both_fp32["v"] = q["v"].bfloat16().float().contiguous()
        rbf = run_prod(libs["prod_bf16"], both_fp32, dwb_f)
        print("  -- both-quantized fp32 staging vs bf16 staging (must be bit-identical) --")
        worst = 0.0
        for k in ALL_OUT:
            d, _ = report(f"cmp.{k}", rb[k], rbf[k])
            worst = max(worst, d)
        print(f"    worst={worst:.3e} [{'OK' if worst == 0.0 else 'DIFF'}]")


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1,
            int(spill.group(1)) if spill else 0)


def part_c():
    print("=" * 80)
    print("Part C: resources (registers / spill / smem)")
    print("=" * 80)
    print(f"  {'build':20s} {'regs':>6s} {'spill':>7s} {'smem':>9s}")
    for tag in BUILDS:
        so, txt = build_v(tag, BUILDS[tag])
        r, sp = ptxas_info(txt)
        lib = load(so)
        lib.b2mma_smem_bytes.restype = ctypes.c_int
        smem = lib.b2mma_smem_bytes()
        print(f"  {tag:20s} {r:6d} {sp:7d} {smem:9d}")


def part_d(cases):
    print("=" * 80)
    print("Part D: timing -- kernel-only dev (one production launch)")
    print("=" * 80)
    libs = {t: load(build(t, e)) for t, e in BUILDS.items()}
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        q = five_bf16(inp)
        dw_f = run_core(libs["core"], q)["dw"]
        dwb_f = dw_f.bfloat16().float().contiguous()
        dwb = dw_f.bfloat16().contiguous()
        qb = dict(q); qb["v"] = q["v"].bfloat16().float().contiguous()
        qvb = dict(q); qvb["v"] = q["v"].bfloat16().contiguous()
        torch.cuda.synchronize()

        def f(tag, d, dwv):
            run_prod(libs[tag], d, dwv)

        dev_bf, _ = profile_dev(lambda: f("prod_bf16", qb, dwb_f))
        dev_vdw, _ = profile_dev(lambda: f("prod_bf16_vdw", qvb, dwb))
        dev_mma, _ = profile_dev(lambda: f("prod_bf16_vdw_mma", qvb, dwb))
        print(f"  {name:20s} bf16={dev_bf:8.1f}us  vdw={dev_vdw:8.1f}us ({dev_vdw/dev_bf:.3f}x)"
              f"  vdw_mma={dev_mma:8.1f}us ({dev_mma/dev_bf:.3f}x)")


def main():
    mode = os.environ.get("B2VDW_MODE", "a")
    if mode == "a":
        part_a(CASES)
    elif mode == "b":
        part_b(CASES)
    elif mode == "c":
        part_c()
    elif mode == "d":
        part_d(CASES)


if __name__ == "__main__":
    main()
