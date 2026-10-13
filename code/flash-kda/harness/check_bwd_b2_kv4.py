"""P7-87 correctness: close the last K!=V boundary named in docs/07 section 89.7 -- the
`BF16_VDW x TMA` cross (and its three-knob extension `BF16_VDW x BF16MMA x TMA`).

Claim (source-level): under `-DB2M_TMA` the `-DB2M_BF16_VDW` flag is a **compile-time no-op**, because
every one of its branches in `b2_mma_probe.cu` lives either in an `#if !B2M_TMA_ON` region
(the non-TMA `Stage` decls `:528/535`, the non-TMA `rds`/`rdsvv` `:638/651`, the non-TMA `prefetch`
`:795`) or is already reached under plain `-DB2M_BF16` (under TMA the `vv`/`dwm` tiles are `tma_t`
= bf16 anyway, `:515/518`; the `rds`/`rd` read path has no `BF16_VDW` branch under TMA, `:636`).
So the flag must not change a single value or a single byte of staging.

This round verifies that claim at K != V by bit-identity:
  (A) bf16_vdw_tma      vs bf16tma        (VDW under TMA; bit-identical, 11 outputs)
  (B) bf16_vdw_mma_tma  vs bf16mma_tma    (VDW stacked on native bf16 mma + TMA)
  (C) non-TMA anchor bf16_vdw vs bf16     (the knob's real effect, already anchored in section 83/86)

Operands are bf16-valued on the wire for every build (so bf16->fp32 widening is exact and any
difference would be a real layout bug, not a dtype artifact).

    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_kv4.py \
        > .../harness/check_bwd_b2_kv4.out.txt 2>&1
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)
from ncu_b2_kv2_driver import build_inputs  # noqa: E402
from check_bwd_b2_prod import run_prod       # noqa: E402

CU = os.path.join(HARNESS, "b2_mma_probe.cu")
# tag -> extra nvcc flags.  TMA bf16 builds pin BVK=64 (a bf16 128 B atom is 64 elements).
BUILDS = {
    "bf16":              ["-DB2M_BF16=1"],
    "bf16_vdw":          ["-DB2M_BF16=1", "-DB2M_BF16_VDW=1"],
    "bf16tma":           ["-DB2M_BF16=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],
    "bf16_vdw_tma":      ["-DB2M_BF16=1", "-DB2M_BF16_VDW=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],
    "bf16mma_tma":       ["-DB2M_BF16=1", "-DB2M_BF16MMA=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],
    "bf16_vdw_mma_tma":  ["-DB2M_BF16=1", "-DB2M_BF16_VDW=1", "-DB2M_BF16MMA=1",
                          "-DB2M_TMA=1", "-DB2M_BVK=64"],
}
OUT = ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk")


def build(tag, V):
    so = "/tmp/b2kv4c_%s_v%d.so" % (tag, V)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC", "-shared",
           "-lcuda", "-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_VD=%d" % V] + BUILDS[tag] + [CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed %s V=%d" % (tag, V))
    return so


def worst(a, b, keys):
    return max((a[k].float() - b[k].float()).abs().max().item() for k in keys)


def to_prod_inp(d):
    d["v_new"] = d["vn"]
    return d


def main():
    print("=" * 80)
    print("P7-87: K!=V closure of the `BF16_VDW x TMA` (and x BF16MMA) boundary (shape B2 T1024 H8)")
    print("=" * 80)
    shape = "b2t1024h8"
    for V in (64, 128, 256):
        libs = {t: ctypes.CDLL(build(t, V)) for t in BUILDS}
        # TMA builds: five bf16 + v bf16 (tag "bf16tma")
        dt = build_inputs(shape, V, "bf16tma")
        ra = run_prod(libs["bf16tma"], to_prod_inp(dict(dt)), dt["dw"])
        rav = run_prod(libs["bf16_vdw_tma"], to_prod_inp(dict(dt)), dt["dw"])
        rb = run_prod(libs["bf16mma_tma"], to_prod_inp(dict(dt)), dt["dw"])
        rbv = run_prod(libs["bf16_vdw_mma_tma"], to_prod_inp(dict(dt)), dt["dw"])
        wa = worst(rav, ra, OUT)
        wb = worst(rbv, rb, OUT)
        # non-TMA anchor (tag "vdw" -> v bf16); the plain `bf16` build stages v as fp32, so give it a
        # widened *copy* (never mutate the bf16 tensor the VDW build reads).
        dv = build_inputs(shape, V, "vdw")
        dc = {k: (v.float().contiguous() if k == "v" else v) for k, v in dv.items()}
        rn = run_prod(libs["bf16"], to_prod_inp(dict(dc)), dv["dw"])
        rnv = run_prod(libs["bf16_vdw"], to_prod_inp(dict(dv)), dv["dw"])
        wc = worst(rnv, rn, OUT)
        print(f"\n# V={V}")
        print(f"  (A) bf16_vdw_tma     vs bf16tma       : max_abs={wa:.3e}  "
              f"[{'BIT-IDENTICAL' if wa == 0 else 'DIFF'}]")
        print(f"  (B) bf16_vdw_mma_tma vs bf16mma_tma   : max_abs={wb:.3e}  "
              f"[{'BIT-IDENTICAL' if wb == 0 else 'DIFF'}]")
        print(f"  (C) bf16_vdw (non-TMA) vs bf16 anchor : max_abs={wc:.3e}  "
              f"[{'BIT-IDENTICAL' if wc == 0 else 'DIFF'}]")
        if wa or wb:
            for k in OUT:
                da = (rav[k].float() - ra[k].float()).abs().max().item()
                dbb = (rbv[k].float() - rb[k].float()).abs().max().item()
                print(f"        {k:6s} (A)={da:.3e}  (B)={dbb:.3e}")


if __name__ == "__main__":
    main()
