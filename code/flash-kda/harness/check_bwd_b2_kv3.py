"""P7-86 correctness: the K!=V **two-knob stacks** whose `ncu` is in docs/07 section 89 must first
be shown algebraically equivalent to their single-knob anchors (the pipeline profiled them, never
checked them at K!=V):

  (A) bf16tma (BF16 x TMA, BVK=64)  vs  bf1664 (BF16, cp.async, BVK=64)
      -> TMA only changes the staging layout, so on bf16-valued operands the 11 production
         outputs must be **bit-identical** (same reason as section 85.1 / section 78).
  (B) bf16mma_tma vs bf16tma  -> only the mma ISA differs: expect only fp32 accumulation order.
  (C) damma_tma vs tma        -> only the DA implementation differs (scalar FMA -> tf32 mma):
         the non-DA outputs bit-identical, dApre/dA at the tf32 floor.

    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_kv3.py \
        > .../harness/check_bwd_b2_kv3.out.txt 2>&1
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
from ncu_b2_kv2_driver import build_inputs, launch_reps, SHAPES  # noqa: E402
from check_bwd_b2_prod import run_prod  # noqa: E402

CU = os.path.join(HARNESS, "b2_mma_probe.cu")
BUILDS = {
    "bf1664":      ["-DB2M_BF16=1", "-DB2M_BVK=64"],
    "bf16tma":     ["-DB2M_BF16=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],
    "bf16mma_tma": ["-DB2M_BF16=1", "-DB2M_BF16MMA=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],
    "tma":         ["-DB2M_TMA=1"],
    "damma_tma":   ["-DB2M_DAMMA=1", "-DB2M_TMA=1"],
}
# outputs excluding the HDW-suppressed `dw` scratch (its buffer is never written)
OUT = ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk")


def build(tag, V):
    so = "/tmp/b2kv3c_%s_v%d.so" % (tag, V)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC", "-shared",
           "-lcuda", "-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_VD=%d" % V] + BUILDS[tag] + \
          [CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed %s V=%d" % (tag, V))
    return so


def worst(a, b, keys):
    return max((a[k].float() - b[k].float()).abs().max().item() for k in keys)


def worst_rel(a, b, keys):
    """max over outputs of max_abs / max|ref| (scale-relative; robust to large-magnitude outs)."""
    w = 0.0
    for k in keys:
        d = (a[k].float() - b[k].float()).abs().max().item()
        m = b[k].float().abs().max().item()
        w = max(w, d / max(m, 1e-12))
    return w


def to_prod_inp(d):
    """remap the driver's `vn` key to the `v_new` key `run_prod` expects (in place)."""
    d["v_new"] = d["vn"]
    return d


def main():
    print("=" * 80)
    print("P7-86: K!=V two-knob-stack correctness (shape B2 T1024 H8, K=128)")
    print("=" * 80)
    shape = "b2t1024h8"
    for V in (64, 128, 256):
        libs = {t: ctypes.CDLL(build(t, V)) for t in BUILDS}
        # --- (A) bf16tma vs bf1664, bf16-valued ops ---
        dt = build_inputs(shape, V, "bf16tma")          # five bf16, v bf16
        dc = build_inputs(shape, V, "bf16")             # five bf16, v fp32
        dc["v"] = dt["v"].float().contiguous()          # exact bf16->fp32 widening
        ra = run_prod(libs["bf16tma"], to_prod_inp(dt), dt["dw"])
        rc = run_prod(libs["bf1664"], to_prod_inp(dc), dc["dw"])
        wa = worst(ra, rc, OUT)
        # --- (B) bf16mma_tma vs bf16tma (same dtype tag) ---
        rb = run_prod(libs["bf16mma_tma"], to_prod_inp(dt), dt["dw"])
        wb = worst_rel(rb, ra, OUT)
        # --- (C) damma_tma vs tma (fp32 wires) ---
        dp = build_inputs(shape, V, "prod")
        rd = run_prod(libs["damma_tma"], to_prod_inp(dp), dp["dw"])
        rt = run_prod(libs["tma"], to_prod_inp(dp), dp["dw"])
        wc = worst_rel(rd, rt, OUT)
        print(f"\n# V={V}")
        print(f"  (A) bf16tma  vs bf1664      : max_abs={wa:.3e}  "
              f"[{'BIT-IDENTICAL' if wa == 0 else 'DIFF'}]")
        print(f"  (B) bf16mma_tma vs bf16tma  : max scale-rel={wb:.3e}  (fp32 accum-order)")
        for k in OUT:
            d = (rb[k].float() - ra[k].float()).abs().max().item()
            m = ra[k].float().abs().max().item()
            print(f"        {k:6s} (B): max_abs={d:.3e}  scale-rel={d/max(m,1e-12):.3e}")
        print(f"  (C) damma_tma vs tma        : max scale-rel={wc:.3e}")
        for k in ("dqg", "dv2", "db", "dApre", "dA"):
            d = (rd[k].float() - rt[k].float()).abs().max().item()
            m = rt[k].float().abs().max().item()
            print(f"        {k:6s} (C): max_abs={d:.3e}  scale-rel={d/max(m,1e-12):.3e}")


if __name__ == "__main__":
    main()
