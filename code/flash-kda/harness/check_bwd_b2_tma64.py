"""P7-50: close the boundary §52.6 left open -- `-DB2M_TMA` was hard-pinned to `BVK==32`
by the SWIZZLE_128B inner-box limit (128 B == 32 fp32), while §49.5's compile scan found
`BVK=64/NS=2` the *best* cp.async tiling (499.1 vs 517.3 us on `B8 T2048 H8`).  So the two
wins (TMA read-path fix + best tiling) could never be combined.

This file adds `BVK==64` TMA support via **SWIZZLE_NONE** (dense 256 B rows -- the same
choice P7-21 made for the 512 B `E^tot` row) and re-measures:

  Part A  `base64` vs `tma64` are **bit-identical** (same mma k-order), `base32` vs
          `base64` is bit-identical too (the k8-block accumulation order is unchanged);
          all land the tf32 noise floor vs torch(fp32) and the identical-op Triton core.
  Part B  kernel-only `dev` (CUPTI) + `wall` for {base32,tma32} x {base64,tma64}
          (+ vecst), i.e. does TMA-on-BVK=64 beat both TMA-on-32 and cp.async-on-64?

The `ncu` byte/duration side lives in `harness/probe_b2_tma64_ncu.sh`
(`harness/probe_b2_tma64_ncu.out.txt`).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_tma64.py \
        > .../harness/check_bwd_b2_tma64.out.txt 2>&1
    B2T64_MODE=a | b
"""
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs      # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                             # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev              # noqa: E402
from check_bwd_b2_core import build_core_triton, run_triton_core, ref_core  # noqa: E402
from check_bwd_b2_tma import build, load, run_cuda, CASES, BT          # noqa: E402

# BVK=32 keeps the SWIZZLE_128B path; BVK=64 exercises the new SWIZZLE_NONE path.
VARIANTS = [
    ("base32", []),
    ("tma32", ["-DB2M_TMA=1"]),
    ("base64", ["-DB2M_BVK=64"]),
    ("tma64", ["-DB2M_TMA=1", "-DB2M_BVK=64"]),
    ("base64v", ["-DB2M_BVK=64", "-DB2M_VECST=1"]),
    ("tma64v", ["-DB2M_TMA=1", "-DB2M_BVK=64", "-DB2M_VECST=1"]),
]


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def part_a(libs):
    print("=" * 96)
    print("Part A: correctness -- tma64 vs base64 BIT-IDENTICAL; base32==base64; tf32 noise floor")
    print("=" * 96)
    tk = build_core_triton()
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        ref = ref_core(inp, BT)
        out = {tag: run_cuda(lib, inp) for tag, lib in libs.items()}
        tri = run_triton_core(tk, inp, BT, num_warps=8)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})  nvt(32)={128//32} nvt(64)={128//64}")
        for fld in ("dq", "dk", "dw", "dgk"):
            d_b64t = (out["base64"][fld].float() - out["tma64"][fld].float()).abs().max().item()
            d_32_64 = (out["base32"][fld].float() - out["base64"][fld].float()).abs().max().item()
            d_t32_t64 = (out["tma32"][fld].float() - out["tma64"][fld].float()).abs().max().item()
            _, mr64 = stats(f"  tma64 vs torch {fld}", out["tma64"][fld], ref[fld])
            _, mr_t = stats(f"  trit   vs torch {fld}", tri[fld], ref[fld])
            print(f"  {fld:4s}  base64-vs-tma64={d_b64t:.3e}  base32-vs-base64={d_32_64:.3e}  "
                  f"tma32-vs-tma64={d_t32_t64:.3e}  (vs torch: tma64 {mr64:.3e} / trit {mr_t:.3e})")
            worst[f"ident64_{fld}"] = max(worst.get(f"ident64_{fld}", 0.0), d_b64t)
            worst[f"bk_{fld}"] = max(worst.get(f"bk_{fld}", 0.0), d_32_64)
            worst[f"torch64_{fld}"] = max(worst.get(f"torch64_{fld}", 0.0), mr64)
    print("\nworst:")
    for fld in ("dq", "dk", "dw", "dgk"):
        print(f"  {fld:4s}  base64-vs-tma64 {worst['ident64_'+fld]:.3e}   "
              f"base32-vs-base64 {worst['bk_'+fld]:.3e}   "
              f"tma64-vs-torch {worst['torch64_'+fld]:.3e}")


def part_b(libs):
    print("\n" + "=" * 96)
    print("Part B: head-to-head kernel-only -- BVK=32 vs 64, cp.async vs TMA")
    print("=" * 96)
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        for lib in libs.values():
            run_cuda(lib, inp)
        torch.cuda.synchronize()
        dev, wall = {}, {}
        for tag, lib in libs.items():
            dev[tag], _ = profile_dev(lambda: run_cuda(lib, inp), reps=40)
            wall[tag] = time_wall(lambda: run_cuda(lib, inp), reps=60)
        print(f"\n  {name}:")
        for tag in libs:
            print(f"    {tag:8s} dev={dev[tag]:9.1f}us  wall={wall[tag]:9.1f}us   "
                  f"dev/base32={dev[tag]/dev['base32']:.3f}x  dev/base64={dev[tag]/dev['base64']:.3f}x")
        print(f"    -> tma64/base64 dev {dev['tma64']/dev['base64']:.3f}x  "
              f"base64/base32 {dev['base64']/dev['base32']:.3f}x  "
              f"tma64/base32 {dev['tma64']/dev['base32']:.3f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    mode = os.environ.get("B2T64_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs)
    print("\nDONE")


if __name__ == "__main__":
    main()
