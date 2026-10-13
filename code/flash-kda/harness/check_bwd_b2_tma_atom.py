"""P7-51: close §53.5's *only* remaining path for `TMA + BVK=64` -- instead of handing the
whole 256 B row to `SWIZZLE_NONE` (P7-50: bit-correct but 2.6-2.8x slower, because dense
256 B stride = 64 words aliases every smem bank), split each operand tile into `NAT=BVK/32`
independent **128 B swizzle atoms** (each its own TMA box of inner width 32), so BVK=64 keeps
the bank-conflict-free swizzle *and* the best (§49.5) tiling.

This file is the correctness + timing half; the `ncu` byte/conflict/duration half lives in
`harness/probe_b2_tma_atom_ncu.sh` (`harness/probe_b2_tma_atom_ncu.out.txt`).

  Part A  `tma_atom64` vs `base64` must be **bit-identical** (same mma k-order); `base32 ==
          base64` for dq/dk/dw (k8 accumulation order unchanged); all land the tf32 noise
          floor vs torch(fp32) and the identical-op Triton core.
  Part B  kernel-only `dev` (CUPTI) + `wall` for {base32, tma32} x {base64, tma_atom64}
          (+vecst): does the atom split finally stack TMA@BVK=64 on top of the best tiling?

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_tma_atom.py \
        > .../harness/check_bwd_b2_tma_atom.out.txt 2>&1
    B2TA_MODE=a | b
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

# CUDA core is tf32 (matches `tl.dot`); true-fp32 torch reference.
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

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16
VARIANTS = [
    ("base32", []),
    ("tma32", ["-DB2M_TMA=1"]),
    ("base64", ["-DB2M_BVK=64"]),
    ("tma_atom64", ["-DB2M_TMA=1", "-DB2M_BVK=64"]),
    ("base64v", ["-DB2M_BVK=64", "-DB2M_VECST=1"]),
    ("tma_atom64v", ["-DB2M_TMA=1", "-DB2M_BVK=64", "-DB2M_VECST=1"]),
]


def build(tag, extra):
    so = f"/tmp/b2_mma_atom_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + ["-lcuda", os.path.join(HARNESS, "b2_mma_probe.cu"),
           "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed")
    return so


def load(so):
    lib = ctypes.CDLL(so)
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_cuda(lib, inp, BT_=BT):
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT_)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2mma_launch(
        ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
        ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
        B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2mma_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw, dgk=dgk)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def part_a(libs):
    print("=" * 96)
    print("Part A: correctness -- tma_atom64 vs base64 BIT-IDENTICAL; base32==base64; tf32 floor")
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
            d_a64_b64 = (out["tma_atom64"][fld].float() - out["base64"][fld].float()).abs().max().item()
            d_a64_t32 = (out["tma_atom64"][fld].float() - out["tma32"][fld].float()).abs().max().item()
            d_32_64 = (out["base32"][fld].float() - out["base64"][fld].float()).abs().max().item()
            _, mr64 = stats(f"  tma_atom64 vs torch {fld}", out["tma_atom64"][fld], ref[fld])
            _, mr_t = stats(f"  trit vs torch {fld}", tri[fld], ref[fld])
            print(f"  {fld:4s}  atom64-vs-base64={d_a64_b64:.3e}  atom64-vs-tma32={d_a64_t32:.3e}  "
                  f"base32-vs-base64={d_32_64:.3e}  (vs torch: atom64 {mr64:.3e} / trit {mr_t:.3e})")
            worst[f"ident_{fld}"] = max(worst.get(f"ident_{fld}", 0.0), d_a64_b64)
            worst[f"torch_{fld}"] = max(worst.get(f"torch_{fld}", 0.0), mr64)
    print("\nworst (over all shapes):")
    for fld in ("dq", "dk", "dw", "dgk"):
        print(f"  {fld:4s}  atom64-vs-base64 {worst['ident_'+fld]:.3e}   "
              f"atom64-vs-torch {worst['torch_'+fld]:.3e}")


def part_b(libs):
    print("\n" + "=" * 96)
    print("Part B: head-to-head kernel-only -- BVK=32/64 x cp.async/TMA(atom)")
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
            print(f"    {tag:12s} dev={dev[tag]:9.1f}us  wall={wall[tag]:9.1f}us   "
                  f"dev/base32={dev[tag]/dev['base32']:.3f}x  dev/base64={dev[tag]/dev['base64']:.3f}x")
        print(f"    -> atom64/base64 dev {dev['tma_atom64']/dev['base64']:.3f}x  "
              f"atom64/base32 {dev['tma_atom64']/dev['base32']:.3f}x  "
              f"tma32/base32 {dev['tma32']/dev['base32']:.3f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    mode = os.environ.get("B2TA_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs)
    print("\nDONE")


if __name__ == "__main__":
    main()
