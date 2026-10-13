"""P7-72: close §54.5's two open points at once --
  (1) `BVK=128` (NAT=4 128 B swizzle atoms) was never tested;
  (2) `atom64`'s L2 bytes (2.39 GB) sit 0.04 GB above `tma32` (2.35 GB) and that delta was
      never attributed (was it the extra atom-boundary L2 re-touch?).

Design (the clean controlled axis is **NAT at fixed NSTAGE**):
  * BVK=128 forces NSTAGE=1 (`sizeof(Shared)` for one atom-major stage = NAT*(3*BT+2*KD)*32*4
    = 4*304*128 B = 155.6 KB; two stages = 311 KB > the 227 KB opt-in dynamic-smem cap, so the
    launcher's `cudaFuncSetAttribute` fails -> rc=-3).  Hence to isolate the atom count from the
    pipeline depth we compare TMA atom32/atom64/atom128 **all at NSTAGE=1** (NAT=1/2/4), and
    separately reproduce the NSTAGE=2 baselines base32/tma32/atom64 from §54.
  * cp.async side: base32/base64/base128, all NSTAGE=1 (base also grows with BVK since it pads
    the whole row `BVP=BVK+4`).

Part A  correctness: `atom128` vs `base128` must be **bit-identical** (same mma k-order);
        `base128` vs `base64` (k8 accumulation order unchanged); all land the tf32 noise floor
        vs torch(fp32) (incl. non-divisible T=250).
Part B  kernel-only `dev` (CUPTI) + `wall`.

The `ncu` half (L2 bytes / srcunit read / bank conflicts / duration, NAT sweep at NSTAGE=1) is
`harness/probe_b2_tma_atom128_ncu.sh` (`..._ncu.out.txt`).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_tma_atom128.py \
        > .../harness/check_bwd_b2_tma_atom128.out.txt 2>&1
    B2TA128_MODE=a | b
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

# (tag, flags).  `ns1-*` variants pin NSTAGE=1 so the NAT sweep is pipeline-depth-controlled.
VARIANTS = [
    ("base32",        ["-DB2M_NS=2"]),
    ("tma32",         ["-DB2M_TMA=1", "-DB2M_NS=2"]),
    ("atom64",        ["-DB2M_TMA=1", "-DB2M_BVK=64", "-DB2M_NS=2"]),
    ("base32ns1",     ["-DB2M_NS=1"]),
    ("base64ns1",     ["-DB2M_BVK=64", "-DB2M_NS=1"]),
    ("base128ns1",    ["-DB2M_BVK=128", "-DB2M_NS=1"]),
    ("tma32ns1",      ["-DB2M_TMA=1", "-DB2M_NS=1"]),
    ("atom64ns1",     ["-DB2M_TMA=1", "-DB2M_BVK=64", "-DB2M_NS=1"]),
    ("atom128ns1",    ["-DB2M_TMA=1", "-DB2M_BVK=128", "-DB2M_NS=1"]),
    ("atom128ns1v",   ["-DB2M_TMA=1", "-DB2M_BVK=128", "-DB2M_NS=1", "-DB2M_VECST=1"]),
]


def build(tag, extra):
    so = f"/tmp/b2_mma_atom128_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + ["-lcuda", os.path.join(HARNESS, "b2_mma_probe.cu"),
           "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
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
    print("=" * 100)
    print("Part A: correctness -- atom128 vs base128 BIT-IDENTICAL; base128==base64 (k-order); tf32 floor")
    print("=" * 100)
    tk = build_core_triton()
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        ref = ref_core(inp, BT)
        out = {tag: run_cuda(lib, inp) for tag, lib in libs.items()}
        tri = run_triton_core(tk, inp, BT, num_warps=8)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})  NAT: 32->1 64->2 128->4")
        for fld in ("dq", "dk", "dw", "dgk"):
            d_a128_b128 = (out["atom128ns1"][fld].float() - out["base128ns1"][fld].float()).abs().max().item()
            d_a128_a64 = (out["atom128ns1"][fld].float() - out["atom64ns1"][fld].float()).abs().max().item()
            d_b128_b64 = (out["base128ns1"][fld].float() - out["base64ns1"][fld].float()).abs().max().item()
            _, mr128 = stats(f"  atom128 vs torch {fld}", out["atom128ns1"][fld], ref[fld])
            _, mr_t = stats(f"  trit vs torch {fld}", tri[fld], ref[fld])
            print(f"  {fld:4s}  atom128-vs-base128={d_a128_b128:.3e}  atom128-vs-atom64={d_a128_a64:.3e}  "
                  f"base128-vs-base64={d_b128_b64:.3e}  (vs torch: atom128 {mr128:.3e} / trit {mr_t:.3e})")
            worst[f"ident_{fld}"] = max(worst.get(f"ident_{fld}", 0.0), d_a128_b128)
            worst[f"torch_{fld}"] = max(worst.get(f"torch_{fld}", 0.0), mr128)
    print("\nworst (over all shapes):")
    for fld in ("dq", "dk", "dw", "dgk"):
        print(f"  {fld:4s}  atom128-vs-base128 {worst['ident_'+fld]:.3e}   "
              f"atom128-vs-torch {worst['torch_'+fld]:.3e}")


def part_b(libs):
    print("\n" + "=" * 100)
    print("Part B: head-to-head kernel-only -- BVK 32/64/128 x cp.async/TMA(atom); NSTAGE pinned")
    print("=" * 100)
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        # validate every variant launches (NS=2 + BVK=128 must be absent; it would rc=-3)
        for tag, lib in libs.items():
            run_cuda(lib, inp)
        torch.cuda.synchronize()
        dev, wall = {}, {}
        for tag, lib in libs.items():
            dev[tag], _ = profile_dev(lambda: run_cuda(lib, inp), reps=40)
            wall[tag] = time_wall(lambda: run_cuda(lib, inp), reps=60)
        print(f"\n  {name}:")
        for tag in libs:
            print(f"    {tag:14s} dev={dev[tag]:9.1f}us  wall={wall[tag]:9.1f}us   "
                  f"dev/base32={dev[tag]/dev['base32']:.3f}x")
        print("    NS=1 NAT sweep (atom32/atom64/atom128): " +
              "  ".join(f"{t}={dev[t]:.1f}" for t in ("tma32ns1", "atom64ns1", "atom128ns1")))
        print("    NS=1 cp.async sweep (base32/64/128):    " +
              "  ".join(f"{t}={dev[t]:.1f}" for t in ("base32ns1", "base64ns1", "base128ns1")))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    mode = os.environ.get("B2TA128_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs)
    print("\nDONE")


if __name__ == "__main__":
    main()
