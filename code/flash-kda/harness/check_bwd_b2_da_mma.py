"""P7-66: `mma.sync`-ize the P7-54 `dApre` segment of B2 (`-DB2M_DAMMA`).

docs/07 §68.4 pinned the DA segment's `184 us` (30% of the bare GEMM core, the single largest
non-GEMM segment) to **scalar shared-memory math**: it doubles the smem bank conflict
(9.4M -> 20.8M) while leaving `tensor%` at 1.6%.  §68.5 named "mma-ize DA" the only remaining
magnitude lever (the other five segments are already mma / pure cross-warp reductions whose cost
is deliverable IO).  `dApre` is `X @ Y^T` twice (base `dv @ v^T`, then `- dw @ kg^T`), exactly the
`dq = do @ h^T` shape the GEMM core already does -- so each warp can compute its whole [BT,BT]
partial with `mma.sync.m16n8k8.tf32` (2 n-tiles x 2 k-steps per term) instead of the scalar
`BW x 8` FMA loops.

  Part A  correctness: CUDA `dApre` vs a true-fp32 torch reference on the *same* `dw` operand
          (from the CUDA GEMM core, hence also tf32); mma vs the P7-54 scalar DA (should differ
          only by the tf32 floor); and bit-identity of mma across non-TMA/TMA32/TMA64 layouts.
  Part B  timing: `da` (scalar) vs `da_mma` (isolated delta over the GEMM-only core) and vs the
          full Triton B2.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_da_mma.py \
        > .../harness/check_bwd_b2_da_mma.out.txt 2>&1
    B2DAMMA_MODE=a | b
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

# the mma segment is tf32 (matches `tl.dot`); the reference is true fp32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import (make_inputs as dqkg_make_inputs,   # noqa: E402
                                   build_kernel as build_dqkg, wy_dqkg_triton)
from check_bwd_ws_reuse import stats, cdiv                            # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                          # noqa: E402
from check_bwd_b2_prod_levels import (run_prod, make as prod_make,     # noqa: E402
                                      build_v as prod_build)

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16
VARIANTS = [
    ("core", []),
    ("core_tma", ["-DB2M_TMA=1"]),
    ("da", ["-DB2M_DA=1"]),
    ("da_tma", ["-DB2M_DA=1", "-DB2M_TMA=1"]),
    ("da_mma", ["-DB2M_DA=1", "-DB2M_DAMMA=1"]),
    ("da_mma_tma", ["-DB2M_DA=1", "-DB2M_DAMMA=1", "-DB2M_TMA=1"]),
    ("da_mma_tma64", ["-DB2M_DA=1", "-DB2M_DAMMA=1", "-DB2M_TMA=1", "-DB2M_BVK=64"]),
]
PROD_SCALAR = ["-DB2M_PROD=1", "-DB2M_HDW=1"]
PROD_MMA = PROD_SCALAR + ["-DB2M_DAMMA=1"]



def build(tag, extra):
    so = f"/tmp/b2_damma_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + ["-lcuda", os.path.join(HARNESS, "b2_mma_probe.cu"),
           "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed")
    return so


def load(so):
    return ctypes.CDLL(so)


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_core(lib, inp):
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2mma_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                          ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
                          B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2mma_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw, dgk=dgk)


def run_da(lib, inp):
    lib.b2da_launch.restype = ctypes.c_int
    lib.b2da_launch.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw_out = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dApre = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2da_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                         ptr(inp["do"]), ptr(inp["dv"]), ptr(inp["k"]), ptr(inp["g"]),
                         ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk), ptr(dApre),
                         B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2da_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw_out, dgk=dgk, dApre=dApre)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def ref_dapre(inp, dw, BT):
    """torch(fp32) reference for the `dApre` segment with the *given* `dw` operand (see P7-54)."""
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dv, vv, k, g = inp["dv"].float(), inp["v"].float(), inp["k"].float(), inp["g"].float()
    dwd = dw.float()
    kg = k * torch.exp2(g)

    def pad4(x):   # [B,T,H,W] -> [B,NT,H,BT,W]
        o = torch.zeros((B, NT * BT, H) + x.shape[3:], device=dev, dtype=torch.float32)
        o[:, :T] = x
        return o.view((B, NT, BT, H) + x.shape[3:]).permute(0, 1, 3, 2, 4).contiguous()

    dv4, v4 = pad4(dv), pad4(vv)          # [B,NT,H,BT,V]
    dw4, kg4 = pad4(dwd), pad4(kg)        # [B,NT,H,BT,K]
    base = torch.matmul(dv4, v4.transpose(-1, -2))     # [B,NT,H,BT(o_t),BT(o_u)]
    dwkg = torch.matmul(dw4, kg4.transpose(-1, -2))    # [B,NT,H,BT(o_t),BT(o_u)]
    dapre = (base - dwkg).permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    return dict(dApre=dapre)


def part_a(libs):
    print("=" * 96)
    print("Part A: correctness -- CUDA dApre vs torch(fp32) [same dw], mma vs scalar, layouts")
    print("=" * 96)
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        dw = run_core(libs["core"], inp)["dw"]          # tf32 dw operand (held fixed)
        ref = ref_dapre(inp, dw, BT)
        out = {tag: run_da(libs[tag], inp)
               for tag in ("da", "da_mma", "da_mma_tma", "da_mma_tma64")}
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for tag in ("da", "da_mma"):
            _, mr = stats(f"  {tag} dApre vs torch", out[tag]["dApre"], ref["dApre"])
            worst[f"{tag}_vs_torch"] = max(worst.get(f"{tag}_vs_torch", 0.0), mr)
        _, mr = stats("  da_mma vs da(scalar)", out["da_mma"]["dApre"], out["da"]["dApre"])
        worst["mma_vs_scalar"] = max(worst.get("mma_vs_scalar", 0.0), mr)
        for tag in ("da_mma_tma", "da_mma_tma64"):
            _, mrid = stats(f"  da_mma vs {tag:12s}", out["da_mma"]["dApre"], out[tag]["dApre"])
            worst[f"mma_vs_{tag}"] = max(worst.get(f"mma_vs_{tag}", 0.0), mrid)
    print("\nworst max_rel over all shapes:")
    for kk in ("da_vs_torch", "da_mma_vs_torch", "mma_vs_scalar",
               "mma_vs_da_mma_tma", "mma_vs_da_mma_tma64"):
        print(f"  {kk:24s} {worst.get(kk, 0.0):.3e}")


def part_b(libs, tk):
    print("\n" + "=" * 96)
    print("Part B: timing -- scalar DA vs mma DA (isolated delta vs GEMM-only core, and full B2)")
    print("=" * 96)
    iscore = lambda tag: tag.startswith("core")
    print(f"  {'case':18s} | {'core':>7s} {'core_tma':>8s} | {'da':>7s} {'da_tma':>7s} | "
          f"{'da_mma':>7s} {'dm_tma':>7s} {'dm_tma64':>8s} | {'fullB2':>7s} | "
          f"{'mma/sc':>6s} {'mma_tma/sc_tma':>14s}")
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        for tag in libs:
            if iscore(tag):
                run_core(libs[tag], inp)
            else:
                run_da(libs[tag], inp)
        torch.cuda.synchronize()
        dev = {}
        for tag in libs:
            if iscore(tag):
                dev[tag], _ = profile_dev(lambda lib=libs[tag]: run_core(lib, inp), reps=40)
            else:
                dev[tag], _ = profile_dev(lambda lib=libs[tag]: run_da(lib, inp), reps=40)
        fn2 = lambda: wy_dqkg_triton(tk, inp, 1.0, BT, num_warps=4)
        dev["fullB2"], _ = profile_dev(fn2, reps=40)
        print(f"  {name:18s} | {dev['core']:7.1f} {dev['core_tma']:8.1f} | {dev['da']:7.1f} "
              f"{dev['da_tma']:7.1f} | "
              f"{dev['da_mma']:7.1f} {dev['da_mma_tma']:7.1f} {dev['da_mma_tma64']:8.1f} | "
              f"{dev['fullB2']:7.1f} | "
              f"{dev['da_mma']/dev['da']:6.2f}x {dev['da_mma_tma']/dev['da_tma']:13.2f}x")


def part_c():
    print("\n" + "=" * 96)
    print("Part C: the *production* kernel (`-DB2M_PROD -DB2M_HDW`) scalar DA vs mma DA")
    print("=" * 96)
    # NB: under `B2M_GATE` the *raw* dq/dk are not stored (production form) -- the gate outputs are
    # `dqg`/`dkg`.  The raw `dq`/`dk` buffers are uninitialised and must NOT be compared.
    lsc = load(prod_build("prod_sc", PROD_SCALAR)[0])
    lmma = load(prod_build("prod_mma", PROD_MMA)[0])
    # production-regime DA marginal: L3 = OUT+DKGB+DA vs L2 = OUT+DKGB (the §68 method).
    lsc2 = load(prod_build("prod_sc_l2", PROD_SCALAR + ["-DB2M_LEVEL=2"])[0])
    lsc3 = load(prod_build("prod_sc_l3", PROD_SCALAR + ["-DB2M_LEVEL=3"])[0])
    lmm2 = load(prod_build("prod_mm_l2", PROD_MMA + ["-DB2M_LEVEL=2"])[0])
    lmm3 = load(prod_build("prod_mm_l3", PROD_MMA + ["-DB2M_LEVEL=3"])[0])
    OUTS = ("dv2", "dvb", "dkgb", "dqg", "dkg", "dg", "db")
    print(f"  {'case':18s} | {'prod(sc)':>9s} {'prod(mma)':>10s} | {'mma/sc':>7s} | "
          f"{'dApre rel':>10s} {'dA rel':>8s} | non-DA")
    for (name, B, T, H, seed) in CASES:
        inp = prod_make(B, T, H, seed)
        rs = run_prod(lsc, inp)
        rm = run_prod(lmma, inp)
        ident = all((rs[f] - rm[f]).abs().max().item() == 0.0 for f in OUTS)
        def rel(a, b):
            d = (a.float() - b.float()).abs().max().item()
            return d / (b.float().abs().max().item() + 1e-12)
        rel_ap, rel_a = rel(rm["dApre"], rs["dApre"]), rel(rm["dA"], rs["dA"])
        torch.cuda.synchronize()
        dsc, _ = profile_dev(lambda: run_prod(lsc, inp))
        dmm, _ = profile_dev(lambda: run_prod(lmma, inp))
        print(f"  {name:18s} | {dsc:9.1f} {dmm:10.1f} | {dmm/dsc:6.2f}x | "
              f"{rel_ap:10.3e} {rel_a:8.3e} | {'bit-exact' if ident else 'DIFF'}")
    print("\n  production-regime DA marginal (dev us; L3=OUT+DKGB+DA minus L2=OUT+DKGB), §68 method:")
    print(f"  {'case':18s} | {'sc L2':>7s} {'sc L3':>7s} {'DA sc':>7s} | "
          f"{'mma L2':>7s} {'mma L3':>7s} {'DA mma':>7s} | {'DA mma/sc':>9s}")
    for (name, B, T, H, seed) in CASES:
        inp = prod_make(B, T, H, seed)
        torch.cuda.synchronize()
        f2 = {}
        for tag, lib in (("sc2", lsc2), ("sc3", lsc3), ("mm2", lmm2), ("mm3", lmm3)):
            f2[tag], _ = profile_dev(lambda lb=lib: run_prod(lb, inp))
        da_sc = f2["sc3"] - f2["sc2"]
        da_mm = f2["mm3"] - f2["mm2"]
        print(f"  {name:18s} | {f2['sc2']:7.1f} {f2['sc3']:7.1f} {da_sc:7.1f} | "
              f"{f2['mm2']:7.1f} {f2['mm3']:7.1f} {da_mm:7.1f} | "
              f"{da_mm/da_sc if da_sc else float('nan'):8.2f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    tk = build_dqkg()
    mode = os.environ.get("B2DAMMA_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs, tk)
    if mode in ("", "c"):
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
