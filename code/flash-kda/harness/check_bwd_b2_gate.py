"""P7-57: the *sixth non-GEMM segment* of B2 in CUDA -- the **gate chain**, i.e. the final
`dq/dk/dg` from the raw GEMM outputs, the raw per-chunk `dgk=[sum_V h*dh]` (already computed)
and `dkgb=A@(-dw)` (P7-53).

FLA `chunk_bwd.py` / `check_bwd_dqkg_kernel.py:173-189`:
    gk_exp = exp2(g)                     # g = chunk-local log2 cumsum of the gate
    gn     = g[last]                     # last valid row of the chunk
    dq     = dq_raw * gk_exp * scale
    dk_mid = dk_raw * exp2(gn - g)
    kdk    = k * dk_mid                  # dk *before* the dkgb*gb term
    dgk    = dgk_raw*exp2(gn) + sum_t kdk
    dg     = q*dq - kdk + (t==last)*dgk + k*exp2(g)*dkgb*beta
    dk     = dk_mid + dkgb*(exp2(g)*beta)

Every mma fragment position of a warp's owned k-column slice is gate-scaled independently; the
only reduction is `sum_t kdk` over the chunk's rows (an intra-warp `shfl` over the mma groupID).
It is a *complete* B2 output triple (`dq`/`dk`/`dg`), so it is verifiable standalone.

  Part A  correctness: CUDA final `dq`/`dk`/`dg` vs the same torch gate applied to the *CUDA*
          raw `dq/dk/dgk` + `dkgb` (isolates the gate arithmetic), bit-identity across
          non-TMA/TMA32/TMA64, and cross-check vs FLA's final `dq`/`dk`/`dg` (tf32 floor).
  Part B  timing: gate kernel vs the same build with the gate segment skipped (`B2M_NODG`), and
          vs the full Triton B2.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_gate.py \
        > .../harness/check_bwd_b2_gate.out.txt 2>&1
    B2G_MODE=a | b
"""
import ctypes
import math
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

# CUDA dvb/dkgb are tf32; the gate arithmetic itself is fp32 scalar.  Keep torch high precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import (make_inputs as dqkg_make_inputs,   # noqa: E402
                                   build_kernel as build_dqkg, wy_dqkg_triton,
                                   fla_wy_dqkg)
from check_bwd_ws_reuse import stats, cdiv                            # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                          # noqa: E402

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16
SCALE = 1.0 / math.sqrt(128)
# `*_nodg` = same gate build with the gate segment skipped (timing baseline).
VARIANTS = [
    ("core",      []),
    ("gate",      ["-DB2M_DKGB=1", "-DB2M_GATE=1"]),
    ("gate_nodg", ["-DB2M_DKGB=1", "-DB2M_GATE=1", "-DB2M_NODG=1"]),
    ("gate_tma",  ["-DB2M_DKGB=1", "-DB2M_GATE=1", "-DB2M_TMA=1"]),
    ("gate_tma64", ["-DB2M_DKGB=1", "-DB2M_GATE=1", "-DB2M_TMA=1", "-DB2M_BVK=64"]),
]
GATE_TAGS = ("gate", "gate_tma", "gate_tma64")
TIMING_TAGS = ("gate_nodg", "gate", "gate_tma", "gate_tma64")


def build(tag, extra):
    so = f"/tmp/b2_gate_{tag}.so"
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


def run_gate(lib, inp, dw):
    lib.b2gate_launch.restype = ctypes.c_int
    lib.b2gate_launch.argtypes = ([ctypes.c_void_p] * 20 + [ctypes.c_float] +
                                  [ctypes.c_int] * 4 + [ctypes.c_void_p])
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw_out = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    dkgb = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dqg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dkg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b2gate_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                           ptr(inp["do"]), ptr(inp["dv"]), ptr(dw),
                           ptr(inp["A"]), ptr(inp["beta"]), ptr(inp["q"]),
                           ptr(inp["k"]), ptr(inp["g"]),
                           ptr(dq), ptr(dk), ptr(dw_out), ptr(dgk), ptr(dkgb),
                           ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                           B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b2gate_launch rc={rc}")
    return dict(dq=dq, dk=dk, dw=dw_out, dgk=dgk, dkgb=dkgb, dqg=dqg, dkg=dkg, dg=dg)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def ref_gate(inp, dq_raw, dk_raw, dgk_raw, dkgb):
    """torch(fp32) gate chain applied to the *given* CUDA raw `dq/dk/dgk` + `dkgb`."""
    q = inp["q"].float()
    k = inp["k"].float()
    g = inp["g"].float()
    beta = inp["beta"].float()
    B, T, H, K = q.shape
    NT = cdiv(T, BT)
    dev = q.device
    idx = torch.arange(T, device=dev)
    c = idx // BT
    last = torch.clamp((c + 1) * BT, max=T) - 1                 # [T] chunk's last valid row
    lastc = torch.clamp((torch.arange(NT, device=dev) + 1) * BT, max=T) - 1   # [NT]
    # gcn[b,c,h,k] = g[b, lastc[c], h, k] ;  gn[b,t,h,k] = g[b, last[t], h, k]
    gcn = g.gather(1, lastc.view(1, NT, 1, 1).expand(B, NT, H, K))    # [B,NT,H,K]
    gn = g.gather(1, last.view(1, T, 1, 1).expand(B, T, H, K))        # [B,T,H,K]
    gk = torch.exp2(g)
    dqg = dq_raw * gk * SCALE
    dkm = dk_raw * torch.exp2(gn - g)
    kdk = k * dkm
    kdkp = torch.zeros(B, NT * BT, H, K, device=dev, dtype=torch.float32)
    kdkp[:, :T] = kdk
    dgkt = dgk_raw * torch.exp2(gcn) + kdkp.view(B, NT, BT, H, K).sum(2)   # [B,NT,H,K]
    dgtok = dgkt.view(B, NT, 1, H, K).expand(B, NT, BT, H, K).reshape(B, NT * BT, H, K)[:, :T]
    is_last = (idx == last).view(1, T, 1, 1).float()
    dg = q * dqg - kdk + is_last * dgtok + k * gk * dkgb * beta.unsqueeze(-1)
    dkg = dkm + dkgb * (gk * beta.unsqueeze(-1))
    return dict(dqg=dqg, dkg=dkg, dg=dg)


def part_a(libs):
    print("=" * 96)
    print("Part A: correctness -- CUDA gate dq/dk/dg, bit-id across layouts, vs FLA (tf32 floor)")
    print("=" * 96)
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        core = run_core(libs["core"], inp)
        out = {tag: run_gate(libs[tag], inp, core["dw"]) for tag in GATE_TAGS}
        ref = ref_gate(inp, core["dq"], core["dk"], core["dgk"], out["gate"]["dkgb"])
        fla = fla_wy_dqkg(inp, SCALE, BT)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for fld in ("dqg", "dkg", "dg"):
            _, mr = stats(f"  {fld} vs torch(same raw)", out["gate"][fld], ref[fld])
            worst[f"{fld}_torch"] = max(worst.get(f"{fld}_torch", 0.0), mr)
        # final dq/dk/dg vs FLA
        _, mr = stats("  dq vs FLA", out["gate"]["dqg"], fla["dq"])
        worst["dq_fla"] = max(worst.get("dq_fla", 0.0), mr)
        _, mr = stats("  dk vs FLA", out["gate"]["dkg"], fla["dk"])
        worst["dk_fla"] = max(worst.get("dk_fla", 0.0), mr)
        _, mr = stats("  dg vs FLA", out["gate"]["dg"], fla["dg"])
        worst["dg_fla"] = max(worst.get("dg_fla", 0.0), mr)
        # bit-identity across stage layouts
        for tag in ("gate_tma", "gate_tma64"):
            for fld in ("dqg", "dkg", "dg"):
                _, mrid = stats(f"  {fld} vs {tag}", out["gate"][fld], out[tag][fld])
                worst[f"{fld}_{tag}"] = max(worst.get(f"{fld}_{tag}", 0.0), mrid)
    print("\nworst max_rel over all shapes:")
    for kk in ("dqg_torch", "dkg_torch", "dg_torch", "dq_fla", "dk_fla", "dg_fla",
               "dqg_gate_tma", "dkg_gate_tma", "dg_gate_tma",
               "dqg_gate_tma64", "dkg_gate_tma64", "dg_gate_tma64"):
        print(f"  {kk:18s} {worst.get(kk, 0.0):.3e}")


def part_b(libs, tk):
    print("\n" + "=" * 96)
    print("Part B: timing -- gate segment delta (vs B2M_NODG) and vs full Triton B2")
    print("=" * 96)
    print(f"  {'case':18s} | {'gate_nodg':>9s} {'gate':>8s} {'gate_tma':>9s} "
          f"{'gate_tma64':>10s} | {'fullB2':>8s} | {'gate/nodg':>9s} {'tma/tma_nodg':>12s}")
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        dw = run_core(libs["core"], inp)["dw"]
        for tag in TIMING_TAGS:
            run_gate(libs[tag], inp, dw)
        torch.cuda.synchronize()
        dev = {}
        for tag in TIMING_TAGS:
            dev[tag], _ = profile_dev(lambda lib=libs[tag]: run_gate(lib, inp, dw), reps=40)
        dev["fullB2"], _ = profile_dev(
            lambda: wy_dqkg_triton(tk, inp, SCALE, BT, num_warps=4), reps=40)
        # tma_nodg baseline would need a separate build; compare tma vs gate only
        print(f"  {name:18s} | {dev['gate_nodg']:9.1f} {dev['gate']:8.1f} "
              f"{dev['gate_tma']:9.1f} {dev['gate_tma64']:10.1f} | {dev['fullB2']:8.1f} | "
              f"{dev['gate']/dev['gate_nodg']:8.2f}x "
              f"{dev['gate_tma']/dev['gate_nodg']:11.2f}x")
    print("\n  note: gate_tma/gate_tma64 also include the TMA read-side win (P7-49/P7-51); the")
    print("  gate-segment delta is gate/gate_nodg (both same staging).")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    tk = build_dqkg()
    mode = os.environ.get("B2G_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs, tk)
    print("\nDONE")


if __name__ == "__main__":
    main()
