"""P7-62: B3 `intra` in the **K2 shape** (CUDA) -- the last remaining structural item
named in ROADMAP / docs/07 §60.5 ("另一条「B3 intra 折进 K2 形状」仍待").

At the deployed CHUNK=16 granularity B3's grid collapses to `NC=1`, so it runs **only** the
diagonal Parts B/D (docs/07 §27.1).  §27 MMA-ized those in Triton (kernel 1.5-2.0x) and §30
fused B2+B3 in Triton; the CUDA production B2 (`b2_mma_probe.cu`, K2 shape) never got the
intra fold.  `b3_intra_probe.cu` implements the diagonal intra in K2 shape (`grid (NT, B*H)`,
8 warps, warp `w` owns the output k columns [w*16, w*16+16)), with `mma.sync.m16n8k8.tf32`
for the four `[BT,BT]@[BT,KD]` matmuls and elementwise gating in the accumulator fragments.

This harness bit-checks it against the Triton MMA B3 (`check_bwd_b3_mma.py::build_intra_mma`)
on the **same** `dAqk`/`dAkk`, and against the per-chunk torch golden (`intra_torch`), for
several shapes incl. a non-divisible `T` and `g=0`; then times it (CUPTI `dev` + event `wall`).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b3_intra_cuda.py \
        > .../harness/check_bwd_b3_intra_cuda.out.txt 2>&1
    B3IC_MODE=a | b | c
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

from check_bwd_intra_kernel import make_inputs, intra_torch, cdiv, stats   # noqa: E402
from check_bwd_b3_mma import build_intra_mma, intra_triton_mma             # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev                   # noqa: E402

SO = "/tmp/b3_intra.so"
BT = 16
BC = 16

# (name, B, T, H, seed) -- K = 128 (kernel hardcodes KD=128)
CASES = [
    ("T=512 H=4", 1, 512, 4, 0),
    ("B2 T=1024 H=8", 2, 1024, 8, 1),
    ("B8 T=2048 H=8", 8, 2048, 8, 2),
    ("T=250 H=3 (non-div)", 1, 250, 3, 3),
    ("T=64 H=1 (single chunk)", 1, 64, 1, 4),
]


def build():
    src = os.path.join(HARNESS, "b3_intra_probe.cu")
    extra = os.environ.get("B3IC_CFLAGS", "").split()
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + [src, "-o", SO]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed")
    if r.stderr.strip():
        print(r.stderr.strip())


def load():
    lib = ctypes.CDLL(SO)
    lib.b3intra_launch.restype = ctypes.c_int
    lib.b3intra_launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_void_p] * 4 + \
        [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.b3intra_smem_bytes.restype = ctypes.c_int
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_cuda(lib, inp):
    B, T, H, K = inp["q"].shape
    assert K == 128, "kernel compiled for K=128 (deployment)"
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    db = torch.empty(B, T, H, device=dev, dtype=torch.float32)
    dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.b3intra_launch(
        ptr(inp["q"]), ptr(inp["k"]), ptr(inp["g"]), ptr(inp["beta"]),
        ptr(inp["dAqk"]), ptr(inp["dAkk"]), ptr(dq), ptr(dk), ptr(db), ptr(dg),
        B, T, H, NT, stream)
    if rc != 0:
        raise RuntimeError(f"b3intra_launch rc={rc}")
    return dict(dq=dq, dk=dk, db=db, dg=dg)


def make(B, T, H, seed, gs=1.0):
    inp = make_inputs(B, T, H, 128, BT, BC, "cuda", dtype=torch.float32, seed=seed)
    if gs != 1.0:
        inp = dict(inp)
        inp["g"] = inp["g"] * gs
    inp = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
    return inp


def part_a(lib):
    print("=" * 96)
    print("Part A: K2-shape CUDA intra vs Triton MMA B3 (same dAqk/dAkk) and vs torch golden")
    print("=" * 96)
    mma_k = build_intra_mma()
    worst = {}
    for (name, B, T, H, seed) in CASES:
        inp = make(B, T, H, seed)
        tri = intra_triton_mma(mma_k, inp, BT, BC, num_warps=4)
        cu = run_cuda(lib, inp)
        gold = intra_torch(inp, BT, BC) if T % BT == 0 else None
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        print("  -- cuda vs Triton MMA --")
        for fld in ("dq", "dk", "db", "dg"):
            _, mr = stats(f"  cuda {fld} vs trit", cu[fld], tri[fld])
            worst[f"ct_{fld}"] = max(worst.get(f"ct_{fld}", 0.0), mr)
        if gold is not None:
            print("  -- cuda vs torch golden --")
            for fld in ("dq", "dk", "db", "dg"):
                _, mr = stats(f"  cuda {fld} vs gold", cu[fld], gold[fld])
                worst[f"cg_{fld}"] = max(worst.get(f"cg_{fld}", 0.0), mr)
            print("  -- Triton MMA vs torch golden (context) --")
            for fld in ("dq", "dk", "db", "dg"):
                _, mr = stats(f"  trit {fld} vs gold", tri[fld], gold[fld])
                worst[f"tg_{fld}"] = max(worst.get(f"tg_{fld}", 0.0), mr)
    print("\nworst-case max_rel over all cases:")
    for fld in ("dq", "dk", "db", "dg"):
        print(f"  {fld:4s}  cuda/trit {worst['ct_'+fld]:.3e}   "
              f"cuda/gold {worst['cg_'+fld]:.3e}   trit/gold {worst['tg_'+fld]:.3e}")
    # g=0 degeneracy (no decay) -- the factored form's stability limit
    print("\n  -- g=0 degeneracy (T=512 H=4) --")
    inp = make(1, 512, 4, seed=7, gs=0.0)
    tri = intra_triton_mma(mma_k, inp, BT, BC, num_warps=4)
    cu = run_cuda(lib, inp)
    for fld in ("dq", "dk", "db", "dg"):
        stats(f"  g=0 cuda {fld} vs trit", cu[fld], tri[fld])


def part_b(lib):
    print("\n" + "=" * 96)
    print("Part B: kernel-only timing -- K2-shape CUDA intra vs Triton MMA B3")
    print("=" * 96)
    mma_k = build_intra_mma()
    print(f"  {'case':22s} | {'CUDA dev':>9s} {'CUDA wall':>9s} | "
          f"{'trit nw2':>9s} {'trit nw4':>9s} | {'CUDA/trit':>9s}")
    for (name, B, T, H, seed) in CASES:
        inp = make(B, T, H, seed)
        run_cuda(lib, inp); torch.cuda.synchronize()
        dcu, _ = profile_dev(lambda: run_cuda(lib, inp), reps=30)
        wcu = time_wall(lambda: run_cuda(lib, inp), reps=50)
        k2 = build_intra_mma()
        intra_triton_mma(k2, inp, BT, BC, num_warps=2); torch.cuda.synchronize()
        d2, _ = profile_dev(lambda: intra_triton_mma(k2, inp, BT, BC, num_warps=2), reps=30)
        k4 = build_intra_mma()
        intra_triton_mma(k4, inp, BT, BC, num_warps=4); torch.cuda.synchronize()
        d4, _ = profile_dev(lambda: intra_triton_mma(k4, inp, BT, BC, num_warps=4), reps=30)
        best = min(d2, d4)
        print(f"  {name:22s} | {dcu:9.1f} {wcu:9.1f} | {d2:9.1f} {d4:9.1f} | "
              f"{dcu/best:8.2f}x")


def part_c(lib):
    print("\n" + "=" * 96)
    print("Part C: static cost")
    print("=" * 96)
    smem = lib.b3intra_smem_bytes()
    print(f"  K2-shape CUDA intra smem = {smem} B ({smem/1024:.1f} KB), "
          f"block = {8*32} threads (8 warps)")
    print(f"  smem-limited blocks/SM (228KB) ~ {233472 // (smem + 1024)}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build()
    lib = load()
    mode = os.environ.get("B3IC_MODE", "")
    if mode in ("", "a"):
        part_a(lib)
    if mode in ("", "b"):
        part_b(lib)
    if mode in ("", "c"):
        part_c(lib)
    print("\nDONE")


if __name__ == "__main__":
    main()
