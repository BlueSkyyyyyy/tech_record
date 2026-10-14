"""P7-88: close the last `K != V` boundary recorded in docs/07 §90.5 -- `K != 128`
(`b2_mma_probe.cu`'s `static_assert(KD == 128, ...)`), the **lane-layout rewrite** the prior
round explicitly deferred.

Every CUDA B2 round so far fixed `K = 128` because the output-N warp split is `BW = KD/NW` and
the `dgk` column reduction / gate chain were written for the 16-wide lane layout (`l & 15`).
This round parameterizes the **base GEMM core** (`dq/dk/dw/dgk`; the six non-GEMM segments stay
`KD == 128`, guarded by a new `#error`) in `K`:

  * the mma `n=8` tile count is `NBW = BW/8` (was hardcoded `2`);
  * the `dgk` column reduction is made generic -- lane `l` owns column `l % BW`, splitting the V
    contraction across the `32/BW` subclasses `l / BW` (at BW=16 this is bit-for-bit the old
    `l & 15` / `l >> 4` / step-2 scheme), and the xor-reduce is `for sh = BW; sh < 32; sh <<= 1`.

`K` only ever appears as the `h/dh` row count (= output N), so the contraction (`V`) is unchanged.

Scope: base GEMM core only.  Scan `K in {64,128,256}` at `V = 128` (isolates `K`), then the pure
crossing cases `(K,V) in {(64,256),(256,64)}`.

Reference / baseline: torch fp32 (`allow_tf32=False`) as the algebraic golden; the identical-op
Triton core (`check_bwd_b2_core.py`, already generic in K,V) where its `[V,K]` smem fits.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=3 python .../harness/check_bwd_b2_k128.py \
        > .../harness/check_bwd_b2_k128.out.txt 2>&1
    B2K_MODE=a | b | d
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
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs          # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                                 # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev                   # noqa: E402
from check_bwd_b2_core import build_core_triton, run_triton_core, ref_core  # noqa: E402

BT = 16
NTHR = 256

# (K, V) pairs: first the K-scan at V=128, then the pure crossing cases.
KV = [(64, 128), (128, 128), (256, 128), (64, 256), (256, 64)]

CASES = [
    ("B8 T2048 H8", 8, 2048, 8),
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]


def so_path(K, V):
    return os.path.join("/tmp", "b2k128_%d_%d.so" % (K, V))


def build(K, V):
    src = os.path.join(HARNESS, "b2_mma_probe.cu")
    extra = ["-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V] + \
        os.environ.get("B2K_CFLAGS", "").split()
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + ["-lcuda", src, "-o", so_path(K, V)]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed for K=%d V=%d" % (K, V))


def load(K, V):
    lib = ctypes.CDLL(so_path(K, V))
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    lib.b2mma_smem_bytes.restype = ctypes.c_int
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_cuda(lib, inp, BT_, K, V):
    B, T, H, K_ = inp["q"].shape
    assert K_ == K and inp["v"].shape[-1] == V
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


def make(B, T, H, K, V, seed=0):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def part_a(libs):
    print("=" * 96)
    print("Part A: correctness -- K2-mma CUDA base GEMM core, K in {64,128,256}")
    print("        vs torch fp32 (golden, allow_tf32=False) and vs identical-op Triton core")
    print("=" * 96)
    worst = {}
    for (K, V) in KV:
        lib = libs[(K, V)]
        print(f"\n#### K = {K}, V = {V} ####")
        # the identical-op Triton core keeps the full [V,K] operand in smem; only run it when
        # K*V fits the deployment (128*128) footprint.
        have_tri = (K * V <= 128 * 128)
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            ref = ref_core(inp, BT)
            cu = run_cuda(lib, inp, BT, K, V)
            tri = None
            if have_tri:
                tk = build_core_triton()
                tri = run_triton_core(tk, inp, BT, num_warps=8)
            print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
            for fld in ("dq", "dk", "dw", "dgk"):
                _, mr_c = stats(f"  [K{K}V{V}] cuda {fld} vs torch", cu[fld], ref[fld])
                worst[f"cuda_{K}_{V}_{fld}"] = max(worst.get(f"cuda_{K}_{V}_{fld}", 0.0), mr_c)
                if tri is not None:
                    _, mr_t = stats(f"  [K{K}V{V}] trit {fld} vs torch", tri[fld], ref[fld])
                    _, mr_ct = stats(f"  [K{K}V{V}] cuda {fld} vs trit ", cu[fld], tri[fld])
                    worst[f"trit_{K}_{V}_{fld}"] = max(worst.get(f"trit_{K}_{V}_{fld}", 0.0), mr_t)
                    worst[f"ct_{K}_{V}_{fld}"] = max(worst.get(f"ct_{K}_{V}_{fld}", 0.0), mr_ct)
    print("\nworst max_rel over all cases:")
    print(f"  {'K':>4s} {'V':>4s} {'field':>5s}  {'cuda/torch':>11s} {'trit/torch':>11s} {'cuda/trit':>10s}")
    for (K, V) in KV:
        for fld in ("dq", "dk", "dw", "dgk"):
            t = worst.get(f"trit_{K}_{V}_{fld}")
            c = worst.get(f"ct_{K}_{V}_{fld}")
            ts = f"{t:11.3e}" if t is not None else f"{'n/a':>11s}"
            cs = f"{c:10.3e}" if c is not None else f"{'n/a':>10s}"
            print(f"  {K:4d} {V:4d} {fld:>5s}  {worst['cuda_'+str(K)+'_'+str(V)+'_'+fld]:11.3e} {ts} {cs}")
    print("  (trit n/a when K*V > 128*128: the identical-op Triton core holds the full [V,K]"
          " operand in smem (>232 KB); CUDA streams V-tiles so it still runs.)")


def part_b(libs):
    print("\n" + "=" * 96)
    print("Part B: identical-op kernel-only timing -- CUDA base GEMM core vs Triton core")
    print("=" * 96)
    print(f"  {'case':18s} {'K':>4s} {'V':>4s} | {'CUDA dev':>9s} {'CUDA wall':>9s} | "
          f"{'trit nw8':>9s} | {'CUDA/trit':>9s}")
    for (K, V) in KV:
        lib = libs[(K, V)]
        have_tri = (K * V <= 128 * 128)
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            run_cuda(lib, inp, BT, K, V); torch.cuda.synchronize()
            dcu, _ = profile_dev(lambda: run_cuda(lib, inp, BT, K, V), reps=30)
            wcu = time_wall(lambda: run_cuda(lib, inp, BT, K, V), reps=50)
            if have_tri:
                tk = build_core_triton()
                run_triton_core(tk, inp, BT, num_warps=8); torch.cuda.synchronize()
                d8, _ = profile_dev(lambda: run_triton_core(tk, inp, BT, num_warps=8), reps=30)
                print(f"  {name:18s} {K:4d} {V:4d} | {dcu:9.1f} {wcu:9.1f} | {d8:9.1f} | "
                      f"{dcu/d8:8.2f}x")
            else:
                print(f"  {name:18s} {K:4d} {V:4d} | {dcu:9.1f} {wcu:9.1f} | {'n/a':>9s} | {'n/a':>8s}")


def part_d(libs):
    print("\n" + "=" * 96)
    print("Part D: static cost (smem scales with KD h/dh rows x BVK V-tile)")
    print("=" * 96)
    for (K, V) in KV:
        smem = libs[(K, V)].b2mma_smem_bytes()
        print(f"  K={K:4d} V={V:4d}: smem = {smem} B ({smem/1024:.1f} KB), block = {NTHR} "
              f"threads; nvt = {V//32} (BVK=32), grid=(NT, B*H)")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for (K, V) in KV:
        build(K, V)
        libs[(K, V)] = load(K, V)
    mode = os.environ.get("B2K_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs)
    if mode in ("", "d"):
        part_d(libs)
    print("\nDONE")


if __name__ == "__main__":
    main()
