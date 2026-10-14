"""P7-81: close the "only K=V=128 tested" boundary of the K2-shaped CUDA B2 **base GEMM
core** (`b2_mma_probe.cu`, `-DB2M_KD`/`-DB2M_VD`).

The KDA state is `S in R^{K x V}`: the key/output-N dim `K` and the value/contraction dim
`V` are, in general, independent.  Every CUDA B2 round so far (P7-46…P7-80) compiled and
verified with the single constant `KD = VD = 128`.  This round parameterizes the two dims
and checks whether the **base GEMM core** (`dq/dk/dw/dgk`; the six non-GEMM segments are
off) is already generic in `K != V` -- it is, because the core indexes the two axes
consistently (`dq/dk/dw/dgk` use `KD`, the `do/vn/dv` operands use `VD`, and the TMA maps
`make_map_dov`/`make_map_hdh` carry `{VD,KD}` separately).

Scope: base GEMM core only.  The six non-GEMM segments partition the V axis by the warp
output width `BW = KD/NW`, so they remain `VD==KD` (guarded by a new `#error` in the .cu).
`K` (the key dim) is fixed at 128 by the warp output partition; we scan `V in {64,128,256}`.

Reference / baseline: torch fp32 (`allow_tf32=False`) as the algebraic golden; the
identical-op Triton core (`check_bwd_b2_core.py`) as the `tl.dot` baseline (it is already
generic in K,V).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=3 python .../harness/check_bwd_b2_kv.py \
        > .../harness/check_bwd_b2_kv.out.txt 2>&1
    B2KV_MODE=a | b | d
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

K = 128
BT = 16
NTHR = 256
VP = {64: "v64.so", 128: "v128.so", 256: "v256.so"}   # V -> per-V shared object

CASES = [
    ("B8 T2048 H8", 8, 2048, 8),
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]
VS = [64, 128, 256]


def so_path(V):
    return os.path.join("/tmp", "b2kv_" + VP[V])


def build(V):
    src = os.path.join(HARNESS, "b2_mma_probe.cu")
    extra = ["-DB2M_VD=%d" % V] + os.environ.get("B2KV_CFLAGS", "").split()
    # -lcuda is needed whenever the TMA path (`-DB2M_TMA`) is on (cuTensorMapEncodeTiled);
    # harmless otherwise.
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + ["-lcuda", src, "-o", so_path(V)]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed for V=%d" % V)


def load(V):
    lib = ctypes.CDLL(so_path(V))
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    lib.b2mma_smem_bytes.restype = ctypes.c_int
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_cuda(lib, inp, BT_, V):
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


def make(B, T, H, V, seed=0):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def part_a(libs):
    print("=" * 96)
    print("Part A: correctness -- K2-mma CUDA base GEMM core (K=128, V in {64,128,256})")
    print("        vs torch fp32 (golden, allow_tf32=False) and vs identical-op Triton core")
    print("=" * 96)
    worst = {}
    for V in VS:
        lib = libs[V]
        print(f"\n#### V = {V}  (mma contraction dim; K = {K} fixed) ####")
        # the identical-op Triton core keeps the full `[V,K]` operand in smem; at V=256 that
        # needs 311 KB > 232 KB, so the Triton baseline cannot run there (CUDA streams V-tiles).
        have_tri = V <= 128
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, V, seed=1)
            ref = ref_core(inp, BT)
            cu = run_cuda(lib, inp, BT, V)
            tri = None
            if have_tri:
                tk = build_core_triton()
                tri = run_triton_core(tk, inp, BT, num_warps=8)
            print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
            for fld in ("dq", "dk", "dw", "dgk"):
                _, mr_c = stats(f"  [{V}] cuda {fld} vs torch", cu[fld], ref[fld])
                worst[f"cuda_{V}_{fld}"] = max(worst.get(f"cuda_{V}_{fld}", 0.0), mr_c)
                if tri is not None:
                    _, mr_t = stats(f"  [{V}] trit {fld} vs torch", tri[fld], ref[fld])
                    _, mr_ct = stats(f"  [{V}] cuda {fld} vs trit ", cu[fld], tri[fld])
                    worst[f"trit_{V}_{fld}"] = max(worst.get(f"trit_{V}_{fld}", 0.0), mr_t)
                    worst[f"ct_{V}_{fld}"] = max(worst.get(f"ct_{V}_{fld}", 0.0), mr_ct)
    print("\nworst max_rel over all cases:")
    print(f"  {'V':>4s} {'field':>5s}  {'cuda/torch':>11s} {'trit/torch':>11s} {'cuda/trit':>10s}")
    for V in VS:
        for fld in ("dq", "dk", "dw", "dgk"):
            t = worst.get(f"trit_{V}_{fld}")
            c = worst.get(f"ct_{V}_{fld}")
            ts = f"{t:11.3e}" if t is not None else f"{'n/a':>11s}"
            cs = f"{c:10.3e}" if c is not None else f"{'n/a':>10s}"
            print(f"  {V:4d} {fld:>5s}  {worst['cuda_'+str(V)+'_'+fld]:11.3e} {ts} {cs}")
    print("  (trit n/a at V=256: the identical-op Triton core holds the full [V,K] operand in"
          " smem -> 311 KB > 232 KB; CUDA streams V-tiles so it runs.)")


def part_b(libs):
    print("\n" + "=" * 96)
    print("Part B: identical-op kernel-only timing -- CUDA base GEMM core vs Triton core")
    print("=" * 96)
    print(f"  {'case':18s} {'V':>4s} | {'CUDA dev':>9s} {'CUDA wall':>9s} | "
          f"{'trit nw8':>9s} | {'CUDA/trit':>9s}")
    for V in VS:
        lib = libs[V]
        have_tri = V <= 128
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, V, seed=1)
            run_cuda(lib, inp, BT, V); torch.cuda.synchronize()
            dcu, _ = profile_dev(lambda: run_cuda(lib, inp, BT, V), reps=30)
            wcu = time_wall(lambda: run_cuda(lib, inp, BT, V), reps=50)
            if have_tri:
                tk = build_core_triton()
                run_triton_core(tk, inp, BT, num_warps=8); torch.cuda.synchronize()
                d8, _ = profile_dev(lambda: run_triton_core(tk, inp, BT, num_warps=8), reps=30)
                print(f"  {name:18s} {V:4d} | {dcu:9.1f} {wcu:9.1f} | {d8:9.1f} | "
                      f"{dcu/d8:8.2f}x")
            else:
                print(f"  {name:18s} {V:4d} | {dcu:9.1f} {wcu:9.1f} | {'n/a':>9s} | {'n/a':>8s}")


def part_d(libs):
    print("\n" + "=" * 96)
    print("Part D: static cost (smem scales with KD rows x BVK V-tile; VD=256 doubles stages)")
    print("=" * 96)
    for V in VS:
        smem = libs[V].b2mma_smem_bytes()
        print(f"  V={V:4d}: smem = {smem} B ({smem/1024:.1f} KB), block = {NTHR} threads; "
              f"nvt = {V//32} (BVK=32), grid=(NT, B*H)")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for V in VS:
        build(V)
        libs[V] = load(V)
    mode = os.environ.get("B2KV_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs)
    if mode in ("", "d"):
        part_d(libs)
    print("\nDONE")


if __name__ == "__main__":
    main()
