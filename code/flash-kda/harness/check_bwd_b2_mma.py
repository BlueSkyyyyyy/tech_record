"""P7-46: second slice of the K2-shaped CUDA port of the fused B2/B3 -- the B2
(`wy_dqkg`) **V-reduction GEMM core** in K2 shape **with `mma.sync` tensor core**.

P7-44 (`check_bwd_b2_core.py` / `b2_core_probe.cu`) killed reading (b) of §46.6
("放弃 tl.dot, plain SIMT fp32 register-resident"): it lost 3-5x to the identical-op
Triton kernel.  P7-45 (§48) then found the fused kernel's ~1.9x headroom is
*register residency of the big operands (h/dh)*, not bytes and not occupancy, and
named the first action: "把大操作数从寄存器保活里拿掉、只留 warp 分片的小累加器".

This harness races `b2_mma_probe.cu` (K2 shape: warp-sharded `[16,16]` accumulators +
`h/dh/do/vn/dv` streamed through smem in V-tiles via `cp.async` double buffering +
`mma.sync.m16n8k8.tf32`) against:

  * the identical-op Triton core (P7-44's `build_core_triton`, tf32 `tl.dot`),
  * torch fp32 (`allow_tf32=False`) as the algebraic reference,
  * the full Triton B2 and the fused B2/B3 (context).

So the *only* changed variable vs P7-44 is the kernel structure (tensor core kept,
operands streamed / not register-resident) -- dtype (fp32/tf32), inputs and op set
are identical.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_mma.py \
        > .../harness/check_bwd_b2_mma.out.txt 2>&1
    B2MM_MODE=a | b | c | d
"""
import ctypes
import math
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

# CUDA core is tf32 (matches `tl.dot`); turn off cublas tf32 so the torch reference is
# true fp32 -- then "CUDA vs Triton" (both tf32) is tight and "CUDA vs torch" shows the
# tf32 noise floor.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import (make_inputs as dqkg_make_inputs,   # noqa: E402
                                   build_kernel as build_dqkg, wy_dqkg_triton)
from check_bwd_b2b3_fused import build_fused_b2b3, fused_b2b3_triton   # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                             # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev               # noqa: E402
from check_bwd_b2_core import (build_core_triton, run_triton_core,     # noqa: E402
                               ref_core, kern_info)

D = 128
SO = "/tmp/b2_mma.so"
NTHR = 256

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]
BT = 16


def build():
    src = os.path.join(HARNESS, "b2_mma_probe.cu")
    extra = os.environ.get("B2MM_CFLAGS", "").split()
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + [src, "-o", SO]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed")


def load():
    lib = ctypes.CDLL(SO)
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    lib.b2mma_smem_bytes.restype = ctypes.c_int
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def run_cuda(lib, inp, BT_):
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    assert K == 128 and V == 128, "kernel is compiled for K=V=128 (deployment)"
    NT = cdiv(T, BT_)
    dev = inp["q"].device
    for name in ("v", "v_new", "h", "dh", "do", "dv"):
        assert inp[name].dtype == torch.float32
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


def make(B, T, H, seed=0):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    inp = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
    return inp


def part_a(lib):
    print("=" * 92)
    print("Part A: correctness -- K2-mma CUDA vs torch(fp32) / vs identical-op Triton core")
    print("=" * 92)
    worst = {}
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        ref = ref_core(inp, BT)
        cu = run_cuda(lib, inp, BT)
        tk = build_core_triton()
        tri = run_triton_core(tk, inp, BT, num_warps=8)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for fld in ("dq", "dk", "dw", "dgk"):
            _, mr_c = stats(f"  cuda {fld} vs torch", cu[fld], ref[fld])
            _, mr_t = stats(f"  trit {fld} vs torch", tri[fld], ref[fld])
            _, mr_ct = stats(f"  cuda {fld} vs trit ", cu[fld], tri[fld])
            worst[f"cuda_{fld}"] = max(worst.get(f"cuda_{fld}", 0.0), mr_c)
            worst[f"trit_{fld}"] = max(worst.get(f"trit_{fld}", 0.0), mr_t)
            worst[f"ct_{fld}"] = max(worst.get(f"ct_{fld}", 0.0), mr_ct)
    print("\nworst max_rel:")
    for fld in ("dq", "dk", "dw", "dgk"):
        print(f"  {fld:4s}  cuda/torch {worst['cuda_'+fld]:.3e}   "
              f"trit/torch {worst['trit_'+fld]:.3e}   cuda/trit {worst['ct_'+fld]:.3e}")


def part_b(lib):
    print("\n" + "=" * 92)
    print("Part B: head-to-head identical-op kernel-only timing -- K2-mma CUDA vs Triton core")
    print("=" * 92)
    print(f"  {'case':18s} | {'CUDA dev':>9s} {'CUDA wall':>9s} | "
          f"{'trit nw4':>9s} {'trit nw8':>9s} | {'CUDA/trit':>9s}")
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        run_cuda(lib, inp, BT); torch.cuda.synchronize()
        dcu, _ = profile_dev(lambda: run_cuda(lib, inp, BT), reps=30)
        wcu = time_wall(lambda: run_cuda(lib, inp, BT), reps=50)
        tk = build_core_triton()
        run_triton_core(tk, inp, BT, num_warps=4); torch.cuda.synchronize()
        d4, _ = profile_dev(lambda: run_triton_core(tk, inp, BT, num_warps=4), reps=30)
        tk8 = build_core_triton()
        run_triton_core(tk8, inp, BT, num_warps=8); torch.cuda.synchronize()
        d8, _ = profile_dev(lambda: run_triton_core(tk8, inp, BT, num_warps=8), reps=30)
        best = min(d4, d8)
        print(f"  {name:18s} | {dcu:9.1f} {wcu:9.1f} | {d4:9.1f} {d8:9.1f} | "
              f"{dcu/best:8.2f}x")


def part_c(lib):
    print("\n" + "=" * 92)
    print("Part C: context -- K2-mma core vs full Triton B2 and full fused B2/B3")
    print("=" * 92)
    dqkg_k = build_dqkg()
    fused_k = build_fused_b2b3()
    tk = build_core_triton()
    print(f"  {'case':18s} | {'CUDAcore':>9s} {'TritCore':>9s} {'fullB2':>9s} "
          f"{'fusedB2B3':>10s} | {'GF/s(core)':>10s}")
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        scale = 1.0 / math.sqrt(128)
        torch.manual_seed(7)
        dAqk = torch.randn(B, T, H, BT, device="cuda") * 0.05

        run_cuda(lib, inp, BT); torch.cuda.synchronize()
        dcu, _ = profile_dev(lambda: run_cuda(lib, inp, BT), reps=20)
        run_triton_core(tk, inp, BT, num_warps=8); torch.cuda.synchronize()
        dtr, _ = profile_dev(lambda: run_triton_core(tk, inp, BT, num_warps=8), reps=20)
        wy_dqkg_triton(dqkg_k, inp, scale, BT, num_warps=4); torch.cuda.synchronize()
        d2, _ = profile_dev(lambda: wy_dqkg_triton(dqkg_k, inp, scale, BT, num_warps=4), reps=20)
        fused_b2b3_triton(fused_k, inp, dAqk, scale, BT); torch.cuda.synchronize()
        df, _ = profile_dev(lambda: fused_b2b3_triton(fused_k, inp, dAqk, scale, BT), reps=20)
        flop = 6.0 * B * T * H * 128 * 128
        gfs = flop / (dcu * 1e-6) / 1e9
        print(f"  {name:18s} | {dcu:9.1f} {dtr:9.1f} {d2:9.1f} {df:10.1f} | "
              f"{gfs:10.1f}")


def part_d(lib):
    print("\n" + "=" * 92)
    print("Part D: static cost")
    print("=" * 92)
    smem = lib.b2mma_smem_bytes()
    print(f"  K2-mma CUDA smem = {smem} B ({smem/1024:.1f} KB), block = {NTHR} threads (8 warps)")
    print(f"  smem-limited blocks/SM (228KB/({smem/1024:.1f}KB+1KB)) ~ {233472 // (smem + 1024)}")
    tk = build_core_triton()
    inp = make(8, 2048, 8, seed=1)
    run_triton_core(tk, inp, BT, num_warps=8)
    r8 = kern_info(tk)
    print(f"  triton core nw=8 : regs={r8[0]} spills={r8[1]} smem={r8[2]/1024:.1f}KB"
          if r8[0] is not None else str(r8))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build()
    lib = load()
    mode = os.environ.get("B2MM_MODE", "")
    if mode in ("", "a"):
        part_a(lib)
    if mode in ("", "b"):
        part_b(lib)
    if mode in ("", "c"):
        part_c(lib)
    if mode in ("", "d"):
        part_d(lib)
    print("\nDONE")


if __name__ == "__main__":
    main()
