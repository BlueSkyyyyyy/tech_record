"""P7-73: bf16-native operand entry for the K2-shaped CUDA B2 (`wy_dqkg`) GEMM core.

`b2_mma_probe.cu` was fp32/tf32-only, so the assembled reverse chain had to `.float()` B1's
bf16 `dh`/`dv2` before feeding B2 (P7-64 §67.4 pinned this).  `-DB2M_BF16` stages the five
streamed operands (`do/vn/dv/h/dh`) as bf16 (halving the global DRAM read and the smem
footprint); the mma stays `m16n8k8.tf32`.  bf16 is a strict subset of tf32, so widening
bf16->fp32 (exact) before the `cvt.rna.tf32` truncation reproduces bf16-mma products exactly
-- only the fp32 accumulation order differs.

This harness races the bf16-native build against the fp32 build:

  Part A  correctness:
    - bf16 CUDA on bf16 operands vs torch fp32 of the *same bf16* operands (exact ref),
    - fp32 CUDA on fp32 operands vs torch fp32 (baseline, context),
    - bf16 CUDA on bf16 operands vs fp32 CUDA on `bf16.float()` operands -- the *deployment
      drop-in* question (does replacing the `.float()`-then-fp32-core with a bf16 core change
      the answer?).
  Part B  kernel-only timing: bf16 vs fp32 build (CUPTI dev + wall).
  Part C  static: smem bytes + registers of both builds.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_bf16.py \
        > .../harness/check_bwd_b2_bf16.out.txt 2>&1
    B2BF_MODE=a | b | c
"""
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs        # noqa: E402
from check_bwd_b2_core import ref_core                                    # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                                # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev                  # noqa: E402

SO_BASE = "/tmp/b2_bf16_base.so"
SO_BF16 = "/tmp/b2_bf16_native.so"
NTHR = 256
BT = 16
OP_KEYS = ("v", "v_new", "h", "dh", "do", "dv")

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 16),
    ("B2 T1024 H8", 2, 1024, 8, 16),
    ("T512 H4", 1, 512, 4, 16),
    ("T250 H3 (non-div)", 1, 250, 3, 16),
]


def build(extra, out):
    src = os.path.join(HARNESS, "b2_mma_probe.cu")
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + [src, "-o", out]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed")


def load(path):
    lib = ctypes.CDLL(path)
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


def to_bf16(inp):
    """The deployment scenario: B1 emits bf16, and today's chain does `.float()` to feed B2."""
    out = dict(inp)
    for k in OP_KEYS:
        out[k] = inp[k].to(torch.bfloat16)
    return out


def widen(inp):
    """bf16 operands cast back to fp32 -- exactly what the current fp32 core is fed."""
    out = dict(inp)
    for k in OP_KEYS:
        out[k] = inp[k].float()
    return out


def part_a(lib_base, lib_bf16):
    print("=" * 96)
    print("Part A: correctness")
    print("  ref = torch fp32 of the *same* operands (allow_tf32=False); max_rel printed")
    print("=" * 96)
    worst = {}
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        inp_bf = to_bf16(inp)
        ref_f = ref_core(inp, BT)
        ref_b = ref_core(inp_bf, BT)
        cu_f = run_cuda(lib_base, inp, BT)
        cu_b = run_cuda(lib_bf16, inp_bf, BT)
        cu_d = run_cuda(lib_base, widen(inp_bf), BT)   # fp32 core on bf16.float() (= deployment)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for fld in ("dq", "dk", "dw", "dgk"):
            _, mr_f = stats(f"  fp32cuda vs torch      {fld}", cu_f[fld], ref_f[fld])
            _, mr_b = stats(f"  bf16cuda vs torch(bf16){fld}", cu_b[fld], ref_b[fld])
            _, mr_d = stats(f"  bf16cuda vs fp32cuda   {fld}", cu_b[fld], cu_d[fld])
            worst[f"f_{fld}"] = max(worst.get(f"f_{fld}", 0.0), mr_f)
            worst[f"b_{fld}"] = max(worst.get(f"b_{fld}", 0.0), mr_b)
            worst[f"d_{fld}"] = max(worst.get(f"d_{fld}", 0.0), mr_d)
    print("\nworst max_rel over all cases:")
    for fld in ("dq", "dk", "dw", "dgk"):
        print(f"  {fld:4s}  fp32/torch {worst['f_'+fld]:.3e}   "
              f"bf16/torch(bf16) {worst['b_'+fld]:.3e}   "
              f"bf16 vs fp32 cuda {worst['d_'+fld]:.3e}")


def part_b(lib_base, lib_bf16):
    print("\n" + "=" * 96)
    print("Part B: kernel-only timing -- bf16-native vs fp32 build (same program shape)")
    print("=" * 96)
    print(f"  {'case':18s} | {'fp32 dev':>9s} {'fp32 wall':>9s} | "
          f"{'bf16 dev':>9s} {'bf16 wall':>9s} | {'bf16/fp32 dev':>13s}")
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        inp_bf = to_bf16(inp)
        run_cuda(lib_base, inp, BT); torch.cuda.synchronize()
        df, _ = profile_dev(lambda: run_cuda(lib_base, inp, BT), reps=30)
        wf = time_wall(lambda: run_cuda(lib_base, inp, BT), reps=50)
        run_cuda(lib_bf16, inp_bf, BT); torch.cuda.synchronize()
        db, _ = profile_dev(lambda: run_cuda(lib_bf16, inp_bf, BT), reps=30)
        wb = time_wall(lambda: run_cuda(lib_bf16, inp_bf, BT), reps=50)
        print(f"  {name:18s} | {df:9.1f} {wf:9.1f} | {db:9.1f} {wb:9.1f} | "
              f"{db/df:12.3f}x")


def part_c(lib_base, lib_bf16):
    print("\n" + "=" * 96)
    print("Part C: static cost")
    print("=" * 96)
    for tag, lib in (("fp32", lib_base), ("bf16", lib_bf16)):
        smem = lib.b2mma_smem_bytes()
        print(f"  {tag:4s}: smem = {smem} B ({smem/1024:.1f} KB), "
              f"smem-limited blocks/SM ~ {233472 // (smem + 1024)}")
    for tag, extra in (("fp32", []), ("bf16", ["-DB2M_BF16"])):
        cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
               "-shared", "-Xptxas", "-v"] + extra + \
              [os.path.join(HARNESS, "b2_mma_probe.cu"), "-o", "/tmp/_b2bf_ptxas.so"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        info = [ln for ln in r.stderr.splitlines()
                if "registers" in ln or "smem" in ln.lower() or "spill" in ln.lower()]
        print(f"  ptxas {tag}: " + " | ".join(info[-3:]))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build([], SO_BASE)
    build(["-DB2M_BF16"], SO_BF16)
    lib_base = load(SO_BASE)
    lib_bf16 = load(SO_BF16)
    mode = os.environ.get("B2BF_MODE", "")
    if mode in ("", "a"):
        part_a(lib_base, lib_bf16)
    if mode in ("", "b"):
        part_b(lib_base, lib_bf16)
    if mode in ("", "c"):
        part_c(lib_base, lib_bf16)
    print("\nDONE")


if __name__ == "__main__":
    main()
