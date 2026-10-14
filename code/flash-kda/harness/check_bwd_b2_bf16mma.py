"""P7-74: native bf16 `mma.sync.m16n8k16` for the K2-shaped CUDA B2 (`wy_dqkg`) GEMM core.

P7-73 (§76) gave the core a bf16-native operand entry (`-DB2M_BF16`) but kept the tf32
`mma.m16n8k8` -- legitimate because bf16 is a strict subset of tf32 (products stay exact, only
the fp32 accumulation order differs).  The residual tf32 limitation (§47.6) is that `.tf32`
fragments are 32-bit, so there is no 16-bit `ldmatrix` and the operands must be read straight
from smem.

`-DB2M_BF16MMA` (requires `-DB2M_BF16`) swaps the core's mma for a native bf16
`mma.sync.m16n8k16` (k16 per instruction).  The A/B fragment *values* are identical to the k8
path (the two-sided packing feeds the exact same operands), so the only difference is the
k-grouping / instruction.

This harness races the bf16-mma build against the P7-73 bf16(tf32) and fp32 builds:

  Part A  correctness:
    - bf16-mma CUDA vs torch fp32 of the *same bf16* operands (exact ref),
    - bf16-mma vs bf16(tf32) build -- isolates the k-grouping change (fragment-layout probe:
      a wrong A/B column mapping shows as O(1), not 1e-7),
    - bf16-mma vs fp32 build on `bf16.float()` -- the deployment drop-in question.
  Part B  kernel-only timing: fp32 / bf16(tf32) / bf16-mma (CUPTI dev + wall).
  Part C  static: smem bytes + registers of all three builds.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_bf16mma.py \
        > .../harness/check_bwd_b2_bf16mma.out.txt 2>&1
    B2BMM_MODE=a | b | c
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

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs        # noqa: E402
from check_bwd_b2_core import ref_core                                    # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                                # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev                  # noqa: E402

SO_BASE = "/tmp/b2_bmm_base.so"
SO_BF16 = "/tmp/b2_bmm_bf16.so"
SO_BMM = "/tmp/b2_bmm_native.so"
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
    """bf16 operands cast back to fp32 -- exactly what the fp32 core is fed."""
    out = dict(inp)
    for k in OP_KEYS:
        out[k] = inp[k].float()
    return out


def part_a(lib_base, lib_bf16, lib_bmm):
    print("=" * 100)
    print("Part A: correctness")
    print("  ref = torch fp32 of the *same* operands (allow_tf32=False); max_rel printed")
    print("=" * 100)
    worst = {}
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        inp_bf = to_bf16(inp)
        ref_f = ref_core(inp, BT)
        ref_b = ref_core(inp_bf, BT)
        cu_f = run_cuda(lib_base, inp, BT)
        cu_b = run_cuda(lib_bf16, inp_bf, BT)
        cu_m = run_cuda(lib_bmm, inp_bf, BT)
        cu_d = run_cuda(lib_base, widen(inp_bf), BT)   # fp32 core on bf16.float() (= deployment)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for fld in ("dq", "dk", "dw", "dgk"):
            _, mr_bm = stats(f"  bf16mma vs torch(bf16){fld}", cu_m[fld], ref_b[fld])
            _, mr_mb = stats(f"  bf16mma vs bf16(tf32)  {fld}", cu_m[fld], cu_b[fld])
            _, mr_md = stats(f"  bf16mma vs fp32(cuda)  {fld}", cu_m[fld], cu_d[fld])
            worst[f"bm_{fld}"] = max(worst.get(f"bm_{fld}", 0.0), mr_bm)
            worst[f"mb_{fld}"] = max(worst.get(f"mb_{fld}", 0.0), mr_mb)
            worst[f"md_{fld}"] = max(worst.get(f"md_{fld}", 0.0), mr_md)
    print("\nworst max_rel over all cases:")
    for fld in ("dq", "dk", "dw", "dgk"):
        print(f"  {fld:4s}  bf16mma/torch(bf16) {worst['bm_'+fld]:.3e}   "
              f"bf16mma vs bf16(tf32) {worst['mb_'+fld]:.3e}   "
              f"bf16mma vs fp32 cuda {worst['md_'+fld]:.3e}")


def part_b(lib_base, lib_bf16, lib_bmm):
    print("\n" + "=" * 100)
    print("Part B: kernel-only timing -- fp32 / bf16(tf32) / bf16-mma (same program shape)")
    print("=" * 100)
    print(f"  {'case':18s} | {'fp32 dev':>9s} | {'bf16 dev':>9s} {'bf16/fp32':>9s} | "
          f"{'bmm dev':>9s} {'bmm/fp32':>9s} {'bmm/bf16':>9s}")
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        inp_bf = to_bf16(inp)
        run_cuda(lib_base, inp, BT); torch.cuda.synchronize()
        df, _ = profile_dev(lambda: run_cuda(lib_base, inp, BT), reps=30)
        run_cuda(lib_bf16, inp_bf, BT); torch.cuda.synchronize()
        db, _ = profile_dev(lambda: run_cuda(lib_bf16, inp_bf, BT), reps=30)
        run_cuda(lib_bmm, inp_bf, BT); torch.cuda.synchronize()
        dm, _ = profile_dev(lambda: run_cuda(lib_bmm, inp_bf, BT), reps=30)
        print(f"  {name:18s} | {df:9.1f} | {db:9.1f} {db/df:8.3f}x | "
              f"{dm:9.1f} {dm/df:8.3f}x {dm/db:8.3f}x")
    print("\n  wall (µs):")
    for (name, B, T, H, btc) in CASES:
        inp = make(B, T, H, seed=1)
        inp_bf = to_bf16(inp)
        wf = time_wall(lambda: run_cuda(lib_base, inp, BT), reps=50)
        wb = time_wall(lambda: run_cuda(lib_bf16, inp_bf, BT), reps=50)
        wm = time_wall(lambda: run_cuda(lib_bmm, inp_bf, BT), reps=50)
        print(f"  {name:18s} | fp32 {wf:8.1f} | bf16 {wb:8.1f} ({wb/wf:.3f}x) | "
              f"bmm {wm:8.1f} ({wm/wf:.3f}x, {wm/wb:.3f}x)")


def part_c(lib_base, lib_bf16, lib_bmm):
    print("\n" + "=" * 100)
    print("Part C: static cost")
    print("=" * 100)
    for tag, lib in (("fp32", lib_base), ("bf16", lib_bf16), ("bmm", lib_bmm)):
        smem = lib.b2mma_smem_bytes()
        print(f"  {tag:5s}: smem = {smem} B ({smem/1024:.1f} KB), "
              f"smem-limited blocks/SM ~ {233472 // (smem + 1024)}")
    for tag, extra in (("fp32", []), ("bf16", ["-DB2M_BF16"]),
                       ("bmm", ["-DB2M_BF16", "-DB2M_BF16MMA"])):
        cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
               "-shared", "-Xptxas", "-v"] + extra + \
              [os.path.join(HARNESS, "b2_mma_probe.cu"), "-o", "/tmp/_b2bmm_ptxas.so"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        info = [ln for ln in r.stderr.splitlines()
                if "registers" in ln or "smem" in ln.lower() or "spill" in ln.lower()]
        print(f"  ptxas {tag}: " + " | ".join(info[-3:]))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build([], SO_BASE)
    build(["-DB2M_BF16"], SO_BF16)
    build(["-DB2M_BF16", "-DB2M_BF16MMA"], SO_BMM)
    lib_base = load(SO_BASE)
    lib_bf16 = load(SO_BF16)
    lib_bmm = load(SO_BMM)
    mode = os.environ.get("B2BMM_MODE", "")
    if mode in ("", "a"):
        part_a(lib_base, lib_bf16, lib_bmm)
    if mode in ("", "b"):
        part_b(lib_base, lib_bf16, lib_bmm)
    if mode in ("", "c"):
        part_c(lib_base, lib_bf16, lib_bmm)
    print("\nDONE")


if __name__ == "__main__":
    main()
