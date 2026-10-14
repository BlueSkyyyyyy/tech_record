"""P7-77: compose the native-bf16 mma (`-DB2M_BF16MMA`) with the TMA operand path
(`-DB2M_TMA`) -- the last `#error` of the bf16-native entry.

P7-74 added a native bf16 `mma.sync.m16n8k16` to the base GEMM core but kept it mutually
exclusive with TMA: `rd2` (the bf16 fragment packer) only knew the non-TMA `OPS` padded
row-major layout, so the combination was rejected by `#error`.  This round generalizes `rd2`
to the same `[NAT][rows][TAW]` 128 B-swizzle atom decode that `rd` uses (both bf16 of a k16
pair fall in the same atom and same 16 B swizzle unit because the fragment column is even),
which removes the exclusion.  No arithmetic changes: the staged bf16 bytes and the fragment
values are exactly those of the non-TMA bf16 mma path, only the smem layout differs.

The comparison is therefore clean: `bmm_tma64` (bf16 mma + TMA staging) must be
**bit-identical** to `bmm64` (bf16 mma + cp.async staging); both differ from the tf32-mma
builds only by the fp32 accumulation order (P7-74).

  Part A  correctness: {bmm_tma64 == bmm64} bit-identity + vs torch(bf16) noise floor.
  Part B  kernel-only timing: dev (CUPTI), the four bf16 / fp32 pairs.
  Part C  static smem + registers.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_bf16mma_tma.py \
        > .../harness/check_bwd_b2_bf16mma_tma.out.txt 2>&1
    B2BT_MODE=a | b | c
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

BT = 16
OP_KEYS = ("v", "v_new", "h", "dh", "do", "dv")
FIELDS = ("dq", "dk", "dw", "dgk")

CASES = [
    ("B8 T2048 H8", 8, 2048, 8),
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]

# (tag, extra flags, input dtype).  All at BVK=64 so the only variable is the staging mechanism
# / mma ISA; TMA builds need `-lcuda` (cuTensorMapEncodeTiled).
VARIANTS = [
    ("bf16_64", ["-DB2M_BF16=1", "-DB2M_BVK=64"], "bf16"),
    ("bmm64", ["-DB2M_BF16=1", "-DB2M_BF16MMA=1", "-DB2M_BVK=64"], "bf16"),
    ("bf16_tma64", ["-DB2M_TMA=1", "-DB2M_BF16=1", "-DB2M_BVK=64"], "bf16"),
    ("bmm_tma64", ["-DB2M_TMA=1", "-DB2M_BF16=1", "-DB2M_BF16MMA=1", "-DB2M_BVK=64"], "bf16"),
]


def build(tag, extra):
    so = f"/tmp/b2_bmmtma_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + ["-lcuda", os.path.join(HARNESS, "b2_mma_probe.cu"),
           "-o", so]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed")
    return so


def load(so):
    lib = ctypes.CDLL(so)
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + \
        [ctypes.c_void_p]
    lib.b2mma_smem_bytes.restype = ctypes.c_int
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


def to_bf16(inp):
    out = dict(inp)
    for k in OP_KEYS:
        out[k] = inp[k].to(torch.bfloat16)
    return out


def part_a(libs):
    print("=" * 96)
    print("Part A: correctness -- {bmm_tma64 == bmm64} and {bmm64 ~ bf16_64} (fp32 accum order)")
    print("=" * 96)
    ident = {}
    worst = {}
    for (name, B, T, H) in CASES:
        inp_f = make(B, T, H, seed=1)
        inp_b = to_bf16(inp_f)
        ref_b = ref_core(inp_b, BT)
        cu = {tag: run_cuda(libs[tag], inp_b) for tag, _e, _d in VARIANTS}
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for fld in FIELDS:
            i_tma = (cu["bmm_tma64"][fld].float() - cu["bmm64"][fld].float()).abs().max().item()
            i_isa = (cu["bmm64"][fld].float() - cu["bf16_64"][fld].float()).abs().max().item()
            _, mr_bt = stats(f"  bmm_tma64 vs torch(bf16) {fld}", cu["bmm_tma64"][fld], ref_b[fld])
            _, mr_b = stats(f"  bmm64     vs torch(bf16) {fld}", cu["bmm64"][fld], ref_b[fld])
            print(f"  {fld:4s}  bmm_tma64-vs-bmm64 max_abs={i_tma:.3e}   "
                  f"bmm64-vs-bf16_64 max_abs={i_isa:.3e}   "
                  f"(vs torch: bmm_tma {mr_bt:.3e} / bmm {mr_b:.3e})")
            ident[f"t_{fld}"] = max(ident.get(f"t_{fld}", 0.0), i_tma)
            ident[f"s_{fld}"] = max(ident.get(f"s_{fld}", 0.0), i_isa)
            worst[f"bt_{fld}"] = max(worst.get(f"bt_{fld}", 0.0), mr_bt)
    print("\nworst over all cases:")
    for fld in FIELDS:
        print(f"  {fld:4s}  bmm_tma64==bmm64 {ident['t_'+fld]:.3e}   "
              f"bmm64-vs-bf16_64 {ident['s_'+fld]:.3e}   "
              f"| bmm_tma64/torch {worst['bt_'+fld]:.3e}")


def part_b(libs):
    print("\n" + "=" * 96)
    print("Part B: kernel-only timing -- bf16_64 / bmm64 / bf16_tma64 / bmm_tma64")
    print("=" * 96)
    print(f"  {'case':18s} | {'bf16_64':>8s} | {'bmm64':>8s} | {'bf16_tma':>8s} | "
          f"{'bmm_tma':>8s} | {'bmm/bf16':>9s} {'bmmtma/bmm':>10s} {'tma/bf16':>9s}")
    for (name, B, T, H) in CASES:
        inp_f = make(B, T, H, seed=1)
        inp_b = to_bf16(inp_f)
        for tag, lib in libs.items():
            run_cuda(lib, inp_b)
        torch.cuda.synchronize()
        dv = {}
        for tag, lib in libs.items():
            dv[tag], _ = profile_dev(lambda lib=lib: run_cuda(lib, inp_b), reps=40)
        print(f"  {name:18s} | {dv['bf16_64']:8.1f} | {dv['bmm64']:8.1f} | "
              f"{dv['bf16_tma64']:8.1f} | {dv['bmm_tma64']:8.1f} | "
              f"{dv['bmm64']/dv['bf16_64']:9.3f} {dv['bmm_tma64']/dv['bmm64']:10.3f} "
              f"{dv['bf16_tma64']/dv['bf16_64']:9.3f}")


def part_c(libs):
    print("\n" + "=" * 96)
    print("Part C: static cost")
    print("=" * 96)
    for tag, _extra, _dt in VARIANTS:
        smem = libs[tag].b2mma_smem_bytes()
        print(f"  {tag:10s}: smem = {smem} B ({smem/1024:.1f} KB)")
    for tag, extra, _ in VARIANTS:
        cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
               "-shared", "-Xptxas", "-v", "-lcuda"] + extra + \
              [os.path.join(HARNESS, "b2_mma_probe.cu"), "-o", "/tmp/_b2bmmtma_ptxas.so"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        info = [ln for ln in r.stderr.splitlines()
                if "registers" in ln or "smem" in ln.lower() or "spill" in ln.lower()]
        print(f"  ptxas {tag:10s}: " + " | ".join(info[-3:]))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra, _dt in VARIANTS:
        libs[tag] = load(build(tag, extra))
    mode = os.environ.get("B2BT_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs)
    if mode in ("", "c"):
        part_c(libs)
    print("\nDONE")


if __name__ == "__main__":
    main()
