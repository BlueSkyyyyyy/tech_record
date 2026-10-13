"""P7-76: close the last `#error` of the bf16-native operand entry -- **bf16 staging × TMA input**.

`b2_mma_probe.cu` gained a bf16 operand entry (P7-73 `-DB2M_BF16`) and a TMA operand path
(P7-49 `-DB2M_TMA`), but the two were mutually exclusive (`#error`): the TMA maps were fp32
(32-element = 128 B atoms).  A bf16 TMA box is **64 elements = 128 B**, so the atom layout is
`NAT = BVK/64` and the swizzle decode (`b ^= ((b>>7)&7)<<4`) is byte-identical (the 128 B atom
row stride shifts `row` into bit 7 either way).  This round only makes the two composable and
verifies the composition; it does not touch any arithmetic.

The comparison is clean because both bf16 builds stage the *same bf16 values* and run the same
`mma.sync.m16n8k8.tf32` in the same k-order -- only the smem layout differs.  So `bftma64` must be
**bit-identical** to `bf16_64` (the cp.async bf16 build), and `tma64` **bit-identical** to
`base64` (fp32), the same identity P7-49 established at BVK=32.

  Part A  correctness: bit-identity pairs + vs torch (allow_tf32=False) noise floor.
  Part B  kernel-only timing: dev (CUPTI) + wall, all four variants.
  Part C  static smem + registers.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=2 python .../harness/check_bwd_b2_bf16_tma.py \
        > .../harness/check_bwd_b2_bf16_tma.out.txt 2>&1
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

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
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
# / operand dtype; TMA builds need `-lcuda` (cuTensorMapEncodeTiled).
VARIANTS = [
    ("base64", [], "fp32"),
    ("tma64", ["-DB2M_TMA=1", "-DB2M_BVK=64"], "fp32"),
    ("bf16_64", ["-DB2M_BF16=1", "-DB2M_BVK=64"], "bf16"),
    ("bftma64", ["-DB2M_TMA=1", "-DB2M_BF16=1", "-DB2M_BVK=64"], "bf16"),
]


def build(tag, extra):
    so = f"/tmp/b2_bftma_{tag}.so"
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
    print("Part A: correctness -- {bftma64==bf16_64} and {tma64==base64} must be BIT-IDENTICAL")
    print("=" * 96)
    ident = {}
    worst = {}
    for (name, B, T, H) in CASES:
        inp_f = make(B, T, H, seed=1)
        inp_b = to_bf16(inp_f)
        ref_f = ref_core(inp_f, BT)
        ref_b = ref_core(inp_b, BT)
        cu = {
            "base64": run_cuda(libs["base64"], inp_f),
            "tma64": run_cuda(libs["tma64"], inp_f),
            "bf16_64": run_cuda(libs["bf16_64"], inp_b),
            "bftma64": run_cuda(libs["bftma64"], inp_b),
        }
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for fld in FIELDS:
            i_bf = (cu["bftma64"][fld].float() - cu["bf16_64"][fld].float()).abs().max().item()
            i_t = (cu["tma64"][fld].float() - cu["base64"][fld].float()).abs().max().item()
            _, mr_bft = stats(f"  bftma64 vs torch(bf16) {fld}", cu["bftma64"][fld], ref_b[fld])
            _, mr_tma = stats(f"  tma64   vs torch(fp32) {fld}", cu["tma64"][fld], ref_f[fld])
            print(f"  {fld:4s}  bftma64-vs-bf16_64 max_abs={i_bf:.3e}   "
                  f"tma64-vs-base64 max_abs={i_t:.3e}   "
                  f"(vs torch: bftma {mr_bft:.3e} / tma {mr_tma:.3e})")
            ident[f"bf_{fld}"] = max(ident.get(f"bf_{fld}", 0.0), i_bf)
            ident[f"t_{fld}"] = max(ident.get(f"t_{fld}", 0.0), i_t)
            worst[f"b_{fld}"] = max(worst.get(f"b_{fld}", 0.0), mr_bft)
            worst[f"t_{fld}"] = max(worst.get(f"t_{fld}", 0.0), mr_tma)
    print("\nworst over all cases:")
    for fld in FIELDS:
        print(f"  {fld:4s}  bftma64==bf16_64 {ident['bf_'+fld]:.3e}   "
              f"tma64==base64 {ident['t_'+fld]:.3e}   "
              f"| bftma64/torch {worst['b_'+fld]:.3e}   tma64/torch {worst['t_'+fld]:.3e}")


def part_b(libs):
    print("\n" + "=" * 96)
    print("Part B: kernel-only timing -- base64 / tma64 / bf16_64 / bftma64")
    print("=" * 96)
    print(f"  {'case':18s} | {'base64':>8s} | {'tma64':>8s} | {'bf16_64':>8s} | "
          f"{'bftma64':>8s} | {'tma64/base':>10s} {'bftma/bf16':>10s}")
    for (name, B, T, H) in CASES:
        inp_f = make(B, T, H, seed=1)
        inp_b = to_bf16(inp_f)
        ins = {"base64": inp_f, "tma64": inp_f, "bf16_64": inp_b, "bftma64": inp_b}
        for tag, lib in libs.items():
            run_cuda(lib, ins[tag])
        torch.cuda.synchronize()
        dv = {}
        for tag, lib in libs.items():
            dv[tag], _ = profile_dev(lambda lib=lib, tag=tag: run_cuda(lib, ins[tag]), reps=40)
        print(f"  {name:18s} | {dv['base64']:8.1f} | {dv['tma64']:8.1f} | "
              f"{dv['bf16_64']:8.1f} | {dv['bftma64']:8.1f} | "
              f"{dv['tma64']/dv['base64']:10.3f} {dv['bftma64']/dv['bf16_64']:10.3f}")


def part_c(libs):
    print("\n" + "=" * 96)
    print("Part C: static cost")
    print("=" * 96)
    for tag, _extra, _dt in VARIANTS:
        smem = libs[tag].b2mma_smem_bytes()
        print(f"  {tag:8s}: smem = {smem} B ({smem/1024:.1f} KB)")
    for tag, extra, _ in VARIANTS:
        cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
               "-shared", "-Xptxas", "-v", "-lcuda"] + extra + \
              [os.path.join(HARNESS, "b2_mma_probe.cu"), "-o", "/tmp/_b2bftma_ptxas.so"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        info = [ln for ln in r.stderr.splitlines()
                if "registers" in ln or "smem" in ln.lower() or "spill" in ln.lower()]
        print(f"  ptxas {tag:8s}: " + " | ".join(info[-3:]))


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
