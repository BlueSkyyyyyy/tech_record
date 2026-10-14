"""P7-89: close the last boundary recorded at the end of P7-88 (`docs/07` §91.6) -- the
**bf16-native operand entry (`-DB2M_BF16`) and native bf16 mma (`-DB2M_BF16MMA`) crossed with
`K != 128`** on the K2-shaped CUDA B2 base GEMM core.

P7-88 parameterized the base GEMM core in `K` (`NBW = BW/8`, generic `dgk` column reduction) but
only ran the fp32/tf32 build; P7-73/74/76/77 added the bf16 entry (staged bf16 operands, tf32
mma) and native bf16 mma (`m16n8k16`) but only ever at `K = V = 128`.  This round races the
existing bf16 builds against the P7-88 `K`-generic base core.

The key invariant is unchanged from P7-73/74: **bf16 (8-bit mantissa) is a strict subset of
tf32 (10-bit)**, so widening bf16->fp32 (exact) before the tf32 `cvt.rna` reproduces bf16-mma
products bit-for-bit -- replacing the deployed `.float()`-then-fp32-core with a bf16 core must
not change the answer.  `K` only appears as the `h/dh` row count (= output N), so it never enters
the V contraction and the bf16 staging (`OPS = BVK + 8`) is `K`-clean.

Parts:
  A  correctness: bf16/bf16mma CUDA on bf16 operands vs torch fp32 of the *same* bf16 operands
     (exact golden), and vs the **fp32 core on `bf16.float()` operands** (the deployment drop-in
     question = `max_abs` must be 0).
  B  kernel-only timing: bf16mma vs fp32, `K` scan at `V = 128` + crossing at `K = 128`.
  C  static: smem bytes (scales with `KD` h/dh rows) + registers of the builds.
  D  TMA cross: `-DB2M_BF16 -DB2M_BF16MMA -DB2M_TMA` (bf16 128 B atoms => `BVK=64`) vs the
     cp.async bf16mma build -- bit-for-bit (the P7-88 staging-only-swaps-transport claim).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=3 python .../harness/check_bwd_b2_k128_bf16.py \
        > .../harness/check_bwd_b2_k128_bf16.out.txt 2>&1
    B2KB_MODE=a | b | c | d
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
from check_bwd_b2_core import ref_core                                   # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                               # noqa: E402
from check_bwd_b1_bf16_e2e import time_wall, profile_dev                 # noqa: E402

BT = 16
NTHR = 256
OP_KEYS = ("v", "v_new", "h", "dh", "do", "dv")

# (K, V) pairs: K-scan at V=128 (isolates K), then the crossing cases.
KV = [(64, 128), (128, 128), (256, 128), (64, 256), (256, 64)]
KV_TIMING = [(64, 128), (128, 128), (256, 128), (128, 64), (128, 256)]

# (tag, nvcc flags).  bf16mma_tma needs BVK=64 (bf16 TMA atom = 64 elems = 128 B).
TAGS = [
    ("fp32",       []),
    ("bf16",       ["-DB2M_BF16"]),
    ("bf16mma",    ["-DB2M_BF16", "-DB2M_BF16MMA"]),
    ("bf16mma_tma", ["-DB2M_BF16", "-DB2M_BF16MMA", "-DB2M_TMA", "-DB2M_BVK=64"]),
]
TAG_FLAGS = dict(TAGS)

CASES = [
    ("B8 T2048 H8", 8, 2048, 8),
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]


def so_path(tag, K, V):
    return os.path.join("/tmp", "b2kb_%s_%d_%d.so" % (tag, K, V))


def build(tag, K, V):
    src = os.path.join(HARNESS, "b2_mma_probe.cu")
    extra = ["-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V] + TAG_FLAGS[tag]
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + ["-lcuda", src, "-o", so_path(tag, K, V)]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed for %s K=%d V=%d" % (tag, K, V))


def load(tag, K, V):
    lib = ctypes.CDLL(so_path(tag, K, V))
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


def make(B, T, H, K, V, seed=1):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def to_bf16(inp):
    out = dict(inp)
    for k in OP_KEYS:
        out[k] = inp[k].to(torch.bfloat16)
    return out


def widen(inp):
    out = dict(inp)
    for k in OP_KEYS:
        out[k] = inp[k].float()
    return out


def part_a(libs):
    print("=" * 96)
    print("Part A: correctness -- bf16 / bf16mma base GEMM core at K != 128")
    print("  ref = torch fp32 of the *same bf16* operands (allow_tf32=False)")
    print("  drop-in = bf16 core on bf16 operands  vs  fp32 core on `bf16.float()` (max_abs must be 0)")
    print("=" * 96)
    worst = {}
    for (K, V) in KV:
        print(f"\n#### K = {K}, V = {V} ####")
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            inp_bf = to_bf16(inp)
            ref_b = ref_core(inp_bf, BT)
            cu_w = run_cuda(libs[("fp32", K, V)], widen(inp_bf), BT, K, V)
            outs = {"bf16": run_cuda(libs[("bf16", K, V)], inp_bf, BT, K, V),
                    "bf16mma": run_cuda(libs[("bf16mma", K, V)], inp_bf, BT, K, V)}
            print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
            for tag, cu in outs.items():
                for fld in ("dq", "dk", "dw", "dgk"):
                    _, mr = stats(f"  [{tag}] {fld} vs torch(bf16)", cu[fld], ref_b[fld])
                    ma, _ = stats(f"  [{tag}] {fld} vs fp32cuda(drop-in)", cu[fld], cu_w[fld])
                    worst[f"{tag}_{K}_{V}_{fld}"] = max(worst.get(f"{tag}_{K}_{V}_{fld}", 0.0), mr)
                    worst[f"{tag}d_{K}_{V}_{fld}"] = max(worst.get(f"{tag}d_{K}_{V}_{fld}", 0.0), ma)
    print("\nworst max_rel vs torch(bf16) / worst max_abs vs fp32-core-drop-in:")
    print(f"  {'tag':10s} {'K':>4s} {'V':>4s} {'field':>5s}  {'max_rel':>10s} {'dropin max_abs':>14s}")
    for (K, V) in KV:
        for tag in ("bf16", "bf16mma"):
            for fld in ("dq", "dk", "dw", "dgk"):
                print(f"  {tag:10s} {K:4d} {V:4d} {fld:>5s}  "
                      f"{worst[f'{tag}_{K}_{V}_{fld}']:10.3e} "
                      f"{worst[f'{tag}d_{K}_{V}_{fld}']:14.3e}")


def part_b(libs):
    print("\n" + "=" * 96)
    print("Part B: kernel-only timing -- bf16mma vs fp32 base GEMM core (K != 128)")
    print("=" * 96)
    print(f"  {'case':18s} {'K':>4s} {'V':>4s} | {'fp32 dev':>9s} | {'bf16 dev':>9s} "
          f"{'bf16/fp32':>10s} | {'bf16mma dev':>12s} {'mma/fp32':>9s}")
    for (K, V) in KV_TIMING:
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            inp_bf = to_bf16(inp)
            run_cuda(libs[("fp32", K, V)], inp, BT, K, V); torch.cuda.synchronize()
            df, _ = profile_dev(lambda: run_cuda(libs[("fp32", K, V)], inp, BT, K, V), reps=30)
            db, _ = profile_dev(lambda: run_cuda(libs[("bf16", K, V)], inp_bf, BT, K, V), reps=30)
            dm, _ = profile_dev(lambda: run_cuda(libs[("bf16mma", K, V)], inp_bf, BT, K, V), reps=30)
            print(f"  {name:18s} {K:4d} {V:4d} | {df:9.1f} | {db:9.1f} {db/df:9.3f}x | "
                  f"{dm:12.1f} {dm/df:8.3f}x")


def part_c():
    print("\n" + "=" * 96)
    print("Part C: static cost (smem scales with KD h/dh rows; bf16 halves operands)")
    print("=" * 96)
    for (K, V) in KV:
        line = f"  K={K:4d} V={V:4d}:"
        for tag, _ in TAGS:
            if tag == "bf16mma_tma" and not (K * V <= 128 * 256):
                continue
            try:
                lib = load(tag, K, V)
                smem = lib.b2mma_smem_bytes()
                line += f"  {tag}={smem/1024:6.1f}KB"
            except OSError:
                line += f"  {tag}=n/a"
        print(line)
    print("\n  ptxas registers (B8T2048H8-relevant build):")
    for tag, flags in TAGS:
        cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
               "-shared", "-Xptxas", "-v", "-DB2M_KD=64", "-DB2M_VD=128"] + flags + \
              [os.path.join(HARNESS, "b2_mma_probe.cu"), "-o", "/tmp/_b2kb_ptxas.so", "-lcuda"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        info = [ln for ln in r.stderr.splitlines()
                if "registers" in ln.lower() or "spill" in ln.lower()]
        print(f"    {tag:11s} (K=64): " + " | ".join(x.strip() for x in info[-2:]))


def part_d(libs):
    print("\n" + "=" * 96)
    print("Part D: TMA cross -- bf16mma_tma (BVK=64) vs bf16mma (cp.async BVK=32), K-scan V=128")
    print("=" * 96)
    worst = {}
    for (K, V) in [(64, 128), (128, 128), (256, 128)]:
        libm = load("bf16mma", K, V)
        libt = load("bf16mma_tma", K, V)
        print(f"\n#### K = {K}, V = {V} ####")
        for (name, B, T, H) in CASES:
            inp_bf = to_bf16(make(B, T, H, K, V, seed=1))
            cm = run_cuda(libm, inp_bf, BT, K, V)
            ct = run_cuda(libt, inp_bf, BT, K, V)
            for fld in ("dq", "dk", "dw", "dgk"):
                ma, _ = stats(f"  [{K},{V}] {fld} TMA vs cpasync", ct[fld], cm[fld])
                worst[f"{K}_{V}_{fld}"] = max(worst.get(f"{K}_{V}_{fld}", 0.0), ma)
    print("\nworst max_abs(TMA - cp.async):")
    for (K, V) in [(64, 128), (128, 128), (256, 128)]:
        print("  K=%4d V=%4d: " % (K, V) +
              "  ".join(f"{fld}={worst[f'{K}_{V}_{fld}']:.3e}" for fld in ("dq", "dk", "dw", "dgk")))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    mode = os.environ.get("B2KB_MODE", "")
    # only build what the selected part needs
    build_pairs = {
        "a": [("fp32", K, V) for (K, V) in KV] + [("bf16", K, V) for (K, V) in KV] +
             [("bf16mma", K, V) for (K, V) in KV],
        "b": [(t, K, V) for (K, V) in KV_TIMING for t in ("fp32", "bf16", "bf16mma")],
        "c": [(t, K, V) for (K, V) in KV for t, _ in TAGS],
        "d": [("bf16mma", K, V) for (K, V) in [(64, 128), (128, 128), (256, 128)]] +
             [("bf16mma_tma", K, V) for (K, V) in [(64, 128), (128, 128), (256, 128)]],
    }
    if mode == "":
        pairs = [p for m in ("a", "b", "c", "d") for p in build_pairs[m]]
    else:
        pairs = build_pairs.get(mode, [])
    seen = set()
    libs = {}
    for (tag, K, V) in pairs:
        if (tag, K, V) in seen:
            continue
        seen.add((tag, K, V))
        build(tag, K, V)
        libs[(tag, K, V)] = load(tag, K, V)
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs)
    if mode in ("", "c"):
        part_c()
    if mode in ("", "d"):
        part_d(libs)
    print("\nDONE")


if __name__ == "__main__":
    main()
