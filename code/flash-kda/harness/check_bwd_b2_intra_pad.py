"""P7-94: close the concrete lever docs/07 §96.5 named for the folded `intra`'s K==128
serial tail -- a **smem staging layout change** (padding / degenerate swizzle), not "fewer bytes".

Background.  §96 (P7-93) attributed `-DB2M_INTRA`'s extra ~26.6% at K==128 to the
**shared-memory read pipe** (`l1tex__data_bank_conflicts...shared_op_ld` x1.55, `shared
wavefronts` x1.26) with bytes / tensor / occupancy all neutral -- i.e. a **critical-path serial
tail**, not a pipe saturation.  §96.5 concluded the K==128 fold "要翻正预算应投在 smem staging
的布局（swizzle/pad 或 bf16 staging），不是省字节".  This round does the cheapest such change:
pad the row stride of the folded diagonal's staging tiles.

Mechanism (why a stride pad should help).  The four intra mmas read the staging `siq/sik/sig`
as `sh->sik[j0][col]` with `j0 = kc + t` (`t = lane & 3`) and `col = warp*BW + nt*8 + g`
(`g = lane >> 2`).  Byte bank = `(row*stride + col) % 32`.  At the natural `stride = KD`
(multiple of 32) `row*stride % 32 == 0`, so the four lanes sharing `g` but differing in `t`
all hit **the same bank** (4-way).  A stride `KD + 8` sends `t` to `8*t`, decorrelating them.
The `[BT,BT]` `dAqk` staging (`sda`) has the analogous `sh->sda[g][j0]` pattern; `BT=16` gives
`stride%32 in {0,16}`, so `+4` (`20*g % 32 in {0,20,8,28,16,4,24,12}`) is conflict-free.

What is / is not expected.  The change cannot alter arithmetic: only the **declared** row stride
of the staging arrays changes; reader and writer both index `[row][col]`, so the results must be
**bit-identical** to the unpadded kernel.  It is a pure smem layout change.

Parts
  A  correctness: each pad variant's `prod_intra` output vs the unpadded `prod_intra` -- all 10
     outputs bit-identical (max_abs == 0), 4 shapes incl. the non-divisible `T=250`.
  B  resources: registers / spill / dynamic smem of the variants (smem grows with the pad; must
     stay <= 227 KiB -- at K==128 `prod_intra` is already 1 block/SM).
  C  timing (CUPTI `dev`, kernel-only): `prod_intra` per variant + `prod` reference, K in
     {64,128}, 2 shapes.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_intra_pad.py \
        > .../harness/check_bwd_b2_intra_pad.out.txt 2>&1
    B2IP_MODE=a | b | c
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
sys.path.insert(0, HARNESS)

from check_bwd_b1_bf16_e2e import profile_dev                       # noqa: E402
from check_bwd_b2_intra import (                                    # noqa: E402
    build, ptxas_info, load, run_prod, run_prod_intra, BT, CU)
from check_bwd_ws_reuse import cdiv                                 # noqa: E402

KS = [64, 128]
MAX_SMEM = 232448   # H100 sm_90 opt-in dynamic smem per block (227 KiB)

# (tag, B2M_INTRA_PAD, B2M_INTRA_SPAD) -- PAD is the q/k/g staging, SPAD the [BT,BT] dAqk tile.
VARIANTS = [
    ("pad0",  0, 0),    # baseline (== unpadded kernel)
    ("q8",    8, 0),    # q/k/g stride +8 (t-decorrelation)
    ("q16",  16, 0),
    ("s4",    0, 4),    # dAqk tile stride +4
    ("q8s4",  8, 4),    # both
]


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def build_variant(so, K, pad, spad, extra=("PROD", "HDW", "INTRA")):
    flags = ["-DB2M_%s=1" % e for e in extra] + \
            ["-DB2M_KD=%d" % K, "-DB2M_VD=%d" % K,
             "-DB2M_INTRA_PAD=%d" % pad, "-DB2M_INTRA_SPAD=%d" % spad]
    txt = build("v", CU, flags, so)
    return txt


_CORE_SOS = {}


def core_so(K):
    """The bare GEMM core built **at the same K/V** (its `h/dh` indexing is K-dependent)."""
    if K not in _CORE_SOS:
        so = "/tmp/b2ip_core_K%d.so" % K
        build("core", CU, ["-DB2M_KD=%d" % K, "-DB2M_VD=%d" % K], so)
        _CORE_SOS[K] = so
    return _CORE_SOS[K]


def make_core_dw(inp, so_core):
    lib = load(so_core)
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.b2mma_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                     ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
                     B, T, H, NT, stream())
    return dw


CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
    ("T64 H1 (single chunk)", 1, 64, 1),
]
OUTS = ("dqg", "dkg", "dg", "db", "dv2", "dvb", "dkgb", "dA", "dApre", "dgk")


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def part_a():
    print("=" * 92)
    print("Part A: correctness -- each pad variant vs unpadded prod_intra (must be bit-identical)")
    print("=" * 92)
    worst = 0.0
    for K in KS:
        sos = {}
        for tag, pad, spad in VARIANTS:
            sos[tag] = "/tmp/b2ip_a_%s_K%d.so" % (tag, K)
            build_variant(sos[tag], K, pad, spad)
            lib = load(sos[tag])
            lib.b2mma_smem_bytes.restype = ctypes.c_int
            print("  built %-6s K=%3d  smem=%7d B  %.1f KiB"
                  % (tag, K, lib.b2mma_smem_bytes(), lib.b2mma_smem_bytes() / 1024.0))
        base = load(sos["pad0"])
        for (name, B, T, H) in CASES:
            inp = _mk(B, T, H, K)
            dw = make_core_dw(inp, core_so(K))
            ref = run_prod_intra(base, inp, dw)
            print("\n# K=%d %s  NT=%d" % (K, name, cdiv(T, BT)))
            for tag, pad, spad in VARIANTS:
                if tag == "pad0":
                    continue
                lib = load(sos[tag])
                v = run_prod_intra(lib, inp, dw)
                w = max(maxabs(v[f], ref[f]) for f in OUTS)
                worst = max(worst, w)
                flag = "BIT-IDENTICAL" if w == 0.0 else "DIFF"
                print("    %-6s vs pad0  worst max_abs=%.3e  [%s]" % (tag, w, flag))
    print("\nPart A worst max_abs over all variants/cases: %.3e  [%s]"
          % (worst, "BIT-IDENTICAL" if worst == 0.0 else "MISMATCH"))


def _mk(B, T, H, K):
    """P7-94 inputs at an arbitrary K (the P7-92 `make` hard-bakes K=V=128)."""
    from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs
    inp = dqkg_make_inputs(B, T, H, K, K, BT, "cuda", dtype=torch.float32, seed=1)
    inp = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
    torch.manual_seed(101)
    NT = cdiv(T, BT)
    o_t = torch.arange(BT, device="cuda")
    low_incl = (o_t[:, None] >= o_t[None, :]).float()
    dAqk = torch.randn(B, NT, H, BT, BT, device="cuda") * low_incl
    dAqk = dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    inp["dAqk"] = dAqk
    return inp


def part_b():
    print("\n" + "=" * 92)
    print("Part B: resources (registers / spill / dynamic smem)")
    print("=" * 92)
    print("  %-16s %6s %7s %9s %10s" % ("build", "regs", "spill", "smem", "occupancy"))
    for K in KS:
        for tag, pad, spad in VARIANTS:
            so = "/tmp/b2ip_b_%s_K%d.so" % (tag, K)
            txt = build_variant(so, K, pad, spad)
            r, sp = ptxas_info(txt)
            lib = load(so)
            lib.b2mma_smem_bytes.restype = ctypes.c_int
            smem = lib.b2mma_smem_bytes()
            occ = "INFEASIBLE" if smem > MAX_SMEM else "ok"
            print("  K=%3d %-10s %6d %7d %9d %10s" % (K, tag, r, sp, smem, occ))


def part_c():
    print("\n" + "=" * 92)
    print("Part C: timing (CUPTI dev, kernel-only) -- prod_intra per variant + prod reference")
    print("=" * 92)
    tc = [("B8 T2048 H8", 8, 2048, 8), ("B2 T1024 H8", 2, 1024, 8)]
    for K in KS:
        so_prod = "/tmp/b2ip_c_prod_K%d.so" % K
        build("prod", CU, ["-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_KD=%d" % K,
                           "-DB2M_VD=%d" % K], so_prod)
        libp = load(so_prod)
        libs = {"__prod__": libp}
        for tag, pad, spad in VARIANTS:
            so = "/tmp/b2ip_c_%s_K%d.so" % (tag, K)
            build_variant(so, K, pad, spad)
            libs[tag] = load(so)
        for (name, B, T, H) in tc:
            inp = _mk(B, T, H, K)
            dw = make_core_dw(inp, core_so(K))
            res = {}

            def bench(tag):
                lib = libs[tag]
                fn = (lambda: run_prod(lib, inp, dw)) if tag == "__prod__" \
                    else (lambda: run_prod_intra(lib, inp, dw))
                fn(); torch.cuda.synchronize()
                d, _ = profile_dev(fn, reps=30)
                res[tag] = d

            for tag in ["__prod__"] + [v[0] for v in VARIANTS]:
                bench(tag)
            base = res["pad0"]
            print("\n# K=%d %s" % (K, name))
            print("  %-8s %10s %10s" % ("variant", "dev(us)", "vs pad0"))
            for tag in ["__prod__"] + [v[0] for v in VARIANTS]:
                rel = res[tag] / base
                print("  %-8s %10.1f %10.3f" % (tag, res[tag], rel))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    pass  # core built per-K via core_so()
    mode = os.environ.get("B2IP_MODE", "")
    if mode in ("", "a"):
        part_a()
    if mode in ("", "b"):
        part_b()
    if mode in ("", "c"):
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
