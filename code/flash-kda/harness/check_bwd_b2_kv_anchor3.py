"""P7-103: close the last boundary §105.7 (P7-102) explicitly left open for the K2-shaped CUDA B2
production kernel (`b2_mma_probe.cu`) at `K != V`: the **`bf16mma` + TMA + `BVK=64`** variant.

§105.7 says verbatim:

  "单 shape 族 (...) `bf16mma` 走非 TMA 路径（`BVK=32`）；`bf16mma_tma`（`BVK=64`）本节**未做数值
   锚定**（只在本文件其它处/编译层验证过）。"

P7-102 even *declared* the macro `BF16_MMA_TMA` (`check_bwd_b2_kv_anchor2.py:65`) but never built or
ran it.  This round finally anchors it.  The subtlety: the TMA+bf16 path is hard-constrained to
`BVK in {64,128}` (`b2_mma_probe.cu:473-474`), while the plain `bf16mma` default is `BVK=32`
(`:85-86`), so comparing `bf16mma_tma` directly to `bf16mma` conflates **two** changes.  We add the
cp.async `BVK=64` control to separate them:

  Part A  three builds per `(K,V)` on the bf16-valued inputs:
            * `bf16mma`        = `-DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_BF16MMA`        (BVK=32, cp.async)
            * `bf16mma_bvk64`  = the above + `-DB2M_BVK=64`                              (BVK=64, cp.async)
            * `bf16mma_tma`    = the above + `-DB2M_TMA=1`                               (BVK=64, TMA)
          and reports
            * `bf16mma_tma` vs `bf16mma_bvk64`: isolates the **staging engine** (expect bit-identical)
            * `bf16mma_bvk64` vs `bf16mma`:     isolates **BVK** = the V-reduction accumulation order
            * all three vs **pure torch** (`wy_dqkg_torch`, bf16-valued inputs; FLA is not same-dtype)
  Part B  resources: registers / spill / smem of the three builds over the `(K,V)` grid.
  Part C  timing (CUPTI `dev` + event `wall`): the two engine/tiling changes in `us`.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=3 python .../harness/check_bwd_b2_kv_anchor3.py \
        > .../harness/check_bwd_b2_kv_anchor3.out.txt 2>&1
    B2KA3_MODE=a | b | c
"""
import ctypes
import os
import re
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs        # noqa: E402
from check_bwd_dqkg_kernel import fla_wy_dqkg, wy_dqkg_torch           # noqa: E402
from check_bwd_ws_reuse import cdiv                                    # noqa: E402
from check_bwd_b2_prod import run_prod, ptr, stream                    # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev, time_wall               # noqa: E402

BT = 16
SCALE = 1.0 / (128 ** 0.5)          # deployment scale (matches run_prod/run_prod_intra)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")

PROD = ["-DB2M_PROD=1", "-DB2M_HDW=1"]
BF16 = PROD + ["-DB2M_BF16=1"]
BF16MMA = BF16 + ["-DB2M_BF16MMA=1"]                            # BVK=32 (default), cp.async
BF16MMA64 = BF16 + ["-DB2M_BF16MMA=1", "-DB2M_BVK=64"]         # BVK=64, cp.async
BF16MMA_TMA = BF16MMA64 + ["-DB2M_TMA=1"]                      # BVK=64, TMA

# five bf16-native streamed operands (`b2_mma_probe.cu:154-162`); q/k/g/beta/A stay fp32.
BF16_KEYS = ("v_new", "h", "dh", "do", "dv")

KV = [(64, 256), (256, 64), (64, 128), (128, 64), (128, 128)]
CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]
FIELDS = ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk")


def build_v(extra, K, V, tag):
    so = "/tmp/b2ka3_%s_%d_%d.so" % (tag, K, V)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v", "-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V]
    cmd += extra + ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed: tag=%s K=%d V=%d" % (tag, K, V))
    return so, r.stderr


def build(extra, K, V, tag):
    return build_v(extra, K, V, tag)[0]


def smem_bytes(so):
    lib = ctypes.CDLL(so)
    lib.b2mma_smem_bytes.restype = ctypes.c_int
    return lib.b2mma_smem_bytes()


def ptxas_reg(txt):
    m = re.search(r"Used (\d+) registers", txt)
    sp = re.search(r"(\d+) bytes spill stores", txt)
    return (int(m.group(1)) if m else -1, int(sp.group(1)) if sp else 0)


def make(B, T, H, K, V, seed=1):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    inp = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
    torch.manual_seed(seed + 100)
    NT = cdiv(T, BT)
    o_t = torch.arange(BT, device="cuda")
    low_incl = (o_t[:, None] >= o_t[None, :]).float()
    dAqk = torch.randn(B, NT, H, BT, BT, device="cuda") * low_incl
    inp["dAqk"] = dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    return inp


def quantize(inp):
    """Three operand dicts carrying the **same numeric values** but the correct "on the wire" layout
    per build family.

    `op_t`-typed operands (`do/vn/dv/h/dh`) are bf16-staged under every `-DB2M_BF16` build (`rd()`).
    The raw-`v` scratch operand goes through `rds()`: **fp32** under non-TMA `-DB2M_BF16` (P7-75, §78),
    **bf16** under `-DB2M_BF16 -DB2M_TMA` (P7-76, one bf16 tensor map covers every operand).  So:

      * `inp_cp`  : 5 streamed operands **bf16 wire**, raw `v` **fp32 wire** (bf16-valued) --
                    what the cp.async `bf16mma`/`bf16mma_bvk64` builds read.
      * `inp_tma` : the same values, raw `v` **bf16 wire** -- what `bf16mma_tma` reads.
      * `inp_ref` : the same values all as fp32 (bf16-valued) -- for the pure-torch anchor, which
                    does `.float()` anyway.
    Feeding different wire layouts with identical values is what lets TMA-vs-cp.async be compared as
    a pure staging change (feeding an fp32 `v` to the TMA build misreads the bytes -> NaN).
    """
    inp_cp, inp_tma, inp_ref = dict(inp), dict(inp), dict(inp)
    for k in BF16_KEYS:
        b = inp[k].bfloat16()
        inp_cp[k] = b.contiguous()
        inp_tma[k] = b.contiguous()
        inp_ref[k] = b.float().contiguous()
    vb = inp["v"].bfloat16()
    inp_cp["v"] = vb.float().contiguous()   # non-TMA keeps raw v fp32
    inp_tma["v"] = vb.contiguous()          # TMA stages raw v bf16
    inp_ref["v"] = vb.float().contiguous()
    return inp_cp, inp_tma, inp_ref


def all_bf16(inp):
    """FLA's fused wy_dqkg kernel requires a uniform operand dtype -- its bf16 path quantizes
    every operand (incl. q/k/g/beta/A), unlike the production kernel (five streamed operands)."""
    out = {}
    for k, v in inp.items():
        out[k] = v.contiguous() if k == "dAqk" else v.bfloat16().contiguous()
    return out


def stats(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    denom = b.float().abs().max().item()
    print("    %-38s max_abs=%.3e  max_rel=%.3e  ref_max=%.3e"
          % (name, d, d / (denom + 1e-12), denom))
    return d, d / (denom + 1e-12)


def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    print("    %-38s max_abs=%.3e  [%s]" % (name, d, "BIT" if d == 0.0 else "DIFF"))
    return d


def zero_dw(B, T, H, K):
    return torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)


# ------------------------------------------------------------------------- Part A
def part_a():
    print("=" * 108)
    print("Part A: bf16mma vs bf16mma_bvk64 vs bf16mma_tma at K!=V -- isolate BVK & TMA + external anchor")
    print("=" * 108)
    worst_ext = {}
    worst_bit = {}
    for (K, V) in KV:
        so_m = build(BF16MMA, K, V, "bf16mma")
        so_m64 = build(BF16MMA64, K, V, "bf16mma64")
        so_mt = build(BF16MMA_TMA, K, V, "bf16mmatma")
        libm, libm64, libmt = ctypes.CDLL(so_m), ctypes.CDLL(so_m64), ctypes.CDLL(so_mt)
        print("\n#### K = %d, V = %d   smem bf16mma=%d bf16mma_bvk64=%d bf16mma_tma=%d ####"
              % (K, V, smem_bytes(so_m), smem_bytes(so_m64), smem_bytes(so_mt)))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            inp_cp, inp_tma, inp_ref = quantize(inp)
            dw = zero_dw(B, T, H, K)
            pm = run_prod(libm, inp_cp, dw)     # BVK=32, cp.async (raw v fp32 wire)
            p64 = run_prod(libm64, inp_cp, dw)  # BVK=64, cp.async
            pt = run_prod(libmt, inp_tma, dw)   # BVK=64, TMA (raw v bf16 wire)
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            for fld in FIELDS:
                worst_bit["tma_vs_bvk64"] = max(worst_bit.get("tma_vs_bvk64", 0.0),
                                                bits("tma.%s vs bvk64" % fld, pt[fld], p64[fld]))
                worst_bit["bvk64_vs_bf16mma"] = max(
                    worst_bit.get("bvk64_vs_bf16mma", 0.0),
                    bits("bvk64.%s vs bf16mma" % fld, p64[fld], pm[fld]))
            # external anchor (pure torch); torch needs T % BT == 0; same bf16 values on the wire
            tor = wy_dqkg_torch(inp_ref, SCALE, BT) if T % BT == 0 else None
            fla_b = fla_wy_dqkg(all_bf16(inp), SCALE, BT)
            for tag, out in (("bf16mma", pm), ("bf16mma_bvk64", p64), ("bf16mma_tma", pt)):
                for fld, tk in (("dqg", "dq"), ("dkg", "dk"), ("dg", "dg"),
                                ("db", "db"), ("dA", "dA")):
                    if tor is not None:
                        _, r = stats("%s.%s vs torch" % (tag, fld), out[fld], tor[tk])
                        k = "torch_%s" % fld
                        worst_ext[k] = max(worst_ext.get(k, 0.0), r)
                    _, r = stats("%s.%s vs FLA" % (tag, fld), out[fld], fla_b[tk])
                    k = "FLA_%s" % fld
                    worst_ext[k] = max(worst_ext.get(k, 0.0), r)
    print("\n" + "-" * 108)
    print("bit-identity of the two isolations (worst max_abs; 0 == bit-identical):")
    for k, v in worst_bit.items():
        print("    %-24s %.3e" % (k, v))
    print("external anchor worst max_rel over all (K,V)/cases (bf16-valued inputs; FLA = all-bf16,注):")
    for fld in ("dqg", "dkg", "dg", "db", "dA"):
        print("    %-5s torch %.3e | FLA %.3e"
              % (fld, worst_ext.get("torch_%s" % fld, -1), worst_ext.get("FLA_%s" % fld, -1)))


# ------------------------------------------------------------------------- Part B
def part_b():
    print("\n" + "=" * 108)
    print("Part B: resources over the (K,V) grid (registers / spill / smem)")
    print("=" * 108)
    print("  %5s %5s | %-14s %8s %8s %10s" % ("K", "V", "tag", "regs", "spill", "smem"))
    for (K, V) in KV:
        for tag, extra in (("bf16mma", BF16MMA), ("bf16mma_bvk64", BF16MMA64),
                           ("bf16mma_tma", BF16MMA_TMA)):
            so, txt = build_v(extra, K, V, tag)
            reg, sp = ptxas_reg(txt)
            print("  %5d %5d | %-14s %8d %8d %10d" % (K, V, tag, reg, sp, smem_bytes(so)))


# ------------------------------------------------------------------------- Part C
def part_c():
    print("\n" + "=" * 108)
    print("Part C: timing -- BVK=32 -> BVK=64 (cp.async) and cp.async -> TMA under bf16mma (K=V=128)")
    print("=" * 108)
    cases = [("B8 T2048 H8", 8, 2048, 8), ("B2 T1024 H8", 2, 1024, 8),
             ("T512 H4", 1, 512, 4), ("T250 H3 (non-div)", 1, 250, 3)]
    K = V = 128
    libm = ctypes.CDLL(build(BF16MMA, K, V, "bf16mma"))
    libm64 = ctypes.CDLL(build(BF16MMA64, K, V, "bf16mma64"))
    libmt = ctypes.CDLL(build(BF16MMA_TMA, K, V, "bf16mmatma"))
    print("  %-20s %12s %12s %12s | %8s %8s" %
          ("case", "bf16mma(us)", "bvk64(us)", "tma(us)", "bvk64/b32", "tma/bvk64"))
    for (name, B, T, H) in cases:
        inp = make(B, T, H, K, V, seed=1)
        inp_cp, inp_tma, _ = quantize(inp)
        dw = zero_dw(B, T, H, K)

        def f_m():
            run_prod(libm, inp_cp, dw)

        def f_m64():
            run_prod(libm64, inp_cp, dw)

        def f_mt():
            run_prod(libmt, inp_tma, dw)

        dev_m, _ = profile_dev(f_m)
        dev_m64, _ = profile_dev(f_m64)
        dev_mt, _ = profile_dev(f_mt)
        print("  %-20s %12.1f %12.1f %12.1f | %8.3f %8.3f"
              % (name, dev_m, dev_m64, dev_mt, dev_m64 / dev_m, dev_mt / dev_m64))
    # wall (event-pair, includes host dispatch) on the two large shapes
    print("\n  wall (event-pair, REP=30 median of 3):")
    print("  %-20s %12s %12s %12s" % ("case", "bf16mma(us)", "bvk64(us)", "tma(us)"))
    for (name, B, T, H) in cases[:2]:
        inp = make(B, T, H, K, V, seed=1)
        inp_cp, inp_tma, _ = quantize(inp)
        dw = zero_dw(B, T, H, K)
        w_m = time_wall(lambda: run_prod(libm, inp_cp, dw))
        w_m64 = time_wall(lambda: run_prod(libm64, inp_cp, dw))
        w_mt = time_wall(lambda: run_prod(libmt, inp_tma, dw))
        print("  %-20s %12.1f %12.1f %12.1f" % (name, w_m, w_m64, w_mt))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    mode = os.environ.get("B2KA3_MODE", "")
    if mode in ("", "a"):
        part_a()
    if mode in ("", "b"):
        part_b()
    if mode in ("", "c"):
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
