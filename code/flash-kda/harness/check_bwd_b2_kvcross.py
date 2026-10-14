"""P7-101: close the two boundaries §103.6 (P7-100) explicitly left open for the K2-shaped CUDA
B2 production kernel (`b2_mma_probe.cu`).

Background.  P7-90 generalized the six-segment production kernel to `K in {64,128,256}`
(*and* generalized the V axis in P7-82), and P7-83/P7-95 anchored the `K!=V` and `K!=128`
axes **separately** -- but every K-axis *numerical* anchor was a **fold-vs-standalone**
consistency (`prod_intra` vs `prod + b3intra`, P7-63/92/95) or a bf16-vs-fp32 drop-in on the
**base GEMM core** (P7-89).  §103.6 says verbatim:

  "`(K,V)` 双非 128 的独立数值核对未做，标待验证"  and
  "`(256,64)` `intra` 的 `rc=-3` 只用代码路径归因，未贴 `sizeof(Shared)` 的确切字节"

This round closes both.

  Part A  independent numerical anchor at `(K,V)` both `!=128` (K<V and K>V) plus the
          `K=V=128` control: the **production** `b2prod` gate triple + `db`/`dA` vs a pure-torch
          per-chunk reference (`wy_dqkg_torch`) and vs FLA (`chunk_kda_bwd_wy_dqkg_fused`), with
          real max_abs / max_rel.  This is the first *independent* (not self-consistency) check
          of the production kernel on the two-axis cross.
  Part A2 `-DB2M_DAMMA` (= `prod_damma`) gate triple must be **bit-identical** to `prod` on the
          same operands at those shapes (DAMMA only rewrites the `dApre/dA` value, §66).
  Part A3 folded `intra` at `(K,V)` both `!=128`: `prod_intra.dqg/dkg/dg/db` vs
          `prod.<...> + standalone b3intra(K)` (the two things the fold replaces).
  Part B  machine-verify the `-DB2M_INTRA` smem boundary: `sizeof(Shared)` (via the exported
          `b2mma_smem_bytes`) for `prod` / `prod_intra` / `prod_bf16_intra` over
          `K in {64,128,256} x V in {64,128,256}`, showing (i) the *intra increment* is a
          function of `K` only (V-independent), (ii) `K=256` fp32 exceeds the 227 KiB opt-in cap,
          (iii) `prod_bf16_intra` fits at `K=256`, and actually *launching* `(256,64)` `intra`
          to read `rc`.

A latent harness bug fixed this round (not a kernel change): `wy_dqkg_torch`
(`check_bwd_dqkg_kernel.py`) sized the zero template for `b_q/b_k/b_g` from the *V*-shaped
`b_do`, so it was only index-correct at `K==V`; the templates now use the `K`-shaped tensors
(bit-identical at `K==V`).

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_kvcross.py \
        > .../harness/check_bwd_b2_kvcross.out.txt 2>&1
    B2KC_MODE=a | a2 | a3 | b
"""
import ctypes
import math
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
from check_bwd_dqkg_kernel import fla_wy_dqkg, wy_dqkg_torch            # noqa: E402
from check_bwd_ws_reuse import cdiv                                     # noqa: E402
from check_bwd_b2_prod import run_prod                                  # noqa: E402
from check_bwd_b2_intra import (run_prod_intra, run_b3, load_b3,        # noqa: E402
                                ptr, stream)

BT = 16
NW = 8                              # warps = b2_mma_probe.cu:72 (constexpr int NW = 8)
SCALE = 1.0 / (128 ** 0.5)          # deployment scale (matches run_prod/run_prod_intra)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")
B3 = os.path.join(HARNESS, "b3_intra_probe.cu")
MAX_SMEM = 232448                   # H100 sm_90 opt-in dynamic smem / block (227 KiB)

PROD = ["-DB2M_PROD=1", "-DB2M_HDW=1"]
DAMMA = PROD + ["-DB2M_DAMMA=1"]
INTRA = PROD + ["-DB2M_INTRA=1"]
BF16_INTRA = INTRA + ["-DB2M_BF16=1"]

# (K,V) both != 128 in both polarities, plus the K=V=128 control.
KV = [(64, 256), (256, 64), (64, 128), (128, 64), (128, 128)]
CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]


def build(extra, K, V, tag):
    so = "/tmp/b2kc_%s_%d_%d.so" % (tag, K, V)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v", "-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V] + extra + \
          ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed: tag=%s K=%d V=%d" % (tag, K, V))
    return so, r.stderr


def build_b3(K):
    so = "/tmp/b2kc_b3_%d.so" % K
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC", "-shared",
           "-DB3I_KD=%d" % K, "-lcuda", B3, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed: b3 K=%d" % K)
    return so


def smem_bytes(so):
    lib = ctypes.CDLL(so)
    lib.b2mma_smem_bytes.restype = ctypes.c_int
    return lib.b2mma_smem_bytes()


def ptxas_reg(txt):
    m = re.search(r"Used (\d+) registers", txt)
    return int(m.group(1)) if m else -1


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


def stats(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    denom = b.float().abs().max().item()
    print("    %-34s max_abs=%.3e  max_rel=%.3e  ref_max=%.3e" % (name, d, d / (denom + 1e-12), denom))
    return d, d / (denom + 1e-12)


def zero_dw(B, T, H, K):
    return torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)


# ------------------------------------------------------------------------- Part A
def part_a():
    print("=" * 100)
    print("Part A: production b2prod at (K,V) both != 128 -- vs pure-torch and FLA (SCALE=1/sqrt128)")
    print("=" * 100)
    worst_t = {}
    worst_f = {}
    worst_t2 = {}
    for (K, V) in KV:
        so, _ = build(PROD, K, V, "prod")
        so2, _ = build(DAMMA, K, V, "damma")
        lib = ctypes.CDLL(so)
        lib2 = ctypes.CDLL(so2)
        print("\n#### K = %d, V = %d   prod smem = %d B   damma smem = %d B ####"
              % (K, V, smem_bytes(so), smem_bytes(so2)))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            dw = zero_dw(B, T, H, K)
            p = run_prod(lib, inp, dw)
            pd = run_prod(lib2, inp, dw)
            # `wy_dqkg_torch` views `[B,T,H,·]` as `[B,NT,BT,H,·]`, so it needs T % BT == 0.
            tor = wy_dqkg_torch(inp, SCALE, BT) if T % BT == 0 else None
            fla = fla_wy_dqkg(inp, SCALE, BT)
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            for fld, tk in (("dqg", "dq"), ("dkg", "dk"), ("dg", "dg"), ("db", "db"), ("dA", "dA")):
                if tor is not None:
                    _, r_t = stats("  prod.%s vs torch.%s" % (fld, tk), p[fld], tor[tk])
                    worst_t["%s" % fld] = max(worst_t.get("%s" % fld, 0.0), r_t)
                _, r_f = stats("  prod.%s vs FLA.%s" % (fld, tk), p[fld], fla[tk])
                worst_f["%s" % fld] = max(worst_f.get("%s" % fld, 0.0), r_f)
            print("    (prod_damma gate triple vs prod -- DAMMA only rewrites dApre/dA):")
            for fld in ("dqg", "dkg", "dg"):
                d, _ = stats("  damma.%s vs prod.%s" % (fld, fld), pd[fld], p[fld])
                worst_t2[fld] = max(worst_t2.get(fld, 0.0), d)
    print("\n" + "-" * 100)
    print("worst max_rel over all (K,V)/cases -- prod vs pure torch / vs FLA:")
    for fld in ("dqg", "dkg", "dg", "db", "dA"):
        print("    %-5s vs torch %.3e   vs FLA %.3e" % (fld, worst_t.get(fld, -1), worst_f.get(fld, -1)))
    print("worst max_abs -- prod_damma gate triple vs prod (expect bit-identical 0):")
    for fld in ("dqg", "dkg", "dg"):
        print("    %-5s %.3e  [%s]" % (fld, worst_t2.get(fld, -1),
                                       "BIT" if worst_t2.get(fld, -1) == 0.0 else "DIFF"))


# ------------------------------------------------------------------------- Part A3
def part_a3():
    print("\n" + "=" * 100)
    print("Part A3: folded intra at (K,V) both != 128 -- prod_intra vs prod + standalone b3intra(K)")
    print("=" * 100)
    worst_fold = {}
    worst_un = {}
    for (K, V) in KV:
        if K == V and K == 128:
            pass
        so_p, _ = build(PROD, K, V, "prod")
        so_i, _ = build(INTRA, K, V, "intra")
        libp = ctypes.CDLL(so_p)
        libi = ctypes.CDLL(so_i)
        libb = load_b3(ctypes.CDLL(build_b3(K)))
        print("\n#### K = %d, V = %d  (prod_intra smem = %d B) ####" % (K, V, smem_bytes(so_i)))
        if smem_bytes(so_i) > MAX_SMEM:
            print("    prod_intra smem %.1f KiB > 227 KiB -- INFEASIBLE, skipped (see Part B)"
                  % (smem_bytes(so_i) / 1024.0))
            continue
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            dw = zero_dw(B, T, H, K)
            p = run_prod(libp, inp, dw)
            pi = run_prod_intra(libi, inp, dw)
            b3 = run_b3(libb, inp, p["dA"])
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            _sa = (pi["dqg"].float() - p["dqg"].float()).abs().max().item()
            print("    (prod_intra - prod).dqg max_abs=%.3e (intra contribution added)" % _sa)
            for fld, ref in (("dqg", lambda: p["dqg"] + b3["dq"]),
                             ("dkg", lambda: p["dkg"] + b3["dk"]),
                             ("dg",  lambda: p["dg"] + b3["dg"]),
                             ("db",  lambda: p["db"] + b3["db"])):
                d, _ = stats("  prod_intra.%s vs prod+b3" % fld, pi[fld], ref())
                worst_fold["%s@%d_%d" % (fld, K, V)] = max(worst_fold.get("%s@%d_%d" % (fld, K, V), 0.0), d)
            for fld in ("dv2", "dvb", "dkgb", "dA", "dApre", "dgk"):
                d = (pi[fld].float() - p[fld].float()).abs().max().item()
                worst_un["%s@%d_%d" % (fld, K, V)] = max(worst_un.get("%s@%d_%d" % (fld, K, V), 0.0), d)
    print("\nworst max_abs over all (K,V)/cases -- fold vs prod+b3 (0 == bit-identical):")
    for (K, V) in KV:
        for fld in ("dqg", "dkg", "dg", "db"):
            print("    K=%3d V=%3d %-4s %.3e   | untouched %.3e/%.3e"
                  % (K, V, fld, worst_fold.get("%s@%d_%d" % (fld, K, V), -1),
                     worst_un.get("dv2@%d_%d" % (K, V), -1),
                     worst_un.get("dA@%d_%d" % (K, V), -1)))


# ------------------------------------------------------------------------- Part B
def part_b():
    print("\n" + "=" * 100)
    print("Part B: -DB2M_INTRA smem boundary -- sizeof(Shared) over K x V (and the launch rc)")
    print("=" * 100)
    print("  %5s %5s | %10s %13s %13s | %10s %10s" %
          ("K", "V", "prod", "prod_intra", "intra-prod", "bf16_intra", "fit_bf16"))
    for K in (64, 128, 256):
        for V in (64, 128, 256):
            sp, _ = build(PROD, K, V, "prod")
            si, _ = build(INTRA, K, V, "intra")
            sb, _ = build(BF16_INTRA, K, V, "bf16intra")
            s_p, s_i, s_b = smem_bytes(sp), smem_bytes(si), smem_bytes(sb)
            print("  %5d %5d | %10d %13d %13d | %10d %10s"
                  % (K, V, s_p, s_i, s_i - s_p, s_b,
                     "OK" if s_b <= MAX_SMEM else "INFEAS"))
    print("\n  analytic intra increment (default pads=0, NW=8): 3*BT*K*4 + BT*BT*4 + NW*BT*4")
    for K in (64, 128, 256):
        print("    K=%3d -> %d B (=%d*3*4 + %d + %d)" %
              (K, 3 * BT * K * 4 + BT * BT * 4 + NW * BT * 4,
               3 * BT * K, BT * BT * 4, NW * BT * 4))
    print("\n  actual launch probe for the fp32 K=256 fold (expect rc=-3 at the smem set-attribute):")
    for V in (64, 128):
        so_i, _ = build(INTRA, 256, V, "intra")
        libi = ctypes.CDLL(so_i)
        B, T, H = 1, 64, 1
        inp = make(B, T, H, 256, V, seed=1)
        dw = zero_dw(B, T, H, 256)
        try:
            run_prod_intra(libi, inp, dw)
            print("    (K=256,V=%d) launched OK  <-- unexpected" % V)
        except RuntimeError as e:
            print("    (K=256,V=%d) launch rejected: %s" % (V, e))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    mode = os.environ.get("B2KC_MODE", "")
    if mode in ("", "a"):
        part_a()
    if mode in ("", "a3"):
        part_a3()
    if mode in ("", "b"):
        part_b()
    print("\nDONE")


if __name__ == "__main__":
    main()
