"""P7-102: close the boundary §104.5 (P7-101) explicitly left open for the K2-shaped CUDA B2
production kernel (`b2_mma_probe.cu`).

§104.2 gave the first *external* (pure-torch + FLA) numeric anchor of the six-segment production
kernel at `(K,V)` both `!= 128`, but all of it was the plain **fp32 / `cp.async`** build.  §104.5
says verbatim:

  "fp32 输出、`cp.async`（未叠 `TMA`/`BF16MMA`）"

This round extends that external anchor to the remaining **staging variants** at `K != V`, and
anchors the **K=256 `INTRA` fold enabling combination** (bf16 staging) to an external reference:

  Part A  staging variants at `K != V`, external anchor (pure-torch + FLA) + bit-identity:
          `prod_tma` (`-DB2M_TMA`), `prod_bf16` (`-DB2M_BF16`, bf16-valued inputs),
          `prod_bf16mma` (`-DB2M_BF16 -DB2M_BF16MMA`).  The gate triple `dqg/dkg/dg` + `db/dA`
          vs `wy_dqkg_torch` / `fla_wy_dqkg`; and the staging drop-ins vs the fp32 `prod`:
            * `prod_tma`   == `prod`            (bit-identical; only the staging engine changes)
            * `prod_bf16`  == `prod` (on bf16-valued inputs; bf16 subset tf32, widening exact)
            * `prod_bf16mma` straight outputs == `prod_bf16` (only the GEMM accumulation order
              changes -> `dv2/dvb/dgk` bit, `db/dkg` fp32-add-order ulp)
  Part B  the **K=256 fold** (`-DB2M_INTRA`) is only launchable with bf16 staging (§104.3: fp32 is
          `rc=-3`).  Anchor `prod_bf16_intra` to an **external** reference:
          `wy_dqkg_torch.dq + intra_torch.dq` (the B2 gate triple + the B3 intra diagonal), and
          the fold self-consistency `prod_bf16_intra` vs `prod_bf16 + standalone b3intra(K)`.
  Part C  resources: registers / spill / smem of the staging builds over the `(K,V)` grid.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_kv_anchor2.py \
        > .../harness/check_bwd_b2_kv_anchor2.out.txt 2>&1
    B2KA2_MODE=a | b | c
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

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs        # noqa: E402
from check_bwd_dqkg_kernel import fla_wy_dqkg, wy_dqkg_torch           # noqa: E402
from check_bwd_ws_reuse import cdiv                                    # noqa: E402
from check_bwd_b2_prod import run_prod, ptr, stream                    # noqa: E402
from check_bwd_b2_intra import run_prod_intra, run_b3, load_b3         # noqa: E402
from check_bwd_intra_kernel import intra_torch                         # noqa: E402

BT = 16
SCALE = 1.0 / (128 ** 0.5)          # deployment scale (matches run_prod/run_prod_intra)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")
B3 = os.path.join(HARNESS, "b3_intra_probe.cu")
MAX_SMEM = 232448                   # H100 sm_90 opt-in dynamic smem / block (227 KiB)

PROD = ["-DB2M_PROD=1", "-DB2M_HDW=1"]
TMA = PROD + ["-DB2M_TMA=1"]
BF16 = PROD + ["-DB2M_BF16=1"]
BF16MMA = BF16 + ["-DB2M_BF16MMA=1"]
BF16_MMA_TMA = BF16 + ["-DB2M_BF16MMA=1", "-DB2M_TMA=1", "-DB2M_BVK=64"]
BF16_INTRA = BF16 + ["-DB2M_INTRA=1"]

# five bf16-native streamed operands (`b2_mma_probe.cu:154-162`); q/k/g/beta/A stay fp32.
BF16_KEYS = ("v_new", "h", "dh", "do", "dv")

KV = [(64, 256), (256, 64), (64, 128), (128, 64), (128, 128)]
CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]
# Part B: K=256 (fold only fits with bf16 staging) + K=128 control + K=64.
KV_INTRA = [(256, 64), (256, 256), (128, 128), (64, 64)]


def build_v(extra, K, V, tag, bvk=None):
    so = "/tmp/b2ka2_%s_%d_%d.so" % (tag, K, V)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v", "-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V]
    if bvk is not None:
        cmd += ["-DB2M_BVK=%d" % bvk]
    cmd += extra + ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed: tag=%s K=%d V=%d" % (tag, K, V))
    return so, r.stderr


def build(extra, K, V, tag, bvk=None):
    return build_v(extra, K, V, tag, bvk)[0]


def build_b3(K):
    so = "/tmp/b2ka2_b3_%d.so" % K
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
    inp_q, inp_b = dict(inp), dict(inp)
    for k in BF16_KEYS:
        b = inp[k].bfloat16()
        inp_q[k] = b.float().contiguous()
        inp_b[k] = b.contiguous()
    return inp_q, inp_b


def all_bf16(inp):
    """FLA's fused wy_dqkg kernel requires a uniform operand dtype, so its bf16 path needs every
    operand (incl. q/k/g/beta/A) bf16 -- unlike the production kernel, which only quantizes the
    five streamed operands."""
    out = {}
    for k, v in inp.items():
        if k in ("dAqk",):
            out[k] = v.contiguous()          # not consumed by fla_wy_dqkg
        else:
            out[k] = v.bfloat16().contiguous()
    return out


def stats(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    denom = b.float().abs().max().item()
    print("    %-36s max_abs=%.3e  max_rel=%.3e  ref_max=%.3e"
          % (name, d, d / (denom + 1e-12), denom))
    return d, d / (denom + 1e-12)


def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    print("    %-36s max_abs=%.3e  [%s]" % (name, d, "BIT" if d == 0.0 else "DIFF"))
    return d


def zero_dw(B, T, H, K):
    return torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)


# ------------------------------------------------------------------------- Part A
def part_a():
    print("=" * 104)
    print("Part A: staging variants of the production kernel at K!=V -- EXTERNAL anchor + bit-identity")
    print("=" * 104)
    worst_ext = {}
    worst_bit = {}
    for (K, V) in KV:
        so_p = build(PROD, K, V, "prod")
        so_t = build(TMA, K, V, "tma")
        so_b = build(BF16, K, V, "bf16")
        so_m = build(BF16MMA, K, V, "bf16mma")
        libp, libt, libb, libm = (ctypes.CDLL(so_p), ctypes.CDLL(so_t),
                                  ctypes.CDLL(so_b), ctypes.CDLL(so_m))
        print("\n#### K = %d, V = %d   smem prod=%d tma=%d bf16=%d bf16mma=%d ####"
              % (K, V, smem_bytes(so_p), smem_bytes(so_t), smem_bytes(so_b), smem_bytes(so_m)))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            inp_q, inp_b = quantize(inp)
            dw = zero_dw(B, T, H, K)
            p = run_prod(libp, inp, dw)            # fp32 cp.async
            pq = run_prod(libp, inp_q, dw)         # fp32 on bf16-valued inputs (reference for drop-in)
            pt = run_prod(libt, inp, dw)           # TMA on fp32
            pb = run_prod(libb, inp_b, dw)         # bf16-native operands
            pm = run_prod(libm, inp_b, dw)         # bf16 operands + native bf16 mma
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            # -- staging bit-identity
            for fld in ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk"):
                worst_bit["tma_vs_prod"] = max(worst_bit.get("tma_vs_prod", 0.0),
                                               bits("tma.%s vs prod" % fld, pt[fld], p[fld]))
                worst_bit["bf16_vs_prod"] = max(worst_bit.get("bf16_vs_prod", 0.0),
                                                bits("bf16.%s vs prod(q)" % fld, pb[fld], pq[fld]))
            for fld in ("dv2", "dvb", "dApre", "dA", "db", "dkgb", "dgk"):
                worst_bit["mma_vs_bf16_straight"] = max(
                    worst_bit.get("mma_vs_bf16_straight", 0.0),
                    bits("mma.%s vs bf16" % fld, pm[fld], pb[fld]))
            for fld in ("dqg", "dkg", "dg"):
                worst_bit["mma_vs_bf16_gate"] = max(
                    worst_bit.get("mma_vs_bf16_gate", 0.0),
                    bits("mma.%s vs bf16" % fld, pm[fld], pb[fld]))
            # -- external anchor (pure torch + FLA); torch needs T % BT == 0
            tor = wy_dqkg_torch(inp, SCALE, BT) if T % BT == 0 else None
            tor_q = wy_dqkg_torch(inp_q, SCALE, BT) if T % BT == 0 else None
            fla = fla_wy_dqkg(inp, SCALE, BT)
            fla_b = fla_wy_dqkg(all_bf16(inp), SCALE, BT)
            for tag, out, torref, fr, grp in (("prod", p, tor, fla, "fp32"),
                                              ("tma", pt, tor, fla, "fp32"),
                                              ("bf16", pb, tor_q, fla_b, "bf16"),
                                              ("bf16mma", pm, tor_q, fla_b, "bf16")):
                for fld, tk in (("dqg", "dq"), ("dkg", "dk"), ("dg", "dg"), ("db", "db"), ("dA", "dA")):
                    if torref is not None:
                        _, r = stats("%s.%s vs torch" % (tag, fld), out[fld], torref[tk])
                        k = "%s_torch_%s" % (grp, fld)
                        worst_ext[k] = max(worst_ext.get(k, 0.0), r)
                    _, r = stats("%s.%s vs FLA" % (tag, fld), out[fld], fr[tk])
                    k = "%s_FLA_%s" % (grp, fld)
                    worst_ext[k] = max(worst_ext.get(k, 0.0), r)
    print("\n" + "-" * 104)
    print("staging bit-identity (worst max_abs; 0 == bit-identical):")
    for k, v in worst_bit.items():
        print("    %-24s %.3e" % (k, v))
    print("external anchor worst max_rel over all (K,V)/cases:")
    print("    (fp32 group = prod/tma vs fp32 torch/FLA; bf16 group = bf16/bf16mma vs torch on "
          "bf16-valued inputs / FLA all-bf16)")
    for fld in ("dqg", "dkg", "dg", "db", "dA"):
        print("    %-5s fp32: torch %.3e  FLA %.3e | bf16: torch %.3e  FLA %.3e"
              % (fld, worst_ext.get("fp32_torch_%s" % fld, -1), worst_ext.get("fp32_FLA_%s" % fld, -1),
                 worst_ext.get("bf16_torch_%s" % fld, -1), worst_ext.get("bf16_FLA_%s" % fld, -1)))


# ------------------------------------------------------------------------- Part B
def part_b():
    print("\n" + "=" * 104)
    print("Part B: K=256 INTRA fold (bf16 staging only) -- EXTERNAL anchor vs wy_dqkg_torch + intra_torch")
    print("=" * 104)
    so_i128 = build(BF16_INTRA, 128, 128, "bf16intra")     # ensures the .cu compiles
    del so_i128
    worst_ext = {}
    worst_fold = {}
    for (K, V) in KV_INTRA:
        so_p = build(BF16, K, V, "bf16")
        so_i = build(BF16_INTRA, K, V, "bf16intra")
        so_pf = build(PROD, K, V, "prod")
        sm_i, sm_p, sm_pf = smem_bytes(so_i), smem_bytes(so_p), smem_bytes(so_pf)
        libp, libi = ctypes.CDLL(so_p), ctypes.CDLL(so_i)
        libb = load_b3(ctypes.CDLL(build_b3(K)))
        print("\n#### K = %d, V = %d   smem prod(fp32)=%d bf16=%d bf16_intra=%d %s ####"
              % (K, V, sm_pf, sm_p, sm_i, "FIT" if sm_i <= MAX_SMEM else "INFEASIBLE"))
        if sm_i > MAX_SMEM:
            print("    bf16_intra smem > 227 KiB -- skipped")
            continue
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            inp_q, inp_b = quantize(inp)
            dw = zero_dw(B, T, H, K)
            p = run_prod(libp, inp_b, dw)          # pre-fold bf16
            pi = run_prod_intra(libi, inp_b, dw)   # folded bf16
            b3 = run_b3(libb, inp_b, p["dA"])      # standalone intra on the same dA
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            _sa = (pi["dqg"].float() - p["dqg"].float()).abs().max().item()
            print("    (prod_intra - prod).dqg max_abs=%.3e (intra contribution added)" % _sa)
            # fold self-consistency
            for fld, ref in (("dqg", lambda: p["dqg"] + b3["dq"]),
                             ("dkg", lambda: p["dkg"] + b3["dk"]),
                             ("dg",  lambda: p["dg"] + b3["dg"]),
                             ("db",  lambda: p["db"] + b3["db"])):
                d = bits("fold.%s vs bf16+b3" % fld, pi[fld], ref())
                worst_fold["%s@%d_%d" % (fld, K, V)] = max(
                    worst_fold.get("%s@%d_%d" % (fld, K, V), 0.0), d)
            # external anchor: wy_dqkg_torch (B2) + intra_torch (B3), on the bf16-valued inputs
            if T % BT == 0:
                tor = wy_dqkg_torch(inp_q, SCALE, BT)
                inp_i = dict(inp_q)
                inp_i["dAkk"] = tor["dA"]
                itr = intra_torch(inp_i, BT, BT)
                for fld, tk in (("dqg", "dq"), ("dkg", "dk"), ("dg", "dg"),
                                ("db", "db"), ("dA", "dA")):
                    if fld == "dA":
                        _, r = stats("bf16_intra.dA vs torch", pi["dA"], tor["dA"])
                    else:
                        _, r = stats("bf16_intra.%s vs torch(B2+B3)" % fld,
                                     pi[fld], tor[tk] + itr[tk])
                    worst_ext["%s@%d_%d" % (fld, K, V)] = max(
                        worst_ext.get("%s@%d_%d" % (fld, K, V), 0.0), r)
    print("\n" + "-" * 104)
    print("fold self-consistency (bit-identity; dg/db fp32 add-order ulp):")
    for (K, V) in KV_INTRA:
        for fld in ("dqg", "dkg", "dg", "db"):
            print("    K=%3d V=%3d %-4s %.3e"
                  % (K, V, fld, worst_fold.get("%s@%d_%d" % (fld, K, V), -1)))
    print("external anchor (max_rel vs wy_dqkg_torch+intra_torch; torch-only, T%BT==0):")
    for (K, V) in KV_INTRA:
        for fld in ("dqg", "dkg", "dg", "db", "dA"):
            print("    K=%3d V=%3d %-4s %.3e"
                  % (K, V, fld, worst_ext.get("%s@%d_%d" % (fld, K, V), -1)))


# ------------------------------------------------------------------------- Part C
def part_c():
    print("\n" + "=" * 104)
    print("Part C: resources over the (K,V) grid (registers / spill / smem)")
    print("=" * 104)
    print("  %5s %5s | %-9s %8s %8s %8s %8s %10s" %
          ("K", "V", "tag", "regs", "spill", "occup", "occsm", "smem"))
    for (K, V) in KV:
        for tag, extra in (("prod", PROD), ("tma", TMA), ("bf16", BF16), ("bf16mma", BF16MMA)):
            so, txt = build_v(extra, K, V, tag)
            reg, sp = ptxas_reg(txt)
            lib = ctypes.CDLL(so)
            lib.b2mma_smem_bytes.restype = ctypes.c_int
            smem = lib.b2mma_smem_bytes()
            print("  %5d %5d | %-9s %8d %8d %8s %8s %10d"
                  % (K, V, tag, reg, sp, "-", "-", smem))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    mode = os.environ.get("B2KA2_MODE", "")
    if mode in ("", "a"):
        part_a()
    if mode in ("", "b"):
        part_b()
    if mode in ("", "c"):
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
