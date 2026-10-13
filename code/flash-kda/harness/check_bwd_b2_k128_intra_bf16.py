"""P7-95: close the last cell §95.6 (P7-92) explicitly left open -- the **bf16-native operand
entry x `intra` fold x `K!=128`** intersection, and answer the sub-question "does the `intra`
staging lower its dtype under a bf16 build?".

Background.  P7-90 generalized the six-segment production kernel to `K in {64,128,256}`; P7-91
(§94) bit-checked the bf16-native entry (`-DB2M_BF16`/`BF16MMA`/`BF16_VDW`) at `K!=128`; P7-92
(§95) generalized the standalone `b3_intra_probe.cu` to `KD` and bit-checked the `-DB2M_INTRA`
fold at `K!=128` (fp32).  The two were never combined: §95.6 says verbatim "bf16 入口 x `intra`
折叠 x `K!=128` 仍未组合，`intra` staging 在 bf16 build 下是否也随之降 dtype 未查".

Source fact (checked this round).  The `intra` staging arrays are declared **unconditionally
`float`** (`b2_mma_probe.cu:621-624`: `siq/sik/sig[BT][KD+pad]`, `sda[BT][BT+pad]`) -- there is
no `-DB2M_BF16` branch on them, so the fold's own staging does **not** lower dtype.  Only the
five *streamed* GEMM operands (`do/vn/dv/h/dh`) shrink under `-DB2M_BF16`.  Hence the bf16 entry
and the fp32 fold compose exactly (bf16 subset tf32 => every `rd()` returns the same fp32 value).

Two consequences this round quantifies
  correctness: `prod_bf16_intra` must be bit-identical to `prod_bf16 + standalone b3intra(K)`
               (same derivation as §95, with the bf16 operand value fixed on both sides).
  capacity:    because the fp32 `intra` staging never shrinks but the five *streamed* operands do,
               the bf16 entry **frees enough smem to make the fold feasible at `K=256`** -- which
               fp32 cannot (`278 KB > 227 KB`, §95.3).  This is exactly the "bf16 staging is a
               capacity lever, not a speed lever" direction §97.6 pointed to.

Parts
  Part A  (a) drop-in: `prod_bf16` vs `prod` on bf16-valued fp32 operands -- bit-identical.
          (b) fold x bf16 x K!=128: `prod_bf16_intra` vs `prod_bf16 + standalone b3intra(K)` --
              `dqg/dkg` bit-identical, `dg/db` up to fp32 add-order ulp, unchanged outputs bit 0.
          (c) `prod_bf16_intra` vs `prod_intra` on the same bf16-valued fp32 operands -- bit 0.
  Part B  static cost (regs / spill / smem) at each `K`; the `K=256` feasibility table.
  Part C  timing (CUPTI `dev`, kernel-only): one `prod_bf16_intra` launch vs `prod_bf16` +
          standalone `b3intra` (bf16 build), per `K`.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_k128_intra_bf16.py \
        > .../harness/check_bwd_b2_k128_intra_bf16.out.txt 2>&1
    B2KIB_MODE=a | b | c
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
from check_bwd_ws_reuse import cdiv                                     # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                           # noqa: E402
from check_bwd_b2_intra import run_prod, run_prod_intra, run_b3, load_b3, ptr, stream  # noqa: E402

BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")
B3 = os.path.join(HARNESS, "b3_intra_probe.cu")

PROD = ["-DB2M_PROD=1", "-DB2M_HDW=1"]
INTR = PROD + ["-DB2M_INTRA=1"]
BF16 = ["-DB2M_BF16=1"]
KS = [64, 128, 256]
MAX_SMEM = 232448   # H100 sm_90 opt-in dynamic smem per block (227 KiB)
CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
    ("T64 H1 (single chunk)", 1, 64, 1),
]
# the five streamed operands that carry the bf16-native entry (P7-73/75/76, §76/§92/§94)
BF16_KEYS = ("v_new", "h", "dh", "do", "dv")


def build(src, so, extra, K=None, V=None):
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v"]
    if K is not None:
        cmd += ["-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V]
    cmd += extra + ["-lcuda", src, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed: %s =%s K=%s V=%s" % (src, extra, K, V))
    return r.stderr


def build_b3(so, K):
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v", "-DB3I_KD=%d" % K, "-lcuda", B3, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed: b3 K=%d" % K)
    return r.stderr


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1, int(spill.group(1)) if spill else 0)


def make(B, T, H, K, V, seed=1):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    inp = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
    torch.manual_seed(seed + 100)
    NT = cdiv(T, BT)
    o_t = torch.arange(BT, device="cuda")
    low_incl = (o_t[:, None] >= o_t[None, :]).float()
    dAqk = torch.randn(B, NT, H, BT, BT, device="cuda") * low_incl
    dAqk = dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    inp["dAqk"] = dAqk
    return inp


def quantize(inp):
    """bf16-valued fp32 copy + bf16 copy of the five streamed operands (dAqk/q/k/g stay fp32)."""
    inp_q = dict(inp)
    inp_b = dict(inp)
    for k in BF16_KEYS:
        b = inp[k].bfloat16()
        inp_q[k] = b.float().contiguous()
        inp_b[k] = b.contiguous()
    return inp_q, inp_b


def zero_dw(B, T, H, K):
    return torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def key_stats(name, pred, gold):
    d = maxabs(pred, gold)
    denom = gold.float().abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print("    %-40s max_abs=%.3e  rel=%.3e  [%s]" % (name, d, d / (denom + 1e-12), flag))
    return d


# ------------------------------------------------------------------------- parts
def builds_for(K):
    so = {}
    tags = {
        "prod": PROD,
        "prod_intra": INTR,
        "prod_bf16": PROD + BF16,
        "prod_bf16_intra": INTR + BF16,
    }
    for t, ex in tags.items():
        p = "/tmp/b2kib_%s_K%d.so" % (t, K)
        build(CU, p, ex, K, K)
        so[t] = p
    build_b3("/tmp/b2kib_b3_K%d.so" % K, K)
    return so


def part_a():
    print("=" * 100)
    print("Part A: bf16-native entry x intra fold x K!=128 (and intra staging dtype question)")
    print("=" * 100)
    worst_dropin = 0.0
    worst_dropin_fold = 0.0
    worst_fold = {}
    worst_un = 0.0
    for K in KS:
        so = builds_for(K)
        lib = {t: ctypes.CDLL(p) for t, p in so.items()}
        libb = load_b3(ctypes.CDLL("/tmp/b2kib_b3_K%d.so" % K))
        for t in ("prod_intra", "prod_bf16_intra"):
            lib[t].b2mma_smem_bytes.restype = ctypes.c_int
        smem = lib["prod_bf16_intra"].b2mma_smem_bytes()
        smem_fp32 = lib["prod_intra"].b2mma_smem_bytes()
        fp32_ok = smem_fp32 <= MAX_SMEM
        print("\n#### K = V = %d   (prod_intra fp32 smem = %d B [%s]; prod_bf16_intra smem = %d B"
              " = %.1f KiB %s) ####"
              % (K, smem_fp32, "OK" if fp32_ok else "INFEASIBLE",
                 smem, smem / 1024.0,
                 "<= 227 KiB OK" if smem <= MAX_SMEM else "> 227 KiB INFEASIBLE"))
        if smem > MAX_SMEM:
            continue
        if not fp32_ok:
            print("  NOTE: fp32 fold INFEASIBLE at K=%d -- skipping the fp32 drop-in comparison,"
                  " but the bf16 fold is run (it fits)." % K)
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, K, seed=1)
            inp_q, inp_b = quantize(inp)          # inp_q: bf16-valued fp32; inp_b: bf16
            dw = zero_dw(B, T, H, K)
            p = run_prod(lib["prod"], inp_q, dw)              # fp32 build, bf16-valued fp32
            pb = run_prod(lib["prod_bf16"], inp_b, dw)         # bf16 build, bf16 tensors
            pbi = run_prod_intra(lib["prod_bf16_intra"], inp_b, dw)  # bf16 fold
            b3 = run_b3(libb, inp, pb["dA"])                   # standalone intra (fp32 q/k/g, dA)
            print("\n# K=%d %s  NT=%d" % (K, name, cdiv(T, BT)))
            for fld in ("dv2", "dvb", "dkgb", "db", "dApre", "dA",
                        "dqg", "dkg", "dg", "dgk"):
                worst_dropin = max(worst_dropin,
                                   key_stats("dropin prod_bf16.%s vs prod" % fld, pb[fld], p[fld]))
            if fp32_ok:
                pi = run_prod_intra(lib["prod_intra"], inp_q, dw)  # fp32 fold, same values
                for fld in ("dv2", "dvb", "dkgb", "db", "dApre", "dA",
                            "dqg", "dkg", "dg", "dgk"):
                    worst_dropin_fold = max(
                        worst_dropin_fold,
                        key_stats("dropin bf16_intra.%s vs fp32_intra" % fld, pbi[fld], pi[fld]))
            # fold correctness under bf16: bf16_intra vs bf16 + standalone b3(K)
            for fld, add in (("dqg", "dq"), ("dkg", "dk"), ("dg", "dg"), ("db", "db")):
                ref = pb[fld].float() + b3[add]
                worst_fold["%s@%d" % (fld, K)] = max(
                    worst_fold.get("%s@%d" % (fld, K), 0.0),
                    key_stats("bf16_intra.%s vs bf16+b3.%s" % (fld, add), pbi[fld], ref))
            # sanity: intra contribution really added
            sa = maxabs(pbi["dqg"], pb["dqg"])
            sb = maxabs(pbi["dqg"] - pb["dqg"], b3["dq"])
            print("    sanity: (bf16_intra-bf16).dqg max_abs=%.3e ; ==b3.dq max_abs=%.3e" % (sa, sb))
            # unchanged outputs stay bit-identical (HDW leaves dw unwritten)
            for fld in ("dv2", "dvb", "dkgb", "dA", "dApre", "dgk"):
                worst_un = max(worst_un,
                               key_stats("unchanged bf16_intra.%s vs bf16" % fld,
                                         pbi[fld], pb[fld]))
    print("\n" + "-" * 100)
    print("Part A: prod_bf16 vs prod (drop-in)          worst max_abs = %.3e  [%s]"
          % (worst_dropin, "BIT-IDENTICAL" if worst_dropin == 0.0 else "MISMATCH"))
    print("Part A: bf16_intra vs fp32_intra (drop-in)   worst max_abs = %.3e  [%s]"
          % (worst_dropin_fold, "BIT-IDENTICAL" if worst_dropin_fold == 0.0 else "MISMATCH"))
    print("Part A: bf16_intra vs bf16 + b3intra(K):")
    for K in KS:
        for fld in ("dqg", "dkg", "dg", "db"):
            v = worst_fold.get("%s@%d" % (fld, K))
            if v is not None:
                print("  K=%3d %-4s %.3e  [%s]" % (K, fld, v, "BIT" if v == 0.0 else "ulp"))
    print("Part A: unchanged outputs worst max_abs = %.3e  [%s]"
          % (worst_un, "BIT-IDENTICAL" if worst_un == 0.0 else "MISMATCH"))


def part_b():
    print("\n" + "=" * 100)
    print("Part B: static cost -- prod / prod_intra / prod_bf16 / prod_bf16_intra at each K")
    print("=" * 100)
    print("  %-18s %5s %6s %7s %10s  %s" % ("build", "K", "regs", "spill", "smem(B)", "block/SM"))
    for K in KS:
        for t, ex in (("prod", PROD), ("prod_intra", INTR),
                      ("prod_bf16", PROD + BF16), ("prod_bf16_intra", INTR + BF16)):
            so = "/tmp/b2kib_r_%s_K%d.so" % (t, K)
            txt = build(CU, so, ex, K, K)
            reg, sp = ptxas_info(txt)
            lib = ctypes.CDLL(so)
            lib.b2mma_smem_bytes.restype = ctypes.c_int
            smem = lib.b2mma_smem_bytes()
            feasible = "feasible" if smem <= MAX_SMEM else "INFEASIBLE(>227KiB)"
            print("  %-18s %5d %6d %7d %10d  %s" % (t, K, reg, sp, smem, feasible))


def part_c():
    print("\n" + "=" * 100)
    print("Part C: timing (CUPTI dev) -- bf16 fold vs bf16 prod + standalone b3intra")
    print("=" * 100)
    print("  %-22s | %9s %14s | %11s | %11s %13s" %
          ("case", "bf16", "bf16_intra", "bf16+b3", "incl/bf16", "fold/(bf16+b3)"))
    tc = [("B2 T1024 H8", 2, 1024, 8), ("T512 H4", 1, 512, 4)]
    for K in KS:
        so = builds_for(K)
        lib = {t: ctypes.CDLL(p) for t, p in so.items()}
        libb = load_b3(ctypes.CDLL("/tmp/b2kib_b3_K%d.so" % K))
        lib["prod_bf16_intra"].b2mma_smem_bytes.restype = ctypes.c_int
        if lib["prod_bf16_intra"].b2mma_smem_bytes() > MAX_SMEM:
            print("  K=%3d  prod_bf16_intra smem %.1f KiB > 227 KiB -- skipped"
                  % (K, lib["prod_bf16_intra"].b2mma_smem_bytes() / 1024.0))
            continue
        for (name, B, T, H) in tc:
            inp = make(B, T, H, K, K, seed=1)
            _, inp_b = quantize(inp)
            dw = zero_dw(B, T, H, K)
            pb0 = run_prod(lib["prod_bf16"], inp_b, dw)

            def f_prod():
                run_prod(lib["prod_bf16"], inp_b, dw)

            def f_fold():
                run_prod_intra(lib["prod_bf16_intra"], inp_b, dw)

            def f_old():
                run_prod(lib["prod_bf16"], inp_b, dw)
                run_b3(libb, inp, pb0["dA"])

            f_prod(); f_fold(); f_old(); torch.cuda.synchronize()
            dp, _ = profile_dev(f_prod, reps=30)
            di, _ = profile_dev(f_fold, reps=30)
            do, _ = profile_dev(f_old, reps=30)
            print("  K=%3d %-16s | %9.1f %14.1f | %11.1f | %11.3f %13.3f" %
                  (K, name, dp, di, do, di / dp, di / do))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    mode = os.environ.get("B2KIB_MODE", "")
    if mode in ("", "a"):
        part_a()
    if mode in ("", "b"):
        part_b()
    if mode in ("", "c"):
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
