"""P7-92: close the last item §94.7 (P7-91) named -- the **`intra` fold x `K!=128`** correction.

Background.  `-DB2M_INTRA` folds the B3 diagonal (the K2-shape `intra`) into the production B2
launch (`b2_mma_probe.cu`; docs/07 §65/§66, P7-63).  At `K==128` that fold was bit-checked
against `prod + standalone b3intra` (§66 Part A, `check_bwd_b2_intra.py`).  P7-83 checked the
fold at `K!=V` (V in {64,128,256}, **K fixed 128**), P7-90 generalized the six-segment core to
`K in {64,128,256}` -- but explicitly left **the fold itself unverified at `K!=128`** and §94.7
pinned the reason: the standalone reference `b3_intra_probe.cu` hard-baked `KD == 128`, so it
could not be built at another `K` to serve as an independent check.

Why no new kernel work is expected.  The intra diagonal is **entirely a K-axis** computation
(`q/k/g`, `dAqk`, `dAkk`, `Mq/Sq/Sk`; docs/07 §65.1): it never touches `v`, and the only
`KD`-dependence is the output-column sharding (warp `w` owns `BW = KD/NW` columns = `NBW = BW/8`
mma `n=8` tiles).  P7-90 already replaced the fold's `nt < 2` accumulators (`idq/idk/idg`) with
`nt < NBW` (`b2_mma_probe.cu:992-1002/1148...`), so `prod_intra` compiles at `K!=128` -- this
round **generalizes the standalone reference to `KD`** (a pure `NBW`-for-2 substitution, byte-
equivalent at `BW==16`) and then bit-checks the two.

Parts
  Part A0 regression: generalized `b3_intra_probe.cu` at `KD==128` vs pre-change (git HEAD)
           standalone -- bit-identical dq/dk/db/dg.
  Part A  fold x `K!=128`: `prod_intra` vs `prod + standalone b3intra(K)` for `K in {64,128,256}`
           (V=K), 4 shapes incl. the non-divisible `T=250` -- `dqg/dkg/dg/db` bit-identical
           (dg up to fp32 add-order ulp), every non-intra output untouched.
  Part A2 sanity: the intra contribution is really added (`prod_intra != prod`, and the
           contribution equals `b3intra` at the same `K`).
  Part B  static cost (regs / spill / smem) of `prod` vs `prod_intra` at each `K`.
  Part C  timing (CUPTI `dev`, kernel-only): one `prod_intra` launch vs `prod` + standalone
           `b3intra`, per `K`.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=4 python .../harness/check_bwd_b2_k128_intra.py \
        > .../harness/check_bwd_b2_k128_intra.out.txt 2>&1
    B2KI_MODE=a0 | a | b | c
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
from check_bwd_ws_reuse import cdiv                                     # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                           # noqa: E402
from check_bwd_b2_intra import run_prod, run_prod_intra, run_b3, load_b3  # noqa: E402

BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")
B3 = os.path.join(HARNESS, "b3_intra_probe.cu")

PROD = ["-DB2M_PROD=1", "-DB2M_HDW=1"]
INTR = PROD + ["-DB2M_INTRA=1"]
KS = [64, 128, 256]
MAX_SMEM = 232448   # H100 sm_90 opt-in dynamic smem per block (227 KiB)
CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
    ("T64 H1 (single chunk)", 1, 64, 1),
]


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


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
    # P7-92: the standalone reference is now K-parameterized via `-DB3I_KD`.
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


def orig_source():
    out = "/tmp/b2_k128_intra_orig.cu"
    r = subprocess.run(["git", "-C", os.path.expanduser("~/proj/tech_record"), "show",
                        "HEAD:code/flash-kda/harness/b3_intra_probe.cu"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("cannot read HEAD b3 source: " + r.stderr)
    open(out, "w").write(r.stdout)
    return out


def make(B, T, H, K, V, seed=1):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    inp = {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}
    # `dAqk = tril(randn, incl diag)` in the [B,T,H,BT] layout the intra kernel consumes.
    torch.manual_seed(seed + 100)
    NT = cdiv(T, BT)
    o_t = torch.arange(BT, device="cuda")
    low_incl = (o_t[:, None] >= o_t[None, :]).float()
    dAqk = torch.randn(B, NT, H, BT, BT, device="cuda") * low_incl
    dAqk = dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    inp["dAqk"] = dAqk
    return inp


def core_dw(inp, so_core):
    """The `dw` the production kernel expects as `dwin` (bare GEMM core, matching K/V)."""
    lib = ctypes.CDLL(so_core)
    lib.b2mma_launch.restype = ctypes.c_int
    lib.b2mma_launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dw = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dgk = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    lib.b2mma_launch(ptr(inp["v"]), ptr(inp["v_new"]), ptr(inp["h"]), ptr(inp["dh"]),
                     ptr(inp["do"]), ptr(inp["dv"]), ptr(dq), ptr(dk), ptr(dw), ptr(dgk),
                     B, T, H, NT, stream())
    return dw


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def key_stats(name, pred, gold):
    d = maxabs(pred, gold)
    denom = gold.float().abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print("    %-30s max_abs=%.3e  rel=%.3e  [%s]" % (name, d, d / (denom + 1e-12), flag))
    return d


# ------------------------------------------------------------------------- parts
def part_a0():
    print("=" * 92)
    print("Part A0: generalized b3_intra_probe.cu at KD==128 vs git HEAD -- bit-identical")
    print("=" * 92)
    build_b3("/tmp/b2ki_b3_new128.so", 128)
    build(orig_source(), "/tmp/b2ki_b3_orig128.so", [], K=None)   # orig has no B3I_KD macro
    ln = load_b3(ctypes.CDLL("/tmp/b2ki_b3_new128.so"))
    lo = load_b3(ctypes.CDLL("/tmp/b2ki_b3_orig128.so"))
    worst = 0.0
    for (name, B, T, H) in CASES:
        inp = make(B, T, H, 128, 128, seed=1)
        # a lower-triangular `dAkk` in the same [B,T,H,BT] layout the kernel consumes.
        torch.manual_seed(7)
        NT = cdiv(T, BT)
        o_t = torch.arange(BT, device="cuda")
        low = (o_t[:, None] > o_t[None, :]).float()
        m = torch.randn(B, NT, H, BT, BT, device="cuda") * low
        dAkk = m.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
        n = run_b3(ln, inp, dAkk)
        o = run_b3(lo, inp, dAkk)
        print("\n# %s" % name)
        for fld in ("dq", "dk", "db", "dg"):
            worst = max(worst, key_stats("orig vs new %s" % fld, n[fld], o[fld]))
    print("\nPart A0 worst max_abs: %.3e  [%s]" %
          (worst, "BIT-IDENTICAL" if worst == 0.0 else "MISMATCH"))
    return worst


def part_a():
    print("\n" + "=" * 92)
    print("Part A: intra fold x K!=128 -- prod_intra vs prod + standalone b3intra(K)")
    print("=" * 92)
    worst = {}
    for K in KS:
        so_p = "/tmp/b2ki_prod_K%d.so" % K
        so_i = "/tmp/b2ki_intra_K%d.so" % K
        so_c = "/tmp/b2ki_core_K%d.so" % K
        so_b = "/tmp/b2ki_b3_K%d.so" % K
        build(CU, so_p, PROD, K, K)
        build(CU, so_i, INTR, K, K)
        build(CU, so_c, [], K, K)
        build_b3(so_b, K)
        libp, libi = ctypes.CDLL(so_p), ctypes.CDLL(so_i)
        libb = load_b3(ctypes.CDLL(so_b))
        print("\n#### K = V = %d ####" % K)
        smem_i = 0
        libi.b2mma_smem_bytes.restype = ctypes.c_int
        smem_i = libi.b2mma_smem_bytes()
        if smem_i > MAX_SMEM:
            print("  prod_intra smem = %d B = %.1f KiB > %d B (227 KiB) -- INFEASIBLE at K=%d"
                  % (smem_i, smem_i / 1024.0, MAX_SMEM, K))
            worst["infeasible@%d" % K] = float(smem_i)
            continue
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, K, seed=1)
            dw = core_dw(inp, so_c)
            p = run_prod(libp, inp, dw)
            pi = run_prod_intra(libi, inp, dw)
            b3 = run_b3(libb, inp, p["dA"])
            print("\n# K=%d %s  NT=%d" % (K, name, cdiv(T, BT)))
            for fld, add in (("dqg", "dq"), ("dkg", "dk"), ("dg", "dg"), ("db", "db")):
                ref = p[fld].float() + b3[add]
                worst["%s@%d" % (fld, K)] = max(worst.get("%s@%d" % (fld, K), 0.0),
                                                key_stats("prod_intra.%s vs prod+b3.%s" % (fld, add),
                                                          pi[fld], ref))
            # A2 sanity: contribution really added + equals standalone b3 on the same dA
            sa = maxabs(pi["dqg"], p["dqg"])
            sb = maxabs(pi["dqg"] - p["dqg"], b3["dq"])
            print("    sanity: (prod_intra-prod).dqg max_abs=%.3e ; ==b3.dq max_abs=%.3e"
                  % (sa, sb))
            worst["sane@%d" % K] = max(worst.get("sane@%d" % K, 0.0), min(sa, 1.0))
            # unchanged outputs stay bit-identical
            for fld in ("dv2", "dvb", "dkgb", "dA", "dApre", "dgk"):  # HDW leaves dw unwritten
                worst["un_%s@%d" % (fld, K)] = max(worst.get("un_%s@%d" % (fld, K), 0.0),
                                                   maxabs(pi[fld], p[fld]))
    print("\nworst-case (fold vs prod+standalone-intra; 0 == bit-identical):")
    for K in KS:
        for fld in ("dqg", "dkg", "dg", "db"):
            print("  K=%3d %-4s %.3e" % (K, fld, worst.get("%s@%d" % (fld, K), 0.0)))
    print("unchanged outputs (must be 0):")
    for K in KS:
        for fld in ("dv2", "dvb", "dkgb", "dA", "dApre", "dgk"):  # HDW leaves dw unwritten
            print("  K=%3d %-6s %.3e" % (K, fld, worst.get("un_%s@%d" % (fld, K), 0.0)))
    print("sanity (intra contribution non-zero, clipped to <=1):")
    for K in KS:
        print("  K=%3d %.3e" % (K, worst.get("sane@%d" % K, 0.0)))
    print("infeasible (smem > 227 KiB):")
    for K in KS:
        if "infeasible@%d" % K in worst:
            print("  K=%3d prod_intra smem=%.1f KiB  [skipped]"
                  % (K, worst["infeasible@%d" % K] / 1024.0))


def part_b():
    print("\n" + "=" * 92)
    print("Part B: resources -- prod vs prod_intra at each K")
    print("=" * 92)
    print("  %-14s %6s %7s %9s" % ("build", "regs", "spill", "smem"))
    for K in KS:
        txt_p = build(CU, "/tmp/b2ki_rp_%d.so" % K, PROD, K, K)
        txt_i = build(CU, "/tmp/b2ki_ri_%d.so" % K, INTR, K, K)
        for tag, txt, so in (("prod@%d" % K, txt_p, "/tmp/b2ki_rp_%d.so" % K),
                             ("prod_intra@%d" % K, txt_i, "/tmp/b2ki_ri_%d.so" % K)):
            r, sp = ptxas_info(txt)
            lib = ctypes.CDLL(so)
            lib.b2mma_smem_bytes.restype = ctypes.c_int
            print("  %-14s %6d %7d %9d" % (tag, r, sp, lib.b2mma_smem_bytes()))


def part_c():
    print("\n" + "=" * 92)
    print("Part C: timing (CUPTI dev) -- one prod_intra launch vs prod + standalone b3intra")
    print("=" * 92)
    print("  %-22s | %8s %10s | %9s | %9s %11s" %
          ("case", "prod", "prod_intra", "prod+b3", "incl/prod", "fold/(p+b3)"))
    tc = [
        ("B2 T1024 H8", 2, 1024, 8),
        ("T512 H4", 1, 512, 4),
    ]
    for K in KS:
        so_p, so_i = "/tmp/b2ki_tp_%d.so" % K, "/tmp/b2ki_ti_%d.so" % K
        so_c, so_b = "/tmp/b2ki_tc_%d.so" % K, "/tmp/b2ki_tb_%d.so" % K
        build(CU, so_p, PROD, K, K)
        build(CU, so_i, INTR, K, K)
        build(CU, so_c, [], K, K)
        build_b3(so_b, K)
        libp, libi = ctypes.CDLL(so_p), ctypes.CDLL(so_i)
        libb = load_b3(ctypes.CDLL(so_b))
        libi.b2mma_smem_bytes.restype = ctypes.c_int
        if libi.b2mma_smem_bytes() > MAX_SMEM:
            print("  K=%3d  prod_intra smem %.1f KiB > 227 KiB -- skipped"
                  % (K, libi.b2mma_smem_bytes() / 1024.0))
            continue
        for (name, B, T, H) in tc:
            inp = make(B, T, H, K, K, seed=1)
            dw = core_dw(inp, so_c)
            p0 = run_prod(libp, inp, dw)

            def f_prod():
                run_prod(libp, inp, dw)

            def f_fold():
                run_prod_intra(libi, inp, dw)

            def f_old():
                run_prod(libp, inp, dw)
                run_b3(libb, inp, p0["dA"])

            f_prod(); f_fold(); f_old(); torch.cuda.synchronize()
            dp, _ = profile_dev(f_prod, reps=30)
            di, _ = profile_dev(f_fold, reps=30)
            do, _ = profile_dev(f_old, reps=30)
            print("  K=%3d %-16s | %8.1f %10.1f | %9.1f | %9.3f %11.3f" %
                  (K, name, dp, di, do, di / dp, di / do))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    mode = os.environ.get("B2KI_MODE", "")
    if mode in ("", "a0"):
        part_a0()
    if mode in ("", "a"):
        part_a()
    if mode in ("", "b"):
        part_b()
    if mode in ("", "c"):
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
