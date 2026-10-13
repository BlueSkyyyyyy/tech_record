"""P7-105: close the last boundary §107.6 (P7-104) explicitly left open for the K2-shaped CUDA
B2 production kernel (`b2_mma_probe.cu`): the **multi-knob stacking** of `bf16mma_tma` with
`-DB2M_DAMMA` / `-DB2M_INTRA` / `BF16_VDW` at `K != V`.

§107.6 says verbatim:

  "未测 `-DB2M_DAMMA`/`-DB2M_INTRA`/`BF16_VDW` 与 `bf16mma_tma` 的 `K≠128` 多旋钮叠加；..."

Every single knob has been verified on its own (DAMMA §66/§69/§86, INTRA §65/§66/§92/§95,
bf16mma_tma §77/§80/§86/§89/§106/§107) and each has been crossed with the bf16 native entry
(§79/§80/§81/§82) -- but the **stack `bf16mma_tma + {DAMMA, INTRA}` at `K != V`** has never run.
This round anchors it, using the exact "base vs stacked twin" discipline the suite uses: nothing
in `b2_mma_probe.cu` changes (only existing macros are combined).

Builds (all `-DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_BF16MMA -DB2M_BVK=64 -DB2M_TMA=1` = "bmt"):

  bmt            = bf16mma_tma                      (base)
  bmt_vdw        = + `-DB2M_BF16_VDW`               (proved a compile-time no-op under TMA, §90)
  bmt_da         = + `-DB2M_DAMMA`                  (DA becomes mma.sync.tf32)
  bmt_intra      = + `-DB2M_INTRA`                  (B3 diagonal folded, needs `dAqk`)
  bmt_da_intra   = + `-DB2M_DAMMA -DB2M_INTRA`      (both; the new corner)

Parts:
  A  correctness at `K != V`, on bf16-valued operands (the TMA wire layout: 5 streamed operands
     bf16, raw `v` bf16):
       * `bmt_vdw`        vs `bmt`          -- expect bit-identical (no-op)
       * `bmt_da`         vs `bmt`          -- non-DA fields bit-identical, only dApre/dA move (tf32)
       * `bmt_intra`      vs `bmt + b3intra(K)`  -- dqg/dkg bit, dg/db fp32 add-order ulp
       * `bmt_da_intra`   vs `bmt_da + b3intra(K)` -- same, on top of the DAMMA base
       * all builds vs **pure torch** (`wy_dqkg_torch`) external anchor (FLA fused is not same
         dtype -- it quantizes q/k/g too, §105.6)
  B  resources (registers / spill / sizeof(Shared)) over the `(K,V)` grid, incl. where the
     INTRA fold becomes infeasible (> 227 KiB).
  C  timing (CUPTI `dev`) at `K=V=128` and one `K!=V`: base vs each stacked twin.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=4 python .../harness/check_bwd_b2_kv_stack.py \
        > .../harness/check_bwd_b2_kv_stack.out.txt 2>&1
    B2KS_MODE=a | b | c
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

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs     # noqa: E402
from check_bwd_dqkg_kernel import fla_wy_dqkg, wy_dqkg_torch         # noqa: E402
from check_bwd_ws_reuse import cdiv                                   # noqa: E402
from check_bwd_b2_prod import run_prod, ptr, stream                  # noqa: E402
from check_bwd_b2_intra import run_prod_intra, run_b3, load_b3       # noqa: E402
from check_bwd_b2_kv_anchor3 import quantize, BF16_KEYS, all_bf16    # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                        # noqa: E402

BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")
B3CU = os.path.join(HARNESS, "b3_intra_probe.cu")

PROD = ["-DB2M_PROD=1", "-DB2M_HDW=1"]
BF16 = PROD + ["-DB2M_BF16=1"]
# bf16mma_tma base: native bf16 mma (k16), BVK=64 (bf16 TMA atom = 64 elem -> BVK in {64,128}),
# TMA staging.
BASE = BF16 + ["-DB2M_BF16MMA=1", "-DB2M_BVK=64", "-DB2M_TMA=1"]
BUILDS = {
    "bmt":          BASE,
    "bmt_vdw":      BASE + ["-DB2M_BF16_VDW=1"],
    "bmt_da":       BASE + ["-DB2M_DAMMA=1"],
    "bmt_intra":    BASE + ["-DB2M_INTRA=1"],
    "bmt_da_intra": BASE + ["-DB2M_DAMMA=1", "-DB2M_INTRA=1"],
}

# correctness at K!=V (both polarities + square control); INTRA feasibility checked in Part B.
KV_CORR = [(64, 128), (128, 64), (128, 128)]
KV_SCAN = KV_CORR + [(256, 128), (64, 256), (256, 64)]
CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]
FIELDS = ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk")
CHECKS_OPS = [("dqg", "dq"), ("dkg", "dk"), ("dg", "dg"), ("db", "db"), ("dA", "dA")]


def build_v(extra, K, V, tag):
    so = "/tmp/b2ks_%s_%d_%d.so" % (tag, K, V)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v", "-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V]
    cmd += extra + ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr
    return so, r.stderr


def build(extra, K, V, tag):
    so, _ = build_v(extra, K, V, tag)
    if so is None:
        raise SystemExit("nvcc failed: tag=%s K=%d V=%d" % (tag, K, V))
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


def build_b3(K):
    so = "/tmp/b2ks_b3_%d.so" % K
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v", "-DB3I_KD=%d" % K, "-lcuda", B3CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("nvcc failed: b3 K=%d" % K)
    return load_b3(ctypes.CDLL(so))


def stats(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    denom = b.float().abs().max().item()
    print("    %-40s max_abs=%.3e  max_rel=%.3e  ref_max=%.3e"
          % (name, d, d / (denom + 1e-12), denom))
    return d, d / (denom + 1e-12)


def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    print("    %-40s max_abs=%.3e  [%s]" % (name, d, "BIT" if d == 0.0 else "DIFF"))
    return d


def zero_dw(B, T, H, K):
    return torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)


# ------------------------------------------------------------------------- Part A
def part_a():
    print("=" * 110)
    print("Part A: bf16mma_tma stacked with DAMMA / INTRA / BF16_VDW at K!=V (bf16-valued operands)")
    print("=" * 110)
    worst_bit, worst_ext = {}, {}
    for (K, V) in KV_CORR:
        libs = {}
        for tag, extra in BUILDS.items():
            so = build(extra, K, V, tag)
            libs[tag] = ctypes.CDLL(so)
        libb3 = build_b3(K)
        print("\n#### K = %d, V = %d   smem bmt=%d bmt_da=%d bmt_intra=%d bmt_da_intra=%d bmt_vdw=%d ####"
              % (K, V, smem_bytes("/tmp/b2ks_bmt_%d_%d.so" % (K, V)),
                 smem_bytes("/tmp/b2ks_bmt_da_%d_%d.so" % (K, V)),
                 smem_bytes("/tmp/b2ks_bmt_intra_%d_%d.so" % (K, V)),
                 smem_bytes("/tmp/b2ks_bmt_da_intra_%d_%d.so" % (K, V)),
                 smem_bytes("/tmp/b2ks_bmt_vdw_%d_%d.so" % (K, V))))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            inp_cp, inp_tma, inp_ref = quantize(inp)
            dw = zero_dw(B, T, H, K)
            base = run_prod(libs["bmt"], inp_tma, dw)
            vdw = run_prod(libs["bmt_vdw"], inp_tma, dw)
            da = run_prod(libs["bmt_da"], inp_tma, dw)
            try:
                intra = run_prod_intra(libs["bmt_intra"], inp_tma, dw)
                da_intra = run_prod_intra(libs["bmt_da_intra"], inp_tma, dw)
                rc_intra = 0
            except RuntimeError as e:
                print("\n# %s  INTRA launch FAILED: %s" % (name, e))
                rc_intra = -1
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            for fld in FIELDS:
                worst_bit["vdw vs bmt"] = max(worst_bit.get("vdw vs bmt", 0.0),
                                              bits("vdw.%s vs bmt" % fld, vdw[fld], base[fld]))
            # DAMMA only touches dApre/dA
            for fld in FIELDS:
                d = bits("da.%s vs bmt" % fld, da[fld], base[fld])
                if fld not in ("dApre", "dA"):
                    worst_bit["da vs bmt (non-DA)"] = max(worst_bit.get("da vs bmt (non-DA)", 0.0), d)
            if rc_intra == 0:
                b3 = run_b3(libb3, inp, base["dA"])
                b3d = run_b3(libb3, inp, da["dA"])
                for fld, tk in (("dqg", "dq"), ("dkg", "dk"), ("dg", "dg"), ("db", "db")):
                    worst_bit["intra vs bmt+b3"] = max(
                        worst_bit.get("intra vs bmt+b3", 0.0),
                        bits("intra.%s vs bmt+b3" % fld, intra[fld], base[fld] + b3[tk]))
                    worst_bit["da_intra vs da+b3"] = max(
                        worst_bit.get("da_intra vs da+b3", 0.0),
                        bits("da_intra.%s vs da+b3" % fld, da_intra[fld], da[fld] + b3d[tk]))
                for fld in FIELDS:
                    if fld in ("dqg", "dkg", "dg", "db"):
                        continue
                    worst_bit["intra untouched"] = max(
                        worst_bit.get("intra untouched", 0.0),
                        bits("intra.%s vs bmt" % fld, intra[fld], base[fld]))
                    worst_bit["dai untouched"] = max(
                        worst_bit.get("dai untouched", 0.0),
                        bits("da_intra.%s vs da" % fld, da_intra[fld], da[fld]))
            # external anchor (pure torch; FLA fused is all-bf16 -> different dtype, note only)
            tor = wy_dqkg_torch(inp_ref, SCALE, BT) if T % BT == 0 else None
            fla_b = fla_wy_dqkg(all_bf16(inp), SCALE, BT)
            for tag, out in (("bmt", base), ("bmt_da", da)):
                for fld, tk in CHECKS_OPS:
                    if tor is not None:
                        _, r = stats("%s.%s vs torch" % (tag, fld), out[fld], tor[tk])
                        worst_ext["torch_%s" % fld] = max(worst_ext.get("torch_%s" % fld, 0.0), r)
                    _, r = stats("%s.%s vs FLA(all-bf16)" % (tag, fld), out[fld], fla_b[tk])
                    worst_ext["FLA_%s" % fld] = max(worst_ext.get("FLA_%s" % fld, 0.0), r)
    print("\n" + "-" * 110)
    print("bit-identity of each stacking (worst max_abs over all (K,V)/cases; 0 == bit-identical):")
    for k in sorted(worst_bit):
        print("    %-28s %.3e" % (k, worst_bit[k]))
    print("external anchor worst max_rel (bf16-valued inputs):")
    for fld, _ in CHECKS_OPS:
        print("    %-5s torch %.3e | FLA(all-bf16) %.3e"
              % (fld, worst_ext.get("torch_%s" % fld, -1), worst_ext.get("FLA_%s" % fld, -1)))


# ------------------------------------------------------------------------- Part B
def part_b():
    print("\n" + "=" * 110)
    print("Part B: resources over the (K,V) grid (registers / spill / sizeof(Shared))")
    print("=" * 110)
    print("  %5s %5s | %-14s %6s %7s %10s %10s" %
          ("K", "V", "tag", "regs", "spill", "smem", ">227KiB"))
    for (K, V) in KV_SCAN:
        for tag in ("bmt", "bmt_da", "bmt_intra", "bmt_da_intra"):
            so, txt = build_v(BUILDS[tag], K, V, tag)
            if so is None:
                print("  %5d %5d | %-14s  (build failed)" % (K, V, tag))
                continue
            reg, sp = ptxas_reg(txt)
            sm = smem_bytes(so)
            print("  %5d %5d | %-14s %6d %7d %10d %10s"
                  % (K, V, tag, reg, sp, sm, "YES" if sm > 232448 else ""))


# ------------------------------------------------------------------------- Part C
def part_c():
    print("\n" + "=" * 110)
    print("Part C: timing (CUPTI dev) -- base bf16mma_tma vs each stacked twin")
    print("=" * 110)
    cases = [("K=128 V=128 B8 T2048 H8", 8, 2048, 8, 128, 128),
             ("K=128 V=64  B8 T2048 H8", 8, 2048, 8, 128, 64),
             ("K=64  V=128 B8 T2048 H8", 8, 2048, 8, 64, 128)]
    for (name, B, T, H, K, V) in cases:
        libs = {t: ctypes.CDLL(build(e, K, V, t)) for t, e in BUILDS.items()}
        inp = make(B, T, H, K, V, seed=1)
        _, inp_tma, _ = quantize(inp)
        dw = zero_dw(B, T, H, K)

        def f(tag, is_intra):
            fn = run_prod_intra if is_intra else run_prod
            return lambda: fn(libs[tag], inp_tma, dw)

        print("\n# %s" % name)
        try:
            fns = {"bmt": f("bmt", False), "bmt_da": f("bmt_da", False),
                   "bmt_intra": f("bmt_intra", True), "bmt_da_intra": f("bmt_da_intra", True)}
            for fn in fns.values():
                fn()
            torch.cuda.synchronize()
            devs = {}
            for tag, fn in fns.items():
                devs[tag], _ = profile_dev(fn, reps=30)
            print("  %-14s %10s %10s %10s %10s" % ("build", "dev(us)", "vs bmt", "", ""))
            for tag in ("bmt", "bmt_da", "bmt_intra", "bmt_da_intra"):
                print("  %-14s %10.1f %10.3f" % (tag, devs[tag], devs[tag] / devs["bmt"]))
        except RuntimeError as e:
            print("  INTRA launch failed at this (K,V): %s" % e)


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    mode = os.environ.get("B2KS_MODE", "")
    if mode in ("", "a"):
        part_a()
    if mode in ("", "b"):
        part_b()
    if mode in ("", "c"):
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
