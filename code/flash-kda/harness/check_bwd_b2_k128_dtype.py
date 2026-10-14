"""P7-91: close the last dtype-entry boundary left by the P7-90 `K`-generalization of the
K2-shaped CUDA B2 **six-segment production kernel** (`b2_mma_probe.cu`, docs/07 section 93.6).

P7-89 (section 92) parameterized `K` for the **base GEMM core** and tested `-DB2M_BF16` /
`-DB2M_BF16MMA` at `K != 128`; P7-90 (section 93) generalized the six non-GEMM segments and
tested `fp32` / `TMA` at `K != 128`; P7-83 (section 86) tested `bf16` / `intra` at `K == 128`,
`V != 128`.  The cell nobody built is the **intersection**: the bf16-native operand entry
(`-DB2M_BF16`, `-DB2M_BF16MMA`, `-DB2M_BF16_VDW`) on the **six-segment production kernel at
`K != 128`**.  Section 93.6 names it explicitly (`-DB2M_BF16_VDW x six-segment K!=128` 待验证).

Why the six-segment `K`-generalization (P7-90) is orthogonal to bf16 staging: `K` only ever
appears as the `h/dh` row count (= output N) and in the K-axis accumulators (`acc3`, gate/`intra`
`idq/idk/idg`); the bf16 entry only changes the **storage dtype** of the five streamed operands
and (with VDW) the raw-`v` scratch.  bf16(8-bit mantissa) subset tf32(10-bit) means every
`rd()` returns the *same* fp32 value whether the value was staged bf16 or fp32 -- and that
argument never mentions `K`.  So bit-identity is a derivation, not a hope.

Parts
  Part A  bf16-native entry x K!=128: `prod_bf16` (bf16 operands) vs `prod` (fp32 on the
          bf16-valued fp32 copies) -- every one of the eleven production outputs bit-identical at
          each (K,V).  Plus `prod_bf16_mma` (`-DB2M_BF16MMA`) vs `prod_bf16` (only the GEMM core
          fp32 accumulation order changes).
  Part B  BF16_VDW x K!=128 (the explicit section 93.6 待验证): `prod_bf16_vdw` (raw `v` also
          bf16-staged) vs `prod_bf16` (raw `v` fp32) -- bit-identical.
  Part C  resources: registers / spill / smem of the builds at each (K,V).
  Part D  timing (kernel-only CUPTI `dev`): prod vs prod_bf16 vs prod_bf16_mma vs prod_bf16_vdw.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=3 python .../harness/check_bwd_b2_k128_dtype.py \
        > .../harness/check_bwd_b2_k128_dtype.out.txt 2>&1
    B2KDT_MODE=a | b | c | d
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
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs   # noqa: E402
from check_bwd_ws_reuse import cdiv                                  # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                        # noqa: E402
from check_bwd_b2_prod import run_prod, ptr, stream                  # noqa: E402

BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")

CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]
# K in {64,128,256} (V=K keeps every build legal); the cross (64,128) exercises the P7-82
# V-axis generic path on the new K and (64,64)/(256,256) exercise the K-axis narrow/wide cases.
KV = [(64, 64), (128, 128), (256, 256), (64, 128)]
PROD = ["-DB2M_PROD=1", "-DB2M_HDW=1"]

# the five streamed operands that carry the bf16-native entry (P7-73/75/76)
BF16_KEYS = ("v_new", "h", "dh", "do", "dv")
BUILDS = ["prod", "prod_bf16", "prod_bf16_mma", "prod_bf16_vdw"]
# NOTE: `dw` is deliberately excluded -- under `-DB2M_HDW` it is an *unwritten* scratch buffer
# (the GEMM core keeps `dw` in-kernel `swall`), so a bit-comparison of it is meaningless
# (this is the same caveat check_bwd_b2_k128_prod.py documents, section 93.3).
ALL_FIELDS = ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk")


def extra_for(tag):
    if tag == "prod":
        return PROD
    if tag == "prod_bf16":
        return PROD + ["-DB2M_BF16=1"]
    if tag == "prod_bf16_mma":
        return PROD + ["-DB2M_BF16=1", "-DB2M_BF16MMA=1"]
    if tag == "prod_bf16_vdw":
        return PROD + ["-DB2M_BF16=1", "-DB2M_BF16_VDW=1"]
    raise KeyError(tag)


def build_v(tag, K, V):
    so = "/tmp/b2kdt_%s_k%d_v%d.so" % (tag, K, V)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v", "-DB2M_KD=%d" % K, "-DB2M_VD=%d" % V] + \
          extra_for(tag) + ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed: %s (K=%d V=%d)" % (tag, K, V))
    return so, r.stderr


def build(tag, K, V):
    return build_v(tag, K, V)[0]


def load(so):
    return ctypes.CDLL(so)


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1,
            int(spill.group(1)) if spill else 0)


def make(B, T, H, K, V, seed=1):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def quantize(inp, with_v=False):
    """Return (fp32-valued copy, bf16 copy).  `with_v` also rounds raw `v` (BF16_VDW wire)."""
    keys = BF16_KEYS + (("v",) if with_v else ())
    inp_q = dict(inp)
    inp_b = dict(inp)
    for k in keys:
        b = inp[k].bfloat16()
        inp_q[k] = b.float().contiguous()
        inp_b[k] = b.contiguous()
    return inp_q, inp_b


def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print("    %-36s max_abs=%.3e  [%s]" % (name, d, flag))
    return d


def zero_dw(B, T, H, K):
    return torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)


def part_a():
    print("=" * 96)
    print("Part A: bf16-native entry at K!=128 -- prod_bf16 vs prod (bit-identity) + mma vs bf16")
    print("=" * 96)
    worst_bf16 = 0.0
    worst_mma = 0.0
    for (K, V) in KV:
        libs = {t: load(build(t, K, V)) for t in ("prod", "prod_bf16", "prod_bf16_mma")}
        print("\n#### K = %d  V = %d ####" % (K, V))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            inp_q, inp_b = quantize(inp)
            dw = zero_dw(B, T, H, K)
            pa = run_prod(libs["prod"], inp_q, dw)
            pb = run_prod(libs["prod_bf16"], inp_b, dw)
            pm = run_prod(libs["prod_bf16_mma"], inp_b, dw)
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            for fld in ALL_FIELDS:
                worst_bf16 = max(worst_bf16, bits("K=%d bf16.%s vs prod" % (K, fld),
                                                   pb[fld], pa[fld]))
            # mma only reorders the GEMM core accumulation -> straight outputs differ by ulps
            for fld in ("dv2", "dvb", "dApre", "dA", "db", "dgk", "dkgb"):
                worst_mma = max(worst_mma, bits("K=%d mma.%s vs bf16" % (K, fld),
                                                pm[fld], pb[fld]))
            for fld in ("dqg", "dkg", "dg"):
                bits("K=%d mma.%s vs bf16" % (K, fld), pm[fld], pb[fld])
    print("\nPart A: prod_bf16 vs prod  worst max_abs = %.3e  [%s]"
          % (worst_bf16, "BIT-IDENTICAL" if worst_bf16 == 0.0 else "MISMATCH"))
    print("Part A: mma vs bf16 straight outputs worst max_abs = %.3e" % worst_mma)
    return worst_bf16


def part_b():
    print("\n" + "=" * 96)
    print("Part B: BF16_VDW at K!=128 (section 93.6 待验证) -- prod_bf16_vdw vs prod_bf16")
    print("=" * 96)
    worst = 0.0
    for (K, V) in KV:
        libb = load(build("prod_bf16", K, V))
        libv = load(build("prod_bf16_vdw", K, V))
        print("\n#### K = %d  V = %d ####" % (K, V))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            # The five streamed operands must be bf16 on the wire for both bf16 builds (`rd()`
            # always reads bf16 under `-DB2M_BF16`).  Raw `v`: the bf16 build stages it fp32 (so
            # feed a bf16-valued fp32 copy), the vdw build stages it bf16 (the bf16 tensor).
            inp_qall, inp_bv = quantize(inp, with_v=True)
            _, inp_b = quantize(inp, with_v=False)
            inp_b_v_fp32 = dict(inp_b)
            inp_b_v_fp32["v"] = inp_qall["v"]
            dw = zero_dw(B, T, H, K)
            pb = run_prod(libb, inp_b_v_fp32, dw)
            pv = run_prod(libv, inp_bv, dw)
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            for fld in ALL_FIELDS:
                worst = max(worst, bits("K=%d vdw.%s vs bf16" % (K, fld), pv[fld], pb[fld]))
    print("\nPart B worst max_abs = %.3e  [%s]"
          % (worst, "BIT-IDENTICAL" if worst == 0.0 else "MISMATCH"))
    return worst


def part_c():
    print("\n" + "=" * 96)
    print("Part C: resources (registers / spill / smem) at K!=128")
    print("=" * 96)
    print("  %-16s %5s %5s %7s %8s %10s" % ("build", "K", "V", "regs", "spill", "smem(B)"))
    for (K, V) in KV:
        for tag in BUILDS:
            so, txt = build_v(tag, K, V)
            reg, sp = ptxas_info(txt)
            lib = load(so)
            lib.b2mma_smem_bytes.restype = ctypes.c_int
            smem = lib.b2mma_smem_bytes()
            print("  %-16s %5d %5d %7d %8d %10d" % (tag, K, V, reg, sp, smem))


def part_d():
    print("\n" + "=" * 96)
    print("Part D: timing -- kernel-only CUPTI `dev` (one launch) at K!=128")
    print("=" * 96)
    for (K, V) in KV:
        libs = {t: load(build(t, K, V)) for t in BUILDS}
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, K, V, seed=1)
            inp_q, inp_b = quantize(inp)
            _, inp_bv = quantize(inp, with_v=True)
            dw = zero_dw(B, T, H, K)
            run_prod(libs["prod"], inp_q, dw)
            torch.cuda.synchronize()
            dp, _ = profile_dev(lambda: run_prod(libs["prod"], inp_q, dw), reps=30)
            db, _ = profile_dev(lambda: run_prod(libs["prod_bf16"], inp_b, dw), reps=30)
            dm, _ = profile_dev(lambda: run_prod(libs["prod_bf16_mma"], inp_b, dw), reps=30)
            dv, _ = profile_dev(lambda: run_prod(libs["prod_bf16_vdw"], inp_bv, dw), reps=30)
            print("  K=%3d V=%3d %-20s prod=%8.1f  bf16=%8.1f (%.3fx)  "
                  "bf16mma=%8.1f (%.3fx)  vdw=%8.1f (%.3fx)"
                  % (K, V, name, dp, db, db / dp, dm, dm / dp, dv, dv / dp))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    mode = os.environ.get("B2KDT_MODE", "")
    if mode in ("", "a"):
        part_a()
    if mode in ("", "b"):
        part_b()
    if mode in ("", "c"):
        part_c()
    if mode in ("", "d"):
        part_d()
    print("\nDONE")


if __name__ == "__main__":
    main()
