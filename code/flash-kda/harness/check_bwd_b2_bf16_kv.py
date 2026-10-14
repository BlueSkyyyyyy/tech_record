"""P7-83: close P7-82's two remaining K!=V coverage holes for the K2-shaped CUDA B2
production kernel (`b2_mma_probe.cu`).

P7-82 (docs/07 section 85) made the base GEMM core and the six non-GEMM segments generic in
`K != V` (V-axis segments partition by their own `BVW = VD/NW`), but explicitly left two crosses
untested:

  (a) the **bf16-native operand entry** (`-DB2M_BF16`, P7-73/75/76/78/80) had only ever been
      built at `K == V == 128`; and
  (b) the **folded `intra` diagonal** (`-DB2M_INTRA`, P7-63/67) had only been built at
      `K == V == 128`.

This round builds both at `K=128, V in {64,128,256}` and anchors them.

Key facts making the expected answer "bit-identical" a *derivation*, not a hope:
  * **bf16 subset tf32**: if the five streamed operands (`do/vn/dv/h/dh`) are bf16-valued, staging
    them as fp32 (fp32 build) vs bf16 (bf16 build) yields the *same* fp32 values at every `rd()`
    (bf16->fp32 widening is exact, `cvt.rna.tf32` a no-op).  The non-GEMM segments read those same
    de-quantized operands, so **every output must be bit-identical** -- and this argument never
    mentions `V`, so it holds verbatim for `K != V`.
  * the folded `intra` is entirely **K-axis** (`q/k/g`, `dAqk`, `dAkk`, `Mq/Sq`), never touches the
    contraction dim `V`; `dAkk` comes from the in-kernel SAND sandwich and the standalone
    `b3_intra_probe.cu` is itself `V`-independent.

Parts
  Part A  bf16 x K!=V: `prod_bf16` (bf16 operands) vs `prod` (fp32 on the bf16-valued fp32 copies)
          -- every one of the ten production outputs bit-identical at each VD.  Plus
          `prod_bf16_mma` (`-DB2M_BF16MMA`) vs `prod_bf16` (only the GEMM core's fp32 accumulation
          order changes).
  Part B  intra-fold x K!=V: `prod_intra` vs `prod` + standalone `b3intra` (the two things the fold
          replaces) -- `dqg/dkg/dg/db` bit-identical, every non-intra output untouched.
  Part C  resources: registers / spill / smem of the builds at each VD.
  Part D  timing (kernel-only CUPTI `dev`): prod vs prod_bf16 vs prod_bf16_mma vs prod_intra at
          K!=V.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=3 python .../harness/check_bwd_b2_bf16_kv.py \
        > .../harness/check_bwd_b2_bf16_kv.out.txt 2>&1
    B2BKV_MODE=a | b | c | d
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
from check_bwd_b2_intra import run_prod_intra, run_b3, load_b3       # noqa: E402

K = 128
BT = 16
SCALE = 1.0 / (128 ** 0.5)
CU = os.path.join(HARNESS, "b2_mma_probe.cu")
SO_B3 = "/tmp/b2bfkv_b3.so"

CASES = [
    ("B2 T1024 H8", 2, 1024, 8),
    ("T512 H4", 1, 512, 4),
    ("T250 H3 (non-div)", 1, 250, 3),
]
VS = [64, 128, 256]

BF16_KEYS = ("v_new", "h", "dh", "do", "dv")

# tags that go through the plain GEMM core (no -DB2M_PROD): unused here (HDW keeps dw in-kernel)
BUILDS = ["prod", "prod_bf16", "prod_bf16_mma", "prod_intra"]


def extra_for(tag):
    base = ["-DB2M_PROD=1", "-DB2M_HDW=1"]
    if tag == "prod":
        return base
    if tag == "prod_bf16":
        return base + ["-DB2M_BF16=1"]
    if tag == "prod_bf16_mma":
        return base + ["-DB2M_BF16=1", "-DB2M_BF16MMA=1"]
    if tag == "prod_intra":
        return base + ["-DB2M_INTRA=1"]
    raise KeyError(tag)


def build_v(tag, V):
    so = "/tmp/b2bfkv_%s_v%d.so" % (tag, V)
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v", "-DB2M_VD=%d" % V] + extra_for(tag) + \
          ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("nvcc failed: %s (V=%d)" % (tag, V))
    return so, r.stderr


def build(tag, V):
    return build_v(tag, V)[0]


def load(so):
    return ctypes.CDLL(so)


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1,
            int(spill.group(1)) if spill else 0)


def make(B, T, H, V, seed=1):
    inp = dqkg_make_inputs(B, T, H, K, V, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def quantize(inp):
    inp_q = dict(inp)
    inp_b = dict(inp)
    for k in BF16_KEYS:
        b = inp[k].bfloat16()
        inp_q[k] = b.float().contiguous()
        inp_b[k] = b.contiguous()
    return inp_q, inp_b


def make_dAqk(B, T, H, seed):
    NT = cdiv(T, BT)
    torch.manual_seed(seed + 100)
    o_t = torch.arange(BT, device="cuda")
    low_incl = (o_t[:, None] >= o_t[None, :]).float()
    dAqk = torch.randn(B, NT, H, BT, BT, device="cuda") * low_incl
    return dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()


def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print("    %-34s max_abs=%.3e  [%s]" % (name, d, flag))
    return d


def part_a():
    print("=" * 92)
    print("Part A: bf16-native operand staging at K!=V -- prod_bf16 vs prod (bit-identity)")
    print("=" * 92)
    worst = 0.0
    worst_mma = 0.0
    for V in VS:
        libs = {t: load(build(t, V)) for t in ("prod", "prod_bf16", "prod_bf16_mma")}
        print("\n#### V = %d  (K = %d fixed) ####" % (V, K))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, V, seed=1)
            inp_q, inp_b = quantize(inp)
            # `dw` is scratch under HDW (unwritten); pass a zero buffer.
            dw = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
            pa = run_prod(libs["prod"], inp_q, dw)
            pb = run_prod(libs["prod_bf16"], inp_b, dw)
            pm = run_prod(libs["prod_bf16_mma"], inp_b, dw)
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            for fld in ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dgk"):
                worst = max(worst, bits("V=%d bf16.%s vs prod" % (V, fld), pb[fld], pa[fld]))
            for fld in ("dv2", "dvb", "dApre", "dA", "db"):
                worst_mma = max(worst_mma, bits("V=%d mma.%s vs bf16" % (V, fld), pm[fld], pb[fld]))
            for fld in ("dqg", "dkg", "dg", "dgk"):
                bits("V=%d mma.%s vs bf16" % (V, fld), pm[fld], pb[fld])
    print("\nPart A: prod_bf16 vs prod  worst max_abs = %.3e  [%s]"
          % (worst, "BIT-IDENTICAL" if worst == 0.0 else "MISMATCH"))
    print("Part A: mma vs bf16 (straight outputs) worst max_abs = %.3e" % worst_mma)
    return worst


def part_b():
    print("\n" + "=" * 92)
    print("Part B: folded intra diagonal at K!=V -- prod_intra vs prod + standalone b3intra")
    print("=" * 92)
    build_so, _ = build_v("prod", 128)   # ensure the .cu compiles; b3 build below
    so_b3 = "/tmp/b2bfkv_b3.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC", "-shared",
           "-lcuda", os.path.join(HARNESS, "b3_intra_probe.cu"), "-o", so_b3]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("b3 nvcc failed")
    libb = load_b3(load(so_b3))
    worst = {}
    for V in VS:
        libp = load(build("prod", V))
        libi = load(build("prod_intra", V))
        print("\n#### V = %d  (K = %d fixed) ####" % (V, K))
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, V, seed=1)
            inp["dAqk"] = make_dAqk(B, T, H, seed=1)
            dw = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
            p = run_prod(libp, inp, dw)
            pi = run_prod_intra(libi, inp, dw)
            b3 = run_b3(libb, inp, p["dA"])
            print("\n# %s  grid=(%d, %d)" % (name, cdiv(T, BT), B * H))
            _sa = (pi["dqg"].float() - p["dqg"].float()).abs().max().item()
            print("    (prod_intra - prod).dqg max_abs=%.3e (intra contribution added)" % _sa)
            for fld, ref in (("dqg", p["dqg"] + b3["dq"]),
                             ("dkg", p["dkg"] + b3["dk"]),
                             ("dg",  p["dg"] + b3["dg"]),
                             ("db",  p["db"] + b3["db"])):
                key = "V%d_%s" % (V, fld)
                worst[key] = max(worst.get(key, 0.0), bits("fold.%s vs prod+b3" % fld, pi[fld], ref))
            for fld in ("dv2", "dvb", "dkgb", "dA", "dApre", "dgk"):
                key = "V%d_un_%s" % (V, fld)
                worst[key] = max(worst.get(key, 0.0),
                                 bits("untouched.%s vs prod" % fld, pi[fld], p[fld]))
    print("\nworst over all V/cases (0 == bit-identical):")
    for V in VS:
        for fld in ("dqg", "dkg", "dg", "db"):
            print("  V=%3d fold.%-4s %.3e" % (V, fld, worst.get("V%d_%s" % (V, fld), -1)))
        for fld in ("dv2", "dvb", "dkgb", "dA", "dApre", "dgk"):
            print("  V=%3d un.%-6s %.3e" % (V, fld, worst.get("V%d_un_%s" % (V, fld), -1)))
    return worst


def part_c():
    print("\n" + "=" * 92)
    print("Part C: resources (registers / spill / smem) at K!=V")
    print("=" * 92)
    print("  %-14s %5s %8s %7s %10s" % ("build", "V", "regs", "spill", "smem"))
    for V in VS:
        for tag in BUILDS:
            so, txt = build_v(tag, V)
            reg, sp = ptxas_info(txt)
            lib = load(so)
            lib.b2mma_smem_bytes.restype = ctypes.c_int
            smem = lib.b2mma_smem_bytes()
            print("  %-14s %5d %8d %7d %10d" % (tag, V, reg, sp, smem))


def part_d():
    print("\n" + "=" * 92)
    print("Part D: timing -- kernel-only CUPTI `dev` (one launch) at K!=V")
    print("=" * 92)
    for V in VS:
        libs = {t: load(build(t, V)) for t in ("prod", "prod_bf16", "prod_bf16_mma", "prod_intra")}
        for (name, B, T, H) in CASES:
            inp = make(B, T, H, V, seed=1)
            inp["dAqk"] = make_dAqk(B, T, H, seed=1)
            inp_q, inp_b = quantize(inp)
            dw = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)

            def f(tag, d):
                if tag == "prod_intra":
                    run_prod_intra(libs[tag], d, dw)
                else:
                    run_prod(libs[tag], d, dw)

            f("prod", inp_q)
            torch.cuda.synchronize()
            dp, _ = profile_dev(lambda: f("prod", inp_q), reps=30)
            db, _ = profile_dev(lambda: f("prod_bf16", inp_b), reps=30)
            dm, _ = profile_dev(lambda: f("prod_bf16_mma", inp_b), reps=30)
            di, _ = profile_dev(lambda: f("prod_intra", inp_q), reps=30)
            print("  V=%3d %-20s prod=%8.1f  bf16=%8.1f (%.3fx)  "
                  "bf16mma=%8.1f (%.3fx)  intra=%8.1f (%.3fx)"
                  % (V, name, dp, db, db / dp, dm, dm / dp, di, di / dp))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("GPU: %s  torch=%s" % (torch.cuda.get_device_name(), torch.__version__))
    mode = os.environ.get("B2BKV_MODE", "")
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
