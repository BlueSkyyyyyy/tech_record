"""P7-78: the bf16-native operand entry **times TMA** inside the six-segment production launch.

P7-73 gave the base K2-shaped B2 GEMM core a bf16-native operand entry (`-DB2M_BF16`); P7-75 lifted
it to the six-segment production launch (`-DB2M_PROD`, non-TMA); P7-76 composed `-DB2M_BF16` with
`-DB2M_TMA` (base GEMM core only); P7-77 composed `-DB2M_BF16MMA` with `-DB2M_TMA` (base GEMM core
only).  Every one of those rounds left the **production launch** (`-DB2M_PROD`) interacting with TMA
as an explicit boundary: P7-75 "仍仅非 TMA", P7-76/P7-77 "仅基础 GEMM 核".  This round closes that
gap -- `-DB2M_PROD -DB2M_BF16 -DB2M_TMA` (and `+ -DB2M_BF16MMA`) compile and are numerically anchored
to the fp32 production launch.

Why it should be a pure drop-in: bf16 ⊂ tf32, so on *bf16-valued* operands the bf16 staging path
widens to the same fp32 value at every `rd()`/`rds()` -- TMA only changes the smem layout.  The one
production-specific subtlety is that under `-DB2M_BF16 -DB2M_TMA` the **fp32 scratch** operands
(`vv` raw v, `dwm` dw) are *also* staged bf16 (a single bf16 tensor map per operand, P7-76), whereas
the non-TMA `-DB2M_BF16` path keeps them fp32 via `rds()`.  So the operating-point comparison feeds
**bf16-valued** `v` and `dw` to both (as they are on the production wire), making the bf16 staging
exact again.

  Part A  bit-identity at a fixed layout: `prod_bf16_tma` vs `prod_tma` (both TMA/BVK=64) and
          `prod_bf16` vs `prod` (both cp.async/BVK=32) -- every output bit-identical.
          Cross-layout (`prod_tma` vs `prod`): the GEMM-core `dw/dqg/dkg` are bit-identical (k8
          accumulation order independent of BVK, P7-53) while `dgk` differs only by the fp32 grouping
          of the V-reduction.  `prod_bf16mma_tma` vs `prod_bf16_tma` = native-bf16 mma: accumulation
          order only.
  Part B  resources (regs / spill / smem) of the five production builds.
  Part C  timing: kernel-only `dev` for prod / prod_tma / prod_bf16 / prod_bf16_tma / bf16mma_tma.
  Part D  torch sanity: `dv2 = beta*(A@dv)` on the bf16-valued operands.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_bf16_tma_prod.py \
        > .../harness/check_bwd_b2_bf16_tma_prod.out.txt 2>&1
    B2TP_MODE=a | b | c | d
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
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dqkg_kernel import make_inputs as dqkg_make_inputs    # noqa: E402
from check_bwd_ws_reuse import cdiv                                   # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                         # noqa: E402
from check_bwd_b2_prod import run_core, run_prod, ptr, stream         # noqa: E402

CASES = [
    ("B8 T2048 H8", 8, 2048, 8, 1),
    ("B2 T1024 H8", 2, 1024, 8, 1),
    ("T512 H4", 1, 512, 4, 1),
    ("T250 H3 (non-div)", 1, 250, 3, 1),
]
BT = 16
CU = os.path.join(HARNESS, "b2_mma_probe.cu")

# The staged operands split in two.  `op_t`-typed operands (`do/vn/dv/h/dh`) go through `rd()`,
# always bf16-staged under `-DB2M_BF16`.  The scratch operands (`v` raw v `vv`, `dw` `dwm`) go
# through `rds()`: **fp32** under non-TMA `-DB2M_BF16` (P7-75), **bf16** under `-DB2M_BF16 -DB2M_TMA`
# (P7-76, one bf16 tensor map covers every operand).  So the "on the wire" bf16 set differs per build.
BF16_OP_T = ("v_new", "h", "dh", "do", "dv")   # always bf16 under -DB2M_BF16
BF16_SCRATCH_TMA = ("v",)                       # bf16 only under -DB2M_BF16 -DB2M_TMA

PROD = ["-DB2M_PROD=1"]
BUILDS = {
    "core":             [],   # plain GEMM core (non-PROD): emits `dw` (b2mma_launch, fp32)
    "prod":             PROD,                                              # fp32  cp.async BVK=32
    "prod_tma32":       PROD + ["-DB2M_TMA=1"],                            # fp32  TMA      BVK=32
    "prod_tma":         PROD + ["-DB2M_TMA=1", "-DB2M_BVK=64"],            # fp32  TMA      BVK=64
    "prod_bf16":        PROD + ["-DB2M_BF16=1"],                           # bf16  cp.async BVK=32
    "prod_bf16_64":     PROD + ["-DB2M_BF16=1", "-DB2M_BVK=64"],           # bf16  cp.async BVK=64
    "prod_bf16_tma":    PROD + ["-DB2M_BF16=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],   # bf16 TMA BVK=64
    "prod_bf16mma_tma": PROD + ["-DB2M_BF16=1", "-DB2M_BF16MMA=1", "-DB2M_TMA=1", "-DB2M_BVK=64"],
}


def build_v(tag, extra):
    so = f"/tmp/b2_bf16tmaprod_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v"] + extra + ["-lcuda", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    return so, r.stderr


def build(tag, extra):
    return build_v(tag, extra)[0]


def load(so):
    return ctypes.CDLL(so)


def make(B, T, H, seed=1):
    inp = dqkg_make_inputs(B, T, H, 128, 128, BT, "cuda", dtype=torch.float32, seed=seed)
    return {k: (v if v.is_contiguous() else v.contiguous()) for k, v in inp.items()}


def quantize(inp):
    """Three operand dicts matching the three build families.

    - `fp32`  : every output-relevant operand fp32 (== the plain production launch's wire format).
    - `bf16`  : `do/vn/dv/h/dh` bf16, raw `v` fp32 (non-TMA `-DB2M_BF16`, P7-75).
    - `bf16t` : `do/vn/dv/h/dh` **and** raw `v` bf16 (`-DB2M_BF16 -DB2M_TMA`, P7-76).
    All three carry the *same numeric values* (each bf16-valued operand is `bf16(x).float()` in the
    fp32 copy), so any correct staging path must read identical operands.
    """
    def bf(x):
        return x.bfloat16()

    fp32 = dict(inp)
    bf16 = dict(inp)
    bf16t = dict(inp)
    for k in BF16_OP_T:
        b = bf(inp[k])
        fp32[k] = b.float().contiguous()
        bf16[k] = b.contiguous()
        bf16t[k] = b.contiguous()
    vb = bf(inp["v"])
    fp32["v"] = vb.float().contiguous()
    bf16["v"] = vb.float().contiguous()   # non-TMA -DB2M_BF16 keeps raw v fp32 (rds)
    bf16t["v"] = vb.contiguous()          # TMA -DB2M_BF16 stages raw v bf16
    return fp32, bf16, bf16t


def stage_dw(dw):
    """dw as fed to the fp32 (rds) path and the bf16 (TMA atom) path -- both bf16-valued."""
    dwb = dw.bfloat16()
    return dwb.float().contiguous(), dwb.contiguous()


# ------------------------------------------------------------------------- parts
def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print(f"    {name:34s} max_abs={d:.3e}  [{flag}]")
    return d


KEYS = ["dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dw", "dgk"]


def part_a(cases):
    print("=" * 78)
    print("Part A: bf16 x TMA in the six-segment production launch -- bit-identity")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in BUILDS.items()}
    w_same = 0.0      # prod_bf16_tma vs prod_tma (bf16 staging at fixed TMA/BVK64)
    w_nontma = 0.0    # prod_bf16 vs prod     (bf16 staging at fixed cp.async/BVK32)
    w_cross = 0.0     # prod_tma vs prod      (cross-layout, fp32): dgk-only fp32 grouping
    w_mma = 0.0       # bf16mma_tma vs bf16_tma: accumulation-order only
    for (name, B, T, H, seed) in cases:
        print(f"\n# {name}")
        inp = make(B, T, H, seed)
        fp32, bf16, bf16t = quantize(inp)
        dw = run_core(libs["core"], fp32)["dw"]
        dwq, dwb = stage_dw(dw)

        pa = run_prod(libs["prod"], fp32, dwq)
        pat = run_prod(libs["prod_tma"], fp32, dwq)
        pb = run_prod(libs["prod_bf16"], bf16, dwq)
        patb = run_prod(libs["prod_bf16_tma"], bf16t, dwb)
        patm = run_prod(libs["prod_bf16mma_tma"], bf16t, dwb)

        print("  -- bf16 staging is exact on bf16-valued operands (same layout) --")
        for k in KEYS:
            w_same = max(w_same, bits(f"bf16_tma.{k} vs tma", patb[k], pat[k]))
        for k in KEYS:
            w_nontma = max(w_nontma, bits(f"bf16.{k} vs prod", pb[k], pa[k]))
        print("  -- cross-layout (fp32): TMA/BVK64 vs cp.async/BVK32 --")
        for k in KEYS:
            d = bits(f"tma.{k} vs prod", pat[k], pa[k])
            if k != "dgk":
                w_cross = max(w_cross, d)
        print("  -- native-bf16 mma: order only --")
        for k in ("dv2", "dvb", "dApre", "dA", "db"):
            w_mma = max(w_mma, bits(f"mma.{k} vs bf16_tma", patm[k], patb[k]))
        for k in ("dw", "dgk", "dqg", "dkg", "dg"):
            bits(f"mma.{k} vs bf16_tma", patm[k], patb[k])
    print(f"\n  prod_bf16_tma vs prod_tma   worst non-dgk: max_abs={w_same:.3e}  "
          f"[{'BIT-IDENTICAL' if w_same == 0.0 else 'MISMATCH'}]")
    print(f"  prod_bf16     vs prod       worst      : max_abs={w_nontma:.3e}  "
          f"[{'BIT-IDENTICAL' if w_nontma == 0.0 else 'MISMATCH'}]")
    print(f"  prod_tma      vs prod       worst non-dgk: max_abs={w_cross:.3e} (BVK ordering)")
    print(f"  bf16mma_tma   vs bf16_tma   worst straight: max_abs={w_mma:.3e}")


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1,
            int(spill.group(1)) if spill else 0)


def part_b():
    print("=" * 78)
    print("Part B: resources (registers / spill / smem)")
    print("=" * 78)
    print(f"  {'build':18s} {'regs':>6s} {'spill':>7s} {'smem':>9s}")
    for tag in BUILDS:
        so, txt = build_v(tag, BUILDS[tag])
        r, sp = ptxas_info(txt)
        lib = load(so)
        lib.b2mma_smem_bytes.restype = ctypes.c_int
        smem = lib.b2mma_smem_bytes()
        print(f"  {tag:18s} {r:6d} {sp:7d} {smem:9d}")


def part_c(cases):
    print("=" * 78)
    print("Part C: timing -- kernel-only dev (one prod launch)")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in BUILDS.items()}
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        fp32, bf16, bf16t = quantize(inp)
        dw = run_core(libs["core"], fp32)["dw"]
        dwq, dwb = stage_dw(dw)
        torch.cuda.synchronize()

        dev_p = profile_dev(lambda: run_prod(libs["prod"], fp32, dwq))[0]
        dev_t32 = profile_dev(lambda: run_prod(libs["prod_tma32"], fp32, dwq))[0]
        dev_t = profile_dev(lambda: run_prod(libs["prod_tma"], fp32, dwq))[0]
        dev_b = profile_dev(lambda: run_prod(libs["prod_bf16"], bf16, dwq))[0]
        dev_b64 = profile_dev(lambda: run_prod(libs["prod_bf16_64"], bf16, dwq))[0]
        dev_bt = profile_dev(lambda: run_prod(libs["prod_bf16_tma"], bf16t, dwb))[0]
        dev_mt = profile_dev(lambda: run_prod(libs["prod_bf16mma_tma"], bf16t, dwb))[0]
        print(f"  {name:20s} prod={dev_p:8.1f}us")
        print(f"    fp32 : tma32={dev_t32:8.1f}({dev_t32/dev_p:.3f}x) tma64={dev_t:8.1f}({dev_t/dev_p:.3f}x)")
        print(f"    bf16 : bf16_32={dev_b:8.1f}({dev_b/dev_p:.3f}x) bf16_64={dev_b64:8.1f}({dev_b64/dev_p:.3f}x)"
              f" bf16_tma64={dev_bt:8.1f}({dev_bt/dev_p:.3f}x) bf16mma_tma64={dev_mt:8.1f}({dev_mt/dev_p:.3f}x)")
        print(f"    TMA@64 vs cp.async@64: fp32 {dev_t/dev_p:.3f}x/'prod' ; bf16 {dev_bt/dev_b64:.3f}x")


def part_d(cases):
    print("=" * 78)
    print("Part D: torch sanity -- prod_bf16_tma dv2 = beta*(A@dv) on bf16-valued operands")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in BUILDS.items()}
    for (name, B, T, H, seed) in cases:
        print(f"\n# {name}")
        inp = make(B, T, H, seed)
        fp32, bf16, bf16t = quantize(inp)
        dw = run_core(libs["core"], fp32)["dw"]
        _, dwb = stage_dw(dw)
        pb = run_prod(libs["prod_bf16_tma"], bf16t, dwb)
        NT = cdiv(T, BT)
        dev = inp["q"].device
        A = inp["A"].float()
        beta = inp["beta"].float()
        dv = fp32["dv"].float()
        Apad = torch.zeros(B, NT * BT, H, BT, device=dev, dtype=torch.float32)
        Apad[:, :T] = A
        dvpad = torch.zeros(B, NT * BT, H, dv.shape[-1], device=dev, dtype=torch.float32)
        dvpad[:, :T] = dv
        A4 = Apad.view(B, NT, BT, H, BT)
        dvv = dvpad.view(B, NT, BT, H, -1)
        dvb_ref = torch.einsum("bcthp,bcthv->bchpv", A4, dvv)
        dvb_ref = dvb_ref.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, dv.shape[-1])[:, :T]
        dv2_ref = dvb_ref * beta.unsqueeze(-1)
        bits("bf16_tma.dv2 vs torch", pb["dv2"], dv2_ref)


def main():
    mode = os.environ.get("B2TP_MODE", "a")
    if mode == "a":
        part_a(CASES)
    elif mode == "b":
        part_b()
    elif mode == "c":
        part_c(CASES)
    elif mode == "d":
        part_d(CASES)


if __name__ == "__main__":
    main()
