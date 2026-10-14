"""P7-75: bf16-native operand staging inside the **six-segment production launch**.

P7-73 gave the *base* K2-shaped B2 GEMM core a bf16-native operand entry (`-DB2M_BF16`: the five
streamed operands `do/vn/dv/h/dh` are staged into smem as bf16, halving their DRAM read and smem
footprint), but explicitly excluded the six non-GEMM segments / `-DB2M_PROD` with an `#error`.
P7-74 then made the GEMM core's mma native bf16 (`-DB2M_BF16MMA`).  This round lifts the P7-73
restriction so the whole production launch can be built `-DB2M_BF16 [-DB2M_BF16MMA]` -- i.e. the
deployment dtype (bf16) is now native end-to-end in the CUDA B2, closing the §67.4 boundary
"the CUDA B2 core was fp32/tf32-only, so the chain had to `.float()` B1's bf16 `dh`/`dv2`".

Key fact making this a pure drop-in: **bf16 ⊂ tf32**.  If the five operands are already bf16-valued,
staging them as fp32 (fp32 build) and staging them as bf16 (bf16 build) yield the *same* fp32 values
at every `rd()` (bf16→fp32 widening is exact, `cvt.rna.tf32` a no-op), so **every** output must be
bit-identical -- the non-GEMM segments consume the same de-quantized operands.  The fp32 scratch
operands (`vv` raw v, `dwm`) are always fp32 and are staged at the fp32 cp.async granularity (U4=4)
independently of the bf16 `op_t`; `rds()` reads them without bf16 de-quantization.

  Part A  bit-identity: `prod_bf16` (bf16 operands) vs `prod` (fp32 operands fed the bf16-valued
          fp32 copies) -- every output bit-identical.  Plus `prod_bf16_mma` vs `prod_bf16` (only the
          GEMM core's fp32 accumulation order changes).
  Part B  resources (regs / spill / smem).
  Part C  timing: kernel-only `dev` prod vs prod_bf16 vs prod_bf16_mma.
  Part D  torch sanity: `dv2 = beta*(A@dv)` on the same operands.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_bf16_prod.py \
        > .../harness/check_bwd_b2_bf16_prod.out.txt 2>&1
    B2BP_MODE=a | b | c | d
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

# the five bf16-streamed operands (keys in the `make_inputs` dict)
BF16_KEYS = ("v_new", "h", "dh", "do", "dv")

BUILDS = {
    "core":          [],   # plain GEMM core (no -DB2M_PROD): emits `dw` (b2mma_launch)
    "prod":          [],
    "prod_bf16":     ["-DB2M_BF16=1"],
    "prod_bf16_mma": ["-DB2M_BF16=1", "-DB2M_BF16MMA=1"],
}


def build_v(tag, extra):
    so = f"/tmp/b2_bf16prod_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", "-Xptxas", "-v"]
    if tag != "core":
        cmd += ["-DB2M_PROD=1"]
    cmd += extra + ["-lcuda", CU, "-o", so]
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
    """Return (inp_q, inp_b): fp32 copies of bf16-valued operands, and the bf16 tensors."""
    inp_q = dict(inp)
    inp_b = dict(inp)
    for k in BF16_KEYS:
        b = inp[k].bfloat16()
        inp_q[k] = b.float().contiguous()
        inp_b[k] = b.contiguous()
    return inp_q, inp_b


# ------------------------------------------------------------------------- parts
def bits(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print(f"    {name:30s} max_abs={d:.3e}  [{flag}]")
    return d


def part_a(cases):
    print("=" * 78)
    print("Part A: prod_bf16 vs prod (fp32 on bf16-valued operands) -- bit-identity")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in BUILDS.items()}
    worst = 0.0
    worst_mma = 0.0
    for (name, B, T, H, seed) in cases:
        print(f"\n# {name}")
        inp = make(B, T, H, seed)
        inp_q, inp_b = quantize(inp)
        dw = run_core(libs["core"], inp_q)["dw"]
        pa = run_prod(libs["prod"], inp_q, dw)
        pb = run_prod(libs["prod_bf16"], inp_b, dw)
        pm = run_prod(libs["prod_bf16_mma"], inp_b, dw)
        keys = ["dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dw", "dgk"]
        for k in keys:
            worst = max(worst, bits(f"bf16.{k} vs prod", pb[k], pa[k]))
        # BF16MMA only changes the GEMM core's fp32 accumulation order.
        for k in ("dv2", "dvb", "dApre", "dA", "db"):
            worst_mma = max(worst_mma, bits(f"mma.{k} vs bf16", pm[k], pb[k]))
        print("    -- mma vs tf32 accumulation-order deltas (expected ~1e-7) --")
        for k in ("dw", "dgk", "dqg", "dkg", "dg"):
            bits(f"mma.{k} vs bf16", pm[k], pb[k])
    print(f"\n  prod_bf16 vs prod  worst: max_abs={worst:.3e}  "
          f"[{'BIT-IDENTICAL' if worst == 0.0 else 'MISMATCH'}]")
    print(f"  mma vs bf16 (straight outputs) worst: max_abs={worst_mma:.3e}")


def ptxas_info(txt):
    reg = re.search(r"Used (\d+) registers", txt)
    spill = re.search(r"(\d+) bytes spill stores", txt)
    return (int(reg.group(1)) if reg else -1,
            int(spill.group(1)) if spill else 0)


def part_b():
    print("=" * 78)
    print("Part B: resources (registers / spill / smem)")
    print("=" * 78)
    print(f"  {'build':16s} {'regs':>6s} {'spill':>7s} {'smem':>9s}")
    for tag in BUILDS:
        so, txt = build_v(tag, BUILDS[tag])
        r, sp = ptxas_info(txt)
        lib = load(so)
        lib.b2mma_smem_bytes.restype = ctypes.c_int
        smem = lib.b2mma_smem_bytes()
        print(f"  {tag:16s} {r:6d} {sp:7d} {smem:9d}")


def part_c(cases):
    print("=" * 78)
    print("Part C: timing -- kernel-only dev (one prod launch)")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in BUILDS.items()}
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        inp_q, inp_b = quantize(inp)
        dw = run_core(libs["core"], inp_q)["dw"]
        torch.cuda.synchronize()

        def f(tag, d):
            run_prod(libs[tag], d, dw)

        dev_p, _ = profile_dev(lambda: f("prod", inp_q))
        dev_b, _ = profile_dev(lambda: f("prod_bf16", inp_b))
        dev_m, _ = profile_dev(lambda: f("prod_bf16_mma", inp_b))
        print(f"  {name:20s} prod={dev_p:8.1f}us  bf16={dev_b:8.1f}us ({dev_b/dev_p:.3f}x)"
              f"  bf16mma={dev_m:8.1f}us ({dev_m/dev_p:.3f}x)")


def part_d(cases):
    print("=" * 78)
    print("Part D: torch sanity -- prod_bf16 dv2 = beta*(A@dv) on the same operands")
    print("=" * 78)
    libs = {t: load(build(t, e)) for t, e in BUILDS.items()}
    for (name, B, T, H, seed) in cases:
        print(f"\n# {name}")
        inp = make(B, T, H, seed)
        inp_q, inp_b = quantize(inp)
        dw = run_core(libs["core"], inp_q)["dw"]
        pb = run_prod(libs["prod_bf16"], inp_b, dw)
        NT = cdiv(T, BT)
        dev = inp["q"].device
        A = inp["A"].float()
        beta = inp["beta"].float()
        # OUT uses the tf32 mma on the *staged* dv, which is bf16-valued here -> reference on
        # the bf16-quantized copy (a raw-fp32 reference would add the bf16 quantization floor).
        dv = inp_q["dv"].float()
        Apad = torch.zeros(B, NT * BT, H, BT, device=dev, dtype=torch.float32)
        Apad[:, :T] = A
        dvpad = torch.zeros(B, NT * BT, H, dv.shape[-1], device=dev, dtype=torch.float32)
        dvpad[:, :T] = dv
        A4 = Apad.view(B, NT, BT, H, BT)
        dvv = dvpad.view(B, NT, BT, H, -1)
        dvb_ref = torch.einsum("bcthp,bcthv->bchpv", A4, dvv)
        dvb_ref = dvb_ref.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, dv.shape[-1])[:, :T]
        dv2_ref = dvb_ref * beta.unsqueeze(-1)
        bits("bf16.dv2 vs torch", pb["dv2"], dv2_ref)


def main():
    mode = os.environ.get("B2BP_MODE", "a")
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
