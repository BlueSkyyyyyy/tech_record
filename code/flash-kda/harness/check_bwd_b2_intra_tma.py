"""P7-67: make the P7-63 `intra` fold (`-DB2M_INTRA`) **composable with the TMA stage layout**
(`-DB2M_TMA`), closing the boundary that docs/07 §67.4 (P7-64) recorded: "`B2M_TMA` 与 `B2M_INTRA`
未打通（TMA 分支未传 `dAqk_in`）".

Root cause (verified in `b2_mma_probe.cu`): the production launch's `#if B2M_TMA_ON` branch of
`b2prod_launch_impl` launched `b2mma_kernel` without the trailing `dAqk` argument that the kernel
signature carries under `B2M_INTRA` -- so `-DB2M_TMA -DB2M_INTRA` did not compile (the non-TMA
`#else` branch forwarded it, the TMA branch did not).  The fix is one argument guard in the TMA
branch (arithmetic untouched).  `b2prod_intra_launch`'s signature is TMA-agnostic (the tensor maps
are encoded *inside* `b2prod_launch_impl` from the plain operand pointers), so no harness-side
launcher change is needed.

Claim verified here: with `-DB2M_PROD -DB2M_INTRA`, the extra `-DB2M_TMA` changes **only the stage
layout** (cp.async -> TMA) and must be **bit-identical** on every output; `dAqk` is read from
global in both layouts, so the folded intra contribution is unaffected.

  Part A  bit-check: `prod_intra_tma` vs `prod_intra` on every output (dv2/dvb/dkgb/db/dApre/dA/
          dqg/dkg/dg/dw/dgk) -- bit-identical (TMA == cp.async, same operands).
  Part B  resources: registers / spill / smem of prod_intra vs prod_intra_tma.
  Part C  timing: CUPTI `dev` + event `wall`, prod_intra vs prod_intra_tma.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_intra_tma.py \
        > .../harness/check_bwd_b2_intra_tma.out.txt 2>&1
    B2IT_MODE=a | b | c
"""
import ctypes
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
sys.path.insert(0, HARNESS)

from check_bwd_b2_intra import (                                    # noqa: E402
    BT, SCALE, CU, CASES, make, _core_dw, run_prod_intra, build,
    load, ptxas_info, SO_CORE,
)
from check_bwd_ws_reuse import cdiv                                  # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev, time_wall             # noqa: E402

SO_INTRA = "/tmp/b2_it_intra.so"
SO_INTRA_TMA = "/tmp/b2_it_intra_tma.so"

FIELDS = ("dv2", "dvb", "dkgb", "db", "dApre", "dA", "dqg", "dkg", "dg", "dw", "dgk")


def build_both():
    txt = build("intra", CU, ["-DB2M_PROD=1", "-DB2M_INTRA=1"], SO_INTRA)
    txt_tma = build("intra_tma", CU, ["-DB2M_PROD=1", "-DB2M_INTRA=1", "-DB2M_TMA=1"],
                    SO_INTRA_TMA)
    return txt, txt_tma


def part_a(cases):
    print("=" * 78)
    print("Part A: prod_intra_tma vs prod_intra -- bit-identity across stage layouts")
    print("=" * 78)
    build_both()
    lib = load(SO_INTRA)
    libt = load(SO_INTRA_TMA)
    worst = 0.0
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        dw = _core_dw(inp)
        p = run_prod_intra(lib, inp, dw)
        pt = run_prod_intra(libt, inp, dw)
        print(f"\n# {name}  NT={cdiv(T,BT)}")
        for fld in FIELDS:
            d = (pt[fld].float() - p[fld].float()).abs().max().item()
            ref = p[fld].float().abs().max().item()
            flag = "OK " if d == 0.0 else "DIFF"
            print(f"    tma.{fld:6s} vs base max_abs={d:.3e}  "
                  f"rel={d/(ref+1e-12):.2e}  ref_max={ref:.3e}  [{flag}]")
            worst = max(worst, d)
    print(f"\n  worst over all tables: max_abs={worst:.3e}  "
          f"[{'BIT-IDENTICAL' if worst == 0.0 else 'MISMATCH'}]")
    return worst


def part_b(txt, txt_tma):
    print("=" * 78)
    print("Part B: resources (registers / spill / smem)")
    print("=" * 78)
    print(f"  {'build':16s} {'regs':>6s} {'spill':>7s} {'smem':>9s}")
    for tag, t, so in (("prod_intra", txt, SO_INTRA), ("prod_intra_tma", txt_tma, SO_INTRA_TMA)):
        r, sp = ptxas_info(t)
        lib = load(so)
        lib.b2mma_smem_bytes.restype = ctypes.c_int
        smem = lib.b2mma_smem_bytes()
        print(f"  {tag:16s} {r:6d} {sp:7d} {smem:9d}")


def part_c(cases):
    print("=" * 78)
    print("Part C: timing -- prod_intra vs prod_intra_tma (CUPTI dev + event wall)")
    print("=" * 78)
    lib = load(SO_INTRA)
    libt = load(SO_INTRA_TMA)
    print(f"  {'case':22s} | {'base dev':>9s} {'tma dev':>9s} | {'ratio':>6s} | "
          f"{'base wall':>9s} {'tma wall':>9s} | {'ratio':>6s}")
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        dw = _core_dw(inp)

        def f_base():
            run_prod_intra(lib, inp, dw)

        def f_tma():
            run_prod_intra(libt, inp, dw)

        f_base(); f_tma(); torch.cuda.synchronize()
        db_, _ = profile_dev(f_base, reps=30)
        dt_, _ = profile_dev(f_tma, reps=30)
        wb = time_wall(f_base, reps=30)
        wt = time_wall(f_tma, reps=30)
        print(f"  {name:22s} | {db_:9.1f} {dt_:9.1f} | {dt_/db_:6.3f} | "
              f"{wb:9.1f} {wt:9.1f} | {wt/wb:6.3f}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build("core", CU, [], SO_CORE)
    mode = os.environ.get("B2IT_MODE", "")
    t = tt = None
    if mode in ("", "a"):
        part_a(CASES)
    if mode in ("", "b"):
        t, tt = build_both()
        part_b(t, tt)
    if mode in ("", "c"):
        part_c(CASES)
    print("\nDONE")


if __name__ == "__main__":
    main()
