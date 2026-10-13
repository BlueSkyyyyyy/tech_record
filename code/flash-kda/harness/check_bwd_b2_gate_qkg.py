"""P7-58: recover the gate-chain cost (§60.5's open lever) by **coalescing the raw `q`/`k`/`g`
into smem once per chunk** (`-DB2M_QKG`) instead of P7-57's fragment-address global reads
(§60.4 = 9.2x global-load sectors).

The staging pass is fully coalesced (one `[BT,KD]` tile each, like `kg`); the gate then reads
`q`/`k`/`g` from shared memory.  It must reproduce the P7-57 gate **bit-for-bit** (same arithmetic,
only the operand fetch changes).

  Part A  correctness: CUDA final `dq`/`dk`/`dg` vs the P7-57 gate build (bit-identity expected),
          vs the same torch gate applied to the CUDA raw (arithmetic), and vs FLA (tf32 floor);
          bit-identity across non-TMA / TMA@32 / TMA@64.
  Part B  timing: gate_qkg vs gate (same staging), plus the TMA variants, and vs full Triton B2.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_gate_qkg.py \
        > .../harness/check_bwd_b2_gate_qkg.out.txt 2>&1
    B2GQ_MODE=a | b
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_b2_gate import (build, load, make, run_core, run_gate,       # noqa: E402
                               ref_gate, CASES, BT, SCALE)                  # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton  # noqa: E402
from check_bwd_ws_reuse import stats, cdiv                                  # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev                               # noqa: E402
from check_bwd_dqkg_kernel import fla_wy_dqkg                               # noqa: E402

# Same signature/launcher; only the operand fetch differs.
VARIANTS = [
    ("core",         []),
    ("gate",         ["-DB2M_DKGB=1", "-DB2M_GATE=1"]),
    ("gate_qkg",     ["-DB2M_DKGB=1", "-DB2M_GATE=1", "-DB2M_QKG=1"]),
    ("gate_qkgp",    ["-DB2M_DKGB=1", "-DB2M_GATE=1", "-DB2M_QKG=1", "-DB2M_QKGP=1"]),
    ("gate_tma",     ["-DB2M_DKGB=1", "-DB2M_GATE=1", "-DB2M_TMA=1"]),
    ("gate_tma_qkg", ["-DB2M_DKGB=1", "-DB2M_GATE=1", "-DB2M_TMA=1", "-DB2M_QKG=1"]),
    ("gate_tma_qkgp", ["-DB2M_DKGB=1", "-DB2M_GATE=1", "-DB2M_TMA=1", "-DB2M_QKG=1", "-DB2M_QKGP=1"]),
    ("gate_tma64_qkg", ["-DB2M_DKGB=1", "-DB2M_GATE=1", "-DB2M_TMA=1", "-DB2M_BVK=64", "-DB2M_QKG=1"]),
]
# Of these, the ones whose dq/dk/dg must match the P7-57 `gate` build bit-for-bit.
QKG_TAGS = ("gate_qkg", "gate_qkgp", "gate_tma_qkg", "gate_tma_qkgp", "gate_tma64_qkg")
TIMING_TAGS = ("gate", "gate_qkg", "gate_qkgp", "gate_tma", "gate_tma_qkg",
               "gate_tma_qkgp", "gate_tma64_qkg")


def part_a(libs):
    print("=" * 96)
    print("Part A: correctness -- gate_qkg (coalesced q/k/g) vs P7-57 gate (bit-id), vs torch/FLA")
    print("=" * 96)
    worst = {}
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        core = run_core(libs["core"], inp)
        base = run_gate(libs["gate"], inp, core["dw"])
        outs = {tag: run_gate(libs[tag], inp, core["dw"]) for tag in QKG_TAGS}
        ref = ref_gate(inp, core["dq"], core["dk"], core["dgk"], base["dkgb"])
        fla = fla_wy_dqkg(inp, SCALE, BT)
        print(f"\n# {name}  grid=({cdiv(T, BT)}, {B*H})")
        for fld in ("dqg", "dkg", "dg"):
            _, mr = stats(f"  {fld} vs torch(same raw)", base[fld], ref[fld])
            worst[f"{fld}_torch"] = max(worst.get(f"{fld}_torch", 0.0), mr)
        for fld, ffla in (("dqg", "dq"), ("dkg", "dk"), ("dg", "dg")):
            _, mr = stats(f"  {ffla} vs FLA", base[fld], fla[ffla])
            worst[f"{ffla}_fla"] = max(worst.get(f"{ffla}_fla", 0.0), mr)
        for tag in QKG_TAGS:
            for fld in ("dqg", "dkg", "dg"):
                _, mrid = stats(f"  {fld} vs {tag}", base[fld], outs[tag][fld])
                worst[f"{fld}_{tag}"] = max(worst.get(f"{fld}_{tag}", 0.0), mrid)
    print("\nworst max_rel over all shapes:")
    for kk in sorted(worst):
        print(f"  {kk:22s} {worst.get(kk, 0.0):.3e}")


def part_b(libs, tk):
    print("\n" + "=" * 96)
    print("Part B: timing -- gate_qkg vs gate (same staging) + TMA, and vs full Triton B2")
    print("=" * 96)
    print(f"\n  {'case':18s} | {'gate':>8s} {'gate_qkg':>9s} {'gate_qkgp':>10s} {'gate_tma':>9s} "
          f"{'tma_qkg':>8s} {'tma_qkgp':>9s} {'tma64_qkg':>10s} | {'fullB2':>8s} | "
          f"{'qkg/gate':>9s} {'qkgp/gate':>10s} {'tqkg/tma':>9s} {'tqkgp/tma':>10s}")
    for (name, B, T, H, _) in CASES:
        inp = make(B, T, H)
        dw = run_core(libs["core"], inp)["dw"]
        for tag in TIMING_TAGS:
            run_gate(libs[tag], inp, dw)
        torch.cuda.synchronize()
        dev = {}
        for tag in TIMING_TAGS:
            dev[tag], _ = profile_dev(lambda lib=libs[tag]: run_gate(lib, inp, dw), reps=40)
        dev["fullB2"], _ = profile_dev(
            lambda: wy_dqkg_triton(tk, inp, SCALE, BT, num_warps=4), reps=40)
        print(f"  {name:18s} | {dev['gate']:8.1f} {dev['gate_qkg']:9.1f} {dev['gate_qkgp']:10.1f} "
              f"{dev['gate_tma']:9.1f} {dev['gate_tma_qkg']:8.1f} {dev['gate_tma_qkgp']:9.1f} "
              f"{dev['gate_tma64_qkg']:10.1f} | {dev['fullB2']:8.1f} | "
              f"{dev['gate_qkg']/dev['gate']:8.3f}x {dev['gate_qkgp']/dev['gate']:9.3f}x "
              f"{dev['gate_tma_qkg']/dev['gate_tma']:8.3f}x {dev['gate_tma_qkgp']/dev['gate_tma']:9.3f}x")
    print("\n  note: qkg stages one coalesced [BT,KD] tile for q/k/g per chunk (like `kg`); `p`")
    print("  pads the row stride to KD+4 to kill the fixed-column row-aliasing bank conflict.")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    libs = {}
    for tag, extra in VARIANTS:
        libs[tag] = load(build(tag, extra))
    tk = build_dqkg()
    mode = os.environ.get("B2GQ_MODE", "")
    if mode in ("", "a"):
        part_a(libs)
    if mode in ("", "b"):
        part_b(libs, tk)
    print("\nDONE")


if __name__ == "__main__":
    main()
