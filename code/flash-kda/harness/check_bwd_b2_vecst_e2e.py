"""P7-71: close the P7-70 boundary (docs/07 §73.6) -- **`-DB2M_VECST` into the end-to-end chain**.

P7-70 (§73) wired the `st2` pairwise store into the `[BT,BT]` `dA`/`dApre` stores and measured it
**only** on the isolated `prod_hdw` build (standalone `da`/`sand`, and the six-segment production
kernel `-DB2M_PROD -DB2M_HDW`).  It left as an explicit boundary: "`-DB2M_VECST` 只测到 `prod_hdw`
（非 TMA、非 INTRA 折叠）；未与 `pci*`（`PROD+HDW+INTRA`）整链、`-DB2M_TMA` 交叉重测".  This round
closes it.

Eight fold kernels (all `-DB2M_PROD -DB2M_HDW -DB2M_INTRA`, i.e. the single-launch B2 that also folds
the B3 diagonal, P7-63), a full 2x2x2 of {DA impl} x {stage layout} x {store width}:

    tag                flags
    ----------------   --------------------------------------------------
    pci                (scalar DA)                                  <- P7-64/P7-69 baseline
    pci_vec            + -DB2M_VECST
    pci_tma            + -DB2M_TMA
    pci_tma_vec        + -DB2M_TMA -DB2M_VECST
    pci_mma            + -DB2M_DAMMA
    pci_mma_vec        + -DB2M_DAMMA -DB2M_VECST
    pci_mma_tma        + -DB2M_DAMMA -DB2M_TMA
    pci_mma_tma_vec    + -DB2M_DAMMA -DB2M_TMA -DB2M_VECST

each chained as `dAv(Triton) -> CUDA-B1 -> CUDA-B2-prod_intra -> reverse cumsum`, eating only
FlashKDA's 16-chunk K1 workspace.

Parts
-----
  A  correctness (4 TCASES incl. non-divisible T=250, B=2):
     (i)  every `*_vec` vs its non-VECST twin -- **all six gradients bit-identical** (VECST only
          changes the store *width*, arithmetic untouched);
     (ii) each variant vs FLA autograd (worst-case max_rel) -- unchanged by VECST.
  B  chain dev/wall (idle card): FLA reverse vs `cf` (Triton fused) vs the eight fold variants,
     plus the `*_vec/base` ratios.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_vecst_e2e.py \
        > .../harness/check_bwd_b2_vecst_e2e.out.txt 2>&1
    B2VE_MODE=correctness | timing
"""
import ctypes
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import stats, D                                     # noqa: E402
from check_bwd_b1_bf16_e2e import prep, fla_golden, time_wall, profile_dev  # noqa: E402
from check_bwd_b1_cuda_e2e import (build as build_b1, load as load_so,      # noqa: E402
                                   TCASES, DEPLOY_FLAGS)
from check_bwd_prod_e2e import build_b2, make_chain_prod                    # noqa: E402
from check_bwd_cudab1_fused_e2e import make_chain_cuda_fused                # noqa: E402

FOLD = ["-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_INTRA=1"]
VEC = ["-DB2M_VECST=1"]
TMA = ["-DB2M_TMA=1"]
MMA = ["-DB2M_DAMMA=1"]

# 8 = {scalar, mma} x {non-TMA, TMA} x {scalar store, VECST}
VARIANTS = [
    ("pci",              FOLD),
    ("pci_vec",          FOLD + VEC),
    ("pci_tma",          FOLD + TMA),
    ("pci_tma_vec",      FOLD + TMA + VEC),
    ("pci_mma",          FOLD + MMA),
    ("pci_mma_vec",      FOLD + MMA + VEC),
    ("pci_mma_tma",      FOLD + MMA + TMA),
    ("pci_mma_tma_vec",  FOLD + MMA + TMA + VEC),
]
# the four (VECST on/off) twins we assert bit-identical
TWINS = [("pci", "pci_vec"), ("pci_tma", "pci_tma_vec"),
         ("pci_mma", "pci_mma_vec"), ("pci_mma_tma", "pci_mma_tma_vec")]
GRADS = ["dq", "dk", "dv", "dbeta", "dg", "dh0"]


def build_all():
    libs = {}
    for tag, flags in VARIANTS:
        so = build_b2(f"vecst_{tag}", flags)
        libs[tag] = ctypes.CDLL(so)
    return libs


def _cmp(a, b, fld):
    return (a[fld].float() - b[fld].float()).abs().max().item()


def part_a(libs, scale, lower_bound):
    print("=" * 104)
    print("Part A: VECST folded kernels in the end-to-end chain vs FLA autograd (and twins bit-identical)")
    print("=" * 104)
    worst = {t: {} for t, _ in VARIANTS}
    worst_tw = {f"{a}-{b}": {} for a, b in TWINS}
    for (name, B, T, H, seed, gs) in TCASES:
        print(f"\n# {name}  B={B} T={T} H={H} seed={seed} g_scale={gs}")
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        chains = {tag: make_chain_prod(libs_b1, libs[tag], True, fi, inp, qn, kn, do, dfin, scale, 4)
                  for tag, _ in VARIANTS}
        res = {tag: chains[tag]() for tag, _ in VARIANTS}
        # (i) twins
        bit = []
        for a, b in TWINS:
            for fld in GRADS:
                d = _cmp(res[b], res[a], fld)
                worst_tw[f"{a}-{b}"][fld] = max(worst_tw[f"{a}-{b}"].get(fld, 0.0), d)
            allz = all(worst_tw[f"{a}-{b}"][fld] == 0 for fld in GRADS)
            bit.append(f"{b[4:]}-{a[4:]}={'BIT' if allz else 'DIFF'}")
        print("      twins:", " ".join(bit))
        # (ii) vs FLA
        for tag, _ in VARIANTS:
            row = []
            for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"), ("dg", "g")]:
                _, m = stats(f"  {tag:16s} {fld} vs FLA", res[tag][fld], gold[gk])
                worst[tag][fld] = max(worst[tag].get(fld, 0.0), m)
            _, m = stats(f"  {tag:16s} dh0 vs FLA",
                         res[tag]["dh0"], gold["h0"].transpose(-1, -2))
            worst[tag]["dh0"] = max(worst[tag].get("dh0", 0.0), m)
            row = " ".join(f"{f}={worst[tag][f]:.2e}" for f in GRADS)
            print(f"      {tag:16s} {row}")
        del chains, res

    print("\nworst-case max_rel vs FLA autograd across cases:")
    print("  " + f"{'fld':6s}" + "".join(f"{t:>17s}" for t, _ in VARIANTS))
    for fld in GRADS:
        print(f"  {fld:6s}" + "".join(f"{worst[t].get(fld, 0.0):17.3e}" for t, _ in VARIANTS))
    print("\nworst-case max_abs (twin vs twin, expect all 0):")
    for a, b in TWINS:
        row = "  ".join(f"{fld}={worst_tw[f'{a}-{b}'].get(fld, 0.0):.1e}" for fld in GRADS)
        print(f"  {b:16s} vs {a:14s} {row}")


def part_b(libs, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 104)
    print("Part B: backward-only chain dev/wall (idle card): FLA vs cf vs the eight fold variants")
    print("=" * 104)
    print(f"  {'case':18s} | {'FLA':>7s} | {'cf':>7s} | " + " ".join(f"{t:>20s}" for t, _ in VARIANTS))
    print(f"  {'':18s} | {'dev':>7s} | {'dev':>7s} | "
          + " ".join(f"{'dev (base->vec)':>20s}" for _, _ in VARIANTS))
    for (name, B, T, H, seed, gs) in TCASES:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        qa = qn.clone().requires_grad_(True)
        ka = kn.clone().requires_grad_(True)
        va = inp["v"].float().clone().requires_grad_(True)
        ba = beta_act.clone().requires_grad_(True)
        ga = g_fla.clone().requires_grad_(True)
        ha = inp["h0"].clone().requires_grad_(True)
        o, s = chunk_kda(qa, ka, va, ga, ba, scale=scale,
                         initial_state=ha.float().transpose(-1, -2), output_final_state=True,
                         use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                         use_beta_sigmoid_in_kernel=False)
        loss = (o.float() * do).sum() + (s.float() * dfin).sum()
        holders = (qa, ka, va, ba, ga, ha)

        def fla_bwd():
            loss.backward(retain_graph=True)
            for t in holders:
                t.grad = None

        ch_cf = make_chain_cuda_fused(libs_b1, fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        chains = {tag: make_chain_prod(libs_b1, libs[tag], True, fi, inp, qn, kn, do, dfin, scale, 4)
                  for tag, _ in VARIANTS}
        ch_cf()
        for c in chains.values():
            c()
        fla_bwd()
        torch.cuda.synchronize()
        d_fl, _ = profile_dev(fla_bwd)
        d_cf, _ = profile_dev(ch_cf)
        d = {tag: profile_dev(chains[tag])[0] for tag, _ in VARIANTS}
        w = {tag: time_wall(chains[tag]) for tag, _ in VARIANTS}
        base_of = {"pci": None, "pci_vec": "pci", "pci_tma": None, "pci_tma_vec": "pci_tma",
                   "pci_mma": None, "pci_mma_vec": "pci_mma", "pci_mma_tma": None,
                   "pci_mma_tma_vec": "pci_mma_tma"}
        ent = []
        for t, _ in VARIANTS:
            b = base_of[t]
            if b is None:
                ent.append(f"{d[t]:8.1f}(-    )")
            else:
                ent.append(f"{d[t]:8.1f}({d[t]/d[b]:.3f})")
        print(f"  {name:18s} | {d_fl:7.1f} | {d_cf:7.1f} | " + " ".join(ent))
        print(f"      wall us -- cf {time_wall(ch_cf):7.1f} "
              + " ".join(f"{t} {w[t]:7.1f}" for t, _ in VARIANTS))
        del ch_cf, chains


libs_b1 = None


def main():
    global libs_b1
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    print("B1 flags:", " ".join(DEPLOY_FLAGS))
    scale = 1.0 / (D ** 0.5)
    lower_bound = -5.0
    print("building CUDA B1 ...")
    libs_b1 = load_so(build_b1())
    print("building eight fold kernels (VECST twin x4) ...")
    libs = build_all()
    mode = os.environ.get("B2VE_MODE", "")
    if mode in ("", "correctness"):
        part_a(libs, scale, lower_bound)
    if mode in ("", "timing"):
        part_b(libs, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
