"""P7-69: close the P7-68 boundary (docs/07 §71.6) -- **DAMMA into the end-to-end chain**.

P7-68 (§71) verified `-DB2M_TMA x -DB2M_DAMMA` only **inside the production kernel** (Part A/B on
synthetic operands) and landed only the **scalar-DA** fold (`pci_tma`) into the P7-64 chain.  It left
as an explicit boundary: "链级只测到 `pci_tma`（折叠版、标量 DA）；`pci_mma`/`pci_mma_tma`（DAMMA
进链）未做端到端".  This round closes it.

Four fold kernels (all `-DB2M_PROD -DB2M_HDW -DB2M_INTRA`, i.e. the single-launch B2 that also folds
the B3 diagonal, P7-63):

    tag            flags
    -----------    ------------------------------------------
    fold           (scalar DA)                                  <- = P7-64 `pci`
    fold_tma       + -DB2M_TMA
    fold_mma       + -DB2M_DAMMA
    fold_mma_tma   + -DB2M_DAMMA -DB2M_TMA

and chain each as `dAv(Triton) -> CUDA-B1 -> CUDA-B2-prod_intra -> reverse cumsum`, eating only
FlashKDA's 16-chunk K1 workspace.

Parts
-----
  A  correctness (6 TCASES incl. non-divisible T, g=0, B=2):
     (i)  `pci_mma_tma` vs `pci_mma` -- **all six gradients bit-identical** (TMA only changes the
          stage layout, §71.2);
     (ii) `pci_mma` vs `pci` -- `dv`/`dh0` bit-identical (both purely B1-determined); `dq/dk/dg/
          dbeta` differ only at the tf32 floor, because the DA rewrite feeds `dA` -> intra -> dq/dk/
          dg and -> db -> dbeta;
     (iii) each variant vs FLA autograd (worst-case max_rel).
  B  chain dev/wall (idle card): FLA reverse vs `cf` (Triton fused) vs the four fold variants, plus
     the `pci_mma_tma/pci_mma` and `pci_mma/pci` ratios.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_damma_e2e.py \
        > .../harness/check_bwd_b2_damma_e2e.out.txt 2>&1
    B2DM_MODE=correctness | timing
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
import check_bwd_prod_e2e as pe                                                  # noqa: E402
from check_bwd_prod_e2e import build_b2, make_chain_prod                    # noqa: E402
from check_bwd_cudab1_fused_e2e import make_chain_cuda_fused                # noqa: E402

# capture the raw B2 outputs of every prod-intra launch (to compare the kernel-level `dApre`/`dA`
# perturbation against how much of it survives into the chain gradients)
_CAP = {}


def _capture():
    orig = pe.run_prod_intra

    def cap_run(lib, inp, dw):
        out = orig(lib, inp, dw)
        _CAP[id(lib)] = out
        return out

    pe.run_prod_intra = cap_run

# four fold kernels (single-launch B2 with B3 diagonal folded in)
FOLD = ["-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_INTRA=1"]
VARIANTS = [
    ("pci",         FOLD),
    ("pci_tma",     FOLD + ["-DB2M_TMA=1"]),
    ("pci_mma",     FOLD + ["-DB2M_DAMMA=1"]),
    ("pci_mma_tma", FOLD + ["-DB2M_DAMMA=1", "-DB2M_TMA=1"]),
]
GRADS = ["dq", "dk", "dv", "dbeta", "dg", "dh0"]


def build_all():
    libs = {}
    for tag, flags in VARIANTS:
        so = build_b2(f"damma_{tag}", flags)
        libs[tag] = ctypes.CDLL(so)
    return libs


def _cmp(a, b, fld):
    return (a[fld].float() - b[fld].float()).abs().max().item()


def part_a(libs, scale, lower_bound):
    print("=" * 96)
    print("Part A: DAMMA / TMA folded kernels in the end-to-end chain vs FLA autograd")
    print("  pci_mma_tma vs pci_mma should be BIT-IDENTICAL; pci_mma vs pci changes only dq/dk/dg/dbeta")
    print("=" * 96)
    worst = {k: {} for k in [t for t, _ in VARIANTS] + ["pci_mma_tma_vs_pci_mma", "pci_mma_vs_pci"]}
    for (name, B, T, H, seed, gs) in TCASES:
        print(f"\n# {name}  B={B} T={T} H={H} seed={seed} g_scale={gs}")
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        res = {}
        chains = {}
        for tag, _ in VARIANTS:
            chains[tag] = make_chain_prod(libs_b1, libs[tag], True, fi, inp, qn, kn, do, dfin, scale, 4)
        for tag, _ in VARIANTS:
            res[tag] = chains[tag]()
        # B2-level (kernel output) perturbation vs chain-level perturbation
        bp = _CAP.get(id(libs["pci"]))
        bm = _CAP.get(id(libs["pci_mma"]))
        if bp is not None and bm is not None:
            for f in ("dApre", "dA", "dkg", "dqg"):
                d = _cmp(bm, bp, f)
                print(f"      B2 {f:6s} mma-sc max_abs={d:.3e}")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"), ("dg", "g")]:
            nz = []
            for tag, _ in VARIANTS:
                _, m = stats(f"  {tag:12s} {fld} vs FLA", res[tag][fld], gold[gk])
                worst[tag][fld] = max(worst[tag].get(fld, 0.0), m)
            d_mt = _cmp(res["pci_mma_tma"], res["pci_mma"], fld)
            d_ms = _cmp(res["pci_mma"], res["pci"], fld)
            worst["pci_mma_tma_vs_pci_mma"][fld] = max(
                worst["pci_mma_tma_vs_pci_mma"].get(fld, 0.0), d_mt)
            worst["pci_mma_vs_pci"][fld] = max(worst["pci_mma_vs_pci"].get(fld, 0.0), d_ms)
            nz.append(f"mma_tma-mma={'BIT-IDENT' if d_mt == 0 else f'{d_mt:.2e}'}")
            nz.append(f"mma-sc={d_ms:.2e}")
            print(f"      [{' '.join(nz)}]")
        for tag, _ in VARIANTS:
            _, m = stats(f"  {tag:12s} dh0 vs FLA",
                         res[tag]["dh0"], gold["h0"].transpose(-1, -2))
            worst[tag]["dh0"] = max(worst[tag].get("dh0", 0.0), m)
        d_mt = _cmp(res["pci_mma_tma"], res["pci_mma"], "dh0")
        d_ms = _cmp(res["pci_mma"], res["pci"], "dh0")
        worst["pci_mma_tma_vs_pci_mma"]["dh0"] = max(
            worst["pci_mma_tma_vs_pci_mma"].get("dh0", 0.0), d_mt)
        worst["pci_mma_vs_pci"]["dh0"] = max(worst["pci_mma_vs_pci"].get("dh0", 0.0), d_ms)
        print(f"      [mma_tma-mma={'BIT-IDENT' if d_mt == 0 else f'{d_mt:.2e}'} mma-sc={d_ms:.2e}]")
        del chains, res

    print("\nworst-case max_rel vs FLA autograd across cases:")
    hdr = "  " + f"{'fld':6s}" + "".join(f"{t:>14s}" for t, _ in VARIANTS)
    print(hdr)
    for fld in GRADS:
        print(f"  {fld:6s}" + "".join(f"{worst[t].get(fld, 0.0):14.3e}" for t, _ in VARIANTS))
    print("\nworst-case max_abs (chain vs chain):")
    for label in ("pci_mma_tma_vs_pci_mma", "pci_mma_vs_pci"):
        row = "  ".join(f"{fld}={worst[label].get(fld, 0.0):.2e}" for fld in GRADS)
        print(f"  {label:24s} {row}")
    print("\n  [expect] pci_mma_tma vs pci_mma: all six BIT-IDENTICAL (max_abs=0)")
    print("  [expect] pci_mma vs pci: dv/dh0 == 0 (B1-determined); dq/dk/dg/dbeta ~ tf32 floor")


def part_b(libs, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 104)
    print("Part B: backward-only chain dev/wall (idle card): FLA vs cf vs the four fold variants")
    print("=" * 104)
    print(f"  {'case':18s} | {'FLA':>7s} | {'cf':>7s} {'pci':>7s} {'p_tma':>7s} {'p_mma':>7s} "
          f"{'p_mtma':>7s} | {'p_mma/':>7s} {'p_mtma/':>8s} {'pci/FLA':>8s}")
    print(f"  {'':18s} | {'dev':>7s} | {'dev':>7s} {'dev':>7s} {'dev':>7s} {'dev':>7s} "
          f"{'dev':>7s} | {'pci':>7s} {'p_mma':>8s} {'':>8s}")
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
        w_cf = time_wall(ch_cf)
        print(f"  {name:18s} | {d_fl:7.1f} | {d_cf:7.1f} {d['pci']:7.1f} {d['pci_tma']:7.1f} "
              f"{d['pci_mma']:7.1f} {d['pci_mma_tma']:7.1f} | {d['pci_mma']/d['pci']:7.3f} "
              f"{d['pci_mma_tma']/d['pci_mma']:8.3f} {d['pci']/d_fl:8.2f}")
        print(f"      wall us -- cf {w_cf:7.1f} pci {w['pci']:7.1f} p_tma {w['pci_tma']:7.1f} "
              f"p_mma {w['pci_mma']:7.1f} p_mtma {w['pci_mma_tma']:7.1f}")
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
    print("building four fold kernels ...")
    libs = build_all()
    mode = os.environ.get("B2DM_MODE", "")
    if mode in ("", "correctness"):
        _capture()
        part_a(libs, scale, lower_bound)
    if mode in ("", "timing"):
        part_b(libs, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
