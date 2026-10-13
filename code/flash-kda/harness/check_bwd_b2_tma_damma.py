"""P7-68: close the two boundaries recorded by P7-67 (docs/07 §70.6) as "待验证":

  (1) `-DB2M_TMA` combined with `-DB2M_DAMMA` **inside the production kernel**
      (`-DB2M_PROD -DB2M_HDW [-DB2M_INTRA]`).  P7-66 verified `da_mma_tma` only in *isolation*
      (`-DB2M_DA -DB2M_DAMMA -DB2M_TMA`, §69 Part A/B); the production regime (six segments in a
      single 1-block/SM launch) was never built with both switches at once.  Open question: does
      the DA segment's mma rewrite change the **sign** of the TMA penalty (§70: TMA is slower by
      8-12% in the production kernel, because per-stage `mbar_wait + __syncthreads` is a net loss
      once the kernel is no longer DRAM bound)?

  (2) the TMA production B2 landing in the P7-64 **end-to-end chain** (`pci_tma` variant).  P7-64
      chained only the non-TMA `prod_hdw[_intra]` (`pc`/`pci`).

Parts
-----
  A  correctness: build the four production variants `sc`, `sc_tma`, `mma`, `mma_tma`
     (`sc` = scalar DA, `mma` = `-DB2M_DAMMA`) and check (i) `*_tma` vs its non-TMA twin is
     **bit-identical on every output** (TMA only changes the stage layout), (ii) `mma*` vs `sc*`
     is bit-identical on the ten non-DA outputs, with only `dApre`/`dA` differing at the tf32
     floor.  6 cases incl. non-divisible T=250, single chunk T=64.
  B  timing: CUPTI `dev` scan of `sc` / `sc_tma` / `mma` / `mma_tma` in the production kernel.
  C  end-to-end chain (idle card): `pci_tma` (TMA fold) vs `pci` (non-TMA fold) vs `cf` --
     six gradients vs FLA autograd + chain dev/wall.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_tma_damma.py \
        > .../harness/check_bwd_b2_tma_damma.out.txt 2>&1
    B2TD_MODE=a | b | c
"""
import ctypes
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

from check_bwd_b2_intra import (                                     # noqa: E402
    BT, SCALE, CU, CASES, make, _core_dw, run_prod, run_prod_intra,
    build, load, ptxas_info, SO_CORE,
)
from check_bwd_ws_reuse import cdiv, stats                              # noqa: E402
from check_bwd_b1_bf16_e2e import profile_dev, time_wall, prep, fla_golden  # noqa: E402

# ---- production variants (all `-DB2M_PROD -DB2M_HDW`) --------------------------
SO_SC = "/tmp/b2_td_sc.so"
SO_SC_TMA = "/tmp/b2_td_sc_tma.so"
SO_MMA = "/tmp/b2_td_mma.so"
SO_MMA_TMA = "/tmp/b2_td_mma_tma.so"
SO_FOLD = "/tmp/b2_td_fold.so"
SO_FOLD_TMA = "/tmp/b2_td_fold_tma.so"

PROD = ["-DB2M_PROD=1", "-DB2M_HDW=1"]
VARS = [
    ("sc", PROD),
    ("sc_tma", PROD + ["-DB2M_TMA=1"]),
    ("mma", PROD + ["-DB2M_DAMMA=1"]),
    ("mma_tma", PROD + ["-DB2M_DAMMA=1", "-DB2M_TMA=1"]),
]

# ten outputs the DA rewrite must not touch + the two it is allowed to change.  NB: under
# `-DB2M_HDW` the production kernel does **not** write the `dw` core output (the `dw`
# round-trip is removed, §64) -- `dw` is a scratch pointer and must not be compared.
NON_DA = ("dv2", "dvb", "dkgb", "db", "dqg", "dkg", "dg", "dgk")
DA_OUT = ("dApre", "dA")
FIELDS = NON_DA + DA_OUT


def build_all():
    txt = {}
    for tag, flags in VARS:
        so = {"sc": SO_SC, "sc_tma": SO_SC_TMA, "mma": SO_MMA, "mma_tma": SO_MMA_TMA}[tag]
        txt[tag] = build(tag, CU, flags, so)
    # fold variants for the e2e chain
    build("fold", CU, PROD + ["-DB2M_INTRA=1"], SO_FOLD)
    build("fold_tma", CU, PROD + ["-DB2M_INTRA=1", "-DB2M_TMA=1"], SO_FOLD_TMA)
    return txt


def part_a(cases):
    print("=" * 92)
    print("Part A: production kernel -- TMA x DAMMA correctness across stage layouts")
    print("=" * 92)
    build_all()
    libs = {tag: load({"sc": SO_SC, "sc_tma": SO_SC_TMA,
                       "mma": SO_MMA, "mma_tma": SO_MMA_TMA}[tag]) for tag, _ in VARS}
    worst = {}
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        dw = _core_dw(inp)
        out = {tag: run_prod(libs[tag], inp, dw) for tag, _ in VARS}
        print(f"\n# {name}  NT={cdiv(T, BT)}")
        for a, b, label in (("sc_tma", "sc", "sc_tma vs sc"),
                            ("mma_tma", "mma", "mma_tma vs mma"),
                            ("mma", "sc", "mma vs sc"),
                            ("mma_tma", "sc_tma", "mma_tma vs sc_tma")):
            nz = []
            for fld in FIELDS:
                d = (out[a][fld].float() - out[b][fld].float()).abs().max().item()
                worst[f"{label}:{fld}"] = max(worst.get(f"{label}:{fld}", 0.0), d)
                if d != 0.0:
                    nz.append(f"{fld}={d:.2e}")
            # only the two DA-rewrite comparisons may differ, and only on dApre/dA.
            allow_da = ("mma" in label and "sc" in label)
            bad = [s for s in nz if not allow_da or s.split("=")[0] not in DA_OUT]
            print(f"  {label:18s} diff fields: {'NONE (bit-identical)' if not nz else ' '.join(nz)}")
            if bad:
                print(f"      !! unexpected non-zero diff: {' '.join(bad)}")

    def wmax(*keys):
        return max(worst.get(k, 0.0) for k in keys)

    print("\nworst-case max_abs over all shapes:")
    for label in ("sc_tma vs sc", "mma_tma vs mma", "mma vs sc", "mma_tma vs sc_tma"):
        nda = wmax(*[f"{label}:{f}" for f in NON_DA])
        da = wmax(*[f"{label}:{f}" for f in DA_OUT])
        print(f"  {label:18s} non-DA {nda:.3e}   dApre/dA {da:.3e}")
    print("\n  [expect] `*_tma vs *` non-DA AND dApre/dA == 0 (stage layout only)")
    print("  [expect] `mma* vs sc*` non-DA == 0, dApre/dA ~ tf32 floor (~3e-4)")


def part_b(cases):
    print("=" * 92)
    print("Part B: production-kernel dev -- sc / sc_tma / mma / mma_tma (does DAMMA flip the TMA sign?)")
    print("=" * 92)
    libs = {tag: load({"sc": SO_SC, "sc_tma": SO_SC_TMA,
                       "mma": SO_MMA, "mma_tma": SO_MMA_TMA}[tag]) for tag, _ in VARS}
    print(f"  {'case':22s} | {'sc':>8s} {'sc_tma':>8s} {'mma':>8s} {'mma_tma':>8s} | "
          f"{'tma/sc':>7s} {'tma_mma/mma':>11s} {'mma/sc':>7s}")
    for (name, B, T, H, seed) in cases:
        inp = make(B, T, H, seed)
        dw = _core_dw(inp)
        for tag, _ in VARS:
            run_prod(libs[tag], inp, dw)
        torch.cuda.synchronize()
        d = {}
        for tag, _ in VARS:
            d[tag], _ = profile_dev(lambda lib=libs[tag]: run_prod(lib, inp, dw), reps=30)
        print(f"  {name:22s} | {d['sc']:8.1f} {d['sc_tma']:8.1f} {d['mma']:8.1f} "
              f"{d['mma_tma']:8.1f} | {d['sc_tma']/d['sc']:7.3f} "
              f"{d['mma_tma']/d['mma']:11.3f} {d['mma']/d['sc']:7.3f}")


def part_c():
    print("=" * 92)
    print("Part C: end-to-end chain -- pci_tma (TMA fold) vs pci (non-TMA fold) vs cf (Triton fused)")
    print("=" * 92)
    from check_bwd_prod_e2e import make_chain_prod
    from check_bwd_cudab1_fused_e2e import make_chain_cuda_fused
    from check_bwd_b1_cuda_e2e import (build as build_b1, load as load_so,      # noqa: E402
                                       TCASES)
    from fla.ops.kda import chunk_kda

    scale = 1.0 / (128 ** 0.5)
    lower_bound = -5.0
    print("building CUDA B1 ...")
    lib_b1 = load_so(build_b1())
    lib_pci = ctypes.CDLL(SO_FOLD)
    lib_pci_tma = ctypes.CDLL(SO_FOLD_TMA)

    worst = {k: {} for k in ("pci", "pci_tma", "cf", "pci_vs_cf", "pci_tma_vs_cf")}
    for (name, B, T, H, seed, gs) in TCASES:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        ch_pci = make_chain_prod(lib_b1, lib_pci, True, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_tma = make_chain_prod(lib_b1, lib_pci_tma, True, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_cf = make_chain_cuda_fused(lib_b1, fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        rp, rt, rc = ch_pci(), ch_tma(), ch_cf()
        print(f"\n# {name}")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"),
                        ("dbeta", "beta"), ("dg", "g")]:
            _, mp = stats(f"  pci     {fld} vs FLA", rp[fld], gold[gk])
            _, mt = stats(f"  pci_tma {fld} vs FLA", rt[fld], gold[gk])
            _, mc = stats(f"  cf      {fld} vs FLA", rc[fld], gold[gk])
            _, dp = stats(f"  pci_tma {fld} vs pci", rt[fld], rp[fld])
            for k, v in (("pci", mp), ("pci_tma", mt), ("cf", mc),
                         ("pci_vs_cf", None), ("pci_tma_vs_cf", None)):
                if v is not None:
                    worst[k][fld] = max(worst[k].get(fld, 0.0), v)
            if fld == "dq":
                print(f"      (pci_tma vs pci {fld}: {'BIT-IDENTICAL' if dp == 0.0 else f'{dp:.2e}'})")
        _, mp = stats("  pci     dh0 vs FLA", rp["dh0"], gold["h0"].transpose(-1, -2))
        _, mt = stats("  pci_tma dh0 vs FLA", rt["dh0"], gold["h0"].transpose(-1, -2))
        _, mc = stats("  cf      dh0 vs FLA", rc["dh0"], gold["h0"].transpose(-1, -2))
        for k, v in (("pci", mp), ("pci_tma", mt), ("cf", mc)):
            worst[k]["dh0"] = max(worst[k].get("dh0", 0.0), v)
        del ch_pci, ch_tma, ch_cf, rp, rt, rc

    print("\nworst-case max_rel across cases:")
    print(f"  {'fld':6s} {'pci vs FLA':>11s} {'pci_tma vs FLA':>15s} {'cf vs FLA':>11s}")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        print(f"  {fld:6s} {worst['pci'][fld]:11.3e} {worst['pci_tma'][fld]:15.3e} "
              f"{worst['cf'][fld]:11.3e}")

    print("\ntiming (idle card), dev = CUPTI, wall = event-pair:")
    print(f"  {'case':18s} | {'cf dev':>7s} {'pci dev':>8s} {'tma dev':>8s} | "
          f"{'pci/FLA':>8s} {'tma/pci':>8s} | {'pci w':>7s} {'tma w':>7s}")
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

        ch_pci = make_chain_prod(lib_b1, lib_pci, True, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_tma = make_chain_prod(lib_b1, lib_pci_tma, True, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_cf = make_chain_cuda_fused(lib_b1, fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        ch_pci(); ch_tma(); ch_cf(); fla_bwd()
        torch.cuda.synchronize()
        d_fl, _ = profile_dev(fla_bwd)
        d_pci, _ = profile_dev(ch_pci)
        d_tma, _ = profile_dev(ch_tma)
        d_cf, _ = profile_dev(ch_cf)
        w_pci = time_wall(ch_pci)
        w_tma = time_wall(ch_tma)
        print(f"  {name:18s} | {d_cf:7.1f} {d_pci:8.1f} {d_tma:8.1f} | "
              f"{d_pci/d_fl:8.2f} {d_tma/d_pci:8.3f} | {w_pci:7.1f} {w_tma:7.1f}")
        del ch_pci, ch_tma, ch_cf


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    build("core", CU, [], SO_CORE)
    mode = os.environ.get("B2TD_MODE", "")
    if mode in ("", "a"):
        part_a(CASES)
    if mode in ("", "b"):
        part_b(CASES)
    if mode in ("", "c"):
        part_c()
    print("\nDONE")


if __name__ == "__main__":
    main()
