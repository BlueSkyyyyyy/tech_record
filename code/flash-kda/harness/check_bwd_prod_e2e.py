"""P7-64: land the **K2-shaped CUDA B2 production kernel** into the assembled reverse chain.

Context
-------
P7-48..P7-63 built, segment by segment, a full K2-shaped CUDA B2 (`b2_mma_probe.cu`): the four
V-reduction GEMMs (core), then the non-GEMM segments `dv2`(OUT) / `dkgb`(DKGB) / `dA`(DA+SAND) /
`db`(DB) / the gated finalize (GATE), merged into a **single production launch** (`-DB2M_PROD`,
P7-60), with the `dw` round-trip removed (`-DB2M_HDW`, P7-61) and the B3 `intra` diagonal folded in
(`-DB2M_INTRA`, P7-63).  But every one of those rounds was verified either **segment-vs-segment** or
against a synthetic operand set (`check_bwd_b2_prod.py` / `check_bwd_b2_intra.py` feed the kernel
`dqkg_make_inputs` data).  The production kernel has **never** been chained end-to-end: it always
sat behind a Triton B2/B3 (`check_bwd_cudab1_fused_e2e.py`, §45), or was measured in isolation.

This round closes that: assemble

    dAv(Triton) -> CUDA-B1 -> CUDA-B2-prod[ -> Triton B3 intra] -> reverse cumsum(B0)

eating only FlashKDA's 16-chunk K1 workspace, and check six gradients vs `torch.autograd`(FLA) and
vs the §45 best chain (`cf` = CUDA-B1 + fused Triton B2/B3).  Two CUDA-B2 variants:

  `pc`   : `-DB2M_PROD -DB2M_HDW` + the standalone Triton B3 `intra` (mirrors the "keep B2/B3 as
           two kernels" deployment conclusion of §66.6);
  `pci`  : `-DB2M_PROD -DB2M_HDW -DB2M_INTRA` -- a **single** CUDA B2 launch emitting all six
           gradients (the `dA`-not-spilled, one-launch form).

Parts
-----
  A  correctness: `pc` / `pci` vs FLA autograd (six gradients, worst-case max_rel), plus vs the
     §45 `cf` chain, 6 cases incl. non-divisible T, g=0, B=2.
  B  chain dev/wall (idle card): FLA reverse vs `cf` vs `pc` vs `pci`, + per-kernel attribution.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_prod_e2e.py \
        > .../harness/check_bwd_prod_e2e.out.txt 2>&1
    PE_MODE=correctness | timing
"""
import ctypes
import os
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

from check_bwd_ws_reuse import stats, CHUNK, D, cdiv                            # noqa: E402
from check_bwd_b1_bf16_e2e import prep, fla_golden, time_wall, profile_dev      # noqa: E402
from check_bwd_b1_cuda_e2e import (build as build_b1, load as load_so,          # noqa: E402
                                   run_cuda_b1, CASES, TCASES, DEPLOY_FLAGS)
from check_bwd_e2e import build_dav_kernel, dav_triton                          # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton    # noqa: E402
from check_bwd_b2_intra import run_prod, run_prod_intra                         # noqa: E402
from check_bwd_cudab1_fused_e2e import make_chain_cuda_fused                    # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                    # noqa: E402

CU = os.path.join(HARNESS, "b2_mma_probe.cu")
FOLD_FLAGS = ["-DB2M_PROD=1", "-DB2M_HDW=1", "-DB2M_INTRA=1"]
PROD_FLAGS = ["-DB2M_PROD=1", "-DB2M_HDW=1"]


def build_b2(tag, extra):
    so = f"/tmp/b2_prod_e2e_{tag}.so"
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared"] + extra + ["-lcuda", CU, "-o", so]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"nvcc failed for {tag}")
    return so


def make_chain_prod(lib_b1, lib_prod, fold, fi, inp, qn, kn, do, dfin, scale, NSTAGE):
    """dAv -> CUDA B1(dht) -> CUDA B2 prod[ -> Triton B3 intra] -> reverse cumsum.

    `fold=True` uses the single `-DB2M_INTRA` launch (B3 diagonal in-kernel); `fold=False` uses
    the `-DB2M_HDW` prod launch plus the standalone Triton B3 `intra`.  Only FlashKDA's K1
    workspace operands (`fi`) are consumed; no FLA forward intermediate.
    """
    dav_k = build_dav_kernel()
    intra_k = build_intra() if not fold else None
    v_f = inp["v"].float()
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    do_b = do.to(torch.bfloat16)
    dht_c = dfin.contiguous()
    B, T, H, K = qn.shape
    dev = qn.device
    # fp32 operands the CUDA prod kernel reads (it is a fp32/tf32 kernel, no bf16 variant)
    qc = qn.contiguous().float()
    kc = kn.contiguous().float()
    vn_f = fi["v_new"].contiguous().float()
    g_f = fi["g_cum"].contiguous().float()
    beta_f = fi["beta"].contiguous().float()
    A_f = fi["Akk"].contiguous().float()
    h_f = fi["h"].contiguous().float()
    dw_dummy = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)   # unused under HDW

    def run():
        dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = run_cuda_b1(lib_b1, fi_b, do_b, dv, scale, CHUNK, NSTAGE,
                                   dht=dht_c, out_f32=False)
        pinp = dict(q=qc, k=kc, v=v_f, v_new=vn_f, g=g_f, beta=beta_f, A=A_f, h=h_f,
                    do=do, dh=dh.float(), dv=dv2.float())
        if fold:
            pinp["dAqk"] = dAqk
            out = run_prod_intra(lib_prod, pinp, dw_dummy)
            dqf, dkf, dbf, dgf = out["dqg"], out["dkg"], out["db"], out["dg"]
        else:
            out = run_prod(lib_prod, pinp, dw_dummy)
            inp3 = dict(q=qc, k=kc, g=g_f, beta=beta_f, dAqk=dAqk, dAkk=out["dA"])
            b3 = intra_triton(intra_k, inp3, CHUNK, CHUNK)
            dqf = out["dqg"] + b3["dq"]
            dkf = out["dkg"] + b3["dk"]
            dbf = out["db"] + b3["db"]
            dgf = out["dg"] + b3["dg"]
        dg = chunk_local_cumsum(dgf, chunk_size=CHUNK, reverse=True)
        return dict(dq=dqf, dk=dkf, dv=out["dv2"], dg=dg, dbeta=dbf, dh0=dh0)

    return run


def part_a(lib_b1, lib_prod, lib_fold, scale, lower_bound):
    print("=" * 100)
    print("Part A: CUDA-B2-prod chains vs FLA autograd, and vs the §45 best chain `cf`")
    print("  pc  = CUDA-B1 + (prod_hdw + Triton B3 intra);  pci = CUDA-B1 + prod_hdw_intra (1 launch)")
    print("=" * 100)
    worst = {k: {} for k in ("pc", "pci", "cf", "pc_vs_cf", "pci_vs_cf")}
    cases = CASES[:int(os.environ.get("PE_CASES", len(CASES)))]
    for (name, B, T, H, seed, gs) in cases:
        print(f"\n# {name}  B={B} T={T} H={H} seed={seed} g_scale={gs}")
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        ch_pc = make_chain_prod(lib_b1, lib_prod, False, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_pci = make_chain_prod(lib_b1, lib_fold, True, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_cf = make_chain_cuda_fused(lib_b1, fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        rpc, rpci, rcf = ch_pc(), ch_pci(), ch_cf()
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"),
                        ("dbeta", "beta"), ("dg", "g")]:
            _, mpc = stats(f"  pc   {fld} vs FLA", rpc[fld], gold[gk])
            _, mpci = stats(f"  pci  {fld} vs FLA", rpci[fld], gold[gk])
            _, mcf = stats(f"  cf   {fld} vs FLA", rcf[fld], gold[gk])
            _, dpc = stats(f"  pc   {fld} vs cf", rpc[fld], rcf[fld])
            _, dpci = stats(f"  pci  {fld} vs cf", rpci[fld], rcf[fld])
            for k, v in (("pc", mpc), ("pci", mpci), ("cf", mcf),
                         ("pc_vs_cf", dpc), ("pci_vs_cf", dpci)):
                worst[k][fld] = max(worst[k].get(fld, 0.0), v)
        _, mpc = stats("  pc   dh0 vs FLA", rpc["dh0"], gold["h0"].transpose(-1, -2))
        _, mpci = stats("  pci  dh0 vs FLA", rpci["dh0"], gold["h0"].transpose(-1, -2))
        _, mcf = stats("  cf   dh0 vs FLA", rcf["dh0"], gold["h0"].transpose(-1, -2))
        for k, v in (("pc", mpc), ("pci", mpci), ("cf", mcf)):
            worst[k]["dh0"] = max(worst[k].get("dh0", 0.0), v)
        del ch_pc, ch_pci, ch_cf, rpc, rpci, rcf

    print("\nworst-case max_rel across cases:")
    print(f"  {'fld':6s} {'pc vs FLA':>11s} {'pci vs FLA':>11s} {'cf vs FLA':>11s}"
          f" {'pc vs cf':>11s} {'pci vs cf':>11s}")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        print(f"  {fld:6s} {worst['pc'][fld]:11.3e} {worst['pci'][fld]:11.3e} "
              f"{worst['cf'][fld]:11.3e} {worst['pc_vs_cf'].get(fld, 0.0):11.3e} "
              f"{worst['pci_vs_cf'].get(fld, 0.0):11.3e}")


def _attrib(prof, sub):
    t, cnt = 0.0, 0
    for ev in prof.key_averages():
        if sub in ev.key:
            t += ev.self_device_time_total
            cnt += ev.count
    return t / max(cnt, 1), t, cnt


def _topk(prof, k=5, reps=15):
    rows = sorted(prof.key_averages(),
                  key=lambda ev: ev.self_device_time_total, reverse=True)[:k]
    return "  ".join(f"{ev.key.split('(')[0][:22]}={ev.self_device_time_total/reps:.0f}" for ev in rows)


def part_b(lib_b1, lib_prod, lib_fold, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 108)
    print("Part B: backward-only chain dev/wall (idle card): FLA vs cf vs pc (prod+intra) vs pci (fold)")
    print("  wall = event-pair / reps (primary); dev = CUPTI over-attributes tiny launches.")
    print("=" * 108)
    hdr = (f"  {'case':18s} | {'FLA dev':>8s} | {'cf dev':>7s} {'pc dev':>7s} {'pci dev':>8s}"
           f" | {'cf w':>7s} {'pc w':>7s} {'pci w':>8s} | {'pc/FLA':>7s} {'pci/FLA':>7s} {'pci/cf':>7s}")
    print(hdr)
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

        ch_cf = make_chain_cuda_fused(lib_b1, fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        ch_pc = make_chain_prod(lib_b1, lib_prod, False, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_pci = make_chain_prod(lib_b1, lib_fold, True, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_cf(); ch_pc(); ch_pci(); fla_bwd()
        torch.cuda.synchronize()
        d_fl, p_fl = profile_dev(fla_bwd)
        d_cf, p_cf = profile_dev(ch_cf)
        d_pc, p_pc = profile_dev(ch_pc)
        d_pci, p_pci = profile_dev(ch_pci)
        w_cf = time_wall(ch_cf); w_pc = time_wall(ch_pc); w_pci = time_wall(ch_pci)
        print(f"  {name:18s} | {d_fl:8.1f} | {d_cf:7.1f} {d_pc:7.1f} {d_pci:8.1f}"
              f" | {w_cf:7.1f} {w_pc:7.1f} {w_pci:8.1f} | {d_pc/d_fl:7.2f} {d_pci/d_fl:7.2f}"
              f" {d_pci/d_cf:7.2f}")
        b1c = _attrib(p_pci, "b1_tma_kernel")[0]
        b2c = _attrib(p_pci, "b2mma_kernel")[0]
        b2p = _attrib(p_pc, "b2mma_kernel")[0]
        print(f"      kernels dev/call -- pci: B1 {b1c:6.1f} prod {b2c:7.1f} | "
              f"pc: prod {b2p:7.1f} | cf:" + _topk(p_cf, 5))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    print("B1 flags:", " ".join(DEPLOY_FLAGS))
    scale = 1.0 / (D ** 0.5)
    lower_bound = -5.0
    print("building CUDA B1 ...")
    lib_b1 = load_so(build_b1())
    so_prod = build_b2("hdw", PROD_FLAGS)
    so_fold = build_b2("hdw_intra", FOLD_FLAGS)
    lib_prod = ctypes.CDLL(so_prod)
    lib_fold = ctypes.CDLL(so_fold)
    mode = os.environ.get("PE_MODE", "")
    if mode in ("", "correctness"):
        part_a(lib_b1, lib_prod, lib_fold, scale, lower_bound)
    if mode in ("", "timing"):
        part_b(lib_b1, lib_prod, lib_fold, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
