"""P7-75 (chain): land the **bf16-native CUDA B2 production launch** into the assembled reverse
chain.

`check_bwd_b2_bf16_prod.py` proved the bf16-native staging is a drop-in (bit-identical to the fp32
prod on bf16-valued operands).  This round chains it, i.e. the deployment dtype (bf16) is now
native end-to-end in the CUDA B2 -- no `.float()` round-trip of B1's bf16 `dh`/`dv2` nor of the
workspace `h`/`v_new`/`do` (the exact tax §67.4 kept as a boundary):

    dAv(Triton) -> CUDA-B1 -> CUDA-B2-prod[ -DB2M_BF16 ][ -> Triton B3 intra] -> reverse cumsum

only consuming FlashKDA's 16-chunk K1 workspace.  Variants:
  `pc`   : fp32 prod (`-DB2M_PROD -DB2M_HDW`) + standalone Triton B3 intra (P7-64 baseline)
  `pcb`  : bf16 prod (`-DB2M_PROD -DB2M_HDW -DB2M_BF16`) + standalone Triton B3 intra
  `pcib` : bf16 single-launch fold (`-DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_INTRA`)
  `cf`   : the §45 best chain (CUDA-B1 + fused Triton B2/B3)

Parts:
  A  correctness: `pcb` / `pcib` vs FLA autograd (six grads), and vs the fp32 `pc`.
  B  chain dev/wall (idle card): FLA vs cf vs pc vs pcb / pcib, per-kernel attribution.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_b2_bf16_e2e.py \
        > .../harness/check_bwd_b2_bf16_e2e.out.txt 2>&1
    BE_MODE=correctness | timing
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
from check_bwd_prod_e2e import make_chain_prod, build_b2, PROD_FLAGS, FOLD_FLAGS  # noqa: E402
from check_bwd_cudab1_fused_e2e import make_chain_cuda_fused                    # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                    # noqa: E402

BF16_KEYS = ["-DB2M_BF16=1"]
B_PROD = PROD_FLAGS + BF16_KEYS
B_FOLD = FOLD_FLAGS + BF16_KEYS


def make_chain_bf16(lib_b1, lib_prod, fold, fi, inp, qn, kn, do, dfin, scale, NSTAGE):
    """Like `make_chain_prod` but the CUDA B2 launch is `-DB2M_BF16` and its five streamed
    operands are fed as **bf16** (B1's native output dtype), skipping every `.float()`."""
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
    # fp32 operands (the six non-GEMM segments read these as fp32)
    qc = qn.contiguous().float()
    kc = kn.contiguous().float()
    g_f = fi["g_cum"].contiguous().float()
    beta_f = fi["beta"].contiguous().float()
    A_f = fi["Akk"].contiguous().float()
    # bf16 streamed operands (native)
    vn_b = fi["v_new"].contiguous().bfloat16()
    h_b = fi["h"].contiguous().bfloat16()
    dw_dummy = torch.zeros(B, T, H, K, device=dev, dtype=torch.float32)

    def run():
        dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = run_cuda_b1(lib_b1, fi_b, do_b, dv, scale, CHUNK, NSTAGE,
                                   dht=dht_c, out_f32=False)
        pinp = dict(q=qc, k=kc, v=v_f, v_new=vn_b, g=g_f, beta=beta_f, A=A_f, h=h_b,
                    do=do_b, dh=dh, dv=dv2)
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


def part_a(lib_b1, lib_prod, lib_fold, lib_bprod, lib_bfold, scale, lower_bound):
    print("=" * 100)
    print("Part A: bf16 prod chains vs FLA autograd, and vs the fp32 `pc` chain")
    print("  pcb  = CUDA-B1 + (prod_hdw_bf16 + Triton B3 intra);  pcib = prod_hdw_bf16_intra")
    print("=" * 100)
    worst = {}
    cases = CASES[:int(os.environ.get("BE_CASES", len(CASES)))]
    for (name, B, T, H, seed, gs) in cases:
        print(f"\n# {name}  B={B} T={T} H={H} seed={seed} g_scale={gs}")
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        ch_pc = make_chain_prod(lib_b1, lib_prod, False, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_pci = make_chain_prod(lib_b1, lib_fold, True, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_pcb = make_chain_bf16(lib_b1, lib_bprod, False, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_pcib = make_chain_bf16(lib_b1, lib_bfold, True, fi, inp, qn, kn, do, dfin, scale, 4)
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        rpc, rpci = ch_pc(), ch_pci()
        rpcb, rpcib = ch_pcb(), ch_pcib()
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"),
                        ("dbeta", "beta"), ("dg", "g")]:
            _, mpcb = stats(f"  pcb   {fld} vs FLA", rpcb[fld], gold[gk])
            _, mpcib = stats(f"  pcib  {fld} vs FLA", rpcib[fld], gold[gk])
            _, dpcb = stats(f"  pcb   {fld} vs pc", rpcb[fld], rpc[fld])
            _, dpci = stats(f"  pcb   {fld} vs pci", rpcb[fld], rpci[fld])
            for k, v in (("pcb", mpcb), ("pcib", mpcib), ("pcb_vs_pc", dpcb),
                         ("pcb_vs_pci", dpci)):
                worst[k] = max(worst.get(k, 0.0), v)
        _, mpcb = stats("  pcb   dh0 vs FLA", rpcb["dh0"], gold["h0"].transpose(-1, -2))
        _, mpcib = stats("  pcib  dh0 vs FLA", rpcib["dh0"], gold["h0"].transpose(-1, -2))
        worst["pcb"] = max(worst.get("pcb", 0.0), mpcb)
        worst["pcib"] = max(worst.get("pcib", 0.0), mpcib)
        del ch_pc, ch_pci, ch_pcb, ch_pcib, rpc, rpci, rpcb, rpcib

    print("\nworst-case max_rel across cases:")
    for k, v in worst.items():
        print(f"  {k:12s} {v:.3e}")


def _attrib(prof, sub):
    t, cnt = 0.0, 0
    for ev in prof.key_averages():
        if sub in ev.key:
            t += ev.self_device_time_total
            cnt += ev.count
    return t / max(cnt, 1), t, cnt


def part_b(lib_b1, lib_prod, lib_fold, lib_bprod, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 104)
    print("Part B: backward-only chain dev/wall (idle card): FLA vs cf vs pc vs pcb / pcib")
    print("  wall = event-pair / reps (primary); dev = CUPTI over-attributes tiny launches.")
    print("=" * 104)
    print(f"  {'case':18s} | {'FLA dev':>8s} | {'cf dev':>7s} {'pc dev':>7s} {'pcb dev':>8s}"
          f" {'pcib dev':>8s} | {'pc/FLA':>7s} {'pcb/FLA':>8s} {'pcb/pc':>7s}")
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
        ch_pcb = make_chain_bf16(lib_b1, lib_bprod, False, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_pcib = make_chain_bf16(lib_b1, lib_fold, True, fi, inp, qn, kn, do, dfin, scale, 4)
        ch_cf(); ch_pc(); ch_pcb(); ch_pcib(); fla_bwd()
        torch.cuda.synchronize()
        d_fl, p_fl = profile_dev(fla_bwd)
        d_cf, _ = profile_dev(ch_cf)
        d_pc, _ = profile_dev(ch_pc)
        d_pcb, p_pcb = profile_dev(ch_pcb)
        d_pcib, _ = profile_dev(ch_pcib)
        w_pc = time_wall(ch_pc); w_pcb = time_wall(ch_pcb); w_pcib = time_wall(ch_pcib)
        print(f"  {name:18s} | {d_fl:8.1f} | {d_cf:7.1f} {d_pc:7.1f} {d_pcb:8.1f}"
              f" {d_pcib:8.1f} | {d_pc/d_fl:7.2f} {d_pcb/d_fl:8.2f} {d_pcb/d_pc:7.2f}")
        print(f"      wall pc/pcb/pcib = {w_pc:.1f}/{w_pcb:.1f}/{w_pcib:.1f} us"
              f"   | pcb kernels B1 {_attrib(p_pcb, 'b1_tma_kernel')[0]:.1f}"
              f" prod {_attrib(p_pcb, 'b2mma_kernel')[0]:.1f}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    print("B1 flags:", " ".join(DEPLOY_FLAGS))
    scale = 1.0 / (D ** 0.5)
    lower_bound = -5.0
    print("building CUDA B1 ...")
    lib_b1 = load_so(build_b1())
    lib_prod = ctypes.CDLL(build_b2("hdw", PROD_FLAGS))
    lib_fold = ctypes.CDLL(build_b2("hdw_intra", FOLD_FLAGS))
    lib_bprod = ctypes.CDLL(build_b2("hdw_bf16", B_PROD))
    lib_bfold = ctypes.CDLL(build_b2("hdw_intra_bf16", B_FOLD))
    mode = os.environ.get("BE_MODE", "")
    if mode in ("", "correctness"):
        part_a(lib_b1, lib_prod, lib_fold, lib_bprod, lib_bfold, scale, lower_bound)
    if mode in ("", "timing"):
        part_b(lib_b1, lib_prod, lib_fold, lib_bprod, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
