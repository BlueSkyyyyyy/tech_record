"""P7-42: combine the two converged optimizations of the reverse chain, which so far
have each been landed *separately*:

  (1) P7-15..P7-41  the K2-shaped hand-written CUDA B1 (`b1_tma_probe.cu`, deploy config
      `-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV -DB1T_EPROXY -DB1T_SPROXY`, `NSTAGE>=4`).  It
      was plugged into the assembled chain only in P7-39 (`check_bwd_b1_cuda_e2e.py`),
      and that chain still used the **unfused** B2 (`wy_dqkg`) + B3 (`intra`).
  (2) P7-30  the **fused** B2+B3 (`check_bwd_b2b3_fused.py`: B3's diagonal folded into
      B2, `dq/dk/dg` accumulators kept in registers, `dA` never spills to DRAM).  Its
      chain (`make_chain_fused`) still used the **Triton** pipe B1 (`dhu_bf16_pipe`).

So neither chain contains both wins.  This script assembles the *fully converged* reverse
chain -- `dAv -> CUDA B1 -> fused B2+B3 -> reverse cumsum(B0)`, eating only FlashKDA's
16-chunk K1 workspace -- and checks it against FLA autograd and against the two
half-converged chains.

Parts
-----
  A  correctness: the combined chain vs FLA autograd (six gradients) and vs each
     half-chain (own delta), 6 cases incl. non-divisible T, g=0, B=2.
  B  chain dev/wall (idle card): FLA reverse vs (CUDA-B1 + unfused) vs (Triton-B1 + fused)
     vs (CUDA-B1 + fused), + per-kernel B1 device attribution.

Run inside `kernel_lab` (idle card), absolute paths:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_cudab1_fused_e2e.py \
        > .../harness/check_bwd_cudab1_fused_e2e.out.txt 2>&1
    CF_MODE=correctness | timing
"""
import ctypes
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import stats, CHUNK, D                                # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                        # noqa: E402
from check_bwd_b1_bf16_e2e import prep, fla_golden, time_wall, profile_dev    # noqa: E402
from check_bwd_b1_cuda_e2e import (build, load, run_cuda_b1, make_chain_cuda,  # noqa: E402
                                   CASES, TCASES, DEPLOY_FLAGS)               # noqa: E402
from check_bwd_b2b3_fused import (build_fused_b2b3, fused_b2b3_triton,         # noqa: E402
                                  make_chain_fused)                           # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                  # noqa: E402


def make_chain_cuda_fused(lib, fi, inp, qn, kn, do, dfin, scale, NSTAGE, nw_f=8):
    """dAv -> CUDA B1 (dht) -> fused B2+B3 -> reverse cumsum.  Only FlashKDA's K1
    workspace operands (`fi`) are consumed; no FLA forward intermediate."""
    dav_k = build_dav_kernel()
    fused_k = build_fused_b2b3()
    v_f = inp["v"].float()
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    do_b = do.to(torch.bfloat16)
    dht_c = dfin.contiguous()

    def run():
        dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        # native bf16 `dh`/`dv2` out of CUDA B1 (deployment dtype); fused B2 loads bf16
        # `dh` directly and needs fp32 `dv` (it feeds a `tl.dot`), so cast only `dv2`.
        dh, dh0, dv2 = run_cuda_b1(lib, fi_b, do_b, dv, scale, CHUNK, NSTAGE,
                                   dht=dht_c, out_f32=False)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi["v_new"], g=fi["g_cum"], beta=fi["beta"],
                    A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2.float())
        b = fused_b2b3_triton(fused_k, inp2, dAqk, scale, CHUNK, num_warps=nw_f)
        dg = chunk_local_cumsum(b["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b["dq"], dk=b["dk"], dv=b["dv2"], dg=dg, dbeta=b["db"],
                    dh0=dh0, dh=dh, dv2=dv2)

    return run


def part_a(lib, scale, lower_bound):
    print("=" * 96)
    print("Part A: combined chain (CUDA-B1 + fused B2/B3) vs FLA, and vs each half-chain")
    print("  half-chains: cu = CUDA-B1 + unfused B2/B3 (P7-39); fu = Triton-B1 + fused (P7-30)")
    print("=" * 96)
    wc, wcu, wfu, wdcu, wdfu = {}, {}, {}, {}, {}
    for (name, B, T, H, seed, gs) in CASES:
        print(f"\n# {name}  B={B} T={T} H={H} seed={seed} g_scale={gs}")
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        ch_cf = make_chain_cuda_fused(lib, fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        ch_cu = make_chain_cuda(lib, fi, inp, qn, kn, do, dfin, scale, NSTAGE=4)
        ch_fu = make_chain_fused(fi, inp, qn, kn, do, dfin, scale, NS=3)
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        rcf, rcu, rfu = ch_cf(), ch_cu(), ch_fu()
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"),
                        ("dbeta", "beta"), ("dg", "g")]:
            _, mc = stats(f"  cf   {fld} vs FLA", rcf[fld], gold[gk])
            _, mcu = stats(f"  cu   {fld} vs FLA", rcu[fld], gold[gk])
            _, mfu = stats(f"  fu   {fld} vs FLA", rfu[fld], gold[gk])
            _, dcu = stats(f"  cf   {fld} vs cu", rcf[fld], rcu[fld])
            _, dfu = stats(f"  cf   {fld} vs fu", rcf[fld], rfu[fld])
            wc[fld] = max(wc.get(fld, 0.0), mc)
            wcu[fld] = max(wcu.get(fld, 0.0), mcu)
            wfu[fld] = max(wfu.get(fld, 0.0), mfu)
            wdcu[fld] = max(wdcu.get(fld, 0.0), dcu)
            wdfu[fld] = max(wdfu.get(fld, 0.0), dfu)
        _, mc = stats("  cf   dh0 vs FLA", rcf["dh0"], gold["h0"].transpose(-1, -2))
        _, mcu = stats("  cu   dh0 vs FLA", rcu["dh0"], gold["h0"].transpose(-1, -2))
        _, mfu = stats("  fu   dh0 vs FLA", rfu["dh0"], gold["h0"].transpose(-1, -2))
        wc["dh0"] = max(wc.get("dh0", 0.0), mc)
        wcu["dh0"] = max(wcu.get("dh0", 0.0), mcu)
        wfu["dh0"] = max(wfu.get("dh0", 0.0), mfu)
        del ch_cf, ch_cu, ch_fu, rcf, rcu, rfu

    print("\nworst-case max_rel across cases:")
    print(f"  {'fld':6s} {'cf vs FLA':>12s} {'cu vs FLA':>12s} {'fu vs FLA':>12s}"
          f" {'cf vs cu':>12s} {'cf vs fu':>12s}")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        print(f"  {fld:6s} {wc[fld]:12.3e} {wcu[fld]:12.3e} {wfu[fld]:12.3e}"
              f" {wdcu.get(fld, 0.0):12.3e} {wdfu.get(fld, 0.0):12.3e}")


def _b1_attrib(prof, sub):
    t, cnt = 0.0, 0
    for ev in prof.key_averages():
        if sub in ev.key:
            t += ev.self_device_time_total
            cnt += ev.count
    return t / max(cnt, 1), t, cnt


def part_b(lib, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    CUDA_NS = [int(x) for x in os.environ.get("CF_NS", "4").split(",")]
    print("\n" + "=" * 104)
    print("Part B: backward-only chain dev/wall (idle card): FLA vs cu vs fu vs combined (cf)")
    print("  wall = event-pair / reps (primary); dev = CUPTI (over-attributes tiny launches).")
    print("=" * 104)
    hdr = f"  {'case':18s} | {'FLA dev':>8s} | {'cu dev':>7s} {'fu dev':>7s}"
    for ns in CUDA_NS:
        hdr += f" | {'cf' + str(ns) + ' dev':>9s}"
    hdr += f" | {'cu w':>7s} {'fu w':>7s}"
    for ns in CUDA_NS:
        hdr += f" {'cf' + str(ns) + ' w':>8s}"
    hdr += " |  cf/cu  cf/fu"
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

        ch_cu = make_chain_cuda(lib, fi, inp, qn, kn, do, dfin, scale, NSTAGE=max(CUDA_NS))
        ch_fu = make_chain_fused(fi, inp, qn, kn, do, dfin, scale, NS=3)
        chs = {ns: make_chain_cuda_fused(lib, fi, inp, qn, kn, do, dfin, scale, NSTAGE=ns)
               for ns in CUDA_NS}
        ch_cu(); ch_fu()
        for c in chs.values():
            c()
        fla_bwd()
        torch.cuda.synchronize()
        d_fl, _ = profile_dev(fla_bwd)
        d_cu, p_cu = profile_dev(ch_cu)
        d_fu, p_fu = profile_dev(ch_fu)
        w_cu = time_wall(ch_cu); w_fu = time_wall(ch_fu)
        line = f"  {name:18s} | {d_fl:8.1f} | {d_cu:7.1f} {d_fu:7.1f}"
        cf_dev, cf_wall, b1c = {}, {}, {}
        for ns in CUDA_NS:
            d_cf, p_cf = profile_dev(chs[ns])
            w_cf = time_wall(chs[ns])
            cf_dev[ns] = d_cf
            cf_wall[ns] = w_cf
            b1c[ns] = _b1_attrib(p_cf, "b1_tma_kernel")[0]
            line += f" | {d_cf:9.1f}"
        line += f" | {w_cu:7.1f} {w_fu:7.1f}"
        for ns in CUDA_NS:
            line += f" {cf_wall[ns]:8.1f}"
        b1t = _b1_attrib(p_fu, "_dhu_kernel")[0]
        line += f" | {min(cf_dev.values()) / d_cu:6.2f}x {min(cf_dev.values()) / d_fu:6.2f}x"
        print(line)
        print(f"      B1 kernel dev: Triton(fu) {b1t:6.1f}us | " +
              " | ".join(f"CUDA(cf{ns}) {b1c[ns]:6.1f}us" for ns in CUDA_NS))


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    print("deploy flags:", " ".join(DEPLOY_FLAGS))
    scale = 1.0 / (D ** 0.5)
    lower_bound = -5.0
    so = build()
    lib = load(so)
    mode = os.environ.get("CF_MODE", "")
    if mode in ("", "correctness"):
        part_a(lib, scale, lower_bound)
    if mode in ("", "timing"):
        part_b(lib, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
