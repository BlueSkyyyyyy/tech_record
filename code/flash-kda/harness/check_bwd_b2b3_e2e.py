"""P7-23: the assembled backward chain's *new* bottleneck is B2/B3, not B1.

P7-22 (`docs/07` §25) landed the Triton software pipeline into B1 and pushed it to
1.0-2.2x FLA `dhu`, yet the chain `dev/FLA` only reached 0.48-0.54x -- because at the
deployed CHUNK=16 granularity the two *per-chunk* GEMMs dominate.  This script pins that
down and asks whether it is recoverable with cheap launch/tiling knobs (the class of lever
that already paid off for B1) or whether it needs a structural rewrite:

  Part A  per-kernel device-time split of the chain (dAv / B1 / B2 / B3 / rcumsum) over the
          4 textbook shapes -- how much of the chain is B2+B3.
  Part B  isolated controlled sweep of B2/B3 over num_warps x num_stages and over the block
          tiling (BK/BV for B2, BC for B3) at BT=16 -- is any cheap knob left on the table?
  Part C  land the only in-chain win found (B3 num_warps) into the assembled chain, verified
          against FLA autograd and the default chain, and time it chain-level.
  Part D  ncu bound analysis lives in `probe_b2b3_ncu.out.txt` (see docs/07 §26): B2 is
          DRAM-bound (h/dh re-read + 247 regs -> 12.5% occ), B3 is FMA/L1-bound with ~0
          tensor-core use.

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle card:
    CUDA_VISIBLE_DEVICES=1 python .../check_bwd_b2b3_e2e.py
    B2B3_MODE=timing | correctness | split | sweep
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import D, CHUNK, stats, cdiv  # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton  # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton  # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton  # noqa: E402
from check_bwd_b1_pipe_e2e import make_chain_fn_pipe  # noqa: E402
from check_bwd_b1_bf16_e2e import (prep, fla_golden, time_wall, profile_dev)  # noqa: E402
from fla.ops.utils import chunk_local_cumsum  # noqa: E402

CASES = [
    ("T=128 H=2", 1, 128, 2, 0, 1.0),
    ("T=256 H=4", 1, 256, 4, 1, 1.0),
    ("T=250 H=3 (non-div)", 1, 250, 3, 2, 1.0),
    ("T=512 H=4 g=2", 1, 512, 4, 3, 2.0),
    ("B2 T=513 H=2 (non-div)", 2, 513, 2, 4, 1.0),
    ("T=192 H=2 g=0", 1, 192, 2, 5, 0.0),
]
TCASES = [
    ("T=512 H=4", 1, 512, 4, 0, 1.0),
    ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
    ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
    ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0),
]
KEYS = ["_dav_kernel", "_dhu_kernel", "wy_dqkg", "intra", "chunk_local_cumsum"]


def split_profile(prof, reps=15):
    agg = {}
    for ev in prof.key_averages():
        for kk in KEYS:
            if kk in ev.key:
                agg[kk] = agg.get(kk, 0.0) + ev.self_device_time_total / reps
    return agg


# cache the pipelined-B1 kernel so we don't rebuild per case (Triton JIT is keyed by constexpr)
_PIPE_KERNEL = [None]


def _get_pipe_kernel():
    if _PIPE_KERNEL[0] is None:
        from check_bwd_b1_pipe_e2e import build_dhu_bf16_pipe
        _PIPE_KERNEL[0] = build_dhu_bf16_pipe()
    return _PIPE_KERNEL[0]


# --------------------------------------------------------------------------------- Parts
def part_split(cases, scale, lower_bound):
    print("=" * 78)
    print("Part A: per-kernel device-time split of the assembled chain (P7-22 pipe B1, NS=3)")
    print("=" * 78)
    print(f"  {'case':18s} | {'chain dev':>9s} | " + " ".join(f"{k.replace('_',''):>10s}" for k in KEYS))
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        ch = make_chain_fn_pipe(fi, inp, qn, kn, do, dfin, scale, NS=3)
        ch()
        torch.cuda.synchronize()
        d, prof = profile_dev(ch, reps=15)
        agg = split_profile(prof)
        b2b3 = agg.get("wy_dqkg", 0) + agg.get("intra", 0)
        line = f"  {name:18s} | {d:9.1f} | " + " ".join(f"{agg.get(k,0):10.1f}" for k in KEYS)
        line += f"   B2+B3={b2b3/d*100:4.1f}%"
        print(line)


def part_sweep(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part B: isolated B2/B3 sweep -- num_warps x num_stages and block tiling (BT=16)")
    print("=" * 78)
    dqkg_k = build_dqkg()
    intra_k = build_intra()
    dav_k = build_dav_kernel()
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        v_f = inp["v"].float()
        dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        dh = torch.randn(B, cdiv(T, CHUNK), H, D, D, device="cuda")
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi["v_new"], g=fi["g_cum"], beta=fi["beta"],
                    A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv)
        inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=fi["beta"], dAqk=dAqk,
                    dAkk=torch.randn_like(dAqk))
        print(f"\n# {name}")
        print("  B2 wy_dqkg  num_warps x num_stages:")
        for nw in (2, 4, 8):
            row = f"    nw={nw}: "
            for ns in (1, 2, 3):
                fn = lambda: wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK, num_warps=nw, num_stages=ns)
                fn(); torch.cuda.synchronize()
                d, _ = profile_dev(fn, reps=15)
                row += f"ns{ns}={d:6.1f}  "
            print(row)
        print("  B3 intra   num_warps x num_stages:")
        for nw in (2, 4, 8):
            row = f"    nw={nw}: "
            for ns in (1, 2, 3):
                fn = lambda: intra_triton(intra_k, inp3, CHUNK, CHUNK, num_warps=nw, num_stages=ns)
                fn(); torch.cuda.synchronize()
                d, _ = profile_dev(fn, reps=15)
                row += f"ns{ns}={d:6.1f}  "
            print(row)


def part_land(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part C: land B3 num_warps=2 into the chain -- correctness vs FLA and time")
    print("=" * 78)
    _get_pipe_kernel()
    NS = 3
    # correctness cases
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b = dict(fi)
        for kk in ("qg", "kg", "w"):
            fi_b[kk] = fi[kk].to(torch.bfloat16)
        do_b = do.to(torch.bfloat16)
        ch_def = make_chain_fn_pipe(fi_b, inp, qn, kn, do, dfin, scale, NS=NS, num_warps=4)
        ch_nw2 = _make_chain_nw(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, 2)
        r_def, r_nw2 = ch_def(), ch_nw2()
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        print(f"\n# {name}")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"), ("dg", "g")]:
            _, mr = stats(f"  nw2 {fld} vs FLA", r_nw2[fld], gold[gk])
            worst[fld] = max(worst.get(fld, 0.0), mr)
            _, mr1 = stats(f"  nw2 {fld} vs nw4", r_nw2[fld], r_def[fld])
        _, mr = stats("  nw2 dh0 vs FLA", r_nw2["dh0"], gold["h0"].transpose(-1, -2))
        worst["dh0"] = max(worst.get("dh0", 0.0), mr)
    print("\nworst-case max_rel (B3 nw=2 chain vs FLA):")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        print(f"  {fld:6s}  {worst[fld]:.3e}")
    return worst


def _make_chain_nw(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, nw_b2, nw_b3):
    """Chain identical to make_chain_fn_pipe but with B2/B3 num_warps exposed."""
    from check_bwd_b1_pipe_e2e import dhu_bf16_pipe_triton
    dhu_k = _get_pipe_kernel()
    dqkg_k = build_dqkg()
    intra_k = build_intra()
    dav_k = build_dav_kernel()
    v_f = inp["v"].float()

    def run():
        dAqk, dv = dav_triton(dav_k, fi_b["v_new"], fi_b["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, dv, scale, CHUNK,
                                            NS=NS, dht=dfin)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi_b["v_new"], g=fi_b["g_cum"],
                    beta=fi_b["beta"], A=fi_b["Akk"], h=fi_b["h"], do=do, dh=dh, dv=dv2)
        b2 = wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK, num_warps=nw_b2)
        inp3 = dict(q=qn, k=kn, g=fi_b["g_cum"], beta=fi_b["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])
        b3 = intra_triton(intra_k, inp3, CHUNK, CHUNK, num_warps=nw_b3)
        dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"],
                    dv=b2["dv"], dg=dg, dbeta=b2["db"] + b3["db"], dh0=dh0,
                    dh=dh, dv2=dv2)

    return run


def part_timing(cases, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 78)
    print("Part C2: chain timing -- B3 warps 4 (default) vs 2, and per-kernel dev")
    print("=" * 78)
    _get_pipe_kernel()
    NS = 3
    hdr = f"  {'case':18s} | {'FLA dev':>8s} | {'nw4 dev':>8s} {'nw2 dev':>8s} {'nw2/nw4':>8s} |"
    print(hdr)
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b = dict(fi)
        for kk in ("qg", "kg", "w"):
            fi_b[kk] = fi[kk].to(torch.bfloat16)
        do_b = do.to(torch.bfloat16)
        ch4 = make_chain_fn_pipe(fi_b, inp, qn, kn, do, dfin, scale, NS=NS, num_warps=4)
        ch2 = _make_chain_nw(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, 2)

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

        ch4(); ch2(); fla_bwd()
        torch.cuda.synchronize()
        d4, p4 = profile_dev(ch4)
        d2, p2 = profile_dev(ch2)
        d_fl, p_fl = profile_dev(fla_bwd)
        a4, a2 = split_profile(p4), split_profile(p2)
        print(f"  {name:18s} | {d_fl:8.1f} | {d4:8.1f} {d2:8.1f} {d2/d4:7.2f}x |"
              f"  B3 nw4={a4.get('intra',0):6.1f} nw2={a2.get('intra',0):6.1f}"
              f"  B2={a2.get('wy_dqkg',0):6.1f}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    mode = os.environ.get("B2B3_MODE", "")
    if mode in ("", "split"):
        part_split(TCASES, scale, lower_bound)
    if mode in ("", "sweep"):
        part_sweep(TCASES[:int(os.environ.get("B2B3_SCASES", 2))], scale, lower_bound)
    if mode in ("", "correctness"):
        part_land(CASES[:int(os.environ.get("B2B3_CASES", len(CASES)))], scale, lower_bound)
    if mode in ("", "timing"):
        part_timing(TCASES, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
