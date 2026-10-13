"""P7-26: B2 `wy_dqkg` -- the `h`/`dh` bf16 lever (§26.6 item (1)) and the `i_k` re-read
(§26.6 item (2)), both landed / attributed in the *real assembled chain*.

Background.  P7-23 (`docs/07` §26) pinned the chain-level bottleneck to B2, `ncu`
showed it DRAM-bound (59% SOL) with `h`/`dh` the dominant traffic, and §26.6 listed two
"structural" levers: (1) store `h`/`dh` as bf16, (2) drop the `i_k`-loop re-read of the
V-side operands.  P7-24/P7-25 then took B3 (MMA) and B2 (register spill) off the table.
This script closes items (1) and (2) with a *positive-or-negative* decision each.

The deployed dtype mix -- read straight off the chain -- is what matters and what the
earlier B2 profiles never used (they fed all-bf16):
    q, k, v, do, dv, h, dh : fp32      v_new, g, A, beta : bf16
so `h`+`dh` are **72.5% of B2's read bytes** (2*128*128*4 B per chunk).

   Part A  isolated kernel, deployed mix: bf16 vs fp32 `h`/`dh` -- correctness (bit-close)
           + CUPTI/do_bench timing.  Expect ~1.1x (the §26.5 number, now on the deployed mix).
   Part B  occupancy sweep (BK/BV/num_warps): is there a <=170-reg config that pays?  (No.)
   Part C  land bf16 `h`/`dh` into the real chain: the harness must `.to(bf16)` the operands
           each call, which is a *separate* large elementwise kernel -- net-negative.
   Part D  per-kernel attribution in the chain: the B2 kernel itself does speed up (~1.10x
           on the large shape), so the lever is real but only with *native* bf16 operands.

Run inside `kernel_lab` (idle card):
    CUDA_VISIBLE_DEVICES=1 python .../check_bwd_b2_hdh.py
    B2H_MODE=correctness | timing | sweep | chain
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import D, CHUNK, stats, cdiv                                   # noqa: E402
from check_bwd_b2_struct import build_dqkg_late, wy_dqkg_late_triton                   # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra                          # noqa: E402
from check_bwd_b3_mma import build_intra_mma, intra_triton_mma                          # noqa: E402
from check_bwd_b1_pipe_e2e import build_dhu_bf16_pipe, dhu_bf16_pipe_triton            # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                                  # noqa: E402
from check_bwd_b1_bf16_e2e import (prep, fla_golden, time_wall, profile_dev)           # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                            # noqa: E402


def _prep_b(fi, do):
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    return fi_b, do.to(torch.bfloat16)


def _run_b2(kernel, inp, scale, BT, BK=64, BV=64, num_warps=4, num_stages=1, ret_handle=False):
    """Local runner exposing BK/BV (and the compiled handle) -- the P7-25 runner hardcodes them."""
    B, T, H, K = inp["q"].shape
    V = inp["v"].shape[-1]
    NT = cdiv(T, BT)
    dev = inp["q"].device
    dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    dv2 = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    db = torch.empty(B, T, H, device=dev, dtype=torch.float32)
    dA = torch.empty(B, T, H, BT, device=dev, dtype=torch.float32)
    h = kernel[(NT, B * H)](
        inp["q"], inp["k"], inp["v"], inp["v_new"], inp["g"], inp["beta"], inp["A"], inp["h"],
        inp["do"], inp["dh"], inp["dv"], dq, dk, dv2, db, dg, dA,
        T, NT, scale, H=H, K=K, V=V, BT=BT, BK=min(BK, K), BV=min(BV, V),
        num_warps=num_warps, num_stages=num_stages)
    if ret_handle:
        return h
    return dict(dq=dq, dk=dk, dv=dv2, db=db, dg=dg, dA=dA)


def _b2_ops(inp, fi_b, do, dfin, scale):
    """Deployed-mix operands for B2 (mirrors `check_bwd_b2_struct._make_chain`)."""
    dav_k = build_dav_kernel()
    dhu_k = build_dhu_bf16_pipe()
    _, dv = dav_triton(dav_k, fi_b["v_new"], fi_b["Aqk"], do, scale, CHUNK)
    dh, dh0, dv2 = dhu_bf16_pipe_triton(dhu_k, fi_b, do.to(torch.bfloat16), dv, scale, CHUNK,
                                        NS=3, dht=dfin)
    return dict(v=inp["v"].float(), v_new=fi_b["v_new"], g=fi_b["g_cum"],
                beta=fi_b["beta"], A=fi_b["Akk"], h=fi_b["h"],
                do=do, dh=dh, dv=dv), dv, dh, dv2


# ------------------------------------------------------------------------- Part A
def part_correctness():
    print("=" * 78)
    print("Part A: isolated B2, deployed dtype mix -- bf16 vs fp32 h/dh")
    print("=" * 78)
    scale = 1.0 / math.sqrt(D)
    k = build_dqkg_late()
    worst = {}
    for (name, B, T, H, seed, gs) in [
        ("T=512 H=4", 1, 512, 4, 0, 1.0),
        ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
        ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
        ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0),
    ]:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, -5.0, seed, gs)
        fi_b, _ = _prep_b(fi, do)
        ops, dv, dh, dv2 = _b2_ops(inp, fi_b, do, dfin, scale)
        base = dict(ops, q=qn, k=kn)
        bf = dict(base, h=base["h"].to(torch.bfloat16), dh=base["dh"].to(torch.bfloat16))
        r0 = _run_b2(k, base, scale, CHUNK)
        r1 = _run_b2(k, bf, scale, CHUNK)
        print(f"\n# {name}  (h/dh fp32 -> bf16)")
        for fld in ["dq", "dk", "dv", "db", "dg", "dA"]:
            _, mr = stats(f"  delta {fld}", r1[fld], r0[fld])
            worst[fld] = max(worst.get(fld, 0.0), mr)
    print("\nworst-case bf16-vs-fp32 h/dh delta (max_rel):",
          {kk: f"{v:.1e}" for kk, v in worst.items()})


# ------------------------------------------------------------------------- Part A2
def part_timing():
    print("=" * 78)
    print("Part A2: isolated B2 timing, deployed mix -- fp32 vs bf16 h/dh")
    print("=" * 78)
    scale = 1.0 / math.sqrt(D)
    k = build_dqkg_late()
    print(f"  {'case':18s} | {'fp32 dev':>9s} {'bf16 dev':>9s} {'dev x':>6s} "
          f"| {'fp32 bench':>10s} {'bf16 bench':>10s} {'bench x':>7s}")
    for (name, B, T, H, seed) in [("T=512 H=4", 1, 512, 4, 0),
                                  ("B2 T=1024 H=8", 2, 1024, 8, 1),
                                  ("B8 T=2048 H=8", 8, 2048, 8, 2),
                                  ("T=250 H=3 nondiv", 1, 250, 3, 3)]:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, -5.0, seed, 1.0)
        fi_b, _ = _prep_b(fi, do)
        ops, dv, dh, dv2 = _b2_ops(inp, fi_b, do, dfin, scale)
        base = dict(ops, q=qn, k=kn)
        bf = dict(base, h=base["h"].to(torch.bfloat16), dh=base["dh"].to(torch.bfloat16))
        f0 = lambda: _run_b2(k, base, scale, CHUNK)
        f1 = lambda: _run_b2(k, bf, scale, CHUNK)
        f0(); f1(); torch.cuda.synchronize()
        d0, _ = profile_dev(f0, reps=20)
        d1, _ = profile_dev(f1, reps=20)
        b0 = triton.testing.do_bench(f0, warmup=25, rep=200) * 1e3
        b1 = triton.testing.do_bench(f1, warmup=25, rep=200) * 1e3
        print(f"  {name:18s} | {d0:9.1f} {d1:9.1f} {d1/d0:5.2f}x "
              f"| {b0:10.1f} {b1:10.1f} {b1/b0:6.2f}x")


# ------------------------------------------------------------------------- Part B
def part_sweep():
    print("=" * 78)
    print("Part B: occupancy sweep -- BK/BV/num_warps (regs, spills, dev)")
    print("=" * 78)
    scale = 1.0 / math.sqrt(D)
    k = build_dqkg_late()
    for (name, B, T, H) in [("T=512 H=4", 1, 512, 4), ("B8 T=2048 H=8", 8, 2048, 8)]:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, -5.0, 0, 1.0)
        fi_b, _ = _prep_b(fi, do)
        ops, dv, dh, dv2 = _b2_ops(inp, fi_b, do, dfin, scale)
        base = dict(ops, q=qn, k=kn)
        print(f"\n# {name}")
        for BK in (32, 64, 128):
            for BV in (32, 64):
                for nw in (4, 8):
                    try:
                        h = _run_b2(k, base, scale, CHUNK, BK=BK, BV=BV, num_warps=nw,
                                    ret_handle=True)
                    except Exception as e:
                        print(f"  BK={BK:3d} BV={BV:3d} nw={nw}: FAIL {str(e)[:40]}")
                        continue
                    fn = lambda: _run_b2(k, base, scale, CHUNK, BK=BK, BV=BV, num_warps=nw)
                    fn(); torch.cuda.synchronize()
                    dev, _ = profile_dev(fn, reps=10)
                    print(f"  BK={BK:3d} BV={BV:3d} nw={nw}: regs={h.n_regs:3d} "
                          f"spill={h.n_spills:3d} dev={dev:8.1f}us")


# --------------------------------------------------------------------------- chain
def _make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, nw_b3, hdh_bf16):
    dhu_k = build_dhu_bf16_pipe()
    dqkg_k = build_dqkg_late()
    dav_k = build_dav_kernel()
    intra_k = build_intra_mma()
    v_f = inp["v"].float()

    def run():
        dAqk, dv = dav_triton(dav_k, fi_b["v_new"], fi_b["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, dv, scale, CHUNK, NS=NS, dht=dfin)
        h = fi_b["h"]
        dh2 = dh
        if hdh_bf16:
            h = h.to(torch.bfloat16)
            dh2 = dh.to(torch.bfloat16)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi_b["v_new"], g=fi_b["g_cum"],
                    beta=fi_b["beta"], A=fi_b["Akk"], h=h, do=do, dh=dh2, dv=dv2)
        b2 = wy_dqkg_late_triton(dqkg_k, inp2, scale, CHUNK)
        inp3 = dict(q=qn, k=kn, g=fi_b["g_cum"], beta=fi_b["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])
        b3 = intra_triton_mma(intra_k, inp3, CHUNK, CHUNK, num_warps=nw_b3)
        dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"], dv=b2["dv"],
                    dg=dg, dbeta=b2["db"] + b3["db"], dh0=dh0)

    return run


CASES = [("T=128 H=2", 1, 128, 2, 0, 1.0),
         ("T=256 H=4", 1, 256, 4, 1, 1.0),
         ("T=250 H=3 (non-div)", 1, 250, 3, 2, 1.0),
         ("T=512 H=4 g=2", 1, 512, 4, 3, 2.0),
         ("B2 T=513 H=2 (non-div)", 2, 513, 2, 4, 1.0),
         ("T=192 H=2 g=0", 1, 192, 2, 5, 0.0)]
TCASES = [("T=512 H=4", 1, 512, 4, 0, 1.0),
          ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
          ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
          ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0)]


def part_chain(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part C: land bf16 h/dh into the assembled chain -- correctness vs FLA + baseline")
    print("=" * 78)
    NS = 3
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b, do_b = _prep_b(fi, do)
        ch_fp = _make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, hdh_bf16=False)
        ch_bf = _make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, hdh_bf16=True)
        r_fp, r_bf = ch_fp(), ch_bf()
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        gk = {"dq": "q", "dk": "k", "dv": "v", "dbeta": "beta", "dg": "g"}
        print(f"\n# {name}")
        for fld in gk:
            _, mr = stats(f"  bf16 {fld} vs FLA", r_bf[fld], gold[gk[fld]])
            worst[fld] = max(worst.get(fld, 0.0), mr)
        _, mr = stats("  bf16 dh0 vs FLA", r_bf["dh0"], gold["h0"].transpose(-1, -2))
        worst["dh0"] = max(worst.get("dh0", 0.0), mr)
        for fld in gk:
            _, mr = stats(f"  delta {fld} (bf16-fp32)", r_bf[fld], r_fp[fld])
    print("\nworst-case max_rel (bf16-h/dh chain vs FLA):",
          {kk: f"{v:.1e}" for kk, v in worst.items()})


def part_chain_timing(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part D: chain timing -- fp32 vs bf16 h/dh (naive cast) + B2-kernel attribution")
    print("=" * 78)
    NS = 3
    from torch.profiler import ProfilerActivity, profile

    def b2_attr(fn, reps=15):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
        tot, cnt = {}, {}
        for ev in p.key_averages():
            tot[ev.key] = ev.self_device_time_total
            cnt[ev.key] = ev.count
        per = {k2: (tot[k2] / max(cnt[k2], 1)) for k2 in tot}
        b2k = [k2 for k2 in per if "wy_dqkg" in k2]
        return (per[b2k[0]] if b2k else float("nan"))

    print(f"  {'case':18s} | {'fp32 dev':>8s} {'bf16 dev':>8s} {'dev x':>6s} "
          f"| {'fp32 wall':>9s} {'bf16 wall':>9s} | {'B2k fp':>8s} {'B2k bf':>8s} {'B2 x':>6s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b, do_b = _prep_b(fi, do)
        ch_fp = _make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, hdh_bf16=False)
        ch_bf = _make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, 4, hdh_bf16=True)
        ch_fp(); ch_bf(); torch.cuda.synchronize()
        d0, _ = profile_dev(ch_fp, reps=15)
        d1, _ = profile_dev(ch_bf, reps=15)
        w0 = time_wall(ch_fp, reps=50)
        w1 = time_wall(ch_bf, reps=50)
        b0 = b2_attr(ch_fp, 15)
        b1 = b2_attr(ch_bf, 15)
        print(f"  {name:18s} | {d0:8.1f} {d1:8.1f} {d1/d0:5.2f}x "
              f"| {w0:9.1f} {w1:9.1f} | {b0:8.1f} {b1:8.1f} {b1/b0:5.2f}x")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    mode = os.environ.get("B2H_MODE", "")
    if mode in ("", "correctness"):
        part_correctness()
    if mode in ("", "timing"):
        part_timing()
    if mode in ("", "sweep"):
        part_sweep()
    if mode in ("", "chain"):
        part_chain(CASES[:int(os.environ.get("B2H_CASES", len(CASES)))], scale, lower_bound)
        part_chain_timing(TCASES, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
