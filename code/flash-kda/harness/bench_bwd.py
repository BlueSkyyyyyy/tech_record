"""P7-7: backward-pass timing — our workspace-driven chain vs FLA `chunk_kda` autograd.

`bench_fwd.py` (P7-1) profiled the *forward* only.  The backward side (`docs/06`
P6-2..P6-8) was validated for *correctness* and only ever timed kernel-by-kernel with a
naive stopwatch.  This script gives the backward the same treatment the forward got in P7:

  * ``wall``  -- mean per-call wall time of ``REPS`` calls under one CUDA-event pair
                 (includes Python/launch host overhead), median of ``rounds`` rounds;
  * ``dev``   -- device (pure-kernel) time from ``torch.profiler`` (CUPTI), summed over
                 all kernels and divided by the number of calls.

Two things are timed, both **backward-only**:

  ours   the assembled chain ``dAv -> B1(dhu) -> B2(wy_dqkg) -> B3(intra) ->
         reverse-cumsum(B0)`` (the P6-7/P6-8 chain), driven entirely off FlashKDA's own K1
         workspace (six fields + B0).  The K1 workspace and the forward intermediates are
         precomputed once (not timed); only the backward Graph runs inside the loop.
  FLA    ``torch.autograd`` through ``fla.ops.kda.chunk_kda``, using ``retain_graph=True``
         on a pre-built graph so only the *backward* kernels run in the timed loop.

Host tax = ``wall - dev`` isolates Python/launch overhead (the P7-6 finding for the
forward: FLA's forward is dominated by ~0.8 ms of per-kernel Python launch wrapping).

Also prints a per-kernel breakdown (CUPTI) and a kernel count.

Usage (inside kernel_lab, GPU):
    python ~/proj/tech_record/code/flash-kda/harness/bench_bwd.py
"""
import math
import os
import statistics
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import (make_inputs, flash_forward, reconstruct, D, CHUNK,
                                LOG2E)                                            # noqa: E402
from check_bwd_dhu_kernel import build_kernel as build_dhu, dhu_triton         # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton   # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton   # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                         # noqa: E402

LN2 = math.log(2.0)
WARMUP = 8
REPS = 30
PROF_REPS = 15
ROUNDS = 3


# --------------------------------------------------------------------------- timing
def time_wall(fn, reps=REPS, rounds=ROUNDS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        means.append(s.elapsed_time(e) / reps)
    return statistics.median(means) * 1e3          # ms -> us


def profile_dev(fn, reps=PROF_REPS):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    dev = sum(ev.self_device_time_total for ev in prof.key_averages())
    return dev / reps, prof                         # us, profiler handle


# ------------------------------------------------------------------- ours chain
def make_chain_fn(fi, inp, qn, kn, do, dfin, scale):
    """Pre-build every Triton kernel once; return a closure running only the backward."""
    dav_k = build_dav_kernel()
    dhu_k = build_dhu()
    dqkg_k = build_dqkg()
    intra_k = build_intra()
    from fla.ops.utils import chunk_local_cumsum
    h0T = inp["h0"].float().permute(0, 1, 3, 2).contiguous()
    v_f = inp["v"].float()

    def run():
        dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = dhu_triton(dhu_k, fi, do, dv, h0T, scale, CHUNK, dht=dfin)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi["v_new"], g=fi["g_cum"], beta=fi["beta"],
                    A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
        b2 = wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK)
        inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=fi["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])
        b3 = intra_triton(intra_k, inp3, CHUNK, CHUNK)
        dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"],
                    dv=b2["dv"], dg=dg, dbeta=b2["db"] + b3["db"], dh0=dh0)

    return run


# ------------------------------------------------------------------- FLA backward
def make_fla_bwd_fn(qn, kn, v, beta_act, g_fla, h0, scale, do, dfin):
    """Autograd backward on a pre-built graph; ``retain_graph`` keeps it re-runnable."""
    from fla.ops.kda import chunk_kda
    qa = qn.clone().requires_grad_(True)
    ka = kn.clone().requires_grad_(True)
    va = v.clone().requires_grad_(True)
    ba = beta_act.clone().requires_grad_(True)
    ga = g_fla.clone().requires_grad_(True)
    ha = h0.clone().requires_grad_(True)
    o, s = chunk_kda(qa, ka, va, ga, ba, scale=scale,
                     initial_state=ha.float().transpose(-1, -2),
                     output_final_state=True, use_qk_l2norm_in_kernel=False,
                     use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
    loss = (o.float() * do).sum() + (s.float() * dfin).sum()
    holders = (qa, ka, va, ba, ga, ha)

    def run():
        loss.backward(retain_graph=True)
        for t in holders:
            t.grad = None

    return run


# ------------------------------------------------------------------- FlashKDA fwd
def make_flash_fwd_fn(inp, scale, lower_bound):
    import flash_kda
    B, T, H, _ = inp["q"].shape
    from check_bwd_ws_reuse import cdiv
    out = torch.empty(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    fin = torch.empty(B, H, D, D, device="cuda", dtype=torch.bfloat16)
    h0 = inp["h0"].bfloat16().transpose(-1, -2).contiguous()

    def run():
        flash_kda.fwd(inp["q"].contiguous(), inp["k"].contiguous(),
                      inp["v"].contiguous(), inp["g"].contiguous(),
                      inp["beta"].contiguous(), scale, out,
                      A_log=inp["A_log"].clone(), dt_bias=inp["dt_bias"].clone(),
                      lower_bound=lower_bound, initial_state=h0, final_state=fin)
    return run


# ------------------------------------------------------------------- FLA fwd+bwd
def make_fla_fwdbwd_fn(qn, kn, v, beta_act, g_fla, h0, scale, do, dfin):
    from fla.ops.kda import chunk_kda

    def run():
        qa = qn.clone().requires_grad_(True)
        ka = kn.clone().requires_grad_(True)
        va = v.clone().requires_grad_(True)
        ba = beta_act.clone().requires_grad_(True)
        ga = g_fla.clone().requires_grad_(True)
        ha = h0.clone().requires_grad_(True)
        o, s = chunk_kda(qa, ka, va, ga, ba, scale=scale,
                         initial_state=ha.float().transpose(-1, -2),
                         output_final_state=True, use_qk_l2norm_in_kernel=False,
                         use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
        ((o.float() * do).sum() + (s.float() * dfin).sum()).backward()

    return run


# --------------------------------------------------------------------------- runner
def run_case(name, B, T, H, seed=0, g_scale=1.0):
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    inp = make_inputs(B, T, H, seed, "cuda", g_scale)
    fw = flash_forward(inp, scale, lower_bound, "cuda")
    o_rec, fin_rec, fi = reconstruct(fw, inp, lower_bound, scale)

    torch.manual_seed(seed + 77)
    do = torch.randn(B, T, H, D, device="cuda")
    dfin = torch.randn(B, H, D, D, device="cuda") * 0.01

    qn = torch.nn.functional.normalize(inp["q"].float(), dim=-1)
    kn = torch.nn.functional.normalize(inp["k"].float(), dim=-1)
    beta_act = torch.sigmoid(inp["beta"].float())
    inner = inp["g"].float() + inp["dt_bias"].view(1, 1, H, D)
    g_fla = lower_bound * torch.sigmoid(torch.exp(inp["A_log"].view(1, 1, H, 1)) * inner)

    chain = make_chain_fn(fi, inp, qn, kn, do, dfin, scale)
    fla_bwd = make_fla_bwd_fn(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"],
                              scale, do, dfin)
    fla_fb = make_fla_fwdbwd_fn(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"],
                                scale, do, dfin)
    fk_fwd = make_flash_fwd_fn(inp, scale, lower_bound)

    chain()
    fla_bwd()
    torch.cuda.synchronize()

    wall_ours = time_wall(chain)
    wall_fla = time_wall(fla_bwd)
    dev_ours, prof_ours = profile_dev(chain)
    dev_fla, prof_fla = profile_dev(fla_bwd)
    wall_fb = time_wall(fla_fb)
    dev_fb, _ = profile_dev(fla_fb)
    wall_fk = time_wall(fk_fwd)
    dev_fk, _ = profile_dev(fk_fwd)

    def nkern(prof):
        return sum(ev.count for ev in prof.key_averages()) // PROF_REPS

    print(f"\n{name:24s} B={B} T={T} H={H} D={D} g_scale={g_scale}")
    print(f"  backward-only   ours  wall={wall_ours:9.1f}us  dev={dev_ours:9.1f}us  "
          f"host={wall_ours-dev_ours:8.1f}us  kernels={nkern(prof_ours):3d}")
    print(f"  backward-only   FLA   wall={wall_fla:9.1f}us  dev={dev_fla:9.1f}us  "
          f"host={wall_fla-dev_fla:8.1f}us  kernels={nkern(prof_fla):3d}")
    print(f"  ratio ours/FLA   wall={wall_ours/wall_fla:5.2f}x  dev={dev_ours/dev_fla:5.2f}x")
    print(f"  FlashKDA fwd          wall={wall_fk:9.1f}us  dev={dev_fk:9.1f}us  "
          f"kernels=           3")
    print(f"  full pipe ours (fwd+chain) ~ dev {dev_fk+dev_ours:7.1f}us  "
          f"FLA fwd+bwd wall={wall_fb:9.1f}us dev={dev_fb:8.1f}us")
    return dict(name=name, B=B, T=T, H=H,
                wall_ours=wall_ours, dev_ours=dev_ours, host_ours=wall_ours - dev_ours,
                wall_fla=wall_fla, dev_fla=dev_fla, host_fla=wall_fla - dev_fla,
                nk_ours=nkern(prof_ours), nk_fla=nkern(prof_fla),
                wall_fk=wall_fk, dev_fk=dev_fk, wall_fb=wall_fb, dev_fb=dev_fb)


def kernel_breakdown(name, B, T, H, seed=0, g_scale=1.0):
    from torch.profiler import ProfilerActivity, profile
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    inp = make_inputs(B, T, H, seed, "cuda", g_scale)
    fw = flash_forward(inp, scale, lower_bound, "cuda")
    _, _, fi = reconstruct(fw, inp, lower_bound, scale)
    torch.manual_seed(seed + 77)
    do = torch.randn(B, T, H, D, device="cuda")
    dfin = torch.randn(B, H, D, D, device="cuda") * 0.01
    qn = torch.nn.functional.normalize(inp["q"].float(), dim=-1)
    kn = torch.nn.functional.normalize(inp["k"].float(), dim=-1)
    beta_act = torch.sigmoid(inp["beta"].float())
    inner = inp["g"].float() + inp["dt_bias"].view(1, 1, H, D)
    g_fla = lower_bound * torch.sigmoid(torch.exp(inp["A_log"].view(1, 1, H, 1)) * inner)

    chain = make_chain_fn(fi, inp, qn, kn, do, dfin, scale)
    fla_bwd = make_fla_bwd_fn(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"],
                              scale, do, dfin)

    print(f"\n{'='*96}\nkernel breakdown [{name}] B={B} T={T} H={H} "
          f"(device us/call, {PROF_REPS} calls)")
    for label, fn in [("ours chain", chain), ("FLA backward", fla_bwd)]:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(PROF_REPS):
                fn()
            torch.cuda.synchronize()
        print(f"  -- {label} --")
        for ev in prof.key_averages():
            us = ev.self_device_time_total / PROF_REPS
            if us > 0.5:
                print(f"     {us:10.1f} us  {ev.count // PROF_REPS:3d}x  {ev.key[:64]}")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    cases = [
        ("T=512 H=4", 1, 512, 4, 0, 1.0),
        ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
        ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
        ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0),
    ]
    cases = cases[:int(os.environ.get("BWD_CASES", len(cases)))]
    rows = []
    for (name, B, T, H, seed, gs) in cases:
        rows.append(run_case(name, B, T, H, seed=seed, g_scale=gs))

    for (name, B, T, H, seed, gs) in cases[:2]:
        kernel_breakdown(name, B, T, H, seed=seed, g_scale=gs)

    print(f"\n{'='*96}")
    print(f"  {'case':20s} | {'ours wall':>9s} {'ours dev':>8s} {'ours host':>9s} | "
          f"{'FLA wall':>8s} {'FLA dev':>7s} {'FLA host':>8s} | {'w-o/FLA':>7s} "
          f"{'d-o/FLA':>7s} | {'#k o/f':>7s}")
    for r in rows:
        print(f"  {r['name']:20s} | {r['wall_ours']:9.1f} {r['dev_ours']:8.1f} "
              f"{r['host_ours']:9.1f} | {r['wall_fla']:8.1f} {r['dev_fla']:7.1f} "
              f"{r['host_fla']:8.1f} | {r['wall_ours']/r['wall_fla']:7.2f} "
              f"{r['dev_ours']/r['dev_fla']:7.2f} | "
              f"{r['nk_ours']:3d}/{r['nk_fla']:3d}")
    print("\nDONE")


if __name__ == "__main__":
    main()
