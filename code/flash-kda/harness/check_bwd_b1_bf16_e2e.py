"""P7-9: land the P7-8 "cheapest lever" (B1 on native bf16 workspace operands) into the
*assembled* workspace-driven backward chain, then re-benchmark end-to-end.

P7-8 (`check_bwd_dhu_dtype.py`, `docs/07` §11) showed that the standalone B1 kernel with
bf16 operands (bf16-mma, fp32 state accumulation — exactly FLA) is **2.2-2.5x** faster than
the fp32/tf32 path and already beats FLA's `dhu`.  But that experiment swapped operands *by
hand* on the isolated kernel: the assembled chain (`check_bwd_ws_reuse.py`, P6-8) still feeds
B1 **fp32** operands, because `reconstruct` casts the entire workspace with `.float()` —
the actual FlashKDA fields are bf16.  So the P7-8 win was never realized in the real chain.

This script closes that loop:

  Part A  correctness of the full chain with a bf16-mma B1
          (`dAv -> B1bf16 -> B2 -> B3 -> rcumsum(B0)`) vs `torch.autograd`(FLA), same 6 cases
          as P6-8 (incl. non-divisible T, g=0, B=2).  Also prints fp32-B1 vs bf16-B1 delta so
          the error the operand switch *adds* is explicit.
  Part B  backward-only timing: ours(fp32 B1) vs ours(bf16 B1) vs FLA, `wall` + `dev`
          (CUPTI), plus a per-kernel breakdown of both our variants.

Only B1 changes; B2/B3 keep the tf32 path (P7-8 showed `dAv/B2/B3` are already ahead).

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed):
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_b1_bf16_e2e.py
    B1BF_CASES=1 ...   # correctness only
    B1BF_TIMING_ONLY=1 ...  # timing only
"""
import math
import os
import statistics
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton
import triton.language as tl

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import (make_inputs, flash_forward, reconstruct, D, CHUNK,   # noqa: E402
                                LOG2E, stats, cdiv)
from check_bwd_dhu_kernel import build_kernel as build_dhu, dhu_triton             # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton       # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton       # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                       # noqa: E402

LN2 = math.log(2.0)


# ------------------------------------------------------------- bf16-mma B1 (FLA-exact)
def build_dhu_bf16():
    """Same recurrence as `check_bwd_dhu_kernel._dhu_kernel`, but every `tl.dot` casts its
    operands to the operand dtype (bf16) — exactly FLA (`b_dh.to(b_k.dtype)`); the running
    state `b_dh` stays fp32.  Adds the `HAS_DHT` final-state-gradient seed (P7-7)."""
    @triton.jit
    def _dhu_kernel_bf16(
        q_ptr, k_ptr, w_ptr, do_ptr, dv_ptr, etot_ptr, dht_ptr,
        dh_ptr, dh0_ptr, dv2_ptr,
        T, NT, scale,
        HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BV: tl.constexpr, HAS_DHT: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_nh = tl.program_id(1)
        i_n = i_nh // HV
        i_h = i_nh % HV
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        o_k = tl.arange(0, K)
        if HAS_DHT:
            p_dht = dht_ptr + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
            b_dh = tl.load(p_dht, mask=m_v[None, :], other=0.0).to(tl.float32)
        else:
            b_dh = tl.zeros([K, BV], dtype=tl.float32)
        bo_t = dh_ptr.dtype.element_ty
        for c in range(NT - 1, -1, -1):
            p_dh = dh_ptr + ((i_n * NT + c) * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_dh, b_dh.to(bo_t), mask=m_v[None, :])
            o_t = c * BT + tl.arange(0, BT)
            m_t = o_t < T
            base = (i_n * T + o_t)[:, None]
            p_k = k_ptr + base * (HV * K) + i_h * K + o_k[None, :]
            p_q = q_ptr + base * (HV * K) + i_h * K + o_k[None, :]
            p_w = w_ptr + base * (HV * K) + i_h * K + o_k[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)
            b_q = tl.load(p_q, mask=m_t[:, None], other=0.0)
            b_w = tl.load(p_w, mask=m_t[:, None], other=0.0)
            p_do = do_ptr + base * (HV * V) + i_h * V + o_v[None, :]
            p_dv = dv_ptr + base * (HV * V) + i_h * V + o_v[None, :]
            m_dv = m_t[:, None] & m_v[None, :]
            b_do = tl.load(p_do, mask=m_dv, other=0.0)
            b_dv = tl.load(p_dv, mask=m_dv, other=0.0)
            p_e = etot_ptr + ((i_n * NT + c) * HV + i_h) * K + o_k
            b_e = tl.load(p_e)
            dt = b_k.dtype
            b_du = tl.dot(b_k, b_dh.to(dt)) + b_dv
            p_dv2 = dv2_ptr + base * (HV * V) + i_h * V + o_v[None, :]
            tl.store(p_dv2, b_du.to(dv2_ptr.dtype.element_ty), mask=m_dv)
            b_dh = (b_dh * b_e[:, None]
                    + tl.dot(tl.trans(b_q), b_do.to(dt)) * scale
                    - tl.dot(tl.trans(b_w), b_du.to(dt)))
        p_dh0 = dh0_ptr + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(dh0_ptr.dtype.element_ty), mask=m_v[None, :])

    return _dhu_kernel_bf16


def dhu_bf16_triton(kernel, fi, do, dv, scale, BT, BV=32, dht=None):
    """Feed B1 its operands in bf16 straight from the workspace (native `kd/qd/kr` are bf16;
    `w`/B0 gate are computed fp32 then rounded, exactly FLA stores them bf16).

    Only the *mma operands* are bf16 (that is where the P7-8 win is: bf16 tensor cores at
    2x tf32 + half the loaded bytes); the running state and the `dh`/`dv2` outputs stay
    **fp32** so the downstream B2/B3 kernels — which the P6-8/P6-7 chain feeds fp32 — are
    unchanged.  (P7-8 also stored `dh` bf16, matching FLA; that is a *consumer* choice, not
    part of the B1 speed lever.)"""
    qg = fi["qg"].to(torch.bfloat16)
    kg = fi["kg"].to(torch.bfloat16)
    w = fi["w"].to(torch.bfloat16)
    g_cum = fi["g_cum"]
    B, T, HV, K = qg.shape
    V = do.shape[-1]
    NT = cdiv(T, BT)
    last_idx = torch.clamp((torch.arange(NT, device=qg.device) + 1) * BT - 1, max=T - 1)
    etot = torch.exp2(g_cum.float()[:, last_idx]).contiguous()
    dh = torch.empty(B, NT, HV, K, V, dtype=torch.float32, device=qg.device)
    dh0 = torch.empty(B, HV, K, V, dtype=torch.float32, device=qg.device)
    dv2 = torch.empty(B, T, HV, V, dtype=torch.float32, device=qg.device)
    d_do = do.to(torch.bfloat16) if do.dtype != torch.bfloat16 else do
    d_dv = dv.to(torch.bfloat16) if dv.dtype != torch.bfloat16 else dv
    has_dht = dht is not None
    dht_arg = dht.contiguous() if has_dht else dh0
    grid = (cdiv(V, BV), B * HV)
    kernel[grid](qg.contiguous(), kg.contiguous(), w.contiguous(), d_do.contiguous(),
                 d_dv.contiguous(), etot, dht_arg, dh, dh0, dv2, T, NT, scale,
                 HV=HV, K=K, V=V, BT=BT, BV=BV, HAS_DHT=has_dht, num_warps=4, num_stages=1)
    return dh, dh0, dv2


# ------------------------------------------------------------------ our chain variants
def make_chain_fn(fi, inp, qn, kn, do, dfin, scale, b1_bf16):
    """Pre-build every Triton kernel once; return a closure running only the backward.

    For the bf16-B1 variant the workspace-native bf16 operands (`qg/kg/w`) and `do` are
    prepared *once* outside the timed closure — in the real chain they are already bf16, so
    this is the honest kernel-time comparison (no per-call harness cast tax)."""
    from check_bwd_e2e import build_dav_kernel, dav_triton
    dav_k = build_dav_kernel()
    dhu_k = build_dhu_bf16() if b1_bf16 else build_dhu()
    dqkg_k = build_dqkg()
    intra_k = build_intra()
    h0T = inp["h0"].float().permute(0, 1, 3, 2).contiguous()
    v_f = inp["v"].float()
    if b1_bf16:
        fi_b = dict(fi)
        for kk in ("qg", "kg", "w"):
            fi_b[kk] = fi[kk].to(torch.bfloat16)
        do_b = do.to(torch.bfloat16)

    def run():
        dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        if b1_bf16:
            dh, dh0, dv2 = dhu_bf16_triton(dhu_k, fi_b, do_b, dv, scale, CHUNK, dht=dfin)
        else:
            dh, dh0, dv2 = dhu_triton(dhu_k, fi, do, dv, h0T, scale, CHUNK, dht=dfin)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi["v_new"], g=fi["g_cum"], beta=fi["beta"],
                    A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
        b2 = wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK)
        inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=fi["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])
        b3 = intra_triton(intra_k, inp3, CHUNK, CHUNK)
        dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"],
                    dv=b2["dv"], dg=dg, dbeta=b2["db"] + b3["db"], dh0=dh0,
                    dh=dh, dv2=dv2)

    return run


def fla_golden(qn, kn, v, beta_act, g_fla, h0, scale, do, dfin):
    from fla.ops.kda import chunk_kda
    q = qn.clone().requires_grad_(True)
    k = kn.clone().requires_grad_(True)
    vv = v.clone().requires_grad_(True)
    beta = beta_act.clone().requires_grad_(True)
    g = g_fla.clone().requires_grad_(True)
    hh = h0.clone().requires_grad_(True)
    o, s = chunk_kda(q, k, vv, g, beta, scale=scale,
                     initial_state=hh.float().transpose(-1, -2),
                     output_final_state=True, use_qk_l2norm_in_kernel=False,
                     use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
    loss = (o.float() * do).sum() + (s.float() * dfin).sum()
    loss.backward()
    return dict(q=q.grad, k=k.grad, v=vv.grad, beta=beta.grad, g=g.grad, h0=hh.grad)


# --------------------------------------------------------------------------- timing
def time_wall(fn, reps=30, rounds=3, warmup=8):
    for _ in range(warmup):
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


def profile_dev(fn, reps=15):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return sum(ev.self_device_time_total for ev in prof.key_averages()) / reps, prof


def nkern(prof, reps=15):
    return sum(ev.count for ev in prof.key_averages()) // reps


# --------------------------------------------------------------------------- runner
def prep(B, T, H, scale, lower_bound, seed=0, g_scale=1.0):
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
    return inp, fi, qn, kn, beta_act, g_fla, do, dfin


def part_a(cases, scale, lower_bound):
    print("=" * 78)
    print("Part A: chain with bf16-mma B1 vs FLA autograd (and fp32-B1 chain, P6-8)")
    print("=" * 78)
    worst_bf, worst_fp, worst_delta = {}, {}, {}
    for (name, B, T, H, seed, gs) in cases:
        print(f"\n# {name}  B={B} T={T} H={H} seed={seed} g_scale={gs}")
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        chain_fp = make_chain_fn(fi, inp, qn, kn, do, dfin, scale, b1_bf16=False)
        chain_bf = make_chain_fn(fi, inp, qn, kn, do, dfin, scale, b1_bf16=True)
        ours_fp = chain_fp()
        ours_bf = chain_bf()
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        print("  -- chain-bf16-B1 vs FLA --")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"), ("dg", "g")]:
            _, mr = stats(f"bf16 {fld}", ours_bf[fld], gold[gk])
            worst_bf[fld] = max(worst_bf.get(fld, 0.0), mr)
        _, mr = stats("bf16 dh0", ours_bf["dh0"], gold["h0"].transpose(-1, -2))
        worst_bf["dh0"] = max(worst_bf.get("dh0", 0.0), mr)
        print("  -- chain-fp32-B1 vs FLA (P6-8 baseline) --")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"), ("dg", "g")]:
            _, mr = stats(f"fp32 {fld}", ours_fp[fld], gold[gk])
            worst_fp[fld] = max(worst_fp.get(fld, 0.0), mr)
        _, mr = stats("fp32 dh0", ours_fp["dh0"], gold["h0"].transpose(-1, -2))
        worst_fp["dh0"] = max(worst_fp.get("dh0", 0.0), mr)
        print("  -- bf16-B1 chain vs fp32-B1 chain (the operand swap's own delta) --")
        for fld in ["dq", "dk", "dv", "dbeta", "dg"]:
            _, mr = stats(f"delta {fld}", ours_bf[fld], ours_fp[fld])
            worst_delta[fld] = max(worst_delta.get(fld, 0.0), mr)
        stats("B1 dh  bf16 vs fp32", ours_bf["dh"], ours_fp["dh"])
    print("\nworst-case max_rel across cases:")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        print(f"  {fld:6s}  bf16B1-vs-FLA {worst_bf[fld]:.3e}   fp32B1-vs-FLA {worst_fp[fld]:.3e}")


def part_b(cases, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 78)
    print("Part B: backward-only timing (ours fp32-B1 / ours bf16-B1 / FLA), wall + dev (CUPTI)")
    print("=" * 78)
    print(f"  {'case':20s} | {'fp32 wall':>9s} {'fp32 dev':>8s} | {'bf16 wall':>9s} {'bf16 dev':>8s} "
          f"| {'FLA wall':>8s} {'FLA dev':>7s} | {'bf16/fp32 dev':>13s} {'bf16/FLA dev':>12s}")
    rows = []
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        chain_fp = make_chain_fn(fi, inp, qn, kn, do, dfin, scale, b1_bf16=False)
        chain_bf = make_chain_fn(fi, inp, qn, kn, do, dfin, scale, b1_bf16=True)

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

        chain_fp(); chain_bf(); fla_bwd()
        torch.cuda.synchronize()
        w_fp, w_bf, w_fl = time_wall(chain_fp), time_wall(chain_bf), time_wall(fla_bwd)
        d_fp, p_fp = profile_dev(chain_fp)
        d_bf, p_bf = profile_dev(chain_bf)
        d_fl, p_fl = profile_dev(fla_bwd)
        print(f"  {name:20s} | {w_fp:9.1f} {d_fp:8.1f} | {w_bf:9.1f} {d_bf:8.1f} "
              f"| {w_fl:8.1f} {d_fl:7.1f} | {d_bf/d_fp:12.2f}x {d_bf/d_fl:11.2f}x")
        rows.append((name, w_fp, d_fp, w_bf, d_bf, w_fl, d_fl, p_fp, p_bf))
    print("\n  per-kernel (device us/call):")
    for (name, *_rest, p_fp, p_bf) in rows:
        keys = ["_dhu_kernel", "wy_dqkg", "intra", "dav", "chunk_local_cumsum"]
        print(f"  -- {name} --")
        for label, prof in [("fp32-B1", p_fp), ("bf16-B1", p_bf)]:
            agg = {}
            for ev in prof.key_averages():
                for kk in keys:
                    if kk in ev.key:
                        agg[kk] = agg.get(kk, 0.0) + ev.self_device_time_total / 15
            print(f"     {label:8s} " + "  ".join(f"{k}={agg.get(k,0):.1f}" for k in keys))
    return rows


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    cases = [
        ("T=128 H=2", 1, 128, 2, 0, 1.0),
        ("T=256 H=4", 1, 256, 4, 1, 1.0),
        ("T=250 H=3 (non-div)", 1, 250, 3, 2, 1.0),
        ("T=512 H=4 g=2", 1, 512, 4, 3, 2.0),
        ("B2 T=513 H=2 (non-div)", 2, 513, 2, 4, 1.0),
        ("T=192 H=2 g=0", 1, 192, 2, 5, 0.0),
    ]
    tcases = [
        ("T=512 H=4", 1, 512, 4, 0, 1.0),
        ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
        ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
        ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0),
    ]
    only = os.environ.get("B1BF_MODE", "")
    if only != "timing":
        part_a(cases[:int(os.environ.get("B1BF_CASES", len(cases)))], scale, lower_bound)
    if only != "correctness":
        part_b(tcases[:int(os.environ.get("B1BF_TCASES", len(tcases)))], scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
