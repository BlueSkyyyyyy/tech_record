"""P7-22: land the P7-14 `tl.range` software pipeline into the *assembled* backward chain's
B1 (`dhu`), which §12/P7-9 left at `num_stages=1`, and re-benchmark end-to-end.

Motivation
----------
P7-14 (`docs/07` §17) showed, on the *isolated* bf16 Triton B1 prototype, that a software
pipeline (`tl.range(..., num_stages)`) alone brings B1 from 1.96x to 1.00x / 0.85x FLA --
"镜像 K2 的 TMA 输入流水是全部收益来源".  But the *assembled* chain (`check_bwd_b1_bf16_e2e.py`,
P7-9, `docs/07` §12) still builds its B1 with `num_stages=1`; the pipeline win was never
realized in the real chain.  This script closes that loop:

  Part A  correctness of the chain with a pipelined B1 vs FLA (same 6 cases as P6-8/P7-9,
          incl. non-divisible T, g=0, B=2) and vs the num_stages=1 chain (pipeline must not
          change the result beyond fp32 accumulation ordering noise).
  Part B  backward-only chain timing, wall + dev (CUPTI), for NS in {1,2,3,4} -- plus the
          per-kernel B1 device time so the chain-level win is attributable.
  Part C  isolated B1 kernel-only sweep over BT (16 = deployment granularity, 64) x NS, to
          reconcile with §17's standalone numbers in the *same* kernel build.

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle card:
    CUDA_VISIBLE_DEVICES=1 python .../check_bwd_b1_pipe_e2e.py
    B1P_MODE=timing ...   # skip Part A
    B1P_MODE=correctness ...
"""
import math
import os
import statistics
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton
import triton.language as tl

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import (make_inputs, flash_forward, reconstruct, D, CHUNK,   # noqa: E402
                                stats, cdiv)
from check_bwd_dhu_kernel import build_kernel as build_dhu, dhu_triton             # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton       # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton       # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                             # noqa: E402
from check_bwd_b1_bf16_e2e import (make_chain_fn, fla_golden, time_wall,           # noqa: E402
                                   profile_dev, prep)
from fla.ops.utils import chunk_local_cumsum                                       # noqa: E402


# --------------------------------------------- bf16-mma B1 with a tl.range software pipeline
def build_dhu_bf16_pipe(ns_default=1):
    """Same recurrence and same bf16-mma operands as `check_bwd_b1_bf16_e2e.build_dhu_bf16`
    (fp32 running state, every `tl.dot` casts to the operand dtype = exactly FLA), but the
    reverse chunk loop is expressed as a *forward* `tl.range` so Triton can software-pipeline
    the global loads (`q/k/w/do/dv/e`) across iterations.  `NS` is a runtime constexpr so a
    single build is swept."""

    @triton.jit
    def _dhu_kernel_bf16_pipe(
        q_ptr, k_ptr, w_ptr, do_ptr, dv_ptr, etot_ptr, dht_ptr,
        dh_ptr, dh0_ptr, dv2_ptr,
        T, NT, scale,
        HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BV: tl.constexpr, HAS_DHT: tl.constexpr,
        NS: tl.constexpr,
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
        for cc in tl.range(0, NT, num_stages=NS):
            c = NT - 1 - cc
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

    return _dhu_kernel_bf16_pipe


def dhu_bf16_pipe_triton(kernel, fi, do, dv, scale, BT, BV=32, dht=None, NS=1, num_warps=4):
    """Identical operand prep to `dhu_bf16_triton`, only the launch differs (`NS`)."""
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
                 HV=HV, K=K, V=V, BT=BT, BV=BV, HAS_DHT=has_dht, NS=NS,
                 num_warps=num_warps, num_stages=1)
    return dh, dh0, dv2


# --------------------------------------------------------------------- chain with pipe B1
def make_chain_fn_pipe(fi, inp, qn, kn, do, dfin, scale, NS, BV=32, num_warps=4):
    dav_k = build_dav_kernel()
    dhu_k = build_dhu_bf16_pipe()
    dqkg_k = build_dqkg()
    intra_k = build_intra()
    v_f = inp["v"].float()
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    do_b = do.to(torch.bfloat16)

    def run():
        dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, dv, scale, CHUNK,
                                            BV=BV, dht=dfin, NS=NS, num_warps=num_warps)
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


# --------------------------------------------------------------------------------- parts
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
NS_LIST = [1, 2, 3, 4]


def part_a(cases, scale, lower_bound):
    print("=" * 78)
    print("Part A: chain with pipelined bf16-mma B1 vs FLA and vs NS=1 chain")
    print("=" * 78)
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        print(f"\n# {name}  B={B} T={T} H={H} seed={seed} g_scale={gs}")
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        ch1 = make_chain_fn_pipe(fi, inp, qn, kn, do, dfin, scale, NS=1)
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        r1 = ch1()
        for NS in NS_LIST:
            chN = make_chain_fn_pipe(fi, inp, qn, kn, do, dfin, scale, NS=NS)
            rN = chN()
            print(f"  -- NS={NS} vs FLA / vs NS=1 --")
            for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"),
                            ("dbeta", "beta"), ("dg", "g")]:
                _, mr = stats(f"  NS{NS} {fld} vs FLA", rN[fld], gold[gk])
                worst[f"{fld}"] = max(worst.get(fld, 0.0), mr)
                if NS != 1:
                    _, mr1 = stats(f"  NS{NS} {fld} vs NS1", rN[fld], r1[fld])
            _, mr = stats(f"  NS{NS} dh0 vs FLA", rN["dh0"], gold["h0"].transpose(-1, -2))
            worst["dh0"] = max(worst.get("dh0", 0.0), mr)
    print("\nworst-case max_rel across cases (pipelines vs FLA):")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        print(f"  {fld:6s}  {worst[fld]:.3e}")


def part_b(cases, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 78)
    print("Part B: backward-only chain timing, wall + dev (CUPTI), NS sweep + FLA")
    print("=" * 78)
    hdr = f"  {'case':18s} | {'FLA wall':>8s} {'FLA dev':>8s} |"
    for NS in NS_LIST:
        hdr += f" NS{NS} wall / dev / (dev/FLA) |"
    print(hdr)
    rows = []
    for (name, B, T, H, seed, gs) in cases:
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

        chains = {NS: make_chain_fn_pipe(fi, inp, qn, kn, do, dfin, scale, NS=NS)
                  for NS in NS_LIST}
        for NS in NS_LIST:
            chains[NS]()
        fla_bwd()
        torch.cuda.synchronize()
        w_fl = time_wall(fla_bwd)
        d_fl, p_fl = profile_dev(fla_bwd)
        line = f"  {name:18s} | {w_fl:8.1f} {d_fl:8.1f} |"
        devs = {}
        profs = {}
        for NS in NS_LIST:
            w = time_wall(chains[NS])
            d, p = profile_dev(chains[NS])
            devs[NS] = d
            profs[NS] = p
            line += f" {w:6.1f} {d:6.1f} ({d/d_fl:4.2f}x) |"
        print(line)
        rows.append((name, d_fl, devs, profs, p_fl))
    print("\n  B1 kernel-only (device us/call) per NS:")
    for (name, d_fl, devs, profs, p_fl) in rows:
        def b1(prof):
            tot = 0.0
            for ev in prof.key_averages():
                if "_dhu_kernel" in ev.key:
                    tot += ev.self_device_time_total / 15
            return tot
        print(f"  {name:18s} FLA_dhu=" + " ".join(
            f"{x:.1f}" for x in [next((ev.self_device_time_total / 15
              for ev in p_fl.key_averages() if "dhu" in ev.key), float('nan'))]) +
            "  ours=" + " ".join(f"NS{NS}:{b1(profs[NS]):.1f}" for NS in NS_LIST))
    return rows


def part_c(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part C: isolated B1 kernel-only, BT (16 deploy / 64) x NS sweep")
    print("=" * 78)
    dhu_k = build_dhu_bf16_pipe()
    print("  (CUPTI device us/call for the B1 kernel alone, BV=32)")
    warm = None
    if os.environ.get("B1P_CLOCKWARM"):
        warm = (torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16),
                torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16))
        print("  (clock-warm: heavy bf16 GEMM immediately before each measurement)")
    print(f"  {'case':18s} {'BT':>3s} | " + " ".join(f"NS{NS:>2d}" for NS in NS_LIST))
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b = dict(fi)
        for kk in ("qg", "kg", "w"):
            fi_b[kk] = fi[kk].to(torch.bfloat16)
        do_b = do.to(torch.bfloat16)
        dv = torch.randn_like(do)
        for BT in (16, 32, 64):
            row = f"  {name:18s} {BT:3d} | "
            for NS in NS_LIST:
                fn = lambda: dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, dv, scale, BT,
                                                  dht=dfin, NS=NS)
                try:
                    fn()
                    torch.cuda.synchronize()
                    if warm is not None:
                        for _ in range(30):
                            torch.mm(warm[0], warm[1])
                        torch.cuda.synchronize()
                    d, _ = profile_dev(fn, reps=15)
                    row += f"{d:6.1f} "
                except triton.runtime.errors.OutOfResources:
                    row += f"{'OOM':>6s} "
                    torch.cuda.synchronize()
            print(row)


def part_d(scale, lower_bound):
    """Measurement-artifact probe: an isolated tiny-grid launch of the B1 kernel is
    attributed a *higher* CUPTI device time than the same kernel launched back-to-back
    after `dav` inside one profiled window.  Explains why the isolated Part C absolutes
    are ~1.2-1.3x above the in-chain Part B attribution; the chain (Part B) is the honest
    number for chain claims, Part C is used only for relative trends."""
    from check_bwd_e2e import build_dav_kernel, dav_triton
    print("\n" + "=" * 78)
    print("Part D: isolated vs back-to-back (after dav) attribution of the B1 kernel alone")
    print("=" * 78)
    B, T, H = 1, 512, 4
    inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, 0, 1.0)
    dav_k = build_dav_kernel()
    dhu_k = build_dhu_bf16_pipe()
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    do_b = do.to(torch.bfloat16)

    def b1_attrib(prof):
        t, cnt = 0.0, 0
        for ev in prof.key_averages():
            if "_dhu_kernel" in ev.key:
                t += ev.self_device_time_total
                cnt += ev.count
        return t / max(cnt, 1), cnt

    print(f"  {'case':12s} | {'isolated dev':>12s} | {'combo B1 attrib':>15s} (count)")
    for NS in (1, 3):
        fn = lambda: dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, torch.randn_like(do), scale,
                                          CHUNK, dht=dfin, NS=NS)
        fn()
        torch.cuda.synchronize()
        d_iso, _ = profile_dev(fn, reps=15)

        def combo():
            _, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, CHUNK)
            dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, dv, scale, CHUNK, dht=dfin, NS=NS)

        combo()
        torch.cuda.synchronize()
        _, prof = profile_dev(combo, reps=15)
        d_cmb, cnt = b1_attrib(prof)
        print(f"  T512 H4 NS{NS} | {d_iso:12.1f} | {d_cmb:15.1f} ({cnt})")


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    mode = os.environ.get("B1P_MODE", "")
    if mode not in ("timing", "b", "c"):
        part_a(CASES[:int(os.environ.get("B1P_CASES", len(CASES)))], scale, lower_bound)
    if mode in ("", "b", "timing"):
        part_b(TCASES[:int(os.environ.get("B1P_TCASES", len(TCASES)))], scale, lower_bound)
    if mode in ("", "c", "timing"):
        part_c(TCASES[:int(os.environ.get("B1P_TCASES", len(TCASES)))], scale, lower_bound)
    if mode in ("", "d", "timing"):
        part_d(scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
