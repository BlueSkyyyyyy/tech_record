"""P7-28: B3 `intra` consumes B2's `dq/dk/dg` *accumulator in place* -- drop the chain's
outside-the-kernel `b2 + b3` elementwise sums.

Motivation (§27.6 / §27.7)
--------------------------
After the P7-24 MMA rewrite B3 is L2/mem-bound (DRAM 57% SOL, L2 74%).  But the assembled
chain (`check_bwd_b1_pipe_e2e.make_chain_fn_pipe`, §25) runs B3 on a **freshly zeroed**
`dq/dk/dg` and then materializes three full `[B,T,H,K]` elementwise sums outside the kernel:

    dq = b2["dq"] + b3["dq"]        # + dk, and dg = cumsum(b2["dg"] + b3["dg"])

That is ~3 x (2 reads + 1 write) of a `B*T*H*K` fp32 tensor per backward.  It is also *not*
what FLA does: FLA's `chunk_kda_bwd_intra` treats the incoming `dq/dk/dg` as an
**accumulator** (reads them, adds the intra contribution, writes to a separate `dq2`):

    chunk_intra.py:559-560   b_dq2 = b_dq2 + tl.load(p_dq);  tl.store(p_dq2, ...)
    chunk_intra.py:662-667   b_dg2 += ... + tl.load(p_dg);  b_dk2 += tl.load(p_dk); store

so our "zero + sum outside" is a *harness convenience* that wastes one full read of the
zero buffer and then a second read of B3's output, plus a write, for each of `dq/dk/dg`.

This script makes B3 consume B2's `dq/dk/dg` **in place** (each `(token, k)` element is
produced by exactly one program -- the grid's `i_k`/`i_i` axes partition `K` and the inner
sub-block -- so no cross-program accumulation is needed; only `db` needs the atomic add on
top of B2's value, exactly as before).  The three outside adds vanish.

  Part A   isolated kernel: in-place accumulate vs (zero-fill B3) + outside add -- expect
           **bit-identical** (`b + 0` is exact); plus the FLA contract check: FLA's intra
           with a non-zero incoming accumulator == FLA(zeros) + accumulator.
  Part B   chain: correctness with the in-place B3 vs the current chain (bit-identical) and
           vs FLA `torch.autograd`; chain `dev` (CUPTI) + `wall`, plus a per-kernel split
           showing the removed elementwise adds.
  Part C   same with a bf16 operands sanity variant is unnecessary -- the kernel arithmetic
           is untouched; we only change *where* the accumulator lives.

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle card:
    CUDA_VISIBLE_DEVICES=1 python .../check_bwd_b3_accum.py
    B3A_MODE=correctness | chain | timing
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import triton
import triton.language as tl

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import D, CHUNK, stats, cdiv                                    # noqa: E402
from check_bwd_intra_kernel import make_inputs                                          # noqa: E402
from check_bwd_b3_mma import build_intra_mma, intra_triton_mma, CASES, TCASES           # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton            # noqa: E402
from check_bwd_b1_pipe_e2e import build_dhu_bf16_pipe, dhu_bf16_pipe_triton             # noqa: E402
from check_bwd_e2e import build_dav_kernel, dav_triton                                  # noqa: E402
from check_bwd_b1_bf16_e2e import prep, fla_golden, time_wall, profile_dev              # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                            # noqa: E402


# --------------------------------------------------------------- B3 into caller-provided buffers
def intra_mma_into(kernel, inp, BT, BC, dq, dk, db, dg, num_warps=4, num_stages=1):
    """Same launch as `intra_triton_mma`, but writes into the caller's buffers.

    When the buffers already hold B2's output, this realizes FLA's *accumulator* contract
    (the kernel's `b + tl.load(out)` becomes an in-place accumulate); when they are zeros
    it is exactly the previous behavior."""
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, BT)
    NC = cdiv(BT, BC)
    BK = min(32, triton.next_power_of_2(K))
    NK = cdiv(K, BK)
    grid = (NK * NC, NT, B * H)
    kernel[grid](
        inp["q"], inp["k"], inp["g"], inp["beta"], inp["dAqk"], inp["dAkk"],
        dq, dk, db, dg,
        T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC,
        num_warps=num_warps, num_stages=num_stages)
    return dict(dq=dq, dk=dk, db=db, dg=dg)


# --------------------------------------------------------------------------- Part A
def part_a():
    print("=" * 78)
    print("Part A: in-place accumulate == (zero B3) + outside add ; and FLA's contract")
    print("=" * 78)
    mma_k = build_intra_mma()
    worst = {}
    for (B, T, H, K, BT, BC, seed) in [
        (2, 512, 4, 128, 64, 16, 1), (1, 250, 3, 128, 64, 16, 2),
        (1, 128, 2, 64, 16, 16, 6),
    ]:
        print(f"\n=== B={B} T={T} H={H} K={K} BT={BT} BC={BC} seed={seed} ===")
        inp = make_inputs(B, T, H, K, BT, BC, "cuda", dtype=torch.float32, seed=seed)
        tri0 = intra_triton_mma(mma_k, inp, BT, BC)          # zero-filled, inside kernel
        acc = {n: torch.randn_like(tri0[n]) for n in ["dq", "dk", "db", "dg"]}
        # baseline: B3 on zeros, then add acc outside (fp32 exact order: (x + 0) + acc)
        base = {n: tri0[n] + acc[n] for n in ["dq", "dk", "db", "dg"]}
        # in-place: B3 accumulates into acc directly
        ip = intra_mma_into(mma_k, inp, BT, BC, acc["dq"], acc["dk"], acc["db"], acc["dg"])
        print("  -- in-place vs (zero + outside add) --")
        for n in ["dq", "dk", "db", "dg"]:
            _, mr = stats(f"  ip {n}", ip[n], base[n])
            worst[n] = max(worst.get(n, 0.0), mr)

    print("\n  -- FLA contract: intra(nonzero accum) == intra(zero) + accum --")
    from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
    B, T, H, K, BT, BC, seed = 2, 512, 4, 128, 64, 16, 1
    inp = make_inputs(B, T, H, K, BT, BC, "cuda", dtype=torch.float32, seed=seed)
    dev = inp["q"].device

    def fla_into(acc):
        dq = acc["dq"].clone()
        dk = acc["dk"].clone()
        db = acc["db"].clone()
        dg = acc["dg"].clone()
        o = chunk_kda_bwd_intra(
            q=inp["q"], k=inp["k"], g=inp["g"], beta=inp["beta"],
            dAqk=inp["dAqk"], dAkk=inp["dAkk"], dq=dq, dk=dk, db=db, dg=dg,
            chunk_size=BT, safe_gate=False)
        return dict(zip(["dq", "dk", "db", "dg"], o))

    zero = {"dq": torch.zeros(B, T, H, K, device=dev, dtype=torch.float32),
            "dk": torch.zeros(B, T, H, K, device=dev, dtype=torch.float32),
            "dg": torch.zeros(B, T, H, K, device=dev, dtype=torch.float32),
            "db": torch.zeros(B, T, H, device=dev, dtype=torch.float32)}
    z = fla_into(zero)
    acc = {n: torch.randn_like(z[n]) for n in z}
    a = fla_into(acc)
    for n in ["dq", "dk", "db", "dg"]:
        _, mr = stats(f"  fla {n}", a[n], z[n] + acc[n])
        worst["fla_" + n] = max(worst.get("fla_" + n, 0.0), mr)

    print("\nworst-case max_rel:")
    for kk in sorted(worst):
        print(f"  {kk:10s}  {worst[kk]:.3e}")


# --------------------------------------------------------------------------- chain
def _prep_b(fi, inp, do):
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    return fi_b, do.to(torch.bfloat16)


def make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, accum):
    """The §25 chain (`dAv -> pipe B1 -> B2 -> MMA B3 -> reverse cumsum`), with B3 either
    on a fresh zero buffer + outside sums (`accum=False`) or accumulating B2's dq/dk/dg/db
    in place (`accum=True`)."""
    dav_k = build_dav_kernel()
    dhu_k = build_dhu_bf16_pipe()
    dqkg_k = build_dqkg()
    intra_k = build_intra_mma()
    v_f = inp["v"].float()

    def run():
        dAqk, dv = dav_triton(dav_k, fi_b["v_new"], fi_b["Aqk"], do, scale, CHUNK)
        dh, dh0, dv2 = dhu_bf16_pipe_triton(dhu_k, fi_b, do_b, dv, scale, CHUNK,
                                            NS=NS, dht=dfin)
        inp2 = dict(q=qn, k=kn, v=v_f, v_new=fi_b["v_new"], g=fi_b["g_cum"],
                    beta=fi_b["beta"], A=fi_b["Akk"], h=fi_b["h"], do=do, dh=dh, dv=dv2)
        b2 = wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK)
        inp3 = dict(q=qn, k=kn, g=fi_b["g_cum"], beta=fi_b["beta"],
                    dAqk=dAqk, dAkk=b2["dA"])
        dev = qn.device
        if accum:
            b3 = intra_mma_into(intra_k, inp3, CHUNK, CHUNK,
                                b2["dq"], b2["dk"], b2["db"], b2["dg"])
            dg = chunk_local_cumsum(b3["dg"], chunk_size=CHUNK, reverse=True)
            return dict(dq=b3["dq"], dk=b3["dk"], dv=b2["dv"], dg=dg,
                        dbeta=b3["db"], dh0=dh0, dh=dh, dv2=dv2)
        z = lambda *s: torch.zeros(*s, device=dev, dtype=torch.float32)
        b3 = intra_mma_into(intra_k, inp3, CHUNK, CHUNK,
                            z(*b2["dq"].shape), z(*b2["dk"].shape),
                            z(*b2["db"].shape), z(*b2["dg"].shape))
        dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=CHUNK, reverse=True)
        return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"], dv=b2["dv"], dg=dg,
                    dbeta=b2["db"] + b3["db"], dh0=dh0, dh=dh, dv2=dv2)

    return run


def part_chain(cases, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part B: chain correctness -- in-place B3 vs current chain vs FLA autograd")
    print("=" * 78)
    NS = 3
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b, do_b = _prep_b(fi, inp, do)
        ch0 = make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, accum=False)
        ch1 = make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, accum=True)
        r0, r1 = ch0(), ch1()
        gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)
        print(f"\n# {name}")
        print("  -- in-place vs current chain --")
        for fld in ["dq", "dk", "dv", "dbeta", "dg"]:
            _, mr = stats(f"  delta {fld}", r1[fld], r0[fld])
            worst[f"delta_{fld}"] = max(worst.get(f"delta_{fld}", 0.0), mr)
        print("  -- in-place chain vs FLA --")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"), ("dg", "g")]:
            _, mr = stats(f"  ip {fld} vs FLA", r1[fld], gold[gk])
            worst[fld] = max(worst.get(fld, 0.0), mr)
        _, mr = stats("  ip dh0 vs FLA", r1["dh0"], gold["h0"].transpose(-1, -2))
        worst["dh0"] = max(worst.get("dh0", 0.0), mr)
    print("\nworst-case max_rel (in-place chain):")
    for kk in sorted(worst):
        print(f"  {kk:16s}  {worst[kk]:.3e}")


def part_timing(cases, scale, lower_bound):
    from fla.ops.kda import chunk_kda
    print("\n" + "=" * 78)
    print("Part C: chain timing -- current (outside sum) vs in-place B3, dev + wall, + FLA")
    print("=" * 78)
    NS = 3
    print(f"  {'case':18s} | {'FLA dev':>8s} {'base dev':>8s} {'ip dev':>8s} {'ip/base':>8s}"
          f" | {'base wall':>9s} {'ip wall':>8s} {'wall x':>7s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fi, qn, kn, beta_act, g_fla, do, dfin = prep(B, T, H, scale, lower_bound, seed, gs)
        fi_b, do_b = _prep_b(fi, inp, do)
        ch0 = make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, accum=False)
        ch1 = make_chain(fi_b, inp, qn, kn, do_b, do, dfin, scale, NS, accum=True)
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

        ch0(); ch1(); fla_bwd()
        torch.cuda.synchronize()
        df, _ = profile_dev(fla_bwd)
        d0, p0 = profile_dev(ch0)
        d1, p1 = profile_dev(ch1)
        w0 = time_wall(ch0, reps=50)
        w1 = time_wall(ch1, reps=50)
        print(f"  {name:18s} | {df:8.1f} {d0:8.1f} {d1:8.1f} {d1/d0:7.2f}x | "
              f"{w0:9.1f} {w1:8.1f} {w1/w0:6.2f}x")
        # per-kernel split: show the elementwise adds gone
        split0 = {}
        for ev in p0.key_averages():
            split0[ev.key] = split0.get(ev.key, 0.0) + ev.self_device_time_total / 15
        split1 = {}
        for ev in p1.key_averages():
            split1[ev.key] = split1.get(ev.key, 0.0) + ev.self_device_time_total / 15
        add0 = sum(v for k, v in split0.items() if "elementwise" in k or "add" in k.lower())
        add1 = sum(v for k, v in split1.items() if "elementwise" in k or "add" in k.lower())
        print(f"      elementwise/add-attributed dev: base={add0:7.1f}us  ip={add1:7.1f}us")
        print(f"      total elementwise kernels: base={sum(ev.count for ev in p0.key_averages() if 'elementwise' in ev.key)}"
              f"  ip={sum(ev.count for ev in p1.key_averages() if 'elementwise' in ev.key)}")
    return


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}  "
          f"triton={triton.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    mode = os.environ.get("B3A_MODE", "")
    if mode in ("", "correctness"):
        part_a()
    if mode in ("", "chain"):
        part_chain(CASES[:int(os.environ.get("B3A_CASES", len(CASES)))], scale, lower_bound)
    if mode in ("", "timing"):
        part_timing(TCASES, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
