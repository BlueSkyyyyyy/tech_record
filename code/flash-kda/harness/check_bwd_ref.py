"""P5-2: validate ``src/kda_bwd_ref.py`` (hand-derived adjoint) vs ``torch.autograd``.

The forward is ``kda_ref.kda_token_recurrent`` (per-token gated delta rule, single
head, ``[T, D]``). We drive ``torch.autograd`` through it and compare against the
explicit chain-rule backward ``kda_bwd_ref.kda_token_backward`` for every input
gradient (``dq, dk, dv, dg, dbeta``) plus ``dh0`` (with a random upstream grad on
the final state as well). A central finite-difference spot check on a few entries
is included so that autograd itself is not the sole oracle.

Usage (inside kernel_lab, CPU is enough):
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_ref.py
"""

import os
import math
import sys

import torch

SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, SRC)
from kda_bwd_ref import kda_gate_backward, kda_token_backward  # noqa: E402
from kda_ref import kda_gate, kda_token_recurrent, l2_normalize  # noqa: E402


def stats(a, b):
    d = (a.float() - b.float()).abs()
    ref = b.float().abs()
    return d.max().item(), (d.max() / (ref.max() + 1e-12)).item(), d.mean().item()


def run_case(T, D, g_scale, seed, dtype):
    torch.manual_seed(seed)
    q = l2_normalize(torch.randn(T, D)).to(dtype)
    k = l2_normalize(torch.randn(T, D)).to(dtype)
    v = torch.randn(T, D).to(dtype)
    g = (-torch.rand(T, D) * g_scale).to(dtype)
    beta = torch.rand(T).to(dtype)
    h0 = torch.randn(D, D).to(dtype)
    scale = 1.0 / math.sqrt(D)
    dout = torch.randn(T, D).to(dtype)
    dfin = torch.randn(D, D).to(dtype)

    qa = q.clone().requires_grad_(True)
    ka = k.clone().requires_grad_(True)
    va = v.clone().requires_grad_(True)
    ga = g.clone().requires_grad_(True)
    ba = beta.clone().requires_grad_(True)
    ha = h0.clone().requires_grad_(True)
    o, Sf = kda_token_recurrent(qa, ka, va, ga, ba, scale, ha)
    (o * dout).sum().add((Sf * dfin).sum()).backward()

    mine = kda_token_backward(q, k, v, g, beta, scale, dout, h0, dfin)

    rows = {}
    for name, mn, au in [("dq", mine["dq"], qa.grad), ("dk", mine["dk"], ka.grad),
                         ("dv", mine["dv"], va.grad), ("dg", mine["dg"], ga.grad),
                         ("dbeta", mine["dbeta"], ba.grad), ("dh0", mine["dh0"], ha.grad)]:
        rows[name] = stats(mn, au)
    return rows


def finite_diff_check(D=8, T=6, seed=123):
    """Central diff on a few entries of each tensor vs the explicit backward."""
    torch.manual_seed(seed)
    dtype = torch.float64
    q = l2_normalize(torch.randn(T, D)).to(dtype)
    k = l2_normalize(torch.randn(T, D)).to(dtype)
    v = torch.randn(T, D).to(dtype)
    g = (-torch.rand(T, D) * 2.0).to(dtype)
    beta = torch.sigmoid(torch.randn(T)).to(dtype)
    h0 = torch.randn(D, D).to(dtype)
    scale = 1.0 / math.sqrt(D)
    dout = torch.randn(T, D).to(dtype)
    dfin = torch.randn(D, D).to(dtype)

    def loss_of(qq, kk, vv, gg, bb, hh):
        o, Sf = kda_token_recurrent(qq, kk, vv, gg, bb, scale, hh)
        return (o * dout).sum() + (Sf * dfin).sum()

    mine = kda_token_backward(q, k, v, g, beta, scale, dout, h0, dfin)
    eps = 1e-6
    pi_of = {"dq": 0, "dk": 1, "dv": 2, "dg": 3, "dbeta": 4, "dh0": 5}
    picks = [("dq", (0, 1)), ("dk", (2, 3)), ("dv", (4, 0)),
             ("dg", (1, 2)), ("dbeta", (3,)), ("dh0", (0, 5))]
    print("  finite-difference spot check (fp64, central diff eps=1e-6):")
    for name, idx in picks:
        args = [q, k, v, g, beta, h0]
        pi = pi_of[name]
        plus = [t.clone() for t in args]
        minus = [t.clone() for t in args]
        plus[pi][idx] += eps
        minus[pi][idx] -= eps
        fd = (loss_of(*plus) - loss_of(*minus)) / (2 * eps)
        ana = mine[name][idx]
        print(f"    {name:6s} idx={str(idx):10s} fd={fd:+.8f}  adjoint={ana:+.8f}  "
              f"absdiff={abs(fd - ana):.2e}")


def gate_chain_check():
    """Chain rule through ``kda_gate`` -> raw logits / A_log / dt_bias."""
    print("  gate-activation chain check (kda_gate -> g_logits, A_log, dt_bias):")
    torch.manual_seed(7)
    for (T, H, D, lb) in [(8, 2, 16, -5.0), (64, 4, 64, -3.0), (200, 3, 128, -5.0)]:
        gl = torch.randn(T, H, D, dtype=torch.float64)
        al = torch.randn(H, dtype=torch.float64)
        db = torch.randn(H, D, dtype=torch.float64)
        dg = torch.randn(T, H, D, dtype=torch.float64)
        gla, ala, dba = gl.clone().requires_grad_(True), al.clone().requires_grad_(True), db.clone().requires_grad_(True)
        (kda_gate(gla, ala, dba, lb) * dg).sum().backward()
        mg, ma, md = kda_gate_backward(dg, gl, al, db, lb)
        for nm, mn, au in [("g_logits", mg, gla.grad), ("A_log", ma, ala.grad), ("dt_bias", md, dba.grad)]:
            ma_, mr_, _ = stats(mn, au)
            print(f"    T={T} H={H} D={D} lb={lb}  d{nm:9s} max_abs={ma_:.2e} max_rel={mr_:.2e}")


def main():
    print("=" * 78)
    print("P5-2  explicit adjoint  vs  torch.autograd  (kda_token_recurrent)")
    print("      fields: dq dk dv dg dbeta dh0;  metric max_abs / max_rel / mean_abs")
    print("=" * 78)

    cases = []
    for T in [1, 16, 17, 64, 128, 512]:
        cases.append((T, 16, 2.0, 0, torch.float64))
    cases.append((48, 32, 2.0, 3, torch.float64))
    for gs in [0.0, 1.0, 4.0]:
        cases.append((256, 64, gs, 1, torch.float64))
    for seed in [2, 5, 9]:
        cases.append((200, 64, 2.0, seed, torch.float64))
    for T in [16, 64, 256]:
        cases.append((T, 64, 2.0, 0, torch.float32))

    hdr = "  T     D  g    seed dtype        " + "".join(f"{n:>26s}" for n in
          ["dq", "dk", "dv", "dg", "dbeta", "dh0"])
    print(hdr)
    agg64 = {n: [0.0, 0.0] for n in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]}
    agg32 = {n: [0.0, 0.0] for n in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]}
    for (T, D, gs, seed, dtype) in cases:
        rows = run_case(T, D, gs, seed, dtype)
        tag = "fp64" if dtype == torch.float64 else "fp32"
        line = f"  {T:<5d} {D:<3d} {gs:<4.1f} {seed:<4d} {tag:11s} "
        for n in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
            ma, mr, _ = rows[n]
            line += f"{ma:>10.2e}/{mr:<9.2e}  "
            agg = agg64 if dtype == torch.float64 else agg32
            agg[n][0] = max(agg[n][0], ma)
            agg[n][1] = max(agg[n][1], mr)
        print(line)

    print("\n  worst-case across cases (max_abs / max_rel):")
    for tag, agg in [("fp64", agg64), ("fp32", agg32)]:
        line = f"    {tag}: "
        for n in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
            line += f"{n}={agg[n][0]:.2e}/{agg[n][1]:.2e}  "
        print(line)

    print()
    finite_diff_check()
    print()
    gate_chain_check()
    print("\nDONE")


if __name__ == "__main__":
    main()
