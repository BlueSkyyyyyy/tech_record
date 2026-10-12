"""P3-3: recurrent-state precision — does storing S in bf16 between chunks cost accuracy?

FlashKDA keeps the on-chip recurrent state ``S`` in **bf16** and only the *update arithmetic*
(old-state read -> fp32 * g_total + fp32 gemm delta -> round to bf16) is fp32 (deep-dive §3).
This harness isolates that decision with the pure-PyTorch chunked reference:

  * golden      : fp64 token-recurrent (``kda_token_recurrent``)
  * fp32        : chunked, everything fp32
  * state_bf16  : chunked, running ``S`` rounded to bf16 **between chunks**, update math fp32
  * state_bf16_arith : same, but the per-chunk multiply-add is also rounded to bf16
  * ops_bf16    : ``state_bf16`` + all gemm operands (kd/qd/ki/kr/U/INV/Mqk) bf16 == kernel-like

Metrics are max_abs / max_rel of output ``o`` and final state ``S`` vs the fp64 golden.

Usage (inside kernel_lab, absolute path):
    python ~/proj/tech_record/code/flash-kda/harness/check_state_bf16.py
"""

import math
import sys

import torch

sys.path.insert(0, "~/proj/tech_record/code/flash-kda/src")
from kda_ref import kda_token_recurrent, l2_normalize  # noqa: E402

LOG2E = 1.4426950408889634


def qb(x):
    """Round to bf16 and back, keeping the working dtype."""
    return x.to(torch.bfloat16).to(x.dtype)


def chunked_ablate(q, k, v, g, beta, scale=1.0, h0=None, chunk=16,
                   state_store_bf16=False, gemm_bf16=False, other_ops_bf16=False,
                   update_arith="fp32"):
    """Chunked KDA forward with switchable precision, mirroring ``kda_ref.kda_chunked_fwd``.

    ``state_store_bf16`` : round the *carried* recurrent state ``S`` to bf16 between chunks
                           (kernel behaviour); the update math itself stays fp32 unless
                           ``update_arith='bf16'``.
    ``gemm_bf16``        : the state read into the ``@ S`` gemms is bf16 (as an MMA operand).
    ``other_ops_bf16``   : every other matmul / inverse operand (kd/qd/ki/kr/U/INV/Mqk) is bf16.
    """
    T, D = q.shape
    dt = q.dtype
    S = torch.zeros(D, D, dtype=dt, device=q.device) if h0 is None else h0.to(dt).clone()
    out = torch.empty(T, D, dtype=dt, device=q.device)

    for t0 in range(0, T, chunk):
        C = min(chunk, T - t0)
        sl = slice(t0, t0 + C)
        qc, kc, vc, gc_, bc = q[sl], k[sl], v[sl], g[sl], beta[sl]
        if C < chunk:
            pad = chunk - C
            qc = torch.cat([qc, qc.new_zeros(pad, D)], 0)
            kc = torch.cat([kc, kc.new_zeros(pad, D)], 0)
            vc = torch.cat([vc, vc.new_zeros(pad, D)], 0)
            gc_ = torch.cat([gc_, gc_.new_zeros(pad, D)], 0)
            bc = torch.cat([bc, bc.new_zeros(pad)], 0)

        gcum = gc_.cumsum(dim=0)
        gtot = gcum[-1]
        E = torch.exp2(gcum)
        etot = torch.exp2(gtot)

        kd = kc * E
        qd = qc * E * scale
        ki = kc * torch.exp2(-gcum)
        kr = ki * etot[None, :]

        S_read = qb(S) if gemm_bf16 else S
        if other_ops_bf16:
            kd, qd, ki, kr, S_read = map(qb, (kd, qd, ki, kr, S_read))
        uprime = (vc - kd @ S_read) * bc[:, None]
        if other_ops_bf16:
            uprime = qb(uprime)

        L = torch.tril(kd @ ki.transpose(0, 1), diagonal=-1) * bc[:, None]
        Mqk = torch.tril(qd @ ki.transpose(0, 1))
        if other_ops_bf16:
            L, Mqk = qb(L), qb(Mqk)

        eye = torch.eye(chunk, dtype=dt, device=q.device)
        P = torch.linalg.solve(eye + L, eye)
        if other_ops_bf16:
            P = qb(P)

        U = P @ uprime
        o = qd @ S_read + Mqk @ U
        delta = kr.transpose(0, 1) @ U

        if state_store_bf16:
            if update_arith == "bf16":
                S = qb(qb(qb(S) * etot[:, None]) + qb(delta))
            else:
                S = qb(qb(S) * etot[:, None] + delta)
        else:
            S = etot[:, None] * S + delta

        out[sl] = o[:C]
    return out, S


def stats(label, gold, pred):
    d = (gold.float() - pred.float()).abs()
    ref = gold.float().abs()
    max_abs = d.max().item()
    max_rel = (d.max() / (ref.max() + 1e-12)).item()
    mean_abs = d.mean().item()
    print(f"  {label:22s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  mean_abs={mean_abs:.3e}")
    return max_abs, max_rel, mean_abs


def run_case(T, g_scale, D=128, seed=0, h0_random=True, work_dtype=torch.float32):
    torch.manual_seed(seed)
    q = torch.randn(T, D, dtype=torch.float64)
    k = torch.randn(T, D, dtype=torch.float64)
    v = torch.randn(T, D, dtype=torch.float64)
    g = -torch.rand(T, D, dtype=torch.float64) * g_scale
    beta = torch.rand(T, dtype=torch.float64)
    h0 = torch.randn(D, D, dtype=torch.float64) if h0_random else None
    scale = 1.0 / math.sqrt(D)

    qn64 = l2_normalize(q)
    kn64 = l2_normalize(k)

    # fp64 golden token recurrence
    o_gold, s_gold = kda_token_recurrent(qn64, kn64, v, g, beta, scale, h0)

    # working-precision inputs (bf16 rounding is inserted explicitly in the ablation)
    qn = qn64.to(work_dtype)
    kn = kn64.to(work_dtype)
    v = v.to(work_dtype)
    g = g.to(work_dtype)
    beta = beta.to(work_dtype)
    h0w = None if h0 is None else h0.to(work_dtype)

    cfg = {
        "fp32":               dict(),
        "store_bf16":         dict(state_store_bf16=True),
        "gemm_bf16(st_fp32)": dict(gemm_bf16=True, other_ops_bf16=True),
        "kernel_bf16":        dict(state_store_bf16=True, gemm_bf16=True, other_ops_bf16=True),
        "kernel_arith_bf16":  dict(state_store_bf16=True, gemm_bf16=True, other_ops_bf16=True,
                                   update_arith="bf16"),
    }
    print(f"\n{'='*78}\n[case T={T} D={D} g={g_scale} h0={'rand' if h0_random else 'zero'} "
          f"work={str(work_dtype).split('.')[-1]}]"
          f"  |o|max={o_gold.abs().max():.3f} |S|max={s_gold.abs().max():.3f}")
    rows = {}
    print(" output o (chunked dtype variants vs fp64 token golden):")
    for name, kw in cfg.items():
        o, s = chunked_ablate(qn, kn, v, g, beta, scale, h0w, **kw)
        rows[name] = (o, s)
        stats(name, o_gold, o)

    print(" final state S (chunked dtype variants vs fp64 token golden):")
    for name in cfg:
        o, s = rows[name]
        stats(name, s_gold, s)

    # key claim of deep-dive §3: bf16 *storage* adds nothing once operands are bf16
    print(" §3 claim: bf16 state storage vs fp32 storage (both with bf16 gemms):")
    stats("kernel_bf16 - st_fp32", rows["gemm_bf16(st_fp32)"][0], rows["kernel_bf16"][0])
    stats("kernel_bf16 - st_fp32", rows["gemm_bf16(st_fp32)"][1], rows["kernel_bf16"][1])
    print(" fp32-FMA vs bf16-FMA update arithmetic (both kernel-bf16 storage):")
    stats("arith_bf16 - fp32FMA", rows["kernel_bf16"][0], rows["kernel_arith_bf16"][0])
    stats("arith_bf16 - fp32FMA", rows["kernel_bf16"][1], rows["kernel_arith_bf16"][1])


def main():
    print("### P3-3 bf16 recurrent-state precision ablation (CPU, PyTorch)")
    for T in [16, 64, 256, 1024]:
        run_case(T, g_scale=1.0)
    for T in [256, 1024]:
        run_case(T, g_scale=4.0)
        run_case(T, g_scale=0.0, h0_random=False)
    print("\nDONE")


if __name__ == "__main__":
    main()
