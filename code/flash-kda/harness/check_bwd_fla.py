"""P5-3: cross-check the explicit per-token adjoint (``src/kda_bwd_ref.py``) against
the backward of FLA ``chunk_kda`` (``torch.autograd`` through the Triton chunked
forward) on realistic ``D=128`` multi-head BF16 inputs.

Setup (so both sides act on *identical* numerics; no gate/l2norm ambiguity):

* ``use_qk_l2norm_in_kernel=False`` -- we pre-normalise q/k and pass the normalised
  tensors to FLA; the reference uses the very same (BF16-rounded) values.
* ``use_gate_in_kernel=False`` -- we pass the pre-computed natural-log decay
  ``g_fla = g_log2 * ln(2)`` (FLA state decay ``= exp(g_fla) = 2**g_log2``), so the
  reference's log2 decay agrees exactly.  The gate-activation chain
  (``g_logits/A_log/dt_bias``) is validated separately in P5-2.
* ``use_beta_sigmoid_in_kernel=False`` -- we pass ``sigmoid(beta_logits)`` directly.
* ``initial_state`` default layout ``[N, H, K, V]`` == the reference's ``S``.

Loss ``L = <o, dout> + <S_T, dfinal_state>``; FLA gives ``q/k/v/g/beta/h0`` grads
via autograd, the reference gives them by the hand-derived chain rule (run in
fp64, so the reported error is the *BF16 kernel* backward error, not ours).

Usage (inside kernel_lab, GPU):
    python /ssd/.../code/flash-kda/harness/check_bwd_fla.py
"""

import os
import math
import sys

import torch

SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, SRC)
from kda_bwd_ref import kda_token_backward  # noqa: E402
from kda_ref import (kda_token_recurrent, l2_normalize,  # noqa: E402
                     kda_gate, kda_beta)

LN2 = math.log(2.0)
LOG2E = 1.4426950408889634


def stats(gold, pred):
    d = (gold.float() - pred.float()).abs()
    ref = gold.float().abs()
    return d.max().item(), (d.max() / (ref.max() + 1e-12)).item(), d.mean().item()


def run_case(name, raw, scale, lower_bound):
    from fla.ops.kda import chunk_kda

    q_raw, k_raw, v_raw, beta_raw, g_logit, A_log, dt_bias, h0 = raw
    B, T, H, D = q_raw.shape
    dev = q_raw.device

    # --- derive the exact per-head decay-level operands (fp32) ---
    qn = l2_normalize(q_raw.float())                       # [B,T,H,D]
    kn = l2_normalize(k_raw.float())
    beta_act = kda_beta(beta_raw.float())                  # [B,T,H]
    g_dec = kda_gate(g_logit.float(), A_log.float(), dt_bias.float(), lower_bound)  # log2 <=0
    g_fla = g_dec * LN2                                    # natural-log decay for FLA

    # BF16 tensors actually fed to FLA (the reference reuses these exact values)
    qb = qn.to(torch.bfloat16)
    kb = kn.to(torch.bfloat16)
    vb = v_raw.float().to(torch.bfloat16)
    bb = beta_act.to(torch.bfloat16)
    gb = g_fla.to(torch.bfloat16)
    hb = h0.float()

    dout = (torch.randn(B, T, H, D, device=dev) * 0.1).to(torch.bfloat16)
    dfin = (torch.randn(B, H, D, D, device=dev) * 0.01).to(torch.float32)

    qa = qb.clone().requires_grad_(True)
    ka = kb.clone().requires_grad_(True)
    va = vb.clone().requires_grad_(True)
    ba = bb.clone().requires_grad_(True)
    ga = gb.clone().requires_grad_(True)
    ha = hb.clone().requires_grad_(True)

    o, s = chunk_kda(qa, ka, va, ga, ba, scale=scale, initial_state=ha,
                     output_final_state=True, use_qk_l2norm_in_kernel=False,
                     use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
    loss = (o * dout).sum() + (s * dfin).sum()
    loss.backward()

    print(f"\n{'='*80}\nCase [{name}]  B={B} T={T} H={H} D={D}  "
          f"|o|max={o.float().abs().max().item():.3f}  |s|max={s.abs().max().item():.3f}")

    # fp64 reference forward (same BF16 inputs) -- sanity that setup matches FLA.
    fwd_rows, bwd_rows = [], []
    for b in range(B):
        for h in range(H):
            qh = qb[b, :, h].double()
            kh = kb[b, :, h].double()
            vh = vb[b, :, h].double()
            gh = g_dec[b, :, h].double()
            bh = bb[b, :, h].double()
            o_rec, s_rec = kda_token_recurrent(qh, kh, vh, gh, bh, scale, hb[b, h].double())
            fwd_rows.append((stats(o[b, :, h], o_rec), stats(s[b, h], s_rec)))

            mine64 = kda_token_backward(qh, kh, vh, gh, bh, scale,
                                        dout[b, :, h].double(), hb[b, h].double(),
                                        dfin[b, h].double())
            mine32 = kda_token_backward(qb[b, :, h].float(), kb[b, :, h].float(),
                                        vb[b, :, h].float(), g_dec[b, :, h].float(),
                                        bb[b, :, h].float(), scale,
                                        dout[b, :, h].float(), hb[b, h].float(),
                                        dfin[b, h].float())
            # g_fla = ln2 * g_log2  =>  dL/dg_log2 = (dL/dg_fla) * ln2.
            # (Getting this backwards inflates dg by 1/ln2^2 ~= 2.08x, i.e. max_rel ~0.52.)
            fl = dict(dq=qa.grad[b, :, h].float(), dk=ka.grad[b, :, h].float(),
                      dv=va.grad[b, :, h].float(), dbeta=ba.grad[b, :, h].float(),
                      dg=ga.grad[b, :, h].float() * LN2, dh0=ha.grad[b, h].float())
            for fld in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
                bwd_rows.append((fld, stats(fl[fld], mine64[fld])[:2],
                                 stats(fl[fld], mine32[fld])[:2]))

    print("  forward sanity (FLA bf16 vs fp64 token-recurrent on same bf16 inputs):")
    o_ma = max(r[0][0] for r in fwd_rows)
    o_mr = max(r[0][1] for r in fwd_rows)
    s_ma = max(r[1][0] for r in fwd_rows)
    s_mr = max(r[1][1] for r in fwd_rows)
    print(f"    o  max_abs={o_ma:.3e} max_rel={o_mr:.3e}")
    print(f"    S  max_abs={s_ma:.3e} max_rel={s_mr:.3e}")

    print("  backward: FLA autograd vs explicit adjoint "
          "(max_abs / max_rel);  [vs fp64 ref]  [vs fp32 ref]")
    agg = {}
    for fld in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
        rows = [r for r in bwd_rows if r[0] == fld]
        m64 = (max(r[1][0] for r in rows), max(r[1][1] for r in rows))
        m32 = (max(r[2][0] for r in rows), max(r[2][1] for r in rows))
        agg[fld] = (m64, m32)
        print(f"    {fld:6s} f64 {m64[0]:.3e}/{m64[1]:.3e}    "
              f"f32 {m32[0]:.3e}/{m32[1]:.3e}")
    return agg


def main():
    torch.manual_seed(0)
    dev = "cuda"
    lower_bound = -5.0
    D = 128
    scale = 1.0 / math.sqrt(D)

    cases = [
        ("T=128 H=2 g=N(0,1)", 1, 128, 2, 1.0),
        ("T=512 H=4 g=N(0,2)", 1, 512, 4, 2.0),
        ("T=1024 H=2 base", 1, 1024, 2, 2.0),
        ("T=1023 H=8 g=N(0,1)", 1, 1023, 8, 1.0),
        ("T=512 g=0", 1, 512, 2, 0.0),
    ]
    overall = {}
    for (name, B, T, H, gs) in cases:
        q_raw = torch.randn(B, T, H, D, device=dev) * 0.5
        k_raw = torch.randn(B, T, H, D, device=dev) * 0.5
        v_raw = torch.randn(B, T, H, D, device=dev)
        beta_raw = torch.randn(B, T, H, device=dev)
        g_logit = torch.randn(B, T, H, D, device=dev) * gs
        A_log = torch.randn(H, device=dev)
        dt_bias = torch.randn(H, D, device=dev)
        h0 = torch.randn(B, H, D, D, device=dev) * 0.5
        agg = run_case(name, (q_raw, k_raw, v_raw, beta_raw, g_logit, A_log, dt_bias, h0),
                       scale, lower_bound)
        for fld, (m64, m32) in agg.items():
            cur = overall.setdefault(fld, [0.0, 0.0])
            cur[0] = max(cur[0], m64[1])
            cur[1] = max(cur[1], m32[1])

    print(f"\n{'='*80}\nworst-case max_rel across cases (FLA vs fp64 ref / FLA vs fp32 ref):")
    for fld in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
        print(f"  {fld:6s} {overall[fld][0]:.3e}   {overall[fld][1]:.3e}")
    print("\nDONE")


if __name__ == "__main__":
    main()
