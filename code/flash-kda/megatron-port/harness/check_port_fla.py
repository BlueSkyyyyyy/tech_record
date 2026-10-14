"""megatron-port M4: anchor the port's PyTorch reference against the actual FLA
``chunk_kda`` Triton kernels (forward + backward) on GPU.

Why this exists
---------------
``src/kda_ref.py`` is the golden we will diff the Megatron port against, but it has
only ever been checked against *itself* (token-recurrence vs chunked, fp64 CPU).  Before
we trust it as a porting oracle it must agree with the operators we are actually porting:
FLA ``fla/ops/kda`` chunk kernels.  This harness feeds both sides the *same* bf16
operands and reports the residual, which is the FLA bf16-kernel error (the reference runs
in fp64).

Two parts, so the two possible contract mistakes are separated:

* **Part A -- explicit operands** (``use_gate_in_kernel=False`` etc.): both sides get the
  exact same bf16 ``q/k/v/g/beta`` (``g`` = natural-log activated decay, ``beta`` =
  post-sigmoid) that the port's ``kda_gate``/``kda_l2norm`` produced, so the residual is
  purely the **chunk algebra** (L, (I+L)^-1, U/W, state recurrence, o).
* **Part B -- fused gate in kernel** (``use_gate_in_kernel=True``,
  ``use_beta_sigmoid_in_kernel=True``): FLA computes ``-exp(A_log)*softplus(g+dt_bias)``
  and ``sigmoid(beta)`` internally from raw logits; the port does it in
  ``kda_gate``/``torch.sigmoid``.  The residual here is the **gate activation contract**
  (softplus vs the ``safe_gate`` clamp, dt_bias layout, log-space units).

Run (inside kernel_lab, GPU):
    FLA_DISABLE_BACKEND_DISPATCH=1 python3 \
        ~/proj/tech_record/code/flash-kda/megatron-port/harness/check_port_fla.py
"""

import math
import os
import sys

import torch

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)
from kda_ref import kda_chunked_fwd, kda_gate, kda_l2norm  # noqa: E402

LN2 = math.log(2.0)


def stats(gold, pred):
    g = gold.float()
    p = pred.float()
    d = (g - p).abs()
    return d.max().item(), (d.max() / (g.abs().max() + 1e-12)).item()


def make_inputs(B, T, H, K, V, seed, device):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float64)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float64)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float64)
    g_logit = torch.randn(B, T, H, K, device=device, dtype=torch.float64)
    # KDA-realistic gate params: A_log ~ log U(1,16) (Megatron A_init_range), small dt_bias.
    A_log = torch.log(torch.empty(H, device=device, dtype=torch.float64).uniform_(1.0, 16.0))
    dt_bias = 0.1 * torch.randn(H * K, device=device, dtype=torch.float64)
    beta_logit = torch.randn(B, T, H, device=device, dtype=torch.float64)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float64)
    return q, k, v, g_logit, A_log, dt_bias, beta_logit, h0


def part_a(device, cases):
    from fla.ops.kda import chunk_kda

    print("\n" + "=" * 86)
    print("[Part A] explicit operands  (use_gate_in_kernel=False, same bf16 q/k/v/g/beta)")
    print("          port fp64 chunked  vs  FLA bf16 chunk_kda  (fwd o/S + autograd grads)")
    print("=" * 86)
    rows = []
    for (B, T, H, K, V, seed) in cases:
        q, k, v, g_logit, A_log, dt_bias, beta_logit, h0 = make_inputs(
            B, T, H, K, V, seed, device)
        qn = kda_l2norm(q).to(torch.bfloat16)
        kn = kda_l2norm(k).to(torch.bfloat16)
        vb = v.to(torch.bfloat16)
        gb = kda_gate(g_logit, A_log, dt_bias).to(torch.bfloat16)          # natural log
        bb = torch.sigmoid(beta_logit).to(torch.bfloat16)
        scale = K ** -0.5

        dout = (torch.randn(B, T, H, V, device=device) * 0.1).to(torch.bfloat16)
        dfin = (torch.randn(B, H, K, V, device=device) * 0.01).float()

        # ---- FLA side (autograd through the Triton kernels) ----
        qa = qn.clone().requires_grad_(True)
        ka = kn.clone().requires_grad_(True)
        va = vb.clone().requires_grad_(True)
        ga = gb.clone().requires_grad_(True)
        ba = bb.clone().requires_grad_(True)
        ha = h0.float().clone().requires_grad_(True)
        o_f, s_f = chunk_kda(qa, ka, va, ga, ba, scale=scale, initial_state=ha,
                             output_final_state=True, use_qk_l2norm_in_kernel=False,
                             use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
        (o_f.float() * dout.float()).sum().add((s_f * dfin).sum()).backward()

        # ---- port side (fp64 autograd; exact) ----
        qr = qn.double().requires_grad_(True)
        kr = kn.double().requires_grad_(True)
        vr = vb.double().requires_grad_(True)
        gr = gb.double().requires_grad_(True)
        br = bb.double().requires_grad_(True)
        hr = h0.double().requires_grad_(True)
        o_r, s_r = kda_chunked_fwd(qr, kr, vr, gr, br, scale=scale, initial_state=hr,
                                   output_final_state=True, chunk_size=64)
        (o_r * dout.double()).sum().add((s_r * dfin.double()).sum()).backward()

        print(f"\n  B={B} T={T} H={H} K={K} V={V} |o|max={o_f.float().abs().max().item():.3f} "
              f"|s|max={s_f.abs().max().item():.3f}")
        om, orr = stats(o_r, o_f)
        sm, srr = stats(s_r, s_f)
        print(f"    fwd  o: max_abs={om:.3e} max_rel={orr:.3e}    "
              f"S: max_abs={sm:.3e} max_rel={srr:.3e}")
        grads = [("dq", qr, qa), ("dk", kr, ka), ("dv", vr, va),
                 ("dg", gr, ga), ("dbeta", br, ba), ("dh0", hr, ha)]
        for nm, gr_ref, gr_fla in grads:
            gm, grr = stats(gr_ref.grad, gr_fla.grad)
            print(f"    bwd  {nm:>5}: max_abs={gm:.3e} max_rel={grr:.3e}")
            rows.append((nm, gm, grr))
    return rows


def part_b(device, cases):
    from fla.ops.kda import chunk_kda

    print("\n" + "=" * 86)
    print("[Part B] fused gate in kernel  (use_gate_in_kernel=True, use_beta_sigmoid=True)")
    print("          port kda_gate+sigmoid+kda_chunked_fwd  vs  FLA chunk_kda")
    print("=" * 86)
    rows = []
    for (B, T, H, K, V, seed) in cases:
        q, k, v, g_logit, A_log, dt_bias, beta_logit, h0 = make_inputs(
            B, T, H, K, V, seed, device)
        qn = kda_l2norm(q).to(torch.bfloat16)
        kn = kda_l2norm(k).to(torch.bfloat16)
        vb = v.to(torch.bfloat16)
        scale = K ** -0.5

        dout = (torch.randn(B, T, H, V, device=device) * 0.1).to(torch.bfloat16)
        dfin = (torch.randn(B, H, K, V, device=device) * 0.01).float()

        # ---- FLA side: raw logits in, gate fused ----
        qa = qn.clone().requires_grad_(True)
        ka = kn.clone().requires_grad_(True)
        va = vb.clone().requires_grad_(True)
        gra = g_logit.to(torch.bfloat16).clone().requires_grad_(True)
        bra = beta_logit.to(torch.bfloat16).clone().requires_grad_(True)
        ha = h0.float().clone().requires_grad_(True)
        A_f = A_log.float().clone().requires_grad_(True)
        dt_f = dt_bias.float().clone().requires_grad_(True)
        o_f, s_f = chunk_kda(qa, ka, va, gra, bra, scale=scale, initial_state=ha,
                             output_final_state=True, use_qk_l2norm_in_kernel=False,
                             use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                             A_log=A_f, dt_bias=dt_f)
        (o_f.float() * dout.float()).sum().add((s_f * dfin).sum()).backward()

        # ---- port side: same raw logits through kda_gate + sigmoid, fp64 ----
        qr = qn.double().requires_grad_(True)
        kr = kn.double().requires_grad_(True)
        vr = vb.double().requires_grad_(True)
        gr = g_logit.double().requires_grad_(True)
        br = beta_logit.double().requires_grad_(True)
        hr = h0.double().requires_grad_(True)
        Ar = A_log.double().requires_grad_(True)
        dtr = dt_bias.double().requires_grad_(True)
        g_act = kda_gate(gr, Ar, dtr)
        beta_act = torch.sigmoid(br)
        o_r, s_r = kda_chunked_fwd(qr, kr, vr, g_act, beta_act, scale=scale,
                                   initial_state=hr, output_final_state=True, chunk_size=64)
        (o_r * dout.double()).sum().add((s_r * dfin.double()).sum()).backward()

        print(f"\n  B={B} T={T} H={H} K={K} V={V}")
        om, orr = stats(o_r, o_f)
        sm, srr = stats(s_r, s_f)
        print(f"    fwd  o: max_abs={om:.3e} max_rel={orr:.3e}    "
              f"S: max_abs={sm:.3e} max_rel={srr:.3e}")
        gset = [("dq", qr, qa), ("dk", kr, ka), ("dv", vr, va),
                ("dg_logit", gr, gra), ("dbeta_logit", br, bra), ("dh0", hr, ha)]
        for nm, gr_ref, gr_fla in gset:
            if gr_fla.grad is None:
                print(f"    bwd  {nm:>11}: FLA gave no grad")
                continue
            gm, grr = stats(gr_ref.grad, gr_fla.grad)
            print(f"    bwd  {nm:>11}: max_abs={gm:.3e} max_rel={grr:.3e}")
            rows.append((nm, gm, grr))
        for nm, gr_ref, gr_fla in [("dA_log", Ar, A_f), ("ddt_bias", dtr, dt_f)]:
            if gr_fla.grad is None:
                print(f"    bwd  {nm:>11}: FLA gave no grad")
                continue
            gm, grr = stats(gr_ref.grad, gr_fla.grad)
            print(f"    bwd  {nm:>11}: max_abs={gm:.3e} max_rel={grr:.3e}")
    return rows


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    print("=" * 86)
    print("megatron-port M4: anchor src/kda_ref.py against FLA chunk_kda (GPU)")
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
    print("=" * 86)

    cases = [
        (1, 128, 2, 128, 128, 1),
        (2, 512, 4, 128, 128, 2),
        (1, 250, 3, 128, 128, 3),   # non-divisible T (chunk 64 -> pad)
        (2, 256, 2, 64, 96, 4),     # K != V
    ]
    part_a(device, cases)
    part_b(device, cases)

    print("\n" + "=" * 86)
    print("Done. (Part A = chunk algebra; Part B = gate activation contract.  Residual is the")
    print("FLA bf16-kernel error because the port reference runs in fp64.)")
    print("=" * 86)


if __name__ == "__main__":
    main()
