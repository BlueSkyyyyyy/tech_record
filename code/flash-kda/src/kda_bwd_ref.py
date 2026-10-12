"""KDA backward reference: explicit per-token adjoint of the gated delta rule.

Companion to :mod:`kda_ref`. The forward (``kda_ref.kda_token_recurrent``) is, per
token ``t`` (state ``S in R^{K x V}``, ``alpha = 2**g_t``, ``beta`` already
sigmoid-activated):

    pred_t   = (k_t * alpha_t) @ S_{t-1}          # [V]
    delta_t  = beta_t * (v_t - pred_t)            # [V]
    S_t      = diag(alpha_t) S_{t-1} + k_t delta_t^T
    o_t      = scale * q_t @ S_t                  # [V]

:func:`kda_token_backward` re-runs the forward to cache ``S_{t-1}`` / ``S_t`` and
then walks ``t = T-1 .. 0`` accumulating the exact adjoints
(``dq, dk, dv, dg, dbeta`` and ``dh0``). No autograd is used: every line is the
hand-derived chain rule, so agreement with ``torch.autograd`` validates the
derivation itself (P5-2), not just the arithmetic.

Run ``python kda_bwd_ref.py`` to cross-check against ``torch.autograd`` on CPU.
"""

import math

import torch

LOG2E = 1.4426950408889634


def kda_token_backward(q, k, v, g, beta, scale, dout, h0=None, dfinal_state=None):
    """Exact backward of the per-token gated delta rule.

    All tensors are ``[T, D]`` (single head) except ``beta`` ``[T]``, ``h0`` and
    ``dfinal_state`` ``[D, D]``. ``dout`` is the upstream gradient of ``o`` and
    ``dfinal_state`` the upstream gradient of the returned final state ``S_T``
    (both optional but ``dout`` is required). Returns a dict with
    ``dq, dk, dv, dg, dbeta, dh0``.
    """
    T, D = q.shape
    dt = q.dtype if q.dtype.is_floating_point else torch.float32
    q = q.to(dt)
    k = k.to(dt)
    v = v.to(dt)
    g = g.to(dt)
    beta = beta.to(dt)
    dout = dout.to(dt)

    S_in = torch.empty(T, D, D, dtype=dt, device=q.device)   # S_{t-1}
    S_out = torch.empty(T, D, D, dtype=dt, device=q.device)  # S_t
    pred = torch.empty(T, D, dtype=dt, device=q.device)
    delta = torch.empty(T, D, dtype=dt, device=q.device)
    alpha = torch.empty(T, D, dtype=dt, device=q.device)

    S = torch.zeros(D, D, dtype=dt, device=q.device) if h0 is None else h0.to(dt).clone()
    for t in range(T):
        a = torch.exp2(g[t])
        p = (k[t] * a) @ S
        d = beta[t] * (v[t] - p)
        S_in[t] = S
        S = a[:, None] * S + k[t][:, None] * d[None, :]
        S_out[t] = S
        pred[t] = p
        delta[t] = d
        alpha[t] = a

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dg = torch.zeros_like(g)
    dbeta = torch.zeros(T, dtype=dt, device=q.device)
    ln2 = math.log(2.0)

    dS = torch.zeros(D, D, dtype=dt, device=q.device)
    if dfinal_state is not None:
        dS = dS + dfinal_state.to(dt)

    for t in range(T - 1, -1, -1):
        a = alpha[t]
        # adjoint of S_t: recurrence inflow + direct output path o_t = scale q_t S_t
        G = dS + scale * (q[t][:, None] @ dout[t][None, :])

        # dq_t = scale * S_t @ dO_t   (S_t is [K,V])
        dq[t] = scale * (S_out[t] @ dout[t])

        # S_t = diag(a) S_{t-1} + k_t delta_t^T
        dS = a[:, None] * G
        dk_t = G @ delta[t]             # [K]
        g_delta = G.transpose(0, 1) @ k[t]  # [V]

        # delta_t = beta_t (v_t - pred_t)
        dbeta[t] = (g_delta * (v[t] - pred[t])).sum()
        dres = beta[t] * g_delta        # [V], grad wrt (v_t - pred_t)
        dv[t] = dres
        dpred = -dres                   # [V]

        # pred_t = (a * k_t) @ S_{t-1}
        dS = dS + (a * k[t])[:, None] @ dpred[None, :]
        g_kalpha = dpred @ S_in[t].transpose(0, 1)  # [K]
        dk_t = dk_t + a * g_kalpha
        dalpha = k[t] * g_kalpha
        # diag(a) S_{t-1} path: dL/da_j += sum_v G[j,v] S_{t-1}[j,v]
        dalpha = dalpha + (G * S_in[t]).sum(dim=1)

        dk[t] = dk_t
        # a = 2**g  =>  da/dg = ln2 * a
        dg[t] = dalpha * (ln2 * a)

    dh0 = dS
    return dict(dq=dq, dk=dk, dv=dv, dg=dg, dbeta=dbeta, dh0=dh0)


def kda_gate_backward(dg_decay, g_logits, A_log, dt_bias=None, lower_bound=-5.0,
                      LOG2E=LOG2E):
    """Chain the decay-level adjoint ``dg_decay`` back to the raw gate logits.

    Forward (``kda_ref.kda_gate``): ``C = lower_bound * log2(e)``, ``a = exp(A_log)``
    (per head), ``u = a * (g_logits + dt_bias)``, ``g_decay = C * sigmoid(u)``.

    Returns ``(dg_logits, dA_log, ddt_bias)`` with the same shapes as the inputs.
    Inputs may be batched ``[T, H, D]`` (or ``[H, D]`` for ``dt_bias`` / ``[H]`` for
    ``A_log``); ``g_logits`` drives the batch shape.
    """
    H, D = A_log.shape[0], (dt_bias.shape[-1] if dt_bias is not None else g_logits.shape[-1])
    g = g_logits.to(torch.float32)
    gs = g + dt_bias.view(H, D) if dt_bias is not None else g
    a = A_log.view(H).exp()                      # [H]
    u = a.view(1, H, 1) * gs.view(-1, H, D)      # [N, H, D]
    s = torch.sigmoid(u)
    C = lower_bound * LOG2E
    du = dg_decay.to(torch.float32).view(-1, H, D) * (C * s * (1.0 - s))
    dg_scaled = du * a.view(1, H, 1)
    dg_logits = dg_scaled.view_as(g_logits)
    dA_log = (du * gs.view(-1, H, D)).sum(dim=(0, 2)) * a  # [H]
    ddt_bias = None
    if dt_bias is not None:
        ddt_bias = dg_scaled.sum(dim=0)                    # [H, D]
    return dg_logits, dA_log, ddt_bias


def _stats(name, a, b):
    d = (a.float() - b.float()).abs()
    ref = b.float().abs()
    print(f"  {name:20s} max_abs={d.max().item():.3e}  "
          f"max_rel={(d.max() / (ref.max() + 1e-12)).item():.3e}  mean_abs={d.mean().item():.3e}")


def _self_test():
    from kda_ref import kda_token_recurrent, l2_normalize

    torch.manual_seed(0)
    for dtype in [torch.float64, torch.float32]:
        print(f"\n===== dtype={dtype} =====")
        for (T, D) in [(1, 8), (16, 16), (20, 12), (48, 16)]:
            for use_state in [False, True]:
                for g_scale in [1.0, 4.0, 0.0]:
                    q = l2_normalize(torch.randn(T, D)).to(dtype)
                    k = l2_normalize(torch.randn(T, D)).to(dtype)
                    v = torch.randn(T, D).to(dtype)
                    g = (-torch.rand(T, D) * g_scale).to(dtype)
                    beta = torch.rand(T).to(dtype)
                    h0 = (torch.randn(D, D) if use_state else None)
                    h0 = h0.to(dtype) if h0 is not None else None
                    scale = 1.0 / math.sqrt(D)
                    dout = torch.randn(T, D).to(dtype)
                    dfin = torch.randn(D, D).to(dtype) if use_state else None

                    # autograd reference
                    qa = q.clone().requires_grad_(True)
                    ka = k.clone().requires_grad_(True)
                    va = v.clone().requires_grad_(True)
                    ga = g.clone().requires_grad_(True)
                    ba = beta.clone().requires_grad_(True)
                    ha = h0.clone().requires_grad_(True) if h0 is not None else None
                    o, Sf = kda_token_recurrent(qa, ka, va, ga, ba, scale, ha)
                    loss = (o * dout).sum()
                    if dfin is not None:
                        loss = loss + (Sf * dfin).sum()
                    loss.backward()

                    mine = kda_token_backward(q, k, v, g, beta, scale, dout, h0, dfin)
                    tag = f"T={T} D={D} state={use_state} g={g_scale}"
                    print(f"[{tag}]")
                    _stats("dq", mine["dq"], qa.grad)
                    _stats("dk", mine["dk"], ka.grad)
                    _stats("dv", mine["dv"], va.grad)
                    _stats("dg", mine["dg"], ga.grad)
                    _stats("dbeta", mine["dbeta"], ba.grad)
                    if ha is not None:
                        _stats("dh0", mine["dh0"], ha.grad)
    print("\nDONE")


if __name__ == "__main__":
    _self_test()
