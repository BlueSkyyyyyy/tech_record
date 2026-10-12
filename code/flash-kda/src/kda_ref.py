"""KDA forward reference: token-recurrent golden vs CHUNK=16 blocked form (pure PyTorch, CPU).

Conventions (match FlashKDA / `tests/torch_ref.py`):

* ``q, k, v``: ``[T, D]`` (single head), ``k``/``q`` already L2-normalised.
* ``g``: ``[T, D]`` **log2 decay**, ``g <= 0`` (per key-channel). Decay factor is ``2**g``.
* ``beta``: ``[T]`` write gate, already sigmoid-activated, ``beta in (0, 1)``.
* state ``S``: ``[D, D]`` = ``[K, V]``; per-token update and output
  ``o_t = scale * q_t @ S_t``.
* ``scale``: output scale (typically ``1/sqrt(D)``); applied to ``q``.

Two independent implementations:

* :func:`kda_token_recurrent` -- naive per-token gated delta rule (fp32 golden);
* :func:`kda_chunked_fwd` -- the blocked (chunk) algebra that FlashKDA K1/K2 mirror.

Run ``python kda_ref.py`` to cross-check them on CPU.
"""

import math

import torch

LOG2E = 1.4426950408889634


def l2_normalize(x, eps=1e-6):
    """Row-wise L2 normalisation (simple version; FlashKDA uses a shuffle-tree order)."""
    return x / x.pow(2).sum(dim=-1, keepdim=True).add(eps).sqrt()


def kda_gate(g_logits, A_log, dt_bias=None, lower_bound=-5.0):
    """KDA gate activation -> log2 decay (matches FlashKDA `torch_ref.py`).

    ``g = lower_bound * log2(e) * sigmoid(exp(A_log) * (g_logits + dt_bias))``, ``<= 0``.
    """
    H, D = g_logits.shape[-2:]
    g = g_logits.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, D)
    a = A_log.view(H, 1).exp()
    return (lower_bound * LOG2E) * torch.sigmoid(a * g)


def kda_beta(beta_logits):
    return torch.sigmoid(beta_logits.float())


def kda_token_recurrent(q, k, v, g, beta, scale=1.0, h0=None):
    """Per-token gated delta rule (golden reference), fp32.

    State ``S`` is ``[K, V]``. For each token ``t`` with ``alpha = 2**g_t``:

        S_t = diag(alpha_t) S_{t-1} + k_t outer (beta_t (v_t - (k_t * alpha_t) @ S_{t-1}))
        o_t = scale * q_t @ S_t
    """
    T, D = q.shape
    dt = torch.float64 if (q.is_floating_point() and q.dtype == torch.float64) else torch.float32
    q = q.to(dt)
    k = k.to(dt)
    v = v.to(dt)
    g = g.to(dt)
    beta = beta.to(dt)
    S = torch.zeros(D, D, dtype=dt, device=q.device) if h0 is None else h0.to(dt).clone()
    outs = []
    for t in range(T):
        alpha = torch.exp2(g[t])                        # [K]
        pred = (k[t] * alpha) @ S                       # [V]
        delta = beta[t] * (v[t] - pred)                 # [V]
        S = alpha[:, None] * S + k[t][:, None] * delta[None, :]
        outs.append(scale * (q[t] @ S))
    return torch.stack(outs, dim=0), S


def _chunk_inv(L):
    """(I + L)^-1 for strictly-lower triangular L [C, C]. Uses a dense solve for the
    reference; the CUDA kernel uses an 8x8 forward substitution + HMMA merge."""
    C = L.shape[0]
    eye = torch.eye(C, dtype=L.dtype, device=L.device)
    return torch.linalg.solve(eye + L, eye)


def kda_chunked_fwd(q, k, v, g, beta, scale=1.0, h0=None, chunk=16):
    """Blocked (chunk) KDA forward, pure PyTorch, fp32/fp64.

    For each chunk with cumulative log2-decay ``gc = cumsum(g)`` and ``gtot = gc[-1]``:

        E = 2**gc;  qd = q*E*scale;  kd = k*E;  ki = k / E;  kr = ki * 2**gtot
        L   = tril(kd @ ki^T, -1) * beta
        Mqk = tril(qd @ ki^T)
        P   = (I + L)^-1
        U   = P @ ((v - kd @ S) * beta)
        o   = qd @ S + Mqk @ U
        S   = kr^T @ U + 2**gtot[:, None] * S
    """
    T, D = q.shape
    dt = q.dtype if q.dtype.is_floating_point else torch.float32
    q = q.to(dt)
    k = k.to(dt)
    v = v.to(dt)
    g = g.to(dt)
    beta = beta.to(dt)
    S = torch.zeros(D, D, dtype=dt, device=q.device) if h0 is None else h0.to(dt).clone()
    out = torch.empty(T, D, dtype=dt, device=q.device)

    for t0 in range(0, T, chunk):
        C = min(chunk, T - t0)
        sl = slice(t0, t0 + C)
        qc, kc, vc, gc_, bc = q[sl], k[sl], v[sl], g[sl], beta[sl]
        if C < chunk:  # pad; zero k/beta/g make padding inert
            pad = chunk - C
            qc = torch.cat([qc, qc.new_zeros(pad, D)], 0)
            kc = torch.cat([kc, kc.new_zeros(pad, D)], 0)
            vc = torch.cat([vc, vc.new_zeros(pad, D)], 0)
            gc_ = torch.cat([gc_, gc_.new_zeros(pad, D)], 0)
            bc = torch.cat([bc, bc.new_zeros(pad)], 0)

        gcum = gc_.cumsum(dim=0)                     # [C, K]
        gtot = gcum[-1]                              # [K]
        E = torch.exp2(gcum)
        etot = torch.exp2(gtot)

        kd = kc * E
        qd = qc * E * scale
        ki = kc * torch.exp2(-gcum)
        kr = ki * etot[None, :]

        L = torch.tril(kd @ ki.transpose(0, 1), diagonal=-1) * bc[:, None]
        Mqk = torch.tril(qd @ ki.transpose(0, 1))
        P = _chunk_inv(L)

        uprime = (vc - kd @ S) * bc[:, None]
        U = P @ uprime
        o = qd @ S + Mqk @ U
        S = kr.transpose(0, 1) @ U + etot[:, None] * S

        out[sl] = o[:C]
    return out, S


def _stats(name, a, b):
    d = (a.float() - b.float()).abs()
    ref = b.float().abs()
    max_abs = d.max().item()
    max_rel = (d.max() / (ref.max() + 1e-12)).item()
    print(f"  {name:24s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  mean_abs={d.mean().item():.3e}")


def _self_test():
    torch.manual_seed(0)
    D = 16
    for dtype in [torch.float64, torch.float32]:
        print(f"\n===== dtype={dtype} =====")
        for (T, H) in [(1, 1), (16, 1), (20, 2), (48, 3)]:
            for use_state in [False, True]:
                for g_scale in [1.0, 4.0, 0.0]:
                    q = torch.randn(T, H, D)
                    k = torch.randn(T, H, D)
                    v = torch.randn(T, H, D)
                    g = -torch.rand(T, H, D) * g_scale
                    beta = torch.rand(T, H)
                    h0 = torch.randn(H, D, D) if use_state else None
                    scale = 1.0 / math.sqrt(D)

                    o_rec = torch.empty(T, H, D)
                    o_chk = torch.empty(T, H, D)
                    s_rec = torch.empty(H, D, D)
                    s_chk = torch.empty(H, D, D)
                    for h in range(H):
                        qr = l2_normalize(q[:, h]).to(dtype)
                        kr = l2_normalize(k[:, h]).to(dtype)
                        vr = v[:, h].to(dtype)
                        gr = g[:, h].to(dtype)
                        br = beta[:, h].to(dtype)
                        h0h = None if h0 is None else h0[h].to(dtype)
                        or_, sr_ = kda_token_recurrent(qr, kr, vr, gr, br, scale, h0h)
                        oc_, sc_ = kda_chunked_fwd(qr, kr, vr, gr, br, scale, h0h, chunk=16)
                        o_rec[:, h], s_rec[h] = or_.float(), sr_.float()
                        o_chk[:, h], s_chk[h] = oc_.float(), sc_.float()

                    tag = f"T={T} H={H} state={use_state} g={g_scale}"
                    print(f"[{tag}]")
                    _stats("o chunk vs token", o_chk, o_rec)
                    _stats("S chunk vs token", s_chk, s_rec)
    print("DONE")


if __name__ == "__main__":
    _self_test()
