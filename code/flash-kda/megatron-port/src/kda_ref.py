# Copyright (c) 2026, tech_record contributors.
#
# Minimal, dependency-light PyTorch reference implementation of Kimi Delta Attention (KDA),
# used to (a) nail down the math of the FLA ``chunk_kda`` operators before porting them into
# Megatron-LM ``megatron/core/ssm/gated_delta_net`` and (b) provide a golden to diff against.
#
# Design notes / conventions (aligned with FLA ``fla/ops/kda``):
#   * Layout is ``[B, T, H, *]`` (FLA's public API), NOT Megatron's ``[T, B, H, *]`` SBHD.
#     Megatron only ever transposes before calling the kernel; see ``kda.py::_forward_compute``.
#   * ``g`` here is the **activated log-decay** (<= 0), one value per key channel, i.e. the
#     output of ``kda_gate``. Inside the chunk kernels FLA carries it in **log2** units
#     (``chunk_local_cumsum(..., scale=RCP_LN2)``); this reference uses **natural** log for
#     readability, which is isomorphic: ``g_log2 = g_nat / ln 2``.
#   * ``beta`` must already be post-sigmoid in ``(0, 1)`` (FLA's ``use_beta_sigmoid_in_kernel``
#     only wraps ``torch.sigmoid`` before the chunk kernel).
#   * The state is ``[B, H, K, V]`` (FLA default); ``state_v_first`` transposes it to ``[V, K]``.
#
# Everything runs on CPU in float64 for the self-test, so it is safe to import and run
# without a GPU.
#
# Self-test:
#   python3 src/kda_ref.py
# It checks, in fp64:
#   1. token recurrence  ==  chunked forward (o and final state),
#   2. chunked forward gradients (autograd) == token recurrence gradients,
#   3. the closed-form gate inverse (softplus) matches autograd through the gate.

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Gate: raw logits -> per-channel log-decay (natural log)
# --------------------------------------------------------------------------------------
def kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """FLA-compatible KDA gate activation.

    Args:
        g:        raw gate logits ``[..., H, K]`` (before activation).
        A_log:    per-head log step size, shape ``[H]`` (fp32 in practice).
        dt_bias:  per-channel step-size bias, shape ``[H * K]`` (fp32 in practice).
        lower_bound: if set, use the *safe gate* variant
                  ``g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
                  instead of ``g = -exp(A_log) * softplus(g + dt_bias)``.
                  When ``A_log is None`` it reduces to ``lower_bound * sigmoid(g + dt_bias)``.

    Returns:
        Activated log-decay ``[..., H, K]`` (dtype of ``g`` after float promotion).

    Mirrors ``fla/ops/kda/gate.py::naive_kda_gate`` / ``naive_kda_lowerbound_gate``.
    """
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    if lower_bound is None:
        assert A_log is not None, "A_log is required for the non-lower-bound gate."
        return -A_log.view(H, 1).float().exp() * F.softplus(g)
    if A_log is not None:
        g = A_log.view(H, 1).float().exp() * g
    return lower_bound * torch.sigmoid(g)


def kda_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization over the last dim, matching FLA ``l2norm`` (q/k only)."""
    return x / x.float().pow(2).sum(-1, keepdim=True).add(eps).sqrt().to(x.dtype)


# --------------------------------------------------------------------------------------
# Golden: token-by-token gated delta rule
# --------------------------------------------------------------------------------------
def kda_token_recurrent(
    q: torch.Tensor,          # [B, T, H, K]
    k: torch.Tensor,          # [B, T, H, K]
    v: torch.Tensor,          # [B, T, H, V]
    g: torch.Tensor,          # [B, T, H, K]  activated log-decay (<=0)
    beta: torch.Tensor,       # [B, T, H]     post-sigmoid
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # [B, H, K, V]
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference O(T) recurrence:

        a_t = exp(g_t)                         # [B,H,K] per-channel decay
        p_t = (a_t * k_t) @ S_{t-1}             # [B,H,V]
        d_t = beta_t * (v_t - p_t)              # [B,H,V]
        S_t = diag(a_t) S_{t-1} + k_t d_t^T     # [B,H,K,V]
        o_t = scale * q_t @ S_t                 # [B,H,V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    dtype = q.dtype
    # float64 internally for a clean golden.
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    S = torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device)
    if initial_state is not None:
        S = S + initial_state.double()
    o = torch.empty(B, T, H, V, dtype=torch.float64, device=q.device)
    for t in range(T):
        a = torch.exp(g[:, t])                                   # [B,H,K]
        kt = k[:, t]                                             # [B,H,K]
        p = torch.einsum("bhk,bhkv->bhv", a * kt, S)             # [B,H,V]
        d = beta[:, t].unsqueeze(-1) * (v[:, t] - p)             # [B,H,V]
        S = a.unsqueeze(-1) * S + kt.unsqueeze(-1) * d.unsqueeze(-2)
        o[:, t] = scale * torch.einsum("bhk,bhkv->bhv", q[:, t], S)
    return o.to(dtype), (S.to(dtype) if output_final_state else None)


# --------------------------------------------------------------------------------------
# Chunked forward (the shape FLA's chunk_kda kernels compute)
# --------------------------------------------------------------------------------------
def _chunk_cumsum(g: torch.Tensor) -> torch.Tensor:
    """Inclusive cumulative sum over the *within-chunk* time dim (dim=2) of a split ``[B,NT,C,...]``."""
    return torch.cumsum(g, dim=2)


def _invert_lower_i_plus(A: torch.Tensor) -> torch.Tensor:
    """Return ``(I + A)^{-1}`` for strictly-lower-triangular ``A`` (last two dims).

    This is the ``solve_tril`` / forward-substitution step of the KDA intra-chunk kernel
    (``fla/ops/kda/chunk_intra.py``). We vectorize it over leading dims with an O(C) loop,
    exactly mirroring ``torch_chunk_gated_delta_rule`` (``gdn.py:706-710``).
    """
    C = A.shape[-1]
    # The recursion below evaluates (I - X)^{-1}; feed it X = -A to obtain (I + A)^{-1}.
    M = -A.clone()
    for i in range(1, C):
        row = M[..., i, :i].clone()
        sub = M[..., :i, :i].clone()
        M[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    eye = torch.eye(C, dtype=A.dtype, device=A.device)
    return M + eye


def kda_chunked_fwd(
    q: torch.Tensor,          # [B, T, H, K]
    k: torch.Tensor,          # [B, T, H, K]
    v: torch.Tensor,          # [B, T, HV, V]  (HV == H, or a GVA multiple of H)
    g: torch.Tensor,          # [B, T, HV, K] activated log-decay (<=0)
    beta: torch.Tensor,       # [B, T, HV]     post-sigmoid
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # [B, HV, K, V] or [B, HV, V, K] if state_v_first
    output_final_state: bool = False,
    chunk_size: int = 64,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunkwise form; the algebra the FLA forward kernels implement.

    Per chunk ``C`` (local index ``i``), with ``g_cum_i = sum_{j<=i} g_j`` and
    ``E_i = exp(g_cum_i)``, ``E_tot = E_{C-1}`` (all per key channel):

        k_dec_i = k_i * E_i          k_inv_i = k_i / E_i
        q_dec_i = q_i * E_i          k_rest_i = k_inv_i * E_tot

        L_ij       = beta_i * (k_dec_i . k_inv_j),  i > j        (strictly lower)
        Aqk_ij     = (q_dec_i . k_inv_j) * scale,    i >= j       (lower incl. diag)
        T          = (I + L)^{-1}
        U          = T @ (beta * v)                              ("u")
        W          = T @ (beta * k_dec)                          ("w")
        v_new      = U - W @ S_in
        S_out      = diag(E_tot) S_in + k_rest^T @ v_new
        o          = scale * (q_dec @ S_in) + Aqk @ v_new

    Note ``w``/``u``/``qg``/``kg`` are the exact tensors FLA saves for backward
    (``chunk_fwd.py:71-84``) and ``v_new`` is what the state kernel writes.

    **GVA** (grouped value attention): FLA allows ``HV > H`` (``HV % H == 0``); the
    value head ``hv`` shares the q/k head ``hv // (HV // H)`` (``chunk_intra.py:79``).
    We reproduce that exactly with ``repeat_interleave`` on q/k, whose autograd is the
    matching group-sum, so ``dq``/``dk`` come out over the original ``H`` heads.

    **state_v_first**: FLA's ``initial_state``/``final_state`` use the V-first layout
    ``[..., V, K]`` when set (instead of the default ``[..., K, V]``); the recurrence is
    on the transpose either way.
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    dtype = q.dtype
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    if HV != H:
        assert HV % H == 0, f"GVA requires HV % H == 0, got HV={HV}, H={H}"
        grp = HV // H
        q = q.repeat_interleave(grp, dim=2)
        k = k.repeat_interleave(grp, dim=2)
    H = HV  # from here H counts the *value/state* heads

    pad = (-T) % chunk_size
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
    NT = (T + pad) // chunk_size

    def split(x):
        # [B, NT, C, H, *]
        return x.reshape(B, NT, chunk_size, *x.shape[2:])

    q, k, v, g, beta = split(q), split(k), split(v), split(g), split(beta)

    g_cum = _chunk_cumsum(g)                       # [B,NT,C,H,K]
    E = torch.exp(g_cum)                           # may underflow to 0 -> harmless
    E_tot = E[:, :, -1]                            # [B,NT,H,K]

    k_dec = k * E
    q_dec = q * E
    # k_rest_i = k_i * exp(g_tot - g_i); the exponent is <= 0, so no overflow.
    k_rest = k * torch.exp(g_cum[:, :, -1:] - g_cum)

    # Intra-chunk decay is computed pairwise as exp(g_cum_i - g_cum_j) so that
    # *only* the masked (i >= j) entries are ever exponentiated. Doing it via
    # k_dec_i . k_inv_j would blow up, because k_inv = k / E overflows when the
    # cumulative decay is large (the same reason FLA anchors g at a local
    # reference before exp2; see docs/03-optimization.md).
    gc = g_cum                                     # [B,NT,C,H,K]
    diff = gc.unsqueeze(3) - gc.unsqueeze(2)       # [B,NT,C_i,C_j,H,K]
    lower_incl = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                       device=q.device), diagonal=0).view(1, 1, chunk_size, chunk_size, 1, 1)
    diff = torch.where(lower_incl, diff, torch.full_like(diff, float("-inf")))
    D = torch.exp(diff)                            # masked pairwise decay

    # beta_i as [B,NT,H,C,1] to scale the *row* index i of the intra matrices.
    betaT = beta.permute(0, 1, 3, 2).unsqueeze(-1)
    # L_ij = beta_i (k_i . k_j) D_ij, strictly lower. Per (value-)head.
    L = torch.einsum("bnihk,bnjhk,bnijhk->bnhij", k, k, D) * betaT
    strict_lower = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                         device=q.device), diagonal=-1)
    L = L * strict_lower
    # Aqk_ij = (q_i . k_j) D_ij * scale, lower incl. diagonal.
    Aqk = torch.einsum("bnihk,bnjhk,bnijhk->bnhij", q, k, D) * scale

    Tmat = _invert_lower_i_plus(L)                 # (I + L)^{-1}, [B,NT,H,C,C]
    U = torch.einsum("bnhij,bnjhv->bnihv", Tmat, v * beta.unsqueeze(-1))
    Wm = torch.einsum("bnhij,bnjhk->bnihk", Tmat, k_dec * beta.unsqueeze(-1))

    S = torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device)
    if initial_state is not None:
        s0 = initial_state.double()
        if state_v_first:
            s0 = s0.transpose(-1, -2)
        S = S + s0

    o = torch.empty(B, NT, chunk_size, H, V, dtype=torch.float64, device=q.device)
    for c in range(NT):
        S_in = S
        Wsin = torch.einsum("bihk,bhkv->bihv", Wm[:, c], S_in)     # [B,C,H,V]
        v_new = U[:, c] - Wsin
        o[:, c] = (scale * torch.einsum("bihk,bhkv->bihv", q_dec[:, c], S_in)
                   + torch.einsum("bhij,bjhv->bihv", Aqk[:, c], v_new))
        S = (E_tot[:, c].unsqueeze(-1) * S_in
             + torch.einsum("bihk,bihv->bhkv", k_rest[:, c], v_new))

    o = o.reshape(B, NT * chunk_size, H, V)[:, :T]
    fstate = S.transpose(-1, -2) if state_v_first else S
    return o.to(dtype), (fstate.to(dtype) if output_final_state else None)


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------
def _rand_case(B, T, H, K, V, seed, chunk_size):
    torch.manual_seed(seed)
    dev = "cpu"
    q = torch.randn(B, T, H, K, device=dev, dtype=torch.float64)
    k = torch.randn(B, T, H, K, device=dev, dtype=torch.float64)
    v = torch.randn(B, T, H, V, device=dev, dtype=torch.float64)
    g_raw = torch.randn(B, T, H, K, device=dev, dtype=torch.float64)
    # KDA-realistic gate parameters (Megatron A_init_range=(1,16), inverse-softplus dt_bias),
    # so that the per-step decay stays bounded and the outputs are O(1).
    A_log = torch.log(torch.empty(H, device=dev, dtype=torch.float64).uniform_(1.0, 16.0))
    dt_bias = 0.1 * torch.randn(H * K, device=dev, dtype=torch.float64)
    beta = torch.rand(B, T, H, device=dev, dtype=torch.float64)
    g = kda_gate(g_raw, A_log, dt_bias)
    h0 = torch.randn(B, H, K, V, device=dev, dtype=torch.float64)
    return q, k, v, g, beta, h0


def _main():
    torch.manual_seed(0)
    print("=" * 78)
    print("KDA reference self-test (fp64, CPU)")
    print("=" * 78)

    # --- Part A: token recurrence vs chunked forward -------------------------------
    cases = [
        (1, 16, 2, 16, 16, 64),
        (2, 33, 3, 24, 20, 16),
        (1, 64, 2, 32, 32, 64),
        (2, 130, 4, 32, 48, 32),
    ]
    print("\n[Part A] token recurrence vs chunked forward (o and final state)")
    ok = True
    for (B, T, H, K, V, C) in cases:
        q, k, v, g, beta, h0 = _rand_case(B, T, H, K, V, seed=1, chunk_size=C)
        o_tok, s_tok = kda_token_recurrent(q, k, v, g, beta, initial_state=h0,
                                           output_final_state=True)
        o_chk, s_chk = kda_chunked_fwd(q, k, v, g, beta, initial_state=h0,
                                       output_final_state=True, chunk_size=C)
        eo = (o_tok - o_chk).abs().max().item() / (o_tok.abs().max().item() + 1e-30)
        es = (s_tok - s_chk).abs().max().item() / (s_tok.abs().max().item() + 1e-30)
        tag = "OK " if max(eo, es) < 1e-10 else "BAD"
        ok &= tag == "OK "
        print(f"  {tag} B={B} T={T} H={H} K={K} V={V} C={C:>2}  "
              f"rel|do|={eo:.2e}  rel|dS|={es:.2e}")

    # --- Part B: autograd through chunked == autograd through token ---------------
    print("\n[Part B] gradients: autograd(chunked) vs autograd(token)")
    B, T, H, K, V, C = 1, 48, 2, 16, 16, 16
    q, k, v, g, beta, h0 = _rand_case(B, T, H, K, V, seed=2, chunk_size=C)
    dout = torch.randn_like(kda_token_recurrent(q, k, v, g, beta)[0])

    def grads(fn, **kw):
        args = [q.clone().requires_grad_(), k.clone().requires_grad_(),
                v.clone().requires_grad_(), g.clone().requires_grad_(),
                beta.clone().requires_grad_(), h0.clone().requires_grad_()]
        # NOTE: h0 must go through initial_state, never positionally (scale comes first).
        o, _ = fn(*args[:5], initial_state=args[5], **kw)
        o.backward(dout)
        return [a.grad for a in args]

    gtok = grads(kda_token_recurrent)
    gchk = grads(kda_chunked_fwd, chunk_size=C)
    names = ["dq", "dk", "dv", "dg", "dbeta", "dh0"]
    for n, a, b in zip(names, gtok, gchk):
        e = (a - b).abs().max().item() / (a.abs().max().item() + 1e-30)
        tag = "OK " if e < 1e-8 else "BAD"
        ok &= tag == "OK "
        print(f"  {tag} {n:>5}: rel|diff|={e:.2e}")

    # --- Part C: gate inverse -----------------------------------------------------
    print("\n[Part C] gate: naive closed form vs autograd through softplus")
    B, T, H, K = 2, 8, 3, 5
    g_raw = torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=True)
    A_log = torch.randn(H, dtype=torch.float64, requires_grad=True)
    dt_bias = torch.randn(H * K, dtype=torch.float64, requires_grad=True)
    y = kda_gate(g_raw, A_log, dt_bias)
    y.sum().backward()
    # analytic: y = -exp(A) sum_k softplus(z_k); dz/dg=1; dA=-exp(A)*sum softplus(z_k)
    z = g_raw.detach() + dt_bias.detach().view(H, -1)
    expA = A_log.detach().exp()
    dg_ref = -expA.unsqueeze(-1) * torch.sigmoid(z)
    dA_ref = (-expA.unsqueeze(-1) * F.softplus(z)).sum(dim=(0, 1, 3))  # per-head [H]
    # NOTE: ``kda_gate`` casts to fp32 (to mirror FLA), so the analytic check is only
    # fp32-tight. Compare relatively on the per-head gradient dA.
    eg = (g_raw.grad - dg_ref).abs().max().item() / (dg_ref.abs().max().item() + 1e-30)
    ea = (A_log.grad - dA_ref).abs().max().item() / (dA_ref.abs().max().item() + 1e-30)
    tag = "OK " if max(eg, ea) < 1e-4 else "BAD"
    ok &= tag == "OK "
    print(f"  {tag} rel dg={eg:.2e}   rel dA={ea:.2e}  (fp32 gate math)")

    print("\n" + "=" * 78)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
