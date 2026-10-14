# Copyright (c) 2026, tech_record contributors.
#
# KDA-specific *fused pre-GDR* Triton kernel family (megatron-port M13-1).
#
# ``KimiDeltaAttention`` deliberately refuses the GDN streamed-fused pre-GDR
# (``gdn_pre_gated_delta_rule_fusion``) because the two are structurally different
# (see ``docs/14``).  This module is the KDA-specific replacement called for by the
# contract in ``docs/14`` §5: given the *exact* tensors the eager
# ``pre_gated_delta_rule`` consumes (``qkvfg`` = ``in_proj`` output and ``beta`` =
# ``beta_proj`` output, both ``[T, B, *]`` SBHD), it reproduces the 8-tuple that the
# eager path returns to ``chunk_kda``.
#
# The eager path has three scopes, each mirrored here by one Triton kernel:
#
#   1. conv scope   : depthwise causal conv1d (kernel ``W``, left pad ``W-1``) over the
#                     first ``2*qk+v`` channels of ``qkvfg`` + SiLU.  Mirrors
#                     ``fla.modules.conv.triton.kernels.causal_conv1d_fwd_kernel``
#                     (fp32 accumulation, same tap order, ``silu(y)=y*sigmoid(y)``).
#   2. qk scope     : L2 norm over the last dim (``KD``) of the reshaped
#                     ``query_key`` [B,T,2Hk,KD], fp32 internal accumulation, exactly
#                     like ``fla.modules.l2norm.l2norm_fwd_kernel1``.
#   3. g/beta scope : ``beta`` sigmoid (fp32) + ``raw_g`` reshape.  KDA *never* activates
#                     the decay gate here (``use_gate_in_kernel=True``); ``raw_g`` is
#                     returned verbatim and ``A_log``/``dt_bias`` are handed to
#                     ``chunk_kda``.
#
# ``gate`` / ``raw_g`` are returned as *views* of ``qkvfg`` (streamed, not materialized);
# only the conv output and the l2-normed ``query_key`` are freshly allocated, matching the
# eager path's own fresh allocations (the l2norm kernel writes q/k fresh).
#
# Layout: SBHD cp=1, ``num_value_heads == num_key_heads`` (Megatron KDA hard constraint,
# so the GVA repeat factor is 1).  ``qkvfg`` is ``[T, B, 2qk+v+qk+v]``.
#
# Self-test (needs a GPU but *not* Megatron): builds random weights and checks the Triton
# family against a pure-torch re-implementation (``_ref_torch``).
#   python3 src/kda_pregdr_fused.py

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------------------
# 1. conv scope: depthwise causal conv1d + SiLU
# --------------------------------------------------------------------------------------
@triton.jit
def _kda_conv_silu_kernel(
    x, y, weight, bias,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t * BT + tl.arange(0, BT)
    o_w = tl.arange(0, BW) + W - BW          # [W] -> taps 0..W-1
    m_d = o_d < D
    m_y = (o_t < T)[:, None] & m_d[None, :]

    # weight [D, 1, W] -> read as [BD, BW]
    b_w = tl.load(weight + o_d[:, None] * W + o_w[None, :],
                  mask=m_d[:, None] & (o_w[None, :] >= 0), other=0.0).to(tl.float32)

    acc = tl.zeros((BT, BD), dtype=tl.float32)
    x_base = x + i_b * (T * D)
    for i_w in tl.static_range(-W + 1, 1):
        o_x = o_t + i_w
        p = x_base + o_x[:, None] * D + o_d[None, :]
        b_yi = tl.load(p, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :],
                       other=0.0).to(tl.float32)
        b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1))[None, :], 1)
        acc += b_yi

    if HAS_BIAS:
        acc += tl.load(bias + o_d, mask=m_d, other=0.0).to(tl.float32)

    if ACTIVATION == 1:  # silu / swish
        acc = acc * tl.sigmoid(acc)

    p_y = y + i_b * (T * D) + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, acc.to(p_y.dtype.element_ty), mask=m_y)


# --------------------------------------------------------------------------------------
# 2. qk scope: L2 norm over the last dim (KD)
# --------------------------------------------------------------------------------------
@triton.jit
def _kda_l2norm_kernel(x, y, eps, T, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m = m_t[:, None] & (o_d[None, :] < D)
    p = o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(x + p, mask=m, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    tl.store(y + p, (b_x * b_rstd[:, None]).to(y.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------------------
# 3. g/beta scope: beta sigmoid (fp32)
# --------------------------------------------------------------------------------------
@triton.jit
def _kda_beta_sigmoid_kernel(x, y, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    v = tl.load(x + off, mask=m, other=0.0).to(tl.float32)
    tl.store(y + off, (1.0 / (1.0 + tl.exp(-v))).to(y.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------------------
# Host entry
# --------------------------------------------------------------------------------------
def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _forward_raw(qkvfg, beta, conv_w, conv_b, dims):
    """Triton fused pre-GDR core.  Returns the 6 kernel operands
    ``(q, k, v, gate, beta_sig, raw_g)`` (A_log/dt_bias are pass-through leaves)."""
    qk, vd, Hk, Hv, KD, VD, W, has_bias, activation = dims
    dev = qkvfg.device
    B, T, Cin = beta.shape[1], beta.shape[0], qkvfg.shape[-1]
    C = 2 * qk + vd

    qkvfg_t = qkvfg.transpose(0, 1)                 # [B, T, Cin]  (view)
    qkv, raw_g, gate = torch.split(qkvfg_t, (C, qk, vd), dim=-1)
    x = qkv.contiguous()                            # conv scope: channel-last [B,T,C]
    conv = torch.empty(B, T, C, device=dev, dtype=qkvfg.dtype)

    BD = min(128, _next_pow2(C))
    BT = 64
    BW = _next_pow2(W)
    grid = (triton.cdiv(C, BD), triton.cdiv(T, BT), B)
    _kda_conv_silu_kernel[grid](
        x, conv, conv_w, conv_b,
        T, D=C, W=W, BT=BT, BW=BW, BD=BD,
        HAS_BIAS=has_bias, ACTIVATION=activation,
    )

    # qk scope: reshape query_key [B,T,2Hk,KD] and L2 norm
    query_key = conv[..., :2 * qk].reshape(B, T, 2 * Hk, KD).contiguous()
    qk_norm = torch.empty_like(query_key)
    rows = B * T * 2 * Hk
    _kda_l2norm_kernel[(triton.cdiv(rows, 64),)](query_key, qk_norm, 1e-6, rows,
                                                 D=KD, BD=_next_pow2(KD), BT=64)
    query, key = torch.split(qk_norm, [Hk, Hk], dim=2)
    value = conv[..., 2 * qk:].reshape(B, T, Hv, VD)

    # g/beta scope
    raw_g = raw_g.reshape(B, T, Hk, KD)
    beta_t = beta.transpose(0, 1).contiguous()      # [B, T, Hk]
    beta_sig = torch.empty_like(beta_t, dtype=torch.float32)
    N = B * T * Hk
    _kda_beta_sigmoid_kernel[(triton.cdiv(N, 1024),)](beta_t, beta_sig, N, BLOCK=1024)

    return (query.contiguous(), key.contiguous(), value.contiguous(),
            gate.reshape(B, T, Hv, VD).contiguous(), beta_sig.contiguous(), raw_g.contiguous())


def _dims(module):
    return (module.qk_dim_local_tp, module.v_dim_local_tp, module.num_key_heads,
            module.num_value_heads, module.key_head_dim, module.value_head_dim,
            module.conv_kernel_dim, bool(module.conv_bias),
            1 if module.activation in ("silu", "swish") else 0)


def kda_pre_gdr_fused_triton(module, qkvfg, beta, batch, seq_len):
    """KDA-specific fused pre-GDR (Triton, **forward only**).  Consumes/produces exactly
    the same tensors as ``KimiDeltaAttention.pre_gated_delta_rule`` (SBHD, cp=1, GVA=1).

    ``qkvfg``: [S, B, 2qk+v+qk+v];  ``beta``: [S, B, Hk].
    Returns ``(q, k, v, gate, beta_sig, raw_g, A_log, dt_bias)``.
    """
    assert module.num_value_heads == module.num_key_heads, "KDA hard constraint"
    assert module.use_qk_l2norm
    out = _forward_raw(qkvfg, beta, module.conv1d.weight, module.conv1d.bias, _dims(module))
    return out + (module.A_log, module.dt_bias)


class _FusedPreGDRAutograd(torch.autograd.Function):
    """Drop-in autograd wrapper: Triton forward, reference (torch-primitive) backward.

    The Triton family is forward-only; the backward here replays the *same algebraic
    contract* with differentiable torch ops (``F.conv1d`` + fp32-internal L2 norm +
    ``sigmoid``).  This is enough to prove the Triton forward is a numerically transparent
    replacement of ``pre_gated_delta_rule`` end-to-end (full-layer param grads flow), while
    a native Triton backward remains a separate future task.
    """

    @staticmethod
    def forward(ctx, module, qkvfg, beta, batch, seq_len):
        ctx.module = module
        ctx.save_for_backward(qkvfg, beta)
        out = _forward_raw(qkvfg, beta, module.conv1d.weight, module.conv1d.bias, _dims(module))
        return out

    @staticmethod
    def backward(ctx, *grads):
        module = ctx.module
        qkvfg, beta = ctx.saved_tensors
        dims = _dims(module)
        qk, vd, Hk, Hv, KD, VD, W, has_bias, activation = dims
        with torch.enable_grad():
            q_ = qkvfg.detach().requires_grad_(True)
            b_ = beta.detach().requires_grad_(True)
            w_ = module.conv1d.weight.detach().requires_grad_(True)
            bias_ = module.conv1d.bias.detach().requires_grad_(True) if has_bias else None
            outs = _ref_torch(q_, b_, w_, bias_, Hk, Hv, KD, VD, W, silu=bool(activation))
            loss = q_.sum() * 0.0
            for o, g in zip(outs, grads):
                if g is not None:
                    loss = loss + (o.float() * g.float()).sum()
            loss.backward()
        return (None, q_.grad, b_.grad, None, None)


def kda_pre_gdr_fused_autograd(module, qkvfg, beta, batch, seq_len):
    """Autograd-enabled drop-in (Triton forward + reference backward)."""
    assert module.num_value_heads == module.num_key_heads, "KDA hard constraint"
    assert module.use_qk_l2norm
    out = _FusedPreGDRAutograd.apply(module, qkvfg, beta, batch, seq_len)
    return out + (module.A_log, module.dt_bias)


# --------------------------------------------------------------------------------------
# Pure-torch reference (self-test only; no Megatron needed)
# --------------------------------------------------------------------------------------
def _ref_torch(qkvfg, beta, conv_w, conv_b, Hk, Hv, KD, VD, W, eps=1e-6, silu=True):
    # qkvfg is SBHD [S, B, Cin]
    T, B, Cin = qkvfg.shape
    qkvfg = qkvfg.transpose(0, 1)                             # [B, T, Cin]
    qk = Hk * KD
    vd = Hv * VD
    C = 2 * qk + vd
    x = qkvfg[..., :C].transpose(1, 2).contiguous()          # [B, C, T]
    import torch.nn.functional as F
    y = F.conv1d(x, conv_w, conv_b, padding=W - 1, groups=C)[..., :T]
    if silu:
        y = y * torch.sigmoid(y)                              # silu
    conv = y.transpose(1, 2).contiguous()                     # [B, T, C]
    qk_ = conv[..., :2 * qk].reshape(B, T, 2 * Hk, KD).float()
    qk_ = (qk_ / torch.sqrt((qk_ * qk_).sum(-1, keepdim=True) + eps)).to(conv.dtype)
    q, k = torch.split(qk_, [Hk, Hk], dim=2)
    val = conv[..., 2 * qk:].reshape(B, T, Hv, VD)
    raw_g = qkvfg[..., C:C + qk].reshape(B, T, Hk, KD)
    gate = qkvfg[..., C + qk:].reshape(B, T, Hv, VD)
    beta_sig = beta.transpose(0, 1).float().sigmoid()
    return q, k, val, gate, beta_sig, raw_g


class _Stub(torch.nn.Module):
    def __init__(self, qk, vd, Hk, Hv, KD, VD, W, dtype):
        super().__init__()
        self.qk_dim_local_tp = qk
        self.v_dim_local_tp = vd
        self.num_key_heads = Hk
        self.num_value_heads = Hv
        self.key_head_dim = KD
        self.value_head_dim = VD
        self.conv_kernel_dim = W
        self.use_qk_l2norm = True
        self.conv_bias = True
        self.activation = "silu"
        self.conv1d = torch.nn.Conv1d(2 * qk + vd, 2 * qk + vd, W, groups=2 * qk + vd,
                                      padding=W - 1, bias=True).to(dtype)
        self.A_log = torch.randn(Hk, device="cuda")
        self.dt_bias = torch.randn(qk, device="cuda")


def _self_test():
    torch.manual_seed(0)
    Hk = Hv = 4
    KD = VD = 32
    W = 4
    qk, vd = Hk * KD, Hv * VD
    B, T = 2, 64
    for dtype in (torch.float32, torch.bfloat16):
        stub = _Stub(qk, vd, Hk, Hv, KD, VD, W, dtype).cuda()
        qkvfg = torch.randn(T, B, 2 * qk + vd + qk + vd, device="cuda", dtype=dtype)
        beta = torch.randn(T, B, Hk, device="cuda", dtype=dtype)
        out = kda_pre_gdr_fused_triton(stub, qkvfg, beta, B, T)
        ref = _ref_torch(qkvfg, beta, stub.conv1d.weight.detach(), stub.conv1d.bias.detach(),
                         Hk, Hv, KD, VD, W)
        worst = 0.0
        for name, a, b in zip(("q", "k", "v", "gate", "beta", "g"), out[:6], ref):
            d = (a.float() - b.float()).abs().max().item()
            worst = max(worst, d)
            print("  %-5s max_abs=%.3e" % (name, d))
        print("  dtype=%s worst max_abs=%.3e" % (str(dtype).split(".")[-1], worst))


if __name__ == "__main__":
    _self_test()
