# Copyright (c) 2026, tech_record contributors.
#
# KDA-specific *fused pre-GDR* native Triton backward (megatron-port M13-2).
#
# M13-1 (``src/kda_pregdr_fused.py``) implemented the KDA-specific fused pre-GDR as a
# forward-only three-kernel Triton family; its autograd wrapper replayed the *eager*
# algebra with differentiable torch ops to get gradients.  That reference backward has two
# consequences, one documented (an extra eager replay) and one **not** documented:
#
#   * it is a torch-primitive fallback, not the fused kernel's own backward; and
#   * it never returns gradients for ``conv1d.weight`` / ``conv1d.bias`` (the reference
#     inside the function used *detached* clones), so M13-1's Part C "full-layer"
#     comparison silently **skipped** the conv parameters (``if ga is None or gb is None:
#     continue``) -- see ``docs/16``.
#
# This module closes both: it adds the three backward kernels the forward family implies
#
#   1. ``conv (+SiLU)`` bwd : ``d_silu`` then depthwise causal-conv dX / dW / dbias,
#   2. ``L2-norm`` bwd       : ``dx = rstd*dy - rstd^3 * x * (dy . x)``,
#   3. ``sigmoid`` bwd       : ``dx = y*(1-y)*dy``,
#
# and re-wires the autograd ``Function`` so that ``conv1d.weight``/``conv1d.bias`` are
# *inputs* of the function (their grads therefore flow to the real parameters).
#
# Everything is fp32-internal; only the pre-GDR operands keep their (bf16/fp32) dtype.
# Nothing here touches the Megatron-LM tree.
#
# Self-test (needs a GPU but *not* Megatron):
#   python3 src/kda_pregdr_bwd.py

import os
import importlib.util

import torch
import triton
import triton.language as tl


def _load_fused():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "kda_pregdr_fused.py")
    spec = importlib.util.spec_from_file_location("kda_pregdr_fused", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fused = _load_fused()
_kda_conv_silu_kernel = fused._kda_conv_silu_kernel
_kda_l2norm_kernel = fused._kda_l2norm_kernel
_kda_beta_sigmoid_kernel = fused._kda_beta_sigmoid_kernel
_next_pow2 = fused._next_pow2
_dims = fused._dims


# --------------------------------------------------------------------------------------
# 1. SiLU backward: grad wrt post-SiLU conv output -> grad wrt pre-activation (fp32)
# --------------------------------------------------------------------------------------
@triton.jit
def _kda_silu_bwd_kernel(dy, pre, dx, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    g = tl.load(dy + off, mask=m, other=0.0).to(tl.float32)
    a = tl.load(pre + off, mask=m, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-a))
    da = g * sig * (1.0 + a * (1.0 - sig))          # silu'(a)
    tl.store(dx + off, da, mask=m)


# --------------------------------------------------------------------------------------
# 2. depthwise causal-conv1d backward: dX (fp32 out), dW / dbias (atomic accumulators)
#
#   forward (channel-last, taps w=0..W-1):  a[t,c] = sum_w x[t+w-(W-1), c] * wgt[c,w]
#   dX[m,c]     = sum_w da[m+(W-1)-w, c] * wgt[c,w]
#   dW[c,w]     = sum_{b,t} x[b, t+w-(W-1), c] * da[b,t,c]
#   dbias[c]    = sum_{b,t} da[b,t,c]
# --------------------------------------------------------------------------------------
@triton.jit
def _kda_conv_bwd_kernel(da, x, wgt, dx, dw, dbias,
                         T, OST, OB, OOFF,
                         D: tl.constexpr, W: tl.constexpr,
                         BT: tl.constexpr, BD: tl.constexpr):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t * BT + tl.arange(0, BT)
    m_d = o_d < D
    m_xy = (o_t < T)[:, None] & m_d[None, :]
    o_w = tl.arange(0, W)

    b_w = tl.load(wgt + o_d[:, None] * W + o_w[None, :],
                  mask=m_d[:, None], other=0.0).to(tl.float32)          # [BD, W]

    da_base = da + i_b * (T * D)
    x_base = x + i_b * (T * D)
    b_da = tl.load(da_base + o_t[:, None] * D + o_d[None, :], mask=m_xy,
                   other=0.0).to(tl.float32)                            # [BT, BD]

    # dX[o_t] = sum_s da[o_t+s] * wgt[W-1-s]
    acc_dx = tl.zeros((BT, BD), dtype=tl.float32)
    for s in tl.static_range(0, W):
        o_ts = o_t + s
        b_das = tl.load(da_base + o_ts[:, None] * D + o_d[None, :],
                        mask=((o_ts < T))[:, None] & m_d[None, :], other=0.0).to(tl.float32)
        b_wc = tl.sum(b_w * (o_w == (W - 1 - s))[None, :], 1)           # [BD]
        acc_dx += b_das * b_wc[None, :]
    tl.store(dx + i_b * OB + o_t[:, None] * OST + OOFF + o_d[None, :],
             acc_dx, mask=m_xy)                                         # dx is fp32

    # dW / dbias via atomics
    for w in tl.static_range(0, W):
        o_tx = o_t + w - (W - 1)
        b_xs = tl.load(x_base + o_tx[:, None] * D + o_d[None, :],
                       mask=((o_tx >= 0) & (o_tx < T))[:, None] & m_d[None, :],
                       other=0.0).to(tl.float32)
        part = tl.sum(b_xs * b_da, 0)                                  # [BD]
        tl.atomic_add(dw + o_d * W + w, part, mask=m_d)
    tl.atomic_add(dbias + o_d, tl.sum(b_da, 0), mask=m_d)


# --------------------------------------------------------------------------------------
# 3. L2-norm backward (over the last dim)
# --------------------------------------------------------------------------------------
@triton.jit
def _kda_l2norm_bwd_kernel(x, dy, dx, eps, T,
                           D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m = (o_t < T)[:, None] & (o_d < D)[None, :]
    p = o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(x + p, mask=m, other=0.0).to(tl.float32)
    b_dy = tl.load(dy + p, mask=m, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    dot = tl.sum(b_dy * b_x, 1)
    dx_ = rstd[:, None] * b_dy - (rstd * rstd * rstd)[:, None] * b_x * dot[:, None]
    tl.store(dx + p, dx_, mask=m)


# --------------------------------------------------------------------------------------
# 4. sigmoid backward (y = sigmoid(x), dy -> dx = y*(1-y)*dy)
# --------------------------------------------------------------------------------------
@triton.jit
def _kda_sigmoid_bwd_kernel(y, dy, dx, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    b_y = tl.load(y + off, mask=m, other=0.0).to(tl.float32)
    b_dy = tl.load(dy + off, mask=m, other=0.0).to(tl.float32)
    tl.store(dx + off, b_dy * b_y * (1.0 - b_y), mask=m)


# --------------------------------------------------------------------------------------
# 5. M13-3 fused producers: write ``da`` (grad wrt pre-SiLU conv output) directly.
#
# The M13-2 host path materialised four fp32 [B,T,C] intermediates on the way to ``da``:
#   * ``dqk_norm = cat([dq, dk])`` (a torch CatArrayBatchedCopy),
#   * ``dqk_in`` (L2-norm backward output),
#   * ``dconv`` (assembled from ``dqk_in`` and ``dv`` via two slice-copies),
#   * ``da``   (SiLU backward output).
# Plus ``fd() = .detach().float().contiguous()`` casts of every incoming gradient.  A
# per-kernel profile (``harness/profile_port_pregdr_bwd.py``) shows those torch elementwise
# kernels are ~half the backward's device time.  These two kernels produce ``da`` in place,
# reading the raw (bf16/fp32) gradients and casting on load:
#
#   * qk channels: L2-norm backward of ``cat([dq,dk])`` composed with ``silu'(pre)``;
#   * v  channels: ``dv * silu'(pre)``.
# --------------------------------------------------------------------------------------
@triton.jit
def _kda_l2norm_silu_qk_bwd_kernel(dq, dk, x, pre, da, eps, ROWS,
                                   Hk: tl.constexpr, KD: tl.constexpr,
                                   C: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    i = tl.program_id(0)
    o_r = i * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_r = o_r < ROWS
    m = m_r[:, None] & (o_d < KD)[None, :]
    rmod = o_r % (2 * Hk)
    is_q = rmod < Hk
    is_k = rmod >= Hk
    tok = o_r // (2 * Hk)
    hh = rmod

    px = o_r[:, None] * KD + o_d[None, :]                       # x = query_key [rows, KD]
    b_x = tl.load(x + px, mask=m, other=0.0).to(tl.float32)

    hq = tok * Hk + hh                                          # valid when is_q
    hk = tok * Hk + (hh - Hk)                                   # valid when !is_q
    pdq = hq[:, None] * KD + o_d[None, :]
    pdk = hk[:, None] * KD + o_d[None, :]
    b_dy = tl.load(dq + pdq, mask=m & is_q[:, None], other=0.0).to(tl.float32) \
        + tl.load(dk + pdk, mask=m & is_k[:, None], other=0.0).to(tl.float32)

    rstd = 1.0 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    dot = tl.sum(b_dy * b_x, 1)
    dxk = rstd[:, None] * b_dy - (rstd * rstd * rstd)[:, None] * b_x * dot[:, None]

    pp = tok[:, None] * C + hh[:, None] * KD + o_d[None, :]     # pre/da qk region
    b_pre = tl.load(pre + pp, mask=m, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-b_pre))
    tl.store(da + pp, dxk * (sig * (1.0 + b_pre * (1.0 - sig))), mask=m)


@triton.jit
def _kda_silu_v_bwd_kernel(dv, pre, da, ROWS,
                           Hv: tl.constexpr, VD: tl.constexpr, C: tl.constexpr,
                           VOFF: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    i = tl.program_id(0)
    o_r = i * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_r = o_r < ROWS
    m = m_r[:, None] & (o_d < VD)[None, :]
    tok = o_r // Hv
    hh = o_r % Hv
    pdv = o_r[:, None] * VD + o_d[None, :]
    b_dy = tl.load(dv + pdv, mask=m, other=0.0).to(tl.float32)
    pp = tok[:, None] * C + VOFF + hh[:, None] * VD + o_d[None, :]
    b_pre = tl.load(pre + pp, mask=m, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-b_pre))
    tl.store(da + pp, b_dy * (sig * (1.0 + b_pre * (1.0 - sig))), mask=m)


# --------------------------------------------------------------------------------------
# Forward with saves (adds pre-SiLU conv output + conv input + pre-norm query_key)
# --------------------------------------------------------------------------------------
def _forward_raw_save(qkvfg, beta, conv_w, conv_b, dims):
    qk, vd, Hk, Hv, KD, VD, W, has_bias, activation = dims
    dev = qkvfg.device
    B, T = beta.shape[1], beta.shape[0]
    C = 2 * qk + vd

    qkvfg_t = qkvfg.transpose(0, 1)
    qkv, raw_g, gate = torch.split(qkvfg_t, (C, qk, vd), dim=-1)
    x = qkv.contiguous()
    conv = torch.empty(B, T, C, device=dev, dtype=qkvfg.dtype)
    pre = torch.empty(B, T, C, device=dev, dtype=qkvfg.dtype)

    BD = min(128, _next_pow2(C))
    BT = 64
    BW = _next_pow2(W)
    conv_b_eff = conv_b if conv_b is not None else x      # dummy pointer, unused when !HAS_BIAS
    grid = (triton.cdiv(C, BD), triton.cdiv(T, BT), B)
    _kda_conv_silu_kernel[grid](x, conv, conv_w, conv_b_eff, T, D=C, W=W, BT=BT, BW=BW, BD=BD,
                                HAS_BIAS=has_bias, ACTIVATION=activation)
    if activation:
        # same accumulation order, activation disabled -> exact pre-SiLU value
        _kda_conv_silu_kernel[grid](x, pre, conv_w, conv_b_eff, T, D=C, W=W, BT=BT, BW=BW, BD=BD,
                                    HAS_BIAS=has_bias, ACTIVATION=0)
    else:
        pre.copy_(conv)

    query_key = conv[..., :2 * qk].reshape(B, T, 2 * Hk, KD).contiguous()
    qk_norm = torch.empty_like(query_key)
    rows = B * T * 2 * Hk
    _kda_l2norm_kernel[(triton.cdiv(rows, 64),)](query_key, qk_norm, 1e-6, rows,
                                                 D=KD, BD=_next_pow2(KD), BT=64)
    query, key = torch.split(qk_norm, [Hk, Hk], dim=2)
    value = conv[..., 2 * qk:].reshape(B, T, Hv, VD)

    raw_g = raw_g.reshape(B, T, Hk, KD)
    beta_t = beta.transpose(0, 1).contiguous()
    beta_sig = torch.empty_like(beta_t, dtype=torch.float32)
    N = B * T * Hk
    _kda_beta_sigmoid_kernel[(triton.cdiv(N, 1024),)](beta_t, beta_sig, N, BLOCK=1024)

    ops = (query.contiguous(), key.contiguous(), value.contiguous(),
           gate.reshape(B, T, Hv, VD).contiguous(), beta_sig.contiguous(), raw_g.contiguous())
    saves = dict(x=x, pre=pre, query_key=query_key, beta_sig=beta_sig,
                 B=B, T=T, C=C, qk=qk, vd=vd, Hk=Hk, Hv=Hv, KD=KD, VD=VD, W=W)
    return ops, saves


# --------------------------------------------------------------------------------------
# Backward host
# --------------------------------------------------------------------------------------
def _backward_raw(qkvfg, beta, conv_w, saves, grads, dims, fused=True):
    qk, vd, Hk, Hv, KD, VD, W, has_bias, activation = dims
    dev = qkvfg.device
    B, T, C = saves["B"], saves["T"], saves["C"]
    x = saves["x"]                    # conv input [B,T,C]
    pre = saves["pre"]                # pre-SiLU conv output [B,T,C]
    query_key = saves["query_key"]    # pre-norm qk [B,T,2Hk,KD]
    beta_sig = saves["beta_sig"]      # [B,T,Hk] fp32

    dq, dk, dv, dgate, dbeta_sig, draw_g = grads
    z = lambda o: torch.zeros_like(o) if o is None else o
    dq, dk, dv = z(dq), z(dk), z(dv)
    dgate, dbeta_sig, draw_g = z(dgate), z(dbeta_sig), z(draw_g)

    Ctot = C + qk + vd
    if fused:
        # The producers index ``dq/dk/dv`` linearly (contiguous).  Autograd may hand a
        # *broadcast* gradient (e.g. ``sum(o)`` -> stride-0 expanded view); M13-2 hid that
        # behind ``fd(...).contiguous()``.  Materialise only when needed (no-op when dense).
        dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
        # --- fused producers: write ``da`` (grad wrt pre-SiLU conv output) in place.
        # qk channels: L2-norm bwd composed with SiLU' (no cat / dqk_in / dconv).
        da = torch.empty(B, T, C, device=dev, dtype=torch.float32)
        rows_qk = B * T * 2 * Hk
        _kda_l2norm_silu_qk_bwd_kernel[(triton.cdiv(rows_qk, 64),)](
            dq, dk, query_key, pre, da, 1e-6, rows_qk,
            Hk=Hk, KD=KD, C=C, BT=64, BD=_next_pow2(KD))
        # v channels: SiLU'.
        rows_v = B * T * Hv
        _kda_silu_v_bwd_kernel[(triton.cdiv(rows_v, 64),)](
            dv, pre, da, rows_v, Hv=Hv, VD=VD, C=C, VOFF=2 * qk, BT=64,
            BD=_next_pow2(VD))
        # --- conv backward writes dx straight into dqkvfg_t[..., :C] (no ``dx`` copy) ---
        dqkvfg_t = torch.empty(B, T, Ctot, device=dev, dtype=torch.float32)
        dweight = torch.zeros(C, W, device=dev, dtype=torch.float32)
        dbias = torch.zeros(C, device=dev, dtype=torch.float32)
        BD = min(128, _next_pow2(C))
        BT = 64
        grid = (triton.cdiv(C, BD), triton.cdiv(T, BT), B)
        _kda_conv_bwd_kernel[grid](da, x, conv_w, dqkvfg_t, dweight, dbias,
                                   T, Ctot, T * Ctot, 0, D=C, W=W, BT=BT, BD=BD)
    else:
        def fd(t):
            return t.detach().float().contiguous()

        # --- L2-norm backward: cat(dq,dk) over the [Hk,Hk] split -> grad wrt query_key ---
        dqk_norm = torch.cat([fd(dq), fd(dk)], dim=2).contiguous()   # [B,T,2Hk,KD] fp32
        dqk_in = torch.empty_like(dqk_norm)
        rows = B * T * 2 * Hk
        _kda_l2norm_bwd_kernel[(triton.cdiv(rows, 64),)](
            query_key, dqk_norm, dqk_in, 1e-6, rows, D=KD, BD=_next_pow2(KD), BT=64)

        # --- assemble grad wrt the post-SiLU conv output (all C channels) ---
        dconv = torch.empty(B, T, C, device=dev, dtype=torch.float32)
        dconv[..., :2 * qk] = dqk_in.reshape(B, T, 2 * qk)
        dconv[..., 2 * qk:] = fd(dv).reshape(B, T, vd)

        # --- SiLU backward -> da (fp32) ---
        da = torch.empty_like(dconv)
        Nc = B * T * C
        _kda_silu_bwd_kernel[(triton.cdiv(Nc, 1024),)](dconv, pre, da, Nc, BLOCK=1024)

        # --- conv backward -> dx / dweight / dbias ---
        dx = torch.empty(B, T, C, device=dev, dtype=torch.float32)
        dweight = torch.zeros(C, W, device=dev, dtype=torch.float32)
        dbias = torch.zeros(C, device=dev, dtype=torch.float32)
        BD = min(128, _next_pow2(C))
        BT = 64
        grid = (triton.cdiv(C, BD), triton.cdiv(T, BT), B)
        _kda_conv_bwd_kernel[grid](da, x, fd(conv_w), dx, dweight, dbias,
                                   T, C, T * C, 0, D=C, W=W, BT=BT, BD=BD)
        dqkvfg_t = torch.empty(B, T, Ctot, device=dev, dtype=torch.float32)
        dqkvfg_t[..., :C] = dx

    # --- sigmoid backward (casts its input on load) ---
    dbeta = torch.empty_like(beta_sig)
    Nb = B * T * Hk
    _kda_sigmoid_bwd_kernel[(triton.cdiv(Nb, 1024),)](
        beta_sig.reshape(-1), dbeta_sig.reshape(-1), dbeta.reshape(-1), Nb, BLOCK=1024)

    # --- assemble dqkvfg [T,B,Cin]: raw_g / gate pass-through copies (cast on copy_) ---
    dqkvfg_t[..., C:C + qk].copy_(draw_g.reshape(B, T, qk))
    dqkvfg_t[..., C + qk:].copy_(dgate.reshape(B, T, vd))
    dqkvfg = dqkvfg_t.transpose(0, 1).to(qkvfg.dtype)

    dbeta_out = dbeta.transpose(0, 1).to(beta.dtype)
    return dqkvfg, dbeta_out, dweight.view(dweight.shape[0], 1, dweight.shape[1]), dbias


class _FusedPreGDRNativeAutograd(torch.autograd.Function):
    """Drop-in autograd: Triton forward (M13-1) + **native Triton backward** (M13-2).

    ``conv1d.weight``/``conv1d.bias`` are function inputs, so their gradients reach the
    real parameters (M13-1's reference backward dropped them).
    """

    @staticmethod
    def forward(ctx, module, qkvfg, beta, conv_w, conv_b, batch, seq_len):
        dims = _dims(module)
        out, saves = _forward_raw_save(qkvfg, beta, conv_w, conv_b, dims)
        ctx.dims = dims
        ctx.saves = saves
        ctx.conv_w = conv_w
        ctx.has_conv_bias = conv_b is not None
        ctx.save_for_backward(qkvfg, beta)
        return out

    @staticmethod
    def backward(ctx, *grads):
        qkvfg, beta = ctx.saved_tensors
        dqkvfg, dbeta, dweight, dbias = _backward_raw(
            qkvfg, beta, ctx.conv_w, ctx.saves, grads, ctx.dims)
        dbias_out = dbias if ctx.has_conv_bias else None
        return (None, dqkvfg, dbeta, dweight, dbias_out, None, None)


def kda_pre_gdr_fused_native(module, qkvfg, beta, batch, seq_len):
    """Autograd-enabled drop-in with the native Triton backward (SBHD, cp=1, GVA=1)."""
    assert module.num_value_heads == module.num_key_heads, "KDA hard constraint"
    assert module.use_qk_l2norm
    out = _FusedPreGDRNativeAutograd.apply(
        module, qkvfg, beta, module.conv1d.weight, module.conv1d.bias, batch, seq_len)
    return out + (module.A_log, module.dt_bias)


# --------------------------------------------------------------------------------------
# Self-test (no Megatron)
# --------------------------------------------------------------------------------------
class _Stub(torch.nn.Module):
    def __init__(self, qk, vd, Hk, Hv, KD, VD, W, dtype):
        super().__init__()
        C = 2 * qk + vd
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
        self.conv1d = torch.nn.Conv1d(C, C, W, groups=C, padding=W - 1, bias=True).to(dtype)
        self.A_log = torch.randn(Hk)
        self.dt_bias = torch.randn(qk)


def _self_test():
    """Isolated backward check: drive the three backward kernels and compare against an
    analytic torch reference that consumes the *same* saved intermediates (so the
    comparison isolates the kernel implementation, not the forward backend)."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    Hk = Hv = 4
    KD = VD = 32
    W = 4
    qk, vd = Hk * KD, Hv * VD
    C = 2 * qk + vd
    dev = "cuda"

    for dtype in (torch.float32, torch.bfloat16):
        B, T = 2, 64
        print("  dtype=%s" % str(dtype).split(".")[-1])
        stub = _Stub(qk, vd, Hk, Hv, KD, VD, W, dtype).to(dev)
        torch.nn.init.normal_(stub.conv1d.weight, std=0.3)
        w = stub.conv1d.weight.detach()
        b = stub.conv1d.bias.detach()
        qkvfg = torch.randn(T, B, C + qk + vd, device=dev, dtype=dtype)
        beta = torch.randn(T, B, Hk, device=dev, dtype=dtype)
        dims = (qk, vd, Hk, Hv, KD, VD, W, True, 1)
        outs, saves = _forward_raw_save(qkvfg, beta, w, b, dims)
        up = [torch.randn(B, T, *s, device=dev)
              for s in ((Hk, KD), (Hk, KD), (Hv, VD), (Hv, VD), (Hk,), (Hk, KD))]
        dqkvfg, dbeta, dweight, dbias = _backward_raw(qkvfg, beta, w, saves, up, dims)

        # analytic torch reference on the same saved intermediates
        x = saves["x"].float()
        pre = saves["pre"].float()
        qk_in = saves["query_key"].float()
        dy = torch.cat([up[0].float(), up[1].float()], dim=2)
        rstd = 1 / torch.sqrt((qk_in * qk_in).sum(-1, keepdim=True) + 1e-6)
        dot = (dy * qk_in).sum(-1, keepdim=True)
        dqk_ref = rstd * dy - rstd ** 3 * qk_in * dot
        dconv = torch.cat([dqk_ref.reshape(B, T, 2 * qk), up[2].float().reshape(B, T, vd)], dim=2)
        sig = torch.sigmoid(pre)
        da = dconv * sig * (1 + pre * (1 - sig))
        xr = x.detach().clone().requires_grad_(True)
        wr = w.float().detach().clone().requires_grad_(True)
        br = b.float().detach().clone().requires_grad_(True)
        co = F.conv1d(xr.transpose(1, 2), wr, br, padding=W - 1, groups=C)[..., :T].transpose(1, 2)
        (da * co).sum().backward()
        dx_native = dqkvfg[:, :, :C].permute(1, 0, 2).float()
        dbeta_ref = up[4].float() * torch.sigmoid(beta.transpose(0, 1).float()) * \
            (1 - torch.sigmoid(beta.transpose(0, 1).float()))
        print("     dweight=%.3e  dbias=%.3e  dx=%.3e  dbeta=%.3e"
              % ((dweight.float() - wr.grad).abs().max().item(),
                 (dbias.float() - br.grad).abs().max().item(),
                 (dx_native - xr.grad).abs().max().item(),
                 (dbeta.permute(1, 0, 2).float() - dbeta_ref).abs().max().item()))


if __name__ == "__main__":
    _self_test()
