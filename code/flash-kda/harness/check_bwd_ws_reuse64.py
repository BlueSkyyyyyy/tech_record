"""P7-12: run the *whole* backward chain at 64-chunk granularity, sourced from FlashKDA's
16-chunk K1 workspace -- closing the last integration gap of docs/07 §13.5 (ii).

P6-8 (`check_bwd_ws_reuse.py`) drove `dAv -> B1 -> B2 -> B3 -> rcumsum(B0)` off the workspace
at CHUNK=16.  P7-10/P7-11 (`check_bwd_b1_rechunk.py`, `check_bwd_rechunk_pass.py`, docs/07
§13-§14) then showed B1@64 is correct and 2.1-2.3x faster, recovering the 64-chunk operands B1
needs (`qg/kg/w/g_cum/P64`) from the workspace *without* a forward recompute.  But that chain
stayed *mixed*: B2/B3 kept running on 16-chunk operands, so B1@64 only landed `dh` at the 64
boundaries while B2/B3 wanted it at every 16 boundary -- the "chain is not clean" caveat of
§13.5 (ii).

This script makes the chain **uniformly 64**: it recovers *every* operand B2/B3/`dAv` need at
64-chunk, from the 16-chunk workspace, and runs the full backward at `BT=64`:

  Part A  recovered 64 operands vs FLA forward@64
          (`qg/kg/w/g_cum/Aqk/Akk(P64)/v_new(U64)/h`), i.e. the "is the recovery faithful?"
          oracle at the resolution of the *intra* matrices too.
  Part B  full chain `dAv@64 -> B1@64 -> B2@64 -> B3@64 -> rcumsum(B0)` vs `torch.autograd`
          through `fla.ops.kda.chunk_kda`, six gradients, incl. non-divisible T / g=0 / B=2.
  Part C  backward-only timing: workspace->64 chain vs the P6-8 workspace->16 chain.

What is *new* vs P7-11: the 64 operands on top of `qg/kg/w/g_cum` are the **intra-attention
matrix** `Aqk64`, the WY inverse `Akk=P64`, the transformed value `U64 = v_new`, and the
chunk-input state `h64`.  All four come from the same 16-chunk fields:
  * `P64`  is the 4x4 block forward-substitution merge from the stored `P16` (P7-11, no new
    inverse);
  * `Aqk64` off-diagonal blocks are `qd16_i (kr_j 2^{e_ij})^T`, e_ij = sum_{s=j+1}^{i-1} G16_s
    (<= 0, folded into the *smaller* `kr` factor; diagonal blocks are the workspace `Mqk`);
  * `U64` and `h64` follow the same intra-block recurrence `u = P diag(beta)(v - k_d S)` at
    64, with the chunk-input state `h64 = h16[:, 0::4]` (the physical state is chunk-size
    independent, so it is just the 16-level state sampled at group boundaries).

Working precision in this torch prototype is fp64 (the `ki16 = kr 2^{-G16}` block can reach
~2^58); the fused Triton pass (P7-11) already folds it into fp32 (`_recover64_kernel`).  The
point here is the **end-to-end correctness at 64**, not the fusion.

Non-divisible T: everything is zero-padded to `Tp = NG*64` and the chain is run at `T=Tp`
(pad tokens have q=v=beta=g=0, so they are inert); the FLA golden sees the same padded inputs
so the comparison is apples-to-apples.

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed), idle card:
    CUDA_VISIBLE_DEVICES=1 python .../harness/check_bwd_ws_reuse64.py
    W64_MODE=correctness|timing
"""
import math
import os
import statistics
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_ws_reuse import (make_inputs, flash_forward, D, CHUNK, LOG2E,   # noqa: E402
                                cdiv, stats)
from check_bwd_b1_rechunk import b0_gate                                        # noqa: E402
from check_bwd_dhu_kernel import build_kernel as build_dhu, dhu_triton          # noqa: E402
from check_bwd_b1_bf16_e2e import build_dhu_bf16, dhu_bf16_triton               # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton    # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton    # noqa: E402
from fla.ops.utils import chunk_local_cumsum                                     # noqa: E402

S = CHUNK        # 16
BT = 64          # the re-chunk granule


# ------------------------------------------------------------------ 64-chunk operand recovery
def build64(fw, inp, lower_bound, scale, dt=torch.float64):
    """Recover *all* 64-chunk backward operands from the 16-chunk K1 workspace + B0 gate.

    Returns an FLA-`fi`-shaped dict, token-major and padded to ``Tp = NG*64``.  ``h`` is
    ``[B,NG,H,K,V]`` (chunk-input state) and ``Aqk/Akk`` are ``[B,Tp,H,64]`` row-major as the
    `dAv`/B2/B3 kernels expect (t index = intra-chunk column, last axis = row; see P6-8).
    """
    B, T, H, _ = inp["q"].shape
    dev = inp["q"].device
    NT16 = fw["NT"]
    Sg = BT // S
    NG = cdiv(NT16, Sg)
    NTp = NG * Sg
    Tp = NG * BT
    C = S
    g_cum16, beta16 = b0_gate(inp, lower_bound)             # [B,T,H,D], [B,T,H]  (log2)

    def pad_nt(x, last):
        z = torch.zeros(B, NTp, H, *last, dtype=dt, device=dev)
        z[:, :NT16] = x.to(dt)
        return z

    kd = pad_nt(fw["kd"], (C, D)).view(B, NG, Sg, H, C, D)
    qd = pad_nt(fw["qd"], (C, D)).view(B, NG, Sg, H, C, D)
    kr = pad_nt(fw["kr"], (C, D)).view(B, NG, Sg, H, C, D)
    P16g = pad_nt(fw["inv"], (C, C)).view(B, NG, Sg, H, C, C)
    M16g = pad_nt(fw["mqk"], (C, C)).view(B, NG, Sg, H, C, C)
    Et16 = pad_nt(fw["gt"], (D,))                           # [B,NTp,H,D]  (E_tot 16)

    # ---- per-sub-chunk gate totals G16 and within-group offsets ----
    gc16_full = torch.zeros(B, NTp * C, H, D, dtype=dt, device=dev)
    gc16_full[:, :T] = g_cum16.to(dt)[:, :T]
    if NT16 * C > T:                                        # carry K1's padded tile tail
        gc16_full[:, T:NT16 * C] = g_cum16.to(dt)[:, T - 1:T]
    g16r = gc16_full.view(B, NTp, C, H, D)
    gate_tok = torch.empty_like(g16r)
    gate_tok[:, :, 0] = g16r[:, :, 0]
    gate_tok[:, :, 1:] = g16r[:, :, 1:] - g16r[:, :, :-1]
    g64 = torch.zeros(B, Tp, H, D, dtype=dt, device=dev)
    g64[:, :NT16 * C] = gate_tok[:, :NT16].reshape(B, NT16 * C, H, D)
    gc4 = g64.view(B, NG, Sg, C, H, D).cumsum(3)            # [B,NG,Sg,C,H,D] 16-local log2
    G16g = gc4[:, :, :, -1]                                 # [B,NG,Sg,H,D]
    offg = torch.zeros_like(G16g)
    if Sg > 1:
        offg[:, :, 1:] = G16g[:, :, :-1].cumsum(2)
    G64 = offg[:, :, -1] + G16g[:, :, -1]                   # [B,NG,H,D] group gate total
    E64 = torch.exp2(G64)

    def tok(x):                                             # [B,NG,Sg,H,C,D] -> [B,Tp,H,D]
        return x.permute(0, 1, 2, 4, 3, 5).reshape(B, Tp, H, D)

    off_f = offg.unsqueeze(4)                               # [B,NG,Sg,H,1,D]
    qg64 = tok(qd * torch.exp2(off_f)) / scale
    exp_kg = (G64[:, :, None, :, :] - offg - G16g).unsqueeze(4)
    kg64 = tok(kr * torch.exp2(exp_kg))                     # exponent <= 0, bounded
    g_cum64 = (gc4 + offg.unsqueeze(3)).reshape(B, Tp, H, D)   # offg: [.,Sg,1,H,D]

    beta_tok = torch.zeros(B, Tp, H, dtype=dt, device=dev)
    beta_tok[:, :T] = beta16.to(dt)
    beta_g = beta_tok.view(B, NG, BT, H).permute(0, 1, 3, 2)     # [B,NG,H,BT]
    beta_sg = beta_tok.view(B, NG, Sg, C, H)                     # [B,NG,Sg,C,H]

    # ---- 64x64 intra blocks: off-diagonal A (L) and Aqk ----
    A64 = torch.zeros(B, NG, H, BT, BT, dtype=dt, device=dev)
    Aq64 = torch.zeros(B, NG, H, BT, BT, dtype=dt, device=dev)
    for s in range(Sg):
        Aq64[:, :, :, s * C:(s + 1) * C, s * C:(s + 1) * C] = M16g[:, :, s]
    for st in range(Sg):
        for su in range(st):
            if st - su > 1:
                e = G16g[:, :, su + 1:st].sum(2)            # [B,NG,H,D], <= 0
                krs = kr[:, :, su] * torch.exp2(e).unsqueeze(3)
            else:
                krs = kr[:, :, su]
            A64[:, :, :, st * C:(st + 1) * C, su * C:(su + 1) * C] = \
                torch.einsum("bnhtd,bnhud->bnhtu", kd[:, :, st], krs)
            Aq64[:, :, :, st * C:(st + 1) * C, su * C:(su + 1) * C] = \
                torch.einsum("bnhtd,bnhud->bnhtu", qd[:, :, st], krs)

    # ---- P64 = (I+L64)^{-1} via 4x4 block forward-substitution (reuses stored P16) ----
    L64 = torch.tril(A64, -1) * beta_g[..., None, :]

    def blk(X, st, su):
        return X[..., st * C:(st + 1) * C, su * C:(su + 1) * C]

    P0, P1, P2, P3 = P16g[:, :, 0], P16g[:, :, 1], P16g[:, :, 2], P16g[:, :, 3]
    L10, L20, L21 = blk(L64, 1, 0), blk(L64, 2, 0), blk(L64, 2, 1)
    L30, L31, L32 = blk(L64, 3, 0), blk(L64, 3, 1), blk(L64, 3, 2)
    P10 = -P1 @ L10 @ P0
    P21 = -P2 @ L21 @ P1
    P32 = -P3 @ L32 @ P2
    P20 = -P2 @ (L20 @ P0 + L21 @ P10)
    P31 = -P3 @ (L31 @ P1 + L32 @ P21)
    P30 = -P3 @ (L30 @ P0 + L31 @ P10 + L32 @ P20)
    P64 = torch.zeros(B, NG, H, BT, BT, dtype=dt, device=dev)
    for s, P in enumerate([P0, P1, P2, P3]):
        P64[..., s * C:(s + 1) * C, s * C:(s + 1) * C] = P
    P64[..., 1 * C:2 * C, 0:1 * C] = P10
    P64[..., 2 * C:3 * C, 0:1 * C] = P20
    P64[..., 2 * C:3 * C, 1 * C:2 * C] = P21
    P64[..., 3 * C:4 * C, 0:1 * C] = P30
    P64[..., 3 * C:4 * C, 1 * C:2 * C] = P31
    P64[..., 3 * C:4 * C, 2 * C:3 * C] = P32

    # w64 = P64 diag(beta) kd64
    kd64r = (kd * torch.exp2(off_f)).permute(0, 1, 3, 2, 4, 5).reshape(B, NG, H, BT, D)
    w64r = torch.einsum("bnhtu,bnhud->bnhtd", P64, kd64r * beta_g[..., None])
    w64 = w64r.permute(0, 1, 3, 2, 4).reshape(B, Tp, H, D)

    # ---- chunk-input state h64 = h16[:, 0::Sg] (physical state, chunk-size independent) ----
    v_pad = torch.zeros(B, Tp, H, D, dtype=dt, device=dev)
    v_pad[:, :T] = inp["v"].to(dt)[:, :T]
    v4 = v_pad.view(B, NTp, C, H, D).permute(0, 1, 3, 2, 4)      # [B,NTp,H,C,D]
    beta_ch = beta_tok.view(B, NTp, C, H).permute(0, 1, 3, 2)    # [B,NTp,H,C]
    kd16v = kd.reshape(B, NTp, H, C, D)
    kr16v = kr.reshape(B, NTp, H, C, D)
    P16v = P16g.reshape(B, NTp, H, C, C)
    St = inp["h0"].to(dt).permute(0, 1, 3, 2).contiguous()  # [B,H,K,V] from S^T=[B,H,V,K]
    h16 = torch.zeros(B, NTp, H, D, D, dtype=dt, device=dev)
    for c in range(NTp):
        h16[:, c] = St
        pred = torch.einsum("bhtk,bhkv->bhtv", kd16v[:, c], St)
        u = torch.einsum("bhtu,bhuv->bhtv", P16v[:, c],
                         (v4[:, c] - pred) * beta_ch[:, c].unsqueeze(-1))
        St = Et16[:, c, :, :, None] * St + torch.einsum("bhtk,bhtv->bhkv", kr16v[:, c], u)
    h64 = h16[:, 0::Sg]                                     # [B,NG,H,K,V]

    # ---- transformed value U64 = P64 diag(beta)(v - kd64 h64) ----
    v_g = v_pad.view(B, NG, BT, H, D).permute(0, 1, 3, 2, 4)     # [B,NG,H,BT,D]
    U64 = torch.zeros(B, Tp, H, D, dtype=dt, device=dev)
    for g in range(NG):
        pred = torch.einsum("bhtk,bhkv->bhtv", kd64r[:, g], h64[:, g])
        u = torch.einsum("bhtu,bhuv->bhtv", P64[:, g],
                         (v_g[:, g] - pred) * beta_g[:, g].unsqueeze(-1))
        U64[:, g * BT:(g + 1) * BT] = u.permute(0, 2, 1, 3)

    Aqk_rm = Aq64.permute(0, 1, 3, 2, 4).reshape(B, Tp, H, BT)
    Akk_rm = P64.permute(0, 1, 3, 2, 4).reshape(B, Tp, H, BT)

    fi = dict(qg=qg64, kg=kg64, w=w64, g_cum=g_cum64, beta=beta_tok,
              Aqk=Aqk_rm, Akk=Akk_rm, v_new=U64, h=h64)
    return fi, dict(E64=E64, Tp=Tp, NG=NG, NT16=NT16)


# ------------------------------------------------------------------------- chain at 64
_KERNELS = {}


def kernels():
    """Build every Triton kernel once (Triton JIT-builds per call otherwise -> host tax)."""
    if not _KERNELS:
        from check_bwd_e2e import build_dav_kernel
        _KERNELS.update(dav=build_dav_kernel(), dhu=build_dhu(), dhu_bf16=build_dhu_bf16(),
                        dqkg=build_dqkg(), intra=build_intra())
    return _KERNELS


def run_chain64(fi, qn, kn, v, h0T, do, dfin, scale, b1_bf16=True):
    from check_bwd_e2e import dav_triton

    # the recovery is fp64; the chain kernels consume fp32/bf16 (P6-8 fed fp32 operands)
    fi = {k: v.float() for k, v in fi.items()}
    K = kernels()
    dAqk, dv = dav_triton(K["dav"], fi["v_new"], fi["Aqk"], do, scale, BT)

    dqkg_k, intra_k = K["dqkg"], K["intra"]
    if b1_bf16:
        dhu_k = K["dhu_bf16"]
        fi_b = dict(fi)
        for kk in ("qg", "kg", "w"):
            fi_b[kk] = fi[kk].to(torch.bfloat16)
        dh, dh0, dv2 = dhu_bf16_triton(dhu_k, fi_b, do.to(torch.bfloat16), dv, scale,
                                       BT, dht=dfin)
    else:
        dh, dh0, dv2 = dhu_triton(K["dhu"], fi, do, dv, h0T, scale, BT, dht=dfin)

    inp2 = dict(q=qn, k=kn, v=v, v_new=fi["v_new"], g=fi["g_cum"], beta=fi["beta"],
                A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
    b2 = wy_dqkg_triton(dqkg_k, inp2, scale, BT)
    inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=fi["beta"], dAqk=dAqk, dAkk=b2["dA"])
    b3 = intra_triton(intra_k, inp3, BT, BT)
    dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=BT, reverse=True)
    return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"], dv=b2["dv"], dg=dg,
                dbeta=b2["db"] + b3["db"], dh0=dh0, dh=dh, dv2=dv2, dAqk=dAqk)


# ------------------------------------------------------------------ FLA forward / golden
def fla_forward64(qn, kn, v, beta_act, g_fla, h0, scale, BT=64):
    """FLA forward@64 on the *natural-log* gate `g_fla` (same convention as P6-8/P7-10:
    FLA's chunk-local cumsum with scale=RCP_LN2 turns it into the log2 `g_cum`)."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    from fla.ops.utils import chunk_local_cumsum

    g_cum = chunk_local_cumsum(g_fla, scale=1.4426950408889634, chunk_size=BT)
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=qn, k=kn, v=v, gk=g_cum, beta=beta_act, scale=scale, chunk_size=BT,
        disable_recompute=True)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g_cum, initial_state=h0, output_final_state=True, chunk_size=BT)
    return dict(g_cum=g_cum, w=w, u=u, qg=qg, kg=kg, Aqk=Aqk, Akk=Akk,
                v_new=v_new, h=h, final_state=final_state)


def fla_golden(qn, kn, v, beta_act, g_fla, h0, scale, do, dfin):
    from fla.ops.kda import chunk_kda
    q = qn.clone().requires_grad_(True)
    k = kn.clone().requires_grad_(True)
    vv = v.clone().requires_grad_(True)
    beta = beta_act.clone().requires_grad_(True)
    g = g_fla.clone().requires_grad_(True)
    hh = h0.clone().requires_grad_(True)
    o, s = chunk_kda(q, k, vv, g, beta, scale=scale,
                     initial_state=hh.float().transpose(-1, -2), output_final_state=True,
                     use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                     use_beta_sigmoid_in_kernel=False)
    loss = (o.float() * do).sum() + (s.float() * dfin).sum()
    loss.backward()
    return dict(q=q.grad, k=k.grad, v=vv.grad, beta=beta.grad, g=g.grad, h0=hh.grad)


# --------------------------------------------------------------------------- prep
def pad_tok(x, Tp):
    if x.shape[1] == Tp:
        return x
    z = torch.zeros(x.shape[0], Tp, *x.shape[2:], device=x.device, dtype=x.dtype)
    z[:, :x.shape[1]] = x
    return z


def prep(B, T, H, scale, lower_bound, seed=0, g_scale=1.0):
    inp = make_inputs(B, T, H, seed, "cuda", g_scale)
    fw = flash_forward(inp, scale, lower_bound, "cuda")
    fi, diag = build64(fw, inp, lower_bound, scale)
    Tp = diag["Tp"]
    torch.manual_seed(seed + 77)
    do = torch.randn(B, T, H, D, device="cuda")
    dfin = torch.randn(B, H, D, D, device="cuda") * 0.01

    qn = torch.nn.functional.normalize(inp["q"].float(), dim=-1)
    kn = torch.nn.functional.normalize(inp["k"].float(), dim=-1)
    beta_act = torch.sigmoid(inp["beta"].float())
    inner = inp["g"].float() + inp["dt_bias"].view(1, 1, H, D)
    g_fla = lower_bound * torch.sigmoid(torch.exp(inp["A_log"].view(1, 1, H, 1)) * inner)

    # everything padded to Tp (pad tokens inert: q=v=beta=g=0); the chain runs at T=Tp
    qnp, knp, vp = pad_tok(qn, Tp), pad_tok(kn, Tp), pad_tok(inp["v"].float(), Tp)
    bp, gp = pad_tok(beta_act, Tp), pad_tok(g_fla, Tp)
    dop = pad_tok(do, Tp)
    h0 = inp["h0"]
    return inp, fw, fi, diag, qnp, knp, vp, bp, gp, dop, dfin, h0


# --------------------------------------------------------------------------- parts
def part_a(cases, scale, lower_bound):
    print("=" * 92)
    print("Part A: recovered 64-chunk operands (16-chunk workspace) vs FLA forward@64")
    print("=" * 92)
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fw, fi, diag, qnp, knp, vp, bp, gp, dop, dfin, h0 = prep(
            B, T, H, scale, lower_bound, seed, gs)
        Tp = diag["Tp"]
        # FLA's `initial_state` is [K,V]; the harness `h0` is FlashKDA's S^T == [V,K] (P4-2).
        F = fla_forward64(qnp.to(torch.bfloat16), knp.to(torch.bfloat16),
                          vp.to(torch.bfloat16), bp.to(torch.bfloat16),
                          gp.to(torch.bfloat16), h0.transpose(-1, -2), scale, BT=64)
        print(f"\n# {name}  B={B} T={T} H={H}  Tp={Tp}  NT64={diag['NG']}")
        # Token-major fields: compare only the T valid tokens (pad region is inert for the
        # chain; FLA's pad tiles can hold uninitialised memory).
        pairs = [("qg", "qg", "qg"), ("kg", "kg", "kg"), ("w", "w", "w"),
                 ("g_cum", "g_cum", "g_cum"), ("v_new(U64)", "v_new", "v_new"),
                 ("Akk(P64)", "Akk", "Akk")]
        for label, rec_key, fla_key in pairs:
            a = fi[rec_key].float()[:, :T]
            b = F[fla_key].float()[:, :T]
            b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
            _, mr = stats(f"rec64  {label:12s}", a, b)
            worst[label] = max(worst.get(label, 0.0), mr)
        # h: chunk 0 is the passed initial state (trivial); compare the *computed* states.
        NT64 = diag["NG"]
        a = fi["h"].float()[:, 1:NT64]
        b = F["h"].float()[:, 1:NT64]
        _, mr = stats("rec64  h(S_in)      ", a, b)
        worst["h(S_in)"] = max(worst.get("h(S_in)", 0.0), mr)
        # `Aqk` is deliberately *not* tabulated against FLA's raw buffer: FLA stores the
        # defined (lower-triangular) `M_qk` into one triangle of a [BT,BT] tile and leaves
        # the other uninitialised, and its `q_dec @ k_inv^T` overflows fp32 in deep-decay
        # chunks (the very overflow P7-10 §13.4 folds `2^{e_ij}` to avoid).  Its recovery is
        # validated end-to-end by Part B (`dAv`/B3 consume it).
    print("\nworst-case max_rel across cases:")
    for label in worst:
        print(f"  {label:12s} {worst[label]:.3e}")


def part_b(cases, scale, lower_bound):
    print("\n" + "=" * 92)
    print("Part B: full chain @64 (workspace-recovered) vs torch.autograd(FLA), T=Tp")
    print("=" * 92)
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        inp, fw, fi, diag, qnp, knp, vp, bp, gp, dop, dfin, h0 = prep(
            B, T, H, scale, lower_bound, seed, gs)
        Tp = diag["Tp"]
        h0T = h0.float().permute(0, 1, 3, 2).contiguous()
        ours = run_chain64(fi, qnp, knp, vp, h0T, dop, dfin, scale, b1_bf16=True)
        gold = fla_golden(qnp, knp, vp, bp, gp, h0, scale, dop, dfin)
        print(f"\n# {name}  B={B} T={T} H={H}  Tp={Tp}")
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"),
                        ("dg", "g")]:
            _, mr = stats(f"64 {fld:6s}", ours[fld][:, :T], gold[gk][:, :T])
            worst[fld] = max(worst.get(fld, 0.0), mr)
        _, mr = stats("64 dh0   ", ours["dh0"], gold["h0"].transpose(-1, -2))
        worst["dh0"] = max(worst.get("dh0", 0.0), mr)
    print("\nworst-case max_rel across cases (chain@64 vs autograd):")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        print(f"  {fld:6s} {worst[fld]:.3e}")


def part_c(cases, scale, lower_bound):
    print("\n" + "=" * 92)
    print("Part C: backward-only timing — workspace->64 chain vs workspace->16 chain (P6-8)")
    print("=" * 92)
    print(f"  {'case':18s} | {'ours64 wall':>11s} {'ours64 dev':>10s} | "
          f"{'ours16 wall':>11s} {'ours16 dev':>10s} | {'dev 64/16':>9s}")
    for (name, B, T, H, seed, gs) in cases:
        inp, fw, fi, diag, qnp, knp, vp, bp, gp, dop, dfin, h0 = prep(
            B, T, H, scale, lower_bound, seed, gs)
        h0T = h0.float().permute(0, 1, 3, 2).contiguous()

        from check_bwd_ws_reuse import reconstruct
        _, _, fi16 = reconstruct(fw, inp, lower_bound, scale)
        fi16 = dict(fi16)
        fi16["beta"] = torch.sigmoid(inp["beta"].float())

        def run16():
            return run_chain16(fw, inp, fi16, qnp[:T], knp[:T], inp["v"].float(),
                               h0T, dop[:T], dfin, scale)

        def run64():
            return run_chain64(fi, qnp, knp, vp, h0T, dop, dfin, scale, b1_bf16=True)

        run16(); run64()
        torch.cuda.synchronize()
        w16, d16, p16 = time_wall_dev(run16)
        w64, d64, p64 = time_wall_dev(run64)
        print(f"  {name:18s} | {w64:11.1f} {d64:10.1f} | {w16:11.1f} {d16:10.1f} | "
              f"{d64/d16:9.2f}")
        keys = ["_dav_kernel", "_dhu_kernel", "wy_dqkg", "intra", "chunk_local_cumsum"]
        for label, prof in [("64", p64), ("16", p16)]:
            agg = {}
            for ev in prof.key_averages():
                for kk in keys:
                    if kk in ev.key:
                        agg[kk] = agg.get(kk, 0.0) + ev.self_device_time_total / 15
            print(f"       {label}: " + "  ".join(f"{k}={agg.get(k, 0):.1f}" for k in keys))


def run_chain16(fw, inp, fi, qn, kn, v, h0T, do, dfin, scale):
    """The P6-8 chain at CHUNK=16 (baseline for Part C)."""
    from check_bwd_e2e import dav_triton
    K = kernels()
    dAqk, dv = dav_triton(K["dav"], fi["v_new"], fi["Aqk"], do, scale, CHUNK)
    fi_b = dict(fi)
    for kk in ("qg", "kg", "w"):
        fi_b[kk] = fi[kk].to(torch.bfloat16)
    dh, dh0, dv2 = dhu_bf16_triton(K["dhu_bf16"], fi_b, do.to(torch.bfloat16), dv, scale,
                                   CHUNK, dht=dfin)
    dqkg_k, intra_k = K["dqkg"], K["intra"]
    inp2 = dict(q=qn, k=kn, v=v, v_new=fi["v_new"], g=fi["g_cum"], beta=fi["beta"],
                A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
    b2 = wy_dqkg_triton(dqkg_k, inp2, scale, CHUNK)
    inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=fi["beta"], dAqk=dAqk, dAkk=b2["dA"])
    b3 = intra_triton(intra_k, inp3, CHUNK, CHUNK)
    dg = chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=CHUNK, reverse=True)
    return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"], dv=b2["dv"], dg=dg,
                dbeta=b2["db"] + b3["db"], dh0=dh0)


# --------------------------------------------------------------------------- timing
def time_wall_dev(fn, reps=30, rounds=3, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    med = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        med.append(s.elapsed_time(e) / reps)
    wall = statistics.median(med) * 1e3
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(15):
            fn()
        torch.cuda.synchronize()
    dev = sum(ev.self_device_time_total for ev in prof.key_averages()) / 15
    return wall, dev, prof


def main():
    assert torch.cuda.is_available(), "need GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    ccases = [
        ("T=256 H=2", 1, 256, 2, 0, 1.0),
        ("T=250 H=3 non-div", 1, 250, 3, 2, 1.0),
        ("B2 T=513 H=2 non-div", 2, 513, 2, 4, 1.0),
        ("T=192 H=2 g=0", 1, 192, 2, 5, 0.0),
    ]
    tcases = [
        ("T=512 H=4", 1, 512, 4, 0, 1.0),
        ("B2 T=1024 H=8", 2, 1024, 8, 1, 1.0),
        ("B8 T=2048 H=8", 8, 2048, 8, 2, 1.0),
        ("T=250 H=3 nondiv", 1, 250, 3, 3, 1.0),
    ]
    mode = os.environ.get("W64_MODE", "")
    if mode != "timing":
        part_a(ccases, scale, lower_bound)
        part_b(ccases, scale, lower_bound)
    if mode != "correctness":
        part_c(tcases, scale, lower_bound)
    print("\nDONE")


if __name__ == "__main__":
    main()
