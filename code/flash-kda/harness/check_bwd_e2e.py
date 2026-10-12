"""P6-7 evidence: end-to-end composition of the minimal backward kernels.

Sections §6.2-§6.8 (P6-2..P6-6) landed each backward segment *in isolation*:
B1 state-adjoint (`dhu`), B2 `wy_dqkg`, B3 `intra`, B0 gate.  Each was compared to its
FLA sub-kernel with FLA-provided intermediate inputs.  That never exercised the *chain*
(the output of one segment feeding the input of the next) nor the one segment that had
not been written yet: the **`dAv` output branch** (FLA `chunk_kda_bwd_kernel_dAv`,
fla/ops/kda/chunk_bwd.py:44-114).

This script closes both gaps:

  Part A  a minimal `dAv` Triton kernel  vs  FLA `chunk_kda_bwd_dAv`
          (dAqk = do @ U^T * scale ; dv = M_qk @ do).
  Part B  full chain  dAv -> B1(dhu) -> B2(wy_dqkg) -> B3(intra) -> reverse-cumsum(B0)
          run on the *real* FLA forward intermediates, compared against
          ``torch.autograd`` through ``fla.ops.kda.chunk_kda`` for all six gradients
          dq, dk, dv, dg, dbeta, dh0.

The chain is what a target-card backward has to reproduce: dAv -> B1 -> B2 -> B3 are
the four GEMM stages of §5.2, and the tail reverse-cumsum is the B0 gate chain (already
validated on its own in §6.8).

Run inside `kernel_lab` (GPU):
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_e2e.py
"""
import math
import os
import sys

# FLA >=0.5 auto-selects a TileLang backend for the KDA backward; docs/06 reads the
# Triton reference, so pin the dispatch to it (same trap as §6.6).
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
SRC = "~/proj/tech_record/code/flash-kda/src"
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dhu_kernel import build_kernel as build_dhu, dhu_triton          # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton    # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton    # noqa: E402

RCP_LN2 = 1.4426950408889634
LN2 = math.log(2.0)


def cdiv(a, b):
    return (a + b - 1) // b


def stats(name, pred, gold):
    d = (pred.float() - gold.float()).abs()
    denom = gold.float().abs().max().item()
    max_abs = d.max().item()
    max_rel = max_abs / (denom + 1e-12)
    print(f"  {name:26s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  ref_max={denom:.3e}")
    return max_abs, max_rel


# --------------------------------------------------------------------------- inputs
def make_inputs(B, T, H, D, scale, lower_bound, device, seed=0, use_gate=False):
    torch.manual_seed(seed)
    q_raw = torch.randn(B, T, H, D, device=device) * 0.5
    k_raw = torch.randn(B, T, H, D, device=device) * 0.5
    v_raw = torch.randn(B, T, H, D, device=device)
    beta_raw = torch.randn(B, T, H, device=device)
    g_logit = torch.randn(B, T, H, D, device=device)          # raw gate logits
    A_log = torch.randn(H, device=device)
    dt_bias = torch.randn(H, D, device=device)
    h0 = torch.randn(B, H, D, D, device=device) * 0.5
    return dict(q_raw=q_raw, k_raw=k_raw, v_raw=v_raw, beta_raw=beta_raw,
                g_logit=g_logit, A_log=A_log, dt_bias=dt_bias, h0=h0)


def derive_operands(inp, lower_bound):
    """Reuse `check_bwd_fla`'s setup so FLA and our chain act on identical numerics."""
    from kda_ref import l2_normalize, kda_gate, kda_beta
    qn = l2_normalize(inp["q_raw"].float())                   # [B,T,H,D] (fp32)
    kn = l2_normalize(inp["k_raw"].float())
    beta_act = kda_beta(inp["beta_raw"].float())              # sigmoid, [B,T,H]
    g_dec = kda_gate(inp["g_logit"].float(), inp["A_log"].float(),
                     inp["dt_bias"].float(), lower_bound)     # log2 <= 0, [B,T,H,D]
    return qn, kn, beta_act, g_dec


# ---------------------------------------------------------------- FLA forward/setup
def fla_forward_intermediates(qn, kn, v, beta_act, g_dec, h0, scale, BT):
    """The exact forward quantities the backward consumes (mirrors §6.1 `fla_forward`)."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    from fla.ops.utils import chunk_local_cumsum

    g_fla = g_dec * LN2
    g_cum = chunk_local_cumsum(g_fla, scale=RCP_LN2, chunk_size=BT)   # log2 chunk cumsum
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=qn, k=kn, v=v, gk=g_cum, beta=beta_act, scale=scale, chunk_size=BT,
        disable_recompute=True)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g_cum, initial_state=h0, output_final_state=True,
        chunk_size=BT)
    return dict(g_cum=g_cum, w=w, u=u, qg=qg, kg=kg, Aqk=Aqk, Akk=Akk,
                v_new=v_new, h=h, final_state=final_state)


def fla_dav(qn, kn, v_new, do, Aqk, scale, BT):
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv
    dA, dv = chunk_kda_bwd_dAv(q=qn, k=kn, v=v_new, do=do, A=Aqk, scale=scale,
                               chunk_size=BT)
    return dA, dv


def fla_chain_grads(fi, qn, kn, v, beta_act, h0, do, dht, scale, BT):
    """FLA's own internal decomposition, returning every intermediate (P6-1 map)."""
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv, chunk_kda_bwd_wy_dqkg_fused
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
    from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
    from fla.ops.utils import chunk_local_cumsum

    dAqk, dv = chunk_kda_bwd_dAv(q=qn, k=kn, v=fi["v_new"], do=do, A=fi["Aqk"],
                                 scale=scale, chunk_size=BT)
    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
        q=fi["qg"], k=fi["kg"], w=fi["w"], gk=fi["g_cum"], h0=h0, dht=dht,
        do=do, dv=dv, scale=scale, chunk_size=BT)
    dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
        q=qn, k=kn, v=v, v_new=fi["v_new"], g=fi["g_cum"], beta=beta_act, A=fi["Akk"],
        h=fi["h"], do=do, dh=dh, dv=dv2, scale=scale, chunk_size=BT)
    dg_cum = dg.clone()
    dq, dk, db, dg = chunk_kda_bwd_intra(
        q=qn, k=kn, g=fi["g_cum"], beta=beta_act, dAqk=dAqk, dAkk=dAkk,
        dq=dq, dk=dk, db=db, dg=dg, chunk_size=BT, safe_gate=False)
    dg = chunk_local_cumsum(dg, chunk_size=BT, reverse=True)
    return dict(dq=dq, dk=dk, dv=dv, db=db, dg=dg, dh0=dh0,
                dh=dh, dv2=dv2, dg_cum=dg_cum, dAqk=dAqk, dAkk=dAkk)


# ------------------------------------------------------------------- dAv minimal kernel
def build_dav_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _dav_kernel(
        v_ptr, A_ptr, do_ptr, dv_ptr, dA_ptr, T, scale,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    ):
        i_t = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // HV
        i_hv = i_bh % HV

        # non-varlen: one sequence per batch row, bos = i_b * T
        v_ptr += (i_b * T * HV + i_hv) * V
        do_ptr += (i_b * T * HV + i_hv) * V
        dv_ptr += (i_b * T * HV + i_hv) * V
        A_ptr += (i_b * T * HV + i_hv) * BT
        dA_ptr += (i_b * T * HV + i_hv) * BT

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        o_A = tl.arange(0, BT)
        m_AT = (o_A[:, None] < BT) & m_t[None, :]
        # A[o_A, o_t]; M_qk = upper-inclusive in this orientation (fla chunk_bwd.py:83-90)
        p_A = A_ptr + o_A[:, None] + o_t[None, :] * (HV * BT)
        b_A = tl.load(p_A, mask=m_AT, other=0.0)
        m_A = (o_t[:, None] <= o_t[None, :]) & m_t[:, None] & m_t[None, :]
        b_A = tl.where(m_A, b_A, 0.0).to(do_ptr.dtype.element_ty)

        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = o_v < V
            m_vT = m_v[:, None] & m_t[None, :]
            m_tv = m_t[:, None] & m_v[None, :]
            p_v = v_ptr + o_v[:, None] + o_t[None, :] * (HV * V)         # [BV, BT] = v^T
            p_do = do_ptr + o_t[:, None] * (HV * V) + o_v[None, :]       # [BT, BV]
            p_dv = dv_ptr + o_t[:, None] * (HV * V) + o_v[None, :]
            b_v = tl.load(p_v, mask=m_vT, other=0.0)
            b_do = tl.load(p_do, mask=m_tv, other=0.0)
            b_dA += tl.dot(b_do, b_v)                                    # do @ v^T
            b_dv = tl.dot(b_A, b_do.to(b_A.dtype))                       # M_qk @ do
            tl.store(p_dv, b_dv.to(dv_ptr.dtype.element_ty), mask=m_tv)

        m_dA = m_t[:, None] & (o_A[None, :] < BT)
        p_dA = dA_ptr + o_t[:, None] * (HV * BT) + o_A[None, :]
        b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.0)
        tl.store(p_dA, b_dA.to(dA_ptr.dtype.element_ty), mask=m_dA)

    return _dav_kernel


def dav_triton(kernel, v_new, Aqk, do, scale, BT, BV=64):
    B, T, HV, V = do.shape
    K = Aqk.shape[-2] if Aqk.dim() == 5 else None
    H = HV  # test config has H == HV
    NT = cdiv(T, BT)
    dev = do.device
    dv = torch.empty_like(do)
    dA = torch.empty(B, T, HV, BT, device=dev, dtype=torch.float32)
    grid = (NT, B * HV)
    kernel[grid](
        v_new.contiguous(), Aqk.contiguous(), do.contiguous(), dv, dA, T, scale,
        H=H, HV=HV, K=64, V=V, BT=BT, BK=32, BV=BV,
        num_warps=4, num_stages=1)
    return dA, dv


# -------------------------------------------------------------------------- runner
def run_case(B, T, H, D, BT, BC, scale, lower_bound, device, seed=0):
    from fla.ops.kda import chunk_kda

    inp = make_inputs(B, T, H, D, scale, lower_bound, device, seed=seed)
    qn, kn, beta_act, g_dec = derive_operands(inp, lower_bound)
    v = inp["v_raw"].float()
    h0 = inp["h0"].float()

    torch.manual_seed(seed + 77)
    do = torch.randn(B, T, H, D, device=device)
    dfin = torch.randn(B, H, D, D, device=device) * 0.01

    # ---- FLA autograd golden (same fp32 operands, all in-kernel transforms off) ----
    qa = qn.clone().requires_grad_(True)
    ka = kn.clone().requires_grad_(True)
    va = v.clone().requires_grad_(True)
    ba = beta_act.clone().requires_grad_(True)
    ga = (g_dec * LN2).clone().requires_grad_(True)          # natural-log decay
    ha = h0.clone().requires_grad_(True)
    o, s = chunk_kda(qa, ka, va, ga, ba, scale=scale, initial_state=ha,
                     output_final_state=True, use_qk_l2norm_in_kernel=False,
                     use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
    loss = (o * do).sum() + (s * dfin).sum()
    loss.backward()
    gold = dict(dq=qa.grad, dk=ka.grad, dv=va.grad, dbeta=ba.grad,
                dg=ga.grad, dh0=ha.grad)

    # ---- our chain on the real FLA forward intermediates ----
    fi = fla_forward_intermediates(qn, kn, v, beta_act, g_dec, h0, scale, BT)

    dav_k = build_dav_kernel()
    dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, BT)

    # Part A: dAv alone vs FLA
    dA_f, dv_f = fla_dav(qn, kn, fi["v_new"], do, fi["Aqk"], scale, BT)
    print("  -- Part A: dAv kernel vs FLA chunk_kda_bwd_dAv --")
    stats("dAqk", dAqk, dA_f)
    stats("dv  (Mqk@do)", dv, dv_f)

    # Part B: chain
    dhu_k = build_dhu()
    dqkg_k = build_dqkg()
    intra_k = build_intra()

    dh, dh0, dv2 = dhu_triton(dhu_k, fi, do, dv, h0, scale, BT, dht=dfin)
    inp2 = dict(q=qn, k=kn, v=v, v_new=fi["v_new"], g=fi["g_cum"], beta=beta_act,
                A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
    b2 = wy_dqkg_triton(dqkg_k, inp2, scale, BT)
    inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=beta_act,
                dAqk=dAqk, dAkk=b2["dA"])
    b3 = intra_triton(intra_k, inp3, BT, BC)

    dq = b2["dq"] + b3["dq"]
    dk = b2["dk"] + b3["dk"]
    db = b2["db"] + b3["db"]
    dg_cum = b2["dg"] + b3["dg"]
    dv_final = b2["dv"]                                   # wy_dqkg's dv2 = dL/dv

    # B0 tail: reverse chunk-local cumsum of dg_cum.  FLA's chunk_local_cumsum
    # defaults to scale=RCP_LN2, so its reverse already maps dg (wrt the scaled
    # log2 g_cum) back to dg wrt the natural-log g input -- i.e. straight to
    # ga.grad.  Do NOT multiply by an extra ln2 here: the `*LN2` used by P5-3
    # (`check_bwd_fla.py`) was only to convert ga.grad into the log2-gate units
    # of the fp64 reference.  Getting this wrong shows up as a uniform RCP_LN2
    # scale (fitted scale ~1.44), still 0.44 in max_rel.
    from fla.ops.utils import chunk_local_cumsum
    dg = chunk_local_cumsum(dg_cum, chunk_size=BT, reverse=True)

    print("  -- Part B: chain dAv -> B1 -> B2 -> B3 -> rev-cumsum  vs  torch.autograd --")
    stats("dq", dq, gold["dq"])
    stats("dk", dk, gold["dk"])
    stats("dv", dv_final, gold["dv"])
    stats("dg", dg, gold["dg"])
    stats("dbeta", db, gold["dbeta"])
    stats("dh0", dh0, gold["dh0"])
    print(f"     (|o|max={o.float().abs().max().item():.3f},  "
          f"|S|max={s.float().abs().max().item():.3f})")

    # ---- Part C: FLA's own internal decomposition (sanity that the setup mirrors it) ----
    fc = fla_chain_grads(fi, qn, kn, v, beta_act, h0, do, dfin, scale, BT)
    print("  -- Part C: FLA internal decomposition vs torch.autograd --")
    for fld in ["dq", "dk", "dv", "dg", "db", "dh0"]:
        stats(fld, fc[fld], gold["dbeta" if fld == "db" else fld])
    print("  -- intermediate-level: ours vs FLA decomposition --")
    stats("dh", dh, fc["dh"])
    stats("dv2", dv2, fc["dv2"])
    stats("dAqk", dAqk, fc["dAqk"])
    stats("dAkk (b2.dA)", b2["dA"], fc["dAkk"])
    stats("dg_cum (b2 only)", b2["dg"], fc["dg_cum"])
    stats("dg (rev-cumsum)", dg, fc["dg"])

    # Part D: feed our B2/B3 FLA's *own* B1 outputs -> isolates kernel correctness
    # from the propagation of B1's tf32 rounding through the decay chain.
    b2s = wy_dqkg_triton(dqkg_k, dict(q=qn, k=kn, v=v, v_new=fi["v_new"], g=fi["g_cum"],
                                      beta=beta_act, A=fi["Akk"], h=fi["h"], do=do,
                                      dh=fc["dh"], dv=fc["dv2"]), scale, BT)
    b3s = intra_triton(intra_k, dict(q=qn, k=kn, g=fi["g_cum"], beta=beta_act,
                                     dAqk=fc["dAqk"], dAkk=fc["dAkk"]), BT, BC)
    dk_sw = b2s["dk"] + b3s["dk"]
    dg_sw = chunk_local_cumsum(b2s["dg"] + b3s["dg"], chunk_size=BT, reverse=True)
    print("  -- swap: our B2/B3 fed FLA's B1 outputs (vs FLA decomposition) --")
    stats("dk", dk_sw, fc["dk"])
    stats("dg (rev-cumsum)", dg_sw, fc["dg"])
    return dict(dq=dq, dk=dk, dv=dv_final, dg=dg, dbeta=db, dh0=dh0), gold


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "need GPU"
    print("device:", torch.cuda.get_device_name(0), "torch", torch.__version__)
    dev = "cuda"
    D = 128
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0

    cases = [
        ("T=256  H=2 BT=64", 1, 256, 2, 64, 16, 0),
        ("T=512  H=4 BT=64", 2, 512, 4, 64, 16, 1),
        ("T=1024 H=8 BT=64", 1, 1024, 8, 64, 16, 2),
        ("T=250  H=3 BT=64 (non-div T)", 1, 250, 3, 64, 16, 3),
    ]
    cases = cases[:int(os.environ.get("E2E_CASES", len(cases)))]
    worst = {}
    for (name, B, T, H, BT, BC, seed) in cases:
        print("\n" + "#" * 78)
        print(f"# config: {name}  B={B} H={H} D={D} seed={seed}")
        print("#" * 78)
        ours, gold = run_case(B, T, H, D, BT, BC, scale, lower_bound, dev, seed=seed)
        for fld in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
            mr = ((ours[fld].float() - gold[fld].float()).abs().max()
                  / (gold[fld].float().abs().max() + 1e-12)).item()
            worst[fld] = max(worst.get(fld, 0.0), mr)

    print("\n" + "=" * 78)
    print("worst-case max_rel across cases (our chain vs FLA autograd):")
    for fld in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
        print(f"  {fld:6s} {worst[fld]:.3e}")
    print("DONE")


if __name__ == "__main__":
    main()
