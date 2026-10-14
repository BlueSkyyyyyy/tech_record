"""P6-8: run the full backward chain *directly off FlashKDA's own K1 workspace*.

P6-7 assembled `dAv -> B1(dhu) -> B2(wy_dqkg) -> B3(intra) -> reverse-cumsum(B0)` on the
FLA *forward intermediates* (`qg/kg/w/Akk/Aqk/v_new/h`).  But the target-card backward is
supposed to *reuse FlashKDA's K1 workspace* (docs/06 §5.2, docs/08 §6): six per-tile fields
`k_dec/q_dec/k_rest/E_tot/P/Mqk` (csrc/smxx/utils.cuh:70-76) plus a lightweight **B0** gate
recompute for `beta`/`g_cum`.  P6-6 Part E only cross-checked the workspace `kGTotal` against
B0's `exp2(g_cum[-1])`; nobody has actually *driven the backward from the workspace*.

This script closes that last integration gap, entirely on FlashKDA's own forward:

  Part A  reconstruct the *forward* from the workspace: `o = q_dec S_in + M_qk U`,
          `S_out = E_tot S_in + k_rest^T U`, using only the 6 workspace fields + B0.
          Compared against FlashKDA's own `out`/`final_state` -> proves the fields are
          read with the right layout/semantics (bf16 floor expected).
  Part B  reconstruct every backward operand (`qg=q_dec/scale`, `kg=k_rest`, `Akk=P`,
          `Aqk=Mqk`, `w=P diag(beta) k_dec`, `v_new=U`, `h=S_in`) and compare each against
          an *independent* analytic recomputation (`kda_ref` formulas at CHUNK=16).
  Part C  run the full backward chain (`dAv -> B1 -> B2 -> B3 -> reverse-cumsum B0`) on the
          workspace-derived operands, at CHUNK=16, and compare `dq,dk,dv,dg,dbeta,dh0` with
          `torch.autograd` through `fla.ops.kda.chunk_kda` (P6-7 setup: same activation-unit
          operands, all in-kernel transforms off).

Run inside `kernel_lab` (GPU, FlashKDA built, FLA installed):
    python ~/proj/tech_record/code/flash-kda/harness/check_bwd_ws_reuse.py
"""
import math
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_dhu_kernel import build_kernel as build_dhu, dhu_triton          # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton    # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton    # noqa: E402
from check_bwd_b0_gate import b0_gate_fwd                                     # noqa: E402

RCP_LN2 = 1.4426950408889634
LOG2E = RCP_LN2
LN2 = math.log(2.0)
CHUNK = 16
D = 128


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
def make_inputs(B, T, H, seed, device, g_scale=1.0):
    torch.manual_seed(seed)
    q = (torch.randn(B, T, H, D, device=device) * 0.5).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=device) * 0.5).to(torch.bfloat16)
    v = torch.randn(B, T, H, D, device=device).to(torch.bfloat16)
    beta = torch.randn(B, T, H, device=device).to(torch.bfloat16)
    g = (torch.randn(B, T, H, D, device=device) * g_scale).to(torch.bfloat16)
    A_log = (torch.rand(H, device=device) * 0.5)
    dt_bias = (torch.randn(H, D, device=device) * 0.1)
    h0 = (torch.randn(B, H, D, D, device=device) * 0.1).to(torch.bfloat16)   # S^T == [B,H,V,K]
    return dict(q=q, k=k, v=v, beta=beta, g=g, A_log=A_log, dt_bias=dt_bias, h0=h0)


def flash_forward(inp, scale, lower_bound, device):
    from flash_kda_C import get_workspace_size, fwd as fwd_raw

    B, T, H, _ = inp["q"].shape
    NT = cdiv(T, CHUNK)
    total_tiles = B * NT
    n_ht = H * total_tiles
    ws = torch.empty(get_workspace_size(B * T, H, B), dtype=torch.uint8, device=device)
    out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=device)
    fin = torch.empty(B, H, D, D, dtype=torch.bfloat16, device=device)
    fwd_raw(inp["q"].contiguous(), inp["k"].contiguous(), inp["v"].contiguous(),
            inp["g"].contiguous(), inp["beta"].contiguous(), scale, out, ws,
            inp["A_log"].clone(), inp["dt_bias"].clone(), lower_bound,
            initial_state=inp["h0"], final_state=fin)
    torch.cuda.synchronize()

    def field(off, nbytes, dt):
        return ws[off:off + n_ht * nbytes].view(dt).view(H, total_tiles, -1)

    kd = field(0, CHUNK * D * 2, torch.bfloat16)
    qd = field(n_ht * CHUNK * D * 2, CHUNK * D * 2, torch.bfloat16)
    kr = field(n_ht * 2 * CHUNK * D * 2, CHUNK * D * 2, torch.bfloat16)
    gt = field(n_ht * 3 * CHUNK * D * 2, D * 4, torch.float32)
    inv = field(n_ht * (3 * CHUNK * D * 2 + D * 4), CHUNK * CHUNK * 2, torch.bfloat16)
    mqk = field(n_ht * (3 * CHUNK * D * 2 + D * 4 + CHUNK * CHUNK * 2),
                CHUNK * CHUNK * 2, torch.bfloat16)

    def to_bnt(x, last):
        return x.view(H, B, NT, *last).permute(1, 2, 0, *range(3, 3 + len(last))).contiguous()

    return dict(
        out=out, fin=fin,
        kd=to_bnt(kd, (CHUNK, D)), qd=to_bnt(qd, (CHUNK, D)), kr=to_bnt(kr, (CHUNK, D)),
        gt=to_bnt(gt, (D,)), inv=to_bnt(inv, (CHUNK, CHUNK)),
        mqk=to_bnt(mqk, (CHUNK, CHUNK)), NT=NT, total_tiles=total_tiles)


def b0_gate(inp, lower_bound):
    """B0 recompute of the per-token activated gate (log2 units) and beta (sigmoid)."""
    gc, gbeta = b0_gate_fwd(inp["g"], inp["A_log"], inp["dt_bias"], inp["beta"],
                            lower_bound, CHUNK, scale=LOG2E)
    return gc, gbeta   # gc = g_cum [B,T,H,D] (log2, <=0); gbeta = sigmoid(beta_raw)


# ------------------------------------------------------- reconstruct fwd + bwd operands
def reconstruct(fw, inp, lower_bound, scale):
    """Turn the 6 workspace fields + B0 into all forward intermediates the backward needs.

    Returns o_rec, fin_rec, and the operand dict (FLA-`fi` conventions, CHUNK=16).
    """
    B, T, H, _ = inp["q"].shape
    NT = fw["NT"]
    Tpad = NT * CHUNK
    dev = inp["q"].device
    g_cum, beta_act = b0_gate(inp, lower_bound)
    # pad the tail chunk to CHUNK (K1 stores CHUNK-wide tiles; the pad rows are inert)
    v_pad = torch.zeros(B, Tpad, H, D, device=dev)
    v_pad[:, :T] = inp["v"].float()
    beta_pad = torch.zeros(B, Tpad, H, device=dev)
    beta_pad[:, :T] = beta_act

    kd = fw["kd"].float()                       # [B,NT,H,CHUNK,D]  k_dec
    qd = fw["qd"].float()                       # q_dec (includes scale)
    kr = fw["kr"].float()                       # k_rest
    Et = fw["gt"]                               # [B,NT,H,D] E_tot
    P = fw["inv"].float()                       # [B,NT,H,CHUNK,CHUNK] P[o_t,o_A]
    Mqk = fw["mqk"].float()                     # [B,NT,H,CHUNK,CHUNK]

    # chunk states S_in [B,NT,H,K,V] and U=v_new [B,T,H,V]
    S = inp["h0"].float().permute(0, 1, 3, 2).contiguous()      # [B,H,K,V] from [B,H,V,K]
    h = torch.zeros(B, NT, H, D, D, device=dev)
    U = torch.zeros(B, T, H, D, device=dev)
    o_rec = torch.zeros(B, T, H, D, device=dev)
    for c in range(NT):
        h[:, c] = S
        t0 = c * CHUNK
        o_t = t0 + torch.arange(CHUNK, device=dev)
        m_t = o_t < T
        vc = v_pad[:, t0:t0 + CHUNK].permute(0, 2, 1, 3)                     # [B,H,CH,V]
        kdc = kd[:, c]                                                       # [B,H,CH,K]
        qdc = qd[:, c]
        krc = kr[:, c]
        Pc = P[:, c]                                                         # [B,H,o_t,o_A]
        Mc = Mqk[:, c]
        bc = beta_pad[:, t0:t0 + CHUNK].permute(0, 2, 1)                     # [B,H,CH]
        pred = torch.einsum("bhtk,bhkv->bhtv", kdc, S)
        u = torch.einsum("bhtu,bhuv->bhtv", Pc, (vc - pred) * bc.unsqueeze(-1))
        n = min(CHUNK, T - t0)
        o_rec[:, t0:t0 + n] = (torch.einsum("bhtk,bhkv->bhtv", qdc, S)
                               + torch.einsum("bhtu,bhuv->bhtv", Mc, u))[:, :, :n].permute(0, 2, 1, 3)
        U[:, t0:t0 + n] = u[:, :, :n].permute(0, 2, 1, 3)
        S = Et[:, c, :, :, None] * S + torch.einsum("bhtk,bhtv->bhkv", krc, u)
    fin_rec = S.permute(0, 1, 3, 2)                                          # [B,H,V,K]
    h_out = h                                                                # [B,NT,H,K,V]

    # bwd operands (FLA `fi` layout). Workspace fields are [B,NT,H,CHUNK,D]; flatten the
    # (NT,H,CHUNK) axes into the FLA token-major [B,T,H,D] (token order: chunk-major, H inner).
    qg = (qd / scale).permute(0, 1, 3, 2, 4).reshape(B, Tpad, H, D)[:, :T].contiguous()
    kg = kr.permute(0, 1, 3, 2, 4).reshape(B, Tpad, H, D)[:, :T].contiguous()
    # w = P diag(beta) k_dec, per chunk
    w = torch.zeros(B, T, H, D, device=dev)
    for c in range(NT):
        t0 = c * CHUNK
        n = min(CHUNK, T - t0)
        bc = beta_pad[:, t0:t0 + CHUNK].permute(0, 2, 1)   # [B,H,CH]
        kd_c = kd[:, c]                                    # [B,H,CH,K]
        w_c = torch.einsum("bhto,bhok->bhtk", P[:, c], kd_c * bc.unsqueeze(-1))
        w[:, t0:t0 + n] = w_c[:, :, :n].permute(0, 2, 1, 3)
    # Aqk from q_dec and k_inv; workspace only stores Mqk=tril(Aqk), which is all dAv uses.
    Aqk_rowmajor = Mqk.permute(0, 1, 3, 2, 4).reshape(B, Tpad, H, CHUNK)[:, :T].contiguous()
    Akk = P.permute(0, 1, 3, 2, 4).reshape(B, Tpad, H, CHUNK)[:, :T].contiguous()

    fi = dict(qg=qg.contiguous(), kg=kg.contiguous(), w=w.contiguous(),
              g_cum=g_cum.contiguous(), beta=beta_act.contiguous(),
              Aqk=Aqk_rowmajor, Akk=Akk, v_new=U.contiguous(),
              h=h_out.contiguous(), Mqk=fw["mqk"])
    return o_rec, fin_rec, fi


# --------------------------------------------- analytic operands (independent ground truth)
def analytic_operands(inp, qn, kn, beta_act, lower_bound, scale):
    """Every backward operand recomputed *from scratch* (kda_ref formulas, fp32, CHUNK=16).

    This is the "is the reconstruction faithful?" oracle for Part B: the workspace-derived
    operands (Part A's reconstruct) must equal these to the bf16 floor.
    """
    from kda_ref import l2_normalize, kda_gate
    B, T, H, _ = inp["q"].shape
    NT = cdiv(T, CHUNK)
    Tpad = NT * CHUNK
    dev = inp["q"].device
    g_dec = kda_gate(inp["g"], inp["A_log"], inp["dt_bias"], lower_bound)   # log2 gate
    v = inp["v"].float()
    # pad the partial tail chunk with zeros (kda_ref convention: pad g/q/k/v/beta = 0 so the
    # pad rows are inert and E_tot == the last valid row's decay)
    def pad(x, last):
        if x.shape[1] == Tpad:
            return x
        z = torch.zeros(B, Tpad, H, *last, device=dev, dtype=x.dtype)
        z[:, :T] = x
        return z
    qn4 = pad(qn, (D,)).reshape(B, NT, CHUNK, H, D)
    kn4 = pad(kn, (D,)).reshape(B, NT, CHUNK, H, D)
    v4 = pad(v, (D,)).reshape(B, NT, CHUNK, H, D)
    g4 = pad(g_dec, (D,)).reshape(B, NT, CHUNK, H, D)
    b4 = pad(beta_act, ()).reshape(B, NT, CHUNK, H)
    gcum = g4.cumsum(2)
    Et = torch.exp2(gcum[:, :, -1])
    E = torch.exp2(gcum)
    qd4 = qn4 * E * scale
    kd4 = kn4 * E
    ki4 = kn4 * torch.exp2(-gcum)
    kr4 = ki4 * Et[:, :, None]
    A4 = torch.einsum("bnchd,bnghd->bnhcg", kd4, ki4)          # [B,NT,H,o_t,o_A]
    bih = b4.permute(0, 1, 3, 2)                               # [B,NT,H,CH]
    L4 = torch.tril(A4, -1) * bih[..., None]
    P4 = torch.linalg.inv(torch.eye(CHUNK, device=dev) + L4)
    Mg4 = torch.tril(torch.einsum("bnchd,bnghd->bnhcg", qd4, ki4))
    w4 = torch.einsum("bnhcg,bnghd->bnhcd", P4, kd4 * b4[..., None])
    S = inp["h0"].float().permute(0, 1, 3, 2).contiguous()
    h_a = torch.zeros(B, NT, H, D, D, device=dev)
    U_a = torch.zeros(B, T, H, D, device=dev)
    for c in range(NT):
        h_a[:, c] = S
        pred = torch.einsum("bchd,bhdv->bhcv", kd4[:, c], S)
        upr = (v4[:, c].permute(0, 2, 1, 3) - pred) * bih[:, c][..., None]
        u = torch.einsum("bhcg,bhgv->bhcv", P4[:, c], upr)
        n = min(CHUNK, T - c * CHUNK)
        U_a[:, c * CHUNK:c * CHUNK + n] = u[:, :, :n].permute(0, 2, 1, 3)
        S = Et[:, c, :, :, None] * S + torch.einsum("bchd,bhcv->bhdv", kr4[:, c], u)
    toT = lambda x: x.permute(0, 1, 3, 2, 4).reshape(B, NT * CHUNK, H, x.shape[-1])[:, :T]
    return dict(qg=(qd4 / scale).reshape(B, Tpad, H, D)[:, :T].contiguous(),
                kg=kr4.reshape(B, Tpad, H, D)[:, :T].contiguous(), w=toT(w4).contiguous(),
                Aqk=toT(Mg4).contiguous(), Akk=toT(P4).contiguous(),
                v_new=U_a.contiguous(), h=h_a.contiguous())


# ------------------------------------------------------------------ our chain (CHUNK=16)
def run_chain(fi, inp, qn, kn, do, dfin, scale, lower_bound):
    B, T, H, _ = inp["q"].shape
    BT = CHUNK

    # dAv (output branch): dAqk, dv
    from check_bwd_e2e import build_dav_kernel, dav_triton
    dav_k = build_dav_kernel()
    dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, BT)

    dhu_k = build_dhu()
    dqkg_k = build_dqkg()
    intra_k = build_intra()

    dh, dh0, dv2 = dhu_triton(dhu_k, fi, do, dv, inp["h0"].float().permute(0, 1, 3, 2).contiguous(),
                              scale, BT, dht=dfin)
    inp2 = dict(q=qn, k=kn, v=inp["v"].float(), v_new=fi["v_new"], g=fi["g_cum"],
                beta=fi["beta"], A=fi["Akk"], h=fi["h"], do=do, dh=dh, dv=dv2)
    b2 = wy_dqkg_triton(dqkg_k, inp2, scale, BT)
    inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=fi["beta"],
                dAqk=dAqk, dAkk=b2["dA"])
    b3 = intra_triton(intra_k, inp3, BT, BT)

    dq = b2["dq"] + b3["dq"]
    dk = b2["dk"] + b3["dk"]
    db = b2["db"] + b3["db"]
    dg_cum = b2["dg"] + b3["dg"]
    dv_final = b2["dv"]

    # B0 tail: reverse chunk-local cumsum maps dg_cum back to dg wrt the natural-log gate
    # (FLA `chunk_local_cumsum` default scale=RCP_LN2; do NOT add an extra ln2, see §6.9).
    from fla.ops.utils import chunk_local_cumsum
    dg = chunk_local_cumsum(dg_cum, chunk_size=BT, reverse=True)
    return dict(dq=dq, dk=dk, dv=dv_final, dg=dg, dbeta=db,
                dh0=dh0, dAqk=dAqk, dAkk=b2["dA"], dh=dh, dv2=dv2)


# ------------------------------------------------------------------------ FLA golden
def fla_golden(qn, kn, v, beta_act, g_fla, h0, scale, do, dfin):
    """FLA autograd on the *activations* (P6-7 setup): both sides get the same fp32/b16
    operands, all in-kernel transforms off, so our dq/dk/dv/dbeta/dg are directly
    comparable.  ``g_fla`` is the natural-log gate; ``qn/kn`` L2-normalised."""
    from fla.ops.kda import chunk_kda
    B, T, H, _ = qn.shape
    q = qn.clone().requires_grad_(True)
    k = kn.clone().requires_grad_(True)
    vv = v.clone().requires_grad_(True)
    beta = beta_act.clone().requires_grad_(True)
    g = g_fla.clone().requires_grad_(True)
    hh = h0.clone().requires_grad_(True)
    o, s = chunk_kda(q, k, vv, g, beta, scale=scale, initial_state=hh.float().transpose(-1, -2),
                     output_final_state=True, use_qk_l2norm_in_kernel=False,
                     use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
    loss = (o.float() * do).sum() + (s.float() * dfin).sum()
    loss.backward()
    return dict(q=q.grad, k=k.grad, v=vv.grad, beta=beta.grad, g=g.grad, h0=hh.grad)


# -------------------------------------------------------------------------- runner
def run_case(B, T, H, scale, lower_bound, device, seed=0, g_scale=1.0):
    inp = make_inputs(B, T, H, seed, device, g_scale)
    fw = flash_forward(inp, scale, lower_bound, device)
    o_rec, fin_rec, fi = reconstruct(fw, inp, lower_bound, scale)

    print("  -- Part A: forward reconstructed from workspace vs FlashKDA's own out/fin --")
    stats("o_rec  vs flash out", o_rec, fw["out"])
    stats("fin_rec vs flash fin", fin_rec, fw["fin"])

    torch.manual_seed(seed + 77)
    do = torch.randn(B, T, H, D, device=device)
    dfin = torch.randn(B, H, D, D, device=device) * 0.01

    # shared activation-unit operands (P6-7 setup): L2-normalised q/k, sigmoid'd beta,
    # natural-log gate; both our chain and the FLA golden consume exactly these.
    qn = torch.nn.functional.normalize(inp["q"].float(), dim=-1)
    kn = torch.nn.functional.normalize(inp["k"].float(), dim=-1)
    beta_act = torch.sigmoid(inp["beta"].float())
    # per-token natural-log gate == lower_bound * sigmoid(...) (LOG2E*LN2 == 1); FLA's
    # local cumsum then reproduces our B0 g_cum exactly.
    inner = inp["g"].float() + inp["dt_bias"].view(1, 1, H, D)
    g_fla = lower_bound * torch.sigmoid(torch.exp(inp["A_log"].view(1, 1, H, 1)) * inner)

    print("  -- Part B: workspace-derived operands vs analytic (kda_ref formulas, CHUNK=16) --")
    an = analytic_operands(inp, qn, kn, beta_act, lower_bound, scale)
    for nm in ["qg", "kg", "w", "Aqk", "Akk", "v_new", "h"]:
        stats(nm, fi[nm], an[nm])

    ours = run_chain(fi, inp, qn, kn, do, dfin, scale, lower_bound)
    gold = fla_golden(qn, kn, inp["v"].float(), beta_act, g_fla, inp["h0"], scale, do, dfin)

    print("  -- Part C: workspace-driven chain vs torch.autograd(FLA, same activations) --")
    stats("dq", ours["dq"], gold["q"])
    stats("dk", ours["dk"], gold["k"])
    stats("dv", ours["dv"], gold["v"])
    stats("dbeta (post-sigmoid)", ours["dbeta"], gold["beta"])
    stats("dg (natural-log)", ours["dg"], gold["g"])
    stats("dh0", ours["dh0"], gold["h0"].transpose(-1, -2))
    return ours, gold


def main():
    assert torch.cuda.is_available(), "need GPU"
    print("device:", torch.cuda.get_device_name(0), "torch", torch.__version__)
    dev = "cuda"
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    cases = [
        ("T=128 H=2", 1, 128, 2, 0, 1.0),
        ("T=256 H=4", 1, 256, 4, 1, 1.0),
        ("T=250 H=3 (non-div)", 1, 250, 3, 2, 1.0),
        ("T=512 H=4 g=2", 1, 512, 4, 3, 2.0),
        ("B2 T=513 H=2 (non-div)", 2, 513, 2, 4, 1.0),
        ("T=192 H=2 g=0", 1, 192, 2, 5, 0.0),
    ]
    cases = cases[:int(os.environ.get("WS_CASES", len(cases)))]
    worst = {}
    for (name, B, T, H, seed, gs) in cases:
        print("\n" + "#" * 78)
        print(f"# {name}  B={B} T={T} H={H} seed={seed} g_scale={gs}")
        print("#" * 78)
        ours, gold = run_case(B, T, H, scale, lower_bound, dev, seed=seed, g_scale=gs)
        for fld, gk in [("dq", "q"), ("dk", "k"), ("dv", "v"), ("dbeta", "beta"),
                        ("dg", "g")]:
            gr = gold[gk].float().abs().max().item()
            mr = ((ours[fld].float() - gold[gk].float()).abs().max() / (gr + 1e-12)).item()
            worst[fld] = max(worst.get(fld, 0.0), mr)
        gr = gold["h0"].transpose(-1, -2).float().abs().max().item()
        mr = ((ours["dh0"].float() - gold["h0"].transpose(-1, -2).float()).abs().max()
              / (gr + 1e-12)).item()
        worst["dh0"] = max(worst.get("dh0", 0.0), mr)

    print("\n" + "=" * 78)
    print("worst-case max_rel across cases (workspace-driven chain vs FLA autograd):")
    for fld in ["dq", "dk", "dv", "dbeta", "dg", "dh0"]:
        print(f"  {fld:6s} {worst[fld]:.3e}")
    print("DONE")


if __name__ == "__main__":
    main()
