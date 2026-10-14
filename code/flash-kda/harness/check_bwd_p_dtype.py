"""P6-9 evidence: does storing P (= A_L^{-1} = FLA `Akk`) in bf16 hurt the backward?

docs/06 §5.3 leaves one precision item open: the inverse-adjoint
``dA_L = -P^T (b_dA) P^T`` (式 7.5, FLA `chunk_bwd.py:302-303`) is sensitive to
``P``, and the target-card K1 workspace stores ``P`` as bf16 (``kINV``, 512 B/tile,
utils.cuh:70-76). §5.3 speculates an fp32 copy ("增存 fp32 `P`", +1 KB/tile) might
be needed. This script *measures* whether that is worth it.

Four parts, all on the fp32 FLA forward intermediates (same setup as check_bwd_e2e):

  Part A  dtype audit of the FLA forward intermediates: with fp32 operands the
          forward returns fp32 `Akk`; with bf16 operands it returns bf16 `Akk`
          (chunk_intra.py:819 `torch.zeros(... dtype=k.dtype)`).  Pins what "stored
          P" actually is, and corrects the §3.3 note that `b_A.to(b_A.dtype)` is a
          *forced* bf16 cast (it is a no-op -- the dtype is whatever A is stored in).
  Part B  analytic sensitivity of ``-P^T dP P^T`` to bf16(P), isolated.
  Part C  B2 (`wy_dqkg`) run twice with ``A=P_fp32`` vs ``A=bf16(P).fp32``
          (isolates the *storage quantization* of P; the adjoint dot stays fp32/tf32.
          P is only consumed by B2 in the backward -- B1 takes w/v_new and is
          P-independent -- so this is the whole P-storage effect).
  Part D  full chain dAv -> B1 -> B2 -> B3 -> rev-cumsum with the two stored-P
          precisions, six gradients vs `torch.autograd` -- end-to-end cost of bf16 P.

Run inside `kernel_lab` (idle GPU):
    CUDA_VISIBLE_DEVICES=1 python ~/proj/tech_record/code/flash-kda/harness/check_bwd_p_dtype.py
"""
import math
import os
import sys

# FLA >=0.5 auto-selects a TileLang backend; pin the Triton reference (same trap as §6.6).
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = os.path.expanduser("~/proj/tech_record/code/flash-kda/harness")
SRC = os.path.expanduser("~/proj/tech_record/code/flash-kda/src")
sys.path.insert(0, HARNESS)
sys.path.insert(0, SRC)

from check_bwd_e2e import (make_inputs, derive_operands, fla_forward_intermediates,   # noqa: E402
                           build_dav_kernel, dav_triton, cdiv)
from check_bwd_dhu_kernel import build_kernel as build_dhu, dhu_triton               # noqa: E402
from check_bwd_dqkg_kernel import build_kernel as build_dqkg, wy_dqkg_triton        # noqa: E402
from check_bwd_intra_kernel import build_kernel as build_intra, intra_triton         # noqa: E402

LN2 = math.log(2.0)
RCP_LN2 = 1.4426950408889634


def stats(name, pred, gold):
    d = (pred.float() - gold.float()).abs()
    denom = gold.float().abs().max().item()
    max_abs = d.max().item()
    max_rel = max_abs / (denom + 1e-12)
    print(f"  {name:30s} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  ref_max={denom:.3e}")
    return max_abs, max_rel


def stats_pair(name, a, b):
    """symmetric delta between two runs (a = fp32-P baseline, b = bf16-P run)."""
    d = (a.float() - b.float()).abs()
    denom = a.float().abs().max().item()
    max_abs = d.max().item()
    max_rel = max_abs / (denom + 1e-12)
    print(f"  {name:30s} D_max_abs={max_abs:.3e}  D_max_rel={max_rel:.3e}  |a|max={denom:.3e}")
    return max_abs, max_rel


# ---------------------------------------------------------------- Part A: dtype audit
def part_a_dtype_audit(device, scale, lower_bound):
    print("\n" + "=" * 78)
    print("Part A: FLA forward intermediate dtype audit (what is 'stored P'?)")
    print("=" * 78)
    for in_dtype in (torch.float32, torch.bfloat16):
        inp = make_inputs(1, 256, 2, 128, scale, lower_bound, device, seed=0)
        qn, kn, beta_act, g_dec = derive_operands(inp, lower_bound)
        c = lambda x: x.to(in_dtype)  # noqa: E731
        v = inp["v_raw"].to(in_dtype)
        h0 = inp["h0"].to(in_dtype)
        fi = fla_forward_intermediates(c(qn), c(kn), v, c(beta_act), c(g_dec), h0, scale, 64)
        print(f"  inputs {str(in_dtype):18s}: Aqk={fi['Aqk'].dtype}  Akk(P)={fi['Akk'].dtype}"
              f"  w={fi['w'].dtype}  v_new={fi['v_new'].dtype}  h={fi['h'].dtype}")


# --------------------------------------------- Part B: analytic sensitivity of the adjoint
def part_b_adjoint_sensitivity(fi, device):
    print("\n" + "=" * 78)
    print("Part B: analytic sensitivity of dA_L = -P^T dP P^T to bf16(P)")
    print("=" * 78)
    P = fi["Akk"].float()                                  # [B,T,H,BT]
    B, T, H, BT = P.shape
    NT = cdiv(T, BT)
    P4 = P.view(B, NT, BT, H, BT).permute(0, 1, 3, 2, 4).contiguous()   # [B,NT,H,BT,BT]
    P4 = P4[:, :, :, :, :].tril()                          # P is unit-lower in (o_t,o_A)
    # strictly-lower mask, FLA chunk_bwd.py:301 `m_A = o_t[:,None] > o_t[None,:]`
    mlow = torch.tril(torch.ones(BT, BT, device=device), -1).bool()
    torch.manual_seed(123)
    dP = torch.where(mlow, torch.randn_like(P4), torch.zeros_like(P4))
    dA_fp32 = -torch.einsum("...ji,...jk,...kl->...il", P4, dP, P4)      # -P^T dP P^T
    P4_bf = P4.to(torch.bfloat16).float()
    dA_bf = -torch.einsum("...ji,...jk,...kl->...il", P4_bf, dP, P4_bf)
    stats("P bf16 vs P fp32 (stored)", P4_bf, P4)
    stats("dA_L(P_bf16) vs dA_L(P_fp32)", dA_bf, dA_fp32)


# ---------------------------------------------------------- Part C: isolated B2 with P dtype
def part_c_b2_isolated(fi, qn, kn, v, beta_act, do, dh, dv, scale, BT, dqkg_k):
    print("\n" + "=" * 78)
    print("Part C: B2 (wy_dqkg) with A=P_fp32 vs A=bf16(P), all other operands fp32")
    print("=" * 78)
    P32 = fi["Akk"]
    # (i) storage quantization only: bf16-rounded value, dot stays fp32/tf32
    Pq = P32.to(torch.bfloat16).to(torch.float32)
    base = dict(q=qn, k=kn, v=v, v_new=fi["v_new"], g=fi["g_cum"], beta=beta_act,
                h=fi["h"], do=do, dh=dh, dv=dv)
    b2_32 = wy_dqkg_triton(dqkg_k, dict(base, A=P32), scale, BT)
    b2_q = wy_dqkg_triton(dqkg_k, dict(base, A=Pq), scale, BT)
    print("  -- our Triton B2: A=bf16(P).fp32 vs A=P_fp32 (P storage quantization) --")
    for fld in ["dq", "dk", "dv", "db", "dg", "dA"]:
        stats_pair(fld, b2_32[fld], b2_q[fld])
    print("  (note: P is only fed to B2 in the backward -- B1 uses w/v_new, P-independent.\n"
          "   A true bf16 MMA path additionally rounds the other dot operand; that axis is\n"
          "   the all-bf16 floor already measured in docs/06 §6.6 Part C, not P-specific.)")
    return b2_32, b2_q


# ------------------------------------------------------------------- full chain assembly
def run_chain(fi, qn, kn, v, beta_act, h0, do, dfin, scale, BT, BC, P,
              dav_k, dhu_k, dqkg_k, intra_k):
    from fla.ops.utils import chunk_local_cumsum
    dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, BT)
    dh, dh0, dv2 = dhu_triton(dhu_k, fi, do, dv, h0, scale, BT, dht=dfin)
    inp2 = dict(q=qn, k=kn, v=v, v_new=fi["v_new"], g=fi["g_cum"], beta=beta_act,
                A=P, h=fi["h"], do=do, dh=dh, dv=dv2)
    b2 = wy_dqkg_triton(dqkg_k, inp2, scale, BT)
    inp3 = dict(q=qn, k=kn, g=fi["g_cum"], beta=beta_act, dAqk=dAqk, dAkk=b2["dA"])
    b3 = intra_triton(intra_k, inp3, BT, BC)
    out = dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"], db=b2["db"] + b3["db"],
               dv=b2["dv"], dh0=dh0,
               dg=chunk_local_cumsum(b2["dg"] + b3["dg"], chunk_size=BT, reverse=True))
    out["dh"], out["dv2"] = dh, dv2
    return out


def run_case(B, T, H, D, BT, BC, scale, lower_bound, device, seed):
    from fla.ops.kda import chunk_kda
    inp = make_inputs(B, T, H, D, scale, lower_bound, device, seed=seed)
    qn, kn, beta_act, g_dec = derive_operands(inp, lower_bound)
    v = inp["v_raw"].float()
    h0 = inp["h0"].float()
    torch.manual_seed(seed + 77)
    do = torch.randn(B, T, H, D, device=device)
    dfin = torch.randn(B, H, D, D, device=device) * 0.01

    # autograd golden (same fp32 operands, all in-kernel transforms off)
    qa, ka, va = qn.clone().requires_grad_(True), kn.clone().requires_grad_(True), v.clone().requires_grad_(True)
    ba, ga, ha = beta_act.clone().requires_grad_(True), (g_dec * LN2).clone().requires_grad_(True), h0.clone().requires_grad_(True)
    o, s = chunk_kda(qa, ka, va, ga, ba, scale=scale, initial_state=ha,
                     output_final_state=True, use_qk_l2norm_in_kernel=False,
                     use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
    ((o * do).sum() + (s * dfin).sum()).backward()
    gold = dict(dq=qa.grad, dk=ka.grad, dv=va.grad, dbeta=ba.grad, dg=ga.grad, dh0=ha.grad)

    fi = fla_forward_intermediates(qn, kn, v, beta_act, g_dec, h0, scale, BT)
    dav_k, dhu_k = build_dav_kernel(), build_dhu()
    dqkg_k, intra_k = build_dqkg(), build_intra()

    P32 = fi["Akk"]
    Pq = P32.to(torch.bfloat16).to(torch.float32)   # stored-P = bf16 value, dot kept fp32/tf32
    ours32 = run_chain(fi, qn, kn, v, beta_act, h0, do, dfin, scale, BT, BC, P32,
                       dav_k, dhu_k, dqkg_k, intra_k)
    oursbf = run_chain(fi, qn, kn, v, beta_act, h0, do, dfin, scale, BT, BC, Pq,
                       dav_k, dhu_k, dqkg_k, intra_k)

    print("  -- Part D: chain P_fp32 vs autograd --")
    w32 = {}
    for fld in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
        gname = "db" if fld == "dbeta" else fld
        w32[fld] = stats(fld, ours32[gname], gold[fld])[1]
    print("  -- Part D: chain P_bf16 vs autograd --")
    wbf = {}
    for fld in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
        gname = "db" if fld == "dbeta" else fld
        wbf[fld] = stats(fld, oursbf[gname], gold[fld])[1]
    print("  -- Part D: chain P_bf16-stored vs chain P_fp32 (P-attributable delta) --")
    dP = {}
    for fld in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
        gname = "db" if fld == "dbeta" else fld
        dP[fld] = stats_pair(fld, ours32[gname], oursbf[gname])[1]
    return w32, wbf, dP


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "need GPU"
    print("device:", torch.cuda.get_device_name(0), "torch", torch.__version__)
    dev = "cuda"
    D = 128
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0

    part_a_dtype_audit(dev, scale, lower_bound)

    cases = [
        ("T=256  H=2 BT=64", 1, 256, 2, 64, 16, 0),
        ("T=512  H=4 BT=64", 2, 512, 4, 64, 16, 1),
        ("T=1024 H=8 BT=64", 1, 1024, 8, 64, 16, 2),
        ("T=250  H=3 BT=64 (non-div T)", 1, 250, 3, 64, 16, 3),
    ]
    worst = {"w32": {}, "wbf": {}, "dP": {}}
    for (name, B, T, H, BT, BC, seed) in cases:
        print("\n" + "#" * 78)
        print(f"# config: {name}  B={B} H={H} D={D} seed={seed}")
        print("#" * 78)
        inp = make_inputs(B, T, H, D, scale, lower_bound, dev, seed=seed)
        qn, kn, beta_act, g_dec = derive_operands(inp, lower_bound)
        v = inp["v_raw"].float()
        h0 = inp["h0"].float()
        torch.manual_seed(seed + 77)
        do = torch.randn(B, T, H, D, device=dev)
        dfin = torch.randn(B, H, D, D, device=dev) * 0.01
        fi = fla_forward_intermediates(qn, kn, v, beta_act, g_dec, h0, scale, BT)
        if seed == 0:
            part_b_adjoint_sensitivity(fi, dev)
        dav_k, dhu_k = build_dav_kernel(), build_dhu()
        dqkg_k, intra_k = build_dqkg(), build_intra()
        # B1 output (P-independent) reused for the isolated B2 part
        dAqk, dv = dav_triton(dav_k, fi["v_new"], fi["Aqk"], do, scale, BT)
        dh, dh0, dv2 = dhu_triton(dhu_k, fi, do, dv, h0, scale, BT, dht=dfin)
        if seed == 0:
            part_c_b2_isolated(fi, qn, kn, v, beta_act, do, dh, dv2, scale, BT, dqkg_k)
        w32, wbf, dP = run_case(B, T, H, D, BT, BC, scale, lower_bound, dev, seed)
        for k in worst:
            src = {"w32": w32, "wbf": wbf, "dP": dP}[k]
            for fld, val in src.items():
                worst[k][fld] = max(worst[k].get(fld, 0.0), val)

    print("\n" + "=" * 78)
    print("worst-case max_rel across cases:")
    print(f"  {'field':8s} {'chain(P_fp32)':>16s} {'chain(P_bf16)':>16s} {'P_bf16-vs-fp32':>16s}")
    for fld in ["dq", "dk", "dv", "dg", "dbeta", "dh0"]:
        print(f"  {fld:8s} {worst['w32'][fld]:16.3e} {worst['wbf'][fld]:16.3e} {worst['dP'][fld]:16.3e}")
    print("DONE")


if __name__ == "__main__":
    main()
