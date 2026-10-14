"""megatron-port M7: extend the FLA numeric anchor (``check_port_fla.py``) to the
gate/layout **boundaries** that M6 never touched:

* ``safe_gate=True`` + ``lower_bound`` -- the bounded gate
  ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` (clamps decay to ``[lower_bound, 0)``,
  enables the TensorCore intra kernel ``chunk_intra.py:825``); requires ``use_gate_in_kernel=True``.
* ``allow_neg_eigval`` -- with ``use_beta_sigmoid_in_kernel=True`` the kernel computes
  ``beta = 2 * sigmoid(beta_logit)`` in ``(0, 2)`` instead of ``sigmoid`` (``chunk.py:62``).
* **GVA** (grouped value attention, ``HV > H``): value head ``hv`` shares q/k head
  ``hv // (HV // H)`` (``chunk_intra.py:79``); ``g``/``beta``/``v``/state carry ``HV`` heads.
* ``state_v_first`` -- ``initial_state`` / ``final_state`` use the V-first ``[..., V, K]``
  layout instead of the default ``[..., K, V]`` (``chunk_delta_h.py:702``).

Same methodology as M6: feed both sides the *same* bf16 operands, run the port reference in
fp64, and read the residual as the FLA bf16-kernel error.  ``src/kda_ref.py`` was extended with
GVA (``repeat_interleave`` on q/k, matching FLA's group-sum autograd) and ``state_v_first``.

Run (inside kernel_lab, GPU):
    FLA_DISABLE_BACKEND_DISPATCH=1 python3 \
        ~/proj/tech_record/code/flash-kda/megatron-port/harness/check_port_fla_m7.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

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


def gate_fp64(g_logit, A_log, dt_bias, lower_bound):
    """Exact fp64 gate (no fp32 downcast), mirrors ``fla/ops/kda/gate.py``."""
    z = g_logit + dt_bias.view(g_logit.shape[-2], -1)
    if lower_bound is None:
        return -A_log.view(-1, 1).exp() * F.softplus(z)
    return lower_bound * torch.sigmoid(A_log.view(-1, 1).exp() * z)


def make_inputs(B, T, H, K, V, seed, device, hv=None, tame=False):
    """Return raw tensors; value/gate heads = ``hv`` (default ``H``)."""
    HV = hv or H
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float64)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float64)
    v = torch.randn(B, T, HV, V, device=device, dtype=torch.float64)
    s = 0.4 if tame else 1.0
    g_logit = s * torch.randn(B, T, HV, K, device=device, dtype=torch.float64)
    hi = 4.0 if tame else 16.0
    A_log = torch.log(torch.empty(HV, device=device, dtype=torch.float64).uniform_(1.0, hi))
    dt_bias = 0.1 * torch.randn(HV * K, device=device, dtype=torch.float64)
    beta_logit = torch.randn(B, T, HV, device=device, dtype=torch.float64)
    h0 = torch.randn(B, HV, K, V, device=device, dtype=torch.float64)
    return q, k, v, g_logit, A_log, dt_bias, beta_logit, h0


def _fla_side(mode, qt, kt, vt, bt, ht, scale, **kw):
    """Run FLA and return (o, S) or (o, S, grads...). ``mode`` in {'A','B','C','D'}."""
    from fla.ops.kda import chunk_kda
    if mode in ("A", "B"):
        o, s = chunk_kda(qt, kt, vt, kw["g"], bt, scale=scale, initial_state=ht,
                         output_final_state=True,
                         use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
                         state_v_first=kw.get("state_v_first", False))
        return o, s
    # fused gate modes C/D
    o, s = chunk_kda(qt, kt, vt, kw["gra"], kw["bra"], scale=scale, initial_state=ht,
                     output_final_state=True,
                     use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                     A_log=kw["A_f"], dt_bias=kw["dt_f"],
                     safe_gate=kw.get("safe_gate", False),
                     lower_bound=kw.get("lower_bound", None),
                     allow_neg_eigval=kw.get("allow_neg_eigval", False),
                     state_v_first=kw.get("state_v_first", False))
    return o, s


def _ref_side(mode, qr, kr, vr, br, hr, scale, chunk_size, **kw):
    """Run the fp64 port reference. Inputs already wrap ``requires_grad_``."""
    if mode in ("A", "B"):
        o, s = kda_chunked_fwd(qr, kr, vr, kw["g"], br, scale=scale, initial_state=hr,
                               output_final_state=True, chunk_size=chunk_size,
                               state_v_first=kw.get("state_v_first", False))
        return o, s
    g_act = gate_fp64(kw["gr"], kw["Ar"], kw["dtr"], kw.get("lower_bound", None))
    beta_act = torch.sigmoid(br)
    if kw.get("allow_neg_eigval", False):
        beta_act = 2.0 * torch.sigmoid(br)
    o, s = kda_chunked_fwd(qr, kr, vr, g_act, beta_act, scale=scale, initial_state=hr,
                           output_final_state=True, chunk_size=chunk_size,
                           state_v_first=kw.get("state_v_first", False))
    return o, s


def part0():
    """Reference-internal identities (fp64 CPU, no GPU): the two new reference
    features must reduce to the already-validated H==HV path.

    * GVA: ``kda_chunked_fwd`` with ``HV = H*G`` == token recurrence run on q/k
      expanded by ``repeat_interleave(G)`` (same head grouping as FLA).
    * state_v_first: the V-first path must leave ``o`` unchanged and return the
      transposed state.
    """
    from kda_ref import kda_token_recurrent
    print("\n" + "=" * 90)
    print("[Part 0] reference-internal identities (fp64 CPU)")
    print("=" * 90)
    torch.manual_seed(7)
    B, T, H, K, V, G = 2, 96, 2, 16, 20, 2
    HV = H * G
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, HV, V, dtype=torch.float64)
    g = -torch.rand(B, T, HV, K, dtype=torch.float64)
    beta = torch.rand(B, T, HV, dtype=torch.float64)
    h0 = torch.randn(B, HV, K, V, dtype=torch.float64)

    o_chk, s_chk = kda_chunked_fwd(q, k, v, g, beta, initial_state=h0,
                                   output_final_state=True, chunk_size=64)
    qe = q.repeat_interleave(G, dim=2)
    ke = k.repeat_interleave(G, dim=2)
    o_tok, s_tok = kda_token_recurrent(qe, ke, v, g, beta, initial_state=h0,
                                       output_final_state=True)
    eo = (o_chk - o_tok).abs().max().item() / (o_tok.abs().max().item() + 1e-30)
    es = (s_chk - s_tok).abs().max().item() / (s_tok.abs().max().item() + 1e-30)
    print(f"  GVA  chunked vs token-recurrence(expanded): rel|do|={eo:.2e} rel|dS|={es:.2e}")

    h0v = h0.transpose(-1, -2).contiguous()
    o_svf, s_svf = kda_chunked_fwd(q, k, v, g, beta, initial_state=h0v,
                                   output_final_state=True, chunk_size=64, state_v_first=True)
    eo2 = (o_svf - o_chk).abs().max().item()
    es2 = (s_svf - s_chk.transpose(-1, -2)).abs().max().item()
    print(f"  svf  o unchanged: |do|max={eo2:.2e}   S_svf == S^T: |d|max={es2:.2e}")
    print(f"       shapes: o={tuple(o_svf.shape)} S={tuple(s_svf.shape)}")


def run(mode, B, T, H, K, V, seed, device, hv=None, tame=False, chunk_size=64,
        safe_gate=False, lower_bound=None, allow_neg_eigval=False, state_v_first=False,
        tag=""):
    from fla.ops.kda import chunk_kda  # noqa: F401
    HV = hv or H
    q, k, v, g_logit, A_log, dt_bias, beta_logit, h0 = make_inputs(
        B, T, H, K, V, seed, device, hv=HV, tame=tame)
    qn = kda_l2norm(q).to(torch.bfloat16)
    kn = kda_l2norm(k).to(torch.bfloat16)
    vb = v.to(torch.bfloat16)
    scale = K ** -0.5
    dout = (torch.randn(B, T, HV, V, device=device) * 0.1).to(torch.bfloat16)
    dfin = (torch.randn(B, HV, V, K, device=device) if state_v_first
            else torch.randn(B, HV, K, V, device=device)).float() * 0.01

    # ---- FLA side ----
    qa = qn.clone().requires_grad_(True)
    ka = kn.clone().requires_grad_(True)
    va = vb.clone().requires_grad_(True)
    ha = h0.to(torch.float32)
    if state_v_first:
        ha = ha.transpose(-1, -2).contiguous()
    ha = ha.clone().requires_grad_(True)

    kw_f = dict(state_v_first=state_v_first)
    if mode in ("A", "B"):
        g_bf = kda_gate(g_logit, A_log, dt_bias).to(torch.bfloat16)
        b_bf = torch.sigmoid(beta_logit).to(torch.bfloat16)
        ga = g_bf.clone().requires_grad_(True)
        ba = b_bf.clone().requires_grad_(True)
        kw_f["g"] = ga
        o_f, s_f = _fla_side(mode, qa, ka, va, ba, ha, scale, **kw_f)
        grads_f = [("dq", qa), ("dk", ka), ("dv", va), ("dg", ga), ("dbeta", ba), ("dh0", ha)]
        gnames = [g[0] for g in grads_f]
    else:
        gra = g_logit.to(torch.bfloat16).clone().requires_grad_(True)
        bra = beta_logit.to(torch.bfloat16).clone().requires_grad_(True)
        A_f = A_log.float().clone().requires_grad_(True)
        dt_f = dt_bias.float().clone().requires_grad_(True)
        kw_f.update(gra=gra, bra=bra, A_f=A_f, dt_f=dt_f,
                    safe_gate=safe_gate, lower_bound=lower_bound,
                    allow_neg_eigval=allow_neg_eigval)
        o_f, s_f = _fla_side(mode, qa, ka, va, bra, ha, scale, **kw_f)
        grads_f = [("dq", qa), ("dk", ka), ("dv", va), ("dg_logit", gra),
                   ("dbeta_logit", bra), ("dh0", ha), ("dA_log", A_f), ("ddt_bias", dt_f)]
        gnames = [g[0] for g in grads_f]
    (o_f.float() * dout.float()).sum().add((s_f.float() * dfin).sum()).backward()

    # ---- port reference side (fp64) ----
    qr = qn.double().requires_grad_(True)
    kr = kn.double().requires_grad_(True)
    vr = vb.double().requires_grad_(True)
    hr = h0.double()
    if state_v_first:
        hr = hr.transpose(-1, -2)
    hr = hr.clone().requires_grad_(True)
    kw_r = dict(state_v_first=state_v_first)
    if mode in ("A", "B"):
        gr = g_bf.double().requires_grad_(True)
        br = b_bf.double().requires_grad_(True)
        kw_r["g"] = gr
        o_r, s_r = _ref_side(mode, qr, kr, vr, br, hr, scale, chunk_size, **kw_r)
        grads_r = [("dq", qr), ("dk", kr), ("dv", vr), ("dg", gr), ("dbeta", br), ("dh0", hr)]
    else:
        gr = g_logit.double().requires_grad_(True)
        br = beta_logit.double().requires_grad_(True)
        Ar = A_log.double().requires_grad_(True)
        dtr = dt_bias.double().requires_grad_(True)
        kw_r.update(gr=gr, Ar=Ar, dtr=dtr,
                    lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        o_r, s_r = _ref_side(mode, qr, kr, vr, br, hr, scale, chunk_size, **kw_r)
        grads_r = [("dq", qr), ("dk", kr), ("dv", vr), ("dg_logit", gr),
                   ("dbeta_logit", br), ("dh0", hr), ("dA_log", Ar), ("ddt_bias", dtr)]
    (o_r * dout.double()).sum().add((s_r * dfin.double()).sum()).backward()

    # ---- compare ----
    hdr = (f"  [{tag}] B={B} T={T} H={H} HV={HV} K={K} V={V} chunk={chunk_size}"
           + (f" lb={lower_bound}" if lower_bound is not None else ""))
    print(hdr)
    print(f"    |o|max={o_f.float().abs().max().item():.3f}  |s|max={s_f.float().abs().max().item():.3f}")
    om, orr = stats(o_r, o_f)
    sm, srr = stats(s_r, s_f)
    print(f"    fwd  o: max_abs={om:.3e} max_rel={orr:.3e}    "
          f"S: max_abs={sm:.3e} max_rel={srr:.3e}")
    worst = max(orr, srr)
    for (nm, rr), (_, rf) in zip(grads_r, grads_f):
        if rf.grad is None:
            print(f"    bwd  {nm:>11}: FLA gave no grad")
            continue
        gm, grr = stats(rr.grad, rf.grad)
        print(f"    bwd  {nm:>11}: max_abs={gm:.3e} max_rel={grr:.3e}")
        worst = max(worst, grr)
    return worst


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    print("=" * 90)
    print("megatron-port M7: anchor src/kda_ref.py against FLA chunk_kda gate/layout boundaries")
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
    print("=" * 90)

    part0()

    print("\n" + "=" * 90)
    print("[Part A] GVA (grouped value attention, HV>H) -- explicit operands")
    print("  FLA: value head hv shares q/k head hv//(HV//H); ref expands q/k via repeat_interleave")
    print("=" * 90)
    run("A", 2, 256, 2, 128, 128, 11, device, hv=4, tag="GVA x2")
    run("A", 1, 250, 2, 64, 96, 12, device, hv=8, tag="GVA x4 non-div T")
    run("A", 1, 512, 4, 128, 128, 13, device, hv=8, tag="GVA x2 big")

    print("\n" + "=" * 90)
    print("[Part B] state_v_first -- explicit operands (initial_state/final_state in V-first [V,K])")
    print("=" * 90)
    run("B", 2, 256, 2, 128, 128, 21, device, state_v_first=True, tag="svf")
    run("B", 1, 250, 2, 64, 96, 22, device, hv=4, state_v_first=True, tag="svf+GVA x2 non-div T")

    print("\n" + "=" * 90)
    print("[Part C] safe_gate + lower_bound=-5 -- fused gate (TensorCore bounded-decay branch)")
    print("=" * 90)
    for case in [(2, 256, 2, 128, 128, 31), (1, 250, 2, 64, 96, 32), (1, 128, 4, 128, 128, 33)]:
        B, T, H, K, V, sd = case
        run("C", B, T, H, K, V, sd, device, tame=True, safe_gate=True, lower_bound=-5.0,
            tag=f"safe lb=-5")
    run("C", 2, 256, 2, 128, 128, 34, device, hv=4, tame=True, safe_gate=True, lower_bound=-5.0,
        tag="safe+GVA x2")

    print("\n" + "=" * 90)
    print("[Part D] allow_neg_eigval -- beta = 2*sigmoid(beta_logit) in (0,2), fused gate")
    print("=" * 90)
    run("D", 2, 256, 2, 128, 128, 41, device, tame=True, allow_neg_eigval=True, tag="neg_eigval")
    run("D", 1, 250, 2, 64, 96, 42, device, tame=True, allow_neg_eigval=True, tag="neg_eigval non-div T")
    run("D", 1, 250, 2, 64, 96, 43, device, hv=4, tame=True, allow_neg_eigval=True,
        tag="neg_eigval+GVA x2")

    print("\n" + "=" * 90)
    print("Done. Residual = FLA bf16-kernel error (reference is fp64). M6 default path unchanged.")
    print("=" * 90)


if __name__ == "__main__":
    main()
