# M13-2: the KDA-specific *fused pre-GDR* **native Triton backward**.
#
# M13-1 (``src/kda_pregdr_fused.py``) built the fused pre-GDR as a forward-only Triton
# family and used a *torch-primitive* reference backward in its autograd wrapper.  That
# wrapper has two problems, one of which was never noticed:
#
#   * it replays the eager algebra in torch (not the fused kernel's own backward); and
#   * it returns gradients only for ``qkvfg`` / ``beta`` -- **``conv1d.weight`` /
#     ``conv1d.bias`` grads are dropped** (the reference inside the function used detached
#     clones), so M13-1's Part C "full-layer" comparison silently skipped them via
#     ``if ga is None or gb is None: continue``.
#
# ``src/kda_pregdr_bwd.py`` closes both: three backward kernels (SiLU -> depthwise causal
# conv dX/dW/dbias, L2-norm, sigmoid) and an autograd ``Function`` that takes the conv
# params as *inputs*.  This harness anchors the backward, without touching Megatron-LM:
#
#   Part A  isolated backward kernels vs an analytic torch reference on the **same** saved
#           intermediates (isolates the kernel, not the conv backend);
#   Part B  end-to-end pre-GDR backward (native) vs ``torch.autograd`` of the torch-primitive
#           forward, across shapes/dtypes (the bf16 residual is forward quantization);
#   Part C  full-layer drop-in with the native backward: forward, input grad, and **all**
#           parameter grads -- explicitly including the ``conv1d`` params M13-1 skipped;
#   Part D  timing: isolated eager pre-GDR vs the native fused family (forward+backward).
#
# M13-3 adds the fused producers (``_kda_l2norm_silu_qk_bwd_kernel`` / ``_kda_silu_v_bwd_kernel``
# write ``da`` in place, dropping the cat/dqk_in/dconv/da materialisations + ``fd()`` casts)
# and two more parts:
#   Part E  ``conv_bias=True`` end-to-end on a stub module with a **non-zero** bias (the real
#           Megatron KDA layer builds conv1d with ``bias=None``, so ``dbias`` is never
#           exercised there -- see M13-2's ``dbias=nan``);
#   Part F  fused vs unfused backward on the **same** saved intermediates (proves the fusion
#           is numerically transparent).
#
# Container import workarounds are the same as M10-M13 (nvrx version patch +
# ``EP_SUPPRESS_NCCL_CHECK=1``).
#
# Run (GPU, single process; pick an idle card):
#   scripts/lab.sh exec "cd \$MEGATRON_LM && CUDA_VISIBLE_DEVICES=1 MEGATRON_LM_DIR=\$MEGATRON_LM \
#       python <abs>/harness/check_port_pregdr_bwd_kernel.py"

import argparse
import importlib.util
import os
import sys
import types

os.environ.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import torch.nn.functional as F


def _import_local(modname, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "src", filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_M13 = None


def _import_m13():
    # reuse M13-1's harness helpers (Megatron import, build_kda, capture, timers);
    # cached so every caller sees the same instance (with ``_import_megatron`` globals set).
    global _M13
    if _M13 is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "check_port_pregdr_kernel.py")
        spec = importlib.util.spec_from_file_location("m13", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _M13 = mod
    return _M13


_KEYS = ["q", "k", "v", "gate", "beta", "g"]


def _mk_grads(outs, seed=0):
    g = torch.Generator(device=outs[0].device).manual_seed(seed)
    return [torch.randn(o.shape, device=o.device, generator=g).float() for o in outs]


# ---------------------------------------------------------------------------------------
# Part A: isolated backward kernels vs analytic torch reference on the same intermediates
# ---------------------------------------------------------------------------------------
def part_a(kda, bwd, dtype, cases):
    print("  -- Part A: isolated backward kernels vs analytic reference (%s)"
          % str(dtype).split(".")[-1])
    m13 = _import_m13()
    dims = bwd._dims(kda)
    qk, vd, Hk, Hv, KD, VD, W, has_bias, activation = dims
    C = 2 * qk + vd
    w = kda.conv1d.weight.detach()
    b = kda.conv1d.bias.detach() if kda.conv1d.bias is not None else None
    for (T, B) in cases:
        cap = m13._capture(kda, dtype, T, B, seed=T * 7 + B)
        qkvfg, beta = cap["qkvfg"], cap["beta_in"]
        outs, saves = bwd._forward_raw_save(qkvfg, beta, w, b, dims)
        up = _mk_grads(outs, seed=T + B)
        dqkvfg, dbeta, dweight, dbias = bwd._backward_raw(qkvfg, beta, w, saves, up, dims)

        x = saves["x"].float()
        pre = saves["pre"].float()
        qk_in = saves["query_key"].float()
        dy = torch.cat([up[0].float(), up[1].float()], dim=2)          # [B,T,2Hk,KD]
        rstd = 1 / torch.sqrt((qk_in * qk_in).sum(-1, keepdim=True) + 1e-6)
        dot = (dy * qk_in).sum(-1, keepdim=True)
        dqk_ref = rstd * dy - rstd ** 3 * qk_in * dot
        dconv = torch.cat([dqk_ref.reshape(B, T, 2 * qk), up[2].float().reshape(B, T, vd)], dim=2)
        sig = torch.sigmoid(pre)
        da = dconv * sig * (1 + pre * (1 - sig))
        xr = x.detach().clone().requires_grad_(True)
        wr = w.float().detach().clone().requires_grad_(True)
        br = b.float().detach().clone().requires_grad_(True) if b is not None else None
        co = F.conv1d(xr.transpose(1, 2), wr, br, padding=W - 1, groups=C)[..., :T].transpose(1, 2)
        (da * co).sum().backward()
        dx_native = dqkvfg[:, :, :C].permute(1, 0, 2).float()
        dbeta_ref = up[4].float() * torch.sigmoid(beta.transpose(0, 1).float()) * \
            (1 - torch.sigmoid(beta.transpose(0, 1).float()))
        dbi_err = (dbias.float() - br.grad).abs().max().item() if br is not None else float("nan")
        print("     T=%-5d B=%d  dweight=%.3e dbias=%.3e dx=%.3e dbeta=%.3e"
              % (T, B,
                 (dweight.float() - wr.grad).abs().max().item(),
                 dbi_err,
                 (dx_native - xr.grad).abs().max().item(),
                 (dbeta.permute(1, 0, 2).float() - dbeta_ref).abs().max().item()))


# ---------------------------------------------------------------------------------------
# Part B: end-to-end backward vs torch.autograd through the torch-primitive forward
# ---------------------------------------------------------------------------------------
def _ref_fwd_diff(qkvfg, beta, w, b, dims):
    qk, vd, Hk, Hv, KD, VD, W, has_bias, activation = dims
    T, B, Cin = qkvfg.shape
    C = 2 * qk + vd
    q_ = qkvfg.transpose(0, 1)                                     # [B,T,Cin]
    x = q_[..., :C].transpose(1, 2).contiguous()
    y = F.conv1d(x, w, b, padding=W - 1, groups=C)[..., :T]
    y = y * torch.sigmoid(y) if activation else y
    conv = y.transpose(1, 2).contiguous()
    qk_ = conv[..., :2 * qk].reshape(B, T, 2 * Hk, KD).float()
    qk_ = (qk_ / torch.sqrt((qk_ * qk_).sum(-1, keepdim=True) + 1e-6)).to(conv.dtype)
    q, k = torch.split(qk_, [Hk, Hk], dim=2)
    val = conv[..., 2 * qk:].reshape(B, T, Hv, VD)
    raw_g = q_[..., C:C + qk].reshape(B, T, Hk, KD)
    gate = q_[..., C + qk:].reshape(B, T, Hv, VD)
    beta_sig = beta.transpose(0, 1).float().sigmoid()
    return q, k, val, gate, beta_sig, raw_g


def part_b(kda, bwd, dtype, cases):
    print("  -- Part B: native backward vs torch.autograd (torch-primitive forward) (%s)"
          % str(dtype).split(".")[-1])
    m13 = _import_m13()
    dims = bwd._dims(kda)
    for (T, B) in cases:
        cap = m13._capture(kda, dtype, T, B, seed=T * 11 + B)
        qkvfg0, beta0 = cap["qkvfg"], cap["beta_in"]
        w0 = kda.conv1d.weight.detach()
        b0 = kda.conv1d.bias.detach() if kda.conv1d.bias is not None else None
        outs = bwd._forward_raw_save(qkvfg0, beta0, w0, b0, dims)[0]
        up = _mk_grads(outs, seed=T * 3 + B)

        # native: Function takes conv params as inputs (bias may be absent)
        qn = qkvfg0.detach().clone().requires_grad_(True)
        bn = beta0.detach().clone().requires_grad_(True)
        wn = w0.detach().clone().requires_grad_(True)
        bnn = b0.detach().clone().requires_grad_(True) if b0 is not None else None
        outn = bwd._FusedPreGDRNativeAutograd.apply(kda, qn, bn, wn, bnn, B, T)
        (sum((o.float() * g).sum() for o, g in zip(outn, up))).backward()

        # reference: differentiable torch forward
        qr = qkvfg0.detach().clone().requires_grad_(True)
        br2 = beta0.detach().clone().requires_grad_(True)
        wr = w0.detach().clone().requires_grad_(True)
        br = b0.detach().clone().requires_grad_(True) if b0 is not None else None
        outr = _ref_fwd_diff(qr, br2, wr, br, dims)
        (sum((o.float() * g).sum() for o, g in zip(outr, up))).backward()

        dq = (qn.grad.float() - qr.grad.float()).abs().max().item()
        db = (bn.grad.float() - br2.grad.float()).abs().max().item()
        dw = (wn.grad.float() - wr.grad).abs().max().item()
        dbi = (bnn.grad.float() - br.grad).abs().max().item() if br is not None else float("nan")
        print("     T=%-5d B=%d  dqkvfg=%.3e dbeta=%.3e dweight=%.3e dbias=%.3e"
              % (T, B, dq, db, dw, dbi))


# ---------------------------------------------------------------------------------------
# Part C: full-layer drop-in with the native backward (all params, incl. conv1d)
# ---------------------------------------------------------------------------------------
def part_c(bwd, dtype, T=64, B=2):
    print("  -- Part C: full-layer drop-in (native fused pre-GDR, same weights) (%s)"
          % str(dtype).split(".")[-1])
    m13 = _import_m13()
    m13.model_parallel_cuda_manual_seed(123)
    kda = m13.build_kda(m13.make_config(dtype))
    state = {k: v.detach().clone() for k, v in kda.state_dict().items()}
    m13.model_parallel_cuda_manual_seed(123)
    kda_f = m13.build_kda(m13.make_config(dtype))
    kda_f.load_state_dict(state)

    def patched(self, qkvfg, beta, batch, seq_len, *args, **kwargs):
        return bwd.kda_pre_gdr_fused_native(self, qkvfg, beta, batch, seq_len)

    kda_f.pre_gated_delta_rule = types.MethodType(patched, kda_f)

    torch.manual_seed(T * 13 + B)
    xa = torch.randn(T, B, 128, device="cuda", dtype=dtype, requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    kda.zero_grad(set_to_none=True)
    kda_f.zero_grad(set_to_none=True)
    oa, _ = kda(xa, None)
    ob, _ = kda_f(xb, None)
    d_out = (oa.float() - ob.float()).abs().max().item()
    oa.float().sum().backward()
    ob.float().sum().backward()
    d_hs = (xa.grad.float() - xb.grad.float()).abs().max().item()

    pa = dict(kda.named_parameters())
    dp = 0.0
    scale = 0.0
    n_cmp = 0
    n_skip = 0
    conv_dw = conv_db = None
    for name, p in kda_f.named_parameters():
        ga, gb = pa[name].grad, p.grad
        if ga is None or gb is None:
            n_skip += 1
            continue
        n_cmp += 1
        d = (ga.float() - gb.float()).abs().max().item()
        dp = max(dp, d)
        scale = max(scale, ga.float().abs().max().item())
        if name == "conv1d.weight":
            conv_dw = (ga.float().abs().max().item(), gb.float().abs().max().item())
        if name == "conv1d.bias":
            conv_db = (ga.float().abs().max().item(), gb.float().abs().max().item())
    print("     out max_abs=%.3e  hs.grad max_abs=%.3e  param.grad max_abs=%.3e (rel=%.3e)"
          % (d_out, d_hs, dp, dp / (scale + 1e-30)))
    fdw = conv_dw if conv_dw is not None else (float("nan"), float("nan"))
    fdb = conv_db if conv_db is not None else (float("nan"), float("nan"))
    print("     params compared=%d skipped=%d   conv1d.weight.grad (eager=%.3e native=%.3e)  "
          "conv1d.bias.grad (eager=%.3e native=%.3e)"
          % (n_cmp, n_skip, fdw[0], fdw[1], fdb[0], fdb[1]))


# ---------------------------------------------------------------------------------------
# Part D: timing (isolated eager pre-GDR vs native fused family, forward+backward)
# ---------------------------------------------------------------------------------------
def part_d(kda, bwd, dtype, cases):
    print("  -- Part D: timing, isolated pre-GDR fwd+bwd (%s)" % str(dtype).split(".")[-1])
    m13 = _import_m13()
    cp_group = m13.ps.get_context_parallel_group()

    def eager(qkvfg, beta, batch, seq_len):
        return kda.pre_gated_delta_rule(qkvfg, beta, batch, seq_len, 1, cp_group)

    def fused_call(qkvfg, beta, batch, seq_len):
        return bwd.kda_pre_gdr_fused_native(kda, qkvfg, beta, batch, seq_len)

    for (T, B) in cases:
        cap = m13._capture(kda, dtype, T, B, seed=T * 5 + B)
        q = cap["qkvfg"].detach().clone().requires_grad_(True)
        be = cap["beta_in"].detach().clone().requires_grad_(True)
        ba, sl = cap["batch"], cap["seq_len"]

        def run_eager():
            outs = eager(q, be, ba, sl)
            sum(o.float().sum() for o in outs[:6]).backward()

        def run_fused():
            outs = fused_call(q, be, ba, sl)
            sum(o.float().sum() for o in outs[:6]).backward()

        te_wall = m13._time(run_eager)
        tf_wall = m13._time(run_fused)
        te_dev = m13._dev_time(run_eager)
        tf_dev = m13._dev_time(run_fused)
        print("     T=%-5d B=%d  wall: eager=%8.2f native=%8.2f (%.3fx) | "
              "dev: eager=%8.2f native=%8.2f (%.3fx)"
              % (T, B, te_wall, tf_wall, te_wall / tf_wall, te_dev, tf_dev, te_dev / tf_dev))


# ---------------------------------------------------------------------------------------
# Part E: conv_bias=True end-to-end (stub module, non-zero bias) -- the real layer has
# bias=None, so this is the only place dbias is driven non-zero through autograd.
# ---------------------------------------------------------------------------------------
def part_e(bwd, dtype, cases):
    print("  -- Part E: conv_bias=True end-to-end vs torch.autograd (stub, non-zero bias) (%s)"
          % str(dtype).split(".")[-1])
    Fm = bwd.fused
    Hk = Hv = 4
    KD = VD = 32
    W = 4
    qk, vd = Hk * KD, Hv * VD
    C = 2 * qk + vd
    dev = "cuda"
    for (T, B) in cases:
        stub = Fm._Stub(qk, vd, Hk, Hv, KD, VD, W, dtype).to(dev)
        torch.nn.init.normal_(stub.conv1d.weight, std=0.3)
        torch.nn.init.normal_(stub.conv1d.bias, std=0.2)
        qkvfg = torch.randn(T, B, C + qk + vd, device=dev, dtype=dtype)
        beta = torch.randn(T, B, Hk, device=dev, dtype=dtype)
        q = qkvfg.detach().clone().requires_grad_(True)
        be = beta.detach().clone().requires_grad_(True)
        out = bwd.kda_pre_gdr_fused_native(stub, q, be, B, T)
        g = _mk_grads(out[:6], seed=T * 17 + B)
        (sum((o.float() * gg).sum() for o, gg in zip(out[:6], g))).backward()

        qr = qkvfg.detach().clone().requires_grad_(True)
        br = beta.detach().clone().requires_grad_(True)
        wr = stub.conv1d.weight.detach().clone().requires_grad_(True)
        bir = stub.conv1d.bias.detach().clone().requires_grad_(True)
        ref = Fm._ref_torch(qr, br, wr, bir, Hk, Hv, KD, VD, W)
        (sum((o.float() * gg).sum() for o, gg in zip(ref, g))).backward()

        dq = (q.grad.float() - qr.grad.float()).abs().max().item()
        db = (be.grad.float() - br.grad.float()).abs().max().item()
        dw = (stub.conv1d.weight.grad.float() - wr.grad.float()).abs().max().item()
        dbi = (stub.conv1d.bias.grad.float() - bir.grad.float()).abs().max().item()
        print("     T=%-5d B=%d  dqkvfg=%.3e dbeta=%.3e dweight=%.3e dbias=%.3e  "
              "(bias.grad native=%.3e ref=%.3e)"
              % (T, B, dq, db, dw, dbi,
                 stub.conv1d.bias.grad.float().abs().max().item(),
                 bir.grad.float().abs().max().item()))


# ---------------------------------------------------------------------------------------
# Part F: fused vs unfused backward on identical saved intermediates / grads.
# ---------------------------------------------------------------------------------------
def part_f(kda, bwd, dtype, cases):
    print("  -- Part F: fused vs unfused backward (same intermediates/grads) (%s)"
          % str(dtype).split(".")[-1])
    m13 = _import_m13()
    dims = bwd._dims(kda)
    C = 2 * dims[0] + dims[1]
    for (T, B) in cases:
        cap = m13._capture(kda, dtype, T, B, seed=T * 19 + B)
        qkvfg, beta = cap["qkvfg"], cap["beta_in"]
        w = kda.conv1d.weight.detach()
        b = kda.conv1d.bias
        outs, saves = bwd._forward_raw_save(qkvfg, beta, w, b, dims)
        up = _mk_grads(outs, seed=T * 23 + B)
        f = bwd._backward_raw(qkvfg, beta, w, saves, up, dims, fused=True)
        u = bwd._backward_raw(qkvfg, beta, w, saves, up, dims, fused=False)
        names = ("dqkvfg", "dbeta", "dweight", "dbias")
        line = []
        for nm, a, c in zip(names, f, u):
            line.append("%s=%.3e" % (nm, (a.float() - c.float()).abs().max().item()))
        print("     T=%-5d B=%d  %s" % (T, B, "  ".join(line)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--skip-c", action="store_true")
    args = ap.parse_args()
    dtypes = [getattr(torch, t.strip()) for t in args.dtypes.split(",")]

    m13 = _import_m13()
    mega = m13._import_megatron()
    print("Megatron-LM:", mega)
    bwd = _import_local("kda_pregdr_bwd", "kda_pregdr_bwd.py")
    m13.init_dist()

    cases_a = [(64, 2), (128, 2), (250, 1), (256, 1), (1024, 2)]
    cases_b = [(64, 2), (250, 1), (128, 2)]
    cases_t = [(64, 2), (256, 2), (1024, 2), (1024, 8)]

    for dtype in dtypes:
        print("=== dtype=%s ===" % str(dtype).split(".")[-1])
        m13.model_parallel_cuda_manual_seed(123)
        kda = m13.build_kda(m13.make_config(dtype))
        part_a(kda, bwd, dtype, cases_a)
        part_b(kda, bwd, dtype, cases_b)
        part_c(bwd, dtype)
        part_f(kda, bwd, dtype, cases_b)
        part_d(kda, bwd, dtype, cases_t)
        part_e(bwd, dtype, cases_b)

    m13.ps.destroy_model_parallel()
    print("ALL DONE")


if __name__ == "__main__":
    main()
