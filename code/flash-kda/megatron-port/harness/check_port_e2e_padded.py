# M10-3: packed THD *with padding* (``cu_seqlens_q_padded``) anchor of the real
# Megatron-LM KDA layer.
#
# M10-2 anchored the packed/THD path but only with ``cu_seqlens_q == cu_seqlens_kv``
# and no padding (``cu_seqlens_q_padded`` left ``None``); that was explicitly listed as
# an uncovered boundary in ``docs/10`` §7.  This harness closes it by exercising the
# padding branch of the layer:
#
#   kda.py:275-281  ``_resolve_cu_seqlens(packed.cu_seqlens_q_padded, packed.cu_seqlens_q,
#                    seq_len_global, ...)``  -> when ``..._padded`` is present it is what
#                    the core operator (and causal-conv1d, kda.py:580/605) actually get;
#   common.py:482   the resolver requires ``cu_seqlens[-1] == total_sequence_length`` and
#                    per-seq lengths divisible by ``cp_size`` (=1 here);
#   kda.py:643      ``_validate_packed_cu_seqlens`` still demands ``cu_seqlens_q ==
#                    cu_seqlens_kv`` (padded on both sides).
#
# The physical token stream therefore follows the *padded* boundaries; each real
# sequence's tokens sit at the front of its padded slot, followed by fill tokens.
#
#   Part A  padded-pack fwd/bwd smoke (finite grads, output shape [T_padded,1,H]);
#   Part B  core-operator anchor in padded mode: FLA ``chunk_kda`` (with the padded
#           ``cu_seqlens`` it is handed) vs the fp64 reference slicing by the same padded
#           boundaries;
#   Part C  fill-sensitivity attribution on the captured core operands: flip the fill of
#           ``q/k/v/g/beta`` one at a time and see which one perturbs *real* core outputs;
#   Part D  module-level padding isolation (FLA core, no-grad): per padded segment, change
#           only that segment's fill and split the real-token response into
#           (own slot) vs (other slots) — the safety property is that *other* slots see
#           exactly zero;
#   Part E  full-module padded anchor: FLA core vs fp64 ref core under identical padded
#           ``packed_seq_params``.
#
# Same two container import-time workarounds as M10-1/M10-2.
#
# Run (GPU, single process):
#   scripts/lab.sh exec "cd \$MEGATRON_LM && MEGATRON_LM_DIR=\$MEGATRON_LM \
#       python <abs>/harness/check_port_e2e_padded.py"

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import check_port_e2e_packed as pk  # noqa: E402  (reuse M10-2 helpers)


# ---------------------------------------------------------------------------------------
# padded PackedSeqParams
# ---------------------------------------------------------------------------------------
def make_packed_seq_params_padded(cu_actual, cu_padded):
    """Build THD params whose *physical* layout is ``cu_padded``.

    ``cu_seqlens_q``/``..._kv`` are the true (unpadded) boundaries; ``..._padded`` carry
    the memory layout the layer must use.  ``max_seqlen_*`` uses the padded max (that is
    what the real token stream sees), matching how Megatron builds padded THD batches.
    """
    cu_a = torch.IntTensor(cu_actual).cuda()
    cu_p = torch.IntTensor(cu_padded).cuda()
    m_padded = int((cu_p[1:] - cu_p[:-1]).max().item())
    return pk.PackedSeqParams(
        cu_seqlens_q=cu_a,
        cu_seqlens_kv=cu_a,
        cu_seqlens_q_padded=cu_p,
        cu_seqlens_kv_padded=cu_p,
        max_seqlen_q=m_padded,
        max_seqlen_kv=m_padded,
        qkv_format="thd",
    )


def segments(cu_actual, cu_padded):
    """Yield (u0, u1, p0, p1): real [u0,u1) maps into padded slot [p0,p1)."""
    for i in range(len(cu_actual) - 1):
        u0, u1 = int(cu_actual[i]), int(cu_actual[i + 1])
        p0 = int(cu_padded[i])
        yield u0, u1, p0, p0 + (u1 - u0)


def slot_bounds(cu_padded):
    return [(int(cu_padded[i]), int(cu_padded[i + 1])) for i in range(len(cu_padded) - 1)]


def build_padded_input(content, cu_actual, cu_padded, fill):
    """Scatter ``content`` [T_actual,1,H] into a padded [T_padded,1,H] stream.

    ``fill`` is a scalar or a [T_padded,1,H] tensor used for the fill slots.
    """
    T_pad = int(cu_padded[-1])
    H = content.shape[-1]
    out = torch.zeros(T_pad, 1, H, device=content.device, dtype=content.dtype)
    if torch.is_tensor(fill):
        out.copy_(fill)
    else:
        out.fill_(float(fill))
    for u0, u1, p0, p1 in segments(cu_actual, cu_padded):
        out[p0:p1] = content[u0:u1]
    return out


def clone_module(kda, config):
    pk.model_parallel_cuda_manual_seed(4242)
    m = pk.build_kda(config)
    m.load_state_dict({k: v.detach().clone() for k, v in kda.state_dict().items()})
    return m


# ---------------------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------------------
def part_a(kda, config, cases, dtype):
    print("  -- Part A: padded pack fwd/bwd smoke (%s)" % str(dtype).split(".")[-1])
    ok = True
    for cu_a, cu_p in cases:
        T = int(cu_p[-1])
        psp = make_packed_seq_params_padded(cu_a, cu_p)
        torch.manual_seed(T * 100 + len(cu_a))
        hs = torch.randn(T, 1, config.hidden_size, device="cuda", dtype=dtype,
                         requires_grad=True)
        kda.zero_grad(set_to_none=True)
        out, bias = kda(hs, None, packed_seq_params=psp)
        ok_shape = out.shape == hs.shape and out.dtype == dtype and bias is None
        out.float().square().mean().backward()
        finite_hs = torch.isfinite(hs.grad).all().item()
        bad = [n for n, p in kda.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all().item()]
        ok = ok and ok_shape and finite_hs and not bad
        print("     a=%s p=%s  T_pad=%4d  shape/dtype=%s  hs.grad finite=%s  bad=%d"
              % (cu_a, cu_p, T, ok_shape, finite_hs, len(bad)))
    return ok


# ---------------------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------------------
def part_b(ref, fla_gdr, captured, cu_padded):
    print("  -- Part B: padded core-operator anchor (FLA vs fp64 ref, padded slicing)")
    keys = ["q", "k", "v", "g", "beta", "A_log", "dt_bias"]
    cu = captured.get("cu_seqlens")
    assert cu is not None, "expected the packed layer to pass cu_seqlens to the core"
    assert cu.tolist() == cu_padded, (
        "core got %s, expected the padded boundaries %s" % (cu.tolist(), cu_padded))
    print("     core cu_seqlens (resolved) = %s  (padded; actual differs)" % cu.tolist())
    args = {k: captured[k].detach().clone() for k in keys}
    base = dict(
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=bool(captured.get("use_qk_l2norm_in_kernel", False)),
        use_gate_in_kernel=bool(captured.get("use_gate_in_kernel", True)),
        safe_gate=bool(captured.get("safe_gate", True)),
        lower_bound=captured.get("lower_bound", -5.0),
        cu_seqlens=cu, cp_context=None,
    )

    def run(gdr, x):
        return gdr(**{**x, **base})

    o_f, _ = run(fla_gdr, {k: v.clone().requires_grad_(True) for k, v in args.items()})
    o_r, _ = run(pk.make_ref_gdr(ref), {k: v.clone().requires_grad_(True) for k, v in args.items()})
    print("     fwd o: max_abs=%.3e  rel=%.3e"
          % ((o_f.float() - o_r.float()).abs().max().item(), pk.relerr(o_f, o_r)))
    xf = {k: v.clone().requires_grad_(True) for k, v in args.items()}
    of, _ = run(fla_gdr, xf)
    of.float().square().mean().backward()
    xr = {k: v.clone().requires_grad_(True) for k, v in args.items()}
    orr, _ = run(pk.make_ref_gdr(ref), xr)
    orr.float().square().mean().backward()
    worst = 0.0
    for k in ["q", "k", "v", "g", "beta"]:
        if xf[k].grad is None or xr[k].grad is None:
            continue
        d = (xf[k].grad.float() - xr[k].grad.float()).abs().max().item()
        worst = max(worst, d)
        print("     grad %-5s max_abs=%.3e  rel=%.3e" % (k, d, pk.relerr(xf[k].grad, xr[k].grad)))
    print("     core worst grad max_abs=%.3e" % worst)


# ---------------------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------------------
def part_c(fla_gdr, captured, cu_actual, cu_padded):
    """Which captured operand's *fill* perturbs real-token core outputs?"""
    print("  -- Part C: fill-sensitivity attribution on captured core operands")
    keys = ["q", "k", "v", "g", "beta", "A_log", "dt_bias"]
    cu = captured["cu_seqlens"]
    base = dict(initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=bool(captured.get("use_qk_l2norm_in_kernel", False)),
                use_gate_in_kernel=bool(captured.get("use_gate_in_kernel", True)),
                safe_gate=bool(captured.get("safe_gate", True)),
                lower_bound=captured.get("lower_bound", -5.0),
                cu_seqlens=cu, cp_context=None)
    ops = {k: captured[k].detach().clone() for k in keys}
    T = ops["q"].shape[1]
    # real physical positions = for each segment, the [p0, p1) real slice of its slot
    real = []
    for u0, u1, p0, p1 in segments(cu_actual, cu_padded):
        real += list(range(p0, p1))
    fill = [t for t in range(T) if t not in real]
    torch.manual_seed(11)
    ops2 = {k: ops[k].clone() for k in keys}
    for k in ["q", "k", "v", "g", "beta"]:
        for t in fill:
            ops2[k][:, t] = torch.randn_like(ops2[k][:, t])

    with torch.no_grad():
        o1, _ = fla_gdr(**{k: ops[k] for k in keys}, **base)
        for probe in ["q", "k", "v", "g", "beta"]:
            mix = {k: ops[k] for k in keys}
            m = dict(mix)
            tmp = ops[probe].clone()
            for t in fill:
                tmp[:, t] = ops2[probe][:, t]
            m[probe] = tmp
            om, _ = fla_gdr(**{k: m[k] for k in keys}, **base)
            d = (o1[0].float() - om[0].float()).abs().amax(dim=(1, 2))  # [T]
            dreal = d[real]
            tmax = real[int(dreal.argmax().item())]
            print("     flip fill of %-5s: real-token max=%.3e (at t=%d)  n_nonzero=%d/%d"
                  % (probe, dreal.max().item(), tmax,
                     int((dreal > 0).sum().item()), len(real)))


# ---------------------------------------------------------------------------------------
# Part D
# ---------------------------------------------------------------------------------------
def part_d(kda, config, cases, dtype, seeds=(1, 2, 3)):
    """Per-segment fill isolation on the real layer (FLA core, forward only)."""
    print("  -- Part D: module padding isolation (per-segment fill flip, %s)"
          % str(dtype).split(".")[-1])
    worst_own = worst_cross = 0.0
    for cu_a, cu_p in cases:
        T_act, T_pad = int(cu_a[-1]), int(cu_p[-1])
        psp = make_packed_seq_params_padded(cu_a, cu_p)
        segs = list(segments(cu_a, cu_p))
        for seed in seeds:
            torch.manual_seed(seed * 131 + T_pad)
            content = torch.randn(T_act, 1, config.hidden_size, device="cuda", dtype=dtype)
            fA = torch.randn(T_pad, 1, config.hidden_size, device="cuda", dtype=dtype)
            fB = torch.randn(T_pad, 1, config.hidden_size, device="cuda", dtype=dtype)
            xA = build_padded_input(content, cu_a, cu_p, fA)
            with torch.no_grad():
                oA, _ = kda(xA, None, packed_seq_params=psp)
            bounds = slot_bounds(cu_p)
            worst_seg_own = worst_seg_cross = 0.0
            for i in range(len(segs)):
                s0, s1 = bounds[i]
                _, _, p0, p1 = segs[i]
                xM = xA.clone()
                xM[p1:s1] = fB[p1:s1]  # only segment i's *fill tail* differs (real untouched)
                with torch.no_grad():
                    oM, _ = kda(xM, None, packed_seq_params=psp)
                d = (oA[:, 0].float() - oM[:, 0].float()).abs().amax(dim=-1)  # [T]
                own = d[segs[i][2]:segs[i][3]].max().item()
                cross = max(d[segs[j][2]:segs[j][3]].max().item()
                            for j in range(len(segs)) if j != i)
                worst_seg_own = max(worst_seg_own, own)
                worst_seg_cross = max(worst_seg_cross, cross)
            worst_own = max(worst_own, worst_seg_own)
            worst_cross = max(worst_cross, worst_seg_cross)
    print("     worst over %d cases x %d seeds, real-token positions:" % (len(cases), len(seeds)))
    print("       own-slot fill flip      : out max_abs=%.3e" % worst_own)
    print("       other-slot fill flip    : out max_abs=%.3e  (cross-segment isolation)"
          % worst_cross)
    return worst_own, worst_cross


# ---------------------------------------------------------------------------------------
# Part E
# ---------------------------------------------------------------------------------------
def part_e(kda, kda_r, config, cases, dtype):
    print("  -- Part E: full-module padded (FLA-core vs ref-core, identical weights)")
    worst_out = worst_hs = worst_p = 0.0
    for cu_a, cu_p in cases:
        T = int(cu_p[-1])
        psp = make_packed_seq_params_padded(cu_a, cu_p)
        torch.manual_seed(T * 100 + len(cu_a))
        hs_a = torch.randn(T, 1, config.hidden_size, device="cuda", dtype=dtype,
                           requires_grad=True)
        hs_b = hs_a.detach().clone().requires_grad_(True)
        kda.zero_grad(set_to_none=True)
        kda_r.zero_grad(set_to_none=True)
        out_a, _ = kda(hs_a, None, packed_seq_params=psp)
        out_b, _ = kda_r(hs_b, None, packed_seq_params=psp)
        do = (out_a.float() - out_b.float()).abs().max().item()
        worst_out = max(worst_out, do)
        out_a.float().square().mean().backward()
        out_b.float().square().mean().backward()
        dh = (hs_a.grad.float() - hs_b.grad.float()).abs().max().item()
        worst_hs = max(worst_hs, dh)
        dp = 0.0
        pa = dict(kda.named_parameters())
        for name, p in kda_r.named_parameters():
            ga, gb = pa[name].grad, p.grad
            if ga is None or gb is None:
                continue
            dp = max(dp, (ga.float() - gb.float()).abs().max().item())
        worst_p = max(worst_p, dp)
        print("     a=%s p=%s T_pad=%4d  out=%.3e hs.grad=%.3e param.grad=%.3e"
              % (cu_a, cu_p, T, do, dh, dp))
    print("     worst: out=%.3e hs.grad=%.3e param.grad=%.3e"
          % (worst_out, worst_hs, worst_p))
    return worst_out, worst_hs, worst_p


def part_f(kda, config, cases, dtype, seeds=(1, 2, 3)):
    """Padded real-token outputs vs the unpadded layout on the real layer (FLA core)."""
    print("  -- Part F: padded vs unpadded real-token equivalence (FLA core, %s)"
          % str(dtype).split(".")[-1])
    worst_o = worst_g = 0.0
    for cu_a, cu_p in cases:
        T_act, T_pad = int(cu_a[-1]), int(cu_p[-1])
        psp_p = make_packed_seq_params_padded(cu_a, cu_p)
        psp_u = pk.make_packed_seq_params(cu_a)
        for seed in seeds:
            torch.manual_seed(seed * 313 + T_pad)
            content = torch.randn(T_act, 1, config.hidden_size, device="cuda", dtype=dtype)
            fill = torch.randn(T_pad, 1, config.hidden_size, device="cuda", dtype=dtype)
            xu = content.detach().clone().requires_grad_(True)
            xp = build_padded_input(content, cu_a, cu_p, fill).requires_grad_(True)
            ou, _ = kda(xu, None, packed_seq_params=psp_u)
            op, _ = kda(xp, None, packed_seq_params=psp_p)
            ou.float().square().mean().backward()
            op.float().square().mean().backward()
            for u0, u1, p0, p1 in segments(cu_a, cu_p):
                worst_o = max(worst_o, (op[p0:p1, 0].float() - ou[u0:u1, 0].float())
                              .abs().max().item())
                worst_g = max(worst_g, (xp.grad[p0:p1, 0].float() - xu.grad[u0:u1, 0].float())
                              .abs().max().item())
    print("     worst over %d cases x %d seeds (real pos): out=%.3e  hs.grad=%.3e"
          % (len(cases), len(seeds), worst_o, worst_g))
    return worst_o, worst_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="float32,bfloat16")
    # each case is "actual | padded" cu_seqlens (padded total = physical length)
    ap.add_argument("--cases", default="0,50,110|0,64,128;0,50,100,220|0,64,128,256")
    args = ap.parse_args()

    cases = []
    for c in args.cases.split(";"):
        a_s, p_s = c.split("|")
        cases.append(([int(x) for x in a_s.split(",")], [int(x) for x in p_s.split(",")]))
    dtypes = [getattr(torch, t.strip()) for t in args.dtypes.split(",")]

    mega = pk._import_megatron()
    print("Megatron-LM:", mega)
    ref = pk._load_kda_ref()
    pk.init_dist()

    for dtype in dtypes:
        print("=== dtype=%s cases=%s ===" % (str(dtype).split(".")[-1], cases))
        kda, kda_r, config, captured, fla_gdr = pk.build_module_case(ref, dtype)
        part_a(kda, config, cases, dtype)
        cu_a0, cu_p0 = cases[0]
        psp0 = make_packed_seq_params_padded(cu_a0, cu_p0)
        torch.manual_seed(int(cu_p0[-1]) * 100 + len(cu_a0))
        hs = torch.randn(int(cu_p0[-1]), 1, config.hidden_size, device="cuda", dtype=dtype,
                         requires_grad=True)
        kda.zero_grad(set_to_none=True)
        kda(hs, None, packed_seq_params=psp0)  # populates `captured`
        part_b(ref, fla_gdr, captured, cu_p0)
        part_c(fla_gdr, captured, cu_a0, cu_p0)
        part_d(kda, config, cases, dtype)
        part_e(kda, kda_r, config, cases, dtype)
        part_f(kda, config, cases, dtype)

    pk.ps.destroy_model_parallel()
    print("ALL DONE")


if __name__ == "__main__":
    main()
