# M10-2: packed / THD (varlen) anchor of the *real* Megatron-LM KDA layer.
#
# M10-1 anchored the real ``KimiDeltaAttention`` module in the default SBHD path
# (``packed_seq_params=None``, B>=1, one sequence per batch row).  This harness extends
# the same two-granularity anchor to the **packed THD** path
# (``PackedSeqParams(qkv_format="thd", cu_seqlens=...)``, batch==1, several sequences
# flattened along the token dim):
#
#   Part A  packed THD fwd/bwd smoke (finite grads, output shape [T_total,1,H]);
#   Part B  core-operator anchor in packed mode: capture the operands the real layer hands
#           ``self.gated_delta_rule`` (now with ``cu_seqlens``) and compare FLA ``chunk_kda``
#           vs the fp64 reference on *those* operands (reference runs a per-sequence loop,
#           exactly the M8-1 varlen semantics);
#   Part C  full-module packed anchor: swap the core operator for the fp64 reference in a
#           second (weight-identical) module and compare output + ``hidden_states.grad`` +
#           parameter grads, both with ``packed_seq_params``;
#   Part D  packed == unpacked self-consistency: the same sequence content fed as either
#           [T,B,H] (SBHD) or the flattened [T*B,1,H] pack must agree (fwd + hs.grad +
#           param.grad) — mirrors Megatron's own test_kda_packed_matches_unpacked_cp1.
#
# Two container import-time workarounds are needed (same as M10-1, both unrelated to KDA):
# patch ``nvidia_resiliency_ext.__version__`` to satisfy the Megatron >=0.6.0 assertion and
# set ``EP_SUPPRESS_NCCL_CHECK=1`` to bypass DeepEP's libnccl byte comparison.
#
# Run (GPU, single process):
#   scripts/lab.sh exec "cd \$MEGATRON_LM && MEGATRON_LM_DIR=\$MEGATRON_LM \
#       python <abs>/harness/check_port_e2e_packed.py"

import argparse
import importlib.util
import os
import socket
import sys

os.environ.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------
# Import plumbing (identical to check_port_e2e.py)
# ---------------------------------------------------------------------------------------
def _load_kda_ref():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "src", "kda_ref.py")
    spec = importlib.util.spec_from_file_location("kda_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_megatron():
    mega = os.environ.get("MEGATRON_LM_DIR", os.path.expanduser("~/proj/Megatron-LM"))
    if not os.path.isdir(mega):
        for cand in ("~/proj/Megatron-LM", "~/proj/Megatron-LM"):
            if os.path.isdir(cand):
                mega = cand
                break
    if mega not in sys.path:
        sys.path.insert(0, mega)
    try:
        import nvidia_resiliency_ext as _nvrx

        _nvrx.__version__ = "0.6.0"
    except Exception as exc:  # pragma: no cover
        print("  [warn] nvrx version patch skipped:", exc)
    global ps, ProcessGroupCollection, hybrid_stack_spec, KimiDeltaAttention
    global model_parallel_cuda_manual_seed, TransformerConfig, PackedSeqParams
    from megatron.core import parallel_state as ps
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.ssm.gated_delta_net import KimiDeltaAttention
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    return mega


def init_dist():
    if not torch.distributed.is_initialized():
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        torch.cuda.set_device(0)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(
            backend="nccl", world_size=1, rank=0, init_method="env://"
        )
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(1, 1)


# ---------------------------------------------------------------------------------------
# Module construction (mirrors M10-1 / tests/unit_tests/ssm/gated_delta_net/test_kda.py)
# ---------------------------------------------------------------------------------------
def make_config(dtype):
    bf16 = dtype == torch.bfloat16
    return TransformerConfig(
        hidden_size=128,
        num_layers=1,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        normalization="RMSNorm",
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=bf16,
        params_dtype=dtype,
        use_cpu_initialization=True,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        linear_cp_mode="headwise",
        experimental_attention_variant="kda",
        is_hybrid_model=True,
        transformer_impl="transformer_engine",
    )


def build_kda(config):
    spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
    return KimiDeltaAttention(
        config,
        submodules=spec.submodules,
        layer_number=1,
        conv_init=1.0,
        pg_collection=ProcessGroupCollection(
            tp=ps.get_tensor_model_parallel_group(),
            cp=ps.get_context_parallel_group(),
        ),
    ).cuda()


def make_packed_seq_params(cu_seqlens):
    cu = torch.IntTensor(cu_seqlens).cuda()
    seqlens = cu[1:] - cu[:-1]
    m = int(seqlens.max().item())
    return PackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=m,
        max_seqlen_kv=m,
        qkv_format="thd",
    )


def _ref_core(ref, q, k, v, g, beta, A_log, dt_bias, safe_gate, lower_bound,
              cu_seqlens, initial_state):
    """fp64 chunked reference forward, honouring cu_seqlens with a per-sequence loop.

    This is the same semantics M8-1 anchored for FLA varlen: each packed sequence starts
    from a zero state and is processed independently (the real layer always calls the core
    with ``initial_state=None, output_final_state=False``).
    """
    act_g = ref.kda_gate(g, A_log, dt_bias, lower_bound if safe_gate else None)
    if cu_seqlens is None:
        o, fs = ref.kda_chunked_fwd(
            q, k, v, act_g, beta, initial_state=initial_state,
            output_final_state=False, chunk_size=64,
        )
        return o, fs
    cu = cu_seqlens.tolist()
    outs = []
    for i in range(len(cu) - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        oi, _ = ref.kda_chunked_fwd(
            q[:, s:e], k[:, s:e], v[:, s:e], act_g[:, s:e], beta[:, s:e],
            initial_state=None, output_final_state=False, chunk_size=64,
        )
        outs.append(oi)
    return torch.cat(outs, dim=1), None


def make_ref_gdr(ref):
    def gdr(
        q, k, v, g, beta, A_log, dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        safe_gate=False,
        lower_bound=None,
        cu_seqlens=None,
        cp_context=None,
        **kwargs,
    ):
        assert cp_context is None, "reference path is single-rank (cp=1)"
        o, fs = _ref_core(
            ref, q, k, v, g, beta, A_log, dt_bias, safe_gate, lower_bound,
            cu_seqlens, initial_state,
        )
        o = o.to(q.dtype)
        fs = None if fs is None else fs.to(q.dtype)
        return o, fs

    return gdr


def relerr(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


# ---------------------------------------------------------------------------------------
# Module cases
# ---------------------------------------------------------------------------------------
def build_module_case(ref, dtype, seed=123):
    """Return (kda_fla, kda_ref, config, captured, fla_orig)."""
    model_parallel_cuda_manual_seed(seed)
    config = make_config(dtype)
    kda = build_kda(config)
    state = {k: v.detach().clone() for k, v in kda.state_dict().items()}

    captured = {}

    def recorder(**kw):
        captured.clear()
        captured.update(kw)
        return _fla_orig(**kw)

    _fla_orig = kda.gated_delta_rule
    kda.gated_delta_rule = recorder

    model_parallel_cuda_manual_seed(seed)
    kda_r = build_kda(config)
    kda_r.load_state_dict(state)
    kda_r.gated_delta_rule = make_ref_gdr(ref)
    return kda, kda_r, config, captured, _fla_orig


# ---------------------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------------------
def part_a(kda, config, packs, dtype):
    print("  -- Part A: packed THD fwd/bwd smoke (%s)" % str(dtype).split(".")[-1])
    ok = True
    for cu_list in packs:
        T = cu_list[-1]
        psp = make_packed_seq_params(cu_list)
        torch.manual_seed(T * 100 + len(cu_list))
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
        print("     cu=%s T=%4d  shape/dtype=%s  hs.grad finite=%s  nonfinite_params=%d"
              % (cu_list, T, ok_shape, finite_hs, len(bad)))
    return ok


def part_b(ref, fla_gdr, captured):
    print("  -- Part B: packed core-operator anchor on captured operands (FLA vs fp64 ref)")
    keys = ["q", "k", "v", "g", "beta", "A_log", "dt_bias"]
    cu = captured.get("cu_seqlens")
    assert cu is not None, "expected the packed layer to pass cu_seqlens to the core"
    args = {k: captured[k].detach().clone() for k in keys}
    safe = bool(captured.get("safe_gate", True))
    lower = captured.get("lower_bound", -5.0)
    print("     captured q shape=%s cu_seqlens=%s" % (tuple(args["q"].shape), cu.tolist()))
    base = dict(
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=bool(captured.get("use_qk_l2norm_in_kernel", False)),
        use_gate_in_kernel=bool(captured.get("use_gate_in_kernel", True)),
        safe_gate=safe,
        lower_bound=lower,
        cu_seqlens=cu,
        cp_context=None,
    )

    def run(gdr, x):
        return gdr(**{**x, **base})

    o_f, _ = run(fla_gdr, {k: v.clone().requires_grad_(True) for k, v in args.items()})
    o_r, _ = run(make_ref_gdr(ref), {k: v.clone().requires_grad_(True) for k, v in args.items()})
    print("     fwd o: shapes %s vs %s  max_abs=%.3e  rel=%.3e"
          % (tuple(o_f.shape), tuple(o_r.shape),
             (o_f.float() - o_r.float()).abs().max().item(), relerr(o_f, o_r)))

    xf = {k: v.clone().requires_grad_(True) for k, v in args.items()}
    of, _ = run(fla_gdr, xf)
    of.float().square().mean().backward()
    xr = {k: v.clone().requires_grad_(True) for k, v in args.items()}
    orr, _ = run(make_ref_gdr(ref), xr)
    orr.float().square().mean().backward()
    worst = 0.0
    for k in ["q", "k", "v", "g", "beta"]:
        if xf[k].grad is None or xr[k].grad is None:
            print("     grad %-5s : MISSING" % k)
            continue
        d = (xf[k].grad.float() - xr[k].grad.float()).abs().max().item()
        worst = max(worst, d)
        print("     grad %-5s max_abs=%.3e  rel=%.3e" % (k, d, relerr(xf[k].grad, xr[k].grad)))
    print("     core worst grad max_abs=%.3e" % worst)
    return o_f, o_r


def part_c(kda, kda_r, config, packs, dtype):
    print("  -- Part C: full-module packed (FLA-core vs ref-core, identical weights)")
    worst_out = worst_hs = worst_p = 0.0
    for cu_list in packs:
        T = cu_list[-1]
        psp = make_packed_seq_params(cu_list)
        torch.manual_seed(T * 100 + len(cu_list))
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
        print("     cu=%s T=%4d  out max_abs=%.3e  hs.grad max_abs=%.3e  param.grad max_abs=%.3e"
              % (cu_list, T, do, dh, dp))
    print("     worst: out=%.3e  hs.grad=%.3e  param.grad=%.3e" % (worst_out, worst_hs, worst_p))
    return worst_out, worst_hs, worst_p


def part_d(kda, config, dtype, T=64, B=2):
    """Packed == unpacked self-consistency on the *real* layer (FLA core both sides).

    Uses two modules with identical weights so both a forward and a backward pass can run
    without double-accumulating grads. The packed input is the SBHD input transposed to
    [B,T,H] then flattened to [T*B,1,H], with cu_seqlens = [0,T,2T,...] — i.e. each batch
    row becomes one sequence, exactly as Megatron's own packed test builds it.
    """
    print("  -- Part D: packed == unpacked self-consistency (T=%d B=%d)" % (T, B))
    model_parallel_cuda_manual_seed(999)
    kda_u = build_kda(config)
    state = {k: v.detach().clone() for k, v in kda_u.state_dict().items()}
    kda_p = build_kda(config)
    kda_p.load_state_dict(state)

    torch.manual_seed(T * 100 + B)
    x = torch.randn(T, B, config.hidden_size, device="cuda", dtype=dtype)

    xu = x.detach().clone().requires_grad_(True)
    out_u, _ = kda_u(xu, None)

    xp = x.transpose(0, 1).contiguous().reshape(T * B, 1, config.hidden_size)
    xp = xp.detach().clone().requires_grad_(True)
    cu = [i * T for i in range(B + 1)]
    psp = make_packed_seq_params(cu)
    out_p, _ = kda_p(xp, None, packed_seq_params=psp)

    ref_u = out_u.transpose(0, 1).reshape(T * B, config.hidden_size)
    d_out = (out_p[:, 0].float() - ref_u.float()).abs().max().item()

    out_u.float().square().mean().backward()
    out_p.float().square().mean().backward()
    # pack the upstream grad the same way and compare hs grads token-by-token
    gu = xu.grad.transpose(0, 1).reshape(T * B, config.hidden_size)
    gp = xp.grad[:, 0]
    d_hs = (gp.float() - gu.float()).abs().max().item()

    dp = 0.0
    pu = dict(kda_u.named_parameters())
    for name, p in kda_p.named_parameters():
        ga, gb = pu[name].grad, p.grad
        if ga is None or gb is None:
            continue
        dp = max(dp, (ga.float() - gb.float()).abs().max().item())
    print("     out max_abs=%.3e  hs.grad max_abs=%.3e  param.grad max_abs=%.3e"
          % (d_out, d_hs, dp))
    return d_out, d_hs, dp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="float32,bfloat16")
    # each pack is a cu_seqlens list; sequences include non-64-divisible lengths
    ap.add_argument("--packs", default="0,64,128;0,50,130;0,64,100,250")
    args = ap.parse_args()

    packs = [[int(x) for x in p.split(",")] for p in args.packs.split(";")]
    dtypes = [getattr(torch, d.strip()) for d in args.dtypes.split(",")]

    mega = _import_megatron()
    print("Megatron-LM:", mega)
    ref = _load_kda_ref()
    init_dist()

    for dtype in dtypes:
        print("=== dtype=%s packs=%s ===" % (str(dtype).split(".")[-1], packs))
        kda, kda_r, config, captured, fla_gdr = build_module_case(ref, dtype)
        part_a(kda, config, packs, dtype)
        # Part B needs captured operands from a real packed forward.
        psp0 = make_packed_seq_params(packs[0])
        torch.manual_seed(packs[0][-1] * 100 + len(packs[0]))
        hs = torch.randn(packs[0][-1], 1, config.hidden_size, device="cuda", dtype=dtype,
                         requires_grad=True)
        kda.zero_grad(set_to_none=True)
        kda(hs, None, packed_seq_params=psp0)  # populates `captured`
        part_b(ref, fla_gdr, captured)
        part_c(kda, kda_r, config, packs, dtype)
        part_d(kda, config, dtype)

    ps.destroy_model_parallel()
    print("ALL DONE")


if __name__ == "__main__":
    main()
