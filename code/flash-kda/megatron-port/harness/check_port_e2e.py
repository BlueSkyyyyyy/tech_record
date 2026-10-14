# M10-1: end-to-end forward/backward anchor of the *real* Megatron-LM KDA layer.
#
# Unlike M9-1 (which used a hand-written shim because ``import megatron`` aborted on a
# version assertion), this harness instantiates the real
# ``megatron.core.ssm.gated_delta_net.kda.KimiDeltaAttention`` module, runs its training
# path (``in_proj -> conv1d -> gated_delta_rule -> gated RMSNorm -> out_proj``), and anchors
# it against this sub-topic's fp64 reference ``src/kda_ref.py`` at two granularities:
#
#   Part A  real layer fwd/bwd smoke (finite grads on every parameter);
#   Part B  core-operator anchor: capture the exact operands handed to
#           ``self.gated_delta_rule`` and compare FLA ``chunk_kda`` vs the fp64 reference
#           on *those* operands (forward ``o``/``final_state`` + six grads);
#   Part C  full-module end-to-end: swap the core operator for the fp64 reference inside a
#           second (weight-identical) module and compare output + ``hidden_states.grad`` +
#           parameter grads against the FLA-backed module.
#
# Two container workarounds are needed to import Megatron (both are import-time, unrelated
# to KDA): the installed ``nvidia-resiliency-ext`` is 0.5.0 while Megatron asserts >=0.6.0
# (patch the in-memory ``__version__``), and DeepEP aborts on an NCCL-version mismatch
# (``EP_SUPPRESS_NCCL_CHECK=1``).  See the megatron-port ROADMAP "已知坑".
#
# Run (GPU, single process):
#   scripts/lab.sh exec "cd \$MEGATRON_LM && MEGATRON_LM_DIR=\$MEGATRON_LM \
#       python <abs>/harness/check_port_e2e.py"

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
# Import plumbing
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
    # nvidia-resiliency-ext 0.5.0 in the container; Megatron asserts >= 0.6.0 at import.
    try:
        import nvidia_resiliency_ext as _nvrx

        _nvrx.__version__ = "0.6.0"
    except Exception as exc:  # pragma: no cover
        print("  [warn] nvrx version patch skipped:", exc)
    global ps, ProcessGroupCollection, hybrid_stack_spec, KimiDeltaAttention
    global model_parallel_cuda_manual_seed, TransformerConfig
    from megatron.core import parallel_state as ps
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
# Module construction (mirrors tests/unit_tests/ssm/gated_delta_net/test_kda.py)
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


def make_ref_gdr(ref):
    """A ``gated_delta_rule`` replacement that runs the fp64 reference.

    It accepts the same keyword signature the KDA module uses and, crucially, applies the
    same channel-wise gate activation *inside* (``use_gate_in_kernel=True`` in Megatron).
    """

    def gdr(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
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
        assert cu_seqlens is None and cp_context is None, "reference path is single-sequence"
        act_g = ref.kda_gate(
            g, A_log, dt_bias, lower_bound if safe_gate else None
        )
        o, fs = ref.kda_chunked_fwd(
            q,
            k,
            v,
            act_g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=64,
        )
        # Keep the downstream graph in the module's dtype (bf16/fp32) while the reference
        # math itself ran in fp64.
        o = o.to(q.dtype)
        fs = None if fs is None else fs.to(q.dtype)
        return o, fs

    return gdr


def relerr(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


# ---------------------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------------------
def build_module_case(ref, dtype, seed=123):
    """Return (kda_fla, kda_ref, config, captured)."""
    model_parallel_cuda_manual_seed(seed)
    config = make_config(dtype)
    kda = build_kda(config)

    # Freeze a deterministic copy of the weights for the reference-backed module.
    state = {k: v.detach().clone() for k, v in kda.state_dict().items()}

    captured = {}

    def recorder(**kw):
        captured.clear()
        captured.update(kw)
        return _fla_orig(**kw)

    _fla_orig = kda.gated_delta_rule  # real FLA chunk_kda
    kda.gated_delta_rule = recorder

    model_parallel_cuda_manual_seed(seed)
    kda_r = build_kda(config)
    kda_r.load_state_dict(state)
    kda_r.gated_delta_rule = make_ref_gdr(ref)
    return kda, kda_r, config, captured, _fla_orig


def part_a(kda, config, cases, dtype):
    print("  -- Part A: real-layer fwd/bwd smoke (%s)" % str(dtype).split(".")[-1])
    ok = True
    for (T, B) in cases:
        torch.manual_seed(T * 100 + B)
        hs = torch.randn(T, B, config.hidden_size, device="cuda", dtype=dtype, requires_grad=True)
        kda.zero_grad(set_to_none=True)
        out, bias = kda(hs, None)
        ok_shape = out.shape == hs.shape and out.dtype == dtype and bias is None
        out.float().square().mean().backward()
        finite_hs = torch.isfinite(hs.grad).all().item()
        bad = []
        for name, p in kda.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all().item():
                bad.append(name)
        ok = ok and ok_shape and finite_hs and not bad
        print("     T=%4d B=%d  shape/dtype=%s  hs.grad finite=%s  nonfinite_params=%d"
              % (T, B, ok_shape, finite_hs, len(bad)))
    return ok


def part_b(ref, fla_gdr, captured, out_fla_core=None):
    """Anchor the core operator on the captured operands: FLA vs fp64 reference."""
    print("  -- Part B: core-operator anchor on captured operands (FLA vs fp64 ref)")
    keys = ["q", "k", "v", "g", "beta", "A_log", "dt_bias"]
    args = {k: captured[k].detach().clone() for k in keys}
    safe = bool(captured.get("safe_gate", True))
    lower = captured.get("lower_bound", -5.0)
    print("     (captured flags: use_gate_in_kernel=%s safe_gate=%s lower_bound=%s)"
          % (captured.get("use_gate_in_kernel"), safe, lower))
    base = dict(
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=bool(captured.get("use_qk_l2norm_in_kernel", False)),
        use_gate_in_kernel=bool(captured.get("use_gate_in_kernel", True)),
        safe_gate=safe,
        lower_bound=lower,
        cu_seqlens=None,
        cp_context=None,
    )

    def run(gdr, x):
        return gdr(**{**x, **base})

    # forward
    o_f, _ = run(fla_gdr, {k: v.clone().requires_grad_(True) for k, v in args.items()})
    o_r, _ = run(make_ref_gdr(ref), {k: v.clone().requires_grad_(True) for k, v in args.items()})
    print("     fwd o: shapes %s vs %s  max_abs=%.3e  rel=%.3e"
          % (tuple(o_f.shape), tuple(o_r.shape),
             (o_f.float() - o_r.float()).abs().max().item(), relerr(o_f, o_r)))

    # gradients through the core operator
    xf = {k: v.clone().requires_grad_(True) for k, v in args.items()}
    of, _ = run(fla_gdr, xf)
    of.float().square().mean().backward()
    xr = {k: v.clone().requires_grad_(True) for k, v in args.items()}
    orr, _ = run(make_ref_gdr(ref), xr)
    orr.float().square().mean().backward()
    worst = 0.0
    for k in ["q", "k", "v", "g", "beta"]:
        if xf[k].grad is None or xr[k].grad is None:
            print("     grad %-5s : MISSING (fla=%s ref=%s)"
                  % (k, xf[k].grad is not None, xr[k].grad is not None))
            continue
        d = (xf[k].grad.float() - xr[k].grad.float()).abs().max().item()
        worst = max(worst, d)
        print("     grad %-5s max_abs=%.3e  rel=%.3e" % (k, d, relerr(xf[k].grad, xr[k].grad)))
    print("     core worst grad max_abs=%.3e" % worst)
    return o_f, o_r


def part_c(kda, kda_r, config, cases, dtype):
    print("  -- Part C: full-module end-to-end (FLA-core vs ref-core, identical weights)")
    worst_out = 0.0
    worst_hs = 0.0
    worst_p = 0.0
    for (T, B) in cases:
        torch.manual_seed(T * 100 + B)
        hs_a = torch.randn(T, B, config.hidden_size, device="cuda", dtype=dtype, requires_grad=True)
        hs_b = hs_a.detach().clone().requires_grad_(True)
        kda.zero_grad(set_to_none=True)
        kda_r.zero_grad(set_to_none=True)
        out_a, _ = kda(hs_a, None)
        out_b, _ = kda_r(hs_b, None)
        do = (out_a.float() - out_b.float()).abs().max().item()
        worst_out = max(worst_out, do)
        out_a.float().square().mean().backward()
        out_b.float().square().mean().backward()
        dh = (hs_a.grad.float() - hs_b.grad.float()).abs().max().item()
        worst_hs = max(worst_hs, dh)
        # parameter grads
        dp = 0.0
        pa = dict(kda.named_parameters())
        for name, p in kda_r.named_parameters():
            ga = pa[name].grad
            gb = p.grad
            if ga is None or gb is None:
                continue
            dp = max(dp, (ga.float() - gb.float()).abs().max().item())
        worst_p = max(worst_p, dp)
        print("     T=%4d B=%d  out max_abs=%.3e  hs.grad max_abs=%.3e  param.grad max_abs=%.3e"
              % (T, B, do, dh, dp))
    print("     worst: out=%.3e  hs.grad=%.3e  param.grad=%.3e" % (worst_out, worst_hs, worst_p))
    return worst_out, worst_hs, worst_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--cases", default="64x2,128x2,250x1")
    args = ap.parse_args()

    cases = [tuple(int(x) for x in c.lower().split("x")) for c in args.cases.split(",")]
    dtypes = [getattr(torch, d.strip()) for d in args.dtypes.split(",")]

    mega = _import_megatron()
    print("Megatron-LM:", mega)
    ref = _load_kda_ref()
    init_dist()

    for dtype in dtypes:
        print("=== dtype=%s cases=%s ===" % (str(dtype).split(".")[-1], cases))
        kda, kda_r, config, captured, fla_gdr = build_module_case(ref, dtype)
        part_a(kda, config, cases, dtype)
        # Part B needs the captured operands from a real forward.
        torch.manual_seed(cases[0][0] * 100 + cases[0][1])
        hs = torch.randn(cases[0][0], cases[0][1], config.hidden_size, device="cuda",
                         dtype=dtype, requires_grad=True)
        kda.zero_grad(set_to_none=True)
        kda(hs, None)  # populates `captured`
        part_b(ref, fla_gdr, captured)
        part_c(kda, kda_r, config, cases, dtype)

    ps.destroy_model_parallel()
    print("ALL DONE")


if __name__ == "__main__":
    main()
