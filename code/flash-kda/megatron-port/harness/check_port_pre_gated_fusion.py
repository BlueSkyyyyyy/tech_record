# M12-1: the ``gdn_pre_gated_delta_rule_fusion`` gap for KDA.
#
# The Megatron GatedDeltaNet (GDN) module has an optional *streamed fused pre-GDR*
# preprocessing (``fused_streamed_pre_gated_delta_rule``, a Triton kernel family in
# ``megatron/core/fusions/fused_pre_gated_delta_rule.py``) that replaces the eager
# ``pre_gated_delta_rule`` python path (conv1d + split + l2norm + repeat + gate/beta
# activation + layout).  KDA deliberately does **not** wire that kernel: the config
# validation and the layer forward both raise ``NotImplementedError``.
#
# This harness does three things, none of which modify the Megatron-LM tree:
#
#   Part A  reproduce the two guards (config-time ``__post_init__`` and the runtime
#           forward guard) verbatim;
#   Part B  implement a pure-PyTorch *KDA-specific* fused pre-GDR reference and anchor
#           it against the operands the real ``KimiDeltaAttention`` layer hands to the
#           core (captured by wrapping ``pre_gated_delta_rule``) -- per-tensor max_abs/rel;
#   Part C  prove the reference is a *drop-in* replacement: swap ``pre_gated_delta_rule``
#           for the fused reference in a weight-identical module and compare full-layer
#           forward output + ``hidden_states.grad`` + parameter grads against the stock
#           layer (this is what a future KDA Triton fusion must reproduce);
#   Part D  quantify the "streamed" claim: which eager intermediates are already views
#           (free) vs materialized contiguous copies, and the byte cost of the copies a
#           fused kernel would fold away.
#
# Two container import-time workarounds are needed (same as M10-1..M11-2, unrelated to KDA):
# patch ``nvidia_resiliency_ext.__version__`` and set ``EP_SUPPRESS_NCCL_CHECK=1``.
#
# Run (GPU, single process):
#   scripts/lab.sh exec "cd \$MEGATRON_LM && MEGATRON_LM_DIR=\$MEGATRON_LM \
#       python <abs>/harness/check_port_pre_gated_fusion.py"

import argparse
import importlib.util
import os
import socket
import sys
import types

os.environ.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------
# Import plumbing (identical to check_port_e2e*.py)
# ---------------------------------------------------------------------------------------
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
    global model_parallel_cuda_manual_seed, TransformerConfig
    global fla_causal_conv1d, fla_l2norm
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.ssm.gated_delta_net import KimiDeltaAttention
    from megatron.core.ssm.gated_delta_net.common import causal_conv1d as fla_causal_conv1d
    from megatron.core.ssm.gated_delta_net.common import l2norm as fla_l2norm
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


def make_config(dtype, fusion=False):
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
        gdn_pre_gated_delta_rule_fusion=fusion,
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


# ---------------------------------------------------------------------------------------
# Part A: reproduce the two guards
# ---------------------------------------------------------------------------------------
def part_a(dtype):
    print("  -- Part A: reproduce the two KDA guard rails (%s)" % str(dtype).split(".")[-1])
    # (1) config-time guard (TransformerConfig.__post_init__)
    msg1 = None
    try:
        make_config(dtype, fusion=True)
    except NotImplementedError as exc:
        msg1 = str(exc)
    print("     config __post_init__ guard : %s" % ("NotImplementedError: " + msg1 if msg1
                                                    else "NOT RAISED (unexpected)"))
    # (2) runtime forward guard (kda.py)
    model_parallel_cuda_manual_seed(123)
    kda = build_kda(make_config(dtype, fusion=False))
    kda.gdn_pre_gated_delta_rule_fusion = True
    msg2 = None
    try:
        hs = torch.randn(32, 2, 128, device="cuda", dtype=dtype)
        kda(hs, None)
    except NotImplementedError as exc:
        msg2 = str(exc)
    print("     forward guard (kda.py)      : %s" % ("NotImplementedError: " + msg2 if msg2
                                                    else "NOT RAISED (unexpected)"))
    return msg1 is not None and msg2 is not None


# ---------------------------------------------------------------------------------------
# KDA-specific pure-PyTorch fused pre-GDR reference (single function, "streamed").
#
# It consumes exactly what ``KimiDeltaAttention.pre_gated_delta_rule`` consumes
# (``qkvfg`` = in_proj output, ``beta`` = beta_proj output, both ``[T, B, *]``) and
# returns the same 8-tuple the eager method returns.  It mirrors the *deterministic*
# eager path in ``kda.py`` (``F.conv1d`` + slice; no causal_conv1d needed) and is the
# algebraic contract a future KDA Triton fusion must reproduce.
# ---------------------------------------------------------------------------------------
def _fla_l2norm_torch(x, eps=1e-6):
    # Mirrors FLA's Triton ``l2norm_fwd_kernel1``: accumulate x^2 in fp32, take
    # rstd = 1/sqrt(sum+eps) in fp32, multiply in fp32, store back at x.dtype.
    xf = x.float()
    rstd = 1.0 / torch.sqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)
    return (xf * rstd).to(x.dtype)


def kda_pre_gdr_fused_ref(module, qkvfg, beta, batch, seq_len, conv_impl="fla",
                          l2norm_impl="fla"):
    """KDA-specific fused pre-GDR reference (cp=1, SBHD).

    ``conv_impl`` selects the depthwise causal-conv backend: ``"fla"`` calls the same
    FLA ``causal_conv1d`` the eager layer uses (bit-exact contract), ``"torch"`` uses
    ``F.conv1d`` (what a self-contained fused kernel would implement).  ``l2norm_impl``
    likewise selects the FLA Triton ``l2norm`` or the torch fp32-internal cell.  Using
    both FLA primitives reproduces the eager path bit-exactly; the residual with torch
    primitives isolates primitive-level rounding.
    """
    qk = module.qk_dim_local_tp
    v = module.v_dim_local_tp
    Hk = module.num_key_heads
    Hv = module.num_value_heads
    KD = module.key_head_dim
    VD = module.value_head_dim

    qkvfg = qkvfg.transpose(0, 1)          # [B, T, Cin]
    beta = beta.transpose(0, 1)            # [B, T, Hk]
    qkv, raw_g, gate = torch.split(qkvfg, (2 * qk + v, qk, v), dim=-1)
    gate = gate.reshape(batch, seq_len, -1, VD)

    # depthwise causal conv1d over the qkv channels, then SiLU (mirrors eager path).
    kernel_seq_len = qkv.shape[1]
    conv_w = module.conv1d.weight
    conv_b = module.conv1d.bias if module.conv_bias else None
    if conv_impl == "fla":
        x = qkv.contiguous()                            # [B, T, C] channel-last
        qkv, _ = fla_causal_conv1d(
            x=x,
            weight=conv_w.squeeze(1),
            bias=conv_b,
            activation=module.activation,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
            cp_context=None,
        )
    else:
        x = qkv.transpose(1, 2).contiguous()            # [B, C, T]
        conv_out = F.conv1d(
            input=x, weight=conv_w, bias=conv_b,
            stride=module.conv1d.stride, padding=module.conv1d.padding,
            dilation=module.conv1d.dilation, groups=module.conv_dim_local_tp,
        )
        qkv = module.act_fn(conv_out[..., :kernel_seq_len]).transpose(1, 2).contiguous()

    # q/k l2norm + split (+ optional GVA repeat)
    query_key, value = torch.split(qkv, [2 * qk, v], dim=-1)
    query_key = query_key.reshape(batch, seq_len, -1, KD)
    value = value.reshape(batch, seq_len, -1, VD)
    if module.use_qk_l2norm:
        query_key = query_key.contiguous()
        if l2norm_impl == "fla" and fla_l2norm is not None:
            query_key = fla_l2norm(query_key)
        else:
            query_key = _fla_l2norm_torch(query_key)
    query, key = torch.split(query_key, [Hk, Hk], dim=2)
    repeat_factor = Hv // Hk
    if repeat_factor > 1:
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)

    # raw decay gate (activated in-kernel by chunk_kda) + beta sigmoid
    raw_g = raw_g.reshape(batch, seq_len, Hk, KD)
    beta = beta.reshape(batch, seq_len, Hk).float().sigmoid().contiguous()

    return (
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate.contiguous(),
        beta,
        raw_g.contiguous(),
        module.A_log,
        module.dt_bias,
    )


def relerr(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


# ---------------------------------------------------------------------------------------
# Part B: capture the eager operands, anchor the fused reference per tensor
# ---------------------------------------------------------------------------------------
_GOLDEN_KEYS = ["q", "k", "v", "gate", "beta", "g"]


def part_b(dtype, T=64, B=2):
    print("  -- Part B: fused pre-GDR reference vs eager pre_gated_delta_rule (%s, T=%d B=%d)"
          % (str(dtype).split(".")[-1], T, B))
    model_parallel_cuda_manual_seed(123)
    kda = build_kda(make_config(dtype, fusion=False))
    cap = {}

    orig = kda.pre_gated_delta_rule

    def recorder(qkvfg, beta, batch, seq_len, *args, **kwargs):
        out = orig(qkvfg, beta, batch, seq_len, *args, **kwargs)
        cap.update(dict(qkvfg=qkvfg.detach(), beta_in=beta.detach(),
                        out=[t.detach() if torch.is_tensor(t) else t for t in out],
                        batch=batch, seq_len=seq_len))
        return out

    kda.pre_gated_delta_rule = recorder
    torch.manual_seed(T * 7 + B)
    hs = torch.randn(T, B, 128, device="cuda", dtype=dtype)
    kda(hs, None)

    worst = 0.0
    for conv_impl, l2_impl in (("fla", "fla"), ("torch", "fla"),
                               ("fla", "torch"), ("torch", "torch")):
        ref = kda_pre_gdr_fused_ref(kda, cap["qkvfg"], cap["beta_in"], cap["batch"],
                                    cap["seq_len"], conv_impl=conv_impl,
                                    l2norm_impl=l2_impl)
        print("     [conv=%s, l2norm=%s]" % (conv_impl, l2_impl))
        for i, name in enumerate(_GOLDEN_KEYS):
            g = cap["out"][i].float()
            r = ref[i].float()
            d = (g - r).abs().max().item()
            if conv_impl == "fla" and l2_impl == "fla":
                worst = max(worst, d)
            print("        %-5s shape=%-22s max_abs=%.3e  rel=%.3e"
                  % (name, tuple(g.shape), d, relerr(g, r)))
    ref = kda_pre_gdr_fused_ref(kda, cap["qkvfg"], cap["beta_in"],
                                cap["batch"], cap["seq_len"], conv_impl="fla",
                                l2norm_impl="fla")
    ref2 = kda_pre_gdr_fused_ref(kda, cap["qkvfg"], cap["beta_in"],
                                 cap["batch"], cap["seq_len"], conv_impl="fla",
                                 l2norm_impl="fla")
    self_worst = max((ref[i].float() - ref2[i].float()).abs().max().item()
                     for i in range(5))
    print("     [control] fused-ref run-to-run self-consistency max_abs=%.3e"
          % self_worst)
    # A_log / dt_bias are pass-through leaves
    print("     A_log/dt_bias pass-through identical: %s"
          % bool(torch.equal(cap["out"][6], ref[6]) and torch.equal(cap["out"][7], ref[7])))
    return cap, ref, worst


# ---------------------------------------------------------------------------------------
# Part C: drop-in replacement of pre_gated_delta_rule -> fused reference
# ---------------------------------------------------------------------------------------
def part_c(dtype, T=64, B=2):
    print("  -- Part C: full-layer drop-in (eager pre-GDR vs fused reference, same weights)")
    model_parallel_cuda_manual_seed(123)
    kda = build_kda(make_config(dtype, fusion=False))
    state = {k: v.detach().clone() for k, v in kda.state_dict().items()}

    model_parallel_cuda_manual_seed(123)
    kda_f = build_kda(make_config(dtype, fusion=False))
    kda_f.load_state_dict(state)

    def fused_pre_gated_delta_rule(self, qkvfg, beta, batch, seq_len, *args, **kwargs):
        return kda_pre_gdr_fused_ref(self, qkvfg, beta, batch, seq_len,
                                     conv_impl="fla", l2norm_impl="fla")

    kda_f.pre_gated_delta_rule = types.MethodType(fused_pre_gated_delta_rule, kda_f)

    torch.manual_seed(T * 13 + B)
    xa = torch.randn(T, B, 128, device="cuda", dtype=dtype, requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    kda.zero_grad(set_to_none=True)
    kda_f.zero_grad(set_to_none=True)
    oa, _ = kda(xa, None)
    ob, _ = kda_f(xb, None)
    d_out = (oa.float() - ob.float()).abs().max().item()
    # count-invariant loss (same token count on both sides)
    oa.float().sum().backward()
    ob.float().sum().backward()
    d_hs = (xa.grad.float() - xb.grad.float()).abs().max().item()
    dp = 0.0
    scale = 0.0
    pa = dict(kda.named_parameters())
    for name, p in kda_f.named_parameters():
        ga, gb = pa[name].grad, p.grad
        if ga is None or gb is None:
            continue
        dp = max(dp, (ga.float() - gb.float()).abs().max().item())
        scale = max(scale, ga.float().abs().max().item())
    print("     out max_abs=%.3e   hs.grad max_abs=%.3e   param.grad max_abs=%.3e "
          "(max|param.grad|=%.3e, rel=%.3e)"
          % (d_out, d_hs, dp, scale, dp / (scale + 1e-30)))
    return d_out, d_hs, dp


# ---------------------------------------------------------------------------------------
# Part D: which eager intermediates are views vs materialized contiguous copies
# ---------------------------------------------------------------------------------------
def part_d(dtype, T=64, B=2):
    print("  -- Part D: streamed/layout accounting (%s)" % str(dtype).split(".")[-1])
    itemsize = torch.tensor([], dtype=dtype).element_size()
    model_parallel_cuda_manual_seed(123)
    kda = build_kda(make_config(dtype, fusion=False))
    cap = {}
    orig = kda.pre_gated_delta_rule

    def recorder(qkvfg, beta, batch, seq_len, *args, **kwargs):
        out = orig(qkvfg, beta, batch, seq_len, *args, **kwargs)
        cap.update(dict(qkvfg=qkvfg.detach(), beta_in=beta.detach(),
                        out=[t for t in out], batch=batch, seq_len=seq_len))
        return out

    kda.pre_gated_delta_rule = recorder
    torch.manual_seed(T * 7 + B)
    hs = torch.randn(T, B, 128, device="cuda", dtype=dtype)
    kda(hs, None)

    qkvfg_ptr = cap["qkvfg"].untyped_storage().data_ptr()
    total = 0
    for i, name in enumerate(_GOLDEN_KEYS):
        t = cap["out"][i]
        n = t.numel() * itemsize
        # A view shares the input projection's storage (no fresh allocation); a fresh
        # tensor means ``.contiguous()`` / the l2norm kernel materialized a copy.
        is_view = t.untyped_storage().data_ptr() == qkvfg_ptr
        if not is_view:
            total += n
        print("     %-5s view_of_qkvfg=%-5s contiguous=%-5s  bytes=%8d"
              % (name, str(is_view), str(t.is_contiguous()), n))
    print("     freshly-allocated core-operand bytes: %d (%.1f KiB)"
          % (total, total / 1024.0))
    print("     NOTE: for this SBHD cp=1 layout the eager outputs (except q/k, which the"
          " l2norm kernel writes fresh)")
    print("           are already contiguous, so a fused pre-GDR saves kernel launches +"
          " activation traffic, not a copy of the final operands.")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--B", type=int, default=2)
    args = ap.parse_args()
    dtypes = [getattr(torch, d.strip()) for d in args.dtypes.split(",")]

    mega = _import_megatron()
    print("Megatron-LM:", mega)
    init_dist()

    for dtype in dtypes:
        print("=== dtype=%s ===" % str(dtype).split(".")[-1])
        part_a(dtype)
        part_b(dtype, args.T, args.B)
        part_c(dtype, args.T, args.B)
        part_d(dtype, args.T, args.B)

    ps.destroy_model_parallel()
    print("ALL DONE")


if __name__ == "__main__":
    main()
