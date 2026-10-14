# M13-1: the KDA-specific *fused pre-GDR* Triton kernel family.
#
# M12-1 (``check_port_pre_gated_fusion.py`` + ``docs/14``) nailed the contract for a
# KDA-specific fused pre-GDR: given ``qkvfg`` (in_proj) and ``beta`` (beta_proj) it must
# reproduce the 8-tuple the eager ``pre_gated_delta_rule`` hands to ``chunk_kda`` --
# conv+SiLU over the qkv channels, q/k L2-norm, ``raw_g`` reshape, ``beta`` sigmoid, and
# **no** decay activation.  ``src/kda_pregdr_fused.py`` implements that contract as a
# three-kernel Triton family.  This harness anchors it and times it, without touching the
# Megatron-LM tree:
#
#   Part A  per-tensor correctness of the Triton family vs the real layer's eager
#           ``pre_gated_delta_rule`` (captured by wrapping the method) across shapes/dtypes;
#   Part B  isolate residual vs the pure-torch-primitive reference (conv/l2 = torch) to
#           separate ``conv-backend`` rounding from algebraic error;
#   Part C  drop-in: swap ``pre_gated_delta_rule`` for the Triton family in a weight-identical
#           module, compare full-layer forward + ``hidden_states.grad`` + parameter grads;
#   Part D  timing: isolated eager pre-GDR (FLA ``causal_conv1d`` + ``l2norm``) vs the Triton
#           family, CUDA-event event-pair, warmup + N iterations.
#
# Container import workarounds are the same as M10-M12 (nvrx version patch +
# ``EP_SUPPRESS_NCCL_CHECK=1``).
#
# Run (GPU, single process; pick an idle card):
#   scripts/lab.sh exec "cd \$MEGATRON_LM && CUDA_VISIBLE_DEVICES=1 MEGATRON_LM_DIR=\$MEGATRON_LM \
#       python <abs>/harness/check_port_pregdr_kernel.py"

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


def _import_megatron():
    mega = os.environ.get("MEGATRON_LM_DIR", os.path.expanduser("~/proj/Megatron-LM"))
    if not os.path.isdir(mega):
        for cand in (os.path.expanduser("~/proj/Megatron-LM"), os.path.expanduser("~/proj/Megatron-LM")):
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
    from megatron.core import parallel_state as ps
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.ssm.gated_delta_net import KimiDeltaAttention
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    return mega


def _import_local(modname, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "src", filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        gdn_pre_gated_delta_rule_fusion=False,
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


def relerr(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


_KEYS = ["q", "k", "v", "gate", "beta", "g"]


def _capture(kda, dtype, T, B, seed):
    cap = {}
    orig = kda.pre_gated_delta_rule

    def recorder(qkvfg, beta, batch, seq_len, *args, **kwargs):
        out = orig(qkvfg, beta, batch, seq_len, *args, **kwargs)
        cap.update(dict(qkvfg=qkvfg.detach(), beta_in=beta.detach(),
                        out=[t.detach() if torch.is_tensor(t) else t for t in out],
                        batch=batch, seq_len=seq_len))
        return out

    kda.pre_gated_delta_rule = recorder
    torch.manual_seed(seed)
    hs = torch.randn(T, B, 128, device="cuda", dtype=dtype)
    kda(hs, None)
    kda.pre_gated_delta_rule = orig
    return cap


# ---------------------------------------------------------------------------------------
# Part A: Triton family vs eager pre_gated_delta_rule
# ---------------------------------------------------------------------------------------
def part_a(dtype, cases, fused):
    print("  -- Part A: Triton pre-GDR family vs eager pre_gated_delta_rule (%s)"
          % str(dtype).split(".")[-1])
    model_parallel_cuda_manual_seed(123)
    kda = build_kda(make_config(dtype))
    for (T, B) in cases:
        cap = _capture(kda, dtype, T, B, seed=T * 7 + B)
        out = fused.kda_pre_gdr_fused_triton(kda, cap["qkvfg"], cap["beta_in"],
                                             cap["batch"], cap["seq_len"])
        worst = 0.0
        line = []
        for i, name in enumerate(_KEYS):
            d = (cap["out"][i].float() - out[i].float()).abs().max().item()
            worst = max(worst, d)
            line.append("%s=%.3e" % (name, d))
        print("     T=%-4d B=%d  %s  | worst=%.3e" % (T, B, "  ".join(line), worst))
    return kda


# ---------------------------------------------------------------------------------------
# Part B: isolate residual vs the torch-primitive reference (conv/l2 = torch)
# ---------------------------------------------------------------------------------------
def part_b(dtype, fused, refmod, cases):
    print("  -- Part B: vs pure-torch-primitive reference (conv=torch, l2norm=torch) (%s)"
          % str(dtype).split(".")[-1])
    model_parallel_cuda_manual_seed(123)
    kda = build_kda(make_config(dtype))
    for (T, B) in cases:
        cap = _capture(kda, dtype, T, B, seed=T * 11 + B)
        out = fused.kda_pre_gdr_fused_triton(kda, cap["qkvfg"], cap["beta_in"],
                                             cap["batch"], cap["seq_len"])
        ref = refmod.kda_pre_gdr_fused_ref(kda, cap["qkvfg"], cap["beta_in"], cap["batch"],
                                           cap["seq_len"], conv_impl="torch",
                                           l2norm_impl="torch")
        worst = 0.0
        line = []
        for i, name in enumerate(_KEYS):
            d = (out[i].float() - ref[i].float()).abs().max().item()
            worst = max(worst, d)
            line.append("%s=%.3e" % (name, d))
        print("     T=%-4d B=%d  %s  | worst=%.3e" % (T, B, "  ".join(line), worst))


# ---------------------------------------------------------------------------------------
# Part C: drop-in replacement of pre_gated_delta_rule
# ---------------------------------------------------------------------------------------
def part_c(dtype, fused, T=64, B=2):
    print("  -- Part C: full-layer drop-in (eager pre-GDR vs Triton family, same weights)")
    model_parallel_cuda_manual_seed(123)
    kda = build_kda(make_config(dtype))
    state = {k: v.detach().clone() for k, v in kda.state_dict().items()}
    model_parallel_cuda_manual_seed(123)
    kda_f = build_kda(make_config(dtype))
    kda_f.load_state_dict(state)

    def patched(self, qkvfg, beta, batch, seq_len, *args, **kwargs):
        # Triton forward + reference (torch-primitive) backward: the family is forward-only.
        return fused.kda_pre_gdr_fused_autograd(self, qkvfg, beta, batch, seq_len)

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


# ---------------------------------------------------------------------------------------
# Part D: timing (isolated eager pre-GDR vs Triton family)
# ---------------------------------------------------------------------------------------
def _time(fn, warmup=25, iters=100):
    """event-pair wall (includes host if the host is the bottleneck)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def _dev_time(fn, warmup=25, iters=50):
    """CUPTI self device time summed over kernels / iters (us)."""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = sum(e.self_device_time_total for e in prof.key_averages())
    return tot / iters


def part_d(dtype, fused, cases):
    print("  -- Part D: timing, isolated pre-GDR (%s)" % str(dtype).split(".")[-1])
    model_parallel_cuda_manual_seed(123)
    kda = build_kda(make_config(dtype))
    cp_group = ps.get_context_parallel_group()

    def eager(qkvfg, beta, batch, seq_len):
        return kda.pre_gated_delta_rule(qkvfg, beta, batch, seq_len, 1, cp_group)

    def ours(qkvfg, beta, batch, seq_len):
        return fused.kda_pre_gdr_fused_triton(kda, qkvfg, beta, batch, seq_len)

    for (T, B) in cases:
        cap = _capture(kda, dtype, T, B, seed=T * 5 + B)
        q, be, ba, sl = cap["qkvfg"], cap["beta_in"], cap["batch"], cap["seq_len"]
        te_wall = _time(lambda: eager(q, be, ba, sl))
        tf_wall = _time(lambda: ours(q, be, ba, sl))
        te_dev = _dev_time(lambda: eager(q, be, ba, sl))
        tf_dev = _dev_time(lambda: ours(q, be, ba, sl))
        print("     T=%-5d B=%d  wall: eager=%8.2f triton=%8.2f (%.3fx) | "
              "dev: eager=%8.2f triton=%8.2f (%.3fx)"
              % (T, B, te_wall, tf_wall, te_wall / tf_wall, te_dev, tf_dev, te_dev / tf_dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--B", type=int, default=2)
    args = ap.parse_args()
    dtypes = [getattr(torch, d.strip()) for d in args.dtypes.split(",")]

    mega = _import_megatron()
    print("Megatron-LM:", mega)
    fused = _import_local("kda_pregdr_fused", "kda_pregdr_fused.py")
    refmod = _import_ref()
    init_dist()

    cases_a = [(64, 2), (128, 2), (250, 1), (256, 1), (1024, 2)]
    cases_t = [(64, 2), (256, 2), (1024, 2), (1024, 8)]

    for dtype in dtypes:
        print("=== dtype=%s ===" % str(dtype).split(".")[-1])
        part_a(dtype, cases_a, fused)
        part_b(dtype, fused, refmod, [(64, 2), (250, 1)])
        part_c(dtype, fused)
        part_d(dtype, fused, cases_t)

    ps.destroy_model_parallel()
    print("ALL DONE")


def _import_ref():
    # reuse the M12-1 harness's pure-torch reference verbatim
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "check_port_pre_gated_fusion.py")
    spec = importlib.util.spec_from_file_location("m12_ref", path)
    mod = importlib.util.module_from_spec(spec)
    # stub heavy imports are not triggered (module import only defines funcs)
    spec.loader.exec_module(mod)
    # torch-primitive path does not touch the FLA globals; define them defensively.
    mod.fla_causal_conv1d = None
    mod.fla_l2norm = None
    return mod


if __name__ == "__main__":
    main()
