# M14-1: wire the KDA-specific *native* fused pre-GDR into the ``gdn_pre_gated_delta_rule_fusion``
# path -- the last roadmapped megatron-port item ("接到 kda.py:399 的 guard") -- as a **runtime
# shim**, so the Megatron-LM tree stays untouched (as every prior round).
#
# M12-1 (``docs/14``) proved the GDN fusion kernel cannot be reused for KDA and gave the KDA
# contract; M13-1/2/3 built the KDA-specific Triton forward + native backward
# (``src/kda_pregdr_fused.py`` / ``src/kda_pregdr_bwd.py``) and anchored them *as a drop-in for
# ``pre_gated_delta_rule``*.  What was never done: make the layer's own
# ``gdn_pre_gated_delta_rule_fusion`` flag dispatch to them (both guards say ``raise`` today).
#
# ``src/kda_pregdr_fusion_shim.py`` lifts the two guard sites at runtime
# (``transformer_config.py:1936-1938`` and ``kda.py:399-402``) and routes the layer's
# ``pre_gated_delta_rule`` call to ``kda_pre_gdr_fused_native`` (forward **and** native backward).
#
#   Part A  reproduce both upstream guards with the shim OFF (config + forward);
#   Part B  install the shim, show the guards are lifted (flag survives construction + forward);
#   Part C  numeric: flag-driven fused path vs eager path, same weights -- forward, input grad,
#           and **all** parameter grads (conv1d.weight included);
#   Part D  timing: full-layer (fwd+bwd) eager vs fused, event-pair ``wall`` + CUPTI ``dev``.
#
# Container import workarounds are the same as M10-M13 (nvrx version patch +
# ``EP_SUPPRESS_NCCL_CHECK=1``).
#
# Run (GPU, single process; pick an idle card):
#   scripts/lab.sh exec "cd \$MEGATRON_LM && CUDA_VISIBLE_DEVICES=1 MEGATRON_LM_DIR=\$MEGATRON_LM \
#       python <abs>/harness/check_port_pregdr_fusion_wire.py"

import importlib.util
import os

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
    global _M13
    if _M13 is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "check_port_pregdr_kernel.py")
        spec = importlib.util.spec_from_file_location("m13", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _M13 = mod
    return _M13


def _cfg(m13, dtype, fusion):
    bf16 = dtype == torch.bfloat16
    return m13.TransformerConfig(
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


# ---------------------------------------------------------------------------------------
# Part A: reproduce the two upstream guards (shim OFF)
# ---------------------------------------------------------------------------------------
def part_a(m13, dtype):
    print("  -- Part A: upstream guards reproduce (shim OFF, %s)" % str(dtype).split(".")[-1])
    try:
        _cfg(m13, dtype, True)
        print("     config guard: NOT raised (unexpected)")
    except NotImplementedError as e:
        print("     config guard: raised NotImplementedError(%r)" % str(e)[:44])

    m13.model_parallel_cuda_manual_seed(123)
    kda = m13.build_kda(_cfg(m13, dtype, False))
    kda.gdn_pre_gated_delta_rule_fusion = True           # force the module attribute
    torch.manual_seed(0)
    hs = torch.randn(64, 2, 128, device="cuda", dtype=dtype)
    try:
        kda(hs, None)
        print("     forward guard: NOT raised (unexpected)")
    except NotImplementedError as e:
        print("     forward guard: raised NotImplementedError(%r)" % str(e)[:44])


# ---------------------------------------------------------------------------------------
# Part B: install the shim; both guards lifted
# ---------------------------------------------------------------------------------------
def part_b(m13, shim, native, dtype):
    print("  -- Part B: shim lifts both guards (%s)" % str(dtype).split(".")[-1])
    shim.install(m13.KimiDeltaAttention, m13.TransformerConfig, native.kda_pre_gdr_fused_native)
    try:
        cfg = _cfg(m13, dtype, True)
        print("     config guard: lifted (config built with fusion=True, "
              "flag=%s)" % cfg.gdn_pre_gated_delta_rule_fusion)
    except NotImplementedError as e:
        print("     config guard: STILL raised ->", e)
        return None
    m13.model_parallel_cuda_manual_seed(123)
    kda = m13.build_kda(cfg)
    print("     module attr gdn_pre_gated_delta_rule_fusion =",
          kda.gdn_pre_gated_delta_rule_fusion)
    torch.manual_seed(1)
    hs = torch.randn(64, 2, 128, device="cuda", dtype=dtype)
    try:
        out, _ = kda(hs, None)
        print("     forward: ran through fused path, out shape=%s finite=%s"
              % (tuple(out.shape), bool(torch.isfinite(out).all())))
    except Exception as e:  # noqa: BLE001
        print("     forward: FAILED ->", type(e).__name__, e)
    return kda


# ---------------------------------------------------------------------------------------
# Part C: numeric -- flag-driven fused path vs eager, same weights
# ---------------------------------------------------------------------------------------
def part_c(m13, dtype, T=64, B=2):
    print("  -- Part C: flag-driven fused vs eager, same weights (%s)"
          % str(dtype).split(".")[-1])
    m13.model_parallel_cuda_manual_seed(123)
    base = m13.build_kda(_cfg(m13, dtype, False))
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    m13.model_parallel_cuda_manual_seed(123)
    fused = m13.build_kda(_cfg(m13, dtype, True))        # shim already installed
    fused.load_state_dict(state)
    assert fused.gdn_pre_gated_delta_rule_fusion is True
    assert base.gdn_pre_gated_delta_rule_fusion is False

    torch.manual_seed(T * 13 + B)
    xa = torch.randn(T, B, 128, device="cuda", dtype=dtype, requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    base.zero_grad(set_to_none=True)
    fused.zero_grad(set_to_none=True)
    oa, _ = base(xa, None)
    ob, _ = fused(xb, None)
    d_out = (oa.float() - ob.float()).abs().max().item()
    oa.float().sum().backward()
    ob.float().sum().backward()
    d_hs = (xa.grad.float() - xb.grad.float()).abs().max().item()

    dp = 0.0
    scale = 0.0
    n_cmp = 0
    base_p = dict(base.named_parameters())
    for name, p in fused.named_parameters():
        ga, gb = base_p[name].grad, p.grad
        if ga is None and gb is None:
            continue
        if ga is None or gb is None:
            print("     [skip] %s grad mismatch presence" % name)
            continue
        n_cmp += 1
        dp = max(dp, (ga.float() - gb.float()).abs().max().item())
        scale = max(scale, ga.float().abs().max().item())
    convw = "conv1d.weight.grad equal=%s" % (
        bool(torch.equal(base.conv1d.weight.grad, fused.conv1d.weight.grad)))
    print("     out max_abs=%.3e  hs.grad max_abs=%.3e  param.grad max_abs=%.3e "
          "(max|param.grad|=%.3e, rel=%.3e, compared=%d)  | %s"
          % (d_out, d_hs, dp, scale, dp / (scale + 1e-30), n_cmp, convw))


# ---------------------------------------------------------------------------------------
# Part D: timing (full-layer fwd+bwd)
# ---------------------------------------------------------------------------------------
def _step(kda, T, B, dtype, seed):
    torch.manual_seed(seed)
    x = torch.randn(T, B, 128, device="cuda", dtype=dtype, requires_grad=True)
    kda.zero_grad(set_to_none=True)
    o, _ = kda(x, None)
    o.float().sum().backward()


def part_d(m13, dtype, cases):
    print("  -- Part D: full-layer fwd+bwd timing, eager vs fused (%s)"
          % str(dtype).split(".")[-1])
    m13.model_parallel_cuda_manual_seed(123)
    base = m13.build_kda(_cfg(m13, dtype, False))
    fused = m13.build_kda(_cfg(m13, dtype, True))
    for i, (T, B) in enumerate(cases):
        seed = T * 5 + B
        te_wall = m13._time(lambda: _step(base, T, B, dtype, seed), warmup=10, iters=30)
        tf_wall = m13._time(lambda: _step(fused, T, B, dtype, seed), warmup=10, iters=30)
        te_dev = m13._dev_time(lambda: _step(base, T, B, dtype, seed), warmup=10, iters=30)
        tf_dev = m13._dev_time(lambda: _step(fused, T, B, dtype, seed), warmup=10, iters=30)
        print("     T=%-5d B=%d  wall: eager=%8.2f fused=%8.2f (%.3fx) | "
              "dev: eager=%8.2f fused=%8.2f (%.3fx)"
              % (T, B, te_wall, tf_wall, te_wall / tf_wall, te_dev, tf_dev, te_dev / tf_dev))


def main():
    m13 = _import_m13()
    mega = m13._import_megatron()
    print("Megatron-LM:", mega)
    shim = _import_local("kda_fusion_shim", "kda_pregdr_fusion_shim.py")
    native = _import_local("kda_pregdr_native", "kda_pregdr_bwd.py")
    m13.init_dist()

    dtypes = (torch.float32, torch.bfloat16)
    # Part A must run with the shim OFF.
    for dtype in dtypes:
        print("=== dtype=%s (guards) ===" % str(dtype).split(".")[-1])
        part_a(m13, dtype)

    for dtype in dtypes:
        print("=== dtype=%s (shim ON) ===" % str(dtype).split(".")[-1])
        part_b(m13, shim, native, dtype)
        part_c(m13, dtype)
        part_d(m13, dtype, [(64, 2), (256, 2), (1024, 2), (1024, 8)])

    m13.ps.destroy_model_parallel()
    print("ALL DONE")


if __name__ == "__main__":
    main()
