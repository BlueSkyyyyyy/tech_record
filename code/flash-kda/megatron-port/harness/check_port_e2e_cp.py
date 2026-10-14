# M11-1: real-layer *context-parallel* (cp>1) numerical anchor vs the fp64 reference.
#
# M8-1/M8-2 anchored the FLA *operator* under CP (``build_cp_context`` + real NCCL ranks).
# M10-1/M10-2/M10-3 anchored the *real* ``KimiDeltaAttention`` module end-to-end but only at
# cp=1.  This harness closes that gap: it runs the real layer with a genuine
# ``context_parallel_size = world`` process group (``mp.spawn`` + NCCL) and anchors the
# CP-sharded output / input-grad / parameter grads against the **fp64 reference core** run on
# the full sequence at cp=1, for both ``linear_cp_mode`` values.
#
# Two input layouts are covered:
#   * ``packed`` : a THD ``PackedSeqParams`` batch ([T_total,1,H]); single-sequence packs are
#                  the supported CP path here (see the finding in docs/12);
#   * ``sbhd``   : an SBHD minibatch ([T,B,H]) with no ``packed_seq_params`` -- this is
#                  Megatron's own supported multi-sequence CP layout (headwise; chunkwise
#                  requires B==1).
#
# Why two stages (not one process): the fp64 reference core has no cross-rank state exchange,
# so it cannot sit inside a cp>1 module (CP *is* the cross-rank state handoff the layer
# performs).  Stage 1 builds a cp=1 module, swaps ``gated_delta_rule`` for the fp64 reference
# (``make_ref_gdr``, M10-2, packed-aware), runs the full-sequence fwd/bwd and dumps
# ``o``/``hs.grad``/param-grads + frozen weights.  Stage 2 spawns ``world`` NCCL ranks,
# rebuilds the real layer at cp=world from the same weights, feeds each rank its zigzag token
# shard of the *same* full input, runs fwd/bwd, and compares the rank-local tensors against
# the stage-1 tensors sharded identically (``get_pos_emb_on_this_cp_rank`` -- exactly
# Megatron's test recipe).
#
# The backward uses a count-invariant loss (``out.float().sum()``; ``mean`` would scale CP
# input grads by the local token count), so a
# forward *layout* mismatch shows in ``o`` while ``hs.grad`` can still look small; the harness
# prints both.
#
# Two container import-time workarounds (same as M10-*, unrelated to KDA):
#   * patch ``nvidia_resiliency_ext.__version__`` to satisfy the Megatron >=0.6.0 assertion;
#   * set ``EP_SUPPRESS_NCCL_CHECK=1`` to bypass DeepEP's libnccl byte comparison.
# Single-node NCCL needs ``NCCL_NVLS_ENABLE=0``.
#
# Run (GPU, N idle devices):
#   scripts/lab.sh exec "cd \$MEGATRON_LM && CUDA_VISIBLE_DEVICES=1,2 \
#       python <abs>/harness/check_port_e2e_cp.py --worlds 2 --arms headwise,chunkwise"

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEGA_DEFAULT = os.path.expanduser("~/proj/Megatron-LM")


# ---------------------------------------------------------------------------------------
# Import plumbing (identical to M10-1/M10-2)
# ---------------------------------------------------------------------------------------
def _load_kda_ref():
    path = os.path.join(_HERE, "..", "src", "kda_ref.py")
    spec = importlib.util.spec_from_file_location("kda_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_megatron():
    mega = os.environ.get("MEGATRON_LM_DIR") or _MEGA_DEFAULT
    if mega not in sys.path:
        sys.path.insert(0, mega)
    try:
        import nvidia_resiliency_ext as _nvrx

        _nvrx.__version__ = "0.6.0"
    except Exception as exc:  # pragma: no cover
        print("  [warn] nvrx version patch skipped:", exc)
    global ps, ProcessGroupCollection, hybrid_stack_spec, KimiDeltaAttention
    global model_parallel_cuda_manual_seed, TransformerConfig, PackedSeqParams
    global get_tensor_on_this_cp_rank
    from megatron.core import parallel_state as ps
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.ssm.gated_delta_net import KimiDeltaAttention
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.models.common.embeddings.rope_utils import (
        get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
    )
    return mega


# ---------------------------------------------------------------------------------------
# Config / module construction (mirrors M10-1 / test_kda.py)
# ---------------------------------------------------------------------------------------
def make_config(dtype, cp_size, arm):
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
        context_parallel_size=cp_size,
        sequence_parallel=False,
        linear_cp_mode=arm,
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


def make_input(hidden, dtype, layout, cu_list, batch, seed):
    """Full (unsharded) input, generated deterministically from ``seed`` on every rank."""
    torch.manual_seed(seed)
    if layout == "packed":
        return torch.randn(cu_list[-1], 1, hidden, device="cuda", dtype=dtype)
    return torch.randn(cu_list[-1], batch, hidden, device="cuda", dtype=dtype)


def _ref_core(ref, q, k, v, g, beta, A_log, dt_bias, safe_gate, lower_bound, cu_seqlens):
    """fp64 chunked reference forward; packed -> per-sequence loop, SBHD -> batch-native."""
    act_g = ref.kda_gate(g, A_log, dt_bias, lower_bound if safe_gate else None)
    if cu_seqlens is None:
        o, _ = ref.kda_chunked_fwd(
            q, k, v, act_g, beta, initial_state=None, output_final_state=False, chunk_size=64
        )
        return o, None
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
            ref, q, k, v, g, beta, A_log, dt_bias, safe_gate, lower_bound, cu_seqlens
        )
        o = o.to(q.dtype)
        return o, None

    return gdr


def relerr(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


def shard(t, layout, cp_group):
    """Full -> this rank's zigzag shard (Megatron's ``get_pos_emb_on_this_cp_rank`` recipe)."""
    local = get_tensor_on_this_cp_rank(t, 0, cp_group)
    if layout == "packed":
        return local.transpose(0, 1).contiguous().view(-1, 1, t.shape[-1])
    return local


# ---------------------------------------------------------------------------------------
# Stage 1: cp=1 fp64 reference on the full sequence
# ---------------------------------------------------------------------------------------
def stage1_reference(ref, dtype, layout, cu_list, batch, seed, out_dir):
    _import_megatron()
    if not torch.distributed.is_initialized():
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(0)
        torch.distributed.init_process_group("nccl", world_size=1, rank=0, init_method="env://")
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(1, 1, context_parallel_size=1)

    config = make_config(dtype, cp_size=1, arm="headwise")
    model_parallel_cuda_manual_seed(seed)
    kda = build_kda(config)
    state = {k: v.detach().clone() for k, v in kda.state_dict().items()}

    model_parallel_cuda_manual_seed(seed)
    kda_r = build_kda(config)
    kda_r.load_state_dict(state)
    kda_r.gated_delta_rule = make_ref_gdr(ref)

    x = make_input(config.hidden_size, dtype, layout, cu_list, batch, seed).requires_grad_(True)
    psp = make_packed_seq_params(cu_list) if layout == "packed" else None

    kda_r.zero_grad(set_to_none=True)
    out_r, _ = kda_r(x, None, packed_seq_params=psp)
    out_r.float().sum().backward()

    ref_pg = {n: (p.grad.detach().float().cpu() if p.grad is not None else None)
              for n, p in kda_r.named_parameters()}
    save = {
        "state": {k: v.float().cpu() for k, v in state.items()},
        "o_ref": out_r.detach().float().cpu(),
        "grad_ref": x.grad.detach().float().cpu(),
        "param_grad_ref": ref_pg,
        "cu_list": cu_list,
        "layout": layout,
        "batch": batch,
        "dtype": str(dtype),
        "seed": seed,
    }
    torch.save(save, os.path.join(out_dir, "ref.pt"))
    ps.destroy_model_parallel()
    print("  [stage1] cp=1 fp64-ref lay=%s cu=%s B=%d: out shape=%s |o|max=%.4f"
          % (layout, cu_list, batch, tuple(out_r.shape), out_r.abs().max().item()), flush=True)


# ---------------------------------------------------------------------------------------
# Stage 2: cp=world real layer, compared to stage-1 tensors sharded identically
# ---------------------------------------------------------------------------------------
def _cp_worker(rank, world, arm, port, out_dir):
    os.environ.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    import torch.distributed as dist

    dist.init_process_group("nccl", rank=rank, world_size=world, init_method="env://")
    torch.cuda.set_device(rank % torch.cuda.device_count())

    _import_megatron()
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(1, 1, context_parallel_size=world)

    save = torch.load(os.path.join(out_dir, "ref.pt"), map_location="cpu")
    dtype = getattr(torch, save["dtype"].split(".")[-1])
    layout, cu_list, batch = save["layout"], save["cu_list"], save["batch"]
    config = make_config(dtype, cp_size=world, arm=arm)
    model_parallel_cuda_manual_seed(save["seed"])
    kda = build_kda(config)
    sd = kda.state_dict()
    kda.load_state_dict({k: v.to(device="cuda", dtype=sd[k].dtype) for k, v in save["state"].items()})

    x_full = make_input(config.hidden_size, dtype, layout, cu_list, batch, save["seed"])
    psp = make_packed_seq_params(cu_list) if layout == "packed" else None
    cp_group = ps.get_context_parallel_group()
    x_local = shard(x_full, layout, cp_group).detach().requires_grad_(True)

    kda.zero_grad(set_to_none=True)
    out, bias = kda(x_local, None, packed_seq_params=psp)
    out.float().sum().backward()

    o_ref = save["o_ref"].to(device="cuda", dtype=torch.float32)
    g_ref = save["grad_ref"].to(device="cuda", dtype=torch.float32)
    o_ref_local = shard(o_ref, layout, cp_group)
    g_ref_local = shard(g_ref, layout, cp_group)

    d_out = (out.float() - o_ref_local).abs().max().item()
    d_grad = (x_local.grad.float() - g_ref_local).abs().max().item()
    # Replicated parameters: each CP rank holds a token shard, so its grad is a *partial*
    # sum; the full-sequence grad is the all-reduce over the CP group (data-parallel-style).
    for p in kda.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=cp_group)

    d_param = 0.0
    worst_name = ""
    for name, p in kda.named_parameters():
        rg = save["param_grad_ref"].get(name)
        if p.grad is None or rg is None:
            continue
        rg = rg.to(device="cuda", dtype=torch.float32)
        dd = (p.grad.float() - rg).abs().max().item()
        if dd > d_param:
            d_param, worst_name = dd, name
    pgr = save["param_grad_ref"].get(worst_name)
    pgrel = (d_param / (pgr.abs().max().item() + 1e-30)) if pgr is not None else float("nan")

    if rank == 0:
        print("  [stage2] world=%d arm=%-9s lay=%-6s out_local=%s  o max_abs=%.3e rel=%.3e | "
              "hs.grad max_abs=%.3e rel=%.3e | param.grad max_abs=%.3e rel=%.3e (%s)"
              % (world, arm, layout, tuple(out.shape), d_out, relerr(out, o_ref_local),
                 d_grad, relerr(x_local.grad, g_ref_local), d_param, pgrel, worst_name), flush=True)
        if os.environ.get("M11_DEBUG"):
            print("  [dbg] |grad_cp|max=%.3e |grad_ref_local|max=%.3e |grad_ref_full|max=%.3e "
                  "|o_ref|max=%.3e"
                  % (x_local.grad.abs().max().item(), g_ref_local.abs().max().item(),
                     g_ref.abs().max().item(), o_ref.abs().max().item()), flush=True)

    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------------------
def run_case(dtype, layout, arm, world, cu_list, batch, seed, out_dir, ref):
    import torch.multiprocessing as mp

    stage1_reference(ref, dtype, layout, cu_list, batch, seed, out_dir)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.set_start_method("spawn", force=True)
    mp.spawn(_cp_worker, args=(world, arm, port, out_dir), nprocs=world, join=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="2")
    ap.add_argument("--arms", default="headwise,chunkwise")
    ap.add_argument("--dtypes", default="bfloat16,float32")
    ap.add_argument("--layouts", default="packed,sbhd")
    ap.add_argument("--cu", default="0,128")        # single-sequence packed pack
    ap.add_argument("--batch", type=int, default=2)  # SBHD batch
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    worlds = [int(w) for w in args.worlds.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]
    dtypes = [getattr(torch, d.strip()) for d in args.dtypes.split(",")]
    layouts = [l.strip() for l in args.layouts.split(",")]
    cu_list = [int(x) for x in args.cu.split(",")]

    print("=" * 100)
    print("megatron-port M11-1: real-layer cp>1 anchor vs fp64 reference")
    print("=" * 100)

    out_dir = os.path.join(_HERE, "_cptmp_m11")
    os.makedirs(out_dir, exist_ok=True)

    for dtype in dtypes:
        for layout in layouts:
            for arm in arms:
                for world in worlds:
                    if layout == "sbhd" and arm == "chunkwise" and args.batch > 1:
                        continue  # chunkwise CP SBHD requires B==1
                    print("=== dtype=%s layout=%s arm=%s world=%d cu=%s B=%d seed=%d ==="
                          % (str(dtype).split(".")[-1], layout, arm, world, cu_list, args.batch, args.seed))
                    ref = _load_kda_ref()
                    run_case(dtype, layout, arm, world, cu_list, args.batch, args.seed, out_dir, ref)

    print("ALL DONE")


if __name__ == "__main__":
    main()
