# M11-2: close the M11-1 leave-out -- **packed THD *multi-sequence* CP**.
#
# M11-1 (docs/12) anchored the real ``KimiDeltaAttention`` under cp>1 for (a) packed THD
# *single*-sequence and (b) SBHD multi-sequence, and left packed THD *multi*-sequence as an
# explicit boundary: with a plain ``PackedSeqParams(cu_seqlens=[0,64,128], qkv_format="thd")``
# and a per-token global-zigzag shard, the errors ballooned to O(1) and the layer printed the
# ``conversion.py:120`` FutureWarning ("THD PackedSeqParams is missing precomputed
# context-parallel layout routes").
#
# Root cause (this harness): it was **not** the route at all (the conversion rebuilds an
# equivalent route from ``cu_seqlens`` whenever the precomputed one is absent). It was the
# *input shard*: Megatron's THD CP *zigzag* layout partitions **each packed sequence
# independently** into ``2*cp_size`` chunks
# (``context_parallel_layout/routes.py::_build_thd_layout_segments``), whereas
# ``get_pos_emb_on_this_cp_rank`` -- the M11-1 shard recipe -- zigzags the **whole concatenated
# token axis**. For a single sequence the two coincide; for a multi-sequence pack they differ
# (e.g. cu=[0,64,128], cp=2: per-seq zigzag gives rank0 tokens {0..15,48..63,64..79,112..127},
# the global zigzag gives {0..31,96..127}). Feeding the global-zigzag shard therefore hands the
# layer a buffer whose rows are not the ones its route expects -> O(1).
#
# This harness (M11-2) shards by the **per-sequence zigzag** the layer actually uses, and
# prebuilds the THD CP routes through the public ``prebuild_thd_cp_partition_routes`` API so the
# fallback FutureWarning disappears (the documented "prebuild routes when constructing the
# batch" contract). Stage 1 / Stage 2 are otherwise identical to M11-1.
#
# Run (GPU, N idle devices):
#   scripts/lab.sh exec "cd \$MEGATRON_LM && CUDA_VISIBLE_DEVICES=1,2 \
#       python <abs>/harness/check_port_e2e_cp_thd.py --worlds 2"

import argparse
import importlib.util
import os
import socket
import sys

os.environ.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEGA_DEFAULT = "~/proj/Megatron-LM"


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
    global prebuild_thd_cp_partition_routes, get_pos_emb_on_this_cp_rank
    from megatron.core.models.common.embeddings.rope_utils import (
        get_pos_emb_on_this_cp_rank,
    )
    from megatron.core import parallel_state as ps
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.ssm.gated_delta_net import KimiDeltaAttention
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
    return mega


def make_config(dtype, cp_size, arm):
    bf16 = dtype == torch.bfloat16
    import torch.nn.functional as F

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


def make_packed_seq_params(cu_list, cp_group=None):
    cu = torch.IntTensor(cu_list).cuda()
    seqlens = cu[1:] - cu[:-1]
    m = int(seqlens.max().item())
    psp = PackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=m,
        max_seqlen_kv=m,
        qkv_format="thd",
        cp_group=cp_group,
    )
    if cp_group is not None and cp_group.size() > 1:
        # Documented steady-state contract: prebuild the THD CP routes at batch construction
        # instead of letting the conversion fall back to a device-to-host rebuild + warning.
        prebuild_thd_cp_partition_routes(psp, cp_group)
    return psp


# ---------------------------------------------------------------------------------------
# Per-sequence zigzag sharding -- the layout the THD CP routes actually consume
# ---------------------------------------------------------------------------------------
def per_seq_zigzag_local_indices(cu_list, cp_size, cp_rank):
    """Global token indices owned by ``cp_rank`` under Megatron's THD zigzag layout.

    Mirrors ``context_parallel_layout/routes.py::_build_thd_layout_segments``: each packed
    sequence is split into ``2*cp_size`` equal chunks; the rank owns chunk ``cp_rank`` then
    chunk ``2*cp_size - cp_rank - 1``, per sequence, sequences visited in order.
    """
    idx = []
    for s, e in zip(cu_list[:-1], cu_list[1:]):
        seq_len = e - s
        chunk = seq_len // (2 * cp_size)
        first = s + cp_rank * chunk
        second = s + (2 * cp_size - cp_rank - 1) * chunk
        idx.extend(range(first, first + chunk))
        idx.extend(range(second, second + chunk))
    return idx


def shard_thd(t, cu_list, cp_group, mode="perseq"):
    if mode == "global":
        # M11-1 recipe: zigzag the whole concatenated token axis (wrong for multi-sequence packs).
        return get_pos_emb_on_this_cp_rank(t, 0, cp_group)
    rank = cp_group.rank()
    idx = per_seq_zigzag_local_indices(cu_list, cp_group.size(), rank)
    sel = torch.tensor(idx, device=t.device, dtype=torch.long)
    return t.index_select(0, sel)


def _ref_core(ref, q, k, v, g, beta, A_log, dt_bias, safe_gate, lower_bound, cu_seqlens):
    act_g = ref.kda_gate(g, A_log, dt_bias, lower_bound if safe_gate else None)
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
        initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False, safe_gate=False, lower_bound=None, cu_seqlens=None,
        cp_context=None, **kwargs,
    ):
        assert cp_context is None, "reference path is single-rank (cp=1)"
        o, _ = _ref_core(ref, q, k, v, g, beta, A_log, dt_bias, safe_gate, lower_bound, cu_seqlens)
        return o.to(q.dtype), None

    return gdr


def relerr(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


# ---------------------------------------------------------------------------------------
# Stage 1: cp=1 fp64 reference on the full (multi-sequence) pack
# ---------------------------------------------------------------------------------------
def stage1_reference(ref, dtype, cu_list, seed, out_dir):
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

    torch.manual_seed(seed)
    x = torch.randn(cu_list[-1], 1, config.hidden_size, device="cuda", dtype=dtype).requires_grad_(True)
    psp = make_packed_seq_params(cu_list)

    kda_r.zero_grad(set_to_none=True)
    out_r, _ = kda_r(x, None, packed_seq_params=psp)
    out_r.float().sum().backward()

    ref_pg = {n: (p.grad.detach().float().cpu() if p.grad is not None else None)
              for n, p in kda_r.named_parameters()}
    torch.save({
        "state": {k: v.float().cpu() for k, v in state.items()},
        "o_ref": out_r.detach().float().cpu(),
        "grad_ref": x.grad.detach().float().cpu(),
        "param_grad_ref": ref_pg,
        "cu_list": cu_list,
        "dtype": str(dtype),
        "seed": seed,
    }, os.path.join(out_dir, "ref_thd.pt"))
    ps.destroy_model_parallel()
    print("  [stage1] cp=1 fp64-ref cu=%s: out=%s |o|max=%.4f"
          % (cu_list, tuple(out_r.shape), out_r.abs().max().item()), flush=True)


# ---------------------------------------------------------------------------------------
# Stage 2: cp=world real layer, per-sequence-zigzag shard, vs stage-1 sharded identically
# ---------------------------------------------------------------------------------------
def _cp_worker(rank, world, arm, port, out_dir, shard_mode="perseq"):
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

    save = torch.load(os.path.join(out_dir, "ref_thd.pt"), map_location="cpu")
    dtype = getattr(torch, save["dtype"].split(".")[-1])
    cu_list = save["cu_list"]
    config = make_config(dtype, cp_size=world, arm=arm)
    model_parallel_cuda_manual_seed(save["seed"])
    kda = build_kda(config)
    sd = kda.state_dict()
    kda.load_state_dict({k: v.to(device="cuda", dtype=sd[k].dtype) for k, v in save["state"].items()})

    torch.manual_seed(save["seed"])
    x_full = torch.randn(cu_list[-1], 1, config.hidden_size, device="cuda", dtype=dtype)
    cp_group = ps.get_context_parallel_group()
    psp = make_packed_seq_params(cu_list, cp_group=cp_group)
    x_local = shard_thd(x_full, cu_list, cp_group, shard_mode).detach().requires_grad_(True)

    kda.zero_grad(set_to_none=True)
    out, bias = kda(x_local, None, packed_seq_params=psp)
    out.float().sum().backward()

    o_ref = save["o_ref"].cuda()
    g_ref = save["grad_ref"].cuda()
    o_ref_local = shard_thd(o_ref, cu_list, cp_group, shard_mode)
    g_ref_local = shard_thd(g_ref, cu_list, cp_group, shard_mode)

    d_out = (out.float() - o_ref_local).abs().max().item()
    d_grad = (x_local.grad.float() - g_ref_local).abs().max().item()
    for p in kda.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=cp_group)

    d_param, worst_name = 0.0, ""
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
        print("  [stage2] world=%d arm=%-9s cu=%-18s T_loc=%d  o max_abs=%.3e rel=%.3e | "
              "hs.grad max_abs=%.3e rel=%.3e | param.grad max_abs=%.3e rel=%.3e (%s)"
              % (world, arm, str(cu_list), out.shape[0], d_out, relerr(out, o_ref_local),
                 d_grad, relerr(x_local.grad, g_ref_local), d_param, pgrel, worst_name),
              flush=True)

    dist.barrier()
    dist.destroy_process_group()


def run_case(dtype, arm, world, cu_list, seed, out_dir, ref, shard_mode="perseq"):
    import torch.multiprocessing as mp

    cu = [int(x) for x in cu_list.split(",")]
    stage1_reference(ref, dtype, cu, seed, out_dir)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.set_start_method("spawn", force=True)
    mp.spawn(_cp_worker, args=(world, arm, port, out_dir, shard_mode), nprocs=world, join=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="2")
    ap.add_argument("--arms", default="headwise,chunkwise")
    ap.add_argument("--dtypes", default="bfloat16,float32")
    # (world, cu) pairs; every seq length % (2*world)==0 and total % (4*world)==0 by construction.
    # The pack separator is '+' so the argument survives the docker-exec shell without quoting.
    ap.add_argument("--cases", default="2:0,64,128+2:0,64,160+4:0,64,128,192,256")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--shard", default="perseq", choices=["perseq", "global"],
                    help="perseq = per-sequence zigzag (the layer's THD layout); "
                         "global = M11-1's whole-axis zigzag (control / reproduces the O(1) bug)")
    args = ap.parse_args()

    worlds = [int(w) for w in args.worlds.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]
    dtypes = [getattr(torch, d.strip()) for d in args.dtypes.split(",")]
    cases = [(int(c.split(":")[0]), c.split(":")[1]) for c in args.cases.split("+")]

    print("=" * 100)
    print("megatron-port M11-2: packed THD *multi-sequence* CP anchor vs fp64 reference")
    print("=" * 100)

    out_dir = os.path.join(_HERE, "_cptmp_m11b")
    os.makedirs(out_dir, exist_ok=True)

    for dtype in dtypes:
        for arm in arms:
            for world, cu_list in cases:
                if world not in worlds:
                    continue
                print("=== dtype=%s arm=%s world=%d cu=%s shard=%s seed=%d ==="
                      % (str(dtype).split(".")[-1], arm, world, cu_list, args.shard, args.seed))
                ref = _load_kda_ref()
                run_case(dtype, arm, world, cu_list, args.seed, out_dir, ref, args.shard)

    print("ALL DONE")


if __name__ == "__main__":
    main()
