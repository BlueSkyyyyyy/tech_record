"""End-to-end latency of a *multi-operator* MPK (Mirage Persistent Kernel)
megakernel on real hardware (P26-1, follows docs/09 section 8 / 11).

Where mpk_latency.py builds a single rmsnorm_layer, this script builds a full
dense transformer MLP block -- a 4-operator DAG inside ONE persistent kernel:

    x --rmsnorm--> h --linear(gate+up)--> mid --silu_mul--> silu
                                                    --linear_with_residual--> out

It sweeps the batch size (which changes both the tensor sizes and the rmsnorm
task count / grid) and, for each, reports the per-invocation latency
(host launch -> double persistent kernel terminates) plus a PyTorch
kernels-per-operator reference for the same block.

Design note: each batch builds its own PersistentKernel because grid_dim for
rmsnorm_layer is (batch,1,1) -- the operator graph is batch-specialized, exactly
as in mpk_latency.py / the official test_mode tests.

Run inside kernel_lab on the *clean* scheduler-first build (see
harness/mpk_latency_multi.sh, which applies harness/mpk_sched_first.patch and
sets MPK_EXTRA_DEFINES=-DMPK_DBG_SCHED_FIRST -- no kernel-code changes):

  MIRAGE_HOME=/ssd/mirage_mpk_build CUDA_VISIBLE_DEVICES=7 \\
    python3 -u mpk_latency_multi.py --device 0 --batch-list 1,8,16,32
"""

import argparse
import os
import statistics
import time

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def grid_for_linear(output_dim):
    """grid_dim.x for linear_layer, matching demo/qwen3/demo.py."""
    if output_dim % 96 == 0:
        return output_dim // 96
    elif output_dim % 64 == 0:
        return output_dim // 64
    else:
        raise AssertionError(f"Unsupported linear output_dim={output_dim}")


def torch_rmsnorm(x, weight, eps=1e-5):
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def torch_mlp_block(x, w_norm, w_gate, w_up, w_down, residual):
    """Reference: rmsnorm -> gate+up -> silu_mul -> down+residual."""
    h = torch_rmsnorm(x, w_norm)
    ref_gate = h.float() @ w_gate.float().T
    ref_up = h.float() @ w_up.float().T
    ref_silu = torch.nn.functional.silu(ref_gate) * ref_up
    return (ref_silu @ w_down.float().T + residual.float()).to(x.dtype)


def time_torch(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def build_and_time(args, nw, ns, tensors, compile_dir):
    (x, w_norm, w_gate, w_up, w_down, residual, out) = tensors
    hidden = args.hidden
    inter = args.intermediate
    fused_outdim = 2 * inter
    B = args.batch

    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = nw
    params["num_local_schedulers"] = ns
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = B
    params["max_num_batched_requests"] = B
    # default page_size=1, max_seq_length=1 -> one page per request; the
    # graph-build KV-capacity check needs max_num_pages >= batch (see
    # persistent_kernel.py:_check_kv_capacity).
    params["max_num_pages"] = max(B, 1)
    pk = PersistentKernel(**params)

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)

    x_dt = pk.attach_input(x, name="x")
    w_norm_dt = pk.attach_input(w_norm, name="w_norm")
    w_gate_dt = pk.attach_input(w_gate, name="w_gate")
    w_up_dt = pk.attach_input(w_up, name="w_up")
    w_down_dt = pk.attach_input(w_down, name="w_down")
    residual_dt = pk.attach_input(residual, name="residual")
    h_dt = pk.attach_input(torch.zeros(B, hidden, dtype=x.dtype, device=x.device), name="h")
    mid_dt = pk.attach_input(torch.zeros(B, fused_outdim, dtype=x.dtype, device=x.device), name="mid")
    silu_dt = pk.attach_input(torch.zeros(B, inter, dtype=x.dtype, device=x.device), name="silu")
    out_dt = pk.attach_input(out, name="out")

    num_tasks_gatedup = grid_for_linear(fused_outdim)
    w_gatedup_dt = pk.shuffle_tensors(
        inputs=[w_gate_dt, w_up_dt],
        shuffled_dim=0,
        num_groups=num_tasks_gatedup // 2,
        name="w_gatedup",
    )

    # 4-operator DAG inside one persistent kernel
    pk.rmsnorm_layer(input=x_dt, weight=w_norm_dt, output=h_dt,
                     grid_dim=(B, 1, 1), block_dim=block_dim)
    pk.linear_layer(input=h_dt, weight=w_gatedup_dt, output=mid_dt,
                    grid_dim=(num_tasks_gatedup, 1, 1), block_dim=block_dim)
    pk.silu_mul_layer(input=mid_dt, output=silu_dt,
                      grid_dim=(num_tasks_gatedup // 2, 1, 1), block_dim=block_dim)
    pk.linear_with_residual_layer(input=silu_dt, weight=w_down_dt,
                                  residual=residual_dt, output=out_dt,
                                  grid_dim=(hidden // 64, 1, 1), block_dim=block_dim)

    t0 = time.time()
    pk.compile(output_dir=compile_dir)
    compile_s = time.time() - t0

    out.zero_()
    pk()
    torch.cuda.synchronize()
    ref = torch_mlp_block(x, w_norm, w_gate, w_up, w_down, residual)
    max_diff = (out.float() - ref.float()).abs().max().item()

    for _ in range(args.warmup):
        pk()
    torch.cuda.synchronize()

    lat = []
    evs = []
    for _ in range(args.iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        pk()
        e.record()
        evs.append((s, e))
    torch.cuda.synchronize()
    for s, e in evs:
        lat.append(s.elapsed_time(e))

    pk.finalize()
    return dict(compile_s=compile_s, max_diff=max_diff,
                mean=statistics.mean(lat), median=statistics.median(lat),
                mn=min(lat), mx=max(lat),
                sd=statistics.pstdev(lat) if len(lat) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-list", type=str, default="1,8,16,32")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--intermediate", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--configs", type=str, default="128:16")
    ap.add_argument("--output-dir", type=str,
                    default=os.environ.get("MIRAGE_HOME", "."))
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
    dtype = torch.bfloat16
    hidden = args.hidden
    inter = args.intermediate

    torch.manual_seed(42)
    x = torch.randn(64, hidden, dtype=dtype, device=device)
    w_norm = torch.randn(hidden, dtype=dtype, device=device)
    w_gate = torch.randn(inter, hidden, dtype=dtype, device=device) * 0.01
    w_up = torch.randn(inter, hidden, dtype=dtype, device=device) * 0.01
    w_down = torch.randn(hidden, inter, dtype=dtype, device=device) * 0.01
    residual = torch.randn(64, hidden, dtype=dtype, device=device)
    out = torch.zeros(64, hidden, dtype=dtype, device=device)

    print(f"[env] multi-op megakernel: rmsnorm -> linear(gate+up) -> silu_mul -> "
          f"linear+residual; hidden={hidden} inter={inter} dtype={dtype} "
          f"iters={args.iters} warmup={args.warmup}", flush=True)

    nw, ns = (int(v) for v in args.configs.split(",")[0].split(":"))

    print(f"{'batch':>6} {'workers':>7} {'sched':>5} {'max|out-ref|':>14} "
          f"{'mean(us)':>10} {'median':>8} {'min':>8} {'max':>8} {'sd':>7} "
          f"{'torch(us)':>10}", flush=True)

    for cfg in args.configs.split(","):
        nw, ns = (int(v) for v in cfg.split(":"))
        for B in (int(v) for v in args.batch_list.split(",")):
            args.batch = B
            tensors = (x[:B].contiguous(), w_norm, w_gate, w_up, w_down,
                       residual[:B].contiguous(), out[:B])
            # torch reference timing for the same block
            torch_lat = time_torch(
                lambda: torch_mlp_block(x[:B], w_norm, w_gate, w_up, w_down,
                                        residual[:B]),
                args.iters, args.warmup)
            try:
                r = build_and_time(args, nw, ns, tensors, args.output_dir)
            except Exception as exc:  # compile-time constraint / build failure
                msg = (str(exc).strip().splitlines()[-1]
                       if str(exc).strip() else repr(exc))
                print(f"{B:>6} {nw:>7} {ns:>5} {'BUILD_FAIL':>14} "
                      f"{torch_lat * 1e3:>10.3f}  <-- {type(exc).__name__}: "
                      f"{msg[:120]}", flush=True)
                continue
            print(f"{B:>6} {nw:>7} {ns:>5} {r['max_diff']:>14.6e} "
                  f"{r['mean'] * 1e3:>10.3f} {r['median'] * 1e3:>8.3f} "
                  f"{r['mn'] * 1e3:>8.3f} {r['mx'] * 1e3:>8.3f} "
                  f"{r['sd'] * 1e3:>7.3f} {torch_lat * 1e3:>10.3f}  "
                  f"(compile {r['compile_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
