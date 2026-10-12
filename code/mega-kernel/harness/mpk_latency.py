"""End-to-end latency characterization of an MPK (Mirage Persistent Kernel)
megakernel on real hardware.

Builds the same single-layer RMSNorm minimal megakernel as mpk_smoke.py (random
tensors, test_mode, no HuggingFace weights), then measures per-invocation
latency (host launch -> persistent kernel terminates) for several
worker/scheduler configs, plus a naive PyTorch kernels-per-operator reference
for context.

Per-iteration latency is captured with a pair of CUDA events around each
`pk()` call; because the launch path makes the default stream wait on the
worker/scheduler done-events, `event.elapsed_time` for the pair measures the
*full* launch+execute+terminate round trip of one megakernel invocation.

Requires the convergence workaround (see harness/mpk_latency_run.patch):
`include/.../persistent_kernel.cuh` spin-counter + `-DMPK_DBG_WATCHDOG`.

Run inside kernel_lab:
  MIRAGE_HOME=/ssd/mirage_mpk_build CUDA_VISIBLE_DEVICES=7 \
    python3 -u mpk_latency.py --iters 200 --warmup 20 --device 0
"""

import argparse
import os
import statistics
import time

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_rmsnorm(x, weight, eps=1e-5):
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def time_torch_rmsnorm(x, w, iters, warmup):
    for _ in range(warmup):
        torch_rmsnorm(x, w)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        torch_rmsnorm(x, w)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def build_and_time(args, nw, ns, x, w, out, compile_dir):
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = nw
    params["num_local_schedulers"] = ns
    params["mpi_rank"] = 0
    params["world_size"] = 1
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    out_dt = pk.attach_input(out, name="out")

    target_cc = pk.target_cc
    block_dim = (256, 1, 1) if target_cc >= 90 else (128, 1, 1)
    pk.rmsnorm_layer(input=x_dt, weight=w_dt, output=out_dt,
                     grid_dim=(args.batch, 1, 1), block_dim=block_dim)

    t0 = time.time()
    pk.compile(output_dir=compile_dir)
    compile_s = time.time() - t0

    out.zero_()
    t0 = time.time()
    pk()
    torch.cuda.synchronize()
    first_ms = (time.time() - t0) * 1e3

    ref = torch_rmsnorm(x, w)
    max_diff = (out - ref).abs().max().item()

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
    return dict(compile_s=compile_s, first_ms=first_ms, max_diff=max_diff,
                mean=statistics.mean(lat), median=statistics.median(lat),
                mn=min(lat), mx=max(lat), iters=len(lat),
                sd=statistics.pstdev(lat) if len(lat) > 1 else 0.0,
                lat=lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--configs", type=str, default="128:16,64:16,16:16")
    ap.add_argument("--output-dir", type=str,
                    default=os.environ.get("MIRAGE_HOME", "."))
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
    dtype = torch.bfloat16

    x = torch.randn(args.batch, args.hidden, dtype=dtype, device=device)
    w = torch.randn(args.hidden, dtype=dtype, device=device)
    out = torch.zeros(args.batch, args.hidden, dtype=dtype, device=device)

    print(f"[env] mirage; batch={args.batch} hidden={args.hidden} "
          f"dtype={dtype} iters={args.iters} warmup={args.warmup}", flush=True)

    torch_lat = time_torch_rmsnorm(x, w, args.iters, args.warmup)
    print(f"[torch] naive RMSNorm (kernels-per-operator) "
          f"{torch_lat * 1e3:.3f} us/call", flush=True)

    print(f"{'workers':>8} {'sched':>6} {'max|out-ref|':>14} "
          f"{'mean(us)':>10} {'median':>8} {'min':>8} {'max':>8} {'sd':>7}", flush=True)

    for cfg in args.configs.split(","):
        nw, ns = (int(v) for v in cfg.split(":"))
        r = build_and_time(args, nw, ns, x, w, out, args.output_dir)
        print(f"{nw:>8} {ns:>6} {r['max_diff']:>14.6e} "
              f"{r['mean'] * 1e3:>10.3f} {r['median'] * 1e3:>8.3f} "
              f"{r['mn'] * 1e3:>8.3f} {r['mx'] * 1e3:>8.3f} "
              f"{r['sd'] * 1e3:>7.3f}  (compile {r['compile_s']:.1f}s, "
              f"first {r['first_ms']:.1f}ms, tested {r['iters']} iters)",
              flush=True)


if __name__ == "__main__":
    main()
