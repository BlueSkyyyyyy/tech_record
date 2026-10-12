"""Minimal MPK (Mirage Persistent Kernel) smoke test / micro-benchmark.

Builds a single-layer RMSNorm PersistentKernel in test_mode with *random*
tensors (no HuggingFace weights), compiles it, runs it on the GPU, checks
correctness against a PyTorch reference and reports per-call latency.

This exercises the full MPK pipeline end-to-end on real hardware:
  Python graph build -> tGraph/runtime codegen -> nvcc JIT -> single
  persistent kernel launch (workers + schedulers) -> in-kernel event
  scheduling -> output.

Run inside the kernel_lab container with MIRAGE_HOME pointing at the mirage
checkout, e.g.:

  MIRAGE_HOME=/ssd/mirage_mpk_build python3 mpk_smoke.py --workers 4 --schedulers 4
"""

import argparse
import os
import sys
import time

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_rmsnorm(x, weight, eps=1e-5):
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--schedulers", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=16)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output-dir", type=str,
                   default=os.environ.get("MIRAGE_HOME", "."))
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
    dtype = torch.bfloat16

    x = torch.randn(args.batch, args.hidden, dtype=dtype, device=device)
    w = torch.randn(args.hidden, dtype=dtype, device=device)
    out = torch.zeros(args.batch, args.hidden, dtype=dtype, device=device)

    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = args.workers
    params["num_local_schedulers"] = args.schedulers
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

    print(f"[cfg] workers={args.workers} schedulers={args.schedulers} "
          f"target_cc={target_cc} shape=({args.batch},{args.hidden})", flush=True)

    t0 = time.time()
    pk.compile(output_dir=args.output_dir)
    print(f"[compile] {time.time() - t0:.1f} s", flush=True)

    print("[run] launching persistent kernel ...", flush=True)
    t0 = time.time()
    pk()
    torch.cuda.synchronize()
    print(f"[run] first call {time.time() - t0:.3f} s", flush=True)

    ref = torch_rmsnorm(x, w)
    max_diff = (out - ref).abs().max().item()
    print(f"[check] max|out-ref| = {max_diff:.6e}", flush=True)

    for _ in range(args.warmup):
        pk()
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(args.iters):
        pk()
    ender.record()
    torch.cuda.synchronize()
    total_ms = starter.elapsed_time(ender)
    print(f"[timing] {args.iters} iters, {total_ms / args.iters:.6f} ms/call "
          f"(total {total_ms:.3f} ms)", flush=True)

    ok = max_diff < 0.05
    print(f"[result] {'PASSED' if ok else 'FAILED'}", flush=True)
    pk.finalize()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
