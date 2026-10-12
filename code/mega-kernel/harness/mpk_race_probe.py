"""Single-shot convergence probe for the MPK worker wait-loop race.

Mirrors the official tests/runtime_python/test_mode/test_rmsnorm_testmode.py:
compile the minimal single-layer RMSNorm PersistentKernel, launch it, and
require torch.cuda.synchronize() to return. Exit codes:
    0   PASS  (converged and numerically correct)
    1   WRONG (converged but wrong output)
    124 HANG  (did not converge within --hang-timeout)

The reference is computed *after* the launch so the GPU is idle between nvcc
and pk(), matching the official test's timing (the race is timing sensitive).

Environment:
  MIRAGE_HOME  path to the mirage checkout (default /ssd/mirage_mpk_build)

Run inside kernel_lab, e.g.:
  MIRAGE_HOME=/ssd/mirage_mpk_build CUDA_VISIBLE_DEVICES=7 \
    python3 -u mpk_race_probe.py --workers 128 --schedulers 16
"""

import argparse
import os
import sys
import threading
import time

import torch

import mirage  # noqa: F401
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_rmsnorm(x, weight, eps=1e-5):
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def sync_with_timeout(timeout_s):
    done = {"ok": False}

    def worker():
        torch.cuda.synchronize()
        done["ok"] = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout_s)
    return done["ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--schedulers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--hang-timeout", type=float, default=30.0)
    ap.add_argument("--prewarm", type=str, default="none",
                    choices=["none", "sync", "rmsnorm", "rmsnorm_diff",
                             "mean", "mul", "copy", "matmul", "sleep",
                             "sumrows", "sumall", "softmax", "clamp",
                             "cast", "alloc", "cast_sum", "addrow",
                             "to_same", "to_half", "copyto32", "to64"])
    ap.add_argument("--prewarm-ms", type=float, default=200.0)
    ap.add_argument("--device", type=int, default=7)
    ap.add_argument("--output-dir", type=str,
                    default=os.environ.get("MIRAGE_HOME", "."))
    args = ap.parse_args()

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
    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.rmsnorm_layer(input=x_dt, weight=w_dt, output=out_dt,
                     grid_dim=(args.batch, 1, 1), block_dim=block_dim)

    t0 = time.time()
    pk.compile(output_dir=args.output_dir)
    print(f"[probe] compiled in {time.time() - t0:.1f}s "
          f"workers={args.workers} sched={args.schedulers}", flush=True)

    if args.prewarm != "none":
        t0 = time.time()
        if args.prewarm == "sync":
            torch.cuda.synchronize()
        elif args.prewarm == "rmsnorm":
            torch_rmsnorm(x, w)
            torch.cuda.synchronize()
        elif args.prewarm == "rmsnorm_diff":
            x2 = torch.randn_like(x)
            w2 = torch.randn_like(w)
            torch_rmsnorm(x2, w2)
            torch.cuda.synchronize()
        elif args.prewarm == "mean":
            x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            torch.cuda.synchronize()
        elif args.prewarm == "mul":
            out.copy_(x * w)
            torch.cuda.synchronize()
        elif args.prewarm == "copy":
            out.copy_(x)
            torch.cuda.synchronize()
        elif args.prewarm == "sumrows":
            x.to(torch.float32).sum(dim=-1)
            torch.cuda.synchronize()
        elif args.prewarm == "sumall":
            x.to(torch.float32).sum()
            torch.cuda.synchronize()
        elif args.prewarm == "softmax":
            torch.softmax(x.to(torch.float32), dim=-1)
            torch.cuda.synchronize()
        elif args.prewarm == "clamp":
            torch.clamp(x, min=0.0)
            torch.cuda.synchronize()
        elif args.prewarm == "cast":
            # Large cast output (same size class as the passing reductions).
            y = x.to(torch.float32)
            torch.cuda.synchronize()
            del y
        elif args.prewarm == "alloc":
            # Allocate the same size class a float32 cast would, no compute.
            y = torch.empty(args.batch, args.hidden, dtype=torch.float32,
                            device=device)
            torch.cuda.synchronize()
            del y
        elif args.prewarm == "cast_sum":
            # Cast (allocates 2x) followed by a row reduction: passing combo.
            x.to(torch.float32).sum(dim=-1)
            torch.cuda.synchronize()
        elif args.prewarm == "addrow":
            # Elementwise add on a float32 tensor of the cast size class.
            y = torch.empty(args.batch, args.hidden, dtype=torch.float32,
                            device=device)
            y.add_(1.0)
            torch.cuda.synchronize()
            del y
        elif args.prewarm == "to_same":
            # .to() with the same dtype: no kernel, only the Python wrapper.
            x.to(torch.bfloat16)
            torch.cuda.synchronize()
        elif args.prewarm == "to_half":
            # Narrowing cast bf16 -> fp16 (different from widening bf16 -> f32).
            x.to(torch.float16)
            torch.cuda.synchronize()
        elif args.prewarm == "copyto32":
            # bf16 -> fp32 widening cast issued through copy_ (not .to()).
            y = torch.empty(args.batch, args.hidden, dtype=torch.float32,
                            device=device)
            y.copy_(x)
            torch.cuda.synchronize()
            del y
        elif args.prewarm == "to64":
            # Even wider cast bf16 -> fp64.
            x.to(torch.float64)
            torch.cuda.synchronize()
        elif args.prewarm == "matmul":
            a = torch.randn(2048, 2048, dtype=dtype, device=device)
            for _ in range(50):
                a = a @ a.T
            torch.cuda.synchronize()
        elif args.prewarm == "sleep":
            torch.cuda._sleep(int(args.prewarm_ms * 1.5e6))
            torch.cuda.synchronize()
        print(f"[probe] prewarm={args.prewarm} done in "
              f"{time.time() - t0:.2f}s", flush=True)

    # Launch and require convergence within --hang-timeout.
    t0 = time.time()
    pk()
    if not sync_with_timeout(args.hang_timeout):
        print(f"[probe] HANG after {args.hang_timeout:.0f}s "
              f"(launch returned in {(time.time() - t0) * 1e3:.1f} ms)",
              flush=True)
        os._exit(124)
    dt = (time.time() - t0) * 1e3

    ref = torch_rmsnorm(x, w)
    d = (out - ref).abs().max().item()
    print(f"[probe] trial 0: converged in {dt:.2f} ms, "
          f"max|out-ref|={d:.6e}", flush=True)

    ok = d < 0.05
    print(f"[probe] RESULT={'PASS' if ok else 'FAIL'} "
          f"max|out-ref|={d:.6e}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
