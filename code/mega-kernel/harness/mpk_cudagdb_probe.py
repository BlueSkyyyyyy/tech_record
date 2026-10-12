"""Launch the minimal MPK single-layer RMSNorm megakernel and, on the *plain*
(buggy) build, keep the process alive so an external debugger (cuda-gdb) can
attach and inspect the device-side state of the hung worker/scheduler kernels.

This is the "candidate (1)" of docs/09 section 10.7 -- device-side event-level
evidence -- implemented with cuda-gdb instead of in-kernel printf, so it does
NOT perturb the timing-sensitive race the way instrumentation does (docs/09
section 9.3/9.4).

Env:
  MIRAGE_HOME  path to the mirage checkout (default /ssd/mirage_mpk_build)

Run inside kernel_lab:
  MIRAGE_HOME=/ssd/mirage_mpk_build CUDA_VISIBLE_DEVICES=7 \
    python3 -u mpk_cudagdb_probe.py --workers 128 --schedulers 16
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--schedulers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--prewarm", type=str, default="none",
                    choices=["none", "cast_1elem"])
    ap.add_argument("--alive-after", type=float, default=300.0,
                    help="seconds to keep the process alive after launch so a "
                         "debugger can attach; prints a marker first")
    ap.add_argument("--device", type=int, default=7)
    ap.add_argument("--output-dir", type=str,
                    default=os.environ.get("MIRAGE_HOME", "."))
    args = ap.parse_args()

    # P28-1: the JIT-compiled launcher .so is built inside a
    # tempfile.TemporaryDirectory() that mirage deletes as soon as compile()
    # returns, so cuda-gdb cannot map its (debug) symbols on attach. With
    # MPK_KEEP_TEMPDIR=1 we keep that dir alive so the -G build can be
    # inspected. Set before PersistentKernel.compile().
    if os.environ.get("MPK_KEEP_TEMPDIR") == "1":
        import tempfile as _tf

        class _KeepTempDir(_tf.TemporaryDirectory):
            def cleanup(self):
                import sys as _sys
                print(f"[probe] keeping tempdir {self.name}", file=_sys.stderr, flush=True)

        _tf.TemporaryDirectory = _KeepTempDir

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

    if args.prewarm == "cast_1elem":
        z = torch.zeros(1, dtype=dtype, device=device).to(torch.float32)
        torch.cuda.synchronize()
        del z
        print("[probe] prewarm=cast_1elem", flush=True)

    t0 = time.time()
    pk()
    print(f"[probe] pk() returned in {(time.time() - t0) * 1e3:.1f} ms", flush=True)

    print(f"[probe] LAUNCHED pid={os.getpid()} alive={args.alive_after:.0f}s "
          f"prewarm={args.prewarm}", flush=True)

    # Keep the CUDA context and both persistent kernels resident. Poll
    # convergence in a daemon thread only for reporting; do NOT exit so the
    # debugger can attach to a live GPU context.
    done = {"ok": False}

    def waiter():
        torch.cuda.synchronize()
        done["ok"] = True

    threading.Thread(target=waiter, daemon=True).start()

    deadline = time.time() + args.alive_after
    while time.time() < deadline:
        if done["ok"]:
            print("[probe] CONVERGED (unexpected for the plain build)", flush=True)
            break
        time.sleep(1.0)
    else:
        print("[probe] STILL HUNG at deadline", flush=True)

    sys.exit(0 if done["ok"] else 124)


if __name__ == "__main__":
    main()
