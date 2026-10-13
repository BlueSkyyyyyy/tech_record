"""Standalone driver for MPK's Hopper CuTe weight-stationary GEMM task body.

Builds tests/runtime_python/cute/hopper/gemm.cu (the `linear_mpk` entry, which
calls kernel::linear_cutlass_ws_hopper from
tasks/cute/hopper/gemm_ws_mpk.cuh) with the *correct* include dirs, then runs
the two configurations the wrapper instantiates (OUTPUT_SIZE=64,
BATCH_SIZE in {8,16}, REDUCTION_SIZE=4096) and checks correctness vs a torch
reference, plus an event-timed benchmark.

The as-shipped setup.py of that test omits
`include/mirage/persistent_kernel` from include_dirs, so it fails to compile
(runtime_header.h not found); we supply the include path explicitly instead of
patching the snapshot.
"""
import os
import sys
import time
from contextlib import contextmanager

import torch
from torch.utils.cpp_extension import load


@contextmanager
def _silence_fd1():
    """The snapshot's wrapper printf()s a banner on *every* launch; hide it."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)

SNAP = os.environ.get("MIRAGE_SNAP", "/ssd/mirage_mpk_build")
HOPPER_DIR = os.path.join(SNAP, "tests/runtime_python/cute/hopper")
TASKS_HOPPER = os.path.join(
    SNAP, "include/mirage/persistent_kernel/tasks/cute/hopper")
PK_INCLUDE = os.path.join(SNAP, "include/mirage/persistent_kernel")
CUTLASS_INCLUDE = os.path.join(SNAP, "deps/cutlass/include")
CUTLASS_UTIL = os.path.join(SNAP, "deps/cutlass/tools/util/include")

dev = torch.device("cuda")
torch.cuda.set_device(0)
print(f"[info] cuda={torch.version.cuda} torch={torch.__version__} "
      f"snap={SNAP}")

t0 = time.time()
mod = load(
    name="mpk_cute_hopper_ws",
    sources=[os.path.join(HOPPER_DIR, "gemm.cu")],
    extra_include_paths=[
        HOPPER_DIR,
        TASKS_HOPPER,
        PK_INCLUDE,
        os.path.join(SNAP, "include"),
        CUTLASS_INCLUDE,
        CUTLASS_UTIL,
    ],
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-DMIRAGE_GRACE_HOPPER",
        "-DMIRAGE_PROFILE_HOPPER",
        "-DMIRAGE_BACKEND_USE_CUDA",
    ],
    verbose=False,
    extra_ldflags=["-lcuda", "-L/usr/local/cuda/lib64/stubs"],
)
print(f"[info] build done in {time.time()-t0:.1f}s")

REDUCTION = 4096
OUTPUT = 64


def run_case(batch):
    g = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.randn((batch, REDUCTION), device="cuda", dtype=torch.bfloat16,
                    generator=g)
    weight = torch.randn((OUTPUT, REDUCTION), device="cuda",
                         dtype=torch.bfloat16, generator=g)
    residual = torch.randn((batch, OUTPUT), device="cuda",
                           dtype=torch.bfloat16, generator=g)
    output = torch.empty((batch, OUTPUT), device="cuda",
                         dtype=torch.bfloat16)

    with _silence_fd1():
        mod.linear_mpk(weight, x, residual, output)
    torch.cuda.synchronize()

    ref = (x.float() @ weight.float().T + residual.float())
    diff = (output.float() - ref).abs()
    rel = diff.max().item() / max(ref.abs().max().item(), 1e-6)
    ok = torch.allclose(output.float(), ref, atol=1e-2, rtol=1e-2)
    print(f"[correctness] B={batch} max_abs_diff={diff.max().item():.5f} "
          f"max_rel={rel:.2e} allclose={ok}")

    with _silence_fd1():
        for _ in range(20):
            mod.linear_mpk(weight, x, residual, output)
        torch.cuda.synchronize()
        iters = 200
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            mod.linear_mpk(weight, x, residual, output)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
    flops = 2.0 * batch * OUTPUT * REDUCTION
    gbps = (batch * REDUCTION + OUTPUT * REDUCTION) * 2 / (ms * 1e-3) / 1e9
    print(f"[perf] B={batch} O={OUTPUT} K={REDUCTION} avg_ms={ms:.4f} "
          f"tflops={flops/(ms*1e-3)/1e12:.4f} read_GBps={gbps:.1f}")


for b in (8, 16):
    run_case(b)
print("[done]")
