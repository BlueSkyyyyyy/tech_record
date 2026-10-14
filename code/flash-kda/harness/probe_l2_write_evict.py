"""P7-111: driver for `harness/probe_l2_write_evict.cu` -- a self-contained
microbenchmark of the *write path* that isolates §99.5 / §102.4's open question:

    why does `dram__bytes_write` move (+35 MB bf16-vs-fp32 at V=256; ~19 MB
    across ncu replays) while the store instructions / L1 store sectors are
    bitwise identical?

Hypothesis (see the .cu header): inside one ncu window `dram__bytes_write` is
"the fraction of dirty L2 lines flushed before the window closes", set by L2
eviction pressure (a concurrent read stream) and by whether the output fits in
L2 -- *not* the output size.  Four read/write orderings, output `--outmb` and
read `--readmb` footprints swept from here.

`ncu` collection lives in `harness/probe_l2_write_evict_ncu.sh`
(`harness/probe_l2_write_evict_ncu.out.txt`).

Run inside `kernel_lab`:
    CUDA_VISIBLE_DEVICES=1 python .../harness/probe_l2_write_evict.py \
        --mode 0 --outmb 8 --readmb 0
"""
import argparse
import ctypes
import os
import subprocess
import sys

import torch

HARNESS = os.path.dirname(os.path.abspath(__file__))
CU = os.path.join(HARNESS, "probe_l2_write_evict.cu")

MODES = {0: "store-only", 1: "read-then-store", 2: "store-then-read",
         3: "interleaved store/read"}


def build(so="/tmp/probe_l2_write_evict.so"):
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", CU, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit("nvcc failed")
    return so


def load(so):
    lib = ctypes.CDLL(so)
    lib.probe_launch.restype = ctypes.c_int
    lib.probe_launch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_long, ctypes.c_long, ctypes.c_void_p,
                                 ctypes.c_int, ctypes.c_int]
    return lib


def run_once(lib, mode, rbuf, out, nout, nread, sink, grid, block):
    return lib.probe_launch(mode, ctypes.c_void_p(rbuf.data_ptr()),
                            ctypes.c_void_p(out.data_ptr()), nout, nread,
                            ctypes.c_void_p(sink.data_ptr()), grid, block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--outmb", type=float, default=8.0)
    ap.add_argument("--readmb", type=float, default=0.0)
    ap.add_argument("--grid", type=int, default=264)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-build", action="store_true",
                    help="use existing /tmp/probe_l2_write_evict.so (ncu: nvcc hangs "
                         "under ncu, so prebuild outside)")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    torch.manual_seed(0)
    so = "/tmp/probe_l2_write_evict.so"
    if not (args.no_build and os.path.exists(so)):
        so = build(so)
    lib = load(so)

    MB = 1024 * 1024
    nout = max(16, (int(args.outmb * MB) // 4 // 16) * 16)   # args.outmb is MiB
    nread = max(0, (int(args.readmb * MB) // 4 // 16) * 16)
    rbuf = torch.randn(max(nread, 4), device="cuda", dtype=torch.float32)
    out = torch.zeros(nout, device="cuda", dtype=torch.float32)
    sink = torch.zeros(args.grid * args.block, device="cuda", dtype=torch.float32)

    rc = run_once(lib, args.mode, rbuf, out, nout, nread, sink, args.grid, args.block)
    if rc != 0:
        raise SystemExit(f"probe_launch rc={rc} (mode={args.mode})")
    torch.cuda.synchronize()
    oss = float(out.sum().item())

    for _ in range(3):
        run_once(lib, args.mode, rbuf, out, nout, nread, sink, args.grid, args.block)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(args.reps):
        run_once(lib, args.mode, rbuf, out, nout, nread, sink, args.grid, args.block)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / args.reps

    print(f"mode={args.mode}({MODES[args.mode]}) outMB={args.outmb} readMB={args.readmb} "
          f"grid={args.grid} block={args.block} nout={nout} nread={nread}")
    print(f"  out.sum={oss:.4e}  wall={ms*1000:.1f} us")


if __name__ == "__main__":
    main()
