"""P7-62: minimal `ncu` driver for the K2-shape CUDA B3 `intra` (`b3_intra_probe.cu`).

Launches exactly one CUDA kernel (`b3intra_kernel`) on synthesized operands so
`ncu -k regex:b3intra_kernel` targets it unambiguously.  `--which ours` runs the CUDA
probe; `--which triton` runs the Triton MMA B3 (`_intra_kernel_mma`) as the reference.

Usage (inside kernel_lab):
    python ncu_b3_intra_cuda_driver.py --which ours --B 8 --T 2048 --H 8
    ncu -k regex:b3intra_kernel -c 1 --launch-skip 3 --set full \\
        python ncu_b3_intra_cuda_driver.py --which ours --B 8 --T 2048 --H 8
"""
import argparse
import ctypes
import os
import subprocess
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

D = 128
HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

SO = "/tmp/b3_intra.so"


def build():
    src = os.path.join(HARNESS, "b3_intra_probe.cu")
    cmd = ["nvcc", "-std=c++17", "-O3", "-arch=sm_90a", "-Xcompiler", "-fPIC",
           "-shared", src, "-o", SO]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise SystemExit("nvcc failed")


def load():
    lib = ctypes.CDLL(SO)
    lib.b3intra_launch.restype = ctypes.c_int
    lib.b3intra_launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_void_p] * 4 + \
        [ctypes.c_int] * 4 + [ctypes.c_void_p]
    return lib


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def synth(B, T, H, bt, dev):
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=dev), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=dev), dim=-1)
    g = -torch.rand(B, T, H, D, device=dev) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev))
    dAqk = torch.randn(B, T, H, bt, device=dev)
    dAkk = torch.randn(B, T, H, bt, device=dev)
    return dict(q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk)


def cdiv(a, b):
    return (a + b - 1) // b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["ours", "triton"], required=True)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--bt", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof", type=int, default=1)
    ap.add_argument("--no-build", action="store_true",
                    help="reuse an existing %s (nvcc can hang when run under ncu)" % SO)
    args = ap.parse_args()

    inp = synth(args.B, args.T, args.H, args.bt, "cuda")
    B, T, H, K = inp["q"].shape
    NT = cdiv(T, args.bt)
    dev = inp["q"].device

    if args.which == "ours":
        if not args.no_build:
            build()
        lib = load()
        dq = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
        dk = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
        db = torch.empty(B, T, H, device=dev, dtype=torch.float32)
        dg = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        def call():
            rc = lib.b3intra_launch(
                ptr(inp["q"]), ptr(inp["k"]), ptr(inp["g"]), ptr(inp["beta"]),
                ptr(inp["dAqk"]), ptr(inp["dAkk"]), ptr(dq), ptr(dk), ptr(db), ptr(dg),
                B, T, H, NT, stream)
            assert rc == 0, rc
    else:
        from check_bwd_b3_mma import build_intra_mma, intra_triton_mma
        k = build_intra_mma()
        call = lambda: intra_triton_mma(k, inp, args.bt, args.bt, num_warps=args.nw)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.prof):
        call()
    torch.cuda.synchronize()
    print(f"DONE which={args.which} B={B} T={T} H={H} bt={args.bt} nw={args.nw}")


if __name__ == "__main__":
    main()
