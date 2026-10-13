"""P7-76 `ncu` driver: attribute bf16-native TMA staging vs bf16 cp.async on the base GEMM core.

Loads a prebuilt `.so` (ncu builds must be done *outside* ncu -- nvcc hangs under ncu here) and
launches exactly one kernel so the profile holds a single launch.

    python .../ncu_b2_bf16_tma_driver.py --mode bftma64 --shape b8t2048h8 --reps 1
"""
import argparse
import ctypes
import sys

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

SHAPES = {"b8t2048h8": (8, 2048, 8), "t512h4": (1, 512, 4), "b2t1024h8": (2, 1024, 8)}
# mode -> (so path, input dtype)
MODES = {
    "base64": ("/tmp/b2_bftma_base64.so", torch.float32),
    "tma64": ("/tmp/b2_bftma_tma64.so", torch.float32),
    "bf16_64": ("/tmp/b2_bftma_bf16_64.so", torch.bfloat16),
    "bftma64": ("/tmp/b2_bftma_bftma64.so", torch.bfloat16),
}
BT = 16


def make_core(B, T, H, dt):
    dev = "cuda"
    K = V = 128
    g = torch.Generator(device=dev).manual_seed(0)
    NT = (T + BT - 1) // BT
    v = torch.randn(B, T, H, V, device=dev, generator=g).to(dt)
    vn = torch.randn(B, T, H, V, device=dev, generator=g).to(dt)
    do = torch.randn(B, T, H, V, device=dev, generator=g).to(dt)
    dv = torch.randn(B, T, H, V, device=dev, generator=g).to(dt)
    h = torch.randn(B, NT, H, K, V, device=dev, generator=g).to(dt)
    dh = torch.randn(B, NT, H, K, V, device=dev, generator=g).to(dt)
    return dict(v=v, vn=vn, do=do, dv=dv, h=h, dh=dh)


def _run(lib, x, B, T, H, reps):
    NT = (T + BT - 1) // BT
    dq = torch.empty(B, T, H, 128, device="cuda")
    dk = torch.empty(B, T, H, 128, device="cuda")
    dw = torch.empty(B, T, H, 128, device="cuda")
    dgk = torch.empty(B, NT, H, 128, device="cuda")

    def p(t):
        return ctypes.c_void_p(t.data_ptr())
    fn = lib.b2mma_launch
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(reps):
        rc = fn(p(x["v"]), p(x["vn"]), p(x["h"]), p(x["dh"]),
                p(x["do"]), p(x["dv"]), p(dq), p(dk), p(dw), p(dgk),
                B, T, H, NT, stream)
        assert rc == 0, rc
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), default="bftma64")
    ap.add_argument("--shape", default="b8t2048h8")
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    B, T, H = SHAPES[a.shape]
    so, dt = MODES[a.mode]
    _run(ctypes.CDLL(so), make_core(B, T, H, dt), B, T, H, a.reps)


if __name__ == "__main__":
    main()
