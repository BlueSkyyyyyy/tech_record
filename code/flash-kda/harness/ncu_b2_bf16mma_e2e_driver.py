"""P7-79 `ncu` driver: profile the **bf16 folded production** CUDA B2 launch and its native
bf16-mma (`-DB2M_BF16MMA`) / mma-DA (`-DB2M_DAMMA`) variants on identical operands, to see what
the ~1-2% chain win of `pcibm`/`pcibmd` over `pcib` (docs §82 Part B) corresponds to at the pipe
level.

Dependency-free (torch + ctypes) so the profile holds exactly one kernel; the `.so` must be
pre-built *outside* ncu (nvcc hangs under ncu in this container, P7-16).  Operands are fed at the
**dtype the `-DB2M_BF16` kernel expects**: the five streamed operands `do/vn/dv/h/dh` are bf16
(`rd()`), the fp32 scratch `vv`(raw `v`) / `dwm`(`dw`) stay fp32 (`rds()`).

    B2BM_SO=/tmp/b2_bf16mma_b_fold.so B2BM_BF16=1 python .../ncu_b2_bf16mma_e2e_driver.py \
        --shape b8t2048h8
"""
import argparse
import ctypes
import os
import sys

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

HARNESS = "~/proj/tech_record/code/flash-kda/harness"
sys.path.insert(0, HARNESS)

SHAPES = {"b8t2048h8": (8, 2048, 8), "b2t1024h8": (2, 1024, 8), "t512h4": (1, 512, 4),
          "t250h3": (1, 250, 3)}
BT = 16
SCALE = 1.0 / (128 ** 0.5)


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="b8t2048h8", choices=list(SHAPES))
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    B, T, H = SHAPES[args.shape]
    dev = "cuda"
    K = V = 128
    NT = (T + BT - 1) // BT
    bf16 = bool(int(os.environ.get("B2BM_BF16", "0")))
    streamed = torch.bfloat16 if bf16 else torch.float32
    gg = torch.Generator(device=dev).manual_seed(0)
    mk = lambda *s: torch.empty(*s, device=dev)
    # fp32 scratch (raw `v` = vv, `dw` = dwm) -- always fp32
    v = torch.randn(B, T, H, V, device=dev, generator=gg)
    dw = torch.randn(B, T, H, K, device=dev, generator=gg)
    # five streamed operands (bf16 under -DB2M_BF16)
    vn = torch.randn(B, T, H, V, device=dev, generator=gg).to(streamed)
    do = torch.randn(B, T, H, V, device=dev, generator=gg).to(streamed)
    dv = torch.randn(B, T, H, V, device=dev, generator=gg).to(streamed)
    h = torch.randn(B, NT, H, K, V, device=dev, generator=gg).to(streamed)
    dh = torch.randn(B, NT, H, K, V, device=dev, generator=gg).to(streamed)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=gg), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=gg), dim=-1)
    g = -torch.rand(B, T, H, K, device=dev, generator=gg)
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev, generator=gg))
    A = torch.randn(B, T, H, BT, device=dev, generator=gg) * 0.1
    o_t = torch.arange(BT, device=dev)
    dAqk = (torch.randn(B, NT, H, BT, BT, device=dev, generator=gg) *
            (o_t[:, None] >= o_t[None, :]).float())
    dAqk = dAqk.permute(0, 1, 3, 2, 4).reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    dq, dk, dwo = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    dgk = mk(B, NT, H, K)
    dv2, dvb = mk(B, T, H, V), mk(B, T, H, V)
    dkgb = mk(B, T, H, K)
    dqg, dkg, dg = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, K)
    db = torch.zeros(B, T, H, device=dev)
    dApre, dA = mk(B, T, H, BT), mk(B, T, H, BT)

    lib = ctypes.CDLL(os.environ["B2BM_SO"])
    fn = lib.b2prod_intra_launch
    fn.restype = ctypes.c_int
    fn.argtypes = ([ctypes.c_void_p] * 24 + [ctypes.c_float] + [ctypes.c_void_p] * 5 +
                   [ctypes.c_int] * 4 + [ctypes.c_void_p])
    st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(args.reps):
        rc = fn(ptr(v), ptr(vn), ptr(h), ptr(dh), ptr(do), ptr(dv), ptr(k), ptr(g), ptr(dw),
                ptr(dq), ptr(dk), ptr(dwo), ptr(dgk), ptr(A), ptr(beta), ptr(dv2), ptr(dvb),
                ptr(dkgb), ptr(q), ptr(k), ptr(g), ptr(dqg), ptr(dkg), ptr(dg), SCALE,
                ptr(db), ptr(dApre), ptr(A), ptr(dA), ptr(dAqk), B, T, H, NT, st)
        if rc != 0:
            raise SystemExit(f"b2prod_intra_launch rc={rc}")
    torch.cuda.synchronize()
    print(f"b2prod_intra_launch ok  shape={args.shape} bf16={bf16} reps={args.reps}")


if __name__ == "__main__":
    main()
