import argparse
import torch
from transformer_engine_torch import rmsnorm_fwd, rmsnorm_bwd

TE_DType = {
    torch.float32: 0,
    torch.bfloat16: 1,
    torch.float16: 2,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--bwd", action="store_true", help="profile rmsnorm_bwd instead of fwd")
    args = ap.parse_args()

    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    rows, cols = args.rows, args.cols
    device = "cuda"

    x = torch.randn(rows, cols, device=device, dtype=dt)
    w = torch.randn(cols, device=device, dtype=dt)

    if not args.bwd:
        op = lambda: rmsnorm_fwd(x, w, 1e-5, None, None, TE_DType[dt], 0, False)
    else:
        _, _, rsigma = rmsnorm_fwd(x, w, 1e-5, None, None, TE_DType[dt], 0, False)
        dy = torch.randn(rows, cols, device=device, dtype=dt)
        op = lambda: rmsnorm_bwd(dy, x, rsigma, w, 0, False)

    # warmup
    for _ in range(3):
        op()
    torch.cuda.synchronize()

    for _ in range(args.iters):
        op()
    torch.cuda.synchronize()
    print(f"done ({rows}x{cols} {args.dtype} {'bwd' if args.bwd else 'fwd'})")


if __name__ == "__main__":
    main()
