#!/usr/bin/env python3
"""KV cache 显存账本：从模型结构算出能服务多少并发。

对应专题第 2 篇《PagedAttention：块、块表与 KV 显存账本》。

用法：
    python3 kv_budget.py                       # 默认按 Llama-3-8B / H800 算
    python3 kv_budget.py --layers 61 --kv-heads 8 --head-dim 128 \
        --weight-gb 120 --gpu-gb 640 --util 0.9
    python3 kv_budget.py --test                # 自测

只依赖标准库。它不是 vLLM 的替代实现，而是把文章里的公式做成可改参数的算盘：
    KV/token = 2 * layers * kv_heads * head_dim * dtype_bytes
    page     = block_size * KV/token
    blocks   = (gpu_gb * util - weight_gb - reserve_gb) * 2^30 // page
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

GIB = 1 << 30


@dataclass(frozen=True)
class ModelSpec:
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # bf16/fp16
    block_size: int = 16

    def kv_bytes_per_token(self) -> int:
        # K 和 V 各一份；GQA/MQA 只影响 kv_heads，不影响公式形状。
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    def page_bytes(self) -> int:
        return self.block_size * self.kv_bytes_per_token()


def blocks_for(avail_bytes: int, page_bytes: int) -> int:
    return max(0, avail_bytes // page_bytes)


def concurrency(blocks: int, seq_len: int, block_size: int) -> int:
    blocks_per_req = -(-seq_len // block_size)  # ceil
    return blocks // blocks_per_req if blocks_per_req else 0


def budget(
    spec: ModelSpec,
    gpu_gb: float,
    util: float,
    weight_gb: float,
    reserve_gb: float = 2.0,
) -> dict:
    requested = int(gpu_gb * util * GIB)
    avail = requested - int(weight_gb * GIB) - int(reserve_gb * GIB)
    page = spec.page_bytes()
    n = blocks_for(avail, page)
    return {
        "requested_bytes": requested,
        "available_bytes": avail,
        "page_bytes": page,
        "blocks": n,
        "total_kv_tokens": n * spec.block_size,
    }


def fmt(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    raise AssertionError("unreachable")


def report(spec: ModelSpec, b: dict, seq_lens: list[int]) -> None:
    print(f"每 token KV     : {fmt(spec.kv_bytes_per_token())}")
    print(f"每块（{spec.block_size} token）: {fmt(b['page_bytes'])}")
    print(f"请求显存        : {fmt(b['requested_bytes'])}")
    print(f"可用 KV 显存    : {fmt(b['available_bytes'])}")
    print(f"块数            : {b['blocks']:,}")
    print(f"KV token 总量   : {b['total_kv_tokens']:,}")
    print()
    print(f"{'序列长度':>10} {'每请求块数':>12} {'最多并发':>10}")
    for seq in seq_lens:
        per_req = -(-seq // spec.block_size)
        print(f"{seq:>10,} {per_req:>12,} {concurrency(b['blocks'], seq, spec.block_size):>10,}")


def _test() -> None:
    # Llama-3-8B：32 层、8 个 KV head、head_dim 128、bf16。
    llama = ModelSpec(layers=32, kv_heads=8, head_dim=128)
    assert llama.kv_bytes_per_token() == 131072, llama.kv_bytes_per_token()
    assert llama.page_bytes() == 2 * 1024 * 1024

    # 80 GB 卡、0.9 util、16 GB 权重、2 GB 预留 -> 约 54 GiB KV。
    b = budget(llama, gpu_gb=80, util=0.9, weight_gb=16, reserve_gb=2)
    assert b["blocks"] == 27648, b["blocks"]
    assert concurrency(b["blocks"], 8192, 16) == 54
    assert concurrency(b["blocks"], 500, 16) >= 860

    # 连续预留的对照：每个请求按 max_model_len 占满。
    contiguous = int(80 * 0.9 * GIB - 16 * GIB - 2 * GIB) // (8192 // 16 * llama.page_bytes())
    assert contiguous == 54, contiguous

    # 换 MLA：DeepSeek-V3 每层只存 512+64 的 latent，按 61 层、1 个"head"算。
    mla = ModelSpec(layers=61, kv_heads=1, head_dim=576, dtype_bytes=2)
    assert mla.kv_bytes_per_token() == 2 * 61 * 1 * 576 * 2

    # 边界：显存不够时块数为 0，而不是负数。
    assert budget(llama, gpu_gb=24, util=0.9, weight_gb=32)["blocks"] == 0
    print("kv_budget: all tests passed")


def main() -> None:
    p = argparse.ArgumentParser(description="KV cache 显存预算")
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype-bytes", type=int, default=2, choices=[1, 2])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--gpu-gb", type=float, default=80)
    p.add_argument("--util", type=float, default=0.9)
    p.add_argument("--weight-gb", type=float, default=16)
    p.add_argument("--reserve-gb", type=float, default=2.0)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if args.test:
        _test()
        return

    spec = ModelSpec(
        layers=args.layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dtype_bytes=args.dtype_bytes,
        block_size=args.block_size,
    )
    b = budget(spec, args.gpu_gb, args.util, args.weight_gb, args.reserve_gb)
    print(f"模型: layers={spec.layers} kv_heads={spec.kv_heads} "
          f"head_dim={spec.head_dim} dtype={spec.dtype_bytes}B block={spec.block_size}")
    report(spec, b, args.seq_lens)


if __name__ == "__main__":
    main()
