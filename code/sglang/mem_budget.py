#!/usr/bin/env python3
"""显存账本：SGLang 的 mem_fraction_static 自动推导 vs vLLM 的 gpu_memory_utilization。

对应专题第 3 篇（KV 池与显存账本）。两个框架都把「给 KV 多少」从手填常数
改成了自动推导，但推导的**输入**不同：

    SGLang  从 chunked_prefill_size / CUDA Graph 的 max_bs 反推「激活与图要占多少」，
            剩下的比例才给「权重 + KV 池」（arg_groups/memory_hook.py）
    vLLM    默认 0.92 / 1.0（extensible KV cache），再靠启动时一次显存 profile 定块数
            （vllm/config/cache.py 的 DEFAULT_GPU_MEMORY_UTILIZATION）

用法：
    python3 mem_budget.py                  # 8B / MLA 两种模型的账本
    python3 mem_budget.py --gpu 40 --tp 2
    python3 mem_budget.py --test
"""

from __future__ import annotations

import argparse

MB = 1  # 本文件内所有显存量统一用 MB 为单位


# ---------------------------------------------------------------------------
# 每 token KV 字节数
# ---------------------------------------------------------------------------
def kv_bytes_per_token_mha(layers: int, kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """MHA/GQA：每层存 K 和 V 两份。

    对应 memory_pool.py::MHATokenToKVPool 的张量形状
    [num_tokens, kv_heads, head_dim] × 2（K/V）× layers。
    """
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def kv_bytes_per_token_mla(
    layers: int, kv_lora_rank: int = 512, qk_rope_head_dim: int = 64, dtype_bytes: int = 2
) -> int:
    """MLA：每层只存一份 latent（c_kv）+ 一份 rope key，没有独立的 V。

    对应 memory_pool.py::MLATokenToKVPool。DeepSeek-V3 式配置：
    layers=61、kv_lora_rank=512、qk_rope_head_dim=64、bf16。
    """
    return layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes


# ---------------------------------------------------------------------------
# SGLang：mem_fraction_static 自动推导
# ---------------------------------------------------------------------------
def sglang_reserved_mb(
    gpu_mb: float,
    tp: int = 1,
    pp: int = 1,
    chunked_prefill_size: int = 8192,
    max_prefill_tokens: int = 16384,
    decode_graph_max_bs: int = 256,
    prefill_graph_bs_num: int = 58,
    is_mla: bool = False,
    deepep: bool = False,
    attn_dp_size: int = 1,
) -> tuple[float, dict[str, float]]:
    """复刻 arg_groups/memory_hook.py::handle_gpu_memory_settings 的 reserved_mem。

    单位全部是 MB。逐项来源：
      512                        恒定的元数据/对齐余量
      activation_tokens * 1.5    chunked_prefill_size（或 max_prefill_tokens）线性估计
      tp * pp / 8 * 1024         并行度越大，通信/临时缓冲越多
      decode graph max_bs * 2    每单位 batch 的图缓冲
      prefill graph            非 MLA：bs 档位数 * 8；MLA：固定 1536（fa3 开销大）
      >60GB 卡的下限 10 * 1024   给大卡留够 headroom（reserved_mem = max(reserved, 10GB)）
      DeepEP a2a                2 * 1024（decode 图里 capture 的 all-to-all 缓冲）
    """
    activation_tokens = max(chunked_prefill_size, 2048)
    if chunked_prefill_size <= 0:  # -1 表示关闭 chunked prefill
        activation_tokens = max(max_prefill_tokens, 2048)

    parts: dict[str, float] = {}
    parts["constant"] = 512.0
    parts["activation"] = activation_tokens * 1.5
    parts["parallel"] = tp * pp / 8 * 1024
    parts["decode_graph"] = decode_graph_max_bs * 2
    if attn_dp_size > 1:  # DP attention 要更多 padding
        parts["dp_attn_pad"] = decode_graph_max_bs * attn_dp_size * 3
        if decode_graph_max_bs > 300:
            parts["dp_attn_pad_large"] = decode_graph_max_bs * attn_dp_size * 1.5
    parts["prefill_graph"] = 1536.0 if is_mla else prefill_graph_bs_num * 8.0
    if deepep:
        parts["deepep_a2a"] = 2048.0

    total = sum(parts.values())
    if gpu_mb > 60 * 1024:
        if total < 10 * 1024:
            parts["large_gpu_floor"] = 10 * 1024 - total
            total = 10 * 1024
    return total, parts


def sglang_static_fraction(gpu_mb: float, **kw) -> float:
    """mem_fraction_static = (gpu - reserved) / gpu，保留三位小数。"""
    reserved, _ = sglang_reserved_mb(gpu_mb, **kw)
    return round((gpu_mb - reserved) / gpu_mb, 3)


def sglang_max_total_tokens(gpu_gib: float, weights_gib: float, bytes_per_token: int,
                            overhead_gib: float = 0.6, **kw) -> int:
    """mem_fraction_static 覆盖「权重 + KV 池」，所以 KV 预算 = fraction*gpu - 权重 - 杂项。

    单位一律用 GiB（`gpu_gib` 是标称 GiB，如 80）。`overhead_gib` 是权重之外
    非 KV 的固定占用（通信 buffer、临时张量等），memory_hook 只预留了激活水位，
    这部分仍是实打实从池子里扣的。

    简化：忽略 page 对齐与固定池（真实实现里还有 req_to_token 表、
    mamba/swa 池、expert slot 等，见 kv_cache_configurator.py）。
    """
    gpu_mib = gpu_gib * 1024
    frac = sglang_static_fraction(gpu_mib, **kw)
    kv_mib = frac * gpu_mib - (weights_gib + overhead_gib) * 1024
    return max(0, int(kv_mib * 1024 * 1024 / bytes_per_token))


# ---------------------------------------------------------------------------
# vLLM：gpu_memory_utilization → 块数
# ---------------------------------------------------------------------------
def vllm_num_blocks(
    gpu_gib: float,
    weights_gib: float,
    bytes_per_token: int,
    utilization: float = 0.92,
    block_tokens: int = 16,
    activation_gib: float = 4.0,
) -> int:
    """块数 = 可用字节 / 每块字节。

    真实 vLLM 先按 `util * 显存` 给 worker 一个上限，再用一次前向 profile 出
    「权重 + 激活峰值 + 图缓冲」的真实占用，剩下的才切成块。这里用
    `activation_gib` 一阶近似那部分（8k batched tokens 量级约 3–5 GB），
    SGLang 侧对应 memory_hook 里的 activation 预留项。
    """
    kv_gib = utilization * gpu_gib - weights_gib - activation_gib
    if kv_gib <= 0:
        return 0
    per_block = block_tokens * bytes_per_token
    return int(kv_gib * 1024**3 / per_block)


def concurrency(tokens: int, avg_ctx: int, avg_out: int = 256) -> int:
    """粗略并发：每个请求平均占 avg_ctx + avg_out/2（生成一半）个 token。"""
    return int(tokens / (avg_ctx + avg_out / 2))


# ---------------------------------------------------------------------------
# 默认场景
# ---------------------------------------------------------------------------
def scenarios(args):
    rows = []
    # 1) Llama-3-8B bf16：32 层、8 KV head、head_dim 128
    mha = kv_bytes_per_token_mha(layers=32, kv_heads=8, head_dim=128)
    weights_8b = 8.03 * 2e9 / 2**30  # 8.03B 参数 × 2 字节 = W8 GB ≈ 14.96 GiB
    frac = sglang_static_fraction(args.gpu * 1024, tp=args.tp, pp=args.pp, is_mla=False)
    s_tokens = sglang_max_total_tokens(args.gpu, weights_8b, mha, tp=args.tp, pp=args.pp)
    v_blocks = vllm_num_blocks(args.gpu, weights_8b, mha)
    rows.append(
        ("Llama-3-8B (GQA, MHA)", mha, weights_8b, frac, s_tokens, v_blocks * 16, v_blocks)
    )

    # 2) DeepSeek-V3 式 MLA：61 层、latent 512 + rope 64，权重按 fp8 1 字节算
    mla = kv_bytes_per_token_mla(layers=61)
    weights_ds = 671 * 1e9 / 2**30  # 671B × 1 字节(fp8) ≈ 625 GiB，单机放不下，只作对照
    frac_mla = sglang_static_fraction(args.gpu * 1024, tp=args.tp, pp=args.pp, is_mla=True)
    rows.append(("DeepSeek 式 MLA (每层 576 维)", mla, weights_ds, frac_mla, None, None, None))
    return rows


def report(args):
    print(
        f"GPU {args.gpu} GB × tp={args.tp} pp={args.pp}，"
        f"chunked_prefill=8192、decode graph max_bs=256\n"
    )
    rows = scenarios(args)
    hdr = f"{'模型':<26}{'KV/token':>11}{'权重GB':>9}{'mem_frac':>10}{'SGLang tok':>12}{'vLLM tok':>10}{'并发@4k':>9}"
    print(hdr)
    print("-" * (len(hdr) + 2))
    for name, bpt, w, frac, s_tok, v_tok, v_blocks in rows:
        if s_tok is None:
            print(f"{name:<26}{bpt:>10}B{w:>9.0f}{frac:>10.3f}{'放不下':>12}{'—':>10}{'—':>9}")
            continue
        print(
            f"{name:<26}{bpt:>10}B{w:>9.1f}{frac:>10.3f}{s_tok:>12,}{v_tok:>10,}"
            f"{concurrency(min(s_tok, v_tok), 4096):>9}"
        )

    # 账本明细
    reserved, parts = sglang_reserved_mb(args.gpu * 1024, tp=args.tp, pp=args.pp)
    print(f"\nSGLang reserved_mem = {reserved:.0f} MB（{(reserved / 1024):.1f} GB）")
    for k, v in parts.items():
        print(f"  {k:<18}{v:>8.0f} MB")
    print(
        f"  → mem_fraction_static = ({args.gpu * 1024:.0f} - {reserved:.0f}) / "
        f"{args.gpu * 1024:.0f} = {sglang_static_fraction(args.gpu * 1024, tp=args.tp, pp=args.pp):.3f}"
    )
    print(
        "\n读法：\n"
        "  · SGLang 的 fraction 是**反推**出来的：先决定 chunked_prefill 与 graph bs，再算余量。\n"
        "  · vLLM 的 utilization 是**正着**给的（默认 0.92），块数由启动时的 profile 校准。\n"
        "  · 同样 80GB 卡、8B 模型，两者 KV 池容量差在几个百分点（差在各自给激活/图留多少）。\n"
        "  · 真正决定并发的不是框架，是 KV/token 与平均上下文长度之比。"
    )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
def _test():
    # 1) 经典数字：Llama-3-8B 的 KV = 2*32*8*128*2 = 131072 B = 128 KiB/token
    assert kv_bytes_per_token_mha(32, 8, 128) == 131072
    # 单 token 1.5 层... 反例：TP 分片后每卡只存一部分，这里是逻辑总量，不分片
    assert kv_bytes_per_token_mha(32, 8, 128, dtype_bytes=1) == 65536

    # 2) MLA：61 层 × 576 维 × 2 字节
    assert kv_bytes_per_token_mla(61) == 61 * 576 * 2

    # 3) reserved 单调性：chunked_prefill 越大，预留越多
    r1, _ = sglang_reserved_mb(80 * 1024, chunked_prefill_size=2048)
    r2, _ = sglang_reserved_mb(80 * 1024, chunked_prefill_size=16384)
    assert r2 > r1

    # 4) 大卡有 10GB 下限
    r_small, _ = sglang_reserved_mb(40 * 1024, chunked_prefill_size=2048, decode_graph_max_bs=8)
    r_big, _ = sglang_reserved_mb(80 * 1024, chunked_prefill_size=2048, decode_graph_max_bs=8)
    assert r_big >= 10 * 1024 and r_small < 10 * 1024

    # 5) fraction 落在合理区间，且随 tp 增大而减小
    f1 = sglang_static_fraction(80 * 1024, tp=1)
    f8 = sglang_static_fraction(80 * 1024, tp=8)
    assert 0.5 < f8 < f1 < 1.0, (f1, f8)

    # 6) MLA 的 prefill graph 预留是固定 1536，与 bs 档位数无关
    _, p_a = sglang_reserved_mb(80 * 1024, is_mla=True, prefill_graph_bs_num=1)
    _, p_b = sglang_reserved_mb(80 * 1024, is_mla=True, prefill_graph_bs_num=64)
    assert p_a["prefill_graph"] == p_b["prefill_graph"] == 1536

    # 7) vLLM：块数 × 16 ≤ 容量；utilization 越小块数越少；放不下时为 0
    W8 = 8.03 * 2e9 / 2**30  # 14.96 GiB
    b = kv_bytes_per_token_mha(32, 8, 128)
    n1 = vllm_num_blocks(80, W8, b, utilization=0.92)
    n2 = vllm_num_blocks(80, W8, b, utilization=0.8)
    assert 0 < n2 < n1
    assert vllm_num_blocks(80, W8, b) * 16 > 420_000  # 8B 在 80GB 上能装 40 万+ token
    assert vllm_num_blocks(20, 30, b) == 0
    # activation 预留越大，块数越少（对应 profile 出来的激活峰值）
    assert vllm_num_blocks(80, W8, b, activation_gib=8) < vllm_num_blocks(80, W8, b, activation_gib=2)

    # 8) 并发随上下文变长而下降
    t = sglang_max_total_tokens(80, W8, b)
    assert concurrency(t, 8192) < concurrency(t, 1024)

    print("mem_budget: all tests passed")
    print(f"  8B MHA   : {b} B/token = {b / 1024:.0f} KiB")
    print(f"  MLA(61L) : {kv_bytes_per_token_mla(61)} B/token = {kv_bytes_per_token_mla(61) / 1024:.0f} KiB")
    print(f"  80GB/8B  : SGLang {sglang_max_total_tokens(80, W8, b):,} tok, "
          f"vLLM {vllm_num_blocks(80, W8, b) * 16:,} tok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=float, default=80.0, help="单卡显存 GB")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        _test()
    else:
        report(args)


if __name__ == "__main__":
    main()
