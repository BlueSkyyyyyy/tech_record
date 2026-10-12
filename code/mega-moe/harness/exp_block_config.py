#!/usr/bin/env python3
"""Mega MoE block/config 启发式的纯 Python 复刻 + 与 DeepGEMM 真实 binding 对拍。

背景（见 docs/03-mega-moe-kernel.md §2）：
  `get_block_config_for_mega_moe` 是**纯 host 启发式**（csrc/jit_kernels/heuristics/mega_moe.hpp:88），
  不依赖 SM100 硬件；DeepGEMM 通过 pybind 暴露了 `deep_gemm.get_block_m_for_mega_moe`。
  因此虽然端到端 kernel 是 SM100 专属，**block_m 的选择逻辑在 H100 上可以真机对拍**。

本脚本做两件事：
  1. 用 Python 按源码逐行复刻 block_m / store_block_m / block_k 的选取；
  2. 在 H100 上调用真实 `deep_gemm.get_block_m_for_mega_moe` 逐点比对（应 100% 一致）；
  3. 附带复刻 `get_pipeline_config_for_mega_moe`（smem 预算 → num_stages），打印配置表。

只依赖 numpy / math，不需要 GPU；在 `kernel_lab`（H100）里跑以证明 binding 可用。
"""

import math
import sys


def _align(x, a):
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------
# 复刻 heuristics/mega_moe.hpp:88-127 `get_block_config_for_mega_moe`
# ---------------------------------------------------------------------------
def block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank,
                              num_topk, num_tokens, mma_kind):
    # :94 期望每专家 token 数 + 一个 sigma 的路由余量
    num_expected_tokens = float(num_tokens) * num_ranks * num_topk / num_experts
    num_covered_tokens = num_expected_tokens + math.sqrt(num_expected_tokens)

    # :97-110 每个 M block 大约多付 128 行的权重重载，尽量少 block、小 tile
    block_m = 16 if num_expected_tokens <= 10 else 32
    if num_expected_tokens > 24:
        num_blocks = int(math.ceil(num_covered_tokens / 240))
        if num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14:
            num_blocks = 2
        for candidate in (64, 128, 192, 240):
            block_m = candidate
            if num_blocks * candidate >= num_covered_tokens:
                break

    # :112-115 非对称 store block 省 smem
    fp8_fp4 = mma_kind == "fp8xfp4"
    if block_m <= 16:
        store_l1 = 8
    elif block_m <= 64:
        store_l1 = 16
    elif block_m <= 192:
        store_l1 = 32
    else:
        store_l1 = 24 if fp8_fp4 else 40
    store_l2 = 8 if (block_m == 240 and fp8_fp4) else store_l1

    # :117 block_k = 128 / elem_bytes
    elem_bytes = 2 if mma_kind == "bf16xbf16" else 1
    block_k = 128 // elem_bytes

    # :125-126 cluster=2，epilogue threads = 2*128
    return dict(cluster=2, block_m=block_m, store_l1=store_l1, store_l2=store_l2,
                block_k=block_k, num_epilogue_threads=256, block_n=128)


# ---------------------------------------------------------------------------
# 复刻 heuristics/mega_moe.hpp:129-195 `get_pipeline_config_for_mega_moe`
# ---------------------------------------------------------------------------
def pipeline_config(smem_capacity, num_experts, hidden, block_m, block_n, block_k,
                    num_bytes_per_pull, store_l1, store_l2, sf_block_m, sf_block_n,
                    gran_k, num_dispatch_warps, num_epilogue_warps, mma_kind):
    KB = 1024
    num_elem = 2 if mma_kind == "bf16xbf16" else 1
    with_sf = mma_kind != "bf16xbf16"

    load_block_m = block_m // 2

    smem_expert_count = _align(num_experts * 4, KB)
    smem_send = _align(num_bytes_per_pull * num_dispatch_warps, KB)
    smem_dispatch = smem_expert_count + smem_send

    epi_wg = num_epilogue_warps // 4
    # :155-157 L1 输出 FP8 (block_n/2)，L2 输出 BF16
    smem_cd_l1 = epi_wg * store_l1 * (block_n // 2) * 2 * num_elem
    smem_cd_l2 = epi_wg * store_l2 * block_n * 2
    smem_cd = _align(max(smem_cd_l1, smem_cd_l2), KB)

    # TaskInfo<true> 对齐到 16B，8 个 uint32 字段 => 32B
    smem_task_info = 2 * 32
    # :165 dispatch + tmem full/empty(2*2) + combine(2/epi warp) + schedule(2*2)
    smem_barriers = (num_dispatch_warps + 2 * 2 + num_epilogue_warps * 2 + 2 * 2) * 8
    smem_amax = store_l1 * num_epilogue_warps * 4 if with_sf else 0
    smem_tmem_ptr = 4

    smem_sfa = sf_block_m * (block_k // gran_k) if with_sf else 0
    smem_sfb = sf_block_n * (block_k // gran_k) if with_sf else 0
    smem_a = load_block_m * block_k * num_elem
    smem_b = block_n * block_k * num_elem
    smem_per_stage = smem_a + smem_b + smem_sfa + smem_sfb + 2 * 8

    smem_fixed = smem_dispatch + smem_cd + smem_amax + smem_barriers + smem_task_info + smem_tmem_ptr
    num_stages = (smem_capacity - smem_fixed) // smem_per_stage
    return dict(num_stages=num_stages, smem_fixed=smem_fixed, smem_per_stage=smem_per_stage,
                smem_total=smem_fixed + num_stages * smem_per_stage)


def full_config(num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk,
                hidden, intermediate_hidden, mma_kind, num_dispatch_warps=4,
                num_epilogue_warps=8, smem_capacity=232448):
    bc = block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank,
                                   num_topk, num_tokens, mma_kind)
    block_m, block_n, block_k = bc["block_m"], bc["block_n"], bc["block_k"]
    num_elem = 2 if mma_kind == "bf16xbf16" else 1
    # :223-229 pull bytes: hidden*elem 反复减半直到 <= 8192(fp8)/4096(bf16)
    limit = 8192 if mma_kind != "bf16xbf16" else 4096
    nbpull = hidden * num_elem
    while nbpull > limit:
        nbpull //= 2
    sf_block_m = _align(block_m, 128) if mma_kind != "bf16xbf16" else 0
    sf_block_n = _align(block_n, 128) if mma_kind != "bf16xbf16" else 0
    pc = pipeline_config(smem_capacity, num_experts, hidden, block_m, block_n, block_k,
                         nbpull, bc["store_l1"], bc["store_l2"], sf_block_m, sf_block_n,
                         32, num_dispatch_warps, num_epilogue_warps, mma_kind)
    return {**bc, "num_bytes_per_pull": nbpull, "sf_block_m": sf_block_m,
            "sf_block_n": sf_block_n, **pc}


def main():
    try:
        import deep_gemm
        real = deep_gemm.get_block_m_for_mega_moe
        has_real = True
    except Exception as e:  # pragma: no cover
        print(f"[warn] 无法 import deep_gemm，仅打印复刻结果: {e}")
        has_real = False

    # 复刻 DeepSeek-V3 量级的典型形状（+ 若干极端）
    cases = [
        # (num_ranks, num_experts, max_tokens/rank, num_tokens, num_topk, mma_kind)
        (8, 256, 4096, 4096, 8, "fp8xfp4"),
        (8, 256, 4096, 512, 8, "fp8xfp4"),
        (8, 256, 4096, 128, 8, "fp8xfp4"),
        (8, 256, 4096, 32, 8, "fp8xfp4"),
        (8, 256, 4096, 8, 8, "fp8xfp4"),
        (8, 64, 4096, 4096, 8, "fp8xfp4"),
        (2, 16, 2048, 4096, 4, "fp8xfp8"),
        (8, 256, 4096, 4096, 8, "bf16xbf16"),
        (32, 256, 4096, 4096, 8, "fp8xfp4"),
    ]

    print(f"{'ranks':>5} {'exp':>4} {'tok':>6} {'topk':>4} {'mma':>10} "
          f"{'blkM':>5} {'stL1':>4} {'stL2':>4} {'blkK':>5} {'pull':>6} "
          f"{'stages':>6} {'smemKiB':>8} {'real_match':>10}")
    all_match = True
    for (nr, ne, mt, tok, k, mma) in cases:
        c = full_config(nr, ne, mt, tok, k, 7168, 2048, mma)
        match = "-"
        if has_real:
            got = real(nr, ne, mt, tok, k, mma)
            match = "OK" if got == c["block_m"] else f"MISMATCH({got})"
            if got != c["block_m"]:
                all_match = False
        print(f"{nr:>5} {ne:>4} {tok:>6} {k:>4} {mma:>10} "
              f"{c['block_m']:>5} {c['store_l1']:>4} {c['store_l2']:>4} {c['block_k']:>5} "
              f"{c['num_bytes_per_pull']:>6} {c['num_stages']:>6} "
              f"{c['smem_total']/1024:>8.1f} {match:>10}")

    if has_real:
        print(f"\n复刻 vs DeepGEMM binding：{'全部一致' if all_match else '存在不一致'}")
        return 0 if all_match else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
