"""P10-1 实验：BF16 Mega MoE 的 host 侧配置与布局语义（H100 可跑部分）。

BF16 Mega MoE（`csrc/apis/mega_moe.hpp:283-369` / `sm100_bf16_mega_moe.cuh`）主体是 SM100 专属，
但以下三件事是**架构无关**的 host 侧逻辑，可以在 H100 上直接调用 DeepGEMM 并核对：

  1) block config 启发式 `get_block_config_for_mega_moe`（`heuristics/mega_moe.hpp:62-105`）：
     block_m 与 dtype 无关；block_k = 128/元素字节（BF16=64、FP8/FP4=128），
     store_block_m 在 MXFP8FP4 与 BF16 间略有不同。用 `deep_gemm.get_block_m_for_mega_moe` 对拍。
  2) symmetric buffer 尺寸 `get_symm_buffer_size_for_mega_moe`（`mega_moe.hpp:38-...`）：
     BF16 走 `"bf16xbf16"` 时 with_sf=false，没有 SF 缓冲；但激活元素 2B vs 1B 让总字节更大。
     用 `_C.get_symm_buffer_size_for_mega_moe` 实测并 slice 出 view 检查 dtype/shape。
  3) 权重重排 `transform_weights_for_mega_moe`（`deep_gemm/mega/__init__.py:146-163`）：
     BF16 分支只做 L1 gate/up 的 gran-8 交错，L2 原样返回（无 SF、无 UTCCP 转置）。

来源核对（`~/github/DeepGEMM`，公开版 commit `057ca59`）：
  - csrc/apis/mega_moe.hpp:9-35, 38-133, 283-369
  - csrc/jit_kernels/heuristics/mega_moe.hpp:55-105
  - csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp:40-54, 84-86
  - deep_gemm/mega/__init__.py:112-163
不涉及 SM100 kernel，仅验证 host 侧配置与棋盘式重排。
"""
import sys

import torch


def align(x, y):
    return (x + y - 1) // y * y


# ---- 复刻 get_block_config_for_mega_moe（heuristics/mega_moe.hpp:62-105）----
def block_config_ref(num_ranks, num_experts, num_tokens, num_topk, mma_kind):
    num_expected_tokens = float(num_tokens) * num_ranks * num_topk / num_experts
    num_covered_tokens = num_expected_tokens + num_expected_tokens ** 0.5

    block_m = 16 if num_expected_tokens <= 10 else 32
    if num_expected_tokens > 24:
        import math
        num_blocks = int(math.ceil(num_covered_tokens / 240))
        if num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14:
            num_blocks = 2
        for candidate in (64, 128, 192, 240):
            block_m = candidate
            if num_blocks * candidate >= num_covered_tokens:
                break

    if block_m <= 16:
        store_l1 = 8
    elif block_m <= 64:
        store_l1 = 16
    elif block_m <= 192:
        store_l1 = 32
    else:
        store_l1 = 24 if mma_kind == 'fp8xp4' else 40
    store_l2 = 8 if (block_m == 240 and mma_kind == 'fp8xfp4') else store_l1
    block_k = 128 // (2 if mma_kind == 'bf16' else 1)
    return block_m, store_l1, store_l2, block_k


def interleave_weights_ref(t, gran=8):
    # deep_gemm/mega/__init__.py:112-126 的纯 torch 复刻
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.stack([gate, up], dim=2).reshape(g, n, *rest)


def deinterleave_weights_ref(t, gran=8):
    g, n, *rest = t.shape
    half = n // 2
    pairs = t.reshape(g, half // gran, 2, gran, *rest)
    gate = pairs[:, :, 0].reshape(g, half, *rest)
    up = pairs[:, :, 1].reshape(g, half, *rest)
    return torch.cat([gate, up], dim=1)


def main():
    import deep_gemm
    from deep_gemm import _C
    from deep_gemm.mega import transform_weights_for_mega_moe

    ok = True
    dev = 'cuda'
    print(f'device: {torch.cuda.get_device_name(0)}')

    # ============ 1) block config 对拍（BF16 vs FP8/FP4 ============
    print('\n== 1) block_m 启发式：API vs 复刻（BF16 / FP8/FP4）==')
    shapes = [
        # num_ranks, num_experts, num_max_tokens, num_tokens, num_topk
        (8, 64, 4096, 8, 1), (8, 64, 4096, 8, 2), (8, 64, 256, 8, 4),
        (8, 256, 4096, 16, 2), (8, 8, 4096, 1, 1), (8, 256, 4096, 64, 2),
        (4, 64, 8192, 32, 2), (8, 512, 4096, 128, 2),
        (8, 64, 4096, 256, 2), (8, 64, 4096, 1024, 2), (8, 64, 8192, 4096, 2),
        (8, 8, 8192, 8192, 2), (4, 128, 8192, 4096, 4),
    ]
    for (R, E, maxT, T, K) in shapes:
        bm_api_bf16 = _C.get_block_m_for_mega_moe(R, E, maxT, T, K, 'bf16xbf16')
        bm_api_fp8 = _C.get_block_m_for_mega_moe(R, E, maxT, T, K, 'fp8xfp4')
        bm_ref_bf16, s1, s2, bk = block_config_ref(R, E, T, K, 'bf16')
        bm_ref_fp8, s1f, s2f, bkf = block_config_ref(R, E, T, K, 'fp8xfp4')
        flag = (bm_api_bf16 == bm_ref_bf16 == bm_api_fp8 == bm_ref_fp8)
        ok &= flag
        print(f'   R={R} E={E} maxT={maxT} T={T} K={K}: '
              f'block_m API(bf16/fp8)={bm_api_bf16}/{bm_api_fp8} ref={bm_ref_bf16} {"" if flag else "MISMATCH"} '
              f'| BF16 store_l1/l2={s1}/{s2} block_k={bk} ; FP8 block_k={bkf}')
    print('   block_m 与 dtype 无关（公式不含 mma_kind）；差异只在 block_k 与 store_block_m')

    # pull 字节：num_bytes_per_pull = hidden*elem_bytes 反复 /2 直到 <= max（heuristics:223-229）
    print('\n   拉取块粒度 num_bytes_per_pull（heuristics/mega_moe.hpp:223-229）：')
    for hidden in (7168, 4096, 2048):
        def pull(mma):
            elem = 2 if mma == 'bf16' else 1
            cap = 4096 if mma == 'bf16' else 8192
            n = hidden * elem
            while n > cap:
                n //= 2
            return n
        print(f'     hidden={hidden}: BF16={pull("bf16")}B（cap 4096） FP8={pull("fp8")}B（cap 8192）')

    # ============ 2) symmetric buffer 尺寸与 view dtype ============
    print('\n== 2) symmetric buffer 尺寸 / view（bf16xbf16 vs fp8xfp4）==')
    for (R, E, maxT, K, H, I, S) in [(8, 64, 4096, 8, 7168, 2048, 0),
                                     (8, 256, 4096, 8, 7168, 2048, 1)]:
        nb_bf16, sl_bf16 = _C.get_symm_buffer_size_for_mega_moe(R, E, maxT, K, H, I, 'bf16xbf16', 'swiglu', S)
        nb_fp8, sl_fp8 = _C.get_symm_buffer_size_for_mega_moe(R, E, maxT, K, H, I, 'fp8xfp4', 'swiglu', S)
        buf_b = torch.zeros(nb_bf16, dtype=torch.int8, device=dev)
        buf_f = torch.zeros(nb_fp8, dtype=torch.int8, device=dev)
        views_b = sl_bf16(buf_b)
        views_f = sl_fp8(buf_f)
        # 输入 token buffer dtype：x 是 view[0]，l1_acts 是 view[8]，l2_acts_sf 是 view[11]
        xb, xsfb, _, _, _, _, _, _, l1b, l1sfb, l2b, l2sfb = views_b
        xf, xsff, _, _, _, _, _, _, l1f, l1sff, l2f, l2sff = views_f
        print(f'   R={R} E={E} S={S}: bytes bf16={nb_bf16/1e6:.1f}MB fp8={nb_fp8/1e6:.1f}MB '
              f'ratio={nb_bf16/nb_fp8:.4f}')
        def desc(t):
            return 'None' if t is None else f'{t.dtype} (numel {t.numel()})'
        print(f'     x dtype bf16={xb.dtype} (numel {xb.numel()}) fp8={xf.dtype}')
        print(f'     x_sf bf16={desc(xsfb)} fp8={desc(xsff)}')
        print(f'     l1_acts shape bf16={tuple(l1b.shape)} fp8={tuple(l1f.shape)}; '
              f'l1_sf bf16={desc(l1sfb)} fp8 shape={tuple(l1sff.shape)}')
        if S > 0:
            print(f'     shared_l2_acts shape bf16={tuple(views_b[6].shape)} fp8={tuple(views_f[6].shape)}')

    # ============ 3) BF16 权重重排 ============
    print('\n== 3) transform_weights_for_mega_moe（BF16 分支）==')
    torch.manual_seed(0)
    for (E, N1, K1, N2, K2) in [(8, 1024, 512, 512, 512), (4, 512, 256, 256, 256), (1, 2048, 1024, 1024, 1024)]:
        l1 = torch.randn(E, N1, K1, dtype=torch.bfloat16, device=dev)
        l2 = torch.randn(E, N2, K2, dtype=torch.bfloat16, device=dev)
        t1, t2 = transform_weights_for_mega_moe(l1, l2)
        ref1 = interleave_weights_ref(l1)
        d_inter = (t1 - ref1).abs().max().item()
        l2_same = t2.data_ptr() == l2.data_ptr()
        # 反交错能恢复原权重
        rec = deinterleave_weights_ref(t1)
        d_rec = (rec - l1).abs().max().item()
        # 交错只改变行序，是纯置换：集合不变
        same_set = torch.equal(torch.sort(t1.reshape(-1))[0], torch.sort(l1.reshape(-1))[0])
        flag = (d_inter == 0.0 and l2_same and d_rec == 0.0 and same_set)
        ok &= flag
        print(f'   E={E} N1={N1} K1={K1}: interleave diff={d_inter} recovert={d_rec} '
              f'l2_unchanged={l2_same} set_equal={same_set} {"" if flag else "MISMATCH"}')

    # 2D 输入（单专家）走 squeeze 分支
    l1_2d = torch.randn(1024, 512, dtype=torch.bfloat16, device=dev)
    l2_2d = torch.randn(512, 512, dtype=torch.bfloat16, device=dev)
    t1_2d, t2_2d = transform_weights_for_mega_moe(l1_2d, l2_2d)
    d2 = (t1_2d - interleave_weights_ref(l1_2d.unsqueeze(0)).squeeze(0)).abs().max().item()
    ok &= (d2 == 0.0 and t1_2d.dim() == 2)
    print(f'   2D 输入: shape={tuple(t1_2d.shape)} diff={d2}')

    # ============ 4) 与 FP8 分支对比：FP8 额外做 SF 的 UTCCP 转置 ============
    print('\n== 4) 对比：FP8/FP4 分支额外转置 SF，BF16 无 SF ==')
    l1w = torch.randn(4, 512, 256, dtype=torch.bfloat16, device=dev)
    l1sf = torch.randint(0, 1 << 20, (4, 512, 2), dtype=torch.int32, device=dev)
    l2w = torch.randn(4, 256, 256, dtype=torch.bfloat16, device=dev)
    l2sf = torch.randint(0, 1 << 20, (4, 256, 2), dtype=torch.int32, device=dev)
    # FP8 分支：先交错 SF 再 UTCCP 转置
    from deep_gemm.mega import _interleave_weights, _transpose_sf_for_utccp
    (f1w, f1sf), (f2w, f2sf) = transform_weights_for_mega_moe((l1w, l1sf), (l2w, l2sf))
    sf_utccp_ok = torch.equal(f1sf, _transpose_sf_for_utccp(_interleave_weights(l1sf)))
    w_inter_ok = torch.equal(f1w, _interleave_weights(l1w))
    l2_sf_transposed = torch.equal(f2sf, _transpose_sf_for_utccp(l2sf))
    ok &= sf_utccp_ok and w_inter_ok and l2_sf_transposed
    print(f'   FP8: l1 w 仅交错={w_inter_ok}；l1 sf 与「交错后立即转置」一致={sf_utccp_ok}'
          f'（说明做了 UTCCP 4x32 转置）；l2 sf 仅转置={l2_sf_transposed}')
    print(f'   BF16: 无 SF，故无 UTCCP 步骤（仅 `_interleave_weights`）')

    print(f'\nALL_OK = {ok}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
