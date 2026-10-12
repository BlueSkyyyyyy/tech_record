#!/usr/bin/env python3
"""Mega MoE 的 host 编排层与对称缓冲区布局 —— 纯 Python 复刻 + 与真实 binding 对拍。

背景（见 docs/42-mega-moe-host-buffer.md）：
  Mega MoE 主 kernel 是 SM100 专属，本机 H100（sm90）跑不了端到端。但**它之前的一整层 host
  编排**（Python `SymmBuffer` / C++ `get_symm_buffer_size_for_mega_moe` / `layout::MegaMoEBuffer`
  及其 ring 容量公式）**全是纯 host 计算，架构无关**，DeepGEMM 通过 pybind 暴露成
  `deep_gemm._C.get_token_alignment_for_mega_moe` 与 `deep_gemm._C.get_symm_buffer_size_for_mega_moe`。

本脚本做四件事（全部在 H100 真机跑）：
  A) 复刻 `layout/mega_moe.cuh` 的 `MegaMoESignals` 结构体布局 + `Workspace`/`Data`/`Buffer`/
     `MegaMoEBuffer` 的**链式 offset 分配**，用 C++ struct 布局规则逐字段算出 12 个 tensor view
     的 (offset, shape, stride, dtype)，与真实 binding 返回的 view **逐字段对拍**；
  B) 复刻 ring 容量公式（`scheduler/mega_moe.cuh:get_num_max_live_pool_blocks` +
     `get_num_l1_warmup_waves` + 迭代所有 kCandidateBlockM 取 max + align 1920），
     用返回的 `num_bytes` 与 `num_ring_tokens` / `num_sf_ring_tokens` 交叉验证；
  C) 用 size-1 进程组构造真实 `SymmBuffer`（world_size==1 走 torch allocator，无需 symmetric
     memory），验证 view、别名（x == shared_l1_acts）、M-major SF、以及 `base` 复用语义；
  D) 打印 ring 容量随形状的变化，说明「为什么要在所有候选 BLOCK_M 上取最大」。

只依赖 torch + deep_gemm（已在 kernel_lab 安装）。GPU 仅用于把 symm buffer 放在 cuda 上。
"""

import math
import sys

import torch
import deep_gemm

_C = deep_gemm._C


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def align(x, a):
    return (x + a - 1) // a * a


def ceil_div(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# A. 复刻 layout/mega_moe.cuh 的结构体与 buffer 链
# ---------------------------------------------------------------------------
K_NUM_CANDIDATE_BLOCK_MS = 8
K_CANDIDATE_BLOCK_M = [8, 16, 32, 64, 96, 128, 192, 240]
K_MAX_CANDIDATE_BLOCK_M = 240
K_MIN_CANDIDATE_BLOCK_M = 8
K_LCM_CANDIDATE_BLOCK_M = 1920

K_NUM_MAX_RANKS = 72                      # sym_buffer.cuh:7
K_NUM_MAX_EXPERTS = 2048                  # layout/mega_moe.cuh:56
K_NUM_MAX_EXPERTS_PER_RANK = 512          # :57
K_NUM_MAX_RING_BLOCKS = 1 << 20           # :58
K_NUM_MAX_SHARED_L2_BLOCKS = 1 << 15      # :59
K_NUM_DEVICE_LOCALITY_DOMAINS = 2         # types.cuh:54


def sizeof_mega_moe_signals():
    """按 C 结构体规则逐字段算 sizeof(MegaMoESignals)（layout/mega_moe.cuh:54-90）。

    字段顺序与 alignas 完全照抄源码；`offsetof(..., combine_ready_grid_idx)==128` 的
    static_assert（:91）会在这里顺带复核。
    """
    # (name, nbytes, alignment)
    fields = [
        ('grid_sync_count', 4 * 4, 4),                       # uint32[kNumMaxGridSyncCounters=4]
        ('nvl_barrier_counter', 4, 4),
        ('nvl_barrier_signals', 2 * 4, 4),                   # int[2]
        ('l1_task_count', K_NUM_DEVICE_LOCALITY_DOMAINS * 4, 4),
        ('l2_task_count', K_NUM_DEVICE_LOCALITY_DOMAINS * 4, 4),
        ('shared_l1_task_count', K_NUM_DEVICE_LOCALITY_DOMAINS * 4, 4),
        ('shared_l2_task_count', K_NUM_DEVICE_LOCALITY_DOMAINS * 4, 4),
        ('combine_ready_grid_idx', K_NUM_MAX_RANKS * 8, 128),
        ('peer_grid_idx', K_NUM_MAX_RANKS * 8, 128),
        ('expert_send_count', K_NUM_MAX_EXPERTS * 8, 128),
        ('expert_recv_count', K_NUM_MAX_EXPERTS * 8, 8),
        ('expert_recv_count_sum', K_NUM_MAX_EXPERTS_PER_RANK * 8, 8),
        ('l1_full_count', K_NUM_MAX_RING_BLOCKS * 4, 4),
        ('l1_empty_count', K_NUM_MAX_RING_BLOCKS * 4, 4),
        ('l2_full_mask', K_NUM_MAX_RING_BLOCKS * 8, 8),
        ('l2_empty_count', K_NUM_MAX_RING_BLOCKS * 4, 4),
        ('shared_l2_full_count', K_NUM_MAX_SHARED_L2_BLOCKS * 4, 4),
    ]
    off, offsets = 0, {}
    for name, size, al in fields:
        off = align(off, al)
        offsets[name] = off
        off += size
    assert offsets['combine_ready_grid_idx'] == 128, offsets['combine_ready_grid_idx']
    return align(off, 128), offsets


def get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank):
    # layout/mega_moe.cuh:20-28
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return align(
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (K_MAX_CANDIDATE_BLOCK_M - 1),
        K_LCM_CANDIDATE_BLOCK_M)


def get_num_sf_ring_tokens(num_ring_tokens, block_m):
    # layout/mega_moe.cuh:31-34
    return (num_ring_tokens // block_m) * align(block_m, 128)


def get_num_max_shared_sf_tokens(num_max_tokens_per_rank):
    # layout/mega_moe.cuh:36-40
    return ceil_div(num_max_tokens_per_rank, K_MIN_CANDIDATE_BLOCK_M) * 128


# --- scheduler/mega_moe.cuh:17-80 -------------------------------------------------
def get_num_l1_warmup_waves(num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters):
    num_first_l2_wave_m_blocks = ceil_div(num_clusters, num_l2_n_clusters)
    num_l1_warmup_clusters_for_first_l2_wave = ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters)
    diff = (num_l1_n_clusters - num_l2_n_clusters) if num_l1_n_clusters > num_l2_n_clusters else 0
    num_warmup_waves_for_interleave_schedule = ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * diff, num_clusters) + 1
    return max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule)


def get_num_max_live_pool_blocks(num_total_m_blocks, num_sms, hidden, intermediate_hidden):
    # scheduler/mega_moe.cuh:48-80
    kMegaMoEBlockN = 128
    kNumCTAsPerCluster = 2
    assert (intermediate_hidden * 2) % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0
    assert hidden % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0
    num_clusters = num_sms // kNumCTAsPerCluster
    num_l1_n_clusters = intermediate_hidden * 2 // (kNumCTAsPerCluster * kMegaMoEBlockN)
    num_l2_n_clusters = hidden // (kNumCTAsPerCluster * kMegaMoEBlockN)
    num_l1_clusters = num_total_m_blocks * num_l1_n_clusters
    num_l1_waves = ceil_div(num_l1_clusters, num_clusters)
    num_min_l1_warmup_waves = get_num_l1_warmup_waves(
        num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters)
    num_l1_warmup_waves = min(num_min_l1_warmup_waves, num_l1_waves)
    num_l1_warmup_clusters = min(num_l1_warmup_waves * num_clusters, num_l1_clusters)
    num_live_blocks_after_warmup = ceil_div(num_l1_warmup_clusters, num_l1_n_clusters)
    frontier_growth = (ceil_div(num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters),
                                num_l2_n_clusters)
                       if num_l2_n_clusters > num_l1_n_clusters else 0)
    wave_margin = ceil_div(num_clusters, min(num_l1_n_clusters, num_l2_n_clusters))
    return min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin)


# --- csrc/apis/mega_moe.hpp:32-153 ------------------------------------------------
def symm_buffer_size_reimpl(num_sms, num_ranks, num_experts, num_max_tokens_per_rank,
                            num_topk, hidden, intermediate_hidden, with_sf,
                            num_shared_experts=0):
    """复刻 `get_symm_buffer_size_for_mega_moe`：ring 容量 + MegaMoEBuffer 全部 offset。

    返回 (num_bytes, num_ring_tokens, num_sf_ring_tokens, views)。
    views: list of (name, offset, shape, stride, dtype_str)；stride=None 表示缺省连续。
    """
    num_experts_per_rank = num_experts // num_ranks
    num_active_topk = min(num_topk, num_experts_per_rank)
    num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk
    shared_intermediate_hidden = intermediate_hidden * num_shared_experts

    # 迭代所有候选 BLOCK_M 取 ring 容量上界（mega_moe.hpp:56-63）
    num_ring_tokens = 0
    for block_m in K_CANDIDATE_BLOCK_M:
        num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank
        num_live_pool_blocks = get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, intermediate_hidden)
        num_ring_tokens = max(num_ring_tokens, num_live_pool_blocks * block_m)
    num_ring_tokens = align(num_ring_tokens, K_LCM_CANDIDATE_BLOCK_M)

    num_sf_ring_tokens = 0
    if with_sf:
        for block_m in K_CANDIDATE_BLOCK_M:
            num_sf_ring_tokens = max(
                num_sf_ring_tokens, get_num_sf_ring_tokens(num_ring_tokens, block_m))

    # --- Workspace.get_num_bytes (layout/mega_moe.cuh:128-144) ---
    signals_size, _ = sizeof_mega_moe_signals()
    num_max_pool_tokens = get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank)
    ws_bytes = signals_size
    ws_bytes += num_experts * num_max_tokens_per_rank * 4
    ws_bytes += num_max_pool_tokens * 12
    ws_bytes = align(ws_bytes, 16)

    # --- Data/Buffer 链 (layout/mega_moe.cuh:387-449) ---
    num_mma_elem_bytes = 1 if with_sf else 2
    input_token_row = hidden * num_mma_elem_bytes
    bf16_token_row = hidden * 2
    intermediate_token_row = intermediate_hidden * num_mma_elem_bytes
    shared_intermediate_token_row = shared_intermediate_hidden * num_mma_elem_bytes
    input_sf_row = (hidden // 32) if with_sf else 0          # 字节；int32 数量 hidden/128
    intermediate_sf_row = (intermediate_hidden // 32) if with_sf else 0
    shared_intermediate_sf_row = (shared_intermediate_hidden // 32) if with_sf else 0
    input_topk_idx_row = num_topk * 8
    input_topk_weights_row = num_topk * 4
    l1_topk_weights_row = 4
    num_max_shared_sf_tokens = get_num_max_shared_sf_tokens(num_max_tokens_per_rank) if with_sf else 0

    def buf(row, n_ranks, n_tokens, base):
        return (base, row * n_tokens * n_ranks)

    base = ws_bytes
    input_token_buffer = buf(input_token_row, 1, num_max_tokens_per_rank, base)
    input_sf_buffer = buf(input_sf_row, 1, num_max_tokens_per_rank, input_token_buffer[0] + input_token_buffer[1])
    input_topk_idx_buffer = buf(input_topk_idx_row, 1, num_max_tokens_per_rank,
                                (input_sf_buffer[0] + input_sf_buffer[1]) if with_sf
                                else (input_token_buffer[0] + input_token_buffer[1]))
    input_topk_weights_buffer = buf(input_topk_weights_row, 1, num_max_tokens_per_rank,
                                    input_topk_idx_buffer[0] + input_topk_idx_buffer[1])

    shared_l1_token_buffer = input_token_buffer
    shared_l1_sf_buffer = buf(input_sf_row, 1,
                              num_max_shared_sf_tokens if num_shared_experts > 0 else 0,
                              input_topk_weights_buffer[0] + input_topk_weights_buffer[1])
    shared_l2_token_buffer = buf(shared_intermediate_token_row, 1,
                                 num_max_tokens_per_rank if num_shared_experts > 0 else 0,
                                 (shared_l1_sf_buffer[0] + shared_l1_sf_buffer[1]) if with_sf
                                 else (input_topk_weights_buffer[0] + input_topk_weights_buffer[1]))
    shared_l2_sf_buffer = buf(shared_intermediate_sf_row, 1,
                              num_max_shared_sf_tokens if num_shared_experts > 0 else 0,
                              shared_l2_token_buffer[0] + shared_l2_token_buffer[1])

    l1_token_base = ((shared_l2_sf_buffer[0] + shared_l2_sf_buffer[1]) if with_sf
                     else (shared_l2_token_buffer[0] + shared_l2_token_buffer[1])) \
        if num_shared_experts > 0 else (input_topk_weights_buffer[0] + input_topk_weights_buffer[1])
    l1_token_buffer = buf(input_token_row, 1, num_ring_tokens, l1_token_base)
    l1_sf_buffer = buf(input_sf_row, 1, num_sf_ring_tokens, l1_token_buffer[0] + l1_token_buffer[1])
    l1_topk_weights_buffer = buf(l1_topk_weights_row, 1, num_ring_tokens,
                                 (l1_sf_buffer[0] + l1_sf_buffer[1]) if with_sf
                                 else (l1_token_buffer[0] + l1_token_buffer[1]))
    l2_token_buffer = buf(intermediate_token_row, 1, num_ring_tokens,
                          l1_topk_weights_buffer[0] + l1_topk_weights_buffer[1])
    l2_sf_buffer = buf(intermediate_sf_row, 1, num_sf_ring_tokens,
                       l2_token_buffer[0] + l2_token_buffer[1])
    combine_token_buffer = buf(bf16_token_row, num_topk + (1 if num_shared_experts > 0 else 0),
                               num_max_tokens_per_rank,
                               (l2_sf_buffer[0] + l2_sf_buffer[1]) if with_sf
                               else (l2_token_buffer[0] + l2_token_buffer[1]))

    total = combine_token_buffer[0] + combine_token_buffer[1]

    # --- slice_input_buffers (mega_moe.hpp:96-151) ---
    tok_dtype = 'fp8' if with_sf else 'bf16'
    views = [
        ('x', input_token_buffer[0], (num_max_tokens_per_rank, hidden), None, tok_dtype),
    ]
    if with_sf:
        views.append(('x_sf', input_sf_buffer[0],
                      (num_max_tokens_per_rank, hidden // 128), None, 'int32'))
    views += [
        ('topk_idx', input_topk_idx_buffer[0], (num_max_tokens_per_rank, num_topk), None, 'int64'),
        ('topk_weights', input_topk_weights_buffer[0], (num_max_tokens_per_rank, num_topk), None, 'fp32'),
        ('shared_l1_acts', input_token_buffer[0], (num_max_tokens_per_rank, hidden), None, tok_dtype),
    ]
    if with_sf and num_shared_experts > 0:
        views.append(('shared_l1_acts_sf', shared_l1_sf_buffer[0],
                      (num_max_shared_sf_tokens, hidden // 128),
                      (1, num_max_shared_sf_tokens), 'int32'))
    if num_shared_experts > 0:
        views.append(('shared_l2_acts', shared_l2_token_buffer[0],
                      (num_max_tokens_per_rank, shared_intermediate_hidden), None, tok_dtype))
    if with_sf and num_shared_experts > 0:
        views.append(('shared_l2_acts_sf', shared_l2_sf_buffer[0],
                      (num_max_shared_sf_tokens, shared_intermediate_hidden // 128),
                      (1, num_max_shared_sf_tokens), 'int32'))
    views.append(('l1_acts', l1_token_buffer[0], (num_ring_tokens, hidden), None, tok_dtype))
    if with_sf:
        views.append(('l1_acts_sf', l1_sf_buffer[0], (num_sf_ring_tokens, hidden // 128),
                      (1, num_sf_ring_tokens), 'int32'))
    views.append(('l2_acts', l2_token_buffer[0],
                  (num_ring_tokens, intermediate_hidden), None, tok_dtype))
    if with_sf:
        views.append(('l2_acts_sf', l2_sf_buffer[0],
                      (num_sf_ring_tokens, intermediate_hidden // 128),
                      (1, num_sf_ring_tokens), 'int32'))
    return total, num_ring_tokens, num_sf_ring_tokens, views


DTYPE_MAP = {
    'fp8': torch.float8_e4m3fn,
    'bf16': torch.bfloat16,
    'int32': torch.int32,
    'int64': torch.int64,
    'fp32': torch.float32,
}


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
def run_configs(num_sms):
    print('=' * 78)
    print('A/B) 复刻 buffer 布局 + ring 容量，与真实 binding 逐字段对拍')
    print('=' * 78)

    cfgs = [
        # (R, E, Tmax, topk, hidden, inter, mma_type, shared)
        (8, 256, 2048, 8, 7168, 2048, 'fp8xfp4', 0),
        (8, 256, 4096, 8, 7168, 2048, 'fp8xfp4', 1),
        (1, 256, 4096, 8, 7168, 2048, 'fp8xfp8', 0),
        (4, 128, 8192, 6, 4096, 1408, 'fp8xfp4', 2),
        (16, 512, 1024, 4, 2048, 512, 'bf16xbf16', 0),
        (64, 1024, 512, 8, 3072, 1536, 'bf16xbf16', 1),
        (2, 64, 32, 2, 1024, 256, 'fp8xfp4', 0),
    ]

    all_ok = True
    for (R, E, Tmax, topk, hidden, inter, mma_type, shared) in cfgs:
        with_sf = mma_type != 'bf16xbf16'
        n_bytes, ring, sf_ring, views = symm_buffer_size_reimpl(
            num_sms, R, E, Tmax, topk, hidden, inter, with_sf, shared)

        real_n, slicer = _C.get_symm_buffer_size_for_mega_moe(
            R, E, Tmax, topk, hidden, inter, mma_type, 'swiglu', shared)
        buf = torch.zeros(real_n, dtype=torch.int8, device='cuda')
        real_views = slicer(buf)

        tag = (f'R={R} E={E} Tmax={Tmax} topk={topk} H={hidden} I={inter} '
               f'{mma_type} shared={shared}')
        print(f'\n--- {tag} ---')
        print(f'  ring_tokens={ring}  sf_ring_tokens={sf_ring}')
        print(f'  num_bytes  reimpl={n_bytes}  real={real_n}  '
              f'{"MATCH" if n_bytes == real_n else "MISMATCH !! "}')
        if n_bytes != real_n:
            all_ok = False

        # 期望 view 名（按 C++ tuple 顺序）：undefined 的用 None 占位
        expected_names = ['x', 'x_sf', 'topk_idx', 'topk_weights',
                          'shared_l1_acts', 'shared_l1_acts_sf', 'shared_l2_acts', 'shared_l2_acts_sf',
                          'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf']
        reimpl_by_name = {v[0]: v for v in views}

        for i, name in enumerate(expected_names):
            rv = real_views[i]
            defined = rv is not None and getattr(rv, 'numel', lambda: 0)() > 0 and rv.dim() > 0
            exp = reimpl_by_name.get(name)
            if not defined:
                # C++ 里 x_sf / shared_*_sf / *_sf 在无 SF 时返回空 tensor
                if exp is not None and exp[2][0] * (exp[2][1] if len(exp[2]) > 1 else 1) > 0:
                    print(f'    {name}: 期望非空但实际 undefined !!')
                    all_ok = False
                continue
            if exp is None:
                print(f'    {name}: 实际有定义但复刻里没算 !!')
                all_ok = False
                continue
            _, off, shape, stride, dt = exp
            r_off = rv.data_ptr() - buf.data_ptr()
            r_shape = tuple(rv.shape)
            r_stride = tuple(rv.stride())
            r_dt = rv.dtype
            exp_stride = stride if stride is not None else tuple(
                _contig_stride(shape))
            ok = (r_off == off and r_shape == shape and r_stride == tuple(exp_stride)
                  and r_dt == DTYPE_MAP[dt])
            all_ok = all_ok and ok
            print(f'    {name:20s} off={r_off:>11d} shape={str(r_shape):18s} '
                  f'stride={str(r_stride):20s} {str(r_dt).split(".")[-1]:8s} '
                  f'{"OK" if ok else "FAIL (exp off=%d shape=%s stride=%s %s)" % (off, shape, tuple(exp_stride), dt)}')

    return all_ok


def _contig_stride(shape):
    s = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        s[i] = s[i + 1] * shape[i + 1]
    return s


def show_ring_ladder(num_sms):
    print()
    print('=' * 78)
    print('D) ring 容量随形状变化：为什么要在所有候选 BLOCK_M 上取 max')
    print('=' * 78)
    R, E, Tmax, topk, hidden, inter = 8, 256, 4096, 8, 7168, 2048
    with_sf = True
    num_experts_per_rank = E // R
    num_active_topk = min(topk, num_experts_per_rank)
    num_max_routed_tokens = Tmax * R * num_active_topk
    print(f'  配置 R={R} E={E} Tmax={Tmax} topk={topk} H={hidden} I={inter}; '
          f'num_sms={num_sms}; routed_tokens={num_max_routed_tokens}')
    print(f'  {"BLOCK_M":>8} {"pool_blocks":>12} {"live_blocks":>12} {"live*BM":>10} {"sf_ring(BM)":>12}')
    best = 0
    for block_m in K_CANDIDATE_BLOCK_M:
        pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank
        live = get_num_max_live_pool_blocks(pool_blocks, num_sms, hidden, inter)
        live_tok = live * block_m
        # sf_ring 用最终 ring（此处先展示该 block_m 单独算的，随后统一）
        print(f'  {block_m:>8} {pool_blocks:>12} {live:>12} {live_tok:>10}')
        best = max(best, live_tok)
    ring = align(best, K_LCM_CANDIDATE_BLOCK_M)
    print(f'  → max(live*BM)={best}, align 1920 → num_ring_tokens={ring}')
    sf_ladder = [(bm, get_num_sf_ring_tokens(ring, bm)) for bm in K_CANDIDATE_BLOCK_M]
    print('  各候选 BLOCK_M 对应的 sf_ring_tokens：', sf_ladder)
    print(f'  → num_sf_ring_tokens = max = {max(v for _, v in sf_ladder)}')


def test_symm_buffer_class():
    print()
    print('=' * 78)
    print('C) 真实 SymmBuffer 类（world_size==1 → torch allocator）与 base 复用')
    print('=' * 78)
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group('gloo', init_method='tcp://127.0.0.1:29511',
                                rank=0, world_size=1)
    from deep_gemm.mega import SymmBuffer, _interleave_weights, _transpose_sf_for_utccp

    R, E, Tmax, topk, hidden, inter = 1, 8, 512, 4, 2048, 512
    sb = SymmBuffer(dist.group.WORLD, E, Tmax, topk, hidden, inter, mma_type='fp8xfp4')
    print(f'  x {tuple(sb.x.shape)} {sb.x.dtype}  x_sf {tuple(sb.x_sf.shape)} {sb.x_sf.dtype}')
    print(f'  topk_idx {tuple(sb.topk_idx.shape)} {sb.topk_idx.dtype}  '
          f'topk_weights {tuple(sb.topk_weights.shape)} {sb.topk_weights.dtype}')
    print(f'  l1_acts {tuple(sb.l1_acts.shape)}  l1_acts_sf {tuple(sb.l1_acts_sf.shape)} '
          f'stride={tuple(sb.l1_acts_sf.stride())}')
    print(f'  l2_acts {tuple(sb.l2_acts.shape)}  l2_acts_sf {tuple(sb.l2_acts_sf.shape)} '
          f'stride={tuple(sb.l2_acts_sf.stride())}')

    # 别名：shared_l1_acts 就是 x
    print(f'  shared_l1_acts is x ? {sb.shared_l1_acts.data_ptr() == sb.x.data_ptr()}')

    # 写入并在 buffer 里原位读回
    sb.x.zero_()
    sb.x[0, 0] = 1.0
    sb.x[1, 3] = 2.0
    print(f'  write/read x: {sb.x[0,0].item()} {sb.x[1,3].item()} (expect 1.0 2.0)')

    # 复用：更小的 num_bytes 用 base 复用
    try:
        sb2 = SymmBuffer(dist.group.WORLD, E, Tmax, topk, hidden, inter,
                         mma_type='fp8xfp4', base=sb)
        print(f'  reuse smaller config: OK, same buffer ? '
              f'{sb2.buffer.data_ptr() == sb.buffer.data_ptr()}')
    except Exception as e:
        print(f'  reuse smaller config FAILED: {type(e).__name__}: {e}')

    # 复用一个更大的配置应被拒绝（num_bytes > base.buffer.nbytes）
    try:
        SymmBuffer(dist.group.WORLD, E, 4096, topk, hidden, inter, mma_type='fp8xfp4', base=sb)
        print('  reuse larger config: !! 意外通过')
    except Exception as e:
        msg = str(e).strip().splitlines()
        print(f'  reuse larger config rejected: {type(e).__name__}: '
              f'{msg[-1] if msg else str(e)[:80]}')


def main():
    if not torch.cuda.is_available():
        print('CUDA 不可用')
        return 1
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    from deep_gemm.testing.utils import get_arch_major
    print(f'GPU: {torch.cuda.get_device_name(0)}  SMs={num_sms}  '
          f'arch_major={get_arch_major()}')
    print(f'get_token_alignment_for_mega_moe() = '
          f'{_C.get_token_alignment_for_mega_moe()} (期望 kLCMCandidateBlockM=1920)')

    ok = run_configs(num_sms)
    show_ring_ladder(num_sms)
    test_symm_buffer_class()

    print()
    print('=' * 78)
    print('总判定：', 'ALL PASS' if ok else 'HAS MISMATCH !!')
    print('=' * 78)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
