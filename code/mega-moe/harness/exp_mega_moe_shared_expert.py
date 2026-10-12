#!/usr/bin/env python3
"""Mega MoE 的共享专家（shared expert）分支 —— host 对拍 + 调度语义 + H100 数值。

背景（见 docs/44-mega-moe-shared-expert.md）：
  DeepSeek 的 MoE 每层除了 top-k 路由的专家，还有一个**每 token 恒定激活的共享专家**。
  Mega MoE 把它融进**同一个 mega-kernel**：共享专家不参与 dispatch（直接吃本 rank 的原始
  token），它的 L1/L2 是另外两条 `BlockPhase::SharedLinear1/SharedLinear2`，结果写进 combine
  buffer 的一个额外槽，由 combine 与路由专家输出**在 fp32 里一起相加**（共享专家不带 top-k 权重）。

  主 kernel 是 SM100 专属，本机 H100（sm90）跑不了端到端。但：
    * host 编排层（`get_symm_buffer_size_for_mega_moe` / `MegaMoEBuffer`）纯 host、架构无关，
      DeepGEMM 通过 pybind 暴露 —— 与真实 binding 逐字段对拍（Part A）；
    * 调度器 `scheduler/mega_moe.cuh` 的形状/决策公式可复刻并核对（Part B）；
    * UTCCP 的 SF token 索引映射（device `transform_sf_token_idx`）与测试 host 参考
      （`_to_shared_mega_moe_sf_layout`）逐点对拍（Part C）；
    * **共享专家的数学**（x@W1ᵀ → SwiGLU → @W2ᵀ，无权重，加进 combine）**架构无关**，
      本机用真实 SM90 BF16 GEMM kernel 跑出来与 fp64 参考对拍（Part D）。

只依赖 torch + deep_gemm（已在 kernel_lab 安装）。
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


K_CANDIDATE_BLOCK_M = [8, 16, 32, 64, 96, 128, 192, 240]
K_MIN_CANDIDATE_BLOCK_M = 8
K_MAX_CANDIDATE_BLOCK_M = 240
K_LCM_CANDIDATE_BLOCK_M = 1920
K_NUM_MAX_SHARED_L2_BLOCKS = 1 << 15
K_NUM_DEVICE_LOCALITY_DOMAINS = 2
K_BLOCK_N = 128                      # BLOCK_N == LAYOUT_AD_M (kernel :168)


def get_num_max_shared_sf_tokens(num_max_tokens_per_rank):
    # layout/mega_moe.cuh:36-40
    return ceil_div(num_max_tokens_per_rank, K_MIN_CANDIDATE_BLOCK_M) * 128


def num_shared_l2_pool_blocks(num_max_tokens_per_rank):
    # layout/mega_moe.cuh:117 + Workspace ctor
    return ceil_div(num_max_tokens_per_rank, K_MIN_CANDIDATE_BLOCK_M)


# ---------------------------------------------------------------------------
# A. host binding：共享专家相关的 buffer view 逐字段对拍
# ---------------------------------------------------------------------------
DTYPE_MAP = {
    'fp8': torch.float8_e4m3fn,
    'bf16': torch.bfloat16,
    'int32': torch.int32,
    'int64': torch.int64,
    'fp32': torch.float32,
}

# C++ slice_input_buffers 返回的 12 个 view（顺序固定，见 mega_moe.hpp:148-150）
VIEW_NAMES = ['x', 'x_sf', 'topk_idx', 'topk_weights',
              'shared_l1_acts', 'shared_l1_acts_sf', 'shared_l2_acts', 'shared_l2_acts_sf',
              'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf']


def run_host_view_checks():
    print('=' * 78)
    print('A) host binding：共享专家 view（shape/stride/dtype/别名）逐字段对拍')
    print('=' * 78)

    # (R, E, Tmax, topk, hidden, inter, mma_type, shared)
    cfgs = [
        (8, 256, 4096, 8, 7168, 2048, 'fp8xfp4', 0),
        (8, 256, 4096, 8, 7168, 2048, 'fp8xfp4', 1),
        (4, 128, 8192, 6, 4096, 1408, 'fp8xfp4', 2),
        (16, 512, 1024, 4, 2048, 512, 'bf16xbf16', 0),
        (64, 1024, 512, 8, 3072, 1536, 'bf16xbf16', 1),
    ]

    all_ok = True
    for (R, E, Tmax, topk, hidden, inter, mma_type, shared) in cfgs:
        with_sf = mma_type != 'bf16xbf16'
        shared_inter = inter * shared
        n_bytes, slicer = _C.get_symm_buffer_size_for_mega_moe(
            R, E, Tmax, topk, hidden, inter, mma_type, 'swiglu', shared)
        buf = torch.zeros(n_bytes, dtype=torch.int8, device='cuda')
        views = {name: slicer(buf)[i] for i, name in enumerate(VIEW_NAMES)}

        tag = (f'R={R} E={E} Tmax={Tmax} topk={topk} H={hidden} I={inter} '
               f'{mma_type} shared={shared}')
        print(f'\n--- {tag} ---')

        def defined(t):
            return t is not None and t.dim() > 0 and t.numel() > 0

        # 1) shared_l1_acts 恒 == x（复用 input_token_buffer，layout:415 / hpp:114）
        same = views['shared_l1_acts'].data_ptr() == views['x'].data_ptr()
        print(f'  shared_l1_acts is x              : {same} '
              f'{"OK" if same else "FAIL"}  (layout:415, hpp:114)')
        all_ok &= same

        # 2) shared_l1_acts_sf / shared_l2_acts_sf 的行数 = get_num_max_shared_sf_tokens(Tmax)
        exp_sf_tokens = get_num_max_shared_sf_tokens(Tmax) if (with_sf and shared > 0) else 0
        if with_sf and shared > 0:
            for nm in ['shared_l1_acts_sf', 'shared_l2_acts_sf']:
                v = views[nm]
                shape_ok = tuple(v.shape)[0] == exp_sf_tokens
                stride_ok = tuple(v.stride()) == (1, exp_sf_tokens)
                print(f'  {nm:20s} shape={tuple(v.shape)} stride={tuple(v.stride())} '
                      f'{"OK" if (shape_ok and stride_ok) else "FAIL"}  '
                      f'(expect row={exp_sf_tokens}, col-major stride=(1,{exp_sf_tokens}))')
                all_ok &= shape_ok and stride_ok
        else:
            for nm in ['shared_l1_acts_sf', 'shared_l2_acts_sf']:
                print(f'  {nm:20s} undefined: {not defined(views[nm])}  (shared==0 或无 SF)')

        # 3) shared_l2_acts shape = (Tmax, E_s*inter)
        if shared > 0:
            v = views['shared_l2_acts']
            exp_shape = (Tmax, shared_inter)
            ok = tuple(v.shape) == exp_shape
            print(f'  shared_l2_acts     shape={tuple(v.shape)} '
                  f'{"OK" if ok else "FAIL"}  (expect {exp_shape} = (Tmax, E_s*I))')
            all_ok &= ok
        else:
            print(f'  shared_l2_acts     undefined: {not defined(views["shared_l2_acts"])}')

        # 4) shared_l1_acts_sf 的 K 维 = hidden/128；shared_l2_acts_sf 的 K 维 = E_s*inter/128
        if with_sf and shared > 0:
            v1 = views['shared_l1_acts_sf']
            v2 = views['shared_l2_acts_sf']
            ok1 = tuple(v1.shape)[1] == hidden // 128
            ok2 = tuple(v2.shape)[1] == shared_inter // 128
            print(f'  shared SF K-dim    l1={tuple(v1.shape)[1]} (={hidden}//128={hidden // 128}) '
                  f'l2={tuple(v2.shape)[1]} (={shared_inter}//128={shared_inter // 128}) '
                  f'{"OK" if (ok1 and ok2) else "FAIL"}')
            all_ok &= ok1 and ok2

        # 5) E_s=0 时 shared view 必须整个不存在（0 元素），且 tma align 标志一致
        if shared == 0:
            for nm in ['shared_l1_acts_sf', 'shared_l2_acts', 'shared_l2_acts_sf']:
                nd = not defined(views[nm])
                print(f'  {nm:20s} undefined when shared==0: {nd}  '
                      f'{"OK" if nd else "FAIL"}')
                all_ok &= nd

    return all_ok


# ---------------------------------------------------------------------------
# B. 调度语义：形状关系 / 早期决策 / 任务数（复刻 scheduler/mega_moe.cuh）
# ---------------------------------------------------------------------------
def shared_shape_relations(hidden, inter, num_shared_experts):
    # scheduler/mega_moe.cuh:182-185
    L1_shape_n = inter * 2
    L1_shape_k = hidden
    L2_shape_n = hidden
    L2_shape_k = inter
    return {
        'SHARED_L1_SHAPE_N': L1_shape_n * num_shared_experts,
        'SHARED_L1_SHAPE_K': L1_shape_k,
        'SHARED_L2_SHAPE_N': L2_shape_n,
        'SHARED_L2_SHAPE_K': L2_shape_k * num_shared_experts,
    }


def is_shared_l2_early(num_tokens, block_m, shape_n, kNumSMs):
    # scheduler/mega_moe.cuh:443-444（kNumSharedL2Clusters = SHARED_L2_SHAPE_N / BLOCK_N / 2）
    kNumSharedL2Clusters = shape_n // K_BLOCK_N // 2
    return ceil_div(num_tokens, block_m) * kNumSharedL2Clusters <= kNumSMs // 2, kNumSharedL2Clusters


def run_scheduler_checks(num_sms):
    print()
    print('=' * 78)
    print('B) 共享专家的「形状关系」与「L2 提前/推迟」决策（复刻 scheduler/mega_moe.cuh）')
    print('=' * 78)

    all_ok = True
    # 形状关系：SHARED_L1_SHAPE_N = L1_SHAPE_N * E_s（N 轴），SHARED_L2_SHAPE_K = L2_SHAPE_K * E_s（K 轴）
    for (hidden, inter, es) in [(7168, 2048, 1), (7168, 2048, 2), (4096, 1408, 4)]:
        rel = shared_shape_relations(hidden, inter, es)
        # 静态断言：SHARED_* % (BLOCK_N*2)==0 且 % BLOCK_K==0（BLOCK_K=128 见 heuristics）
        ok = (rel['SHARED_L1_SHAPE_N'] % (K_BLOCK_N * 2) == 0
              and rel['SHARED_L2_SHAPE_N'] % (K_BLOCK_N * 2) == 0
              and rel['SHARED_L1_SHAPE_K'] % 128 == 0
              and rel['SHARED_L2_SHAPE_K'] % 128 == 0)
        print(f'  H={hidden} I={inter} E_s={es}: '
              f'SHARED_L1_SHAPE_N={rel["SHARED_L1_SHAPE_N"]}={inter * 2}*{es} '
              f'SHARED_L2_SHAPE_K={rel["SHARED_L2_SHAPE_K"]}={inter}*{es} '
              f'{"OK" if ok else "FAIL"}')
        all_ok &= ok

    # L2 提前/推迟决策表：随 BLOCK_M 变化（num_tokens 固定）
    print()
    print('  is_shared_l2_early = ceil_div(num_tokens, BLOCK_M) * (SHARED_L2_SHAPE_N/128/2) <= num_sms/2')
    print(f'  num_sms={num_sms}, H=7168 I=2048 E_s=1 (hidden 固定)，num_tokens 变化：')
    print(f'  {"num_tokens":>10} {"BLOCK_M":>8} {"m_blocks":>8} {"L2clusters":>10} {"early?":>7}')
    for num_tokens in [256, 2048, 8192]:
        for block_m in [64, 128, 240]:
            early, l2c = is_shared_l2_early(num_tokens, block_m, 7168, num_sms)
            print(f'  {num_tokens:>10} {block_m:>8} {ceil_div(num_tokens, block_m):>8} '
                  f'{l2c:>10} {str(early):>7}')
    early, l2c = is_shared_l2_early(2048, 240, 7168, num_sms)
    ok = (l2c == 7168 // 128 // 2) and (early == (ceil_div(2048, 240) * l2c <= num_sms // 2))
    all_ok &= ok

    # 共享 L2 的信号容量：num_shared_l2_pool_blocks = ceil_div(Tmax, 8) <= 32768
    print()
    print('  shared L2 信号容量：num_shared_l2_pool_blocks = ceil_div(Tmax, kMinCandidateBlockM)')
    print(f'  {"Tmax":>10} {"pool_blocks":>12} {"<=32768?":>9}  (layout:117/125, signals:59)')
    for Tmax in [512, 4096, 16384, 262144]:
        pb = num_shared_l2_pool_blocks(Tmax)
        okpb = pb <= K_NUM_MAX_SHARED_L2_BLOCKS
        print(f'  {Tmax:>10} {pb:>12} {str(okpb):>9}')
        all_ok &= okpb

    return all_ok


# ---------------------------------------------------------------------------
# C. UTCCP 的 SF token 索引映射：device vs host 参考逐点对拍
# ---------------------------------------------------------------------------
def device_transform_sf_token_idx(token_idx_in_expert, block_m):
    # kernel :145-149（SF_BLOCK_M = align(block_m, 128)）
    sf_block_m = align(block_m, 128)
    idx = token_idx_in_expert % block_m
    return token_idx_in_expert // block_m * sf_block_m + \
        (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3)


def host_to_shared_layout_index(block_idx, m_idx, block_m):
    # test_mega_moe.py:36-50（aligned_block_m = align(block_m,128)）
    aligned_block_m = align(block_m, 128)
    transposed_m_idx = (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
    return block_idx * aligned_block_m + transposed_m_idx


def run_utccp_mapping_check():
    print()
    print('=' * 78)
    print('C) UTCCP SF token 索引映射：device `transform_sf_token_idx` vs host 参考逐点对拍')
    print('=' * 78)
    all_ok = True
    total = 0
    for block_m in [8, 16, 32, 64, 96, 128, 192, 240]:
        num_m_blocks = 5
        seen = set()
        mism = 0
        for block_idx in range(num_m_blocks):
            for m_idx in range(block_m):
                t = block_idx * block_m + m_idx
                dv = device_transform_sf_token_idx(t, block_m)
                hv = host_to_shared_layout_index(block_idx, m_idx, block_m)
                if dv != hv:
                    mism += 1
                seen.add(dv)
        total += 1
        # 双射：映射到 [0, num_m_blocks * align(block_m,128)) 且无重复
        span = num_m_blocks * align(block_m, 128)
        bijective = (len(seen) == num_m_blocks * block_m) and (max(seen) < span)
        print(f'  BLOCK_M={block_m:>3}  mism={mism}  bijective_into_{span}={bijective} '
              f'{"OK" if (mism == 0 and bijective) else "FAIL"}')
        all_ok &= (mism == 0 and bijective)
    print(f'  → {total} 个 BLOCK_M 全部逐点一致且为单射')
    return all_ok


# ---------------------------------------------------------------------------
# D. H100 数值：共享专家前向（真实 SM90 BF16 GEMM）+ combine 语义
# ---------------------------------------------------------------------------
def silu(x):
    return x * torch.sigmoid(x)


def run_numeric(num_sms):
    print()
    print('=' * 78)
    print('D) H100 数值：共享专家前向（真实 deep_gemm.bf16_gemm_nt）+ combine「无权重相加」')
    print('=' * 78)
    all_ok = True

    torch.manual_seed(0)
    dev = 'cuda'
    T, H, I = 1024, 1024, 512
    Es = 2
    S = I * Es                              # 共享专家的中间维 = E_s * I

    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    W1s = torch.randn(S * 2, H, dtype=torch.bfloat16, device=dev) * 0.05   # 共享 L1: [2S, H]
    W2s = torch.randn(H, S, dtype=torch.bfloat16, device=dev) * 0.05       # 共享 L2: [H, S]

    # --- 真实 kernel 前向 ---
    l1 = torch.empty(T, S * 2, dtype=torch.bfloat16, device=dev)
    deep_gemm.bf16_gemm_nt(x, W1s, l1)
    # SwiGLU（gate=l1[:, :S], up=l1[:, S:]）
    act = (silu(l1[:, :S].float()) * l1[:, S:].float()).to(torch.bfloat16)
    y_shared = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    deep_gemm.bf16_gemm_nt(act, W2s, y_shared)

    # --- fp64 参考 ---
    x64, W1s64, W2s64 = x.double(), W1s.double(), W2s.double()
    l1_64 = x64 @ W1s64.t()
    act_64 = silu(l1_64[:, :S]) * l1_64[:, S:]
    y_ref = act_64 @ W2s64.t()

    diff = (y_shared.double() - y_ref)
    rel = diff.norm() / y_ref.norm()
    print(f'  形状: T={T} H={H} I={I} E_s={Es} S=E_s*I={S}  '
          f'W1s={tuple(W1s.shape)} W2s={tuple(W2s.shape)}')
    print(f'  共享专家前向 vs fp64 参考: relerr={rel.item():.3e} '
          f'{"OK (<2e-2)" if rel.item() < 2e-2 else "FAIL"}')
    all_ok &= rel.item() < 2e-2

    # --- 「N 轴 / K 轴」：把共享专家看成「更宽的单个 MLP」vs「E_s 个小专家拼接」等价性 ---
    # 用 (T,S) 的恒等说明 L1 的 N = 2S、L2 的 K = S（乘上权重即等价）。
    assert W1s.shape[0] == 2 * S and W2s.shape[1] == S
    print(f'  形状关系坐实: SHARED_L1_SHAPE_N=2S={2 * S}, SHARED_L2_SHAPE_K=S={S}  OK')

    # --- combine 语义：y = Σ_k w_k * routed_k + shared （共享专家无权重，只加一次）---
    num_experts, topk = 4, 2
    We1 = torch.randn(num_experts, 2 * I, H, dtype=torch.bfloat16, device=dev) * 0.05
    We2 = torch.randn(num_experts, H, I, dtype=torch.bfloat16, device=dev) * 0.05
    topk_idx = torch.stack([torch.randperm(num_experts, device=dev)[:topk] for _ in range(T)])
    topk_w = torch.rand(T, topk, device=dev)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(torch.bfloat16)

    # 真 kernel 逐专家做路由专家（简化为按槽位 gather 的 dense GEMM）
    routed_out = torch.zeros(T, H, dtype=torch.float32, device=dev)      # combine 的 fp32 累加
    for k in range(topk):
        idx = topk_idx[:, k]
        xg = torch.index_select(x, 0, idx)
        l1g = torch.empty(T, 2 * I, dtype=torch.bfloat16, device=dev)
        # 逐 token 的专家不同 -> 用 index_select on weight 不便；逐专家分组做
        for e in range(num_experts):
            mask = idx == e
            ne = int(mask.sum())
            if ne == 0:
                continue
            xe = x[mask]
            l1e = torch.empty(ne, 2 * I, dtype=torch.bfloat16, device=dev)
            deep_gemm.bf16_gemm_nt(xe, We1[e], l1e)
            ae = (silu(l1e[:, :I].float()) * l1e[:, I:].float()).to(torch.bfloat16)
            ye = torch.empty(ne, H, dtype=torch.bfloat16, device=dev)
            deep_gemm.bf16_gemm_nt(ae, We2[e], ye)
            routed_out[mask] += ye.float() * topk_w[mask, k].float().unsqueeze(-1)

    y_combined = routed_out + y_shared.float()
    # fp64 参考
    routed_64 = torch.zeros(T, H, dtype=torch.float64, device=dev)
    for k in range(topk):
        idx = topk_idx[:, k]
        for e in range(num_experts):
            mask = idx == e
            if int(mask.sum()) == 0:
                continue
            xe = x[mask].double()
            l1e = xe @ We1[e].double().t()
            ae = silu(l1e[:, :I]) * l1e[:, I:]
            ye = ae @ We2[e].double().t()
            routed_64[mask] += ye * topk_w[mask, k].double().unsqueeze(-1)
    y_comb_ref = routed_64 + y_ref
    relc = (y_combined.double() - y_comb_ref).norm() / y_comb_ref.norm()
    print(f'  combine: y = Σ_k w_k·routed_k + shared（共享项不加权、只加一次） vs fp64: '
          f'relerr={relc.item():.3e} {"OK (<3e-2)" if relc.item() < 3e-2 else "FAIL"}')
    all_ok &= relc.item() < 3e-2

    # --- 反例：把共享项错误地「按每个 top-k 槽加一次」（重复 topk 次），误差应显著变大 ---
    y_wrong = routed_out + y_shared.float() * topk
    relw = (y_wrong.double() - y_comb_ref).norm() / y_comb_ref.norm()
    print(f'  反例（共享项重复 topk={topk} 次）: relerr={relw.item():.3e} '
          f'{"OK（显著大于正确解）" if relw.item() > relc.item() else "FAIL（反例未放大）"}')
    all_ok &= relw.item() > relc.item()

    return all_ok


def main():
    if not torch.cuda.is_available():
        print('CUDA 不可用')
        return 1
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    from deep_gemm.testing.utils import get_arch_major
    print(f'GPU: {torch.cuda.get_device_name(0)}  SMs={num_sms}  arch_major={get_arch_major()}')

    ok_a = run_host_view_checks()
    ok_b = run_scheduler_checks(num_sms)
    ok_c = run_utccp_mapping_check()
    ok_d = run_numeric(num_sms)

    print()
    print('=' * 78)
    all_ok = ok_a and ok_b and ok_c and ok_d
    print(f'  分项: A(host view)={ok_a}  B(schedule)={ok_b}  C(utccp)={ok_c}  D(numeric)={ok_d}')
    print('总判定：', 'ALL PASS' if all_ok else 'HAS FAILURE !!')
    print('=' * 78)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
