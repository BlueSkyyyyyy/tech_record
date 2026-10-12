#!/usr/bin/env python3
# P12-1 实验：DeepGEMM 稀疏注意力 indexer 的「稀疏 MQA logits」语义（H100 上只做架构无关部分）。
#
# 参考实现：~/github/DeepGEMM  commit 057ca59
#   host API     csrc/apis/attention.hpp:284-442（get_sparse_mqa_logits_metadata / fp8_fp4_sparse_mqa_logits
#                / paged 变体），注册 :445-476
#   jit wrapper  csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp:15-202
#   layout       deep_gemm/include/deep_gemm/layout/sparse_mqa_logits.cuh
#   metadata 调度 deep_gemm/include/deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh
#   主 kernel     deep_gemm/include/deep_gemm/impls/sm100_sparse_mqa_logits.cuh
#
# 稀疏 MQA logits 语义（DeepSeek 稀疏注意力 indexer 的第二级打分）：
#   indexer 先用一个轻量网络给「query-token × KV-块」打粗分并选出 top-k 个 KV 块
#   （即 sparse_kv_block_indices，每行按块号升序、前缀有效），本算子在**被选中的稀疏块内**
#   计算细粒度 logits：
#       logits_out[q, s*SPARSE_BLOCK_KV + i]
#           = Σ_h relu(<q[q,h,:], kv[block_q[s]*SPARSE_BLOCK_KV + i, :]>) * w[q,h]
#   即把 dense MQA logits 在被选位置的取值按「槽 s / 块内偏移 i」**压缩**存进 [T, NB*SB] 的输出。
#   主 kernel 是 **SM100 专属**（block-scaled UMMA / MXFP8 / MXFP4），H100 **跑不了**；
#   本实验在 H100 上用真实 dense SM90 kernel 取到全量 logits，再验证：
#     (A) metadata 尺寸/布局公式；(B) 稀疏选择+两 token 合并去重+压缩列映射；(C) unaligned ks 偏移；
#     (D) host 契约在 sm90 被拒（坐实 SM100-only）。
#
# 运行：bash harness/exp_sparse_mqa_logits.sh > harness/exp_sparse_mqa_logits.out.txt
import torch
import deep_gemm
from deep_gemm.utils import per_custom_dims_cast_to_fp8

torch.manual_seed(0)
DEV = "cuda"
SMS = torch.cuda.get_device_properties(0).multi_processor_count
ARCH = torch.cuda.get_device_capability()  # (9, 0) on H100


def reldiff(x, ref):
    denom = ref.abs().max().item()
    return (x.double() - ref.double()).abs().max().item() / (denom if denom > 0 else 1.0)


def align(x, a):
    return (x + a - 1) // a * a


def ceil_div(x, y):
    return -(-x // y)


def to_mqa_weights(weights: torch.Tensor) -> torch.Tensor:
    es = weights.element_size()
    stride = (weights.size(1) * es + 15) // 16 * 16 // es
    storage = torch.empty((weights.size(0), stride), device=weights.device, dtype=weights.dtype)
    storage[:, :weights.size(1)].copy_(weights)
    return storage


def ref_mqa_logits(q, kv, weights, ks, ke):
    # 权威 fp32 参考：逐 KV chunk 累加，越界填 -inf
    seq_len_kv = kv.shape[0]
    seq_len = q.shape[0]
    q = q.float()
    k = kv.float()
    w = weights.transpose(0, 1).contiguous()  # [H, T]
    kv_chunk = max(1, (256 * 1024 * 1024) // max(1, seq_len * q.shape[1] * 4))
    positions = torch.arange(0, seq_len_kv, device=DEV)
    logits = torch.empty((seq_len, seq_len_kv), dtype=torch.float, device=DEV)
    for n0 in range(0, seq_len_kv, kv_chunk):
        n1 = min(n0 + kv_chunk, seq_len_kv)
        score = torch.einsum('mhd,nd->hmn', q, k[n0:n1])
        chunk = torch.einsum('hmn,hm->mn', score.relu(), w)
        cols = positions[n0:n1]
        mask = (cols[None, :] >= ks[:, None]) & (cols[None, :] < ke[:, None])
        logits[:, n0:n1] = chunk.masked_fill(~mask, float('-inf'))
    return logits


print("=" * 84)
print(f"H100 sm{ARCH[0]}{ARCH[1]}  num_sms={SMS}  torch={torch.__version__}  deep_gemm={deep_gemm.__path__}")
print("=" * 84)

# ---------------------------------------------------------------------------
# A. metadata 布局与尺寸公式（对应 layout/sparse_mqa_logits.cuh + jit hpp:20-36）
# ---------------------------------------------------------------------------
print("\n[A] sparse MQA metadata 布局/尺寸公式复刻")
print("  struct 尺寸：MetadataHeader=16 KVSplitHeader=16 KVBlockInfo=8 ScheduleEntry=16 "
      "WorkspaceState=384 QBlockInfo=8")


def get_sparse_split_kv(is_fp4):
    # jit_kernels/impls/sm100_sparse_mqa_logits.hpp:15-18
    return 640 if is_fp4 else 512


def num_metadata_bytes(num_q_tokens, num_max_sparse_blocks, sparse_block_kv, split_kv, is_paged, num_sms):
    # 复刻 jit_kernels/impls/sm100_sparse_mqa_logits.hpp:20-36
    assert num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096
    assert sparse_block_kv in (8, 16)
    num_kv_blocks_per_split = split_kv // sparse_block_kv
    num_kv_split_bytes = 16 + num_kv_blocks_per_split * 8
    if is_paged:
        num_max_kv_splits = num_q_tokens * ceil_div(num_max_sparse_blocks, num_kv_blocks_per_split)
    else:
        num_max_kv_splits = ceil_div(num_q_tokens, 2) * ceil_div(2 * num_max_sparse_blocks, num_kv_blocks_per_split)
    num_max_schedule_entries = align(num_max_kv_splits, num_sms)
    return 16 + num_max_kv_splits * num_kv_split_bytes + num_max_schedule_entries * 16


print(f"  {'dtype':>5} {'SPLIT_KV':>9} {'SB':>3} {'KVBlocks/Split':>15} | "
      f"{'T':>6} {'NB':>5} {'paged':>6} {'bytes':>10} {'splits':>8} {'entry_bytes':>11}")
A_CASES = [
    (False, 8, 4096, 64, False),
    (False, 16, 4096, 64, False),
    (True, 8, 4096, 64, False),
    (True, 16, 4096, 64, False),
    (False, 8, 2048, 32, True),
    (False, 16, 8192, 128, False),
]
for is_fp4, sb, T, NB, paged in A_CASES:
    split = get_sparse_split_kv(is_fp4)
    nkb = split // sb
    b = num_metadata_bytes(T, NB, sb, split, paged, SMS)
    n_splits = (T * ceil_div(NB, nkb)) if paged else (ceil_div(T, 2) * ceil_div(2 * NB, nkb))
    print(f"  {'fp4' if is_fp4 else 'fp8':>5} {split:>9} {sb:>3} {nkb:>15} | "
          f"{T:>6} {NB:>5} {str(paged):>6} {b:>10} {n_splits:>8} {b // max(1, T):>11}")

# ---------------------------------------------------------------------------
# B. 稀疏选择 + 两 token 合并去重 + 压缩列映射（H100 dense SM90 kernel 取真值）
# ---------------------------------------------------------------------------
print("\n[B] 稀疏选择语义（用真实 SM90 dense kernel 取全量 logits，再按稀疏块抽取/压缩）")
print(f"  {'T':>5} {'S':>6} {'H':>3} {'D':>4} {'SB':>3} {'NB':>4} {'blk/req':>8} | "
      f"{'vs_ref(full)':>12} {'gather==L':>10} {'relerr(sel)':>11} {'压缩比':>7} {'去重省':>7}")


def build_sparse_indices(nblk_per_token, candidate_blocks_per_token, NB_max, device):
    # 每 token 的稀疏块号：升序、唯一、落在有效块范围内；前 nblk 有效，其余 padding
    idx = torch.zeros((len(nblk_per_token), NB_max), dtype=torch.int, device=device)
    for t, (nb, ncb) in enumerate(zip(nblk_per_token, candidate_blocks_per_token)):
        if nb > 0:
            perm = torch.randperm(int(ncb), device=device)[:nb].sort().values
            idx[t, :nb] = perm
    return idx


for T, S, H, D, SB, NB in [
    (64, 4096, 64, 128, 8, 16),
    (64, 4096, 32, 64, 16, 32),
    (128, 8192, 64, 128, 8, 64),
]:
    num_kv_blocks = ceil_div(S, SB)
    # 每 token 有效 KV 长度（对齐到 SB），前一半 token 短一半长
    ke = torch.empty(T, dtype=torch.int, device=DEV)
    half = T // 2
    ke[:half] = align(int(S * 0.75), SB)
    ke[half:] = S
    ks = torch.zeros(T, dtype=torch.int, device=DEV)
    nblk = torch.clamp((ke - ks + SB - 1) // SB, max=min(NB, num_kv_blocks)).to(torch.int)
    # indexer 选出的块：每 token 在有效范围内取 nblk 个块（候选窗口限 4*NB，制造块重叠）
    cand = ((ke - ks + SB - 1) // SB).clamp(max=min(num_kv_blocks, 4 * NB)).tolist()
    sparse_idx = build_sparse_indices(nblk.tolist(), cand, NB, DEV)

    q = torch.randn(T, H, D, device=DEV, dtype=torch.bfloat16)
    kv = torch.randn(S, D, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(T, H, device=DEV, dtype=torch.float)
    kw = to_mqa_weights(weights)

    q_in = q.to(torch.float8_e4m3fn)
    q_sim = q_in.to(torch.bfloat16)
    kv_in, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv_sim = (kv_in.float() * kv_sf.unsqueeze(1)).to(torch.bfloat16)

    ref = ref_mqa_logits(q, kv, kw.float(), ks, ke)
    max_seqlen_k = int((ke - ks).max().item())
    L = deep_gemm.fp8_fp4_mqa_logits(
        q=(q_in, None), kv=(kv_in, kv_sf), weights=kw,
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=max_seqlen_k)
    assert L.dtype == torch.float

    # 展开成 [T, S]
    dense = torch.full((T, S), float('-inf'), device=DEV)
    for t in range(T):
        dense[t, ks[t]:ke[t]] = L[t, :ke[t] - ks[t]]
    full_rel = reldiff(dense.masked_fill(dense == float('-inf'), 0),
                       ref.masked_fill(ref == float('-inf'), 0))

    # 压缩输出：out[t, s*SB+i] = dense[t, idx[t,s]*SB+i]  （对齐情形 block 起点即 blk*SB）
    out = torch.zeros(T, NB * SB, device=DEV)
    for t in range(T):
        nb = int(nblk[t].item())
        for s in range(nb):
            b = int(sparse_idx[t, s].item())
            out[t, s * SB:(s + 1) * SB] = dense[t, b * SB:(b + 1) * SB]
    # gather 是否等于 dense 对应位置（逐位）
    gather_eq = True
    for t in range(T):
        nb = int(nblk[t].item())
        for s in range(nb):
            b = int(sparse_idx[t, s].item())
            if not torch.equal(out[t, s * SB:(s + 1) * SB], dense[t, b * SB:(b + 1) * SB]):
                gather_eq = False
    # 独立 fp32 参考：在被选位置从 `ref`（fp32 全量）gather
    ref_sel = torch.zeros(T, NB * SB, device=DEV)
    for t in range(T):
        nb = int(nblk[t].item())
        blocks = sparse_idx[t, :nb].long()
        cols = (blocks[:, None] * SB + torch.arange(SB, device=DEV)[None, :]).reshape(-1)
        ref_sel[t, :nb * SB] = ref[t, cols]
    minnblk = int(nblk.min().item())
    sel_rel = reldiff(out[:, :minnblk * SB], ref_sel[:, :minnblk * SB])

    # 两 token 合并去重：Q-block 内 KV 块并集 vs 各自之和
    saves, sums = 0, 0
    for t0 in range(0, T, 2):
        na = int(nblk[t0].item())
        nb = int(nblk[min(t0 + 1, T - 1)].item())
        ua = set(sparse_idx[t0, :na].tolist())
        ub = set(sparse_idx[min(t0 + 1, T - 1), :nb].tolist())
        sums += na + nb
        saves += (na + nb) - len(ua | ub)
    compute = int((nblk * SB).sum().item())
    dense_cost = int((ke - ks).sum().item())
    print(f"  {T:>5} {S:>6} {H:>3} {D:>4} {SB:>3} {NB:>4} {num_kv_blocks:>8} | "
          f"{full_rel:>12.3e} {str(gather_eq):>10} {sel_rel:>11.3e} "
          f"{dense_cost / compute:>6.2f}x {saves:>5}/{sums:<4}")

# ---------------------------------------------------------------------------
# C. unaligned ks 偏移语义：块 i 覆盖 [i*SB+off, (i+1)*SB+off)，off = ks % SB
# ---------------------------------------------------------------------------
print("\n[C] use_unaligned_ks 偏移语义（块起点 = 逻辑块号*SB + ks%SB，跨块边界）")
T, S, SB = 8, 4096, 8
ks = torch.full((T,), 5, dtype=torch.int, device=DEV)   # off = 5
ke = torch.full((T,), S, dtype=torch.int, device=DEV)
off = int(ks[0].item()) % SB
# 逻辑块号 3 覆盖的绝对 KV token 区间
for blk in [0, 3, 100]:
    lo = blk * SB + off
    print(f"  逻辑块 {blk:>4}: 绝对 token [{lo}, {lo + SB})   "
          f"（对齐语义下块内首 token = {blk * SB}，偏移 {off} 使块跨原块边界）")
print(f"  indexer 需保证块号升序且落在 [0, ceil((ke-ks)/SB))：off 使首块可能回看 ks 之前的 token")

# ---------------------------------------------------------------------------
# D. host 契约：稀疏 API 在 sm90 被拒（SM100 专属）
# ---------------------------------------------------------------------------
print("\n[D] host 契约（sm90 应全部因 arch_major==10 断言被拒）")


def expect_fail(name, fn):
    try:
        fn()
        print(f"  [!] {name}: 未报错（与预期不符）")
    except Exception as ex:
        msg = str(ex).splitlines()[-1][:96]
        print(f"  [ok] {name}: {type(ex).__name__}: {msg}")


T = 32
q_in = torch.randn(T, 64, 128, device=DEV).to(torch.float8_e4m3fn)
kv_in, kv_sf = per_custom_dims_cast_to_fp8(torch.randn(S, 128, device=DEV, dtype=torch.bfloat16), (0,), False)
kw = to_mqa_weights(torch.randn(T, 64, device=DEV).float())
ks = torch.zeros(T, dtype=torch.int, device=DEV)
ke = torch.full((T,), S, dtype=torch.int, device=DEV)
sparse_idx = build_sparse_indices([16] * T, [S // 8] * T, 16, DEV)

expect_fail("get_sparse_mqa_logits_metadata", lambda: deep_gemm.get_sparse_mqa_logits_metadata(
    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, num_kv_tokens=S,
    sparse_kv_block_indices=sparse_idx, qk_dtype=torch.float8_e4m3fn,
    sparse_block_kv=8, use_unaligned_ks=False))
expect_fail("fp8_fp4_sparse_mqa_logits", lambda: deep_gemm.fp8_fp4_sparse_mqa_logits(
    q=(q_in, None), kv=(kv_in, kv_sf), weights=kw,
    metadata=torch.zeros(16, dtype=torch.uint8, device=DEV),
    num_max_sparse_blocks=16, sparse_block_kv=8, use_unaligned_ks=False))
expect_fail("get_paged_sparse_mqa_logits_metadata", lambda: deep_gemm.get_paged_sparse_mqa_logits_metadata(
    context_lens=torch.ones(T, dtype=torch.int, device=DEV) * 4096,
    block_table=torch.zeros(T, 64, dtype=torch.int, device=DEV),
    indices=torch.zeros(T, dtype=torch.int, device=DEV),
    page_kv=64, sparse_kv_block_indices=sparse_idx,
    qk_dtype=torch.float8_e4m3fn, sparse_block_kv=8))

print("\n完成。")
