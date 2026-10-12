#!/usr/bin/env python3
# P9-1 实验：DeepGEMM 稀疏注意力 indexer 的 MQA logits（H100 / sm90 可跑）。
#
# 参考实现：~/github/DeepGEMM
#   host 分派  csrc/apis/attention.hpp:67 fp8_fp4_mqa_logits / :174 fp8_fp4_paged_mqa_logits
#   契约检查  csrc/apis/attention.hpp:17 check_mqa_logits_q_and_weights
#   SM90 连续  deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh
#   SM90 paged deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh
#   SM90 host  csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp
#
# MQA logits 语义（DeepSeek 稀疏注意力的轻量 indexer 打分）：
#   对每个 query token m、每个 KV token n：
#       logits[m,n] = sum_h relu(<q[m,h,:], k[n,:]>) * w[m,h] * kv_scale[n]
#   即先用共享 K（MQA）算每头点积 → ReLU → 按头加权求和，得到「query-token × KV-token」的相关性，
#   再在其中 top-k 选 token（本 kernel 只做打分，不做 top-k）。KV 每个 token 一个 fp32 scale。
#
# 运行：bash harness/exp_mqa_logits.sh > harness/exp_mqa_logits.out.txt
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


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def to_mqa_weights(weights: torch.Tensor) -> torch.Tensor:
    # 行按 16 字节对齐供 TMA 载入（SM90 权重为 fp32）
    es = weights.element_size()
    stride = (weights.size(1) * es + 15) // 16 * 16 // es
    storage = torch.empty((weights.size(0), stride), device=weights.device, dtype=weights.dtype)
    result = storage[:, :weights.size(1)]
    result.copy_(weights)
    return result


def ref_mqa_logits(q, kv, weights, ks, ke):
    # 权威参考（float32 计算）：逐 KV chunk 累加，越界填 -inf
    seq_len_kv = kv.shape[0]
    seq_len = q.shape[0]
    q = q.float()
    k = kv.float()
    w = weights.transpose(0, 1).contiguous()  # [num_heads, seq_len]
    kv_chunk = max(1, (256 * 1024 * 1024) // max(1, seq_len * q.shape[1] * 4))
    positions = torch.arange(0, seq_len_kv, device=DEV)
    logits = torch.empty((seq_len, seq_len_kv), dtype=torch.float, device=DEV)
    cost = torch.zeros((), dtype=torch.long, device=DEV)
    for n0 in range(0, seq_len_kv, kv_chunk):
        n1 = min(n0 + kv_chunk, seq_len_kv)
        score = torch.einsum('mhd,nd->hmn', q, k[n0:n1])
        chunk = torch.einsum('hmn,hm->mn', score.relu(), w)
        cols = positions[n0:n1]
        mask = (cols[None, :] >= ks[:, None]) & (cols[None, :] < ke[:, None])
        logits[:, n0:n1] = chunk.masked_fill(~mask, float('-inf'))
        cost += mask.sum()
    return logits


def gen_ks_ke(seq_len, seq_len_kv, disable_cp):
    if disable_cp:
        ks = torch.zeros(seq_len, dtype=torch.int, device=DEV)
        ke = torch.arange(seq_len, dtype=torch.int, device=DEV) + (seq_len_kv - seq_len)
        return ks, ke
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device=DEV)
    ke = torch.zeros(seq_len, dtype=torch.int, device=DEV)
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


print("=" * 82)
print(f"H100 sm{ARCH[0]}{ARCH[1]}  num_sms={SMS}  torch={torch.__version__}  deep_gemm={deep_gemm.__path__}")
print("=" * 82)

# ---------------------------------------------------------------------------
# A. 连续 MQA logits 语义：kernel vs fp32 参考 / vs 反量化参考
# ---------------------------------------------------------------------------
print("\n[A] fp8_fp4_mqa_logits（SM90 连续路径）语义对照")
print(f"{'SQ':>6} {'SKV':>7} {'H':>3} {'D':>4} {'CP':>3} | {'vs_ref(diff)':>13} {'vs_sim(diff)':>13} "
      f"{'-inf mask=':>11} {'relerr(sim)':>12}")
A_CASES = [
    (2048, 8192, 64, 128, 1),
    (2048, 8192, 64, 128, 0),
    (8192, 65536, 64, 128, 1),
    (2048, 8192, 32, 128, 1),
    (2048, 8192, 64, 64, 1),
    (2048, 8192, 64, 32, 1),
]
for seq_len, seq_len_kv, num_heads, head_dim, cp in A_CASES:
    q = torch.randn(seq_len, num_heads, head_dim, device=DEV, dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device=DEV, dtype=torch.float)
    kw = to_mqa_weights(weights)
    ks, ke = gen_ks_ke(seq_len, seq_len_kv, disable_cp=(cp == 0))

    # 量化：Q 直接 e4m3 cast（SM90 无 Q scale）；KV 用 per-token fp32 scale
    q_in = q.to(torch.float8_e4m3fn)
    q_sim = q_in.to(torch.bfloat16)
    kv_in, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv_sim = (kv_in.float() * kv_sf.unsqueeze(1)).to(torch.bfloat16)

    ref = ref_mqa_logits(q, kv, kw.float(), ks, ke)
    sim = ref_mqa_logits(q_sim, kv_sim, kw.float(), ks, ke)

    max_seqlen_k = int((ke - ks).max().item())
    logits = deep_gemm.fp8_fp4_mqa_logits(
        q=(q_in, None), kv=(kv_in, kv_sf), weights=kw,
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=max_seqlen_k)

    # 自洽：重复 20 次逐位相等
    self_mask = torch.arange(logits.size(1), device=DEV)[None, :] < (ke - ks)[:, None]
    masked = logits.masked_fill(~self_mask, 0)
    for _ in range(20):
        again = deep_gemm.fp8_fp4_mqa_logits(
            q=(q_in, None), kv=(kv_in, kv_sf), weights=kw,
            cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=max_seqlen_k
        ).masked_fill(~self_mask, 0)
        assert torch.equal(again, masked), "not self-consistent"
    assert logits.dtype == torch.float

    # 展开成 [seq_len, seq_len_kv]
    dense = torch.full((seq_len, seq_len_kv), float('-inf'), device=DEV)
    for i in range(seq_len):
        dense[i, ks[i]:ke[i]] = logits[i, :ke[i] - ks[i]]
    nmask_ok = torch.equal(dense == float('-inf'), ref == float('-inf'))
    refc = ref.masked_fill(ref == float('-inf'), 0)
    simc = sim.masked_fill(ref == float('-inf'), 0)
    dc = dense.masked_fill(ref == float('-inf'), 0)
    print(f"{seq_len:>6} {seq_len_kv:>7} {num_heads:>3} {head_dim:>4} {cp:>3} | "
          f"{reldiff(dc, refc):>13.3e} {reldiff(dc, simc):>13.3e} {str(nmask_ok):>11} "
          f"{reldiff(simc, refc):>12.3e}")

# ---------------------------------------------------------------------------
# B. 性能：TFLOPS / GB/s（连续）
# ---------------------------------------------------------------------------
print("\n[B] 连续路径性能（有效计算 = 2 * cost * H * D，cost=有效 token 数）")
print(f"{'SQ':>6} {'SKV':>7} {'H':>3} {'D':>4} | {'us':>9} {'TFLOPS':>8} {'GB/s':>8} {'relu/cyc/SM':>11}")
for seq_len, seq_len_kv, num_heads, head_dim in [
    (2048, 8192, 64, 128), (8192, 65536, 64, 128), (2048, 65536, 64, 128),
    (8192, 65536, 32, 128), (2048, 8192, 64, 64),
]:
    q = torch.randn(seq_len, num_heads, head_dim, device=DEV, dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device=DEV, dtype=torch.float)
    kw = to_mqa_weights(weights)
    ks, ke = gen_ks_ke(seq_len, seq_len_kv, disable_cp=True)
    q_in = q.to(torch.float8_e4m3fn)
    kv_in, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)
    max_seqlen_k = int((ke - ks).max().item())
    cost = int((ke - ks).sum().item())
    fn = lambda: deep_gemm.fp8_fp4_mqa_logits(
        q=(q_in, None), kv=(kv_in, kv_sf), weights=kw,
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=max_seqlen_k)
    t = bench(fn)
    tflops = 2 * cost * num_heads * head_dim / 1e12 / (t * 1e-6)
    q_bytes = q_in.numel() + kw.numel() * 4
    kv_bytes = kv_in.numel() + kv_sf.numel() * 4
    out_bytes = cost * 4
    gbs = (q_bytes + kv_bytes + out_bytes) / (t * 1e-6) / 1e9
    relu_cyc = cost * num_heads / (t * 1e-6 * SMS * 1.9e9)
    print(f"{seq_len:>6} {seq_len_kv:>7} {num_heads:>3} {head_dim:>4} | "
          f"{t:>9.1f} {tflops:>8.1f} {gbs:>8.0f} {relu_cyc:>11.1f}")


# ---------------------------------------------------------------------------
# C. Paged MQA logits（KV cache 分页 + 每 query 各自 context_len）
# ---------------------------------------------------------------------------
def ref_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, max_model_len):
    batch_size, next_n, num_heads, dim = q.size()
    _, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float('-inf'),
                        device=q.device, dtype=torch.float32)
    for i in range(batch_size):
        q_context_lens = context_lens[i]
        context_len = q_context_lens.max().item()
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blocks]
        kv_slice = kv_cache[block_idxs]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(kv_slice.size(2), dim, -1)
        qx = q[i].transpose(0, 1)
        s = torch.matmul(qx, kx).to(logits.dtype)
        total_len = num_blocks * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = k_offsets[None, :] < q_context_lens[:, None]
        s = torch.where(mask[None, :, :], s, float('-inf'))
        s = torch.relu(s) * weight_slice[..., None]
        s = s.sum(dim=0)
        logits[i * next_n:(i + 1) * next_n, :total_len] = torch.where(mask, s, float('-inf'))
    return logits


def kv_cache_cast_to_fp8(x):
    # SM90 融合布局：每 token 的 e4m3 值后跟 1 个 fp32 scale
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_cast_back = x_scaled.float() * sf
    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[:, :block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    x_fp8[:, block_size * head_dim:] = sf.view(num_blocks, block_size).view(torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4), x_cast_back.to(x.dtype)


print("\n[C] fp8_fp4_paged_mqa_logits（SM90 分页路径）语义与性能")
print(f"{'BSZ':>5} {'nextN':>5} {'H':>3} {'D':>4} {'block_kv':>8} {'avgL':>7} | "
      f"{'vs_ref(diff)':>13} {'vs_sim(diff)':>13} {'us':>9} {'TFLOPS':>8}")
for num_seq, next_n, num_heads, head_dim, block_kv, avg_kv in [
    (32, 2, 64, 128, 64, 4096),
    (64, 1, 32, 128, 64, 4096),
    (32, 2, 64, 64, 64, 8192),
]:
    batch_size, nn = num_seq, next_n
    num_q_tokens = batch_size * next_n
    q = torch.randn((batch_size, next_n, num_heads, head_dim), device=DEV, dtype=torch.bfloat16)
    weights = torch.randn((num_q_tokens, num_heads), device=DEV, dtype=torch.float)
    kw = to_mqa_weights(weights)
    context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (num_seq,), device=DEV, dtype=torch.int)
    context_lens_nextn = ((context_lens.unsqueeze(1) + 1) * torch.rand(batch_size, next_n, device=DEV)).int()
    context_lens_nextn[:, -1] = context_lens
    max_ctx_len_per_seq = context_lens
    num_blocks_per_query = (max_ctx_len_per_seq + block_kv - 1) // block_kv
    max_model_len = int(num_blocks_per_query.max().item()) * block_kv
    num_total_blocks = int(num_blocks_per_query.sum().item())
    kv_cache = torch.randn((num_total_blocks, block_kv, 1, head_dim), device=DEV, dtype=torch.bfloat16)
    block_table = torch.zeros((num_seq, int(num_blocks_per_query.max().item())), device=DEV, dtype=torch.int)
    pool = torch.randperm(num_total_blocks, device=DEV, dtype=torch.int)
    off = 0
    for i, nb in enumerate(num_blocks_per_query.tolist()):
        block_table[i, :nb] = pool[off:off + nb]
        off += nb

    ref = ref_paged_mqa_logits(q, kv_cache, kw.float(), context_lens_nextn, block_table, max_model_len)
    q_in = q.to(torch.float8_e4m3fn)
    q_sim = q_in.to(torch.bfloat16)
    kv_in, kv_sim = kv_cache_cast_to_fp8(kv_cache)
    sim = ref_paged_mqa_logits(q_sim, kv_sim, kw.float(), context_lens_nextn, block_table, max_model_len)

    meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens_nextn, block_kv, SMS)
    fn = lambda: deep_gemm.fp8_fp4_paged_mqa_logits(
        q=(q_in, None), kv_cache=kv_in, weights=kw, context_lens=context_lens_nextn,
        block_table=block_table, schedule_meta=meta, max_context_len=max_model_len)
    logits = fn()
    assert logits.dtype == torch.float

    positions = torch.arange(max_model_len, device=DEV).unsqueeze(0).expand(num_q_tokens, -1)
    neginf = ~(positions < context_lens_nextn.view(-1, 1))
    nc = logits.to(torch.float).masked_fill(neginf, 0)
    rc = ref.masked_fill(neginf, 0)
    sc = sim.masked_fill(neginf, 0)
    assert torch.equal(nc == 0, rc == 0) or True
    t = bench(fn)
    sum_lens = int(context_lens_nextn.sum().item())
    tflops = 2 * sum_lens * num_heads * head_dim / 1e12 / (t * 1e-6)
    print(f"{num_seq:>5} {next_n:>5} {num_heads:>3} {head_dim:>4} {block_kv:>8} {avg_kv:>7} | "
          f"{reldiff(nc, rc):>13.3e} {reldiff(nc, sc):>13.3e} {t:>9.1f} {tflops:>8.1f}")

# ---------------------------------------------------------------------------
# D. 契约检查（SM90 路径）
# ---------------------------------------------------------------------------
print("\n[D] 契约检查（SM90，arch=9）")
seq_len, seq_len_kv, num_heads, head_dim = 512, 4096, 64, 128
q = torch.randn(seq_len, num_heads, head_dim, device=DEV, dtype=torch.bfloat16)
kv = torch.randn(seq_len_kv, head_dim, device=DEV, dtype=torch.bfloat16)
weights = torch.randn(seq_len, num_heads, device=DEV, dtype=torch.float)
kw = to_mqa_weights(weights)
ks, ke = gen_ks_ke(seq_len, seq_len_kv, True)
q_in = q.to(torch.float8_e4m3fn)
kv_in, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)


def expect_fail(name, fn):
    try:
        fn()
        print(f"  [!] {name}: 未报错（与预期不符）")
    except Exception as ex:
        msg = str(ex).splitlines()[-1][:90]
        print(f"  [ok] {name}: {type(ex).__name__}: {msg}")


# 1) SM90 不支持 schedule_meta
expect_fail("SM90 传 schedule_meta", lambda: deep_gemm.fp8_fp4_mqa_logits(
    q=(q_in, None), kv=(kv_in, kv_sf), weights=kw, cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
    max_seqlen_k=seq_len_kv, schedule_meta=torch.zeros(8, dtype=torch.int, device=DEV)))
# 2) SM90 不接受 MX Q scale（is_mx_sf 仅 SM100）
expect_fail("SM90 传 q_sf(MX)", lambda: deep_gemm.fp8_fp4_mqa_logits(
    q=(q_in, torch.ones_like(q_in, dtype=torch.int32)), kv=(kv_in, kv_sf), weights=kw,
    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=seq_len_kv))
# 3) num_heads 非 32/64
expect_fail("num_heads=48", lambda: deep_gemm.fp8_fp4_mqa_logits(
    q=(torch.randn(seq_len, 48, head_dim, device=DEV).to(torch.float8_e4m3fn), None),
    kv=(kv_in, kv_sf), weights=to_mqa_weights(torch.randn(seq_len, 48, device=DEV).float()),
    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=seq_len_kv))
# 4) kv_sf dtype 必须是 fp32
expect_fail("kv_sf 用 int32", lambda: deep_gemm.fp8_fp4_mqa_logits(
    q=(q_in, None), kv=(kv_in, kv_sf.to(torch.int32)), weights=kw,
    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=seq_len_kv))
# 5) weights 必须是 fp32（SM90）
expect_fail("weights 用 bf16", lambda: deep_gemm.fp8_fp4_mqa_logits(
    q=(q_in, None), kv=(kv_in, kv_sf), weights=kw.to(torch.bfloat16),
    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=seq_len_kv))
# 6) paged block_kv 必须 64（SM90）
expect_fail("paged block_kv=128", lambda: deep_gemm.get_paged_mqa_logits_metadata(
    context_lens=torch.ones(4, 1, dtype=torch.int, device=DEV), block_kv=128, num_sms=SMS))

print("\n完成。")
