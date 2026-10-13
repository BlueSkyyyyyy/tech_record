"""P95-1 实验：Mega MoE 的 FP4 路由权重路径（`fp8xfp4` / `kPackedFP4`）。

Mega MoE 的 FP8/FP4 主 kernel（`sm100_fp8_fp4_mega_moe.cuh`）是 SM100 专属、本机跑不了，
但「FP4 路由权重长什么样、host 契约怎么定、packed 表示怎么进 TMA/SMEM、权重重排是否与
packing 相容、FP4 相对 FP8 精度/显存差多少」这些**是架构无关的 host + 张量语义**，可在 H100 上
用真实 DeepGEMM host API + 参考量化器直接核对。

四组实验：
  1) host 配置面（真实 `_C`）：`symm buffer` 尺寸 fp8xfp8 == fp8xfp4、block_m 与 mma_type 无关、
     权重 SF 是 int32 packed UE8M0。
  2) packed FP4 表示契约：`per_token_cast_to_fp4(gran_k=32)` → int8 `[N, K/2]`、逻辑 K 翻倍、
     与 `cast_back_from_fp4` 逐位往返。
  3) 权重重排：`transform_weights_for_mega_moe` 的 FP8/FP4 tuple 路径对其 packed int8 权重与
     int32 SF 的作用，并证明「N 轴 gran-8 交错」与「K 轴 2-per-byte packing」**可交换**（逐位）。
  4) 精度/显存：小 MoE 上 FP8 权重 vs FP4 权重（激活都 FP8）端到端误差与权重字节数对比。

来源核对（`~/github/DeepGEMM`，公开版 commit `057ca59`）：
  - csrc/apis/mega_moe.hpp:155-283（入口/契约）、csrc/utils/math.hpp:11（kPackedFP4）
  - csrc/utils/layout.hpp:57-64,163-170（逻辑形状 / 权重形状）
  - csrc/jit_kernels/impls/runtime_utils.hpp:63-100,297-323（to_string / TMA dtype / 权重描述符）
  - csrc/jit_kernels/heuristics/mega_moe.hpp:73-82,112-117,214-229
  - deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:152-164,803-810,918
  - deep_gemm/mega/__init__.py:112-164、deep_gemm/utils/math.py:99-167
  - tests/test_mega_moe.py:102-119
不涉及 SM100 kernel，仅验证 host 侧表示与张量语义。
"""
import math
import sys

import torch


def align(x, y):
    return (x + y - 1) // y * y


def rel_err(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


# =================== 1) host 配置面（真实 _C） ===================
def part1_host_config():
    from deep_gemm import _C
    ok = True
    print('== 1) host 配置面（真实 _C，H100）==')

    # symm buffer：FP4 只在权重侧，激活恒 FP8，故 fp8xfp4 == fp8xfp8，与 bf16 不同
    shapes = [(8, 64, 4096, 8, 7168, 2048, 0), (8, 256, 4096, 8, 7168, 2048, 1)]
    for (R, E, maxT, K, H, I, S) in shapes:
        nb8, sl8 = _C.get_symm_buffer_size_for_mega_moe(R, E, maxT, K, H, I, 'fp8xfp8', 'swiglu', S)
        nb4, sl4 = _C.get_symm_buffer_size_for_mega_moe(R, E, maxT, K, H, I, 'fp8xfp4', 'swiglu', S)
        nbb, _ = _C.get_symm_buffer_size_for_mega_moe(R, E, maxT, K, H, I, 'bf16xbf16', 'swiglu', S)
        flag = (nb8 == nb4) and (nbb != nb8)
        ok &= flag
        print(f'   R={R} E={E} S={S}: fp8xfp8={nb8/1e6:.1f}MB fp8xfp4={nb4/1e6:.1f}MB '
              f'bf16={nbb/1e6:.1f}MB  fp8xfp4==fp8xfp8:{nb8 == nb4} {"" if flag else "MISMATCH"}')

    # 激活/SF view dtype：x 是 fp8，x_sf/l1_acts_sf 是 int32（packed UE8M0）
    buf = torch.zeros(nb4, dtype=torch.int8, device='cuda')
    views = sl4(buf)
    x, x_sf = views[0], views[1]
    l1_acts, l1_acts_sf = views[8], views[9]
    l2_acts, l2_acts_sf = views[10], views[11]
    sf_int = (x_sf.dtype == torch.int32 and l1_acts_sf.dtype == torch.int32
              and l2_acts_sf.dtype == torch.int32)
    ok &= sf_int
    print(f'   view dtypes: x={x.dtype} l1_acts={l1_acts.dtype} '
          f'x_sf={x_sf.dtype} l1_acts_sf={l1_acts_sf.dtype} l2_acts_sf={l2_acts_sf.dtype} '
          f'(SF 全 packed UE8M0=int32: {sf_int})')
    print(f'   x_sf shape={tuple(x_sf.shape)}  l2_acts shape={tuple(l2_acts.shape)}')

    # block_m：fp8xfp4 与 fp8xfp8 完全一致（公式里没有 mma_kind）
    print('   block_m 启发式（heuristics/mega_moe.hpp:88-127）：')
    for (R, E, maxT, T, K) in [(8, 64, 4096, 8, 2), (8, 256, 4096, 64, 2),
                               (8, 8, 8192, 8192, 2), (8, 512, 4096, 128, 2)]:
        b8 = _C.get_block_m_for_mega_moe(R, E, maxT, T, K, 'fp8xfp8')
        b4 = _C.get_block_m_for_mega_moe(R, E, maxT, T, K, 'fp8xfp4')
        bb = _C.get_block_m_for_mega_moe(R, E, maxT, T, K, 'bf16xbf16')
        flag = (b8 == b4 == bb)
        ok &= flag
        print(f'     R={R} E={E} T={T} K={K}: block_m fp8xfp8={b8} fp8xfp4={b4} bf16={bb} '
              f'{"OK" if flag else "MISMATCH"}')
    return ok


# =================== 2) packed FP4 表示契约 ===================
def part2_packed_repr():
    from deep_gemm.utils.math import (per_token_cast_to_fp4, cast_back_from_fp4,
                                      _dequantize_from_fp4_e2m1)
    ok = True
    print('\n== 2) packed FP4 表示契约（deep_gemm.utils.math）==')
    torch.manual_seed(0)
    for (m, n) in [(256, 512), (128, 2048), (64, 7168)]:
        x = torch.randn(m, n, dtype=torch.bfloat16, device='cuda') * 0.1
        packed, sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
        # 契约：int8、[m, align(n,32)/2]（K 轴 2-per-byte），SF 数 = align(n,32)/32
        shape_ok = (packed.dtype == torch.int8 and packed.shape == (m, align(n, 32) // 2)
                    and sf.shape == (m, align(n, 32) // 32))
        # 逻辑 K 翻倍
        logical_k = packed.shape[-1] * 2
        # 往返：cast_back_from_fp4 == 手工 dequant(pack) * SF（gran 32）
        back = cast_back_from_fp4(packed, sf, gran_k=32)
        lo = (packed.to(torch.int16) & 0x0F).to(torch.int8)
        hi = ((packed.to(torch.int16) >> 4) & 0x0F).to(torch.int8)
        codes = torch.empty(m, packed.shape[1] * 2, dtype=torch.int8, device='cuda')
        codes[:, 0::2], codes[:, 1::2] = lo, hi
        manual = _dequantize_from_fp4_e2m1(codes)
        gi = torch.arange(codes.shape[1], device='cuda') // 32
        manual = manual * sf[:, gi]
        rt = torch.equal(back, manual)
        # SF 是 2 的幂（UE8M0 的 ceil），且 sf>=1e-4/6 的量化下限
        sf_pow2 = bool(torch.all((sf.view(torch.int32) & 0x7FFFFF) == 0))
        flag = shape_ok and rt and logical_k == n
        ok &= flag
        print(f'   m={m} n={n}: packed={tuple(packed.shape)} dtype={packed.dtype} '
              f'sf={tuple(sf.shape)} logical_K={logical_k} roundtrip_bit_exact={rt} '
              f'sf_is_pow2={sf_pow2} {"" if flag else "MISMATCH"}')
    return ok


# =================== 3) 权重重排与 packing 可交换 ===================
def part3_transform():
    from deep_gemm.mega import (transform_weights_for_mega_moe,
                                _interleave_weights, _transpose_sf_for_utccp)
    from deep_gemm.utils.math import per_token_cast_to_fp4
    ok = True
    print('\n== 3) transform_weights_for_mega_moe（FP4 tuple 分支）==')
    torch.manual_seed(1)
    E, I, H = 4, 512, 256          # l1: [E, 2I, H] ; l2: [E, H, I]
    l1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device='cuda') * 0.1
    l2 = torch.randn(E, H, I, dtype=torch.bfloat16, device='cuda') * 0.1

    # 造 packed FP4 权重 + fp32 SF（tests/test_mega_moe.py:103-110 的做法）
    def to_fp4_packed(w):
        g, n, k = w.shape
        pw = torch.empty(g, n, k // 2, dtype=torch.int8, device='cuda')
        sf = torch.empty(g, n, k // 32, dtype=torch.float, device='cuda')
        for i in range(g):
            pw[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
        return pw, sf

    l1w, l1sf = to_fp4_packed(l1)
    l2w, l2sf = to_fp4_packed(l2)
    # SF 走 int32 packed（Mega MoE 契约要求 kInt；此处模拟 transform 的 MN-major int32 槽）
    l1sf_i = l1sf.view(torch.int32)
    l2sf_i = l2sf.view(torch.int32)

    (t1w, t1sf), (t2w, t2sf) = transform_weights_for_mega_moe((l1w, l1sf_i), (l2w, l2sf_i))

    w_inter_ok = torch.equal(t1w, _interleave_weights(l1w))
    l1_sf_ok = torch.equal(t1sf, _transpose_sf_for_utccp(_interleave_weights(l1sf_i)))
    l2_sf_ok = torch.equal(t2sf, _transpose_sf_for_utccp(l2sf_i))
    l2w_ok = t2w.data_ptr() == l2w.data_ptr()
    flag = w_inter_ok and l1_sf_ok and l2_sf_ok and l2w_ok
    ok &= flag
    print(f'   E={E} l1[2I={2*I},H={H}] -> l1w{tuple(t1w.shape)} l1sf{tuple(t1sf.shape)}')
    print(f'   l1 w 仅交错={w_inter_ok}；l1 sf 与「交错后 UTCCP 转置」一致={l1_sf_ok}；'
          f'l2 sf 仅转置={l2_sf_ok}；l2 w 原样（同对象）={l2w_ok} {"" if flag else "MISMATCH"}')

    # 关键：N 轴 gran-8 交错与 K 轴 2-per-byte packing 可交换（逐位）
    def unpack(p):
        lo = (p.to(torch.int16) & 0x0F).to(torch.int8)
        hi = ((p.to(torch.int16) >> 4) & 0x0F).to(torch.int8)
        out = torch.empty(p.shape[0], p.shape[1], p.shape[2] * 2, dtype=torch.int8, device='cuda')
        out[:, :, 0::2], out[:, :, 1::2] = lo, hi
        return out

    inter_then_unpack = unpack(_interleave_weights(l1w))
    # 先 unpack 再沿 N 交错，应与「先交错再 unpack」逐位相等（packing 沿 K，与 N 置换独立）
    unpacked = unpack(l1w)
    inter_unpacked = _interleave_weights(unpacked)
    commute = torch.equal(inter_then_unpack, inter_unpacked)
    ok &= commute
    print(f'   pack/unpack 与 N-交错可交换（逐位）: {commute} '
          f'(unpack∘interleave == interleave∘unpack: {commute})')
    return ok


# =================== 4) 精度 / 显存：FP8 权重 vs FP4 权重 ===================
def part4_accuracy():
    from deep_gemm.utils.math import (per_token_cast_to_fp8, cast_back_from_fp8,
                                      per_token_cast_to_fp4, cast_back_from_fp4)
    print('\n== 4) 小 MoE：FP8 权重 vs FP4 权重的端到端误差与字节数 ==')
    torch.manual_seed(2)
    T, H, I, E, K = 256, 512, 512, 8, 2
    x = torch.randn(T, H, dtype=torch.bfloat16, device='cuda')
    W1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device='cuda') * (1.0 / math.sqrt(H))
    W2 = torch.randn(E, H, I, dtype=torch.bfloat16, device='cuda') * (1.0 / math.sqrt(I))
    logits = torch.randn(T, E, device='cuda')
    tw, ti = torch.topk(torch.softmax(logits, -1), K, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True)

    def mx_fp8_2d(t):
        # 每行 gran-32 UE8M0 量化再反量化（隔离量化误差）
        f, sf = per_token_cast_to_fp8(t, use_ue8m0=True, gran_k=32)
        return cast_back_from_fp8(f, sf, gran_k=32)

    def mx_fp4_rows(t):
        # per-expert [N,K] -> packed int8 + sf -> 反量化
        p, sf = per_token_cast_to_fp4(t, use_ue8m0=True, gran_k=32)
        return cast_back_from_fp4(p, sf, gran_k=32)

    # 被 dispatch 到各专家的 token（模拟 expanded 视图）
    def run(weight_quant):
        out = torch.zeros(T, H, dtype=torch.float64, device='cuda')
        for e in range(E):
            # 该专家收到的 (token, weight) —— 这里取所有 token 的一个子集
            tok = torch.arange(T, device='cuda')
            sel = torch.nonzero((ti == e).any(-1)).flatten()
            if sel.numel() == 0:
                continue
            xs = x[sel].float().double()                    # [n,H] 精确
            xq = mx_fp8_2d(x[sel].float()).double()         # [n,H] 激活 FP8 量化
            w1q = weight_quant(W1[e].float()).double()       # [2I,H]
            h = xq @ w1q.t()                                 # [n,2I]
            g, u = h[:, :I], h[:, I:]
            act = (g / (1 + torch.exp(-g))) * u              # SwiGLU（精确激活，只隔离 GEMM 量化）
            actq = mx_fp8_2d(act.float()).double()           # L1 输出 FP8 量化（doc 99 的语义）
            w2q = weight_quant(W2[e].float()).double()       # [H,I]
            y = actq @ w2q.t()                               # [n,H]
            wcol = tw[sel, (ti[sel] == e).float().argmax(-1)].double().unsqueeze(-1)
            out.index_add_(0, sel, y * wcol)
        return out

    ref = torch.zeros(T, H, dtype=torch.float64, device='cuda')
    for e in range(E):
        sel = torch.nonzero((ti == e).any(-1)).flatten()
        if sel.numel() == 0:
            continue
        h = x[sel].float().double() @ W1[e].float().double().t()
        g, u = h[:, :I], h[:, I:]
        act = (g / (1 + torch.exp(-g))) * u
        y = act @ W2[e].float().double().t()
        wcol = tw[sel, (ti[sel] == e).float().argmax(-1)].double().unsqueeze(-1)
        ref.index_add_(0, sel, y * wcol)

    # jitter：activations FP8 两次都一样，权重 FP8 vs FP4 不同
    def wq8(t):
        return mx_fp8_2d(t)

    def wq4(t):
        return mx_fp4_rows(t)

    y8 = run(wq8)
    y4 = run(wq4)
    r8, r4 = rel_err(y8, ref), rel_err(y4, ref)
    # 权重字节数（per expert）
    b8 = 2 * I * H * 1 + H * I * 1              # FP8：W1+W2 各 1B
    b4 = 2 * I * H * 0.5 + H * I * 0.5          # FP4：各 0.5B
    # 量化本身：权重相对 bf16 的误差
    q8 = rel_err(torch.stack([wq8(W1[e].float()) for e in range(E)]).double(),
                 W1.float().double())
    q4 = rel_err(torch.stack([wq4(W1[e].float()) for e in range(E)]).double(),
                 W1.float().double())
    print(f'   端到端 relerr vs fp64： 权重FP8={r8:.4e}  权重FP4={r4:.4e}（激活都 FP8 gran32）')
    print(f'   权重量化本身（W1，相对 bf16）： FP8={q8:.4e}  FP4={q4:.4e}')
    print(f'   每专家权重字节： FP8={b8}B  FP4={b4}B  比={b8 / b4:.2f}x')
    ok = (r4 < 5e-1) and (r4 >= r8) and (abs(b4 * 2 - b8) < 1e-6)
    print(f'   FP4 权重误差更大（{r4 >= r8}）且权重显存减半（{b4 * 2 == b8}）: {ok}')
    return ok


def main():
    print(f'device: {torch.cuda.get_device_name(0)}')
    ok = True
    ok &= part1_host_config()
    ok &= part2_packed_repr()
    ok &= part3_transform()
    ok &= part4_accuracy()
    print(f'\nALL_OK = {ok}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
