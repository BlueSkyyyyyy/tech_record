"""P47-1 H100 evidence: Mega Gate 的端到端验收测试与路由 oracle
(`tests/test_mega_gate.py`)。

Mega Gate 主体 kernel（`sm100_bf16_mega_gate`）是 SM100 专属
（`csrc/apis/mega_gate.hpp:159` 断言 `arch_major == 10`），H100(sm90) 跑不了。
但它的**验收测试**里那套路由 oracle —— sqrt-softplus 打分、带 bias 排名、
top-k 权重归一化、logical→physical 专家映射的逆、top-k 边界 crossing 容忍、
mask / fix_routing / force_random 语义 —— 全是纯 torch，架构无关，H100 真机可跑。

本脚本直接 import 真实 `tests/test_mega_gate.py` 作为模块，复用它的 helper
（`make_inputs` / `make_outputs` / `check_topk_weights` / `make_physical_map`），
用一个「完全正确的 torch 代理 gate」喂给它，验证这套 oracle 自洽，
再用扰动反例验证它真的能抓到错。
"""
import os
import sys
import inspect
import importlib.util

import torch
import torch.nn.functional as F

_CANDIDATES = [
    os.environ.get('DEEPGEMM_TESTS', ''),
    os.path.expanduser('~/github/DeepGEMM/tests'),
    os.path.expanduser('~/github/DeepGEMM/tests'),
    os.path.expanduser('~/github/DeepGEMM/tests'),
]
TESTS_DIR = next(p for p in _CANDIDATES if p and os.path.isfile(os.path.join(p, 'test_mega_gate.py')))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major

PASS = 0
FAIL = 0


def check(name, cond, extra=''):
    global PASS, FAIL
    cond = bool(cond)
    if cond:
        PASS += 1
        print(f'  [PASS] {name} {extra}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {extra}')
    return cond


def hdr(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


# --- 直接加载真实 test_mega_gate.py 作为模块，复用它的 helper ---
spec = importlib.util.spec_from_file_location('test_mega_gate', os.path.join(TESTS_DIR, 'test_mega_gate.py'))
TMG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TMG)
print(f'arch_major = {get_arch_major()} (H100=9)')
print(f'deep_gemm  = {deep_gemm.__file__}')
print(f'test module = {inspect.getsourcefile(TMG)}')


# ---------------------------------------------------------------------------
# 一个「完全正确的 torch 代理 gate」：语义与 kernel 一致（不含 GEMM 精度差）
# ---------------------------------------------------------------------------
def reference_gate(inputs, num_topk, fix_routing_mask=None, force_random=None,
                   fixed_idx=None, with_unmapped=False, ep_rank=0, seed=0, mask_override=None):
    x, weight = inputs['x'], inputs['weight']
    T, E = x.shape[0], weight.shape[0]
    dev = x.device
    logits = (x.float() @ weight.float().t())          # bf16 操作数 -> fp32 累加
    scores = F.softplus(logits).sqrt()                 # 与 kernel `sqrt_softplus` 同义（见 §B）
    ranking = scores.clone()
    # 排名 bias（测试 :130-135）：image token 用 image_bias「替换」普通 bias，否则用 bias
    if inputs['bias'] is not None or inputs['image_bias'] is not None:
        rb = inputs['bias'] if inputs['bias'] is not None else torch.zeros(E, device=dev)
        if inputs['image_bias'] is not None:
            rb = torch.where(inputs['image_token_mask'][:, None], inputs['image_bias'][None, :], rb[None, :])
        else:
            rb = rb[None, :]
        ranking = ranking + rb

    active = mask_override if mask_override is not None else inputs['mask']
    if active is None:
        active = torch.ones(T, dtype=torch.bool, device=dev)
    # 确定性 top-k：ranking 降序，平局取最小专家号（kernel `:83-89`）
    order = torch.argsort(ranking, dim=1, descending=True, stable=True)
    sel = order[:, :num_topk].clone()
    if fixed_idx is not None and fix_routing_mask is not None:
        fixed = active & fix_routing_mask
        sel[fixed] = fixed_idx[fixed]

    ts = scores.gather(1, sel)
    w = ts / (ts.sum(1, keepdim=True) + 1e-20) * TMG.ROUTED_SCALING_FACTOR

    fr = torch.zeros(T, dtype=torch.bool, device=dev) if force_random is None else (active & force_random)
    unmapped = sel.clone()

    # logical -> physical（重复专家；kernel `epilogue:198-207`）
    physical = sel.clone()
    if inputs['to_physical_map'] is not None:
        tpm, lc = inputs['to_physical_map'], inputs['logical_count']
        for t in range(T):
            if not bool(active[t]) or bool(fr[t]):
                continue
            for j in range(num_topk):
                l = int(sel[t, j])
                if l < 0:
                    continue
                nd = int(lc[l])
                dup = (ep_rank + t * 23333) % nd
                physical[t, j] = int(tpm[l, dup])

    # force_random：从 [0, sum(logical_count)) 里取 num_topk 个互不相同的专家，权重 > 0
    if bool(fr.any()):
        n_phys = int(inputs['logical_count'].sum()) if inputs['to_physical_map'] is not None else E
        g = torch.Generator(device='cpu').manual_seed(seed)
        for t in torch.nonzero(fr).flatten().tolist():
            perm = torch.randperm(n_phys, generator=g)[:num_topk]
            physical[t] = perm.to(dev)
            w[t] = 0.5
        unmapped[fr] = -1

    physical[~active] = -1
    w[~active] = 0.0
    unmapped[~active] = -1

    outs = {
        'topk_idx': physical.to(torch.int64),
        'topk_weights': w.to(torch.float32),
        'unmapped_topk_idx': unmapped.to(torch.int64) if with_unmapped else None,
    }
    return outs, logits


def run_check(inputs, outs, logits, num_topk, **kw):
    TMG.check_topk_weights(inputs, outs, logits, num_topk=num_topk, **kw)


def expect_raise(name, fn):
    try:
        fn()
        return check(name, False, 'expected AssertionError, got none')
    except AssertionError as e:
        msg = str(e).splitlines()[0][:60] if str(e) else ''
        return check(name, True, f'AssertionError: {msg}')


def inverse(p, num_logical):
    # 测试的物理->logical 逆映射（tests/test_mega_gate.py:121-128）
    return torch.where(p < num_logical, p, p - num_logical)


# ===========================================================================
hdr('A. 测试常量与用例枚举（源码事实 + 真实模块）')
# ===========================================================================
check('A1 NUM_TOPK==6 / NUM_SHARED_EXPERTS==1 / ROUTED_SCALING_FACTOR==1.5',
      (TMG.NUM_TOPK, TMG.NUM_SHARED_EXPERTS, TMG.ROUTED_SCALING_FACTOR) == (6, 1, 1.5))
check('A2 VLLM_NUM_EXTRA_EXPERTS==32', TMG.VLLM_NUM_EXTRA_EXPERTS == 32)
cases = list(TMG.enumerate_cases())
check('A3 enumerate_cases 共 45 个', len(cases) == 45, f'len={len(cases)}')
check('A4 第一个是 (1,4096,384)', cases[0] == (1, 4096, 384), f'{cases[0]}')
check('A5 含 4-wave K 尾块用例 (32768,4096,512)', (32768, 4096, 512) in cases)
check('A6 GATE_CONFIGS 11 组', len(TMG.GATE_CONFIGS) == 11, f'len={len(TMG.GATE_CONFIGS)}')

# ===========================================================================
hdr('B. 打分函数：kernel `sqrt_softplus` vs 测试参考 `softplus(·).sqrt()`')
# ===========================================================================
# kernel 公式（epilogue/sm100_mega_gate.cuh:19-23）： value>20 ? value : log1p(exp(value))，再 sqrt
xs = torch.linspace(-40, 40, 4001, device='cuda')
k_sp = torch.where(xs > 20.0, xs, torch.log1p(torch.exp(xs)))
t_sp = F.softplus(xs)
d = (k_sp - t_sp).abs().max().item()
check('B1 threshold-20 折衷式与 F.softplus 逐点一致（max|Δ|）', d == 0.0, f'max|Δ|={d:.3e}')
# 与真值 log1p(exp)（fp64）比：linear 分支（x>20）的近似误差 ~ e^-x / x，
# 远小于 fp32 舍入（~2^-23），故整体误差由 fp32 舍入主导。
sp64 = torch.log1p(torch.exp(xs.double()))
rel = ((k_sp.double() - sp64).abs() / sp64.abs()).max().item()
lin = xs[xs > 20.0]
approx_err = float((torch.exp(-lin) / lin).max())
check('B2 与 fp64 真值 max rel < 1e-6（受 fp32 舍入主导）', rel < 1e-6, f'max rel={rel:.3e}')
check('B2b linear 分支解析近似误差 e^-x/x << fp32 舍入', approx_err < 1e-8,
      f'e^-x/x max={approx_err:.3e}')
# 恰好在 20 的分支点
v20 = torch.tensor([20.0], device='cuda')
k20 = float(torch.where(v20 > 20.0, v20, torch.log1p(torch.exp(v20)))[0])
t20 = float(F.softplus(v20)[0])
check('B3 x==20 两式都走 log1p 分支', abs(k20 - t20) == 0.0, f'k={k20:.10f} t={t20:.10f}')
print(f'  · log1p(exp(20))={k20:.12f}, softplus(20)={t20:.12f}, 线性分支值=20.0')

# ===========================================================================
hdr('C. `check_topk_weights` 自洽：完全正确的代理 gate 通过（阈值 1e-11）')
# ===========================================================================
torch.manual_seed(0)
T, E, H, K = 1024, 384, 4096, TMG.NUM_TOPK
inputs = TMG.make_inputs(T, H, E)          # with_bias=True, with_physical_map=True
outs, logits = reference_gate(inputs, K)   # 不给 unmapped -> 走 physical 逆映射分支
# 让 physical 逆映射确实被用到：proxy 输出 physical，无 unmapped
wd = TMG.check_topk_weights(inputs, outs, logits, None, num_topk=K)
check('C1 正确代理 gate weight_diff < 1e-11', wd < 1e-11, f'weight_diff={wd:.3e}')
check('C2 正确代理 gate weight_diff == 0（同 scores 同归一化）', wd == 0.0, f'{wd:.3e}')
# calc_diff 的判据强度：weights 量级 ~1.5/K/...
w_scale = float(outs['topk_weights'].abs().mean())
print(f'  · top-k 权重均值 |w| ~ {w_scale:.4f}, calc_diff 阈值 1e-11 对应的相对误差 ~ { (2e-11)**0.5:.1e}')

# 扰动 1：权重整体乘 (1+1e-3) -> calc_diff ~ 5e-7 > 1e-11
bad = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in outs.items()}
bad['topk_weights'] = bad['topk_weights'] * (1.0 + 1e-3)
expect_raise('C3 权重偏 1e-3 -> calc_diff 触发断言（:143）',
             lambda: run_check(inputs, bad, logits, K))

# 扰动 2：权重偏 1e-6 -> calc_diff ~ 5e-13 < 1e-11，应仍然通过（阈值不是无限严）
bad2 = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in outs.items()}
bad2['topk_weights'] = bad2['topk_weights'] * (1.0 + 1e-6)
cd = float(calc_diff(bad2['topk_weights'], outs['topk_weights']))
check('C4 权重偏 1e-6 -> calc_diff ~1e-12 < 1e-11 仍通过', cd < 1e-11, f'calc_diff={cd:.3e}')

# 扰动 3：把一个已选专家换成明显更差的未选专家（权重保持一致）-> crossing 超容忍（:155）
bad3 = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in outs.items()}
ranking = F.softplus(logits).sqrt() + inputs['bias'][None, :]
order = torch.argsort(ranking, dim=1, descending=True, stable=True)
sel_logical = inverse(outs['topk_idx'], E)             # 逆映射还原 logical（< E）
repl = sel_logical.clone()
for t in range(T):
    nxt = int(order[t, K].item())                       # 第 (K+1) 名未选专家
    repl[t, -1] = nxt                                   # 换掉最后一个选中项
s_log = F.softplus(logits).sqrt()
ts = s_log.gather(1, repl)
bad3['topk_idx'] = repl                                 # 全 < E，逆映射为自身
bad3['topk_weights'] = (ts / (ts.sum(1, keepdim=True) + 1e-20) * TMG.ROUTED_SCALING_FACTOR).float()
expect_raise('C5 换掉一个选中专家（权重自洽）-> crossing 越界断言（:155）',
             lambda: run_check(inputs, bad3, logits, K))

# 扰动 4：重复专家号 -> 唯一性断言（:149）（权重设为与新选择一致，先过 :143）
bad4 = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in outs.items()}
dup_log = inverse(outs['topk_idx'], E).clone()
dup_log[:, 1] = dup_log[:, 0]
ts4 = F.softplus(logits).sqrt().gather(1, dup_log)
bad4['topk_idx'] = dup_log
bad4['topk_weights'] = (ts4 / (ts4.sum(1, keepdim=True) + 1e-20) * TMG.ROUTED_SCALING_FACTOR).float()
expect_raise('C6 两个槽选同一专家 -> 唯一性断言（:149）',
             lambda: run_check(inputs, bad4, logits, K))

# ===========================================================================
hdr('D. logical -> physical 专家映射与它的逆（测试 :121-128 与 :40-49 的隐式耦合）')
# ===========================================================================
tpm, lc = TMG.make_physical_map(E)
check('D1 map 形状 (E, 33)', tuple(tpm.shape) == (E, TMG.VLLM_NUM_EXTRA_EXPERTS + 1), f'{tuple(tpm.shape)}')
check('D2 第 0 列 = 自身（logical 直通）', torch.equal(tpm[:, 0], torch.arange(E, device='cuda', dtype=torch.int32)))
check('D3 第 1 列前 32 行 = E + arange(32)（重复槽按 logical 顺序编号）',
      torch.equal(tpm[:32, 1], (E + torch.arange(32, device='cuda', dtype=torch.int32))))
check('D4 logical_count：前 32 为 2、其余为 1',
      bool((lc[:32] == 2).all()) and bool((lc[32:] == 1).all()))

# 逆映射（测试 :121-128）：p < num_logical ? p : p - num_logical
ok = True
for l in range(E):
    for dup in range(int(lc[l])):
        p = int(tpm[l, dup])
        if int(inverse(torch.tensor([p], device='cuda'), E)[0]) != l:
            ok = False
check('D5 对每个 (logical, duplicate) 槽，逆映射都还原出 logical', ok)
# 逆映射不是普遍成立：只有当重复槽恰好是「num_logical + logical」时才成立
bad_map = tpm.clone()
bad_map[0, 1] = E + 5          # 破坏「重复槽 = E + logical」的顺序
check('D6 反例：把重复槽改成 E+5，逆映射还原错误（说明它是构造的隐式约定）',
      int(inverse(bad_map[0, 1], E)) != 0, f'inverse={int(inverse(bad_map[0,1],E))}')

# 重复分散公式 (rank + t*23333) % nd 按 token 均衡（kernel epilogue:205）
toks = torch.arange(1 << 16, device='cuda')
dup2 = (0 + toks * 23333) % 2
frac = float((dup2 == 0).float().mean())
check('D7 重复分散 (rank+t*23333)%2 在两个副本上均衡', abs(frac - 0.5) < 1e-3, f'frac(copy0)={frac:.5f}')

# ===========================================================================
hdr('E. 路由控制：mask / fix_routing / force_random / image_bias / 极端 bias')
# ===========================================================================
# E1: mask 语义（-1 索引、权重 0，测试 :167-171）
torch.manual_seed(1)
inputs_m = TMG.make_inputs(64, 4096, 128, with_mask=True)
outs_m, logits_m = reference_gate(inputs_m, K)
wd_m = TMG.check_topk_weights(inputs_m, outs_m, logits_m, inputs_m['mask'], num_topk=K)
check('E1 mask 生效（inactive 恒 -1/0，active 正确）', wd_m == 0.0, f'weight_diff={wd_m:.3e}')

# E2: image_bias（对 image token 用 image_bias 排名，测试 :132-135）
torch.manual_seed(2)
inputs_i = TMG.make_inputs(128, 4096, 128, with_image_bias=True)
outs_i, logits_i = reference_gate(inputs_i, K)
wd_i = TMG.check_topk_weights(inputs_i, outs_i, logits_i, num_topk=K)
check('E2 image_bias 排名语义一致', wd_i == 0.0, f'weight_diff={wd_i:.3e}')
# 反例：忽略 image_bias（只用普通 bias）得到不同的 top-k
i_ignore = inputs_i.copy()
i_ignore['image_bias'] = None
i_ignore['image_token_mask'] = None
outs_ig, _ = reference_gate(i_ignore, K)
sel_i = inverse(outs_i['topk_idx'], 128)
sel_ig = inverse(outs_ig['topk_idx'], 128)
frac_diff = float((sel_i != sel_ig).any(1).float().mean())
check('E2b 忽略 image_bias 时 top-k 选择与用 image_bias 不同',
      frac_diff > 0.05, f'不同 token 比例={frac_diff:.3f}')

# E3: 极端 bias（测试 :272-275）：bias=1e8 选最大 bias 专家、image_bias=-1e8
torch.manual_seed(3)
inputs_x = TMG.make_inputs(128, 4096, 384, with_image_bias=True)
inputs_x['bias'].fill_(1e8)
inputs_x['image_bias'].fill_(-1e8)
outs_x, logits_x = reference_gate(inputs_x, K)
wd_x = TMG.check_topk_weights(inputs_x, outs_x, logits_x, num_topk=K)
check('E3 极端 bias 下 oracle 仍自洽', wd_x < 1e-11, f'weight_diff={wd_x:.3e}')

# E4: fix_routing（测试 :236-247）：fixed token 保留给定路由
torch.manual_seed(4)
inputs_f = TMG.make_inputs(256, 4096, 384)
tok = torch.arange(256, device='cuda')
mask_f = tok % 11 != 0
fixed_mask = (tok % 3 == 0) & mask_f
fr_mask = (tok % 5 == 0) & mask_f
fixed_idx = torch.randint(0, 384, (256, K), dtype=torch.int64, device='cuda')
outs_f, logits_f = reference_gate(inputs_f, K, fix_routing_mask=fixed_mask,
                                  force_random=fr_mask, fixed_idx=fixed_idx,
                                  with_unmapped=True, mask_override=mask_f)
# 保留 unmapped（check_routing_controls 的用法）
wd_f = TMG.check_topk_weights(inputs_f, outs_f, logits_f, mask_f, fr_mask, fixed_mask, num_topk=K)
check('E4 fix_routing + force_random + mask 组合 oracle 自洽', wd_f == 0.0, f'weight_diff={wd_f:.3e}')
fixed_only = fixed_mask & mask_f & ~fr_mask        # 测试 :246 的 fixed 定义（排除 force_random）
check('E5 fixed token 的 unmapped 等于给定 fixed_idx',
      torch.equal(outs_f['unmapped_topk_idx'][fixed_only], fixed_idx[fixed_only]))
fr_rows = fr_mask & mask_f
check('E6 force_random 行：idx 互不相同且 unmapped==-1',
      bool((outs_f['unmapped_topk_idx'][fr_rows] == -1).all()) and
      all(len(set(outs_f['topk_idx'][t].tolist())) == K for t in torch.nonzero(fr_rows).flatten().tolist()))

# ===========================================================================
hdr('F. 退化用例：全零打分 -> arange；零 token')
# ===========================================================================
inputs_z = TMG.make_inputs(8, 4096, 128, with_bias=False, with_physical_map=False)
inputs_z['x'].zero_()
inputs_z['weight'].zero_()
outs_z, logits_z = reference_gate(inputs_z, 8)
check('F1 全零打分（所有 score 相等）确定性 top-k = arange(num_topk)',
      torch.equal(outs_z['topk_idx'], torch.arange(8, device='cuda').expand(8, 8)),
      f'{outs_z["topk_idx"][0].tolist()}')
# 核验：sqrt(softplus(0)) = sqrt(ln2)
check('F2 softplus(0)=ln2，sqrt 后 = sqrt(ln2)', abs(float(F.softplus(torch.tensor(0.0))) - 0.6931472) < 1e-6)

inputs_0 = TMG.make_inputs(0, 4096, 128, with_physical_map=False)
outs_0, logits_0 = reference_gate(inputs_0, K)
check('F3 num_tokens==0 输出 shape (0,K)', tuple(outs_0['topk_idx'].shape) == (0, K))
wd_0 = TMG.check_topk_weights(inputs_0, outs_0, logits_0, num_topk=K)
check('F4 num_tokens==0 oracle 直接返回 0', wd_0 == 0.0)

# ===========================================================================
hdr('G. host 契约与 test 门控（源码事实）')
# ===========================================================================
src_run = inspect.getsource(TMG.run_mega_gate)
src_corr = inspect.getsource(TMG.check_correctness)
src_ref = inspect.getsource(TMG.run_reference_gemm)
check('G1 参考 logits 走 deep_gemm.cublaslt_gemm_nt（bf16）', 'cublaslt_gemm_nt' in src_ref)
check('G2 重复 10 次输出 assert_bitwise_equal（:185-191）', 'assert_bitwise_equal' in src_corr)
check('G3 融合 kernel 入口 deep_gemm.bf16_mega_gate', 'bf16_mega_gate' in src_run)
check('G4 arch gate `get_arch_major()==10`（H100 上两用例都被跳过）',
      get_arch_major() == 9)

# host 契约（csrc/apis/mega_gate.hpp）源码事实
mg_src = open(os.path.join(os.path.dirname(TESTS_DIR), 'csrc/apis/mega_gate.hpp')).read()
for tag, needle in [
    ('G5 hidden % 256 == 0', 'hidden % kHiddenAlignment == 0'),
    ('G6 num_routed_experts <= 512', 'kNumMaxRoutedExperts = 512'),
    ('G7 num_shared_experts in {1,2}', 'num_shared_experts == 1 or num_shared_experts == 2'),
    ('G8 scoring_func == "sqrtsoftplus"', 'scoring_func == "sqrtsoftplus"'),
    ('G9 arch_major == 10 闸门', 'DG_HOST_ASSERT(arch_major == 10)'),
    ('G10 fix_routing 需要 unmapped', 'fix_routing_mask.has_value()) {\n        DG_HOST_ASSERT(unmapped_topk_idx.has_value())'),
]:
    check(tag, needle in mg_src)

# ===========================================================================
hdr('H. 归一化权重公式（scores / (sum + 1e-20) * routed_scaling_factor）')
# ===========================================================================
torch.manual_seed(5)
T2, E2 = 512, 384
inp2 = TMG.make_inputs(T2, 4096, E2)
outs2, logits2 = reference_gate(inp2, K)
scores2 = F.softplus(logits2).sqrt()
sel2 = inverse(outs2['topk_idx'], E2)
ts2 = scores2.gather(1, sel2)
manual = ts2 / (ts2.sum(1, keepdim=True) + 1e-20) * 1.5
check('H1 权重 == 手写 scores/(sum+1e-20)*1.5（逐位）', torch.equal(manual, outs2['topk_weights']))
check('H2 每个 token 的 top-k 未加权分数和 > 0', bool((ts2.sum(1) > 0).all()))
# 未归一化（漏掉 /sum）会明显偏离：反例
wrong = (ts2 * 1.5)
cd_w = float(calc_diff(wrong, outs2['topk_weights']))
check('H3 漏掉归一化 -> calc_diff 很大（反例）', cd_w > 0.1, f'calc_diff={cd_w:.3f}')

print('\n' + '=' * 78)
print(f'总判定: {PASS} PASS / {FAIL} FAIL')
print('=' * 78)
sys.exit(1 if FAIL else 0)
