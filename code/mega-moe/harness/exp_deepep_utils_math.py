"""P104-1 实验：DeepEP 的 MoE 语义工具层 `deep_ep/utils/math.py`。

目标源码（`~/github/DeepEP@93eb6eb`，按文件路径直接 exec 真实源码——math.py 只 import torch，
不需要 deep_ep 包，也绕过 `deep_ep/__init__.py` 的 `check_nccl_so()`）：

  * calc_diff            :5-9    —— 与 DeepGEMM `testing/numeric.py` 同名但**不同定义**的判据
  * safe_div             :12-19
  * ceil_div / align     :22-27
  * per_token_cast_to_fp8:31-39  —— per-128 token 量化器（**未打包 UE8M0**，返回 fp32 scale）
  * per_token_cast_back  :43-56  —— 反量化 + **int scale 解码分支**（`uint8<<23` 重解释）
  * inplace_unique       :59-71  —— top-k 专家去重 oracle（combine 归约的语义规范）
  * create_grouped_scores:74-79  —— **group-limited 路由**的专家掩码
  * hash_tensor(s)       :82-93
  * count_bytes          :96-103

对照：DeepGEMM `deep_gemm/testing/numeric.py`（calc_diff/count_bytes）与
`deep_gemm/utils/math.py`（per_token_cast_to_fp8 / pack_ue8m0_to_int）按路径加载。

运行：bash scripts/lab.sh exec "python /abs/path/harness/exp_deepep_utils_math.py"
"""
import hashlib
import importlib.util
import os
import sys

import torch

# `per_token_cast_to_fp8/back` 带 `@torch.compile(dynamic=True)`（math.py:30/42）。dynamo 对该
# 动态形状的 `.view` 有已知图断裂，故本实验强制 eager（语义与编译版一致，仅不做图优化）。
torch._dynamo.config.disable = True

DEP = os.environ.get('DEEPEP_ROOT', '~/github/DeepEP')
DGEMM = os.environ.get('DEEPGEMM_ROOT', '~/github/DeepGEMM')
if not os.path.isdir(DEP):
    DEP = os.path.expanduser('~/github/DeepEP')
if not os.path.isdir(DGEMM):
    DGEMM = os.path.expanduser('~/github/DeepGEMM')

torch.manual_seed(0)
DEV = 'cuda'
FAILS = []


def load(path, name):
    src = open(path, 'rb').read()
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m, hashlib.sha256(src).hexdigest()


def check(label, cond, extra=''):
    tag = '[PASS]' if cond else '[FAIL]'
    if not cond:
        FAILS.append(label)
    print(f'  {tag} {label} {extra}')


print('=' * 78)
print('P104-1  DeepEP utils/math.py —— H100 真机语义核验')
print('=' * 78)
print(f'torch {torch.__version__}  cuda {torch.version.cuda}  dev {torch.cuda.get_device_name(0)}')

m, h_m = load(f'{DEP}/deep_ep/utils/math.py', 'dep_math')
dgn, h_dgn = load(f'{DGEMM}/deep_gemm/testing/numeric.py', 'dg_numeric')
dgm, h_dgm = load(f'{DGEMM}/deep_gemm/utils/math.py', 'dg_math')
print(f'deep_ep/utils/math.py           sha256={h_m}')
print(f'deep_gemm/testing/numeric.py    sha256={h_dgn}')
print(f'deep_gemm/utils/math.py         sha256={h_dgm}')
print()

# ---------------------------------------------------------------------------------------
print('Part A — calc_diff: DeepEP 与 DeepGEMM 同名不同定义，以及 +1 位移的后果')
# ---------------------------------------------------------------------------------------
for dtype in (torch.float32, torch.bfloat16):
    x = (torch.randn(4096, dtype=dtype, device=DEV) * 3).to(torch.float32)
    y = (torch.randn(4096, dtype=dtype, device=DEV) * 3).to(torch.float32)
    xs, ys = x.double() + 1, y.double() + 1
    d_dep = m.calc_diff(x, y)
    d_sq = ((xs - ys) ** 2).sum() / (xs * xs + ys * ys).sum()
    d_cos = 1 - 2 * (xs * ys).sum() / (xs * xs + ys * ys).sum()
    check(f'A1 {dtype} DeepEP calc_diff == 归一化平方误差(x+1,y+1)',
          abs(d_dep - float(d_sq)) < 1e-12, f'{d_dep:.6e} vs {float(d_sq):.6e}')
    check(f'A1b {dtype} 余弦式 == 平方误差式（代数恒等）',
          abs(d_dep - float(d_cos)) < 1e-12)
    # DeepEP(x,y) 先把两边 +1(d为double)再算；DeepGEMM(x+1,y+1) 则在 double 域。
    # 故对齐时要把 +1 在 double 里做，否则 fp32 的 x+1 会再引入一次舍入。
    d_dg = float(dgn.calc_diff(xs, ys))
    check(f'A2 {dtype} DeepEP calc_diff(x,y) == DeepGEMM calc_diff((x+1).double,(y+1).double)',
          abs(d_dep - d_dg) < 1e-12, f'{d_dep:.6e} vs {d_dg:.6e}')
    d_same = float(dgn.calc_diff(x, y))
    check(f'A2b {dtype} 但 DeepEP(x,y) != DeepGEMM(x,y)（+1 位移造成差异）',
          abs(d_dep - d_same) > 1e-6, f'DeepEP {d_dep:.6e} vs DeepGEMM {d_same:.6e}')

# A3 全零张量：+1 位移恰好把 0/0 变成 64/128 -> 0，避免 NaN；DeepGEMM 则靠显式 guard。
z = torch.zeros(64, device=DEV)
d_dep_z = m.calc_diff(z, z)
d_dg_z = float(dgn.calc_diff(z, z))
print(f'  A3  全零: DeepEP calc_diff={d_dep_z}  (nan={d_dep_z != d_dep_z})   '
      f'DeepGEMM calc_diff={d_dg_z}')
check('A3 全零张量两边都得 0.0（DeepEP 靠 +1 位移避开 0/0，DeepGEMM 靠显式 guard）',
      d_dep_z == 0.0 and d_dg_z == 0.0)

# A4 位移破坏尺度不变性
xx = torch.tensor([1.0, 2.0, 3.0, -1.0], device=DEV)
yy = torch.tensor([1.1, 1.9, 3.4, -0.8], device=DEV)
row = []
for c in (0.1, 1.0, 10.0, 100.0):
    row.append((c, m.calc_diff(xx * c, yy * c), float(dgn.calc_diff(xx * c, yy * c))))
print('  A4  scale c -> (DeepEP, DeepGEMM):', '  '.join(
    f'c={c:g}:({a:.4e},{b:.4e})' for c, a, b in row))
dep_spread = max(a for _, a, _ in row) - min(a for _, a, _ in row)
dg_spread = max(b for _, _, b in row) - min(b for _, _, b in row)
check('A4 DeepEP calc_diff 随缩放变化（+1 位移破坏尺度不变），DeepGEMM 近似不变',
      dep_spread > 1e-4 and dg_spread < 1e-6,
      f'DeepEP spread={dep_spread:.2e} DeepGEMM spread={dg_spread:.2e}')

# A5 非负性（2ab<=a^2+b^2）
vals = []
for _ in range(200):
    p = torch.randn(128, device=DEV) * (10 ** torch.randint(-3, 3, (1,)).item())
    q = torch.randn(128, device=DEV) * (10 ** torch.randint(-3, 3, (1,)).item())
    vals.append(m.calc_diff(p, q))
check('A5 calc_diff 恒 >= 0（200 组随机，含 1e-3~1e3 量级）', min(vals) >= 0.0,
      f'min={min(vals):.3e} max={max(vals):.3e}')

# ---------------------------------------------------------------------------------------
print()
print('Part B — per_token_cast_to_fp8 / per_token_cast_back: 与 DeepGEMM 量化器的异同')
# ---------------------------------------------------------------------------------------
m_, n_ = 64, 100  # n 非 128 的倍数，检验 padding 切片
xb = (torch.randn(m_, n_, device=DEV) * 2).to(torch.bfloat16)
xf, sf = m.per_token_cast_to_fp8(xb)
print(f'  B shape: x_fp8={tuple(xf.shape)} sf={tuple(sf.shape)} '
      f'({xf.dtype}, {sf.dtype})')
check('B0 输出 shape 保留 (m,n) 与 sf=(m,ceil(n/128))',
      tuple(xf.shape) == (m_, n_) and tuple(sf.shape) == (m_, (n_ + 127) // 128))

# B1 round-trip
xbk = m.per_token_cast_back(xf, sf)
rel = ((xbk.float() - xb.float()).norm() / xb.float().norm()).item()
print(f'  B1 反量化回 bf16 relerr={rel:.4e}  dtype={xbk.dtype}')
check('B1 round-trip relerr 落在 fp8 per-128 量化层 (<4e-2)', rel < 4e-2)

# B2 与 DeepGEMM 非 UE8M0 分支：scale 逐位一致；但**乘子**写法不同（448/amax vs 1/(amax/448)）
xg, sfg = dgm.per_token_cast_to_fp8(xb, use_ue8m0=False)
bit_sf = torch.equal(sf, sfg)
ndiff = int((xf.view(torch.uint8) != xg.view(torch.uint8)).sum())
tot = xf.numel()
print(f'  B2 vs DeepGEMM(use_ue8m0=False): scale 逐位={bit_sf}  '
      f'fp8 不同元素={ndiff}/{tot}')
check('B2 DeepEP 与 DeepGEMM 的 fp32 scale 逐位一致', bit_sf)
check('B2b 但 fp8 有极少量元素差 1（乘子 448/amax vs 1/(amax/448) 的舍入差）',
      ndiff < tot * 0.01,
      f'{ndiff}/{tot} = {100.0 * ndiff / tot:.3f}%')

# B3 int-scale 解码分支 (:51-53)：仅当 scale 数 %4==0（无 padding）时 layout 对齐
m2, n2 = 64, 512
xb2 = (torch.randn(m2, n2, device=DEV) * 2).to(torch.bfloat16)
xf8_u, sf_u = dgm.per_token_cast_to_fp8(xb2, use_ue8m0=True)
xf8_up, sf_packed = dgm.per_token_cast_to_fp8(xb2, use_ue8m0=True, use_packed_ue8m0=True)
dec = (sf_packed.view(torch.uint8).to(torch.int) << 23).view(torch.float)
xbk_int = m.per_token_cast_back(xf8_u, sf_packed)
xbk_dec = m.per_token_cast_back(xf8_u, dec)
print(f'  B3 int-scale: packed={tuple(sf_packed.shape)} 解码 scale==UE8M0 fp32 scale? '
      f'{torch.equal(dec, sf_u)}  cast_back(int)==cast_back(fp32)? {torch.equal(xbk_int, xbk_dec)}')
check('B3 packed-int -> (uint8<<23) 解码 == DeepGEMM 的 UE8M0 fp32 scale',
      torch.equal(dec, sf_u))
check('B3b per_token_cast_back 的 int 分支 (:51-53) 与显式 fp32 scale 路径逐位一致',
      torch.equal(xbk_int, xbk_dec) and xbk_int.dtype == torch.bfloat16)

# B4 空张量早退 (:44-45)
xe = torch.empty(0, n_, dtype=torch.float8_e4m3fn, device=DEV)
sfe = torch.empty(0, (n_ + 127) // 128, device=DEV)
xbe = m.per_token_cast_back(xe, sfe)
check('B4 空张量早退 (:44-45) 返回 bf16 且 shape 不变', xbe.dtype == torch.bfloat16
      and tuple(xbe.shape) == (0, n_))

# B5 UE8M0 模式下：DeepEP 无 UE8M0，故其 scale 与 DeepGEMM 的 UE8M0 上取整 scale 不同
check('B5 DeepEP 量化器**不做** UE8M0 上取整（scale 非 2 的幂）',
      not bool(((sf.view(torch.int) & 0x7FFFFF) == 0).all()))

# ---------------------------------------------------------------------------------------
print()
print('Part C — inplace_unique: top-k 专家去重 oracle（combine 归约的语义规范）')
# ---------------------------------------------------------------------------------------
T, k, E = 2048, 8, 32
x = torch.randint(0, E, (T, k), device=DEV, dtype=torch.int32)
# 一半行掺入重复 + -1 padding
x[: T // 2, -2:] = x[: T // 2, :2]      # 制造重复
x[T // 2:, -3:] = -1                    # 制造 padding
orig = x.clone()
m.inplace_unique(x, E)


def oracle_unique(rows, num_slots):
    out = torch.full_like(rows, -1)
    for i in range(rows.size(0)):
        vals = sorted({int(v) for v in rows[i].tolist() if v >= 0}, reverse=True)
        out[i, :min(num_slots, len(vals))] = torch.tensor(
            vals[:min(num_slots, rows.size(1))], dtype=rows.dtype, device=rows.device)
    return out


ref = oracle_unique(orig, E)
check('C1 inplace_unique == 独立 oracle（去重、降序、-1 填充）', torch.equal(x, ref))
# 集合相等性
sets_ok = all(set(int(v) for v in x[i].tolist() if v >= 0)
              == set(int(v) for v in orig[i].tolist() if v >= 0) for i in range(T))
check('C1b 每行有效专家集合不变（仅去重/排序）', sets_ok)
# -1 全在尾部（有效位掩码沿行非增）
validmask = (x >= 0).to(torch.int)
noninc = bool((validmask[:, 1:] - validmask[:, :-1] <= 0).all())
check('C1c 所有 -1 落在每行尾部（有效位掩码非增）', noninc)

# C2 第一次按 count 排序 (:66) 是死代码：它只是为了把「出现 0 次」的 bin 移到某处再 mask，
#    但 mask 是**按 bin 下标**做的 (:67)，与排序结果无关；第二次按值排序 (:68) 才是决定输出的排序。
def unique_no_countsort(rows, num_slots):
    r = rows.clone()
    xp = r.masked_fill(r < 0, num_slots)
    bc = torch.zeros((r.size(0), num_slots + 1), dtype=r.dtype, device=r.device)
    bc.scatter_add_(1, xp, torch.ones_like(xp))
    bc = bc[:, :num_slots]
    idx = torch.arange(num_slots, device=r.device).expand(r.size(0), -1).clone()
    idx.masked_fill_(bc == 0, -1)
    idx = torch.sort(idx, descending=True, dim=-1).values
    out = torch.full_like(r, -1)
    vl = min(num_slots, r.size(1))
    out[:, :vl] = idx[:, :vl]
    return out


o2 = unique_no_countsort(orig, E)
check('C2 第一次按 count 的排序 (:66) 对最终结果无影响（真正的排序是 :68 按值降序）',
      torch.equal(o2, ref))

# C3 与 DeepGEMM combine 的 compute_topk_slots 顺序取向对比
#     P58: compute_topk_slots 用 __ffs 从低位挤到前 → **升序**；DeepEP 这里是降序
x_asc = x.clone()
check('C3 DeepEP 输出为**降序**（首列 >= 末列），对照 DeepGEMM combine 的升序去重',
      bool((x[:, 0] >= x[:, -1]).all()))

# ---------------------------------------------------------------------------------------
print()
print('Part D — create_grouped_scores: group-limited（device-limited）路由掩码')
# ---------------------------------------------------------------------------------------
tokens, experts, G = 512, 32, 4
eg = experts // G
scores = torch.randn(tokens, experts, device=DEV)
group_idx = torch.stack([torch.randperm(G, device=DEV)[:2] for _ in range(tokens)])
out = m.create_grouped_scores(scores.clone(), group_idx, G)
# 独立复刻：只保留被选中 group 的专家分数
sel = torch.zeros(tokens, G, dtype=torch.bool, device=DEV)
sel.scatter_(1, group_idx, True)
ref_d = (scores.view(tokens, G, eg) * sel.unsqueeze(-1)).view(tokens, experts)
check('D1 create_grouped_scores 仅保留被选中 group 的专家分数', torch.equal(out, ref_d))
# 每 token 非零专家数 == 选中组数(2) * eg(=8) = 16
nz_per_tok = (out != 0).sum(1)
check('D2 每 token 恰好保留 2 组的 16 个专家（未选中组整段清零）',
      bool((nz_per_tok == 2 * eg).all()),
      f'min={int(nz_per_tok.min())} max={int(nz_per_tok.max())} (期望 {2 * eg})')

# ---------------------------------------------------------------------------------------
print()
print('Part E — safe_div / hash_tensor(s) / count_bytes')
# ---------------------------------------------------------------------------------------
check('E1 safe_div(0,0)=0', m.safe_div(0, 0) == 0)
try:
    m.safe_div(3, 0)
    ok = False
except ZeroDivisionError:
    ok = True
check('E1b safe_div(3,0) 抛 ZeroDivisionError', ok)
check('E1c safe_div(6,3)=2', m.safe_div(6, 3) == 2)

a = torch.randint(-10, 10, (5, 4), device=DEV)
b = torch.randint(-10, 10, (3,), device=DEV)
nbytes_dep = m.count_bytes(a, None, (b, [a, None]))
nbytes_dg = dgn.count_bytes(a, None, (b, [a, None]))
print(f'  E2 count_bytes: DeepEP={nbytes_dep}  DeepGEMM={nbytes_dg}  '
      f'手算={a.numel()*a.element_size()*2+b.numel()*b.element_size()}')
check('E2 count_bytes 递归计数（跳过 None；两个库实现逐字相同）与手算一致',
      nbytes_dep == nbytes_dg == (a.numel() * a.element_size() * 2 + b.numel() * b.element_size()))

hi = m.hash_tensor(a)
check('E3 hash_tensor == int32 view 之和（与元素顺序无关）',
      hi == int(a.view(torch.int).sum().item()))
# XOR 递归
h1 = m.hash_tensors(a, (b, None))
h2 = m.hash_tensor(a) ^ m.hash_tensor(b)
check('E3b hash_tensors 递归 XOR 语义', h1 == h2)

print()
print('=' * 78)
print(f'结果: {"ALL PASS" if not FAILS else "FAIL: " + ", ".join(FAILS)}')
print('=' * 78)
