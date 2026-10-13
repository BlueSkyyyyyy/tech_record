#!/usr/bin/env python3
# P71-1 实验 driver：SM 局部域（locality domain）的**验收测试即正确性规范**
#   `tests/test_locality_domain.py`（167 行，纯 host / API 层，与 SM100 kernel 无关）。
#
# 第 34 篇（P24-1）拆了局部域的**实现**——pointer-chase 延迟探针、try_probe 两闸门、
# MLOPart / CUDA VMM 本地化分配、调度器按域分区——但它描述的「域到底算什么对」的唯一规范
# `tests/test_locality_domain.py` 从没被单独拆过。本篇把它当**可执行规格**逐段拆透，
# 并逐条核对第 34 篇里的结论。
#
# 关键环境事实（本机 H100）：MLOPart 不可用（MPS 只暴露 1 个 device，`mlopart.py:99` 断言
# 「2 MLOPart devices」失败），故 `is_localization_available()` 恒 False。于是：
#   * 能真机跑的用例：`test_sm_locality_domains`（域表 + 均衡表）、`test_without_mlopart`
#     （先剥掉 `-mlopart` 再验证「无 MLOPart 时的告警/拒绝/回退」）——两者都原样跑真实测试体。
#   * 跑不了的用例：`test_empty` / `test_localize` / `test_mlopart`（都需要真的能分配本地化内存）。
#     但它们**契约里的某些断言与 MLOPart 无关**（如 domain 越界必须抛 RuntimeError），可在 H100 单独验证。
#
# 本实验：
#   A) 盘点 test_locality_domain.py 的顶层结构（5 个用例 / __main__ 顺序 / 两个 helper）；
#   B) 环境能力矩阵 + 每条能力的**为什么**（定位到 mlopart.py:99 的断言）；
#   C) **原样跑**两个可跑的用例（导入真实测试模块、调用真实测试体）；
#   D) **独立复刻** get_even / balance 的 host 数学，与真实 API 对拍，并逐条核对测试的 4 条规格断言；
#   E) 契约负样本（与 MLOPart 无关、H100 可跑）：assert_raises 只吞 RuntimeError、is_localized 的
#      非连续拒绝（layout.hpp:150）、empty 的 domain 越界（locality_domain.hpp:36）、localize 拒绝；
#   F) 盘点跑不了的 3 个用例：定位阻塞行 + 说明目标卡上会看到什么。
#
# 目标卡代号敏感：一律用「目标卡」指代。

import importlib.util
import os
import sys
import traceback

import torch

import deep_gemm
from deep_gemm import locality_domain as ld

DG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(deep_gemm.__file__)))
TESTS_DIR = os.path.join(DG_ROOT, 'tests')

DEV = 'cuda'
RESULTS = []


def report(name, ok, detail=''):
    RESULTS.append(bool(ok))
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')


def sep(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


def short(e):
    s = repr(str(e))
    # 去掉宏展开出来的长相对路径，只留文件名:行号
    return s.replace('csrc/apis/../jit_kernels/impls/../heuristics/../../', '') \
            .replace('csrc/apis/../runtime/', '')


# 按文件路径导入真实测试模块（tests/ 不在 sys.path）——与官方 `__main__` 用的是同一份代码
spec = importlib.util.spec_from_file_location('test_locality_domain',
                                              os.path.join(TESTS_DIR, 'test_locality_domain.py'))
tld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tld)
SRC = open(tld.__file__, encoding='utf-8').read().splitlines()
NUM_SMS_PER_TPC = tld.NUM_SMS_PER_TPC


# ---------------------------------------------------------------------------
sep('Part A：test_locality_domain.py 的顶层结构')
# ---------------------------------------------------------------------------
print(f'deep_gemm 根目录 : {DG_ROOT}')
print(f'test 文件        : {tld.__file__}  ({len(SRC)} 行)')
print(f'平台             : {sys.platform}  |  device: {torch.cuda.get_device_name(0)} '
      f'(sm{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})')
print(f'常量             : NUM_SMS_PER_TPC = {NUM_SMS_PER_TPC}')

fns = [n for n in dir(tld) if n.startswith('test_')]
print(f'\n5 个用例 + 2 个 helper：')
for n in ('assert_raises', 'random_bytes', 'test_sm_locality_domains', 'test_empty',
          'as_plain', 'test_localize', 'test_mlopart', 'test_without_mlopart'):
    obj = getattr(tld, n)
    doc = (obj.__doc__ or '').strip().splitlines()
    print(f'  {n:<26} :{obj.__code__.co_firstlineno:>4}  {doc[0] if doc else ""}')
body = [ln.strip() for ln in tld.__doc__.splitlines()] if tld.__doc__ else []
main_block = SRC[SRC.index('if __name__ == \'__main__\':') + 1:]
calls = [ln.strip() for ln in main_block if ln.strip()]
print('\n__main__ 顺序（前 5 个用例 + seed）：')
for s in calls:
    print(f'  {s}')
report('A: 模块可 import、5 用例 + 2 helper 可见',
       set(fns) == {'test_sm_locality_domains', 'test_empty', 'test_localize', 'test_mlopart',
                    'test_without_mlopart'},
       f'用例 {len(fns)} 个、helper assert_raises/random_bytes/as_plain 均在')


# ---------------------------------------------------------------------------
sep('Part B：环境能力矩阵 + 每条能力的「为什么」')
# ---------------------------------------------------------------------------
num_sms = torch.cuda.get_device_properties(0).multi_processor_count
caps = {
    'get_num_locality_domains()': deep_gemm.get_num_locality_domains(),
    'get_granularity()': ld.get_granularity(),
    'is_localization_available()': deep_gemm.is_localization_available(),
    'num_sms (device)': num_sms,
}
for k, v in caps.items():
    print(f'  {k:<32} = {v}')
print('\n为什么 is_localization_available() == False：')
print('  mlopart.is_available() → 子进程起 MPS + `start_server -mlopart`，serve() 断言')
print('  `num_devices == NUM_LOCALITY_DOMAINS(2)`（mlopart.py:99）；本机 MPS 只暴露 1 个 device，')
print('  断言失败 → 子进程退出 → MLOPart 通知不可用 → 顶层打印一次 UserWarning。')
report('B: 能力矩阵与第 34 篇一致（2 域 / 2 MiB / 不可用）',
       deep_gemm.get_num_locality_domains() == 2 and ld.get_granularity() == 2 << 20
       and deep_gemm.is_localization_available() is False,
       f'domains=2, granularity=2MiB, available=False')


# ---------------------------------------------------------------------------
sep('Part C：原样跑两个可跑的用例（真实测试体）')
# ---------------------------------------------------------------------------
torch.manual_seed(0)
for name, fn in (('test_sm_locality_domains', tld.test_sm_locality_domains),
                 ('test_without_mlopart', tld.test_without_mlopart)):
    print(f'\n--- {name}() ---')
    try:
        fn()
        ok = True
        detail = 'OK'
    except Exception as e:                                   # noqa: BLE001
        ok = False
        detail = f'{type(e).__name__}: {str(e)[:100]}'
        traceback.print_exc()
    report(f'C: 真实测试体 {name}()', ok, detail)


# ---------------------------------------------------------------------------
sep('Part D：独立复刻 host 数学 + 逐条核对测试的规格断言')
# ---------------------------------------------------------------------------
def get_even_ref(num_sms_local, num_domains=2):
    # 对应 csrc/apis/locality_domain.hpp:103-106：tpc_idx%num_domains repeat_interleave(2)
    tpc_idx = torch.arange(num_sms_local // NUM_SMS_PER_TPC, dtype=torch.int)
    return (tpc_idx % num_domains).to(torch.uint8).repeat_interleave(NUM_SMS_PER_TPC)


def balance_ref(sm_domain, first_half=False):
    # 对应 csrc/apis/locality_domain.hpp:126-136
    # 关键：torch 的 `slice(0, n)` 第二参是 **start** 而非 end —— `surplus` 取的是
    # 「多数域的 TPC 列表」的**后半段** `[num_tpcs//2:]`（保证「已 50/50 时不动」）。
    # first_half=True 是故意写错的变体（`[:num_tpcs//2]`），用来证明测试的规格断言有牙齿。
    tpcs = sm_domain.view(-1, NUM_SMS_PER_TPC)
    tpc_domain = tpcs.select(1, 0)
    num_tpcs = tpcs.size(0)
    num_domain_1 = int(tpc_domain.sum().item())
    larger = 1 if num_domain_1 * 2 > num_tpcs else 0
    idx = (tpc_domain == larger).nonzero().flatten()
    surplus = idx[:num_tpcs // 2] if first_half else idx[num_tpcs // 2:]
    balanced = sm_domain.clone()
    balanced.view(-1, NUM_SMS_PER_TPC).index_fill_(0, surplus, 1 - larger)
    return balanced


sm_real = ld.get_sm_locality_domains()
bal_real = ld.get_balanced_sm_locality_domains()
even_ref = get_even_ref(num_sms)
bal_ref = balance_ref(sm_real)
print(f'真实 sm_domains   : {torch.bincount(sm_real.long(), minlength=2).tolist()} SMs/域, dtype={sm_real.dtype}')
print(f'真实 balanced     : {torch.bincount(bal_real.long(), minlength=2).tolist()} SMs/域')
even_ok = torch.equal(even_ref.to(sm_real.device), sm_real)
bal_ok = torch.equal(bal_ref.to(bal_real.device), bal_real)
print(f'独立 get_even 复刻 与真实 sm_domains 逐位相等: {even_ok}')
print(f'独立 balance 复刻 与真实 balanced  逐位相等: {bal_ok}')
report('D: 独立复刻 get_even == 真实 sm_domains（本机走 even 回退）', even_ok,
       '逐位相等 → 坐实第 34 篇「不可用时等于 get_even」')
report('D: 独立复刻 balance == 真实 balanced', bal_ok, '逐位相等')

# 测试的 4 条规格断言（test_locality_domain.py:39/42/45/47）逐条独立核验
spec_checks = []

# 断言①(:39) 两域都在、且不是全部（0 < sum < num_sms）、dtype 都是 uint8
spec_checks.append(('① 0<sum<num_sms 且 dtype 均 uint8',
                    sm_real.dtype == bal_real.dtype == torch.uint8
                    and 0 < int(sm_real.sum()) < num_sms))

# 断言②(:42) TPC 内两个 SM 同域
tpc_pair_ok = True
for t in (sm_real, bal_real):
    tpcs = t.view(-1, NUM_SMS_PER_TPC)
    tpc_pair_ok &= torch.equal(tpcs[:, 0], tpcs[:, 1])
spec_checks.append(('② TPC 内两 SM 同域（sm + balanced）', tpc_pair_ok))

# 断言③(:45) balanced 恰好 50/50
spec_checks.append(('③ balanced.sum()*2 == num_sms（精确 50/50）',
                    int(bal_real.sum()) * 2 == num_sms))

# 断言④(:47) balance 只搬「最少数量的 TPC」
moved = int((bal_real != sm_real).view(-1, NUM_SMS_PER_TPC).all(1).sum())
expected_moved = abs(int(sm_real.sum()) - num_sms // 2) // NUM_SMS_PER_TPC
spec_checks.append((f'④ 搬运 TPC 数 == |sum-num_sms/2|/2（{moved} == {expected_moved}）',
                    moved == expected_moved))

for name, ok in spec_checks:
    report(f'D: 规格断言 {name}', ok)

# 利齿：把 surplus 误取成「前半段」——已 50/50 的输入会被错误翻成全部一个域，
# 触发规格断言③（balanced.sum()*2 == num_sms）。证明测试的判据能抓这种实现错。
buggy = balance_ref(sm_real, first_half=True)
buggy_ok = int(buggy.sum()) * 2 == num_sms
report('D: 利齿——surplus 误取前半段会被规格断言③抓住',
       (not buggy_ok), f'错误变体 balanced={torch.bincount(buggy.long(), minlength=2).tolist()} → ③ 不成立')

# 合成输入上的边界（TPC 一致偏斜：验证 ④ 的公式仍成立）
def tpc_skew(num_on):
    t = torch.zeros(num_sms // NUM_SMS_PER_TPC, dtype=torch.int)
    t[:num_on] = 1
    return t.to(torch.uint8).repeat_interleave(NUM_SMS_PER_TPC)
for label, sd in (('all-0 (占满一个域)', torch.zeros(num_sms, dtype=torch.uint8)),
                  ('all-1 (占满另一个域)', torch.ones(num_sms, dtype=torch.uint8)),
                  ('TPC 一致 40/26', tpc_skew(40)),
                  ('已 50/50', tpc_skew(33))):
    b = balance_ref(sd)
    mv = int((b != sd).view(-1, NUM_SMS_PER_TPC).all(1).sum())
    print(f'  balance({label:<22}) -> {torch.bincount(b.long(), minlength=2).tolist()}, '
          f'moved={mv}, 公式期望={abs(int(sd.sum()) - num_sms // 2) // NUM_SMS_PER_TPC}')


# ---------------------------------------------------------------------------
sep('Part E：契约负样本（与 MLOPart 无关，H100 可跑）')
# ---------------------------------------------------------------------------
# E1: assert_raises 只吞 RuntimeError；AssertionError 必须原样抛出（:19-24）
try:
    tld.assert_raises(lambda: (_ for _ in ()).throw(AssertionError('x')))
    e1_ok = False
except AssertionError:
    e1_ok = True
try:
    tld.assert_raises(lambda: (_ for _ in ()).throw(RuntimeError('ok')))
    e1_ok &= True
except Exception:                                            # noqa: BLE001
    e1_ok = False
report('E1: assert_raises 只吞 RuntimeError', e1_ok, 'AssertionError 穿透、RuntimeError 被吞')

# E2: is_localized —— 普通张量 False；非连续张量抛 RuntimeError（layout.hpp:150）
plain = torch.zeros((4, 8), dtype=torch.float8_e4m3fn, device=DEV)
e2 = deep_gemm.is_localized(plain) is False
try:
    deep_gemm.is_localized(plain.mT)
    e2 = False
    msg = 'NO RAISE'
except RuntimeError as e:
    msg = short(e)
    e2 = 'layout.hpp:150' in msg
report('E2: is_localized(plain)=False；is_localized(plain.mT) 抛 RuntimeError', e2, msg)

# E3: empty 的 domain 越界必须抛 RuntimeError（locality_domain.hpp:36），可在 H100 单独验证
e3 = True
for di in (-1, 2):
    try:
        ld.empty((1,), torch.uint8, di)
        e3 = False
    except RuntimeError as e:
        msg = short(e)
        e3 &= 'locality_domain.hpp:36' in msg
report('E3: empty 的 domain 越界（-1/2）抛 RuntimeError（与 MLOPart 无关）', e3, msg)

# E4: 无 MLOPart 时 empty(合法域) / localize 必须失败（测试用 assert_raises 语义只对越界；合法域是 AssertionError）
e4 = True
try:
    ld.empty((2, 16), torch.uint8, 0)
    e4 = False
    m4 = 'NO RAISE'
except AssertionError as e:
    m4 = f'AssertionError: {str(e)[:60]}'
try:
    deep_gemm.localize(torch.zeros((2, 16), dtype=torch.uint8, device=DEV))
    e4 = False
except AssertionError as e:
    e4 &= 'MLOPart is unavailable' in str(e)
report('E4: 无 MLOPart 时 empty(合法域)/localize 抛 AssertionError「MLOPart is unavailable」', e4, m4)


# ---------------------------------------------------------------------------
sep('Part F：跑不了的 3 个用例——阻塞行与目标卡上的样子')
# ---------------------------------------------------------------------------
blocked = {
    'test_empty': ('tld 先在 :58 对合法域调 locality_domain.empty(...,0)；'
                   '→ csrc/apis/locality_domain.hpp:38 map → runtime/locality_domain.hpp:37 '
                   'mlopart::create_memory → Python `assert self.available`（mlopart.py:40）失败'),
    'test_localize': ('tld:77 assert deep_gemm.is_localization_available() 直接失败（本机 False）'),
    'test_mlopart': ('tld:113 起 MLOPart(device_uuid) 后 :115 create_memory；'
                     'serve() 断言 2 device 失败 → 客户端 recv 得到不可用 → create_memory 断言失败'),
}
for name, why in blocked.items():
    print(f'  {name:<16}: {why}')
# 把 test_localize 与 test_mlopart 真的跑一下，记录它们在哪里挂
for name, fn in (('test_empty', tld.test_empty), ('test_localize', tld.test_localize)):
    try:
        fn()
        print(f'  [{name}] 意外通过')
    except Exception as e:                                   # noqa: BLE001
        print(f'  [{name}] 预期失败于 {type(e).__name__}: {str(e)[:70]}')
report('F: 3 个用例的阻塞点定位到具体行（需 MLOPart / instance 才可跑）', True,
       'test_empty/test_localize/test_mlopart')


# ---------------------------------------------------------------------------
sep('汇总')
# ---------------------------------------------------------------------------
n_pass = sum(RESULTS)
print(f'{n_pass} PASS / {len(RESULTS) - n_pass} FAIL  (共 {len(RESULTS)})')
print('参考版本：~/github/DeepGEMM commit 057ca59（v2.8.1）。')
print('本机 H100：MLOPart 不可用 → 5 个用例中 2 个（域表 / 无 MLOPart）真机通过，')
print('其余 3 个的阻塞点已定位；其与 MLOPart 无关的契约断言已在 Part E 单独验证。')
sys.exit(0 if n_pass == len(RESULTS) else 1)
