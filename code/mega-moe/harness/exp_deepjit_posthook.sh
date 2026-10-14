#!/usr/bin/env bash
# P37-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证 DeepJIT 的 post_hook 后处理机制 +
#   磁盘产物（meta.json / cubin）/ CUDA↔Ascend 后端抽象。
#   ./harness/exp_deepjit_posthook.sh
# 对应源码 src/deepjit_posthook.cpp，输出见 harness/exp_deepjit_posthook.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DJ=${HOME}/github/DeepGEMM/third-party/deep_jit/include
TORCH=/usr/local/lib/python3.12/dist-packages/torch
PYSITE=/usr/local/lib/python3.12/dist-packages
SRC=${HOME}/proj/tech_record/code/mega-moe/src/deepjit_posthook.cpp

echo "== DeepJIT 版本坐标 =="
( cd ${HOME}/github/DeepGEMM/third-party/deep_jit && git log --oneline -1 ) 2>/dev/null || true
sha256sum $DJ/deep_jit/backend/cuda/options.hpp $DJ/deep_jit/backend/cuda/backend.hpp \
          $DJ/deep_jit/runtime/runtime.hpp $DJ/deep_jit/runtime/config.hpp \
          $DJ/deep_jit/cache/disk.hpp $DJ/deep_jit/backend/ascend/options.hpp
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1

echo
echo "== 编译 DeepJIT 宿主实验程序（直接 include 真实头文件）=="
rm -rf /tmp/deepjit_posthook /tmp/deepjit_posthook_root
g++ -std=c++20 -O2 \
  -I$DJ -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
  -I$PYSITE/pybind11/include -I/usr/include/python3.12 -I/usr/local/cuda/include \
  "$SRC" -o /tmp/deepjit_posthook \
  -L$TORCH/lib -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu \
  -Wl,-rpath,$TORCH/lib \
  -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcudart -lpython3.12 -ldl -lpthread
echo "build OK"

echo
echo "== 主实验（单进程）=="
/tmp/deepjit_posthook > /tmp/deepjit_posthook.raw.txt
cat /tmp/deepjit_posthook.raw.txt

echo
echo "== 跨进程冷读取 hook 内容（同一路径、两次独立进程）=="
/tmp/deepjit_posthook coldwrite "AAA" >> /tmp/deepjit_posthook.raw.txt
/tmp/deepjit_posthook coldwrite "BBB" >> /tmp/deepjit_posthook.raw.txt
grep '^COLDWRITE' /tmp/deepjit_posthook.raw.txt

echo
echo "== Python 侧判定 =="
python3 - <<'PY'
import json, re, os
raw = open('/tmp/deepjit_posthook.raw.txt').read()
lines = raw.splitlines()
ok = True
def fail(msg):
    global ok; ok = False; print('  !! ' + msg)

def one(pat):
    m = [l for l in lines if re.match(pat, l)]
    return m[-1] if m else None

# 1) 基线产物
l = one(r'NOHOOK files')
if l:
    d = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', l)}
    print('1) 无 hook 产物: cu/cubin/meta/committed = %d/%d/%d/%d' %
          (d['cu'], d['cubin'], d['meta'], d['committed']))
    if not all(d.values()): fail('基线产物缺文件')
mpath = one(r'NOHOOK meta_path=(\S+)').split(' meta_path=')[-1]
meta = json.load(open(mpath))
print('   meta.json post_hook = %r  (期望 null)' % meta['compiler_options']['post_hook'])
if meta['compiler_options']['post_hook'] is not None: fail('无 hook 时 meta.post_hook 非 null')
print('   meta.json command 去换行首段: %s' % meta['command'].split()[0])

# 2) 良性 hook
l = one(r'MARK digest=(\S+) distinct_from_nohook=(\d+) mark_lines=(-?\d+)')
d1, dist, ml = l.split()[1].split('=')[1], int(l.split()[2].split('=')[1]), int(l.split()[3].split('=')[1])
print('2) hook digest=%s distinct=%d hook 跑的副作用行数=%d' % (d1, dist, ml))
if dist != 1: fail('加 hook 未改变缓存键')
if ml != 1: fail('hook 未执行（副作用缺失）')
mpath = one(r'MARK meta_path=(\S+)').split(' meta_path=')[-1]
meta = json.load(open(mpath))
print('   meta.json post_hook = %r (期望 hooks/h_mark.py)' % meta['compiler_options']['post_hook'])
if meta['compiler_options']['post_hook'] != 'hooks/h_mark.py': fail('meta 未记录 hook 路径')

# 3) hook 内容哈希 + 线程内缓存
l = one(r'HOOKHASH before=(\S+) after_inproc_modify=(\S+) threadlocal_cached=(\d+)')
cached = int(l.split()[-1].split('=')[1])
print('3) 同进程改 hook 文件后哈希不变(线程内按路径缓存) = %d' % cached)
if cached != 1: fail('线程内 hook 哈希缓存未生效')
l = one(r'MARK2 digest=(\S+) distinct=(\d+)')
if int(l.split()[-1].split('=')[1]) != 1: fail('新 hook 路径未产生新键')
print('   新路径 h_mark2.py → 新 digest %s' % l.split()[1].split('=')[1])

# 4) override 语义
l = one(r'OVERRIDE .*nullopt_inherits=(\d+) empty_is_configured=(\d+) empty_value=(\S*)')
d = dict(re.findall(r'(nullopt_inherits|empty_is_configured|empty_value)=(\S*)', l))
inh = int(d['nullopt_inherits']); emp = int(d['empty_is_configured'])
print('4) override: nullopt 继承=%d 空串仍算已配置=%d (值=%r)' % (inh, emp, d['empty_value']))
if inh != 1: fail('nullopt 未继承默认 hook')
if emp != 1: fail('空串未被当作已配置 hook')

# 5) 失败 hook
l = one(r'FAILHOOK threw=(\d+) published_hit=(\d+)')
threw, pub = int(l.split()[1].split('=')[1]), int(l.split()[2].split('=')[1])
print('5) 失败 hook: 抛异常=%d 被错误发布=%d' % (threw, pub))
if threw != 1 or pub != 0: fail('失败 hook 语义错误')

# 6) 破坏 hook
l = one(r'CORRUPT digest=(\S+) cubin_head=(\S+)')
head = l.split()[2].split('=')[1]
print('6) 破坏 hook: 产物 cubin 头 = %r（说明 hook 在发布前就地改写）' % head)
if head != 'HOOK': fail('hook 未作用于待发布产物')
l = one(r'CORRUPT load_threw=(\d+)')
if int(l.split('=')[1]) != 1: fail('坏 cubin 竟然 load 成功')
print('   加载坏 cubin 失败 = %s' % l.split('=')[1])

# 7) 命中不重跑 hook
l = one(r'LOAD launched=(\d+) mark_lines_after_load=(-?\d+)')
ll = int(l.split()[1].split('=')[1]); ml = int(l.split()[2].split('=')[1])
print('7) compile() 加载+发射=%d；launch 后副作用行数仍=%d（未重跑）' % (ll, ml))
if ll != 1 or ml != 1: fail('cache 命中重跑了 hook')
l = one(r'LOAD disk_hit_same_obj=(\d+) mark_lines_after_disk_hit=(-?\d+)')
dl = int(l.split()[1].split('=')[1]); ml2 = int(l.split()[2].split('=')[1])
print('   清内存缓存后磁盘命中对象非空=%d 副作用行数=%d' % (dl, ml2))
if dl != 1 or ml2 != 1: fail('磁盘命中重跑了 hook')

# 8) 跨进程冷读取
cw = [l for l in lines if l.startswith('COLDWRITE')]
if len(cw) == 2:
    h0 = cw[0].split('hookhash=')[1].split()[0]
    h1 = cw[1].split('hookhash=')[1].split()[0]
    d0 = cw[0].split('digest=')[1].split()[0]
    d1b = cw[1].split('digest=')[1].split()[0]
    print('8) 跨进程冷读取: hookhash %s vs %s；digest %s vs %s' % (h0[:12], h1[:12], d0[:12], d1b[:12]))
    if h0 == h1 or d0 == d1b: fail('冷读取未区分不同 hook 内容')
else:
    fail('缺少 COLDWRITE 输出')

print()
print('总判定: ' + ('ALL PASS' if ok else 'FAIL'))
PY
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
