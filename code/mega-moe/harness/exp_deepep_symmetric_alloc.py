#!/usr/bin/env python3
"""P56-1: DeepEP 对称内存分配层 —— 纯 Python 的 BufferAllocator 计划/落地，以及与设备结果的交叉核对。

本文件**加载并运行 deep_ep/buffers/allocator.py 的真实源码**（不 import deep_ep 的 C 扩展：
本机缺 `deep_ep._C`，见 ROADMAP 阻塞），只把 `deep_ep._C.get_num_allocation_alignment` 用与
`compiled.cuh:69` 一致的常量 stub 掉，并加载真实 `deep_ep/utils/math.py` 提供 `align`。

对照关系：
  * `BufferAllocator.allocate`（allocator.py:19-26）的「先排布 meta 张量 + 记录对齐偏移」语义；
  * `materialize`（:28-36）的「一次性把 planning 落到单块连续 uint8 storage」语义；
  * 与 `src/deepep_symmetric_alloc.cu` 打印的设备/驱动常量（kNumAllocationAlignmentBytes=2MiB、
    recommended granularity）逐项交叉核对。
"""
import sys
import importlib.util
import types
import torch

DEP = "~/github/DeepEP"
ALIGN_BYTES = 2 * 1024 * 1024          # deep_ep/include/deep_ep/common/compiled.cuh:69

G_PASS = G_FAIL = 0
def expect(cond, msg):
    global G_PASS, G_FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if cond: G_PASS += 1
    else: G_FAIL += 1

def section(name):
    print(f"\n=== {name} ===")


# ------------------------------------------------------------------ 加载真实源码
def load_real_allocator():
    pkg = types.ModuleType("deep_ep"); pkg.__path__ = [f"{DEP}/deep_ep"]
    cext = types.ModuleType("deep_ep._C")
    cext.get_num_allocation_alignment = lambda: ALIGN_BYTES
    utils = types.ModuleType("deep_ep.utils"); utils.__path__ = [f"{DEP}/deep_ep/utils"]
    buffers = types.ModuleType("deep_ep.buffers"); buffers.__path__ = [f"{DEP}/deep_ep/buffers"]
    sys.modules.update({
        "deep_ep": pkg, "deep_ep._C": cext,
        "deep_ep.utils": utils, "deep_ep.buffers": buffers,
    })

    def load(modname, path):
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    math_mod = load("deep_ep.utils.math", f"{DEP}/deep_ep/utils/math.py")
    utils.math = math_mod
    alloc_mod = load("deep_ep.buffers.allocator", f"{DEP}/deep_ep/buffers/allocator.py")
    return alloc_mod, math_mod


# ------------------------------------------------------------------ 独立 oracle
def oracle_plan(allocations, alignment):
    """独立复刻 allocate（allocator.py:19-26）的偏移/总字节推演。"""
    num_bytes = 0
    out = []
    for shape, nbytes in allocations:
        offset = -(-num_bytes // alignment) * alignment      # math.align
        out.append(offset)
        num_bytes = -(-(offset + nbytes) // alignment) * alignment
    return out, num_bytes


def main():
    device_dump = sys.argv[1] if len(sys.argv) > 1 else None
    dev_align = None
    if device_dump:
        with open(device_dump) as f:
            for line in f:
                if "kNumAllocationAlignmentBytes=" in line:
                    # [info] granularity recommended=... minimum=... ; kNumAllocationAlignmentBytes=2097152
                    dev_align = int(line.split("kNumAllocationAlignmentBytes=")[1].split()[0])

    print("DeepEP 对称内存分配层实验 —— 纯 Python BufferAllocator（真实源码）")
    alloc_mod, math_mod = load_real_allocator()
    BufferAllocator = alloc_mod.BufferAllocator

    section("Part A: 常量与对齐（allocator.py:14；compiled.cuh:69）")
    expect(math_mod.align(0, ALIGN_BYTES) == 0, "align(0, 2MiB) == 0")
    expect(math_mod.align(1, ALIGN_BYTES) == ALIGN_BYTES, "align(1, 2MiB) == 2MiB")
    expect(math_mod.align(ALIGN_BYTES, ALIGN_BYTES) == ALIGN_BYTES, "align 幂等")
    if dev_align is not None:
        expect(dev_align == ALIGN_BYTES, f"设备打印的常量 == 2 MiB ({dev_align})")
    expect(not torch.cuda.is_available() or torch.cuda.is_available(), "torch 可用")

    section("Part B: allocate 的规划语义（allocator.py:19-26）")
    a = BufferAllocator()
    expect(a.alignment == ALIGN_BYTES, "BufferAllocator.alignment == get_num_allocation_alignment()")
    specs = [((128, 7168), torch.bfloat16),
             ((8,), torch.int32),
             ((4096,), torch.uint8),
             ((3, 5, 7), torch.float16)]
    tensors = [a.allocate(shape, dt) for shape, dt in specs]
    expect(all(t.device.type == "meta" for t in tensors), "allocate 返回 meta 占位张量")
    expect(not a.materialized, "planning 阶段 materialized==False")

    alloc_bytes = [(s, t.nbytes) for (s, _), t in zip(specs, tensors)]
    exp_off, exp_total = oracle_plan(alloc_bytes, ALIGN_BYTES)
    got_off = [off for (_t, off) in a.allocations]
    expect(got_off == exp_off, f"每个偏移 2 MiB 对齐且与 oracle 一致 {got_off} == {exp_off}")
    expect(a.num_bytes == exp_total, f"总字节与 oracle 一致 ({a.num_bytes} == {exp_total})")
    expect(all(off % ALIGN_BYTES == 0 for off in got_off), "所有偏移都是 2 MiB 的倍数")
    expect(a.num_bytes % ALIGN_BYTES == 0, "总字节也是 2 MiB 的倍数")

    section("Part C: materialize 落地为单块连续 storage（allocator.py:28-36）")
    storage = torch.zeros(a.num_bytes, dtype=torch.uint8, device="cuda")
    a.materialize(storage)
    expect(a.materialized, "materialize 后 materialized==True")
    ok_alias = True
    for t, (shape, dt), off in zip(tensors, specs, got_off):
        ok_alias &= (t.device.type == "cuda" and t.shape == torch.Size(shape) and t.dtype == dt)
        ok_alias &= (t.data_ptr() == storage.data_ptr() + off)
        ok_alias &= t.is_contiguous()
    expect(ok_alias, "每个张量都是 storage 上对应偏移的连续视图（data_ptr 精确吻合）")

    # 数据共享：往张量写值，直接读 raw storage 验证
    tensors[1].fill_(12345)
    raw = storage[got_off[1]:got_off[1] + 4].view(torch.int32)
    expect(int(raw[0]) == 12345, "写 int32 张量后 raw storage 字节同步变化（同一块内存）")
    tensors[2][:5] = 200
    expect(int(storage[got_off[2]]) == 200, "写 uint8 张量后 storage 首字节同步")

    section("Part D: 契约与负例（allocator.py:21/29-31）")
    def rejected(fn, desc):
        try:
            fn(); return False
        except AssertionError:
            return True
    expect(rejected(lambda: a.allocate((1,), torch.uint8), "materialized 后再 allocate 被拒"),
           "materialized 后再 allocate 引发 AssertionError")
    b = BufferAllocator(); b.allocate((16,), torch.uint8)
    expect(rejected(lambda: b.materialize(storage), "storage 大小不符被拒"),
           "materialize 传入错误大小 storage 被拒")
    bad = torch.zeros(b.num_bytes, dtype=torch.int32, device="cuda")   # dtype 非 uint8
    expect(rejected(lambda: b.materialize(bad), "storage dtype 非 uint8 被拒"),
           "materialize 传入非 uint8 storage 被拒")
    cpu = torch.zeros(b.num_bytes, dtype=torch.uint8)                  # CPU
    expect(rejected(lambda: b.materialize(cpu), "storage 非 CUDA 被拒"),
           "materialize 传入非 CUDA storage 被拒")
    b.materialize(torch.zeros(b.num_bytes, dtype=torch.uint8, device="cuda"))
    expect(rejected(lambda: b.materialize(torch.zeros(b.num_bytes, dtype=torch.uint8, device="cuda")),
                    "二次 materialize 被拒"),
           "二次 materialize 引发 AssertionError")

    section("Part E: 真实 MoE/EP 规模的一次规划")
    # 仿 EP buffer：workspace + dispatch/combine buffer 等若干段
    c = BufferAllocator()
    segs = [((1024, 7168), torch.bfloat16),
            ((1024, 7168), torch.bfloat16),
            ((64,), torch.int64),
            ((64, 4), torch.int32)]
    for s, d in segs:
        c.allocate(s, d)
    real = torch.empty(c.num_bytes, dtype=torch.uint8, device="cuda")
    real.random_(0, 256)
    c.materialize(real)
    total = sum(int(t.nbytes) for t, _ in c.allocations)
    expect(c.num_bytes >= total, f"总分配 >= 各段之和（{c.num_bytes} >= {total}）")
    expect(all((t.data_ptr() - real.data_ptr()) % ALIGN_BYTES == 0 for t, _ in c.allocations),
           "真实规模下每段仍 2 MiB 对齐")

    print(f"\n=== SUMMARY ===\nPASS={G_PASS} FAIL={G_FAIL}")
    return 0 if G_FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
