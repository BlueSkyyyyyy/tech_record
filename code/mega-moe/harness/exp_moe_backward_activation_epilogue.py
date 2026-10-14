#!/usr/bin/env python3
# P146-1 host 侧验证：`epilogue_class` 的 host 接线（第 156 篇配套）
#   ① `get_epilogue_operator_type()` 返回的**类型字符串** 1:1 映射到 `operators.cuh` 的 device struct；
#   ② JIT 源码注入：`get_epilogue_operator_type()` 被 std::format 填进 kernel 模板实参（哪一段、第几个 {}）；
#   ③ 重实现 `resolve_epilogue_class` 的三输入互斥（epilogue_class / alpha / sfd），核对每种组合；
#   ④ 结论：SwiGLU 反向需要「两输出 + 两 SFD」，塞不进当前单 SED 的契约。
# 纯 host / Python，架构无关；行号对着 ~/github/DeepGEMM 真实核对。
import os, re, sys

DG = os.environ.get("DEEPGEMM_DIR", os.path.expanduser("~/github/DeepGEMM"))
EPI = os.path.join(DG, "csrc/jit_kernels/impls/epilogue_class.hpp")
OPC = os.path.join(DG, "deep_gemm/include/deep_gemm/epilogue/operators.cuh")
BF16HPP = os.path.join(DG, "csrc/jit_kernels/impls/sm100_bf16_gemm.hpp")

def read(p):
    with open(p) as f:
        return f.read()

def sec(name):
    print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)

# ---------------------------------------------------------------------------
sec("① host 类 → device operator 的类型字符串映射")
epi = read(EPI)
# 抓 "epilogue::operators::Xxx"
types = re.findall(r'return\s+"(epilogue::operators::\w+)"', epi)
opc = read(OPC)
structs = re.findall(r'struct\s+(\w+)\s*[:\{]', opc)
print(f"  epilogue_class.hpp 返回的 operator 类型字符串（{len(types)} 个）：")
for t in types:
    name = t.split("::")[-1]
    ok = "OK" if name in structs else "MISSING"
    print(f"    {t:42s} -> operators.cuh struct {name:24s} [{ok}]")
print(f"  operators.cuh 定义的 operator struct：{structs}")
# 断言：每个 host 类型字符串都能在 operators.cuh 找到 struct
assert all(t.split("::")[-1] in structs for t in types), "类型字符串与 device struct 不匹配"
print("  => host 类型字符串与 device operator 1:1；新增 operator 的**唯一**接线点就是这里多返回一个字符串。")

# ---------------------------------------------------------------------------
sec("② JIT 源码注入：类型字符串成了 kernel 模板实参")
bf = read(BF16HPP)
# 找 std::format(R"( ... )", ...) 里的模板体
m = re.search(r'jit->compile\(tag,\s*std::format\(R"\((.*?)\)"', bf, re.S)
tpl = m.group(1)
n_brace = tpl.count("{}")
print(f"  sm100_bf16_gemm.hpp 的 JIT 模板体有 {n_brace} 个 '{{}}' 占位符。")
# 找 format 实参里 operator 类型的那一项（get_epilogue_operator_type()）
body_after = bf[m.end():]
# 第 4 行末尾就是 args.epilogue_class->get_epilogue_operator_type(), args.gemm_desc.tc_util
print("  模板实参里与 epilogue 相关的两个（sm100_bf16_gemm.hpp:69-70）：")
for line in bf.splitlines():
    if "get_epilogue_operator_type" in line or "tc_util)" in line:
        print("    " + line.strip())
print("  => operator 的**类型**随字符串注入 .cu；operator 的**状态**随 `__grid_constant__` 参数传入")
print("     （sm100_bf16_gemm.cuh:39、sm100_fp8_fp4_gemm_1d1d.cuh:41）。")

# ---------------------------------------------------------------------------
sec("③ resolve_epilogue_class 的互斥（epilogue_class.hpp:111-127 的 Python 复刻）")
def resolve(has_epilogue_class, has_alpha, has_sfd, c_is_none, d_dtype):
    n = int(has_epilogue_class) + int(has_alpha) + int(has_sfd)
    if n > 1:
        return "ASSERT: epilogue_class / alpha / sfd 三者互斥"
    if has_alpha:
        return "AlphaEpilogue (check: d bf16/fp32)"
    if has_sfd:
        if not c_is_none or d_dtype != "e4m3":
            return "ASSERT: FP8 量化要求 not c 且 d==E4M3"
        return "FP8QuantizationEpilogue (单输出 D + 单 SFD)"
    if has_epilogue_class:
        return "用户自定义 EpilogueClass"
    return "IdentityEpilogue (默认)"

cases = [
    ("默认（三者都无）",              (False, False, False, True,  "bf16")),
    ("alpha=0.5",                    (False, True,  False, True,  "bf16")),
    ("sfd（FP8 输出对）",             (False, False, True,  True,  "e4m3")),
    ("sfd + c（非法）",               (False, False, True,  False, "e4m3")),
    ("sfd + d=fp32（非法）",          (False, False, True,  True,  "fp32")),
    ("epilogue_class + alpha（非法）", (True,  True,  False, True,  "bf16")),
    ("epilogue_class + sfd（非法）",   (True,  False, True,  True,  "e4m3")),
]
for tag, args in cases:
    print(f"    {tag:32s} -> {resolve(*args)}")

# ---------------------------------------------------------------------------
sec("④ 单输出契约放不下 SwiGLU 反向（要 2 张量 + 2 SFD）")
print("  当前 EpilogueClass 只表达：1 个 D（tensor_map_cd）+ 可选 1 个 SFD（`output_sfd()` :38）。")
print("  FP8 输出对还要求 `not c`（epilogue_class.hpp:84）——连 C 都被占掉。")
print("  SwiGLU 反向需要：d(gate) 张量 + d(up) 张量 + 各自 SFD = 2 数据 + 2 SF。")
print("  => 必须扩 3 处契约：(a) EpilogueOperatorArgs 加逐元素输入/第二输出指针；")
print("     (b) resolve/check 支持双输出；(c) sm100_store_cd.cuh 的 is_same_v 分支加一条双写。")
print("  SM90 分支直接拒绝自定义 epilogue：gemm.hpp:238 “Custom epilogues require SM100”。")

print("\n[done] host 接线与契约缺口核对完成；数值逐位对拍见 src/moe_backward_activation_epilogue.cu。")
