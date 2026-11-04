#!/usr/bin/env python3
"""权限的局部与全局：为什么"每个动作都合规"仍然会泄密。

对应专题第 11 篇《记忆与安全：从 MemGPT 到 CaMeL》。

用法：
    python3 taint.py
    python3 taint.py --test

只依赖标准库。它实现一个极小的 capability/taint 检查器：

    read_email   → 产出 private 数据
    web_fetch    → 产出 untrusted 数据
    send_email   → 外部 sink
    transfer     → 外部 sink（带金额）

两个检查器：

    per_action：只看单次调用（工具名 + 参数类型），对应大多数"逐动作审批"。
    global    ：追踪数据标签，禁止 private 数据流向外部 sink，
                也禁止 untrusted 数据影响控制流（收件人、金额、工具选择）。

脚本的核心是一个反例：`read_email` 和 `send_email` 单独看都合规，
连起来却是把私人邮件发到攻击者地址。逐动作监控没有"关系"视角，
所以这类 2-safety 违规它看不见。
"""

from __future__ import annotations

from dataclasses import dataclass, field


PRIVATE = "private"
UNTRUSTED = "untrusted"
PUBLIC = "public"


@dataclass
class Value:
    label: set[str] = field(default_factory=lambda: {PUBLIC})
    data: str = ""


@dataclass
class Call:
    tool: str
    args: dict[str, Value]
    out: Value | None = None


def per_action(call: Call) -> tuple[bool, str]:
    """逐动作检查：工具存在、参数齐全。它不关心数据从哪来、要到哪去。"""
    schemas = {
        "read_email": {"id"},
        "web_fetch": {"url"},
        "send_email": {"to", "body"},
        "transfer": {"to", "amount"},
    }
    if call.tool not in schemas:
        return False, f"unknown tool {call.tool}"
    missing = schemas[call.tool] - set(call.args)
    if missing:
        return False, f"missing args {sorted(missing)}"
    return True, "ok"


def global_check(plan: list[Call]) -> tuple[bool, str]:
    """全局检查：private 不得进外部 sink；untrusted 不得影响控制流。"""
    external = {"send_email", "transfer"}
    for call in plan:
        ok, why = per_action(call)
        if not ok:
            return False, why
        labels = set().union(*(v.label for v in call.args.values())) if call.args else {PUBLIC}
        if call.tool in external:
            if PRIVATE in labels:
                return False, f"{call.tool}: private data reaches an external sink ({sorted(labels)})"
            if UNTRUSTED in labels:
                return False, f"{call.tool}: untrusted data controls an external action ({sorted(labels)})"
        if call.out is not None:
            # 输出继承输入的标签：这是 taint 传播，也是 CaMeL 的核心
            call.out.label |= labels
    return True, "ok"


def demo() -> None:
    private_mail = Value({PRIVATE}, "salary: 123456")
    attacker = Value({UNTRUSTED}, "attacker@example.com")
    plan = [
        Call("read_email", {"id": Value({PUBLIC}, "42")}, private_mail),
        Call("send_email", {"to": attacker, "body": private_mail}),
    ]
    print("per-action:")
    for c in plan:
        print(f"  {c.tool:<12}{per_action(c)}")
    print("\nglobal:")
    ok, why = global_check(plan)
    print(f"  allowed={ok}  reason={why}")
    print("\n逐动作检查两次都说 ok，全局检查在第二次调用拒绝——差别只在'数据从哪来'。")

    # 良性计划：公开数据发出去是允许的
    good = [Call("send_email", {"to": Value({PUBLIC}, "a@b.c"), "body": Value({PUBLIC}, "hi")})]
    ok, why = global_check(good)
    print(f"\npublic-only plan: allowed={ok} ({why})")


def test() -> None:
    private_mail = Value({PRIVATE}, "secret")
    attacker = Value({UNTRUSTED}, "evil")
    bad = [
        Call("read_email", {"id": Value({PUBLIC}, "1")}, private_mail),
        Call("send_email", {"to": attacker, "body": private_mail}),
    ]
    assert all(per_action(c)[0] for c in bad)
    ok, why = global_check(bad)
    assert not ok and "private data" in why

    # untrusted 数据控制收件人：即使正文是公开的，也应拒绝
    bad2 = [Call("send_email", {"to": attacker, "body": Value({PUBLIC}, "hi")})]
    ok, why = global_check(bad2)
    assert not ok and "untrusted" in why

    # 公开数据外发允许
    good = [Call("send_email", {"to": Value({PUBLIC}, "a@b.c"), "body": Value({PUBLIC}, "hi")})]
    assert global_check(good)[0]

    # taint 会传播：read_email 的输出被 web_fetch 的结果污染后，仍然不能外发
    v = Value({PRIVATE}, "x")
    v.label |= {UNTRUSTED}
    assert not global_check([Call("transfer", {"to": Value({PUBLIC}, "acct"), "amount": v})])[0]
    print("taint: all tests passed")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    test() if args.test else demo()


if __name__ == "__main__":
    main()
