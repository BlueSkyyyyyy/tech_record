#!/usr/bin/env python3
"""一个最小 agent loop：模型输出 → 工具调用 → 观察 → 再输出。

对应专题第 2 篇《Agent Loop 解剖》。

用法：
    python3 loop_sim.py
    python3 loop_sim.py --max-steps 3
    python3 loop_sim.py --test

只依赖标准库。这里没有真实模型，`ScriptedModel` 按预先写好的剧本返回
OpenAI Responses 风格的 item：assistant 文本 + 一个或多个 function_call。
重点是 loop 的骨架：

    while True:
        resp = model(messages, tools)
        messages.append(resp)
        if not resp.tool_calls: break
        for call in resp.tool_calls:          # 可并行
            messages.append(run_tool(call))
        if steps >= max_steps: break

真实 harness 会在这里加上：token 预算、压缩、审批、沙箱、流式事件、
取消、重试、工具输出截断。脚本只保留"闭环"本身，方便看清终止条件。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Response:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ScriptedModel:
    """按剧本返回响应；剧本走完后返回结束语。"""

    def __init__(self, script: list[Response]):
        self.script = script
        self.i = 0
        self.calls = 0

    def __call__(self, messages: list[dict], tools: list[dict]) -> Response:
        self.calls += 1
        if self.i >= len(self.script):
            return Response(text="done")
        r = self.script[self.i]
        self.i += 1
        return r


def tool_read_file(args: dict) -> str:
    path = args.get("path", "")
    if path == "missing.py":
        raise FileNotFoundError(path)
    return f"# contents of {path}\nprint('hello')\n"


def tool_run_tests(args: dict) -> str:
    return "1 passed, 0 failed"


def tool_finish(args: dict) -> str:
    return "SUBMITTED: " + args.get("summary", "")


TOOLS = {
    "read_file": tool_read_file,
    "run_tests": tool_run_tests,
    "finish": tool_finish,
}


def execute(call: ToolCall) -> dict:
    """执行一个工具调用，永远返回一条 tool 消息（错误也返回，不抛出）。"""
    fn = TOOLS.get(call.name)
    if fn is None:
        out = f"ERROR: unknown tool {call.name!r}"
    else:
        try:
            out = fn(call.args)
        except Exception as e:  # harness 的职责：把异常变成观察
            out = f"ERROR: {type(e).__name__}: {e}"
    return {"type": "function_call_output", "call_id": call.id, "output": out}


def run(model: ScriptedModel, max_steps: int = 8, parallel: bool = True) -> dict:
    messages: list[dict] = [{"role": "user", "content": "fix the bug"}]
    steps = 0
    tool_calls = 0
    while steps < max_steps:
        resp = model(messages, tools=[{"name": n} for n in TOOLS])
        steps += 1
        messages.append({"role": "assistant", "content": resp.text, "tool_calls": len(resp.tool_calls)})
        if not resp.tool_calls:
            return {"status": "completed", "steps": steps, "tool_calls": tool_calls, "messages": messages}
        if resp.tool_calls[0].name == "finish":
            # finish 是终止工具：执行后仍然把结果写回，然后结束
            messages.append(execute(resp.tool_calls[0]))
            tool_calls += 1
            return {"status": "submitted", "steps": steps, "tool_calls": tool_calls, "messages": messages}
        if parallel and len(resp.tool_calls) > 1:
            with ThreadPoolExecutor(max_workers=len(resp.tool_calls)) as ex:
                outputs = list(ex.map(execute, resp.tool_calls))
        else:
            outputs = [execute(c) for c in resp.tool_calls]
        messages.extend(outputs)
        tool_calls += len(outputs)
    return {"status": "max_steps", "steps": steps, "tool_calls": tool_calls, "messages": messages}


def default_script() -> list[Response]:
    return [
        Response(
            text="先读两个文件，再跑测试。",
            tool_calls=[
                ToolCall("c1", "read_file", {"path": "app.py"}),
                ToolCall("c2", "read_file", {"path": "test_app.py"}),
            ],
        ),
        Response(
            text="读一个不存在的文件，看看错误怎么回来。",
            tool_calls=[ToolCall("c3", "read_file", {"path": "missing.py"})],
        ),
        Response(
            text="改完了，跑测试。",
            tool_calls=[ToolCall("c4", "run_tests", {})],
        ),
        Response(
            text="提交。",
            tool_calls=[ToolCall("c5", "finish", {"summary": "patched app.py"})],
        ),
    ]


def report(result: dict) -> None:
    print(f"status={result['status']}  steps={result['steps']}  tool_calls={result['tool_calls']}")
    for m in result["messages"]:
        if "call_id" in m:
            print(f"  tool[{m['call_id']}] -> {m['output'][:60]}")
        elif m["role"] == "assistant":
            print(f"  model: {m['content']} (calls={m['tool_calls']})")


def test() -> None:
    r = run(ScriptedModel(default_script()), max_steps=8)
    assert r["status"] == "submitted"
    assert r["steps"] == 4
    assert r["tool_calls"] == 5
    outs = [m["output"] for m in r["messages"] if "output" in m]
    assert any("FileNotFoundError" in o for o in outs)
    assert outs[-1].startswith("SUBMITTED:")

    # 不并行也应当得到相同的消息条数
    r2 = run(ScriptedModel(default_script()), max_steps=8, parallel=False)
    assert len(r2["messages"]) == len(r["messages"])

    # 步数上限必须生效，且不会执行到 finish
    r3 = run(ScriptedModel(default_script()), max_steps=2)
    assert r3["status"] == "max_steps" and r3["steps"] == 2

    # 未知工具变成观察，而不是异常
    bad = [Response(tool_calls=[ToolCall("x", "nope", {})])]
    r4 = run(ScriptedModel(bad), max_steps=2)
    assert any("unknown tool" in m.get("output", "") for m in r4["messages"])
    print("loop_sim: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--no-parallel", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        test()
        return
    result = run(ScriptedModel(default_script()), args.max_steps, not args.no_parallel)
    report(result)


if __name__ == "__main__":
    main()
