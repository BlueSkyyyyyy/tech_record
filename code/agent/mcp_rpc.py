#!/usr/bin/env python3
"""MCP 的最小可运行骨架：JSON-RPC 2.0 over stdio。

对应专题第 3 篇《工具层：schema、MCP 与 Code Mode》。

用法：
    python3 mcp_rpc.py                 # 启动 client，走一遍完整握手
    python3 mcp_rpc.py --server        # 作为 stdio server（通常由 client 拉起）
    python3 mcp_rpc.py --test

只依赖标准库。它实现了 MCP 生命周期里最关键的五个方法：

    initialize                → 协商版本与能力
    notifications/initialized → 客户端确认（通知，无 id）
    tools/list                → 返回工具名、描述、inputSchema
    tools/call                → 执行工具，返回 content 数组
    ping                      → 健康检查

真实 MCP 还有 resources/prompts/sampling/roots/elicitation、OAuth、 transports
（stdio / Streamable HTTP）。这里只保留"host 如何发现并调用一个工具"这条主链，
以及两个容易被忽略的细节：请求 id 必须原样回传；工具错误要放在 result.isError
里，而不是 JSON-RPC error（后者表示协议层失败）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

PROTOCOL_VERSION = "2025-06-18"

TOOLS = {
    "add": {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    "fail": {
        "name": "fail",
        "description": "Always returns a tool-level error.",
        "inputSchema": {"type": "object", "properties": {}},
    },
}


def handle(req: dict) -> dict | None:
    method = req.get("method")
    rid = req.get("id")
    if rid is None:  # notification：不回复
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "mini-mcp", "version": "0.1.0"},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": list(TOOLS.values())}}
    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "add":
            result = {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}
        elif name == "fail":
            result = {"content": [{"type": "text", "text": "tool failed on purpose"}], "isError": True}
        else:
            # 工具不存在属于协议层错误：method 找到了，但参数指向未知工具
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32602, "message": f"unknown tool: {name}"},
            }
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            resp = handle(req)
        if resp is not None:
            print(json.dumps(resp, ensure_ascii=False), flush=True)


class Client:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.next_id = 1

    def request(self, method: str, params: dict | None = None) -> dict:
        rid = self.next_id
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        resp = json.loads(line)
        assert resp["id"] == rid, "响应 id 必须与请求 id 一致"
        return resp

    def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        assert self.proc.stdin
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=5)


def demo() -> None:
    c = Client()
    init = c.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}})
    print("initialize:", json.dumps(init["result"], ensure_ascii=False))
    c.notify("notifications/initialized")

    tools = c.request("tools/list")["result"]["tools"]
    print("tools:", [t["name"] for t in tools])

    add = c.request("tools/call", {"name": "add", "arguments": {"a": 2, "b": 3}})
    print("add:", add["result"]["content"][0]["text"])

    fail = c.request("tools/call", {"name": "fail", "arguments": {}})
    print("fail:", fail["result"]["isError"], fail["result"]["content"][0]["text"])

    missing = c.request("tools/call", {"name": "nope", "arguments": {}})
    print("unknown tool:", missing["error"])
    c.close()


def test() -> None:
    c = Client()
    init = c.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}})
    assert init["result"]["protocolVersion"] == PROTOCOL_VERSION
    c.notify("notifications/initialized")
    assert {t["name"] for t in c.request("tools/list")["result"]["tools"]} == {"add", "fail"}
    assert c.request("tools/call", {"name": "add", "arguments": {"a": 2, "b": 3}})["result"]["content"][0]["text"] == "5"
    assert c.request("tools/call", {"name": "fail", "arguments": {}})["result"]["isError"] is True
    assert c.request("tools/call", {"name": "nope", "arguments": {}})["error"]["code"] == -32602
    assert c.request("nope")["error"]["code"] == -32601
    c.close()
    print("mcp_rpc: all tests passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.server:
        serve()
    elif args.test:
        test()
    else:
        demo()


if __name__ == "__main__":
    main()
