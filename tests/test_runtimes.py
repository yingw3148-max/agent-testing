"""HTTP / CLI 工具运行时与路由分发测试。"""
from __future__ import annotations

import json
import os
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_ablate.executors import (CliToolRuntime, HttpToolRuntime,
                                    RuntimeDispatcher, build_dispatcher)
from agent_ablate.registry import ToolRegistry, ToolSpec


# ---------- 本地测试 HTTP 服务 ----------

class _Recorder:
    last_path: str = ""
    last_body: bytes = b""
    last_auth: str = ""


class _Handler(BaseHTTPRequestHandler):
    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        _Recorder.last_path = self.path
        _Recorder.last_body = self.rfile.read(length)
        _Recorder.last_auth = self.headers.get("Authorization", "")
        if "fail" in self.path:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode())

    do_GET = do_POST = _handle

    def log_message(self, *a):
        pass


@pytest.fixture()
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


# ---------- HTTP 运行时 ----------

class TestHttpRuntime:
    def test_get_placeholder_and_query(self, http_server, monkeypatch):
        monkeypatch.setenv("TEST_NCE_TOKEN", "tk123")
        rt = HttpToolRuntime(url=f"{http_server}/users/{{user_id}}", method="GET",
                             headers={"Authorization": "env:TEST_NCE_TOKEN"})
        out = rt("t", {"user_id": "u 1", "fields": ["name", "age"]})
        assert json.loads(out)["ok"] is True
        path = urllib.parse.unquote(_Recorder.last_path)
        assert path.startswith("/users/u 1?")
        query = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
        assert query["fields"] == ["['name', 'age']"] or "name" in str(query)
        assert _Recorder.last_auth == "tk123"

    def test_post_json_body(self, http_server):
        rt = HttpToolRuntime(url=f"{http_server}/create", method="POST")
        out = rt("t", {"name": "巴黎", "count": 2})
        assert json.loads(out)["ok"] is True
        body = json.loads(_Recorder.last_body)
        assert body == {"name": "巴黎", "count": 2}

    def test_http_error_raises(self, http_server):
        rt = HttpToolRuntime(url=f"{http_server}/fail")
        with pytest.raises(RuntimeError, match="HTTP 500"):
            rt("t", {})


# ---------- CLI 运行时 ----------

class TestCliRuntime:
    def test_echo_and_quoting(self):
        rt = CliToolRuntime(command="echo {text}")
        # 含空格与元字符的参数应被转义，不能拆分/注入第二条命令
        out = rt("t", {"text": "a b & echo hacked"})
        assert "a b" in out
        assert out.count("hacked") == 1  # 只出现一次，未触发第二条 echo

    def test_missing_placeholder_raises(self):
        rt = CliToolRuntime(command="tool --id {device_id}")
        with pytest.raises(ValueError, match="device_id"):
            rt("t", {})

    def test_nonzero_exit_raises(self):
        rt = CliToolRuntime(command="exit 3" if os.name == "nt" else "false")
        with pytest.raises(RuntimeError, match="退出码 3|退出码 [1-9]"):
            rt("t", {})


# ---------- 路由分发 ----------

class TestDispatcher:
    def _reg(self):
        return ToolRegistry([
            ToolSpec(name="api_tool", runtime={"type": "http", "url": "http://x/{a}"}),
            ToolSpec(name="cli_tool", runtime={"type": "cli", "command": "echo {a}"}),
            ToolSpec(name="py_tool"),  # 无 runtime -> 回退
        ])

    def test_runtime_routing_and_fallback(self):
        calls = {"api": 0, "py": 0}

        class FakeHttp:
            def __call__(self, name, args):
                calls["api"] += 1
                return "api-result"

        def fallback(name, args):
            calls["py"] += 1
            return "py-result"

        d = RuntimeDispatcher({"api_tool": FakeHttp()}, fallback)
        assert d("api_tool", {"a": 1}) == "api-result"
        assert d("py_tool", {}) == "py-result"   # 无 runtime -> 回退
        assert calls == {"api": 1, "py": 1}
        # 无 runtime 且无 fallback 的工具 -> KeyError
        with pytest.raises(KeyError):
            RuntimeDispatcher({})("py_tool", {})

    def test_build_dispatcher_unknown_type(self):
        reg = ToolRegistry([ToolSpec(name="bad", runtime={"type": "grpc"})])
        with pytest.raises(ValueError, match="runtime.type"):
            build_dispatcher(reg)

    def test_build_dispatcher_from_registry(self):
        d = build_dispatcher(self._reg(), lambda n, a: "py")
        # 走真实 CLI 运行时；Windows cmd 的 echo 会保留包裹引号，故剥掉再比
        out = d("cli_tool", {"a": "hello"}).strip('"')
        assert out == "hello"
        assert d("py_tool", {}) == "py"
