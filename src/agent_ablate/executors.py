"""NCE 工具执行层：HTTP API 接口工具、命令行工具 + 按工具路由。

每个工具在 nce_tools.yaml 里通过 runtime 段声明执行方式：

```yaml
tools:
  # —— NCE API 接口工具 ——
  - name: nce_query_user
    description: 查询 NCE 系统用户信息
    parameters:
      type: object
      properties:
        user_id: {type: string}
      required: [user_id]
      additionalProperties: false
    runtime:
      type: http
      url: https://nce.example.com/api/users/{user_id}   # {参数名} 占位
      method: GET                                        # GET/DELETE: 其余参数拼 query
      headers:                                           # POST/PUT/PATCH: 其余参数作 JSON body
        Authorization: "env:NCE_API_TOKEN"               # "env:变量名" 从环境变量读取
      timeout: 30

  # —— 命令行工具 ——
  - name: nce_device_status
    description: 查询 NCE 设备状态
    parameters:
      type: object
      properties:
        device_id: {type: string}
      required: [device_id]
      additionalProperties: false
    runtime:
      type: cli
      command: nce-cli device status --id {device_id}    # {参数名} 占位，自动 shell 转义
      timeout: 60
      workdir: /opt/nce                                  # 可选
      env: {NCE_PROFILE: prod}                           # 可选，并入子进程环境
```

未声明 runtime 的工具回退到配置的 Python 执行器（executor: demo / 自定义）。

安全说明：命令模板来自你自己的配置文件（可信），模型只能提供参数值，
占位符替换时对参数值做 shell 转义；命令最终经系统 shell 执行，模板本身
不要引入不可信内容。
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import urllib.parse
import urllib.request
from typing import Any, Callable

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_]\w*)\}")


def _quote_arg(value: Any) -> str:
    """参数值转 shell 安全字符串（Windows 下用双引号转义，其余用 shlex.quote）。"""
    s = str(value)
    if os.name == "nt":
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return shlex.quote(s)


def _render(template: str, args: dict, quote: Callable[[Any], str]) -> str:
    """替换 {arg} 占位符；替换后仍残留占位符说明缺少必填参数，抛错记为失败调用。"""

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in args:
            raise ValueError(f"命令/URL 缺少参数 '{key}' 的值")
        return quote(args[key])

    return _PLACEHOLDER.sub(sub, template)


class HttpToolRuntime:
    """NCE API 接口工具：把工具调用翻译成 HTTP 请求。"""

    def __init__(self, url: str, method: str = "POST", headers: dict | None = None,
                 timeout: int = 30):
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.timeout = timeout

    def __call__(self, name: str, args: dict) -> str:
        # URL 模板中的参数从 args 中“消费”掉，其余按 method 拼 query 或 JSON body
        used = set(_PLACEHOLDER.findall(self.url))
        url = _render(self.url, args, lambda v: urllib.parse.quote(str(v), safe=""))
        rest = {k: v for k, v in args.items() if k not in used}

        headers: dict[str, str] = {}
        for k, v in self.headers.items():
            v = str(v)
            headers[k] = os.environ.get(v[4:], "") if v.startswith("env:") else v

        data = None
        if self.method in ("GET", "DELETE", "HEAD"):
            if rest:
                sep = "&" if urllib.parse.urlparse(url).query else "?"
                url = url + sep + urllib.parse.urlencode(rest)
        else:
            data = json.dumps(rest).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        req = urllib.request.Request(url, data=data, method=self.method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {e.code}: {body}") from e


class CliToolRuntime:
    """命令行工具：渲染命令模板后经系统 shell 执行，stdout 作为结果。"""

    def __init__(self, command: str, timeout: int = 60, workdir: str | None = None,
                 env: dict | None = None):
        self.command = command
        self.timeout = timeout
        self.workdir = workdir or None
        self.env = env or {}

    def __call__(self, name: str, args: dict) -> str:
        cmd = _render(self.command, args, _quote_arg)
        env = {**os.environ, **{k: str(v) for k, v in self.env.items()}}
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=self.timeout, cwd=self.workdir, env=env)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"命令超时（{self.timeout}s）: {cmd}") from None
        if proc.returncode != 0:
            raise RuntimeError(f"命令退出码 {proc.returncode}: "
                               f"{proc.stderr.strip()[:500] or '(无 stderr)'}")
        return proc.stdout.strip()


# 注册表 runtime.type -> 运行时类（可扩展，如 type: grpc）
RUNTIME_TYPES: dict[str, type] = {
    "http": HttpToolRuntime,
    "cli": CliToolRuntime,
}


class RuntimeDispatcher:
    """按工具名路由：有 runtime 声明走对应运行时，否则回退 Python 执行器。"""

    def __init__(self, runtimes: dict[str, Callable], fallback: Callable | None = None):
        self.runtimes = runtimes
        self.fallback = fallback

    def __call__(self, name: str, args: dict) -> str:
        rt = self.runtimes.get(name)
        if rt is not None:
            return rt(name, args)
        if self.fallback is not None:
            return self.fallback(name, args)
        raise KeyError(f"工具 '{name}' 未配置执行方式（runtime 或 executor）")


def build_dispatcher(tool_reg, fallback: Callable | None = None) -> RuntimeDispatcher:
    """根据工具注册表的 runtime 声明构造分发器。"""
    runtimes: dict[str, Callable] = {}
    for tool_name, cfg in tool_reg.runtimes().items():
        cfg = dict(cfg)
        rtype = cfg.pop("type", None)
        cls = RUNTIME_TYPES.get(rtype or "")
        if cls is None:
            raise ValueError(f"工具 '{tool_name}' 的 runtime.type 未知: {rtype!r}，"
                             f"支持: {sorted(RUNTIME_TYPES)}")
        runtimes[tool_name] = cls(**cfg)
    return RuntimeDispatcher(runtimes, fallback)
