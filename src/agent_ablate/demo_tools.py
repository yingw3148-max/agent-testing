"""内置示例 NCE 工具执行器，对应 configs/nce_tools.yaml 里的演示工具。

真实实验时用自己的执行器替换（见 README“接入说明”）：
executor 是任意 callable(name: str, args: dict) -> str。
"""
from __future__ import annotations

import datetime
import json
import re


def _get_weather(args: dict) -> str:
    city = args["city"]
    unit = args.get("unit", "c")
    if unit not in ("c", "f"):
        raise ValueError(f"unit 枚举越界: {unit!r}，只允许 'c'/'f'")
    temp_c = 22
    temp = temp_c if unit == "c" else round(temp_c * 9 / 5 + 32, 1)
    return json.dumps({"city": city, "weather": "晴", "temperature": temp, "unit": unit},
                      ensure_ascii=False)


def _calculator(args: dict) -> str:
    expr = args["expression"]
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
        raise ValueError("表达式含非法字符")
    return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - 字符集已白名单限制


def _search_knowledge(args: dict) -> str:
    q = args["query"].lower()
    kb = {
        "agent": "Agent 是由模型驱动的、可调用工具完成任务的智能体系统。",
        "消融": "消融实验通过逐一增减组件来度量各组件对整体性能的贡献。",
    }
    for k, v in kb.items():
        if k in q:
            return v
    return "知识库中没有匹配内容。"


def _get_current_time(args: dict) -> str:
    tz = args.get("timezone", "UTC")
    if not re.fullmatch(r"[A-Za-z_/]+", tz):
        raise ValueError("timezone 格式错误")
    return datetime.datetime.now(datetime.timezone.utc).isoformat() + f" ({tz})"


DEMO_EXECUTORS = {
    "get_weather": _get_weather,
    "calculator": _calculator,
    "search_knowledge": _search_knowledge,
    "get_current_time": _get_current_time,
}


def demo_executor(name: str, args: dict) -> str:
    fn = DEMO_EXECUTORS.get(name)
    if fn is None:
        raise KeyError(f"未实现的演示工具: {name}")
    return fn(args)
