"""OpenAI 兼容平台：驱动任意 OpenAI 兼容端点上的模型跑 ReAct 循环。

适用：vLLM、Ollama(--openai)、Moonshot/Kimi、DeepSeek 等任何
``POST {base_url}/chat/completions`` 服务。框架自行维护消息循环、
解析 tool_calls、执行 NCE 工具与 Skill 加载，并把每一次调用、每一轮
token 消耗记入 Trace。

模型配置（experiment.yaml 的 model 段）::

    model:
      name: kimi-k2-0905-preview
      base_url: https://api.moonshot.cn/v1
      api_key_env: MOONSHOT_API_KEY     # 从环境变量读 key，不落盘
      temperature: 0

Skill 加载约定：模型通过调用名为 skill_tool_name（默认 "Skill"）的工具
加载 Skill，参数里给出 skill / name / skill_name 字段。
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from ..models import CallRecord, Task, Trace
from .base import PlatformContext

MAX_CONTENT = 4000  # 单条结果注入上下文的长度上限


def chat_completion(base_url: str, payload: dict[str, Any], api_key: str | None = None) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OpenAICompatiblePlatform:
    def __init__(self, ctx: PlatformContext):
        self.ctx = ctx
        self.model = ctx.model
        self.base_url = self.model.get("base_url", "http://localhost:8000/v1")
        self.api_key = self.model.get("api_key") or os.environ.get(
            self.model.get("api_key_env", ""), None)

    # ---------- 主循环 ----------

    def run(self, task: Task, group_id: str) -> Trace:
        ctx = self.ctx
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(task)},
            {"role": "user", "content": task.query},
        ]
        trace = Trace(task_id=task.task_id, group_id=group_id)
        seq = 0

        for turn in range(1, ctx.max_turns + 1):
            payload: dict[str, Any] = {
                "model": self.model.get("name", "default"),
                "temperature": self.model.get("temperature", 0),
                "messages": messages,
            }
            tools = ctx.tools.openai_tools()
            if tools:
                payload["tools"] = tools
            data = chat_completion(self.base_url, payload, self.api_key)
            usage = data.get("usage") or {}
            trace.input_tokens += usage.get("prompt_tokens", 0)
            trace.output_tokens += usage.get("completion_tokens", 0)

            msg = data["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                trace.turns = turn
                trace.final_answer = msg.get("content")
                trace.status = "completed"
                return trace

            messages.append({"role": "assistant", "content": msg.get("content"),
                             "tool_calls": tool_calls})
            for tc in tool_calls:
                seq += 1
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc["function"].get("arguments")}
                call = self._execute(seq, turn, name, args)
                trace.calls.append(call)
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": self._tool_msg(call)})

        trace.turns = ctx.max_turns
        trace.status = "hit_limit"
        trace.final_answer = None
        return trace

    # ---------- 工具执行 ----------

    def _execute(self, seq: int, turn: int, name: str, args: dict) -> CallRecord:
        ctx = self.ctx
        if name == ctx.skill_tool_name:
            return self._load_skill(seq, turn, args)

        call = CallRecord(seq=seq, turn=turn, kind="nce", name=name, arguments=args)
        if not ctx.tools.exists(name):
            call.error = f"工具 '{name}' 不存在"
            return call
        if ctx.executor is None:
            call.error = "未配置 NCE 工具执行器（executor）"
            return call
        try:
            result = ctx.executor(name, args)
            call.result_text = result if isinstance(result, str) else json.dumps(
                result, ensure_ascii=False)
        except Exception as e:  # 执行失败也记入轨迹，供纠错率统计
            call.error = f"{type(e).__name__}: {e}"
        return call

    def _load_skill(self, seq: int, turn: int, args: dict) -> CallRecord:
        ctx = self.ctx
        skill_name = (args.get("skill") or args.get("name")
                      or args.get("skill_name") or "")
        call = CallRecord(seq=seq, turn=turn, kind="skill", name=str(skill_name),
                          arguments=args)
        skill = ctx.skills.get(str(skill_name)) if skill_name else None
        if skill is None:
            call.error = f"Skill '{skill_name}' 不存在"
        else:
            call.result_text = (skill.content or skill.description)[:MAX_CONTENT]
        return call

    # ---------- 提示词 ----------

    def _system_prompt(self, task: Task) -> str:
        ctx = self.ctx
        lines = ["你是一个智能体，按任务需要逐步调用工具完成用户请求。"]
        skills = ctx.skills.describe()
        if skills:
            lines.append("可用 Skill（通过 Skill 工具按名称加载，加载后可按其说明使用）：")
            for s in skills:
                lines.append(f"- {s['name']}: {s['description']}")
        else:
            lines.append("当前没有可用 Skill，不要调用 Skill 工具。")
        tools = ctx.tools.describe()
        if tools:
            lines.append("可用 NCE 工具（严格按 JSON Schema 传参）：")
            for t in tools:
                lines.append(f"- {t['name']}: {t['description']}")
        lines.append("完成所有调用后，用自然语言给出最终答案。")
        return "\n".join(lines)

    @staticmethod
    def _tool_msg(call: CallRecord) -> str:
        if call.error is not None:
            return f"ERROR: {call.error}"
        return (call.result_text or "")[:MAX_CONTENT]
