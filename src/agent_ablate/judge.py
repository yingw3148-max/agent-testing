"""LLM Judge（可选）：判定规则难以覆盖的语义类指标。

用途：
1. Skill 无效调用中“加载与任务无关的 Skill”（Skill 本身存在但无贡献）。
2. NCE 参数值错误中的“语义张冠李戴”（参数名合法但取值与任务意图不符）。
3. 端到端准确率的开放式答案判定（match=judge）。

不配置 judge 时框架完全离线运行，语义类判定自动跳过
（Skill 无效调用只统计“重复加载”规则，参数值错误只统计规则可判定部分）。
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


class Judge:
    """Judge 协议。自定义 Judge 实现同样的一组方法即可。"""

    def skill_relevant(self, query: str, skill_name: str, skill_description: str) -> bool:
        raise NotImplementedError

    def param_semantic_ok(self, query: str, tool_name: str, arguments: dict) -> bool:
        raise NotImplementedError

    def judge_answer(self, query: str, gold: str, answer: str) -> bool:
        raise NotImplementedError


class LLMJudge(Judge):
    """基于任意 OpenAI 兼容端点的 Judge。"""

    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 temperature: float = 0.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    # ---- HTTP ----

    def _ask(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"].get("content") or ""

    @staticmethod
    def _yes(text: str) -> bool:
        return "yes" in text.strip().lower()

    # ---- 判定项 ----

    def skill_relevant(self, query: str, skill_name: str, skill_description: str) -> bool:
        prompt = (
            "你是一个严格的评测员。判断加载以下 Skill 是否对完成用户任务有贡献。\n"
            f"任务：{query}\nSkill 名称：{skill_name}\nSkill 描述：{skill_description}\n"
            "只回答 yes 或 no。"
        )
        return self._yes(self._ask(prompt))

    def param_semantic_ok(self, query: str, tool_name: str, arguments: dict) -> bool:
        prompt = (
            "你是一个严格的评测员。工具调用中每个参数名都合法，但参数取值可能与任务意图不符"
            "（语义张冠李戴，例如给错误的城市查天气）。判断是否语义正确。\n"
            f"任务：{query}\n工具：{tool_name}\n参数：{json.dumps(arguments, ensure_ascii=False)}\n"
            "只回答 yes 或 no。"
        )
        return self._yes(self._ask(prompt))

    def judge_answer(self, query: str, gold: str, answer: str) -> bool:
        prompt = (
            "你是一个严格的评测员。判断智能体的最终答案是否正确完成任务要求。\n"
            f"任务：{query}\n参考答案：{gold}\n智能体答案：{answer}\n"
            "只回答 yes 或 no。"
        )
        return self._yes(self._ask(prompt))


def build_judge(cfg: dict | None) -> Judge | None:
    """根据配置构造 Judge；未配置返回 None。"""
    if not cfg:
        return None
    api_key = cfg.get("api_key") or os.environ.get(cfg.get("api_key_env", ""), None)
    return LLMJudge(model=cfg["model"], base_url=cfg["base_url"], api_key=api_key,
                    temperature=cfg.get("temperature", 0.0))
