"""Skill 注册表与 NCE 工具注册表。

- Skill 注册表：定义某版本（初始/优化）下智能体可用 Skill 集合，
  用于判定 Skill 幻觉与重复加载。
- NCE 工具注册表：定义工具名与 JSON Schema，
  用于判定工具名幻觉、参数幻觉（Schema 未定义参数）与参数值错误
  （类型错误 / 枚举越界 / 格式错误等规则可判定的部分）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Skill:
    name: str
    description: str = ""
    content: str = ""  # 加载后注入给模型的 Skill 正文


class SkillRegistry:
    def __init__(self, skills: list[Skill] | None = None):
        self._skills: dict[str, Skill] = {s.name: s for s in (skills or [])}

    @classmethod
    def from_yaml(cls, path: str) -> "SkillRegistry":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        skills = [Skill(**{k: v for k, v in s.items() if k in ("name", "description", "content")})
                  for s in data.get("skills", [])]
        return cls(skills)

    def exists(self, name: str) -> bool:
        return name in self._skills

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def describe(self) -> list[dict[str, str]]:
        return [{"name": s.name, "description": s.description} for s in self._skills.values()]

    def __len__(self) -> int:
        return len(self._skills)


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    schema: dict[str, Any] = field(default_factory=lambda: {
        "type": "object", "properties": {}, "additionalProperties": False})
    # 执行方式声明，两种取值：
    #   runtime: {type: http, url: ..., method: GET, headers: {...}}   NCE API 接口工具
    #   runtime: {type: cli, command: "...", timeout: 60}              命令行工具
    # 不声明 runtime 的工具回退到 Python 执行器（executor 配置）。
    runtime: dict[str, Any] | None = None


_TYPE_CHECKERS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec] | None = None):
        self._tools: dict[str, ToolSpec] = {t.name: t for t in (tools or [])}

    @classmethod
    def from_yaml(cls, path: str) -> "ToolRegistry":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        tools = [ToolSpec(name=t["name"],
                          description=t.get("description", ""),
                          schema=t.get("parameters") or t.get("schema") or {},
                          runtime=t.get("runtime"))
                 for t in data.get("tools", [])]
        return cls(tools)

    def exists(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def runtimes(self) -> dict[str, dict[str, Any]]:
        """返回声明了 runtime 的工具名 -> runtime 配置。"""
        return {t.name: t.runtime for t in self._tools.values() if t.runtime}

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "schema": t.schema}
                for t in self._tools.values()]

    def openai_tools(self) -> list[dict[str, Any]]:
        """转成 OpenAI function-calling 格式。"""
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.schema}}
                for t in self._tools.values()]

    # ---------- 幻觉/错误判定 ----------

    def unknown_params(self, name: str, args: dict) -> list[str]:
        """返回 Schema 未定义的参数名列表（参数幻觉）。"""
        spec = self._tools.get(name)
        if spec is None:
            return []
        props = spec.schema.get("properties", {}) or {}
        if spec.schema.get("additionalProperties", False):
            return []
        return [k for k in args if k not in props]

    def validate_values(self, name: str, args: dict) -> list[str]:
        """规则可判定的参数值错误：必填缺失、类型错误、枚举越界、pattern/min/max。

        返回错误描述列表，空列表表示通过。语义张冠李戴类错误需 LLM Judge，
        不在这里判定。
        """
        spec = self._tools.get(name)
        if spec is None:
            return []
        schema = spec.schema
        props = schema.get("properties", {}) or {}
        errors: list[str] = []

        for req in schema.get("required", []) or []:
            if req not in args:
                errors.append(f"缺少必填参数 '{req}'")

        for key, value in args.items():
            sub = props.get(key)
            if sub is None:
                continue
            where = f"参数 '{key}'"
            expected = sub.get("type")
            if expected and expected in _TYPE_CHECKERS and not _TYPE_CHECKERS[expected](value):
                errors.append(f"{where} 类型错误：期望 {expected}，实际 {type(value).__name__}")
                continue
            if "enum" in sub and value not in sub["enum"]:
                errors.append(f"{where} 枚举越界：{value!r} 不在 {sub['enum']} 中")
            if "pattern" in sub and isinstance(value, str):
                if not re.search(sub["pattern"], value):
                    errors.append(f"{where} 格式错误：不匹配 pattern {sub['pattern']!r}")
            if "minimum" in sub and isinstance(value, (int, float)) and value < sub["minimum"]:
                errors.append(f"{where} 小于 minimum {sub['minimum']}")
            if "maximum" in sub and isinstance(value, (int, float)) and value > sub["maximum"]:
                errors.append(f"{where} 大于 maximum {sub['maximum']}")
        return errors

    def __len__(self) -> int:
        return len(self._tools)
