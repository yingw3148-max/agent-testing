"""平台适配层接口。

平台职责：接收任务 + 当前实验组的模型/Skill/NCE 配置，真实或模拟地驱动智能体，
产出结构统一的 Trace（调用记录、轮次、token、最终答案）。

指标判定不在平台内做，全部由 evaluator 依据注册表统一完成，
保证跨平台口径一致。

接入新平台两种方式：
1. 继承 AgentPlatform 写成类，配置里用 ``dotted.path:ClassName`` 引用；
2. 直接加进 platforms/__init__.py 的 PLATFORM_REGISTRY。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..models import Task, Trace
from ..registry import SkillRegistry, ToolRegistry

# NCE 工具执行器：由使用方提供 name -> callable(arguments) -> result_text
ToolExecutor = Callable[[str, dict], str]


class AgentPlatform(Protocol):
    def run(self, task: Task, group_id: str) -> Trace:
        ...


@dataclass
class PlatformContext:
    """构建平台时注入的实验组上下文。"""

    group_id: str
    model: dict[str, Any] = field(default_factory=dict)      # name / base_url / api_key_env ...
    skills: SkillRegistry = field(default_factory=SkillRegistry)  # 空注册表 = 该组无 Skill
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    executor: ToolExecutor | None = None                     # NCE 工具执行器
    max_turns: int = 12
    skill_tool_name: str = "Skill"                           # 平台上加载 Skill 的工具名
    extra: dict[str, Any] = field(default_factory=dict)      # 平台自定义参数
