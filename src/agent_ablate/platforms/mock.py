"""Mock 平台：从任务自带的 mock_plan 直接还原 Trace。

用于：
- 无 API 环境下端到端演示与回归测试；
- 离线开发报告/指标链路。

mock_plan 格式（写在任务 JSONL 的 mock_plan 字段里）::

    {
      "answer": "最终答案",
      "turns": 3,
      "input_tokens": 1200,
      "output_tokens": 300,
      "status": "completed",
      "calls": [
        {"turn": 1, "kind": "skill", "name": "pdf_reader",
         "arguments": {}, "result_text": "loaded"},
        {"turn": 2, "kind": "nce", "name": "get_weather",
         "arguments": {"city": "Paris", "unit": "c"}, "result_text": "晴"},
        {"turn": 3, "kind": "nce", "name": "calculator",
         "arguments": {"expression": "1/0"}, "error": "division by zero"}
      ]
    }
"""
from __future__ import annotations

from ..models import CallRecord, Task, Trace
from .base import PlatformContext


class MockPlatform:
    def __init__(self, ctx: PlatformContext):
        self.ctx = ctx

    def run(self, task: Task, group_id: str) -> Trace:
        plan = task.mock_plan or {}
        calls = []
        for i, c in enumerate(plan.get("calls", [])):
            calls.append(CallRecord(seq=i + 1, **{k: v for k, v in c.items()
                                                  if k != "seq"}))
        turns = plan.get("turns") or max((c.turn for c in calls), default=0)
        return Trace(
            task_id=task.task_id,
            group_id=group_id,
            final_answer=plan.get("answer"),
            calls=calls,
            turns=turns,
            status=plan.get("status", "completed"),
            input_tokens=plan.get("input_tokens", 0),
            output_tokens=plan.get("output_tokens", 0),
        )
