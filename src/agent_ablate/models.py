"""核心数据模型：任务、调用记录、执行轨迹。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


def _filter_kwargs(cls: type, d: dict) -> dict:
    """只保留 dataclass 定义的字段，忽略多余键。"""
    names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in d.items() if k in names}


@dataclass
class Task:
    """一条评测任务。

    match 取值：
    - exact：归一化后完全相等
    - contains：gold 与答案互相包含（适用于短答案）
    - judge：交给 LLM Judge 判定（需在配置中启用 judge）
    """

    task_id: str
    query: str
    gold: str | None = None
    match: str = "exact"
    mock_plan: dict[str, Any] | None = None  # mock 平台使用的脚本化计划
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(**_filter_kwargs(cls, d))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CallRecord:
    """一次工具/Skill 调用记录（平台适配层负责产出）。"""

    seq: int = 0            # 任务内序号，从 1 开始
    turn: int = 1           # 所在交互轮次（第几个“推理—行动”循环）
    kind: str = "nce"       # "skill" | "nce"
    name: str = ""          # skill 名或 NCE 工具名
    arguments: dict = field(default_factory=dict)
    result_text: str | None = None
    error: str | None = None  # 非空表示调用执行失败

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def from_dict(cls, d: dict) -> "CallRecord":
        return cls(**_filter_kwargs(cls, d))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trace:
    """一个任务在一次实验组中的完整执行轨迹。"""

    task_id: str
    group_id: str
    final_answer: str | None = None
    calls: list[CallRecord] = field(default_factory=list)
    turns: int = 0
    status: str = "completed"  # completed | failed | hit_limit
    input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        d = dict(d)
        d["calls"] = [CallRecord.from_dict(c) for c in d.get("calls", [])]
        return cls(**_filter_kwargs(cls, d))

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def read_tasks_jsonl(path: str) -> list[Task]:
    tasks: list[Task] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(Task.from_dict(json.loads(line)))
    return tasks


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
