"""轨迹评估器：把平台产出的 Trace 逐调用分类并统计。

分类规则（与指标定义一一对应）：

Skill 调用：
- skill_hallucination：Skill 不存在于当前实验组的注册表。
- skill_invalid：Skill 存在但无贡献。规则可判定的部分 = 同一任务内重复加载
  同一 Skill；若启用 LLM Judge，再叠加“与任务无关的 Skill”。

NCE 工具调用（先判幻觉，再判无效，两阶段互斥）：
- nce_name_hallucination：工具名不存在。
- nce_param_hallucination：工具名存在但含 Schema 未定义参数。
- nce_param_value_error：参数名合法但取值错误（规则判定 + 可选 Judge 语义判定）。
- nce_redundant：同一工具以相同参数重复调用且返回结果未变化（超出首次部分）。

纠错（self-recovery）：
- 失败调用 = 执行报错（error 非空）或被标记为参数值错误的调用。
- 一次失败调用被“纠正” = 同一任务内其后存在至少一次成功调用。
- 任务级统计 n_failures / n_corrected，聚合时计算纠错率。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from .judge import Judge
from .models import Task, Trace
from .registry import SkillRegistry, ToolRegistry

# 调用类别常量
OK = "ok"
SKILL_HALLUC = "skill_hallucination"
SKILL_INVALID = "skill_invalid"
NCE_NAME_HALLUC = "nce_name_hallucination"
NCE_PARAM_HALLUC = "nce_param_hallucination"
NCE_VALUE_ERR = "nce_param_value_error"
NCE_REDUNDANT = "nce_redundant"
FAILED = "failed"  # 附带说明，叠加在真实类别之上


@dataclass
class CallLabel:
    seq: int
    category: str
    note: str = ""


@dataclass
class TaskEvaluation:
    """单个任务 × 单个实验组的评估结果。"""

    task_id: str
    group_id: str
    # 调用量
    n_skill_calls: int = 0
    n_nce_calls: int = 0
    # Skill 指标计数
    skill_halluc: int = 0
    skill_invalid: int = 0
    # NCE 指标计数
    nce_name_halluc: int = 0
    nce_param_halluc: int = 0
    nce_redundant: int = 0
    nce_value_err: int = 0
    # 纠错
    n_failures: int = 0
    n_corrected: int = 0
    # 过程量
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "completed"
    correct: bool = False
    graded: bool = False  # 该任务是否有 gold 可判分
    labels: list[CallLabel] = field(default_factory=list)

    @property
    def nce_halluc(self) -> int:
        """NCE 工具幻觉（合计）= 工具名幻觉 + 参数幻觉。"""
        return self.nce_name_halluc + self.nce_param_halluc

    @property
    def nce_invalid(self) -> int:
        """NCE 无效调用（合计）= 冗余调用 + 参数值错误。"""
        return self.nce_redundant + self.nce_value_err

    def to_dict(self) -> dict:
        d = asdict(self)
        d["nce_halluc"] = self.nce_halluc
        d["nce_invalid"] = self.nce_invalid
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskEvaluation":
        d = dict(d)
        labels = [CallLabel(**l) for l in d.pop("labels", [])]
        for k in ("nce_halluc", "nce_invalid"):
            d.pop(k, None)
        names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(labels=labels, **{k: v for k, v in d.items() if k in names})


def check_answer(task: Task, answer: str | None, judge: Judge | None = None) -> bool:
    """端到端准确率判定。无答案一律算错。"""
    if task.gold is None or answer is None:
        return False
    gold_norm = " ".join(str(task.gold).split()).strip().casefold()
    ans_norm = " ".join(str(answer).split()).strip().casefold()
    if task.match == "exact":
        return gold_norm == ans_norm
    if task.match == "contains":
        return gold_norm in ans_norm or ans_norm in gold_norm
    if task.match == "judge":
        if judge is None:
            raise ValueError(f"任务 {task.task_id} 使用 match=judge 但未配置 judge")
        return judge.judge_answer(task.query, str(task.gold), str(answer))
    raise ValueError(f"未知 match 模式: {task.match}")


def _norm_args(args: dict) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def evaluate_task(task: Task, trace: Trace, skill_reg: SkillRegistry,
                  tool_reg: ToolRegistry, judge: Judge | None = None) -> TaskEvaluation:
    ev = TaskEvaluation(task_id=trace.task_id, group_id=trace.group_id)
    ev.turns = trace.turns
    ev.input_tokens = trace.input_tokens
    ev.output_tokens = trace.output_tokens
    ev.status = trace.status
    ev.graded = task.gold is not None
    ev.correct = check_answer(task, trace.final_answer, judge) if ev.graded else False

    loaded_skills: set[str] = set()
    last_result_by_key: dict[str, str | None] = {}  # (name+args) -> 上一次成功调用的结果
    failed_seqs: list[int] = []
    success_seqs: list[int] = []

    for call in trace.calls:
        if call.kind == "skill":
            ev.n_skill_calls += 1
            if not skill_reg.exists(call.name):
                ev.skill_halluc += 1
                note = f"Skill '{call.name}' 不存在"
                ev.labels.append(CallLabel(call.seq, SKILL_HALLUC, note))
                if call.error:
                    ev.labels.append(CallLabel(call.seq, FAILED, call.error))
            else:
                reason = None
                if call.name in loaded_skills:
                    reason = "重复加载同一 Skill"
                elif judge is not None:
                    desc = skill_reg.get(call.name).description  # type: ignore[union-attr]
                    if not judge.skill_relevant(task.query, call.name, desc):
                        reason = "加载与任务无关的 Skill"
                if reason:
                    ev.skill_invalid += 1
                    ev.labels.append(CallLabel(call.seq, SKILL_INVALID, reason))
                else:
                    loaded_skills.add(call.name)
                    ev.labels.append(CallLabel(call.seq, OK))
        else:  # nce
            ev.n_nce_calls += 1
            if not tool_reg.exists(call.name):
                ev.nce_name_halluc += 1
                ev.labels.append(CallLabel(call.seq, NCE_NAME_HALLUC,
                                           f"工具 '{call.name}' 不存在"))
            else:
                unknown = tool_reg.unknown_params(call.name, call.arguments)
                if unknown:
                    ev.nce_param_halluc += 1
                    ev.labels.append(CallLabel(call.seq, NCE_PARAM_HALLUC,
                                               f"Schema 未定义参数: {', '.join(unknown)}"))
                else:
                    value_errs = tool_reg.validate_values(call.name, call.arguments)
                    if not value_errs and judge is not None:
                        if not judge.param_semantic_ok(task.query, call.name, call.arguments):
                            value_errs = ["语义错误：参数取值与任务意图不符（Judge 判定）"]
                    key = (call.name, _norm_args(call.arguments))
                    is_redundant = (call.error is None
                                    and last_result_by_key.get(key) == call.result_text)
                    if value_errs:
                        ev.nce_value_err += 1
                        ev.labels.append(CallLabel(call.seq, NCE_VALUE_ERR, "; ".join(value_errs)))
                    elif is_redundant:
                        ev.nce_redundant += 1
                        ev.labels.append(CallLabel(call.seq, NCE_REDUNDANT,
                                                   "相同工具相同参数重复调用，结果未变化"))
                    else:
                        ev.labels.append(CallLabel(call.seq, OK))
                    if call.error is None:
                        last_result_by_key[key] = call.result_text

        # 纠错统计：失败 = 执行报错或参数值错误
        is_value_err = any(l.seq == call.seq and l.category == NCE_VALUE_ERR
                           for l in ev.labels)
        if call.error is not None or is_value_err:
            ev.n_failures += 1
            failed_seqs.append(call.seq)
            if call.error:
                ev.labels.append(CallLabel(call.seq, FAILED, call.error))
        else:
            success_seqs.append(call.seq)

    # 失败调用若其后存在成功调用，记为一次成功纠错
    for seq in failed_seqs:
        if any(s > seq for s in success_seqs):
            ev.n_corrected += 1
    return ev
