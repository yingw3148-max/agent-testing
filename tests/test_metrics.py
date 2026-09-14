"""指标与评估器单元测试。

覆盖：
- 端到端 demo 任务集在三种 Skill 口径下的分类计数
- 聚合的绝对值/比例口径、除零保护、纠错率
- 序列化往返
"""
from __future__ import annotations

import os

import pytest

from agent_ablate.evaluator import (NCE_PARAM_HALLUC, NCE_REDUNDANT,
                                    NCE_VALUE_ERR, SKILL_HALLUC, SKILL_INVALID,
                                    evaluate_task)
from agent_ablate.metrics import aggregate
from agent_ablate.models import CallRecord, Task, Trace, read_tasks_jsonl
from agent_ablate.platforms.mock import MockPlatform
from agent_ablate.platforms.base import PlatformContext
from agent_ablate.registry import SkillRegistry, ToolRegistry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_INIT = os.path.join(ROOT, "configs", "skills_initial.yaml")
SKILL_OPT = os.path.join(ROOT, "configs", "skills_optimized.yaml")
TOOLS = os.path.join(ROOT, "configs", "nce_tools.yaml")
TASKS = os.path.join(ROOT, "data", "tasks.jsonl")


def run_group(skills_path):
    tasks = read_tasks_jsonl(TASKS)
    skills = SkillRegistry.from_yaml(skills_path) if skills_path else SkillRegistry()
    tools = ToolRegistry.from_yaml(TOOLS)
    platform = MockPlatform(PlatformContext(group_id="g"))
    evals = []
    for t in tasks:
        trace = platform.run(t, "g")
        evals.append(evaluate_task(t, trace, skills, tools))
    return evals


class TestSkillClassification:
    def test_base_group_all_skill_calls_are_halluc(self):
        evals = run_group(None)
        t2 = next(e for e in evals if e.task_id == "t2")
        # t2 调用了 web_search / pdf_reader / pdf_reader，无 Skill 注册表 -> 全部幻觉
        assert t2.skill_halluc == 3
        assert t2.skill_invalid == 0

    def test_init_registry_mixed(self):
        evals = run_group(SKILL_INIT)
        t2 = next(e for e in evals if e.task_id == "t2")
        # web_search 不在初始版 -> 幻觉；pdf_reader 首次有效、第二次重复加载无效
        assert t2.skill_halluc == 1
        assert t2.skill_invalid == 1

    def test_opt_registry_no_halluc(self):
        evals = run_group(SKILL_OPT)
        t2 = next(e for e in evals if e.task_id == "t2")
        assert t2.skill_halluc == 0
        assert t2.skill_invalid == 1  # 仅重复加载


class TestNceClassification:
    def test_value_error_and_redundant_and_halluc(self):
        evals = run_group(None)
        t3 = next(e for e in evals if e.task_id == "t3")
        assert t3.nce_name_halluc == 1   # get_stock
        assert t3.nce_param_halluc == 1  # lang 未定义
        assert t3.nce_value_err == 1     # unit=celsius 枚举越界
        assert t3.nce_redundant == 1     # 相同参数相同结果重复调用

    def test_labels_marked(self):
        evals = run_group(None)
        t3 = next(e for e in evals if e.task_id == "t3")
        cats = {(l.seq, l.category) for l in t3.labels}
        assert (1, NCE_VALUE_ERR) in cats      # unit=celsius 枚举越界
        assert (3, NCE_REDUNDANT) in cats      # 相同参数相同结果重复调用
        assert (5, NCE_PARAM_HALLUC) in cats   # lang 未定义


class TestCorrection:
    def test_failed_then_success_counts_as_corrected(self):
        evals = run_group(None)
        t4 = next(e for e in evals if e.task_id == "t4")
        assert t4.n_failures == 1
        assert t4.n_corrected == 1

    def test_uncorrected_failure(self):
        skills = SkillRegistry()
        tools = ToolRegistry.from_yaml(TOOLS)
        task = Task(task_id="x", query="q", gold="2", match="exact")
        trace = Trace(task_id="x", group_id="g", final_answer="2", turns=2, calls=[
            CallRecord(seq=1, turn=1, kind="nce", name="calculator",
                       arguments={"expression": "1+"}, error="bad"),
            CallRecord(seq=2, turn=2, kind="nce", name="calculator",
                       arguments={"expression": "2+"}, error="bad"),
        ])
        ev = evaluate_task(task, trace, skills, tools)
        assert ev.n_failures == 2
        assert ev.n_corrected == 0


class TestAggregation:
    def test_abs_vs_ratio(self):
        evals = run_group(SKILL_OPT)
        m = aggregate("g", evals)
        # 绝对值 = 各任务次数均值；4 个任务都有 gold
        assert m["n_tasks"] == 4
        assert m["n_graded"] == 4
        # t1..t4 轮数 3,4,5,3 -> 均值 3.75
        assert m["avg_turns"] == pytest.approx(3.75)
        # Skill 调用总数：t1=1, t2=3, t3=0, t4=0 -> 4；幻觉 0 -> 比例 0
        assert m["total_skill_calls"] == 4
        assert m["skill_halluc_ratio_pct"] == 0.0
        # 无效 Skill：t2 重复 1 次 -> 均值 0.25，比例 25%
        assert m["skill_invalid_abs"] == pytest.approx(0.25)
        assert m["skill_invalid_ratio_pct"] == pytest.approx(25.0)
        # NCE 调用总数：t1=2, t2=1, t3=5, t4=2 -> 10
        assert m["total_nce_calls"] == 10
        # 幻觉：t3 名1+参1=2 -> 均值 0.5，比例 20%
        assert m["nce_halluc_abs"] == pytest.approx(0.5)
        assert m["nce_halluc_ratio_pct"] == pytest.approx(20.0)
        # 无效：t3 冗余1+值错1=2 -> 均值 0.5，比例 20%
        assert m["nce_invalid_ratio_pct"] == pytest.approx(20.0)
        # 纠错：失败 4 次（t3: 值错/名幻觉/参幻觉各 1，t4: 1），
        # 仅 t3 值错与 t4 报错在其后有成功调用 -> 2/4 = 50%
        assert m["failures_abs"] == pytest.approx(1.0)
        assert m["correction_abs"] == pytest.approx(0.5)
        assert m["correction_ratio_pct"] == pytest.approx(50.0)
        # 准确率：t1 contains 对、t2 对、t3 对、t4 exact 对 -> 100
        assert m["accuracy_pct"] == pytest.approx(100.0)

    def test_zero_division_guard(self):
        ev = evaluate_task(
            Task(task_id="x", query="q", gold=None),
            Trace(task_id="x", group_id="g", final_answer=None, calls=[]),
            SkillRegistry(), ToolRegistry())
        m = aggregate("g", [ev])
        assert m["skill_halluc_ratio_pct"] == 0.0
        assert m["nce_halluc_ratio_pct"] == 0.0
        assert m["correction_ratio_pct"] == 0.0
        assert m["accuracy_pct"] == 0.0  # 无 gold 不计入分母，比例为 0

    def test_accuracy_partial(self):
        evals = run_group(SKILL_OPT)
        evals[0].correct = False  # t1 判错
        m = aggregate("g", evals)
        assert m["accuracy_pct"] == pytest.approx(75.0)


class TestSerialization:
    def test_eval_roundtrip(self):
        evals = run_group(SKILL_OPT)
        d = evals[1].to_dict()
        from agent_ablate.evaluator import TaskEvaluation
        ev2 = TaskEvaluation.from_dict(d)
        assert ev2.skill_invalid == evals[1].skill_invalid
        assert ev2.labels == evals[1].labels
