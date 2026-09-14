"""指标聚合。

所有“率”类指标同时给出两种口径（与指标定义一致）：
- 绝对值（_abs）：各任务该指标次数的平均值（先按任务计数，再对任务取平均）。
- 比例（_ratio_pct）：全部任务的该项总次数 ÷ 对应调用总次数 × 100%。
  分母为 0 时比例记 0，避免除零。
"""
from __future__ import annotations

from typing import Any, Callable

from .evaluator import TaskEvaluation

MetricFn = Callable[[TaskEvaluation], int]


def aggregate(group_id: str, evals: list[TaskEvaluation]) -> dict[str, Any]:
    n = len(evals)
    total: MetricFn = lambda attr: sum(getattr(e, attr) for e in evals)  # noqa: E731
    absm: MetricFn = lambda attr: (total(attr) / n) if n else 0.0  # noqa: E731

    total_skill = total("n_skill_calls")
    total_nce = total("n_nce_calls")
    total_fail = total("n_failures")
    graded = [e for e in evals if e.graded]

    def ratio(num_attr: str, denom: int) -> float:
        return 100.0 * total(num_attr) / denom if denom else 0.0

    m: dict[str, Any] = {
        "group_id": group_id,
        "n_tasks": n,
        "n_graded": len(graded),
        # 端到端准确率
        "accuracy_pct": (100.0 * sum(e.correct for e in graded) / len(graded)) if graded else 0.0,
        # token
        "total_input_tokens": total("input_tokens"),
        "avg_input_tokens": absm("input_tokens"),
        "total_output_tokens": total("output_tokens"),
        "avg_output_tokens": absm("output_tokens"),
        # 交互轮数
        "avg_turns": absm("turns"),
        # 工具调用次数
        "total_skill_calls": total_skill,
        "avg_skill_calls": absm("n_skill_calls"),
        "total_nce_calls": total_nce,
        "avg_nce_calls": absm("n_nce_calls"),
        # Skill 幻觉
        "skill_halluc_abs": absm("skill_halluc"),
        "skill_halluc_ratio_pct": ratio("skill_halluc", total_skill),
        # Skill 无效调用
        "skill_invalid_abs": absm("skill_invalid"),
        "skill_invalid_ratio_pct": ratio("skill_invalid", total_skill),
        # NCE 工具幻觉（合计 = 名 + 参数）
        "nce_halluc_abs": absm("nce_halluc"),
        "nce_halluc_ratio_pct": ratio("nce_halluc", total_nce),
        "nce_name_halluc_abs": absm("nce_name_halluc"),
        "nce_name_halluc_ratio_pct": ratio("nce_name_halluc", total_nce),
        "nce_param_halluc_abs": absm("nce_param_halluc"),
        "nce_param_halluc_ratio_pct": ratio("nce_param_halluc", total_nce),
        # NCE 无效调用（合计 = 冗余 + 参数值错误）
        "nce_invalid_abs": absm("nce_invalid"),
        "nce_invalid_ratio_pct": ratio("nce_invalid", total_nce),
        "nce_redundant_abs": absm("nce_redundant"),
        "nce_redundant_ratio_pct": ratio("nce_redundant", total_nce),
        "nce_value_err_abs": absm("nce_value_err"),
        "nce_value_err_ratio_pct": ratio("nce_value_err", total_nce),
        # 失败与纠错
        "failures_abs": absm("n_failures"),
        "correction_abs": absm("n_corrected"),
        "correction_ratio_pct": (100.0 * total("n_corrected") / total_fail) if total_fail else 0.0,
    }
    return m


# 报告中使用的指标展示顺序与中文名
METRIC_LABELS: list[tuple[str, str]] = [
    ("accuracy_pct", "端到端准确率 (%)"),
    ("avg_turns", "平均交互轮数"),
    ("avg_input_tokens", "平均输入 token"),
    ("avg_output_tokens", "平均输出 token"),
    ("avg_skill_calls", "平均 SKILL 调用频次"),
    ("avg_nce_calls", "平均 NCE 调用频次"),
    ("skill_halluc_abs", "Skill 幻觉次数(均值)"),
    ("skill_halluc_ratio_pct", "Skill 幻觉率 (%)"),
    ("skill_invalid_abs", "Skill 无效调用次数(均值)"),
    ("skill_invalid_ratio_pct", "Skill 无效调用率 (%)"),
    ("nce_halluc_abs", "NCE 工具幻觉次数(均值)"),
    ("nce_halluc_ratio_pct", "NCE 工具幻觉率 (%)"),
    ("nce_name_halluc_abs", "NCE 工具名幻觉次数(均值)"),
    ("nce_name_halluc_ratio_pct", "NCE 工具名幻觉率 (%)"),
    ("nce_param_halluc_abs", "NCE 参数幻觉次数(均值)"),
    ("nce_param_halluc_ratio_pct", "NCE 参数幻觉率 (%)"),
    ("nce_invalid_abs", "NCE 无效调用次数(均值)"),
    ("nce_invalid_ratio_pct", "NCE 无效调用率 (%)"),
    ("nce_redundant_abs", "NCE 冗余调用次数(均值)"),
    ("nce_redundant_ratio_pct", "NCE 冗余调用率 (%)"),
    ("nce_value_err_abs", "NCE 参数值错误次数(均值)"),
    ("nce_value_err_ratio_pct", "NCE 参数值错误率 (%)"),
    ("correction_abs", "成功纠错次数(均值)"),
    ("correction_ratio_pct", "纠错率 (%)"),
]
