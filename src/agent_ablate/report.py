"""消融对比报告生成（Markdown + 汇总 CSV）。"""
from __future__ import annotations

import csv
import os
from typing import Any

from .metrics import METRIC_LABELS


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return "\n".join(lines)


def build_report(results: dict[str, dict], meta: dict[str, dict],
                 baseline_id: str | None) -> str:
    gids = list(results.keys())
    base = results.get(baseline_id) if baseline_id else None

    out: list[str] = ["# 智能体消融实验报告", ""]
    out.append(f"- 实验组数：{len(gids)}")
    out.append(f"- 每组任务数：{results[gids[0]]['n_tasks'] if gids else 0}")
    if baseline_id:
        out.append(f"- 基线组：`{baseline_id}`")
    out.append("")

    # 表 1：总览
    out += ["## 一、总览", ""]
    headers = ["实验组", "模型", "Skill 版本", "准确率(%)", "平均轮数",
               "平均输入token", "平均输出token", "平均Skill调用", "平均NCE调用"]
    rows = []
    for gid in gids:
        m = results[gid]
        rows.append([gid, meta.get(gid, {}).get("model", "-"),
                     meta.get(gid, {}).get("skills", "-"),
                     m["accuracy_pct"], m["avg_turns"], m["avg_input_tokens"],
                     m["avg_output_tokens"], m["avg_skill_calls"], m["avg_nce_calls"]])
    out.append(_md_table(headers, rows))
    out.append("")

    # 表 2：全部指标明细
    out += ["## 二、指标明细", ""]
    headers = ["指标"] + gids
    rows = [[label] + [results[gid][key] for gid in gids]
            for key, label in METRIC_LABELS if key in results[gids[0]]]
    out.append(_md_table(headers, rows))
    out.append("")

    # 表 3：相对基线的变化
    if base:
        out += [f"## 三、相对基线 `{baseline_id}` 的变化", ""]
        out.append("降低量 = 基线 − 变体（准确率、纠错率为反向：变体 − 基线）。")
        out.append("")
        headers = ["指标", "基线值"] + [f"{gid} 变化量" for gid in gids if gid != baseline_id]
        rows = []
        for key, label in METRIC_LABELS:
            if key not in base:
                continue
            b = base[key]
            higher_better = key.startswith("accuracy") or key.startswith("correction")
            delta = []
            for gid in gids:
                if gid == baseline_id:
                    continue
                v = results[gid][key]
                d = (v - b) if higher_better else (b - v)
                delta.append(d)
            rows.append([label, b] + delta)
        out.append(_md_table(headers, rows))
        out.append("")

        out += ["## 四、相对降低率（相对基线）", ""]
        out.append("相对降低率 = 降低量 ÷ 基线值 × 100%（准确率、纠错率为相对提升率；基线为 0 时记 -）。")
        out.append("")
        headers = ["指标"] + [gid for gid in gids if gid != baseline_id]
        rows = []
        for key, label in METRIC_LABELS:
            if key not in base or base[key] == 0:
                continue
            b = base[key]
            higher_better = key.startswith("accuracy") or key.startswith("correction")
            rel = []
            for gid in gids:
                if gid == baseline_id:
                    continue
                v = results[gid][key]
                d = (v - b) if higher_better else (b - v)
                rel.append(f"{100.0 * d / b:.1f}%")
            rows.append([label] + rel)
        out.append(_md_table(headers, rows))
        out.append("")
    return "\n".join(out)


def write_summary_csv(results: dict[str, dict], out_path: str) -> None:
    gids = list(results.keys())
    keys = [k for k, _ in METRIC_LABELS if k in results[gids[0]]]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + gids)
        for key in keys:
            w.writerow([key] + [results[g][key] for g in gids])
