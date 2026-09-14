"""实验运行器：按配置驱动各实验组跑任务、评估、产出结果文件。

输出目录结构::

    outputs/
      tasks.jsonl              # 任务集备份（eval 子命令依赖）
      traces/{group}.jsonl     # 原始执行轨迹（每行一个 Trace）
      evals/{group}.jsonl      # 逐任务评估结果（含每个调用的分类标签）
      metrics.json             # 各组聚合指标
      per_task/{group}.csv     # 逐任务指标明细（Excel 可直接打开）
      report.md                # 中文消融对比报告
"""
from __future__ import annotations

import copy
import csv
import importlib
import json
import os
from typing import Any

import yaml

from .evaluator import TaskEvaluation, evaluate_task
from .judge import build_judge
from .metrics import aggregate
from .models import Trace, read_tasks_jsonl, write_jsonl, read_jsonl
from .platforms.base import PlatformContext
from .platforms.mock import MockPlatform
from .platforms.openai_compatible import OpenAICompatiblePlatform
from .registry import SkillRegistry, ToolRegistry

PLATFORM_REGISTRY = {
    "mock": MockPlatform,
    "openai_compatible": OpenAICompatiblePlatform,
}

# 各组元信息（模型名 / Skill 版本），仅用于报告展示
GROUP_META: dict[str, dict[str, str]] = {}


# ---------- 配置与对象构造 ----------

def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_object(dotted: str):
    """按 'pkg.module:Class' 或 'pkg.module.Class' 加载对象。"""
    if ":" in dotted:
        mod, name = dotted.split(":", 1)
    else:
        mod, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_platform(group_cfg: dict, ctx: PlatformContext):
    name = group_cfg.get("platform", "mock")
    cls = PLATFORM_REGISTRY.get(name)
    if cls is None:
        cls = load_object(name)  # 自定义平台：dotted.path:Class
    return cls(ctx)


def build_executor(cfg: dict | None, tool_reg: ToolRegistry):
    """构造执行入口：注册表中声明了 runtime 的工具走 HTTP/CLI 运行时，
    其余回退到 Python 执行器（缺省/ 'demo' -> 内置演示执行器；否则 dotted.path:factory）。"""
    fallback = None
    if cfg is None or cfg == "demo":
        from .demo_tools import demo_executor
        fallback = demo_executor
    elif isinstance(cfg, str):
        fallback = load_object(cfg)()
    elif isinstance(cfg, dict) and "type" in cfg:
        obj = load_object(cfg["type"])
        fallback = obj(**{k: v for k, v in cfg.items() if k != "type"})
    else:
        raise ValueError(f"无法解析 executor 配置: {cfg!r}")
    from .executors import build_dispatcher
    return build_dispatcher(tool_reg, fallback)


# ---------- 运行 ----------

def run_experiment(config_path: str, tasks_path: str, out_dir: str,
                   only_groups: list[str] | None = None) -> dict[str, dict]:
    cfg = load_config(config_path)
    defaults = cfg.get("defaults", {})
    tool_reg = ToolRegistry.from_yaml(cfg["nce_tools"])
    judge = build_judge(cfg.get("judge"))
    tasks = read_tasks_jsonl(tasks_path)

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "tasks.jsonl"), [t.to_dict() for t in tasks])

    results: dict[str, dict] = {}
    for group in cfg["groups"]:
        gid = group["id"]
        if only_groups and gid not in only_groups:
            continue
        merged = deep_merge(defaults, group)
        GROUP_META[gid] = {
            "model": (merged.get("model") or {}).get("name", "-"),
            "skills": (merged.get("skills") or "无"),
        }

        skills = (SkillRegistry.from_yaml(merged["skills"])
                  if merged.get("skills") else SkillRegistry())
        ctx = PlatformContext(
            group_id=gid,
            model=merged.get("model") or {},
            skills=skills,
            tools=tool_reg,
            executor=build_executor(merged.get("executor"), tool_reg),
            max_turns=merged.get("max_turns", 12),
            skill_tool_name=merged.get("skill_tool_name", "Skill"),
            extra={k: v for k, v in merged.items() if k not in
                   ("id", "platform", "model", "skills", "executor",
                    "max_turns", "skill_tool_name")},
        )
        platform = build_platform(merged, ctx)

        trace_dir = os.path.join(out_dir, "traces")
        eval_dir = os.path.join(out_dir, "evals")
        os.makedirs(trace_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        traces: list[Trace] = []
        evals: list[TaskEvaluation] = []
        for task in tasks:
            print(f"[{gid}] 运行任务 {task.task_id} ...")
            trace = platform.run(task, gid)
            traces.append(trace)
            evals.append(evaluate_task(task, trace, skills, tool_reg, judge))

        write_jsonl(os.path.join(trace_dir, f"{gid}.jsonl"),
                    [t.to_dict() for t in traces])
        write_jsonl(os.path.join(eval_dir, f"{gid}.jsonl"),
                    [e.to_dict() for e in evals])
        results[gid] = aggregate(gid, evals)

    finalize_outputs(cfg, results, out_dir)
    return results


def eval_only(config_path: str, out_dir: str) -> dict[str, dict]:
    """离线重评：读取 traces/ 重新评估（换 judge / 换注册表口径时用）。"""
    cfg = load_config(config_path)
    defaults = cfg.get("defaults", {})
    tool_reg = ToolRegistry.from_yaml(cfg["nce_tools"])
    judge = build_judge(cfg.get("judge"))
    tasks = {t.task_id: t for t in read_tasks_jsonl(os.path.join(out_dir, "tasks.jsonl"))}

    results: dict[str, dict] = {}
    trace_dir = os.path.join(out_dir, "traces")
    for fname in sorted(os.listdir(trace_dir)):
        gid = fname[:-6] if fname.endswith(".jsonl") else fname
        group = next((g for g in cfg["groups"] if g["id"] == gid), {})
        merged = deep_merge(defaults, group)
        GROUP_META[gid] = {
            "model": (merged.get("model") or {}).get("name", "-"),
            "skills": (merged.get("skills") or "无"),
        }
        skills = (SkillRegistry.from_yaml(merged["skills"])
                  if merged.get("skills") else SkillRegistry())
        evals = [evaluate_task(tasks[t["task_id"]], Trace.from_dict(t), skills,
                               tool_reg, judge)
                 for t in read_jsonl(os.path.join(trace_dir, fname))]
        os.makedirs(os.path.join(out_dir, "evals"), exist_ok=True)
        write_jsonl(os.path.join(out_dir, "evals", fname),
                    [e.to_dict() for e in evals])
        results[gid] = aggregate(gid, evals)

    finalize_outputs(cfg, results, out_dir)
    return results


def finalize_outputs(cfg: dict, results: dict[str, dict], out_dir: str) -> None:
    baseline = cfg.get("baseline") or (cfg["groups"][0]["id"] if cfg.get("groups") else None)
    ordered = {g["id"]: results[g["id"]] for g in cfg["groups"] if g["id"] in results}

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline, "groups": ordered, "meta": GROUP_META},
                  f, ensure_ascii=False, indent=2)

    write_per_task_csvs(ordered, out_dir)

    from .report import build_report, write_summary_csv
    from .report_html import write_html_report
    write_summary_csv(ordered, os.path.join(out_dir, "metrics_summary.csv"))
    report = build_report(ordered, GROUP_META, baseline)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    html_path = write_html_report(out_dir, GROUP_META)
    print(f"HTML 可视化报告: {html_path}")


def write_per_task_csvs(results: dict[str, dict], out_dir: str) -> None:
    dir_path = os.path.join(out_dir, "per_task")
    os.makedirs(dir_path, exist_ok=True)
    cols = ["task_id", "n_skill_calls", "n_nce_calls", "skill_halluc", "skill_invalid",
            "nce_name_halluc", "nce_param_halluc", "nce_redundant", "nce_value_err",
            "n_failures", "n_corrected", "turns", "input_tokens", "output_tokens",
            "correct", "status"]
    for gid in results:
        eval_path = os.path.join(out_dir, "evals", f"{gid}.jsonl")
        if not os.path.exists(eval_path):
            continue
        with open(os.path.join(dir_path, f"{gid}.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in read_jsonl(eval_path):
                w.writerow(row)
