"""命令行入口。

用法：
    agent-ablate run   --config configs/experiment.yaml --tasks data/tasks.jsonl --out outputs
    agent-ablate eval  --config configs/experiment.yaml --out outputs   # 离线重评 traces
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-ablate",
                                     description="智能体消融实验测评框架")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="运行实验（驱动各平台执行任务并评估）")
    p_run.add_argument("--config", required=True, help="实验配置 YAML")
    p_run.add_argument("--tasks", required=True, help="任务集 JSONL")
    p_run.add_argument("--out", default="outputs", help="输出目录")
    p_run.add_argument("--groups", default=None,
                       help="只跑指定实验组，逗号分隔，如 base,post_skill_opt")

    p_eval = sub.add_parser("eval", help="离线重评：对已有 traces 重新计算指标")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--out", default="outputs", help="含 traces/ 的输出目录")

    p_dash = sub.add_parser("dashboard", help="由 metrics.json 重新生成 HTML 可视化报告与数据文件")
    p_dash.add_argument("--out", default="outputs", help="含 metrics.json 的输出目录")

    p_serve = sub.add_parser("serve", help="启动本地服务访问可视化报告（数据热更新最稳方式）")
    p_serve.add_argument("--out", default="outputs", help="含 report.html 的输出目录")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    from .runner import eval_only, run_experiment

    if args.cmd == "run":
        only = args.groups.split(",") if args.groups else None
        results = run_experiment(args.config, args.tasks, args.out, only)
    elif args.cmd == "eval":
        results = eval_only(args.config, args.out)
    elif args.cmd == "dashboard":
        from .report_html import write_html_report
        path = write_html_report(args.out)
        print(f"HTML 可视化报告: {path}")
        return 0
    else:
        return _serve(args.out, args.port)

    print("\n===== 各组指标摘要 =====")
    for gid, m in results.items():
        print(f"[{gid}] 准确率 {m['accuracy_pct']:.1f}% | 平均轮数 {m['avg_turns']:.2f} | "
              f"Skill幻觉率 {m['skill_halluc_ratio_pct']:.1f}% | "
              f"NCE幻觉率 {m['nce_halluc_ratio_pct']:.1f}% | "
              f"NCE无效率 {m['nce_invalid_ratio_pct']:.1f}% | "
              f"纠错率 {m['correction_ratio_pct']:.1f}%")
    print(f"\n结果已写入 {args.out}/（report.md / metrics.json / metrics_summary.csv / traces / evals）")
    return 0


def _serve(out_dir: str, port: int) -> int:
    import functools
    import http.server

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=out_dir)
    print(f"请打开 http://localhost:{port}/report.html （Ctrl+C 停止）")
    http.server.ThreadingHTTPServer(("", port), handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
