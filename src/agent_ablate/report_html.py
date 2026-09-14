"""HTML 可视化报告：页面与数据分离，支持动态更新。

产出两个文件：

- ``report.html``：单文件页面（ECharts 已内联，离线可开），通过
  ``<script src="report_data.js">`` 读取数据，并每 3 秒轮询重新加载
  （重新跑 run/eval 后无需重新生成页面，刷新或静待轮询即可看到新数据）；
- ``report_data.js``：``window.ABLATE_DATA = {...}``，由 run/eval/dashboard 生成。

浏览方式（二选一）：
- 直接双击 report.html（file:// 下用 script 标签加载，Chrome/Edge/Firefox 均可用）；
- ``agent-ablate serve --out outputs`` 起本地服务后访问
  http://localhost:8000/report.html（HTTP 方式最稳妥）。

每个图表右上角工具栏可“保存为图片”下载 PNG。
"""
from __future__ import annotations

import json
import os
import time

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

POLL_INTERVAL_MS = 3000

# 相对基线展示“相对降低率”的指标（越低越好）
_REDUCTION_KEYS = [
    ("avg_turns", "平均交互轮数"),
    ("avg_input_tokens", "平均输入token"),
    ("avg_output_tokens", "平均输出token"),
    ("skill_halluc_ratio_pct", "Skill幻觉率"),
    ("skill_invalid_ratio_pct", "Skill无效调用率"),
    ("nce_halluc_ratio_pct", "NCE工具幻觉率"),
    ("nce_invalid_ratio_pct", "NCE无效调用率"),
    ("nce_redundant_ratio_pct", "NCE冗余调用率"),
    ("nce_value_err_ratio_pct", "NCE参数值错误率"),
]

_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>智能体消融实验报告</title>
<script>/*__ECHARTS__*/</script>
<script src="report_data.js"></script>
<style>
  body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 0; background: #f5f6fa; color: #2d3436; }
  header { background: #2d3436; color: #fff; padding: 18px 28px; display: flex; align-items: baseline; gap: 20px; }
  header h1 { margin: 0; font-size: 20px; }
  header .meta { font-size: 13px; color: #b2bec3; }
  header .status { margin-left: auto; font-size: 13px; color: #55efc4; }
  header button { background: #0984e3; color: #fff; border: 0; border-radius: 4px; padding: 5px 14px; cursor: pointer; }
  .grid { display: flex; flex-wrap: wrap; gap: 16px; padding: 20px 28px; }
  .card { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 12px 16px; }
  .card h2 { font-size: 15px; margin: 4px 0 0; color: #636e72; font-weight: 600; }
  .chart { width: 560px; height: 360px; }
  .chart-wide { width: 1140px; height: 400px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { border: 1px solid #dfe6e9; padding: 6px 10px; text-align: center; }
  th { background: #f1f2f6; }
  td:first-child, th:first-child { text-align: left; }
  .links { padding: 0 28px 28px; font-size: 13px; }
  .links a { color: #0984e3; margin-right: 16px; }
  .tip { color: #b2bec3; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>智能体消融实验报告</h1>
  <div class="meta" id="meta"></div>
  <button onclick="refreshNow()">立即刷新数据</button>
  <div class="status" id="status">正在载入数据…</div>
</header>
<div class="grid">
  <div class="card"><h2>端到端准确率 (%)</h2><div id="c_accuracy" class="chart"></div></div>
  <div class="card"><h2>平均交互轮数</h2><div id="c_turns" class="chart"></div></div>
  <div class="card"><h2>平均 token 消耗</h2><div id="c_tokens" class="chart"></div></div>
  <div class="card"><h2>平均工具调用频次</h2><div id="c_calls" class="chart"></div></div>
  <div class="card"><h2>Skill 幻觉率 / 无效调用率 (%)</h2><div id="c_skill" class="chart"></div></div>
  <div class="card"><h2>NCE 工具幻觉率（细分，%）</h2><div id="c_nce_halluc" class="chart"></div></div>
  <div class="card"><h2>NCE 无效调用率（细分，%）</h2><div id="c_nce_invalid" class="chart"></div></div>
  <div class="card"><h2>纠错率 (%)</h2><div id="c_correction" class="chart"></div></div>
  <div class="card" style="width:1140px"><h2>相对基线降低率（%，正值=优于基线）</h2><div id="c_reduction" class="chart-wide"></div></div>
</div>
<div class="grid">
  <div class="card" style="width:1140px"><h2>指标明细表</h2><div id="table"></div></div>
</div>
<div class="links">
  数据下载：<a href="metrics_summary.csv" download>metrics_summary.csv（组×指标汇总）</a>
  <a href="metrics.json" download>metrics.json</a>
  <span class="tip">提示：每个图表右上角工具栏可“保存为图片”下载 PNG；页面每几秒自动检测数据更新。</span>
</div>
<script>
const POLL_MS = /*__POLL_MS__*/;
let CHARTS = [];
let METRICS = {}, META = {}, BASELINE = null, REDUCTION = {groups: [], keys: [], rows: []};

function mk(id, opt) {
  const c = echarts.init(document.getElementById(id), null, {renderer: 'canvas'});
  opt.toolbox = { feature: { saveAsImage: { name: id, pixelRatio: 2 } } };
  opt.tooltip = { trigger: 'axis' };
  c.setOption(opt);
  CHARTS.push(c);
}
const bar = (cats) => (name, key, color) => ({ name, type: 'bar',
  data: cats.map(g => METRICS[g][key]),
  label: { show: true, position: 'top', fontSize: 11 }, itemStyle: { color } });

function renderAll(data) {
  METRICS = data.groups; META = data.meta || {}; BASELINE = data.baseline;
  REDUCTION = data.reduction || {groups: [], keys: [], rows: []};
  const cats = Object.keys(METRICS);
  const B = bar(cats);

  CHARTS.forEach(c => c.dispose());
  CHARTS = [];

  document.getElementById('meta').textContent =
    `实验组 ${cats.length} 个 · 每组任务 ${METRICS[cats[0]].n_tasks} 个 · 基线组：${BASELINE}`;
  document.getElementById('status').textContent =
    '数据更新于 ' + (data.generated_at || '-') + ' · 每 ' + (POLL_MS/1000) + 's 自动检测';

  mk('c_accuracy', { legend: {}, xAxis: { type: 'category', data: cats },
    yAxis: { type: 'value', max: 100 }, series: [B('准确率', 'accuracy_pct', '#0984e3')] });
  mk('c_turns', { legend: {}, xAxis: { type: 'category', data: cats },
    yAxis: { type: 'value', minInterval: 1 }, series: [B('平均轮数', 'avg_turns', '#6c5ce7')] });
  mk('c_tokens', { legend: {}, xAxis: { type: 'category', data: cats }, yAxis: { type: 'value' },
    series: [B('输入token', 'avg_input_tokens', '#00b894'), B('输出token', 'avg_output_tokens', '#e17055')] });
  mk('c_calls', { legend: {}, xAxis: { type: 'category', data: cats }, yAxis: { type: 'value' },
    series: [B('SKILL调用', 'avg_skill_calls', '#0984e3'), B('NCE调用', 'avg_nce_calls', '#e84393')] });
  mk('c_skill', { legend: {}, xAxis: { type: 'category', data: cats }, yAxis: { type: 'value' },
    series: [B('Skill幻觉率', 'skill_halluc_ratio_pct', '#d63031'),
             B('Skill无效调用率', 'skill_invalid_ratio_pct', '#e84393')] });
  mk('c_nce_halluc', { legend: {}, xAxis: { type: 'category', data: cats }, yAxis: { type: 'value' },
    series: [B('工具名幻觉率', 'nce_name_halluc_ratio_pct', '#d63031'),
             B('参数幻觉率', 'nce_param_halluc_ratio_pct', '#e17055'),
             B('幻觉率合计', 'nce_halluc_ratio_pct', '#b2bec3')] });
  mk('c_nce_invalid', { legend: {}, xAxis: { type: 'category', data: cats }, yAxis: { type: 'value' },
    series: [B('冗余调用率', 'nce_redundant_ratio_pct', '#fdcb6e'),
             B('参数值错误率', 'nce_value_err_ratio_pct', '#e17055'),
             B('无效调用率合计', 'nce_invalid_ratio_pct', '#b2bec3')] });
  mk('c_correction', { legend: {}, xAxis: { type: 'category', data: cats },
    yAxis: { type: 'value', max: 100 },
    series: [B('纠错率', 'correction_ratio_pct', '#00b894'),
             B('成功纠错次数(均值)', 'correction_abs', '#55efc4')] });

  if (REDUCTION.rows.length) {
    mk('c_reduction', { legend: {}, xAxis: { type: 'category', data: REDUCTION.groups },
      yAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
      series: REDUCTION.keys.map((k, i) => ({ name: k.label, type: 'bar',
        data: REDUCTION.rows[i] })) });
  } else {
    document.getElementById('c_reduction').innerHTML =
      '<div class="tip" style="padding:40px">基线组无数据或未指定 baseline，跳过。</div>';
  }

  let html = '<table><tr><th>指标</th>' + cats.map(g =>
    `<th>${g}<br><span class="tip">${(META[g]||{}).model||'-'} / ${(META[g]||{}).skills||'-'}</span></th>`).join('') + '</tr>';
  for (const [label, vals] of (data.table_rows || [])) {
    html += '<tr><td>' + label + '</td>' + vals.map(v => '<td>' + v + '</td>').join('') + '</tr>';
  }
  document.getElementById('table').innerHTML = html + '</table>';

  window.addEventListener('resize', () => CHARTS.forEach(c => c.resize()));
}

// ---- 动态加载 report_data.js（带 cache-bust），供轮询与手动刷新 ----
function loadData(cb) {
  const s = document.createElement('script');
  s.src = 'report_data.js?t=' + Date.now();
  s.onload = () => { if (window.ABLATE_DATA) cb(window.ABLATE_DATA); s.remove(); };
  s.onerror = () => { document.getElementById('status').textContent = '数据加载失败，重试中…'; s.remove(); };
  document.head.appendChild(s);
}
function refreshNow() { loadData(renderAll); }
setInterval(refreshNow, POLL_MS);
if (window.ABLATE_DATA) renderAll(window.ABLATE_DATA);
else refreshNow();
</script>
</body>
</html>
"""


def _load_echarts() -> str:
    with open(os.path.join(STATIC_DIR, "echarts.min.js"), encoding="utf-8") as f:
        return f.read()


def _relative_reductions(groups: list[str], metrics: dict, baseline: str | None) -> dict:
    """计算各组相对基线的相对降低率（%），基线为 0 的指标记 null。"""
    if not baseline or baseline not in metrics:
        return {"groups": [], "keys": [], "rows": []}
    others = [g for g in groups if g != baseline]
    keys: list[dict] = []
    rows: list[list[float | None]] = []
    for k, lbl in _REDUCTION_KEYS:
        if k not in metrics[baseline]:
            continue
        b = metrics[baseline][k]
        keys.append({"key": k, "label": lbl})
        rows.append([round(100.0 * (b - metrics[g][k]) / b, 1) if b else None
                     for g in others])
    return {"groups": others, "keys": keys, "rows": rows}


def build_data_payload(groups_data: dict, meta: dict, baseline: str | None) -> dict:
    groups = list(groups_data.keys())
    from .metrics import METRIC_LABELS
    table_rows = [[label, [f"{groups_data[g][key]:.2f}" for g in groups]]
                  for key, label in METRIC_LABELS if key in groups_data[groups[0]]]
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": baseline,
        "groups": groups_data,
        "meta": meta,
        "reduction": _relative_reductions(groups, groups_data, baseline),
        "table_rows": table_rows,
    }


def write_html_report(out_dir: str, meta: dict | None = None) -> str:
    """读取 out_dir/metrics.json，生成/更新 report.html 与 report_data.js。"""
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)
    payload = build_data_payload(data["groups"], data.get("meta") or meta or {},
                                 data.get("baseline"))

    # 数据文件：页面通过 script 标签加载，run/eval 更新它即可刷新页面
    data_path = os.path.join(out_dir, "report_data.js")
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("window.ABLATE_DATA = ")
        json.dump(payload, f, ensure_ascii=False)
        f.write(";\n")

    # 页面：首次生成；已存在则不覆盖（避免用户打开的页面被重置，数据走 report_data.js 热更新）
    html_path = os.path.join(out_dir, "report.html")
    if not os.path.exists(html_path):
        page = _PAGE.replace("/*__ECHARTS__*/", _load_echarts())
        page = page.replace("/*__POLL_MS__*/", str(POLL_INTERVAL_MS))
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(page)
    return html_path
