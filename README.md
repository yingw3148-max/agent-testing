# agent-ablate：智能体消融实验测评框架

面向智能体（Agent）消融实验的测评框架：同一套任务集，在多个「模型 × Skill 版本」
实验组上运行，统一采集轨迹并计算全部指标，产出中文对比报告。

## 实验组设置

默认 6 组（`configs/experiment.yaml`）：

| 组 id | 含义 |
|---|---|
| `base` | 基础模型（型号） |
| `base_skill_init` | 基础模型 + 初始 Skill |
| `base_skill_opt` | 基础模型 + 优化 Skill |
| `post` | 后训练模型（型号） |
| `post_skill_init` | 后训练模型 + 初始 Skill |
| `post_skill_opt` | 后训练模型 + 优化 Skill |

组间差异完全由配置驱动：模型（`model`）、Skill 版本（`skills` 指向的注册表文件）、
平台（`platform`）均可按组覆盖，可任意增删组。

## 架构

```
任务集(JSONL) ──> 平台适配层(mock / openai_compatible / 自定义) ──> Trace(原始轨迹)
注册表(Skill 版本 + NCE 工具 Schema) ──> 评估器 evaluator ──> 逐任务评估(含每个调用的分类标签)
                                                      └────> 聚合 metrics ──> report.md / CSV / JSON
```

- **平台只产出轨迹，不做指标判定**：每个调用的存在性、参数合法性、冗余性等全部由
  评估器依据注册表统一判定，保证跨平台口径一致。
- **可插拔**：平台、NCE 工具执行器、LLM Judge 都是接口 + 注册表方式扩展。

```
src/agent_ablate/
├── models.py            # Task / CallRecord / Trace
├── registry.py          # SkillRegistry、ToolRegistry（Schema 校验、未知参数、取值规则）
├── evaluator.py         # 逐调用分类 + 纠错统计 + 准确率判定
├── metrics.py           # 聚合：所有指标的绝对值与比例口径
├── judge.py             # 可选 LLM Judge（语义类判定）
├── runner.py            # 实验编排、输出物落盘
├── report.py            # 中文 Markdown 报告 + CSV
├── cli.py               # 命令行入口
├── demo_tools.py        # 内置演示 NCE 工具执行器
└── platforms/
    ├── base.py          # AgentPlatform 协议 + PlatformContext
    ├── mock.py          # 脚本化假平台（演示/测试，无需 API）
    └── openai_compatible.py  # 任意 OpenAI 兼容端点（vLLM/Ollama/Moonshot/...）
```

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .[dev]   # Windows
# source .venv/bin/activate && pip install -e .[dev] # Linux/macOS

# 运行 demo（mock 平台，无需 API，直接复现任务里的脚本化轨迹）
.venv/Scripts/agent-ablate run --config configs/experiment.yaml --tasks data/tasks.jsonl --out outputs

# 跑测试
.venv/Scripts/python.exe -m pytest
```

产出（`outputs/`）：

- `report.html` + `report_data.js`：**可视化报告**（页面与数据分离，支持动态更新）。
  ECharts 已内联进页面，离线可开；含准确率/轮数/token/调用频次/幻觉与无效细分/纠错率/
  相对基线降低率等图表，**每个图表右上角工具栏可“保存为图片”下载 PNG**
- **动态更新**：页面每 3 秒自动重新加载 `report_data.js`，`run` / `eval` /
  `dashboard` 重跑后**无需重新生成页面**，浏览器开着就能自动刷新出最新数据
  （也可点页面右上角“立即刷新数据”按钮）。页面文件只在首次生成，之后只更新数据文件
- 本地服务方式（数据热更新最稳，推荐长时间跑实验时配合用）：
  `agent-ablate serve --out outputs --port 8000` 后访问 `http://localhost:8000/report.html`
- `traces/{组}.jsonl`：原始执行轨迹（每次调用的名称、参数、结果、报错、轮次、token）
- `evals/{组}.jsonl`：逐任务评估结果，**含每个调用的分类标签**（幻觉/无效/冗余/值错）
- `metrics.json` / `metrics_summary.csv`：各组聚合指标
- `per_task/{组}.csv`：逐任务指标明细（Excel 可直接打开）
- `report.md`：总览 + 指标明细 + 相对基线变化量 + 相对降低率

只重新生成网页（改了数据或样式后）：

```bash
.venv/Scripts/agent-ablate dashboard --out outputs
```

离线重评（换了 judge 或注册表口径，不想重跑模型时）：

```bash
.venv/Scripts/agent-ablate eval --config configs/experiment.yaml --out outputs
```

## 接入真实模型

把组的 `platform` 改为 `openai_compatible` 并填模型信息（框架自己驱动
ReAct 循环、解析 tool_calls、执行工具、统计每轮 token）：

```yaml
groups:
  - id: post_skill_opt
    platform: openai_compatible
    model:
      name: posttrained-model-X          # 后训练模型名
      base_url: https://api.example.com/v1
      api_key_env: EXAMPLE_API_KEY       # 从环境变量读 key，不落盘
      temperature: 0
    skills: configs/skills_optimized.yaml
```

约定：模型通过调用名为 `skill_tool_name`（默认 `Skill`）的工具加载 Skill，
参数里给出 `skill` / `name` / `skill_name` 字段。

## 接入新平台 / 执行器 / Judge

- **新平台**：实现 `platforms/base.py` 里的 `AgentPlatform` 协议
  （`run(task, group_id) -> Trace`），配置里写 `platform: pkg.module:ClassName`。
- **NCE 工具执行器**：三种方式按工具路由——
  1. **HTTP API 接口工具**：在 `nce_tools.yaml` 的工具下声明
     ```yaml
     runtime:
       type: http
       url: https://nce.example.com/api/users/{user_id}   # {参数名} 占位
       method: GET          # GET/DELETE：其余参数拼 query；POST/PUT/PATCH：其余参数作 JSON body
       headers: {Authorization: "env:NCE_API_TOKEN"}      # "env:" 前缀读环境变量
       timeout: 30
     ```
  2. **命令行工具**：
     ```yaml
     runtime:
       type: cli
       command: nce-cli device status --id {device_id}    # {参数名} 占位，自动 shell 转义
       timeout: 60
       # workdir: /opt/nce        # 可选
       # env: {NCE_PROFILE: prod} # 可选
     ```
     stdout 作为结果，非零退出码/超时记为调用失败（进入纠错率统计）。
  3. **Python 执行器**（未声明 runtime 的工具回退到这里）：任意
     `callable(name, args) -> str`，配置 `executor: demo` 用内置演示，
     `executor: pkg.module:factory` 用你自己的。
- **LLM Judge**（可选，判定“Skill 是否与任务无关”“参数语义张冠李戴”“开放式答案”）：

  ```yaml
  judge:
    model: judge-model
    base_url: https://api.example.com/v1
    api_key_env: JUDGE_API_KEY
  ```
  不配置则完全离线运行，语义类判定自动跳过（Skill 无效只统计重复加载，
  参数值错误只统计规则可判定部分）。

## 任务集格式

`data/tasks.jsonl` 每行一个任务：

```json
{"task_id": "t1", "query": "用户输入", "gold": "参考答案",
 "match": "exact | contains | judge",
 "mock_plan": {"answer": "最终答案", "turns": 3, "input_tokens": 1500, "output_tokens": 300,
               "calls": [{"turn": 1, "kind": "skill", "name": "pdf_reader", "arguments": {}, "result_text": "loaded"}]}}
```

`mock_plan` 仅 mock 平台使用；真实平台忽略该字段。

## 指标口径

所有“率”同时输出两种口径：

- **绝对值（均值）**：先数出每个任务中的次数，再对全部任务取平均；
- **比例**：全部任务的该项总次数 ÷ 对应调用总次数 × 100%（分母为 0 记 0）。

| 指标 | 判定规则 |
|---|---|
| Skill 幻觉 | 调用的 Skill 名不在该组注册表 |
| Skill 无效 | Skill 存在但无贡献：规则判定“同一任务重复加载”；启用 Judge 后追加“与任务无关” |
| NCE 工具名幻觉 | 工具名不在注册表 |
| NCE 参数幻觉 | 工具名存在，但含 Schema 未定义参数 |
| NCE 参数值错误 | 参数名合法但取值错误：必填缺失/类型错误/枚举越界/pattern/min/max（规则），语义张冠李戴（Judge） |
| NCE 冗余调用 | 同一工具相同参数重复调用且返回结果未变化（超出首次部分） |
| 纠错率 | 失败调用（执行报错或参数值错误）其后出现成功调用记为纠错；比例 = 成功纠错 ÷ 失败总次数 |
| 平均交互轮数 | 各任务“推理—行动”循环次数的平均 |
| token | 来自平台 API 的 usage 统计（输入/输出分列） |
| 端到端准确率 | gold 与最终答案比对：`exact` / `contains` / `judge`，无 gold 的任务不计入分母 |

## 配置说明

`configs/experiment.yaml` 顶层字段：

- `nce_tools`：NCE 工具注册表（Schema 用于全部幻觉/参数/取值判定）
- `baseline`：指标对比的基线组 id
- `judge`：可选 LLM Judge
- `defaults`：各组公共默认（平台、max_turns、执行器等）
- `groups`：实验组列表，`id` 必填，其余字段覆盖 defaults

`skills: `（空）表示该组无 Skill，任何 Skill 调用都计为幻觉。
