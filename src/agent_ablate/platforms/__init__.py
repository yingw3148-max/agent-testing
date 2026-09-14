"""智能体平台适配层。

内置：
- mock：脚本化假平台，用于演示与测试（无需 API）。
- openai_compatible：任意 OpenAI 兼容端点（vLLM / Moonshot / 本地服务等）。

自定义平台：实现 ``AgentPlatform`` 协议（run 方法返回 Trace），
并在配置中用 ``平台名`` 或 ``dotted.path:ClassName`` 引用。
"""
