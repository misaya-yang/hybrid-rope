# 默认上下文入口整理（2026-09-13）

目标：常用入口只展示当前固定表全窗口质量主线，让历史计划、过时优先级和旁线按需读取。

## 目录层次

- 根 `index.md`：研究、论文、证据、实验四个入口。
- `docs/research/next_stage_20260912/index.md`：当前目标、确认流水线、面板合同和开发结果。
- `experiments/index.md`：任务质量确认；replay仅作按需方法研究。
- `paper-2027/HANDOFF.md`：当前修订、证据与构建；过去交付与审稿记录另存。
- 各层 `CATALOG_20260913.md`：整理前完整目录。旧主计划保存在 `PLAN_HISTORY_20260913.md`，旧论文交接摘要保存在 `HANDOFF_HISTORY_20260913.md`。这些快照都与原入口同目录，原有相对链接继续有效。

旧LoRA Top1、S1–S4队列、Agent-range、位置能力/算子旁线、早期理论和外部指导不再在默认索引中展开。其科学结果与代码没有删除、迁移或降级；具体研究任务仍可沿完整目录定位和复用。证据owner和注册身份不变。

## 验证与边界

文档检查覆盖107个受管理文档、1444个本地链接、30项资产与57次来源检查，零错误。全部快照也纳入链接检查。未运行模型、改动远端队列、提交或推送。

此改动控制仓库的默认阅读路线，不是模型上下文的技术隔离机制。显式打开历史文件、全仓库搜索、已有会话历史仍可能带入旧材料；全局插件技能列表也不受目录整理控制。AGENTS已明确完整目录和历史方案按具体问题读取。

## 2026-09-14 Astra减负

- 合并仓库AGENTS中的重复执行/检查规则，保留研究归因、授权延续、证据身份与最小验证边界；全局两条AGENTS保持不变。
- 当前研究与流水线入口分别从2850/8765字符缩为736/939字符，固定指向当前方法合同；移除实验入口中的fixed-u当前队列定位。
- 修改前内容完整保存在[研究历史目录](../research/next_stage_20260912/CATALOG_20260914.md)和[流水线历史目录](../../experiments/fixed_rope_three_interfaces_20260913/CATALOG_20260914.md)，只按需读取。没有迁移实验代码或原始证据。
- 本机9个个人技能仅缩短触发描述：audit、optimize、polish、critique、harden、distill、overdrive、motion-landing-page-builder、hatch-pet。UI任务范围更明确，技能正文及调用策略不变；未修改插件缓存。
- 本机ignored的`.codex/config.toml`对本项目停用独立langchain-docs、pencil、playwright MCP，保留全局配置。此覆盖不随Git分发；未停用文献搜索、OpenAI文档、shell或桌面内建浏览器。
- TOML解析通过；CLI合并配置验证被既有全局features格式不兼容阻断（map/boolean）。桌面端重载后的工具清单尚未核实，不宣称当前会话已卸载工具或实测token/延迟下降。
- 文档检查无新增错误：115个受管理文档、1528个本地链接；仍有既有证据快照哈希不一致，未为通过导航检查修改科学登记。7个技能直接通过验证；optimize/harden的旧可选字段被当前validator拒绝，保留这些字段并单独验证更新后的描述，正文未改。

依据：[OpenAI Astra技能与提示词建议](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。此维护记录不加入默认研究阅读链。
