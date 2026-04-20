# EVQ-Cosh 技术债务审计报告

> **审计日期**: 2026-03-22
> **审计范围**: `/hybrid-rope/` 全仓库 (124 Python 文件, ~3.4GB)
> **项目性质**: NeurIPS 论文投稿级研究代码, 1 人核心开发 + AI 辅助

---

## 1. 项目整体健康快照

**中度债务, 主要集中在测试完全缺失、实验脚本膨胀、和可复现性基础设施不足**。核心库 (`scripts/lib/rope/`, 530行) 质量良好, 但外围实验脚本 (70+ phase scripts, 多个超过 1000 行) 存在大量重复代码和硬编码。对于 NeurIPS 投稿级项目, 文档和实验追踪做得非常优秀 (40+ markdown docs), 但代码层面的工程健康度远低于文档质量。

---

## 2. 技术债务分类清单

### 2.1 代码质量与可读性

#### TD-01: 巨型函数 — eval_longbench.py main() 545 行
- **位置**: `scripts/supporting_eval/eval_longbench.py`, Lines 1828-2373
- **描述**: `main()` 函数长达 545 行, 嵌套 6 层, 内含 3 个嵌套函数定义 (`run_single_generation`, `iter_generation_groups`, `run_group_with_backoff`)。这不是函数, 是一个小型应用程序塞进了一个函数里。
- **影响**: Debug 一个评估失败需要读完整个 545 行上下文; 无法单独测试任何子功能; 改一处容易引入副作用。估计 debug 时间 3-5x 正常水平。
- **严重程度**: **High**
- **修复难度**: **L** (需要拆分为多个函数, 理清状态传递)
- **建议修复方案**: 将 main() 拆为 `setup_experiment()`, `run_evaluations()`, `aggregate_results()`, `save_outputs()` 四个顶层函数。内嵌函数提升为模块级函数, 通过参数传递状态。

#### TD-02: 超大文件 — 5 个文件超过 1000 行
- **位置**:
  - `eval_longbench.py` — 2375 行
  - `phase16_formula_optimality_sweep.py` — 1671 行
  - `prepare_mixed_prior_dataset_v1.py` — 1580 行
  - `run_evq_sweep.py` — 1312 行
  - `llama3_continued_pretrain.py` — 1236 行
- **描述**: 研究代码的典型症状 — 实验脚本从小开始, 不断追加功能, 缺乏重构节点。
- **影响**: 每次修改都需要理解大量上下文; 同一文件多人/多时期修改容易冲突; IDE 响应变慢。
- **严重程度**: **Medium**
- **修复难度**: **XL** (涉及 5 个核心文件, 每个需要独立重构)
- **建议修复方案**: 论文投稿前不建议大动。投稿后按使用频率排序重构, 优先拆 eval_longbench.py (评估框架, 复用率最高)。

#### TD-03: 广泛的裸 except 捕获 — 56 处
- **位置**: 20 个文件, 56 处 `except Exception` 或 `except:`
  - 重灾区: `eval_niah_recall.py` (11处), `eval_longbench.py` (8+处), `test4_attention_isomorphism.py` (6处), `llama3_continued_pretrain.py` (5处)
- **描述**: 大量 `except Exception: pass` 模式, 静默吞掉错误。最典型的案例: `eval_longbench.py` L437-449 双层 try-except, 尝试 delattr 失败后再 setattr, 两层都 `pass` — 完全不知道哪个操作失败了、为什么失败。
- **影响**: 生产中 (论文实验中) 的隐性 bug 来源。一个频率注入错误可能被 except 吞掉, 导致实验结果用了错误的 RoPE 配置而不自知。**这对论文可信度是潜在风险**。
- **严重程度**: **High**
- **修复难度**: **M** (逐文件替换为具体异常类型, 添加 logging)
- **建议修复方案**: 对核心路径 (lib/rope/ 和 train.py) 优先修复 — 将 `except Exception` 替换为 `except (ValueError, TypeError)` 等具体类型, 加 `logging.warning()`。评估脚本的 except 可以暂时保留但加日志。

#### TD-04: 魔法数字散布
- **位置**:
  - `prepare_mixed_prior_dataset_v1.py` L74: `TRAIN_TRUNCATE_HEAD_CAP = 500`
  - `prepare_mixed_prior_dataset_v1.py` L271: `if i % 17 == 0:`, L278: `if i % 24 == 0:` — 完全不知道 17 和 24 是什么
  - `phase8f_multi_seed.py` L26-27: `BASE = 500000.0`, `DIM = 64` (有的有注释, 有的没有)
  - 各种脚本中的 `lr=6e-4`, `BATCH=2`, `TOKENS=50_000_000` 散布在脚本顶部
- **描述**: 实验超参数直接硬编码在脚本中, 没有统一的配置管理。不同脚本的同一参数可能不一致。
- **影响**: 复现实验需要逐文件检查参数; 参数不一致导致实验对比失效; Reviewer 追查参数来源困难。
- **严重程度**: **Medium**
- **修复难度**: **M** (为论文核心实验提取统一 config)
- **建议修复方案**: 为 Claim 1-6 对应的核心实验创建 `configs/` 目录, 每个实验一个 YAML 配置文件。脚本改为读取配置。非核心实验保持现状。

#### TD-05: 硬编码路径 — 不可移植
- **位置**: `scripts/data_prep/prepare_mixed_prior_dataset_v1.py` L39-43
  ```python
  DEFAULT_LONGALPACA = "/root/autodl-tmp/dfrope/datasets/..."
  DEFAULT_TOKENIZER = "/root/autodl-tmp/dfrope/ms_models/..."
  ```
- **描述**: 数据准备脚本硬编码了特定机器 (autodl-tmp) 的绝对路径。在任何其他机器上运行都会失败。
- **影响**: Reviewer 或合作者无法复现数据准备步骤; 换机器训练需要手动改路径。
- **严重程度**: **Medium**
- **修复难度**: **S** (改为 `os.getenv("EVQ_DATA_ROOT", default)`)
- **建议修复方案**: 用环境变量 + argparse 默认值替代。在 REPRODUCE.md 中添加环境变量说明。

### 2.2 架构与设计

#### TD-06: 核心库与实验脚本边界模糊
- **位置**: `scripts/lib/rope/` (530行核心库) vs `scripts/core_text_phases/` (70个脚本)
- **描述**: 核心库设计良好 (4 个文件, 职责清晰: learnable_evq / schedules / inject / attn_hist), 但实验脚本中大量重复了种子设置、设备检测、模型加载等模板代码。核心的 `train.py` (643行) 是唯一的共享训练入口, 但很多 phase 脚本绕过它直接训练。
- **影响**: 修改训练流程需要改多个地方; 不同 phase 可能使用了微妙不同的训练设置而不自知。
- **严重程度**: **Medium**
- **修复难度**: **L** (需要提取共享工具函数)
- **建议修复方案**: 创建 `scripts/lib/utils.py` 提取: `set_seed()`, `get_device()`, `setup_logging()`, `load_model_and_tokenizer()` 等共享逻辑。Phase 脚本改为导入这些函数。

#### TD-07: `models/` 目录为空
- **位置**: `models/` (空目录)
- **描述**: 目录存在但为空, 所有模型代码在 `scripts/lib/rope/`。名称误导。
- **影响**: 新人 (包括 Reviewer 看代码) 会困惑 "模型代码在哪"。
- **严重程度**: **Low**
- **修复难度**: **S** (删除或添加 README 说明)
- **建议修复方案**: 删除空目录, 或者将 `scripts/lib/rope/` 移入 `models/rope/` 使结构更直观。

### 2.3 测试与覆盖率

#### TD-08: 测试覆盖率 = 0%
- **位置**: 整个仓库
- **描述**: **124 个 Python 文件, 0 个测试文件**。唯一找到的 `test_once.py` 是一个 video 实验脚本, 不是单元测试。没有 pytest 配置, 没有 conftest.py, 没有 CI 运行测试。
- **影响**: 这是本审计最严重的发现。核心库 `learnable_evq.py` 中的 EVQ 频率计算是论文的数学核心 — 如果这里有一个 off-by-one 或 edge case bug, 所有实验结果都不可信。目前唯一的 "测试" 是 "实验结果看起来对"。
- **严重程度**: **Critical**
- **修复难度**: **M** (核心库测试) / **L** (全面测试)
- **建议修复方案**:
  1. 立即为 `scripts/lib/rope/` 写测试 (最高优先级):
     - `test_learnable_evq.py`: 测试 EVQ 频率计算的数学正确性 (τ=0 退化为 geometric, 端点行为, 对称性)
     - `test_inject.py`: 测试频率注入是否真正改变了模型的 inv_freq
     - `test_schedules.py`: 测试 schedule 归一化
  2. 为 `train.py` 写冒烟测试 (tiny model, 10 steps, 检查 loss 下降)
  3. 添加 `pytest.ini` 和运行说明

### 2.4 安全与合规

- **无明显问题**。没有发现硬编码凭证、API key、或 .env 文件。`.gitignore` 配置合理。HuggingFace token 通过标准 `huggingface-cli login` 流程管理。

### 2.5 性能与可扩展性

#### TD-09: eval_longbench.py 内存管理
- **位置**: `scripts/supporting_eval/eval_longbench.py` L437-449
- **描述**: 旋转位置编码缓存清理使用 try-except-pass 模式, 如果清理失败, 缓存可能累积导致 OOM。
- **影响**: 长序列评估时潜在 OOM 风险, 且因为 except pass 不会有任何错误提示。
- **严重程度**: **Medium**
- **修复难度**: **S** (改为具体异常 + logging)
- **建议修复方案**: 替换为 `if hasattr(module, attr): setattr(module, attr, None)`, 不需要 try-except。

### 2.6 运维与部署

#### TD-10: 无 CI/CD、无 Docker、无自动化
- **位置**: 仓库根目录
- **描述**: 无 `.github/workflows/`, 无 `Dockerfile`, 无 `docker-compose.yml`。研究项目可以接受, 但对 NeurIPS 投稿的可复现性有影响。
- **影响**: Reviewer 无法一键复现; 换机器需要手动配环境; 依赖冲突无法提前发现。
- **严重程度**: **Low** (研究项目标准) / **Medium** (NeurIPS 投稿标准)
- **修复难度**: **M**
- **建议修复方案**: 添加一个简单的 `Dockerfile` (基于 pytorch/pytorch 镜像 + pip install -r requirements.txt)。不需要完整 CI, 但 Dockerfile 让 Reviewer 可以 `docker build && docker run`。

#### TD-11: 依赖版本未锁定
- **位置**: `requirements.txt`
- **描述**: 所有依赖都是 `>=` 约束 (如 `torch>=2.0.0`, `transformers>=4.40.0`)。没有 lock file。
- **影响**: 6 个月后 pip install 可能安装完全不同版本的库, 导致结果不可复现。transformers 库的 breaking changes 尤其频繁。
- **严重程度**: **High** (对论文可复现性)
- **修复难度**: **S** (运行 `pip freeze > requirements-lock.txt`)
- **建议修复方案**: 保留当前 requirements.txt 作为 "最低要求", 新增 `requirements-lock.txt` 记录精确版本。在 REPRODUCE.md 中推荐使用 lock file。

### 2.7 文档与知识

#### TD-12: 文档丰富但缺少代码级文档
- **位置**: 仓库整体
- **描述**: 项目级文档极其优秀 (40+ markdown 文件, experiment registry, claims map, reproducibility guide, theory validation)。但 Python 代码级文档不足 — 核心库 `learnable_evq.py` 的 type hint 覆盖率约 73%, 部分公共函数缺少 docstring。
- **影响**: 对论文层面无影响; 对代码维护有中等影响。新人看文档知道做了什么, 但看代码不一定知道怎么改。
- **严重程度**: **Low**
- **修复难度**: **S-M**
- **建议修复方案**: 为 `scripts/lib/rope/` 4 个文件的所有公共函数添加 Google-style docstring。其他文件随用随补。

### 2.8 其他

#### TD-13: 实验残留 — 29 个 phase 脚本, 20 个不同 phase 前缀
- **位置**: `scripts/core_text_phases/`
- **描述**: 70 个实验脚本中, 有 29 个 phase-prefixed 脚本, 覆盖 20 个不同的 phase 编号 (phase8d 到 phase21b)。部分是历史实验、已废弃的变体。混杂在一起难以区分哪些是论文核心、哪些是探索性实验。
- **影响**: 新人 (或 6 个月后的你自己) 需要交叉对照 EXPERIMENT_REGISTRY.md 才能知道哪个脚本对应论文的哪个实验。
- **严重程度**: **Low**
- **修复难度**: **M**
- **建议修复方案**: 在 `scripts/core_text_phases/README.md` 中添加一个表格, 标注每个脚本对应的论文 Claim/Table。废弃脚本移入 `scripts/archive/`。

#### TD-14: 无代码质量工具链
- **位置**: 仓库根目录
- **描述**: 无 linter (flake8/ruff), 无 formatter (black/yapf), 无 type checker (mypy), 无 pre-commit hooks。
- **影响**: 代码风格不一致; 潜在 bug 无法被静态分析发现; 新贡献者可能引入风格不同的代码。
- **严重程度**: **Low** (solo research project)
- **修复难度**: **S** (10 分钟配置 ruff)
- **建议修复方案**: 添加 `ruff.toml` 到根目录, 配置基本规则 (E, F, W)。不需要 pre-commit hook, 手动 `ruff check .` 即可。

---

## 3. 优先级排序与 Roadmap 建议

按 **影响 × 紧急度 / 修复成本** 排序:

| 排名 | ID | 债务项 | 严重度 | 修复量 | 理由 |
|:----:|:---:|--------|:------:|:------:|------|
| **1** | TD-08 | 核心库零测试 | Critical | M | **论文数学核心无测试保护**。如果 EVQ 频率计算有边界 bug, 所有实验结果不可信。投稿前必须修。 |
| **2** | TD-11 | 依赖版本未锁定 | High | S | **5 分钟搞定, 可复现性的底线**。`pip freeze` 一行命令。 |
| **3** | TD-03 | 56 处裸 except | High | M | **核心路径** (lib/rope/ + train.py) 优先修, 评估脚本后修。防止实验结果被静默错误污染。 |
| **4** | TD-05 | 硬编码路径 | Medium | S | **15 分钟搞定**。环境变量替代, REPRODUCE.md 更新。 |
| **5** | TD-01 | main() 545 行 | High | L | 投稿后重构。现在不动, 避免引入新 bug。 |
| **6** | TD-04 | 魔法数字 | Medium | M | 为论文核心实验创建 config YAML。 |
| **7** | TD-06 | 共享工具缺失 | Medium | L | 投稿后统一提取 utils。 |
| **8** | TD-10 | 无 Dockerfile | Medium | M | 写一个基础 Dockerfile 助力复现。 |
| **9** | TD-14 | 无 linter | Low | S | 添加 ruff.toml, 10 分钟。 |
| **10** | TD-13 | 实验残留 | Low | M | README 表格标注核心脚本。 |

### 建议执行顺序

```
投稿前 (2-3 人天):
  ① TD-11: pip freeze → requirements-lock.txt              [0.5h]
  ② TD-08: 为 lib/rope/ 写核心单元测试                      [1 天]
  ③ TD-03: 修 lib/rope/ 和 train.py 的 except              [3h]
  ④ TD-05: 硬编码路径 → 环境变量                            [0.5h]
  ⑤ TD-14: 添加 ruff.toml                                  [15min]

投稿后 (5-7 人天):
  ⑥ TD-01 + TD-02: 重构 eval_longbench.py                  [2 天]
  ⑦ TD-04 + TD-06: config 管理 + 共享 utils                [2 天]
  ⑧ TD-10: Dockerfile                                       [0.5 天]
  ⑨ TD-13: 实验脚本整理归档                                 [1 天]
```

**完成 Top 5 项预计: 2-3 人天** (其中 TD-08 核心库测试占大头)。

---

## 4. 预防未来债务的建议

### 4.1 为 lib/rope/ 建立 "黄金标准" 测试
核心库是论文的数学基础。每次改动 learnable_evq.py 后, 自动验证: τ=0 退化为 geometric; EVQ 频率在 [0,1] 内单调; sinh/arcsinh 数值稳定性 (τ=10+ 不溢出); K=1 边界情况; 与 numpy 参考实现交叉验证。

### 4.2 新实验脚本模板化
创建 `scripts/core_text_phases/TEMPLATE.py`, 包含标准化的 argparse、seed 设置、设备检测、结果保存路径。所有新 phase 脚本从模板开始, 减少重复代码。

### 4.3 实验参数集中管理
将论文核心实验的超参数提取到 `configs/claim1.yaml`, `configs/claim2.yaml` 等。脚本读取 config 而非硬编码。Reviewer 只需看 config 目录就能验证所有参数。

### 4.4 每月 `pip freeze` 快照
训练环境的精确依赖版本是可复现性的保障。建议每次重要实验前做一次 freeze, 存入 `artifacts/env_snapshots/YYYY-MM-DD_freeze.txt`。

### 4.5 `ruff check` 作为提交前自检
不需要强制 pre-commit hook (solo 项目成本太高), 但在重要提交前手动跑 `ruff check scripts/lib/` 检查核心库质量。

---

## 特别说明

这份审计基于一个 NeurIPS 投稿级研究项目的标准。相比工业生产代码, 研究代码有其特殊性:

1. **实验脚本的 "一次性" 特征是合理的** — 70 个 phase 脚本中很多跑过一次就不再改了, 对它们做完美重构 ROI 为负。
2. **文档质量远超平均水平** — 40+ 篇结构化的实验报告、theory validation、reviewer defense 文档, 这在研究项目中极为罕见。
3. **核心库 (lib/rope/) 质量良好** — 530 行, 4 个文件, 职责清晰, 命名合理, ~73% type hint 覆盖。唯一缺的是测试。

**最大的风险在 TD-08**: 论文的数学核心 (`learnable_evq.py`) 没有测试保护。如果 `evq_cosh_inv_freq()` 的实现与论文公式有任何偏差, 所有实验结果的可信度都受影响。这是投稿前唯一真正 "必须修" 的项目。

---

准备好开始修复哪一项了吗？我们可以从最高优先级的开始，一步步来。建议先从 TD-11 (5分钟锁定依赖版本) 和 TD-08 (核心库测试) 开始。
