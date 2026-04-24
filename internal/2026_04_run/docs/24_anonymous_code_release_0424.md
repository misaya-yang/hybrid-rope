# EVQ-Cosh 匿名代码包准备指南 · 2026-04-24

**目的**：配合 NeurIPS 2026 checklist Q5 (`[Yes]`)，在提交时附上 anonymous supplementary code zip。
**约束**：双盲，**必须不包含**任何能识别作者、机构、集群、wandb、github handle 的信息。
**预计时间**：1–1.5 小时（纯打包，不跑实验）。

---

## 0. 包结构目标

```
evq-cosh-supp/
├── README.md                      # 本包的复现指南，必须匿名
├── LICENSE                        # MIT 或 Apache-2.0
├── requirements.txt               # pinned，无 wandb
├── evq/                           # 核心实现
│   ├── __init__.py
│   ├── inverse_freq.py            # phi_k(tau) 反CDF warp + tau*=d/√L rule
│   ├── warp.py                    # 等价的 RoPE base_freqs 替换钩子
│   └── surrogate.py               # C_app, K_app kernel + collision score
├── analyses/                      # 离线数值分析（纯 NumPy/SciPy）
│   ├── tau_formula_validation.py  # 99-run sweep 重绘 fig6
│   ├── lambda_curvature.py        # c_coll vs c_pred 表
│   ├── surrogate_validation.py    # 12-config 碰撞缩减表
│   └── stiffness_sweep.py         # Table 9 Sp family
├── experiments/                   # 训练 recipe（config only，无自定义集群脚本）
│   ├── configs/
│   │   ├── mha_454M_L512.yaml     # EVQ×YaRN primary
│   │   ├── mha_125M_L128.yaml     # DAPE-style PE-dominant
│   │   └── mla_432M_L8K.yaml      # MLA scarce-channel
│   └── run.py                     # 最小训练入口（HF + Accelerate，无内部调度器）
├── eval/
│   ├── passkey.py                 # PK@8K / 12K / 16K
│   ├── niah.py                    # NIAH multi-key/value
│   ├── ruler.py                   # 可选
│   └── ppl.py                     # 全序列 + per-document
└── repro.md                       # 每个 primary table/figure 的复现命令
```

---

## 1. 必须移除/替换的内容

| 项 | 处理 |
|---|---|
| wandb API key / entity / project | 全部替换为 `os.environ["WANDB_MODE"] = "disabled"` |
| 作者邮箱、姓名、ORCID | 删 |
| 集群路径 (`/mnt/projects/xxx`, `/shared/team/yyy`) | 替换为 `./data`、`./checkpoints` |
| 内部 slurm 脚本、submitit 配置 | 删，留一个最小的 `torchrun` 示例 |
| `.git/config` 中的 user/email | 打包前 `git archive` 而不是 `zip -r .` |
| 内部 issue 编号、JIRA、Slack 链接 | 全文 grep 删 |
| wandb run URL、dashboard 截图 | 删 |
| 作者机构的 checkpoints hub 路径 | 删，用 HF Hub public model name 代替 |
| `internal/` 整个目录 | **不进包** |
| `results/` 下的 wandb export、raw logs | 不进包，只进必要的 CSV |

## 2. 打包命令（照搬）

```bash
# 在 repo 根目录
cd /path/to/hybrid-rope

# 清理 __pycache__、.DS_Store
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
find . -name .DS_Store -delete 2>/dev/null || true

# 构建 supp 目录
mkdir -p /tmp/evq-cosh-supp
rsync -av \
  --exclude='.git' \
  --exclude='internal/' \
  --exclude='paper/' \
  --exclude='.claude/' \
  --exclude='.codex/' \
  --exclude='results/' \
  --exclude='*.wandb' \
  --exclude='wandb/' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='checkpoints/' \
  --exclude='.venv/' \
  --exclude='.DS_Store' \
  --exclude='docs/' \
  --exclude='experiments/_raw/' \
  . /tmp/evq-cosh-supp/

# 重新写 README（见下文模板）
cp internal/2026_04_run/docs/24_anonymous_code_release_0424.md /tmp/evq-cosh-supp/INTERNAL_NOTES.md  # 删掉这行
rm -f /tmp/evq-cosh-supp/INTERNAL_NOTES.md

# 做一次 identity 扫描
cd /tmp/evq-cosh-supp
grep -rEn "wandb|@[A-Za-z0-9._-]+\.(com|edu|org|cn)|github\.com/[A-Za-z0-9-]+" . \
  --include="*.py" --include="*.md" --include="*.yaml" --include="*.toml" \
  | grep -v '^Binary' \
  | grep -vE "example\.com|anonymous|ANONYMOUS"
# ↑ 必须全部 review 并清除

# 打包
cd /tmp
zip -r evq-cosh-supp.zip evq-cosh-supp/ -x '*.pyc' '*.DS_Store'

# 检查大小 (NeurIPS supp zip 上限一般 100MB)
ls -lh evq-cosh-supp.zip
```

## 3. README.md 匿名模板（直接贴到 supp 根目录）

```markdown
# EVQ-Cosh: Supplementary Code

Anonymous supplementary code for the NeurIPS 2026 submission on variational frequency
allocation for rotary position embedding.

## Quick start

1. `pip install -r requirements.txt`
2. `python analyses/tau_formula_validation.py` — regenerates the tau-sweep figure from
   pre-computed best-tau points (no GPU).
3. `python analyses/surrogate_validation.py` — regenerates the 12-config collision-
   reduction table (no GPU).
4. `python analyses/lambda_curvature.py` — regenerates Table 7 (c_coll vs c_pred).

## Reproducing primary training experiments

See `repro.md`. Each primary-tier experiment (EVQ×YaRN 454M, PE-dominant DAPE-style,
MLA 432M) has a config file; launch with

    torchrun --nproc_per_node=8 experiments/run.py \
      --config experiments/configs/mha_454M_L512.yaml \
      --seed 42

Upstream datasets are from their respective public releases (FineWeb-Edu, TinyStories,
QuALITY, RULER). Our passkey-mix composition script is in `data/prepare_passkey_mix.py`.

## License

MIT.
```

## 4. 最后一道 identity grep（提交前必做）

```bash
cd /tmp/evq-cosh-supp
# 姓名、邮箱、机构、集群、wandb、slack
grep -rEni "your_name|real_name|@gmail|@outlook|@anthropic|@openai|@meta|@google|\\bslack\\b|jira|linear\\.app|wandb|/mnt/|/shared/|your_org" . --include="*.py" --include="*.md" --include="*.yaml" --include="*.toml"

# 预期输出: 零。若非零，逐个清理。
```

## 5. 提交时

1. OpenReview 提交表里 supplementary material 那一栏上传 `evq-cosh-supp.zip`。
2. 正文 PDF 不变。
3. Checklist Q5 已改为 `[Yes]`，指向这个 supp（**我已经在 `paper/main.tex` 里改好**）。

---

## 时间预算

- 打包 + identity grep: 45 min
- 写 repro.md (列出每个 primary 表/图的命令): 20 min
- 最后 check: 15 min

**总计 ~1.5 h，零 GPU。**
