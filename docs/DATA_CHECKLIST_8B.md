# 8B LongInst 实验数据集完整清单

> **环境约束**: 服务器无外网，仅支持 `hf-mirror.com` 镜像 或 `modelscope.cn`
>
> **前置环境变量** (每次登录都要执行):
> ```bash
> export PATH="/root/miniconda3/bin:$PATH"
> export HF_ENDPOINT=https://hf-mirror.com
> export PYTHONUNBUFFERED=1
> pip install -U huggingface_hub --break-system-packages 2>/dev/null || true
> ```

---

## 数据总览

| # | 数据 | 用途 | 必需? | 预估大小 |
|---|------|------|-------|----------|
| 1 | Meta-Llama-3-8B-Instruct | 基座模型 | **必需** | ~15 GB |
| 2 | LongAlpaca-12k (→ min64 过滤版) | 长指令训练核心 | **必需** | ~470 MB (过滤后) |
| 3 | WikiText-103-raw-v1 (train.txt) | wiki 短文本混合训练 | **必需** | ~500 MB |
| 4 | LongBench (21 task jsonl) | 评测 (gate + full LB21) | **必需** | ~120 MB |
| 5 | LongQA.jsonl | 补充长指令数据 | **可选** | ~未知 |

---

## 1. Meta-Llama-3-8B-Instruct (基座模型)

**脚本参数**: `--base_model_path`
**默认路径**: `/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct`

这是一个 **gated model**，需要先在 HuggingFace 上申请访问权限，获得 token 后才能下载。

### 方法 A: ModelScope (推荐，无需 token，国内直连)

```bash
pip install modelscope --break-system-packages

# 下载到指定目录
python -c "
from modelscope import snapshot_download
snapshot_download(
    'LLM-Research/Meta-Llama-3-8B-Instruct',
    local_dir='/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct'
)
"
```

ModelScope 页面: https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct

### 方法 B: HF 镜像 (需要 token)

```bash
# 先登录 (需要在 huggingface.co 申请 Llama 3 权限并创建 Read token)
huggingface-cli login --token hf_YOUR_TOKEN_HERE

# 下载
huggingface-cli download \
  meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --resume-download
```

### 验证

```bash
python -c "
from transformers import AutoTokenizer, AutoConfig
path = '/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct'
cfg = AutoConfig.from_pretrained(path)
tok = AutoTokenizer.from_pretrained(path)
assert cfg.num_hidden_layers == 32, f'Expected 32 layers, got {cfg.num_hidden_layers}'
assert cfg.hidden_size == 4096, f'Expected 4096 hidden, got {cfg.hidden_size}'
assert cfg.num_attention_heads == 32
assert cfg.num_key_value_heads == 8, 'GQA 8 expected'
print(f'✅ Model OK: {cfg.model_type}, layers={cfg.num_hidden_layers}, '
      f'hidden={cfg.hidden_size}, heads={cfg.num_attention_heads}, '
      f'kv_heads={cfg.num_key_value_heads}, vocab={tok.vocab_size}')
print(f'   rope_theta={cfg.rope_theta}')
"
```

**预期输出**: `✅ Model OK: llama, layers=32, hidden=4096, heads=32, kv_heads=8, vocab=128256`

---

## 2. LongAlpaca-12k → min64 过滤版 (核心训练数据)

**脚本参数**: `--longalpaca_path`
**默认路径**: `/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl`

这是**最关键的训练数据**。原始数据为 12000 条，过滤 `assistant_tokens >= 64` 后得到 9526 条。

**来源**: `Yukang/LongAlpaca-12k` (HuggingFace)

### 步骤 1: 下载原始 LongAlpaca-12k

```bash
mkdir -p /root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload

# 方法 A: huggingface-cli (推荐)
huggingface-cli download \
  --repo-type dataset \
  Yukang/LongAlpaca-12k \
  --local-dir /root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/raw \
  --resume-download

# 方法 B: Python (如果 cli 不好用)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Yukang/LongAlpaca-12k',
    repo_type='dataset',
    local_dir='/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/raw'
)
"
```

下载后检查是否有 `LongAlpaca-12k.json` 文件 (注意: HF 上可能是 parquet 格式，需要转换):

```bash
ls -la /root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/raw/
```

如果下载的是 parquet 格式，需要先转成 jsonl:

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('Yukang/LongAlpaca-12k', split='train')
print(f'Total rows: {len(ds)}')
print(f'Columns: {ds.column_names}')
print(f'Sample[0] keys: {list(ds[0].keys())}')
ds.to_json('/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.jsonl', lines=True)
print('Saved to LongAlpaca-12k.jsonl')
"
```

### 步骤 2: 过滤生成 min64 版本

```bash
python - <<'PY'
import json

input_path = "/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.jsonl"
output_path = "/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl"

kept = 0
dropped_short = 0
dropped_missing = 0

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        # 字段名可能是 instruction/output 或其他格式
        instruction = obj.get("instruction", obj.get("input", "")).strip()
        output = obj.get("output", obj.get("response", "")).strip()
        if not instruction or not output:
            dropped_missing += 1
            continue
        # 用空格分词粗略估计 token 数 (实际应该用 tokenizer)
        # 但为保持与原审计一致，这里用字符数 / 4 近似
        assistant_tokens_approx = len(output.split())
        if assistant_tokens_approx < 64:
            dropped_short += 1
            continue
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

print(f"Input: {kept + dropped_short + dropped_missing}")
print(f"Kept: {kept}")
print(f"Dropped (too short): {dropped_short}")
print(f"Dropped (missing fields): {dropped_missing}")
PY
```

**注意**: 如果你之前的机器上已有审计过的 `min64` 文件，**强烈建议直接 scp 过来**而非重新生成，因为可能存在 tokenizer-based filtering vs word-based 的差异:

```bash
# 从旧机器 scp（如果可以）
scp old_server:/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl \
    /root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/
```

### 验证

```bash
python - <<'PY'
import json, hashlib

path = "/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl"
h = hashlib.sha256()
n = 0
for line in open(path, "rb"):
    h.update(line)
    n += 1
sha = h.hexdigest()
print(f"行数: {n}")
print(f"SHA256: {sha}")

# 已审计的参考值
EXPECTED_SHA = "a9e86ac088aae843556a7d88f97d8369bf05e668a5e2d09e59af2784ba476587"
EXPECTED_ROWS = 9526
if sha == EXPECTED_SHA:
    print(f"✅ SHA256 完全匹配审计值")
else:
    print(f"⚠️ SHA256 不匹配！预期: {EXPECTED_SHA}")
    print(f"   如果是重新生成的文件，差异可能来自过滤逻辑细微不同，需要人工确认")
if n == EXPECTED_ROWS:
    print(f"✅ 行数匹配: {n}")
else:
    print(f"⚠️ 行数不匹配: {n} vs 预期 {EXPECTED_ROWS}")
PY
```

---

## 3. WikiText-103-raw-v1 (wiki 训练语料)

**脚本参数**: `--wikitext_train_path`
**默认路径**: `/root/autodl-tmp/wikitext_data/train.txt`

脚本读取纯文本 `train.txt`，按双换行拆段落 (>=300 字符)，构建 wiki continuation 样本。
这对应 `Salesforce/wikitext` 的 `wikitext-103-raw-v1` 配置。

### 下载并导出为 train.txt

```bash
mkdir -p /root/autodl-tmp/wikitext_data

python -c "
from datasets import load_dataset

# HF_ENDPOINT 环境变量会让它自动走镜像
ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
print(f'Loaded {len(ds)} rows')

# 导出为纯文本 (每行一条，空行分隔段落)
with open('/root/autodl-tmp/wikitext_data/train.txt', 'w', encoding='utf-8') as f:
    for row in ds:
        text = row.get('text', '')
        f.write(text + '\n')

import os
size = os.path.getsize('/root/autodl-tmp/wikitext_data/train.txt')
print(f'✅ Saved train.txt, size={size / 1e6:.1f} MB')
"
```

### 验证

```bash
python -c "
path = '/root/autodl-tmp/wikitext_data/train.txt'
with open(path, 'r') as f:
    lines = f.readlines()
    total_chars = sum(len(l) for l in lines)
    non_empty = sum(1 for l in lines if l.strip())
print(f'总行数: {len(lines)}')
print(f'非空行: {non_empty}')
print(f'总字符: {total_chars:,}')
# wikitext-103-raw-v1 train 约有 1.8M 行，~500MB
assert len(lines) > 1_000_000, f'行数太少: {len(lines)}'
assert total_chars > 100_000_000, f'字符太少: {total_chars}'
print('✅ WikiText train.txt OK')
"
```

---

## 4. LongBench (评测数据, 21 tasks)

**脚本参数**: `--longbench_local_data_dir`
**默认路径**: `/root/autodl-tmp/dfrope/ms_datasets/LongBench/data`

需要 21 个 jsonl 文件 (对应 LB21_TASKS)，放在同一个目录下:

```
narrativeqa.jsonl, qasper.jsonl, multifieldqa_en.jsonl, multifieldqa_zh.jsonl,
hotpotqa.jsonl, 2wikimqa.jsonl, musique.jsonl, dureader.jsonl,
gov_report.jsonl, qmsum.jsonl, multi_news.jsonl, vcsum.jsonl,
trec.jsonl, triviaqa.jsonl, samsum.jsonl, lsht.jsonl,
passage_count.jsonl, passage_retrieval_en.jsonl, passage_retrieval_zh.jsonl,
lcc.jsonl, repobench-p.jsonl
```

### 方法 A: 使用 datasets 库逐 task 导出 (推荐)

```bash
mkdir -p /root/autodl-tmp/dfrope/ms_datasets/LongBench/data

python - <<'PYEOF'
from datasets import load_dataset
import json, os

tasks = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader",
    "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
    "lcc", "repobench-p",
]

out_dir = "/root/autodl-tmp/dfrope/ms_datasets/LongBench/data"
os.makedirs(out_dir, exist_ok=True)

for task in tasks:
    out_path = os.path.join(out_dir, f"{task}.jsonl")
    if os.path.exists(out_path):
        print(f"  SKIP {task} (已存在)")
        continue
    print(f"  下载 {task}...", end=" ", flush=True)
    try:
        # 尝试不带 _e 后缀
        ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    except Exception:
        try:
            # 有些 config 名带 _e 后缀
            ds = load_dataset("THUDM/LongBench", f"{task}_e", split="test", trust_remote_code=True)
        except Exception as e:
            print(f"❌ FAILED: {e}")
            continue
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    print(f"✅ {len(ds)} samples")

print("\n=== 完整性检查 ===")
ok = 0
for task in tasks:
    p = os.path.join(out_dir, f"{task}.jsonl")
    if os.path.exists(p):
        n = sum(1 for _ in open(p))
        print(f"  ✅ {task}: {n} samples")
        ok += 1
    else:
        print(f"  ❌ {task}: MISSING")
print(f"\n{ok}/{len(tasks)} tasks ready")
PYEOF
```

### 方法 B: huggingface-cli 整体下载

```bash
huggingface-cli download \
  --repo-type dataset \
  THUDM/LongBench \
  --local-dir /root/autodl-tmp/dfrope/ms_datasets/LongBench_raw \
  --resume-download
```

注意: HuggingFace datasets 仓库可能不是直接提供 jsonl 文件，而是 parquet 或带 loading script，
所以**方法 A (用 datasets 库加载再导出) 更可靠**。

### 验证

```bash
python -c "
import os
tasks = [
    'narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh',
    'hotpotqa', '2wikimqa', 'musique', 'dureader',
    'gov_report', 'qmsum', 'multi_news', 'vcsum',
    'trec', 'triviaqa', 'samsum', 'lsht',
    'passage_count', 'passage_retrieval_en', 'passage_retrieval_zh',
    'lcc', 'repobench-p',
]
d = '/root/autodl-tmp/dfrope/ms_datasets/LongBench/data'
missing = []
for t in tasks:
    p = os.path.join(d, f'{t}.jsonl')
    if not os.path.exists(p):
        missing.append(t)
if missing:
    print(f'❌ 缺失 {len(missing)} tasks: {missing}')
else:
    print(f'✅ 全部 {len(tasks)} tasks 就位')
"
```

---

## 5. LongQA.jsonl (可选，补充长指令数据)

**脚本参数**: `--longqa_path`
**默认路径**: `/root/autodl-tmp/dfrope/datasets/LongQA.jsonl`

**状态: ⚠️ 可选，缺失不影响训练**

代码逻辑 (line 478-480):
```python
for p in long_paths:
    if not p.exists():
        continue  # 文件不存在直接跳过
```

只要 `LongAlpaca-12k.min64.jsonl` 存在，训练就能正常运行。`LongQA` 仅作为额外的长指令数据补充。

**⚠️ 警告**: `YeungNLP/LongQLoRA-Dataset` 中**没有**名为 `LongQA.jsonl` 的文件。
该 repo 包含的是:
- `LongQLoRA-SFT-Data-39k.jsonl` (506 MB) — 混合数据，包含 book summarization + NQ + LongQA子集 + Evol-Instruct
- `LongQLoRA-Pretrain-Data-54k.jsonl` (2.23 GB)

**你的 `LongQA.jsonl` 可能是之前手动从某个来源提取/构建的**。建议:

1. 先跳过这个文件，传空字符串: `--longqa_path ""`
2. 如果确实需要，从旧服务器 scp 过来
3. **不要让 AI 自作主张下载替代品** — 格式/内容不一致会污染实验

---

## 6. 参考 JSON 文件 (已在 git 仓库内)

以下文件在 `artifacts/` 目录中，随 git clone 自动获得，**无需额外下载**:

| 参数 | 仓库内路径 |
|------|-----------|
| `--qwen_seed42_json` | `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json` |
| `--qwen_seed1337_json` | `artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json` |
| `--morning_reference_json` | 同 qwen_seed42_json (默认值) |
| `--known_good_run_config` | `artifacts/new_attnbias_v1/train/llama3_jointopt_v5_sdpa_higher_util_s42/run_config.json` |

验证:
```bash
cd <repo_root>
for f in \
  artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json \
  artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json \
  artifacts/new_attnbias_v1/train/llama3_jointopt_v5_sdpa_higher_util_s42/run_config.json; do
  if [ -f "$f" ]; then echo "✅ $f"; else echo "❌ MISSING: $f"; fi
done
```

---

## 7. 软件依赖

```bash
# flash-attn (Blackwell / RTX Pro 6000 必须)
pip install flash-attn --no-build-isolation --break-system-packages

# 确认版本
python -c "
import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')
import flash_attn; print(f'flash_attn={flash_attn.__version__}')
import transformers; print(f'transformers={transformers.__version__}')
import peft; print(f'peft={peft.__version__}')
import datasets; print(f'datasets={datasets.__version__}')
"
```

---

## 最终一键验证脚本

全部数据就位后，运行此脚本做最终检查:

```bash
python - <<'VERIFY'
import os, json

checks = []

# 1. Model
model_path = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
cfg_file = os.path.join(model_path, "config.json")
if os.path.exists(cfg_file):
    cfg = json.load(open(cfg_file))
    ok = cfg.get("num_hidden_layers") == 32 and cfg.get("hidden_size") == 4096
    checks.append(("Model (Llama-3-8B-Instruct)", ok, model_path))
else:
    checks.append(("Model (Llama-3-8B-Instruct)", False, f"MISSING: {cfg_file}"))

# 2. LongAlpaca min64
la_path = "/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl"
if os.path.exists(la_path):
    n = sum(1 for _ in open(la_path))
    checks.append(("LongAlpaca min64", n > 9000, f"{n} rows"))
else:
    checks.append(("LongAlpaca min64", False, "MISSING"))

# 3. WikiText
wt_path = "/root/autodl-tmp/wikitext_data/train.txt"
if os.path.exists(wt_path):
    size = os.path.getsize(wt_path)
    checks.append(("WikiText train.txt", size > 100_000_000, f"{size/1e6:.0f} MB"))
else:
    checks.append(("WikiText train.txt", False, "MISSING"))

# 4. LongBench
lb_dir = "/root/autodl-tmp/dfrope/ms_datasets/LongBench/data"
tasks = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader",
    "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
    "lcc", "repobench-p",
]
present = sum(1 for t in tasks if os.path.exists(os.path.join(lb_dir, f"{t}.jsonl")))
checks.append(("LongBench (21 tasks)", present == 21, f"{present}/21 tasks"))

# 5. LongQA (optional)
lq_path = "/root/autodl-tmp/dfrope/datasets/LongQA.jsonl"
if os.path.exists(lq_path):
    n = sum(1 for _ in open(lq_path))
    checks.append(("LongQA (可选)", True, f"{n} rows"))
else:
    checks.append(("LongQA (可选)", True, "跳过 (不影响实验)"))

print("=" * 60)
print("  8B LongInst 数据就位检查")
print("=" * 60)
all_ok = True
for name, ok, detail in checks:
    status = "✅" if ok else "❌"
    if not ok and "可选" not in name:
        all_ok = False
    print(f"  {status} {name}: {detail}")
print("=" * 60)
if all_ok:
    print("  🎉 全部必需数据就位，可以开始实验！")
else:
    print("  ⚠️ 有必需数据缺失，请先补齐再运行实验")
print("=" * 60)
VERIFY
```

---

## Operator

Claude (Cowork mode), 2026-02-27

Sources: [Yukang/LongAlpaca-12k](https://huggingface.co/datasets/Yukang/LongAlpaca-12k), [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench), [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), [ModelScope Llama-3-8B-Instruct](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct), [YeungNLP/LongQLoRA-Dataset](https://huggingface.co/datasets/YeungNLP/LongQLoRA-Dataset)
