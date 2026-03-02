# Phase 8H 快速执行指令（先打 τ=0.7/0.8 靶心）

## 服务器信息
- SSH: `ssh -p 24215 root@connect.bjb2.seetacloud.com`
- 密码: `LJlYtCph/ROu`
- 环境: conda base
- GPU: RTX 5090 32GB

## 任务概述

在服务器上执行 base=10K 的 EVQ 实验。理论预测最优 τ*(10K)≈0.76，因此先只跑 τ=0.7 和 τ=0.8，确认哪个更好后再精细扫。

### 已有数据（不需要重跑）

| Method | τ | retrieval | PPL@16K |
|--------|---|-----------|---------|
| geo_4k | — | 0.680 | 274.246 |
| evq1.1_4k | 1.1 | 0.5675 | 282.446 |
| evq1.2_4k | 1.2 | 0.5675 | 317.039 |

## 执行步骤

### 1. 连接服务器

```bash
ssh -p 24215 root@connect.bjb2.seetacloud.com
# 密码: LJlYtCph/ROu
```

### 2. 确认环境和代码位置

```bash
conda activate base
nvidia-smi
# 找到 Phase 8 代码
find /root -name "run_evq_sweep.py" 2>/dev/null
find /root -name "phase8_runner.py" 2>/dev/null
# 找到已有的 base=10K 实验目录
find /root -path "*/base10k*" -name "result.json" 2>/dev/null
find /root -path "*evq_phase8*" -type d 2>/dev/null
```

记下代码目录（例如 `/root/autodl-tmp/hybrid-rope/scripts/m4_evq_sweep/`）和数据目录。

### 3. 阅读已有代码确认接口

进入代码目录，阅读 `run_evq_sweep.py`，确认以下函数的签名：
- `evq_cosh_inv_freq(head_dim, tau, base)` — EVQ 频率生成
- `GPT(cfg, inv_freq)` — 模型创建
- `train_model(model, data, cfg, seed)` — 训练
- `eval_model(model, val_data, eval_lengths)` — PPL 评估
- `geometric_inv_freq(dim, base)` — Geometric 频率

同时确认 passkey 评估的 import：
- `eval_passkey_scratch.py` 或 `eval_passkey.py` 中的 `eval_passkey_nll_gap`

**如果函数签名不同，按实际代码调整下面的脚本。**

### 4. 创建并运行快速扫描脚本

在代码目录下创建 `phase8h_fast.py`：

```python
#!/usr/bin/env python3
"""Phase 8H Fast: 只跑 τ=0.7 和 τ=0.8（理论预测最优区间）"""

import json, math, os, sys, time
from pathlib import Path
import torch

# ── 适配 import ──────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import (
    GPT, load_fineweb, eval_model, train_model,
    evq_cosh_inv_freq, geometric_inv_freq,
    set_seed, DEVICE
)
# passkey 评估 —— 按实际文件名调整
try:
    from eval_passkey_scratch import eval_passkey_nll_gap
except ImportError:
    from eval_passkey import eval_passkey_nll_gap

from transformers import AutoTokenizer

# ── 配置 ─────────────────────────────────────────────
BASE      = 10000.0
DIM       = 64
SEQ       = 4096
TOKENS    = 50_000_000
LR        = 6e-4
BATCH     = 2
SEED      = 42

EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
PK_LENGTHS   = [1024, 2048, 4096, 8192]
PK_TRIALS    = 100

# ── 工作目录（按实际路径调整）──────────────────────
WORK = Path("/root/autodl-tmp/evq_phase8/base10k_8h")
WORK.mkdir(parents=True, exist_ok=True)

# ── Geo baseline ─────────────────────────────────────
GEO = {"retrieval": 0.680, "ppl_16k": 274.246, "mean_nll_gap": 0.1268}

# ── 训练+评估 ────────────────────────────────────────
def run_one(tau):
    tag = f"evq{tau}_4k"
    run_dir = WORK / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    result_file = run_dir / "result.json"

    if result_file.exists():
        print(f"\n[SKIP] {tag}: already done")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  START: {tag}  (τ={tau}, base={BASE}, 50M tokens)")
    print(f"{'='*60}")

    set_seed(SEED)

    cfg = dict(
        vocab_size=50304, hidden_size=1024, num_layers=24,
        num_heads=16, head_dim=DIM, intermediate_size=4096,
        max_position_embeddings=SEQ,
        batch_size=BATCH, train_tokens=TOKENS, seq_len=SEQ, lr=LR,
    )

    inv_freq = evq_cosh_inv_freq(DIM, tau, BASE)
    model = GPT(cfg, inv_freq.clone()).to(DEVICE)

    train_data, val_data = load_fineweb(cfg)

    t0 = time.time()
    train_model(model, train_data, cfg, seed=SEED)
    train_sec = time.time() - t0
    print(f"  Training: {train_sec/60:.1f} min")

    # PPL
    ppl = eval_model(model, val_data, EVAL_LENGTHS)
    ppl_16k = ppl.get("16384", ppl.get(16384, None))
    print(f"  PPL@16K = {ppl_16k}")

    # Passkey
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    filler = tok.encode(
        "The quick brown fox jumps over the lazy dog. " * 20,
        add_special_tokens=False
    )
    pk = eval_passkey_nll_gap(
        model, tok, filler,
        lengths=PK_LENGTHS, depths=[0.5], num_trials=PK_TRIALS,
    )
    g = pk.get("global", {})
    ret  = g.get("retrieval_rate", 0)
    gap  = g.get("mean_nll_gap", 0)
    print(f"  Retrieval = {ret:.4f},  NLL gap = {gap:.4f}")

    # 对比
    ret_vs = (ret / GEO["retrieval"] - 1) * 100
    ppl_vs = (ppl_16k / GEO["ppl_16k"] - 1) * 100 if ppl_16k else None
    print(f"  vs Geo:  retrieval {ret_vs:+.1f}%,  PPL@16K {ppl_vs:+.1f}%")

    result = dict(
        method=tag, tau=tau, retrieval=ret, mean_nll_gap=gap,
        ppl=ppl, ppl_16k=ppl_16k, train_sec=train_sec,
        passkey_detail=pk,
        config=dict(base=BASE, seq=SEQ, tokens=TOKENS, seed=SEED),
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    del model; torch.cuda.empty_cache()
    return result


# ── 主流程 ────────────────────────────────────────────
if __name__ == "__main__":
    results = {}

    for tau in [0.7, 0.8]:
        r = run_one(tau)
        results[f"evq{tau}"] = r

    # ── 汇总 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY  (base=10K, 50M tokens, seed=42)")
    print(f"{'='*60}")
    print(f"  {'Method':<14s} {'τ':>5s} {'Ret':>8s} {'PPL@16K':>9s} {'vs Geo ret':>12s} {'vs Geo PPL':>12s}")
    print(f"  {'geo_4k':<14s} {'—':>5s} {GEO['retrieval']:8.4f} {GEO['ppl_16k']:9.1f} {'baseline':>12s} {'baseline':>12s}")

    for tag, r in results.items():
        ret = r.get("retrieval", 0)
        ppl = r.get("ppl_16k", 0)
        tau_s = str(r.get("tau", "?"))
        ret_d = f"{(ret/GEO['retrieval']-1)*100:+.1f}%" if ret else "—"
        ppl_d = f"{(ppl/GEO['ppl_16k']-1)*100:+.1f}%" if ppl else "—"
        print(f"  {tag:<14s} {tau_s:>5s} {ret:8.4f} {ppl:9.1f} {ret_d:>12s} {ppl_d:>12s}")

    # 加上已有的 1.1/1.2
    print(f"  {'evq1.1':<14s} {'1.1':>5s} {'0.5675':>8s} {'282.4':>9s} {'-16.5%':>12s} {'+3.0%':>12s}")
    print(f"  {'evq1.2':<14s} {'1.2':>5s} {'0.5675':>8s} {'317.0':>9s} {'-16.5%':>12s} {'+15.6%':>12s}")

    # 判定
    best = max(results.values(), key=lambda x: x.get("retrieval", 0))
    beats_geo = best["retrieval"] > GEO["retrieval"]
    print(f"\n  Best EVQ retrieval: τ={best['tau']} → {best['retrieval']:.4f}")
    print(f"  Beats Geo (0.680)?  {'YES ✅' if beats_geo else 'NO ❌'}")

    best_ppl = min(results.values(), key=lambda x: x.get("ppl_16k", 9999) or 9999)
    beats_ppl = (best_ppl.get("ppl_16k") or 9999) < GEO["ppl_16k"]
    print(f"  Best EVQ PPL@16K:  τ={best_ppl['tau']} → {best_ppl['ppl_16k']:.1f}")
    print(f"  Beats Geo (274.2)?  {'YES ✅' if beats_ppl else 'NO ❌'}")

    verdict = "NEXT: Phase B (fine sweep + Hybrid)" if (beats_geo or beats_ppl) else "NEXT: expand to τ=0.4/0.5/0.6 or STOP"
    print(f"\n  VERDICT: {verdict}")

    summary = dict(
        phase="8H-fast", results=results, geo=GEO,
        best_ret=dict(tau=best["tau"], retrieval=best["retrieval"]),
        best_ppl=dict(tau=best_ppl["tau"], ppl_16k=best_ppl["ppl_16k"]),
        beats_geo_ret=beats_geo, beats_geo_ppl=beats_ppl,
        verdict=verdict,
    )
    with open(WORK / "phase8h_fast_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {WORK}/phase8h_fast_summary.json")
```

运行：

```bash
cd <代码目录>   # run_evq_sweep.py 所在目录
python phase8h_fast.py
```

预计 ~50 分钟（2 runs × 25min）。完成后把 `phase8h_fast_summary.json` 内容发回。

### 5. 根据结果决定下一步

**情况 A — τ=0.7 或 0.8 赢了 Geo**:
→ 进 Phase B：在赢家 ±0.1/±0.05 精细扫 + Hybrid r=22/23 对比

**情况 B — PPL 赢但 retrieval 没赢**:
→ 补跑 τ=0.5 和 0.6 看趋势

**情况 C — 全面不赢**:
→ 补跑 τ=0.4 和 0.5，如果还不行则 base=10K 下 EVQ 可能整体不 work

## ⚠️ 注意

1. 只改 τ，其他全部和 geo_4k 一致（base=10000, seed=42, 50M tokens）
2. 脚本有断点续跑功能（检测 result.json 存在则跳过）
3. import 路径按服务器实际调整——如果 `from run_evq_sweep import ...` 报错，检查实际模块名
4. `load_fineweb` 的数据应该已缓存在本地，不需要重新下载
5. 如果遇到显存问题，可能需要调 `eval_chunks` 或 `EVAL_LENGTHS` 中去掉 16384
