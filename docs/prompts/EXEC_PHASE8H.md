# Phase 8H 执行指令

## 服务器信息
- SSH: `ssh -p 24215 root@connect.bjb2.seetacloud.com`
- 密码: `LJlYtCph/ROu`
- 环境: conda base
- GPU: RTX 5090 32GB

## 任务

在服务器上执行 Phase 8H 实验：**base=10K 下 EVQ τ 系统扫描（50M tokens）**。

### 背景

已有数据（base=10K, 50M tokens, seed=42）：
- geo_4k: retrieval=0.680, PPL@16K=274.246
- evq1.1_4k: retrieval=0.5675, PPL@16K=282.446 （败）
- evq1.2_4k: retrieval=0.5675, PPL@16K=317.039 （败）

τ=1.1/1.2 完败 Geometric。现在需要扫描 τ∈[0.2, 1.0] 找到 base=10K 下真正的 τ*。

---

## 第一步：连接服务器，确认环境

```bash
ssh -p 24215 root@connect.bjb2.seetacloud.com
# 密码: LJlYtCph/ROu

conda activate base
nvidia-smi  # 确认 GPU 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 第二步：定位代码和已有数据

```bash
# 找到 Phase 8 代码目录
find /root -name "phase8_runner.py" -o -name "run_evq_sweep.py" 2>/dev/null
find /root -name "evq_phase8" -type d 2>/dev/null

# 确认已有的 base=10K 数据
ls /root/autodl-tmp/evq_phase8/base10k*/  # 或类似路径
# 找到 geo_4k 的结果确认 retrieval=0.680
```

**⚠️ 如果路径不同，请适配。关键是找到：**
1. `run_evq_sweep.py`（包含 `GPT` 类、`train_model()`、`eval_model()`、`evq_cosh_inv_freq()`）
2. `phase8_runner.py`（包含 `_run_passkey()` 和评估逻辑）
3. `eval_passkey_scratch.py` 或 `eval_passkey.py`（passkey 评估）
4. 已完成的 geo_4k 模型/结果目录

## 第三步：创建 Phase 8H 运行脚本

在代码目录下创建 `phase8h_sweep.py`：

```python
#!/usr/bin/env python3
"""
Phase 8H: Base=10K systematic τ sweep (50M tokens).
Sweeps τ ∈ [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] for EVQ.
Then optionally runs Hybrid with r=16/22/23 at the best τ.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch

# ── 自动适配：找到已有代码 ──────────────────────────
# 尝试 import 已有模块
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import (
    GPT, load_fineweb, eval_model, train_model,
    evq_cosh_inv_freq, geometric_inv_freq,
    set_seed, DEVICE
)
from eval_passkey_scratch import eval_passkey_nll_gap
from transformers import AutoTokenizer

# ── 配置 ─────────────────────────────────────────────
BASE = 10000.0
DIM = 64          # head_dim
SEQ = 4096
TOKENS = 50_000_000
LR = 6e-4
BATCH = 2
SEED = 42
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
PK_LENGTHS = [1024, 2048, 4096, 8192]
PK_TRIALS = 100

# Phase A: τ 粗扫
PHASE_A_TAUS = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

# Geo baseline (复用已有数据)
GEO_RESULTS = {
    "method": "geometric",
    "tau": None,
    "retrieval": 0.680,
    "mean_nll_gap": 0.1268,
    "ppl_16k": 274.246,
    "source": "reused from 8G"
}

# 工作目录
WORK = Path("/root/autodl-tmp/evq_phase8/base10k_8h")
WORK.mkdir(parents=True, exist_ok=True)

# ── Hybrid 频率生成（支持自定义 r）──────────────────
def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=1.0, r=16):
    """Hybrid: first r channels geometric, rest EVQ-cosh."""
    n = dim // 2  # 32
    geo = geometric_inv_freq(dim, base).double()

    n_evq = n - r
    if n_evq <= 0:
        return geo.float()

    theta_max_low = geo[r].item()
    theta_min_low = geo[-1].item()

    u = torch.arange(n_evq, dtype=torch.float64) / max(n_evq - 1, 1)
    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))

    evq_part = (theta_min_low ** phi) * (theta_max_low ** (1.0 - phi))
    inv_freq = torch.cat([geo[:r], evq_part])
    return inv_freq.float()

# ── 单次训练+评估 ────────────────────────────────────
def run_single(run_name, inv_freq, run_dir):
    """从零训练 350M，评估 PPL + passkey，保存结果。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    result_file = run_dir / "result.json"
    if result_file.exists():
        print(f"  [SKIP] {run_name}: result.json already exists")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  Running: {run_name}")
    print(f"  Dir: {run_dir}")
    print(f"{'='*60}")

    set_seed(SEED)

    # 模型配置
    cfg = {
        "vocab_size": 50304,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "head_dim": DIM,
        "intermediate_size": 4096,
        "max_position_embeddings": SEQ,
        "batch_size": BATCH,
        "train_tokens": TOKENS,
        "seq_len": SEQ,
        "lr": LR,
    }

    # 创建模型，注入频率
    model = GPT(cfg, inv_freq.clone()).to(DEVICE)

    # 加载数据
    train_data, val_data = load_fineweb(cfg)

    # 训练
    t0 = time.time()
    train_model(model, train_data, cfg, seed=SEED)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.0f}s")

    # PPL 评估
    ppl = eval_model(model, val_data, EVAL_LENGTHS)
    ppl_16k = ppl.get("16384", ppl.get(16384, None))
    print(f"  PPL@16K = {ppl_16k}")

    # Passkey 评估
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    filler_ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 20,
                            add_special_tokens=False)
    pk = eval_passkey_nll_gap(
        model, tok, filler_ids,
        lengths=PK_LENGTHS, depths=[0.5], num_trials=PK_TRIALS,
    )
    g = pk.get("global", {})
    retrieval = g.get("retrieval_rate", 0)
    nll_gap = g.get("mean_nll_gap", 0)
    print(f"  Retrieval = {retrieval:.4f}, NLL gap = {nll_gap:.4f}")

    # 保存
    result = {
        "method": run_name,
        "ppl": ppl,
        "ppl_16k": ppl_16k,
        "retrieval": retrieval,
        "mean_nll_gap": nll_gap,
        "passkey_detail": pk,
        "train_time_sec": train_time,
        "config": {"base": BASE, "seq": SEQ, "tokens": TOKENS, "seed": SEED}
    }

    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # 释放显存
    del model
    torch.cuda.empty_cache()

    return result

# ── Phase A: EVQ 粗扫 ────────────────────────────────
def phase_a():
    print("\n" + "="*60)
    print("  PHASE A: EVQ τ sweep [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]")
    print("="*60)

    all_results = {"geo_4k": GEO_RESULTS}

    for tau in PHASE_A_TAUS:
        run_name = f"evq{tau}_4k"
        inv_freq = evq_cosh_inv_freq(DIM, tau, BASE)
        run_dir = WORK / run_name
        result = run_single(run_name, inv_freq, run_dir)
        all_results[run_name] = {
            "tau": tau,
            "retrieval": result["retrieval"],
            "ppl_16k": result["ppl_16k"],
            "mean_nll_gap": result["mean_nll_gap"],
        }

        # 实时打印对比
        print(f"\n  --- 当前汇总 ---")
        print(f"  {'Method':<15s} {'τ':>5s} {'Retrieval':>10s} {'PPL@16K':>10s} {'vs Geo ret':>12s} {'vs Geo PPL':>12s}")
        geo_ret = GEO_RESULTS["retrieval"]
        geo_ppl = GEO_RESULTS["ppl_16k"]
        for name, r in all_results.items():
            ret = r.get("retrieval", 0) or 0
            ppl = r.get("ppl_16k", 0) or 0
            tau_s = str(r.get("tau", "—"))
            ret_diff = f"{(ret/geo_ret - 1)*100:+.1f}%" if geo_ret and ret else "—"
            ppl_diff = f"{(ppl/geo_ppl - 1)*100:+.1f}%" if geo_ppl and ppl else "—"
            print(f"  {name:<15s} {tau_s:>5s} {ret:10.4f} {ppl:10.1f} {ret_diff:>12s} {ppl_diff:>12s}")

    # 找 peak
    evq_results = {k: v for k, v in all_results.items() if k.startswith("evq")}
    best_ret_name = max(evq_results, key=lambda k: evq_results[k]["retrieval"])
    best_ppl_name = min(evq_results, key=lambda k: evq_results[k]["ppl_16k"] or 9999)

    print(f"\n  Best retrieval: {best_ret_name} = {evq_results[best_ret_name]['retrieval']:.4f}")
    print(f"  Best PPL@16K:   {best_ppl_name} = {evq_results[best_ppl_name]['ppl_16k']:.1f}")
    print(f"  Geo baseline:   retrieval={geo_ret}, PPL@16K={geo_ppl}")

    beats_geo_ret = evq_results[best_ret_name]["retrieval"] > geo_ret
    beats_geo_ppl = (evq_results[best_ppl_name]["ppl_16k"] or 9999) < geo_ppl

    print(f"\n  EVQ beats Geo on retrieval? {'YES ✅' if beats_geo_ret else 'NO ❌'}")
    print(f"  EVQ beats Geo on PPL@16K?   {'YES ✅' if beats_geo_ppl else 'NO ❌'}")

    # 保存汇总
    summary = {
        "phase": "8H-A",
        "all_results": all_results,
        "best_retrieval": {"name": best_ret_name, **evq_results[best_ret_name]},
        "best_ppl": {"name": best_ppl_name, **evq_results[best_ppl_name]},
        "beats_geo_retrieval": beats_geo_ret,
        "beats_geo_ppl": beats_geo_ppl,
        "verdict": "PROCEED_TO_PHASE_B" if (beats_geo_ret or beats_geo_ppl) else "STOP",
    }

    with open(WORK / "phase8h_phaseA_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to {WORK}/phase8h_phaseA_summary.json")
    return summary

# ── Phase B: 精细扫 + Hybrid ─────────────────────────
def phase_b(tau_peak):
    """Phase A 之后手动调用，传入最优 τ。"""
    print(f"\n{'='*60}")
    print(f"  PHASE B: Fine sweep around τ_peak={tau_peak} + Hybrid")
    print(f"{'='*60}")

    all_results = {}

    # 精细 EVQ 扫描
    fine_taus = [
        round(tau_peak - 0.10, 2),
        round(tau_peak - 0.05, 2),
        round(tau_peak + 0.05, 2),
        round(tau_peak + 0.10, 2),
    ]
    fine_taus = [t for t in fine_taus if 0.05 <= t <= 2.0]  # 范围限制

    for tau in fine_taus:
        run_name = f"evq{tau}_4k_fine"
        inv_freq = evq_cosh_inv_freq(DIM, tau, BASE)
        result = run_single(run_name, inv_freq, WORK / run_name)
        all_results[run_name] = {
            "tau": tau, "retrieval": result["retrieval"],
            "ppl_16k": result["ppl_16k"], "mean_nll_gap": result["mean_nll_gap"],
        }

    # Hybrid 对比 (r=16, 22, 23)
    for r in [16, 22, 23]:
        run_name = f"hybrid{tau_peak}_r{r}_4k"
        inv_freq = hybrid_evq_inv_freq(DIM, BASE, tau_peak, r=r)
        result = run_single(run_name, inv_freq, WORK / run_name)
        all_results[run_name] = {
            "tau": tau_peak, "r": r, "retrieval": result["retrieval"],
            "ppl_16k": result["ppl_16k"], "mean_nll_gap": result["mean_nll_gap"],
        }

    # 汇总
    print(f"\n  --- Phase B 汇总 ---")
    geo_ret = GEO_RESULTS["retrieval"]
    geo_ppl = GEO_RESULTS["ppl_16k"]
    print(f"  {'Method':<25s} {'τ':>5s} {'r':>3s} {'Retrieval':>10s} {'PPL@16K':>10s} {'vs Geo ret':>12s}")
    for name, r in all_results.items():
        ret = r.get("retrieval", 0)
        ppl = r.get("ppl_16k", 0)
        ret_diff = f"{(ret/geo_ret - 1)*100:+.1f}%" if geo_ret and ret else "—"
        print(f"  {name:<25s} {str(r.get('tau','—')):>5s} {str(r.get('r','—')):>3s} {ret:10.4f} {ppl:10.1f} {ret_diff:>12s}")

    summary = {"phase": "8H-B", "tau_peak": tau_peak, "results": all_results}
    with open(WORK / "phase8h_phaseB_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to {WORK}/phase8h_phaseB_summary.json")
    return summary


# ── 主入口 ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="A", choices=["A", "B", "AB"])
    parser.add_argument("--tau_peak", type=float, default=None,
                        help="Phase B: τ_peak from Phase A (required for --phase B)")
    args = parser.parse_args()

    if args.phase in ("A", "AB"):
        summary_a = phase_a()

        if args.phase == "AB" and summary_a["verdict"] == "PROCEED_TO_PHASE_B":
            # 自动从 Phase A 的 best τ 进入 Phase B
            tau_peak = summary_a["best_retrieval"]["tau"]
            if not summary_a["beats_geo_retrieval"]:
                tau_peak = summary_a["best_ppl"]["tau"]
            print(f"\n  Auto-entering Phase B with τ_peak={tau_peak}")
            phase_b(tau_peak)
        elif args.phase == "AB":
            print("\n  Phase A verdict: STOP. No τ beats Geo. Skipping Phase B.")

    elif args.phase == "B":
        if args.tau_peak is None:
            print("ERROR: --tau_peak required for Phase B")
            sys.exit(1)
        phase_b(args.tau_peak)

    print("\n  DONE. All results in:", WORK)
```

## 第四步：执行

```bash
cd <代码目录>  # 即 run_evq_sweep.py 所在目录

# 方式1：先跑 Phase A，看结果再决定 Phase B
python phase8h_sweep.py --phase A

# 方式2：全自动（Phase A → 自动判定 → Phase B）
python phase8h_sweep.py --phase AB

# 方式3：手动指定 τ_peak 跑 Phase B（Phase A 之后）
python phase8h_sweep.py --phase B --tau_peak 0.7
```

**预计时间**：
- Phase A: 7 runs × ~25min = ~3h
- Phase B: 7 runs × ~25min = ~3h
- 全自动 AB: ~6h

## 第五步：结果收集

Phase A 完成后，汇总在 `base10k_8h/phase8h_phaseA_summary.json`。
Phase B 完成后，汇总在 `base10k_8h/phase8h_phaseB_summary.json`。
每个单独 run 在各自目录的 `result.json`。

**请将 phase8h_phaseA_summary.json（和 phase8h_phaseB_summary.json 如果有的话）的内容发回给我。**

---

## ⚠️ 重要注意事项

1. **脚本中的 import 路径可能需要适配**：如果 `from run_evq_sweep import ...` 报错，检查实际文件位置并修正 `sys.path`
2. **`evq_cosh_inv_freq` 函数签名可能不同**：已有代码可能是 `evq_cosh_inv_freq(head_dim, tau, base)` 或其他参数顺序，请按实际调整
3. **`eval_passkey_scratch.py` 的 import 名可能不同**：可能是 `eval_passkey` 而非 `eval_passkey_scratch`，请按实际调整
4. **filler_ids 的获取方式**：已有代码可能有不同的 filler 文本生成方式，用已有方式即可
5. **如果某个 τ 的 run 之前已经跑过**（比如 10M 版本），**不要复用**——必须用 50M tokens 重新训练
6. **数据集缓存**：50M tokens 的 FineWeb 数据应该已经在本地缓存了，不需要重新下载
7. **显存**：350M 模型 + 4096 seq_len + batch=2 在 32GB 卡上没问题
8. **所有 run 必须 seed=42**：确保和 Geo baseline 可比
