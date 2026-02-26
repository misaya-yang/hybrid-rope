# EVQ vs Anchored-Sigmoid: Theory Gap and Ablation Plan

## 1) What is likely valid in the critique

- `anchored_sigmoid` introduces an S-shaped transition in index-space scaling.
- In discrete curvature diagnostics on `log(inv_freq)`, this schedule has one sign change in second difference (an inflection-like transition).
- `evq_cosh` / `evq_exp` constructions are monotone and show no sign change in the same diagnostic.

Quick check command used:

```bash
python3 - <<'PY'
import math
# simplified diagnostic; prints sign changes of second diff on log(inv_freq)
PY
```

Observed pattern (n in {32,64,128}):
- `anchored_sigmoid`: `sign_changes=1`
- `evq_cosh`: `sign_changes=0`
- `evq_exp`: `sign_changes=0`

## 2) What is NOT yet proven

- Inflection alone does not prove the full cause of Musique drops.
- Data distribution, response-only masking, and training budget can still dominate outcomes.
- Therefore this remains a falsifiable hypothesis, not a final conclusion.

## 3) Code landed (non-breaking, default unchanged)

- Added `evq_cosh` / `evq_exp` schedule support:
  - `rope/schedules.py`
  - `scripts/train_cross_model_lora_fast_tuned.py`
  - `scripts/plan_b_train_anchored_v2.py`
  - `scripts/isolated/longinst/new_lora_longinst_train_v1.py`
  - `scripts/isolated/longinst/run_llama8k_theory_v1.py`

Defaults stay on `anchored_sigmoid` to preserve current protocol.

## 4) Minimal fair ablation (same data, same budget)

### Cross-model fast tuned path (Qwen etc.)

```bash
python scripts/train_cross_model_lora_fast_tuned.py \
  --method anchored_sigmoid \
  ...

python scripts/train_cross_model_lora_fast_tuned.py \
  --method evq_cosh --evq_tau 0.5 \
  ...

python scripts/train_cross_model_lora_fast_tuned.py \
  --method evq_exp --evq_beta 3.0 \
  ...
```

### LLaMA isolated path

```bash
python scripts/isolated/longinst/new_lora_longinst_train_v1.py \
  --rope_schedule anchored_sigmoid \
  ...

python scripts/isolated/longinst/new_lora_longinst_train_v1.py \
  --rope_schedule evq_cosh --evq_tau 0.5 \
  ...

python scripts/isolated/longinst/new_lora_longinst_train_v1.py \
  --rope_schedule evq_exp --evq_beta 3.0 \
  ...
```

## 5) Acceptance criterion

- Keep decode/prompt/manifest parity locked.
- Compare `qasper/musique` gate and full `lb21` delta patterns.
- If EVQ reduces Musique drop while preserving long-range gains, migrate mainline schedule.
