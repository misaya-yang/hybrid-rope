# Fresh 72-case validation: calibration regresses; OLMo gains remain conditional

All six previously frozen arms completed. This report reads the saved generations;
it does not describe new model runs. The primary objective remains a validated
improvement over MrRoPE on models where the baseline works, not an OLMo-only win.

## Verified result

Equal-task macro accuracy on six RULER task types, with 4 cases per task at 4K
and 8 per task at 16K:

| Frozen arm | 4K | 16K |
|---|---:|---:|
| BM | 77.5000% | 48.2292% |
| b4wide | 79.9306% | 57.2917% |
| Gain-only calibrated b4wide | 78.8889% | 56.5972% |
| Task-decision calibrated b4wide | 79.9306% | 38.8542% |
| Official-index YaRN | 51.0417% | 14.0625% |
| MrRoPE-Pro | 26.5278% | 13.5417% |

The task-decision calibration's 16K loss against its own initializer is
**18.4375 percentage points**, with approximate stratified paired 95% t interval
**[-29.3424, -7.5326] pp** (4 wins, 13 losses, 31 ties). Its short macro gain
against that initializer is zero. Gain-only calibration does not show the same
large long loss: -0.6944 pp, interval [-6.0362, 4.6473]. Thus the completed controls
locate the observed regression in the task-calibration intervention, rather than
in b4wide itself. They do not establish that every possible task-calibration
method fails.

b4wide exceeds MrRoPE by 43.75 pp at 16K on this sample. This is real conditional
evidence for this OLMo checkpoint and protocol. It does not repair the failed
Qwen transfer or prove a universally better allocation. YaRN versus MrRoPE is
only +0.5208 pp at 16K, with both methods weak. This model/length is therefore
unsuitable as the sole evidence for why MrRoPE beats YaRN on healthy baselines.

## Raw verification

Local copy: `results/rope_decision_20260911/fresh72_completed/`.

- 72 unique token prompts; 432 generation records, 72 per arm.
- Every output's prompt hash matches the frozen plan.
- Recomputing the original RULER substring-based scorer from output text gives
  zero mismatches. This verifies consistency with the scorer, not immunity of
  the metric to verbose or malformed answers.
- Frozen plan SHA256:
  `6e91bf13c022e4b25c0555070869c53840c88060337097cc76ab8a341b79c057`.
- Per-file hashes and verification counts:
  `local_raw_verification.json`; full contrasts: `validation_readout.json`.
- Selection and prompt-disjointness records remain in `RESEARCH_OWNER_20260911.md`.
  These are sample-level independent validation results, not full RULER.

## Research decision

Do not expand the 65-parameter calibration from its four selected development
cases. Repeatedly fitting those cases did not transfer to these fresh inputs.
Do not substitute NLL or selected attention margins for task validation.

The next discriminating construction is the already proposed YaRN/MrRoPE
factorial: separate slots where MrRoPE increases frequency from slots where it
decreases frequency relative to official-index YaRN. The max/min hybrid tables
introduce no fitted shape parameters. They test whether the two interventions
have different task effects; neither is presumed better before generation.
Use the healthy Qwen baseline and reuse identified inputs and references.

## Execution state

SSH was recovered through the existing local SOCKS proxy after local DNS failed.
The instance is reachable, but `nvidia-smi` reports `No devices were found` and
there are no live campaign workers. The old gain sweep has 83/350 rows on gain
1.20; the negative LongBridge holdout arm has 60/180 rows. Preserve and resume
only missing work when GPU access returns; a restart must not overwrite results.
