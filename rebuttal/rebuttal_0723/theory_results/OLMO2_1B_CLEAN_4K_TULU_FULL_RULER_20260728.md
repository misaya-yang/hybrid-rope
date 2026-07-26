# OLMo-2 1B Clean 4K Adaptation → Full RULER

Status: **complete negative result / claim guardrail**  
Concerns: `R27bE.2`, `R27bE.5`, `AC.2`

## Question and smallest executable test

1. **Concern addressed.** Does the approximately 1B mature-model result
   transfer beyond the NIAH-like task family used by the successful
   task-specific routing adaptation?
2. **Existing evidence.** A 4K-only EVQ routing curriculum produces a
   same-task 8K NIAH result, but that curriculum is structurally matched to the
   evaluation task. It is not evidence of unseen-task or full-RULER transfer.
3. **Smallest missing evidence.** Remove the custom binding/routing stage,
   preserve 4K-only adaptation, and evaluate all 13 prepared RULER tasks.
4. **Executable plan.** Start from the frozen 4K LongAlign EVQ adapter, run one
   deterministic pass over an official-source Tulu-3 assistant-only view, then
   evaluate 13 RULER tasks at 4K/8K/16K with 20 examples per cell.
5. **Stop condition.** Complete the 39 candidate cells; do not add a new
   training mechanism or train a control after a negative gate.

## Data boundary

The earlier positive routing arm did not copy evaluation rows, identities, or
values. Its training-data script directly calls the official RULER
`scripts/data/synthetic/niah.py` generator with custom templates and seeds.
Training and evaluation therefore share the benchmark generator family. The
result is valid only as benchmark-family-matched capability conversion or a
mechanism diagnostic.

The clean arm used:

- Stage A: an official LongAlign frozen 4K view, full-token next-token loss,
  611 steps and 20,016,360 supervised tokens;
- Stage B: `allenai/tulu-3-sft-olmo-2-mixture-0225` at revision
  `d91a0785ade02942520280fb484866fce41e448f`, assistant-content-and-EOS loss,
  one pass over 3,968 rows, 496 steps and 1,667,032 supervised tokens;
- no explicitly added custom binding, NIAH, or RULER benchmark rows;
- maximum physical training length 4,096 throughout.

“No explicitly added rows” is the defensible boundary. It is not a claim that
every upstream LongAlign or Tulu example has been semantically proved unrelated
to every retrieval-like task.

## Frozen protocol

| Item | Value |
| --- | --- |
| Model | `OLMo-2-0425-1B-Instruct` |
| Model SHA-256 | `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` |
| Frequency | endpoint EVQ-Cosh, \(\tau=2\) |
| `inv_freq` SHA-256 | `917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607` |
| Adaptation | Q/K/V/O LoRA, rank 64, alpha 128 |
| Stage-A adapter SHA-256 | `47e72c5e58be443a3f088787b58df415538f7cdb12e9f76d799dbb2b85155ee0` |
| Final adapter SHA-256 | `c3454e03f6a8c90d581e994dd0954f1079e820b3ece237692023b7708af3ccd9` |
| Final training seed | `20260728` |
| Evaluation | greedy autoregressive, official task-specific RULER scoring |
| Matrix | 13 tasks × 3 lengths × 20 examples = 780 generations |

The length-matched controls were reused from the already completed, same-row
formal matrix: untouched Native RoPE at 4K, untouched Native plus official
Transformers YaRN factor 2 at 8K, and factor 4 at 16K.

## Results

### Training and language modeling

Tulu validation NLL improved from `1.2635` to `1.1410`. Natural-text NLL was:

| Length | Clean EVQ NLL |
| ---: | ---: |
| 4K | 2.5766 |
| 8K | 2.7362 |
| 16K | 2.9714 |

The adapter therefore retained coherent teacher-forced language modeling. The
full-RULER result below is not explained by an obvious NLL explosion.

### Full RULER macro score

| Length | Clean EVQ | Length-matched control | Delta |
| ---: | ---: | ---: | ---: |
| 4K | 0.0974 | 0.6535 Native | -0.5561 |
| 8K | 0.0401 | 0.5872 official YaRN ×2 | -0.5471 |
| 16K | 0.0205 | 0.0908 official YaRN ×4 | -0.0703 |

No clean-EVQ cell strictly exceeded its length-matched control: **0/39**
positive cells. All 15 result groups completed, covering 13 tasks, 39 cells,
and 780 generated examples; no NaN, Inf, OOM, traceback, or failed status was
admitted.

The complete per-task matrix is retained in
`olmo2_1b_clean_4k_tulu_full_ruler_20260728.json`.

## Interpretation

This result is negative for broad capability transfer. It establishes three
important boundaries:

1. The earlier positive 8K NIAH result remains a legitimate
   **held-out-identity, same-task** conversion result, but it cannot be called
   unseen-task, full-RULER, or general downstream transfer.
2. Low natural-text NLL at 8K/16K does not by itself imply usable
   autoregressive long-context task performance.
3. Removing the task-matched binding curriculum does not recover broad RULER
   capability under this small 4K-only adaptation budget.

This experiment does not isolate EVQ allocation shape: it evaluates a complete
EVQ-plus-LoRA adaptation path against stronger length-matched deployment
controls. It must not be used to conclude that EVQ is intrinsically worse than
Native/YaRN under every training protocol.

## Reviewer-facing use

Do not use this as positive evidence. If the rebuttal cites the NIAH
`49/100` and `48/100` result, keep “same-task capability conversion” adjacent
and do not imply full-RULER transfer. If broader downstream capability is
asked directly, disclose that the clean 13-task matrix did not beat the
length-matched controls and keep the corresponding limitation open.

## Provenance

- Training READY SHA-256:
  `abe59548b893fa97b4282eaf8d386421b9631784ae61dbbfc44cde7a1ed37d30`
- Training result SHA-256:
  `d96688b07a2b1b55113ebf1c88b3a05aa6e94ec73a8785cb5c71d399aa5a72c8`
- Training-complete SHA-256:
  `1699f85768512445e1752429199e0d65e8fba4a86b4288e349ba026e89100aa3`
- RULER READY SHA-256:
  `e530f2858adea241a02f907a67c005ba35d97612fe1f38b3568f345b8e37cecf`
- Existing control summary SHA-256:
  `464bd2167d758621b8be03c4345ebd40950e78159c97b3862df93a1a30d699d5`
- Frozen 93-file evidence archive:
  `olmo2_clean_4k_tulu_full_ruler_s20260728.tar.gz`
- Evidence archive SHA-256:
  `0bc43a412d08d890897001d9a147c81ffaa550d58e78e0bb88f8352be4f93dd6`
