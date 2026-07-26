# LLaMA-3-8B matched counterfactual continuation — rebuttal plan

Status: **archived design only / no current rebuttal need / do not run**
Primary concerns: `R27bE.2`, `R27bE.5`, `AC.2`, `AC.4`

> **Supersession notice.** The current rebuttal core is already sendable from
> the completed matched OLMo counterfactual endpoint plus the separate matched
> LLaMA natural-LM and RULER endpoints. No matched LLaMA counterfactual run is
> required or authorized. The protocol below is retained only as future
> experimental-design history and is not an action queue or current evidence.

## 1. Required five-line decision record

1. **Reviewer or AC concern addressed.** The strongest controlled multi-seed
   submitted evidence is below 1B, and the mature 8B evidence needs a clearer
   autoregressive capability endpoint.
2. **Existing evidence.** Matched seed-42 LLaMA-3-8B Native/EVQ LongAlpaca
   parents already establish 16K/32K NLL and remote-source effects. A later
   matched 8K RULER-family continuation gives 16K official macro
   `0.295%` for Native-LoRA and `14.03%` for EVQ-LoRA, but uses ordinary
   answer-only supervision. OLMo-2 1.485B shows that paired counterfactual
   routing can convert 4K training into 8K strict exact.
3. **Smallest missing evidence.** None for the current rebuttal. A matched
   LLaMA counterfactual pair would answer only a narrower future attribution
   question.
4. **Smallest executable plan.** No execution is authorized in the current
   response cycle. The historical design below would continue one Native and
   one EVQ RULER-family adapter only if a later research decision reopens it.
5. **Stop condition.** Stop at design status for the current rebuttal; do not
   launch either arm.

## 2. Historical rationale for the archived design

The current evidence chain is already nearly complete:

\[
\text{8B matched LM adaptation}
\rightarrow
\text{long-position NLL/source use}
\rightarrow
\text{8B task-family 2x length transfer}.
\]

A narrower future ambiguity is whether explicit causal source supervision
makes the 8B result stronger and more reproducible, rather than whether
another frequency mechanism should be invented. The archived design therefore:

- keeps Native and EVQ frequencies fixed at the two existing parent values;
- starts from the matched ordinary-CE RULER-family pair, whose complete
  training chains begin at the same seed-42 LongAlpaca parent pair;
- changes no model architecture;
- introduces no YaRN, virtual positions, long-sequence backward pass,
  homotopy, hybrid frequency table, DC channel, or EVQ-v2 mechanism;
- adds only a matched counterfactual training objective to both arms.

The older `frequency_adaptation_8b` progressive-morph proposal is not used.
Static LoRA cannot exactly transplant one RoPE generator into another, and the
OLMo progressive-morph screen is already negative. Here the parents have
already co-adapted to their own fixed frequency substrates.

## 3. Primary arms and reused controls

### New training arms

| Arm | Parent adapter | Fixed runtime frequency | New training |
| --- | --- | --- | --- |
| `native_cf` | matched Native 13-family adapter, step 516 | exact LLaMA endpoint Native RoPE | 300-step counterfactual continuation |
| `evq_cf` | matched EVQ 13-family adapter, step 516 | exact midpoint EVQ-Cosh, \(\tau=1.414\) | identical 300-step counterfactual continuation |

### Existing no-training controls

- untouched Native LLaMA-3-8B-Instruct;
- matched Native/EVQ LongAlpaca parents;
- step-0 snapshots of the completed Native/EVQ ordinary-CE 13-family
  RULER-mix adapters.

The primary schedule comparison is `native_cf` versus `evq_cf`. Each arm's
step-0 snapshot supplies the within-schedule before/after control for the
counterfactual objective.

## 4. Frozen training contract

- Model/tokenizer bytes: identical to the matched LongAlpaca pair.
- Parent training seed: 42.
- New continuation seed and row order: one frozen shared value.
- Physical sequence length: exactly 8,192; all position IDs are in
  `[0,8191]`.
- LoRA: existing Q/K/V/O rank 64, alpha 128; base model, LM head, and
  frequency table frozen.
- Precision/runtime: BF16, Flash-only SDPA, fused AdamW, TF32, persistent
  Inductor cache, and the fastest already completed LLaMA-8B 8K receipt.
- Global batch: eight; 300 optimizer steps; learning rate \(1\times10^{-5}\);
  10-step warmup and cosine decay matched across arms.
- Save evaluation candidates at steps 100, 200, and 300. Select one shared
  checkpoint step for both arms using the frozen validation rule below, never
  the final test rows.
- No 16K/32K backward pass.

The scientific variables are identical across the two new arms except for the
pre-existing Native versus EVQ frequency substrate and its parent adapter.

## 5. Frozen data and objective

Use the already validated 8K 13-family training view as the positive-row
backbone. Counterfactual batches come only from the eight NIAH families and
variable tracking, where the causal binding is explicit. CWE, FWE, QA1, and
QA2 remain ordinary answer-only rows rather than receiving a fabricated
single-source interpretation.

Freeze the optimizer-step mixture at `50%` counterfactual binding batches,
`40%` task-balanced ordinary 13-family batches, and `10%` LongAlpaca replay.
For every counterfactual semantic group, create two token-length-matched
branches:

- preserve template, filler, query position, answer position, token count,
  distractor count, and position IDs;
- branch \(x^+\) contains source value \(y\);
- branch \(x^-\) swaps that source to a same-token-length value \(y'\);
- supervise \(y\) on \(x^+\) and \(y'\) on \(x^-\);
- keep exact row, key, value, and QA-query overlap at zero between train and
  evaluation. Generator families and templates remain shared, so the claim is
  explicitly task-family-adapted rather than unseen-task transfer.

For a positive/source-swapped pair:

\[
\begin{aligned}
\mathcal L_{\mathrm{answer}}
&=\tfrac12\left[
\mathrm{CE}(y\mid x^+)+\mathrm{CE}(y'\mid x^-)
\right],\\
\mathcal L_{\mathrm{cf}}
&=\tfrac12\operatorname{softplus}
\left(1-\left[\log p(y\mid x^+)-\log p(y'\mid x^+)\right]\right)\\
&\quad+\tfrac12\operatorname{softplus}
\left(1-\left[\log p(y'\mid x^-)-\log p(y\mid x^-)\right]\right),\\
\mathcal L&=\mathcal L_{\mathrm{answer}}+0.5\mathcal L_{\mathrm{cf}}.
\end{aligned}
\]

This is the symmetric source-following objective used by the successful OLMo
design: each branch learns its own correct answer, and each must prefer that
answer over the answer licensed by the other source. Answer CE and margin are
normalized per supervised answer token. Prompt, filler, source, and query
labels are masked; for multi-token answers, the preference term is averaged
over supervised positions where the two answers differ. Natural replay uses
the same assistant-only CE in both arms. No full-token LM loss is allowed to
dominate the routing objective.

## 6. Evaluation matrix

### Primary capability endpoint

- Official task-specific autoregressive RULER scoring.
- Same frozen task set for every arm.
- 13 tasks at 8K, 16K, and 32K.
- Retain both official macro and normalized exact.
- Use the existing \(n=20\) matrix for the first gate. If the frozen 16K EVQ
  result exceeds Native by at least five macro points and has non-zero exact,
  expand inference only to at least \(n=50\) per 16K task.

### Causal source endpoint

On at least 100 disjoint semantic groups per reported length:

- strict autoregressive exact;
- answer NLL and full-vocabulary answer rank;
- original-versus-source-swapped answer following;
- source-removal NLL delta;
- results by source-to-generation distance bucket.

Triplets from one semantic group remain one statistical unit.

### Retention and language-model endpoint

- The same 24 external temporal packs at 8K/16K/32K.
- 8K RULER retention relative to each step-0 ordinary-CE parent.
- Report NLL and PPL, but never use them as substitutes for autoregressive
  capability.

### Frozen checkpoint-selection rule

At steps 0, 100, 200, and 300, score only the held-out validation partition.
Among steps 100/200/300, choose one shared step for both arms: maximize the
two-arm mean 16K score over the nine source-bearing families, subject to
neither arm losing more than five official-macro points on the 8K complete
matrix relative to its own step 0. The final \(n=20\) and expanded
\(n\ge50\) test rows are opened only after the shared step is selected.
Because 16K validation participates in checkpoint selection, the result must
be described as target-length-validated task adaptation, not as untuned
zero-shot extrapolation.

## 7. Pre-registered interpretation

### Positive 2× result

Claim task-adapted 2× capability only if:

1. both new arms learn the 8K canary;
2. `evq_cf` exceeds `native_cf` at 16K by at least five official-macro points;
3. EVQ has non-zero normalized exact at 16K;
4. relative to its own step-0 parent, `evq_cf` improves the pre-registered
   nine-family source-following score or source-causal margin without losing
   more than two points on the complete 16K official macro;
5. source-swap and source-removal effects agree with the capability direction;
6. the result is not driven by one task family.

### 4× boundary

32K must always be retained. If both arms remain at zero, say directly:

> Counterfactual supervision converts the EVQ substrate into measurable 2×
> task capability under the tested protocol, while 4× autoregressive
> capability remains open.

This is the planned narrow disclosure. It is materially relevant and prevents
the 2× result from being read as universal 4× downstream closure.

### Failure cases

- Only one arm learns 8K: report trainability, not length transfer.
- Both learn 8K but tie/fail at 16K: counterfactual supervision does not close
  the 2× gap.
- NLL improves but exact/source causality does not: retain only the LM endpoint.
- EVQ wins 16K only on the same training templates: report template-matched
  adaptation, not held-out generalization.

## 8. Reviewer-facing paragraph if the gate passes

> To separate generic LoRA adaptation from causal long-range use, we continued
> the matched seed-42 Native and EVQ LLaMA-3-8B RULER-family adapters for the
> same 300 physical-8K steps using token-length- and position-matched source
> counterfactuals. Both arms used identical rows, order, optimizer, Q/K/V/O
> rank-64 capacity, one shared checkpoint-selection rule, and evaluation; only
> their previously fixed frequency substrates differed. Using disjoint 16K
> validation rows to select one shared step, EVQ-LoRA reached [X] official
> RULER macro and
> [X] normalized exact, compared with [Y]/[Y] for Native-LoRA, while source
> swap/removal changed the answer probability in the same direction. At 32K,
> [RESULT]. We therefore claim matched, task-adapted 2× capability conversion,
> not unseen-task or universal 4× downstream superiority.

Until the experiment completes, every bracket remains a visible placeholder
and this paragraph is not evidence.

## 9. Estimated paid-GPU envelope

The completed 516-step LLaMA arms each took about 4,032 seconds on RTX Pro
6000. Reusing the compiled 8K path, a 300-step continuation should be planned
at roughly 40–50 minutes per arm plus evaluation. The READY receipt must
replace this estimate with a measured finite-loss/throughput probe before a
paid launch. Do not run both processes concurrently merely to fill memory;
select the configuration that minimizes total GPU seconds to the complete
matched answer.
