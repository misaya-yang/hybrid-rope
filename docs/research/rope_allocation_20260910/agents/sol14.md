# Sol14 — failure ledger and a role-conditioned allocation rule

Date: 2026-09-10. Scope: full assigned failure-dialogue audit plus the assigned project/Pro evidence shard. No GPU work, model download, source edit, or extra agent was used. Archived reports are treated as claims unless supported by the cited primary artifact.

## Result

The strongest rule consistent with this shard is **allocate labeled positional channels to maximize useful-source margin relative to content-conditioned interference, subject to explicit native-operation constraints**. This is one principle but two optimization regimes:

* **Training from scratch:** channel identities are exchangeable before learning, so a population density can be designed under an explicitly prescribed signal field and nuisance covariance. The conditional matched-filter solution is
  \[
  \rho^*(x)=\frac{[C^{-1}h(x)]_+}{\int [C^{-1}h(y)]_+\,dy},
  \]
  with active-set KKT conditions when positivity binds. EVQ-Cosh is recovered only under its special constant-signal and nested slow-tail covariance approximation. It is a training-time initialization/design law; the trained response law can change with the allocation.
* **Frozen deployment:** slots and learned circuits are not exchangeable. Keep the labeled frequency vector \(\nu\), measure signed source-versus-each-distractor score-margin moments on the actual roles, and solve a finite constrained maximin problem. MrRoPE-Pro is a strong feasible reference/heuristic, not an exact solution of the scratch density problem.

For frozen operations \(r\), let the exact pre-RoPE-fixed margin be
\[
m_r(\nu;u)=\sum_j a_{rj}(u)\cos(d_r\nu_j)+b_{rj}(u)\sin(d_r\nu_j)+c_r(u),
\]
where \(u\) indexes content instances and \(r\) identifies a useful source against one competing key. Define \(\mu_r(\nu)=\mathbb E m_r\) and \(v_r(\nu)=\operatorname{Var}m_r\). A concrete rule is
\[
\boxed{
\max_{\nu\in\mathcal F}\ \min_{r\in\mathcal R_{\rm far}}
\frac{\mu_r(\nu)}{\sqrt{v_r(\nu)+\epsilon_r^2}}
\quad\text{s.t.}\quad
\mu_q(\nu)-\gamma_q\sqrt{v_q(\nu)+\epsilon_q^2}\ge0,
\ q\in\mathcal R_{\rm native}.}
\]
Here \(\mathcal F\) fixes endpoint/support policy, positivity, ordering, and any declared movement limits; \(\epsilon\) covers estimation and upstream-transport uncertainty. For a margin with positive mean, Cantelli gives \(P[m_r\le0]\le v_r/(v_r+\mu_r^2)\); a union bound converts the per-distractor constraints into a source-selection probability bound. This is a conditional attention result, not a generated-answer theorem.

Use a local convex approximation only to propose a direction. Accept a table only after exact finite trigonometric replay of every native and far constraint, followed by the requested full-model behavior test. If the measured signed moments fail to rank the already observed MrPro/Smooth/P2 outcomes in the right direction, the rule is missing the operative role, value/readout, or upstream-state effect; do not add another unsigned penalty to force the ranking.

## Non-redundant failure ledger

### F1. Scope expansion replaced the requested deliverable

In the April submission-hygiene episode the task excluded new experiments, numeric changes, and a large theory addition. The assistant nevertheless introduced an oversized appendix derivation, discovered it only during final diff review, and rolled it back (`corpus/sol14_full_dialogue.jsonl:29–56`). The final patch was acceptable only after returning to metadata correction, evidence-tier separation, calibrated wording, and compilation. Constraint: the mathematical object must be the allocation decision the user asked for; an elegant derivation is not permission to enlarge the artifact.

### F2. A proxy was repeatedly treated as the requested outcome

The remote experiment episodes repeatedly reported PPL, partial passkey retrieval, checkpoint presence, or process liveness as if they closed broader claims. The user ultimately specified the exact matrix: lengths 2/4/8/16/24/32/40/48K, four checkpoints (Geo/EVQ before and after continuation), raw and YaRN overlays, and both PPL and passkey (`corpus/sol14_full_dialogue.jsonl:296–303`). Before that correction, the assistant successively proposed a 48K single point, a staged boundary probe, and a smaller matrix. Constraint: preserve model set, length set, overlay state, endpoint, and metric jointly; no one-dimensional proxy can replace the requested tensor of outcomes.

### F3. Runtime diagnosis was asserted before discriminating evidence

After three successful runs and one step-0 OOM, the assistant first reduced microbatch through gradient accumulation, producing a stable but much slower run. The user correctly pointed to compile/CUDA-graph private pools and sequential-run fragmentation as a competing explanation (`corpus/sol14_full_dialogue.jsonl:120–128`). The assistant then suggested disabling `torch.compile(max-autotune)`, which contradicted the throughput objective and was immediately rejected (`:128–130`). Constraint: separate a conservative recovery intervention from a causal diagnosis. In allocation, a movement that creates headroom is not evidence for the claimed frequency mechanism.

### F4. ETA was computed before checking the protocol that determined it

The assistant estimated the whole 1024→2048 pipeline from historical duration, then only after challenge checked whether \(\tau\), microbatch, accumulation, and effective batch changed with length (`corpus/sol14_full_dialogue.jsonl:135–149`). It found the automatic continuation did not match the seed-42 batch protocol and had to patch it. Constraint: allocation comparisons must freeze the full protocol before interpreting duration or outcome; changing support length while silently changing optimizer batch geometry confounds attribution.

### F5. Completion was inferred from a file gate while the actual job had failed

The continuation controller waited for `model.pt` and printed “still training” while the GPU was idle; later stage 2 crashed on a CUDAGraph overwritten-output error (`corpus/sol14_full_dialogue.jsonl:99–112`, `:152–155`). Constraint: proposed, launched, running, checkpointed, evaluated, and accepted are distinct states. A frequency table receipt or geometry certificate is not a capability result.

### F6. Evaluation harness complexity created new invalidity

For the long evaluation, the assistant added staged harnesses, fallbacks, watchers, tokenizer recovery, and matrix redesign before locking the requested evaluation. Launches then failed on import paths, online tokenizer lookup, missing `inv_freq.npy`, and wrong checkpoint paths (`corpus/sol14_full_dialogue.jsonl:264–300`). These were engineering failures, not model results. Constraint: use the smallest existing evaluator that preserves model identity and the requested endpoints; fallbacks must not silently change the scientific condition.

### F7. Wrong checkpoint and wrong architecture made attempted results invalid

The untrained baseline directories contained `ckpt_25%/50%`, not `model.pt`; the assistant chose `ckpt_50%.pt` only after failure. More seriously, it loaded a 454M checkpoint with a generic 500M configuration, leaving four layers missing/randomly initialized, and correctly stopped the run as invalid (`corpus/sol14_full_dialogue.jsonl:302–308`). Constraint: checkpoint SHA/config/slot table/tokenizer/data identity are preconditions, not cleanup. In frozen allocation, labeled slot identity is part of the model.

### F8. The assistant stopped with the requested evaluation uncompleted

After the final user correction and further frustration, the assistant acknowledged the exact matrix but ended without a valid run (`corpus/sol14_full_dialogue.jsonl:309–312`). This is the clearest proposed-versus-implemented-versus-tested distinction in the shard: the matrix was finally specified, a script was attempted, but no valid complete evaluation existed. Constraint: do not call a design, loader smoke, or partial row evidence a result.

## Why unsigned geometry cannot select the table

Four repository facts independently reject an unsigned or unlabeled allocation rule:

1. A clipped-affine approximation with movement MAE 0.001223 still failed native retention; 81.2% of squared residual sat at one slot. Small coordinate error did not imply small functional error (`.agents/explorer_survey_3/handoff.md:14–22`).
2. Permuting an identical frequency multiset doubled OLMo native NLL (3.104234→6.864926) and collapsed Qwen 64K core-4 from 0.7000 to 0 (`.agents/explorer_survey_3/handoff.md:23–30`). Slot labels and learned orientations matter.
3. Fixed-support EVQ beat the comparator at long lengths across three seeds, while support retargeting reversed every long comparison (`.agents/explorer_survey_3/handoff.md:43–57`). Allocation and support are coupled; a scratch optimum cannot simply be transplanted.
4. Smooth_MrBudget improved unresolved energy, unweighted distortion, projection-weighted MSE, and weak-response exposure yet was 9.79 points worse than MrPro on the 128K development panel; P2 traded a +3.54 long gain for −14.31 short, while E1 slot 28 showed only a small development positive (`reports/astra03.md:11–20`). Any positive combination of those unsigned costs is therefore insufficient.

The 128K BM diagnosis adds the correct nuance: compact content at original 128K positions was solvable, while the same visible token/position set read from a dense-prefill KV state was not (`docs/research/ROPE_BM_128K_DIAGNOSIS_20260908.md:24–50`). Phase magnitude alone is not sufficient; content-conditioned state history and competition enter the margin distribution.

## How the rule relates EVQ and MrRoPE without false equivalence

EVQ and MrRoPE answer different conditional problems. Under scratch symmetry and a prescribed exchangeable response model, optimizing channel density is meaningful; the Cosh density is the unique solution of its selected broadband covariance surrogate, not the exact finite-window task risk. Exact finite-window collision energy can instead have an atomic equilibrium, so “correcting” Cosh does not yield a universal smooth law (`reports/astra02.md:5–19,59–103`).

For frozen deployment, the checkpoint has labeled channels and co-adapted content coefficients. MrPro supplies a robust, hand-designed feasible table and useful boundary policy. It should be the reference point for a signed-margin intervention, not reverse-engineered as an optimum by defining \(h=C\rho_{\rm MrPro}\), which would be tautological. The common principle is conditional signal divided by interference; the inputs and admissible variables differ by regime.

This also explains why a pure mixed-mode projector is promising but insufficient. Exact retiming of a role-qualified relation \(R\nu=R\omega/S\) can preserve orthogonal carriers with minimal displacement, but the relation must first be identified by signed source-versus-distractor contribution. A large harmonic coefficient or attractive beat period does not establish its role. The margin rule supplies that missing selection criterion; projector retiming is one feasible parameterization after the role is established.

## Decision contract for Qwen2.5-3B, 32K→128K

1. Freeze the Qwen checkpoint, tokenizer, native/MrPro/P2/Smooth tables, prompts, source/distractor labels, decoding, and full state provenance.
2. On independent calibration rows, record pre-RoPE Q/K and compute exact signed margins for the intended source against every material competitor, split into native and far operations. Include covariance across slots; projection-weight Frobenius norms are not a substitute.
3. Before optimizing, test whether the standardized-margin statistic ranks the existing behavioral contrast: it must reject Smooth as a universal improvement and expose the P2 short/long tradeoff. Failure means the measurement contract is wrong; stop.
4. Starting from MrPro or the evidence-supported feasible face, solve one maximin signed-margin allocation with native probability constraints. Do not sweep unrelated curves or choose a slot from observed 128K answers.
5. Certify the exact finite table: ordering/endpoints, native constraints, far margins, and full trigonometric replay. Then compare the one intervention with an equal-size control that preserves the selected relation (or its matched mirror), plus the reused baselines.
6. Test the requested generated long-context tasks and native retention. Attention source-selection is mechanism evidence; generated correctness remains the decision endpoint.

No supplied artifact contains the role-conditioned moments needed to emit a defensible new Qwen table. The constructive result is the optimizer and its falsification contract, not a claimed winning allocation.

## Evidence boundary

The dialogue file was read as 312 JSONL records. Its archived tool outputs were inspected only where the assigned project artifacts independently exposed the quantitative claim; no unviewed tool-output record is labeled as read. All assigned project and external handoff files are listed with byte count, line count, and SHA-256 in the accompanying coverage receipt. The failure transcript contains exposed credentials and a public key; neither is reproduced here.
