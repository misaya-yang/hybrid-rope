# Handoff Report: Experimental Audit of Zero-Training RoPE Retrofit

- **Agent:** Experimental Auditor (R3)
- **Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_auditor_r3_expertauditor_1`
- **Handoff Type:** Hard (Task complete)
- **Target:** Parent Agent (`f5123604-2261-4239-b583-f59569deb57e`)

---

## 1. Observation

1. **$S=2/4/8$ Behavior Across Scales:**
   - In `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` (lines 74–75, 113–118, 130–142, 298–310):
     - At $s=2$, OLMo 1x PG-19 PPL retention is `0.983986`, downstream macro retention is `1.041220`, and 2x NLL is `2.970433` (vs Native `7.100855`).
     - At $s=4$, log-s4 with $c=0.074$ achieves 1x PG-19 PPL retention of `0.875302`, 5-task macro retention of `0.915103`, 4x PG-19 NLL of `3.081946` (vs Native `7.205538`), and 16K RULER-13 of `0.49859` (vs Native `0.00385`).
     - At $s=8$, arithmetic core-4 is `0.2100` and log core-4 is `0.3025`. Full RULER-13 triggered early stop because `single_key_3` collapsed to `0.00 / 0.20 / 0.25` across 4K/8K/16K.

2. **Ordered Rotary Coupling vs. Multiset Permutation:**
   - In `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` (lines 185–189, 271–279):
     - Permuting final interior frequencies while preserving multiset, endpoints, and gain on OLMo exploded 1x PG-19 NLL from `3.104234` to `6.864926` ($\Delta\text{NLL} = +3.760692$).
     - Permuting interior rotary slot assignments while preserving multiset on Qwen 64K collapsed all four task cells to `0.0000` (destroying the reference macro of `0.7000`).

3. **Knife-Edge 1x Retention Gate:**
   - In `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` (line 239) and `paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md` (lines 58, 141):
     - OLMo arithmetic-s4 reached PPL retention `0.869584` ($< 0.875$, failed gate).
     - OLMo two-parameter $C2$ law reached PPL retention `0.870971` ($< 0.875$, failed gate), despite reconstruction MAE of `0.001223`.
     - Qwen 32K $C2$ reached macro retention `0.868902` ($< 0.875$, failed gate).

4. **Likelihood vs. Generation QA Disconnect:**
   - In `paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` (lines 162–171, 182–191):
     - Far source-use is $+1.438$ (95% CI $[0.720, 2.230]$); far canonical answer NLL drops from `5.123` to `3.695`.
     - Yet 30-row natural QA macro was Native `0.13229` vs Index `0.10174` (delta $-0.03055$, CI $[-0.1168, +0.0486]$).
   - In `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` (lines 35–42):
     - HotpotQA 16K (38 Native-short-correct rows): Native EM/F1 `0/0`, log-s4 `0.079 / 0.153`, YaRN-4 `0.395 / 0.500`. Correct EOS was Native `0/38`, log-s4 `22/38`, YaRN `36/38`.
   - In `paper-2027/research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md` (lines 87–91):
     - At 64K: SQuAD QA (Index `0.150` vs YaRN `0.200`); HotpotQA (Native `0.250`, Index `0.150`, YaRN `0.200`).

5. **Gain Scaling and Attention Normalization:**
   - In `paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` (lines 30–32, 118, 137, 149–151):
     - 512 free headwise gain parameters reduced training loss from `3.5612` to `2.3105`, but dropped HotpotQA F1 from `0.24237` to `0.19439`, crashed EOS from `178/200` to `108/200`, and doubled average generated length from `9.10` to `18.50` tokens (92/200 hitting maximum generation limit).

6. **Basin Barrier in Headwise Spatial Factorization:**
   - In `HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` (lines 25–29, 133–140, 176–177):
     - Log-start two-axis reached HotpotQA F1 `0.25223` but failed Native retention (`0.7714 < 0.875`).
     - Native-start two-axis achieved Native retention `1.0462` but failed long QA completely (HotpotQA F1 `0.02560`, 8/200 EOS).

7. **Table-Weight Co-Adaptation Crossings:**
   - In `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` (lines 364–368, 388–401):
     - 50M TinyStories crossing showed table $\times$ weights interaction of `-3.5367` (5.9x larger than main effects).
   - In `paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md` (lines 205–216):
     - 151.9M crossing showed FMRoPE weights prefer FMRoPE table (`3.426` vs `5.776` NLL); Cosh weights prefer Cosh table (`3.479` vs `4.455` NLL).

8. **2026-09-02 Remote Data Loss Incident:**
   - In `INDEX.md` (line 114) and `paper-2027/HANDOFF.md` (lines 67–70):
     - Zero raw JSON/JSONL files exist in the repository for 2026-09-02 Qwen runs. Parent hashes are recorded in markdown summaries, but remote server shutdown occurred before file retrieval.

---

## 2. Logic Chain

1. From Observation 2, permuting rotary slots while preserving the frequency multiset collapses model performance to zero, demonstrating that Transformer representations are wired to specific ordered rotary pairs.
2. From Observation 7 and Theorem T2 (Transplant Rigidity), position-independent linear transformations cannot compensate for unequal frequency multisets, establishing that pretrained weights are fundamentally co-adapted to their training spectrum.
3. From Observation 1, zero-training static tables provide useful extrapolation up to $S \le 4$, but fail catastrophically at $S=8$. This is explained by Theorem T7: novel phase volume $N_k(S)$ super-doubles as $S$ increases, causing unresolvable attention noise beyond a finite scale horizon.
4. From Observation 3, the Native retention gate ($0.875$) is a knife-edge boundary that cannot be predicted by geometric distance metrics (MAE/RMSE), meaning any static frequency shift risks catastrophic in-window degradation.
5. From Observation 4 and Theorem T8 (Pillar 2), teacher-forced answer likelihood is smooth, whereas autoregressive token generation is piecewise-constant. An $O(1)$ cumulative answer logit gain translates to only $O(0.05 - 0.1)$ nats per token, which is insufficient to overcome $O(1)$ competitor margins on visited greedy paths. Consequently, long NLL improvements do not convert to natural multi-hop QA or EOS emission.
6. From Observation 5, gain scaling acts strictly as an attention temperature multiplier that preserves key argmax. It provides an in-distribution sharpening shortcut under teacher forcing, but ruins autoregressive generation by amplifying incorrect leading tokens.
7. From Observation 6, the Native-start and long-start optimizations inhabit disconnected basins separated by an unbridgeable ridge: Native-start suffers from gradient starvation on aliased slots, while log-start breaks Native retention.
8. Therefore, from Steps 1–7, the failure of mature-checkpoint zero-training retrofit is not an optimization artifact or a curve-family defect, but a **fundamental structural constraint** arising from weight-table co-adaptation, off-arc phase novelty, and piecewise-constant winner-margin dynamics.
9. From Observation 8, all 2026-09-02 Qwen numbers lack repository raw artifacts and must be held strictly as internal decision evidence, labeled `UNSUPPORTED BY REPOSITORY EVIDENCE`.

---

## 3. Caveats

- **Sample Size of Specific Panels:** The far-evidence natural QA panel was conducted on 30 rows, and the HotpotQA Fact D panel evaluated 38 Native-short-correct rows. While paired bootstrap intervals confirm directional collapse, sample variance on these specific subsets is larger than on 520-row RULER matrices.
- **Model Diversity:** Deep causal evaluations focus primarily on OLMo-2-1B, Qwen2.5-1.5B/0.5B, and Gemma-1.1-2B. Larger scales (e.g. 70B+) were not evaluated under zero-training retrofit.
- **Dynamic Session Routing:** These findings govern static single-table deployments. Dynamic serving architectures that route short requests ($\le L_{native}$) to Native RoPE bypass the in-window retention gate by engineering fiat, but do not solve the static one-table problem.

---

## 4. Conclusion

1. **Empirical Status:** The ambition to achieve a single static, zero-training RoPE table that simultaneously preserves 1x Native capability and enables long-context natural multi-hop QA on mature checkpoints is **structurally obstructed and empirically falsified**.
2. **Established Frontier:** Normalized-index static tables succeed as effective engineering solutions for synthetic long retrieval (RULER) and long natural perplexity up to $S \le 4$.
3. **Evidence Ceilings:** The 2026-09-02 Qwen natural QA, NLL, and gain sweep numbers must remain internal and cannot be promoted to submission claims due to unrecovered raw artifacts.

---

## 5. Verification Method

- **Inspect Analysis Document:** View `.agents/teamwork_preview_auditor_r3_expertauditor_1/analysis.md` for complete citations and derivations.
- **Verify Clean Git Status:**
  ```bash
  git status --porcelain
  ```
  Confirms zero modifications to source code, configs, or papers.
- **Verify Missing Raw JSONs:**
  ```bash
  find paper-2027/research/attention-aware-retrofit/evidence/ -name "*20260902*"
  ```
  Returns 0 files, confirming the data gap.
- **Verify Key Citations:**
  - `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`: lines 74–75, 130–142, 185–189, 271–279.
  - `HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`: lines 25–32, 118–120, 137, 175–177.
  - `ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`: lines 147–172, 182–205.
  - `FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`: lines 364–368, 388–401.
