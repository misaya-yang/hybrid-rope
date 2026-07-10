# Fable5 + GPT-5.6 Audit: Repository-Asset Response Matrix

日期：2026-07-10

目标：针对两轮审计逐项判断“仓库已有证据能否回答”，形成 rebuttal 解决方案。本文不修改已提交 PDF，也不把部分历史 raw artifact 缺失误写成实验不可复现。

## 0. Overall verdict

两轮审计真正集中在四类问题：

1. **报告一致性**：Figure 8/Table 21、Figure 9/Table 20；
2. **理论 epistemic boundary**：cosh shape、tau scale、surrogate、pure-tether、waterbed；
3. **baseline / replication**：base、YaRN scale、Primary II seeds、MLA channel/tau；
4. **可复现与应用边界**：配置、PK metric、1B row、LoRA、系统收益。

仓库现有资产足以做到：

- 不撤回三个核心 empirical conclusions；
- 解释并披露两个 stale figure 和一个 Table 21 erratum；
- 用 99-run、base-10K pilot、learnable-tau 记录、MLA channel-count pilot、Primary-I seedwise data 回答审稿人认为“完全缺失”的多项证据；
- 给出完整 rerun path，说明历史原始件不完整不等于不可复现。

仍需未来重跑才能升级的，是 tuned-YaRN leaderboard、Primary-II full 3-seed table、MLA direct tau convention ablation 和 trained `L_eff^J/R_F` measurement。Primary-I AR exact 已从 tracked raw payload 中恢复，不再属于待重跑项。这些限制主张强度，但不推翻已报告的 matched-scale、seed-scoped 或 scarce-channel 结论。

## 1. What remains defended

| Core claim | Why it survives the audits | Primary repository evidence |
| --- | --- | --- |
| Frequency allocation is a third RoPE design axis | The intervention changes only inverse frequencies; the exact conditional minimizer and inverse CDF are unchanged by reporting issues | `paper/sections/03_theory.tex`; `scripts/lib/rope/schedules.py`; `tests/test_rope_core.py` |
| EVQ increases YaRN leverage under the same tested scale | 10% primary row is 3-seed; EVQ+YaRN is 100% at 8K for all three seeds versus Geo+YaRN 58/62/64% | `data/curated/table2_evq_yarn_454m_passkey_10pct.json` |
| PE-dominant allocation-shape result holds in the tested protocol | Table 4 is explicitly seed 42 for the printed Geo/DAPE/EVQ contrast; learned tau is 3-seed; fixed tau=5 also has a 3-seed 8K aggregate (335.7±1.7); a separate 3-seed L=256 sweep supports the operating basin | `paper/tables/table4_pe_dominant.tex`; `data/curated/learnable_tau_128tok_evidence.json`; `results/core_text/phase11/results_phase11_yarn.json` |
| MLA scarce-channel anchor remains valid | The 432M result is 3-seed mean/std with a large 16K effect; a separate within-MLA d_rope=32/16 pilot supports the channel-scarcity direction | `data/curated/table18_mla_3seed_aggregate.json`; `data/curated/mla_channel_count_125m_pilot.json` |
| `tau=d_head/sqrt(L)` is a near-optimal default/basin selector | 99 planned runs: exact best 3/9, top-2 6/9, top-3 8/9, every optimum within 1.5x | `data/curated/phase16_99run_manifest.csv`; `docs/exp/2026-03-09_phase16_formula_optimality_sweep_results.md` |

## 2. R1: Theory audit

| Reviewer issue | Correct judgment | Existing answer / asset | Rebuttal action |
| --- | --- | --- | --- |
| Shape and scale do not come from one objective | **Already disclosed, not a fatal error.** The paper explicitly separates the surrogate-derived cosh family from the post-softmax operating-point selector. | `paper/sections/03_theory.tex` opening and “Shape and scale come from separate layers.” | Quote this distinction directly. Do not claim a unified optimum; do not concede the cosh derivation itself. |
| Surrogate is not exact RoPE kernel | **True approximation, functionally tested.** The claim is conditional; exact-kernel collision reductions are 24–92% over 12 configurations. | `paper/sections/03_theory.tex`; Appendix surrogate-validation tables | Emphasize functional validation and trained-model consequences. Residual/prior sensitivity is future strengthening, not a refutation of the conditional theorem. |
| `L^-1/2` depends on diffuse/Pearson choices; 0.465 vs 0.500 | **Operating-law identity is not the claim; basin membership is.** | 99-run asset gives top-3 8/9 and all optima within 1.5x; paper labels prefactor empirical. | Lead with empirical basin result. Say the exponent is structural within the stated model and the deployed value is a robust default, not unique global optimum. |
| Pure-tether at tau=4 is not quantitatively controlled | **A missing mechanism diagnostic, not a failed empirical result.** Exact cosh allocation is used in training; the Taylor truncation is not used to generate frequencies. | `scripts/lib/rope/schedules.py`; paper explicitly says forced residual is nonzero | Clarify that empirical PPL/PK results do not assume the Taylor approximation is numerically exact at tau=4. Keep `R_F` as a future falsification test. |
| Waterbed inequality does not prove PPL tradeoff | **Correct interpretation boundary.** It proves an allocation-divergence bound; PPL tradeoff is empirical. | `paper/sections/03_theory.tex`; multiscale PPL table | Use “consistent with” for PPL, while defending the mathematical bound on its own terms. |
| Realistic attention-distance prior is untested | **Not closed by the current reviewer-grade assets.** Local extraction code and partial traces exist, but their summaries are internally inconsistent and the cited canonical result JSON is absent. | `scripts/m4_max_36gb/test3_attention_prior.py`; ignored local traces only | State that the theorem is conditional and functional exact-kernel validation is the current check. Do not present the local prior traces as a completed empirical-prior result. |
| Global allocation vs per-head specialization | **Valid extension, not a contradiction.** A global initializer and learned per-head dynamics act at different stages. | `results/attention_viz/attention_stats.npz`; paper Fig. 6 | Defend stage distinctness; per-head tau/CARoPE composition remains future work. |

### R1 ready paragraph

> We agree that shape and scale occupy different epistemic layers, and this separation is already explicit in Sec. 3: the cosh family is the exact minimizer of the stated convex broadband surrogate, while tau is a post-softmax operating-point selector rather than a globally optimal transformer parameter. The practical claim is therefore not identity of a unique exponent or prefactor, but basin membership. Our 99-run sweep across nine (L,H,d_head) settings places the rule exactly first in 3/9, top-2 in 6/9, top-3 in 8/9, with every empirical optimum within 1.5x of the prediction. The exact-kernel functional check (24–92% collision-score reduction over 12 configurations) and trained PPL/PK results test the consequences beyond the surrogate. The pure-tether and diffuse-attention assumptions remain falsifiable mechanism approximations; they do not alter the exact inverse-CDF implementation used in the experiments.

## 3. R2: Empirical and reproducibility audit

| Reviewer issue | Correct judgment | Existing asset | Rebuttal action |
| --- | --- | --- | --- |
| Text effect may exist only at base 500K | **Existing pilot refutes “only at 500K.”** At base 10K, EVQ improves PPL by -11.59% @1K, -22.28% @2K, -21.83% @4K; at 500K the corresponding gains are -14.39%, -28.67%, -32.64%. | `data/curated/text_base_10k_500k_pilot.json` | Report as a 151.9M, L=512, 50M-token, seed-42 supporting pilot. Do not call it a complete tuned-base sweep. |
| Geo/EVQ tuned YaRN scale missing | **Not closed.** Primary I is intentionally matched-scale, not best-tuned dominance. | Table 2 caption; `phase14c_multiscale_evq_yarn.py` supporting runner | Defend the factorial matched-scale question. Promise no tuned-leaderboard conclusion. A scale sweep remains an optional rerun. |
| Primary II is seed 42 | **Correctly disclosed; result is protocol-scoped, not invalid.** Learned tau is 3-seed; fixed tau=5 has a 3-seed PPL@8K aggregate of 335.7±1.7; L=256 supporting curves are also 3-seed. Geo/DAPE still need matched replication. | Table 4; `learnable_tau_128tok_evidence.json`; Table 22 | Keep exact statement: EVQ beats Geo/DAPE in the tested seed-42 protocol, while replicated EVQ rows show the effect is not a one-seed EVQ anomaly. Do not claim all methods are already matched 3-seed. |
| Why learnable tau stops at 1.14 | **Already answered by existing training evidence.** Tau converges to 1.1406±0.0034 across 3 seeds; PPL@128 is nearly flat while 8K improves at larger fixed tau. | `data/curated/learnable_tau_128tok_evidence.json` | Explain objective mismatch: in-range loss cannot observe out-of-range utility. This converts the negative baseline into mechanism evidence without inventing a trajectory. |
| MLA channel-scarcity mechanism is not isolated | **A supporting within-architecture pilot already exists.** MLA d_rope 32: -6.3/-9.1% at 8K/16K; d_rope 16: -47.8/-47.9%, with a weak baseline caveat. | `data/curated/mla_channel_count_125m_pilot.json` | Use only as qualitative support. The 432M 3-seed anchor remains primary; direct tau=d_rope/sqrt(L) is still a separate ablation. |
| Figure 8/Table 21 inconsistent | **Real presentation error; underlying NLL conclusion survives.** | `data/curated/quality_454m_full_eval.json`; `rebuttal/FIGURE_TABLE_AUDIT.md` | Disclose n=200 stale panel and 26.6→24.6 erratum. State NLL values unchanged and use n=2086 aggregate as source of truth. |
| Figure 9/Table 20 inconsistent | **Real stale visualization; qualitative multiscale direction survives.** | `paper/tables/table1_multiscale_raw_ppl.tex`; `rebuttal/FIGURE_TABLE_AUDIT.md` | Correct 454M interpretation to -13.3%; -81.2% belongs to progressive training. Keep Table 20 as heterogeneous supporting consistency, not controlled scaling law. |
| Statistical treatment for Primary I | **Seedwise evidence is already available.** EVQ+YaRN exceeds Geo+YaRN at 8K by +38/+42/+36 pp; EVQ raw exceeds Geo raw by +14/+8/+16 pp. | `data/curated/table2_evq_yarn_454m_passkey_10pct.json` | Report minimum paired direction, not a p-value with n=3. Use mean±std already printed. |
| PK is not AR exact | **Metric distinction retained; AR endpoint recovered separately.** At 8K, Geo+YaRN has 61.3% TF retrieval but 0.0% AR exact in all seeds; EVQ+YaRN has 58.0% mean AR exact (58/18/98%). | `data/curated/primary1_evq_yarn_10pct_raw.json` | Report TF NLL-gap retrieval and AR exact side by side. Preserve the wide seed range and do not relabel either endpoint. |
| 1B reversal means benefit vanishes with training | **Reviewer comparison is confounded.** The 1B run changes L_train from 8K to 4K and data/schedule, and is single seed; it is not a token-only continuation of the primary 500M/8K 3-seed anchor. | `paper/REBUTTAL_PLAYBOOK.md`; `scripts/core_text_phases/run_350m_4k_1b.sh` | Defend the primary conclusion. Present the row as schedule sensitivity and note EVQ+YaRN+FT remains -2.5%; do not call it saturation proof. |
| Manuscript/repo is not reproducible | **Overstated.** Architecture/config tables, locked requirements, canonical schedule, training/eval entrypoints, data-prep and curated expected outputs exist. Historical raw provenance is not required to rerun. | `docs/overview/REPRODUCE.md`; `docs/overview/DATA_PREPARATION.md`; `scripts/package_supplement.py`; code tests | Answer with exact rerun paths and expected directional gates. Do not volunteer machine-loss history in the rebuttal. |

### R2 ready paragraph

> The empirical claims remain scoped but reproducible. Primary I is a matched-scale 2x2 substrate/range test, not a best-tuned YaRN leaderboard; its 8K interaction is consistent in every seed (EVQ+YaRN minus Geo+YaRN: +38/+42/+36 pp). The PE-dominant Geo/DAPE/EVQ comparison is explicitly the retained seed-42 protocol; fixed tau=5 has a separate three-seed 8K aggregate of 335.7±1.7, learned tau is three-seed, and the L=256 operating basin is also multi-seed. Existing supporting assets address two further controls: a 151.9M text pilot at base 10K retains EVQ gains (-22.28% at 2K and -21.83% at 4K), and a within-MLA d_rope=32/16 pilot shows a stronger effect in the more channel-scarce setting, while the 432M three-seed MLA aggregate remains the primary anchor. These pilots do not replace a complete tuned-base or direct MLA-tau ablation, but they rule out the stronger interpretations that the effect exists only at base 500K or is a one-seed EVQ anomaly.

## 4. R3: Application, clarity, and impact audit

| Reviewer issue | Correct judgment | Existing asset | Rebuttal action |
| --- | --- | --- | --- |
| Downstream evidence weak | **Outside the primary claim by design.** The paper explicitly calls accuracy a non-regression check and uses NLL/PK/PPL as mechanism endpoints. | Sec. 4.1; QuALITY n=2086 aggregate; 750M AR exact supporting row | Do not sell QuALITY as SOTA. Defend practical relevance through zero-parameter initialization, matched-scale systems effect, and MLA. |
| When should practitioners use EVQ? | **Repo contains a usable boundary.** Larger benefit appears when a large base/dead-channel regime wastes spectral capacity; video/base and text base pilots show it is not tied to one base. | video base sweep; `text_base_10k_500k_pilot.json`; dead-channel audit | Recommend diagnose alive/dead channels; do not enable unconditionally or claim base-independent dominance. |
| Notation/exposition too complex | **Editorial issue, not scientific invalidity.** | existing epistemic map and evidence-tier table | In rebuttal, answer with one sentence: shape exact under surrogate; tau operating default; K counts rotary pairs; d_eff controls operating scale. |
| No measured compute saving | **Correct.** Zero learned parameters does not imply lower FLOPs. | method implementation | Claim implementation simplicity and no additional learned parameters, not measured compute/latency benefit. |
| Video 0.53 is post-hoc | **Already scoped as directional.** | Appendix B.5 explicitly compares m=1/m=2 and leaves derivation future | Point to existing candor; transfer evidence is the head-to-head/base sweep, not the exact 0.53 decomposition. |
| LoRA rank threshold/generalization | **Supporting hypothesis only.** | LoRA table and rank model | Do not use it to defend the core paper. Keep it model-specific until matched Geo+LoRA and another rank sweep exist. |

## 5. Reporting corrections that do not change conclusions

### Figure 8 / Table 21

Safe answer:

> The reviewer is correct that the submitted Figure 8 was stale: it plotted the superseded n=200 accuracy pilot under a Gold-NLL caption. The full n=2086 aggregate is the source of truth. It also reveals a transcription erratum in Table 21: 8K-raw Geo accuracy is 513/2086=24.59%, i.e. 24.6%, not 26.6%. The Gold-NLL values and their conclusion are unchanged (-30.1% at 8K raw and -21.4% at 16K raw). We will correct the figure and table entry in a revision.

### Figure 9 / Table 20

Safe answer:

> The -81.2% value belongs to a separate single-seed progressive-training experiment and should not have been used as the 454M point in Figure 9. The corresponding three-seed 454M FineWeb-Edu row in Table 20 is -13.3%. Correcting the plot changes the magnitude of that point, not the table’s qualitative observation that each individually scoped row shows a long-range improvement. We treat Table 20 as heterogeneous supporting evidence, not a controlled scaling law.

## 6. Reproducibility position

Do not say “the experiment cannot be reproduced because historical data were lost.” The accurate statement is:

> The repository provides the canonical schedule implementation, locked environment, data-preparation instructions, model configurations, training/evaluation entrypoints, curated expected aggregates, and smoke/unit tests. This is sufficient to rerun the reported protocols from public data. Curated aggregates provide an audit target; historical checkpoints are convenient provenance artifacts, not a logical prerequisite for reproduction from scratch.

Concrete paths:

- core schedule: `scripts/lib/rope/schedules.py`;
- main sweep: `scripts/core_text_phases/run_evq_sweep.py`;
- PE-dominant/DAPE: the exact L=128 protocol is documented in `docs/exp/2026-02-24_128tok_baseline_report.md`; `phase11b_125m_dape.py` is the separate L=256 multi-seed supporting runner;
- matched-scale supporting rerun: `scripts/core_text_phases/phase14c_multiscale_evq_yarn.py`;
- 99-run rule validation: `scripts/core_text_phases/phase16_formula_optimality_sweep.py`;
- QuALITY: `scripts/core_text_phases/phase21b_quality_eval_clean.py`;
- data and expected gates: `docs/overview/DATA_PREPARATION.md`, `docs/overview/REPRODUCE.md`;
- curated targets: `data/curated/*.json`, `data/curated/phase16_99run_manifest.csv`.

## 7. What still merits rerunning, without blocking rebuttal

| Priority | Rerun | Why | Existing entrypoint |
| --- | --- | --- | --- |
| P0 if compute is available | Exact L=128 Geo/DAPE matched seeds 137/256 | upgrades the remaining seed-scoped baseline contrast; fixed EVQ already has a 3-seed aggregate | reconstruct the documented L=128 protocol; do not mislabel the current L=256 `phase11b` runner as Table 4 |
| P1 | Geo/EVQ YaRN scale sweep | upgrades matched-scale conclusion to tuned-scale robustness | adapt existing YaRN evaluators / `phase14c_multiscale_evq_yarn.py` |
| Done | Primary I AR exact recovery | adds the generation endpoint without changing PK definition | recovered tracked Primary-I raw payload |
| P1 | MLA tau=d_rope/sqrt(L) | tests operating convention directly | `run_gqa_evq_experiment.py` with explicit tau |
| P2 | trained `L_eff^J` and `R_F` | strengthens theory mechanism rather than empirical conclusion | existing attention extraction/analysis scripts |

## 8. Fable5 final review checklist

- [ ] Do not equate partial historical artifact loss with non-reproducibility.
- [ ] Do not withdraw the three core conclusions; keep their stated matched-scale/seed/protocol scope.
- [ ] Use the base-10K pilot, learnable-tau record, MLA channel pilot, 99-run manifest, and Primary-I seedwise data.
- [ ] Admit Figure 8/9 errors and the 26.6→24.6 erratum without saying the reviewer misunderstood.
- [ ] Keep the Figure 8 NLL and Table 20 directional conclusions after correcting the visual provenance.
- [ ] Do not upgrade supporting pilots into tuned-baseline or multi-seed primary claims.
- [ ] Separate “can be rerun from code/public data” from “historical checkpoint remains locally archived.”
