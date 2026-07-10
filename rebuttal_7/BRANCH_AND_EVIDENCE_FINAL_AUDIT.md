# Final branch, evidence, and July 22 readiness audit

**Audit date:** 2026-07-10
**Mode:** repository/evidence consolidation only. No new experiments were run.

## Verdict

`main` is now the complete integration branch for every currently visible local and `origin/*` line of work. Every audited ref is an ancestor of `main` and has zero commits in `main..ref`. The final recovery-branch merge records ancestry with the `ours` strategy because its one unique commit had no branch-only path and its useful content was already present or restored in a newer form on `main`.

`rebuttal_7/` is sufficient as the command center for the July 22 response under the current evidence boundary. It indexes both independent committee simulations (18 Fable5 questions plus 17 GPT Pro questions), provides the formal four-question response draft, preserves raw-backed AR exact-match evidence, and records every remaining control as a limitation or placeholder rather than a completed result. Package status correctly remains `draft_with_placeholders`; “battle-ready” does not mean every requested new experiment exists.

## Branch audit

| Ref family | Final relation to `main` | Commits absent from `main` |
| --- | --- | ---: |
| `backup/2026-03-06` and `origin/backup/2026-03-06` | ancestor | 0 |
| `origin/codex/llama8b-positional-distill-pilot` | ancestor | 0 |
| `origin/codex/neurips-2026-submission-gates` | ancestor | 0 |
| `origin/codex/neurips2026-readiness` | ancestor | 0 |
| `origin/codex/rebuttal-7-fable5-fixes-20260710` | ancestor | 0 |
| `origin/codex/rebuttal-evidence-reconciliation-20260710` | ancestor | 0 |
| `origin/codex/rebuttal-evidence-recovery` | ancestor | 0 |
| `origin/codex/rebuttal-path-b-prep-20260610` | ancestor | 0 |
| `origin/main` | ancestor before final push | 0 |

The recovery branch's unique commit was `663cae0` (`prepare asset-grounded rebuttal`). Tree comparison found no path that existed only on that branch. Its missing reviewer-useful supplement entries were restored on `main`; the obsolete core-text passkey evaluator was not reintroduced because `scripts/supporting_eval/eval_passkey_scratch.py` is the canonical helper.

The audit also found a semantic merge regression that ancestry checks alone could not detect: a prior conflict had kept older LoRA, positional-distillation, MLA evaluator, and continued-pretraining files while retaining newer callers/tests. The 24 hardened files shared identically by the integrated rebuttal/positional-distillation branches were restored. This reestablished native geometric controls, frequency-artifact validation, anonymous provenance, RULER variant identity, model-geometry inference, and checkpoint-loaded frequency checks. The corrected playbook and supplement `trace-only` gate were restored from the same validated branch state.

## Curated and ignored evidence

- All 14 tracked `data/curated/` core files remain present: 13 JSON files plus the 99-run CSV. No curated file was deleted or replaced by a cleanup branch.
- `results/` still contains 368 files: 38 tracked and 330 ignored. This audit made no `results/` change.
- The tracked result set includes the Qwen 21-task outputs; they remain supporting/negative evidence rather than a promoted primary claim.
- `results/eval_3seeds_full_results.json` remains byte-identical to the tracked curated MLA evaluator payload.
- `results/350m_mla32_results_final.json` remains local and ignored. It duplicates the seed-42 portion of Primary III and does not close a missing control or checkpoint-provenance gate.
- The old QuALITY `n=200` pilot, three machine-specific Phase 18/19/22--23 reports, incomplete attention-prior traces, and invalid legacy LoRA outputs were explicitly reviewed and not promoted. Full reasons are in `IGNORED_ASSET_RECONCILIATION.md`.

The promotion decision is therefore complete for this checkout: every high-value ignored candidate is either represented by a smaller tracked, provenance-labelled artifact or is excluded for a concrete protocol, anonymity, duplication, or integrity reason.

## July 22 response coverage

The authoritative path is:

1. `README.md` for the response order and current readiness;
2. `FABLE5_RESPONSE_AND_FIX_LEDGER.md` for all 18 Fable5 questions;
3. `GPT_PRO_RESPONSE_CROSSWALK.md` for all 17 GPT Pro questions;
4. `IGNORED_ASSET_RECONCILIATION.md` for evidence promotion/exclusion;
5. `rebuttal/REBUTTAL_RESPONSE_DRAFT.md` for formal responses to the four highest-impact questions.

The strongest closure is the raw-backed 8K autoregressive result: Geo+YaRN retains 61.3% teacher-forced NLL-gap retrieval but records 0.0% AR exact in all three seeds; EVQ+YaRN reaches 58.0% mean AR exact (58/18/98%). The package also makes the QuALITY `n=2086` correction, surrogate-versus-scale separation, and high-base finite-channel scope explicit.

The following remain honest limitations, not missing organizational work: matched Geo/DAPE replication for Primary II, a fully tuned geometric-base or YaRN leaderboard, the direct MLA `d_eff`/tau convention ablation, trained `L_eff^J`/Fisher-forcing measurements, alternative non-geometric trained shapes, and a corrected matched LoRA empirical rerun.

## Paper experiment code workspace

`paper_experiments/` is a new manifest-driven access layer covering 94 canonical files in 10 experiment families: shared RoPE/training, Primary I--III, theory/mechanism, supporting text, video DiT, 8B LoRA, data preparation, and paper figures.

The workspace uses repository-relative symbolic links rather than copied source. `MANIFEST.json` records every source path, family membership, and SHA-256 hash; tests require every link to resolve inside the repository and match its canonical file. Results, checkpoints, ignored archives, and obsolete evaluators are excluded. `scripts/build_paper_experiment_workspace.py` rebuilds the index deterministically.

## Verification record

- `.venv/bin/python -m pytest -q`: **218 passed**.
- `.venv/bin/python scripts/validate_rebuttal_evidence_bundle.py`: **PASS**.
- Curated JSON parse gate: **13/13 valid**; tracked curated total remains **14** including the Phase16 CSV.
- Relevant Python compilation gate: **PASS**.
- `scripts/package_supplement.py` plus `unzip -t`: **PASS**; leak and `trace-only` gates active.
- Staged added-line anonymity scan: **0 matches**.
- Paper experiment workspace: **94/94 links resolved and hash-matched**.
- Branch ancestry audit: **all audited refs are ancestors; zero missing commits**.
- Cache cleanup: repository-local `.pytest_cache`, `.DS_Store`, non-environment `__pycache__`, and `.pyc` artifacts removed; experiment results untouched.

No TeX source changed in this consolidation, so the paper PDF was not rebuilt. No training, checkpoint evaluation, or new numerical experiment was performed.
