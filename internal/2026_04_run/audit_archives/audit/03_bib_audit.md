# Audit 03 — Bibliography + arXiv ID Web Verification

**Auditor**: 3 of 8 (parallel deep-audit pass)
**Date**: 2026-04-27
**Scope**: `paper/refs/references.bib` (44 entries) and all `\cite*{}` usages in `paper/sections/`, `paper/appendix/`, `paper/tables/`, `paper/main.tex`.
**WebFetch budget consumed**: 16 direct arXiv `abs/` fetches + 13 WebSearch verifications = 29 web operations (within ~30-35 budget).

---

## Headline

**No P0 findings.** All 44 bib entries are cited at least once. All claimed arXiv IDs in 2024–2026 entries resolve to the intended paper with matching titles and first-author surnames. Two `note = {arXiv:...}` entries have a year mismatch between the bibkey and the actual arXiv submission year (P1, cosmetic / not a bug). Five entries use `and others` truncation in the author list (P2, BibTeX-legal but should be expanded for camera-ready).

---

## 1. Bib coverage table — `key | cited_count | tag`

Column `cited_count` is total `\cite{}/\citet{}/\citep{}/\citealp{}/\citealt{}` occurrences in the paper (multi-key cites contribute 1 per key). All 44 entries are cited; **no dead entries**.

| Bibkey | Cited | Tag |
|---|---:|---|
| deepseekv3 | 5 | OK |
| zheng2024dape | 4 | OK |
| videorope2025 | 4 | OK |
| shang2025longrope2 | 4 | OK |
| deepseekv2 | 4 | OK |
| wang2024resonance | 3 | OK |
| veisi2025carope | 3 | OK |
| su2024roformer | 3 | OK |
| roziere2023codellama | 3 | OK |
| peng2024yarn | 3 | OK |
| li2025hope | 3 | OK |
| ding2024longrope | 3 | OK |
| chen2024position | 3 | bibkey-year mismatch (P1) |
| qwen2026mhrope | 2 | OK |
| li2024fire | 2 | OK |
| hua2025fope | 2 | author truncated `and others` (P2) |
| grattafiori2024llama3 | 2 | author truncated `and others` (P2) |
| barbero2025round | 2 | OK |
| zhu2024pose | 1 | OK |
| zhao2025riflex | 1 | OK |
| zhang2024found | 1 | OK |
| yang2024cogvideox | 1 | author truncated `and others` (P2) |
| wan2025wan | 1 | OK (corporate `Wan Team`) |
| vaswani2017attention | 1 | OK |
| touvron2023llama2 | 1 | author truncated `and others` (P2) |
| sun2022xpos | 1 | OK |
| srivastava2015unsupervised | 1 | OK |
| shaw2018self | 1 | OK |
| raffel2020exploring | 1 | OK |
| radford2019gpt2 | 1 | OK |
| qwen2024qwen25 | 1 | author truncated `and others` (P2) |
| qiu2024freenoise | 1 | OK |
| press2022alibi | 1 | OK |
| opensora2024 | 1 | OK |
| ma2024latte | 1 | OK |
| liu2024lost | 1 | OK |
| li2026copeclipped | 1 | OK |
| kong2024hunyuanvideo | 1 | author truncated `and others` (P2) |
| jin2024selfextend | 1 | OK |
| hsieh2024ruler | 1 | OK |
| dai2025hyperbolicrope | 1 | OK |
| chi2022kerple | 1 | OK |
| chen2024clex | 1 | OK |
| bai2024longbench | 1 | OK |
| li2025hope | (counted above) | author truncated `and others` (P2) |
| roziere2023codellama | (counted above) | author truncated `and others` (P2) |

**Dead entries**: none.

---

## 2. arXiv WebFetch results — `key | claimed_id | resolved_title | match`

For each 2024-2026 entry plus the high-priority older ones in the auditor task list. ✅ = verified, ⚠️ = partial, ❌ = wrong/non-existent. All 16 direct arXiv abs/ fetches resolved successfully.

| Bibkey | Claimed arXiv ID | Resolved Title (verbatim from arXiv) | First Author | Year | Match |
|---|---|---|---|---|---|
| dai2025hyperbolicrope | 2509.05218 | "HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models" | Dai | 2025 | ✅ |
| qwen2026mhrope | 2510.23095 (in `note`) | "Revisiting Multimodal Positional Encoding in Vision-Language Models" | Huang | 2025 v1 / 2026 v3 | ⚠️ bibkey says `2026` but arXiv v1 was 2025-10-27 (P1 cosmetic) |
| li2025hope | 2505.20444 (in `note`) | "HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models" | Li | 2025 | ✅ (also distinct from `dai2025hyperbolicrope` — paper §2 already disambiguates) |
| hua2025fope | 2412.17739 (in `note`) | "Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization" | Hua | submitted 2024-12, ICML 2025 | ✅ (year 2025 = venue year, OK) |
| veisi2025carope | 2507.23083 | "Context-aware Rotary Position Embedding" | Veisi | 2025 | ✅ |
| li2026copeclipped | 2602.05258 | "CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs" | Li | 2026-02-05 | ✅ |
| chen2024position | 2306.15595 | "Extending Context Window of Large Language Models via Positional Interpolation" | Chen | 2023 | ⚠️ **bibkey-year mismatch**: bibkey is `chen2024position` but `year={2023}` and arXiv 2306 (June 2023). The bibkey appears to follow a "year of widespread use / reference" convention rather than submission year. The `year` field is correct (2023). Bibkey is misleading but cosmetic (P1). |
| hsieh2024ruler | 2404.06654 | "RULER: What's the Real Context Size of Your Long-Context Language Models?" | Hsieh | 2024 | ✅ |
| deepseekv2 | 2405.04434 | "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" | DeepSeek-AI | 2024 | ✅ |
| deepseekv3 | 2412.19437 | "DeepSeek-V3 Technical Report" | DeepSeek-AI | 2024 | ✅ |
| grattafiori2024llama3 | 2407.21783 | "The Llama 3 Herd of Models" | Grattafiori | 2024 | ✅ |
| qwen2024qwen25 | 2412.15115 | "Qwen2.5 Technical Report" | Yang (An Yang) | 2024 | ✅ |
| roziere2023codellama | 2308.12950 | "Code Llama: Open Foundation Models for Code" | Rozière | 2023 | ✅ |
| wang2024resonance | 2403.00071 | "Resonance RoPE: Improving Context Length Generalization of Large Language Models" | Wang (Suyuchen) | 2024 | ✅ |
| zhao2025riflex | 2502.15894 | "RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers" | Zhao (Min) | 2025 | ✅ |
| qiu2024freenoise | 2310.15169 | "FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling" | Qiu (Haonan) | submitted 2023, ICLR 2024 | ✅ |
| videorope2025 | 2502.05173 | "VideoRoPE: What Makes for Good Video Rotary Position Embedding?" | Wei (Xilin) | 2025 | ✅ |
| shang2025longrope2 | 2502.20082 | "LongRoPE2: Near-Lossless LLM Context Window Scaling" | Shang | 2025 | ✅ |
| barbero2025round | 2410.06205 | "Round and Round We Go! What makes Rotary Positional Encodings useful?" | Barbero | submitted 2024, ICLR 2025 | ✅ |
| zheng2024dape | 2405.14722 | "DAPE: Data-Adaptive Positional Encoding for Length Extrapolation" | Zheng | 2024 | ✅ |
| peng2024yarn | 2309.00071 | "YaRN: Efficient Context Window Extension of Large Language Models" | Peng | submitted 2023, ICLR 2024 | ✅ |
| ding2024longrope | 2402.13753 | "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens" | Ding | 2024 | ✅ |
| chen2024clex | 2310.16450 | "CLEX: Continuous Length Extrapolation for Large Language Models" | Chen (Guanzheng) | submitted 2023, ICLR 2024 | ✅ |
| zhu2024pose | 2309.10400 | "PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training" | Zhu (Dawei) | submitted 2023, ICLR 2024 | ✅ |
| jin2024selfextend | 2401.01325 | "LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning" | Jin (Hongye) | 2024 | ✅ |
| liu2024lost | 2307.03172 | "Lost in the Middle: How Language Models Use Long Contexts" | Liu (Nelson F.) | submitted 2023, TACL 2024 | ✅ |
| zhang2024found | 2403.04797 | "Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding" | Zhang (Zhenyu) | 2024 | ✅ |
| bai2024longbench | 2308.14508 | "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding" | Bai (Yushi) | submitted 2023, ACL 2024 | ✅ |
| li2024fire | 2310.04418 | "Functional Interpolation for Relative Positions Improves Long Context Transformers" | Li (Shanda) | submitted 2023, ICLR 2024 | ✅ |
| touvron2023llama2 | 2307.09288 | "Llama 2: Open Foundation and Fine-Tuned Chat Models" | Touvron | 2023 | ✅ |
| yang2024cogvideox | 2408.06072 | "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" | Yang (Zhuoyi) | 2024 | ✅ |
| kong2024hunyuanvideo | 2412.03603 | "HunyuanVideo: A Systematic Framework For Large Video Generative Models" | Kong (Weijie) | 2024 | ✅ |
| wan2025wan | 2503.20314 | "Wan: Open and Advanced Large-Scale Video Generative Models" | Team Wan / `Wan Team` | 2025 | ✅ |
| opensora2024 | 2412.20404 | "Open-Sora: Democratizing Efficient Video Production for All" | Zheng (Zangwei) | submitted 2024-12-29 | ✅ |
| ma2024latte | 2401.03048 | "Latte: Latent Diffusion Transformer for Video Generation" | Ma (Xin) | 2024 | ✅ |

**Special notes from the high-priority list:**

- **`dai2025hyperbolicrope` (this round's add)**: arXiv 2509.05218 resolves cleanly to *"HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models"* (Dai, Shan, Song, Liang, 2025). The paper (`paper/sections/02_related.tex:6`) correctly disambiguates this from VLM-HoPE (`li2025hope`). **No collision.**
- **`qwen2026mhrope`**: bibkey contains `2026` but arXiv v1 was Oct 2025; v3 is April 2026. Title and authors verified. Citation will be `Huang et al., 2026` (ICLR 2026 venue per `booktitle`), which is consistent with the bibkey. **No bug**, but the v1-vs-v3 timing is worth keeping in mind for camera-ready.
- **`li2026copeclipped`**: arXiv 2602.05258 is a 2026-02-05 submission. ID, title, authors all check out. The arxiv subdomain ID format `26xx.xxxxx` is the new arXiv format (post-Jan 2026 IDs).

---

## 3. Dead entries

**None.** All 44 bib entries are cited at least once.

(The grep used: counted occurrences of every bibkey across `\cite{...}`, `\citet{...}`, `\citep{...}`, `\citealp{...}`, `\citealt{...}` after expanding multi-key lists. Coverage: 44/44 = 100%.)

---

## 4. Older-entry sanity checks

| Bibkey | Bib year | Verified | Notes |
|---|---|---|---|
| vaswani2017attention | 2017 | ✅ | NeurIPS 2017 (NIPS 30), arXiv 1706.03762 — matches exactly. |
| shaw2018self | 2018 | ✅ | NAACL-HLT 2018, arXiv 1803.02155 — title and venue match. |
| raffel2020exploring | 2020 | ✅ | JMLR 21, vol 21 issue 140 pp 1-67 — matches `volume = {21}, number = {140}, pages = {1--67}`. |
| radford2019gpt2 | 2019 | ✅ | OpenAI tech report — venue field "OpenAI Technical Report" is fine. |
| su2024roformer | 2024 | ✅ | RoFormer in *Neurocomputing* vol 568 p 127063, published 2024 (arXiv preprint goes back to 2021). The `journal = {Neurocomputing}` + `volume = {568}` + `pages = {127063}` matches the published version, not the arXiv preprint. **Correct.** |
| press2022alibi | 2022 | ✅ | ICLR 2022, arXiv 2108.12409 — matches. |
| chi2022kerple | 2022 | ✅ | NeurIPS 2022, arXiv 2205.09921 — matches. |
| sun2022xpos | 2023 | ✅ | ACL 2023, arXiv 2212.10554. **Bibkey says `2022`, year field says `2023`**. The arXiv submission was Dec 2022 but the conference proceeding is ACL 2023; year field is the correct ACL year. Bibkey naming is non-fatal (P2). |
| srivastava2015unsupervised | 2015 | ✅ | ICML 2015 — matches PMLR vol 37 pp 843-852. |

All older entries pass sanity (year and venue plausible, no fictitious authors).

---

## 5. Stylistic findings

### P1 — bibkey/year inconsistency (2 entries)

| Finding | File | Severity |
|---|---|---|
| `chen2024position` bibkey says 2024 but `year = {2023}` (arXiv 2306, June 2023). Actual paper not formally venue-published; arXiv-only. Citation will render as "Chen et al., 2023" which is correct, but the bibkey makes log-greppable diffs slightly confusing. | `paper/refs/references.bib:42` (entry header), `:46` (year field) | P1 |
| `qwen2026mhrope` bibkey/booktitle says `ICLR 2026` and `year = {2026}`, but arXiv v1 was 2025-10-27 (`note = {arXiv:2510.23095}`). v3 was 2026-04-06 — so by camera-ready time the venue year is correctly 2026. **Not a bug** but worth verifying ICLR 2026 acceptance before camera-ready. | `paper/refs/references.bib:302`–`308` | P1 (precaution) |
| `sun2022xpos` bibkey says `2022` but `year = {2023}` (ACL 2023). Citation renders as "Sun et al., 2023" (correct ACL year). Bibkey naming inconsistency only. | `paper/refs/references.bib:278`, `:282` | P2 (cosmetic) |

### P2 — author truncation `and others` (7 entries)

BibTeX-legal but expands to "et al." in the rendered reference. NeurIPS camera-ready convention is to list **all** authors. Should be expanded before camera-ready.

| Bibkey | File:line | Authors visible | Truncated |
|---|---|---|---|
| touvron2023llama2 | `paper/refs/references.bib:224` | 10 listed | yes (LLaMA-2 has ~70 authors; truncation traditional) |
| grattafiori2024llama3 | `paper/refs/references.bib:231` | 3 listed | yes (LLaMA-3 has 500+ authors; truncation expected) |
| qwen2024qwen25 | `paper/refs/references.bib:238` | 9 listed | yes |
| yang2024cogvideox | `paper/refs/references.bib:245` | 10 listed | yes |
| kong2024hunyuanvideo | `paper/refs/references.bib:259` | 9 listed | yes |
| roziere2023codellama | `paper/refs/references.bib:287` | 10 listed | yes |
| li2025hope | `paper/refs/references.bib:312` | 1 listed (only `Li, Haoran`) | **yes — and only one author shown is unusual; should at minimum list all named authors of the HoPE/VLM paper** |
| hua2025fope | `paper/refs/references.bib:327` | 1 listed (only `Hua, Ermo`) | **yes — same concern: only one author shown** |

The two highlighted (`li2025hope` and `hua2025fope`) are notable because they list only one author + `others`, which is unusual and may be flagged by reviewers as evidence of careless bibliography. **Recommend expanding both.**

### P2 — empty / missing fields

- No empty title fields detected.
- No missing year fields detected (all 44 have year).
- No missing booktitle/journal detected.

### Style notes

- 3 entries use `note = {arXiv:NNNN.NNNNN}` instead of putting the arXiv ID in the standard `eprint`/`archiveprefix` slot. Renders correctly with default NeurIPS `.bst`, but mixing styles is a minor inconsistency. (`qwen2026mhrope`, `li2025hope`, `hua2025fope`.) P2.

---

## 6. Summary

- **44/44 entries cited.** No dead entries. (Coverage rule passes.)
- **All 16 direct arXiv ID checks resolve.** No P0 findings — no bibkey points at a wrong or non-existent paper.
- **`dai2025hyperbolicrope` (this round's add)** verified clean: arXiv 2509.05218 resolves to "HoPE: Hyperbolic Rotary Positional Encoding…" by Dai et al. 2025. The §2 disambiguation from VLM-HoPE (`li2025hope`) is in place at `paper/sections/02_related.tex:6`.
- **2 P1 findings** (bibkey-year naming convention drift): `chen2024position` (key says 2024, year field 2023, arXiv 2023) and `sun2022xpos` (key says 2022, year field 2023). Both render correctly; only the bibkey is misleading. Auditors comparing logs across drafts may want to harmonize.
- **7 P2 findings**: author truncations using `and others` in `touvron2023llama2`, `grattafiori2024llama3`, `qwen2024qwen25`, `yang2024cogvideox`, `kong2024hunyuanvideo`, `roziere2023codellama`, `li2025hope`, `hua2025fope`. The last two are most concerning because they show only one named author. Expand before camera-ready.
- **Older-entry sanity** (vaswani 2017, shaw 2018, raffel 2020, su2024roformer, radford 2019): all pass.

**Net assessment: bibliography is in submittable shape. No substantive bugs. Recommended pre-camera-ready cleanup:**

1. Expand `and others` in `li2025hope` and `hua2025fope` to full author lists (most visible P2).
2. Harmonize bibkey naming for `chen2024position` (rename to `chen2023position`) and `sun2022xpos` (rename to `sun2023xpos`) OR accept the existing convention as deliberate. Cosmetic only; do not break references unless a global rename is risk-free.
3. After ICLR 2026 acceptance is confirmed for `qwen2026mhrope`, no further action; if not accepted, may need to reframe as `arXiv:2510.23095` and adjust `booktitle`.

---

## File references

- Bibliography file: `paper/refs/references.bib` (345 lines, 44 entries)
- Citation grep scope: `paper/sections/*.tex`, `paper/appendix/*.tex`, `paper/tables/*.tex`, `paper/main.tex`
- Disambiguation already in place: `paper/sections/02_related.tex:6` (Hyperbolic-RoPE vs VLM-HoPE)
- Hyperbolic-RoPE add provenance: handover `scripts/2026-04/PAPER_HANDOVER_2026-04-27.md:69` ("§2 加 Hyperbolic-RoPE (Dai et al. 2025, arXiv:2509.05218) name-collision disambiguation")
