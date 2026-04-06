# EVQ-Cosh Citation Audit Report

## Summary

24 citations verified. 5 keys corrected, 2 citations added, 1 unused entry removed, venue metadata updated for 6 entries. All cite keys now match bib keys 1:1.

---

## Key Corrections (reviewer-visible)

| Old Key | New Key | Issue |
|---------|---------|-------|
| `anil2022pose` | `zhu2024pose` | Wrong first author (Anil is not an author; first author is Zhu Dawei). Wrong year (2022 vs ICLR 2024). |
| `wang2025riflex` | `zhao2025riflex` | Wrong first author (Wang is not an author; first author is Zhao Min). |
| `qin2024freenoise` | `qiu2024freenoise` | Wrong first author (Qin is not an author; first author is Qiu Haonan). |
| `zhu2024longrope2` | `shang2025longrope2` | Wrong first author (Zhu is not an author; first author is Shang Ning). Wrong year (2024 vs ICML 2025). |
| `chen2023position` | `chen2024position` | Wrong year (2023 arXiv vs EMNLP 2024 proceedings). |

**Reviewer risk**: Misattributed keys (especially `anil2022pose` for a paper by Zhu et al.) signal sloppy scholarship and can trigger immediate desk-reject suspicion. These are now fixed.

---

## Venue Metadata Updates

| Entry | Change |
|-------|--------|
| `liu2024lost` | Added `volume = {12}, pages = {157--173}` (TACL 2024) |
| `jin2024selfextend` | Added PMLR 235:22099-22114 (ICML 2024 Spotlight) |
| `videorope2025` | Added PMLR v267 (ICML 2025 Oral) |
| `ding2024longrope` | Added PMLR series (ICML 2024) |
| `raffel2020exploring` | Added `volume = {21}, number = {140}, pages = {1--67}` (JMLR) |
| `su2024roformer` | Added `volume = {568}, pages = {127063}` (Neurocomputing) |

---

## Citations Added

| Key | Paper | Venue | Rationale |
|-----|-------|-------|-----------|
| `wang2024resonance` | Resonance RoPE: Improving Context Length Generalization of Large Language Models | ACL Findings 2024 | Closest precedent for frequency-aware RoPE design. Identifies critical frequencies during interpolation. Reviewer-expected. |
| `zhang2024found` | Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding | NeurIPS 2024 | Positional encoding modification for long-context evaluation. Complements `liu2024lost`. Reviewer-expected. |

---

## Citations Removed

| Key | Paper | Reason |
|-----|-------|--------|
| `ge2024contentbiasfvd` | On the Content Bias in Frechet Video Distance | Not cited anywhere in the paper. |

---

## Entries Kept as-is (verified correct)

`hsieh2024ruler` (still arXiv as of audit date), `radford2019gpt2` (technical report, no formal proceedings), all others verified against official proceedings.

---

## Related Work Changes

Rewrote `02_related.tex` with:
- Resonance RoPE integrated into PE design space paragraph as closest precedent
- Found in the Middle integrated into evaluation paragraph
- Sharper axis-based framing: EVQ targets training-time allocation (third axis), distinct from base-frequency tuning (first axis) and inference-time rescaling (second axis)
- Kerple moved from learnable PE paragraph into design space paragraph (it is a PE design, not a learnable/adaptive method)
- FIRE and DAPE descriptions sharpened

---

## Reviewer-Risk Note

The original bib had 4 entries where the citation key's first-author surname did not match the actual first author. This pattern (e.g., citing "Anil et al." when the paper is by Zhu et al.) is a known red flag for reviewers checking citation quality. All instances are now corrected. The `hsieh2024ruler` entry remains as arXiv; if a venue publication appears before camera-ready, update it.
