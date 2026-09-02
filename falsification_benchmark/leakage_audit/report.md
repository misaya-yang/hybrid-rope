# Leakage audit

- **Benchmark:** `hybrid-rope-theory-falsification-v1`
- **Status:** `PASS`
- **Episodes:** 16
- **Violations:** 0
- **Warnings:** 0

## Automated checks

| Check | Verdict |
| --- | :---: |
| `identity_and_order` | **PASS** |
| `chronology` | **PASS** |
| `forbidden_visible_fields` | **PASS** |
| `cross_episode_protected_literals` | **PASS** |
| `cross_episode_protected_phrases` | **PASS** |
| `hidden_owner_paths` | **PASS** |
| `post_start_timestamps` | **PASS** |
| `directional_language_outside_contract` | **PASS** |

## Episode audit

| Episode | Temporal grade | Protected values/phrases | Timestamp boundary | Owner path | Directional language | Manual semantic review |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| TFB-001 | C | PASS | PASS | PASS | PASS | PASS |
| TFB-002 | B | PASS | PASS | PASS | PASS | PASS |
| TFB-003 | B | PASS | PASS | PASS | PASS | PASS |
| TFB-004 | B | PASS | PASS | PASS | PASS | PASS |
| TFB-005 | B | PASS | PASS | PASS | PASS | PASS |
| TFB-006 | B | PASS | PASS | PASS | PASS | PASS |
| TFB-007 | C | PASS | PASS | PASS | PASS | PASS |
| TFB-008 | B | PASS | PASS | PASS | PASS | PASS |
| TFB-009 | B | PASS | PASS | PASS | PASS | PASS |
| TFB-010 | B | PASS | PASS | PASS | PASS | PASS |
| TFB-011 | A | PASS | PASS | PASS | PASS | PASS |
| TFB-012 | A | PASS | PASS | PASS | PASS | PASS |
| TFB-013 | A | PASS | PASS | PASS | PASS | PASS |
| TFB-014 | B | PASS | PASS | PASS | PASS | PASS |
| TFB-015 | A | PASS | PASS | PASS | PASS | PASS |
| TFB-016 | B | PASS | PASS | PASS | PASS | PASS |

## Manual scope

- Each visible packet was compared with its canonical owner and hidden answer.
- Only protocol-intrinsic fields or facts documented before the episode were retained.
- Later packets omit earlier benchmark outcomes so the full visible bundle can be submitted atomically.
- Temporal grades B/C describe provenance strength; they do not waive the zero-leakage gate.

## Artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `experiment_registry.json` | `dec5513410c18a6384e8b23df0d6976ef8f67a29236f585aa3b123b9757f5e7c` |
| `visible_packets/packets.json` | `ed0d06845e88b24e5edeb5f9833c6e00fe863baec56a0a54fbf0c4058ffb39f8` |
| `hidden_answers/answers.json` | `afb9a9aeaffe714330b979c89fa743f0c42ce00190e4d5d26d40421a56402f10` |
| `fresh_theorist_guide.md` | `79ecd1c98471d7192a06a6d32cfad3da426a60f1921146aae13983da83ef1ffa` |
| `evaluator/core.py` | `e46b97d227df643c7d20cd173a1135edceadbd49345c986c149bc10db220ed29` |
| `evaluator/__main__.py` | `5c2b17a3237c77a8c140f876510fb9795222b967798e1627fce868943f43224d` |
| `evaluator/test_evaluator.py` | `56855d3a79d1bf4e6008b7860fa4f1562f87855be49d99285aca9bb20993c9a0` |
| `leakage_audit/audit.py` | `8a0d1027cf0009ee35404f3f5dc75498500a21abf3c99c9fac15e2e6af696a29` |
