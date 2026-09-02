# Theory Falsification Benchmark — experiment registry

- **Benchmark:** `hybrid-rope-theory-falsification-v1`
- **Version:** `1.0.0`
- **Frozen:** 2026-09-02
- **Episodes:** 16
- **Ordering:** best-supported real execution start, not report or archive date
- **Boundary:** completed historical evidence only; no new theory, method, experiment, inference, training, or GPU work

| Order | ID | Execution start | Model | Endpoint | Non-obvious discriminator | Temporal grade |
| ---: | --- | --- | --- | --- | --- | :---: |
| 1 | TFB-001 | 2026-02-27 | 50.9M decoder | TinyStories PPL, 2K–16K | non-monotone finite-grid response | C |
| 2 | TFB-002 | 2026-03-03 | 454.2M decoder | passkey NLL-gap + PPL | substrate × fixed scaling operator | B |
| 3 | TFB-003 | 2026-03-05 | 125M decoder | PPL, 256–8192 | table effect under Kerple+MLP | B |
| 4 | TFB-004 | 2026-03-06 | 750M decoder | PPL / retrieval / AR exact | retrieval ceiling vs exact generation | B |
| 5 | TFB-005 | 2026-03-09 | 50.9M decoder | weighted OOD NLL/PPL | exact optimum vs displaced basin | B |
| 6 | TFB-006 | 2026-03-16 | 129.6M Video-DiT | denoising MSE, 32→128 frames | cross-modal transfer | B |
| 7 | TFB-007 | 2026-03-20 | 125M MHA/GQA/MLA | PPL + passkey NLL-gap | scarcity effect vs architecture monotonicity | C |
| 8 | TFB-008 | 2026-07-12 | LLaMA-3-8B + LoRA | temporal-holdout NLL, 8K–32K | short/long sign crossover | B |
| 9 | TFB-009 | 2026-07-25 | 50.9M decoder on M4 | weighted OOD NLL | pure allocation and shape non-uniqueness | B |
| 10 | TFB-010 | 2026-08-20 | 151.9M decoder | paired tail NLL, 256–2048 | fixed vs target-matched support | B |
| 11 | TFB-011 | 2026-08-21 | 151.9M decoder | paired tail NLL, 256–2048 | selected vs frozen-schedule seed | A |
| 12 | TFB-012 | 2026-08-23 | Qwen/OLMo 1.5B | RULER, 64K/16K | geometric contrast vs profile detail | A |
| 13 | TFB-013 | 2026-08-24 | OLMo-2 1.5B | PG-19 NLL, 4K/8K | long gain vs Native retention | A |
| 14 | TFB-014 | 2026-08-25 | OLMo-2 1.5B | dense NLL + 2Wiki | matched table/weights tail/full mismatch | B |
| 15 | TFB-015 | 2026-08-31 | OLMo-2 1.5B | PG-19 4K NLL | unordered multiset vs ordered coupling | A |
| 16 | TFB-016 | 2026-09-01 | Gemma-1/1.1-2B | RULER 4K–16K | 8K recovery vs 16K collapse | B |

Grades: **A** = surviving pre-result registration; **B** = raw/hash-backed protocol identity; **C** = protocol-only retrospective reconstruction with every measurement and interpretation removed.

Only `visible_packets/packets.json` and `fresh_theorist_guide.md` may be given to a fresh theorist. Registry provenance, hidden answers, audits, and evaluator internals remain coordinator-only until the atomic prediction file is frozen.
