# OLMo-2 1.485B Shared QA/RULER Task Localization

Execution date: 2026-07-28
Run/seed label: `20260730`
Status: `COMPLETE / FULL MATCHED LONG CONTROL AND ROUTED COMPOSITE`
Evidence tier: `POST_SUB_RAW_HASH_BACKED`

## Question

The fresh generic-data Q/K experiment improves long-position natural-text
NLL but does not recover broad task capability after the EVQ frequency
replacement. This experiment tests whether one shared, short Q/K-only
continuation can localize that long-position substrate to both held-out
multi-hop QA and all 13 RULER families without training on evaluation rows.

## Matched parents

Both arms start from the completed fresh generic-data Q/K-only adapters:

| Arm | Parent adapter SHA-256 |
| --- | --- |
| Native | `3b8735c9b0cd6b65238f256ce0d9395c6e835be9ba7583fca485bc1e675c4740` |
| EVQ | `26157b7b586449bcc1a3ad5ff574ae4428640a047bc3fdad90a40e71d2653c79` |

The base checkpoint composite SHA-256 is
`36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f`.
The only registered method variable is the active Native or EVQ frequency
table. Base weights remain frozen; the continuation updates only the existing
Q/K LoRA tensors.

## Registered protocol

- rank 64, alpha 128 Q/K-only LoRA;
- 500 optimizer steps, global batch 8;
- learning rate `5e-5`, 20 warmup steps, fused AdamW, BF16;
- fixed optimizer-step cycle:
  `2Wiki, RULER13, 2Wiki, RULER13, LongAlign replay`;
- 200 2Wiki steps, 200 RULER13 steps, and 100 replay steps;
- within every task microbatch, deterministic
  `continuous-4K : target-8K : target-16K = 1:1:2` phase exposure;
- complete answer-plus-immediate-EOS supervision;
- every physical training sequence is at most 4,096 tokens;
- one matched seed, `20260730`.

This gives each task family the same 200 phase-step dose as the corresponding
successful separate continuation while sharing one natural-replay stream.

## Data and isolation

| View | Rows / role | Manifest SHA-256 | Isolation gate |
| --- | --- | --- | --- |
| 2Wiki phase view | 1,536 train; held-out validation | `9abbc22bcf833762076fd895f9d26706f0cfbecdceb88a2867ced63bf2fdddf9` | zero train/evaluation QA-identity overlap |
| RULER13 phase view | 1,248 train, balanced over 13 families | `0bf1f2158340c7b6190c73c3b1aa0e07e8007e8c28f2da6f36b471528c361161` | generated training and evaluation rows are disjoint; zero exact-row overlap |
| Generic replay | frozen LongAlign 4K view | bound in each READY receipt | no task labels |

This is task-family adaptation on held-out rows, not clean unseen-task
transfer.

## Executed training

Both matched 500-step continuations completed:

| Field | Native | EVQ |
| --- | ---: | ---: |
| Final adapter SHA-256 | `bf0aaeaa3e4e4221f622ea2586d760c47c9c694d315847bad96eba214ff942d3` | `24f484b988a5f083c03552b60ae3f5bebe18d49760f399fa793de39710dad769` |
| Processed input tokens | 16,380,000 | 16,380,000 |
| Optimizer steps | 500 | 500 |
| Measured training seconds | 555.53 | 728.87 |
| Measured input tokens/s | 29,485.47 | 22,473.12 |
| Peak allocated bytes | 22,280,460,800 | 23,290,729,472 |

The executed arms match exactly on:

- selection stream:
  `96534060023a1b26dce504f7056efe6fac2d9e7aa9dc7ed44e4bde2ea86bb480`;
- position stream:
  `2e535b05b443393f1f98549ae2442acd1e11f29b5e67692dd24ddfee41ee46f3`;
- exposure stream:
  `df4889359fe0ecd719afee7a61b0451013a2e7817d970b91e82de4ff0a3c8a3d`;
- 200/200/100 2Wiki/RULER/natural steps;
- 800/800/1,600 contiguous/8K-phase/16K-phase sequence exposures;
- 16,380,000 processed input tokens and the complete optimizer protocol.

Each task contributes 400/400/800 contiguous/8K-phase/16K-phase sequence
exposures. The only active method difference remains the Native versus EVQ
frequency substrate and the resulting learned adapter.

The post-training 32-row teacher-forced diagnostics are:

| View | Offset | Native NLL | EVQ NLL |
| --- | ---: | ---: | ---: |
| 2Wiki | 0 | 0.3527 | 0.5219 |
| 2Wiki | 4,096 | 1.1681 | 0.7799 |
| 2Wiki | 12,289 | 2.1311 | 1.2197 |
| RULER13 | 0 | 0.3014 | 1.8482 |
| RULER13 | 4,096 | 2.0817 | 2.8493 |
| RULER13 | 12,289 | 2.6424 | 3.1437 |

These values diagnose the optimization trajectory; they are not capability
results.

## Strict autoregressive screen

The screen uses 50 held-out 2Wiki rows per length and two held-out examples
per RULER family and length. Every result passed a strict identity gate over
checkpoint, adapter SHA, frequency substrate, data manifest, evaluator hash,
protocol, cell coverage, and raw-generation hash.

| Endpoint | Native 4K | EVQ 4K | Native 8K | EVQ 8K |
| --- | ---: | ---: | ---: | ---: |
| 2Wiki token F1 | 36.27% | 21.50% | 0.95% | 15.19% |
| 2Wiki normalized exact | 32.0% | 16.0% | 0% | 10.0% |
| 2Wiki terminal EOS | 100% | 100% | 30.0% | 100% |
| 13-family RULER official macro | 70.38% | 34.36% | 7.37% | 22.24% |

## Decision gates and stop decisions

The shared adapter is useful only if all of the following hold:

1. EVQ exceeds matched Native at 8K on both 2Wiki and complete-family RULER;
2. EVQ remains within 5 percentage points of Native 2Wiki F1 at 4K;
3. EVQ remains within 10 percentage points of Native RULER macro at 4K;
4. gains appear in autoregressive capability metrics, not only NLL;
5. Native/EVQ selection, position, exposure, family-count, and protocol
   receipts match.

The single-path gate fails because EVQ trails Native by 14.77 percentage
points on 4K 2Wiki F1 and by 36.03 points on 4K RULER. It nevertheless passes
the long-route portion of the gate: at 8K, EVQ has higher 2Wiki F1 and exact
match, 100% terminal EOS, and a 14.87-point RULER advantage over the matched
Native continuation.

No ratio, seed, rank, learning-rate, or loss sweep is authorized. The complete
4K/8K/16K single-path matrix is stopped.

## Request-length routing follow-up

The screen supports evaluating one non-training deployment alternative:

- registered total context budget at most 4K: untouched Native RoPE with the
  adapter disabled;
- evaluated 8K and 16K total context budgets: EVQ-Cosh with this one shared
  QA/RULER Q/K adapter enabled for the entire request.

The choice is made before prefill; it does not switch a KV cache mid-request,
use dual attention, or add another training intervention. It is a
deterministic two-path deployment policy, not a single pure-EVQ configuration.

The full matched 8K/16K evaluation completed for both shared Native and shared
EVQ adapters. The untouched-Native 4K components and every long component pass
the strict identity, raw-row coverage, aggregate-recomputation, and artifact
hash gate.

### Full matched long-context result

Each 2Wiki cell contains 200 greedy autoregressive generations. Each RULER cell
contains 20 held-out examples for every one of the 13 families.

| Endpoint | Native 8K | EVQ 8K | Native 16K | EVQ 16K |
| --- | ---: | ---: | ---: | ---: |
| 2Wiki token F1 | 0.48% | **15.09%** | 0% | **4.63%** |
| 2Wiki normalized exact | 0% | **12.0%** | 0% | **2.5%** |
| 2Wiki terminal EOS | 33.5% | **99.5%** | **91.0%** | 64.5% |
| 13-family RULER official macro | 2.98% | **25.50%** | 0.64% | **5.48%** |

Native's high 16K terminal-EOS rate is not capability: its 2Wiki token F1 and
exact match are both zero. The EVQ advantage is therefore present in strict
autoregressive capability endpoints, not only teacher-forced NLL.

### Derived request-length-routed result

The deterministic policy composes the independently validated branches:

| Total context budget | Active branch | 2Wiki F1 / exact / EOS | RULER macro |
| ---: | --- | ---: | ---: |
| 4K | untouched Native, adapter off | 27.64% / 22.0% / 97.5% | 65.16% |
| 8K | EVQ + shared Q/K adapter | 15.09% / 12.0% / 99.5% | 25.50% |
| 16K | EVQ + shared Q/K adapter | 4.63% / 2.5% / 64.5% | 5.48% |

This table is a deterministic composition of exact branch outputs rather than
a claim that a separate router changes model computation. The branch is chosen
from the registered total context budget before prefill; each request still
uses one attention path and one KV cache.

### All 13 RULER families

The Native 8K/16K columns are the matched shared-adapter long controls. The 4K
column is the routed untouched-Native branch.

| Family | Routed Native 4K | Native shared 8K | EVQ shared 8K | Native shared 16K | EVQ shared 16K |
| --- | ---: | ---: | ---: | ---: | ---: |
| cwe | 0.50% | 0% | 11.00% | 0% | 0% |
| fwe | 48.33% | 25.00% | 50.00% | 8.33% | 40.00% |
| niah_multikey_1 | 85.00% | 0% | 35.00% | 0% | 0% |
| niah_multikey_2 | 95.00% | 0% | 0% | 0% | 0% |
| niah_multikey_3 | 60.00% | 0% | 0% | 0% | 0% |
| niah_multiquery | 70.00% | 1.25% | 25.00% | 0% | 1.25% |
| niah_multivalue | 46.25% | 2.50% | 27.50% | 0% | 0% |
| niah_single_1 | 100.00% | 0% | 70.00% | 0% | 0% |
| niah_single_2 | 95.00% | 10.00% | 80.00% | 0% | 0% |
| niah_single_3 | 100.00% | 0% | 0% | 0% | 0% |
| qa_1 | 70.00% | 0% | 5.00% | 0% | 5.00% |
| qa_2 | 50.00% | 0% | 25.00% | 0% | 25.00% |
| vt | 27.00% | 0% | 3.00% | 0% | 0% |

At 8K, EVQ has non-zero official score in 10/13 families versus 4/13 for the
matched Native continuation. At 16K the counts are 4/13 versus 1/13. These
counts describe the registered finite evaluation; they are not statistical
significance claims.

## Interpretation and claim boundary

The shared continuation demonstrates that one task-family adapter can support
both held-out multi-hop QA and complete-family RULER length transfer on the EVQ
substrate. Request-length routing then preserves the untouched model's 4K path
structurally while using that EVQ adapter only at the evaluated long budgets.

The mandatory boundaries are:

- one matched training seed per frequency arm;
- physical training sequences are at most 4K, but target-8K/16K relative
  phases are explicitly exposed through position IDs;
- training and evaluation rows are disjoint, but the 2Wiki and RULER task
  families are shared;
- this is task-family adaptation, not generic-data or clean unseen-task
  transfer;
- the 2Wiki prompts are deterministically distractor-filled and do not
  reproduce the unmodified LongBench leaderboard protocol;
- the routed result is a two-path deployment policy, not a single pure-EVQ
  configuration and not proof that EVQ is intrinsically no-harm at 4K;
- the absolute 16K result remains limited.

## Additional-ablation decision

No further training ablation is required for this question. The completed
evidence already supplies:

1. matched Native/EVQ generic Q/K parents;
2. one matched shared QA/RULER continuation;
3. held-out QA and all 13 RULER families;
4. strict Native long controls;
5. an untouched-Native 4K deployment anchor.

Q-only/K-only, V/O, rank, alpha, learning-rate, data-ratio, second-seed, and
factor-4 sweeps would not repair the failed single-path 4K gate or change the
proper claim boundary. Allocation-shape causality should continue to use the
separate exact-range fixed-schedule factorial; this task-adapted mature-model
experiment should not be repurposed as a pure Cosh-coordinate ablation.

## Artifact provenance

The final routed owner was written only after the route script validated all
six 4K/8K/16K branch components. The separate generic-parent receipt validates
the selected generic evaluation matrix, while the completion receipt closes
the shared training and screen stage.

Core owner and gate hashes:

| Artifact | SHA-256 |
| --- | --- |
| Final routed metrics and six-component validation owner | `326a0a616d9b472d3f2b4606cdf636b17946ddeb9fb18741a294afadb9e478c3` |
| Generic-parent selected-matrix strict receipt | `5b5541efc9d67e0eec2dc3a4b88a3b3d11613221c553883d77638a73037e5774` |
| Generic training/screen completion receipt | `b425f0d3848a60d0b5c5efb13aff3b549c3bda119e1f5a83f713ee9dad802e0c` |
| Screen gate | `a4b8a5fb6df44543da8080d63030e8adb8ac1a5cc568930620a23a17fdb4bf75` |
| Validator | `48205a58142ba00185cd436a307dc0e82474d1ce414874d8cb7e6adee919c77a` |
| Route owner script | `b9e6acc8b873024db622627ae1e7a2691d6314bb238f543d79e7b870ed047275` |
| EVQ shared training result | `ec18e009c64ed75a2415ac6ddc0184c45ddbf09b6c95169fd652cb45d48771d1` |
| Native shared training result | `716872c02126ea26a96e49958fa206facbc20cfc88728098f414c7974bd5293d` |

Each row below is `results / examples / run-manifest` SHA-256:

| Component | Artifact SHA-256 values |
| --- | --- |
| 2Wiki untouched Native 4K | `3a836250d4d2f32cc5498f9a967e85bd159ec6d636d52579912cb85a6d100a2a` / `d17044cc81081f4471555c5a08c80e84cbb27233b0ac70b1a249dac6e009590b` / `e5c87ce5623b27d7c47c99f325774fcd76a7a654425bb61cc5a80c2c12da24fb` |
| 2Wiki EVQ shared 8K/16K | `6b2fd4e252250c5a42ad9b5713859fce5cd5d227ff36b119d3fc41e08283da25` / `0db2077957a7dc38755fbd69e30ff1e923f5a2aeedafb2d3e9be657a2a91b3ad` / `cafc5064e4907973c040a4b516cff70003543a34766ff5d1cc873dc728b67e34` |
| 2Wiki Native shared 8K/16K | `92986dc2973d2bbceeddbe9bfda1fdd864512bf03d19dde61c69fd3f7dfccb39` / `0e2513afe5975bc42a4b0cffbb6a2e983e5762137a352f0bfc200fa15927d15e` / `3ddb5dbf6d33a922c9c571ad18c1c44c56d81cd4fe6044aa21190fdf0e8fd3c2` |
| RULER untouched Native 4K | `6880005fda82956e977ab65547fa06ffb7c77d09043ad7503a0c05d59e2d5d45` / `a46e92f0364e7b932f3b8a95e6aaac9c120277cf971693153433a4ab455a7acb` / `f48581dc8566fcefc4267d8d94fb443f0550c63cc7c9400650f3def7fc7fa794` |
| RULER EVQ shared 8K/16K | `adb1db6eb1e91653da5869c12a256d622ef352607bdcdef7e2f8b7389e4bf81e` / `1df5b53df3ea5a3a37fdc137525d307aac08b670371131e7bf859d15e3450c70` / `d2e4be30cae8538eb29b65cf37b5989c38073016321cd35b185cb25b3ea5f63e` |
| RULER Native shared 8K/16K | `9cdebfab6ec281c5612faa1330351dde2af7ff67e39d763975409edc88adb29d` / `3d29275a352f3da772b4ba70897d8b1662a023322739fb96fff373862055acf5` / `2bf21beb37b2d490fb65b7fad4fbb09923df20d8c2358e24e3915b28c2c16274` |

The 2Wiki/RULER data-manifest hashes are
`9abbc22bcf833762076fd895f9d26706f0cfbecdceb88a2867ced63bf2fdddf9`
and
`886d94731af31f30698204caa9f30045887c9333d6327e2e1c33555ed57da68c`.
The corresponding evaluator hashes are
`03f3c309b8c826e910d71cf85720acc2b8ec65e63479a243ffe35acbab123720`
and
`92ae680b1c3474c0ed9d1425d523356fecbe4550b589584d9030a877bb6e3b96`.

## Reviewer-facing wording

> On OLMo-2 1.485B, we trained one matched seed of Native/EVQ Q/K-only
> adapters using a shared 2Wiki/RULER-family continuation, with every physical
> sequence at most 4K and target-range phases exposed only through position
> IDs. On held-out rows, Native/EVQ obtains 0/12.0% 2Wiki exact and
> 2.98/25.50% complete 13-family RULER macro at 8K; at 16K the corresponding
> results are 0/2.5% and 0.64/5.48%. For deployment, a deterministic prefill
> policy preserves the untouched Native path for total budgets at most 4K
> (22.0% 2Wiki exact; 65.16% RULER) and activates the EVQ frequency profile
> and shared adapter only at the evaluated 8K/16K budgets. The 2Wiki endpoint
> uses deterministic answer-filtered distractor filling rather than the
> unmodified LongBench protocol. This is single-seed task-family adaptation
> and a two-path deployment policy, not a pure-EVQ no-harm result or
> unseen-task transfer.
