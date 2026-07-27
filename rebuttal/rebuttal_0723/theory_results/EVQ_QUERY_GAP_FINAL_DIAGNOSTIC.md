# OLMo-2 1.485B EVQ Query-Gap Final Diagnostic

Date: 2026-07-28
Evidence tier: `POST_SUB_RAW_HASH_BACKED`
Seed scope: one training seed (`20260728`), 100 fixed evaluation rows per
length

## Final decision

**Select outcome 2: one explicitly authorized 32-step
answer-plus-immediate-EOS continuation was executed after the pre-repair
diagnostic, and no further capability training is needed.**

The final EVQ chain is:

```text
EVQ 300-step parent 95ceeb70...
  → query-gap +100 a0ccd2cf...
  → realized-gap answer+EOS 32 steps 2119c8c5...
```

On real contiguous prompts with greedy autoregressive decoding, the final EVQ
adapter obtains complete expected-answer string exact followed by observed
terminal EOS on `100/100` at 4K, `98/100` at 8K, and `60/100` at 16K.

The fully matched downstream Native chain is:

```text
Native 300-step parent 6570ab94...
  → identical query-gap +100 c140874f...
  → identical realized-gap answer+EOS 32 steps 8b8ffc57...
```

It obtains `95/100`, `18/100`, and `0/100` under the same strict endpoint.
Thus the same downstream supervision does not reproduce the EVQ
length-transfer result on the Native substrate.

The older repair-before-training diagnostic remains valid only as the
pre-repair baseline: adapter `a0ccd2cf...` had first-number retrieval but
`0/100` full-string-plus-EOS at 8K and 16K. It must not be used as the current
final outcome.

## Claim boundary

This result supports:

- 4K-physical-token LoRA training with explicit target-range RoPE phase
  exposure;
- complete numeric task-answer generation plus terminal EOS on real 8K and
  partially on real 16K prompts;
- a matched, same-family EVQ-versus-Native downstream-training comparison;
- no broad 4K language-modeling, RULER-family, short-context MCQA, or basic
  dialogue collapse.

It does not establish:

- training without any exposure to long-range positions;
- clean unseen-task 8K/16K transfer;
- uniform 16K capability;
- multi-seed stability, statistical significance, universal EVQ dominance,
  or long-context SOTA;
- general natural-language QA from a numeric NIAH endpoint.

## Verified identities

| Component | SHA-256 / contract |
| --- | --- |
| Base model | OLMo-2-0425-1B-Instruct, 1,484,916,736 parameters, `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` |
| Fixed EVQ frequency | `917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607` |
| Fixed Native frequency | `dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34` |
| EVQ 300-step parent | `95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a` |
| EVQ query-gap +100 | `a0ccd2cf141300ba4489882dda1324b2f237e65444a9a71d687e8c5fad57ae8b` |
| EVQ final +32 | `2119c8c5683f2b244c3dd9a1f3b2587c50eeb81314a82ec0dacd47f2ec889550` |
| Native 300-step parent | `6570ab94aec68431dd4e261eb3ef342ef72253357aa65a0d37b0a018df2f3f8d` |
| Native query-gap +100 | `c140874fe00849fb9a51a8528dcf6f55215cc28f9e22f6bdf3f649f2ce421dbb` |
| Native final +32 | `8b8ffc577b09a146a8bff5699a81cd550a96f275c2ffc9d556f9a24ec6f6a929` |

Both +100 arms use the same trainer, frozen routing data, seed, 100-step
budget, optimizer/LR semantics, query-offset stream, and physical-token
budget. Both +32 arms use the same rebuilt routing data, seed, optimizer,
answer-plus-EOS objective, realized gap/position/exposure streams, and
32-step budget. The intended scientific difference is the fixed frequency
substrate and the corresponding parent state.

All backward passes contain at most 4,096 physical tokens. Explicit position
IDs expose source-to-query phases extending to the 16K range; the largest
observed +32 position ID is 16,257. The accurate description is therefore
“at most 4K physical tokens with target-range relative-phase exposure.”

## Strict autoregressive capability

The primary endpoint is literal complete answer-string equality plus terminal
EOS from raw token IDs. It is not first-number, substring, teacher forcing,
NLL, or PPL.

| Full string + EOS | EVQ `+100+32` | Native `+100+32` | EVQ − Native |
| --- | ---: | ---: | ---: |
| 4K | **100/100** | **95/100** | `+5` |
| 8K | **98/100** | **18/100** | `+80` |
| 16K | **60/100** | **0/100** | `+60` |

All final generations in both arms terminate with EOS. For EVQ, the
beyond-original-training-gap subsets are `48/50` at 8K and `31/66` at 16K.
The latter is the nearest material boundary: 16K remains
distance-sensitive.

The strict raw generation hashes are:

- EVQ: `6fbde008d2732b8ca8d382b25a8719d854ae2d19dbf565bbda11ce2cfbe02dbd`;
- Native: `d296085c2a48a0141fc5337f2e1a892d7069f80656a1d29522bf28a9f206aec9`.

## Failure classification

| Arm / length | Wrong answer retrieval | Incomplete answer | Narrow format error | EOS failure |
| --- | ---: | ---: | ---: | ---: |
| EVQ 4K | 0 | 0 | 0 | 0 |
| EVQ 8K | 2 | 0 | 0 | 0 |
| EVQ 16K | 40 | 0 | 0 | 0 |
| Native 4K | 0 | 0 | 5 | 0 |
| Native 8K | 82 | 0 | 0 | 0 |
| Native 16K | 100 | 0 | 0 | 0 |

The five Native 4K failures all generate the correct number, then token `13`
(`.`), then EOS. They remain strict failures; the metric was not relaxed.
The direct token-level reason that `.` outranks EOS on these five rows is not
yet established and is not needed for the long-range conclusion.

## Isolated EOS32 control

An auxiliary comparison applies only the identical final 32-step
continuation directly to the original 300-step parents:

| Full string + EOS | EVQ `95ce+32` | Native `6570+32` |
| --- | ---: | ---: |
| 4K | `100/100` | `72/100` |
| 8K | `79/100` | `0/100` |
| 16K | `18/100` | `0/100` |

This isolates the last 32-step intervention, not the complete +100+32
lineage. The complete matched downstream comparison is the primary table
above.

## NLL/PPL remains separate from capability

After query-gap +100:

| Substrate | 4K NLL / PPL | 8K NLL / PPL | 16K NLL / PPL |
| --- | ---: | ---: | ---: |
| EVQ | `2.54411 / 12.73` | `2.69674 / 14.83` | `2.91416 / 18.43` |
| Native | `2.23777 / 9.37` | `3.70708 / 40.73` | `4.82004 / 123.97` |

The +32 continuation changes 4K natural NLL by `+0.001404` for EVQ and
`+0.001953` for Native. These are language-modeling retention measurements;
the capability claim comes only from strict generation.

## 4K RULER/QA retention

Each matrix contains 13 RULER task families with 20 fixed rows per task.

| Arm | Direct parent macro | Final macro | Delta | QA family delta | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| EVQ | `0.27821` | `0.26442` | `-0.01378` | `0.00000` | localized `STOP` |
| Native | `0.58141` | `0.58269` | `+0.00128` | `0.00000` | `PASS` |

EVQ does not show broad collapse, but `niah_single_2` changes from `0.70` to
`0.55`; this localized 15-point regression must remain adjacent to any
no-catastrophic-forgetting statement. Native's largest accepted task decrease
is `niah_multikey_2`, `0.70` to `0.65`.

## General-capability retention

A separate frozen short-context diagnostic uses 100 revision-pinned MMLU rows
and 100 ARC-Challenge rows. Prompts contain 45–146 tokens. The metric is
zero-shot choice-string mean-logprob accuracy, not an official leaderboard
reproduction.

| Arm | MMLU parent → final | ARC parent → final | Macro delta | Gate |
| --- | ---: | ---: | ---: | --- |
| EVQ | `34% → 34%` | `29% → 28%` | `-0.5 pp` | `PASS` |
| Native | `27% → 25%` | `35% → 38%` | `+0.5 pp` | `PASS` |

The frozen-data manifest is
`313a16bc6852629bbfa6d0b4c0253862650c670d018d3d5b8282c908c6cae963`;
the gate is
`f393fee131abbe0a52061990007e552fa83aca10dd67195437947450e498e5ef`.

The five-prompt matched dialogue smoke is unchanged parent-to-final for
Native: arithmetic, exact-token instruction, short explanation, and dialogue
memory are correct with EOS; the sequence answer is semantically correct but
fails the registered literal-only format. The same bounded smoke had already
shown no broad EVQ dialogue collapse. It is qualitative, not a benchmark.

## Rebuttal-ready wording

> On OLMo-2-0425-1B-Instruct (1.485B parameters), every LoRA backward pass
> used at most 4K physical tokens while exposing the query block to
> source-to-query RoPE phases in the target range. Under greedy
> autoregressive evaluation on real contiguous prompts, EVQ plus the fixed
> query-gap +100 and answer-plus-EOS +32 continuation obtains complete
> answer-string exact match with observed terminal EOS on 100/100 4K,
> 98/100 8K, and 60/100 16K examples. Applying the same +100 and +32
> downstream protocol to the matched Native parent obtains 95/100, 18/100,
> and 0/100. The 16K EVQ result remains distance-sensitive (31/66 beyond
> the original maximum training gap), and these are single-seed,
> same-task-family results rather than clean unseen-task transfer. Four-k
> natural NLL changes by only +0.0014; a 13-family retention matrix shows no
> broad collapse but includes one localized 0.70-to-0.55 regression.

## Artifact index

| Artifact | SHA-256 |
| --- | --- |
| Shared query-gap +100 routing-data manifest | `83a745b25893a73749dd85ff66dea5b305be6e4e6dcf16a5d0e13cae935ec63b` |
| Shared query-gap +100 query-offset stream | `1b1a42992948378f4f51b3a8f26d92320cf6fdbcefc2608ff45a82b33df76087` |
| Shared +32 routing-data manifest | `4a2d792fa6f1b6c4dfd6c4d24ad99069f81c3d5b9f8a7334ea6090cb25c52cac` |
| Shared +32 realized-gap target stream | `92f4c3f504a3065a3cd78db580992fad22ee838c5f3b97998bf72302497f83cb` |
| EVQ +32 training result | `fb9002e29ba19198b88114b64b264573683bc1f72d238653a5990270711464e1` |
| EVQ strict n=100 result | `ab0c7521bcc2ee3f56006a3d8d9a90adeec76fc95b2e453d3ca07a882a3e921b` |
| EVQ strict raw generations | `6fbde008d2732b8ca8d382b25a8719d854ae2d19dbf565bbda11ce2cfbe02dbd` |
| EVQ 4K retention gate | `d24e68117d4255ff4699edd700a3b1052a777a956a1dd9a3bc72665755156124` |
| Native query-gap +100 result | `1c196e3b7d773bce621debb22a34ace48f39e9dd028439beef1a9fd3683e9ab6` |
| Native +32 training result | `5bb38d21bb45e20c8f351f9c1b80b5ea2d58a3e47518571c7e599619189ff747` |
| Native strict n=100 result | `19d31d8660730cbd25702f3f6f7d95df8cbb4b695c6f980261bdc4ab222bb5b7` |
| Native strict raw generations | `d296085c2a48a0141fc5337f2e1a892d7069f80656a1d29522bf28a9f206aec9` |
| Native 4K retention gate | `0da606e6aa81515d05a0eea358e56fb02863cff3039d66bf0494fb28601fb563` |
| MCQA retention gate | `f393fee131abbe0a52061990007e552fa83aca10dd67195437947450e498e5ef` |

Machine-readable aggregate:
`evq_query_gap_realized_eos32_20260728/FINAL_METRICS_AND_LINEAGE.json`.
