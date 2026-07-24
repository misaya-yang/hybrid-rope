# MLA shared-operator YaRN factorial: RTX 5090 plan

Status: `OFFLINE_CODE_COMPLETE / REMOTE_PREFLIGHT_PENDING / DO_NOT_LAUNCH`

Last updated: 2026-07-24

## 1. Why this is the next useful experiment

The completed MLA scarcity run gave two different answers:

- raw EVQ did not beat native at the registered 2x endpoint;
- with substrate-specific YaRN mappings, K=8 EVQ became increasingly better
  at 4x/8x extrapolation.

The second result is not a clean interaction test because native uses the
official native-grid operator while EVQ uses a virtual-coordinate
generalization. The next experiment should ask only:

> With exactly the same per-index YaRN correction coefficients and the same
> attention `mscale`, does range scaling increase EVQ's advantage more when
> active rotary frequencies are scarce?

This is a mechanism follow-up. It does not repair the submitted YaRN identity,
prove official YaRN compatibility on EVQ, or provide production-scale evidence.

## 2. Frozen factorial

Reuse the completed 50.1M MLA architecture and training protocol:

- FineWeb-Edu, `L_train=4096`, base 500K;
- fixed `d_rope=64`, `d_nope=0`, 32-pair capacity;
- active budgets K=8 and K=32; inactive pairs use zero-frequency identity
  rotation;
- 299,892,736 tokens per arm, checkpoints near 200M and 300M;
- native endpoint Geo versus EVQ-Cosh `tau=1.414`;
- identical initialization, row order, optimizer, batch, data prefix and
  training code within each seed.

This gives four training arms per seed:

| active pairs | substrate |
| ---: | --- |
| 8 | native |
| 8 | EVQ |
| 32 | native |
| 32 | EVQ |

`tau=1.414` is retained only as the historical empirical MLA setting. This
experiment is not a tau sweep and cannot validate the MLA `d_eff` convention.

## 3. Inference operators

Every checkpoint is evaluated with:

1. `raw`;
2. `position_interpolation`: divide every active frequency by scale;
3. `shared_index_freq_only`: official YaRN correction bounds and linear
   channel-index mask, applied identically to native and EVQ, `mscale=1`;
4. `mscale_only`: raw frequencies plus the common official attention
   amplitude;
5. `shared_index_full`: item 3 plus the common official `mscale`;
6. `virtual_coordinate_full`: official native operator for native and the
   existing virtual-coordinate generalization for EVQ, secondary deployment
   diagnostic only.

At 200M, evaluate only `raw` and `shared_index_full`; these are sufficient for
the frozen direction-stability gate. At 300M, evaluate all six operators. This
reduces total registered inference work by one third (and parity-operator work
by 40%) without changing a training arm, primary endpoint or decision criterion.

Evaluation batches are frozen to 8/4/2/1 windows at 4K/8K/16K/32K. Every batch
therefore contains at most 32K input tokens, no more than the already validated
single-window 32K path. Per-window NLL is still computed independently. This
reduces forward-launch count by 53% without changing anchors or estimands.

The shared-index implementation must satisfy element-wise parity with official
YaRN on native endpoint RoPE. On EVQ it must be labeled
`shared-index YaRN-component control (NOT official YaRN)`.

For base 500K and original length 4096, the frozen official correction ranges
are channel indices `low=1, high=4` at K=8 and `low=7, high=16` at K=32.
These same masks are used for native and EVQ; no virtual-coordinate remapping
is allowed in the shared-index arms.

## 4. Fresh evaluation data

The previous selection and test aggregates have now been observed. They cannot
serve as a new confirmatory split.

Before GPU startup:

- allocate new 16-window selection and 32-window test anchors from validation
  regions that do not overlap any previous 32K window; or use a separately
  pinned FineWeb-Edu validation shard;
- prove train/validation separation and pairwise 32K non-overlap;
- hash the validation tensor, both anchor arrays and the exact training prefix;
- freeze all 4K/8K/16K/32K endpoints even though 16K/32K are the registered
  interaction endpoints.

## 5. Estimands and gate

For K, length L and operator O, define the EVQ advantage:

`D[K,L,O] = NLL(native,K,L,O) - NLL(EVQ,K,L,O)`.

For the shared full operator:

`J[K,L] = D[K,L,shared_index_full] - D[K,L,raw]`.

The scarcity interaction is:

`I = mean(J[8,16K], J[8,32K]) - mean(J[32,16K], J[32,32K])`.

Seed 42 may expand to seeds 43/88 only if all conditions hold on the new
selection split at 300M:

1. `D[8,16K,shared_index_full] > 0` and
   `D[8,32K,shared_index_full] > 0`;
2. their mean is at least `0.05 NLL`;
3. `J[8,16K] > 0`, `J[8,32K] > 0`, and their mean is at least `0.05 NLL`;
4. `I > 0`;
5. at 4K, EVQ-minus-native under the shared full operator is at most
   `+0.02 NLL` for both K=8 and K=32;
6. at 16K and 32K, for both K=8 and K=32,
   `NLL(native,shared_index_full) - NLL(native,raw) <= +0.05`;
7. the signs in conditions 1 and 3 agree at 200M and 300M;
8. the native arm has exact registered-schedule identity, exact official
   correction-mask/`mscale` parity, and at most one FP32 ULP output difference
   from the equation-faithful official path. The one-ULP allowance covers only
   checkpoint FP32 rounding; method identity is never inferred by tolerance.

No interaction may pass solely because one control collapses. Failure of any
condition is a terminal `STOP`; test anchors remain unread and no additional
seed is launched.

After a seed-42 PASS, seeds 43/88 use only the unseen test split. A claim
requires positive K=8 shared-operator advantage and positive scarcity
interaction in every seed. Report all lengths and all six operators regardless
of sign.

### Statistical role of the thresholds

The `0.05 NLL` threshold is a predeclared practical-effect floor, not a
variance-derived significance cutoff. An advantage of 0.05 NLL corresponds to
`exp(-0.05)=0.9512`, or about 4.9% lower PPL for EVQ. Likewise, the 0.02
in-domain-cost ceiling allows about 2.0% higher PPL. The selection split exists
only to prevent spending on clearly weak directions; it is not used for the
final claim.

The confirmatory unit is the training seed (`n=3`). The 32 windows are paired
repeated measurements and must never be treated as 32 independent seeds.
Final reporting includes:

- every per-anchor Native-minus-EVQ difference and operator interaction;
- seed-level means and t-based 95% intervals;
- a separate precision grade stating whether the seed-level interval excludes
  zero.

The registered practical claim still requires all seeds to agree and the mean
effect to exceed 0.05 NLL. If its seed-level interval crosses zero, wording must
remain “directionally consistent at three seeds,” not “statistically
significant.”

The prior scarcity aggregate preserves exact means and provenance hashes, but
the 99-file per-window retrieval bundle is no longer discoverable in the
current local workspace. It therefore cannot be used to retrofit an empirical
MDE or change these thresholds. The new summary retains paired anchor effects
directly so this evidence is not lost again.

## 6. Cost and execution

The completed workload measured 12.30 minutes per 300M-token arm. Expected
training time is therefore:

- seed-42 gate: four arms, about 49 minutes;
- seeds 43/88 confirmation: eight arms, about 98 minutes;
- full PASS path: about 2.5 hours plus evaluation and serialization.

The RTX 5090 profile in `docs/overview/RTX5090_BLACKWELL_PROFILE.md` remains the
runtime default. Use BF16, Flash-only SDPA, `torch.compile(default)`, fused
AdamW and the persistent Inductor cache, but repeat the discarded K=8/K=32
probe before paid training.

Do not open a GPU instance until the new data manifest, operator parity tests,
READY receipt, exact launch command, disk budget and stop/cleanup path all pass
off GPU.

## 7. Current implementation state

The package now includes the frozen protocol, source-manifest validation,
fresh-anchor generator, two-level READY receipts, fail-closed seed/test access,
five operator evaluators, seed-42 gate, three-seed summary, proof-backed
checkpoint cleanup integration, persistent compile-cache handling and a
single launcher. Terminal PASS and STOP paths both generate a hash-backed
Markdown report. A compact artifact monitor sleeps for five minutes between
snapshots instead of streaming logs.

Fresh runtime tests pass locally for:

- official/shared-index equations and identical masks;
- registered FP32 native checkpoint parity within one ULP;
- rejection of near-but-not-identical training schedules;
- fresh-window non-overlap and source hash validation;
- terminal gate behavior, including K=32 native-control collapse;
- refusal to train seeds 43/88 or read test before a PASS gate;
- complete three-seed summary construction.

No server-side fresh anchors or READY receipt exist yet, and no GPU run has
started. Before launch, run the CPU-only `preflight` command on the target data
disk, then independently inspect both receipts and the exact command. Code
readiness is not launch authorization.
