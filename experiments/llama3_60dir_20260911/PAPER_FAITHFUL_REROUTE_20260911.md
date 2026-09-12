# Paper-faithful reroute: progressive radix conversion before broad search

Date: 2026-09-11

## Decision

Pause the remaining Plan B S queue after the in-flight
`D13a_STATIC_ENDPOINT` arm.  Do not treat the 20-direction catalogue as the
source of a method.  Replace the next block with one comparison that directly
tests the mechanism and operating regime used by MrRoPE.

## What the paper actually changes

For the middle RoPE band, YaRN induces a regressive sequence of per-digit radix
factors.  MrRoPE-Pro keeps the same high/middle/low partition and YaRN attention
gain, but makes the log-radix increments increase arithmetically with dimension.
Equivalently, its cumulative frequency compression starts slowly at the
high-frequency edge and becomes steeper toward the low-frequency edge.  The
claimed benefit is therefore not generic frequency search: it is preservation
of high-frequency/local information while assigning the required extension to
less local, slower channels.

MrRoPE-Pro is an `s`-indexed family, not one universal table.  Both
`lambda_j = s**epsilon_j` and the YaRN attention gain depend on the deployment
factor.  At `s=1` it is exactly native RoPE; at `s=4` and `s=16` it is a
different operator.  Therefore the completed fixed-4 MR result is a valid
baseline for a 32K-target deployment, but it is not the paper's 128K-target
baseline.

The paper's Llama experiment deploys an 8K checkpoint to a 128K target
(`s=16`).  Its reported RULER gap is small at 8K--32K and becomes largest at
64K--128K.  The old Plan B uses a fixed `s=4` table and stops at 32K, so it does
not exercise the regime carrying the paper's main claim.

The paper reports its fixed-16 methods on 8K inputs but does not include the
unmodified `s=1` model in the relevant PPL, NIAH, or RULER comparisons.  It
therefore does not establish that deploying the 128K table preserves native
8K capability.

## Immediate experiment

Use the already validated Llama runner and frozen S rows.  At 8K and 32K,
compare:

1. `MR`: paper Eq. 14 progressive conversion at `s=16`;
2. `BM`: the one existing theory-derived candidate, replacing MrRoPE-Pro's
   arithmetic increments with a boundary-matched progressive filling;
3. `OfficialYaRN`: the pinned index-ramp implementation at `s=16`.

All arms use Meta-Llama-3-8B-Instruct, the checkpoint-derived native frequency
grid, the same band `(18, 35)`, the same analytic YaRN gain
`1 + 0.1 ln(16)`, greedy decoding, and the frozen S rows (32 at 8K and 64 at
32K).  Reuse the completed native (`s=1`) and fixed-4 results rather than rerun
them.  This gives both the paper-scale method comparison and the native-length
tax of deploying an `s=16` operator.  This is a development comparison, not an
independent confirmation.

Primary endpoint: the existing RULER-derived partial-recall macro, equally
weighted over the eight tasks.  Save raw outputs.  `strict`, full-string exact,
and EOS remain diagnostics computed from the same generation and are not
selection gates.  No extra generation is spent on them.

The combined existing/new cells form a small scale-policy factorial:
`native(s=1)` and `{MR, BM, YaRN}(s=4, s=16)` at 8K and 32K.  Report method
labels with their `s`; never collapse `MR(s=4)` and `MR(s=16)` into one
baseline.  This also evaluates the algebraically specified length-matched
policy `s_eff=max(1, planned_sequence_length/W)`: it is native at 8K, reuses
the fixed-4 cells at 32K, and becomes the paper's fixed-16 operator at 128K.
Treat this as a scale-policy candidate, not as evidence for a new frequency
allocation law.

The first decision is directional rather than a significance claim:

- first report each `s=16` arm's 8K change from native `s=1`; the paper reports
  its extension methods at 8K but provides no native baseline, so it does not
  establish lossless native-length retention;
- if BM improves the 32K primary endpoint over both MR and OfficialYaRN and the
  gain is not confined to one anomalous row, promote the same three rules to a
  compact 64K stress panel;
- if BM does not improve, stop this construction rather than expanding a
  coefficient or table search; use the paired failures to derive the next
  change to progressive allocation;
- regardless of the 32K ordering, do not use it to refute the paper's 64K--128K
  claim, because that regime has not yet been measured.

## Execution identity

The dry run must report target 131072, `low=18`, `high=35`, `n=17`, and finite
MR/BM/OfficialYaRN operators.  Results live under
`/root/autodl-tmp/llama3_mrrope_s16_20260911/` and must not be mixed with the
fixed-4 Plan B result directory.

## First completed result

The 8K+32K development comparison completed with 96 valid scored rows per arm.
On the 32K primary macro, BM(s=16) scored 0.596875, MR(s=16) 0.532031, and
OfficialYaRN(s=16) 0.449219.  BM therefore gained +6.48pp over MR and +14.77pp
over YaRN.  Against MR, five task macros improved and three tied; 19 individual
rows changed (13 wins, 6 losses), so the gain was not a single-row artifact.
The simultaneous source-bootstrap interval was [-3.93,+16.90]pp versus MR and
[+4.41,+25.12]pp versus YaRN.  This is a promotion signal, not confirmation.

The same fixed-16 deployment incurred a large native-length cost relative to
native(s=1)=0.867708 at 8K: MR 0.679688 (-18.80pp), BM 0.642188 (-22.55pp),
and YaRN 0.632812 (-23.49pp).  Thus BM(s=16) is currently a long-context Pareto
point, not a lossless replacement.

The registered branch fired.  A compact 64K panel (four rows per task, balanced
retrieval depths) is being prepared while the useful 16K scale-curve cells run.
An unattended supervisor will run one BM 64K feasibility row first and, only if
it succeeds, the complete MR/BM/OfficialYaRN 64K comparison.

The 16K scale-curve cells subsequently completed: BM(s=16)=0.685417 versus
MR(s=16)=0.571875 and OfficialYaRN(s=16)=0.571875, a +11.35pp primary gain over
either baseline.  The one-row 64K feasibility run completed without OOM; the
full 32-row-per-arm 64K comparison is now running near the 32 GiB device limit.

The compact 64K comparison completed: BM(s=16)=0.514063,
MR(s=16)=0.460417, and OfficialYaRN(s=16)=0.351562.  BM gained +5.36pp over MR
and +16.25pp over YaRN.  The 32-row panel is deliberately small, but BM changed
14 rows versus MR (10 wins, 4 losses), so the signal again is not one example.

This same-length result makes the next comparison the scale policy, not another
frequency shape.  A matched `s=8` MR/BM/OfficialYaRN run on the identical 64K
panel is now active.  It directly tests whether setting the operator to the
required context length recovers performance relative to the paper's fixed
128K-target (`s=16`) deployment.

The 64K `s=8` comparison then completed.  MR(s=8)=0.545313 was the best rule,
followed by BM(s=16)=0.514063, BM(s=8)=0.490104, MR(s=16)=0.460417, and
OfficialYaRN(s=16)=0.351562; OfficialYaRN(s=8) collapsed to 0 on this panel.
Matching the scale raised MR by +8.49pp, but lowered BM by -2.40pp.  Therefore
scale is not a nuisance label: it interacts with the filling rule.  The earlier
BM(s=16) advantage over MR(s=16) is real for a fixed 128K-target deployment,
but BM does not beat the proper 64K-target MrRoPE baseline.  A 16K `s=2`
comparison is now running to complete the prescribed length-matched curve
`s in {1,2,4,8}` without adding another frequency shape.

At 16K, the matched `s=2` cells produced a large shape separation:
BM=0.837500, OfficialYaRN=0.293490, and MR=0.260417.  BM also remained slightly
above the best over-provisioned `s=4` baseline (OfficialYaRN=0.809375).  This is
not explained by a different gain within the `s=2` comparison because all
three arms share the same analytic `g_2`; however BM has sum_m=37 while MR has
sum_m=103/3.  The next and only new factor is therefore an exact area-matched
MR control with amplitude `(37)/(103/3)=111/103`.  `MR_AEQBM` and BM both have
sum_m=37 at `s=2`; their paired comparison tests whether merely increasing
MR's total compression is sufficient without a coefficient sweep.

That amplitude control completed at 0.783333, below BM's 0.837500 by 5.42pp.
It rules out total log-compression area alone as a sufficient explanation, but
it is not a pure filling-shape contrast: multiplying the whole MR exponent
table by `111/103` also changes the low-frequency endpoint from exponent `1`
to `111/103` (effective scale `s^(111/103)`).  The exact clean control is
`UNI(s=2)`: its middle increments are constant rather than BM's symmetric
taper, while both operators retain endpoints 0/1 and have sum_m=37.  That
single endpoint-and-area-matched comparison is now running on the same rows.

## Mechanism behind the clean shape contrast

Write `q=0,...,n` for the middle-band coordinate.  BM assigns increment
weights `w_k=k(n+1-k)`, so its cumulative exponent has the closed form

```
m_BM(q) = q(q+1)(3n+2-2q) / [n(n+1)(n+2)].
```

In the continuum this is the cubic smoothstep `3x^2-2x^3`.  It is the unique
lowest-degree curve satisfying both endpoint values and zero endpoint slopes:
`m(0)=0`, `m(1)=1`, and `m'(0)=m'(1)=0`.  MrRoPE's limiting `x^2` is smooth at
the native/high-frequency join but has a slope jump where it meets the fully
scaled low-frequency plateau; UNI's `x` has a slope jump at both joins.  BM is
therefore not an arbitrary table: it removes both join shocks in the local
log-radix increment while preserving the native and fully-scaled plateaus.
Equivalently, that cubic is the minimizer of the bending energy
`integral_0^1 (m''(x))^2 dx` under those four boundary constraints.  This is a
construction rationale for the operator, not a claim that bending energy is
itself the task loss; BM > UNI is the falsifiable behavioral consequence being
tested.

The symmetry `w_k=w_(n+1-k)` implies `m_BM(q)+m_BM(n-q)=1`, hence its exact
middle-band area is `(n+1)/2`, identical to UNI's linear ramp.  Consequently
BM versus UNI at fixed `s`, gain, band, checkpoint, and rows isolates the
two-boundary taper from endpoints and total log-compression.  The preregistered
direction is BM > UNI on the equal-task partial-score macro; per-task changes
are descriptive rather than a selection rule.

While that development contrast runs, a fresh 16K V panel is being generated
from an independent seed and disjoint stage/question range.  If the development
direction survives, the confirmatory run will compare BM, UNI, and the proper
MR(s=2) baseline on those unseen rows; no amplitude or shape sweep is added.

The development contrast completed with UNI=0.697396 versus BM=0.837500, a
+14.01pp BM gain.  At row level BM had 17 wins, 3 losses, and 44 ties.  Because
the operators share the same band, gain, endpoints, and exact sum_m=37, this is
the first clean evidence in this reroute that the two-boundary taper itself
matters on the 16K `s=2` regime.  It remains development evidence on reused S
rows.  The fresh V panel completed independently with 128 rows (16 per task),
and the locked BM/UNI/MR(s=2) confirmation is now running on it.

On that fresh V panel, the preregistered clean shape contrast confirmed:
BM=0.791406 versus UNI=0.637500, a +15.39pp gain.  BM had 35 row wins, 8 losses,
and 85 ties; a task-stratified row bootstrap gave a descriptive 95% interval
of [+9.30,+21.58]pp.  Six of eight task macros improved and two tied.  Thus the
BM taper effect at 16K `s=2` replicated on unseen rows and is not attributable
to endpoint choice, gain, total exponent area, or one task.  The paired MR(s=2)
arm on the same V rows is still running; until it completes, this is a confirmed
mechanism contrast rather than the final candidate-versus-MrRoPE verdict.

The fresh V comparison is now complete.  BM=0.791406 versus MR(s=2)=0.271484,
a +51.99pp candidate gain; BM had 78 row wins, 4 losses, and 46 ties, with a
task-stratified row-bootstrap 95% interval of [+44.48,+59.32]pp.  Seven task
macros improved and one tied.  Together with BM=0.791406 versus UNI=0.637500,
this independently confirms both the practical improvement over the correct
scale-specific MrRoPE baseline and the narrower two-boundary-taper contrast on
Llama-3-8B at 16K `s=2`.

The preregistered transfer branch has fired.  A fixed Qwen2.5-3B-Instruct
comparison is running at its native 32K and extended 64K lengths with `s=2`:
MrPro, BM, and UNI only.  Its 36-row panel uses a new seed and QA offset; the
checkpoint, inputs, tables, runner, and scorer passed their pinned identity
checks before launch.  This is the cross-model test of the prediction that the
BM taper helps in the modest exact-extension regime; the previous Qwen 128K
`s=4` null/loss remains valid and is not being overwritten.

## Intake from `RoPE_Theory_Driven_No_Search_Protocol_20260911.md`

The protocol's useful principles are adopted: one official task endpoint per
generation, exact cache reuse, BM-versus-UNI as the fixed same-area mechanism
contrast, independent confirmation only for a locked method, and a strict
separation between EVQ learning-time evidence and frozen three-band deployment.
Its older `Llama-only` execution boundary and default fixed `DEPLOY4` queue are
not treated as current authorization or as a command to rerun a large matrix.

The current scale curve shows that absolute context length is not the only
variable.  On Llama's length-matched deployments, BM beats MR at `s=2`/16K but
loses at `s=4`/32K and `s=8`/64K.  On the first Qwen `s=2`/64K development
panel, however, BM is +3.26pp over MR even though the absolute length is 64K.
The working explanation is therefore a relative-scale interaction: BM applies
more middle-band compression than MR.  At `s=2` this can repair insufficient
reach; as `ln(s)` grows, the same exponent difference becomes a larger phase
perturbation and can trade away local/mid-frequency resolution.  The slow
plateau still scales by the full `s`, which is why MR can support 16x while its
more conservative middle band protects learned local computations.

The Llama comparisons give an exact, useful decomposition.  At `s=2`/16K on
fresh V rows, UNI-MR is +36.60pp and BM-UNI is +15.39pp, summing to BM-MR
+51.99pp: both more total compression and BM's taper help.  At `s=4`/32K on S
rows, UNI-MR is -21.74pp while BM-UNI is +18.88pp, summing to BM-MR -2.86pp:
the taper still helps relative to equal-area UNI, but the larger compression
area now costs more than it gains.  This is the mechanism-derived opening for
one new arm, not a coefficient search.

### Locked next Llama arm: `MR_AREA_SMOOTH`

For middle-band increments `k=1,...,n`, use the discrete
`BetaBinomial(n-1, alpha=4, beta=2)` mass

```
w_k proportional to k(k+1)(k+2)(n+1-k).
```

Its mean increment index is `(2n+1)/3`, so its cumulative exponent area is
exactly `(n+2)/3`, identical to MR.  Its continuum CDF is `5x^4-4x^5`, with
zero slope at both endpoints.  The integers `(4,2)` are the smallest beta
parameters greater than one whose mean is `2/3`; no parameter was fitted to a
score.  This gives the exact MR area and plateaus while tapering both joins.

Prediction: on Llama `s=4`/32K, it should exceed MR if BM's same-area advantage
over UNI is genuinely a boundary-taper effect and BM's net loss is caused by
its larger area.  Failure rule: if it does not beat the cached MR macro on the
fixed S rows, discard this construction; do not vary beta parameters.  The
operator is implemented, its 93 local tool tests pass, and the remote dry run
reports `sum_m=34.33333333333333`, endpoints 0/1, finite phases, and a distinct
frequency hash.  It remains an implemented analytic branch, but is not in the
active queue while the newer window-coverage question is primary.

The protocol's EVQ E3 matched-scaling factorial is scientifically valuable,
but no 432M checkpoints are present on the current GPU host.  It is therefore
an asset-blocked later experiment, not a reason to leave this paid GPU idle.
The large Y1/Y2 and PAPER16 matrices are also deferred: current matched-scale
method evidence and independent transfer are more decision-relevant than
rerunning a broad fixed-4 or fixed-16 catalogue.

## New primary question: anytime coverage for accumulating agent context

The next research objective is not maximum-length leaderboard performance.  An
agent begins near the native window and accumulates context over many decisions;
an extreme-horizon table is useful only if its whole prefix trajectory remains
usable.  For a base window `W`, fixed maximum demand `S`, method `M`, and input
ratio `r=L/W`, define the fixed-from-token-one curve

```
A_M(r; S) = task macro at length rW using the table M(scale=S).
```

The primary development utility is the equal-milestone area
`U_M(S)=mean_{r=1,...,S} A_M(r;S)`.  Also report the native tax at `r=1`, the
endpoint score at `r=S`, and every pointwise delta versus both MR and YaRN.  A
strong "gentler throughout" claim requires nonnegative evidence at every
milestone, not an average that hides one collapsed region.  In a later real
agent benchmark, equal milestone weights must be replaced by the observed
distribution of agent decisions over context length and supplemented with
closed-loop trajectory success.

This fixes one application parameter--the maximum required horizon--and lets
the frequency rule compete.  If the deployment requirement is `S=6`, the fair
primary baselines are MR(scale=6) and YaRN(scale=6), also fixed from token one.
MR(scale=16) is a target-mismatch reference that tests the criticism, not the
fair main baseline.  A per-length oracle that changes `s` and rebuilds history
is also not a free deployment baseline: standard cached hidden states were
computed under the old operator, so exact switching requires a prefix refill.

Existing local evidence already refutes the proposition that fixed MR(scale=16)
is optimal at all shorter ratios.  On the same S rows, its 8K score is 0.679688
versus Native 0.867708; at 16K, tested MR(scale=4)=0.794010 exceeds fixed-16
MR=0.571875; at 32K, MR(scale=4)=0.569010 exceeds fixed-16 MR=0.532031.  On the
same compact 64K panel, MR(scale=8)=0.545313 exceeds fixed-16 MR=0.460417.
These are local eight-task development comparisons (the 64K cell uses the
separate compact panel), not a full RULER-13 or agent-trajectory result.

The claim is specifically target-scale mismatch, not that `scale=input_ratio`
is always optimal.  At 16K, MR(scale=2)=0.260417 is worse than fixed-16 MR,
while MR(scale=4)=0.794010 is best among the tested MR tables.  Therefore the
scientific question is whether one extreme fixed table is dominated along its
prefix path, and what gentler fixed horizon gives the best coverage--not a
naive rule that changes `s` to the current length.

Before `ANYTIME_S6_D`, complete the triangular MR scale-policy matrix on the
existing rows.  Already available cells are retained; only MR(scale=2) at 8K
and MR(scale=8) at 8K/16K/32K are missing.  Together with the cached scale-1,
scale-4, scale-16, and 64K scale-8 cells, this yields:

```
input ratio 1x: MR scales 1,2,4,8,16
input ratio 2x: MR scales   2,4,8,16
input ratio 4x: MR scales     4,8,16
input ratio 8x: MR scales       8,16
```

This is a response map for MrRoPE's explicit deployment parameter, not a
frequency-formula search.  It tests whether fixed scale-16 is optimal or
systematically pays sub-horizon regret and whether the best scale has a stable
safety margin above the realized input ratio.  The scale-2 and scale-8 missing
cells are dry-run validated and will execute automatically after the active
Qwen confirmation, before the fixed-S=6 curve.

The triangular matrix is now complete and gives a coherent development signal:

| input | tested MR table scales and macros | best scale | fixed-16 regret |
|---|---|---:|---:|
| 8K (1x) | s1 .867708, s2 .920833, s4 .781250, s8 .675000, s16 .679688 | 2 | 24.11pp |
| 16K (2x) | s2 .260417, s4 .794010, s8 .707552, s16 .571875 | 4 | 22.21pp |
| 32K (4x) | s4 .569010, s8 .597917, s16 .532031 | 8 | 6.59pp |
| 64K (8x) | s8 .545312, s16 .460417 | 8 | 8.49pp |

At 1x, 2x, and 4x the best tested MR table has a twofold safety margin
`s_table=2r`; at 8x the best tested cell is s8 rather than s16.  Fixed s16 is
not the best tested MR configuration at any sub-horizon and pays 6.59--24.11pp
regret.  This is a striking but still small-panel development result.  It
supports studying a finite-horizon safety-margin rule and whole-curve static
method, not selecting the pointwise envelope as a deployable policy.

### Locked development experiment: `ANYTIME_S6_D`

- checkpoint: the same original Meta-Llama-3-8B-Instruct;
- fixed maximum: `S=6`, so the deployment table never changes during a run;
- milestones: 8K, 16K, 24K, 32K, 40K, and 48K;
- methods: MR, BM, and public index-YaRN, all with the same scale-6 gain;
- native reference: Native only at 8K;
- data: eight existing RULER-derived tasks, four fresh rows per task/milestone,
  one generation per method and row;
- primary: equal-milestone/equal-task partial-score macro; secondary reporting
  is every milestone, every task, and paired wins/losses;
- role: development coverage curve only.  The rows are independently generated
  per milestone, not a literal growing agent history.

The fixed prediction is BM has higher trajectory utility and smaller early
tax than fixed MR/YaRN while retaining a usable 48K endpoint.  If it fails to
beat both baselines across the curve, do not tune `S` or its shape from these
scores.  If it succeeds, construct a separate nested-prefix agent H split whose
same scenarios accumulate tool/history blocks across milestones; only that
split can support the agent-trajectory claim.

`ANYTIME_S6_D` is complete:

| fixed scale-6 policy | 8K | 16K | 24K | 32K | 40K | 48K | equal-milestone mean | 8K tax vs Native |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MR | .732812 | .803125 | .618750 | .679688 | .746354 | .551562 | .688715 | -11.72pp |
| index-YaRN | .748438 | .828646 | .667188 | .604688 | .729167 | .596875 | .695833 | -10.16pp |
| BM | .806250 | .854688 | .693750 | .768750 | .722917 | .467188 | .718924 | -4.38pp |

Native on the new 8K rows is .850000.  The originally reported .718924 is the
equal-milestone arithmetic mean, not the log-length trapezoidal AUC proposed in
the later Agent-Range specification.  Recomputed over 8K--48K, the log-length
AUCs are MR .717043, YaRN .726928, and BM .767314, so BM improves by +5.03pp
and +4.04pp respectively.  BM also has much smaller native-window tax.
It beats both baselines at every milestone through 32K.  But it loses slightly
at 40K and collapses relative to both at the declared 48K endpoint: -8.44pp
versus MR and -12.97pp versus YaRN.  Thus the intended whole-curve method has
not yet been achieved.  The result isolates the method-design target: retain
BM's 8K--32K coverage while repairing the 40K--48K tail, rather than maximizing
another average or reducing the claimed horizon after seeing the result.

Because all three policies share the same scale-6 gain, their pairwise curve
differences are frequency-table effects.  The common 8K tax can still include
gain.  The queued scale/gain factorial is now running and will determine
whether the final static construction should change the common temperature,
the middle profile, or both.  BM's endpoint loss despite its larger exponent
area rules out the simple story that more compression alone fixes the tail.

## Review of the supplied Pro analysis on anytime coverage

The analysis gets the research object right: distinguish realized input ratio
`r`, table-construction scale `s`, and maximum deployment horizon `S`; optimize
an interval/trajectory utility rather than an endpoint; include dependency
distance as well as total input length; and treat a fixed table and a dynamic
cache-aware policy as different algorithms.  Its warning that BM is smoother
but not gentler is exactly correct: BM has a larger exponent than MR at every
internal slot.

Two statements are stale relative to completed local evidence.  First,
MR(scale=2) at 16K is already known: 0.260417, far below MR(scale=4)=0.794010.
Second, the newly completed MR(scale=2) 8K cell is 0.920833, above the cached
Native 0.867708 on this S panel.  Thus neither `s=r` nor `s=1` is automatically
the empirical optimum on a small task panel; the scale-policy response must be
mapped before proposing a rule.

The proposed Fisher/KL-weighted convex program is a legitimate later research
direction, but it is not an analytic zero-calibration method.  It uses task
data to estimate per-length preferred configurations and sensitivity matrices,
and local output KL is still a proxy for generated task success.  It therefore
must be labeled a calibrated static optimizer, use a separate confirmation
split, and cannot precede the cheaper scale/gain attribution now queued.

The literature-neighbor warning is also correct.  YaRN already describes
dynamic scaling for progressively growing autoregressive sequences and warns
that RoPE-aware KV caching needs special handling.  CLEX learns continuous
frequency dynamics across sampled length scales, so it is a training-based
neighbor rather than a frozen-static baseline.  Jet-Long is the closest recent
deployment neighbor: it preserves a native local view, dynamically remaps the
remote view, and corrects cached rotations on the fly.  Therefore the novelty
cannot be “first to care about the whole length range.”  The intended claim is
narrower: a single frozen static exponent allocation with standard incremental
inference that improves coverage under a declared finite horizon, followed by
comparison with a real dynamic baseline on quality and cost.

## Review of the supplied Agent-Range RIBB execution specification

The specification is valuable as a research contract.  In particular, it
correctly separates current ratio `r`, table factor `s`, and declared horizon
`S`; keeps one static table and stable KV-cache semantics; embeds Uni `(1,1)`,
MR `(2,1)`, and BM `(2,2)` in one interpretable beta-increment family; requires
matched-gain attribution; and reserves held-out and nested Agent trajectories
for claims that actually need them.  The Qwen reversal makes its calibration /
confirmation separation especially important: a 36-row development panel had
BM above MR at 64K, but the independent 108-row confirmation reversed that
ordering.

It is not executable unchanged.  Its result anchors predate the completed
matrix: MR(scale=2) at 16K is known to be .260417, and the best tested 32K MR
cell is scale=8 at .597917, making scale-16 regret 6.59pp rather than the
document's 3.70pp comparison against scale=4.  The current S=6 panel is a
192-row development panel at 8/16/24/32/40/48K; it does not contain the
specification's 2K/4K/12K retention cells and cannot become held-out evidence
after it selects a candidate.

The proposed sequential `profile first, gain second` optimization is also too
strong an assumption.  The completed 8K factorial cells `(frequency scale,
gain scale)` are `(1,1)=.867708`, `(16,1)=.690625`, `(1,16)=.537500`, and
`(16,16)=.679688`, giving a large crossover interaction of +31.93pp.  Matched
gain is still mandatory for a frequency-only claim, but the selected profile
must receive a bounded profile-by-gain interaction check before freezing the
complete deployment configuration.

RIBB is zero weight training, but its proposed NLL/KL screen and generation
selection make it a *calibrated* static method, not a universal analytic or
zero-calibration rule.  Its two-parameter family is therefore a useful fallback
after a cheaper falsification, not a reason to build a result cube, twelve-cell
screen, and Agent harness before table feasibility is known.  The grid should
also include exact `mu=2/3`, not decimal `.667`, or the `alpha>=1,beta>=1`
filter can accidentally exclude exact MR through rounding.

The observed cross-model difference does not yet identify native window length
as the cause.  The current implementation already recomputes the correction
band from each checkpoint's `(W, theta, head_dim)`, but it holds the normalized
BM shape `(alpha,beta)=(2,2)` fixed.  Llama-8K and Qwen-32K also differ in theta,
training distribution, architecture, tokenizer, and evaluation rows.  The
correct hypothesis is therefore that band geometry alone may be insufficient
and that checkpoint-conditioned shape or gain may be needed; it is not yet that
shape must be a function of `W` alone.  A direct low-cost interaction test is
the one-dimensional exact bridge `alpha=2, beta in {1,1.5,2}` on both models:
MR and BM are reusable endpoints, so only the `beta=1.5` cell is new per model.

All current range numbers remain development evidence.  The Llama panel has
only four rows per task and length, independently generated across milestones;
it omits 2K/4K/12K, full RULER, natural QA, and nested Agent histories.  The
Qwen confirmation has no untouched Native arm.  These panels can choose which
mechanism deserves a held-out run, but cannot establish native preservation,
agent suitability, universal transfer, or a final paper claim.  Any surviving
candidate needs a new prompt split, adequate paired uncertainty, raw full
generations with EOS/cap metadata, broader task coverage, and then same-history
Agent checkpoints.

The next preregistered candidate is consequently `MR_AREA_SMOOTH(s=6)`: it
matches MR's discrete exponent area while making both increment boundaries
smooth.  It directly tests whether BM's early/mid gain can be retained without
its excess cumulative displacement and 48K collapse.  It runs on the frozen D
panel after the active factorial.  A positive full curve advances to a new
held-out split; a negative result justifies the bounded RIBB family, rather than
an immediate twelve-profile sweep or a claim that static range optimization is
impossible.

### Measurement-to-method chain

1. The MR triangle estimates sub-horizon regret and whether a stable safety
   margin exists between realized `r` and a useful table scale.
2. `ANYTIME_S6_D` identifies where each fixed table loses along the trajectory.
3. A preregistered scale-frequency x gain-scale factorial separates table
   distortion, attention-temperature distortion, and their interaction.  At
   8K it uses `{frequency scale 1,16} x {gain scale 1,16}`; at 16K/32K it uses
   `{frequency scale 4,16} x {gain scale 4,16}`.  Existing corners are reused;
   only four mixed controls are generated.
4. The result selects a mechanism branch, not a score-ranked curve.  If gain
   dominates, define one fixed trajectory-calibrated gain by
   `log(s_gain)=sum_i w_i log(r_i)` for preregistered Agent milestone weights,
   while preserving the endpoint frequency table.  If frequency placement
   dominates, construct one endpoint-constrained coverage allocation; the
   existing MR-area smooth table is one analytic diagnostic, while a
   Fisher-fitted table would be a separately labeled calibrated method.  If the
   factors interact strongly, run one necessary joint 2x2 before combining.
5. The single derived method returns to the same S=6 D curve.  Only a method
   with improved interval utility, no hidden prefix collapse, and usable 48K
   endpoint advances to nested fixed-history replay and then closed-loop Agent
   trajectories.  A dynamic YaRN-style policy or Jet-Long-like implementation
   is a final systems baseline, not a substitute for the static-method test.

This sequence explicitly turns measurement into construction and falsification;
it does not stop at describing that fixed scale-16 is suboptimal.

### Completed scale-frequency x gain-scale attribution

The factorial is complete on the aligned S rows:

| input | extreme/extreme | extreme/gentle | gentle/extreme | gentle/gentle | best tested cell |
|---|---:|---:|---:|---:|---|
| 8K | f16/g16 .679688 | f16/g1 .690625 | f1/g16 .537500 | f1/g1 .867708 | f1/g1 |
| 16K | f16/g16 .571875 | f16/g4 .625000 | f4/g16 .748958 | f4/g4 .794010 | f4/g4 |
| 32K | f16/g16 .532031 | f16/g4 .574219 | f4/g16 .410677 | f4/g4 .569010 | f16/g4 |

At 16K, changing f4 to f16 costs about 17pp under either gain while changing
g4 to g16 costs 4.5--5.3pp; the interaction is only -0.81pp.  At 8K and 32K,
however, interactions are +31.93pp and +11.61pp respectively.  At 32K the
extreme frequency table paired with gentle g4 is slightly better than f4/g4,
while high gain makes f4 collapse.  There is consequently no length-invariant
claim that either frequency or gain alone dominates.  A frequency-only
candidate remains valid only at matched gain, and any selected profile needs a
small profile-by-gain check before the complete fixed configuration is frozen.

The post-factor candidate supervisor transitioned directly into
`MR_AREA_SMOOTH(s=6)` on the 192-row D curve.  This first run deliberately keeps
the common g6 gain so it isolates frequency allocation against the existing
MR/BM/YaRN arms.  A gentler trajectory-level gain is considered only after the
profile result, with interaction explicitly tested rather than assumed away.

`MR_AREA_SMOOTH(s=6)` is complete and decisively refutes its prediction.  Its
8/16/24/32/40/48K macros are
`.734375/.796875/.600000/.695833/.685417/.031250`; the equal-milestone mean is
`.590625` and log-length AUC is `.680737`, both below MR, YaRN, and BM.  The 48K
failure is broad rather than one-task noise: seven of eight task macros are
zero and only FWE is nonzero at `.25`.  Matching MR's total exponent area while
smoothing both increment boundaries is therefore not sufficient; the late-
concentrated placement is a candidate explanation for a length-specific
instruction/query readout failure, not yet an isolated cause.

The next minimal RIBB test is now running at matched g6:
`RIBB_A2_B1P5`, with `alpha=2,beta=1.5`.  It lies pointwise strictly between MR
`(2,1)` and BM `(2,2)` on every internal transition slot, so both endpoints are
reused and only one new cell is required.  Its falsifiable target is a genuine
MR--BM tradeoff: retain enough of BM's 8K--32K gain while moving the 48K result
toward MR/YaRN.  If this exact bridge cannot improve the useful frontier, do
not expand to the full twelve-profile grid merely because it is available.

`RIBB_A2_B1P5(g6)` is complete.  Its 8/16/24/32/40/48K macros are
`.756250/.871875/.679688/.694271/.744792/.036458`; equal-milestone mean is
`.630556` and log-length AUC is `.730135`.  It improves several mid-range
points but collapses at 48K almost as completely as `MR_AREA_SMOOTH`.  Because
its exponent profile is pointwise strictly between the non-collapsed MR and BM
profiles, task quality is not a convex or even monotone interpolation in this
coordinate; a wider beta grid would therefore be score search, not a justified
smooth tradeoff.

The necessary 48K interaction check with the identical
`alpha=2,beta=1.5` frequency table and unit Q/K gain is complete: task-equal
partial and strict are both zero, with EOS rate `.50`.  Lowering g6 to g1 does
not rescue the endpoint on these rows, so a simple common-temperature fix is
ruled out for this profile.  Frequency attribution remains provisional until
MR and BM reproduce at 48K under the current runner/evaluator hash; that exact
same-panel audit is queued after the active Qwen interaction cell.

The failure is behavioral, not merely an EOS formatting miss.  RIBB g6 emits
background-like continuations or invented prose on the failed 48K queries;
its strict and full-string-plus-EOS rates are both zero, while EOS is `.8125`.
The metrics also answer different claims on the baselines: at 48K BM has
partial `.467188`, cap rate `.34375`, and full-string-plus-EOS `.15625`, versus
MR `.551562/.0625/.25` and YaRN `.596875/.03125/.28125`.  At 8K BM's partial
`.80625` is below Native `.85`, yet its full-string-plus-EOS `.53125` is above
Native `.375`.  Thus the partial-score “native tax” cannot be generalized to
every output contract, and all metrics must remain separately named.

The matched-gain Qwen scale-2 `alpha=2,beta=1.5` interaction cell is complete
against the existing 108-row inputs, with the exact MrPro result reused.  At
32K, RIBB is `1.000000`, versus MR `.888889`, BM `.916667`, and UNI `.983333`;
this cell has only two rows per task, so the apparent lead is a development
signal rather than a reliable endpoint estimate.  At 64K, RIBB is `.840625`,
essentially tied with MR `.844271` (-0.36pp) while exceeding BM `.824132`
(+1.65pp) and UNI `.814583` (+2.60pp).  RIBB versus MR has 8 wins, 8 losses,
and 80 ties at 64K.  Its 64K task deltas against MR are mixed: multikey -6.25pp,
multiquery +1.56pp, VT +2.50pp, with single-key, FWE, and QA tied.  EOS rates
are `.7500/.8646` at 32K/64K, versus MR `.8333/.8750`, BM `.8333/.8438`, and
UNI `.9167/.8646`.

This rules out a general statement that the bridge itself fails on Qwen, but it
does not provide a long-context gain over MR or identify native window length
as the cause of the Llama/Qwen difference.  It remains a development
model-by-profile interaction: the reused panel has already been observed and
still lacks an untouched Native arm.  The queued current-runner 48K Llama
MR/BM identity audit has now started; the profile-by-gain endpoint supervisor
will proceed only if that audit reproduces the frozen rows and scores.
