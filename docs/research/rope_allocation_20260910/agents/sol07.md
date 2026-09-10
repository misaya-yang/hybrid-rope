# Sol07 — nongeometric experiment audit and predictive allocation rule

## Decision

The screen does not support a geometry-only allocation rule. The measurable computation that best explains a real win is the signed answer-decision margin under the actual frozen model, decomposed into prefix-formation and readout contributions. In the slot-28 multikey case, the Mr/Mr, Mr/E1, E1/Mr, E1/E1 correct-minus-competitor margins are -2.125, -1.125, -1.000, +0.250 nats. Prefix formation contributes +1.125, readout +1.000, and the factorial remainder only +0.250. Nearly additive continuous changes cross a discrete greedy boundary.

The constructive rule is to allocate log-frequency gaps by cross-fitted marginal decision utility. MrRoPE supplies the frozen deployment baseline; EVQ supplies a from-scratch initialization or regularizer. Gain remains an independent amplitude coordinate.

## Actual experiment audit

All Qwen3B task numbers below are historical development evidence unless marked transfer. The runner verifies model, input, configuration, and baseline identities and records candidate contracts (worker.py:53-102,147-200). This makes comparisons paired, not independently validated.

| Intervention | Outcome | Supported reading |
|---|---|---|
| E1 slot 28, predecessor exponent | 32K tie; 128K 78.125 to 83.333%, 2 wins/0 losses | Local headroom. Multikey has a continuous margin explanation involving prefix and readout; multiquery is readout-sensitive. |
| E1 slot 29, successor exponent | 32K +8.333 pp; 128K -0.208 pp, 2/1 | Short gain is one QA row; long VT gain and multiquery loss nearly cancel. |
| E1 pair 28+29 | 32K tie; 128K -4.167 pp, 0/1 | Individual gains do not compose. Utilities must be recomputed after every edit. |
| MrPro gain .074 | 32K 98.333%; 128K 75.347%, 4/2 | Gain has a large length-dependent effect and changes amplitude, not phase arcs. |
| BM gain .074 | 32K 100%; 128K 70%, 6/4 | Against same-gain MrPro, table effect is +1.667 pp short and -5.347 pp long. |
| FullLagP2, gain .074 | 32K 72.917%; 128K 81.667% | Long gain is QA/VT with multikey/FWE cost. It beats official MrPro +3.542 pp long and loses -14.306 pp short. P2 changes high gaps negligibly and creates a sharp gap around 29, so it does not support EVQ-style high-gap donation. |
| E7 local projection | 32K +2.778 pp; 128K -9.514 pp, 6/7 | Local-output preservation failed as downstream selector. Ideal finite math is sound; deployment arithmetic differs. |
| E8 zero slot 51 | 12-row long subset -13.889 pp | Strong conditional target-mass score did not predict generation. |
| LongBridge slower/faster | slower -6.667/+1.944 pp; faster 0/-4.167 pp | Direction matters on this panel. Slower is VT-specific with FWE cost and leaves within-band gaps unchanged. |
| Smooth(MrBudget) | 32K tie; 128K -9.792 pp | Minimum roughness at fixed MrPro budget is insufficient. |
| HighGapToLong | -17.083/-10.764 pp, 0/7 | This donation quantum and recipient construction fails. EVQ training theory is untested. |
| Layer/group/E9/E10 initial screens | current summaries are 12-row ties | Non-results for ranking; partial-panel macros cannot be mixed with 36-row macros. |

Frozen cross-model slot-28 transfer is heterogeneous. Qwen7B ties at 32K and loses 1.667 pp at 128K (1/1). OLMo1B loses 3.333 pp at 4K and gains 3.819 pp at 16K (4/2), from a low absolute baseline. Checkpoint-relative recalibration is the defensible hypothesis. No completed independent holdout summary exists in the local mirror.

## What computations explain ordering

For target answer token y* and actual competitors C, define

\[
M_t(\nu,g)=\ell_t(y^*;\nu,g)-\log\sum_{c\in C}\exp \ell_t(c;\nu,g).
\]

On a gold-prefix calibration trajectory, a finite candidate a has delta margin
\[
\Delta M_{e,t}^a=M_{e,t}(\nu^a,g)-M_{e,t}(\nu^0,g).
\]
This has the right sign and includes values, output projection, and later layers through final logits. Free generation remains the outcome. It currently explains one winner case; predictive ordering across unseen examples is still untested.

Prefix/read crossovers define P=M(a,0)-M(0,0), R=M(0,a)-M(0,0), and I=M(a,a)-M(a,0)-M(0,a)+M(0,0). The positive multikey case gives P=1.125, R=1.000, I=0.250. This is the strongest current mechanistic measurement.

The conditional selector in select.py:24-68 reconstructs selected-key logits while holding omitted mass/output and upstream states fixed. Its aggregation averages task/length cells and takes the worse split (select.py:96-108). It nominated both E1 winners, but also E2/E8 losers, layer/group ties, and cannot explain the bad E1 pair. It is a proposal filter, not an objective.

NLL is a preservation guard. E1 changes tasks with sub-millinat NLL shifts, and BM/P2 rankings change when gain is matched. Equal-weight chord separation is periodic and nonmonotone; slot 28 wins while reducing it at three of five audited distances. Smooth(MrBudget) improves geometry and loses long tasks. HighGapToLong preserves endpoints and moves a declared budget but loses everywhere. Geometry and budget are descriptors or constraints, not utility.

E7 cleanly separates implementation failure from theory failure. Linear predicted NMSE is 7.9366844e-8; exact ideal finite relative-phase is 7.9364028e-8; FP32 absolute-coordinate is 7.9325473e-8; BF16 absolute-coordinate is 1.2807392e-5. The derivative predicts the ideal intervention; BF16 arithmetic produces about 161 times squared error, or 12.7 times amplitude. Downstream loss separately shows the protected local object is not sufficient. See local_precision.py:20-50.

The original old_probability times exp(delta) normalization could underflow. Log-weight normalization at select.py:49-58 repairs it and changes E5 runner-up from layer 27 to 32. Layer-27 outputs remain valid outcomes of an invalidly ranked proposal. This is selector implementation failure, not retroactive result corruption.

## Constructive predictive rule

Parameterize a monotone table by log gaps
\[
x_j=\log\nu_j-\log\nu_{j+1}\ge0,\qquad\sum_jx_j=R.
\]
Fix endpoints or total range R when required. A transfer a=(i to j, eta) removes eta from donor gap i and adds it to recipient j, then reconstructs frequencies cumulatively.

For frozen deployment estimate
\[
\widehat U_a=Q_{0.2,e}\left[\sum_t w_{e,t}\Delta M^a_{e,t}\right]
-\lambda_N[\Delta NLL_a-\epsilon_N]_+
-\lambda_S Regress_a
-\lambda_B BF16Mismatch_a.
\]
Group examples by task and physical relation distance. The lower quantile prevents one threshold crossing from dominating. Regress penalizes negative margins on baseline-correct examples. BF16Mismatch is the measured ideal-versus-deployed discrepancy. Use complete forward passes for a small grounded set of adjacent-step transfers; conditional replay only pre-orders them. Select on folds A/B and require the sign to predict fold C.

Algorithm:

1. Start from target-checkpoint MrPro, keeping validated endpoints and gain.
2. Form finite donor-to-recipient transfers sized by an existing adjacent exponent step or declared physical phase quantum.
3. Rank by cross-fitted utility, not target mass, NLL, chord distance, or roughness.
4. Apply the best positive transfer and recompute. The failed 28+29 pair forbids additive composition.
5. Stop when no transfer has positive lower-confidence utility. Freeze and test untouched examples.

## From scratch versus frozen deployment

Use the same gap coordinates but different marginal utilities. From scratch:
\[
J_{\rm train}(x,g)=\min_W E_{\mathcal D_{\rm train}}L(W,x,g)+\rho\Omega_{\rm EVQ}(x).
\]
Weights co-adapt. The EVQ Cosh density is an initialization or regularizer. At an interior constrained optimum, active gaps equalize marginal training value, with KKT inequalities at zero gaps. Marginals must be estimated while allowing matched weight adaptation.

Frozen deployment uses
\[
J_{\rm frozen}(x,g;W_0)=E L_{\rm decision}(W_0,x,g)+preservation+numerical\ penalties.
\]
Weights cannot absorb a cold swap, so MrPro is the safe reference and only prospectively validated small exchanges are justified. HighGapToLong failing frozen does not refute EVQ from-scratch theory; an EVQ training optimum does not predict a cold-swap win.

Unified statement: allocate log-frequency gap mass until marginal task value is equalized under the adaptation regime actually allowed. EVQ supplies an analytic training prior, MrRoPE a deployment baseline, and signed decision utility the evidence for deviations.

## Minimum decisive next experiment

Use the prepared new-seed 128K multikey, multiquery, VT, and QA cohort with calibration A/B, prediction C, and untouched confirmation partitions. Compute teacher-forced answer margins for MrPro and the two frozen single-slot edits, including regressions on baseline-correct cases. Pre-register candidate-by-task sign predictions, then test margins and free generations on C. If ordering holds, test the frozen choice untouched.

Primary question: does predicted delta margin order s28_less, s29_more, and MrPro, and does positive change precede generation gain? Controls are same-gain MrPro, deployed BF16 arithmetic, and prefix/read interventions on a few discordant cases. The completed MrPro gain-.074 arm should be analyzed, not rerun.

Success supports the constrained exchange rule. Failure means calibration-trajectory margins remain too state/task-specific; the fallback is full-forward finite task loss on coarser gap blocks, not geometry-only selection.

## Evidence pointers and coverage

- Outcomes: docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:38-47,576-630 and docs/research/PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md:15-92.
- Selector: experiments/nongeometric_screen/select.py:24-68,96-108,136-186.
- Runner: experiments/nongeometric_screen/worker.py:53-102,147-200.
- Transfer: experiments/nongeometric_screen/cross_model.py:14-31,34-57,109-112.
- Per-method values were re-read from results/nongeometric_screen_20260909/results/*/summary.json; partial and complete panels were kept separate.

Read all 38 assigned files in full. No assigned-file omissions. Additionally inspected local per-method summaries, transfer summaries, selector receipts, and E7 precision receipt. No GPU job was launched and no runtime or paper source was edited.
