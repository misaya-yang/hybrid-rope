Two things did not fit the character limit of our first response. Both are material the metareview asks for, so we add them here rather than leave them out. Nothing below is new evidence.

**A. The three conditions, in one table.**

| Metareview condition | What we ran | Result |
| --- | --- | --- |
| 1. Novelty over FMRoPE | FMRoPE moves the band, \\(b \\to b(T)\\); EVQ-Cosh redistributes channels inside it, \\(u_i \\to \\varphi_\\tau(u_i)\\) | FMRoPE's own abstract describes its effect as shifting the band toward lower frequencies, exponents uniform in i; the interior distribution is what we change |
| 2. Direct controlled comparison | Exact-range control and three-seed factorial: both sampled extrema and log span pinned, only 30 interior frequencies changed | OOD NLL **0.478/0.205/0.113** at 512/1K/2K; **32/32, 27/32, 22/32** anchors; direction holds in 10/12 configurations |
| 3. Stronger evaluation | Matched Native/EVQ chains at 1.485B and 8B: strict generation, 13-family RULER, real-document multi-hop QA | 8K full-answer-plus-EOS exact **18/100 → 98/100**; 8B 16K RULER macro **0.295% → 14.03%**; 2Wiki 8K exact **0% → 17.5%** |

The levels differ in what is parameterized and at what stage, not in what is numerically achievable — which is why the exact-range control, not the formula, is what settles the question.

**B. Evidence that the gain is remote-content use, not local reconstruction.**

This is the part of the answer to "stronger evaluation" we had to cut, and we think it is the strongest mechanistic result we have at scale. In the matched 8B adaptation, Native/EVQ NLL is 1.919/2.309 at 8K, **4.691/3.181** at 16K and 6.899/4.851 at 32K, with the 16K and 32K direction holding in **all 24 evaluation packs and all three domains**. At true 16K, deleting the remote gold block from every head worsens EVQ NLL by **1.506** while leaving the matched Native control essentially unchanged, and median target-block hit@16 rises from **18.75% to 64.06%**.

A generation endpoint can be met by a model that has learned the task format. A model that degrades sharply when the distant block is removed, and that retrieves it more often, is using it. Those two readouts point the same way here.

**C. Sampling, stated plainly.**

The metareview's concern about diagnostic-heavy validation deserves an explicit accounting rather than an aggregate. Three agreeing seeds: the fixed-schedule ladder, the exact-range factorial, the base/head-dimension controls, the 432M MLA study (PPL@16K **138.8±5.5 → 95.6±4.1** with only 16 rotary channels). One training seed per arm, with the arms matched: the 1.485B and 8B adaptations, with a second independently trained EVQ seed reproducing the OLMo strict-retrieval result (69/100 and 67/100 at 8K against Native 0/100). One trajectory: the 1.485B from-scratch branch. The submission also reported a 750M continuation whose 8K strict autoregressive exact goes from **0% to 77.5%** (Table 12) and 129M/382M bidirectional 3D-RoPE video-DiT results (Table 14).

**D. What we would ask the committee to weigh.**

The metareview's three conditions were the right ones, and the experiments that answer them exist because they were named. Our claim after all of it is narrower than the submission's: training-time exponent allocation is a separately identifiable design axis — identifiable under controls that fix the positional operator, parameter count, tuning budget, reference grid, deformation magnitude and spectral range — whose effect survives to 8B in free-running generation and reaches real-document multi-hop QA. We do not claim Cosh is a universal optimum, that τ = d/√L is exact, or that EVQ replaces target-aware range scaling; under retargeting the FMRoPE rule is stronger, and we report that.

The in-window trade-off is real and we report it in both settings rather than around it: **+0.0381 NLL** at 4K when training from scratch, and 13-family RULER macro **82.16% → 37.51%** when retrofitting a mature checkpoint, improving to 42.44% when adaptation is restricted to Q and K. Oka et al. report the same ordering for base selection, with FMRoPE underperforming conventional RoPE in short contexts, which suggests the trade-off is structural to this design space rather than specific to our rule.
