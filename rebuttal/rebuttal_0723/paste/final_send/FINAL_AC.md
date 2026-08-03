Three items did not fit the character limit of our first response: the condition-to-evidence map, the 8B mechanistic attribution, and the seed accounting. All are material the metareview asks for; none is new evidence.

**A. The three conditions, in one table.**

| Metareview condition | What we ran | Result |
| --- | --- | --- |
| 1. Novelty over FMRoPE | FMRoPE moves the band, b → b(T); EVQ-Cosh redistributes channels inside it, u_i → φ_τ(u_i) | FMRoPE's abstract: shifts the band toward lower frequencies, exponents uniform in i; the interior distribution is what we change |
| 2. Direct controlled comparison | Exact-range control + three-seed factorial: sampled extrema and log span pinned, only 30 interior frequencies changed | OOD NLL **0.478/0.205/0.113** at 512/1K/2K; **32/32, 27/32, 22/32** anchors; direction holds in 10/12 configurations |
| 3. Stronger evaluation | Matched 1.485B/8B chains: strict generation, 13-family RULER, real-document QA; plus the pre-specified 1.485B from-scratch comparison | 8K strict exact **18/100 → 98/100**; 8B 16K RULER macro **0.295% → 14.03%**; 2Wiki 8K exact **0% → 17.5%**; from-scratch NLL **+0.0381** at 4K, **−0.0437/−0.1351** at 8K/16K |

The levels differ in what is parameterized and at what stage, not in what is numerically achievable — which is why the exact-range control, not the formula, is what settles the question.

**B. Evidence that the gain is remote-content use, not local reconstruction.**

This is the part of "stronger evaluation" we had to cut — the strongest mechanistic result we have at scale. In the matched 8B adaptation, Native/EVQ NLL is 1.919/2.309 at 8K, **4.691/3.181** at 16K and 6.899/4.851 at 32K, with the 16K and 32K direction holding in **all 24 evaluation packs and all three domains**. At true 16K, deleting the remote gold block from every head worsens EVQ NLL by **1.506** while leaving the matched Native control essentially unchanged, and median target-block hit@16 rises from **18.75% to 64.06%**.

A generation endpoint can be met by a model that has learned the task format. A model that degrades sharply when the distant block is removed, and that retrieves it more often, is using it. Those two readouts point the same way here.

**C. Sampling, stated plainly.**

Three agreeing seeds: the fixed-schedule ladder, the exact-range factorial, the base/head-dimension controls, and the 432M MLA study (PPL@16K **138.8±5.5 → 95.6±4.1**, 16 rotary channels). One training seed per arm, arms matched: the 1.485B and 8B adaptations — a second independently trained EVQ seed reproduces the OLMo strict-retrieval result (69/100 and 67/100 at 8K against Native 0/100). One trajectory: the 1.485B from-scratch branch. Submitted breadth at these tiers: the 750M continuation (8K strict autoregressive exact **0% → 77.5%**, Table 12).

The claim is stated at its most precise level: training-time exponent allocation is a separately identifiable design axis whose effect survives to 8B in free-running generation and reaches real-document multi-hop QA. The boundaries — no universal Cosh optimality, τ as an operating convention, FMRoPE stronger under retargeting — are stated in our response and unchanged.
