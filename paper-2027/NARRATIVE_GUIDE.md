# Manuscript narrative

The central object is the RoPE exponent distribution. Define it, explain the
controlled behavior it changes, then explain its positional geometry. Present
Cosh as one explicit allocation construction, followed by adjustments of frozen
models. The fixed-range finding and weight-table crossings organize the paper.

1. Write the positive argument: question, controlled finding, analysis, design,
   interpretation. Put actual experiment conditions next to the result.
2. Use `x = a + R z` as a definition and a control for range. The contribution
   is the theory, construction, and controlled findings about exponent
   allocation; changing coordinates alone is not the research result.
3. The first figure should make the intervention and its empirical claim
   visible. Additional figures should resolve a scientific question.
4. Compare related work by mathematical operation, training stage, and
   finding. Keep the closest-work discussion compact; LongRoPE includes both
   dimension-wise scaling and its position threshold.
5. Keep method identities and metrics exact. Frequency blending, log shifts,
   and cumulative radix products have distinct formulas. Full-output token
   F1, autoregressive exact match, teacher-forced retrieval, and PPL retain
   their own names.
6. Preserve useful earlier experiments with their actual provenance. Choose
   main-text space by contribution to the argument, with full task details
   and long derivations in the appendix.

The reference-paper lessons remain simple: MrRoPE supplies a focused question;
Decoupling connects definitions to measurable behavior; Deconstructing orders
experiments by competing explanations; RePo/PPE make the first visual concrete;
GRAPE/Selective RoPE connect theory, method, and evidence around one object.
