# Rebuttal Writing Guide for NeurIPS

## Rebuttal Principles

A rebuttal is a negotiation, not a fight. Your goal is to change a reviewer's mind by
addressing their specific concerns with evidence, not by arguing they're wrong.

### The 3 Rules

1. **Never be defensive.** Thank the reviewer for the concern, then address it factually.
2. **Lead with evidence, not argument.** "We ran the additional experiment (new Table R1)" beats
   "We respectfully disagree."
3. **Prioritize ruthlessly.** Address the concerns most likely to change scores first.

## Response Structure

For each reviewer concern, use this structure:

```
**[Concern topic in bold]**

We thank Reviewer X for this observation. [1-sentence acknowledgment that shows you understood.]

[Direct response with evidence — numbers, references, or reasoning.]

[If applicable: what you will add/change in the revision.]
```

### Length and Formatting
- NeurIPS typically allows ~5000 characters or 1 page per reviewer
- Use markdown-style bold for topic headers
- Keep individual responses to 3-5 sentences
- Include a "Summary of Changes" at the top listing all revisions

## Common Concern Types and Strategies

### "Limited scale / small models"
**Strategy**: Show scale consistency across what you have, compare to competing methods' scales,
and argue the mechanism is scale-independent.
```
Our 5-scale chain (50M–750M) is the broadest from-scratch study in this area, exceeding
[competing method] (125M only). The improvement direction is consistent at every scale with
no diminishing returns. The mechanism modifies only [X], which depends on [architectural
constant], not model width.
```

### "Missing baseline comparison"
**Strategy**: If you can run it in 2 weeks, run it and include the result. If not, explain
why the comparison is not directly applicable.
```
We have added [Baseline X] results in new Table R1. [Baseline X] achieves [Y], compared to
our [Z], confirming [claim]. We note that [Baseline X] requires [additional cost], while our
method requires [less].
```

### "Novelty concerns"
**Strategy**: Articulate the precise technical novelty — what insight enables the result that
prior work lacks?
```
The key insight is not [surface-level description] but rather [deeper principle]. Specifically,
prior work [A, B, C] assumes [X], which prevents [Y]. We show that by [formulating differently],
the problem admits a closed-form solution. This is methodologically distinct from [prior work].
```

### "Writing quality / clarity"
**Strategy**: Concede, fix it, and move on. Don't argue about writing.
```
We thank the reviewer for this feedback. We have revised [Section X] to [specific improvement].
The updated version clarifies [specific point].
```

### "Statistical significance / single seed"
**Strategy**: Provide error bars if you have them, explain seed variance from multi-seed runs,
and label single-seed results appropriately.
```
Our primary claims are anchored in 3-seed evidence (Tables 1-3). The 750M result is explicitly
labeled as "supporting evidence." For context, our 3-seed runs show max inter-seed variance
of ~3% at 16K, making the single-seed 750M result (45.9% improvement) well outside noise range.
```

## Tone Calibration

| Situation | Tone |
|-----------|------|
| Reviewer found a real issue | "We thank the reviewer for identifying this. We have corrected..." |
| Reviewer misunderstood | "We apologize for the unclear presentation. To clarify: [explanation]" |
| Reviewer's concern is unfounded | "We appreciate this question. In fact, [evidence that addresses it]" |
| Reviewer asks for impossible experiments | "This is an excellent suggestion for future work. Within the rebuttal period, we have added [partial result]" |

**Never write**: "The reviewer is incorrect" or "We disagree with the reviewer."
**Instead**: "We appreciate this concern. The answer lies in [Section X / Table Y]..."

## After Rebuttal

If the rebuttal is successful and you're revising for camera-ready:
- Make every promised change
- Highlight changes in blue or use `\usepackage{changes}`
- Add a "Changes from Submission" section at the start of the appendix
- Thank reviewers in the acknowledgments
