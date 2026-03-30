# NeurIPS Paper Checklist Guide

The NeurIPS checklist is mandatory and reviewers read it. Treat it seriously —
thoughtful justifications signal a mature submission.

## How to Answer

Each item gets one of: `\answerYes{}`, `\answerNo{}`, `\answerNA{}`

- **Yes**: Provide a specific justification pointing to where in the paper this is addressed
- **No**: Explain why not and what you did instead
- **NA**: Explain why this item doesn't apply to your work

Never give one-word justifications. The checklist is your chance to preempt reviewer concerns.

## Item-by-Item Guidance

### 1. Claims
> Do the main claims accurately reflect the paper's contributions and scope?

**Good justification**: "The abstract and introduction state three specific claims: [X, Y, Z].
Each maps directly to experimental evidence in [Tables/Figures]."

### 2. Limitations
> Does the paper discuss limitations?

This is where scale limitations, single-seed caveats, and scope restrictions go. Be honest.
A transparent limitations section actually increases reviewer confidence.

### 3. Theory Assumptions and Proofs
> Does the paper provide full assumptions and correct proofs?

If you have theoretical results:
- State where assumptions are listed (main body + appendix)
- Confirm proofs are in the appendix
- Note any approximation steps explicitly

### 4. Experimental Reproducibility
> Does the paper disclose all information needed to reproduce results?

List: model architecture, hyperparameters, dataset details, random seeds, compute resources.
Point to specific sections/tables/appendices.

### 5. Open Access to Data and Code
> Does the paper provide open access?

During review: `\answerNo{}` is acceptable with "Code and data will be released upon
acceptance" or "Anonymous review package does not include code."

### 6. Experimental Settings
> Are all training and test details specified?

Point to the experimental setup section and appendix details.

### 7. Statistical Significance
> Does the paper report error bars?

Describe your multi-seed protocol. If some results are single-seed, state this explicitly
and label them as supporting evidence.

### 8. Compute Resources
> Does the paper report compute requirements?

Include: GPU type, number of GPUs, training time, total GPU-hours.

### 9. Code of Ethics
> Does the research conform to the NeurIPS Code of Ethics?

For most ML methodology papers: "This work studies [technical topic] and does not involve
human subjects, sensitive data, or dual-use concerns."

### 10. Broader Impacts
> Does the paper discuss societal impacts?

Even pure methodology papers should mention potential downstream impacts.
Be balanced: mention both potential benefits and risks of the technology.

### 11. Safeguards
> Are there safeguards for responsible release?

For methodology papers without deployed systems: `\answerNA{}` with justification.

### 12-15. Licenses, Assets, Crowdsourcing, IRB
These are straightforward — answer factually based on what you used.

## Common Mistakes

- Giving `\answerYes{}` without pointing to the specific section
- Being vague: "We discuss limitations" vs. "Section 6 discusses three specific limitations:
  scale gap (50M-750M only), single-seed large-scale evidence, and broadband approximation scope"
- Defensive tone in justifications — just state facts
- Forgetting to include the checklist at all (instant desk reject at some venues)
