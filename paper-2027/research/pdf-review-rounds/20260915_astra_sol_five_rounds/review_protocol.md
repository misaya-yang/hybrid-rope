# Review protocol and calibration

## Separation of responsibilities

Two independently initialized agents, Astra and Sol, receive the same frozen
manuscript and the same prompt within a round. No reviewer receives prior
reviews, author expectations, desired scores, repository material, or the
reference-paper result. Only the integrator reads the repository and decides
which proposed revisions are justified.

The review scores are internal judgments. The requested independent assessment
of MrRoPE gave4/10 despite its ICLR2026 Oral acceptance. Its Markdown input also
lacked full figures, unlike the reviewed PDFs. This is evidence against treating
these raw scores as calibrated acceptance/presentation predictions; it is not
evidence that every criticism in that review is wrong. A low or high reference
score is never supplied as a target to the manuscript reviewers.

For every concern, the integrator checks the actual claim, the cited passage,
existing answers and counterevidence, and whether it changes the central result.
A missing local file does not establish missing research. Optional future
experiments do not become mandatory limitations. An unsupported reviewer claim
is recorded as a reviewer error; a repeated valid clarity issue is repaired.

## R03/R04 single-review rubric

Both agents assess the specific question, novelty as presented, contribution,
mathematical correctness, experimental support and narrative. They read the
whole main paper and consult relevant appendices. The output is central
contribution, strengths, up to five concerns with locators/counterevidence/
consequence/minimal repair, separate optional suggestions, recommendation,
internal score and confidence. The supplied internal anchors were2 reject,
4 weak reject,5 borderline,6 weak accept,8 strong accept,10 exceptional.
The reference Markdown review uses this rubric, with section/paragraph locators
in place of PDF pages and no external figure access.

## R05–R07 exact common prompt template

Only ROUND and each agent's own output filename vary. The output filename is
chosen from the agent task name; the substantive prompt is identical across
models and across these three rounds.

> Review /tmp/rope-pdf-only-ROUND/input.pdf as a simulated ICLR 2027 review panel. Your only substantive source is this frozen PDF. Do not read repository files, memory, other reviews, author conversations, or external literature; do not browse. You may extract text, render pages and check printed mathematics. Read the full main paper, visually inspect figures/layout, and consult relevant appendices before raising concerns. Do not edit the manuscript.
>
> Produce four distinct reviewer perspectives: R1 novelty and significance; R2 theory and method; R3 experiments and practical value; R4 overall argument and narrative. Each evaluates the whole paper, with its stated emphasis, without a predetermined attitude or score. These are perspectives within your context, not claims of four independently executed agents.
>
> For each reviewer give: central contribution as understood, strongest evidence, and up to three important concerns. Each concern must name PDF page/section, the actual claim affected, concrete evidence, relevant counterevidence or an answer already in the appendix, and the smallest useful repair. Classify it as an established error, unsupported claim, explanatory problem, or optional extension. Missing PDF detail does not prove an experiment or artifact does not exist. Judge the actual claims; SOTA, universal task-optimality, and every possible extension are not acceptance prerequisites. Evaluate the contribution's positive significance as carefully as limitations. Finish each review with recommendation, internal 1–10 score and 1–5 confidence, with decisive reasons; no target score is given.
>
> Then write an AC meta-review: reconcile disagreements, recheck contested facts against the PDF, discard misreadings, distinguish decision-relevant issues from optional improvements, and assess what new knowledge and practical value the paper establishes. Give a reasoned overall recommendation and internal score, not an arithmetic average, plus the highest-value manuscript revisions. State the limits of novelty assessment from PDF alone. Include a compact four-score/AC table.
>
> Write in Chinese to /tmp/rope-pdf-only-ROUND/reviews/<your task name's final component>.md. Return the path, scores, AC judgment and concise summary. State the material actually inspected.
