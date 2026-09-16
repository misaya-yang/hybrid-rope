# Common final comparative review prompt

Perform the final comparative review of manuscript_c.pdf versus manuscript_b.pdf.
These two PDFs are your only substantive sources; do not read repository, memory,
conversations, external sources or another agent's review. You may extract/render
PDFs and check printed mathematics. Read both main papers and relevant appendices,
inspect figures, and evaluate actual claims fairly; no target score or preferred
version is prescribed. Retain four perspectives: novelty/significance, theory/method,
experiments/practical value, narrative/clarity, each with C/B score and confidence.
Give an AC recommendation and an explicit regression audit: lost evidence, incorrect
changes, weakened explanations or useful additions. For material concerns provide
page/section, evidence/counterevidence, classification and smallest repair. Distinguish
optional future experiments from established flaws. Do not require universal task
optimality or wins. Treat previous judgments as revisable hypotheses. Focus on the
central claim and accurate separation of Cosh training/adaptation and frozen deployment.
Return a concise complete review, preferably within1200 words; do not write or edit files.

Both agents received the same full wording and absolute PDF paths. Agent identities
are recorded in the calling task: revision_v1_comparison_astra (gpt-6-astra) and
revision_v1_comparison_sol (gpt-5.6-sol). The four perspectives are within each
agent, not eight independently executed reviewers.
