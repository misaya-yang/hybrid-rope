# PDF comparison prompt

Replace `{PDF_A}`, `{PDF_B}` and `{VENUE}`. Send the resulting text unchanged to each independent reviewer. Supply no author-side findings or target score. For a single-paper review, remove the second path and comparative/regression request.

> Independently assess the revised manuscript {PDF_A} against its pre-edit baseline {PDF_B} for {VENUE}. These PDFs are your only substantive sources. You may extract their text and render pages, but must not read TeX, repository files, reports, web sources, memory, conversations or other reviews. Read both main papers and relevant appendices, including figures and tables. Give four perspectives: novelty/significance, theory/method, experimental/practical value, and narrative/clarity, followed by an AC synthesis and a concrete regression audit. For each material finding identify its page/section, supporting and contrary evidence, effect on the stated contribution, and smallest useful repair. Distinguish real errors, lost evidence, important reproducibility gaps, presentation weaknesses and optional extensions. Inspect the rendered formula before reporting a suspected extraction error. Assess the claims actually made; neither a preferred version nor a score is prescribed. Return a concise complete review. Do not edit files or spawn agents.

## Prompt design basis

The explicit task, permitted context and output contract follow OpenAI's [prompt-engineering guidance](https://developers.openai.com/api/docs/guides/prompt-engineering), checked 2026-09-17. The specific PDF boundary, model pairing and editorial criteria come from the author's instructions. They are not official conference rules.
