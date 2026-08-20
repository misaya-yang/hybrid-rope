# Historical provenance and reproduction docs

This directory is the NeurIPS-era provenance, reproduction, terminology, and
implementation-audit layer. It remains useful, but it is not the current ICLR
2027 claim or action authority.

For current work, read first:

1. `../../AGENTS.md`
2. `../../paper-2027/HANDOFF.md`
3. `../../paper-2027/research/README.md`

## What this directory still owns

| Need | File |
| --- | --- |
| Historical paper-to-code/data map | `PAPER_CLAIMS_MAP.md` |
| Sanitized result provenance and hashes | `RESULT_PROVENANCE_MANIFEST.md` |
| Reproduction paths | `REPRODUCE.md` |
| Dataset identities and preparation | `DATA_PREPARATION.md` |
| Metric and protocol vocabulary | `TERMS_AND_PROTOCOLS.md` |
| Script/result support audit | `EXPERIMENT_CODE_RESULT_AUDIT.md` |
| Historical wording audit | `PAPER_DESCRIPTION_AUDIT.md` |
| Blackwell runtime receipts | `RTX5090_BLACKWELL_PROFILE.md` |

## Authority rule

When a current ICLR claim routes to an experiment originally documented here,
use this directory to recover code, data, and provenance, then verify the
canonical experiment owner. Do not let an old paper description, “current”
label, table number, or rebuttal status override:

- the active `paper-2027/` source;
- `paper-2027/research/` claim routing;
- the raw/hash-backed owner;
- the current handoff.

NLL/PPL, teacher-forced passkey NLL-gap, strict autoregressive exact match,
2Wiki, RULER, downstream QA, and causal source-use remain different endpoints.
Historical local or report-backed evidence is not automatically reviewer-safe.
