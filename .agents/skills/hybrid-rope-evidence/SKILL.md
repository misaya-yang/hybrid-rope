---
name: hybrid-rope-evidence
description: Trace or register Hybrid-RoPE experimental evidence when checking paper claims or adding a result to the evidence registry.
---

# Hybrid-RoPE evidence

Deliver a claim traceable to its actual result owner, with its supported scope and
remaining source gaps. For a new result, update only the relevant owner, nearest
index, and registry entry; update the claim map if the manuscript claim changes.

## Locate the relevant evidence

- Use the [evidence index](../../../paper-2027/research/evidence/index.md) and
  [asset registry](../../../paper-2027/research/evidence/asset_registry.json) for
  asset identity, availability, and source paths.
- For a specific manuscript statement, use the
  [claim map](../../../paper-2027/research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md).
- For experiments not yet registered in the paper, follow their local index from
  the [experiment index](../../../experiments/index.md). Do not treat an old
  queue, external proposal, or implementation-ready status as a completed run.

Read the relevant entries and sources, not every linked report. Preserve source
paths and existing valid receipts. A report supports report-backed claims; raw-row
verification requires inspecting the corresponding raw rows. Missing ignored
artifacts are an availability limitation, not a reason to reconstruct results.

## Preserve the scientific identity

For the comparison at hand, retain model/checkpoint, table/support, gain, data
revision and rows, lengths, decoder, metric and aggregation identities. Distinguish
a local development subset from a published or independent benchmark. Pure-z
attribution needs the other factors controlled. Complete-output exact requires
the full raw-token answer and terminal EOS when that is the requested contract.
Report costs and local regressions alongside gains; do not infer an unmeasured
absolute score from paired differences.

## Finish the update

Use document-relative links and repository-root-relative registry paths. Follow
[maintenance guidance](../../../docs/maintenance/index.md) for inventory refresh
when files are added; run `python3 scripts/check_repository_docs.py` after navigation
changes. This checks document/source consistency, not model capability. Do not
launch experiments merely to complete an evidence-registration request.
