# FMRoPE vs EVQ: 151.9M / L=256 / 500M-token replication

Status: conditional follow-up to the registered 100M-token diagnostic.

This run preserves the parent experiment's model, three frequency arms, seed,
FineWeb-Edu source shards, optimizer family, batch sizes, evaluation anchors,
and NLL metrics. Only the training horizon changes:

- requested tokens: 500,000,000;
- consumed tokens: 499,974,144 complete global batches;
- optimizer steps: 7,629;
- warmup steps: 762 (the same approximately 10% schedule fraction).

The three arms are Paper-Geo base 500K, paper-faithful local FMRoPE
base 256, and EVQ-Cosh tau 4 / base 500K. The validation source remains the
disjoint pinned shard 004. Results must be reported separately from the 100M
run and remain a single-seed, 3.29-token-per-parameter diagnostic rather than a
training-saturation claim.
