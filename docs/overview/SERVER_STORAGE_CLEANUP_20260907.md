# Server storage cleanup — 2026-09-07 UTC

- **Status:** completed operational cleanup; no model experiment performed.
- **Scope:** user-authorized cleanup in no-GPU mode, including system/data disks;
  retain one or two small base models and reconstruction-critical assets.
- **Protocol:** snapshot exact files, classify against revision v5, archive
  non-weight historical material locally, verify hashes, reject drift/open files,
  delete only the frozen selection, then inspect surviving files and disk space.
- **Receipt:** private local bundle `20260907T030155Z`; archive SHA-256
  `de298453459d0991ef5eb0519746926cf8ea7490dee379c4fb3a2fb4f03c3cbf`.
- **Evidence limit:** storage and file-presence/integrity observations only;
  neither a model-quality result nor a runtime/GPU qualification.

## Space reclaimed

| Disk | Available before (GiB) | Available after (GiB) | Reclaimed (GiB) |
| --- | ---: | ---: | ---: |
| System | 9.41 | 16.07 | 6.66 |
| Data | 1.92 | 48.49 | 46.57 |

Total reclaimed: **53.23 GiB**; **55,454** inventoried
file/symlink entries removed, plus empty retired directories.

## Retained reconstruction assets

- Complete OLMo-2-0425-1B-Instruct (about 1.485B) and Qwen2.5-1.5B-Instruct.
- Four principal Geo/Cosh scratch checkpoints for seeds 137 and 256.
- Four final comparison adapters, the OLMo teacher cache, current raw/tokenized
  data and tokenizers, release008/release012 and recent audit code/results.
- The existing runtime and platform services; no package environment was rebuilt.

**Pre-existing gap:** the principal scratch seed42 pair was not found on the
inspected data disk before cleanup. E1 still needs its exact original weights;
other exploratory seed42 weights cannot substitute for them.

Non-core OLMo/Qwen 7B and Qwen 0.5B weights, old exploratory runs/code,
non-final/obsolete adapters and optimizer states, compiler/download caches,
and designated non-core activation/teacher caches were removed. Historical
non-weight data/code/results were first migrated to a verified local archive;
removed non-core weights and large regenerable caches were not backed up.

## Validation and follow-up

The local archive contains 4,540 verified files. Before deletion, 125 full weight
hashes were recorded, including both retained base models matched to the prior
asset owner. After deletion, all selected paths were absent, 10 protected weights
were present, and 2,982 retained files kept the same size/inode/mtime. The live
platform-monitor database changed during operation and was confirmed open by
its platform service; it was never in the deletion set.

Seven retained historical scripts reference removed non-core model paths.
Treat their old arms and restart instructions as inactive; do not silently
redownload or substitute another model. Exact affected paths and archive members
are in the private receipt bundle. Future protocols must recheck asset presence.

No GPU device nodes were present, and no training/inference was launched. No
model execution or post-cleanup full weight rehash was performed. This cleanup
changes physical artifact locations, not the scientific meaning of old results.
