# Server storage cleanup — 2026-09-08

- Status: COMPLETE. Explicitly requested cleanup of old research data and caches;
  no power-state change, model inference or GPU job.
- Deleted 2,392 files, freeing approximately **39.64 GiB** across the system and
  data filesystems. The frozen deletion list excluded model weights and any
  currently open files.

| Filesystem | Available before | Available after | Reclaimed |
| --- | ---: | ---: | ---: |
| System | 15.98 GiB | 27.73 GiB | 11.75 GiB |
| Data | 5.60 GiB | 33.49 GiB | 27.89 GiB |

Removed old downloaded FineWeb/PG19 corpora, tokenized training copies, inactive
task-input copies, Q/K/V activation and decision-trace tensors, and teacher
caches. These old large caches and inputs must no longer be reported as available
for replay. Small result/identity receipts and existing source code remain.

The 19 inventoried base, scratch, final and adapter weight files were retained;
their size, inode and modification time were checked before and after deletion.
Model configurations, tokenizers, the installed runtime, and the current OLMo
screen preparation were retained. The OLMo base subsequently matched its canonical
SHA256 during the one preparation-time weight read.

Cleanup plan SHA256:
`8e1e25c1f9b0f0a5c6704b630916392de0a6abc8af6959a797b82b1aeab04dda`.
The exact private file list and completion receipt remain in the current server
phase root. No repository archive or model redownload was required.
