# Continue on another machine

The repository is the portable source of current state. Start with
[root index](../../index.md), then select manuscript work or experiments.
The [paper handoff](../../paper-2027/HANDOFF.md), author contract and repository
skills carry the decisions; personal Codex memory and old chats are optional.

## Transfer the actual working tree

At this cleanup, the source branch is `09_09` and the checkout contains
uncommitted manuscript, report and documentation changes. A matching Git commit
alone does not prove the PC has the current work. No commit or push is implied
by documentation cleanup.

Two valid transfer paths:

1. Sync the reviewed changes through the project's normal Git workflow.
   Preserve any PC-side dirty work before applying incoming changes.
2. Use the local continuation overlay produced with this cleanup. It contains
   the modified/new Git-visible files, base commit and hashes, not `.git`,
   checkpoints, ignored raw, credentials or model caches. It is an overlay for
   an existing repository clone, not a replacement for its history.

For the overlay, read its `CONTINUATION_MANIFEST.json` first. The base commit
must be available in the destination clone. A safe application target is a new
detached worktree created at that exact base, not a dirty working directory:

```text
git worktree add --detach <new-empty-worktree-path> <manifest-base-commit>
```

Extract the overlay into that worktree, preserving relative paths. On Windows,
use an archive tool or `Expand-Archive`; on Linux/macOS, `unzip` works. Review
`git status` and the manifest. If it lists deletions, review them individually;
the current export is not an instruction to erase PC-side work. Do not blindly
replace files in another active checkout.

The cleanup report records the local export artifact and verification outcome.
The archive must actually be transferred to the PC; creating it here does not
mean the PC is synchronized.

To make a later overlay, run `python scripts/export_continuation_overlay.py`
with a new ZIP filename under `internal/local_snapshots/`. It refuses to replace
an existing archive and checks that inputs did not change during collection.

## Verify useful state, then choose one route

```text
git status --short
python scripts/check_repository_docs.py
```

Use `python3` where appropriate. Check the current title/abstract and manuscript
hash against the continuation manifest. The active manuscript is
`paper-2027/main.pdf`; the submission source ZIP is a different artifact from
the continuation overlay.

| Task | Read next |
|---|---|
| Edit the manuscript | `paper-2027/HANDOFF.md`, then `.agents/skills/hybrid-rope-paper-editing/SKILL.md` |
| Review a revision | `.agents/skills/hybrid-rope-regression-review/SKILL.md` and that round's immutable PDFs |
| Analyze or run an authorized experiment | `experiments/index.md`, relevant owner, then `docs/research/protocols/EXPERIMENT_WORKFLOW.md` |
| Register a completed result | `.agents/skills/hybrid-rope-evidence/SKILL.md` |
| Locate old work | `experiments/DIRECTORY_MAP.md` or the historical catalogs, only for the named question |

Repository skills should be available to a new session in this checkout. If
the client does not discover them, read the indicated `SKILL.md` directly;
do not depend on copying a personal skills directory or Mac absolute paths.

A concise first message to a new session is:

> Continue from this repository's index.md for the task I give next. Use its
> current result owners and project skills, preserve the frozen title/abstract,
> and distinguish historical plans from authorized work. Verify the local
> working-tree state before changing files.

## Environment and data boundaries

Documentation and report inspection require no model. For paper builds, use
a shell with `pdflatex`/`bibtex` or Tectonic; Poppler supports visual checks.
On Windows, WSL or an equivalent working shell can run `compile.sh`.
Figure/report regeneration uses the dependencies declared by its script;
the paper figure layer uses Python, NumPy and Matplotlib. Use a local environment
that actually contains them rather than copying a Mac interpreter path.

`package_source.py` needs repository history for its pinned runtime sources, or
the extracted submission archive's bundled runtime tree. Keep that distinction
when making a source bundle on the PC.

Large raw, models and tokenizer caches remain at their experiment owners.
Do not redownload or rerun to compensate for their absence on a second laptop.
Known SSH ports in dated records are connection hints, not current credentials,
machine health or authorization. The latest user-supplied experiment endpoint
was port 20225 on `connect.westd.seetacloud.com`; earlier ports may belong to
different or stopped rental instances. Use current author/owner information
for a remote task, and do not restart a machine merely for a documentation check.

OpenReview state is external: the assistant filled and previewed the final
abstract, but did not click submit. Check with the author before reporting the
actual submission status; no session migration should submit the form.
