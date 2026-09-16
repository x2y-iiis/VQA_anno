# Upstream provenance

- Repository: `git@github.com:piao-0429/doubao_las_annotation.git`
- Base commit: `286db70d9bd5fa0ce4cffb6de9232276b8cb025f`
- Vendored on: 2026-09-05
- Source directory: `/mnt/doubao_las_annotation`

## Local fixes

- Merge temporally contiguous Step 4 segments when both normalized `skill` and
  rewritten `description` are identical. This removes artificial duplicate
  subtask boundaries created by LAS segmentation or lossy English rewriting.

The vendored package includes the source working tree's human-actor and
single-head-video support. Git metadata, virtual environments, runtime output,
generated comparisons, and credential files are intentionally excluded.
