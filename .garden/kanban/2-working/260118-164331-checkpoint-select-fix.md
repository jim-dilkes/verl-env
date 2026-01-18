# Fix Checkpoint Selection Sort Order

**Type:** fix
**Branch:** fix/various-bugfix
**Created:** 2026-01-18
**Started:** 2026-01-18
**Completed:** —

## Goal
Fix sbatch scripts to select the most recent checkpoint snapshot by modification time instead of alphabetically.

## Scope
- [x] Identify all affected sbatch files (152 files use this pattern)
- [x] Update to use `ls -t` (sort by mtime) so `head -n 1` gets newest
- [x] Verify fix works correctly

## Out of Scope
- Changing to explicit snapshot hashes (decided against)
- Modifying checkpoint selection logic outside sbatch files

## Key Decisions
- Use `ls -t` to sort by modification time (newest first)
- `head -n 1` then correctly picks the latest snapshot
- Fix all affected sbatch files at once

## Working Notes
### 2026-01-18 - Feature Started
**Bug:** sbatch scripts comment says "pick latest snapshot" but code uses `ls ... | head -n 1` which picks first alphabetically.

**Root cause:** `ls` without `-t` returns alphabetically sorted results.

**Fix:** Change `ls -d $SNAPSHOT_DIR/*` to `ls -dt $SNAPSHOT_DIR/*` in all affected scripts.

**Affected files:** 152 sbatch files use this pattern.

## Original Notes
- Snapshot selection comment says "latest", but command picks the first directory (`head -n 1`) rather than most-recent by mtime. Both new Overcooked scripts do this in OC_PPO_4B_multiact_eps0.sbatch and OC_PPO_4B_multiact_eps02.sbatch. Not a code bug, but can silently select the wrong checkpoint.
