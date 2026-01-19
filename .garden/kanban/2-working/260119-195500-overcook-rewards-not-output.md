# Fix Overcooked Rewards Not Propagating

**Type:** fix
**Branch:** fix/overcook-rewards-not-output
**Created:** 2026-01-19
**Started:** 2026-01-19
**Completed:** —

## Goal
Trace and fix the reward flow from core Overcooked env through wrapper to training/eval metrics - rewards currently showing as zero/missing in both contexts despite being visible in interactive mode.

## Scope
- [ ] Trace reward flow: core env → wrapper → trainer/evaluator
- [ ] Identify where rewards are being lost/zeroed
- [ ] Implement fix for the reward propagation issue
- [ ] Verify rewards appear correctly in training
- [ ] Verify rewards appear correctly in eval metrics
- [ ] Document whether each reward type is configurable
- [ ] Check if pickup-onion/ingredient reward exists

## Out of Scope
- General refactoring of Overcooked wrapper (unless directly needed for fix)
- Changes to other environments
- Performance optimizations

## Key Decisions
- Focus on wrapper layer first (interactive mode works → core env likely fine)
- Fix both training and eval contexts

## Working Notes
<!-- Session handoff and working context goes here -->
### 2026-01-19 - Feature Started
**Context:** User reports zero/missing rewards in Overcooked env during training and eval. Interactive interface shows rewards correctly, so core `overcooked_ai` env is likely fine. Suspected: wrapper layer losing rewards.

**Investigation targets:**
- `verl/envs/environments/` - Overcooked wrapper/adapter
- Reward aggregation/formatting in wrapper step()
- How trainer/evaluator consume reward data

**Additional asks:**
- Report on reward configurability
- Check for onion/ingredient pickup reward
