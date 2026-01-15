# Overcooked Learning Debug

## Status
- Created: 2026-01-12

## Problem
Overcooked agents are not learning at all - they can't get a single positive reward. Main issue seems to be that they never try to interact, so can't get rewards.

## Things to Try
1. Modify prompt - encourage interaction/exploration
2. Multi-act, with and without epsilon - could be a great example of it working?
3. Entropy based approaches as previously used

## Goals
- [ ] Try prompt modifications
- [ ] Test multi-act with epsilon
- [ ] Test multi-act without epsilon
- [ ] Try entropy-based approaches

## Notes
Could be a great demonstration case if multi-act exploration helps here.