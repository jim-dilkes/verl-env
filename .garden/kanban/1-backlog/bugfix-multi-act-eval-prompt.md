# Bugfix: Multi-act Eval Prompt

## Status
- Created: 2026-01-12

## Problem
Multi-act context mediated exploration is almost correct, however the evaluations don't actually use the correct prompt.

## Goals
- [ ] Fix eval to use correct multi-act prompt
- [ ] Test with 25% epsilon (much higher than current)

## Context
- Related to: context-driven-action-selection (completed)
