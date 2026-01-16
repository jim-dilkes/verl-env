# Auto WandB Analysis

**Type:** feat
**Branch:** feat/auto-wandb-analysis
**Created:** 2026-01-16
**Started:** 2026-01-16
**Completed:** —

## Goal
Build Python tooling to automate wandb experiment analysis - compare runs, query by config/metrics, generate text-friendly summaries for both human review and Claude Code consumption.

## Scope
- [ ] Core wandb API wrapper for fetching runs by project/filters
- [ ] Config diff utility - show what hyperparams differ between runs
- [ ] Final metrics comparison table (reward/score across runs)
- [ ] Learning curve extraction (key metrics over training steps)
- [ ] CLI interface for common queries
- [ ] Markdown report generation for experiment summaries
- [ ] "Experiment history" overview - what we've tried, outcomes, trajectory

## Out of Scope
- Visualization/charts (text-only output)
- Auto-alerting/notifications
- Live monitoring of in-progress runs
- Complex database/caching (use local CSVs of processed wandb data)

## Key Decisions
- Location: `analysis/` dir in verl repo
- Data flow: Fresh wandb imports → local CSVs → analysis
- Output formats: CLI tables + markdown + LLM-friendly text
- Dual purpose: Human use + Claude Code tooling for future sessions

## Working Notes
### 2026-01-16 - Feature Started
Interview summary:
- Primary use case: post-hoc analysis of completed experiments
- Need both run comparison and experiment tracking queries
- Outputs for terminal (quick checks) and markdown (notes/sharing)
- Key insight: should be usable by Claude Code to pull data and summarize experiments
- Metrics focus: config diffs, final scores, learning curves
