@BRISK.md

<!-- brisk-session-manager -->
## Brisk Session Manager
Always use the AskUserQuestion tool when you need input from the user.
Never ask questions in plain text output.
<!-- /brisk-session-manager -->

## Scratchpad
`.brisk/scratchpad/` contains persistent notes for cross-session context:
- `project-overview.md` - Research focus, key questions, novel contributions
- `codebase-structure.md` - Files we modify vs read-only, PhD notes location
- `environment-interface.md` - LLMAgentsWrapper interface, observation format, factory functions
- `exploration-metrics.md` - State visitation, action entropy, validity tracking metrics
- `experimental-configs.md` - Active experiments, key params, ideas to try
- `fastsnake-env.md` - FastSnake env details and TODO

Read these at session start for project context.

## Scope of Work
Only modify files in:
- `verl/trainer/ppo/ray_multistep_trainer.py`
- `verl/trainer/ppo/multi_env_evaluator.py`
- `verl/envs/environments/` (all subdirs)
- `verl/envs/captioners/` (all files)
- `experiments/` (config files)

## External Resources
- FastSnake repo: `github.com/jim-dilkes/FastSnake` (cloned to `verl/envs/environments/FastSnake/`)

## PhD Notes
Located at `/Users/jim/Documents/PhD/Research Projects/4) Exploration in SDM for LLMs/`:
- `Log Book/` - Dated experiment logs, job queues, findings (e.g., `2025-11-25 - Full runs...`)
- `Implementation/` - Implementation notes (`Exploration Metrics.md`, `Bugfix Exploration Metrics.md`)
- `Research Plan/` - Research direction docs (`Expl-RLMSDM - Overarching Plan.md`, Loss/Context mediated exploration)
- `Ideas to try.md` - Future experiment ideas (episode sampling, fewer tokens, Muon optimizer, etc.)
- `Novelty, Paper Plan.md` - Paper positioning, comparison table vs RAGEN/ARPO/Search-R1/etc.
- `{C} Kanban/` - Feature planning board. **EDITABLE** - add implementation notes, design decisions, progress updates to feature files here.

Consult these for research context, experimental rationale, and implementation decisions.

## Kanban Progress Tracking
Keep `{C} Kanban/` updated throughout work:
- Starting feature → update status, add implementation notes
- Significant progress → log decisions, blockers, approaches tried
- Completing work → mark done, note follow-ups or learnings

Do this proactively; don't wait for user to ask.
