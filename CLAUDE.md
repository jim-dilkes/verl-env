@BRISK.md

## Scratchpad
`.brisk/scratchpad/` contains persistent notes for cross-session context:
- `project-overview.md` - Research focus, key questions, novel contributions
- `codebase-structure.md` - Files we modify vs read-only, PhD notes location
- `environment-interface.md` - LLMAgentsWrapper interface, observation format, factory functions
- `exploration-metrics.md` - State visitation, action entropy, validity tracking metrics
- `experimental-configs.md` - Active experiments, key params, ideas to try
- `fastsnake-env.md` - FastSnake env details and TODO

Read these at session start for project context.

## Environment
- Conda environment: `conda activate verlog`
- SLURM jobs: Run on iridis via `ssh iridis3`, repo at `~/verl-env`, then `./submit_sbatch <dir>`

## Scope of Work
Only modify files in:
- `verl/trainer/ppo/ray_multistep_trainer.py`
- `verl/trainer/ppo/multi_env_evaluator.py`
- `verl/envs/environments/` (all subdirs)
- `verl/envs/captioners/` (all files)
- `experiments/` (config files)

## Config Registration
New config parameters must be registered in the appropriate YAML before use in sbatch overrides.
- Prompt configs: `verl/trainer/config/prompt/<env>.yaml` (e.g., `snake.yaml`)
- Hydra will reject overrides for unregistered keys

Example: To add `prompt.prompt.my_new_param=True`, first add `my_new_param: false` to the prompt yaml.

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

## Feature Testing Requirements
All new features MUST include a test that runs on cluster login nodes (direct python, not sbatch).

**Baseline test script:** `experiments/snake/test_login_node.sh`
- Model: Qwen3-0.6B-Base
- 1 critic warmup step, 3 training steps
- Tuned for 24GB L4 login nodes (train_batch=16, n_rollouts=4, micro_batch=2)
- Single GPU, gpu_mem_util=0.30

**Usage:**
1. If feature doesn't need config changes: just run `bash experiments/snake/test_login_node.sh`
2. If feature needs specific configs: copy the script, add your overrides, document in PR

This ensures features work before committing to full cluster runs.

<!-- brisk-session-manager -->
## Session Guidelines
Always use the AskUserQuestion tool when you need input from the user.
Never ask questions in plain text output.

## Project Documentation
- **SPEC.md**: Contains the project specification and requirements. Read this
  at the start of each session to understand the project goals and constraints.
- **PROGRESS.md**: Track your work here. Update this file when completing
  significant tasks or milestones. Check it at session start to see what's
  been done and what remains.

## Scratchpad
A `.brisk/scratchpad/` directory exists for your working notes. Use it to:
- Track findings and insights as you explore the codebase
- Note decisions and their rationale
- Keep context that would be useful across conversation turns
- Draft plans before implementation

Files in scratchpad are gitignored and persist across session restarts.
Write notes proactively—they help you maintain context.
<!-- /brisk-session-manager -->
