# Codebase Structure and Scope

## Files We Modify (scope of work)
```
verl/trainer/ppo/
├── ray_multistep_trainer.py    # Main multi-step training loop
├── multi_env_evaluator.py      # Multi-env evaluation + entropy probing

verl/envs/
├── environments/
│   ├── FastSnake/              # Snake game (submodule from jim-dilkes/FastSnake)
│   ├── FrozenLake/             # Simple grid navigation
│   ├── webshop/                # E-commerce search/click (submodule)
│   ├── babyai_text/            # Grid nav + manipulation
│   ├── crafter/                # Resource gathering game
│   ├── minihack/               # Dungeon exploration
│   ├── nle/                    # Full NetHack
│   ├── textworld/              # Text adventures
│   ├── babaisai/               # Rule manipulation puzzles
│   ├── overcooked/             # Multi-agent cooking game (JaxMARL)
│   ├── env_wrapper.py          # Base wrapper interface
│   └── __init__.py             # Factory: make_env(), get_action_extraction_fn()
├── captioners/
│   ├── naive.py                # <think><plan><action> format
│   ├── cot.py                  # THINK: ... ACTION: ...
│   ├── base.py                 # BaseCaptioner class
│   └── prompt_builder/         # History management (HistoryPromptBuilder)

experiments/
├── BAI/                        # BabyAI experiment configs
├── snake/                      # FastSnake experiment configs
├── webshop/                    # WebShop experiment configs
└── training_efficiency_notes.md
```

## Core verl Library (read-only context)
- `verl/protocol.py` - DataProto class for data transfer
- `verl/trainer/ppo/ray_trainer.py` - Original single-step trainer
- `verl/trainer/ppo/core_algos.py` - PPO/GRPO algorithms, advantage estimation
- `verl/workers/` - FSDP/Megatron workers
- `verl/single_controller/` - Ray coordination

## Runtime Patches (sitecustomize)
This repo includes `sitecustomize.py`, which Python auto-imports on startup (via the standard `site` module) when the repo is on `PYTHONPATH` / you run from the repo root.

Why it exists:
- Under Python 3.12, `multiprocess==0.70.18` (a transitive dependency of `datasets`) can raise a noisy shutdown-time exception:
	`AttributeError: '_thread.RLock' object has no attribute '_recursion_count'`
- The patch suppresses that teardown traceback by making `multiprocess.resource_tracker` compatible with Python 3.12.

Notes:
- This does not affect training logic; it only prevents an “Exception ignored in: ResourceTracker.__del__” message at interpreter exit.
- If/when `multiprocess` publishes a fixed release for your environment, this can be removed.

## PhD Notes Location
`/Users/jim/Documents/PhD/Research Projects/4) Exploration in SDM for LLMs/`
- `Log Book/` - Experiment logs and findings
- `Implementation/` - Implementation notes
- `Research Plan/` - Research direction docs
- `Ideas to try.md` - Future experiment ideas
- `{C} Kanban/` - **EDITABLE** Feature planning board
