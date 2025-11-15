## Goal
Port the multi-turn environment support from the Verlog fork into the latest `verl` with minimal churn to upstream files. The key additions are the environment stack (`verl/envs/**`), the multi-environment evaluator, and a PPO trainer that can drive multi-step rollouts with VecEnv.

---

## Components Identified in Verlog

### 1. Environment Runtime
- `verl/envs/vec_env.py`: launches each environment + captioner worker, supports grouped seeding, skip actions for frozen environments, memory stats, etc.
- `verl/envs/environments/**`: environment factories, wrappers, instruction prompts, reward shaping knobs (format penalties, binary rewards).
- `verl/envs/captioners/**`: prompt builders that convert environment state into text.

These directories do not exist in upstream `verl` and can be copied verbatim.

### 2. Evaluation Utilities
- `verl/trainer/ppo/multi_env_evaluator.py`: instantiates VecEnv per evaluation config, runs multi-environment validation loops, logs per-env metrics and sample episodes.
- Hydra configs under `verl/trainer/config/evaluation/*.yaml` describing evaluation suites.

### 3. Trainer / Algo Changes
- Verlog’s `ray_trainer.py` embeds:
  - VecEnv creation and shared validation env.
  - Freeze-completed-episodes logic using `__SKIP__` actions and `frozen_mask`.
  - MultiEnvEvaluator integration for periodic validation.
  - Environment seeding utilities and GRPO grouping awareness.
  - Batched rollout structure with `DataProto.insert` for each environment step.
- `core_algos.compute_gae_advantage_return` gained a `frozen_mask` term to ignore inactive episodes.
- `main_ppo.py` loads optional custom reward functions before instantiating the trainer.

---

## Implementation Plan

1. **Copy Environment Stack**
   - Mirror `Verlog/verl/envs` into the new codebase’s `verl/envs`.
   - Ensure any third-party deps (gymnasium, psutil, cloudpickle, etc.) are recorded in requirements if not already present.

2. **Add Multi-Environment Evaluator**
   - Copy `verl/trainer/ppo/multi_env_evaluator.py`.
   - Bring over the evaluation YAMLs that Verlog added (or at least the ones we need) under `trainer/config/evaluation/`.

3. **Introduce a Dedicated Trainer**
   - Add a new `verl/trainer/ppo/ray_multistep_trainer.py` derived from Verlog’s trainer.
   - Strip it down to only the logic that differs from upstream `ray_trainer.py` (VecEnv management, rollout loop, eval hooks) while reusing shared helpers (resource pools, checkpointing, optimizer updates) via imports from the stock trainer where possible.
   - Keep constructor signature and public methods identical (`init_workers()`, `fit()`), so it can be swapped in easily.

4. **Expose Trainer Selection**
   - Extend `main_ppo.py` (or a helper) to load the custom reward hook and select `RayMultistepTrainer` when `config.envs.enable_multistep` (or similar) is true; otherwise fall back to the upstream trainer.
   - Add the enabling flag plus VecEnv-related settings to `trainer/config/ppo_trainer.yaml`.

5. **Shared Logic Updates**
   - Merge the `frozen_mask` support into upstream `core_algos.py` so both trainers can use it.
   - Keep any other small utility changes (e.g., reward manager loading) unified instead of duplicating files.

6. **Evaluation + Hydra Wiring**
   - Update config defaults to include the new evaluation presets as optional overrides.
   - Ensure `hydra.searchpath` (if used) can discover the new YAMLs.

7. **Testing / Validation**
   - Run unit/integration tests for standard PPO to confirm the original trainer still works.
   - Smoke-test the multistep trainer on a toy environment (e.g., FrozenLake) to validate VecEnv rollout, freezing, and evaluation logging.

---

## Notes on Future Maintenance
- Most churn will live inside `ray_multistep_trainer.py`. When upstream `ray_trainer.py` changes, only shared utilities (resource pool helpers, metrics) should need syncing.
- Keep clear documentation/comments in the multistep trainer highlighting deviations from upstream to simplify future merges.

