# Prewarm + Reuse Eval VecEnvs (Keep `fork` Speed, Avoid Late-Fork Deadlock)

**Type:** perf / stability
**Branch:** (tbd)
**Created:** 2026-01-20
**Started:**
**Completed:**

## Problem Statement
We need evaluation-time VecEnv creation to be as fast as `fork` (NFS makes `spawn`/`forkserver` prohibitively slow), but our current evaluation flow creates VecEnv workers *late* in the driver process lifetime (after Ray + torch distributed + vLLM CUDA graphs + threadpools). Forking at that point can deadlock/hang.

Observed symptom (cluster): initial validation hangs immediately after logging `[VecEnv] Using multiprocessing method: fork` for evaluation VecEnv creation.

## Goals
- Preserve `fork`-level speed for evaluation env worker startup on NFS.
- Eliminate evaluation hangs caused by late `fork`.
- Maintain behavioral parity with current evaluation semantics (metrics, seeds, prompts, batching).
- Keep the codebase readable (avoid the “tangled lifecycle” we’ve hit before).

## Non-Goals / Out of Scope
- Rewriting VecEnv IPC protocol.
- Changing environment logic, captioners, or evaluation metric definitions.
- Switching the whole system back to `spawn`/`forkserver` globally.
- Solving every JAX/fork interaction in every env (needs an explicit decision; see questions).

## Current Behavior (Baseline)
### Where VecEnvs are created today
- Training VecEnv is created early in `RayMultistepTrainer.__init__()` via `_make_vec_env(...)`.
- Multi-env evaluation creates VecEnv(s) inside `MultiEnvEvaluator.evaluate(...)` (specifically inside the per-env evaluation body). It does:
  1) Build a temporary config per eval env (overrides n_rollouts/batch_size/prompt settings)
  2) Call `make_vec_env(...)` which instantiates `verl.envs.vec_env.VecEnv`
  3) `VecEnv` starts N worker processes via `multiprocessing.get_context(method).Process(...).start()`
  4) Run rollouts (optionally batched)
  5) Close VecEnv at the end

### Why it breaks
When evaluation VecEnv is created, the process already initialized heavy components that spawn threads / hold internal locks (Ray runtime, torch distributed, vLLM CUDA graph capture, tokenizers parallelism, etc.). Forking at this stage is unsafe and can deadlock.

### Why spawn/forkserver are unacceptable
On NFS, each `spawn`ed worker repeats large import trees; startup can become 10–60+ minutes for large worker counts.

## Proposed Approach (Option 1): Prewarm + Reuse Eval VecEnv Pools
### Key idea
Create the evaluation VecEnv worker processes **early**, at a safe time (before Ray/torch/vLLM thread-heavy init), and keep them alive for reuse across evaluations. Evaluation then becomes “reset + rollout + metrics”, without any new process creation.

This preserves `fork` speed (COW inherits imports once) and avoids late-fork deadlocks.

### High-level lifecycle
We introduce an explicit **Eval VecEnv Pool** concept inside `MultiEnvEvaluator`.

#### Objects
- `MultiEnvEvaluator` owns a cache: `eval_env_pool: Dict[PoolKey, VecEnv]`
- Each cached `VecEnv` is tied to a *specific effective environment configuration* (see “PoolKey” below).

#### Lifecycle states
1) **Constructed**: evaluator exists, pool cache empty.
2) **Prewarmed**: evaluator has created VecEnv(s) for some/all eval environments (processes started).
3) **In Use**: evaluate() resets VecEnv with explicit seeds, runs rollouts.
4) **Idle**: after evaluate(), VecEnv stays alive; no close.
5) **Shutdown**: on trainer shutdown/exception, evaluator closes all cached VecEnvs.

### Proposed control flow (intended)
#### At trainer startup (safe-fork window)
- Trainer creates `MultiEnvEvaluator` (already happens).
- Trainer calls `multi_env_evaluator.prewarm()` **before any worker init that triggers CUDA graphs / distributed initialization**.
- `prewarm()` builds the per-eval temporary config(s) and instantiates VecEnv(s) (using `fork`).

#### During training / validation
- When evaluation is requested:
  - Optionally close training VecEnv (existing behavior) depending on memory decision.
  - `MultiEnvEvaluator.evaluate()` retrieves the already-created VecEnv from its pool cache.
  - It resets with explicit seeds and runs rollouts. It does **not** create or close worker processes.
  - It returns metrics identical to current behavior.

#### At end of run
- Trainer calls `multi_env_evaluator.close()` (or `__del__` safety) to close all pooled VecEnvs.

## PoolKey (How we decide if a VecEnv can be reused)
We will **not** pool by full config hash.

**Final decision (Strategy A):** pool is keyed only by **worker-count** (effectively `batch_size` used for batched eval, otherwise `n_rollouts`).

To preserve parity across different eval configs (layout/prompt/kwargs/etc.) without creating many pools, we will implement `hard_reset` which rebuilds env + captioner *inside existing worker processes*.

This gives:
- No late forks (pools are created once, early)
- Bounded CPU memory/process count (typically 1 pool)
- Full parity on env/captioner config (because the env objects are recreated)

---

## Final Specification (Implementation-Ready)

### Summary
1) Add a **bounded eval VecEnv pool** in `MultiEnvEvaluator`, keyed by worker-count.
2) Add a `VecEnv.hard_reset(...)` API + worker protocol command to rebuild env+captioner inside existing worker processes.
3) Add **Policy 1 guard**: when `vec_env_multiprocessing == fork`, refuse to start worker processes if JAX appears imported in the parent (with an explicit override env var).
4) Ensure trainer calls `multi_env_evaluator.prewarm(...)` **before** heavy runtime init (`init_workers()` / vLLM / torch distributed) so evaluation never needs to fork later.

### Invariants / Constraints (decided)
- **Fork-only performance constraint:** evaluation VecEnv workers must use `fork` for startup speed on NFS.
- **Training-time eval env_name constraint:** during training evals, all eval environments share the same `env_name` as training (cross-env post-training is out-of-scope).
- **Batched eval constraint:** enforce `n_rollouts % batch_size == 0` for batched eval.
- **Parity definition:** statistical parity is acceptable; exact bitwise parity is not required.
- **Training env lifecycle:** keep training VecEnv alive during eval (do not close/recreate around evaluation).

### New/Updated Public APIs

#### `VecEnv.hard_reset(...)` (new)
Add a method on `verl.envs.vec_env.VecEnv`:

- Signature (suggested):
  - `hard_reset(self, *, env_name: str, task: str, config: Any, render_mode: str | None = None) -> None`
- Behavior:
  - Sends a `('hard_reset', payload)` message to every worker.
  - Waits for acknowledgements; if any worker errors, raise a `RuntimeError` with worker rank + traceback string.

Payload requirements:
- Must be **picklable**.
- To avoid OmegaConf pickling issues, send a resolved container:
  - `config_blob = OmegaConf.to_container(config, resolve=True)` (or equivalent)
  - Worker reconstructs with `OmegaConf.create(config_blob)`.

#### Worker protocol extension (new)
Extend the `worker(...)` command loop in `verl/envs/vec_env.py` with a new command:

- `cmd == 'hard_reset'`:
  - Input: `{env_name, task, render_mode, config_blob}`
  - Steps (per worker):
    1) Attempt to `close()` old `env` and cleanup captioner references.
    2) `del env`, `del captioner` and `gc.collect()`.
    3) Recreate:
       - `env = make_env(env_name, task, config, render_mode=render_mode)`
       - `captioner = make_captioner(config)`
    4) Reset internal per-worker counters if needed (`reset_count`, cached image, etc.).
    5) Reply `('ok', None)`.
  - On exception: reply `('error', <stringified traceback>)`.

Notes:
- The worker should import `make_env`/`make_captioner` lazily inside the `hard_reset` handler.
- After `hard_reset`, the parent **must** call `VecEnv.reset(seeds=...)` before stepping.

#### Policy 1 guard (new)
Add a small guard in `VecEnv.__init__` (parent process) when `mp_method == 'fork'`:

- If any of these are present in `sys.modules`: `jax`, `jaxlib`, `jaxmarl` (including submodules), raise a `RuntimeError`.
- Override escape hatch:
  - `VERL_ALLOW_UNSAFE_FORK_WITH_JAX=1` disables the guard.

Rationale:
- Prevent silent regressions where a future import pulls in JAX in the parent, reintroducing fork deadlocks.

### MultiEnvEvaluator changes (pool + prewarm)

#### Pooling model (bounded)
- Maintain a dict: `pool_by_worker_count: Dict[int, VecEnv]`.
- Key is the number of worker processes for the eval VecEnv instance:
  - `worker_count = batch_size` if batched eval is enabled for that env
  - else `worker_count = config.envs.n_rollouts`

#### `prewarm()` (new)
Add `MultiEnvEvaluator.prewarm()` which:
1) Computes all distinct `worker_count` values required by the evaluation config.
2) Creates a `VecEnv` for each `worker_count` **once**.
3) Does not run rollouts; only starts processes.

**Critical ordering constraint:** trainer must call `multi_env_evaluator.prewarm()` **before** `RayPPOTrainer.init_workers()`.

Implementation note:
- Use a “neutral” temporary config for prewarm that is safe to construct (it won’t actually instantiate envs in the parent; env construction happens in workers).
- Because we will `hard_reset` before using the pool for any eval env, the exact prewarm config does not need to match a particular eval; it only needs the worker count and vec env settings.

#### Evaluate flow (updated)
For each eval environment entry:
1) Build `temp_config = _create_env_config(...)` as today (prompt/instruction/env kwargs overrides apply here).
2) Determine `worker_count` and fetch pool `vec_env = pool_by_worker_count[worker_count]`.
   - If missing: **fail fast** (do not late-fork).
3) Call `vec_env.hard_reset(env_name=..., task=..., config=temp_config, render_mode=...)`.
4) Run the existing batched evaluation loop:
   - Compute explicit seeds
   - Call `vec_env.reset(seeds=...)` for each batch
   - Step until episodes finish
5) Do **not** close the VecEnv.

### Configuration
Add config flags with safe defaults:

- `eval.vecenv_pooling.enabled: bool` (default: `true` for NFS deployments)
- `eval.vecenv_pooling.prewarm: bool` (default: `true`)
- `eval.vecenv_pooling.fail_if_missing_pool: bool` (default: `true`)

VecEnv guard escape hatch:
- Environment variable `VERL_ALLOW_UNSAFE_FORK_WITH_JAX=1`

### Error handling / failure modes

- If `hard_reset` fails in any worker:
  - Parent raises `RuntimeError` including worker rank and traceback.
  - Trainer should surface the error and stop the run (safer than continuing with mismatched env state).

- If evaluator needs a worker_count that is not prewarmed:
  - Fail fast with a clear message: “would require late fork; refusing”.

### Performance considerations

- This design ensures:
  - Process creation happens once and early.
  - Switching eval configs is done via `hard_reset` (in-process), avoiding NFS import storms.
- JAX import cost:
  - For Overcooked/JaxMARL, JAX will be imported in worker processes when they first construct the env.
  - Keeping the pool alive amortizes this cost across all evals.

### Observability
Add logs:
- On `prewarm()`: list worker_counts created.
- On each eval env: log `worker_count`, whether `hard_reset` succeeded, and timing for `hard_reset`.
- On shutdown: log closing pooled VecEnvs.

### Acceptance Criteria (final)
- No hangs during initial validation or later periodic eval when `vec_env_multiprocessing=fork`.
- Evaluation does not create new OS processes after prewarm.
- CPU memory does not grow unbounded with number of eval configs (pool bounded by worker-count).
- Metrics continue to match baseline statistically under the same seed policy.

### Test Plan (final)

Unit-ish / local:
1) Create a small VecEnv (e.g. fastsnake) and call `hard_reset` to a different config; verify it can reset/step after rebuild.
2) Add a targeted regression test that `VecEnv` raises if `jax` is imported in parent and mp_method is `fork` (guard), and that the env var override disables it.

Integration:
1) Run a short training job with `eval.vecenv_pooling.enabled=true` and `prewarm=true`:
   - Confirm prewarm logs appear before `init_workers()`.
   - Confirm evaluation logs show `hard_reset` but not new VecEnv construction.
2) Run Overcooked evaluation config once; confirm no late-fork hang.

---

## Implementation Checklist (files + touchpoints)

### `verl/envs/vec_env.py`
- Add Policy 1 guard in `VecEnv.__init__` (fork + jax already imported => error; env var override).
- Add `VecEnv.hard_reset(...)` method.
- Extend `worker(...)` loop to handle `cmd == 'hard_reset'`.

### `verl/trainer/ppo/multi_env_evaluator.py`
- Add pool dict keyed by worker-count.
- Add `prewarm()` and call from trainer before `init_workers()`.
- Modify evaluate path to use pool + `hard_reset` instead of constructing/closing VecEnv per env.

### `verl/trainer/ppo/ray_multistep_trainer.py`
- Ensure `multi_env_evaluator.prewarm()` is invoked before `init_workers()`.
- Stop closing/recreating the training VecEnv around evaluation (per decision), or gate behind a flag defaulting to “keep alive”.

---

## Outstanding Questions
None blocking for implementation based on current decisions.


## Ensuring Behavioral Parity
### Reset / seeding parity
Current evaluation computes explicit seeds per rollout (and per batch) then calls `VecEnv.reset(seeds=...)`. With reuse, we continue doing that. Each evaluation starts with `reset(seeds=...)`, so any prior state is discarded.

### Batched evaluation parity
Current evaluation (for batched mode): creates a VecEnv with `batch_size` workers once per eval call, then loops over batches with different seed slices.

With reuse:
- The VecEnv for that eval env config is created once (with `batch_size` workers) at prewarm time.
- Each evaluation call uses the same VecEnv and still performs the same batch loop. No semantic change.

### Prompt / instruction parity
Current `_create_env_config` sets prompt overrides (notably `prompt.prompt.environment_instruction`). That impacts env initialization.

With reuse:
- We create one VecEnv per eval env config with its own `temp_config` containing those overrides.
- We do NOT attempt to “repurpose” a VecEnv for a different instruction prompt.

### Metric parity
Metrics are computed from rollout data collected during evaluation. Since we are not changing step logic, only the process lifecycle, metrics should be identical within stochasticity bounds (controlled by seeds).

### Cleanup parity
Today, evaluator closes VecEnv after each env evaluation call. With pooling, we won’t close after each eval by default.

We must ensure:
- Worker processes don’t leak resources across eval calls.
- Explicit shutdown closes everything deterministically.

## Risks / Failure Modes
- **Memory/CPU budget**: keeping multiple eval VecEnvs alive may consume too many processes/GB.
- **Deadlocks** if prewarm is still “too late” in initialization order.
- **Config drift**: if evaluation config changes mid-run, we need to rebuild the corresponding pool entry.
- **JAX environments**: some envs may be fundamentally incompatible with `fork` (JAX deadlock risk). We need an explicit policy.

## Instrumentation / Observability
Add (or ensure) logs:
- When prewarm occurs, list pool entries and worker counts.
- When evaluate() uses a pooled VecEnv, log pool key/env name.
- On shutdown, log close success.

Optional: expose metric `eval/*/vecenv_create_time_seconds` should become ~0 after first prewarm.

## Acceptance Criteria
- Cluster run no longer hangs during initial evaluation.
- Evaluation startup time stays close to `fork` baseline (no NFS import storm).
- Metrics and episode generation logging remain unchanged.
- Clean shutdown closes all pooled VecEnv workers.

## Test Plan (proposed)
1) Local smoke test with `scripts/test_eval_timing.py` (ensure repeated evaluate() calls reuse the pool).
2) Small cluster run with 1 eval env, small batch_size; verify:
   - prewarm log appears early
   - no VecEnv creation happens during evaluation
3) Regression: standard training loop still progresses, and evaluation outputs same keys.

## Rollout Plan
- Behind a config flag: `eval.vecenv_pooling.enabled` (name TBD).
- Default OFF until cluster verified.
- After validation, make default ON for NFS clusters.

---

## Architectural Decisions Needed (Do NOT assume)
Please answer these; they determine the exact lifecycle.

### Q1) Prewarm scope: all eval envs or lazy?
**Option A: Prewarm ALL eval env VecEnvs upfront**
- Pros: guaranteed no late forks, simplest runtime behavior
- Cons: potentially huge process/memory footprint (sum of all eval env batch_size workers)

**Option B: Prewarm only a subset upfront (e.g., the envs used in `val_before_train`)**
- Pros: controls footprint
- Cons: requires clear definition of “subset”; still risks late fork if a non-prewarmed eval is triggered later

**Option C: Lazy prewarm on first use BUT still early**
- Pros: minimal upfront cost
- Cons: hard to guarantee “early enough” unless we restrict when eval can run

Decision needed: which option do you want, and what is the max total worker-process budget you can tolerate?

### Q2) Interaction with training env close/recreate
Current trainer closes `self.env` to free memory during multi-env eval, then recreates it after eval.

With pooled eval VecEnvs, we can:
- **Option A: Keep current behavior** (close training env during eval)
  - Pros: preserves memory behavior
  - Cons: total memory savings reduced because pooled eval VecEnvs remain resident
- **Option B: Stop closing training env**
  - Pros: simpler lifecycle
  - Cons: may increase peak memory; might be why close existed

Decision needed: should we keep closing the training env around evaluation?

### Q3) Pool lifetime
- **Option A: Pool lasts entire run** (created at startup, closed at the end)
- **Option B: Pool lasts between eval cycles only** (create before eval, keep for next eval, but allow explicit teardown to reclaim memory)

Decision needed: do you ever need to reclaim the CPU-side memory/processes mid-run?

### Q4) JAX env policy (e.g., overcooked-jaxmarl)
Some envs historically required `spawn` to avoid JAX+fork deadlocks.

Choices:
- **Option A: Allow per-eval-env mp method** (`fork` default, but some envs can force `spawn`)
- **Option B: Hard-require `fork` for all eval envs** (fast, but potentially unsafe for JAX)
- **Option C: Disallow JAX envs in pooled mode** (fail fast)

Decision needed: what should the policy be?

### Q5) PoolKey strictness
- **Option A: Hash full resolved `temp_config`** (safe parity, more pools)
- **Option B: Hand-pick fields** (fewer pools, risk mismatch)

Decision needed: which do you prefer?

### Q6) Safety behavior if pooling is enabled but a needed pool entry is missing
- **Option A: Fail fast with a clear error** (“would require late fork; refusing”)
- **Option B: Fallback to `spawn`** (avoids hang, but may take hours)

### Q7) Do we want `hard_reset` semantics?
If we implement `hard_reset`, we can bound the pool size, but we must redesign parts of VecEnv creation.

Choices:
- **Option A: Pool-by-config only** (no `hard_reset`, simplest implementation, risk many pools)
- **Option B: Pool-by-worker-count + `hard_reset`** (bounded pools, more invasive changes)

Decision needed: do you want to pursue `hard_reset` now, or keep it as a follow-up?

### Q8) Worker-count strategy (if `hard_reset` is chosen)
- **Option A: Maintain one pool per worker-count encountered** (e.g., batch_size values)
  - Pros: minimal changes to VecEnv API
  - Cons: still can grow (but bounded by #distinct batch sizes)
- **Option B: Maintain a single pool at max worker-count and add “inactive workers” support
  - Pros: strictly bounded to 1 pool
  - Cons: requires new VecEnv API to activate a subset and ensure non-active workers do not allocate envs

Decision needed: which strategy is acceptable?

Decision needed: which failure mode is acceptable?

---

## Implementation Notes (sketch; not final)
- Add `MultiEnvEvaluator.prewarm()` and `MultiEnvEvaluator.close()`
- Add internal `get_or_create_vecenv(env_config, eval_name)` that uses pool cache
- Modify evaluation code path to **not** close VecEnv after each eval when pooling enabled
- Ensure trainer calls `prewarm()` at the correct time in initialization order

If `hard_reset` is chosen:
- Extend `verl/envs/vec_env.py` worker protocol with `hard_reset`
- Change VecEnv worker initialization so it can recreate `env`/`captioner` from config payloads

