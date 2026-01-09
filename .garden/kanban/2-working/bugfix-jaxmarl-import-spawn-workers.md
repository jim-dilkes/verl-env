# Bugfix: jaxmarl import fails in spawn multiprocessing workers

## Status
- Created: 2026-01-09
- Started: 2026-01-09
- Completed:

## Scope
**In scope**:
- Diagnose why `import jaxmarl` fails in spawned worker processes
- Fix import/path issue so overcooked env works with `spawn` multiprocessing

**Out of scope**:
- Changing multiprocessing mode (must stay `spawn` for JAX deadlock avoidance)
- General overcooked env changes

## Goals
- [ ] Identify root cause of jaxmarl import failure in spawn workers
- [ ] Implement fix
- [ ] Verify overcooked training runs on cluster

## Acceptance Criteria
- Overcooked PPO job runs past env initialization on SLURM cluster
- Workers successfully import jaxmarl and create environments

## Test Cases
- Submit same sbatch job, verify no `ModuleNotFoundError: No module named 'jaxmarl'`
- Env workers start successfully, training begins

## Constraints
- Must use `spawn` multiprocessing (fork causes JAX deadlock - see commit a97b4dd8)
- Cluster conda env: `/home/jsbd1n24/.conda/envs/verl`

## Context
- Related fix: a97b4dd8 `fix(overcooked): use spawn multiprocessing to avoid JAX deadlock`
- Key files: `verl/envs/vec_env.py`, `verl/envs/environments/__init__.py`, `verl/envs/environments/overcooked/`
- Error traceback shows: worker subprocess → `vec_env.py:270` → `multi_env_evaluator.py:80` → lazy import in `__init__.py:58` → jaxmarl import fails

## Interview Notes
- SLURM job `OC_PPO_4B_initial_1` (ID 512427) on rose09 (a100)
- All 50 workers fail with same error
- jaxmarl is presumably installed in conda env but spawn workers can't find it
- Likely sys.path or environment variable not being inherited by spawn processes
