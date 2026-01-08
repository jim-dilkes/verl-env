# Bugfix: Overcooked Env Gets Stuck

## Status
- Created: 2026-01-08
- Started: 2026-01-08
- Completed: 2026-01-08

## Problem
Overcooked environment hangs during initial evaluation on cluster. Workers spawn but never progress past environment creation.

## Root Cause
Fork multiprocessing + JAX = deadlock.

When parent process has threads (Ray, vLLM, PyTorch), fork copies only main thread. Other threads' locks are left in inconsistent state. When forked workers import JAX, it deadlocks on internal locks.

## Evidence
- Workers log "Memory at start" but never "Memory after env creation"
- CPU efficiency 0.01% (blocking, not computing)
- Confirmed locally: fork + threads + JAX causes crash on macOS, silent deadlock on Linux

## Fix
Changed `envs.vec_env_multiprocessing=fork` → `spawn` in sbatch config.

## Files Changed
- `experiments/overcooked/260108_initial/OC_PPO_4B_initial_1.sbatch` - spawn instead of fork
- `.garden/docs-agent/environments/overcooked-jaxmarl-implementation.md` - documented the issue
