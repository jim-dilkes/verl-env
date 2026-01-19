# Slow VecEnv Creation During Eval

**Type:** perf
**Branch:** perf/slow-vecenv-creation
**Created:** 2026-01-19
**Started:** 2026-01-19
**Completed:** 2026-01-19

## Goal
Reduce VecEnv creation overhead during evaluation (currently ~80% of eval time on some partitions).

## Scope
- [x] Investigate multiprocessing start method impact
- [x] Create test script to compare spawn vs fork vs forkserver
- [x] Verify fork works with fastsnake and overcooked
- [x] Run cluster test to measure real impact with NFS
- [x] Change default from 'spawn' to 'fork'
- [x] Update all sbatch files to use fork

## Out of Scope
- Cross-eval VecEnv caching (fork alone provides 126x speedup - sufficient)
- Changes to babyai environment (separate issue)

## Key Decisions
- **Root cause identified**: Default `spawn` multiprocessing causes each worker to re-import all modules, creating NFS contention on cluster
- **Solution**: Switch to `fork` - workers inherit imports via copy-on-write
- **Cluster testing confirms 126x speedup** (125.2s → 0.99s for 50 workers)

## Working Notes

### 2026-01-19 - Feature Complete

**Changes made:**
1. Changed default in `verl/envs/vec_env.py` from `spawn` to `fork`
2. Updated 14 sbatch files to use `fork`
3. Created test scripts for benchmarking

**Cluster benchmark results (50 workers, fastsnake):**
| Method | Create | Reset | Total | Speedup |
|--------|--------|-------|-------|---------|
| spawn | 2.3s | 33.4s | 125.2s | — |
| fork | 0.17s | 0.015s | 0.99s | **126x** |
| forkserver | 1.06s | 26.5s | 84.5s | 1.5x |

**Expected impact:** Eval VecEnv creation drops from ~1600s to ~15s on A100 partition.

### 2026-01-19 - Investigation Complete

**Root Cause Analysis:**
1. VecEnv uses `spawn` multiprocessing by default
2. `spawn` starts fresh Python interpreter for each worker
3. Each worker re-imports all modules (numpy, torch, JAX, env code)
4. With 50 workers = 50x import overhead hitting NFS simultaneously
5. NFS contention explains partition variance (A100 5x slower than H200)

**Files created:**
- `scripts/test_vecenv_multiprocessing.py` - comparison test
- `experiments/tests/test_vecenv_mp_cluster.sh` - cluster test script

## Original Notes

VecEnv creation accounts for a large portion of evaluation time. Each eval creates a new VecEnv (spawns N worker processes), runs rollouts, then closes it. With multiple evals per evaluation cycle, this overhead compounds.

### Timing breakdown from cluster runs

**StateVisitation eval (300 rollouts, 6 batches of 50):**
```
VecEnv creation:    1642.52s  <- majority of time
VecEnv reset:       24.03s
VecEnv close:       3.95s
Tokenizer:          4.94s
Total eval:         2054.66s
```

VecEnv creation is ~80% of total eval time.

### Partition-specific variance

Same workload shows dramatically different VecEnv creation times across cluster partitions:

| Partition | VecEnv Creation Time |
|-----------|---------------------|
| quad_h200 | ~317s |
| a100 | ~1627s (5x slower) |
| swarm_a100 | ~528s |
