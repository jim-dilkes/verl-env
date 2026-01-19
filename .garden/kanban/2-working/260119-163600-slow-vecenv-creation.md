# Slow VecEnv Creation During Eval

**Type:** perf
**Branch:** perf/slow-vecenv-creation
**Created:** 2026-01-19
**Started:** 2026-01-19
**Completed:** —

## Goal
Reduce VecEnv creation overhead during evaluation (currently ~80% of eval time on some partitions).

## Scope
- [x] Investigate multiprocessing start method impact
- [x] Create test script to compare spawn vs fork vs forkserver
- [x] Verify fork works with fastsnake and overcooked
- [ ] Run cluster test to measure real impact with NFS
- [ ] Change default from 'spawn' to 'fork' (if cluster tests pass)
- [ ] Consider cross-eval VecEnv reuse (if fork alone isn't enough)

## Out of Scope
- Cross-eval VecEnv caching (may revisit if fork isn't sufficient)
- Changes to babyai environment (separate issue)

## Key Decisions
- **Root cause identified**: Default `spawn` multiprocessing causes each worker to re-import all modules, creating NFS contention on cluster
- **Solution**: Switch to `fork` - workers inherit imports via copy-on-write
- **Local testing shows 40x speedup** with fork vs spawn (0.07s vs 2.72s total for 10 workers)

## Working Notes

### 2026-01-19 - Investigation Complete

**Root Cause Analysis:**
1. VecEnv uses `spawn` multiprocessing by default (`config.envs.vec_env_multiprocessing`)
2. `spawn` starts fresh Python interpreter for each worker
3. Each worker re-imports all modules (numpy, torch, JAX, env code)
4. With 50 workers = 50x import overhead hitting NFS simultaneously
5. NFS contention explains partition variance (A100 5x slower than H200)

**Local Test Results (10 workers, fastsnake):**
| Method | Create | Reset | Total | vs spawn |
|--------|--------|-------|-------|----------|
| spawn | 0.04s | 0.73s | 2.72s | baseline |
| fork | 0.01s | 0.003s | 0.07s | **40x faster** |
| forkserver | 0.02s | 0.68s | 2.13s | 1.3x faster |

**Why fork is faster:**
- Workers inherit all imports from parent via copy-on-write
- No NFS reads needed - memory pages shared until written
- Create time: 0.01s vs 0.04s (4x)
- Reset time: 0.003s vs 0.73s (243x) - lazy init already done in parent

**Verified working:**
- fastsnake + fork: ✓
- overcooked + fork: ✓ (JAX works fine)
- babyai has separate bug (not mp-related)

**Files created:**
- `scripts/test_vecenv_multiprocessing.py` - comparison test
- `experiments/tests/test_vecenv_mp_cluster.sh` - cluster test script

**Next steps:**
1. Run `bash experiments/tests/test_vecenv_mp_cluster.sh` on cluster
2. If fork works: change default in vec_env.py from 'spawn' to 'fork'
3. Existing `test_login_node.sh` already uses fork (line 132)

**Note:** The comment "spawn avoids CUDA segfaults when forking" is conservative - VecEnv workers don't use CUDA (environments run CPU-only, CUDA is in actor process).

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

The slowdown appears infrastructure-related (likely NFS/filesystem contention when 50 workers spawn simultaneously and each imports JAX, loads environment code, etc.).

### Per-eval overhead

With 5-7 evals per evaluation cycle, each creating its own VecEnv:
- Best case (H200): 5 × 300s = 25 min just for VecEnv creation
- Worst case (A100): 5 × 1600s = 2+ hours
