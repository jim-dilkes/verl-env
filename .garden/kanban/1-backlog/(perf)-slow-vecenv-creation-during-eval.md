# Slow VecEnv Creation During Eval

**Type:** perf
**Created:** 2026-01-19

## Problem

VecEnv creation accounts for a large portion of evaluation time. Each eval creates a new VecEnv (spawns N worker processes), runs rollouts, then closes it. With multiple evals per evaluation cycle, this overhead compounds.

## Evidence

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

## Scope

The issue affects:
- All evaluations (not just batched ones)
- Overcooked evals more severely (JAX initialization overhead)
- A100/swarm partitions disproportionately

## Notes

- Within-batch VecEnv reuse already implemented (commit 3a91b8a5) - addresses batched eval overhead
- Cross-eval VecEnv reuse attempted but reverted due to complexity/fragility of cache key design
- `vecenv_create_time_seconds` metric now logged for all evals (commit c319c41b)
