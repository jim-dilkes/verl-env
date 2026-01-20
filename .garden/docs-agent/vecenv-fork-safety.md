# VecEnv Fork Safety

## Why Fork is Safe

VecEnv uses `fork` multiprocessing by default (`envs.vec_env_multiprocessing=fork`) because it's ~100x faster than `spawn` on NFS filesystems. Workers inherit imports via copy-on-write, avoiding slow re-imports.

Fork is safe for VecEnv because:

1. **CPU-only workers**: Env and captioner code doesn't use CUDA
2. **Early creation**: VecEnvs are created at trainer init / prewarm, before heavy runtime init
3. **JAX is imported but not used**: The `transformers` library imports JAX as a side-effect (via `MistralForSequenceClassification`), but no JAX computation runs before forking

## JAX Import vs JAX Usage

The key insight is that **importing JAX doesn't start threads** - only actually using JAX does.

```python
# This just loads modules (no threads):
import jax

# This would initialize XLA runtime (threads start):
jax.numpy.array([1, 2, 3])
```

Since we fork before any JAX computation, workers get a clean copy without inherited thread/lock state.

## Historical Context

We originally had a "JAX guard" that checked if JAX was imported before forking and raised an error. This was removed because:

1. `transformers` always imports JAX at module level (unavoidable)
2. The guard would always trigger, making it useless
3. The real safety comes from forking early (before JAX is *used*), not from checking imports

## When Fork Could Be Unsafe

Fork can cause deadlocks if:
- The parent process has started threads that hold locks
- Child inherits the lock state but not the threads
- Child tries to acquire the "held" lock → deadlock

This would happen if VecEnv creation occurred **after** heavy runtime initialization (Ray actors, vLLM engine, model loading). The current design avoids this by:
- Creating training VecEnv early in `RayMultistepTrainer.__init__`
- Prewarming eval VecEnv pools before `init_workers()`

## Fallback: spawn

If fork issues occur in some environment, use `spawn`:

```yaml
envs:
  vec_env_multiprocessing: spawn
```

This is slower (~10-30s per VecEnv creation on NFS) but avoids all fork-related issues.
