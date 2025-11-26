## Training Efficiency Notes

### Pipeline Timing
- Actor update and subsequent weight broadcast must finish before rollout resumes. Tweaks under `actor_rollout_ref.actor.*` (e.g., micro batch, checkpointing) directly shorten rollout idle time.
- Critic updates (`critic.*`) run after weights are sent and can overlap with new rollouts. Speeding them up still reduces total wall-clock, but rollout waits mostly on the actor stage.

### Memory vs Throughput Trade-offs
- **Gradient checkpointing** (`*.model.enable_gradient_checkpointing`): Off = faster, higher activation memory; keep it on for critic/ref if memory is tight.
- **Tensor vs data parallel** (`actor_rollout_ref.rollout.tensor_model_parallel_size`): TP>1 halves per-GPU memory but adds NCCL overhead; TP=1 duplicates the whole model per GPU yet simplifies comms.
- **Padding removal** (`*.model.use_remove_padding`): True trims padded tokens → less activation memory; False keeps behavior simple but always allocates `max_prompt_length`.
- **CUDA graphs vs eager** (`actor_rollout_ref.rollout.enforce_eager`): False enables CUDA graph capture for faster vLLM; True disables graphs (needed for runtime LoRA) and nearly halves throughput.
- **Prefix & chunked prefill** (`actor_rollout_ref.rollout.enable_prefix_caching`, `enable_chunked_prefill`): True reuses KV-cache, needs more memory; False recomputes prompts but lowers KV footprint.
- **Rollout batch caps** (`actor_rollout_ref.rollout.max_num_batched_tokens`, `max_num_seqs`, `max_model_len`): Set close to actual usage, e.g. `actor_rollout_ref.rollout.max_num_batched_tokens=$((micro_batch_size * max_token_len_per_gpu + 256))`.
- **KV cache budget** (`actor_rollout_ref.rollout.gpu_memory_utilization`): Increase for speed (0.45–0.5) or decrease for memory (≤0.35) knowing lower values trigger cache thrashing later in training.
- **Weight broadcast chunking** (`actor_rollout_ref.rollout.update_weights_bucket_megabytes`): 512MB = faster sync but higher bandwidth spikes; 64MB smooths transfers and reduces peak memory.
- **VecEnv multiprocessing** (`envs.vec_env_multiprocessing`): `fork` launches faster but duplicates process memory; `spawn` saves RAM at the cost of startup time.

### Practical Snippets
```bash
# Actor checkpointing off, critic on
actor_rollout_ref.model.enable_gradient_checkpointing=False \
critic.model.enable_gradient_checkpointing=True \

# Tight KV limits
actor_rollout_ref.rollout.max_model_len=1024 \
actor_rollout_ref.rollout.max_num_batched_tokens=$((micro_batch_size * max_token_len_per_gpu + 256)) \

# Memory-friendly rollout config
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.enable_prefix_caching=True \
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \

# Env workers
envs.vec_env_multiprocessing=spawn \
envs.n_rollouts=32 \
```

Use this doc as a checklist when trading off speed vs memory in PPO runs—each flag above was observed to impact either rollout throughput, learner speed, or GPU/Ray memory usage in recent experiments. 

