## 251120 LoRA Perf Sweep

| ID | File | Purpose | Key Config |
|----|------|---------|------------|
| A | `FS_PPO_Q3_4B_lora_A_baseline.sbatch` | Baseline LoRA (rank16) for perf/memory reference. | micro=16; KV util 0.50; token cap `+2048`; LoRA rank16 α32 on actor+critic; checkpoints on; balance off. |
| B | `FS_PPO_Q3_4B_lora_B_noActorCkpt.sbatch` | Measure speed gain when only actor checkpoints are disabled. | A + `actor_rollout_ref.model.enable_gradient_checkpointing=False`. |
| C | `FS_PPO_Q3_4B_lora_C_noCheckpoints.sbatch` | Upper-bound throughput by disabling all checkpoints. | A + both actor/critic ckpt off. |
| D | `FS_PPO_Q3_4B_lora_D_rank32.sbatch` | Higher-capacity adapters (rank32/α64) to see memory vs. quality tradeoff. | A + `lora_rank=32`, `lora_alpha=64` actor+critic. |
| E | `FS_PPO_Q3_4B_lora_E_actorOnly.sbatch` | LoRA only on actor (critic full) to cut memory but keep critic fixed. | A settings but critic LoRA disabled (`lora_rank=0`), actor rank16; checkpoints on. |
| F | `FS_PPO_Q3_4B_lora_F_smallRank.sbatch` | Very low-rank adapters (rank8/α16) to test minimal-memory variant. | A but rank8/α16 actor+critic; checkpoints on. |
| G | `FS_PPO_Q3_4B_lora_G_micro12.sbatch` | Reduce micro-batch to 12 to check throughput vs. stability with LoRA. | A but micro=12, mini-batch scaled to 144, token cap `+2048`; checkpoints on. |
| H | `FS_PPO_Q3_4B_lora_H_balance.sbatch` | Add balanced batching + larger kv (0.52) to test step-time variance. | A but `trainer.balance_batch=True`, KV util 0.52, token cap `+2560`. |
| I | `FS_PPO_Q3_4B_lora_I_bucket256.sbatch` | Larger weight bucket (256MB) to see effect on communication time. | A but `rollout.update_weights_bucket_megabytes=256`. |
| J | `FS_PPO_Q3_4B_lora_J_tp1_micro16.sbatch` | Pure data parallel (TP=1) + LoRA to test comm vs. memory. | A but `tensor_model_parallel_size=1`; token cap `+1536`; checkpoints on. |
| K | `FS_PPO_Q3_4B_lora_K_rank32_noActorCkpt.sbatch` | Combine higher rank + actor checkpoint off to see if memory is ok. | D + `actor_rollout_ref.model.enable_gradient_checkpointing=False`. |
| L | `FS_PPO_Q3_4B_lora_L_rank8_noCkpt.sbatch` | Aggressive low-rank + no checkpoints for max speed/min memory. | B but rank8/α16 actor+critic, critic ckpt off too. |
| M | `FS_PPO_Q3_4B_lora_M_micro12_balance.sbatch` | Micro12 + balanced batching to see if smaller micros stabilize. | G + `trainer.balance_batch=True`. |
| A' | `FS_PPO_Q3_4B_lora_A_noEager.sbatch` | Baseline LoRA but with CUDA graphs re-enabled (no eager). | A + `rollout.enforce_eager=False`, token cap `+2560`, tag `no_eager`. |
| E' | `FS_PPO_Q3_4B_lora_E_actorOnly_noEager.sbatch` | Actor-only LoRA without eager fallback. | E + `rollout.enforce_eager=False`, token cap `+2560`. |
| F' | `FS_PPO_Q3_4B_lora_F_smallRank_noEager.sbatch` | Rank8 adapters in non-eager mode for maximum speed. | F + `rollout.enforce_eager=False`, token cap `+2560`. |
| H' | `FS_PPO_Q3_4B_lora_H_balance_noEager.sbatch` | Balanced batches + KV util 0.52 without eager slowdown. | H + `rollout.enforce_eager=False`. |
| J' | `FS_PPO_Q3_4B_lora_J_tp1_noEager.sbatch` | TP=1 data-parallel geometry with CUDA graphs active. | J + `rollout.enforce_eager=False`. |
| M' | `FS_PPO_Q3_4B_lora_M_micro12_balance_noEager.sbatch` | Micro12 balanced variant in non-eager mode. | M + `rollout.enforce_eager=False`, token cap `+2560`. |

Common knobs: 35 steps, 3 critic warmup iterations, random H100 delay, LoRA WANDB tag, memory telemetry enabled. All original A–M runs keep `rollout.enforce_eager=True`; primed variants run with `False` to restore CUDA graphs.

