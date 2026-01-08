Barely gets through run before seeming to get stuck on first evals - maybe the environment is too slow? Maybe it gets stuck ? Please test and ivnestigate. 

```Running SLURM prolog script on rose11
===============================================================================
Job started on Thu Jan  8 11:08:21 GMT 2026
Job ID          : 510530
Job name        : OC_PPO_4B_initial_1
WorkDir         : /iridisfs/home/jsbd1n24/verl-env
Command         : /home/jsbd1n24/verl-env/experiments/overcooked/260108_initial/OC_PPO_4B_initial_1.sbatch
Partition       : a100
Num hosts       : 1
Num cores       : 8
Num of tasks    : 1
Hosts allocated : rose11
Job Output Follows ...
===============================================================================
```

```[36m(WorkerDict pid=1346899)[0m 
Capturing CUDA graphs (decode, FULL):  93%|█████████▎| 62/67 [00:02<00:00, 23.34it/s]
[36m(WorkerDict pid=1346899)[0m 
Capturing CUDA graphs (decode, FULL):  97%|█████████▋| 65/67 [00:02<00:00, 22.91it/s]
[36m(WorkerDict pid=1346899)[0m 
Capturing CUDA graphs (decode, FULL): 100%|██████████| 67/67 [00:03<00:00, 21.85it/s]
[36m(WorkerDict pid=1346900)[0m /home/jsbd1n24/.conda/envs/verl/lib/python3.12/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:678: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=1346900)[0m   warnings.warn(
[36m(WorkerDict pid=1346900)[0m kwargs: {'n': 1, 'logprobs': 0, 'max_tokens': 256, 'repetition_penalty': 1.0, 'detokenize': False, 'temperature': 1.25, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(WorkerDict pid=1346899)[0m /home/jsbd1n24/.conda/envs/verl/lib/python3.12/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:678: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=1346899)[0m   warnings.warn(
[36m(WorkerDict pid=1346899)[0m kwargs: {'n': 1, 'logprobs': 0, 'max_tokens': 256, 'repetition_penalty': 1.0, 'detokenize': False, 'temperature': 1.25, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(TaskRunner pid=1346284)[0m [RayPPOTrainer] Setting actor_rollout_wg in multi_env_evaluator
[36m(TaskRunner pid=1346284)[0m wandb: Tracking run with wandb version 0.23.1
[36m(TaskRunner pid=1346284)[0m wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
[36m(TaskRunner pid=1346284)[0m wandb: Run data is saved locally in wandb/OC_PPO_4B_initial_run1_slu510530/wandb/offline-run-20260108_111402-04o8tdrv
[36m(TaskRunner pid=1346284)[0m Checkpoint tracker file does not exist: /scratch/jsbd1n24/checkpoints/verl_env/OC_PPO_4B_initial_run1/latest_checkpointed_iteration.txt
[36m(TaskRunner pid=1346284)[0m Training from scratch
[36m(TaskRunner pid=1346284)[0m [RayPPOTrainer] Starting initial validation at global_step=0
[36m(TaskRunner pid=1346284)[0m [RayPPOTrainer] Using multi_env_evaluator initial evaluation
[36m(TaskRunner pid=1346284)[0m [RayPPOTrainer] Closing training env to free memory for multi-env evaluation
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [WARNING] VecEnv.close: Failed to send close command to worker: [Errno 32] Broken pipe
[36m(TaskRunner pid=1346284)[0m [VecEnv] Successfully closed 32 worker processes
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Starting evaluation at global_step=0
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Number of environments to evaluate: 2
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Initial memory usage: 976.9 MB
[36m(TaskRunner pid=1346284)[0m Evaluating environment: Overcooked-CrampedRoom-Greedy
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Creating config for environment: Overcooked-CrampedRoom-Greedy
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Original env_config: {'name': 'Overcooked-CrampedRoom-Greedy', 'n_rollouts': 50, 'episode_length': 40, 'generation': {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'do_sample': False}, 'initial_seed': 0, 'freeze_completed_episodes': True, 'duplication_mode': 'none', 'env_name': 'overcooked', 'task': 'none', 'format_penalty': 0.0, 'instruction_prompt': '[Instructions]\nYou are a helpful assistant. You always respond by wrapping your thoughts in the correct XML tags. Your maximum response length: 200 words (tokens)\nYou are playing Overcooked solo. You control the only chef in a kitchen.\nYour goal is to cook and deliver soups as fast as possible.\n\n[How to Cook]\n1. Pick up ingredients (e.g., onions) from ingredient piles using \'interact\'\n2. Place 3 ingredients in a pot using \'interact\' while facing it\n3. Wait for the soup to cook\n4. Pick up a dish from the dish pile\n5. Pick up the cooked soup from the pot (with dish in hand)\n6. Deliver the soup to the serving counter using \'interact\'\n\n[Available Actions]\n"right": move right,\n"down": move down,\n"left": move left,\n"up": move up,\n"stay": stay in place (wait),\n"interact": interact with object in front of you (pick up, place, or use)\n\n[Rules]\n- You can only hold one object at a time\n- Each soup requires exactly 3 ingredients\n', 'captioner': {'type': 'naive', 'max_text_history': 0}, 'overcooked_kwargs': {'layout_name': 'cramped_room', 'horizon': 40, 'partner_policy': 'none', 'shaped_reward': False, 'pot_cook_time': 5, 'print_visualization': True, 'print_coordinates': True}}
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Training config n_rollouts: 32
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Evaluation env_config n_rollouts: 50
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Set instruction_prompt from eval config (length: 931 chars)
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] After basic overrides - n_rollouts: 50, task: none, env_name: overcooked
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Original captioner config: {'type': 'naive', 'max_text_history': 0, 'max_cot_history': 0, 'max_image_history': 0}
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Environment captioner config: {'type': 'naive', 'max_text_history': 0}
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Final captioner config: {'type': 'naive', 'max_text_history': 0, 'max_cot_history': 0, 'max_image_history': 0}
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Setting overcooked_kwargs: {'layout_name': 'cramped_room', 'horizon': 40, 'partner_policy': 'none', 'shaped_reward': False, 'pot_cook_time': 5, 'print_visualization': True, 'print_coordinates': True}
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Set initial_seed: 0
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Final temp_config.envs.n_rollouts: 50
[36m(TaskRunner pid=1346284)[0m [MultiEnvEvaluator] Final temp_config.envs keys: ['enable', 'n_rollouts', 'episode_length', 'duplication_mode', 'group_initial_seed', 'group_rollout_size', 'env_name', 'task', 'grpo_mode', 'format_penalty', 'instruction_prompt', 'binary_reward', 'freeze_completed_episodes', 'vec_env_multiprocessing', 'captioner', 'babyai_kwargs', 'babaisai_kwargs', 'fastsnake_kwargs', 'crafter_kwargs', 'minihack_kwargs', 'frozenlake_kwargs', 'webshop_kwargs', 'overcooked_kwargs']
[36m(TaskRunner pid=1346284)[0m [VecEnv] Using multiprocessing method: fork
[36m(TaskRunner pid=1346284)[0m [Worker 0] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 1] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 2] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 3] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 4] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 5] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 6] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 7] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 8] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 9] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 10] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 11] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 12] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 13] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 14] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 15] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 16] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 17] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 18] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 19] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 20] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 21] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 22] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 23] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 24] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 25] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 26] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 27] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 28] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 29] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 30] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 31] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 32] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 33] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 34] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 35] Memory at start: 970.22 MB
[36m(TaskRunner pid=1346284)[0m [Worker 36] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 37] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 38] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 39] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 40] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 41] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 42] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 43] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 44] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 45] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 46] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 47] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m 
[36m(TaskRunner pid=1346284)[0m [Worker 48] Memory at start: 970.23 MB
[36m(TaskRunner pid=1346284)[0m [Worker 49] Memory at start: 970.23 MB
==============================================================================
Running epilogue script on rose11.

Submit time  : 2026-01-08T11:08:21
Start time   : 2026-01-08T11:08:21
End time     : 2026-01-08T12:45:24
Elapsed time : 01:37:03 (Timelimit=2-00:00:00)

Job ID: 510530
Cluster: iridis_x
User/Group: jsbd1n24/fp
State: CANCELLED (exit code 0)
Nodes: 1
Cores per node: 8
CPU Utilized: 00:00:03
CPU Efficiency: 0.01% of 12:56:24 core-walltime
Job Wall-clock time: 01:37:03
Memory Utilized: 73.81 GB
Memory Efficiency: 29.52% of 250.00 GB (250.00 GB/node)

```