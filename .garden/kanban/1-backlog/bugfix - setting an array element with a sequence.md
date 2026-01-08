Key sections from slurm logs below. Please investigate in my local copy of the codebase. 


```
Running SLURM prolog script on swarma1002
===============================================================================
Job started on Thu Jan  8 15:06:16 GMT 2026
Job ID          : 510942
Job name        : FS_PPO_4B_decay_A_1
WorkDir         : /iridisfs/home/jsbd1n24/verl-env
Command         : /home/jsbd1n24/verl-env/experiments/snake/260106_entropy_decay/FS_PPO_4B_decay_A_1.sbatch
Partition       : swarm_a100
Num hosts       : 1
Num cores       : 8
Num of tasks    : 1
Hosts allocated : swarma1002
Job Output Follows ...
===============================================================================
```

```Training Progress:   0%|          | 4/1000 [11:18<45:49:11, 165.61s/it]
[36m(TaskRunner pid=1271585)[0m step:4 - actor/entropy:0.36945098638534546 - critic/vf_loss:np.float64(0.316579750584412) - critic/vf_clipfrac:np.float64(0.08957402876985725) - critic/vpred_mean:np.float64(0.6322887133806944) - critic/grad_norm:np.float64(49.0375) - perf/mfu/critic:np.float64(0.04542102367255493) - critic/lr:np.float64(1e-05) - critic/score/mean:0.10353711790393014 - critic/score/max:0.99 - critic/score/min:-2.01 - critic/rewards/mean:0.10353711790393014 - critic/rewards/max:0.99 - critic/rewards/min:-2.01 - critic/advantages/mean:-1.1866295493896344e-17 - critic/advantages/max:5.315892870990613 - critic/advantages/min:-6.21189273005197 - critic/returns/mean:0.42377298118087053 - critic/returns/max:3.624473840066752 - critic/returns/min:-2.01 - critic/values/mean:0.75 - critic/values/max:7.5625 - critic/values/min:-4.71875 - critic/vf_explained_var:-0.390396284093796 - response_length/mean:19.625 - response_length/max:256.0 - response_length/min:0.0 - response_length/clip_ratio:0.0030487803742289543 - response_length_non_aborted/mean:112.43668365478516 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:65.0 - response_length_non_aborted/clip_ratio:0.017467249184846878 - response/aborted_ratio:0.8254573345184326 - prompt_length/mean:81.88033294677734 - prompt_length/max:432.0 - prompt_length/min:0.0 - prompt_length/clip_ratio:0.0 - timing_s/old_log_prob:10.899189329706132 - timing_s/values:32.795669596642256 - timing_s/adv:1.1905829515308142 - timing_s/update_critic:116.13478486053646 - timing_s/step:161.0777970990166 - timing_per_token_ms/adv:0.008939988372673657 - timing_per_token_ms/update_critic:0.8720464416034275 - timing_per_token_ms/values:0.24625995567217762 - perf/total_num_tokens:133175 - perf/time_per_step:161.0777970990166 - perf/throughput:413.3872029493165
[36m(TaskRunner pid=1271585)[0m /iridisfs/home/jsbd1n24/verl-env/verl/trainer/ppo/ray_multistep_trainer.py:1359: FutureWarning: Warning: Function 'verl.trainer.ppo.metric_utils.reduce_metrics' is deprecated. Please use 'verl.utils.metric.reduce_metrics' instead.
[36m(TaskRunner pid=1271585)[0m   actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
Error executing job with overrides: ['data.max_prompt_length=768', 'data.max_response_length=256', 'data.train_batch_size=256', 'data.train_files=examples/data/placeholder.parquet', 'data.val_files=examples/data/placeholder.parquet', 'actor_rollout_ref.actor.ppo_mini_batch_size=128', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024', 'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1024', 'actor_rollout_ref.rollout.tensor_model_parallel_size=2', 'actor_rollout_ref.rollout.max_num_seqs=2048', 'actor_rollout_ref.rollout.max_num_batched_tokens=18944', 'actor_rollout_ref.rollout.engine_kwargs.vllm.max_seq_len_to_capture=2304', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.40', 'actor_rollout_ref.rollout.max_model_len=1024', 'actor_rollout_ref.model.path=/scratch/jsbd1n24/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554', 'actor_rollout_ref.rollout.temperature=1.25', 'actor_rollout_ref.rollout.top_k=-1', 'actor_rollout_ref.rollout.val_kwargs.temperature=0.5', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.mode=sync', 'actor_rollout_ref.actor.entropy_coeff=0.002', 'actor_rollout_ref.actor.entropy_coeff_low=0.0005', 'actor_rollout_ref.actor.entropy_coeff_high=0.02', 'actor_rollout_ref.actor.entropy_coeff_lr=0.002', 'actor_rollout_ref.actor.entropy_low=0.5', 'actor_rollout_ref.actor.entropy_high=0.7', 'actor_rollout_ref.actor.entropy_low_final=0.35', 'actor_rollout_ref.actor.entropy_high_final=0.55', 'actor_rollout_ref.actor.entropy_top_p=0.33', 'actor_rollout_ref.actor.policy_loss.loss_mode=vanilla', 'actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024', 'trainer.balance_batch=False', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'critic.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.rollout.enforce_eager=False', 'actor_rollout_ref.rollout.enable_prefix_caching=True', 'actor_rollout_ref.rollout.load_format=dummy_dtensor', 'actor_rollout_ref.rollout.update_weights_bucket_megabytes=256', '+actor_rollout_ref.model.override_config.torch_dtype=bfloat16', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.model.use_fused_kernels=False', 'actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16', '+critic.model.override_config.torch_dtype=bfloat16', 'critic.model.fsdp_config.model_dtype=bfloat16', 'critic.model.path=/scratch/jsbd1n24/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554', 'critic.ppo_micro_batch_size_per_gpu=16', 'critic.ppo_max_token_len_per_gpu=1280', 'critic.forward_max_token_len_per_gpu=1280', 'algorithm.step_gamma=0.99', 'algorithm.token_gamma=1.0', 'algorithm.step_lam=0.95', 'algorithm.token_lam=1.0', 'algorithm.use_kl_in_reward=False', 'algorithm.kl_ctrl.kl_coef=0.0', 'envs.enable=True', 'envs.n_rollouts=32', 'envs.duplication_mode=none', 'envs.freeze_completed_episodes=True', 'envs.env_name=fastsnake', 'envs.format_penalty=1', 'envs.binary_reward=False', 'envs.captioner.type=naive', 'envs.captioner.max_text_history=0', 'envs.captioner.max_cot_history=0', 'envs.fastsnake_kwargs.width=10', 'envs.fastsnake_kwargs.height=10', 'envs.fastsnake_kwargs.max_rounds=8', 'envs.fastsnake_kwargs.num_external_snakes=1', 'envs.fastsnake_kwargs.num_random_snakes=1', 'envs.fastsnake_kwargs.death_reward=-1', 'envs.vec_env_multiprocessing=fork', 'actor_rollout_ref.rollout.calculate_log_probs=true', 'algorithm.rollout_correction.rollout_is=sequence', 'algorithm.rollout_correction.rollout_is_threshold=3.0', 'algorithm.rollout_correction.rollout_rs=null', 'algorithm.rollout_correction.rollout_rs_threshold=null', 'algorithm.rollout_correction.rollout_rs_threshold_lower=null', 'algorithm.rollout_correction.rollout_token_veto_threshold=null', 'algorithm.rollout_correction.bypass_mode=false', 'algorithm.rollout_correction.use_policy_gradient=false', 'algorithm.rollout_correction.rollout_is_batch_normalize=false', 'trainer.log_val_generations=1', 'trainer.project_name=verl_env', 'trainer.experiment_name=FS_PPO_4B_decay_A_run1_slu510942', 'trainer.group=FS_PPO_4B_decay_A', 'trainer.val_before_train=True', 'trainer.critic_warmup=5', 'trainer.critic_warmup_micro_batch_size_per_gpu=32', 'trainer.n_gpus_per_node=2', 'trainer.nnodes=1', 'trainer.save_freq=50', 'trainer.test_freq=50', 'trainer.render=False', 'trainer.total_epochs=1000', 'trainer.total_training_steps=1000', 'trainer.default_local_dir=/scratch/jsbd1n24/checkpoints/verl_env/FS_PPO_4B_decay_A_run1', 'trainer.max_actor_ckpt_to_keep=1', 'trainer.max_critic_ckpt_to_keep=1', 'trainer.tags=["snake","project:verl-env","vllm:0.11.0","adaptive-entropy","entropy-decay","moderate"]', 'ray_kwargs.ray_init.num_gpus=2', 'evaluation=snake_evals', 'prompt=snake']
[36m(TaskRunner pid=1271585)[0m wandb: 
[36m(TaskRunner pid=1271585)[0m wandb: Run history:
[36m(TaskRunner pid=1271585)[0m wandb:               actor/entropy ▁▁▁▁
[36m(TaskRunner pid=1271585)[0m wandb: behavior/valid_action_ratio ▁
[36m(TaskRunner pid=1271585)[0m wandb:       critic/advantages/max ▁▆█▇
[36m(TaskRunner pid=1271585)[0m wandb:      critic/advantages/mean ▁█▄▄
[36m(TaskRunner pid=1271585)[0m wandb:       critic/advantages/min █▄▁▁
[36m(TaskRunner pid=1271585)[0m wandb:            critic/grad_norm █▃▃▁
[36m(TaskRunner pid=1271585)[0m wandb:                   critic/lr ▁▁▁▁
[36m(TaskRunner pid=1271585)[0m wandb:          critic/returns/max ▄█▁▄
[36m(TaskRunner pid=1271585)[0m wandb:         critic/returns/mean ▅█▁▄
[36m(TaskRunner pid=1271585)[0m wandb:          critic/returns/min ▁▁▁▁
[36m(TaskRunner pid=1271585)[0m wandb:                        +158 ...
[36m(TaskRunner pid=1271585)[0m wandb: 
[36m(TaskRunner pid=1271585)[0m wandb: Run summary:
[36m(TaskRunner pid=1271585)[0m wandb:               actor/entropy 0.36945
[36m(TaskRunner pid=1271585)[0m wandb: behavior/valid_action_ratio 0.9869
[36m(TaskRunner pid=1271585)[0m wandb:       critic/advantages/max 5.31589
[36m(TaskRunner pid=1271585)[0m wandb:      critic/advantages/mean -0.0
[36m(TaskRunner pid=1271585)[0m wandb:       critic/advantages/min -6.21189
[36m(TaskRunner pid=1271585)[0m wandb:            critic/grad_norm 49.0375
[36m(TaskRunner pid=1271585)[0m wandb:                   critic/lr 1e-05
[36m(TaskRunner pid=1271585)[0m wandb:          critic/returns/max 3.62447
[36m(TaskRunner pid=1271585)[0m wandb:         critic/returns/mean 0.42377
[36m(TaskRunner pid=1271585)[0m wandb:          critic/returns/min -2.01
[36m(TaskRunner pid=1271585)[0m wandb:                        +159 ...
[36m(TaskRunner pid=1271585)[0m wandb: 
[36m(TaskRunner pid=1271585)[0m wandb: You can sync this run to the cloud by running:
[36m(TaskRunner pid=1271585)[0m wandb: wandb sync wandb/FS_PPO_4B_decay_A_run1_slu510942/wandb/offline-run-20260108_150939-vifd55go
[36m(TaskRunner pid=1271585)[0m wandb: Find logs at: wandb/FS_PPO_4B_decay_A_run1_slu510942/wandb/offline-run-20260108_150939-vifd55go/logs
[36m(TaskRunner pid=1271585)[0m 
Training Progress:   0%|          | 4/1000 [14:33<60:24:47, 218.36s/it]
Traceback (most recent call last):
  File "/iridisfs/home/jsbd1n24/verl-env/verl/trainer/main_ppo.py", line 44, in main
    run_ppo(config)
  File "/iridisfs/home/jsbd1n24/verl-env/verl/trainer/main_ppo.py", line 98, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/jsbd1n24/.conda/envs/verl/lib/python3.12/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/jsbd1n24/.conda/envs/verl/lib/python3.12/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/jsbd1n24/.conda/envs/verl/lib/python3.12/site-packages/ray/_private/worker.py", line 2967, in get
    values, debugger_breakpoint = worker.get_objects(
                                  ^^^^^^^^^^^^^^^^^^^
  File "/home/jsbd1n24/.conda/envs/verl/lib/python3.12/site-packages/ray/_private/worker.py", line 1015, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ValueError): [36mray::TaskRunner.run()[39m (pid=1271585, ip=127.0.0.1, actor_id=6daa7ef5cdf2784446ab9c7901000000, repr=<main_ppo.TaskRunner object at 0x7fc77b5dcf80>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/iridisfs/home/jsbd1n24/verl-env/verl/trainer/main_ppo.py", line 399, in run
    trainer.fit()
  File "/iridisfs/home/jsbd1n24/verl-env/verl/trainer/ppo/ray_multistep_trainer.py", line 1359, in fit
    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/iridisfs/home/jsbd1n24/verl-env/verl/utils/import_utils.py", line 152, in wrapped
    return obj(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/iridisfs/home/jsbd1n24/verl-env/verl/trainer/ppo/metric_utils.py", line 47, in reduce_metrics
    return reduce_metrics(metrics)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/iridisfs/home/jsbd1n24/verl-env/verl/utils/metric/utils.py", line 74, in reduce_metrics
    metrics[key] = np.mean(processed_val)
                   ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jsbd1n24/.conda/envs/verl/lib/python3.12/site-packages/numpy/_core/fromnumeric.py", line 3860, in mean
    return _methods._mean(a, axis=axis, dtype=dtype,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jsbd1n24/.conda/envs/verl/lib/python3.12/site-packages/numpy/_core/_methods.py", line 119, in _mean
    arr = asanyarray(a)
          ^^^^^^^^^^^^^
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.

```

