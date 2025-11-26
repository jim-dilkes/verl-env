# Hold all pending jobs
for jobid in $(squeue -u $USER -h -o "%i"); do
    scontrol hold $jobid
done

# Submit your new job
sbatch /home/jsbd1n24/verl-env/experiments/snake/251120_run_0_vllm9/FS_PPO_Q3_4B_perf_O_rolloutMem50.sbatch

# Release held jobs
for jobid in $(squeue -u $USER -h -o "%i"); do
    scontrol release $jobid
done