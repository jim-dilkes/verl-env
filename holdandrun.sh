#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  holdandrun.sh --submit <sbatch1> [<sbatch2> ...]
               [--cancel <jobid1> [<jobid2> ...]]
               [--exclude-regex <regex>]
               [--no-release]

What it does:
  1) Holds your PENDING jobs (optionally excluding some by name regex)
  2) Optionally cancels specified jobs
  3) Submits one or more sbatch scripts
  4) Releases held jobs (unless --no-release)

Examples:
  # Queue OC reruns first: hold non-OC pending jobs, cancel OC jobs, resubmit OC scripts,
  # and keep non-OC jobs held until you're ready to release.
  ./holdandrun.sh --exclude-regex '^OC_' --no-release \
    --cancel 528640 528641 528642 528646 528647 \
    --submit \
      /iridisfs/home/$USER/verl-env/experiments/overcooked/260116_action_mode_comparison/OC_PPO_4B_BL_1.sbatch \
      /iridisfs/home/$USER/verl-env/experiments/overcooked/260116_action_mode_comparison/OC_PPO_4B_multi_eps02_1.sbatch \
      /iridisfs/home/$USER/verl-env/experiments/overcooked/260116_action_mode_comparison/OC_PPO_4B_multi_eps0_1.sbatch \
    /iridisfs/home/$USER/verl-env/experiments/overcooked/260115_14B_multi_action_combined/OC_PPO_14B_NT_MA_eps0_1.sbatch \
      /iridisfs/home/$USER/verl-env/experiments/overcooked/260115_14B_multi_action_combined/OC_PPO_14B_NT_MA_eps02_1.sbatch

  # Later, release everything still on hold:
    squeue -u "$USER" -h -t PD -o "%i" | xargs -r scontrol release
EOF
}

exclude_regex='^$'
do_release=true
cancel_ids=()
submit_files=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --exclude-regex)
            exclude_regex="$2"
            shift 2
            ;;
        --no-release)
            do_release=false
            shift
            ;;
        --cancel)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                cancel_ids+=("$1")
                shift
            done
            ;;
        --submit)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                submit_files+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown arg: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ((${#submit_files[@]} == 0)); then
    echo "No sbatch files provided. Use --submit <sbatch...>" >&2
    usage >&2
    exit 2
fi

# Hold PENDING jobs (excluding by name regex)
held_ids=()
while read -r jobid jobname; do
    [[ -z "$jobid" ]] && continue
    if [[ "$jobname" =~ $exclude_regex ]]; then
        continue
    fi
    scontrol hold "$jobid"
    held_ids+=("$jobid")
done < <(squeue -u "$USER" -h -t PD -o "%i %j")

# Cancel requested jobs
if ((${#cancel_ids[@]} > 0)); then
    scancel "${cancel_ids[@]}"
fi

# Submit requested sbatch scripts
for sb in "${submit_files[@]}"; do
    sbatch "$sb"
done

# Release held jobs (optional)
if "$do_release"; then
    for jobid in "${held_ids[@]}"; do
        scontrol release "$jobid" || true
    done
fi