#!/usr/bin/env bash
# Regenerate the progression-report figures + markdown from current caches.
# Doesn't refetch from WandB — see fetch_data.py and fetch_history_per_key.py
# for that. Run those once when you want fresh data.
set -euo pipefail
cd "$(dirname "$0")/../.."   # → repo root

PY="/opt/anaconda3/bin/conda run -n verlog python -m"

$PY analysis.progression_report.plot_performance
$PY analysis.progression_report.plot_entropy_methods
$PY analysis.progression_report.generate_report

echo
echo "==> figures + report at: figures/progression_report/"
ls -1 figures/progression_report/
