#!/usr/bin/env bash
#SBATCH -p gpu03,H100
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -t 5-00:00:00
#SBATCH -J dexi083_optcase
#SBATCH -o results/slurm_logs/%x-%j.out
#SBATCH -e results/slurm_logs/%x-%j.err

set -euo pipefail

set +u
source ~/.bashrc
set -u
if command -v conda >/dev/null 2>&1; then
  conda activate mr_pino || true
fi

ROOT_DIR="/public/home/bairuilianggroup/winfred/workspace/MRI_PINN"
cd "$ROOT_DIR"

mkdir -p results/slurm_logs results/slurm

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/${USER}/slurm_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export MLFLOW_TRACKING_URI="http://172.100.100.100:5000"
export NO_PROXY="localhost,127.0.0.1,::1,172.100.100.100"
export no_proxy="$NO_PROXY"

CASE_CONFIGS=(
  "configs/sim_series/dcpinn_dexi083_optcase9_10.yaml"
  "configs/sim_series/dcpinn_dexi083_optcase10_11.yaml"
  "configs/sim_series/dcpinn_dexi083_optcase11_12.yaml"
  "configs/sim_series/dcpinn_dexi083_optcase12_13.yaml"
  "configs/sim_series/dcpinn_dexi083_optcase13_14.yaml"
  "configs/sim_series/dcpinn_dexi083_optcase14_15.yaml"
  "configs/sim_series/dcpinn_dexi083_optcase15_16.yaml"
)

SUMMARY_PATH="results/slurm_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.summary.tsv"
echo -e "case\tstatus\tlog_path" > "$SUMMARY_PATH"

run_case() {
  local cfg="$1"
  local case_name
  local log_path
  case_name=$(basename "$cfg" .yaml)
  log_path="results/slurm_logs/${case_name}-${SLURM_JOB_ID}.log"

  echo "CASE_START ${case_name} $(date -Iseconds) log=${log_path}"
  if python -u driver.py "$cfg" > "$log_path" 2>&1; then
    echo -e "${case_name}\tDONE\t${log_path}" >> "$SUMMARY_PATH"
    echo "CASE_DONE ${case_name} $(date -Iseconds)"
  else
    local status=$?
    echo -e "${case_name}\tFAIL(${status})\t${log_path}" >> "$SUMMARY_PATH"
    echo "CASE_FAIL ${case_name} status=${status} $(date -Iseconds)"
    return "$status"
  fi
}

pids=()
for cfg in "${CASE_CONFIGS[@]}"; do
  run_case "$cfg" &
  pids+=("$!")
done

fail_count=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail_count=$((fail_count + 1))
  fi
done

echo "PACK_DONE fail_count=${fail_count} summary=${SUMMARY_PATH}"
if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi
