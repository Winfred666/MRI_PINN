#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  sbatch --dependency="${SBATCH_DEPENDENCY}" scripts/slurm/dcpinn_dexi083_optcase_pack1.sbatch
else
  sbatch scripts/slurm/dcpinn_dexi083_optcase_pack1.sbatch
fi
