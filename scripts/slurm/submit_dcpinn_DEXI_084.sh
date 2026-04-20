#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  sbatch --dependency="${SBATCH_DEPENDENCY}" scripts/slurm/dcpinn_DEXI_084_pack1.sbatch
  sbatch --dependency="${SBATCH_DEPENDENCY}" scripts/slurm/dcpinn_DEXI_084_pack2.sbatch
else
  sbatch scripts/slurm/dcpinn_DEXI_084_pack1.sbatch
  sbatch scripts/slurm/dcpinn_DEXI_084_pack2.sbatch
fi
