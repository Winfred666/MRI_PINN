#!/usr/bin/env bash
set -euo pipefail

dependency_args=()
if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  dependency_args=(--dependency="${SBATCH_DEPENDENCY}")
fi

sbatch "${dependency_args[@]}" scripts/slurm/dcpinn_DEXI_084_pack1.sbatch
sbatch "${dependency_args[@]}" scripts/slurm/dcpinn_DEXI_084_pack2.sbatch
