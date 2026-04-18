#!/usr/bin/env bash
set -euo pipefail

sbatch scripts/slurm/dcpinn_DEXI_084_smoke_pack1.sbatch
sbatch scripts/slurm/dcpinn_DEXI_084_smoke_pack2.sbatch
