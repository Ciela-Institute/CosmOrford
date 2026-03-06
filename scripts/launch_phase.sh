#!/usr/bin/env bash
# scripts/launch_phase.sh — Launch all experiments for a given phase
# Usage: ./scripts/launch_phase.sh 1|2|3
set -euo pipefail
PHASE="${1:?Usage: $0 <phase-number>}"

for config in configs/experiments/phase${PHASE}_*.yaml; do
    name=$(basename "$config" .yaml)
    echo "Launching: $name"
    modal run train_modal.py --config "$config" --name "$name"
done
