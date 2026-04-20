#!/usr/bin/env bash
# Regenerate every figure and CSV from scratch.
# Usage:  bash scripts/reproduce.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== 1/4 Boundary sanity checks (M1 milestone) ==="
pytest tests/test_boundaries.py -v

echo "=== 2/4 Full reproducibility pipeline ==="
python -m experiments.reproduce_all

echo "All done. CSVs at results/csv/, figures at results/figures/."
