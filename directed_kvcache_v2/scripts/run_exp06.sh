#!/bin/bash
# Run Exp 06: Surrogate Deep-Dive — Mechanism Decomposition
# Estimated runtime: ~11.5 hours (5h surrogate gen + 6.5h eval)
#
# Usage: bash scripts/run_exp06.sh
set -e

umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NOTEBOOK="06_surrogate_deep_dive.ipynb"
OUTPUT="${NOTEBOOK%.ipynb}_executed.ipynb"
LOG="results/exp06/run.log"

echo "=============================================="
echo "Exp 06: Surrogate Deep-Dive"
echo "=============================================="
echo "Start time: $(date)"
echo "Notebook: $NOTEBOOK"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo ""

mkdir -p results/exp06/surrogates

# Check GPU
python -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}, {props.total_memory / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected')
"

echo ""
echo "Starting notebook execution..."
echo ""

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=86400 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "$OUTPUT" \
    "$NOTEBOOK" \
    2>&1 | tee "$LOG"

echo ""
echo "=============================================="
echo "Exp 06 complete!"
echo "End time: $(date)"
echo "Output notebook: $OUTPUT"
echo "Results: results/exp06/results.json"
echo "=============================================="
