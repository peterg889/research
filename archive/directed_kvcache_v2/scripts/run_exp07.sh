#!/bin/bash
# Run Exp 07: Static Surrogates, Dual-Mode Priming, Intent Routing
# Estimated runtime: ~10.8 hours (5.5h surrogate gen + 5.1h eval + 5min embedding)
#
# Usage: bash scripts/run_exp07.sh
set -e

umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NOTEBOOK="07_static_surrogates_and_routing.ipynb"
OUTPUT="${NOTEBOOK%.ipynb}_executed.ipynb"
LOG="results/exp07/run.log"

echo "=============================================="
echo "Exp 07: Static Surrogates & Routing"
echo "=============================================="
echo "Start time: $(date)"
echo "Notebook: $NOTEBOOK"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo ""

mkdir -p results/exp07/surrogates

# Build notebook from build script
echo "Building notebook from scripts/build_nb07.py..."
python scripts/build_nb07.py
echo ""

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
echo "Exp 07 complete!"
echo "End time: $(date)"
echo "Output notebook: $OUTPUT"
echo "Results: results/exp07/results.json"
echo "=============================================="
