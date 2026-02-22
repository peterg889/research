#!/bin/bash
# Run Exp 08: Mechanism Isolation + Prefix Amplification
# Estimated runtime: ~2 hours (0.6h surrogate gen + 1.2h eval)
#
# Usage: bash scripts/run_exp08.sh
set -e

umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NOTEBOOK="08_mechanism_and_amplification.ipynb"
OUTPUT="${NOTEBOOK%.ipynb}_executed.ipynb"
LOG="results/exp08/run.log"

echo "=============================================="
echo "Exp 08: Mechanism Isolation + Amplification"
echo "=============================================="
echo "Start time: $(date)"
echo "Notebook: $NOTEBOOK"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo ""

mkdir -p results/exp08/surrogates

# Build notebook from build script
echo "Building notebook from scripts/build_nb08.py..."
python scripts/build_nb08.py
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
echo "Exp 08 complete!"
echo "End time: $(date)"
echo "Output notebook: $OUTPUT"
echo "Results: results/exp08/results.json"
echo "=============================================="
