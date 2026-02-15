#!/bin/bash
# Run Exp 11: Long-Document Priming with Natural Questions
# Estimated runtime: ~4-6 hours (30min surrogate gen + 3-5h eval)
#
# Usage: bash scripts/run_exp11.sh
set -e

umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NOTEBOOK="11_long_document_priming.ipynb"
OUTPUT="${NOTEBOOK%.ipynb}_executed.ipynb"
LOG="results/exp11/run.log"

echo "=============================================="
echo "Exp 11: Long-Document Priming (Natural Questions)"
echo "=============================================="
echo "Start time: $(date)"
echo "Notebook: $NOTEBOOK"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo ""

mkdir -p results/exp11/surrogates

# Build notebook from build script
echo "Building notebook from scripts/build_nb11.py..."
python scripts/build_nb11.py
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
echo "Exp 11 complete!"
echo "End time: $(date)"
echo "Output notebook: $OUTPUT"
echo "Results: results/exp11/results.json"
echo "=============================================="
