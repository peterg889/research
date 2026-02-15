#!/bin/bash
# Run Exp 10: Semantic Content Gradient
# Estimated runtime: ~15 hours (2h surrogate gen + 13h eval)
#
# Usage: bash scripts/run_exp10.sh
set -e

umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NOTEBOOK="10_semantic_content_gradient.ipynb"
OUTPUT="${NOTEBOOK%.ipynb}_executed.ipynb"
LOG="results/exp10/run.log"

echo "=============================================="
echo "Exp 10: Semantic Content Gradient"
echo "=============================================="
echo "Start time: $(date)"
echo "Notebook: $NOTEBOOK"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo ""

mkdir -p results/exp10/surrogates

# Build notebook from build script
echo "Building notebook from scripts/build_nb10.py..."
python scripts/build_nb10.py
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
echo "Exp 10 complete!"
echo "End time: $(date)"
echo "Output notebook: $OUTPUT"
echo "Results: results/exp10/results.json"
echo "=============================================="
