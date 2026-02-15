#!/bin/bash
# Run Exp 12: Long-Document Priming Diagnostic Battery
# Tests 9 conditions (repetition, amplification, suffix, no_rope, layers)
# Estimated runtime: ~6-8 hours (5 builds + 9 scores per sample x 400 samples)
#
# Usage: bash scripts/run_exp12.sh
set -e

umask 000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NOTEBOOK="12_long_doc_priming_diagnostic.ipynb"
OUTPUT="${NOTEBOOK%.ipynb}_executed.ipynb"
LOG="results/exp12/run.log"

echo "=============================================="
echo "Exp 12: Long-Doc Priming Diagnostic Battery"
echo "=============================================="
echo "Start time: $(date)"
echo "Notebook: $NOTEBOOK"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo ""
echo "Conditions: bare, prefix_1x, prefix_5x, prefix_20x,"
echo "  amplify_2x, amplify_5x, layers_0_15, suffix, no_rope"
echo "Samples: 400 (100 per length bin)"
echo ""

mkdir -p results/exp12

# Build notebook from build script
echo "Building notebook from scripts/build_nb12.py..."
python scripts/build_nb12.py
echo ""

# Check GPU
python -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}, {props.total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected')
" 2>/dev/null || echo "GPU check skipped"

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
echo "Exp 12 complete!"
echo "End time: $(date)"
echo "Output notebook: $OUTPUT"
echo "Results: results/exp12/results.json"
echo "=============================================="
