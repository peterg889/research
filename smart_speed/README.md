# Entropy-Based Speculative Decoding

A comprehensive experimental framework for evaluating entropy-based speculative decoding, where the drafter model continues generating tokens until its output entropy exceeds a threshold, indicating reduced confidence.

## Key Innovation

Unlike traditional speculative decoding with fixed k-token windows, this approach:
- Dynamically adjusts draft length based on model confidence
- Uses entropy as a proxy for prediction uncertainty
- Potentially achieves better efficiency when drafter confidence varies

## Architecture Overview

### 1. Core Implementation (`implementation_architecture.py`)
- **EntropyComputer**: Supports Shannon, Rényi, and top-k entropy calculations
- **StoppingCriterion**: Multiple strategies (absolute, relative, window, adaptive)
- **EntropyBasedDrafter**: Generates tokens until entropy threshold exceeded
- **EntropyBasedSpeculativeDecoder**: Main decoder with verification logic

### 2. Evaluation Metrics (`evaluation_metrics.py`)
- **SpeculativeDecodingEvaluator**: Tracks generation metrics
- **ComparativeEvaluator**: Compares different methods
- Comprehensive metrics: TPS, acceptance rates, latency distributions, entropy correlations

### 3. Experiment Runner (`experiment_runner.py`)
- Automated experiment execution
- Hyperparameter grid search
- Ablation studies
- Baseline comparisons

### 4. Analysis Pipeline (`data_analysis_pipeline.py`)
- Statistical analysis
- Performance clustering
- Visualization generation
- Comprehensive reporting

## Quick Start

```python
from implementation_architecture import EntropyConfig, EntropyBasedSpeculativeDecoder
from experiment_runner import ExperimentConfig, ExperimentRunner

# Configure experiment
config = ExperimentConfig(
    drafter_model_path="path/to/small-model",
    verifier_model_path="path/to/large-model",
    dataset_name="wikitext",
    dataset_split="test",
    num_samples=1000,
    max_sequence_length=512,
    experiment_name="entropy_exp_v1",
    output_dir="./experiments/entropy_exp_v1"
)

# Run experiments
runner = ExperimentRunner(config)
runner.run_full_experiment()

# Analyze results
from data_analysis_pipeline import AnalysisPipeline
pipeline = AnalysisPipeline(
    experiment_dir="./experiments/entropy_exp_v1",
    output_dir="./analysis/entropy_exp_v1"
)
pipeline.run_full_analysis()
```

## Experiment Plan Details

See `entropy_speculative_decoding_plan.md` for:
- Theoretical foundation
- Detailed methodology
- Expected outcomes
- Timeline

## Key Hyperparameters

- `theta_abs`: Absolute entropy threshold (e.g., 1.5)
- `theta_rel`: Relative entropy increase ratio (e.g., 1.5)
- `strategy`: Stopping strategy (absolute, relative, window, combined, adaptive)
- `entropy_type`: Entropy calculation method (shannon, renyi, topk)
- `max_draft_len`: Maximum draft sequence length

## Results Structure

```
experiments/
├── configs/          # Experiment configurations
├── results/          # Metrics JSON files
├── plots/           # Generated visualizations
└── checkpoints/     # Intermediate results

analysis/
├── processed_data.csv
├── entropy_analysis.png
├── performance_analysis.png
├── comparison_analysis.png
└── final_report.md
```

## Citation

If you use this framework, please cite:
```
@software{entropy_speculative_decoding,
  title={Entropy-Based Speculative Decoding},
  year={2024}
}
```