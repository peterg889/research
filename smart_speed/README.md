# Entropy-Based Speculative Decoding

## Hypothesis

**Can entropy-based stopping criteria achieve better efficiency than fixed-k speculative decoding by dynamically adjusting draft length based on model confidence?**

In standard speculative decoding, a small "drafter" model generates a fixed number (k) of tokens that a larger "verifier" model then accepts or rejects. This experiment tests whether using the drafter's output entropy as a dynamic stopping criterion can outperform fixed-k approaches by:

1. Generating more tokens when the drafter is confident (low entropy)
2. Stopping earlier when the drafter is uncertain (high entropy)

## Experimental Design

### Conditions

| Method | Description |
|--------|-------------|
| **Fixed-k baseline** | Standard speculative decoding with k=4, 6, 8, 10 |
| **Entropy-absolute** | Stop when entropy exceeds absolute threshold θ |
| **Entropy-relative** | Stop when entropy increases by factor r over initial |
| **Entropy-window** | Stop when entropy over sliding window exceeds threshold |
| **Entropy-combined** | Multiple criteria combined |
| **Entropy-adaptive** | Threshold adapts based on running statistics |

### Metrics

- **Primary**: Tokens per second (TPS), acceptance rate
- **Secondary**: Draft length distribution, latency (P50/P90/P99)
- **Diagnostic**: Entropy-acceptance correlation, position-wise acceptance

### Success Criteria

- **H1**: EBSD achieves >10% speedup over best fixed-k baseline on ≥50% of tasks
- **H2**: Optimal entropy threshold correlates with task complexity
- **H3**: Relative entropy stopping is more robust than absolute threshold

## Quick Start

```python
from speculative_decoding import EntropyConfig, EntropyBasedSpeculativeDecoder
from experiment import ExperimentConfig, ExperimentRunner

# Configure experiment
config = ExperimentConfig(
    drafter_model_path="path/to/small-model",
    verifier_model_path="path/to/large-model",
    dataset_name="wikitext",
    dataset_split="test",
    num_samples=1000,
    max_sequence_length=512,
    experiment_name="entropy_exp_v1",
    output_dir="./results/entropy_exp_v1"
)

# Run experiments
runner = ExperimentRunner(config)
runner.run_full_experiment()

# Analyze results
from analysis import AnalysisPipeline
pipeline = AnalysisPipeline(
    experiment_dir="./results/entropy_exp_v1",
    output_dir="./analysis/entropy_exp_v1"
)
pipeline.run_full_analysis()
```

## Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `theta_abs` | Absolute entropy threshold | 0.5 - 2.5 |
| `theta_rel` | Relative entropy ratio | 1.2 - 2.5 |
| `max_draft_len` | Maximum draft tokens | 4 - 24 |
| `entropy_type` | Entropy calculation | shannon, renyi, topk |
| `strategy` | Stopping strategy | absolute, relative, window, combined, adaptive |

## File Structure

```
smart_speed/
├── speculative_decoding.py  # Core entropy-based decoding implementation
├── experiment.py            # Experiment runner with ablations
├── evaluation.py            # Metrics and evaluation
├── analysis.py              # Statistical analysis and visualization
└── README.md
```

## Key References

- Leviathan, Y., et al. (2023). Fast Inference from Transformers via Speculative Decoding.
- Chen, C., et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling.
