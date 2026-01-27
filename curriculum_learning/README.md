# Curriculum Learning for Language Models

## Hypothesis

**Does the order in which training data is presented affect the speed and quality of language model training?**

Curriculum learning theory (Bengio et al., 2009) suggests that presenting training examples in a meaningful order—typically easy-to-hard—can accelerate convergence and improve final model performance. This experiment tests whether strategic data ordering provides measurable benefits for BERT-style masked language model training.

## Experimental Design

### Conditions

We compare **9 curriculum strategies** across two main axes:

| Axis | Strategies | Description |
|------|------------|-------------|
| **Reading Level** | `reading_level_easy_to_hard` | Simple texts first, complex texts later |
| | `reading_level_hard_to_easy` | Complex texts first |
| | `reading_level_staged` | Discrete difficulty tiers |
| **Semantic Topic** | `topic_sequential` | Complete each topic before moving on |
| | `topic_interleaved` | Evenly interleave topics |
| | `topic_largest_first` | Largest topic clusters first |
| **Hybrid** | `hybrid_reading_topic` | Reading level, then topic |
| | `hybrid_topic_reading` | Topic, then reading level |
| **Baseline** | `random` | Random ordering (control) |

### Metrics

- **Primary**: Final validation loss and accuracy after fixed training steps
- **Secondary**: Convergence speed (steps to reach threshold), learning efficiency (area under loss curve)
- **Statistical**: Effect sizes (Cohen's d), paired t-tests, confidence intervals

### Data

Real text data from WikiText, AG News, and IMDB. Each sample is automatically labeled with:
- **Reading level**: Flesch-Kincaid grade score
- **Topic**: BERTopic-derived semantic cluster

## Quick Start

```bash
# Quick test (1K samples, 2 epochs)
python experiment.py --scale=debug

# Research experiment (50K samples, 15 epochs)
python experiment.py --scale=scientific --use-wandb

# Custom run
python experiment.py --scale=pilot --strategies random reading_level_easy_to_hard topic_sequential
```

## Programmatic API

```python
from config import scientific_config
from experiment import UnifiedExperiment

config = scientific_config(
    model_size="bert-base",
    strategies=["random", "reading_level_easy_to_hard", "topic_sequential"]
)

experiment = UnifiedExperiment(config)
results = experiment.run()
```

## Experiment Scales

| Scale | Samples | Epochs | Purpose |
|-------|---------|--------|---------|
| `debug` | 1K | 2 | Verify code works |
| `pilot` | 10K | 5 | Initial validation |
| `scientific` | 50K | 15 | Publication-ready results |
| `large` | 100K | 25 | Comprehensive study |

## File Structure

```
curriculum_learning/
├── config.py          # Experiment configuration (scales, strategies, hyperparameters)
├── experiment.py      # Main experiment runner
├── data_pipeline.py   # Data loading and curriculum ordering
├── model.py           # BERT model implementation
├── evaluation.py      # Statistical analysis
├── tracking.py        # Logging and W&B integration
└── requirements.txt   # Dependencies
```

## Requirements

```bash
pip install -r requirements.txt
```

## Key References

- Bengio, Y., et al. (2009). Curriculum Learning. ICML.
- Platanios, E., et al. (2019). Competence-based Curriculum Learning. NAACL.
