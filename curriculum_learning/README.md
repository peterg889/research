# Streamlined Curriculum Learning Framework

A clean, production-ready system for testing whether **training data order** affects language model learning performance.

## 🚀 Quick Start

```bash
# Debug experiment (1K samples, quick test)
python experiment.py --scale=debug

# Full research experiment (50K samples) 
python experiment.py --scale=scientific --use-wandb

# Custom experiment
python experiment.py --scale=pilot --model-size=bert-base --strategies random reading_level_easy_to_hard
```

## 📊 Research Questions

- **Does curriculum learning improve language model training?**
- **Which curriculum axis is more effective: reading level or semantic topics?**
- **How much improvement does strategic data ordering provide?**

## 🏗️ Architecture

**6 Core Files:**
- **`config.py`** - Single configuration system (debug → large scale)
- **`data_pipeline.py`** - Real data loading with curriculum strategies  
- **`model.py`** - Scalable BERT (mini → base) using HuggingFace
- **`evaluation.py`** - Statistical analysis with publication rigor
- **`experiment.py`** - Main experiment runner (CLI + programmatic)
- **`tracking.py`** - Experiment logging with optional W&B

## 🎯 Curriculum Strategies

**9 strategies across 2 main axes:**

**Reading Level Axis:**
- `reading_level_easy_to_hard` - Simple → complex texts
- `reading_level_hard_to_easy` - Complex → simple texts  
- `reading_level_staged` - Present each difficulty tier completely

**Semantic Topic Axis:**
- `topic_sequential` - Present each topic completely
- `topic_interleaved` - Evenly interleave topics
- `topic_largest_first` - Largest topics first

**Hybrid Approaches:**
- `hybrid_reading_topic` - Reading level first, then topic
- `hybrid_topic_reading` - Topic first, then reading level
- `random` - Baseline random ordering

## 📈 Experiment Scales

| Scale | Samples | Epochs | Use Case |
|-------|---------|--------|----------|
| `debug` | 1K | 2 | Quick testing |
| `pilot` | 10K | 5 | Initial validation |
| `scientific` | 50K | 15 | Publication-ready |
| `large` | 100K | 25 | Comprehensive study |

## 🔬 Scientific Features

- **Real Data**: WikiText + AG News + IMDB (not synthetic)
- **Automatic Labeling**: Flesch-Kincaid reading levels + BERTopic semantic topics
- **Statistical Rigor**: Effect sizes, significance testing, confidence intervals
- **Comprehensive Caching**: Avoids re-computation of expensive operations
- **Resource Monitoring**: Memory usage, training time estimation
- **Reproducibility**: Fixed seeds, configuration tracking

## 💻 Usage Examples

### Programmatic API
```python
from config import scientific_config
from experiment import Experiment

# Full research experiment
config = scientific_config(
    model_size="bert-base",
    strategies=["random", "reading_level_easy_to_hard", "topic_sequential"],
    use_wandb=True
)

experiment = Experiment(config)
results = experiment.run()

print(f"Best strategy: {results['report']['best_strategies']['by_final_performance']}")
```

### Quick Testing
```python
from config import debug_config
from experiment import Experiment

# Quick test
experiment = Experiment(debug_config())
results = experiment.run()
```

## 📊 Output & Results

**Automatic Generation:**
- Statistical analysis report (JSON)
- Training curves and metrics
- Resource usage statistics
- Recommendations for best strategies

**Example Output:**
```
🏆 Best strategy: reading_level_easy_to_hard
✅ Effective strategies: reading_level_easy_to_hard, topic_sequential
💡 Recommendations:
   1. reading_level_easy_to_hard shows 12.3% improvement over random
   2. Topic-based strategies show moderate benefits
   3. Consider hybrid approaches for maximum effect
```

## 🛠️ Requirements

```bash
pip install torch transformers datasets bertopic textstat scipy tqdm
pip install wandb  # Optional for experiment tracking
```

## 🎓 Research Applications

This framework enables research into:
- **Curriculum Learning**: Strategic data ordering for better learning
- **Language Model Training**: Effects of data presentation order
- **Educational Technology**: Optimal content sequencing
- **Transfer Learning**: Curriculum effects in domain adaptation

## 📚 Key Papers

- Bengio et al. (2009): Curriculum Learning
- Platanios et al. (2019): Competence-based Curriculum Learning  
- Recent work on curriculum learning for NLP

## 🔧 Configuration Options

**Model Sizes:** `bert-mini` (11M), `bert-small` (41M), `bert-base` (110M)

**Data Sources:** WikiText, AG News, IMDB (automatically balanced)

**Training:** Masked Language Modeling with proper MLM head and loss

**Evaluation:** Convergence speed, final performance, learning efficiency

---

**Simple. Scalable. Scientific.** 🚀