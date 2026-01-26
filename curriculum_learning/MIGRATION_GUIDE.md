# Migration Guide: Unified Curriculum Learning System

## Overview

The curriculum learning codebase has been consolidated from multiple fragmented components into a unified, cohesive system. This guide explains how to migrate from the old system to the new unified approach.

## What Changed

### 🗂️ File Consolidation

**Removed (obsolete):**
- `experiment_enhanced.py` 
- `experiment_memory_efficient.py`
- `config_fair_comparison.py`
- Separate tracking files

**New Unified Files:**
- `unified_experiment.py` - Single experiment runner with all modes
- Enhanced `config.py` - Includes fair comparison features
- Enhanced `enhanced_tracking.py` - Used by unified system

### 🔄 API Changes

#### Old Way (Multiple Classes)
```python
# OLD: Multiple experiment classes
from experiment import Experiment
from experiment_enhanced import EnhancedExperiment  
from experiment_memory_efficient import MemoryEfficientExperiment
from config_fair_comparison import FairComparisonConfig

# Different classes for different needs
experiment = EnhancedExperiment(config)
```

#### New Way (Unified Interface)
```python
# NEW: Single class with modes
from unified_experiment import UnifiedExperiment, ExperimentMode
from config import fair_comparison_config

# One class, different modes
experiment = UnifiedExperiment(config, ExperimentMode.ENHANCED)
```

## Migration Steps

### 1. Update Imports

**Before:**
```python
from experiment_enhanced import EnhancedExperiment
from config_fair_comparison import FairComparisonConfig
```

**After:**
```python
from unified_experiment import UnifiedExperiment, ExperimentMode
from config import fair_comparison_config
```

### 2. Update Experiment Creation

**Before:**
```python
# Enhanced experiment
config = Config(scale="scientific")
experiment = EnhancedExperiment(config)

# Memory efficient 
experiment = MemoryEfficientExperiment(config)

# Fair comparison
config = FairComparisonConfig(scale="pilot")
experiment = EnhancedExperiment(config)
```

**After:**
```python
# Enhanced experiment
config = Config(scale="scientific")
experiment = UnifiedExperiment(config, ExperimentMode.ENHANCED)

# Memory efficient
experiment = UnifiedExperiment(config, ExperimentMode.MEMORY_EFFICIENT)

# Fair comparison
config = fair_comparison_config(scale="pilot")
experiment = UnifiedExperiment(config, ExperimentMode.FAIR_COMPARISON)
```

### 3. Use Convenience Functions

**Simplest Migration:**
```python
# OLD
from experiment_enhanced import EnhancedExperiment
config = Config(scale="scientific")
experiment = EnhancedExperiment(config)
results = experiment.run()

# NEW
from unified_experiment import run_enhanced_experiment
config = Config(scale="scientific")
results = run_enhanced_experiment(config)
```

## Experiment Modes

### 1. Basic Mode (`ExperimentMode.BASIC`)
- **Use for:** Quick testing, simple experiments
- **Features:** Standard training, minimal tracking
- **Replaces:** Original `Experiment` class

```python
from unified_experiment import run_basic_experiment
results = run_basic_experiment(debug_config())
```

### 2. Enhanced Mode (`ExperimentMode.ENHANCED`)
- **Use for:** Research experiments, detailed analysis
- **Features:** Better metrics, statistical analysis, W&B integration
- **Replaces:** `EnhancedExperiment` class

```python
from unified_experiment import run_enhanced_experiment
results = run_enhanced_experiment(scientific_config(use_wandb=True))
```

### 3. Memory Efficient Mode (`ExperimentMode.MEMORY_EFFICIENT`)
- **Use for:** Large datasets, limited RAM
- **Features:** Aggressive cleanup, memory monitoring
- **Replaces:** `MemoryEfficientExperiment` class

```python
from unified_experiment import run_memory_efficient_experiment
results = run_memory_efficient_experiment(massive_config())
```

### 4. Fair Comparison Mode (`ExperimentMode.FAIR_COMPARISON`)
- **Use for:** Unbiased curriculum comparison research
- **Features:** No early stopping, equal training steps
- **Replaces:** `FairComparisonConfig` + enhanced experiment

```python
from unified_experiment import run_fair_comparison_experiment
config = fair_comparison_config(scale="scientific")
results = run_fair_comparison_experiment(config)
```

## Configuration Changes

### Fair Comparison

**Before:**
```python
from config_fair_comparison import create_fair_pilot_config
config = create_fair_pilot_config()
```

**After:**
```python
from config import fair_comparison_config
config = fair_comparison_config(scale="pilot")
```

### Memory Settings

**Before:**
```python
config = Config(scale="large")
config.memory_efficient = True
experiment = MemoryEfficientExperiment(config)
```

**After:**
```python
config = Config(scale="large", memory_efficient=True)
experiment = UnifiedExperiment(config, ExperimentMode.MEMORY_EFFICIENT)
```

## Results Structure

### Enhanced Results Object

The new system returns a structured `ExperimentResults` object:

```python
@dataclass
class ExperimentResults:
    strategy_results: Dict[str, Dict[str, List[float]]]
    experiment_summary: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    convergence_analysis: Dict[str, Any]
    resource_usage: Dict[str, float]

# Usage
results = run_enhanced_experiment(config)
print(f"Best strategy: {results.experiment_summary['best_strategies']['by_final_performance']}")
print(f"Runtime: {results.resource_usage['total_runtime_hours']:.2f} hours")
```

## Jupyter Notebook Migration

### Old Notebook Structure
```python
# OLD notebook pattern
from experiment_enhanced import EnhancedExperiment
from config_fair_comparison import FairComparisonConfig

config = FairComparisonConfig(scale="pilot")
experiment = EnhancedExperiment(config)
results = experiment.run()
```

### New Notebook Structure
```python
# NEW notebook pattern
from unified_experiment import UnifiedExperiment, ExperimentMode
from config import fair_comparison_config

config = fair_comparison_config(scale="pilot", use_wandb=True)
experiment = UnifiedExperiment(config, ExperimentMode.FAIR_COMPARISON)
results = experiment.run()

# Or even simpler:
from unified_experiment import run_fair_comparison_experiment
results = run_fair_comparison_experiment(config)
```

## Benefits of Migration

### ✅ Reduced Complexity
- **Before:** 3 experiment classes + separate configs
- **After:** 1 unified class with 4 modes

### ✅ Better Maintainability  
- Single codebase to maintain
- Consistent API across all modes
- Shared bug fixes and improvements

### ✅ Enhanced Features
- All modes get latest improvements
- Unified tracking and logging
- Better error handling

### ✅ Easier Testing
- Single interface to test
- Mode switching for different scenarios
- Consistent results structure

## Common Migration Issues

### 1. Import Errors
**Error:** `ModuleNotFoundError: No module named 'experiment_enhanced'`

**Solution:** Update imports to use unified system:
```python
# OLD
from experiment_enhanced import EnhancedExperiment

# NEW  
from unified_experiment import UnifiedExperiment, ExperimentMode
```

### 2. Configuration Issues
**Error:** `AttributeError: 'Config' object has no attribute 'detailed_logging'`

**Solution:** Use `fair_comparison_config()` for fair comparison features:
```python
# OLD
config = FairComparisonConfig(scale="pilot")

# NEW
config = fair_comparison_config(scale="pilot")
```

### 3. Results Access
**Error:** Different results structure

**Solution:** Use the new structured results:
```python
# OLD
best_strategy = results['report']['best_strategies']['by_final_performance']

# NEW
best_strategy = results.experiment_summary['best_strategies']['by_final_performance']
```

## Backward Compatibility

The old experiment classes are **deprecated but still functional** for a transition period. However, they will not receive updates or bug fixes. 

**Recommendation:** Migrate to the unified system as soon as possible to benefit from:
- Latest bug fixes
- Performance improvements  
- New features
- Better documentation and support

## Getting Help

1. **Check this migration guide** for common patterns
2. **Look at the updated notebook** for example usage
3. **Run the test examples** in `unified_experiment.py`
4. **Check the docstrings** for detailed API documentation

## Quick Reference

| Old Component | New Equivalent |
|---------------|----------------|
| `Experiment` | `UnifiedExperiment(..., ExperimentMode.BASIC)` |
| `EnhancedExperiment` | `UnifiedExperiment(..., ExperimentMode.ENHANCED)` |
| `MemoryEfficientExperiment` | `UnifiedExperiment(..., ExperimentMode.MEMORY_EFFICIENT)` |
| `FairComparisonConfig` | `fair_comparison_config()` |
| Multiple tracking classes | `UnifiedExperimentTracker` (automatic) |

The unified system maintains all functionality while providing a cleaner, more maintainable interface.