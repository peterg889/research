"""
Unified Configuration System for Curriculum Learning

Simple, scalable configuration from debug experiments to large-scale research.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json
from pathlib import Path


class Scale(Enum):
    """Experiment scale presets for different research phases"""
    DEBUG = "debug"           # Quick testing: 1K samples, 2 epochs
    PILOT = "pilot"           # Initial validation: 10K samples, 5 epochs  
    SCIENTIFIC = "scientific" # Publication-ready: 50K samples, 15 epochs
    LARGE = "large"          # Comprehensive: 100K samples, 25 epochs
    MASSIVE = "massive"      # Full-scale: 500K samples, 50 epochs
    EXTREME = "extreme"      # Maximum scale: 1M+ samples, 100 epochs


@dataclass
class Config:
    """
    Unified configuration for curriculum learning experiments.
    
    Simple usage:
        config = Config(scale="debug")  # Quick test
        config = Config(scale="scientific")  # Full research
    """
    
    # Core settings
    scale: str = "scientific"
    model_size: str = "bert-small"  # bert-mini, bert-small, bert-base
    strategies: Optional[List[str]] = None  # None = use all strategies
    
    # Experiment settings (auto-configured by scale)
    num_samples: Optional[int] = None
    num_epochs: Optional[int] = None
    num_runs: int = 5
    batch_size: Optional[int] = None
    
    # Model architecture (auto-configured by model_size)
    vocab_size: int = 30522
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    max_seq_length: int = 256
    
    # Training parameters
    learning_rate: float = 2e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Data configuration
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Curriculum parameters
    use_bertopic: bool = True
    min_topic_size: int = 10
    
    # Evaluation parameters
    eval_every_n_steps: int = 250
    save_every_n_steps: int = 1000
    early_stopping_patience: int = 5
    
    # Statistical parameters
    confidence_level: float = 0.95
    effect_size_threshold: float = 0.1
    
    # Infrastructure
    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 4
    cache_dir: str = "./cache"
    results_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"
    
    # Memory efficiency settings
    memory_efficient: bool = False  # Enable memory-efficient mode
    max_memory_gb: float = 16.0  # Maximum memory usage in GB
    
    # Fair comparison settings
    use_early_stopping: bool = True  # Can be disabled for fair comparison
    early_stopping_patience: int = 7
    ensure_equal_steps: bool = False  # Force all strategies to train for same steps
    random_seed: int = 42  # For reproducibility
    
    # Tracking
    use_wandb: bool = False
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        """Auto-configure settings based on scale and model size"""
        # Convert scale string to enum
        if isinstance(self.scale, str):
            self.scale = Scale(self.scale)
        
        # Auto-configure based on scale
        scale_configs = {
            Scale.DEBUG: {
                'num_samples': 1000,
                'num_epochs': 2,
                'batch_size': 16,
                'eval_every_n_steps': 50,
                'min_topic_size': 10
            },
            Scale.PILOT: {
                'num_samples': 10000,
                'num_epochs': 5,
                'batch_size': 32,
                'eval_every_n_steps': 100,
                'min_topic_size': 50
            },
            Scale.SCIENTIFIC: {
                'num_samples': 50000,
                'num_epochs': 15,
                'batch_size': 32,
                'eval_every_n_steps': 250,
                'min_topic_size': 100
            },
            Scale.LARGE: {
                'num_samples': 100000,
                'num_epochs': 25,
                'batch_size': 64,
                'eval_every_n_steps': 500,
                'min_topic_size': 200
            },
            Scale.MASSIVE: {
                'num_samples': 500000,
                'num_epochs': 50,
                'batch_size': 128,
                'eval_every_n_steps': 1000,
                'min_topic_size': 500
            },
            Scale.EXTREME: {
                'num_samples': 1000000,
                'num_epochs': 100,
                'batch_size': 256,
                'eval_every_n_steps': 2000,
                'min_topic_size': 1000
            }
        }
        
        # Apply scale defaults if not explicitly set
        if self.scale in scale_configs:
            defaults = scale_configs[self.scale]
            for key, value in defaults.items():
                if getattr(self, key) is None:
                    setattr(self, key, value)
        
        # Auto-configure model architecture
        model_configs = {
            'bert-mini': {
                'hidden_size': 256,
                'num_hidden_layers': 4,
                'num_attention_heads': 4,
                'intermediate_size': 1024
            },
            'bert-small': {
                'hidden_size': 512,
                'num_hidden_layers': 8,
                'num_attention_heads': 8,
                'intermediate_size': 2048
            },
            'bert-base': {
                'hidden_size': 768,
                'num_hidden_layers': 12,
                'num_attention_heads': 12,
                'intermediate_size': 3072
            }
        }
        
        if self.model_size in model_configs:
            model_config = model_configs[self.model_size]
            for key, value in model_config.items():
                if getattr(self, key) is None:
                    setattr(self, key, value)
        
        # Define all valid strategies
        self.VALID_STRATEGIES = {
            # Baseline
            "random",
            # Reading-level strategies
            "reading_level_easy_to_hard",
            "reading_level_hard_to_easy", 
            "reading_level_staged",
            # Topic-based strategies
            "topic_sequential",
            "topic_interleaved",
            "topic_largest_first",
            # Hybrid strategies
            "hybrid_reading_topic",
            "hybrid_topic_reading",
            # Epoch-interleaving strategies
            "reading_topic_by_epoch",
            "reading_levels_by_epoch",
            "all_strategies_by_epoch"
        }
        
        # Set default strategies if not specified
        if self.strategies is None:
            self.strategies = list(self.VALID_STRATEGIES)
        
        # Set experiment name if not specified
        if self.experiment_name is None:
            self.experiment_name = f"curriculum_{self.scale.value}_{self.model_size}"
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration consistency"""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, "Train/val/test ratios must sum to 1.0"
        assert self.num_samples > 100, "Need at least 100 samples"
        assert self.num_epochs >= 1, "Need at least 1 epoch"
        assert self.hidden_size % self.num_attention_heads == 0, "Hidden size must be divisible by num_attention_heads"
        assert self.intermediate_size >= self.hidden_size, "Intermediate size should be >= hidden size"
        
        # Validate strategies
        invalid_strategies = set(self.strategies) - self.VALID_STRATEGIES
        if invalid_strategies:
            raise ValueError(f"Invalid strategies: {invalid_strategies}. Valid strategies are: {sorted(self.VALID_STRATEGIES)}")
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        path = Path(path)
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON file"""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Scale):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("CURRICULUM LEARNING CONFIGURATION")
        print("=" * 60)
        print(f"Scale: {self.scale.value}")
        print(f"Model: {self.model_size}")
        print(f"Samples: {self.num_samples:,}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Runs per strategy: {self.num_runs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Strategies: {len(self.strategies)}")
        print()
        
        # Model details
        total_params = self._estimate_parameters()
        print(f"Model Parameters: {total_params/1e6:.1f}M")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Layers: {self.num_hidden_layers}")
        print(f"Attention heads: {self.num_attention_heads}")
        print()
        
        # Resource estimates
        steps_per_epoch = self.num_samples // self.batch_size
        total_steps = steps_per_epoch * self.num_epochs * len(self.strategies) * self.num_runs
        estimated_hours = total_steps * 0.1 / 3600  # Rough estimate
        
        print(f"Total training steps: {total_steps:,}")
        print(f"Estimated time: {estimated_hours:.1f} hours")
        print("=" * 60)
    
    def _estimate_parameters(self) -> int:
        """Estimate total model parameters more accurately"""
        # Embeddings
        token_embeddings = self.vocab_size * self.hidden_size
        position_embeddings = self.max_seq_length * self.hidden_size
        token_type_embeddings = 2 * self.hidden_size  # 2 token types
        embedding_layer_norm = 2 * self.hidden_size  # weight + bias
        embedding_params = token_embeddings + position_embeddings + token_type_embeddings + embedding_layer_norm
        
        # Transformer layers (per layer)
        # Self-attention
        qkv_params = 3 * self.hidden_size * self.hidden_size + 3 * self.hidden_size  # weights + biases
        attention_output = self.hidden_size * self.hidden_size + self.hidden_size
        attention_layer_norm = 2 * self.hidden_size
        
        # Feed-forward
        ff_up = self.hidden_size * self.intermediate_size + self.intermediate_size
        ff_down = self.intermediate_size * self.hidden_size + self.hidden_size
        ff_layer_norm = 2 * self.hidden_size
        
        layer_params = qkv_params + attention_output + attention_layer_norm + ff_up + ff_down + ff_layer_norm
        transformer_params = self.num_hidden_layers * layer_params
        
        # MLM head
        mlm_dense = self.hidden_size * self.hidden_size + self.hidden_size
        mlm_layer_norm = 2 * self.hidden_size
        mlm_decoder = self.hidden_size * self.vocab_size + self.vocab_size  # decoder weights + bias
        mlm_params = mlm_dense + mlm_layer_norm + mlm_decoder
        
        return embedding_params + transformer_params + mlm_params


# Convenience functions
def debug_config(**kwargs) -> Config:
    """Quick debug configuration"""
    return Config(scale="debug", **kwargs)


def pilot_config(**kwargs) -> Config:
    """Pilot study configuration"""
    return Config(scale="pilot", **kwargs)


def scientific_config(**kwargs) -> Config:
    """Scientific publication configuration"""
    return Config(scale="scientific", **kwargs)


def large_config(**kwargs) -> Config:
    """Large-scale study configuration"""
    return Config(scale="large", **kwargs)


def massive_config(**kwargs) -> Config:
    """Massive scale configuration (500K samples)"""
    return Config(scale="massive", **kwargs)


def extreme_config(**kwargs) -> Config:
    """Extreme scale configuration (1M+ samples)"""
    return Config(scale="extreme", **kwargs)


def fair_comparison_config(scale: str = "pilot", **kwargs) -> Config:
    """
    Configuration for fair curriculum comparison.
    
    Ensures theoretical validity by:
    - Disabling early stopping (all strategies train equal steps)
    - Fixed seeds for reproducibility
    - Comprehensive logging
    - Multiple runs for statistical validity
    """
    
    # Fair comparison defaults
    fair_defaults = {
        'use_early_stopping': False,  # Critical for fair comparison
        'early_stopping_patience': 999999,  # Effectively disabled
        'eval_every_n_steps': 100,  # Frequent evaluation for better curves
        'random_seed': 42,  # Fixed seed
        'num_runs': 5,  # Multiple runs for statistics
        'detailed_logging': True,
        'experiment_name': f'fair_curriculum_{scale}'
    }
    
    # Merge with provided kwargs
    fair_defaults.update(kwargs)
    
    # Create config with fair comparison settings
    config = Config(scale=scale, **fair_defaults)
    
    # Validate fairness
    _validate_fair_comparison(config)
    
    return config


def _validate_fair_comparison(config: Config):
    """Validate configuration for fair comparison"""
    issues = []
    
    if getattr(config, 'use_early_stopping', True):
        issues.append("Early stopping enabled - strategies may train for different durations")
    
    if config.num_runs < 3:
        issues.append(f"Only {config.num_runs} runs - recommend ≥3 for statistical validity")
    
    if config.eval_every_n_steps > 1000:
        issues.append(f"Evaluation every {config.eval_every_n_steps} steps may miss convergence dynamics")
    
    if "random" not in config.strategies:
        issues.append("Random baseline not included - needed for comparison")
    
    if issues:
        print("⚠️  Fair Comparison Warnings:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nConsider adjusting configuration for more valid comparisons.")
    else:
        print("✅ Configuration validated for fair comparison")


if __name__ == "__main__":
    # Test different configurations
    print("Testing configuration system...\n")
    
    # Debug config
    config = debug_config()
    config.print_summary()
    
    # Custom config
    print("\nCustom configuration:")
    custom = Config(
        scale="pilot",
        model_size="bert-base",
        strategies=["random", "reading_level_easy_to_hard"],
        num_epochs=10
    )
    print(f"Samples: {custom.num_samples}")
    print(f"Epochs: {custom.num_epochs}")
    print(f"Model: {custom.model_size}")
    print(f"Strategies: {custom.strategies}")