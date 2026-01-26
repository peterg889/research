"""
Fair Comparison Configuration for Curriculum Learning

Ensures theoretical validity by maintaining identical conditions
except for data ordering.
"""

from config import Config, Scale
from typing import Optional


class FairComparisonConfig(Config):
    """
    Configuration that ensures fair comparison between strategies.
    
    Key principles:
    - All strategies train for exactly the same number of steps
    - No early stopping (could favor some strategies)
    - Identical initialization seeds
    - Comprehensive logging to capture dynamics
    """
    
    def __init__(self, **kwargs):
        # Set defaults for fair comparison
        defaults = {
            'use_early_stopping': False,  # Disable to ensure equal training
            'early_stopping_patience': 999999,  # Effectively disabled
            'eval_every_n_steps': 100,  # Frequent evaluation
            'save_every_n_steps': 500,  # Regular checkpoints
            'log_every_n_steps': 10,  # Detailed logging for dynamics
            'random_seed': 42,  # Fixed seed for reproducibility
        }
        
        # Override with any provided kwargs
        defaults.update(kwargs)
        
        # Initialize parent
        super().__init__(**defaults)
        
        # Additional fair comparison settings
        self.use_early_stopping = False
        self.ensure_equal_steps = True
        self.detailed_logging = True
    
    def validate_fairness(self):
        """Validate configuration ensures fair comparison"""
        issues = []
        
        if hasattr(self, 'use_early_stopping') and self.use_early_stopping:
            issues.append("Early stopping enabled - strategies may train for different durations")
        
        if self.num_runs < 3:
            issues.append(f"Only {self.num_runs} runs - recommend ≥3 for statistical validity")
        
        if self.eval_every_n_steps > 1000:
            issues.append(f"Evaluation every {self.eval_every_n_steps} steps may miss convergence dynamics")
        
        if issues:
            print("⚠️  Fair Comparison Warnings:")
            for issue in issues:
                print(f"   - {issue}")
            print("\nConsider adjusting configuration for more valid comparisons.")
        else:
            print("✅ Configuration validated for fair comparison")
        
        return len(issues) == 0


def create_fair_pilot_config(**kwargs) -> FairComparisonConfig:
    """Create a pilot configuration ensuring fair comparison"""
    config = FairComparisonConfig(
        scale="large",  # 100K samples
        model_size="bert-small",
        num_epochs=20,  # Fixed epochs for all
        num_runs=3,
        batch_size=32,
        strategies=[
            "random",  # Always include baseline
            "reading_level_easy_to_hard",
            "reading_level_hard_to_easy",
            "topic_sequential",
            "topic_largest_first",
            "hybrid_reading_topic"
        ],
        experiment_name="fair_curriculum_pilot",
        **kwargs
    )
    
    config.validate_fairness()
    return config


def create_fair_scientific_config(**kwargs) -> FairComparisonConfig:
    """Create a scientific configuration ensuring fair comparison"""
    config = FairComparisonConfig(
        scale="extreme",  # 1M samples
        model_size="bert-small",
        num_epochs=50,  # Fixed for all
        num_runs=5,  # Multiple runs for significance
        batch_size=256,
        strategies=[
            # Core strategies
            "random",
            "reading_level_easy_to_hard",
            "reading_level_hard_to_easy",
            "reading_level_staged",
            
            # Topic strategies
            "topic_sequential",
            "topic_interleaved",
            "topic_largest_first",
            
            # Hybrid strategies  
            "hybrid_reading_topic",
            "hybrid_topic_reading",
        ],
        experiment_name="fair_curriculum_scientific",
        **kwargs
    )
    
    config.validate_fairness()
    return config


if __name__ == "__main__":
    print("Testing Fair Comparison Configurations\n")
    
    # Test pilot config
    print("=== Fair Pilot Config ===")
    pilot = create_fair_pilot_config()
    pilot.print_summary()
    
    print("\n=== Fair Scientific Config ===") 
    scientific = create_fair_scientific_config()
    
    # Test with early stopping enabled (should warn)
    print("\n=== Testing Unfair Config ===")
    unfair = FairComparisonConfig(
        scale="pilot",
        use_early_stopping=True,
        num_runs=1
    )
    unfair.validate_fairness()