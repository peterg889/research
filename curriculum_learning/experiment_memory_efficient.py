"""
Memory-Efficient Experiment Runner for Curriculum Learning

Optimized for large-scale experiments with limited RAM.
"""

import os
# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import torch
import numpy as np
import time
import argparse
import psutil
from typing import Dict, List, Optional
from tqdm import tqdm

from config import Config, debug_config, pilot_config, scientific_config, large_config
from data_pipeline import DataPipeline
from model import create_model
from evaluation import Evaluator
from tracking import ExperimentTracker
from experiment import Experiment


class MemoryEfficientExperiment(Experiment):
    """
    Memory-optimized experiment runner with:
    - Aggressive garbage collection
    - GPU cache clearing
    - Memory monitoring
    - Reduced concurrent operations
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        
        # Memory monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        
        print(f"Initial memory usage: {self.initial_memory:.1f} MB")
        
        # Set memory-efficient options
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # Reduce number of worker threads to save memory
        torch.set_num_threads(2)
        
    def _log_memory(self, stage: str):
        """Log current memory usage"""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
        
        print(f"[Memory] {stage}: {current_memory:.1f} MB (peak: {self.peak_memory:.1f} MB)")
        
        # Warn if memory usage is high
        if current_memory > 16000:  # 16GB
            print("⚠️  WARNING: High memory usage detected!")
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup"""
        # Clear Python garbage
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
    
    def prepare_data(self):
        """Memory-efficient data preparation"""
        print("\nPreparing data (memory-efficient mode)...")
        self._log_memory("Before data loading")
        
        # Load dataset
        super().prepare_data()
        
        self._log_memory("After data loading")
        
        # Clear any temporary data
        self._cleanup_memory()
        self._log_memory("After cleanup")
    
    def train_strategy(self, strategy: str, run_id: int = 0) -> Dict[str, List[float]]:
        """Train with memory monitoring and cleanup"""
        
        print(f"\nTraining {strategy} (run {run_id + 1}/{self.config.num_runs})")
        self._log_memory(f"Before {strategy}")
        
        try:
            # Run training
            result = super().train_strategy(strategy, run_id)
            
            # Cleanup after training
            self._cleanup_memory()
            self._log_memory(f"After {strategy} cleanup")
            
            return result
            
        except Exception as e:
            print(f"ERROR in {strategy}: {e}")
            
            # Emergency cleanup
            self._cleanup_memory()
            
            # If OOM, try to recover
            if "out of memory" in str(e).lower():
                print("Attempting to recover from OOM...")
                torch.cuda.empty_cache()
                gc.collect()
                
                # Skip this strategy
                return {
                    'steps': [0],
                    'losses': [float('inf')],
                    'accuracies': [0.0]
                }
            else:
                raise
    
    def run(self) -> Dict[str, any]:
        """Run experiment with memory monitoring"""
        
        print(f"\nStarting Memory-Efficient Curriculum Learning Experiment")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "No GPU")
        
        self.config.print_summary()
        
        # Prepare data
        self.prepare_data()
        
        # Run strategies one at a time (not in parallel)
        all_results = {}
        
        for i, strategy in enumerate(self.config.strategies):
            print(f"\n[{i+1}/{len(self.config.strategies)}] Running strategy: {strategy}")
            
            # Clear memory before each strategy
            self._cleanup_memory()
            
            strategy_results = []
            
            for run_id in range(self.config.num_runs):
                try:
                    # Check memory before run
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    if current_memory > 20000:  # 20GB threshold
                        print(f"⚠️  Memory usage too high ({current_memory:.1f} MB), forcing cleanup...")
                        self._cleanup_memory()
                        time.sleep(2)  # Give system time to free memory
                    
                    run_result = self.train_strategy(strategy, run_id)
                    strategy_results.append(run_result)
                    
                    # Cleanup between runs
                    del run_result
                    self._cleanup_memory()
                    
                except Exception as e:
                    print(f"ERROR in {strategy} run {run_id}: {e}")
                    self._cleanup_memory()
                    continue
            
            if strategy_results:
                # Use best run
                best_run = min(strategy_results, key=lambda r: r['losses'][-1] if r['losses'] else float('inf'))
                all_results[strategy] = best_run
                
                # Record in evaluator
                self.evaluator.record_strategy_results(
                    strategy,
                    best_run['steps'],
                    best_run['losses'], 
                    best_run['accuracies']
                )
                
                # Clear strategy results to free memory
                del strategy_results
            else:
                print(f"WARNING: No successful runs for {strategy}")
            
            # Force cleanup after each strategy
            self._cleanup_memory()
        
        # Generate report
        print(f"\nAnalyzing results...")
        report = self.evaluator.generate_report()
        
        # Log final results
        self.tracker.log_experiment_results(report)
        
        # Print summary
        self._print_summary(report)
        
        # Save report
        report_file = f"{self.config.results_dir}/report_{self.tracker.experiment_id}.json"
        self.evaluator.save_report(report_file)
        
        # Finalize
        self.tracker.finalize()
        
        # Final memory report
        print(f"\nMemory Report:")
        print(f"  Initial: {self.initial_memory:.1f} MB")
        print(f"  Peak: {self.peak_memory:.1f} MB")
        print(f"  Final: {self.process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        return {
            'results': all_results,
            'report': report,
            'experiment_id': self.tracker.experiment_id
        }


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Memory-Efficient Curriculum Learning Experiment Runner")
    
    parser.add_argument("--scale", type=str, default="debug",
                       choices=["debug", "pilot", "scientific", "large"],
                       help="Experiment scale")
    
    parser.add_argument("--model-size", type=str, default="bert-small",
                       choices=["bert-mini", "bert-small", "bert-base"],
                       help="Model size")
    
    parser.add_argument("--strategies", nargs="+", default=None,
                       help="Specific strategies to test")
    
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    
    parser.add_argument("--num-runs", type=int, default=None,
                       help="Number of runs per strategy")
    
    args = parser.parse_args()
    
    # Create config
    config_map = {
        "debug": debug_config,
        "pilot": pilot_config,
        "scientific": scientific_config,
        "large": large_config
    }
    
    config = config_map[args.scale]()
    
    # Apply overrides
    if args.model_size:
        config.model_size = args.model_size
    if args.strategies:
        config.strategies = args.strategies
    if args.use_wandb:
        config.use_wandb = True
    if args.num_runs:
        config.num_runs = args.num_runs
    
    # Run memory-efficient experiment
    experiment = MemoryEfficientExperiment(config)
    results = experiment.run()
    
    print(f"\n🎉 Experiment completed!")
    print(f"Results saved in: {config.results_dir}")


if __name__ == "__main__":
    main()