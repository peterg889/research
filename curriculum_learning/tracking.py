"""
Unified Experiment Tracking for Curriculum Learning

Simple, comprehensive logging system for reproducible research.
"""

import os
import json
import time
import psutil
import torch
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from enum import Enum

from config import Config


class ExperimentTracker:
    """
    Simple experiment tracking with optional W&B integration.
    
    Features:
    - Automatic experiment ID generation
    - Metrics logging with timestamps
    - Resource monitoring
    - Configuration tracking
    - Optional Weights & Biases integration
    """
    
    def __init__(self, config: Config, use_wandb: bool = False):
        self.config = config
        self.use_wandb = use_wandb
        
        # Generate unique experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import random
        rand_suffix = f"{random.randint(1000, 9999)}"
        self.experiment_id = f"{config.experiment_name}_{config.scale.value}_{config.model_size}_{timestamp}_{rand_suffix}"
        
        # Setup logging directories
        self.log_dir = Path(config.results_dir) / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.events_file = self.log_dir / "events.log"
        
        # Initialize W&B if requested
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="curriculum-learning",
                    name=self.experiment_id,
                    group=f"{config.scale.value}_{config.model_size}",
                    tags=[
                        config.scale.value,
                        config.model_size,
                        f"strategies_{len(config.strategies)}",
                        f"runs_{config.num_runs}",
                        f"samples_{config.num_samples//1000}K"
                    ],
                    config=self._serialize_config(config),
                    notes=f"Curriculum learning experiment: {len(config.strategies)} strategies, {config.num_runs} runs each"
                )
                print(f"W&B logging enabled: {wandb.run.url}")
            except ImportError:
                print("WARNING: W&B not available, continuing without it")
                self.use_wandb = False
            except Exception as e:
                print(f"WARNING: W&B initialization failed: {e}")
                self.use_wandb = False
        
        # Resource monitoring
        self.start_time = time.time()
        self.peak_memory = 0
        self.peak_gpu_memory = 0
        
        # Log experiment start
        self._log_event("experiment_started", {
            "experiment_id": self.experiment_id,
            "config": self._serialize_config(config)
        })
        
        print(f"Experiment started: {self.experiment_id}")
        print(f"Logs: {self.log_dir}")
    
    def _serialize_config(self, config: Config) -> Dict[str, Any]:
        """Convert config to JSON-serializable format"""
        result = {}
        for key, value in config.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = self._make_json_serializable(value)
        return result
    
    def log_training_step(self, step: int, strategy: str, metrics: Dict[str, float]):
        """Log training metrics for a step"""
        timestamp = time.time()
        
        # Add metadata
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'strategy': strategy,
            'phase': 'train',
            'metrics': metrics,
            'runtime_seconds': timestamp - self.start_time
        }
        
        # Write to file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to W&B
        if self.use_wandb:
            import wandb
            wandb_metrics = {f"train/{strategy}/{k}": v for k, v in metrics.items()}
            wandb_metrics['step'] = step
            wandb.log(wandb_metrics)
        
        # Update resource monitoring
        self._update_resource_usage()
    
    def log_validation_step(self, step: int, strategy: str, metrics: Dict[str, float]):
        """Log validation metrics"""
        timestamp = time.time()
        
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'strategy': strategy,
            'phase': 'validation',
            'metrics': metrics,
            'runtime_seconds': timestamp - self.start_time
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        if self.use_wandb:
            import wandb
            wandb_metrics = {f"val/{strategy}/{k}": v for k, v in metrics.items()}
            wandb_metrics['step'] = step
            wandb.log(wandb_metrics)
    
    def log_strategy_completion(self, strategy: str, final_metrics: Dict[str, float], 
                              training_time: float):
        """Log completion of a curriculum strategy"""
        self._log_event("strategy_completed", {
            'strategy': strategy,
            'final_metrics': final_metrics,
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60
        })
    
    def log_experiment_results(self, results: Dict[str, Any]):
        """Log final experiment results"""
        self._log_event("experiment_results", results)
        
        if self.use_wandb:
            import wandb
            # Log summary table
            summary_data = []
            for strategy, metrics in results.items():
                if isinstance(metrics, dict) and 'final_loss' in metrics:
                    summary_data.append([
                        strategy,
                        metrics.get('final_loss', 0),
                        metrics.get('final_accuracy', 0),
                        metrics.get('convergence_step', 0)
                    ])
            
            if summary_data:
                wandb.log({
                    "results_summary": wandb.Table(
                        data=summary_data,
                        columns=["Strategy", "Final Loss", "Final Accuracy", "Convergence Step"]
                    )
                })
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log experiment events"""
        timestamp = datetime.now().isoformat()
        
        # Make data JSON serializable
        serializable_data = self._make_json_serializable(data)
        
        with open(self.events_file, 'a') as f:
            f.write(f"[{timestamp}] {event_type}: {json.dumps(serializable_data)}\n")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'value'):  # Handle Enum objects
            return obj.value
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _update_resource_usage(self):
        """Update resource usage statistics"""
        try:
            # CPU and Memory
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.peak_memory = max(self.peak_memory, memory_mb)
            
            # GPU memory if available
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_memory_mb)
        
        except Exception:
            pass  # Don't fail experiment if monitoring fails
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        total_runtime = time.time() - self.start_time
        
        return {
            'experiment_id': self.experiment_id,
            'total_runtime_seconds': total_runtime,
            'total_runtime_hours': total_runtime / 3600,
            'peak_memory_mb': self.peak_memory,
            'peak_gpu_memory_mb': self.peak_gpu_memory,
            'log_directory': str(self.log_dir)
        }
    
    def finalize(self):
        """Finalize experiment tracking"""
        summary = self.get_summary()
        
        self._log_event("experiment_finished", summary)
        
        if self.use_wandb:
            import wandb
            wandb.finish()
        
        print(f"Experiment completed: {self.experiment_id}")
        print(f"Runtime: {summary['total_runtime_hours']:.2f} hours")
        print(f"Peak memory: {summary['peak_memory_mb']:.1f} MB")
        if summary['peak_gpu_memory_mb'] > 0:
            print(f"Peak GPU memory: {summary['peak_gpu_memory_mb']:.1f} MB")


def load_experiment_metrics(log_dir: str) -> Dict[str, Any]:
    """
    Load metrics from a completed experiment.
    
    Args:
        log_dir: Path to experiment log directory
        
    Returns:
        Dictionary with all logged metrics organized by strategy
    """
    
    metrics_file = Path(log_dir) / "metrics.jsonl"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"No metrics file found at {metrics_file}")
    
    # Load all metrics
    all_metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            all_metrics.append(json.loads(line))
    
    # Organize by strategy
    results = {}
    for entry in all_metrics:
        strategy = entry['strategy']
        phase = entry['phase']
        
        if strategy not in results:
            results[strategy] = {'train': [], 'validation': []}
        
        results[strategy][phase].append({
            'step': entry['step'],
            'timestamp': entry['timestamp'],
            'metrics': entry['metrics']
        })
    
    return results


if __name__ == "__main__":
    # Test the tracking system
    from config import debug_config
    
    print("🧪 Testing Experiment Tracking...\n")
    
    config = debug_config()
    tracker = ExperimentTracker(config, use_wandb=False)
    
    # Simulate training
    for step in range(10):
        metrics = {
            'loss': 5.0 - step * 0.3,
            'accuracy': step * 0.05,
            'learning_rate': 1e-4
        }
        tracker.log_training_step(step, "random", metrics)
        
        if step % 3 == 0:
            val_metrics = {
                'val_loss': 5.2 - step * 0.25,
                'val_accuracy': step * 0.04
            }
            tracker.log_validation_step(step, "random", val_metrics)
    
    # Complete strategy
    tracker.log_strategy_completion("random", {'final_loss': 2.0}, 120.0)
    
    # Finalize
    summary = tracker.get_summary()
    print(f"Experiment ID: {summary['experiment_id']}")
    print(f"Runtime: {summary['total_runtime_seconds']:.1f}s")
    
    tracker.finalize()
    
    # Test loading metrics
    print(f"\nTesting metrics loading...")
    loaded_metrics = load_experiment_metrics(tracker.log_dir)
    print(f"Loaded data for strategies: {list(loaded_metrics.keys())}")
    
    print(f"\nTracking system test completed!")