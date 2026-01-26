"""
Improved W&B Logging for Curriculum Learning Experiments

Provides clear visibility into:
a) Whether the model is currently training
b) Whether it's training effectively
c) Early look into experimental outcomes
"""

import time
import wandb
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class ImprovedWandBLogger:
    """Enhanced W&B logging with better organization and real-time insights"""
    
    def __init__(self, config: Any, experiment_id: str):
        self.config = config
        self.experiment_id = experiment_id
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.current_strategy = None
        self.current_epoch = 0
        self.total_strategies = len(config.strategies)
        self.strategy_start_times = {}
        
        # Initialize tracking for experimental outcomes
        self.strategy_best_scores = {}
        self.strategy_convergence_epochs = {}
        self.baseline_score = None
        
    def log_training_step(self, step: int, strategy: str, metrics: Dict[str, float], 
                         epoch: int, progress: float):
        """Log training step with enhanced visibility"""
        
        # Update current strategy tracking
        if strategy != self.current_strategy:
            self.current_strategy = strategy
            self.strategy_start_times[strategy] = time.time()
            
        self.current_epoch = epoch
        
        # Calculate training health metrics
        current_time = time.time()
        time_since_last_log = current_time - self.last_log_time
        self.last_log_time = current_time
        
        # Prepare comprehensive logging
        log_dict = {
            # Core training metrics
            f"train/{strategy}/loss": metrics['loss'],
            f"train/{strategy}/accuracy": metrics['accuracy'],
            f"train/{strategy}/learning_rate": metrics.get('learning_rate', 0),
            
            # Training health indicators
            "training_health/loss_smoothed": metrics['loss'],  # W&B will smooth this
            "training_health/gradient_norm": metrics.get('gradient_norm', 0),
            "training_health/steps_per_second": 1.0 / max(time_since_last_log, 0.001),
            
            # Progress tracking
            "progress/current_epoch": epoch,
            "progress/epoch_progress": progress,
            "progress/strategy_index": self.config.strategies.index(strategy) + 1,
            "progress/total_strategies": self.total_strategies,
            "progress/experiment_progress": self._calculate_experiment_progress(strategy, epoch),
            
            # Enhanced metrics for quality
            "quality/top_5_accuracy": metrics.get('top_5_accuracy', 0),
            "quality/prediction_confidence": metrics.get('prediction_confidence', 0),
            "quality/prediction_entropy": metrics.get('prediction_entropy', 0),
            
            # System status
            "system/is_training": 1,  # Binary indicator
            "system/current_strategy_name": strategy,  # Text, not media
            "system/time_elapsed_hours": (current_time - self.start_time) / 3600,
            
            # Global step
            "step": step
        }
        
        # Add perplexity if loss is reasonable
        if metrics['loss'] < 10:
            log_dict["quality/perplexity"] = 2.71828 ** metrics['loss']
        
        wandb.log(log_dict)
    
    def log_validation_step(self, step: int, strategy: str, metrics: Dict[str, float], epoch: int):
        """Log validation with focus on experimental outcomes"""
        
        val_score = metrics.get('val_accuracy', 0)
        
        # Track best score per strategy
        if strategy not in self.strategy_best_scores or val_score > self.strategy_best_scores[strategy]:
            self.strategy_best_scores[strategy] = val_score
            
        # Track baseline for comparison
        if strategy == 'random' and (self.baseline_score is None or val_score > self.baseline_score):
            self.baseline_score = val_score
        
        # Calculate improvement over baseline
        improvement = 0
        if self.baseline_score and self.baseline_score > 0:
            improvement = ((val_score - self.baseline_score) / self.baseline_score) * 100
        
        log_dict = {
            # Validation metrics
            f"val/{strategy}/loss": metrics['val_loss'],
            f"val/{strategy}/accuracy": val_score,
            f"val/{strategy}/perplexity": 2.71828 ** min(metrics['val_loss'], 10),
            
            # Experimental outcomes preview
            f"outcomes/{strategy}/best_accuracy": self.strategy_best_scores[strategy],
            f"outcomes/{strategy}/improvement_over_baseline": improvement,
            "outcomes/current_best_strategy": max(self.strategy_best_scores, key=self.strategy_best_scores.get),
            "outcomes/current_best_improvement": max([
                ((score - self.baseline_score) / self.baseline_score * 100) if self.baseline_score else 0
                for score in self.strategy_best_scores.values()
            ]),
            
            # Convergence tracking
            f"convergence/{strategy}/epochs_trained": epoch,
            
            "step": step,
            "system/is_training": 1
        }
        
        wandb.log(log_dict)
    
    def log_epoch_summary(self, epoch: int, strategy: str, train_metrics: Dict[str, float],
                         val_metrics: Dict[str, float], converged: bool = False):
        """Log epoch-level summaries"""
        
        # Track convergence
        if converged and strategy not in self.strategy_convergence_epochs:
            self.strategy_convergence_epochs[strategy] = epoch
        
        # Calculate time efficiency
        strategy_time = time.time() - self.strategy_start_times.get(strategy, self.start_time)
        
        log_dict = {
            # Epoch summaries
            f"epoch_summary/{strategy}/train_loss": train_metrics['loss_avg'],
            f"epoch_summary/{strategy}/train_accuracy": train_metrics['acc_avg'],
            f"epoch_summary/{strategy}/val_loss": val_metrics['val_loss'],
            f"epoch_summary/{strategy}/val_accuracy": val_metrics['val_accuracy'],
            
            # Efficiency metrics
            f"efficiency/{strategy}/epochs_to_convergence": self.strategy_convergence_epochs.get(strategy, -1),
            f"efficiency/{strategy}/time_hours": strategy_time / 3600,
            f"efficiency/{strategy}/converged": int(converged),
            
            "epoch": epoch,
            "system/is_training": 1
        }
        
        wandb.log(log_dict)
    
    def log_strategy_complete(self, strategy: str, final_metrics: Dict[str, Any]):
        """Log when a strategy completes training"""
        
        log_dict = {
            f"final/{strategy}/accuracy": final_metrics.get('final_accuracy', 0),
            f"final/{strategy}/loss": final_metrics.get('final_loss', 0),
            f"final/{strategy}/epochs_trained": final_metrics.get('epochs_trained', 0),
            f"final/{strategy}/training_time_hours": final_metrics.get('training_time', 0) / 3600,
            
            # Update progress
            "progress/strategies_completed": len([s for s in self.strategy_best_scores]),
            "system/is_training": 1 if len(self.strategy_best_scores) < self.total_strategies else 0
        }
        
        wandb.log(log_dict)
    
    def log_experiment_complete(self, results: Dict[str, Any]):
        """Log final experimental results"""
        
        # Create summary table
        summary_data = []
        for strategy in self.config.strategies:
            if strategy in self.strategy_best_scores:
                improvement = 0
                if self.baseline_score and self.baseline_score > 0:
                    improvement = ((self.strategy_best_scores[strategy] - self.baseline_score) / 
                                 self.baseline_score * 100)
                
                summary_data.append([
                    strategy,
                    self.strategy_best_scores[strategy],
                    improvement,
                    self.strategy_convergence_epochs.get(strategy, "Did not converge"),
                    "✓" if improvement > 2 else "✗"
                ])
        
        # Log summary table
        table = wandb.Table(
            columns=["Strategy", "Best Accuracy", "Improvement %", "Convergence Epoch", "Significant"],
            data=summary_data
        )
        
        wandb.log({
            "experiment_summary": table,
            "final_baseline_accuracy": self.baseline_score,
            "best_strategy": max(self.strategy_best_scores, key=self.strategy_best_scores.get),
            "max_improvement": max([
                ((score - self.baseline_score) / self.baseline_score * 100) if self.baseline_score else 0
                for score in self.strategy_best_scores.values()
            ]),
            "system/is_training": 0,
            "system/experiment_complete": 1
        })
    
    def _calculate_experiment_progress(self, current_strategy: str, current_epoch: int) -> float:
        """Calculate overall experiment progress as percentage"""
        strategy_idx = self.config.strategies.index(current_strategy)
        strategies_complete = strategy_idx
        current_strategy_progress = current_epoch / self.config.num_epochs
        
        total_progress = (strategies_complete + current_strategy_progress) / self.total_strategies
        return total_progress * 100
    
    def create_custom_charts(self):
        """Define custom W&B charts for better visualization"""
        
        # Chart configs to add to W&B
        charts = [
            # Training health dashboard
            {
                "title": "Training Health Dashboard",
                "charts": [
                    {"y": "training_health/loss_smoothed", "title": "Loss Trend"},
                    {"y": "training_health/gradient_norm", "title": "Gradient Norm"},
                    {"y": "training_health/steps_per_second", "title": "Training Speed"}
                ]
            },
            # Experimental outcomes preview
            {
                "title": "Experimental Outcomes Preview", 
                "charts": [
                    {"y": "outcomes/current_best_improvement", "title": "Best Improvement %"},
                    {"y": ["outcomes/*/best_accuracy"], "title": "Best Accuracies by Strategy"},
                    {"y": ["outcomes/*/improvement_over_baseline"], "title": "Improvements over Baseline"}
                ]
            },
            # Progress tracking
            {
                "title": "Experiment Progress",
                "charts": [
                    {"y": "progress/experiment_progress", "title": "Overall Progress %"},
                    {"y": "progress/strategies_completed", "title": "Strategies Completed"},
                    {"y": "system/time_elapsed_hours", "title": "Time Elapsed (hours)"}
                ]
            }
        ]
        
        return charts