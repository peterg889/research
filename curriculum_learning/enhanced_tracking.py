"""
Enhanced Experiment Tracking with Better Metrics and Statistical Analysis

Addresses the critical issues observed in W&B:
- Low validation accuracy (~5%)
- Noisy learning curves
- Missing statistical analysis
- Poor visualization
"""

import os
import json
import time
import math
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import Config
from tracking import ExperimentTracker


class EnhancedExperimentTracker(ExperimentTracker):
    """
    Enhanced tracking with better metrics, statistical analysis, and visualization
    """
    
    def __init__(self, config: Config, use_wandb: bool = False):
        super().__init__(config, use_wandb)
        
        # Enhanced logging settings
        self.log_frequency = max(10, config.eval_every_n_steps // 10)  # Reduce noise
        self.epoch_metrics = {}  # Store epoch-level metrics
        self.strategy_histories = {}  # Store full training histories
        
        # Statistical tracking
        self.baseline_strategy = "random"
        self.convergence_patience = 5
        self.convergence_threshold = 0.001
        
        print(f"Enhanced tracking initialized:")
        print(f"  Log frequency: every {self.log_frequency} steps")
        print(f"  Baseline strategy: {self.baseline_strategy}")
    
    def log_training_step(self, step: int, strategy: str, metrics: Dict[str, float]):
        """Enhanced training step logging with better metrics"""
        timestamp = time.time()
        
        # Calculate enhanced metrics
        enhanced_metrics = self._calculate_enhanced_metrics(metrics)
        
        # Store in strategy history
        if strategy not in self.strategy_histories:
            self.strategy_histories[strategy] = {
                'steps': [],
                'losses': [],
                'accuracies': [],
                'perplexities': [],
                'learning_rates': [],
                'gradient_norms': [],
                'epochs': []
            }
        
        history = self.strategy_histories[strategy]
        history['steps'].append(step)
        history['losses'].append(enhanced_metrics['loss'])
        history['accuracies'].append(enhanced_metrics['accuracy'])
        history['perplexities'].append(enhanced_metrics['perplexity'])
        history['learning_rates'].append(enhanced_metrics['learning_rate'])
        history['gradient_norms'].append(enhanced_metrics['gradient_norm'])
        
        # Calculate current epoch
        steps_per_epoch = getattr(self.config, 'steps_per_epoch', 1000)
        current_epoch = step // steps_per_epoch
        history['epochs'].append(current_epoch)
        
        # Log to file (always)
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'epoch': current_epoch,
            'strategy': strategy,
            'phase': 'train',
            'metrics': enhanced_metrics,
            'runtime_seconds': timestamp - self.start_time
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\\n')
        
        # Log to W&B (reduced frequency)
        if self.use_wandb and (step % self.log_frequency == 0 or step < 100):
            self._log_to_wandb_enhanced(step, strategy, enhanced_metrics, current_epoch)
    
    def _calculate_enhanced_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate enhanced metrics from basic ones"""
        enhanced = metrics.copy()
        
        # Add perplexity
        enhanced['perplexity'] = math.exp(min(metrics['loss'], 10))  # Cap to prevent overflow
        
        # Add accuracy percentage
        enhanced['accuracy_pct'] = metrics['accuracy'] * 100
        
        # Add learning rate in scientific notation
        enhanced['lr_log'] = math.log10(max(metrics['learning_rate'], 1e-10))
        
        # Gradient norm stability
        enhanced['grad_norm_stable'] = 1.0 if metrics['gradient_norm'] < 10.0 else 0.0
        
        return enhanced
    
    def _log_to_wandb_enhanced(self, step: int, strategy: str, metrics: Dict[str, float], epoch: int):
        """Enhanced W&B logging with better organization"""
        if not self.use_wandb:
            return
            
        import wandb
        
        # Log with better namespacing
        wandb_metrics = {
            # Training metrics
            f"train/{strategy}/loss": metrics['loss'],
            f"train/{strategy}/accuracy": metrics['accuracy'],
            f"train/{strategy}/accuracy_pct": metrics['accuracy_pct'],
            f"train/{strategy}/perplexity": metrics['perplexity'],
            
            # Optimization metrics
            f"optimization/{strategy}/learning_rate": metrics['learning_rate'],
            f"optimization/{strategy}/lr_log": metrics['lr_log'],
            f"optimization/{strategy}/gradient_norm": metrics['gradient_norm'],
            f"optimization/{strategy}/grad_stable": metrics['grad_norm_stable'],
            
            # Meta information
            "step": step,
            "epoch": epoch,
            "strategy_active": strategy,
        }
        
        wandb.log(wandb_metrics)
    
    def log_epoch_summary(self, epoch: int, strategy: str, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float]):
        """Log epoch-level summaries for cleaner visualization"""
        
        # Store epoch metrics
        if strategy not in self.epoch_metrics:
            self.epoch_metrics[strategy] = {
                'epochs': [],
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'train_perplexity': [],
                'val_perplexity': []
            }
        
        epoch_data = self.epoch_metrics[strategy]
        epoch_data['epochs'].append(epoch)
        epoch_data['train_loss'].append(train_metrics['loss_avg'])
        epoch_data['train_acc'].append(train_metrics['acc_avg'])
        epoch_data['val_loss'].append(val_metrics['val_loss'])
        epoch_data['val_acc'].append(val_metrics['val_accuracy'])
        epoch_data['train_perplexity'].append(math.exp(min(train_metrics['loss_avg'], 10)))
        epoch_data['val_perplexity'].append(math.exp(min(val_metrics['val_loss'], 10)))
        
        # Log to W&B
        if self.use_wandb:
            import wandb
            wandb.log({
                f"epoch_summary/{strategy}/train_loss": train_metrics['loss_avg'],
                f"epoch_summary/{strategy}/train_acc": train_metrics['acc_avg'],
                f"epoch_summary/{strategy}/train_perplexity": math.exp(min(train_metrics['loss_avg'], 10)),
                f"epoch_summary/{strategy}/val_loss": val_metrics['val_loss'],
                f"epoch_summary/{strategy}/val_acc": val_metrics['val_accuracy'],
                f"epoch_summary/{strategy}/val_perplexity": math.exp(min(val_metrics['val_loss'], 10)),
                f"epoch_summary/{strategy}/improvement_over_baseline": self._calculate_improvement(strategy, val_metrics['val_accuracy']),
                "epoch": epoch
            })
    
    def _calculate_improvement(self, strategy: str, current_acc: float) -> float:
        """Calculate improvement over baseline strategy"""
        if (strategy == self.baseline_strategy or 
            self.baseline_strategy not in self.epoch_metrics or
            not self.epoch_metrics[self.baseline_strategy]['val_acc']):
            return 0.0
        
        baseline_acc = self.epoch_metrics[self.baseline_strategy]['val_acc'][-1]
        return ((current_acc - baseline_acc) / max(baseline_acc, 1e-8)) * 100
    
    def detect_convergence(self, strategy: str) -> Dict[str, Any]:
        """Detect if strategy has converged"""
        if strategy not in self.epoch_metrics:
            return {'converged': False, 'reason': 'No data'}
        
        val_losses = self.epoch_metrics[strategy]['val_loss']
        
        if len(val_losses) < self.convergence_patience + 1:
            return {'converged': False, 'reason': 'Insufficient data'}
        
        # Check if loss has stopped improving
        recent_losses = val_losses[-self.convergence_patience:]
        min_recent = min(recent_losses)
        best_overall = min(val_losses[:-self.convergence_patience]) if len(val_losses) > self.convergence_patience else float('inf')
        
        if min_recent > best_overall - self.convergence_threshold:
            return {
                'converged': True,
                'reason': 'Loss plateau',
                'convergence_epoch': len(val_losses) - self.convergence_patience,
                'best_loss': best_overall
            }
        
        return {'converged': False, 'reason': 'Still improving'}
    
    def create_comparison_visualizations(self, all_results: Dict[str, Any]):
        """Create comprehensive comparison visualizations"""
        if not self.use_wandb:
            return
        
        import wandb
        
        # 1. Strategy comparison table
        self._create_strategy_table(all_results)
        
        # 2. Learning curves comparison
        self._create_learning_curves_comparison()
        
        # 3. Statistical analysis
        self._create_statistical_analysis()
        
        # 4. Convergence analysis
        self._create_convergence_analysis()
    
    def _create_strategy_table(self, all_results: Dict[str, Any]):
        """Create detailed strategy comparison table"""
        if not self.use_wandb:
            return
            
        import wandb
        
        table_data = []
        
        for strategy, results in all_results.items():
            if strategy in self.epoch_metrics and self.epoch_metrics[strategy]['val_acc']:
                final_val_acc = self.epoch_metrics[strategy]['val_acc'][-1]
                final_val_loss = self.epoch_metrics[strategy]['val_loss'][-1]
                final_val_perplexity = self.epoch_metrics[strategy]['val_perplexity'][-1]
                
                # Calculate convergence info
                convergence_info = self.detect_convergence(strategy)
                convergence_epoch = convergence_info.get('convergence_epoch', len(self.epoch_metrics[strategy]['epochs']))
                
                # Calculate improvement over baseline
                improvement = self._calculate_improvement(strategy, final_val_acc)
                
                table_data.append([
                    strategy,
                    f"{final_val_loss:.4f}",
                    f"{final_val_acc:.4f}",
                    f"{final_val_perplexity:.2f}",
                    f"{improvement:+.2f}%",
                    convergence_epoch,
                    "Yes" if convergence_info['converged'] else "No"
                ])
        
        table = wandb.Table(
            data=table_data,
            columns=["Strategy", "Final Val Loss", "Final Val Acc", "Final Perplexity", 
                    "Improvement vs Baseline", "Convergence Epoch", "Converged"]
        )
        wandb.log({"strategy_comparison_detailed": table})
    
    def _create_learning_curves_comparison(self):
        """Create learning curves comparison plot"""
        if not self.use_wandb or not self.epoch_metrics:
            return
            
        import wandb
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        for strategy, data in self.epoch_metrics.items():
            epochs = data['epochs']
            
            # Loss curves
            ax1.plot(epochs, data['train_loss'], label=f"{strategy} (train)", linewidth=2, alpha=0.7)
            ax2.plot(epochs, data['val_loss'], label=f"{strategy} (val)", linewidth=2)
            
            # Accuracy curves
            ax3.plot(epochs, data['train_acc'], label=f"{strategy} (train)", linewidth=2, alpha=0.7)
            ax4.plot(epochs, data['val_acc'], label=f"{strategy} (val)", linewidth=2)
        
        # Styling
        for ax, title, ylabel in zip([ax1, ax2, ax3, ax4], 
                                   ["Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"],
                                   ["Loss", "Loss", "Accuracy", "Accuracy"]):
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        wandb.log({"learning_curves_detailed": wandb.Image(fig)})
        plt.close(fig)
    
    def _create_statistical_analysis(self):
        """Create statistical analysis of strategy performance"""
        if not self.use_wandb or not self.epoch_metrics:
            return
            
        import wandb
        
        # Extract final accuracies for each strategy
        strategy_scores = {}
        for strategy, data in self.epoch_metrics.items():
            if data['val_acc']:
                strategy_scores[strategy] = data['val_acc'][-1]
        
        if len(strategy_scores) < 2:
            return
        
        # Create statistical comparison
        baseline_score = strategy_scores.get(self.baseline_strategy, 0)
        
        stats_data = []
        for strategy, score in strategy_scores.items():
            if strategy == self.baseline_strategy:
                continue
                
            improvement = ((score - baseline_score) / max(baseline_score, 1e-8)) * 100
            effect_size = abs(improvement) / 10  # Rough effect size estimate
            
            # Effect size interpretation
            if effect_size < 0.2:
                effect_interpretation = "Negligible"
            elif effect_size < 0.5:
                effect_interpretation = "Small"
            elif effect_size < 0.8:
                effect_interpretation = "Medium"
            else:
                effect_interpretation = "Large"
            
            stats_data.append([
                strategy,
                f"{score:.4f}",
                f"{improvement:+.2f}%",
                f"{effect_size:.3f}",
                effect_interpretation
            ])
        
        stats_table = wandb.Table(
            data=stats_data,
            columns=["Strategy", "Final Score", "Improvement vs Baseline", "Effect Size", "Effect Interpretation"]
        )
        wandb.log({"statistical_analysis": stats_table})
    
    def _create_convergence_analysis(self):
        """Analyze convergence patterns"""
        if not self.use_wandb or not self.epoch_metrics:
            return
            
        import wandb
        
        convergence_data = []
        
        for strategy in self.epoch_metrics.keys():
            convergence_info = self.detect_convergence(strategy)
            
            convergence_data.append([
                strategy,
                "Yes" if convergence_info['converged'] else "No",
                convergence_info.get('convergence_epoch', 'N/A'),
                convergence_info['reason'],
                f"{convergence_info.get('best_loss', 0):.4f}" if 'best_loss' in convergence_info else 'N/A'
            ])
        
        convergence_table = wandb.Table(
            data=convergence_data,
            columns=["Strategy", "Converged", "Convergence Epoch", "Reason", "Best Loss"]
        )
        wandb.log({"convergence_analysis": convergence_table})
    
    def finalize(self):
        """Enhanced finalization with comprehensive analysis"""
        # Create final visualizations
        self.create_comparison_visualizations({})
        
        # Generate final report
        summary = self.get_enhanced_summary()
        
        self._log_event("experiment_finished_enhanced", summary)
        
        if self.use_wandb:
            import wandb
            wandb.finish()
        
        print(f"\\nEnhanced Experiment Completed: {self.experiment_id}")
        print(f"Runtime: {summary['total_runtime_hours']:.2f} hours")
        print(f"Strategies analyzed: {summary['strategies_analyzed']}")
        print(f"Best performing strategy: {summary['best_strategy']}")
        print(f"Peak memory: {summary['peak_memory_mb']:.1f} MB")
        if summary['peak_gpu_memory_mb'] > 0:
            print(f"Peak GPU memory: {summary['peak_gpu_memory_mb']:.1f} MB")
    
    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get enhanced experiment summary with detailed analysis"""
        base_summary = super().get_summary()
        
        # Add enhanced analysis
        enhanced_summary = base_summary.copy()
        enhanced_summary.update({
            'strategies_analyzed': len(self.epoch_metrics),
            'total_epochs_trained': sum(len(data['epochs']) for data in self.epoch_metrics.values()),
            'best_strategy': self._find_best_strategy(),
            'convergence_analysis': {strategy: self.detect_convergence(strategy) 
                                  for strategy in self.epoch_metrics.keys()},
            'final_scores': {strategy: data['val_acc'][-1] if data['val_acc'] else 0 
                           for strategy, data in self.epoch_metrics.items()}
        })
        
        return enhanced_summary
    
    def _find_best_strategy(self) -> str:
        """Find the best performing strategy"""
        best_strategy = "unknown"
        best_score = -1
        
        for strategy, data in self.epoch_metrics.items():
            if data['val_acc'] and data['val_acc'][-1] > best_score:
                best_score = data['val_acc'][-1]
                best_strategy = strategy
        
        return best_strategy


class EarlyStopping:
    """Early stopping utility to prevent overtraining"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop
        
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                print(f"Early stopping triggered. Restored best weights from {self.patience} steps ago.")
            return True
        return False


def compute_enhanced_mlm_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive MLM metrics"""
    mask = (labels != -100)
    
    if mask.sum() == 0:
        return {
            'accuracy': 0.0,
            'top_5_accuracy': 0.0,
            'prediction_confidence': 0.0,
            'prediction_entropy': 0.0
        }
    
    # Standard accuracy
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    # Top-5 accuracy
    _, top_5_preds = torch.topk(logits, 5, dim=-1)
    labels_expanded = labels.unsqueeze(-1).expand_as(top_5_preds)
    top_5_correct = (top_5_preds == labels_expanded) & mask.unsqueeze(-1)
    top_5_accuracy = top_5_correct.any(dim=-1).sum().float() / mask.sum().float()
    
    # Prediction confidence and entropy
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    
    # Average over masked tokens only
    masked_max_probs = max_probs[mask]
    masked_entropy = entropy[mask]
    
    return {
        'accuracy': accuracy.item(),
        'top_5_accuracy': top_5_accuracy.item(),
        'prediction_confidence': masked_max_probs.mean().item(),
        'prediction_entropy': masked_entropy.mean().item()
    }