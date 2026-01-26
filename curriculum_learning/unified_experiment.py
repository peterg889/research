"""
Unified Experiment Runner for Curriculum Learning

Consolidates all experiment variants into a single, cohesive system:
- Basic, Enhanced, and Memory-Efficient modes
- Unified tracking with W&B integration
- Fair comparison and statistical analysis
- Configurable early stopping and memory management
"""

import os
# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import torch
import numpy as np
import time
import math
import psutil
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from dataclasses import dataclass
from enum import Enum

from config import Config
from data_pipeline import DataPipeline
from model import create_model
from evaluation import Evaluator


class ExperimentMode(Enum):
    """Experiment execution modes"""
    STANDARD = "standard"           # Standard mode with comprehensive tracking
    MEMORY_EFFICIENT = "memory_efficient"  # Optimized for large datasets (1M+ samples)


@dataclass
class ExperimentResults:
    """Structured experiment results"""
    strategy_results: Dict[str, Dict[str, List[float]]]
    experiment_summary: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    convergence_analysis: Dict[str, Any]
    resource_usage: Dict[str, float]


class UnifiedExperimentTracker:
    """
    Unified tracking system combining all tracking features
    """
    
    def __init__(self, config: Config, mode: ExperimentMode = ExperimentMode.STANDARD):
        self.config = config
        self.mode = mode
        self.use_wandb = config.use_wandb
        
        # Generate unique experiment ID
        from datetime import datetime
        import random
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = f"{random.randint(1000, 9999)}"
        self.experiment_id = f"{config.experiment_name}_{config.scale.value}_{config.model_size}_{timestamp}_{rand_suffix}"
        
        # Setup logging
        from pathlib import Path
        self.log_dir = Path(config.results_dir) / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        
        # Initialize tracking structures
        self.start_time = time.time()
        self.epoch_metrics = {}
        self.strategy_histories = {}
        self.baseline_strategy = "random"
        
        # Mode-specific settings
        self.log_frequency = max(10, config.eval_every_n_steps // 10)
        self.convergence_patience = 5
        self.convergence_threshold = 0.001
        
        # Initialize W&B if requested
        if self.use_wandb:
            self._initialize_wandb()
            # Initialize improved logger
            try:
                from improved_wandb_logging import ImprovedWandBLogger
                self.improved_logger = ImprovedWandBLogger(config, self.experiment_id)
            except ImportError:
                print("  Warning: ImprovedWandBLogger not available, using basic logging")
                self.improved_logger = None
        else:
            self.improved_logger = None
        
        print(f"Unified Experiment Tracker initialized")
        print(f"  Mode: {mode.value}")
        print(f"  Experiment ID: {self.experiment_id}")
        print(f"  W&B logging: {'enabled' if self.use_wandb else 'disabled'}")
    
    def _initialize_wandb(self):
        """Initialize Weights & Biases"""
        try:
            import wandb
            wandb.init(
                project="curriculum-learning",
                name=self.experiment_id,
                group=f"{self.config.scale.value}_{self.config.model_size}",
                tags=[
                    self.config.scale.value,
                    self.config.model_size,
                    self.mode.value,
                    f"strategies_{len(self.config.strategies)}",
                    f"runs_{self.config.num_runs}"
                ],
                config=self._serialize_config(),
                notes=f"Unified curriculum learning experiment ({self.mode.value})"
            )
            print(f"  W&B URL: {wandb.run.url}")
        except Exception as e:
            print(f"  W&B initialization failed: {e}")
            self.use_wandb = False
    
    def _serialize_config(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable format"""
        result = {}
        for key, value in self.config.__dict__.items():
            if hasattr(value, 'value'):  # Enum
                result[key] = value.value
            elif isinstance(value, (list, tuple)):
                result[key] = list(value)
            else:
                result[key] = value
        return result
    
    def log_training_step(self, step: int, strategy: str, metrics: Dict[str, float]):
        """Log training step with mode-appropriate detail"""
        timestamp = time.time()
        
        # Calculate enhanced metrics
        enhanced_metrics = self._calculate_enhanced_metrics(metrics)
        
        # Store in history
        if strategy not in self.strategy_histories:
            self.strategy_histories[strategy] = {
                'steps': [], 'losses': [], 'accuracies': [], 'timestamps': []
            }
        
        history = self.strategy_histories[strategy]
        history['steps'].append(step)
        history['losses'].append(enhanced_metrics['loss'])
        history['accuracies'].append(enhanced_metrics['accuracy'])
        history['timestamps'].append(timestamp)
        
        # Log to file
        import json
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'strategy': strategy,
            'phase': 'train',
            'metrics': enhanced_metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to W&B with appropriate frequency
        if self.use_wandb and (step % self.log_frequency == 0 or step < 100):
            # Use improved logger if available, otherwise fall back to basic logging
            if self.improved_logger and 'epoch' in enhanced_metrics and 'progress' in enhanced_metrics:
                self.improved_logger.log_training_step(
                    step, strategy, enhanced_metrics, 
                    enhanced_metrics.get('epoch', 0),
                    enhanced_metrics.get('progress', 0)
                )
            else:
                self._log_to_wandb(step, strategy, enhanced_metrics, 'train')
    
    def log_validation_step(self, step: int, strategy: str, metrics: Dict[str, float]):
        """Log validation step"""
        timestamp = time.time()
        
        enhanced_metrics = self._calculate_enhanced_metrics(metrics, prefix='val_')
        
        # Log to file
        import json
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'strategy': strategy,
            'phase': 'validation',
            'metrics': enhanced_metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to W&B
        if self.use_wandb:
            # Use improved logger if available, otherwise fall back to basic logging
            if self.improved_logger and 'epoch' in enhanced_metrics:
                self.improved_logger.log_validation_step(
                    step, strategy, enhanced_metrics,
                    enhanced_metrics.get('epoch', 0)
                )
            else:
                self._log_to_wandb(step, strategy, enhanced_metrics, 'val')
    
    def log_epoch_summary(self, epoch: int, strategy: str, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float]):
        """Log epoch-level summary"""
        
        # Store epoch metrics
        if strategy not in self.epoch_metrics:
            self.epoch_metrics[strategy] = {
                'epochs': [], 'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [], 'val_perplexity': []
            }
        
        epoch_data = self.epoch_metrics[strategy]
        epoch_data['epochs'].append(epoch)
        epoch_data['train_loss'].append(train_metrics['loss_avg'])
        epoch_data['train_acc'].append(train_metrics['acc_avg'])
        epoch_data['val_loss'].append(val_metrics['val_loss'])
        epoch_data['val_acc'].append(val_metrics['val_accuracy'])
        epoch_data['val_perplexity'].append(math.exp(min(val_metrics['val_loss'], 10)))
        
        # Log using improved logger if available, otherwise use basic W&B logging
        if self.use_wandb:
            if self.improved_logger:
                # Detect convergence for improved logger
                convergence_info = self.detect_convergence(strategy)
                converged = convergence_info.get('converged', False)
                
                self.improved_logger.log_epoch_summary(
                    epoch, strategy, train_metrics, val_metrics, converged
                )
            else:
                # Fallback to basic W&B logging
                import wandb
                wandb.log({
                    f"epoch_summary/{strategy}/train_loss": train_metrics['loss_avg'],
                    f"epoch_summary/{strategy}/train_acc": train_metrics['acc_avg'],
                    f"epoch_summary/{strategy}/val_loss": val_metrics['val_loss'],
                    f"epoch_summary/{strategy}/val_acc": val_metrics['val_accuracy'],
                    f"epoch_summary/{strategy}/val_perplexity": math.exp(min(val_metrics['val_loss'], 10)),
                    "epoch": epoch
                })
    
    def log_strategy_complete(self, strategy: str, metrics: Dict[str, Any]):
        """Log strategy completion"""
        if self.use_wandb and self.improved_logger:
            self.improved_logger.log_strategy_complete(strategy, metrics)
    
    def _calculate_enhanced_metrics(self, metrics: Dict[str, float], prefix: str = '') -> Dict[str, float]:
        """Calculate enhanced metrics for better analysis"""
        enhanced = metrics.copy()
        
        # Add perplexity if loss is available
        loss_key = f'{prefix}loss' if prefix else 'loss'
        if loss_key in metrics:
            enhanced[f'{prefix}perplexity'] = math.exp(min(metrics[loss_key], 10))
        
        # Add accuracy percentage
        acc_key = f'{prefix}accuracy' if prefix else 'accuracy'
        if acc_key in metrics:
            enhanced[f'{prefix}accuracy_pct'] = metrics[acc_key] * 100
        
        # Add learning rate metrics
        if 'learning_rate' in metrics:
            enhanced['lr_log'] = math.log10(max(metrics['learning_rate'], 1e-10))
        
        # Add stability indicators
        if 'gradient_norm' in metrics:
            enhanced['grad_stable'] = 1.0 if metrics['gradient_norm'] < 10.0 else 0.0
        
        return enhanced
    
    def _log_to_wandb(self, step: int, strategy: str, metrics: Dict[str, float], phase: str):
        """Log to Weights & Biases with organized structure"""
        if not self.use_wandb:
            return
        
        import wandb
        
        # Create organized metrics without the problematic strategy_active field
        wandb_metrics = {}
        for key, value in metrics.items():
            wandb_metrics[f"{phase}/{strategy}/{key}"] = value
        
        wandb_metrics['step'] = step
        # Log current strategy as a proper metric, not a string that W&B might interpret as media
        wandb_metrics['system/current_strategy_index'] = self.config.strategies.index(strategy) if strategy in self.config.strategies else -1
        wandb_metrics['system/is_training'] = 1 if phase == 'train' else 0
        
        wandb.log(wandb_metrics)
    
    def detect_convergence(self, strategy: str) -> Dict[str, Any]:
        """Detect if strategy has converged (enhanced modes only)"""
        if strategy not in self.epoch_metrics:
            return {'converged': False, 'reason': 'No convergence detection'}
        
        val_losses = self.epoch_metrics[strategy]['val_loss']
        
        if len(val_losses) < self.convergence_patience + 1:
            return {'converged': False, 'reason': 'Insufficient data'}
        
        # Check for loss plateau
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
    
    def create_visualizations(self):
        """Create comprehensive visualizations for enhanced modes"""
        if not self.use_wandb:
            return
        
        import wandb
        
        # Strategy comparison table
        if self.epoch_metrics:
            self._create_strategy_comparison_table()
        
        # Learning curves
        self._create_learning_curves()
    
    def _create_strategy_comparison_table(self):
        """Create strategy comparison table"""
        import wandb
        
        table_data = []
        for strategy, data in self.epoch_metrics.items():
            if data['val_acc']:
                final_acc = data['val_acc'][-1]
                final_loss = data['val_loss'][-1]
                final_perplexity = data['val_perplexity'][-1]
                
                convergence_info = self.detect_convergence(strategy)
                convergence_epoch = convergence_info.get('convergence_epoch', len(data['epochs']))
                
                table_data.append([
                    strategy,
                    f"{final_loss:.4f}",
                    f"{final_acc:.4f}",
                    f"{final_perplexity:.2f}",
                    convergence_epoch,
                    "Yes" if convergence_info['converged'] else "No"
                ])
        
        if table_data:
            table = wandb.Table(
                data=table_data,
                columns=["Strategy", "Final Loss", "Final Accuracy", "Final Perplexity",
                        "Convergence Epoch", "Converged"]
            )
            wandb.log({"strategy_comparison": table})
    
    def _create_learning_curves(self):
        """Create learning curves visualization"""
        try:
            import matplotlib.pyplot as plt
            import wandb
            
            if not self.epoch_metrics:
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            for strategy, data in self.epoch_metrics.items():
                epochs = data['epochs']
                
                ax1.plot(epochs, data['train_loss'], label=f"{strategy} (train)", alpha=0.7)
                ax2.plot(epochs, data['val_loss'], label=f"{strategy} (val)")
                ax3.plot(epochs, data['train_acc'], label=f"{strategy} (train)", alpha=0.7)
                ax4.plot(epochs, data['val_acc'], label=f"{strategy} (val)")
            
            for ax, title, ylabel in zip([ax1, ax2, ax3, ax4],
                                       ["Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"],
                                       ["Loss", "Loss", "Accuracy", "Accuracy"]):
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            wandb.log({"learning_curves": wandb.Image(fig)})
            plt.close(fig)
        except ImportError:
            print("Matplotlib not available, skipping learning curves")
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize tracking and return summary"""
        # Create visualizations
        self.create_visualizations()
        
        # Log experiment completion
        if self.use_wandb and self.improved_logger:
            self.improved_logger.log_experiment_complete({
                'strategy_results': self.strategy_histories,
                'final_scores': self.epoch_metrics
            })
        
        # Generate summary
        total_runtime = time.time() - self.start_time
        summary = {
            'experiment_id': self.experiment_id,
            'mode': self.mode.value,
            'total_runtime_hours': total_runtime / 3600,
            'strategies_analyzed': len(self.strategy_histories),
            'log_directory': str(self.log_dir)
        }
        
        # Add analysis
        summary.update({
            'convergence_analysis': {s: self.detect_convergence(s) for s in self.epoch_metrics.keys()},
                'final_scores': {s: data['val_acc'][-1] if data['val_acc'] else 0 
                               for s, data in self.epoch_metrics.items()}
            })
        
        if self.use_wandb:
            import wandb
            wandb.finish()
        
        print(f"Experiment completed: {self.experiment_id}")
        print(f"Runtime: {summary['total_runtime_hours']:.2f} hours")
        
        return summary


class UnifiedExperiment:
    """
    Unified experiment runner that consolidates all variants
    """
    
    def __init__(self, config: Config, mode: ExperimentMode = ExperimentMode.STANDARD):
        self.config = config
        self.mode = mode
        
        # Initialize components
        self.data_pipeline = DataPipeline(config)
        self.evaluator = Evaluator(config)
        self.tracker = UnifiedExperimentTracker(config, mode)
        
        # Setup device
        self.device = self._setup_device()
        
        # Experiment state
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Mode-specific features
        if mode == ExperimentMode.MEMORY_EFFICIENT:
            self._setup_memory_monitoring()
        
        # Setup early stopping based on config
        if config.use_early_stopping:
            self._setup_early_stopping()
        
        print(f"Unified Experiment initialized")
        print(f"  Mode: {mode.value}")
        print(f"  Scale: {config.scale.value}")
        print(f"  Device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup computing device"""
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        
        # Set seeds for reproducibility
        seed = getattr(self.config, 'random_seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        return device
    
    def _setup_memory_monitoring(self):
        """Setup memory monitoring for memory-efficient mode"""
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        
        # Optimize for memory
        torch.set_num_threads(2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"  Memory monitoring enabled. Initial: {self.initial_memory:.1f} MB")
    
    def _setup_early_stopping(self):
        """Setup early stopping for enhanced modes"""
        if self.config.use_early_stopping:
            from enhanced_tracking import EarlyStopping
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=0.001,
                restore_best_weights=True
            )
        else:
            self.early_stopping = None
    
    def _log_memory(self, stage: str):
        """Log memory usage (memory-efficient mode only)"""
        if self.mode != ExperimentMode.MEMORY_EFFICIENT:
            return
        
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
        print(f"[Memory] {stage}: {current_memory:.1f} MB (peak: {self.peak_memory:.1f} MB)")
    
    def _cleanup_memory(self):
        """Cleanup memory (memory-efficient mode only)"""
        if self.mode != ExperimentMode.MEMORY_EFFICIENT:
            return
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Multiple cleanup passes
        for _ in range(3):
            gc.collect()
    
    def prepare_data(self):
        """Load and prepare data"""
        print("\nPreparing data...")
        self._log_memory("Before data loading")
        
        # Load dataset
        self.dataset = self.data_pipeline.load()
        
        # Split data
        self.train_dataset, self.val_dataset, test_dataset = self.dataset.split(
            self.config.train_ratio, self.config.val_ratio
        )
        
        print(f"Data ready:")
        print(f"   Train: {len(self.train_dataset)} samples")
        print(f"   Validation: {len(self.val_dataset)} samples")
        print(f"   Test: {len(test_dataset)} samples")
        
        # Log dataset statistics
        stats = self.dataset.get_statistics()
        print(f"   Topics: {stats['topic_stats']['num_topics']}")
        print(f"   Reading levels: {stats['reading_level_stats']['mean']:.1f} ± {stats['reading_level_stats']['std']:.1f}")
        
        self._log_memory("After data loading")
        self._cleanup_memory()
    
    def train_strategy(self, strategy: str, run_id: int = 0) -> Dict[str, List[float]]:
        """Train model with unified approach"""
        print(f"\n📚 Training {strategy} (run {run_id + 1}/{self.config.num_runs})")
        self._log_memory(f"Before {strategy}")
        
        try:
            # Set run-specific seed for reproducibility
            run_seed = self.config.random_seed + run_id
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed(run_seed)
                torch.cuda.manual_seed_all(run_seed)
            
            # Create fresh model
            model = create_model(self.config, self.device)
            
            # Setup optimizer and scheduler
            if isinstance(model, torch.nn.DataParallel):
                optimizer = model.module.prepare_optimizer()
                total_steps = (len(self.train_dataset) // self.config.batch_size) * self.config.num_epochs
                scheduler = model.module.create_scheduler(optimizer, total_steps)
            else:
                optimizer = model.prepare_optimizer()
                total_steps = (len(self.train_dataset) // self.config.batch_size) * self.config.num_epochs
                scheduler = model.create_scheduler(optimizer, total_steps)
            
            # Train the model
            return self._train_enhanced(model, optimizer, scheduler, strategy, run_id)
            
        except Exception as e:
            print(f"❌ ERROR in {strategy}: {e}")
            self._cleanup_memory()
            
            # Return failed result
            return {
                'steps': [0],
                'losses': [float('inf')],
                'accuracies': [0.0]
            }
    
    def _train_basic(self, model, optimizer, scheduler, strategy: str, run_id: int) -> Dict[str, List[float]]:
        """Basic training loop"""
        # Check epoch interleaving
        is_epoch_interleaving = self.train_dataset.is_epoch_interleaving_strategy(strategy)
        
        # Create validation dataloader
        val_dataloader = self.val_dataset.create_dataloader("random", self.config.batch_size)
        
        # Create train dataloader (if not epoch-interleaving)
        if not is_epoch_interleaving:
            train_dataloader = self.train_dataset.create_dataloader(strategy, self.config.batch_size)
        
        # Training state
        model.train()
        global_step = 0
        
        # Results storage
        steps, losses, accuracies = [], [], []
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"  📖 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Create epoch-specific dataloader if needed
            if is_epoch_interleaving:
                train_dataloader = self.train_dataset.create_dataloader(strategy, self.config.batch_size, epoch)
            
            progress_bar = tqdm(train_dataloader, desc=f"Training {strategy}")
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Create MLM labels
                if isinstance(model, torch.nn.DataParallel):
                    labels = model.module.create_mlm_labels(input_ids, self.train_dataset.tokenizer)
                    masked_input_ids = model.module._last_masked_input_ids
                else:
                    labels = model.create_mlm_labels(input_ids, self.train_dataset.tokenizer)
                    masked_input_ids = model._last_masked_input_ids
                
                # Forward pass
                outputs = model(masked_input_ids, attention_mask, labels)
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                
                # Compute accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                mask = (labels != -100)
                if mask.sum() > 0:
                    accuracy = ((predictions == labels) & mask).float().sum() / mask.sum()
                else:
                    accuracy = torch.tensor(0.0)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Track metrics
                if isinstance(model, torch.nn.DataParallel):
                    gradient_norm = model.module.compute_gradient_norm()
                else:
                    gradient_norm = model.compute_gradient_norm()
                
                # Store results
                steps.append(global_step)
                losses.append(loss.item())
                accuracies.append(accuracy.item())
                
                # Log metrics
                metrics = {
                    'loss': loss.item(),
                    'accuracy': accuracy.item(),
                    'gradient_norm': gradient_norm,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                self.tracker.log_training_step(global_step, strategy, metrics)
                
                # Update progress
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy.item():.4f}"
                })
                
                global_step += 1
                
                # Validation
                if global_step % self.config.eval_every_n_steps == 0:
                    val_metrics = self._validate_model_basic(model, val_dataloader)
                    self.tracker.log_validation_step(global_step, strategy, val_metrics)
        
        # Final validation
        final_val_metrics = self._validate_model_basic(model, val_dataloader)
        
        print(f"  ✅ {strategy} completed")
        print(f"     Final train loss: {losses[-1]:.4f}")
        print(f"     Final train accuracy: {accuracies[-1]:.4f}")
        print(f"     Final val loss: {final_val_metrics['val_loss']:.4f}")
        print(f"     Final val accuracy: {final_val_metrics['val_accuracy']:.4f}")
        
        self._cleanup_memory()
        
        return {
            'steps': steps,
            'losses': losses,
            'accuracies': accuracies
        }
    
    def _train_enhanced(self, model, optimizer, scheduler, strategy: str, run_id: int) -> Dict[str, List[float]]:
        """Enhanced training loop with better metrics and early stopping"""
        from enhanced_tracking import compute_enhanced_mlm_metrics
        
        # Check epoch interleaving
        is_epoch_interleaving = self.train_dataset.is_epoch_interleaving_strategy(strategy)
        
        # Create validation dataloader
        val_dataloader = self.val_dataset.create_dataloader("random", self.config.batch_size)
        
        # Create train dataloader (if not epoch-interleaving)
        if not is_epoch_interleaving:
            train_dataloader = self.train_dataset.create_dataloader(strategy, self.config.batch_size)
        
        # Training state
        model.train()
        global_step = 0
        
        # Enhanced results storage
        results = {
            'steps': [], 'losses': [], 'accuracies': [],
            'epoch_train_losses': [], 'epoch_train_accs': [],
            'epoch_val_losses': [], 'epoch_val_accs': [], 'epochs': []
        }
        
        # Calculate steps per epoch
        steps_per_epoch = len(self.train_dataset) // self.config.batch_size
        self.config.steps_per_epoch = steps_per_epoch
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"  📖 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Epoch-level metrics
            epoch_train_losses = []
            epoch_train_accs = []
            
            # Create epoch-specific dataloader if needed
            if is_epoch_interleaving:
                train_dataloader = self.train_dataset.create_dataloader(strategy, self.config.batch_size, epoch)
            
            model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Training {strategy} (Epoch {epoch + 1})")
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Create MLM labels
                if isinstance(model, torch.nn.DataParallel):
                    labels = model.module.create_mlm_labels(input_ids, self.train_dataset.tokenizer)
                    masked_input_ids = model.module._last_masked_input_ids
                else:
                    labels = model.create_mlm_labels(input_ids, self.train_dataset.tokenizer)
                    masked_input_ids = model._last_masked_input_ids
                
                # Forward pass
                outputs = model(masked_input_ids, attention_mask, labels)
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                
                # Compute enhanced metrics
                enhanced_metrics = compute_enhanced_mlm_metrics(outputs.logits, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Track metrics
                if isinstance(model, torch.nn.DataParallel):
                    gradient_norm = model.module.compute_gradient_norm()
                else:
                    gradient_norm = model.compute_gradient_norm()
                
                # Store results
                results['steps'].append(global_step)
                results['losses'].append(loss.item())
                results['accuracies'].append(enhanced_metrics['accuracy'])
                
                # Store for epoch averaging
                epoch_train_losses.append(loss.item())
                epoch_train_accs.append(enhanced_metrics['accuracy'])
                
                # Enhanced logging metrics
                logging_metrics = {
                    'loss': loss.item(),
                    'accuracy': enhanced_metrics['accuracy'],
                    'top_5_accuracy': enhanced_metrics['top_5_accuracy'],
                    'prediction_confidence': enhanced_metrics['prediction_confidence'],
                    'prediction_entropy': enhanced_metrics['prediction_entropy'],
                    'gradient_norm': gradient_norm,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                
                # Log to tracker
                self.tracker.log_training_step(global_step, strategy, logging_metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{enhanced_metrics['accuracy']:.4f}",
                    'top5': f"{enhanced_metrics['top_5_accuracy']:.4f}",
                    'conf': f"{enhanced_metrics['prediction_confidence']:.3f}"
                })
                
                global_step += 1
                
                # Validation during epoch
                if global_step % self.config.eval_every_n_steps == 0:
                    val_metrics = self._validate_model_enhanced(model, val_dataloader)
                    self.tracker.log_validation_step(global_step, strategy, val_metrics)
            
            # End of epoch processing
            epoch_train_loss_avg = np.mean(epoch_train_losses)
            epoch_train_acc_avg = np.mean(epoch_train_accs)
            
            # Full validation at end of epoch
            val_metrics = self._validate_model_enhanced(model, val_dataloader)
            
            # Store epoch-level results
            results['epochs'].append(epoch)
            results['epoch_train_losses'].append(epoch_train_loss_avg)
            results['epoch_train_accs'].append(epoch_train_acc_avg)
            results['epoch_val_losses'].append(val_metrics['val_loss'])
            results['epoch_val_accs'].append(val_metrics['val_accuracy'])
            
            # Log epoch summary
            train_summary = {
                'loss_avg': epoch_train_loss_avg,
                'acc_avg': epoch_train_acc_avg
            }
            
            self.tracker.log_epoch_summary(epoch, strategy, train_summary, val_metrics)
            
            # The improved logger will handle epoch summary through the regular log_epoch_summary call
            
            print(f"    📊 Epoch {epoch + 1} Summary:")
            print(f"       Train Loss: {epoch_train_loss_avg:.4f}, Train Acc: {epoch_train_acc_avg:.4f}")
            print(f"       Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.4f}")
            
            # Early stopping check
            if self.config.use_early_stopping and self.early_stopping:
                if self.early_stopping(val_metrics['val_loss'], model):
                    print(f"    🛑 Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Check convergence
            convergence_info = self.tracker.detect_convergence(strategy)
            if convergence_info['converged']:
                print(f"    ✅ Convergence detected: {convergence_info['reason']}")
        
        # Final validation
        final_val_metrics = self._validate_model_enhanced(model, val_dataloader)
        
        print(f"  ✅ {strategy} completed")
        print(f"     Final train loss: {results['losses'][-1]:.4f}")
        print(f"     Final train accuracy: {results['accuracies'][-1]:.4f}")
        print(f"     Final val loss: {final_val_metrics['val_loss']:.4f}")
        print(f"     Final val accuracy: {final_val_metrics['val_accuracy']:.4f}")
        
        # Log strategy completion through tracker
        strategy_start_time = (self.tracker.improved_logger.strategy_start_times.get(strategy, self.tracker.start_time) 
                              if self.tracker.improved_logger else self.tracker.start_time)
        self.tracker.log_strategy_complete(strategy, {
            'final_accuracy': final_val_metrics['val_accuracy'],
            'final_loss': final_val_metrics['val_loss'],
            'epochs_trained': len(results['epochs']),
            'training_time': time.time() - strategy_start_time
        })
        
        self._cleanup_memory()
        
        return results
    
    def _validate_model_basic(self, model, val_dataloader) -> Dict[str, float]:
        """Basic validation"""
        model.eval()
        
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Create MLM labels
                if isinstance(model, torch.nn.DataParallel):
                    labels = model.module.create_mlm_labels(input_ids, self.val_dataset.tokenizer)
                    masked_input_ids = model.module._last_masked_input_ids
                else:
                    labels = model.create_mlm_labels(input_ids, self.val_dataset.tokenizer)
                    masked_input_ids = model._last_masked_input_ids
                
                # Forward pass
                outputs = model(masked_input_ids, attention_mask, labels)
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                
                # Compute accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                mask = (labels != -100)
                if mask.sum() > 0:
                    accuracy = ((predictions == labels) & mask).float().sum() / mask.sum()
                else:
                    accuracy = torch.tensor(0.0)
                
                val_losses.append(loss.item())
                val_accuracies.append(accuracy.item())
        
        model.train()
        
        return {
            'val_loss': np.mean(val_losses),
            'val_accuracy': np.mean(val_accuracies)
        }
    
    def _validate_model_enhanced(self, model, val_dataloader) -> Dict[str, float]:
        """Enhanced validation with comprehensive metrics"""
        from enhanced_tracking import compute_enhanced_mlm_metrics
        
        model.eval()
        
        val_losses = []
        val_accuracies = []
        val_top5_accuracies = []
        val_confidences = []
        val_entropies = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Create MLM labels
                if isinstance(model, torch.nn.DataParallel):
                    labels = model.module.create_mlm_labels(input_ids, self.val_dataset.tokenizer)
                    masked_input_ids = model.module._last_masked_input_ids
                else:
                    labels = model.create_mlm_labels(input_ids, self.val_dataset.tokenizer)
                    masked_input_ids = model._last_masked_input_ids
                
                # Forward pass
                outputs = model(masked_input_ids, attention_mask, labels)
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                
                # Compute enhanced metrics
                enhanced_metrics = compute_enhanced_mlm_metrics(outputs.logits, labels)
                
                val_losses.append(loss.item())
                val_accuracies.append(enhanced_metrics['accuracy'])
                val_top5_accuracies.append(enhanced_metrics['top_5_accuracy'])
                val_confidences.append(enhanced_metrics['prediction_confidence'])
                val_entropies.append(enhanced_metrics['prediction_entropy'])
        
        model.train()
        
        return {
            'val_loss': np.mean(val_losses),
            'val_accuracy': np.mean(val_accuracies),
            'val_top_5_accuracy': np.mean(val_top5_accuracies),
            'val_prediction_confidence': np.mean(val_confidences),
            'val_prediction_entropy': np.mean(val_entropies),
            'val_perplexity': math.exp(min(np.mean(val_losses), 10))
        }
    
    def run(self) -> ExperimentResults:
        """Run complete experiment"""
        print(f"\n🔬 Starting Unified Curriculum Learning Experiment")
        print(f"Mode: {self.mode.value}")
        self.config.print_summary()
        
        # Prepare data
        self.prepare_data()
        
        # Run all strategies
        all_results = {}
        
        for strategy in self.config.strategies:
            print(f"\n🎯 Running strategy: {strategy}")
            self._cleanup_memory()
            
            # Multiple runs for statistical reliability
            strategy_results = []
            
            for run_id in range(self.config.num_runs):
                try:
                    # Reset early stopping for each run (if enabled)
                    if hasattr(self, 'early_stopping') and self.early_stopping:
                        from enhanced_tracking import EarlyStopping
                        self.early_stopping = EarlyStopping(
                            patience=self.config.early_stopping_patience,
                            min_delta=0.001,
                            restore_best_weights=True
                        )
                    
                    run_result = self.train_strategy(strategy, run_id)
                    strategy_results.append(run_result)
                    
                    # Cleanup between runs
                    self._cleanup_memory()
                    
                except Exception as e:
                    print(f"❌ ERROR in {strategy} run {run_id}: {e}")
                    self._cleanup_memory()
                    continue
            
            if strategy_results:
                # Use best run based on final loss
                best_run = min(strategy_results, key=lambda r: r['losses'][-1] if r['losses'] else float('inf'))
                all_results[strategy] = best_run
                
                # Record in evaluator
                self.evaluator.record_strategy_results(
                    strategy,
                    best_run['steps'],
                    best_run['losses'],
                    best_run['accuracies']
                )
            else:
                print(f"⚠️  WARNING: No successful runs for {strategy}")
        
        # Generate evaluation report
        print(f"\n📊 Analyzing results...")
        report = self.evaluator.generate_report()
        
        # Finalize tracking
        tracker_summary = self.tracker.finalize()
        
        # Print summary
        self._print_summary(report, tracker_summary)
        
        # Save report
        report_file = f"{self.config.results_dir}/unified_report_{self.tracker.experiment_id}.json"
        self.evaluator.save_report(report_file)
        
        # Prepare final results
        results = ExperimentResults(
            strategy_results=all_results,
            experiment_summary=report,
            statistical_analysis=tracker_summary.get('final_scores', {}),
            convergence_analysis=tracker_summary.get('convergence_analysis', {}),
            resource_usage=self._get_resource_summary()
        )
        
        return results
    
    def _get_resource_summary(self) -> Dict[str, float]:
        """Get resource usage summary"""
        summary = {
            'total_runtime_hours': (time.time() - self.tracker.start_time) / 3600
        }
        
        if self.mode == ExperimentMode.MEMORY_EFFICIENT:
            summary.update({
                'initial_memory_mb': self.initial_memory,
                'peak_memory_mb': self.peak_memory,
                'final_memory_mb': self.process.memory_info().rss / 1024 / 1024
            })
        
        return summary
    
    def _print_summary(self, report: Dict[str, Any], tracker_summary: Dict[str, Any]):
        """Print experiment summary"""
        print("\n" + "="*70)
        print("🎯 UNIFIED CURRICULUM LEARNING RESULTS")
        print("="*70)
        print(f"Mode: {self.mode.value}")
        
        # Basic summary
        summary = report['experiment_summary']
        print(f"Strategies tested: {summary['total_strategies']}")
        print(f"Effective strategies: {summary['effective_strategies']}")
        print(f"Success rate: {summary['effectiveness_rate']*100:.1f}%")
        
        # Summary analysis
        final_scores = tracker_summary.get('final_scores', {})
        if final_scores:
            print(f"\n📊 FINAL SCORES:")
            for strategy, score in final_scores.items():
                print(f"  {strategy}: {score:.4f}")
            
            convergence_analysis = tracker_summary.get('convergence_analysis', {})
            converged_strategies = [s for s, info in convergence_analysis.items() if info.get('converged', False)]
            if converged_strategies:
                print(f"\n🎯 CONVERGED STRATEGIES: {', '.join(converged_strategies)}")
        
        # Resource usage
        if self.mode == ExperimentMode.MEMORY_EFFICIENT:
            print(f"\n💾 MEMORY USAGE:")
            print(f"  Initial: {self.initial_memory:.1f} MB")
            print(f"  Peak: {self.peak_memory:.1f} MB")
        
        print(f"\nRuntime: {tracker_summary['total_runtime_hours']:.2f} hours")
        print("="*70)


# Convenience functions for different modes
def run_standard_experiment(config: Config) -> ExperimentResults:
    """Run standard experiment with comprehensive tracking"""
    experiment = UnifiedExperiment(config, ExperimentMode.STANDARD)
    return experiment.run()


# Legacy function - redirect to standard
def run_enhanced_experiment(config: Config) -> ExperimentResults:
    """Legacy: Run standard experiment"""
    return run_standard_experiment(config)


def run_memory_efficient_experiment(config: Config) -> ExperimentResults:
    """Run memory-efficient experiment"""
    experiment = UnifiedExperiment(config, ExperimentMode.MEMORY_EFFICIENT)
    return experiment.run()


# Legacy function - redirect to standard with early stopping disabled
def run_fair_comparison_experiment(config: Config) -> ExperimentResults:
    """Legacy: Run standard experiment without early stopping"""
    config.use_early_stopping = False
    return run_standard_experiment(config)


if __name__ == "__main__":
    # Example usage
    from config import debug_config, pilot_config
    
    print("🧪 Testing Unified Experiment System...\n")
    
    # Test basic mode
    config = debug_config(use_wandb=False)
    results = run_basic_experiment(config)
    print(f"Basic experiment completed: {len(results.strategy_results)} strategies")
    
    # Test enhanced mode
    # results = run_enhanced_experiment(config)
    # print(f"Enhanced experiment completed")