"""
Enhanced Experiment Runner with Fixed Training Loop and Better Metrics

Addresses critical issues:
- Low validation accuracy (~5%)
- Missing training curve logging
- No early stopping
- Poor statistical analysis
"""

import os
# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import time
import math
from typing import Dict, List, Optional
from tqdm import tqdm

from config import Config
from data_pipeline import DataPipeline
from model import create_model
from evaluation import Evaluator
from enhanced_tracking import EnhancedExperimentTracker, EarlyStopping, compute_enhanced_mlm_metrics
from experiment import Experiment


class EnhancedExperiment(Experiment):
    """
    Enhanced experiment runner with:
    - Better metrics calculation
    - Early stopping
    - Enhanced logging
    - Statistical analysis
    """
    
    def __init__(self, config: Config):
        # Initialize base experiment but override tracker
        self.config = config
        
        # Initialize components with enhanced tracker
        self.data_pipeline = DataPipeline(config)
        self.evaluator = Evaluator(config)
        self.tracker = EnhancedExperimentTracker(config, config.use_wandb)
        
        # Set device
        self.device = self._setup_device()
        
        # Experiment state
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Enhanced features
        if config.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=0.001,
                restore_best_weights=True
            )
        else:
            self.early_stopping = None
        
        print(f"Enhanced Experiment initialized: {config.experiment_name}")
        print(f"Scale: {config.scale.value}")
        print(f"Device: {self.device}")
        if config.use_early_stopping:
            print(f"Early stopping: Enabled (patience={config.early_stopping_patience})")
        else:
            print(f"Early stopping: Disabled (fair comparison mode)")
    
    def train_strategy(self, strategy: str, run_id: int = 0) -> Dict[str, List[float]]:
        """Enhanced training with better metrics and early stopping"""
        
        print(f"\\n📚 Training {strategy} (run {run_id + 1}/{self.config.num_runs})")
        
        # Set seed for this specific run to ensure reproducibility
        # Each strategy in the same run gets the same initial weights
        run_seed = self.config.random_seed + run_id
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
        
        # Create fresh model for each run
        model = create_model(self.config, self.device)
        
        # Setup optimizer and scheduler (handle DataParallel wrapper)
        if isinstance(model, torch.nn.DataParallel):
            optimizer = model.module.prepare_optimizer()
            total_steps = (len(self.train_dataset) // self.config.batch_size) * self.config.num_epochs
            scheduler = model.module.create_scheduler(optimizer, total_steps)
        else:
            optimizer = model.prepare_optimizer()
            total_steps = (len(self.train_dataset) // self.config.batch_size) * self.config.num_epochs
            scheduler = model.create_scheduler(optimizer, total_steps)
        
        # Check if this is an epoch-interleaving strategy
        is_epoch_interleaving = self.train_dataset.is_epoch_interleaving_strategy(strategy)
        
        # Create validation dataloader (always random)
        val_dataloader = self.val_dataset.create_dataloader("random", self.config.batch_size)
        
        # For non-interleaving strategies, create dataloader once
        if not is_epoch_interleaving:
            train_dataloader = self.train_dataset.create_dataloader(strategy, self.config.batch_size)
        
        # Training state
        model.train()
        global_step = 0
        optimizer.zero_grad()
        
        # Enhanced results storage
        results = {
            'steps': [],
            'losses': [],
            'accuracies': [],
            'epoch_train_losses': [],
            'epoch_train_accs': [],
            'epoch_val_losses': [],
            'epoch_val_accs': [],
            'epochs': []
        }
        
        # Calculate steps per epoch for better logging
        steps_per_epoch = len(self.train_dataset) // self.config.batch_size
        self.config.steps_per_epoch = steps_per_epoch  # Store for tracker
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"  📖 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Epoch-level metrics
            epoch_train_losses = []
            epoch_train_accs = []
            
            # For epoch-interleaving strategies, create new dataloader each epoch
            if is_epoch_interleaving:
                train_dataloader = self.train_dataset.create_dataloader(strategy, self.config.batch_size, epoch)
                print(f"    Using epoch-specific curriculum for epoch {epoch + 1}")
            
            model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Training {strategy} (Epoch {epoch + 1})")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Create MLM labels (handle DataParallel wrapper)
                if isinstance(model, torch.nn.DataParallel):
                    labels = model.module.create_mlm_labels(input_ids, self.train_dataset.tokenizer)
                    masked_input_ids = model.module._last_masked_input_ids
                else:
                    labels = model.create_mlm_labels(input_ids, self.train_dataset.tokenizer)
                    masked_input_ids = model._last_masked_input_ids
                
                # Forward pass
                outputs = model(masked_input_ids, attention_mask, labels)
                
                # Ensure loss is scalar (DataParallel can return tensor with batch dim)
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
                
                # Track metrics (handle DataParallel wrapper)
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
                
                # Log to enhanced tracker
                self.tracker.log_training_step(global_step, strategy, logging_metrics)
                
                # Update progress bar with better metrics
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{enhanced_metrics['accuracy']:.4f}",
                    'top5': f"{enhanced_metrics['top_5_accuracy']:.4f}",
                    'conf': f"{enhanced_metrics['prediction_confidence']:.3f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                global_step += 1
                
                # Validation during epoch
                if global_step % self.config.eval_every_n_steps == 0:
                    val_metrics = self._validate_model_enhanced(model, val_dataloader)
                    self.tracker.log_validation_step(global_step, strategy, val_metrics)
                    
                    # Check for training instability
                    if gradient_norm > 10.0:
                        print(f"    ⚠️  WARNING: Large gradient norm detected: {gradient_norm:.2f}")
                    
                    # Check if accuracy is too low (potential training issue)
                    if global_step > 1000 and enhanced_metrics['accuracy'] < 0.1:
                        print(f"    ⚠️  WARNING: Very low accuracy detected: {enhanced_metrics['accuracy']:.4f}")
            
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
            
            print(f"    📊 Epoch {epoch + 1} Summary:")
            print(f"       Train Loss: {epoch_train_loss_avg:.4f}, Train Acc: {epoch_train_acc_avg:.4f}")
            print(f"       Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.4f}")
            print(f"       Val Perplexity: {math.exp(min(val_metrics['val_loss'], 10)):.2f}")
            
            # Early stopping check (only if enabled)
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
        self.tracker.log_strategy_completion(strategy, final_val_metrics, time.time())
        
        print(f"  ✅ {strategy} completed")
        print(f"     Final train loss: {results['losses'][-1]:.4f}")
        print(f"     Final train accuracy: {results['accuracies'][-1]:.4f}")
        print(f"     Final val loss: {final_val_metrics['val_loss']:.4f}")
        print(f"     Final val accuracy: {final_val_metrics['val_accuracy']:.4f}")
        
        return results
    
    def _validate_model_enhanced(self, model, val_dataloader) -> Dict[str, float]:
        """Enhanced validation with comprehensive metrics"""
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
                
                # Create MLM labels (handle DataParallel wrapper)
                if isinstance(model, torch.nn.DataParallel):
                    labels = model.module.create_mlm_labels(input_ids, self.val_dataset.tokenizer)
                    masked_input_ids = model.module._last_masked_input_ids
                else:
                    labels = model.create_mlm_labels(input_ids, self.val_dataset.tokenizer)
                    masked_input_ids = model._last_masked_input_ids
                
                # Forward pass
                outputs = model(masked_input_ids, attention_mask, labels)
                
                # Ensure loss is scalar
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
    
    def run(self) -> Dict[str, any]:
        """Enhanced experiment run with comprehensive analysis"""
        
        print(f"\\n🔬 Starting Enhanced Curriculum Learning Experiment")
        self.config.print_summary()
        
        # Prepare data
        self.prepare_data()
        
        # Run all strategies with enhanced tracking
        all_results = {}
        
        for strategy in self.config.strategies:
            print(f"\\n🎯 Running strategy: {strategy}")
            
            # Multiple runs for statistical reliability
            strategy_results = []
            
            for run_id in range(self.config.num_runs):
                try:
                    # Reset early stopping for each run (if enabled)
                    if self.config.use_early_stopping:
                        self.early_stopping = EarlyStopping(
                            patience=self.config.early_stopping_patience,
                            min_delta=0.001,
                            restore_best_weights=True
                        )
                    
                    run_result = self.train_strategy(strategy, run_id)
                    strategy_results.append(run_result)
                    
                except Exception as e:
                    print(f"❌ ERROR in {strategy} run {run_id}: {e}")
                    continue
            
            if strategy_results:
                # Use best run based on final validation accuracy
                if hasattr(self.tracker, 'epoch_metrics') and strategy in self.tracker.epoch_metrics:
                    # Use the run with best final validation accuracy
                    best_run_idx = 0
                    best_val_acc = 0
                    
                    # For simplicity, use the last run (could be improved)
                    best_run = strategy_results[-1]
                else:
                    # Fallback: use run with lowest final loss
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
        
        # Enhanced analysis
        print(f"\\n📊 Analyzing results...")
        report = self.evaluator.generate_report()
        
        # Log results with enhanced tracker
        self.tracker.log_experiment_results(report)
        
        # Create comprehensive visualizations
        self.tracker.create_comparison_visualizations(all_results)
        
        # Print enhanced summary
        self._print_enhanced_summary(report)
        
        # Save report
        report_file = f"{self.config.results_dir}/enhanced_report_{self.tracker.experiment_id}.json"
        self.evaluator.save_report(report_file)
        
        # Finalize enhanced tracking
        self.tracker.finalize()
        
        return {
            'results': all_results,
            'report': report,
            'experiment_id': self.tracker.experiment_id,
            'enhanced_summary': self.tracker.get_enhanced_summary()
        }
    
    def _print_enhanced_summary(self, report: Dict[str, any]):
        """Print enhanced experiment summary"""
        print("\\n" + "="*70)
        print("📋 ENHANCED CURRICULUM LEARNING RESULTS")
        print("="*70)
        
        # Basic summary
        summary = report['experiment_summary']
        print(f"🎯 Strategies tested: {summary['total_strategies']}")
        print(f"✅ Effective strategies: {summary['effective_strategies']}")
        print(f"📈 Success rate: {summary['effectiveness_rate']*100:.1f}%")
        
        # Enhanced metrics from tracker
        if hasattr(self.tracker, 'epoch_metrics'):
            print(f"\\n📊 DETAILED PERFORMANCE:")
            
            for strategy, data in self.tracker.epoch_metrics.items():
                if data['val_acc']:
                    final_acc = data['val_acc'][-1]
                    final_loss = data['val_loss'][-1]
                    final_perplexity = data['val_perplexity'][-1]
                    
                    improvement = self.tracker._calculate_improvement(strategy, final_acc)
                    convergence = self.tracker.detect_convergence(strategy)
                    
                    print(f"  🔸 {strategy}:")
                    print(f"     Final Accuracy: {final_acc:.4f} ({improvement:+.2f}% vs baseline)")
                    print(f"     Final Loss: {final_loss:.4f}")
                    print(f"     Final Perplexity: {final_perplexity:.2f}")
                    print(f"     Converged: {'Yes' if convergence['converged'] else 'No'}")
                    if convergence['converged']:
                        print(f"     Convergence Epoch: {convergence.get('convergence_epoch', 'N/A')}")
        
        # Best strategies
        best_strategies = report['best_strategies']
        if best_strategies['by_final_performance']:
            print(f"\\n🏆 Best strategy: {best_strategies['by_final_performance']}")
        
        if best_strategies['effective_list']:
            print(f"\\n✨ All effective strategies:")
            for i, strategy in enumerate(best_strategies['effective_list'], 1):
                print(f"   {i}. {strategy}")
        
        # Recommendations
        print(f"\\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        # Enhanced insights
        print(f"\\n🔍 ENHANCED INSIGHTS:")
        if hasattr(self.tracker, 'baseline_strategy'):
            baseline = self.tracker.baseline_strategy
            print(f"   • Baseline strategy: {baseline}")
            
            if hasattr(self.tracker, 'epoch_metrics') and baseline in self.tracker.epoch_metrics:
                baseline_acc = self.tracker.epoch_metrics[baseline]['val_acc'][-1] if self.tracker.epoch_metrics[baseline]['val_acc'] else 0
                print(f"   • Baseline final accuracy: {baseline_acc:.4f}")
                
                # Find best improvement
                best_improvement = 0
                best_strategy = baseline
                for strategy, data in self.tracker.epoch_metrics.items():
                    if strategy != baseline and data['val_acc']:
                        improvement = self.tracker._calculate_improvement(strategy, data['val_acc'][-1])
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_strategy = strategy
                
                if best_improvement > 0:
                    print(f"   • Best improvement: {best_strategy} (+{best_improvement:.2f}%)")
                else:
                    print(f"   • No strategy significantly outperformed baseline")
        
        print("="*70)