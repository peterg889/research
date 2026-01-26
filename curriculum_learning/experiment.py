"""
Unified Experiment Runner for Curriculum Learning

Single entry point for all curriculum learning experiments.
Simple, scalable, and easy to use from debug to large-scale research.
"""

import os
# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import time
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm

from config import Config, debug_config, pilot_config, scientific_config, large_config
from data_pipeline import DataPipeline
from model import create_model
from evaluation import Evaluator
from tracking import ExperimentTracker


class Experiment:
    """
    Main experiment runner for curriculum learning research.
    
    Usage:
        # Quick test
        experiment = Experiment(debug_config())
        results = experiment.run()
        
        # Full research
        experiment = Experiment(scientific_config(use_wandb=True))
        results = experiment.run()
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.data_pipeline = DataPipeline(config)
        self.evaluator = Evaluator(config)
        self.tracker = ExperimentTracker(config, config.use_wandb)
        
        # Set device
        self.device = self._setup_device()
        
        # Experiment state
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
        print(f"Experiment initialized: {config.experiment_name}")
        print(f"Scale: {config.scale.value}")
        print(f"Device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup computing device with reproducibility"""
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        
        # Set seeds for reproducibility using config seed
        seed = getattr(self.config, 'random_seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False  # For exact reproducibility
        
        return device
    
    def prepare_data(self):
        """Load and split data"""
        print("\nPreparing data...")
        
        # Load full dataset
        self.dataset = self.data_pipeline.load()
        
        # Split into train/val/test
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
    
    def train_strategy(self, strategy: str, run_id: int = 0) -> Dict[str, List[float]]:
        """Train model with single curriculum strategy"""
        
        print(f"\nTraining {strategy} (run {run_id + 1}/{self.config.num_runs})")
        
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
        optimizer.zero_grad()  # Initialize gradients
        
        # Results storage
        steps = []
        losses = []
        accuracies = []
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"📚 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # For epoch-interleaving strategies, create new dataloader each epoch
            if is_epoch_interleaving:
                train_dataloader = self.train_dataset.create_dataloader(strategy, self.config.batch_size, epoch)
                print(f"Using epoch-specific curriculum for epoch {epoch + 1}")
            
            progress_bar = tqdm(train_dataloader, desc=f"Training {strategy} (Epoch {epoch + 1})")
            
            for batch in progress_bar:
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
                
                
                # Compute accuracy (on masked tokens only)
                predictions = torch.argmax(outputs.logits, dim=-1)
                mask = (labels != -100)
                if mask.sum() > 0:
                    accuracy = ((predictions == labels) & mask).float().sum() / mask.sum()
                else:
                    accuracy = torch.tensor(0.0)
                
                # Backward pass (accumulate gradients)
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
                steps.append(global_step)
                losses.append(loss.item())
                accuracies.append(accuracy.item())
                
                # Log to W&B and tracker
                metrics = {
                    'loss': loss.item(),
                    'accuracy': accuracy.item(),
                    'gradient_norm': gradient_norm,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                self.tracker.log_training_step(global_step, strategy, metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy.item():.4f}"
                })
                
                global_step += 1
                
                # Validation
                if global_step % self.config.eval_every_n_steps == 0:
                    val_metrics = self._validate_model(model, val_dataloader)
                    self.tracker.log_validation_step(global_step, strategy, val_metrics)
                
                # Check for training instability
                if gradient_norm > 10.0:  # Simple gradient norm check
                    print(f"WARNING: Large gradient norm detected: {gradient_norm:.2f}")
        
        # Final validation
        final_val_metrics = self._validate_model(model, val_dataloader)
        self.tracker.log_strategy_completion(strategy, final_val_metrics, time.time())
        
        print(f"{strategy} completed")
        print(f"   Final loss: {losses[-1]:.4f}")
        print(f"   Final accuracy: {accuracies[-1]:.4f}")
        
        return {
            'steps': steps,
            'losses': losses,
            'accuracies': accuracies
        }
    
    def _validate_model(self, model, val_dataloader) -> Dict[str, float]:
        """Run validation"""
        model.eval()
        
        val_losses = []
        val_accuracies = []
        
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
                
                # Ensure loss is scalar (DataParallel can return tensor with batch dim)
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                
                # Compute metrics
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
    
    def run(self) -> Dict[str, any]:
        """Run complete experiment"""
        
        print(f"\nStarting Curriculum Learning Experiment")
        self.config.print_summary()
        
        # Prepare data
        self.prepare_data()
        
        # Run all strategies
        all_results = {}
        
        for strategy in self.config.strategies:
            print(f"\nRunning strategy: {strategy}")
            
            # Multiple runs for statistical reliability
            strategy_results = []
            
            for run_id in range(self.config.num_runs):
                try:
                    run_result = self.train_strategy(strategy, run_id)
                    strategy_results.append(run_result)
                    
                except Exception as e:
                    print(f"ERROR in {strategy} run {run_id}: {e}")
                    continue
            
            if strategy_results:
                # Aggregate results across runs (use the best run for evaluation)
                best_run = min(strategy_results, key=lambda r: r['losses'][-1])
                all_results[strategy] = best_run
                
                # Record in evaluator
                self.evaluator.record_strategy_results(
                    strategy,
                    best_run['steps'],
                    best_run['losses'], 
                    best_run['accuracies']
                )
            else:
                print(f"WARNING: No successful runs for {strategy}")
        
        # Generate evaluation report
        print(f"\nAnalyzing results...")
        report = self.evaluator.generate_report()
        
        # Log final results
        self.tracker.log_experiment_results(report)
        
        # Print summary
        self._print_summary(report)
        
        # Save report
        report_file = f"{self.config.results_dir}/report_{self.tracker.experiment_id}.json"
        self.evaluator.save_report(report_file)
        
        # Finalize tracking
        self.tracker.finalize()
        
        return {
            'results': all_results,
            'report': report,
            'experiment_id': self.tracker.experiment_id
        }
    
    def _print_summary(self, report: Dict[str, any]):
        """Print experiment summary"""
        print("\n" + "="*60)
        print("CURRICULUM LEARNING RESULTS")
        print("="*60)
        
        summary = report['experiment_summary']
        print(f"Strategies tested: {summary['total_strategies']}")
        print(f"Effective strategies: {summary['effective_strategies']}")
        print(f"Success rate: {summary['effectiveness_rate']*100:.1f}%")
        
        best_strategies = report['best_strategies']
        if best_strategies['by_final_performance']:
            print(f"\nBest strategy: {best_strategies['by_final_performance']}")
        
        if best_strategies['effective_list']:
            print(f"Effective strategies: {', '.join(best_strategies['effective_list'])}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("="*60)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Curriculum Learning Experiment Runner")
    
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
    
    # Run experiment
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"\n🎉 Experiment completed!")
    print(f"Results saved in: {config.results_dir}")


if __name__ == "__main__":
    main()