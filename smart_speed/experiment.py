"""
Experiment runner for Entropy-Based Speculative Decoding with ablations and hyperparameter sweeps
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import itertools
from dataclasses import dataclass
import json
import os
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from speculative_decoding import (
    EntropyConfig, EntropyBasedSpeculativeDecoder,
    StoppingStrategy, EntropyType
)
from evaluation import (
    SpeculativeDecodingEvaluator, ComparativeEvaluator
)


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Model paths
    drafter_model_path: str
    verifier_model_path: str
    
    # Data configuration
    dataset_name: str
    dataset_split: str
    num_samples: int
    max_sequence_length: int
    
    # Experiment settings
    experiment_name: str
    output_dir: str
    seed: int = 42
    device: str = "cuda"
    
    # Baseline configurations
    baseline_k_values: List[int] = None
    
    def __post_init__(self):
        if self.baseline_k_values is None:
            self.baseline_k_values = [4, 6, 8, 10]


class HyperparameterGrid:
    """Define hyperparameter search space"""
    
    def __init__(self):
        self.param_space = {
            'theta_abs': [0.5, 1.0, 1.5, 2.0, 2.5],
            'theta_rel': [1.2, 1.5, 2.0, 2.5],
            'window_size': [1, 3, 5],
            'entropy_type': ['shannon', 'renyi', 'topk'],
            'strategy': ['absolute', 'relative', 'window', 'combined', 'adaptive'],
            'min_draft_len': [1, 2],
            'max_draft_len': [8, 12, 16, 20],
            'temperature': [0.8, 1.0, 1.2],
            'adaptive_alpha': [0.05, 0.1, 0.2]
        }
        
    def get_grid_configs(self, base_config: EntropyConfig) -> List[EntropyConfig]:
        """Get all configurations for grid search"""
        configs = []
        
        # Get all combinations
        keys = list(self.param_space.keys())
        values = [self.param_space[k] for k in keys]
        
        for combination in itertools.product(*values):
            config_dict = dict(zip(keys, combination))
            
            # Create config object
            config = EntropyConfig(**config_dict)
            configs.append(config)
            
        return configs
    
    def get_ablation_configs(self, base_config: EntropyConfig) -> Dict[str, List[EntropyConfig]]:
        """Get configurations for ablation studies"""
        ablations = {}
        
        # 1. Entropy type ablation
        ablations['entropy_type'] = []
        for entropy_type in ['shannon', 'renyi', 'topk']:
            config = EntropyConfig(
                strategy=base_config.strategy,
                theta_abs=base_config.theta_abs,
                entropy_type=entropy_type
            )
            ablations['entropy_type'].append(config)
        
        # 2. Stopping strategy ablation
        ablations['stopping_strategy'] = []
        for strategy in ['absolute', 'relative', 'window', 'combined', 'adaptive']:
            config = EntropyConfig(
                strategy=strategy,
                theta_abs=base_config.theta_abs,
                theta_rel=base_config.theta_rel,
                window_size=base_config.window_size,
                entropy_type=base_config.entropy_type
            )
            ablations['stopping_strategy'].append(config)
        
        # 3. Threshold sensitivity
        ablations['threshold_sensitivity'] = []
        base_theta = base_config.theta_abs
        for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            config = EntropyConfig(
                strategy=base_config.strategy,
                theta_abs=base_theta * factor,
                entropy_type=base_config.entropy_type
            )
            ablations['threshold_sensitivity'].append(config)
        
        # 4. Max draft length ablation
        ablations['max_draft_length'] = []
        for max_len in [4, 8, 12, 16, 20, 24]:
            config = EntropyConfig(
                strategy=base_config.strategy,
                theta_abs=base_config.theta_abs,
                max_draft_len=max_len,
                entropy_type=base_config.entropy_type
            )
            ablations['max_draft_length'].append(config)
        
        # 5. Temperature effect
        ablations['temperature'] = []
        for temp in [0.6, 0.8, 1.0, 1.2, 1.4]:
            config = EntropyConfig(
                strategy=base_config.strategy,
                theta_abs=base_config.theta_abs,
                temperature=temp,
                entropy_type=base_config.entropy_type
            )
            ablations['temperature'].append(config)
        
        return ablations


class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.setup_logging()
        self.setup_output_dir()
        self.set_seeds()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_output_dir(self):
        """Create output directory structure"""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'configs').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        
    def set_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def load_models(self) -> Tuple[Any, Any, Any]:
        """Load drafter and verifier models"""
        # This is a placeholder - implement actual model loading
        self.logger.info(f"Loading drafter model from {self.config.drafter_model_path}")
        self.logger.info(f"Loading verifier model from {self.config.verifier_model_path}")
        
        # Example loading code (replace with actual implementation)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.drafter_model_path)
        drafter_model = AutoModelForCausalLM.from_pretrained(
            self.config.drafter_model_path,
            torch_dtype=torch.float16,
            device_map=self.config.device
        )
        verifier_model = AutoModelForCausalLM.from_pretrained(
            self.config.verifier_model_path,
            torch_dtype=torch.float16,
            device_map=self.config.device
        )
        
        return drafter_model, verifier_model, tokenizer
    
    def load_dataset(self) -> List[str]:
        """Load evaluation dataset"""
        self.logger.info(f"Loading dataset {self.config.dataset_name}")
        
        # Placeholder - implement actual dataset loading
        # Example: load from HuggingFace datasets
        from datasets import load_dataset
        
        dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        texts = dataset['text'][:self.config.num_samples]
        
        return texts
    
    def run_baseline_experiments(self, 
                               drafter_model,
                               verifier_model,
                               tokenizer,
                               test_data: List[str]) -> Dict[str, SpeculativeDecodingEvaluator]:
        """Run baseline experiments with fixed k values"""
        self.logger.info("Running baseline experiments")
        baseline_results = {}
        
        for k in self.config.baseline_k_values:
            self.logger.info(f"Running baseline with k={k}")
            evaluator = SpeculativeDecodingEvaluator()
            
            # Run evaluation with fixed k
            # This is a simplified version - implement actual baseline
            for text in tqdm(test_data, desc=f"Baseline k={k}"):
                # Simulate baseline speculative decoding
                # In practice, implement actual fixed-k speculative decoding
                pass
            
            baseline_results[f'fixed_k_{k}'] = evaluator
            
        return baseline_results
    
    def run_entropy_experiment(self,
                             config: EntropyConfig,
                             drafter_model,
                             verifier_model,
                             tokenizer,
                             test_data: List[str]) -> SpeculativeDecodingEvaluator:
        """Run experiment with specific entropy configuration"""
        self.logger.info(f"Running experiment with config: {config}")
        
        # Create decoder
        decoder = EntropyBasedSpeculativeDecoder(
            drafter_model=drafter_model,
            verifier_model=verifier_model,
            tokenizer=tokenizer,
            entropy_config=config
        )
        
        # Create evaluator
        evaluator = SpeculativeDecodingEvaluator()
        
        # Run generation on test data
        for text in tqdm(test_data, desc="Generating"):
            input_ids = tokenizer.encode(text, return_tensors='pt').to(self.config.device)
            
            # Generate sequence
            start_time = time.time()
            generated_ids = self._generate_sequence(decoder, input_ids)
            end_time = time.time()
            
            # Update metrics
            sequence_time = end_time - start_time
            num_tokens = generated_ids.shape[1] - input_ids.shape[1]
            evaluator.update_sequence(sequence_time, num_tokens)
        
        # Get final statistics from decoder
        decoder_stats = decoder.get_statistics()
        
        return evaluator
    
    def _generate_sequence(self, decoder, input_ids, max_length=None):
        """Generate a sequence using the decoder"""
        if max_length is None:
            max_length = self.config.max_sequence_length
            
        current_ids = input_ids
        
        while current_ids.shape[1] < max_length:
            result = decoder.generate_token(current_ids)
            current_ids = result['output_ids']
            
            # Check for EOS token
            if self._has_eos_token(current_ids):
                break
                
        return current_ids
    
    def _has_eos_token(self, token_ids):
        """Check if sequence contains EOS token"""
        # Implement based on your tokenizer
        return False
    
    def run_ablation_studies(self,
                           drafter_model,
                           verifier_model,
                           tokenizer,
                           test_data: List[str]) -> Dict[str, Dict[str, SpeculativeDecodingEvaluator]]:
        """Run all ablation studies"""
        self.logger.info("Running ablation studies")
        
        # Define base configuration
        base_config = EntropyConfig(
            strategy='absolute',
            theta_abs=1.5,
            entropy_type='shannon',
            max_draft_len=12
        )
        
        # Get ablation configurations
        grid = HyperparameterGrid()
        ablation_configs = grid.get_ablation_configs(base_config)
        
        ablation_results = {}
        
        for ablation_name, configs in ablation_configs.items():
            self.logger.info(f"Running ablation: {ablation_name}")
            ablation_results[ablation_name] = {}
            
            for i, config in enumerate(configs):
                config_name = f"{ablation_name}_{i}"
                evaluator = self.run_entropy_experiment(
                    config, drafter_model, verifier_model, tokenizer, test_data
                )
                ablation_results[ablation_name][config_name] = evaluator
                
                # Save intermediate results
                self.save_intermediate_results(config_name, config, evaluator)
        
        return ablation_results
    
    def run_hyperparameter_search(self,
                                drafter_model,
                                verifier_model,
                                tokenizer,
                                test_data: List[str],
                                search_type: str = 'grid') -> Dict[str, SpeculativeDecodingEvaluator]:
        """Run hyperparameter search"""
        self.logger.info(f"Running {search_type} search")
        
        base_config = EntropyConfig()
        grid = HyperparameterGrid()
        
        if search_type == 'grid':
            configs = grid.get_grid_configs(base_config)
        else:
            # Implement other search strategies (random, bayesian, etc.)
            configs = grid.get_grid_configs(base_config)[:50]  # Sample subset
        
        search_results = {}
        best_score = -float('inf')
        best_config = None
        
        for i, config in enumerate(configs):
            config_name = f"search_{i}"
            self.logger.info(f"Testing configuration {i+1}/{len(configs)}")
            
            evaluator = self.run_entropy_experiment(
                config, drafter_model, verifier_model, tokenizer, test_data
            )
            
            # Compute score (customize based on your objectives)
            metrics = evaluator.compute_summary_metrics()
            score = metrics['tokens_per_second'] * metrics['overall_acceptance_rate']
            
            if score > best_score:
                best_score = score
                best_config = config
                
            search_results[config_name] = evaluator
            
            # Save checkpoint
            self.save_search_checkpoint(i, config, evaluator, score)
        
        self.logger.info(f"Best configuration: {best_config}")
        self.logger.info(f"Best score: {best_score}")
        
        return search_results, best_config
    
    def save_intermediate_results(self, name: str, config: EntropyConfig, evaluator: SpeculativeDecodingEvaluator):
        """Save intermediate results"""
        # Save config
        config_path = self.output_dir / 'configs' / f'{name}_config.json'
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        # Save metrics
        metrics = evaluator.compute_summary_metrics()
        metrics_path = self.output_dir / 'results' / f'{name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_search_checkpoint(self, iteration: int, config: EntropyConfig, 
                             evaluator: SpeculativeDecodingEvaluator, score: float):
        """Save search checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'config': config.__dict__,
            'metrics': evaluator.compute_summary_metrics(),
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.output_dir / 'checkpoints' / f'checkpoint_{iteration}.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def run_full_experiment(self):
        """Run complete experiment pipeline"""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        # Load models and data
        drafter_model, verifier_model, tokenizer = self.load_models()
        test_data = self.load_dataset()
        
        # 1. Run baseline experiments
        baseline_results = self.run_baseline_experiments(
            drafter_model, verifier_model, tokenizer, test_data
        )
        
        # 2. Run ablation studies
        ablation_results = self.run_ablation_studies(
            drafter_model, verifier_model, tokenizer, test_data
        )
        
        # 3. Run hyperparameter search
        search_results, best_config = self.run_hyperparameter_search(
            drafter_model, verifier_model, tokenizer, test_data
        )
        
        # 4. Run final evaluation with best config
        self.logger.info("Running final evaluation with best configuration")
        final_evaluator = self.run_entropy_experiment(
            best_config, drafter_model, verifier_model, tokenizer, test_data
        )
        
        # 5. Generate comprehensive report
        self.generate_final_report(
            baseline_results, ablation_results, search_results, 
            best_config, final_evaluator
        )
        
        self.logger.info("Experiment completed successfully")
    
    def generate_final_report(self, 
                            baseline_results: Dict,
                            ablation_results: Dict,
                            search_results: Dict,
                            best_config: EntropyConfig,
                            final_evaluator: SpeculativeDecodingEvaluator):
        """Generate final comprehensive report"""
        # Create comparative evaluator
        comparative = ComparativeEvaluator()
        
        # Add all methods
        for name, evaluator in baseline_results.items():
            comparative.add_method_results(name, evaluator)
            
        comparative.add_method_results("entropy_best", final_evaluator)
        
        # Generate report
        report_path = self.output_dir / f"{self.config.experiment_name}_report.md"
        comparative.generate_report(str(report_path))
        
        # Save best configuration
        best_config_path = self.output_dir / "best_config.json"
        with open(best_config_path, 'w') as f:
            json.dump(best_config.__dict__, f, indent=2)
        
        # Generate summary plots
        self.generate_summary_plots(ablation_results)
    
    def generate_summary_plots(self, ablation_results: Dict):
        """Generate summary plots for ablation studies"""
        import matplotlib.pyplot as plt
        
        for ablation_name, results in ablation_results.items():
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Extract metrics
            configs = list(results.keys())
            tps_values = []
            acc_rates = []
            
            for config_name, evaluator in results.items():
                metrics = evaluator.compute_summary_metrics()
                tps_values.append(metrics['tokens_per_second'])
                acc_rates.append(metrics['overall_acceptance_rate'])
            
            # Plot TPS
            axes[0].bar(range(len(configs)), tps_values)
            axes[0].set_xlabel('Configuration')
            axes[0].set_ylabel('Tokens per Second')
            axes[0].set_title(f'{ablation_name} - Generation Speed')
            
            # Plot acceptance rate
            axes[1].bar(range(len(configs)), acc_rates)
            axes[1].set_xlabel('Configuration')
            axes[1].set_ylabel('Acceptance Rate')
            axes[1].set_title(f'{ablation_name} - Acceptance Rate')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / f'{ablation_name}_summary.png')
            plt.close()


if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig(
        drafter_model_path="path/to/drafter",
        verifier_model_path="path/to/verifier",
        dataset_name="wikitext",
        dataset_split="test",
        num_samples=100,
        max_sequence_length=512,
        experiment_name="entropy_speculative_v1",
        output_dir="./experiments/entropy_speculative_v1"
    )
    
    runner = ExperimentRunner(config)
    runner.run_full_experiment()