"""
Comprehensive evaluation metrics for Entropy-Based Speculative Decoding
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score


@dataclass
class GenerationMetrics:
    """Container for generation metrics"""
    # Timing metrics
    total_time: float = 0.0
    drafter_time: float = 0.0
    verifier_time: float = 0.0
    
    # Token metrics
    total_tokens_generated: int = 0
    total_tokens_drafted: int = 0
    total_tokens_accepted: int = 0
    
    # Sequence metrics
    num_sequences: int = 0
    draft_lengths: List[int] = field(default_factory=list)
    acceptance_lengths: List[int] = field(default_factory=list)
    
    # Entropy metrics
    entropy_trajectories: List[np.ndarray] = field(default_factory=list)
    acceptance_by_entropy: Dict[float, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {'accepted': 0, 'total': 0}))
    
    # Position-wise metrics
    position_acceptance: Dict[int, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {'accepted': 0, 'total': 0}))
    
    # Quality metrics
    perplexities: List[float] = field(default_factory=list)
    
    # Latency distribution
    token_latencies: List[float] = field(default_factory=list)
    sequence_latencies: List[float] = field(default_factory=list)


class SpeculativeDecodingEvaluator:
    """Evaluator for speculative decoding methods"""
    
    def __init__(self, entropy_bins: int = 20):
        self.entropy_bins = entropy_bins
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = GenerationMetrics()
        self.baseline_metrics = GenerationMetrics()
        
    def update_generation_step(self,
                             draft_length: int,
                             accepted_length: int,
                             draft_time: float,
                             verify_time: float,
                             entropies: Optional[torch.Tensor] = None,
                             draft_tokens: Optional[torch.Tensor] = None,
                             accepted_mask: Optional[torch.Tensor] = None):
        """Update metrics for a single generation step"""
        self.metrics.total_tokens_drafted += draft_length
        self.metrics.total_tokens_accepted += accepted_length
        self.metrics.total_tokens_generated += max(1, accepted_length)
        
        self.metrics.drafter_time += draft_time
        self.metrics.verifier_time += verify_time
        self.metrics.total_time += draft_time + verify_time
        
        self.metrics.draft_lengths.append(draft_length)
        self.metrics.acceptance_lengths.append(accepted_length)
        
        # Update position-wise acceptance
        for i in range(draft_length):
            self.metrics.position_acceptance[i]['total'] += 1
            if i < accepted_length:
                self.metrics.position_acceptance[i]['accepted'] += 1
        
        # Update entropy-based acceptance
        if entropies is not None and accepted_mask is not None:
            self._update_entropy_acceptance(entropies, accepted_mask)
            self.metrics.entropy_trajectories.append(entropies.cpu().numpy())
    
    def _update_entropy_acceptance(self, entropies: torch.Tensor, accepted_mask: torch.Tensor):
        """Update acceptance statistics binned by entropy"""
        entropy_values = entropies.cpu().numpy()
        accepted_values = accepted_mask.cpu().numpy()
        
        # Bin entropies
        entropy_range = (0, 5)  # Typical entropy range
        bins = np.linspace(entropy_range[0], entropy_range[1], self.entropy_bins + 1)
        
        for i, (entropy, accepted) in enumerate(zip(entropy_values, accepted_values)):
            bin_idx = np.digitize(entropy, bins) - 1
            bin_idx = np.clip(bin_idx, 0, self.entropy_bins - 1)
            bin_center = (bins[bin_idx] + bins[bin_idx + 1]) / 2
            
            self.metrics.acceptance_by_entropy[bin_center]['total'] += 1
            if accepted:
                self.metrics.acceptance_by_entropy[bin_center]['accepted'] += 1
    
    def update_sequence(self, 
                       sequence_time: float,
                       num_tokens: int,
                       perplexity: Optional[float] = None):
        """Update metrics for a complete sequence"""
        self.metrics.num_sequences += 1
        self.metrics.sequence_latencies.append(sequence_time)
        
        if num_tokens > 0:
            avg_token_latency = sequence_time / num_tokens
            self.metrics.token_latencies.extend([avg_token_latency] * num_tokens)
        
        if perplexity is not None:
            self.metrics.perplexities.append(perplexity)
    
    def compute_summary_metrics(self) -> Dict[str, float]:
        """Compute summary metrics"""
        metrics = {}
        
        # Efficiency metrics
        if self.metrics.total_time > 0:
            metrics['tokens_per_second'] = self.metrics.total_tokens_generated / self.metrics.total_time
        else:
            metrics['tokens_per_second'] = 0.0
        
        if self.metrics.total_tokens_drafted > 0:
            metrics['overall_acceptance_rate'] = self.metrics.total_tokens_accepted / self.metrics.total_tokens_drafted
        else:
            metrics['overall_acceptance_rate'] = 0.0
        
        # Draft length statistics
        if self.metrics.draft_lengths:
            metrics['avg_draft_length'] = np.mean(self.metrics.draft_lengths)
            metrics['std_draft_length'] = np.std(self.metrics.draft_lengths)
            metrics['max_draft_length'] = np.max(self.metrics.draft_lengths)
            metrics['min_draft_length'] = np.min(self.metrics.draft_lengths)
        
        # Acceptance length statistics
        if self.metrics.acceptance_lengths:
            metrics['avg_acceptance_length'] = np.mean(self.metrics.acceptance_lengths)
            metrics['std_acceptance_length'] = np.std(self.metrics.acceptance_lengths)
        
        # Time breakdown
        if self.metrics.total_time > 0:
            metrics['drafter_time_ratio'] = self.metrics.drafter_time / self.metrics.total_time
            metrics['verifier_time_ratio'] = self.metrics.verifier_time / self.metrics.total_time
        
        # Latency percentiles
        if self.metrics.token_latencies:
            metrics['token_latency_p50'] = np.percentile(self.metrics.token_latencies, 50)
            metrics['token_latency_p90'] = np.percentile(self.metrics.token_latencies, 90)
            metrics['token_latency_p99'] = np.percentile(self.metrics.token_latencies, 99)
        
        # Quality metrics
        if self.metrics.perplexities:
            metrics['avg_perplexity'] = np.mean(self.metrics.perplexities)
        
        return metrics
    
    def compute_position_acceptance_curve(self) -> Tuple[List[int], List[float]]:
        """Compute acceptance rate by position"""
        positions = sorted(self.metrics.position_acceptance.keys())
        acceptance_rates = []
        
        for pos in positions:
            stats = self.metrics.position_acceptance[pos]
            if stats['total'] > 0:
                rate = stats['accepted'] / stats['total']
            else:
                rate = 0.0
            acceptance_rates.append(rate)
        
        return positions, acceptance_rates
    
    def compute_entropy_acceptance_curve(self) -> Tuple[List[float], List[float]]:
        """Compute acceptance rate by entropy"""
        entropy_bins = sorted(self.metrics.acceptance_by_entropy.keys())
        acceptance_rates = []
        
        for entropy in entropy_bins:
            stats = self.metrics.acceptance_by_entropy[entropy]
            if stats['total'] > 0:
                rate = stats['accepted'] / stats['total']
            else:
                rate = 0.0
            acceptance_rates.append(rate)
        
        return entropy_bins, acceptance_rates
    
    def compute_entropy_correlation(self) -> Dict[str, float]:
        """Compute correlation between entropy and acceptance"""
        correlations = {}
        
        # Flatten entropy trajectories and acceptance data
        all_entropies = []
        all_accepted = []
        
        for trajectory, accept_len in zip(self.metrics.entropy_trajectories, 
                                        self.metrics.acceptance_lengths):
            for i, entropy in enumerate(trajectory):
                all_entropies.append(entropy)
                all_accepted.append(1 if i < accept_len else 0)
        
        if len(all_entropies) > 1:
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(all_entropies, all_accepted)
            correlations['pearson_r'] = pearson_r
            correlations['pearson_p'] = pearson_p
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(all_entropies, all_accepted)
            correlations['spearman_r'] = spearman_r
            correlations['spearman_p'] = spearman_p
            
            # Mutual information
            # Discretize for MI calculation
            entropy_bins = np.histogram_bin_edges(all_entropies, bins=20)
            entropy_discrete = np.digitize(all_entropies, entropy_bins)
            mi = mutual_info_score(entropy_discrete, all_accepted)
            correlations['mutual_information'] = mi
        
        return correlations
    
    def plot_analysis(self, save_path: Optional[str] = None):
        """Generate analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Draft length distribution
        ax = axes[0, 0]
        ax.hist(self.metrics.draft_lengths, bins=20, alpha=0.7)
        ax.set_xlabel('Draft Length')
        ax.set_ylabel('Frequency')
        ax.set_title('Draft Length Distribution')
        
        # 2. Position-wise acceptance rate
        ax = axes[0, 1]
        positions, acceptance_rates = self.compute_position_acceptance_curve()
        ax.plot(positions, acceptance_rates, 'b-', linewidth=2)
        ax.set_xlabel('Position')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title('Acceptance Rate by Position')
        ax.grid(True, alpha=0.3)
        
        # 3. Entropy vs acceptance rate
        ax = axes[0, 2]
        entropy_bins, entropy_acceptance = self.compute_entropy_acceptance_curve()
        ax.plot(entropy_bins, entropy_acceptance, 'r-', linewidth=2)
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title('Acceptance Rate by Entropy')
        ax.grid(True, alpha=0.3)
        
        # 4. Entropy trajectories
        ax = axes[1, 0]
        for i, trajectory in enumerate(self.metrics.entropy_trajectories[:20]):
            ax.plot(trajectory, alpha=0.3)
        ax.set_xlabel('Position')
        ax.set_ylabel('Entropy')
        ax.set_title('Sample Entropy Trajectories')
        ax.grid(True, alpha=0.3)
        
        # 5. Token latency distribution
        ax = axes[1, 1]
        if self.metrics.token_latencies:
            ax.hist(self.metrics.token_latencies, bins=30, alpha=0.7)
            ax.axvline(np.median(self.metrics.token_latencies), color='r', 
                      linestyle='--', label=f'Median: {np.median(self.metrics.token_latencies):.3f}')
            ax.set_xlabel('Token Latency (s)')
            ax.set_ylabel('Frequency')
            ax.set_title('Token Latency Distribution')
            ax.legend()
        
        # 6. Acceptance length vs draft length scatter
        ax = axes[1, 2]
        ax.scatter(self.metrics.draft_lengths, self.metrics.acceptance_lengths, alpha=0.5)
        ax.plot([0, max(self.metrics.draft_lengths)], 
               [0, max(self.metrics.draft_lengths)], 
               'r--', label='y=x')
        ax.set_xlabel('Draft Length')
        ax.set_ylabel('Accepted Length')
        ax.set_title('Draft vs Accepted Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class ComparativeEvaluator:
    """Compare different speculative decoding methods"""
    
    def __init__(self):
        self.results = {}
        
    def add_method_results(self, method_name: str, evaluator: SpeculativeDecodingEvaluator):
        """Add results from a method"""
        self.results[method_name] = {
            'evaluator': evaluator,
            'summary': evaluator.compute_summary_metrics(),
            'position_curve': evaluator.compute_position_acceptance_curve(),
            'entropy_curve': evaluator.compute_entropy_acceptance_curve(),
            'correlations': evaluator.compute_entropy_correlation()
        }
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table"""
        data = []
        
        for method, results in self.results.items():
            row = {'Method': method}
            row.update(results['summary'])
            row.update({f'entropy_{k}': v for k, v in results['correlations'].items()})
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Tokens per second comparison
        ax = axes[0, 0]
        methods = list(self.results.keys())
        tps_values = [self.results[m]['summary']['tokens_per_second'] for m in methods]
        ax.bar(methods, tps_values)
        ax.set_ylabel('Tokens per Second')
        ax.set_title('Generation Speed Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Acceptance rate comparison
        ax = axes[0, 1]
        acc_rates = [self.results[m]['summary']['overall_acceptance_rate'] for m in methods]
        ax.bar(methods, acc_rates)
        ax.set_ylabel('Acceptance Rate')
        ax.set_title('Overall Acceptance Rate')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Position-wise acceptance curves
        ax = axes[1, 0]
        for method, results in self.results.items():
            positions, rates = results['position_curve']
            ax.plot(positions, rates, label=method, linewidth=2)
        ax.set_xlabel('Position')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title('Position-wise Acceptance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Average draft length comparison
        ax = axes[1, 1]
        avg_lengths = [self.results[m]['summary']['avg_draft_length'] for m in methods]
        ax.bar(methods, avg_lengths)
        ax.set_ylabel('Average Draft Length')
        ax.set_title('Draft Length Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def generate_report(self, output_path: str):
        """Generate comprehensive comparison report"""
        report = []
        report.append("# Speculative Decoding Methods Comparison Report\n")
        
        # Summary table
        report.append("## Summary Metrics\n")
        df = self.generate_comparison_table()
        report.append(df.to_markdown(index=False))
        report.append("\n")
        
        # Best performer analysis
        report.append("## Best Performers\n")
        
        # Find best for each metric
        metrics_to_maximize = ['tokens_per_second', 'overall_acceptance_rate', 'avg_draft_length']
        metrics_to_minimize = ['token_latency_p50', 'token_latency_p90']
        
        for metric in metrics_to_maximize:
            if metric in df.columns:
                best_method = df.loc[df[metric].idxmax(), 'Method']
                best_value = df[metric].max()
                report.append(f"- **{metric}**: {best_method} ({best_value:.3f})\n")
        
        for metric in metrics_to_minimize:
            if metric in df.columns:
                best_method = df.loc[df[metric].idxmin(), 'Method']
                best_value = df[metric].min()
                report.append(f"- **{metric}**: {best_method} ({best_value:.3f})\n")
        
        # Entropy correlation analysis
        report.append("\n## Entropy Correlation Analysis\n")
        for method, results in self.results.items():
            corr = results['correlations']
            if corr:
                report.append(f"\n### {method}\n")
                report.append(f"- Pearson correlation: {corr.get('pearson_r', 'N/A'):.3f} (p={corr.get('pearson_p', 'N/A'):.3f})\n")
                report.append(f"- Spearman correlation: {corr.get('spearman_r', 'N/A'):.3f} (p={corr.get('spearman_p', 'N/A'):.3f})\n")
                report.append(f"- Mutual Information: {corr.get('mutual_information', 'N/A'):.3f}\n")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(''.join(report))
        
        # Save comparison plots
        self.plot_comparison(output_path.replace('.md', '_comparison.png'))
        
        # Save individual method plots
        for method, results in self.results.items():
            evaluator = results['evaluator']
            plot_path = output_path.replace('.md', f'_{method}_analysis.png')
            evaluator.plot_analysis(plot_path)