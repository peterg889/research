"""
Data Analysis Pipeline for Entropy-Based Speculative Decoding Experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class ExperimentDataLoader:
    """Load and organize experiment data"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.configs = {}
        self.metrics = {}
        self.raw_data = {}
        
    def load_all_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load all experiment data into structured format"""
        # Load configurations
        config_files = list((self.experiment_dir / 'configs').glob('*.json'))
        for config_file in config_files:
            exp_name = config_file.stem.replace('_config', '')
            with open(config_file, 'r') as f:
                self.configs[exp_name] = json.load(f)
        
        # Load metrics
        metrics_files = list((self.experiment_dir / 'results').glob('*_metrics.json'))
        for metrics_file in metrics_files:
            exp_name = metrics_file.stem.replace('_metrics', '')
            with open(metrics_file, 'r') as f:
                self.metrics[exp_name] = json.load(f)
        
        # Create DataFrame
        df = self._create_dataframe()
        
        return df, self.raw_data
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create unified DataFrame from loaded data"""
        rows = []
        
        for exp_name in self.configs.keys():
            if exp_name not in self.metrics:
                continue
                
            row = {
                'experiment': exp_name,
                **self.configs[exp_name],
                **self.metrics[exp_name]
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add derived metrics
        if 'tokens_per_second' in df.columns and 'overall_acceptance_rate' in df.columns:
            df['efficiency_score'] = df['tokens_per_second'] * df['overall_acceptance_rate']
        
        return df


class EntropyAnalyzer:
    """Analyze entropy patterns and their relationship to performance"""
    
    def __init__(self, df: pd.DataFrame, raw_data: Dict):
        self.df = df
        self.raw_data = raw_data
        
    def analyze_entropy_patterns(self) -> Dict[str, Any]:
        """Analyze entropy patterns across experiments"""
        analysis = {}
        
        # Group by entropy type
        if 'entropy_type' in self.df.columns:
            entropy_groups = self.df.groupby('entropy_type')
            
            analysis['by_entropy_type'] = {
                'mean_tps': entropy_groups['tokens_per_second'].mean().to_dict(),
                'mean_acceptance': entropy_groups['overall_acceptance_rate'].mean().to_dict(),
                'mean_draft_length': entropy_groups['avg_draft_length'].mean().to_dict()
            }
        
        # Analyze threshold impact
        if 'theta_abs' in self.df.columns:
            analysis['threshold_correlation'] = {
                'tps_correlation': self.df[['theta_abs', 'tokens_per_second']].corr().iloc[0, 1],
                'acceptance_correlation': self.df[['theta_abs', 'overall_acceptance_rate']].corr().iloc[0, 1]
            }
        
        # Analyze strategy effectiveness
        if 'strategy' in self.df.columns:
            strategy_groups = self.df.groupby('strategy')
            analysis['by_strategy'] = {
                'efficiency_score': strategy_groups['efficiency_score'].mean().to_dict(),
                'consistency': strategy_groups['efficiency_score'].std().to_dict()
            }
        
        return analysis
    
    def find_optimal_thresholds(self) -> Dict[str, float]:
        """Find optimal thresholds for different objectives"""
        optimal = {}
        
        # Optimize for different objectives
        objectives = {
            'max_tps': 'tokens_per_second',
            'max_acceptance': 'overall_acceptance_rate',
            'max_efficiency': 'efficiency_score',
            'min_latency': 'token_latency_p50'
        }
        
        for obj_name, metric in objectives.items():
            if metric in self.df.columns and 'theta_abs' in self.df.columns:
                if obj_name.startswith('min'):
                    best_idx = self.df[metric].idxmin()
                else:
                    best_idx = self.df[metric].idxmax()
                    
                optimal[obj_name] = {
                    'theta_abs': self.df.loc[best_idx, 'theta_abs'],
                    'value': self.df.loc[best_idx, metric],
                    'config': self.df.loc[best_idx].to_dict()
                }
        
        return optimal
    
    def plot_entropy_analysis(self, save_dir: Optional[str] = None):
        """Generate entropy analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Threshold vs Performance
        if 'theta_abs' in self.df.columns:
            ax = axes[0, 0]
            ax.scatter(self.df['theta_abs'], self.df['tokens_per_second'], alpha=0.6)
            ax.set_xlabel('Absolute Threshold')
            ax.set_ylabel('Tokens per Second')
            ax.set_title('Threshold vs Generation Speed')
            
            # Add trend line
            z = np.polyfit(self.df['theta_abs'], self.df['tokens_per_second'], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(self.df['theta_abs'].min(), self.df['theta_abs'].max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', alpha=0.8)
        
        # 2. Strategy comparison
        if 'strategy' in self.df.columns:
            ax = axes[0, 1]
            strategy_data = self.df.groupby('strategy')['efficiency_score'].mean().sort_values()
            strategy_data.plot(kind='barh', ax=ax)
            ax.set_xlabel('Efficiency Score')
            ax.set_title('Strategy Effectiveness')
        
        # 3. Entropy type comparison
        if 'entropy_type' in self.df.columns:
            ax = axes[0, 2]
            entropy_pivot = self.df.pivot_table(
                index='entropy_type',
                values=['tokens_per_second', 'overall_acceptance_rate'],
                aggfunc='mean'
            )
            entropy_pivot.plot(kind='bar', ax=ax)
            ax.set_ylabel('Value')
            ax.set_title('Entropy Type Comparison')
            ax.legend(['TPS', 'Acceptance Rate'])
        
        # 4. Draft length distribution by strategy
        if 'strategy' in self.df.columns and 'avg_draft_length' in self.df.columns:
            ax = axes[1, 0]
            for strategy in self.df['strategy'].unique():
                data = self.df[self.df['strategy'] == strategy]['avg_draft_length']
                ax.hist(data, alpha=0.5, label=strategy, bins=15)
            ax.set_xlabel('Average Draft Length')
            ax.set_ylabel('Frequency')
            ax.set_title('Draft Length by Strategy')
            ax.legend()
        
        # 5. Acceptance rate heatmap
        if 'theta_abs' in self.df.columns and 'window_size' in self.df.columns:
            ax = axes[1, 1]
            pivot = self.df.pivot_table(
                index='theta_abs',
                columns='window_size',
                values='overall_acceptance_rate',
                aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax)
            ax.set_title('Acceptance Rate Heatmap')
        
        # 6. Efficiency score distribution
        ax = axes[1, 2]
        ax.hist(self.df['efficiency_score'], bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(self.df['efficiency_score'].mean(), color='r', linestyle='--', 
                  label=f'Mean: {self.df["efficiency_score"].mean():.2f}')
        ax.set_xlabel('Efficiency Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Efficiency Score Distribution')
        ax.legend()
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(Path(save_dir) / 'entropy_analysis.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()


class PerformanceAnalyzer:
    """Analyze performance characteristics"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def analyze_performance_factors(self) -> pd.DataFrame:
        """Analyze which factors most impact performance"""
        # Select numeric columns for correlation analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        config_cols = [col for col in numeric_cols if col not in 
                      ['tokens_per_second', 'overall_acceptance_rate', 'efficiency_score']]
        metric_cols = ['tokens_per_second', 'overall_acceptance_rate', 'efficiency_score']
        
        # Compute correlations
        correlations = {}
        for metric in metric_cols:
            if metric in self.df.columns:
                corr_values = {}
                for col in config_cols:
                    if col in self.df.columns:
                        corr, p_value = stats.pearsonr(self.df[col], self.df[metric])
                        corr_values[col] = {'correlation': corr, 'p_value': p_value}
                correlations[metric] = corr_values
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(correlations).T
        
        return corr_df
    
    def identify_performance_clusters(self, n_clusters: int = 4) -> Dict[str, Any]:
        """Identify performance clusters using clustering"""
        # Select features for clustering
        feature_cols = ['tokens_per_second', 'overall_acceptance_rate', 
                       'avg_draft_length', 'efficiency_score']
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        if len(available_features) < 2:
            return {}
        
        X = self.df[available_features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=min(2, len(available_features)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = clusters
        
        for i in range(n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == i]
            cluster_analysis[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'mean_tps': cluster_data['tokens_per_second'].mean(),
                'mean_acceptance': cluster_data['overall_acceptance_rate'].mean(),
                'dominant_strategy': cluster_data['strategy'].mode().iloc[0] if 'strategy' in cluster_data.columns else None
            }
        
        return {
            'clusters': cluster_analysis,
            'pca_components': X_pca,
            'cluster_labels': clusters,
            'explained_variance': pca.explained_variance_ratio_
        }
    
    def plot_performance_analysis(self, save_dir: Optional[str] = None):
        """Generate performance analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Performance scatter plot
        ax = axes[0, 0]
        scatter = ax.scatter(self.df['overall_acceptance_rate'], 
                           self.df['tokens_per_second'],
                           c=self.df['efficiency_score'],
                           cmap='viridis',
                           alpha=0.6,
                           s=50)
        ax.set_xlabel('Acceptance Rate')
        ax.set_ylabel('Tokens per Second')
        ax.set_title('Performance Landscape')
        plt.colorbar(scatter, ax=ax, label='Efficiency Score')
        
        # 2. Factor importance
        ax = axes[0, 1]
        factor_importance = self.analyze_performance_factors()
        if not factor_importance.empty:
            # Plot top factors for efficiency score
            if 'efficiency_score' in factor_importance.index:
                factors = factor_importance.loc['efficiency_score']
                factor_df = pd.DataFrame([
                    {'factor': k, 'correlation': v['correlation']} 
                    for k, v in factors.items()
                ])
                factor_df = factor_df.sort_values('correlation', key=abs, ascending=False).head(10)
                
                factor_df.plot(x='factor', y='correlation', kind='barh', ax=ax)
                ax.set_xlabel('Correlation with Efficiency')
                ax.set_title('Factor Importance')
        
        # 3. Clustering visualization
        ax = axes[1, 0]
        cluster_results = self.identify_performance_clusters()
        if 'pca_components' in cluster_results:
            pca_data = cluster_results['pca_components']
            clusters = cluster_results['cluster_labels']
            
            scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], 
                               c=clusters, cmap='tab10', alpha=0.6)
            ax.set_xlabel(f'PC1 ({cluster_results["explained_variance"][0]:.1%} var)')
            ax.set_ylabel(f'PC2 ({cluster_results["explained_variance"][1]:.1%} var)')
            ax.set_title('Performance Clusters')
            
        # 4. Latency distribution comparison
        ax = axes[1, 1]
        if 'token_latency_p50' in self.df.columns:
            # Compare latency percentiles
            latency_cols = ['token_latency_p50', 'token_latency_p90', 'token_latency_p99']
            available_cols = [col for col in latency_cols if col in self.df.columns]
            
            if available_cols:
                latency_data = self.df[available_cols].mean()
                latency_data.plot(kind='bar', ax=ax)
                ax.set_ylabel('Latency (s)')
                ax.set_title('Latency Percentiles')
                ax.set_xticklabels(['P50', 'P90', 'P99'], rotation=0)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(Path(save_dir) / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()


class ComparisonAnalyzer:
    """Compare entropy-based approach with baselines"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def compare_with_baselines(self) -> pd.DataFrame:
        """Compare entropy methods with fixed-k baselines"""
        # Separate entropy and baseline experiments
        entropy_df = self.df[~self.df['experiment'].str.startswith('fixed_k')]
        baseline_df = self.df[self.df['experiment'].str.startswith('fixed_k')]
        
        if baseline_df.empty or entropy_df.empty:
            return pd.DataFrame()
        
        # Get best entropy configuration
        best_entropy_idx = entropy_df['efficiency_score'].idxmax()
        best_entropy = entropy_df.loc[best_entropy_idx]
        
        # Compare with each baseline
        comparisons = []
        for idx, baseline in baseline_df.iterrows():
            comparison = {
                'baseline': baseline['experiment'],
                'tps_improvement': (best_entropy['tokens_per_second'] / baseline['tokens_per_second'] - 1) * 100,
                'acceptance_diff': (best_entropy['overall_acceptance_rate'] - baseline['overall_acceptance_rate']) * 100,
                'efficiency_improvement': (best_entropy['efficiency_score'] / baseline['efficiency_score'] - 1) * 100
            }
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)
    
    def statistical_significance_test(self) -> Dict[str, Any]:
        """Test statistical significance of improvements"""
        entropy_df = self.df[~self.df['experiment'].str.startswith('fixed_k')]
        baseline_df = self.df[self.df['experiment'].str.startswith('fixed_k')]
        
        if baseline_df.empty or entropy_df.empty:
            return {}
        
        results = {}
        
        # Test each metric
        metrics = ['tokens_per_second', 'overall_acceptance_rate', 'efficiency_score']
        
        for metric in metrics:
            if metric in self.df.columns:
                entropy_values = entropy_df[metric].values
                baseline_values = baseline_df[metric].values
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(entropy_values, baseline_values)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.std(entropy_values)**2 + np.std(baseline_values)**2) / 2)
                cohens_d = (np.mean(entropy_values) - np.mean(baseline_values)) / pooled_std
                
                results[metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05,
                    'entropy_mean': np.mean(entropy_values),
                    'baseline_mean': np.mean(baseline_values)
                }
        
        return results
    
    def plot_comparison(self, save_dir: Optional[str] = None):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Method comparison
        ax = axes[0, 0]
        
        # Group methods
        entropy_df = self.df[~self.df['experiment'].str.startswith('fixed_k')]
        baseline_df = self.df[self.df['experiment'].str.startswith('fixed_k')]
        
        if not baseline_df.empty and not entropy_df.empty:
            # Box plot comparison
            data_to_plot = []
            labels = []
            
            for metric in ['tokens_per_second']:
                if metric in self.df.columns:
                    data_to_plot.extend([
                        baseline_df[metric].values,
                        entropy_df[metric].values
                    ])
                    labels.extend(['Baseline', 'Entropy-based'])
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels)
                ax.set_ylabel('Tokens per Second')
                ax.set_title('Method Comparison')
        
        # 2. Improvement distribution
        ax = axes[0, 1]
        comparison_df = self.compare_with_baselines()
        if not comparison_df.empty:
            comparison_df['efficiency_improvement'].hist(ax=ax, bins=20)
            ax.set_xlabel('Efficiency Improvement (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Improvement Distribution')
            ax.axvline(0, color='r', linestyle='--', alpha=0.5)
        
        # 3. Statistical significance
        ax = axes[1, 0]
        sig_results = self.statistical_significance_test()
        if sig_results:
            metrics = list(sig_results.keys())
            p_values = [sig_results[m]['p_value'] for m in metrics]
            effect_sizes = [sig_results[m]['cohens_d'] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, p_values, width, label='p-value')
            ax.bar(x + width/2, effect_sizes, width, label="Cohen's d")
            
            ax.set_xlabel('Metric')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45)
            ax.set_ylabel('Value')
            ax.set_title('Statistical Significance')
            ax.axhline(0.05, color='r', linestyle='--', alpha=0.5, label='p=0.05')
            ax.legend()
        
        # 4. Best configuration details
        ax = axes[1, 1]
        best_idx = self.df['efficiency_score'].idxmax()
        best_config = self.df.loc[best_idx]
        
        # Create text summary
        text = f"Best Configuration:\n"
        text += f"Strategy: {best_config.get('strategy', 'N/A')}\n"
        text += f"Entropy Type: {best_config.get('entropy_type', 'N/A')}\n"
        text += f"Threshold: {best_config.get('theta_abs', 'N/A'):.2f}\n"
        text += f"TPS: {best_config.get('tokens_per_second', 0):.2f}\n"
        text += f"Acceptance: {best_config.get('overall_acceptance_rate', 0):.3f}\n"
        text += f"Efficiency: {best_config.get('efficiency_score', 0):.2f}"
        
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=12,
               verticalalignment='center', fontfamily='monospace')
        ax.axis('off')
        ax.set_title('Best Configuration')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(Path(save_dir) / 'comparison_analysis.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()


class AnalysisPipeline:
    """Main analysis pipeline"""
    
    def __init__(self, experiment_dir: str, output_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = ExperimentDataLoader(experiment_dir)
        
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("Loading experiment data...")
        df, raw_data = self.loader.load_all_data()
        
        print(f"Loaded {len(df)} experiments")
        
        # Save processed data
        df.to_csv(self.output_dir / 'processed_data.csv', index=False)
        
        # Run analyses
        print("Running entropy analysis...")
        entropy_analyzer = EntropyAnalyzer(df, raw_data)
        entropy_results = entropy_analyzer.analyze_entropy_patterns()
        optimal_thresholds = entropy_analyzer.find_optimal_thresholds()
        
        # Save entropy results
        with open(self.output_dir / 'entropy_analysis.json', 'w') as f:
            json.dump({
                'patterns': entropy_results,
                'optimal_thresholds': optimal_thresholds
            }, f, indent=2)
        
        entropy_analyzer.plot_entropy_analysis(self.output_dir)
        
        print("Running performance analysis...")
        perf_analyzer = PerformanceAnalyzer(df)
        factor_importance = perf_analyzer.analyze_performance_factors()
        clusters = perf_analyzer.identify_performance_clusters()
        
        # Save performance results
        factor_importance.to_csv(self.output_dir / 'factor_importance.csv')
        with open(self.output_dir / 'performance_clusters.json', 'w') as f:
            json.dump(clusters, f, indent=2, default=str)
        
        perf_analyzer.plot_performance_analysis(self.output_dir)
        
        print("Running comparison analysis...")
        comp_analyzer = ComparisonAnalyzer(df)
        comparison_df = comp_analyzer.compare_with_baselines()
        significance_results = comp_analyzer.statistical_significance_test()
        
        # Save comparison results
        if not comparison_df.empty:
            comparison_df.to_csv(self.output_dir / 'baseline_comparison.csv', index=False)
        
        with open(self.output_dir / 'statistical_significance.json', 'w') as f:
            json.dump(significance_results, f, indent=2)
        
        comp_analyzer.plot_comparison(self.output_dir)
        
        # Generate final report
        self.generate_final_report(df, entropy_results, optimal_thresholds, 
                                 factor_importance, clusters, comparison_df, 
                                 significance_results)
        
        print(f"Analysis complete! Results saved to {self.output_dir}")
    
    def generate_final_report(self, df, entropy_results, optimal_thresholds,
                            factor_importance, clusters, comparison_df, 
                            significance_results):
        """Generate comprehensive final report"""
        report = []
        report.append("# Entropy-Based Speculative Decoding: Analysis Report\n")
        report.append(f"Generated: {pd.Timestamp.now()}\n")
        report.append(f"Total Experiments: {len(df)}\n")
        
        # Executive Summary
        report.append("\n## Executive Summary\n")
        
        best_idx = df['efficiency_score'].idxmax()
        best_config = df.loc[best_idx]
        
        report.append(f"- Best efficiency score: {best_config['efficiency_score']:.3f}\n")
        report.append(f"- Best configuration: {best_config['experiment']}\n")
        report.append(f"- Average improvement over baselines: {comparison_df['efficiency_improvement'].mean():.1f}%\n" if not comparison_df.empty else "")
        
        # Optimal Thresholds
        report.append("\n## Optimal Configurations\n")
        for objective, config in optimal_thresholds.items():
            report.append(f"\n### {objective}\n")
            report.append(f"- Threshold: {config['theta_abs']:.2f}\n")
            report.append(f"- Value: {config['value']:.3f}\n")
        
        # Statistical Significance
        report.append("\n## Statistical Analysis\n")
        for metric, results in significance_results.items():
            report.append(f"\n### {metric}\n")
            report.append(f"- Entropy mean: {results['entropy_mean']:.3f}\n")
            report.append(f"- Baseline mean: {results['baseline_mean']:.3f}\n")
            report.append(f"- p-value: {results['p_value']:.4f}\n")
            report.append(f"- Effect size (Cohen's d): {results['cohens_d']:.3f}\n")
            report.append(f"- Statistically significant: {'Yes' if results['significant'] else 'No'}\n")
        
        # Performance Clusters
        report.append("\n## Performance Clusters\n")
        if 'clusters' in clusters:
            for cluster_name, cluster_info in clusters['clusters'].items():
                report.append(f"\n### {cluster_name}\n")
                report.append(f"- Size: {cluster_info['size']} experiments\n")
                report.append(f"- Mean TPS: {cluster_info['mean_tps']:.2f}\n")
                report.append(f"- Mean Acceptance: {cluster_info['mean_acceptance']:.3f}\n")
        
        # Key Findings
        report.append("\n## Key Findings\n")
        
        # Find most important factors
        if not factor_importance.empty and 'efficiency_score' in factor_importance.index:
            factors = factor_importance.loc['efficiency_score']
            top_factors = sorted(
                [(k, v['correlation']) for k, v in factors.items()],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            report.append("\n### Most Important Factors:\n")
            for factor, corr in top_factors:
                report.append(f"- {factor}: {corr:.3f} correlation\n")
        
        # Strategy effectiveness
        if 'by_strategy' in entropy_results:
            report.append("\n### Strategy Effectiveness:\n")
            strategy_scores = entropy_results['by_strategy']['efficiency_score']
            sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
            for strategy, score in sorted_strategies:
                report.append(f"- {strategy}: {score:.3f}\n")
        
        # Save report
        with open(self.output_dir / 'final_report.md', 'w') as f:
            f.write(''.join(report))


if __name__ == "__main__":
    # Example usage
    pipeline = AnalysisPipeline(
        experiment_dir="./experiments/entropy_speculative_v1",
        output_dir="./analysis/entropy_speculative_v1"
    )
    pipeline.run_full_analysis()