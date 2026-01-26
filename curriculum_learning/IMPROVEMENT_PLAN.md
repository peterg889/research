# Comprehensive Curriculum Learning Experiment Improvement Plan

## 🚨 Critical Issues Identified

### **From W&B Analysis:**
1. **Extremely low validation accuracy (~5%)** - Suggests model isn't learning effectively
2. **Very noisy learning curves** - Too frequent logging obscures patterns
3. **Inconsistent training lengths** - Different strategies run different numbers of steps
4. **Missing key metrics** - No training curves, proper learning rate schedules, statistical comparisons
5. **No convergence detection** - Models may be under/over-training

## 🎯 Priority 1: Fix Core Training Issues

### **1. MLM Training Problems**
**Issue:** 5% validation accuracy suggests fundamental training problems

**Fixes:**
```python
# Add proper MLM validation metrics
def compute_mlm_accuracy(logits, labels):
    """Compute MLM accuracy only on masked tokens"""
    mask = (labels != -100)
    if mask.sum() == 0:
        return 0.0
    
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) & mask
    return correct.sum().float() / mask.sum().float()

# Add perplexity metric
def compute_perplexity(loss):
    return torch.exp(loss).item()

# Add top-k accuracy
def compute_top_k_accuracy(logits, labels, k=5):
    mask = (labels != -100)
    if mask.sum() == 0:
        return 0.0
    
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == labels_expanded) & mask.unsqueeze(-1)
    return correct.any(dim=-1).sum().float() / mask.sum().float()
```

### **2. Learning Rate and Optimizer Issues**
**Issue:** Current LR schedule may be suboptimal

**Fixes:**
```python
# Better learning rate schedule with warmup and decay
def create_better_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    warmup_steps = int(num_training_steps * warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay with minimum LR
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Add learning rate finder
def find_optimal_lr(model, dataloader, max_lr=1e-2, num_iter=100):
    """Find optimal learning rate using the learning rate range test"""
    lrs = []
    losses = []
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(max_lr/1e-8)**(1/num_iter))
    
    for i, batch in enumerate(dataloader):
        if i >= num_iter:
            break
            
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
    
    return lrs, losses
```

### **3. Early Stopping and Convergence Detection**
**Issue:** No automatic stopping when training plateaus

**Fixes:**
```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
```

## 🎯 Priority 2: Enhanced Logging and Monitoring

### **1. Better W&B Integration**
**Issue:** Noisy curves, missing key metrics

**Fixes:**
```python
class EnhancedTracker(ExperimentTracker):
    def __init__(self, config, use_wandb=True):
        super().__init__(config, use_wandb)
        self.log_frequency = config.eval_every_n_steps  # Reduce logging frequency
        
    def log_training_step(self, step, strategy, metrics):
        # Only log every N steps to reduce noise
        if step % self.log_frequency == 0:
            super().log_training_step(step, strategy, metrics)
            
            # Add additional metrics
            if self.use_wandb:
                import wandb
                
                # Log learning phase
                epoch = step // (len(train_dataset) // batch_size)
                wandb.log({
                    f"epoch": epoch,
                    f"train/{strategy}/perplexity": math.exp(metrics['loss']),
                    f"train/{strategy}/lr": metrics['learning_rate'],
                    f"train/{strategy}/grad_norm": metrics['gradient_norm'],
                    "step": step
                })
    
    def log_epoch_summary(self, epoch, strategy, train_metrics, val_metrics):
        """Log epoch-level summaries for cleaner visualization"""
        if self.use_wandb:
            import wandb
            wandb.log({
                f"epoch_summary/{strategy}/train_loss_avg": train_metrics['loss_avg'],
                f"epoch_summary/{strategy}/train_acc_avg": train_metrics['acc_avg'],
                f"epoch_summary/{strategy}/val_loss": val_metrics['val_loss'],
                f"epoch_summary/{strategy}/val_acc": val_metrics['val_accuracy'],
                f"epoch_summary/{strategy}/val_perplexity": math.exp(val_metrics['val_loss']),
                "epoch": epoch
            })
    
    def create_comparison_plots(self, all_results):
        """Create comparison plots across strategies"""
        if not self.use_wandb:
            return
            
        import wandb
        import matplotlib.pyplot as plt
        
        # Strategy comparison table
        table_data = []
        for strategy, results in all_results.items():
            table_data.append([
                strategy,
                results['final_loss'],
                results['final_accuracy'],
                results['convergence_epoch'],
                results['total_time']
            ])
        
        table = wandb.Table(
            data=table_data,
            columns=["Strategy", "Final Loss", "Final Acc", "Convergence Epoch", "Time (min)"]
        )
        wandb.log({"strategy_comparison": table})
        
        # Learning curves comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for strategy, results in all_results.items():
            epochs = range(len(results['epoch_losses']))
            ax1.plot(epochs, results['epoch_losses'], label=strategy, linewidth=2)
            ax2.plot(epochs, results['epoch_accuracies'], label=strategy, linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Learning Curves - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Learning Curves - Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        wandb.log({"learning_curves_comparison": wandb.Image(fig)})
        plt.close(fig)
```

### **2. Statistical Analysis Enhancement**
**Issue:** No proper statistical significance testing

**Fixes:**
```python
class StatisticalAnalyzer:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        
    def compare_strategies(self, strategy_results, baseline="random"):
        """Compare all strategies against baseline with statistical tests"""
        results = {}
        baseline_scores = strategy_results[baseline]['final_scores']
        
        for strategy, data in strategy_results.items():
            if strategy == baseline:
                continue
                
            scores = data['final_scores']
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(scores, baseline_scores)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(scores)-1)*np.var(scores, ddof=1) + 
                                 (len(baseline_scores)-1)*np.var(baseline_scores, ddof=1)) / 
                                (len(scores) + len(baseline_scores) - 2))
            cohens_d = (np.mean(scores) - np.mean(baseline_scores)) / pooled_std
            
            # Confidence interval
            se = pooled_std * np.sqrt(1/len(scores) + 1/len(baseline_scores))
            margin_error = stats.t.ppf((1 + self.confidence_level) / 2, 
                                     len(scores) + len(baseline_scores) - 2) * se
            ci_lower = (np.mean(scores) - np.mean(baseline_scores)) - margin_error
            ci_upper = (np.mean(scores) - np.mean(baseline_scores)) + margin_error
            
            results[strategy] = {
                'mean_improvement': np.mean(scores) - np.mean(baseline_scores),
                'relative_improvement': (np.mean(scores) - np.mean(baseline_scores)) / np.mean(baseline_scores),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'confidence_interval': (ci_lower, ci_upper),
                'is_significant': p_value < (1 - self.confidence_level),
                'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
            }
            
        return results
    
    def _interpret_effect_size(self, d):
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def multiple_comparisons_correction(self, p_values, method='bonferroni'):
        """Apply multiple comparisons correction"""
        if method == 'bonferroni':
            corrected_alpha = 0.05 / len(p_values)
            return [p < corrected_alpha for p in p_values]
        elif method == 'benjamini_hochberg':
            from statsmodels.stats.multitest import multipletests
            _, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
            return corrected_p < 0.05
```

## 🎯 Priority 3: Experiment Design Improvements

### **1. Cross-Validation and Robustness**
**Issue:** Single runs don't provide reliable estimates

**Fixes:**
```python
class RobustExperiment:
    def __init__(self, config):
        self.config = config
        
    def run_with_cross_validation(self, k_folds=5):
        """Run experiment with k-fold cross-validation"""
        from sklearn.model_selection import KFold
        
        # Split data into folds
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        all_results = {}
        
        for strategy in self.config.strategies:
            strategy_results = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
                print(f"Running {strategy} - Fold {fold+1}/{k_folds}")
                
                # Create fold-specific datasets
                train_subset = Subset(self.dataset, train_idx)
                val_subset = Subset(self.dataset, val_idx)
                
                # Train model
                result = self.train_single_fold(strategy, train_subset, val_subset)
                strategy_results.append(result)
                
            all_results[strategy] = {
                'fold_results': strategy_results,
                'mean_score': np.mean([r['final_score'] for r in strategy_results]),
                'std_score': np.std([r['final_score'] for r in strategy_results]),
                'scores': [r['final_score'] for r in strategy_results]
            }
            
        return all_results
    
    def sensitivity_analysis(self, param_grid):
        """Test sensitivity to hyperparameters"""
        results = {}
        
        for params in param_grid:
            config_copy = copy.deepcopy(self.config)
            for key, value in params.items():
                setattr(config_copy, key, value)
                
            experiment = Experiment(config_copy)
            result = experiment.run()
            
            param_key = "_".join([f"{k}={v}" for k, v in params.items()])
            results[param_key] = result
            
        return results
```

### **2. Hyperparameter Optimization**
**Issue:** Fixed hyperparameters may be suboptimal

**Fixes:**
```python
def optimize_hyperparameters(base_config, search_space, n_trials=50):
    """Optimize hyperparameters using Optuna"""
    import optuna
    
    def objective(trial):
        # Sample hyperparameters
        config = copy.deepcopy(base_config)
        config.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        config.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        config.warmup_steps = trial.suggest_int('warmup_steps', 100, 2000)
        config.weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        
        # Run quick experiment
        config.num_epochs = 5  # Quick evaluation
        config.num_runs = 1
        config.strategies = ["random", "reading_level_easy_to_hard"]  # Limited strategies
        
        experiment = Experiment(config)
        results = experiment.run()
        
        # Return improvement over random
        random_score = results['results']['random']['accuracies'][-1]
        curriculum_score = results['results']['reading_level_easy_to_hard']['accuracies'][-1]
        
        return curriculum_score - random_score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value
```

### **3. Enhanced Evaluation Metrics**
**Issue:** Limited metrics don't capture full picture

**Fixes:**
```python
class ComprehensiveEvaluator:
    def __init__(self, config):
        self.config = config
        
    def evaluate_comprehensive(self, model, test_loader, strategy_name):
        """Comprehensive evaluation with multiple metrics"""
        model.eval()
        
        metrics = {
            'accuracy': [],
            'top_5_accuracy': [],
            'perplexity': [],
            'loss': [],
            'prediction_confidence': [],
            'prediction_entropy': []
        }
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                
                labels = model.create_mlm_labels(input_ids, test_loader.dataset.tokenizer)
                masked_input_ids = model._last_masked_input_ids
                
                outputs = model(masked_input_ids, attention_mask, labels)
                
                # Standard metrics
                metrics['loss'].append(outputs.loss.item())
                metrics['perplexity'].append(torch.exp(outputs.loss).item())
                
                # Accuracy metrics
                predictions = torch.argmax(outputs.logits, dim=-1)
                mask = (labels != -100)
                
                if mask.sum() > 0:
                    acc = ((predictions == labels) & mask).float().sum() / mask.sum()
                    metrics['accuracy'].append(acc.item())
                    
                    # Top-5 accuracy
                    _, top5_preds = torch.topk(outputs.logits, 5, dim=-1)
                    labels_expanded = labels.unsqueeze(-1).expand_as(top5_preds)
                    top5_correct = (top5_preds == labels_expanded) & mask.unsqueeze(-1)
                    top5_acc = top5_correct.any(dim=-1).sum().float() / mask.sum()
                    metrics['top_5_accuracy'].append(top5_acc.item())
                    
                    # Prediction confidence and entropy
                    probs = torch.softmax(outputs.logits, dim=-1)
                    max_probs, _ = torch.max(probs, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                    
                    masked_max_probs = max_probs[mask]
                    masked_entropy = entropy[mask]
                    
                    metrics['prediction_confidence'].extend(masked_max_probs.cpu().tolist())
                    metrics['prediction_entropy'].extend(masked_entropy.cpu().tolist())
        
        # Aggregate metrics
        final_metrics = {}
        for key, values in metrics.items():
            if values:
                final_metrics[f'{key}_mean'] = np.mean(values)
                final_metrics[f'{key}_std'] = np.std(values)
                final_metrics[f'{key}_median'] = np.median(values)
        
        return final_metrics
```

## 🎯 Priority 4: Reproducibility and Documentation

### **1. Better Random Seed Management**
```python
def set_reproducible_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    import random
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For datasets library
    os.environ['PYTHONHASHSEED'] = str(seed)
```

### **2. Experiment Versioning and Metadata**
```python
class ExperimentVersion:
    def __init__(self):
        self.git_commit = self._get_git_commit()
        self.timestamp = datetime.now().isoformat()
        self.python_version = sys.version
        self.torch_version = torch.__version__
        self.cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        
    def _get_git_commit(self):
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except:
            return "unknown"
    
    def to_dict(self):
        return {
            'git_commit': self.git_commit,
            'timestamp': self.timestamp,
            'python_version': self.python_version,
            'torch_version': self.torch_version,
            'cuda_version': self.cuda_version
        }
```

## 📊 Summary of Critical Actions

### **Immediate (This Week):**
1. ✅ Fix MLM accuracy calculation and add perplexity
2. ✅ Implement early stopping
3. ✅ Reduce W&B logging frequency
4. ✅ Add statistical significance testing
5. ✅ Fix learning rate schedule

### **Next Week:**
1. ✅ Implement cross-validation
2. ✅ Add comprehensive evaluation metrics
3. ✅ Create better W&B visualizations
4. ✅ Add hyperparameter optimization

### **Ongoing:**
1. ✅ Monitor training more carefully
2. ✅ Document all changes
3. ✅ Version experiments properly

This plan addresses the fundamental issues causing low accuracy and poor experimental quality. The immediate actions should resolve the training problems, while the longer-term improvements will make the research more robust and publication-ready.