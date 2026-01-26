# Entropy-Based Speculative Decoding: Experiment Plan

## 1. Core Concept

### Traditional Speculative Decoding
- Drafter model generates fixed k tokens
- Verifier model evaluates all k tokens in parallel
- Accepts/rejects based on probability matching

### Entropy-Based Speculative Decoding (EBSD)
- Drafter model generates tokens until entropy threshold exceeded
- Dynamic sequence length based on model confidence
- Potential for better efficiency when drafter is confident

## 2. Entropy-Based Stopping Criterion

### Definition
```python
def compute_entropy(logits):
    """
    Compute entropy of output distribution
    H(p) = -Σ p(x) * log(p(x))
    """
    probs = softmax(logits)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy
```

### Stopping Conditions
1. **Absolute Entropy Threshold**: Stop when H(p) > θ_abs
2. **Relative Entropy Increase**: Stop when H(p_t) / H(p_{t-1}) > θ_rel
3. **Sliding Window Average**: Stop when mean(H(p_{t-w:t})) > θ_window
4. **Adaptive Threshold**: θ(t) = θ_base + α * t (increases with position)

### Key Hyperparameters
- `theta_abs`: Absolute entropy threshold (e.g., 0.5, 1.0, 2.0)
- `theta_rel`: Relative entropy increase ratio (e.g., 1.5, 2.0)
- `window_size`: Size of sliding window (e.g., 3, 5)
- `max_draft_len`: Maximum draft sequence length (e.g., 10, 20)
- `min_draft_len`: Minimum draft sequence length (e.g., 1, 2)

## 3. Implementation Architecture

### Core Components

```python
class EntropyBasedSpeculativeDecoder:
    def __init__(self, drafter, verifier, entropy_config):
        self.drafter = drafter
        self.verifier = verifier
        self.entropy_config = entropy_config
        
    def generate_draft(self, context, max_len):
        """Generate draft tokens until entropy threshold"""
        draft_tokens = []
        draft_logits = []
        draft_entropies = []
        
        for i in range(max_len):
            logits = self.drafter(context + draft_tokens)
            entropy = compute_entropy(logits)
            
            if self.should_stop(entropy, draft_entropies, i):
                break
                
            token = sample(logits)
            draft_tokens.append(token)
            draft_logits.append(logits)
            draft_entropies.append(entropy)
            
        return draft_tokens, draft_logits, draft_entropies
    
    def should_stop(self, current_entropy, entropy_history, position):
        """Implement various stopping criteria"""
        # Multiple stopping strategies
        pass
```

### Key Algorithms

1. **Entropy Computation**
   - Standard Shannon entropy
   - Rényi entropy (parameterized)
   - Top-k entropy (only consider top-k tokens)

2. **Acceptance Mechanism**
   - Standard rejection sampling
   - Modified acceptance based on entropy alignment
   - Partial acceptance with backtracking

## 4. Evaluation Metrics

### Primary Metrics
1. **Tokens per Second (TPS)**: Overall generation speed
2. **Acceptance Rate**: Fraction of draft tokens accepted
3. **Draft Length Distribution**: Statistics of generated draft lengths
4. **Speedup Factor**: TPS_EBSD / TPS_baseline

### Secondary Metrics
1. **Entropy-Acceptance Correlation**: How well entropy predicts acceptance
2. **Computational Efficiency**: FLOPs per accepted token
3. **Quality Metrics**: Perplexity, task-specific scores
4. **Latency Distribution**: P50, P90, P99 latencies

### Analysis Metrics
1. **Optimal Threshold Analysis**: Best threshold for different tasks/models
2. **Entropy Trajectory Patterns**: Common entropy evolution patterns
3. **Failure Mode Analysis**: When/why method underperforms

## 5. Experimental Setup

### Models
1. **Drafter Models**
   - Small: 125M - 350M parameters
   - Medium: 1B - 3B parameters
   - Architecture: Same family as verifier (e.g., Llama, Mistral)

2. **Verifier Models**
   - Large: 7B - 70B parameters
   - Must be compatible with drafter tokenizer

### Datasets
1. **Text Generation**
   - CommonCrawl samples
   - Wikipedia articles
   - Code generation (The Stack)
   - Scientific papers (ArXiv)

2. **Task-Specific**
   - Question answering (Natural Questions)
   - Summarization (CNN/DailyMail)
   - Translation (WMT)

### Baselines
1. **Standard Speculative Decoding**: Fixed k values (k=4, 6, 8, 10)
2. **No Speculation**: Direct generation from verifier
3. **Oracle**: Perfect acceptance (upper bound)
4. **Random k**: Randomly varying k in [1, max_k]

## 6. Ablation Studies

### Entropy Calculation
1. Temperature scaling effect
2. Top-k vs full vocabulary entropy
3. Different entropy measures (Shannon, Rényi, Tsallis)

### Stopping Criteria
1. Single vs combined criteria
2. Static vs adaptive thresholds
3. Position-dependent thresholds

### Model Size Ratios
1. Drafter/Verifier size ratios: 1:10, 1:20, 1:50
2. Same architecture vs different architectures

### Task Dependency
1. Performance across different text types
2. Domain-specific threshold tuning
3. Length-dependent behavior

## 7. Hyperparameter Optimization

### Grid Search Parameters
```python
param_grid = {
    'theta_abs': [0.5, 1.0, 1.5, 2.0, 2.5],
    'theta_rel': [1.2, 1.5, 2.0, 3.0],
    'window_size': [1, 3, 5],
    'min_draft_len': [1, 2, 3],
    'max_draft_len': [8, 12, 16, 20],
    'entropy_type': ['shannon', 'renyi', 'topk'],
    'stopping_strategy': ['absolute', 'relative', 'combined']
}
```

### Adaptive Strategies
1. **Bayesian Optimization**: For continuous hyperparameters
2. **Multi-Armed Bandits**: For online threshold adaptation
3. **Reinforcement Learning**: Learn stopping policy

## 8. Implementation Details

### Efficient Entropy Computation
```python
class CachedEntropyComputer:
    def __init__(self, vocab_size, device):
        # Precompute log for efficiency
        self.register_buffer('log_cache', 
                           torch.log(torch.arange(1, vocab_size+1) / vocab_size))
    
    def compute(self, logits):
        # Optimized entropy computation
        pass
```

### Batched Verification
- Verify multiple sequences with different draft lengths
- Efficient padding and masking strategies
- Dynamic batching based on draft lengths

### Memory Management
- Token buffer recycling
- Efficient KV cache management
- Gradient checkpointing for large models

## 9. Expected Outcomes

### Hypotheses
1. **H1**: EBSD achieves higher speedup than fixed-k when drafter confidence varies
2. **H2**: Optimal entropy threshold correlates with task complexity
3. **H3**: Relative entropy increase is more robust than absolute threshold
4. **H4**: Combined criteria outperform single criterion

### Success Criteria
1. >10% speedup over best fixed-k baseline on at least 50% of tasks
2. Acceptance rate remains within 10% of fixed-k methods
3. No quality degradation (same perplexity/task scores)
4. Robust across different model pairs

## 10. Timeline

### Phase 1: Infrastructure (Week 1-2)
- Implement base EBSD framework
- Set up evaluation pipeline
- Integrate with existing models

### Phase 2: Initial Experiments (Week 3-4)
- Run baseline comparisons
- Initial hyperparameter sweep
- Identify promising configurations

### Phase 3: Comprehensive Evaluation (Week 5-6)
- Full ablation studies
- Multi-task evaluation
- Statistical analysis

### Phase 4: Optimization & Analysis (Week 7-8)
- Implement advanced strategies
- Failure mode analysis
- Final results compilation