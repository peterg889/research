# Theoretical Framework for Curriculum Learning Experiments

## Research Question

**Primary Question**: Does the order in which training data is presented affect the speed and quality of neural network training?

**Secondary Questions**:
1. Which ordering strategies lead to faster convergence?
2. Do some orderings result in better final performance?
3. Is there an interaction between data characteristics (reading level, topic) and optimal ordering?

## Theoretical Basis

### Curriculum Learning Hypothesis
Based on Bengio et al. (2009), the core hypothesis is that presenting training examples in a meaningful order (easy → hard) can:
- Accelerate convergence
- Improve final performance
- Reduce training instability

### Why Order Might Matter

1. **Optimization Landscape**: Easy examples may guide the model to better regions of the loss landscape
2. **Feature Learning**: Simple patterns learned early may serve as building blocks
3. **Noise Reduction**: Starting with cleaner/easier examples reduces early noise
4. **Cognitive Analogy**: Mimics human learning (scaffolding)

## Experimental Design

### Control Variables (Held Constant)
- **Model Architecture**: Identical BERT configuration
- **Initialization**: Same random seed for each strategy's fresh model
- **Optimizer**: AdamW with identical hyperparameters  
- **Learning Schedule**: Same warmup and decay
- **Total Data**: Every strategy sees exactly the same samples
- **Training Duration**: Fixed number of epochs (no early stopping for fair comparison)
- **Batch Size**: Constant across strategies

### Independent Variable
- **Data Presentation Order**: The sequence in which training samples are presented

### Dependent Variables
1. **Convergence Speed**
   - Steps to reach 90% of final performance
   - Steps to reach specific loss threshold
   - Rate of loss decrease

2. **Final Performance**
   - Final validation loss
   - Final validation accuracy
   - Final perplexity

3. **Training Efficiency**
   - Area under the loss curve (lower = more efficient)
   - Total gradient updates needed
   - Training stability (loss variance)

## Curriculum Strategies

### 1. Baseline
- **Random**: Standard shuffled ordering (control condition)

### 2. Reading Level Based
- **Easy to Hard**: Start with simple texts (low Flesch-Kincaid)
- **Hard to Easy**: Start with complex texts  
- **Staged**: Easy → Medium → Hard in discrete stages

**Hypothesis**: Easy-to-hard should converge faster by building foundational patterns first

### 3. Topic Based  
- **Sequential**: Group by topic, random within groups
- **Interleaved**: Alternate between topics each batch
- **Largest First**: Start with most common topics

**Hypothesis**: Topic grouping provides coherent semantic contexts

### 4. Hybrid
- **Reading-Topic**: Sort by reading level, then by topic
- **Topic-Reading**: Group by topic, then sort by reading level

**Hypothesis**: Combining criteria may leverage both benefits

### 5. Epoch-Interleaving
- **By Epoch**: Change strategy each epoch

**Hypothesis**: Dynamic ordering may prevent overfitting to specific patterns

## Validity Considerations

### Internal Validity
1. **Randomization**: Multiple runs with different seeds
2. **Control**: Random baseline for comparison
3. **Isolation**: Only ordering varies between conditions

### Threats to Validity
1. **Early Stopping**: Could favor strategies that converge faster
   - **Mitigation**: Disable for primary analysis
   
2. **Model Capacity**: Effects may vary with model size
   - **Mitigation**: Test multiple model sizes
   
3. **Dataset Specificity**: Results may not generalize
   - **Mitigation**: Use diverse data sources

4. **Metric Selection**: Some metrics may favor certain strategies
   - **Mitigation**: Use multiple complementary metrics

## Statistical Analysis Plan

### Primary Analysis
1. **Paired t-tests**: Compare each strategy to random baseline
2. **Effect Sizes**: Cohen's d for practical significance
3. **Multiple Comparisons**: Bonferroni correction

### Secondary Analysis  
1. **Learning Curves**: Convergence rate analysis
2. **Variance Analysis**: Training stability across runs
3. **Interaction Effects**: Strategy × data characteristics

### Minimum Requirements for Validity
- At least 3 runs per strategy (5 preferred)
- Fixed training duration for all strategies
- Identical hyperparameters except ordering
- Comprehensive metric logging

## Expected Outcomes

### If Curriculum Learning Works:
- Easy-to-hard should converge 10-30% faster
- Final performance should be comparable or better
- Learning curves should be smoother

### If Order Doesn't Matter:
- All strategies perform similarly to random
- High variance between runs
- No consistent patterns

### Possible Negative Results:
- Some orderings could hurt performance
- Overfitting to specific ordering patterns
- Increased training instability

## Implementation Checklist

✅ **Essential for Validity**:
- [ ] Fresh model initialization for each strategy
- [ ] Identical hyperparameters across strategies
- [ ] Fixed epoch count (no early stopping)
- [ ] Multiple runs for statistical power
- [ ] Comprehensive logging of all metrics
- [ ] Same data seen by all strategies

⚠️ **Avoid**:
- [ ] Early stopping (biases comparison)
- [ ] Different learning rates per strategy
- [ ] Adaptive batch sizes
- [ ] Strategy-specific hyperparameters

## References

1. Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. ICML.
2. Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. ICML.
3. Kumar, G., Foster, G., Cherry, C., & Krikun, M. (2019). Reinforcement learning based curriculum optimization for neural machine translation. NAACL.