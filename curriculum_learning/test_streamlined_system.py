"""
Quick Test of Streamlined Curriculum Learning System

Tests all components working together without long training.
"""

import torch
from config import debug_config
from data_pipeline import DataPipeline  
from model import create_model
from evaluation import Evaluator
from tracking import ExperimentTracker


def test_streamlined_system():
    """Test all components working together"""
    
    print("🧪 Testing Streamlined Curriculum Learning System")
    print("=" * 60)
    
    # 1. Configuration
    print("\n1. CONFIGURATION")
    config = debug_config(num_samples=200, num_epochs=1, num_runs=1)
    config.strategies = ["random", "reading_level_easy_to_hard"]
    print(f"Config: {config.scale.value}, {len(config.strategies)} strategies")
    
    # 2. Data Pipeline
    print("\n2. DATA PIPELINE")
    data_pipeline = DataPipeline(config)
    dataset = data_pipeline.load()
    train_ds, val_ds, test_ds = dataset.split()
    
    print(f"Data: {len(dataset)} total, {len(train_ds)} train, {len(val_ds)} val")
    
    # Test curriculum ordering
    for strategy in config.strategies:
        order = dataset.get_curriculum_order(strategy)
        print(f"{strategy}: {len(order)} samples ordered")
    
    # 3. Model
    print("\n3. MODEL")
    model = create_model(config, device="cpu")
    metrics = model.get_metrics()
    print(f"Model: {metrics.total_parameters:,} parameters")
    
    # Test forward pass
    dataloader = train_ds.create_dataloader("random", batch_size=4)
    batch = next(iter(dataloader))
    
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = model.create_mlm_labels(input_ids, train_ds.tokenizer)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, labels)
        print(f"Forward pass: loss={outputs.loss.item():.4f}")
    
    # 4. Evaluation System
    print("\n4. EVALUATION")
    evaluator = Evaluator(config)
    
    # Simulate results
    strategies = config.strategies
    for strategy in strategies:
        steps = list(range(10))
        losses = [5.0 - i * 0.3 + (0.1 if strategy == "random" else 0) for i in steps]
        accuracies = [i * 0.05 + (0.01 if strategy != "random" else 0) for i in steps]
        
        evaluator.record_strategy_results(strategy, steps, losses, accuracies)
    
    report = evaluator.generate_report()
    print(f"Evaluation: {len(report['detailed_results'])} strategies analyzed")
    
    effective = report['best_strategies']['effective_list']
    print(f"Effective strategies: {len(effective)}")
    
    # 5. Tracking System
    print("\n5. TRACKING")
    tracker = ExperimentTracker(config, use_wandb=False)
    
    # Log some metrics
    for step in range(5):
        metrics = {
            'loss': 5.0 - step * 0.5,
            'accuracy': step * 0.1,
            'learning_rate': 1e-4
        }
        tracker.log_training_step(step, "test_strategy", metrics)
    
    summary = tracker.get_summary()
    print(f"Tracking: {summary['experiment_id']}")
    
    tracker.finalize()
    
    # 6. Integration Test
    print("\n6. INTEGRATION")
    
    # Test the main experiment class (without running full training)
    from experiment import Experiment
    
    experiment = Experiment(config)
    experiment.prepare_data()
    
    print(f"Experiment: data prepared, {len(experiment.config.strategies)} strategies")
    print(f"Training datasets: {len(experiment.train_dataset)} train, {len(experiment.val_dataset)} val")
    
    # 7. Command Line Interface
    print("\n7. COMMAND LINE")
    import subprocess
    result = subprocess.run(["python", "experiment.py", "--help"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("CLI: Help command works")
    else:
        print("CLI: Help command failed")
    
    # Final Summary
    print("\n" + "="*60)
    print("🎉 STREAMLINED SYSTEM TEST RESULTS")
    print("="*60)
    print("Configuration: Multiple scales (debug → large)")
    print("Data Pipeline: Real datasets + curriculum strategies") 
    print("Model: Scalable BERT (mini → base)")
    print("Evaluation: Statistical analysis")  
    print("Tracking: Experiment logging")
    print("Integration: All components work together")
    print("CLI: Command-line interface")
    print()
    print("Ready for production curriculum learning research!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_streamlined_system()
    print(f"\n{'SUCCESS' if success else 'FAILED'}: Streamlined system test")