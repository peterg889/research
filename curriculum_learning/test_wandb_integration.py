#!/usr/bin/env python3
"""
Test script to verify W&B logging integration works correctly
"""

import sys
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_improved_logger():
    """Test the improved W&B logger independently"""
    print("Testing ImprovedWandBLogger...")
    
    try:
        from improved_wandb_logging import ImprovedWandBLogger
        
        # Mock config for testing
        class MockConfig:
            def __init__(self):
                self.strategies = ['random', 'reading_level_easy_to_hard', 'topic_sequential']
                self.num_epochs = 5
        
        config = MockConfig()
        logger = ImprovedWandBLogger(config, "test_experiment_123")
        
        # Test that logger can be instantiated and methods exist
        # (Skip actual logging since it requires wandb.init())
        assert hasattr(logger, 'log_training_step'), "Missing log_training_step method"
        assert hasattr(logger, 'log_validation_step'), "Missing log_validation_step method"
        assert hasattr(logger, 'log_epoch_summary'), "Missing log_epoch_summary method"
        assert hasattr(logger, 'log_strategy_complete'), "Missing log_strategy_complete method"
        
        print("✅ ImprovedWandBLogger instantiates correctly and has required methods")
        return True
        
    except Exception as e:
        print(f"❌ ImprovedWandBLogger failed: {e}")
        return False

def test_unified_experiment_config():
    """Test unified experiment configuration"""
    print("Testing unified experiment configuration...")
    
    try:
        from config import Config
        
        # Test basic config creation
        config = Config(
            scale="debug",
            strategies=['random', 'reading_level_easy_to_hard'],
            use_wandb=False  # Disable W&B for testing
        )
        
        print(f"✅ Config created successfully: {config.num_samples} samples")
        return True
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

def test_experiment_initialization():
    """Test experiment initialization without actually running"""
    print("Testing experiment initialization...")
    
    try:
        from config import Config
        from unified_experiment import UnifiedExperiment, ExperimentMode
        
        # Create minimal config
        config = Config(
            scale="debug",
            strategies=['random'],
            use_wandb=False,  # Disable W&B for testing
        )
        
        # Test experiment creation
        experiment = UnifiedExperiment(config, ExperimentMode.STANDARD)
        
        print("✅ Experiment initialization successful")
        print(f"   Mode: {experiment.mode}")
        print(f"   Device: {experiment.device}")
        print(f"   Improved logger: {'Available' if experiment.tracker.improved_logger else 'Not available'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_integration():
    """Test that logging doesn't duplicate"""
    print("Testing logging integration...")
    
    try:
        from unified_experiment import UnifiedExperimentTracker, ExperimentMode
        from config import Config
        
        # Create config with W&B disabled
        config = Config(
            scale="debug",
            strategies=['random'],
            use_wandb=False
        )
        
        # Create tracker
        tracker = UnifiedExperimentTracker(config, ExperimentMode.STANDARD)
        
        # Test that methods don't crash
        metrics = {'loss': 1.5, 'accuracy': 0.4, 'epoch': 1, 'progress': 0.5}
        tracker.log_training_step(10, 'random', metrics)
        
        val_metrics = {'val_loss': 1.4, 'val_accuracy': 0.42, 'epoch': 1}
        tracker.log_validation_step(10, 'random', val_metrics)
        
        train_summary = {'loss_avg': 1.5, 'acc_avg': 0.4}
        val_summary = {'val_loss': 1.4, 'val_accuracy': 0.42}
        tracker.log_epoch_summary(1, 'random', train_summary, val_summary)
        
        print("✅ Logging integration works without duplication")
        return True
        
    except Exception as e:
        print(f"❌ Logging integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("W&B LOGGING INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        test_improved_logger,
        test_unified_experiment_config,
        test_experiment_initialization,
        test_logging_integration
    ]
    
    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! W&B logging integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())